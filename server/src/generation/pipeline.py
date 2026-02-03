"""Main generation pipeline orchestration.

This module provides the GenerationPipeline class that manages all image
generation modes for the Sound Dancer AI visualization system.

Generation Modes (GenerationMode enum):
========================================

1. FEEDBACK (default) - Real-time Live Performance
   - First frame: txt2img generation (creates initial image)
   - Subsequent frames: img2img using previous frame as input
   - Best for: Live VJ performance, real-time music visualization
   - FPS: 1-5 fps depending on hardware
   - Features: Auto-reset, preview mode, latent blending for stability

2. KEYFRAME_RIFE - Real-time with Pose Control
   - Generates keyframes every N frames with ControlNet pose guidance
   - Uses RIFE neural interpolation between keyframes
   - Best for: Animated character visualization, dance performances
   - FPS: 2-8 fps effective (1-2 fps keyframes + interpolated frames)
   - Features: Procedural pose animation, audio-reactive poses

Performance Characteristics:
============================
- FEEDBACK: Fastest, ~200-500ms per frame
- KEYFRAME_RIFE: Medium, ~400ms per keyframe + ~10ms interpolation

Hardware Requirements:
======================
- All modes: NVIDIA GPU with 8GB+ VRAM recommended
- ControlNet (KEYFRAME_RIFE): Requires SD 1.5 base model
"""

import gc
import time
import logging
import threading
from typing import Optional

import torch
from PIL import Image
from dataclasses import dataclass

from .stream_diffusion import StreamDiffusionWrapper
from .keyframe_interpolator import KeyframeInterpolationPipeline, KeyframeConfig
from .frame_encoder import FrameEncoder
from .base import GenerationRequest, ImageGenerator
from .backends import create_generator

from ..mapping.audio_mapper import AudioMapper, GenerationParams
from ..server.protocol import MappingConfig
from ..server.protocol import AudioMetrics, GenerationConfig, GenerationMode
from ..presets.defaults import get_preset
from ..config import settings
from ..story.controller import StoryController
from ..story.schema import StoryScript, StoryState
from ..lyrics import KeywordExtractor

logger = logging.getLogger(__name__)


@dataclass
class GeneratedFrame:
    """A generated frame with metadata."""

    image: Image.Image
    jpeg_bytes: bytes
    frame_id: int
    timestamp: float
    params: GenerationParams
    generation_time_ms: float


class GenerationPipeline:
    """Orchestrates the full audio-to-image generation pipeline.

    The pipeline handles:
    - Mode selection and initialization (FEEDBACK, KEYFRAME_RIFE)
    - Audio-to-parameter mapping via AudioMapper
    - Frame generation and encoding
    - Story-driven prompt generation
    - Lyric keyword extraction and injection

    Usage:
        pipeline = GenerationPipeline(config)
        pipeline.initialize()

        # In generation loop:
        frame = pipeline.generate(audio_metrics)
        # frame.jpeg_bytes is ready to send to client

    Configuration:
        Use GenerationConfig to set mode, prompts, and parameters.
        Use update_config() to change settings at runtime.
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        generator: Optional["ImageGenerator"] = None,
    ):
        """Initialize the generation pipeline.

        Args:
            config: Generation configuration
            generator: Optional pre-configured ImageGenerator. If provided,
                      the pipeline will use this instead of creating its own.
                      This enables dependency injection for testing and
                      plug-and-play backend swapping.
        """
        self.config = config or GenerationConfig()
        self._diffusion: Optional[StreamDiffusionWrapper] = None
        self._injected_generator: Optional["ImageGenerator"] = generator
        self._keyframe_pipeline: Optional[KeyframeInterpolationPipeline] = None
        self._encoder = FrameEncoder()
        self._mapper = AudioMapper(
            get_preset(self.config.mapping_preset),
            use_color_organ=(self.config.mapping_preset == "color_organ")
        )

        # Story-driven prompt generation
        self._story_controller = StoryController()

        # Lyric-driven prompt generation
        self._keyword_extractor = KeywordExtractor()

        self._frame_count = 0
        self._last_frame: Optional[Image.Image] = None
        self._base_image: Optional[Image.Image] = None  # Persistent base image for img2img
        self._initialized = False
        self._generation_epoch = 0  # Increments on reset to invalidate in-flight generations

        self._current_audio_metrics: Optional[AudioMetrics] = None

        # FPS tracking
        self._fps_samples: list[float] = []
        self._last_frame_time = 0.0

        # Anchored mode for FLUX - use first txt2img as stable reference
        # Instead of feedback loop (which causes drift), always condition on anchor
        self._anchor_frame: Optional[Image.Image] = None
        self._use_anchored_mode = False  # Auto-enabled for FLUX backends

        # Thread safety lock for cleanup/reinit operations
        # This prevents race conditions when generate() runs in a thread pool
        # while config updates trigger cleanup/reinit on the main thread
        self._state_lock = threading.RLock()

    def initialize(self) -> None:
        """Initialize the generation pipeline.

        If an ImageGenerator was injected via the constructor, it will be used.
        Otherwise, a StreamDiffusionWrapper is created based on configuration.
        """
        if self._initialized:
            return

        logger.info("Initializing generation pipeline...")

        mode = self.config.generation_mode

        # Check if we have an injected generator
        if self._injected_generator is not None:
            logger.info("Using injected ImageGenerator")
            if not self._injected_generator.is_initialized:
                self._injected_generator.initialize()

            # For backward compatibility, we still need _diffusion for some operations
            # If the injected generator is a StreamDiffusionBackend, use its wrapper
            from .backends import StreamDiffusionBackend
            if isinstance(self._injected_generator, StreamDiffusionBackend):
                self._diffusion = self._injected_generator.wrapper
            else:
                # For non-StreamDiffusion backends, _diffusion stays None
                # and we use the abstract interface
                self._diffusion = None

            # Enable FLUX-specific features for FLUX backends
            # FLUX/Klein relies on prompt changes for variation, not strength
            from .backends.flux_klein_backend import FluxKleinBackend
            if isinstance(self._injected_generator, FluxKleinBackend):
                logger.info("FLUX backend detected - enabling aggressive prompt modulation")
                self._mapper.set_flux_mode(True)
                # Enable anchored mode by default for FLUX - avoids feedback drift
                self._use_anchored_mode = True
                logger.info("FLUX backend detected - enabling anchored mode (stable reference frame)")
                # Check periodic refresh setting
                periodic = getattr(self.config, 'periodic_pose_refresh', False)
                logger.info(f"FLUX settings: anchored_mode={self._use_anchored_mode}, periodic_pose_refresh={periodic}")
                if not periodic:
                    logger.info("TIP: Enable 'periodic_pose_refresh' to auto-refresh anchor every 8 beats")
                # Apply periodic refresh setting to mapper
                self._mapper.set_periodic_pose_refresh(periodic)

            self._initialized = True
            logger.info("Generation pipeline ready (injected generator)")
            return

        # Keyframe + RIFE mode - pose-guided keyframes with interpolation
        if mode == GenerationMode.KEYFRAME_RIFE:
            logger.info("Initializing Keyframe + RIFE hybrid pipeline...")

            # Use StreamDiffusion for keyframes (ControlNet disabled by default for speed)
            use_controlnet = getattr(self.config, 'use_controlnet', False)
            controlnet_weight = getattr(self.config, 'controlnet_pose_weight', 0.8)
            acceleration = getattr(self.config, 'acceleration', 'lcm')
            hyper_sd_steps = getattr(self.config, 'hyper_sd_steps', 1)

            self._diffusion = StreamDiffusionWrapper(
                model_id=self.config.model_id,
                width=self.config.width,
                height=self.config.height,
                device=settings.device,
                use_controlnet=use_controlnet,
                controlnet_weight=controlnet_weight,
                acceleration=acceleration,
                hyper_sd_steps=hyper_sd_steps,
            )
            self._diffusion.initialize()

            # Configure procedural pose for keyframes
            use_procedural = getattr(self.config, 'use_procedural_pose', True)
            if use_procedural:
                self._diffusion.set_procedural_pose(True)
                pose_mode = getattr(self.config, 'pose_animation_mode', 'gentle')
                pose_framing = getattr(self.config, 'pose_framing', 'upper_body')
                pose_speed = getattr(self.config, 'pose_animation_speed', 1.0)
                pose_intensity = getattr(self.config, 'pose_animation_intensity', 0.5)
                self._diffusion.set_pose_animation_mode(pose_mode)
                self._diffusion.set_pose_framing(pose_framing)
                self._diffusion.set_pose_animation_speed(pose_speed)
                self._diffusion.set_pose_intensity(pose_intensity)

            # Setup keyframe pipeline with interpolation
            keyframe_config = KeyframeConfig(
                keyframe_interval=getattr(self.config, 'keyframe_interval', 4),
                interpolation_mode="blend",  # Use blend for now, upgrade to RIFE later
                audio_adaptive=True,
            )
            self._keyframe_pipeline = KeyframeInterpolationPipeline(
                generator_fn=self._generate_keyframe_with_pose,
                config=keyframe_config,
                device=settings.device,
            )
            self._keyframe_pipeline.initialize()
            logger.info(f"Keyframe + RIFE pipeline ready (interval={keyframe_config.keyframe_interval}, controlnet={use_controlnet})")

        else:
            # FEEDBACK mode - use backend factory to select stream_diffusion or flux_klein
            backend = getattr(self.config, 'generator_backend', 'stream_diffusion')
            logger.info(f"Using generator backend: {backend}")

            if backend == "flux_klein":
                # Use FLUX.2 Klein backend via factory with session config
                self._injected_generator = create_generator(
                    backend="flux_klein",
                    model_id=getattr(self.config, 'flux_model_id', None) or settings.flux_model_id,
                    width=self.config.width,
                    height=self.config.height,
                    device=settings.device,
                    use_controlnet=getattr(self.config, 'use_controlnet', False),
                    controlnet_weight=getattr(self.config, 'controlnet_pose_weight', 0.8),
                    use_taesd=getattr(self.config, 'use_taesd', False),
                    temporal_coherence="blending" if getattr(self.config, 'latent_blending', True) else "none",
                    acceleration=getattr(self.config, 'acceleration', 'lcm'),
                    hyper_sd_steps=getattr(self.config, 'hyper_sd_steps', 1),
                    # FLUX-specific settings
                    precision=getattr(self.config, 'flux_precision', None) or settings.flux_precision,
                    cpu_offload=getattr(self.config, 'flux_cpu_offload', None) or settings.flux_cpu_offload,
                    inference_steps=getattr(self.config, 'flux_inference_steps', None) or settings.flux_inference_steps,
                    guidance_scale=getattr(self.config, 'flux_guidance_scale', None) or settings.flux_guidance_scale,
                    compile_transformer=getattr(self.config, 'flux_compile', None) or settings.flux_compile,
                    cache_prompt_embeds=getattr(self.config, 'flux_cache_prompt', None) or settings.flux_cache_prompt,
                )
                self._injected_generator.initialize()
                self._diffusion = None  # FLUX doesn't use StreamDiffusionWrapper
                logger.info("FLUX.2 Klein backend ready")
            elif backend == "audio_reactive":
                # Use AudioReactive backend for high-responsiveness music visualization
                self._injected_generator = create_generator(
                    backend="audio_reactive",
                    model_id=self.config.model_id,
                    width=self.config.width,
                    height=self.config.height,
                    device=settings.device,
                    num_inference_steps=getattr(self.config, 'num_inference_steps', 4),
                    use_taesd=getattr(self.config, 'use_taesd', True),
                )
                self._injected_generator.initialize()
                self._diffusion = None  # AudioReactive doesn't use StreamDiffusionWrapper
                logger.info("AudioReactive backend ready")
            else:
                # Default: StreamDiffusion backend
                # Pass SOTA settings to diffusion wrapper
                # Note: ControlNet requires diffusers path (not StreamDiffusion)
                use_controlnet = getattr(self.config, 'use_controlnet', False)
                controlnet_weight = getattr(self.config, 'controlnet_pose_weight', 0.8)
                use_taesd = getattr(self.config, 'use_taesd', False)
                temporal_coherence = getattr(self.config, 'temporal_coherence', 'blending')
                acceleration = getattr(self.config, 'acceleration', 'lcm')
                hyper_sd_steps = getattr(self.config, 'hyper_sd_steps', 1)

                logger.info(f"SOTA settings: controlnet={use_controlnet}, cn_weight={controlnet_weight}, taesd={use_taesd}, temporal={temporal_coherence}, acceleration={acceleration}")

                self._diffusion = StreamDiffusionWrapper(
                    model_id=self.config.model_id,
                    width=self.config.width,
                    height=self.config.height,
                    use_controlnet=use_controlnet,
                    controlnet_weight=controlnet_weight,
                    use_taesd=use_taesd,
                    temporal_coherence=temporal_coherence,
                    acceleration=acceleration,
                    hyper_sd_steps=hyper_sd_steps,
                )
                self._diffusion.initialize()
                logger.info("StreamDiffusion backend ready")

        self._initialized = True
        logger.info("Generation pipeline ready")

    def update_config(self, config: GenerationConfig) -> None:
        """Update generation configuration."""
        model_changed = config.model_id != self.config.model_id
        size_changed = (
            config.width != self.config.width or config.height != self.config.height
        )
        mode_changed = config.generation_mode != self.config.generation_mode

        # Check if SOTA settings changed (requires re-init)
        sota_changed = (
            getattr(config, 'use_controlnet', False) != getattr(self.config, 'use_controlnet', False) or
            getattr(config, 'use_taesd', False) != getattr(self.config, 'use_taesd', False) or
            getattr(config, 'temporal_coherence', 'blending') != getattr(self.config, 'temporal_coherence', 'blending')
        )

        # Check if acceleration method changed (requires re-init to swap LoRA and scheduler)
        acceleration_changed = (
            getattr(config, 'acceleration', 'lcm') != getattr(self.config, 'acceleration', 'lcm') or
            (getattr(config, 'acceleration', 'lcm') == 'hyper-sd' and
             getattr(config, 'hyper_sd_steps', 1) != getattr(self.config, 'hyper_sd_steps', 1))
        )

        self.config = config
        self._mapper.set_preset(get_preset(config.mapping_preset))

        # Enable color organ mode if that preset is selected
        if config.mapping_preset == "color_organ":
            self._mapper.set_color_organ_mode(True)
        elif self._mapper._use_color_organ:
            # Disable if switching away from color organ
            self._mapper.set_color_organ_mode(False)

        # Update periodic pose refresh setting from config
        periodic_pose_refresh = getattr(config, 'periodic_pose_refresh', False)
        self._mapper.set_periodic_pose_refresh(periodic_pose_refresh)

        # Update FLUX anchored mode setting - only for FLUX backends
        from .backends.flux_klein_backend import FluxKleinBackend
        if isinstance(self._injected_generator, FluxKleinBackend):
            flux_anchored = getattr(config, 'flux_anchored_mode', True)
            if flux_anchored != self._use_anchored_mode:
                self.set_anchored_mode(flux_anchored)
        elif self._use_anchored_mode and not isinstance(self._injected_generator, FluxKleinBackend):
            # Disable anchored mode if it was enabled but we're not using FLUX
            self.set_anchored_mode(False)

        # Update dynamic settings on diffusion wrapper without full re-init
        if self._diffusion:
            # Update ControlNet weight dynamically
            cn_weight = getattr(config, 'controlnet_pose_weight', 0.8)
            self._diffusion.set_controlnet_weight(cn_weight)
            # Update pose lock mode
            pose_lock = getattr(config, 'controlnet_pose_lock', True)
            self._diffusion.set_pose_lock(pose_lock)

            # Update procedural pose settings
            use_procedural = getattr(config, 'use_procedural_pose', False)
            logger.info(f"CONFIG UPDATE: use_procedural_pose={use_procedural}")
            self._diffusion.set_procedural_pose(use_procedural)
            if use_procedural:
                pose_mode = getattr(config, 'pose_animation_mode', 'gentle')
                pose_speed = getattr(config, 'pose_animation_speed', 1.0)
                pose_intensity = getattr(config, 'pose_animation_intensity', 0.5)
                pose_framing = getattr(config, 'pose_framing', 'upper_body')
                self._diffusion.set_pose_animation_mode(pose_mode)
                self._diffusion.set_pose_animation_speed(pose_speed)
                self._diffusion.set_pose_intensity(pose_intensity)
                self._diffusion.set_pose_framing(pose_framing)
                logger.debug(f"Updated procedural pose: mode={pose_mode}, speed={pose_speed}, intensity={pose_intensity}, framing={pose_framing}")

                # Update procedural txt2img settings
                procedural_txt2img = getattr(config, 'procedural_use_txt2img', True)
                procedural_seed = getattr(config, 'procedural_fixed_seed', None)
                procedural_blend = getattr(config, 'procedural_blend_weight', 0.4)
                self._diffusion.set_procedural_txt2img_mode(procedural_txt2img)
                self._diffusion.set_procedural_fixed_seed(procedural_seed)
                self._diffusion.set_procedural_blend_weight(procedural_blend)
                logger.debug(f"Updated procedural txt2img: use_txt2img={procedural_txt2img}, seed={procedural_seed}, blend={procedural_blend}")

            logger.debug(f"Updated ControlNet: weight={cn_weight}, pose_lock={pose_lock}, procedural={use_procedural}")

            # Handle LoRA updates (hot-swap)
            if hasattr(config, 'loras'):
                loras = config.loras if config.loras is not None else settings.default_loras
                self._diffusion.load_custom_loras(loras)

        # Handle LoRA updates for injected generators (e.g., AudioReactive backend)
        if self._injected_generator is not None and self._diffusion is None:
            if hasattr(config, 'loras') and hasattr(self._injected_generator, 'load_custom_loras'):
                loras = config.loras if config.loras is not None else settings.default_loras
                self._injected_generator.load_custom_loras(loras)

        if model_changed or size_changed or sota_changed or mode_changed or acceleration_changed:
            # Re-initialize pipeline for new model/size/SOTA settings/mode/acceleration
            # Use lock to prevent race conditions with concurrent generate() calls
            with self._state_lock:
                logger.info(f"Re-initializing pipeline: model={model_changed}, size={size_changed}, sota={sota_changed}, mode={mode_changed}, acceleration={acceleration_changed}")
                self._initialized = False
                self.cleanup()
                self.initialize()

    def update_mapping_config(self, mapping_config: MappingConfig) -> None:
        """Update audio-to-parameter mapping configuration."""
        self._mapper.set_dynamic_config(mapping_config)

        # Update crossfeed/blending settings on the diffusion wrapper
        if self._diffusion and hasattr(mapping_config, 'crossfeed'):
            cf = mapping_config.crossfeed
            self._diffusion.set_crossfeed_config(
                enabled=cf.enabled,
                power=cf.power,
                range_=cf.range,
                decay=cf.decay,
            )

        logger.info(f"Mapping config updated: {len(mapping_config.mappings)} mappings, triggers={mapping_config.triggers}")

    def generate(self, metrics: AudioMetrics) -> GeneratedFrame:
        """Generate a frame based on current audio metrics.

        Thread Safety: This method acquires _state_lock during critical sections
        to prevent race conditions when config updates trigger cleanup/reinit
        while generation is in progress in another thread.
        """
        # Acquire lock for initialization check - generation lock acquired later
        with self._state_lock:
            if not self._initialized:
                self.initialize()
            generation_epoch = self._generation_epoch

        start_time = time.perf_counter()

        # Send audio energy to procedural pose generator for reactive movement
        if self._diffusion:
            # Use bass + mid energy for body movement
            audio_energy = (metrics.bass + metrics.mid * 0.5) / 1.5
            self._diffusion.set_pose_audio_energy(audio_energy)

        # Get dynamic prompt from story controller if a story is active
        effective_config = self.config
        if self._story_controller.has_story:
            story_output = self._story_controller.get_prompt(metrics)
            if story_output.base_prompt:
                # Create a modified config with story-driven prompt
                effective_config = GenerationConfig(
                    **{
                        **self.config.model_dump(),
                        "base_prompt": story_output.base_prompt,
                        "negative_prompt": story_output.negative_prompt or self.config.negative_prompt,
                    }
                )
                logger.debug(
                    f"Story prompt: scene={story_output.scene_id}, "
                    f"transition={story_output.is_transitioning}, "
                    f"energy_blend={story_output.energy_blend_factor:.2f}"
                )

        # Experimental: Full lyric-driven prompt generation
        lyric_driven = getattr(self.config, 'lyric_driven_mode', False)
        if lyric_driven and metrics.lyrics and metrics.lyrics.is_singing:
            lyric_prompt = self._keyword_extractor.text_to_prompt(metrics.lyrics.text)
            if lyric_prompt:
                effective_config = GenerationConfig(
                    **{
                        **effective_config.model_dump(),
                        "base_prompt": lyric_prompt,
                    }
                )
                logger.debug(f"Lyric-driven prompt: {lyric_prompt[:80]}...")

        # Map audio to generation parameters (uses effective_config which may have story prompt)
        params = self._mapper.map(metrics, effective_config)

        # Advance story state after getting prompt
        if self._story_controller.has_story:
            self._story_controller.advance(metrics)

        # Store current metrics for generators that need it
        self._current_audio_metrics = metrics

        # Generate based on mode - hold lock during generation to prevent
        # race conditions with cleanup/reinit from config updates
        with self._state_lock:
            mode = self.config.generation_mode

            if mode == GenerationMode.KEYFRAME_RIFE:
                # Keyframe + RIFE: generate keyframes, interpolate between
                image = self._generate_keyframe_rife(params, metrics)
                logger.info(f"Frame {self._frame_count + 1}: Keyframe+RIFE")

            else:
                # FEEDBACK mode (default): img2img loop with txt2img for first frame
                # Check for auto-reset
                auto_reset = getattr(self.config, 'auto_reset_frames', 0)
                if auto_reset > 0 and self._frame_count > 0 and self._frame_count % auto_reset == 0:
                    logger.info(f"Auto-reset at frame {self._frame_count}")
                    # Use base image if available, otherwise fall back to txt2img
                    if self._base_image is not None:
                        self._last_frame = self._base_image.copy()
                        logger.info("Auto-reset: restored base image")
                    else:
                        self._last_frame = None

                # Check for periodic txt2img (beat-triggered fresh pose)
                if params.force_txt2img:
                    logger.info("Force txt2img triggered (periodic pose refresh) - clearing anchor")
                    self._last_frame = None
                    # In anchored mode, also clear anchor to get fresh reference
                    if self._use_anchored_mode:
                        self._anchor_frame = None
                        logger.info("Anchor frame cleared - next frame will be fresh txt2img")

                # Get input frame for conditioning
                if self._use_anchored_mode:
                    # Anchored mode: always use the anchor frame (first txt2img result)
                    # This avoids feedback drift while still providing visual context
                    input_frame = self._anchor_frame
                else:
                    # Standard feedback mode: use last frame
                    input_frame = self.get_input_frame_for_generation()

                # Feedback with conditioning, or txt2img if no reference frame
                if input_frame is None:
                    logger.info(f"Frame {self._frame_count + 1}: txt2img (no anchor/input frame), seed={params.seed}")
                    image = self._generate_txt2img(params)
                    # In anchored mode, store this as the new anchor
                    if self._use_anchored_mode:
                        self._anchor_frame = image.copy()
                        logger.info("Anchored mode: stored new anchor frame")
                else:
                    # Clamp strength for logging (actual clamping happens in _generate_feedback)
                    clamped_strength = max(0.26, min(0.95, params.strength))
                    if self._use_anchored_mode:
                        logger.info(f"Frame {self._frame_count + 1}: img2img (anchored), seed={params.seed}")
                    else:
                        logger.info(f"Frame {self._frame_count + 1}: img2img strength={clamped_strength:.3f}, seed={params.seed}")
                    image = self._generate_feedback(params, input_frame)

        # Encode to JPEG
        jpeg_bytes = self._encoder.encode(image)

        # Track timing
        generation_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_fps(generation_time_ms)

        # Check if reset happened during generation
        if self._generation_epoch != generation_epoch:
            logger.warning(f"Reset detected during generation (epoch {generation_epoch} -> {self._generation_epoch}), discarding frame")
            # Don't update _last_frame - let the next frame be txt2img
            # CRITICAL: Also re-clear latent history since this frame may have contaminated it
            if self._diffusion:
                self._diffusion.clear_latent_history()
            # Return the frame anyway so client gets something, but don't save state
            return GeneratedFrame(
                image=image,
                jpeg_bytes=jpeg_bytes,
                frame_id=0,  # Invalid frame ID to indicate discarded
                timestamp=time.time(),
                params=params,
                generation_time_ms=generation_time_ms,
            )

        # Store for feedback loop
        self._last_frame = image
        self._frame_count += 1

        return GeneratedFrame(
            image=image,
            jpeg_bytes=jpeg_bytes,
            frame_id=self._frame_count,
            timestamp=time.time(),
            params=params,
            generation_time_ms=generation_time_ms,
        )

    def _generate_txt2img(self, params: GenerationParams) -> Image.Image:
        """Generate image from text only.

        Uses either the injected ImageGenerator or the StreamDiffusionWrapper.
        """
        # Use injected generator if available and no wrapper
        if self._injected_generator is not None and self._diffusion is None:
            request = GenerationRequest(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                guidance_scale=params.guidance_scale,
                seed=params.seed,
                input_image=None,  # txt2img mode
            )
            result = self._injected_generator.generate(request)
            return result.image

        # Default: use StreamDiffusionWrapper
        return self._diffusion.generate_txt2img(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            guidance_scale=params.guidance_scale,
            seed=params.seed,
        )

    def _generate_feedback(self, params: GenerationParams, input_frame: Optional[Image.Image] = None) -> Image.Image:
        """Generate image using previous frame as input.

        Uses either the injected ImageGenerator or the StreamDiffusionWrapper.

        Args:
            params: Generation parameters
            input_frame: Optional explicit input frame (for preview mode).
                         If not provided, uses _last_frame.
        """
        frame = input_frame if input_frame is not None else self._last_frame
        if frame is None:
            # First frame - use txt2img
            return self._generate_txt2img(params)

        # Clamp strength to ensure at least 1 denoising step
        # With few-step inference (3-4 steps), strength < 0.26 results in 0 steps
        # Formula: min_strength = 1.0 / num_inference_steps + epsilon
        min_strength = 0.26  # Safe for 4-step inference
        strength = max(min_strength, min(0.95, params.strength))

        # Use injected generator if available and no wrapper
        if self._injected_generator is not None and self._diffusion is None:
            request = GenerationRequest(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                guidance_scale=params.guidance_scale,
                seed=params.seed,
                input_image=frame,
                strength=strength,  # Use clamped strength
                is_onset=params.is_onset,
                onset_confidence=params.onset_confidence,
                is_beat_seed_jump=params.is_beat_seed_jump,
            )
            result = self._injected_generator.generate(request)
            return result.image

        # Default: use StreamDiffusionWrapper
        return self._diffusion.generate_img2img(
            prompt=params.prompt,
            image=frame,
            strength=strength,  # Use clamped strength
            negative_prompt=params.negative_prompt,
            guidance_scale=params.guidance_scale,
            seed=params.seed,
        )

    def _generate_keyframe_rife(self, params: GenerationParams, metrics: AudioMetrics) -> Image.Image:
        """Generate using Keyframe + RIFE interpolation hybrid.

        Generates pose-guided keyframes with ControlNet, interpolates between them.
        """
        if self._keyframe_pipeline:
            audio_energy = (metrics.bass + metrics.mid * 0.5) / 1.5

            # Send audio energy to procedural pose generator
            if self._diffusion:
                self._diffusion.set_pose_audio_energy(audio_energy)

            frame = self._keyframe_pipeline.generate_frame(
                generation_params={
                    "prompt": params.prompt,
                    "negative_prompt": params.negative_prompt,
                    "guidance_scale": params.guidance_scale,
                    "seed": params.seed,
                },
                audio_energy=audio_energy,
                is_beat=metrics.is_beat,
            )

            # Store keyframe for next iteration's style reference
            # Only store actual keyframes, not interpolated frames
            if self._keyframe_pipeline._frame_count % self._keyframe_pipeline.config.keyframe_interval == 0:
                self._last_frame = frame

            return frame

        # Fallback to txt2img
        return self._generate_txt2img(params)

    def _generate_keyframe_with_pose(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a pose-guided keyframe using ControlNet.

        This generates a new procedural pose and uses img2img with ControlNet
        to create a keyframe that follows the animated pose.
        """
        if not self._diffusion:
            return Image.new('RGB', (self.config.width, self.config.height), color='black')

        # Get keyframe strength from config (default 0.6 for pose-guided keyframes)
        keyframe_strength = getattr(self.config, 'keyframe_strength', 0.6)

        # Use the last frame if available for style consistency
        if self._last_frame is not None:
            # img2img with ControlNet - pose changes while style preserved
            return self._diffusion.generate_img2img(
                prompt=prompt,
                image=self._last_frame,
                strength=keyframe_strength,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        else:
            # First frame - use txt2img
            return self._diffusion.generate_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                seed=seed,
            )

    def set_source_image(self, image: Image.Image) -> None:
        """Set external source image for img2img mode.

        This image persists through resets - on reset, the pipeline will
        return to this base image instead of falling back to txt2img.

        The image is scaled to fit within the target dimensions while preserving
        aspect ratio, then centered on a black background.
        """
        target_w, target_h = self.config.width, self.config.height
        src_w, src_h = image.size

        if (src_w, src_h) != (target_w, target_h):
            # Calculate scale to fit within target while preserving aspect ratio
            scale = min(target_w / src_w, target_h / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)

            # Resize with high quality
            resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Create black background and paste centered
            padded = Image.new('RGB', (target_w, target_h), (0, 0, 0))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            padded.paste(resized, (paste_x, paste_y))
            image = padded

            logger.info(f"Base image scaled from {src_w}x{src_h} to {new_w}x{new_h}, centered in {target_w}x{target_h}")

        self._base_image = image  # Store persistently
        self._last_frame = image  # Also set as current frame
        logger.info(f"Base image set and will persist through resets: {image.size}")

    def clear_base_image(self) -> None:
        """Clear the persistent base image."""
        self._base_image = None
        logger.info("Base image cleared")

    def reset(self) -> None:
        """Reset the pipeline state (clear last frame, counters).

        If a base image was set via set_source_image(), the pipeline will
        reset to that image. Otherwise, it falls back to txt2img.
        """
        logger.info("Pipeline reset - clearing feedback loop")
        self._generation_epoch += 1  # Invalidate any in-flight generations

        # Reset to base image if one was set, otherwise clear for txt2img
        if self._base_image is not None:
            self._last_frame = self._base_image.copy()
            logger.info("Restored base image as starting point")
        else:
            self._last_frame = None

        # Clear anchor frame for FLUX anchored mode
        self._anchor_frame = None

        self._frame_count = 0
        self._fps_samples.clear()

        # Clear latent history in diffusion wrapper
        if self._diffusion:
            self._diffusion.clear_latent_history()

        # Clear keyframe pipeline
        if self._keyframe_pipeline:
            self._keyframe_pipeline.reset()

        # Reset to a new random seed for fresh starting point
        self._mapper.reset_seed()

        if self._base_image is not None:
            logger.info(f"Reset complete (epoch={self._generation_epoch}) - next frame will use base image")
        else:
            logger.info(f"Reset complete (epoch={self._generation_epoch}) - next frame will be txt2img")

    def set_anchored_mode(self, enabled: bool) -> None:
        """Enable/disable anchored mode for FLUX backends.

        In anchored mode, the first txt2img result is used as a stable reference
        for all subsequent generations. This avoids the feedback drift that occurs
        with Klein's conditioning when using the previous frame.

        Args:
            enabled: True to enable anchored mode, False for standard feedback loop
        """
        if enabled != self._use_anchored_mode:
            self._use_anchored_mode = enabled
            if enabled:
                logger.info("Anchored mode enabled - will use first frame as stable reference")
            else:
                logger.info("Anchored mode disabled - using standard feedback loop")
                # Clear anchor so we start fresh with feedback
                self._anchor_frame = None

    def refresh_anchor(self) -> None:
        """Clear the anchor frame to trigger a fresh txt2img on next generation.

        Useful for manually refreshing the visual style without a full reset.
        """
        if self._use_anchored_mode:
            self._anchor_frame = None
            logger.info("Anchor frame cleared - next generation will be txt2img")

    def get_input_frame_for_generation(self) -> Optional[Image.Image]:
        """Get the input frame for img2img generation.

        Priority:
        1. Lock to base image mode → always use base image (no feedback)
        2. Default → last generated frame (feedback loop)
        """
        if getattr(self.config, 'lock_to_base_image', False) and self._base_image is not None:
            return self._base_image
        return self._last_frame

    def get_fps(self) -> float:
        """Get current generation FPS."""
        if not self._fps_samples:
            return 0.0
        return sum(self._fps_samples) / len(self._fps_samples)

    def get_pose_preview(self) -> Optional[bytes]:
        """Get the current pose image as JPEG bytes for preview."""
        if self._diffusion:
            pose_image = self._diffusion.get_current_pose_image()
            if pose_image:
                return self._encoder.encode(pose_image)
        return None

    def _update_fps(self, generation_time_ms: float) -> None:
        """Update FPS tracking."""
        fps = 1000.0 / generation_time_ms if generation_time_ms > 0 else 0
        self._fps_samples.append(fps)
        # Keep last 30 samples for smoothing
        if len(self._fps_samples) > 30:
            self._fps_samples.pop(0)

    # Story control methods
    def load_story(self, story: StoryScript) -> None:
        """Load a story script for dynamic prompt generation."""
        self._story_controller.load_story(story)
        logger.info(f"Story loaded: {story.name} with {len(story.scenes)} scenes")

    def unload_story(self) -> None:
        """Unload the current story, returning to static prompts."""
        self._story_controller.unload_story()
        logger.info("Story unloaded")

    def story_play(self) -> None:
        """Resume story playback."""
        self._story_controller.play()

    def story_pause(self) -> None:
        """Pause story playback."""
        self._story_controller.pause()

    def story_skip_next(self) -> None:
        """Skip to next scene."""
        self._story_controller.skip_to_next_scene()

    def story_skip_prev(self) -> None:
        """Skip to previous scene."""
        self._story_controller.skip_to_prev_scene()

    def story_restart(self) -> None:
        """Restart story from beginning."""
        self._story_controller.reset()

    def get_story_state(self) -> Optional[StoryState]:
        """Get current story state for UI updates."""
        if not self._story_controller.has_story:
            return None
        return self._story_controller.get_state()

    def has_story(self) -> bool:
        """Check if a story is currently loaded."""
        return self._story_controller.has_story

    def cleanup(self) -> None:
        """Release resources."""
        # Log VRAM before cleanup
        vram_before = None
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GenerationPipeline cleanup starting - VRAM used: {vram_before:.2f} GB")

        if self._diffusion:
            self._diffusion.cleanup()
            self._diffusion = None
        if self._injected_generator is not None:
            self._injected_generator.cleanup()
            self._injected_generator = None
        if self._keyframe_pipeline:
            self._keyframe_pipeline.cleanup()
            self._keyframe_pipeline = None

        # Clear image references
        self._last_frame = None
        self._base_image = None
        self._anchor_frame = None
        self._initialized = False

        # Final CUDA cleanup pass to catch any lingering references
        # gc first to release Python references, then synchronize and clear cache
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1024**3
            freed = vram_before - vram_after if vram_before else 0
            logger.info(f"GenerationPipeline cleanup complete - VRAM used: {vram_after:.2f} GB (freed {freed:.2f} GB)")
