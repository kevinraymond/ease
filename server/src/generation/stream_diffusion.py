"""StreamDiffusion wrapper for real-time image generation with SOTA optimizations."""

import gc
import torch
from PIL import Image
from typing import Optional, Callable
import logging
import numpy as np

from ..config import settings
from .procedural_pose import ProceduralPoseGenerator, PoseAnimationMode, PoseFraming, BodyPart

logger = logging.getLogger(__name__)


class TAESDDecoder:
    """Tiny AutoEncoder for Stable Diffusion - 100x faster decode."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16, is_sdxl: bool = False):
        self.device = device
        self.dtype = dtype
        self._decoder = None
        self._is_sdxl = is_sdxl
        self._initialized = False

    def initialize(self) -> None:
        """Load TAESD decoder."""
        if self._initialized:
            return

        try:
            from diffusers import AutoencoderTiny

            model_id = settings.taesd_sdxl_model if self._is_sdxl else settings.taesd_model
            logger.info(f"Loading TAESD decoder: {model_id}")

            self._decoder = AutoencoderTiny.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
            ).to(self.device)

            self._decoder.eval()
            self._initialized = True
            logger.info("TAESD decoder initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to load TAESD: {e}")
            raise

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using TAESD (100x faster than full VAE)."""
        if not self._initialized:
            self.initialize()

        with torch.no_grad():
            # TAESD expects latents scaled differently
            # Standard VAE uses 0.18215, TAESD uses 1.0
            scaled_latents = latents / 0.18215
            images = self._decoder.decode(scaled_latents).sample
            # Denormalize from [-1, 1] to [0, 1]
            images = (images + 1) / 2
            images = images.clamp(0, 1)
            return images

    def cleanup(self) -> None:
        """Release resources."""
        if self._decoder is not None:
            del self._decoder
            self._decoder = None
        self._initialized = False


class StreamDiffusionWrapper:
    """Wrapper for StreamDiffusion pipeline with SOTA optimizations."""

    def __init__(
        self,
        model_id: str = None,
        width: int = 512,
        height: int = 512,
        device: str = None,
        # SOTA settings (override global config)
        use_controlnet: bool = None,
        controlnet_weight: float = None,
        use_taesd: bool = None,
        temporal_coherence: str = None,
        # Acceleration settings (override global config)
        acceleration: str = None,
        hyper_sd_steps: int = None,
    ):
        self.model_id = model_id or settings.model
        self.width = width
        self.height = height
        self.device = device or settings.device
        self.dtype = getattr(torch, settings.dtype)

        # Acceleration method (from param or global settings)
        self._acceleration = acceleration if acceleration is not None else settings.acceleration
        self._hyper_sd_steps = hyper_sd_steps if hyper_sd_steps is not None else settings.hyper_sd_steps

        self._pipe = None
        self._txt2img_pipe = None
        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._controlnet_txt2img_pipe = None  # txt2img with ControlNet for procedural poses
        self._controlnet = None
        self._pose_detector = None
        self._stream = None
        self._last_stream_seed: Optional[int] = None  # Track seed for StreamDiffusion noise updates
        self._initialized = False

        # ControlNet for identity preservation (from param or global settings)
        self._use_controlnet = use_controlnet if use_controlnet is not None else settings.use_controlnet
        self._controlnet_weight = controlnet_weight if controlnet_weight is not None else settings.controlnet_pose_weight
        logger.info(f"StreamDiffusionWrapper init: controlnet_weight={self._controlnet_weight}")

        # TAESD for fast decoding
        self._taesd: Optional[TAESDDecoder] = None
        self._use_taesd = use_taesd if use_taesd is not None else settings.use_taesd

        # StreamDiffusion optimization - enabled by default for stream_diffusion backend
        # Auto-disable when incompatible features are enabled:
        # - ControlNet requires the full diffusers pipeline for pose conditioning
        # - Hyper-SD uses TCDScheduler which StreamDiffusion doesn't support
        self._use_streamdiffusion = not self._use_controlnet
        if self._acceleration == "hyper-sd" and self._use_streamdiffusion:
            logger.info("Hyper-SD requires diffusers path (StreamDiffusion doesn't support TCDScheduler)")
            self._use_streamdiffusion = False
        if self._use_controlnet:
            logger.info("ControlNet enabled - disabling StreamDiffusion (incompatible), using diffusers path")

        # Similarity filter state
        self._last_latents: Optional[torch.Tensor] = None
        self._similarity_filter = settings.similarity_filter

        # Latent blending state for temporal coherence
        self._latent_history: list[torch.Tensor] = []
        # Enable latent blending based on temporal_coherence setting
        self._temporal_coherence = temporal_coherence or "blending"
        self._use_latent_blending = self._temporal_coherence == "blending" or settings.latent_blending

        # ControlNet pose caching
        self._cached_pose: Optional[Image.Image] = None
        self._pose_extracted: bool = False  # Only extract once per reset cycle (when locked)
        self._controlnet_frame_count: int = 0
        # NOTE: Skip frames was causing alternating output (pose/no-pose bouncing)
        self._controlnet_skip_frames: int = 1  # Use ControlNet every frame (was 2, caused bouncing)
        self._generation_epoch: int = 0  # Increments on reset to invalidate in-flight operations
        self._frames_since_reset: int = 0  # Track frames since last reset
        self._pose_extraction_delay: int = 5  # Wait this many frames before extracting pose
        self._pose_lock: bool = True  # True = extract once (locked), False = re-extract periodically (drift)
        self._pose_drift_interval: int = 10  # Inject fresh pose every N frames when drifting
        self._inject_txt2img_for_pose: bool = False  # Flag to inject txt2img on next frame

        # Procedural pose generation (alternative to extracting from images)
        self._procedural_pose_generator: Optional[ProceduralPoseGenerator] = None
        self._use_procedural_pose: bool = False  # True = use procedural, False = extract from image

        # Procedural pose txt2img settings
        self._procedural_use_txt2img: bool = True  # Use txt2img + ControlNet (vs img2img)
        self._procedural_fixed_seed: Optional[int] = None  # Fixed seed for character consistency (None = use 42)
        self._procedural_blend_weight: float = 0.4  # Latent blend weight for frame smoothness

        # Custom LoRA tracking (for hot-swap support)
        self._loaded_loras: dict[str, float] = {}  # name -> weight
        self._lora_dir: str = settings.lora_dir

        # NSFW filtering with fallback to previous frame
        self._nsfw_filter_enabled: bool = settings.nsfw_filter
        self._safety_checker = None
        self._feature_extractor = None
        self._last_safe_frame: Optional[Image.Image] = None
        self._nsfw_frame_count: int = 0  # Track consecutive NSFW frames for logging

    def _get_acceleration_steps(self, for_img2img: bool = False, strength: float = 0.5) -> int:
        """Get the number of inference steps based on acceleration method.

        Args:
            for_img2img: If True, ensures enough steps for img2img after strength scaling
            strength: The img2img strength (used to calculate minimum steps)
        """
        if self._acceleration == "hyper-sd":
            base_steps = self._hyper_sd_steps
            if for_img2img and strength > 0:
                # img2img uses int(steps * strength) effective steps
                # We need at least 1 effective step, so: steps * strength >= 1
                # Therefore: steps >= 1 / strength
                min_steps = max(base_steps, int(1.0 / strength) + 1)
                if min_steps > base_steps:
                    logger.debug(f"Hyper-SD: bumping steps from {base_steps} to {min_steps} for img2img (strength={strength:.2f})")
                return min_steps
            return base_steps
        elif self._acceleration == "lcm":
            return 4  # LCM standard
        else:
            return 20  # Standard diffusion

    def _get_acceleration_guidance(self) -> float:
        """Get the guidance scale based on acceleration method."""
        if self._acceleration == "hyper-sd":
            return 1.0  # Hyper-SD works with low CFG
        elif self._acceleration == "lcm":
            return 1.5  # LCM standard
        else:
            return 7.5  # Standard diffusion

    def _get_hyper_sd_eta(self) -> float:
        """Get the optimal eta for Hyper-SD based on step count.

        Eta controls stochasticity - higher = more random, lower = more deterministic.
        Recommended values from ByteDance:
        - 1-step: eta=1.0 (full stochastic)
        - 2-step: eta=0.5
        - 4-step: eta=0.3
        - 8-step: eta=0.3
        """
        eta_map = {
            1: 1.0,
            2: 0.5,
            4: 0.3,
            8: 0.3,
        }
        return eta_map.get(self._hyper_sd_steps, 0.3)

    def _validate_img2img_strength(self, strength: float, num_steps: int) -> float:
        """Ensure strength results in at least 1 effective step.

        With low step counts (Hyper-SD 1-8 steps) and low strength values,
        the effective denoising steps can become 0, causing empty tensor errors.
        This method ensures at least 1 effective step.
        """
        if num_steps <= 0:
            num_steps = 1
        effective_steps = int(num_steps * strength)
        if effective_steps < 1:
            min_strength = (1.0 / num_steps) + 0.01
            logger.warning(f"Strength {strength:.3f} too low for {num_steps} steps (effective={effective_steps}), bumping to {min_strength:.3f}")
            return min_strength
        return strength

    def initialize(self) -> None:
        """Initialize the generation pipeline."""
        if self._initialized:
            return

        logger.info(f"Initializing pipeline with model: {self.model_id}")
        logger.info(f"  StreamDiffusion: {self._use_streamdiffusion}")
        logger.info(f"  ControlNet: {self._use_controlnet}")
        logger.info(f"  TAESD: {self._use_taesd}")
        logger.info(f"  Temporal coherence: {self._temporal_coherence}")
        logger.info(f"  Latent blending: {self._use_latent_blending}")

        # Initialize TAESD if enabled
        if self._use_taesd:
            try:
                is_sdxl = "xl" in self.model_id.lower() or "sdxl" in self.model_id.lower()
                self._taesd = TAESDDecoder(
                    device=self.device,
                    dtype=self.dtype,
                    is_sdxl=is_sdxl,
                )
                self._taesd.initialize()
            except Exception as e:
                logger.warning(f"TAESD init failed, using standard VAE: {e}")
                self._taesd = None
                self._use_taesd = False

        if self._use_streamdiffusion:
            try:
                self._init_streamdiffusion()
                # StreamDiffusion wrapper is img2img only, so also init diffusers txt2img
                self._init_txt2img_only()
                # Initialize NSFW filter if enabled
                if self._nsfw_filter_enabled:
                    self._init_nsfw_filter()
                self._initialized = True
                logger.info("StreamDiffusion (img2img) + diffusers (txt2img) initialized")
                return
            except Exception as e:
                logger.warning(f"StreamDiffusion init failed, falling back to diffusers: {e}")
                self._use_streamdiffusion = False

        # Fallback to standard diffusers (both txt2img and img2img)
        self._init_diffusers()

        # Initialize NSFW filter if enabled
        if self._nsfw_filter_enabled:
            self._init_nsfw_filter()

        self._initialized = True
        logger.info("Diffusers pipeline initialized successfully")

    def _init_streamdiffusion(self) -> None:
        """Initialize StreamDiffusion pipeline with optimizations using StreamDiffusionWrapper."""
        try:
            from streamdiffusion.wrapper import StreamDiffusionWrapper

            logger.info(f"Loading StreamDiffusionWrapper for img2img: {self.model_id}")

            # Use StreamDiffusionWrapper with mode="img2img" for proper img2img support
            # t_index_list=[35, 45] is optimized for img2img (preserves input structure)
            model_lower = self.model_id.lower()
            use_lcm = "lcm" not in model_lower and "turbo" not in model_lower

            self._stream = StreamDiffusionWrapper(
                model_id_or_path=self.model_id,
                t_index_list=[35],  # Balance between stability and reactivity
                mode="img2img",
                width=self.width,
                height=self.height,
                frame_buffer_size=1,
                use_denoising_batch=settings.stream_batch,
                use_lcm_lora=use_lcm,
                use_tiny_vae=False,  # TAESD has scaling issues
                cfg_type="none",  # No classifier-free guidance for speed
                acceleration="none",  # Skip TensorRT (not installed)
                device=self.device,
                dtype=self.dtype,
            )

            # Prepare with empty prompt (will be updated per-frame)
            self._stream.prepare(
                prompt="",
                negative_prompt="",
                num_inference_steps=50,
                guidance_scale=1.2,
            )

            logger.info("StreamDiffusionWrapper initialized successfully (img2img mode)")

        except ImportError as e:
            logger.warning(f"StreamDiffusionWrapper not available: {e}")
            raise ImportError("StreamDiffusion not installed or wrapper not found")

    def _init_txt2img_only(self) -> None:
        """Initialize only txt2img pipeline (used alongside StreamDiffusion for img2img)."""
        from diffusers import AutoPipelineForText2Image, LCMScheduler
        import torch

        logger.info(f"Loading txt2img pipeline for first-frame generation...")

        # Try loading with fp16 variant first, fall back to default if not available
        try:
            self._txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
                safety_checker=None,
                requires_safety_checker=False,
                device_map=None,
                low_cpu_mem_usage=False,
            ).to(self.device)
        except ValueError as e:
            if "variant" in str(e):
                logger.warning(f"Model {self.model_id} has no fp16 variant, loading default weights")
                self._txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    device_map=None,
                    low_cpu_mem_usage=False,
                ).to(self.device)
            else:
                raise

        self._txt2img_pipe.enable_attention_slicing()

        # Load acceleration LoRA based on settings
        model_lower = self.model_id.lower()
        if "lcm" not in model_lower and "turbo" not in model_lower:
            if self._acceleration == "hyper-sd":
                try:
                    from diffusers import TCDScheduler

                    lora_map = {
                        1: "Hyper-SD15-1step-lora.safetensors",
                        2: "Hyper-SD15-2steps-lora.safetensors",
                        4: "Hyper-SD15-4steps-lora.safetensors",
                        8: "Hyper-SD15-8steps-lora.safetensors",
                    }
                    lora_file = lora_map[self._hyper_sd_steps]

                    self._txt2img_pipe.load_lora_weights(
                        "ByteDance/Hyper-SD",
                        weight_name=lora_file
                    )
                    self._txt2img_pipe.fuse_lora(lora_scale=settings.hyper_sd_lora_scale)
                    self._txt2img_pipe.scheduler = TCDScheduler.from_config(
                        self._txt2img_pipe.scheduler.config
                    )
                    logger.info(f"Hyper-SD LoRA ({self._hyper_sd_steps}-step) loaded for txt2img")
                except Exception as e:
                    logger.warning(f"Could not load Hyper-SD LoRA: {e}")
            elif self._acceleration == "lcm":
                try:
                    self._txt2img_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                    self._txt2img_pipe.fuse_lora()
                    self._txt2img_pipe.scheduler = LCMScheduler.from_config(
                        self._txt2img_pipe.scheduler.config
                    )
                    logger.info("LCM-LoRA loaded for txt2img")
                except Exception as e:
                    logger.warning(f"Could not load LCM-LoRA: {e}")
            else:
                logger.info("No acceleration LoRA for txt2img (using standard scheduler)")

        # Also create img2img pipeline for fallback when variable strength is needed
        # (StreamDiffusion has fixed denoising level)
        # Must be created AFTER LoRA loading so it shares the fused weights
        from diffusers import AutoPipelineForImage2Image
        self._img2img_pipe = AutoPipelineForImage2Image.from_pipe(self._txt2img_pipe)
        self._img2img_pipe.safety_checker = None
        self._img2img_pipe.enable_attention_slicing()
        logger.info("txt2img + img2img fallback pipelines ready")

    def _init_diffusers(self) -> None:
        """Initialize standard diffusers pipeline."""
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
        import torch

        # Load both txt2img and img2img pipelines
        # IMPORTANT: For multi-GPU support, we must avoid meta tensors from accelerate
        # Load to CPU with device_map=None, then manually move to target device
        logger.info(f"Loading diffusers pipeline to {self.device}...")

        # Load to CPU first (no device_map to avoid meta tensors)
        # Try loading with fp16 variant first, fall back to default if not available
        try:
            self._txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
                safety_checker=None,
                requires_safety_checker=False,
                device_map=None,  # Explicitly disable device_map to avoid meta tensors
                low_cpu_mem_usage=False,  # Load full tensors, not meta
            )
        except ValueError as e:
            if "variant" in str(e):
                logger.warning(f"Model {self.model_id} has no fp16 variant, loading default weights")
                self._txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    device_map=None,
                    low_cpu_mem_usage=False,
                )
            else:
                raise

        # Now move to target device
        self._txt2img_pipe = self._txt2img_pipe.to(self.device)

        # img2img shares components to save VRAM
        self._img2img_pipe = AutoPipelineForImage2Image.from_pipe(self._txt2img_pipe)
        self._img2img_pipe.safety_checker = None

        self._pipe = self._txt2img_pipe

        # Enable memory optimizations
        self._txt2img_pipe.enable_attention_slicing()
        self._img2img_pipe.enable_attention_slicing()

        # Initialize ControlNet if enabled
        if self._use_controlnet:
            self._init_controlnet()

        # Load acceleration LoRA (LCM or Hyper-SD) for fast generation
        self._load_acceleration_lora()

        if settings.compile_unet and hasattr(torch, "compile"):
            logger.warning("=" * 60)
            logger.warning("TORCH.COMPILE ENABLED - First generation will be SLOW")
            logger.warning("Compilation happens on first inference (~30-60 seconds)")
            logger.warning("Subsequent generations will be faster")
            logger.warning("=" * 60)
            logger.info("Compiling UNet with torch.compile...")
            self._txt2img_pipe.unet = torch.compile(
                self._txt2img_pipe.unet,
                mode="reduce-overhead",
                fullgraph=True,
            )

    def _init_nsfw_filter(self) -> None:
        """Initialize safety checker for NSFW filtering."""
        try:
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            from transformers import CLIPImageProcessor

            logger.info("Loading safety checker for NSFW filtering...")
            self._safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=self.dtype,
            ).to(self.device)
            self._feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            logger.info("NSFW filter initialized - will use previous frame fallback")
        except Exception as e:
            logger.warning(f"Failed to load safety checker: {e}")
            self._nsfw_filter_enabled = False

    def _check_nsfw_and_fallback(self, image: Image.Image) -> Image.Image:
        """Check if image is NSFW and return previous safe frame if so.

        Args:
            image: The generated image to check

        Returns:
            The original image if safe, or the last safe frame if NSFW
        """
        if not self._nsfw_filter_enabled or self._safety_checker is None:
            # Update last safe frame even when filter disabled (for later enabling)
            self._last_safe_frame = image
            return image

        try:
            # Run safety checker
            safety_input = self._feature_extractor(image, return_tensors="pt").to(self.device)

            # Convert image to numpy for safety checker
            import numpy as np
            image_np = np.array(image)

            # Safety checker returns (images, has_nsfw_concept)
            _, has_nsfw = self._safety_checker(
                images=[image_np],
                clip_input=safety_input.pixel_values,
            )

            if has_nsfw[0]:
                self._nsfw_frame_count += 1
                if self._nsfw_frame_count == 1 or self._nsfw_frame_count % 30 == 0:
                    logger.info(f"NSFW content detected (count={self._nsfw_frame_count}), using previous safe frame")

                if self._last_safe_frame is not None:
                    return self._last_safe_frame
                else:
                    # No previous frame yet, return a blank or the image anyway
                    logger.warning("No previous safe frame available, returning current frame")
                    return image
            else:
                # Safe frame - update cache and reset counter
                self._last_safe_frame = image
                if self._nsfw_frame_count > 0:
                    logger.info(f"Safe frame after {self._nsfw_frame_count} NSFW frames")
                self._nsfw_frame_count = 0
                return image

        except Exception as e:
            logger.warning(f"NSFW check failed: {e}, passing through")
            self._last_safe_frame = image
            return image

    def _init_controlnet(self) -> None:
        """Initialize ControlNet for pose preservation."""
        # Check model compatibility - ControlNet v1.1 requires SD 1.5, not SD-Turbo/SDXL
        model_lower = self.model_id.lower()
        if "turbo" in model_lower or "sdxl" in model_lower or "xl" in model_lower:
            logger.warning(f"ControlNet v1.1 is incompatible with {self.model_id} (requires SD 1.5)")
            logger.warning("Disabling ControlNet. Use runwayml/stable-diffusion-v1-5 or Lykon/dreamshaper-8 for ControlNet support.")
            logger.warning("Falling back to latent blending for temporal coherence.")
            self._use_controlnet = False
            # Re-enable latent blending as fallback for temporal coherence
            self._use_latent_blending = True
            return

        try:
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline
            from controlnet_aux import OpenposeDetector

            logger.info("Initializing ControlNet for pose preservation...")

            # Load OpenPose ControlNet
            self._controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose",
                torch_dtype=self.dtype,
            ).to(self.device)

            # Create ControlNet img2img pipeline (for pose extraction mode)
            self._controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline(
                vae=self._txt2img_pipe.vae,
                text_encoder=self._txt2img_pipe.text_encoder,
                tokenizer=self._txt2img_pipe.tokenizer,
                unet=self._txt2img_pipe.unet,
                controlnet=self._controlnet,
                scheduler=self._txt2img_pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(self.device)

            self._controlnet_pipe.enable_attention_slicing()

            # Create ControlNet txt2img pipeline (for procedural pose mode)
            # This generates fresh images guided purely by pose, without input image conflict
            self._controlnet_txt2img_pipe = StableDiffusionControlNetPipeline(
                vae=self._txt2img_pipe.vae,
                text_encoder=self._txt2img_pipe.text_encoder,
                tokenizer=self._txt2img_pipe.tokenizer,
                unet=self._txt2img_pipe.unet,
                controlnet=self._controlnet,
                scheduler=self._txt2img_pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(self.device)

            self._controlnet_txt2img_pipe.enable_attention_slicing()
            logger.info("ControlNet txt2img pipeline initialized for procedural poses")

            # Initialize pose detector
            self._pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            logger.info("ControlNet initialized successfully")

        except Exception as e:
            logger.warning(f"ControlNet initialization failed: {e}")
            self._use_controlnet = False
            self._controlnet = None
            self._controlnet_pipe = None
            self._pose_detector = None

    def _load_acceleration_lora(self) -> None:
        """Load acceleration LoRA (LCM or Hyper-SD) for fast generation."""
        if self._acceleration == "lcm":
            self._load_lcm_lora()
        elif self._acceleration == "hyper-sd":
            self._load_hyper_sd_lora()
        else:
            logger.info("No acceleration LoRA - using standard scheduler")

    def _load_lcm_lora(self) -> None:
        """Load LCM-LoRA for fast generation across ALL pipelines (consistent style)."""
        try:
            from diffusers import LCMScheduler

            logger.info("Loading LCM-LoRA for fast generation (all pipelines)...")

            # Load LCM-LoRA weights into the base txt2img pipeline
            # Since all pipelines share the same UNet, this affects everything
            self._txt2img_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            self._txt2img_pipe.fuse_lora()  # Fuse for faster inference
            logger.info("LCM-LoRA fused into shared UNet")

            # Switch to LCM scheduler for ALL pipelines
            lcm_scheduler_config = self._txt2img_pipe.scheduler.config

            self._txt2img_pipe.scheduler = LCMScheduler.from_config(lcm_scheduler_config)
            logger.info("  - txt2img: LCMScheduler applied")

            self._img2img_pipe.scheduler = LCMScheduler.from_config(lcm_scheduler_config)
            logger.info("  - img2img: LCMScheduler applied")

            if self._controlnet_pipe is not None:
                self._controlnet_pipe.scheduler = LCMScheduler.from_config(lcm_scheduler_config)
                logger.info("  - controlnet img2img: LCMScheduler applied")

            if self._controlnet_txt2img_pipe is not None:
                self._controlnet_txt2img_pipe.scheduler = LCMScheduler.from_config(lcm_scheduler_config)
                logger.info("  - controlnet txt2img: LCMScheduler applied")

            logger.info("LCM-LoRA loaded - all pipelines now use 4-6 steps with consistent style")

        except Exception as e:
            logger.warning(f"LCM-LoRA loading failed, using standard scheduler (slower): {e}")

    def _load_hyper_sd_lora(self) -> None:
        """Load Hyper-SD LoRA with TCDScheduler for fast 1-8 step generation.

        Hyper-SD (ByteDance) offers state-of-the-art quality at very low step counts.
        Uses TCDScheduler instead of LCMScheduler for optimal results.
        """
        try:
            from diffusers import TCDScheduler

            # Map step count to LoRA file
            lora_map = {
                1: "Hyper-SD15-1step-lora.safetensors",
                2: "Hyper-SD15-2steps-lora.safetensors",
                4: "Hyper-SD15-4steps-lora.safetensors",
                8: "Hyper-SD15-8steps-lora.safetensors",
            }
            lora_file = lora_map[self._hyper_sd_steps]

            logger.info(f"Loading Hyper-SD LoRA for {self._hyper_sd_steps}-step generation...")
            logger.info(f"  LoRA file: {lora_file}")
            logger.info(f"  LoRA scale: {settings.hyper_sd_lora_scale}")
            logger.info(f"  Eta: {self._get_hyper_sd_eta()}")
            logger.info(f"  Base model: {self.model_id}")

            # Load Hyper-SD LoRA from HuggingFace (this may download on first use)
            logger.info("Downloading/loading Hyper-SD LoRA weights...")
            self._txt2img_pipe.load_lora_weights(
                "ByteDance/Hyper-SD",
                weight_name=lora_file,
                adapter_name="hyper-sd"
            )
            logger.info("Hyper-SD LoRA weights loaded, now fusing...")

            # Fuse with recommended scale (0.125 is much lower than LCM's 1.0)
            self._txt2img_pipe.fuse_lora(lora_scale=settings.hyper_sd_lora_scale)
            logger.info("Hyper-SD LoRA fused into shared UNet")

            # Unload adapter after fusing to free memory
            self._txt2img_pipe.unload_lora_weights()
            logger.info("LoRA adapter unloaded (weights are fused)")

            # Apply TCDScheduler to all pipelines
            scheduler_config = self._txt2img_pipe.scheduler.config

            self._txt2img_pipe.scheduler = TCDScheduler.from_config(scheduler_config)
            logger.info("  - txt2img: TCDScheduler applied")

            self._img2img_pipe.scheduler = TCDScheduler.from_config(scheduler_config)
            logger.info("  - img2img: TCDScheduler applied")

            if self._controlnet_pipe is not None:
                self._controlnet_pipe.scheduler = TCDScheduler.from_config(scheduler_config)
                logger.info("  - controlnet img2img: TCDScheduler applied")

            if self._controlnet_txt2img_pipe is not None:
                self._controlnet_txt2img_pipe.scheduler = TCDScheduler.from_config(scheduler_config)
                logger.info("  - controlnet txt2img: TCDScheduler applied")

            logger.info(f"Hyper-SD loaded - all pipelines now use {self._hyper_sd_steps} step(s)")

            # Validate the model works with a quick test
            logger.info("Validating Hyper-SD setup with test generation...")
            try:
                import torch
                with torch.no_grad():
                    test_result = self._txt2img_pipe(
                        prompt="test",
                        num_inference_steps=self._hyper_sd_steps,
                        guidance_scale=0.0,
                        height=64,
                        width=64,
                        eta=self._get_hyper_sd_eta(),
                    )
                logger.info("Hyper-SD validation successful")
            except Exception as val_e:
                logger.error(f"Hyper-SD validation FAILED: {val_e}")
                raise RuntimeError(f"Hyper-SD model validation failed: {val_e}") from val_e

        except Exception as e:
            logger.error(f"Hyper-SD loading failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("Falling back to LCM...")
            # Reset the acceleration mode so helper methods use LCM values
            self._acceleration = "lcm"
            self._load_lcm_lora()

    def load_custom_loras(self, loras: list[dict]) -> None:
        """Load/update custom LoRAs without pipeline restart (hot-swap).

        LCM-LoRA stays fused (permanent, fast). Custom LoRAs are loaded as
        adapters that can be enabled/disabled via set_adapters().

        Args:
            loras: List of LoRA configs: [{path: str, weight: float, name: str (optional)}]
        """
        import os

        if not self._txt2img_pipe:
            logger.warning("Cannot load LoRAs: txt2img pipeline not initialized")
            return

        if not loras:
            # Clear all custom LoRAs
            if self._loaded_loras:
                try:
                    self._txt2img_pipe.unload_lora_weights()
                    self._loaded_loras.clear()
                    logger.info("Cleared all custom LoRAs")
                except Exception as e:
                    logger.warning(f"Failed to unload LoRAs: {e}")
            return

        # Build target state
        target_loras = {}
        for i, lora in enumerate(loras):
            path = lora.get("path", "")
            weight = lora.get("weight", 1.0)
            name = lora.get("name", f"lora_{i}")

            # Resolve relative paths
            if not os.path.isabs(path):
                path = os.path.join(self._lora_dir, path)

            if os.path.exists(path):
                target_loras[name] = {"path": path, "weight": weight}
            else:
                logger.warning(f"LoRA not found: {path}")

        if not target_loras:
            logger.info("No valid LoRAs to load")
            return

        # Load new LoRAs (ones not already loaded)
        for name, config in target_loras.items():
            if name not in self._loaded_loras:
                try:
                    self._txt2img_pipe.load_lora_weights(
                        config["path"],
                        adapter_name=name
                    )
                    logger.info(f"Loaded LoRA: {name} from {config['path']}")
                except Exception as e:
                    logger.warning(f"Failed to load LoRA {name}: {e}")
                    continue

        # Set active adapters with weights
        try:
            adapter_names = list(target_loras.keys())
            adapter_weights = [target_loras[n]["weight"] for n in adapter_names]
            self._txt2img_pipe.set_adapters(adapter_names, adapter_weights)
            logger.info(f"Active LoRAs: {dict(zip(adapter_names, adapter_weights))}")
        except Exception as e:
            logger.warning(f"Failed to set adapters: {e}")

        self._loaded_loras = {n: c["weight"] for n, c in target_loras.items()}

    def _extract_pose(self, image: Image.Image) -> Optional[Image.Image]:
        """Extract pose from image for ControlNet conditioning."""
        if self._pose_detector is None:
            return None

        try:
            pose_image = self._pose_detector(
                image,
                hand_and_face=False,  # Faster
                output_type="pil",
            )
            return pose_image
        except Exception as e:
            logger.warning(f"Pose extraction failed: {e}")
            return None

    def _should_skip_frame(self, latents: torch.Tensor) -> bool:
        """Check if frame can be skipped using stochastic similarity filter."""
        if not self._similarity_filter or self._last_latents is None:
            return False

        # Compute cosine similarity between latent tensors
        with torch.no_grad():
            flat_current = latents.flatten()
            flat_previous = self._last_latents.flatten()
            similarity = torch.nn.functional.cosine_similarity(
                flat_current.unsqueeze(0),
                flat_previous.unsqueeze(0),
            ).item()

        return similarity > settings.similarity_threshold

    def _apply_latent_blending(
        self,
        latents: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Apply latent blending for temporal coherence using lunarring approach."""
        # Defensive check: don't process empty tensors
        if latents.numel() == 0:
            logger.error(f"Empty latents tensor in blending: shape={latents.shape}")
            return latents

        if not self._use_latent_blending or len(self._latent_history) == 0:
            return latents

        # Only blend during initial steps (higher noise levels)
        blend_range = int(total_steps * settings.crossfeed_range)
        if step > blend_range:
            return latents

        # Calculate blend weight based on step
        progress = step / blend_range
        decay = settings.crossfeed_decay ** progress
        blend_weight = settings.crossfeed_power * decay

        # Get previous latent
        prev_latent = self._latent_history[-1]

        # Blend current with previous
        blended = (1 - blend_weight) * latents + blend_weight * prev_latent

        return blended

    def _decode_latents(self, latents: torch.Tensor, use_taesd: bool = None) -> Image.Image:
        """Decode latents to PIL Image, optionally using TAESD for speed."""
        use_taesd = use_taesd if use_taesd is not None else self._use_taesd

        if use_taesd and self._taesd is not None:
            # Fast TAESD decode
            with torch.no_grad():
                images = self._taesd.decode(latents)
                # Convert to PIL
                image_np = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                return Image.fromarray(image_np)
        else:
            # Standard VAE decode (higher quality but slower)
            if self._use_streamdiffusion and self._stream:
                from streamdiffusion.image_utils import postprocess_image
                return postprocess_image(self._stream.decode(latents))
            elif self._txt2img_pipe:
                with torch.no_grad():
                    images = self._txt2img_pipe.vae.decode(
                        latents / self._txt2img_pipe.vae.config.scaling_factor
                    ).sample
                    images = (images / 2 + 0.5).clamp(0, 1)
                    image_np = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    return Image.fromarray(image_np)

        raise RuntimeError("No decoder available")

    def generate_txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 0.0,
        num_inference_steps: int = None,
        seed: Optional[int] = None,
        use_taesd: bool = None,
        latent_callback: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    ) -> Image.Image:
        """Generate image from text prompt."""
        if not self._initialized:
            self.initialize()

        # Use acceleration-appropriate steps and guidance
        steps = self._get_acceleration_steps()
        guidance_scale = self._get_acceleration_guidance()
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # StreamDiffusion wrapper is configured for img2img mode, so use diffusers for txt2img
        # This generates the first frame, which then feeds into the img2img loop
        if False and self._use_streamdiffusion and self._stream:  # Disabled - wrapper is img2img only
            from streamdiffusion.image_utils import postprocess_image

            if seed is not None:
                torch.manual_seed(seed)
            self._stream.update_prompt(prompt)
            output = self._stream()

            # Store latent for blending
            if hasattr(self._stream, 'last_latent'):
                self._update_latent_history(self._stream.last_latent)

            # StreamDiffusion returns tensor, convert to PIL Image
            if isinstance(output, torch.Tensor):
                return postprocess_image(output, output_type="pil")[0]
            return output

        # Diffusers path - used for txt2img (first frame)
        def callback(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]

            # Apply latent blending
            if latent_callback:
                latents = latent_callback(latents, step, steps)
            elif self._use_latent_blending:
                latents = self._apply_latent_blending(latents, step, steps)

            callback_kwargs["latents"] = latents
            return callback_kwargs

        # Build pipeline kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
            "width": self.width,
            "height": self.height,
            "generator": generator,
            "callback_on_step_end": callback if self._use_latent_blending else None,
            "output_type": "latent" if (use_taesd or self._use_taesd) else "pil",
        }

        # Add eta for TCDScheduler (Hyper-SD)
        if self._acceleration == "hyper-sd":
            pipe_kwargs["eta"] = self._get_hyper_sd_eta()

        result = self._txt2img_pipe(**pipe_kwargs)

        if (use_taesd or self._use_taesd) and self._taesd:
            latents = result.images
            self._update_latent_history(latents)
            output_image = self._decode_latents(latents, use_taesd=True)
        else:
            output_image = result.images[0]

        # Apply NSFW filter with fallback to previous safe frame
        return self._check_nsfw_and_fallback(output_image)

    def generate_img2img(
        self,
        prompt: str,
        image: Image.Image,
        strength: float = 0.5,
        negative_prompt: str = "",
        guidance_scale: float = 0.0,
        num_inference_steps: int = None,
        seed: Optional[int] = None,
        use_taesd: bool = None,
        latent_callback: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
    ) -> Image.Image:
        """Generate image from prompt + source image."""
        import time
        t_start = time.perf_counter()

        if not self._initialized:
            self.initialize()

        steps = num_inference_steps or 20
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # Resize input if needed
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)

        # Use acceleration-appropriate guidance and steps
        # For img2img, we need to ensure enough steps after strength scaling
        effective_guidance = self._get_acceleration_guidance()
        steps = self._get_acceleration_steps(for_img2img=True, strength=strength)

        # Validate strength to ensure at least 1 effective step (prevents empty tensor errors)
        strength = self._validate_img2img_strength(strength, steps)

        # When ControlNet is enabled or high strength is requested, use diffusers path
        # StreamDiffusion has a fixed denoising level and doesn't support variable strength
        # For keyframe generation with strength > 0.5, use diffusers for better control
        use_stream = self._use_streamdiffusion and self._stream and not self._use_controlnet and strength <= 0.5
        logger.debug(f"img2img: streamdiffusion={self._use_streamdiffusion}, stream_obj={self._stream is not None}, controlnet={self._use_controlnet}, strength={strength:.2f}, using_stream={use_stream}")

        if use_stream:
            t_preprocess = time.perf_counter()
            if seed is not None:
                torch.manual_seed(seed)
                # Update StreamDiffusion's internal noise if seed changed (for audio reactivity)
                if seed != self._last_stream_seed:
                    self._stream.update_noise(seed)  # Use default scale=1.0
                    self._last_stream_seed = seed

            # StreamDiffusionWrapper img2img: use preprocess_image method
            try:
                # The wrapper's preprocess_image handles resizing and tensor conversion
                image_tensor = self._stream.preprocess_image(image)
                t_inference = time.perf_counter()
                logger.debug(f"  preprocess: {(t_inference - t_preprocess)*1000:.1f}ms")

                # Call the stream with preprocessed image and prompt
                output = self._stream(image=image_tensor, prompt=prompt)
                t_postprocess = time.perf_counter()
                logger.debug(f"  inference: {(t_postprocess - t_inference)*1000:.1f}ms")

                # StreamDiffusionWrapper returns PIL Image directly
                t_end = time.perf_counter()
                logger.debug(f"  total: {(t_end - t_start)*1000:.1f}ms [StreamDiffusion]")
                # Apply NSFW filter with fallback
                return self._check_nsfw_and_fallback(output)

            except Exception as e:
                logger.warning(f"StreamDiffusion img2img failed: {e}, falling back to diffusers")
                # Fall through to diffusers path below

        # Diffusers fallback path
        logger.info(f"DIFFUSERS PATH: controlnet={self._use_controlnet}, cn_pipe={self._controlnet_pipe is not None}, procedural={self._use_procedural_pose}")

        # Use ControlNet pipeline if available for identity preservation
        if self._use_controlnet and self._controlnet_pipe is not None:
            cn_result = self._generate_with_controlnet(
                prompt=prompt,
                image=image,
                strength=strength,
                negative_prompt=negative_prompt,
                guidance_scale=effective_guidance,
                num_inference_steps=steps,
                generator=generator,
                use_taesd=use_taesd,
            )
            # Apply NSFW filter with fallback
            return self._check_nsfw_and_fallback(cn_result)

        # Standard img2img path (diffusers)
        def callback(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]

            if latent_callback:
                latents = latent_callback(latents, step, steps)
            elif self._use_latent_blending:
                latents = self._apply_latent_blending(latents, step, steps)

            callback_kwargs["latents"] = latents
            return callback_kwargs

        logger.debug(f"[diffusers] img2img: strength={strength:.3f}, steps={steps}, guidance={effective_guidance}")
        t_diffusers_start = time.perf_counter()

        # Build pipeline kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "image": image,
            "strength": strength,
            "negative_prompt": negative_prompt,
            "guidance_scale": effective_guidance,
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": callback if self._use_latent_blending else None,
            "output_type": "latent" if (use_taesd or self._use_taesd) else "pil",
        }

        # Add eta for TCDScheduler (Hyper-SD)
        if self._acceleration == "hyper-sd":
            pipe_kwargs["eta"] = self._get_hyper_sd_eta()

        result = self._img2img_pipe(**pipe_kwargs)

        t_diffusers_end = time.perf_counter()
        logger.debug(f"[diffusers] inference: {(t_diffusers_end - t_diffusers_start)*1000:.1f}ms, total: {(t_diffusers_end - t_start)*1000:.1f}ms")

        if (use_taesd or self._use_taesd) and self._taesd:
            latents = result.images
            self._update_latent_history(latents)
            output_image = self._decode_latents(latents, use_taesd=True)
        else:
            output_image = result.images[0]

        # Apply NSFW filter with fallback
        return self._check_nsfw_and_fallback(output_image)

    def _generate_with_controlnet(
        self,
        prompt: str,
        image: Image.Image,
        strength: float,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        generator: torch.Generator,
        use_taesd: bool = None,
    ) -> Image.Image:
        """Generate with ControlNet for pose preservation."""
        import time

        self._controlnet_frame_count += 1
        self._frames_since_reset += 1

        # Log state for debugging
        logger.debug(f"ControlNet state: frame={self._controlnet_frame_count}, frames_since_reset={self._frames_since_reset}, pose_extracted={self._pose_extracted}, cached_pose={self._cached_pose is not None}, weight={self._controlnet_weight:.2f}")

        # Check if we should inject txt2img for a fresh pose (like Reset does)
        # Skip this entirely when using procedural poses - they animate smoothly on their own
        should_inject_txt2img = False
        if not self._use_procedural_pose:
            should_inject_txt2img = getattr(self, '_inject_txt2img_for_pose', False)
            # NOTE: Removed periodic txt2img injection from drift mode - it caused unwanted
            # 2-second interval changes even when audio was paused. Drift mode now only
            # re-extracts pose from the current frame, which is much gentler.

        if should_inject_txt2img:
            self._inject_txt2img_for_pose = False  # Clear flag
            self._pose_extracted = False  # Will extract pose from this new frame
            self._cached_pose = None
            logger.info("Injecting txt2img for fresh pose - this generates a completely new pose")
            fast_steps = self._get_acceleration_steps()
            pipe_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "guidance_scale": self._get_acceleration_guidance(),
                "num_inference_steps": fast_steps,
                "width": self.width,
                "height": self.height,
                "generator": generator,
            }
            if self._acceleration == "hyper-sd":
                pipe_kwargs["eta"] = self._get_hyper_sd_eta()
            return self._txt2img_pipe(**pipe_kwargs).images[0]

        # After reset OR when entering drift mode, skip ControlNet to let pose naturally vary
        # Skip this logic entirely when using procedural poses - we always want CN active with the animated pose
        in_evolution_window = False
        in_drift_window = False
        if not self._use_procedural_pose:
            in_evolution_window = (self._frames_since_reset <= self._pose_extraction_delay and not self._pose_extracted)
            # Also skip CN during drift mode re-extraction periods (when pose just unlocked)
            in_drift_window = (not self._pose_lock and not self._pose_extracted)

        if in_evolution_window or in_drift_window:
            logger.info(f"Skipping ControlNet for pose evolution (drift={not self._pose_lock}, frames_since_reset={self._frames_since_reset}/{self._pose_extraction_delay})")
            # Use higher strength to allow more deviation during evolution
            evolution_strength = min(strength * 1.5, 0.9)
            fast_steps = self._get_acceleration_steps(for_img2img=True, strength=evolution_strength)
            pipe_kwargs = {
                "prompt": prompt,
                "image": image,
                "strength": evolution_strength,
                "negative_prompt": negative_prompt,
                "guidance_scale": self._get_acceleration_guidance(),
                "num_inference_steps": fast_steps,
                "generator": generator,
            }
            if self._acceleration == "hyper-sd":
                pipe_kwargs["eta"] = self._get_hyper_sd_eta()
            return self._img2img_pipe(**pipe_kwargs).images[0]

        # Procedural pose mode: generate animated pose every frame
        # Do this FIRST so we always have the current pose
        should_extract = False  # Initialize for non-procedural mode
        logger.debug(f"POSE DEBUG: use_procedural={self._use_procedural_pose}, generator={self._procedural_pose_generator is not None}, weight={self._controlnet_weight}")
        if self._use_procedural_pose and self._procedural_pose_generator is not None:
            t_pose_start = time.perf_counter()
            self._cached_pose = self._procedural_pose_generator.generate_pose()
            t_pose_end = time.perf_counter()
            self._pose_extracted = True
            logger.debug(f"Procedural pose generated in {(t_pose_end - t_pose_start)*1000:.2f}ms (frame {self._controlnet_frame_count})")

            # With procedural poses, ALWAYS use ControlNet (don't skip frames)
            # because the pose changes every frame and we want to follow it

            # When txt2img mode is enabled for procedural poses, route through the
            # dedicated txt2img + ControlNet path for better pose control
            if self._procedural_use_txt2img and self._controlnet_txt2img_pipe is not None:
                logger.info(f"PROCEDURAL TXT2IMG: Using txt2img + ControlNet for strong pose control")
                return self._generate_controlnet_txt2img(
                    prompt=prompt,
                    pose_image=self._cached_pose,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    use_taesd=use_taesd,
                )
            # Otherwise fall through to img2img path below
        else:
            # For non-procedural mode, optionally skip ControlNet on some frames for speed
            use_cn_this_frame = (self._controlnet_frame_count % self._controlnet_skip_frames) == 0
            if not use_cn_this_frame and self._cached_pose is not None:
                # Fast path: skip ControlNet, just do regular img2img with cached pose
                logger.debug(f"Skipping ControlNet this frame (frame {self._controlnet_frame_count})")
                fast_steps = self._get_acceleration_steps(for_img2img=True, strength=strength)
                pipe_kwargs = {
                    "prompt": prompt,
                    "image": image,
                    "strength": strength,
                    "negative_prompt": negative_prompt,
                    "guidance_scale": self._get_acceleration_guidance(),
                    "num_inference_steps": fast_steps,
                    "generator": generator,
                }
                if self._acceleration == "hyper-sd":
                    pipe_kwargs["eta"] = self._get_hyper_sd_eta()
                return self._img2img_pipe(**pipe_kwargs).images[0]
            # Standard pose extraction logic depends on lock mode:
            # - Locked: Extract once after reset, use forever
            # - Drift: Re-extract periodically to allow pose to evolve
            if not self._pose_extracted:
                # First extraction after reset
                should_extract = True
                logger.info(f"Extracting pose for first time after reset")
            elif not self._pose_lock and (self._controlnet_frame_count % self._pose_drift_interval) == 0:
                # Drift mode: re-extract periodically
                should_extract = True
                logger.info(f"Drift mode: re-extracting pose (every {self._pose_drift_interval} frames)")

        if should_extract:
            # Save current epoch to detect if reset happens during extraction
            extraction_epoch = self._generation_epoch
            t_pose_start = time.perf_counter()
            extracted_pose = self._extract_pose(image)
            t_pose_end = time.perf_counter()

            # Check if a reset happened during extraction (epoch changed)
            if self._generation_epoch != extraction_epoch:
                logger.warning(f"Reset detected during pose extraction (epoch {extraction_epoch} -> {self._generation_epoch}), discarding result")
                # Fall back to regular img2img for this frame
                pipe_kwargs = {
                    "prompt": prompt,
                    "image": image,
                    "strength": strength,
                    "negative_prompt": negative_prompt,
                    "guidance_scale": self._get_acceleration_guidance(),
                    "num_inference_steps": self._get_acceleration_steps(for_img2img=True, strength=strength),
                    "generator": generator,
                }
                if self._acceleration == "hyper-sd":
                    pipe_kwargs["eta"] = self._get_hyper_sd_eta()
                return self._img2img_pipe(**pipe_kwargs).images[0]

            self._cached_pose = extracted_pose
            logger.info(f"Pose extraction complete: {(t_pose_end - t_pose_start)*1000:.1f}ms, success={self._cached_pose is not None}")
            self._pose_extracted = True

            if self._cached_pose is None:
                logger.warning("Pose extraction failed, falling back to standard img2img")
                pipe_kwargs = {
                    "prompt": prompt,
                    "image": image,
                    "strength": strength,
                    "negative_prompt": negative_prompt,
                    "guidance_scale": self._get_acceleration_guidance(),
                    "num_inference_steps": self._get_acceleration_steps(for_img2img=True, strength=strength),
                    "generator": generator,
                }
                if self._acceleration == "hyper-sd":
                    pipe_kwargs["eta"] = self._get_hyper_sd_eta()
                return self._img2img_pipe(**pipe_kwargs).images[0]
        else:
            logger.debug(f"Using cached pose (lock={self._pose_lock}, epoch={self._generation_epoch})")

        # Debug: Log the actual pose being used
        pose_info = f"None" if self._cached_pose is None else f"size={self._cached_pose.size}, mode={self._cached_pose.mode}"

        # BOTH MODES now use img2img + ControlNet for consistent style
        # The only difference is the pose source:
        # - Procedural: animated pose generated each frame
        # - Non-procedural: pose extracted from previous frame

        # Use the user-specified strength directly - let them control the pose variation
        effective_strength = strength
        logger.info(f"CONTROLNET IMG2IMG: strength={effective_strength:.3f}, pose={pose_info}")

        # Use acceleration-appropriate steps for ControlNet generation
        # Must account for img2img strength to ensure at least 1 effective step
        cn_steps = self._get_acceleration_steps(for_img2img=True, strength=effective_strength)

        # Validate strength to ensure at least 1 effective step (prevents empty tensor errors)
        effective_strength = self._validate_img2img_strength(effective_strength, cn_steps)

        # Boost ControlNet weight for procedural mode - need stronger pose guidance
        effective_cn_weight = self._controlnet_weight
        if self._use_procedural_pose:
            effective_cn_weight = min(self._controlnet_weight + 0.2, 1.2)  # Boost up to 1.2

        # Use acceleration-appropriate guidance
        accel_guidance = self._get_acceleration_guidance()
        # Use slightly higher guidance for ControlNet to improve pose adherence
        effective_guidance = min(accel_guidance + 0.5, 2.5)

        logger.info(f"CONTROLNET IMG2IMG: strength={effective_strength:.3f}, pose_weight={effective_cn_weight:.2f}, steps={cn_steps}, pose={pose_info}")

        t_cn_start = time.perf_counter()

        # Ensure control image matches generation resolution to prevent ControlNet dimension mismatch
        control_image = self._cached_pose
        if control_image is not None and control_image.size != (self.width, self.height):
            logger.debug(f"Resizing control image from {control_image.size} to ({self.width}, {self.height})")
            control_image = control_image.resize((self.width, self.height), Image.Resampling.LANCZOS)

        # Also ensure input image matches generation resolution
        if image.size != (self.width, self.height):
            logger.debug(f"Resizing input image from {image.size} to ({self.width}, {self.height})")
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)

        pipe_kwargs = {
            "prompt": prompt,
            "image": image,
            "control_image": control_image,
            "strength": effective_strength,
            "negative_prompt": negative_prompt,
            "guidance_scale": effective_guidance,
            "num_inference_steps": cn_steps,
            "generator": generator,
            "controlnet_conditioning_scale": effective_cn_weight,
            "output_type": "latent" if (use_taesd or self._use_taesd) else "pil",
        }
        if self._acceleration == "hyper-sd":
            pipe_kwargs["eta"] = self._get_hyper_sd_eta()

        result = self._controlnet_pipe(**pipe_kwargs)
        t_cn_end = time.perf_counter()
        logger.info(f"ControlNet inference: {(t_cn_end - t_cn_start)*1000:.0f}ms")

        if (use_taesd or self._use_taesd) and self._taesd:
            latents = result.images
            self._update_latent_history(latents)
            return self._decode_latents(latents, use_taesd=True)
        else:
            return result.images[0]

    def _update_latent_history(self, latents: torch.Tensor) -> None:
        """Update latent history for blending."""
        # Defensive check: don't store empty tensors
        if latents.numel() == 0:
            logger.error(f"Refusing to store empty latents in history: shape={latents.shape}")
            return

        self._latent_history.append(latents.clone())
        # Keep only last 3 latents
        if len(self._latent_history) > 3:
            self._latent_history.pop(0)
        self._last_latents = latents.clone()

    def _generate_controlnet_txt2img(
        self,
        prompt: str,
        pose_image: Image.Image,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        generator: torch.Generator,
        use_taesd: bool = None,
    ) -> Image.Image:
        """Generate using txt2img + ControlNet for procedural poses.

        Uses LCM-LoRA for fast 4-6 step generation with a FIXED SEED so each frame
        generates similar content but with different poses.

        The seed and blend weight are configurable via:
        - _procedural_fixed_seed: Character consistency (default 42)
        - _procedural_blend_weight: Frame smoothness (default 0.4)
        """
        # Use acceleration-appropriate steps
        procedural_steps = self._get_acceleration_steps()

        # High ControlNet weight ensures pose is followed strongly
        cn_weight = min(self._controlnet_weight + 0.1, 1.0)

        # Use acceleration-appropriate guidance
        accel_guidance = self._get_acceleration_guidance()

        # CRITICAL: Use a FIXED seed for procedural mode
        # This ensures each frame generates similar content (same person/scene)
        # Only the pose changes via ControlNet, making blending smooth
        fixed_seed = self._procedural_fixed_seed if self._procedural_fixed_seed is not None else 42
        fixed_generator = torch.Generator(device=self.device)
        fixed_generator.manual_seed(fixed_seed)

        logger.debug(f"Procedural txt2img: seed={fixed_seed}, cn_weight={cn_weight:.2f}, steps={procedural_steps}")

        # Ensure pose image matches generation resolution
        control_image = pose_image
        if control_image is not None and control_image.size != (self.width, self.height):
            logger.debug(f"Resizing pose image from {control_image.size} to ({self.width}, {self.height})")
            control_image = control_image.resize((self.width, self.height), Image.Resampling.LANCZOS)

        pipe_kwargs = {
            "prompt": prompt,
            "image": control_image,  # ControlNet conditioning image
            "negative_prompt": negative_prompt,
            "guidance_scale": accel_guidance,
            "num_inference_steps": procedural_steps,
            "generator": fixed_generator,  # Use fixed seed, not random
            "controlnet_conditioning_scale": cn_weight,
            "width": self.width,
            "height": self.height,
            "output_type": "latent",
        }
        if self._acceleration == "hyper-sd":
            pipe_kwargs["eta"] = self._get_hyper_sd_eta()

        result = self._controlnet_txt2img_pipe(**pipe_kwargs)

        latents = result.images

        # Blend with previous frame for smooth transitions
        # With fixed seed, frames are similar so blending works well
        blend_weight = self._procedural_blend_weight
        if len(self._latent_history) > 0 and blend_weight > 0:
            prev_latents = self._latent_history[-1]
            latents = (1 - blend_weight) * latents + blend_weight * prev_latents
            logger.debug(f"Procedural blend: {blend_weight:.2f} (seed={fixed_seed})")

        self._update_latent_history(latents)

        # Decode latents to image
        if (use_taesd or self._use_taesd) and self._taesd:
            return self._decode_latents(latents, use_taesd=True)
        else:
            return self._decode_latents(latents, use_taesd=False)

    def resize(self, width: int, height: int) -> None:
        """Update output dimensions."""
        self.width = width
        self.height = height
        self._initialized = False

    def set_latent_blending(self, enabled: bool) -> None:
        """Enable/disable latent blending for temporal coherence."""
        self._use_latent_blending = enabled

    def set_crossfeed_config(self, enabled: bool, power: float, range_: float, decay: float) -> None:
        """Update crossfeed/latent blending configuration dynamically.

        Args:
            enabled: Whether to enable latent blending
            power: Blend strength (0=none, 1=full previous)
            range_: Fraction of steps to apply blending
            decay: Decay rate (higher = faster adaptation)
        """
        self._use_latent_blending = enabled
        # Update the global settings so _apply_latent_blending picks them up
        settings.latent_blending = enabled
        settings.crossfeed_power = max(0.0, min(1.0, power))
        settings.crossfeed_range = max(0.0, min(1.0, range_))
        settings.crossfeed_decay = max(0.0, min(1.0, decay))
        logger.info(f"Crossfeed updated: enabled={enabled}, power={power:.2f}, range={range_:.2f}, decay={decay:.2f}")

    def set_similarity_filter(self, enabled: bool) -> None:
        """Enable/disable stochastic similarity filter."""
        self._similarity_filter = enabled

    def set_controlnet_weight(self, weight: float) -> None:
        """Update ControlNet conditioning scale at runtime."""
        self._controlnet_weight = max(0.0, min(1.0, weight))
        logger.debug(f"ControlNet weight updated to {self._controlnet_weight:.2f}")

    def set_pose_lock(self, locked: bool) -> None:
        """Set pose lock mode. Locked = extract once, Unlocked = drift."""
        old_lock = getattr(self, '_pose_lock', True)
        self._pose_lock = locked

        if not locked and old_lock:
            # Switching TO drift mode - do a full reset like clicking Reset button
            # This is necessary because img2img preserves spatial structure even at high strength
            self._pose_extracted = False
            self._cached_pose = None
            self._controlnet_frame_count = 0
            self._frames_since_reset = 0
            # Also clear latent history like Reset does
            self._latent_history.clear()
            self._last_latents = None
            self._generation_epoch += 1
            # Flag to inject txt2img on next frame for fresh pose
            self._inject_txt2img_for_pose = True
            logger.info("Pose lock disabled: will inject txt2img for fresh pose (like Reset)")
        elif locked and not old_lock:
            self._inject_txt2img_for_pose = False
            logger.info("Pose lock enabled: next extracted pose will be locked")

        logger.info(f"Pose lock = {locked}")

    def set_procedural_pose(self, enabled: bool) -> None:
        """Enable/disable procedural pose generation.

        When enabled, poses are generated procedurally (animated) instead of
        being extracted from input images. This allows continuous pose changes
        without a camera/video input.
        """
        logger.info(f"SET_PROCEDURAL_POSE called with enabled={enabled}")
        self._use_procedural_pose = enabled

        if enabled and self._procedural_pose_generator is None:
            # Initialize the procedural pose generator with upper_body framing for 512x512
            self._procedural_pose_generator = ProceduralPoseGenerator(
                width=self.width,
                height=self.height,
                framing=PoseFraming.UPPER_BODY,
            )
            logger.info("Initialized procedural pose generator with upper_body framing")

        if enabled:
            # When enabling procedural mode, set pose_extracted=True to prevent
            # the evolution/drift window from triggering and skipping ControlNet
            self._cached_pose = None
            self._pose_extracted = True  # Changed from False - we'll generate pose each frame
            self._frames_since_reset = 100  # High value to skip evolution window

            # Enable frame-synced animation for smooth low-FPS generation
            # This ensures small, consistent pose changes between frames
            if self._procedural_pose_generator:
                self._procedural_pose_generator.set_frame_sync(True, frames_per_cycle=60)
                # Default to gentle mode for smoother results at low FPS
                self._procedural_pose_generator.set_mode(PoseAnimationMode.GENTLE)
                logger.info("Frame sync enabled with GENTLE mode for smooth low-FPS animation")

            logger.info("Procedural pose mode enabled - poses will be animated")
        else:
            if self._procedural_pose_generator:
                self._procedural_pose_generator.set_frame_sync(False)
            logger.info("Procedural pose mode disabled - will extract from images")

    def set_pose_animation_mode(self, mode: str) -> None:
        """Set the procedural pose animation mode.

        Args:
            mode: One of "idle", "gentle", "dancing", "walking", "waving"
                  "gentle" is recommended for low FPS generation (slow, graceful)
        """
        if self._procedural_pose_generator is None:
            self._procedural_pose_generator = ProceduralPoseGenerator(
                width=self.width,
                height=self.height,
            )

        try:
            animation_mode = PoseAnimationMode(mode.lower())
            self._procedural_pose_generator.set_mode(animation_mode)
            logger.info(f"Pose animation mode set to: {mode}")
        except ValueError:
            logger.warning(f"Unknown pose animation mode: {mode}")

    def set_pose_intensity(self, intensity: float) -> None:
        """Set the intensity of pose animation (0-1)."""
        if self._procedural_pose_generator is None:
            self._procedural_pose_generator = ProceduralPoseGenerator(
                width=self.width,
                height=self.height,
            )

        self._procedural_pose_generator.set_intensity(intensity)
        logger.debug(f"Pose intensity set to: {intensity:.2f}")

    def set_pose_audio_energy(self, energy: float) -> None:
        """Set current audio energy for reactive poses (0-1)."""
        if self._procedural_pose_generator is not None:
            self._procedural_pose_generator.set_audio_energy(energy)

    def set_pose_animation_speed(self, speed: float) -> None:
        """Set animation speed multiplier."""
        if self._procedural_pose_generator is None:
            self._procedural_pose_generator = ProceduralPoseGenerator(
                width=self.width,
                height=self.height,
            )

        self._procedural_pose_generator.set_animation_speed(speed)
        logger.info(f"Pose animation speed set to: {speed:.2f}")

    def set_pose_framing(self, framing: str) -> None:
        """Set pose framing (full_body, upper_body, portrait).

        Args:
            framing: One of "full_body", "upper_body", "portrait"
                     "upper_body" is recommended for 512x512 images
        """
        if self._procedural_pose_generator is None:
            self._procedural_pose_generator = ProceduralPoseGenerator(
                width=self.width,
                height=self.height,
            )

        try:
            pose_framing = PoseFraming(framing.lower())
            self._procedural_pose_generator.set_framing(pose_framing)
            logger.info(f"Pose framing set to: {framing}")
        except ValueError:
            logger.warning(f"Unknown pose framing: {framing}, using upper_body")
            self._procedural_pose_generator.set_framing(PoseFraming.UPPER_BODY)

    def set_procedural_txt2img_mode(self, enabled: bool) -> None:
        """Set whether procedural poses use txt2img or img2img.

        When enabled (recommended), procedural poses use txt2img + ControlNet
        which generates purely from prompt + pose without input image conflicts.

        When disabled, uses img2img + ControlNet which preserves style from the
        input image but may have pose conflicts at lower strength values.
        """
        self._procedural_use_txt2img = enabled
        logger.info(f"Procedural txt2img mode: {enabled}")

    def set_procedural_fixed_seed(self, seed: Optional[int]) -> None:
        """Set the fixed seed for procedural pose generation.

        Using a fixed seed ensures character consistency across frames.
        None defaults to 42.
        """
        self._procedural_fixed_seed = seed
        logger.info(f"Procedural fixed seed: {seed if seed is not None else 'default (42)'}")

    def set_procedural_blend_weight(self, weight: float) -> None:
        """Set the latent blend weight for procedural pose frame smoothness.

        Higher values = smoother transitions but slower pose changes.
        Lower values = snappier pose changes but potentially more flickering.

        Args:
            weight: Blend weight from 0.0 (no blending) to 1.0 (full previous frame)
        """
        self._procedural_blend_weight = max(0.0, min(1.0, weight))
        logger.info(f"Procedural blend weight: {self._procedural_blend_weight:.2f}")

    def get_current_pose_image(self) -> Optional[Image.Image]:
        """Get the current pose image for preview/debugging.

        Returns the cached pose if available, or generates a new procedural pose.
        """
        if self._use_procedural_pose and self._procedural_pose_generator is not None:
            # Generate a fresh procedural pose for preview
            return self._procedural_pose_generator.generate_pose()
        elif self._cached_pose is not None:
            return self._cached_pose
        return None

    def clear_latent_history(self) -> None:
        """Clear latent history for fresh start."""
        self._latent_history.clear()
        self._last_latents = None
        # Clear pose cache so pose can change on reset
        # Setting _pose_extracted=False allows new pose to be captured
        # Increment epoch to invalidate any in-flight pose extraction
        self._generation_epoch += 1
        self._cached_pose = None
        self._pose_extracted = False
        self._controlnet_frame_count = 0
        self._frames_since_reset = 0  # Reset counter to enable pose evolution delay
        logger.info(f"Latent history and pose cache cleared (new epoch={self._generation_epoch}, will skip CN for {self._pose_extraction_delay} frames)")

    def cleanup(self) -> None:
        """Release GPU memory."""
        # Log VRAM before cleanup
        vram_before = None
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"StreamDiffusionWrapper cleanup starting - VRAM used: {vram_before:.2f} GB")

        if self._pipe:
            del self._pipe
            self._pipe = None
        if hasattr(self, '_txt2img_pipe') and self._txt2img_pipe:
            del self._txt2img_pipe
            self._txt2img_pipe = None
        if hasattr(self, '_img2img_pipe') and self._img2img_pipe:
            del self._img2img_pipe
            self._img2img_pipe = None
        if hasattr(self, '_controlnet_pipe') and self._controlnet_pipe:
            del self._controlnet_pipe
            self._controlnet_pipe = None
        if hasattr(self, '_controlnet_txt2img_pipe') and self._controlnet_txt2img_pipe:
            del self._controlnet_txt2img_pipe
            self._controlnet_txt2img_pipe = None
        if hasattr(self, '_controlnet') and self._controlnet:
            del self._controlnet
            self._controlnet = None
        if hasattr(self, '_pose_detector') and self._pose_detector:
            del self._pose_detector
            self._pose_detector = None
        if self._stream:
            del self._stream
            self._stream = None
        if self._taesd:
            self._taesd.cleanup()
            self._taesd = None

        # Clean up additional components
        if self._procedural_pose_generator is not None:
            del self._procedural_pose_generator
            self._procedural_pose_generator = None
        if self._safety_checker is not None:
            del self._safety_checker
            self._safety_checker = None
        if self._feature_extractor is not None:
            del self._feature_extractor
            self._feature_extractor = None
        self._loaded_loras.clear()

        self._initialized = False
        self._latent_history.clear()
        self._last_latents = None

        # Proper CUDA cleanup sequence: gc first to release Python references,
        # then synchronize to wait for async ops, then clear cache
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1024**3
            freed = vram_before - vram_after if vram_before else 0
            logger.info(f"StreamDiffusionWrapper cleanup complete - VRAM used: {vram_after:.2f} GB (freed {freed:.2f} GB)")
