"""StreamDiffusion-based image generation backend.

This backend wraps the existing StreamDiffusionWrapper to implement the
ImageGenerator interface, providing plug-and-play compatibility.
"""

import logging
import time
from typing import Optional

from PIL import Image

from ..base import (
    BaseImageGenerator,
    GenerationRequest,
    GenerationResult,
    GeneratorCapability,
)
from ..stream_diffusion import StreamDiffusionWrapper
from ...config import settings

logger = logging.getLogger(__name__)


class StreamDiffusionBackend(BaseImageGenerator):
    """Image generator using StreamDiffusion.

    Wraps the StreamDiffusionWrapper to provide the ImageGenerator interface.
    Supports txt2img, img2img, ControlNet, latent blending, and various
    acceleration methods (LCM, Hyper-SD).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        device: Optional[str] = None,
        # Feature toggles
        use_controlnet: Optional[bool] = None,
        controlnet_weight: Optional[float] = None,
        use_taesd: Optional[bool] = None,
        temporal_coherence: Optional[str] = None,
        # Acceleration
        acceleration: Optional[str] = None,
        hyper_sd_steps: Optional[int] = None,
    ):
        """Initialize the StreamDiffusion backend.

        Args:
            model_id: Hugging Face model ID or path
            width: Output image width
            height: Output image height
            device: Target device (cuda, cpu)
            use_controlnet: Enable ControlNet for pose preservation
            controlnet_weight: ControlNet conditioning scale
            use_taesd: Use TAESD for fast VAE decoding
            temporal_coherence: Temporal coherence mode ("blending" or "none")
            acceleration: Acceleration method ("lcm", "hyper-sd", "none")
            hyper_sd_steps: Number of steps for Hyper-SD (1, 2, 4, or 8)
        """
        super().__init__()

        self._model_id = model_id or settings.model
        self._width = width
        self._height = height
        self._device = device or settings.device

        # Feature toggles (from params or settings)
        self._use_controlnet = use_controlnet if use_controlnet is not None else settings.use_controlnet
        self._controlnet_weight = controlnet_weight if controlnet_weight is not None else settings.controlnet_pose_weight
        self._use_taesd = use_taesd if use_taesd is not None else settings.use_taesd
        self._temporal_coherence = temporal_coherence or "blending"

        # Acceleration
        self._acceleration = acceleration if acceleration is not None else settings.acceleration
        self._hyper_sd_steps = hyper_sd_steps if hyper_sd_steps is not None else settings.hyper_sd_steps

        # The underlying wrapper
        self._wrapper: Optional[StreamDiffusionWrapper] = None

        # Set capabilities based on configuration
        self._update_capabilities()

    def _update_capabilities(self) -> None:
        """Update capabilities based on current configuration."""
        self._capabilities = {
            GeneratorCapability.TXT2IMG,
            GeneratorCapability.IMG2IMG,
            GeneratorCapability.CUSTOM_LORA,
        }

        if self._use_controlnet:
            self._capabilities.add(GeneratorCapability.CONTROLNET)

        if self._use_taesd:
            self._capabilities.add(GeneratorCapability.TAESD)

        if self._temporal_coherence == "blending":
            self._capabilities.add(GeneratorCapability.LATENT_BLENDING)

        if self._acceleration == "lcm":
            self._capabilities.add(GeneratorCapability.LCM)
        elif self._acceleration == "hyper-sd":
            self._capabilities.add(GeneratorCapability.HYPER_SD)

        # NSFW filter if enabled in settings
        if settings.nsfw_filter:
            self._capabilities.add(GeneratorCapability.NSFW_FILTER)

    def initialize(self) -> None:
        """Initialize the StreamDiffusion pipeline."""
        if self._initialized:
            return

        logger.info(f"Initializing StreamDiffusionBackend: model={self._model_id}")
        logger.info(f"  ControlNet: {self._use_controlnet}")
        logger.info(f"  TAESD: {self._use_taesd}")
        logger.info(f"  Acceleration: {self._acceleration}")

        self._wrapper = StreamDiffusionWrapper(
            model_id=self._model_id,
            width=self._width,
            height=self._height,
            device=self._device,
            use_controlnet=self._use_controlnet,
            controlnet_weight=self._controlnet_weight,
            use_taesd=self._use_taesd,
            temporal_coherence=self._temporal_coherence,
            acceleration=self._acceleration,
            hyper_sd_steps=self._hyper_sd_steps,
        )
        self._wrapper.initialize()

        self._initialized = True
        logger.info("StreamDiffusionBackend initialized")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an image from the request."""
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Extract backend hints
        hints = request.backend_hints
        controlnet_image = hints.get("controlnet_image")
        use_taesd = hints.get("use_taesd", self._use_taesd)

        # Decide txt2img vs img2img
        if request.input_image is None:
            # txt2img mode
            image = self._wrapper.generate_txt2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                use_taesd=use_taesd,
            )
        else:
            # img2img mode
            image = self._wrapper.generate_img2img(
                prompt=request.prompt,
                image=request.input_image,
                strength=request.strength,
                negative_prompt=request.negative_prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                use_taesd=use_taesd,
            )

        generation_time_ms = (time.perf_counter() - start_time) * 1000

        return GenerationResult(
            image=image,
            seed_used=request.seed,
            generation_time_ms=generation_time_ms,
            metadata={
                "backend": "stream_diffusion",
                "model_id": self._model_id,
                "acceleration": self._acceleration,
            },
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._wrapper:
            self._wrapper.cleanup()
            self._wrapper = None

        self._initialized = False
        logger.info("StreamDiffusionBackend cleaned up")

    def warmup(self) -> None:
        """Warm up the pipeline."""
        if not self._initialized:
            self.initialize()

        logger.info("Warming up StreamDiffusionBackend...")
        request = GenerationRequest(
            prompt="warmup test",
            num_inference_steps=1,
        )
        try:
            self.generate(request)
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    # Additional methods for dynamic configuration

    def set_controlnet_weight(self, weight: float) -> None:
        """Update ControlNet conditioning scale at runtime."""
        self._controlnet_weight = weight
        if self._wrapper:
            self._wrapper.set_controlnet_weight(weight)

    def set_pose_lock(self, locked: bool) -> None:
        """Set pose lock mode."""
        if self._wrapper:
            self._wrapper.set_pose_lock(locked)

    def set_procedural_pose(self, enabled: bool) -> None:
        """Enable/disable procedural pose generation."""
        if self._wrapper:
            self._wrapper.set_procedural_pose(enabled)

    def set_pose_animation_mode(self, mode: str) -> None:
        """Set procedural pose animation mode."""
        if self._wrapper:
            self._wrapper.set_pose_animation_mode(mode)

    def set_pose_audio_energy(self, energy: float) -> None:
        """Set audio energy for reactive poses."""
        if self._wrapper:
            self._wrapper.set_pose_audio_energy(energy)

    def set_crossfeed_config(
        self,
        enabled: bool,
        power: float,
        range_: float,
        decay: float,
    ) -> None:
        """Update latent blending configuration."""
        if self._wrapper:
            self._wrapper.set_crossfeed_config(enabled, power, range_, decay)

    def load_custom_loras(self, loras: list[dict]) -> None:
        """Load custom LoRAs."""
        if self._wrapper:
            self._wrapper.load_custom_loras(loras)

    def clear_latent_history(self) -> None:
        """Clear latent history for fresh start."""
        if self._wrapper:
            self._wrapper.clear_latent_history()

    def get_current_pose_image(self) -> Optional[Image.Image]:
        """Get the current pose image for preview."""
        if self._wrapper:
            return self._wrapper.get_current_pose_image()
        return None

    def resize(self, width: int, height: int) -> None:
        """Update output dimensions (requires re-initialization)."""
        self._width = width
        self._height = height
        if self._wrapper:
            self._wrapper.resize(width, height)
            self._initialized = False

    @property
    def wrapper(self) -> Optional[StreamDiffusionWrapper]:
        """Access the underlying wrapper for advanced usage."""
        return self._wrapper
