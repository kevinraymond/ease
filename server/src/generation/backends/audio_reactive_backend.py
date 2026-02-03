"""Audio-reactive backend for high-responsiveness music visualization.

This backend prioritizes audio reactivity over temporal consistency,
making it ideal for music visualizers where dramatic changes on beats
and continuous variation with audio energy are desired.
"""

import logging
from typing import Optional

from PIL import Image

from ..base import (
    BaseImageGenerator,
    GenerationRequest,
    GenerationResult,
    GeneratorCapability,
)
from ..audio_reactive_pipeline import AudioReactivePipeline
from ...config import settings

logger = logging.getLogger(__name__)


class AudioReactiveBackend(BaseImageGenerator):
    """Image generator optimized for audio-reactive visuals.

    Key characteristics:
    - Fast response to seed changes (immediate visual variation)
    - Strength parameter directly controls deviation from input
    - No pipeline state that causes flickering
    - ~60-100ms per frame with LCM + TAESD
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        device: Optional[str] = None,
        num_inference_steps: int = 4,
        use_taesd: bool = True,
        # Ignored params for compatibility
        use_controlnet: Optional[bool] = None,
        controlnet_weight: Optional[float] = None,
        temporal_coherence: Optional[str] = None,
        acceleration: Optional[str] = None,
        hyper_sd_steps: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the audio-reactive backend.

        Args:
            model_id: Hugging Face model ID (SD 1.5 based)
            width: Output image width
            height: Output image height
            device: Target device
            num_inference_steps: Number of LCM steps (2-4 recommended)
            use_taesd: Use TAESD for fast VAE decoding
        """
        super().__init__()

        self._model_id = model_id or settings.model
        self._width = width
        self._height = height
        self._device = device or settings.device
        self._num_steps = num_inference_steps
        self._use_taesd = use_taesd

        self._pipeline: Optional[AudioReactivePipeline] = None
        self._initialized = False

        logger.info(
            f"AudioReactiveBackend created: model={self._model_id}, "
            f"size={self._width}x{self._height}, steps={self._num_steps}"
        )

    @property
    def capabilities(self) -> set[GeneratorCapability]:
        """Return generator capabilities."""
        return {
            GeneratorCapability.IMG2IMG,
            GeneratorCapability.NEGATIVE_PROMPT,
            GeneratorCapability.SEED_CONTROL,
            GeneratorCapability.STRENGTH_CONTROL,
            GeneratorCapability.CUSTOM_LORA,
        }

    @property
    def model_id(self) -> str:
        """Return the model ID."""
        return self._model_id

    def initialize(self) -> None:
        """Initialize the pipeline."""
        if self._initialized:
            return

        import torch

        logger.info("Initializing AudioReactiveBackend...")

        self._pipeline = AudioReactivePipeline(
            model_id=self._model_id,
            width=self._width,
            height=self._height,
            device=self._device,
            dtype=torch.float16,
            num_inference_steps=self._num_steps,
            use_taesd=self._use_taesd,
        )
        self._pipeline.initialize()

        self._initialized = True
        logger.info("AudioReactiveBackend initialized")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an image from the request.

        Args:
            request: Generation request with prompt, seed, strength, etc.

        Returns:
            GenerationResult with the output image
        """
        import time

        if not self._initialized:
            self.initialize()

        t_start = time.perf_counter()

        # Get input image (required for img2img)
        if request.input_image is None:
            # Generate a gray placeholder for txt2img-like behavior
            input_image = Image.new("RGB", (self._width, self._height), (128, 128, 128))
            # Use higher strength for txt2img
            strength = max(request.strength or 0.9, 0.9)
        else:
            input_image = request.input_image
            strength = request.strength or 0.6

        # Generate with onset effects and beat seed jump
        if self._pipeline is None:
            raise RuntimeError("AudioReactiveBackend not initialized")
        output_image = self._pipeline.generate(
            image=input_image,
            prompt=request.prompt,
            seed=request.seed or 42,
            strength=strength,
            negative_prompt=request.negative_prompt or "",
            guidance_scale=request.guidance_scale if request.guidance_scale is not None else 1.5,
            is_onset=request.is_onset,
            onset_confidence=request.onset_confidence,
            is_beat_seed_jump=request.is_beat_seed_jump,
        )

        t_end = time.perf_counter()
        generation_time = t_end - t_start

        return GenerationResult(
            image=output_image,
            seed_used=request.seed or 42,
            generation_time_ms=generation_time * 1000,
        )

    def cleanup(self) -> None:
        """Release resources."""
        logger.info("AudioReactiveBackend cleanup...")

        if self._pipeline is not None:
            self._pipeline.cleanup()
            self._pipeline = None

        self._initialized = False
        logger.info("AudioReactiveBackend cleanup complete")

    def warmup(self) -> None:
        """Warm up the pipeline."""
        if not self._initialized:
            self.initialize()

        logger.info("Warming up AudioReactiveBackend...")

        # Pipeline already warms up during initialize
        # Do an extra generation to ensure everything is ready
        dummy = Image.new("RGB", (self._width, self._height), (100, 100, 100))
        if self._pipeline is None:
            raise RuntimeError("AudioReactiveBackend not initialized")
        self._pipeline.generate(
            image=dummy,
            prompt="warmup test",
            seed=12345,
            strength=0.5,
        )

        logger.info("AudioReactiveBackend warmed up")

    def load_custom_loras(self, loras: list[dict]) -> None:
        """Load custom LoRAs (including LyCORIS/LoKr formats).

        Args:
            loras: List of LoRA configs, each with 'path' and 'weight' keys.
                   Paths are relative to the configured lora_dir.
        """
        if not self._initialized:
            self.initialize()

        if self._pipeline is not None:
            self._pipeline.load_custom_loras(loras)
