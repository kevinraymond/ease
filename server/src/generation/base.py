"""Abstract interface for image generation backends.

This module defines the plug-and-play interface for image generation systems.
Different backends can implement this interface to provide image generation
through various methods (StreamDiffusion, ComfyUI, external APIs, etc.).

Example usage:
    from generation import create_generator

    generator = create_generator("stream_diffusion", model_id="Lykon/dreamshaper-8")
    generator.initialize()

    # Generate from text
    request = GenerationRequest(prompt="a woman dancing")
    result = generator.generate(request)
    image = result.image

    # Generate from text + image (img2img)
    request = GenerationRequest(
        prompt="a woman dancing",
        input_image=previous_frame,
        strength=0.4,
    )
    result = generator.generate(request)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Protocol, runtime_checkable

from PIL import Image


class GeneratorCapability(Enum):
    """Capabilities that an image generator may support."""

    TXT2IMG = auto()  # Text-to-image generation
    IMG2IMG = auto()  # Image-to-image transformation
    CONTROLNET = auto()  # ControlNet conditioning (pose, depth, etc.)
    LATENT_BLENDING = auto()  # Temporal coherence via latent blending
    STREAMING = auto()  # StreamDiffusion-style continuous generation
    NSFW_FILTER = auto()  # Built-in NSFW content filtering
    TAESD = auto()  # Fast TAESD VAE decoding
    CUSTOM_LORA = auto()  # Custom LoRA loading
    HYPER_SD = auto()  # Hyper-SD acceleration
    LCM = auto()  # LCM acceleration
    NEGATIVE_PROMPT = auto()  # Negative prompt support
    SEED_CONTROL = auto()  # Seed control for reproducibility
    STRENGTH_CONTROL = auto()  # Strength control for img2img


@dataclass
class GenerationRequest:
    """Request for image generation.

    Attributes:
        prompt: Text prompt for generation
        negative_prompt: Negative prompt (things to avoid)
        seed: Random seed for reproducibility (None = random)
        guidance_scale: Classifier-free guidance scale
        input_image: Input image for img2img (None = txt2img mode)
        strength: Denoising strength for img2img (0-1, higher = more change)
        num_inference_steps: Number of denoising steps (None = backend default)
        is_onset: Whether this frame is on an audio onset/transient
        onset_confidence: Confidence of onset detection (0-1)
        backend_hints: Backend-specific data (ControlNet images, LoRA configs, etc.)
    """

    prompt: str
    negative_prompt: str = ""
    seed: Optional[int] = None
    guidance_scale: float = 1.5
    input_image: Optional[Image.Image] = None
    strength: float = 0.5
    num_inference_steps: Optional[int] = None
    is_onset: bool = False
    onset_confidence: float = 0.0
    is_beat_seed_jump: bool = False  # Beat triggered seed jump - use txt2img injection
    backend_hints: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from image generation.

    Attributes:
        image: The generated image
        seed_used: The actual seed used (useful when seed was None)
        generation_time_ms: Time taken for generation in milliseconds
        metadata: Additional backend-specific metadata
    """

    image: Image.Image
    seed_used: Optional[int] = None
    generation_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ImageGenerator(Protocol):
    """Protocol for image generation backends.

    Backends must implement this interface to be usable with the EASE system.
    Use duck typing - any class implementing these methods will work.
    """

    @property
    def capabilities(self) -> set[GeneratorCapability]:
        """Set of capabilities supported by this generator."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the generator has been initialized."""
        ...

    def initialize(self) -> None:
        """Initialize the generator.

        Load models, set up pipelines, etc. May be called multiple times
        (should be idempotent after first initialization).
        """
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an image from the request.

        Args:
            request: Generation parameters

        Returns:
            GenerationResult with the generated image and metadata

        Raises:
            RuntimeError: If generator is not initialized
        """
        ...

    def cleanup(self) -> None:
        """Clean up resources.

        Release GPU memory, close connections, etc.
        Should be safe to call multiple times.
        """
        ...

    def warmup(self) -> None:
        """Warm up the generator.

        Run a test generation to pre-compile/optimize. Optional but
        recommended for best first-frame performance.
        """
        ...

    def supports_capability(self, capability: GeneratorCapability) -> bool:
        """Check if this generator supports a capability.

        Args:
            capability: The capability to check

        Returns:
            True if supported, False otherwise
        """
        ...


class BaseImageGenerator(ABC):
    """Abstract base class for image generators.

    Provides default implementations for common functionality.
    Subclasses must implement the abstract methods.
    """

    def __init__(self):
        self._initialized = False
        self._capabilities: set[GeneratorCapability] = set()

    @property
    def capabilities(self) -> set[GeneratorCapability]:
        """Set of capabilities supported by this generator."""
        return self._capabilities

    @property
    def is_initialized(self) -> bool:
        """Whether the generator has been initialized."""
        return self._initialized

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the generator. Subclasses must implement."""
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an image. Subclasses must implement."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources. Subclasses must implement."""
        pass

    def warmup(self) -> None:
        """Warm up the generator with a test generation."""
        if not self._initialized:
            self.initialize()

        # Run a minimal generation to warm up
        request = GenerationRequest(
            prompt="test warmup",
            num_inference_steps=1,
        )
        try:
            self.generate(request)
        except Exception:
            pass  # Warmup failures are non-fatal

    def supports_capability(self, capability: GeneratorCapability) -> bool:
        """Check if this generator supports a capability."""
        return capability in self._capabilities

    def _ensure_initialized(self) -> None:
        """Ensure the generator is initialized."""
        if not self._initialized:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
