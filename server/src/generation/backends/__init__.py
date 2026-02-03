"""Image generation backend implementations and factory.

This module provides the registry of available backends and a factory
function to create generators based on configuration.
"""

import logging
from typing import Callable, Optional

from ..base import ImageGenerator, BaseImageGenerator, GeneratorCapability
from .stream_diffusion_backend import StreamDiffusionBackend
from .flux_klein_backend import FluxKleinBackend
from .audio_reactive_backend import AudioReactiveBackend

logger = logging.getLogger(__name__)


# Registry of available backends
# Maps backend name to factory function
_BACKEND_REGISTRY: dict[str, Callable[..., ImageGenerator]] = {
    "stream_diffusion": StreamDiffusionBackend,
    "flux_klein": FluxKleinBackend,
    "audio_reactive": AudioReactiveBackend,
}


def register_backend(name: str, factory: Callable[..., ImageGenerator]) -> None:
    """Register a new backend.

    Args:
        name: Unique name for the backend
        factory: Factory function or class that creates the generator
    """
    if name in _BACKEND_REGISTRY:
        logger.warning(f"Overwriting existing backend: {name}")
    _BACKEND_REGISTRY[name] = factory
    logger.info(f"Registered backend: {name}")


def list_backends() -> list[str]:
    """List available backend names."""
    return list(_BACKEND_REGISTRY.keys())


def create_generator(
    backend: str = "stream_diffusion",
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
    # Backend-specific kwargs
    **kwargs,
) -> ImageGenerator:
    """Create an image generator based on backend type.

    Args:
        backend: Backend name ("stream_diffusion", etc.)
        model_id: Hugging Face model ID or path
        width: Output image width
        height: Output image height
        device: Target device (cuda, cpu)
        use_controlnet: Enable ControlNet for pose preservation
        controlnet_weight: ControlNet conditioning scale
        use_taesd: Use TAESD for fast VAE decoding
        temporal_coherence: Temporal coherence mode ("blending" or "none")
        acceleration: Acceleration method ("lcm", "hyper-sd", "none")
        hyper_sd_steps: Number of steps for Hyper-SD
        **kwargs: Additional backend-specific arguments

    Returns:
        Configured ImageGenerator instance

    Raises:
        ValueError: If backend is not registered
    """
    if backend not in _BACKEND_REGISTRY:
        available = ", ".join(_BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend: {backend}. Available: {available}")

    factory = _BACKEND_REGISTRY[backend]

    logger.info(f"Creating generator: backend={backend}, model={model_id}")

    return factory(
        model_id=model_id,
        width=width,
        height=height,
        device=device,
        use_controlnet=use_controlnet,
        controlnet_weight=controlnet_weight,
        use_taesd=use_taesd,
        temporal_coherence=temporal_coherence,
        acceleration=acceleration,
        hyper_sd_steps=hyper_sd_steps,
        **kwargs,
    )


def create_generator_from_settings() -> ImageGenerator:
    """Create a generator from global settings.

    Reads configuration from the global Settings object and creates
    the appropriate generator.

    Returns:
        Configured ImageGenerator instance
    """
    from ...config import settings

    backend = getattr(settings, "generator_backend", "stream_diffusion")

    # Common kwargs for all backends
    kwargs = {
        "width": settings.width,
        "height": settings.height,
        "device": settings.device,
        "use_controlnet": settings.use_controlnet,
        "controlnet_weight": settings.controlnet_pose_weight,
        "use_taesd": settings.use_taesd,
        "temporal_coherence": "blending" if settings.latent_blending else "none",
        "acceleration": settings.acceleration,
        "hyper_sd_steps": settings.hyper_sd_steps,
    }

    # Backend-specific configuration
    if backend == "audio_reactive":
        kwargs["num_inference_steps"] = getattr(settings, "audio_reactive_steps", 4)
        kwargs["use_taesd"] = settings.use_taesd if settings.use_taesd else True
    elif backend == "flux_klein":
        kwargs["model_id"] = settings.flux_model_id or None
        kwargs["precision"] = settings.flux_precision
        kwargs["cpu_offload"] = settings.flux_cpu_offload
        kwargs["inference_steps"] = settings.flux_inference_steps
        kwargs["guidance_scale"] = settings.flux_guidance_scale
        kwargs["compile_transformer"] = settings.flux_compile
        kwargs["cache_prompt_embeds"] = settings.flux_cache_prompt
    else:
        kwargs["model_id"] = settings.model

    return create_generator(backend=backend, **kwargs)


__all__ = [
    "BaseImageGenerator",
    "GeneratorCapability",
    "StreamDiffusionBackend",
    "FluxKleinBackend",
    "AudioReactiveBackend",
    "register_backend",
    "list_backends",
    "create_generator",
    "create_generator_from_settings",
]
