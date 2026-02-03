"""Generation module - AI image generation pipeline.

## Plug-and-Play Generator Interface

Use the ImageGenerator interface for plug-and-play backend swapping:

```python
from generation import create_generator, create_generator_from_settings

# Create from global settings (recommended)
generator = create_generator_from_settings()
generator.initialize()

# Or create with specific backend
generator = create_generator("stream_diffusion", model_id="Lykon/dreamshaper-8")
generator.initialize()

# Generate images
request = GenerationRequest(prompt="a woman dancing")
result = generator.generate(request)
image = result.image
```

## Available Backends

- `stream_diffusion`: StreamDiffusion-based generation (default)

## Dependency Injection

You can inject a pre-configured generator into the pipeline:

```python
generator = create_generator("stream_diffusion")
pipeline = GenerationPipeline(config, generator=generator)
pipeline.initialize()
```
"""

# Abstract interface - these have no circular dependencies
from .base import (
    ImageGenerator,
    BaseImageGenerator,
    GenerationRequest,
    GenerationResult,
    GeneratorCapability,
)

# Low-level components (for advanced usage)
from .stream_diffusion import StreamDiffusionWrapper
from .frame_encoder import FrameEncoder


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == "GenerationPipeline":
        from .pipeline import GenerationPipeline
        return GenerationPipeline
    elif name == "GeneratedFrame":
        from .pipeline import GeneratedFrame
        return GeneratedFrame
    elif name == "create_generator":
        from .backends import create_generator
        return create_generator
    elif name == "create_generator_from_settings":
        from .backends import create_generator_from_settings
        return create_generator_from_settings
    elif name == "register_backend":
        from .backends import register_backend
        return register_backend
    elif name == "list_backends":
        from .backends import list_backends
        return list_backends
    elif name == "StreamDiffusionBackend":
        from .backends import StreamDiffusionBackend
        return StreamDiffusionBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Abstract interface
    "ImageGenerator",
    "BaseImageGenerator",
    "GenerationRequest",
    "GenerationResult",
    "GeneratorCapability",
    # Backend factory (lazy loaded)
    "create_generator",
    "create_generator_from_settings",
    "register_backend",
    "list_backends",
    "StreamDiffusionBackend",
    # Pipeline (lazy loaded)
    "GenerationPipeline",
    "GeneratedFrame",
    # Low-level components
    "StreamDiffusionWrapper",
    "FrameEncoder",
]
