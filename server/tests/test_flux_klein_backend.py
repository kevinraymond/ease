"""Manual test script for FLUX.2 [klein] backend.

Run with:
    cd server && uv run python tests/test_flux_klein_backend.py

This script validates:
1. Backend initialization with auto-precision detection
2. txt2img generation
3. img2img generation
4. Memory cleanup

Note: Requires CUDA GPU and diffusers>=0.32.0
"""
import sys
import os
import time

# Add server directory to path for proper package imports
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

import torch
from PIL import Image


def test_gpu_detection():
    """Test GPU VRAM detection and precision selection."""
    from src.generation.backends.flux_klein_backend import (
        get_gpu_vram_gb,
        detect_optimal_precision,
    )

    print("\n=== GPU Detection Test ===")
    vram_gb = get_gpu_vram_gb()
    print(f"Detected GPU VRAM: {vram_gb:.2f} GB")

    precision = detect_optimal_precision(vram_gb)
    print(f"Recommended precision: {precision}")

    # Validate logic
    if vram_gb >= 16:
        assert precision == "bf16", f"Expected bf16 for {vram_gb}GB, got {precision}"
    elif vram_gb >= 10:
        assert precision == "fp8", f"Expected fp8 for {vram_gb}GB, got {precision}"
    else:
        assert precision == "nvfp4", f"Expected nvfp4 for {vram_gb}GB, got {precision}"

    print("GPU detection: PASSED")
    return vram_gb, precision


def test_backend_initialization(precision: str = "auto", compile_transformer: bool = False):
    """Test backend initialization."""
    from src.generation.backends.flux_klein_backend import FluxKleinBackend

    print(f"\n=== Backend Initialization Test (precision={precision}, compile={compile_transformer}) ===")

    backend = FluxKleinBackend(
        width=512,
        height=512,
        precision=precision,
        cpu_offload=not compile_transformer,  # Disable offload when compiling
        compile_transformer=compile_transformer,
    )

    print(f"Backend created: {backend}")
    print(f"Capabilities: {backend.capabilities}")

    # Don't initialize yet - that downloads models
    print("Backend creation: PASSED")
    return backend


def test_txt2img(backend):
    """Test text-to-image generation."""
    from src.generation.base import GenerationRequest

    print("\n=== txt2img Generation Test ===")

    # Initialize (downloads model if needed)
    print("Initializing backend (may download model)...")
    start = time.time()
    backend.initialize()
    print(f"Initialization took: {time.time() - start:.1f}s")

    # Warmup
    print("Warming up...")
    backend.warmup()

    # Generate
    print("Generating image...")
    request = GenerationRequest(
        prompt="a serene mountain landscape at sunset, digital art, highly detailed",
        negative_prompt="blurry, low quality",
        num_inference_steps=4,
        seed=42,
    )

    start = time.time()
    result = backend.generate(request)
    gen_time = time.time() - start

    print(f"Generation time: {gen_time:.2f}s ({result.generation_time_ms:.0f}ms internal)")
    print(f"Seed used: {result.seed_used}")
    print(f"Image size: {result.image.size}")
    print(f"Metadata: {result.metadata}")

    # Save result
    output_path = "/tmp/flux_klein_txt2img.png"
    result.image.save(output_path)
    print(f"Saved to: {output_path}")

    print("txt2img: PASSED")
    return result.image


def test_img2img(backend, input_image: Image.Image):
    """Test image-to-image generation."""
    from src.generation.base import GenerationRequest

    print("\n=== img2img Generation Test (Klein uses conditioning, strength ignored) ===")

    # Klein uses image conditioning (concatenation), not noise-based img2img
    # Strength is ignored - variation comes from seed/prompt changes
    request = GenerationRequest(
        prompt="a mountain landscape at night with stars, digital art",
        negative_prompt="blurry, low quality",
        input_image=input_image,
        strength=0.7,  # Ignored by Klein - always uses conditioning
        num_inference_steps=4,
        seed=123,
    )

    start = time.time()
    result = backend.generate(request)
    gen_time = time.time() - start

    print(f"Generation time: {gen_time:.2f}s ({result.generation_time_ms:.0f}ms internal)")
    print(f"Seed used: {result.seed_used}")
    print(f"Image size: {result.image.size}")

    # Save result
    output_path = "/tmp/flux_klein_img2img.png"
    result.image.save(output_path)
    print(f"Saved to: {output_path}")
    print("img2img: PASSED")

    return result.image


def test_cleanup(backend):
    """Test resource cleanup."""
    print("\n=== Cleanup Test ===")

    # Get VRAM before cleanup
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM allocated before cleanup: {vram_before:.2f} GB")

    backend.cleanup()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_after = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM allocated after cleanup: {vram_after:.2f} GB")

    assert not backend.is_initialized, "Backend should not be initialized after cleanup"
    print("Cleanup: PASSED")


def test_factory_creation():
    """Test creating backend through factory function."""
    from src.generation.backends import create_generator, list_backends

    print("\n=== Factory Creation Test ===")

    backends = list_backends()
    print(f"Available backends: {backends}")
    assert "flux_klein" in backends, "flux_klein should be registered"

    # Create via factory (don't initialize - just test creation)
    generator = create_generator(
        backend="flux_klein",
        width=512,
        height=512,
        precision="auto",
    )
    print(f"Created generator: {type(generator).__name__}")

    print("Factory creation: PASSED")


def main():
    """Run all tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Test FLUX.2 Klein backend")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--skip-download-prompt", action="store_true", help="Skip download confirmation")
    args = parser.parse_args()

    print("=" * 60)
    print("FLUX.2 [klein] Backend Test Suite")
    if args.compile:
        print("  (torch.compile ENABLED)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. FLUX requires GPU.")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Test GPU detection
    vram_gb, precision = test_gpu_detection()

    # Test factory registration
    test_factory_creation()

    # Ask user before downloading model
    if not args.skip_download_prompt:
        print("\n" + "=" * 60)
        print("The following tests will download the FLUX model (~12GB).")
        print("=" * 60)
        response = input("Continue with model download? [y/N]: ")
        if response.lower() != "y":
            print("Skipping model tests.")
            return

    # Create backend for generation tests
    backend = test_backend_initialization(precision, compile_transformer=args.compile)

    try:
        # Test txt2img
        txt2img_result = test_txt2img(backend)

        # Test img2img
        test_img2img(backend, txt2img_result)

    finally:
        # Always cleanup
        test_cleanup(backend)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nGenerated images saved to:")
    print("  - /tmp/flux_klein_txt2img.png")
    print("  - /tmp/flux_klein_img2img.png (uses image conditioning for frame continuity)")


if __name__ == "__main__":
    main()
