#!/usr/bin/env python3
"""
EASE Model Downloader

Downloads required models from HuggingFace for first-time setup.
Models are cached in ~/.cache/huggingface/ and reused across runs.
"""

import os
import sys
from pathlib import Path


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return True
        else:
            print("Warning: CUDA not available. AI features will be disabled.")
            return False
    except ImportError:
        print("Warning: PyTorch not installed yet.")
        return False


def download_diffusion_model(model_id: str):
    """Download the base Stable Diffusion model."""
    print(f"\nDownloading base model: {model_id}")
    print("This may take a while on first run (~2-4 GB)...")

    try:
        from diffusers import StableDiffusionPipeline
        import torch

        # Just download, don't load to GPU
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # Don't move to device - just verify download
        del pipe
        print(f"Model {model_id} downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def download_controlnet_models():
    """Download ControlNet models for pose guidance (optional)."""
    print("\nDownloading ControlNet OpenPose model (optional, ~1.4 GB)...")

    try:
        from diffusers import ControlNetModel
        import torch

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16,
        )
        del controlnet
        print("ControlNet model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Note: ControlNet download failed (optional feature): {e}")
        return False


def download_rife_models():
    """Download RIFE frame interpolation models."""
    print("\nChecking RIFE models for frame interpolation...")

    server_dir = Path(__file__).parent.parent / "server"
    rife_dir = server_dir / "models" / "rife"
    rife_dir.mkdir(parents=True, exist_ok=True)

    # RIFE models are downloaded on demand by keyframe_interpolator.py
    # Just create the directory structure
    print(f"RIFE model directory: {rife_dir}")
    print("RIFE models will be downloaded on first use.")
    return True


def main():
    print("=" * 50)
    print("EASE Model Downloader")
    print("=" * 50)

    # Check CUDA availability
    check_cuda()

    # Get model ID from environment or use default
    model_id = os.environ.get("EASE_MODEL", "Lykon/dreamshaper-8")

    # Download main model (critical)
    base_model_success = download_diffusion_model(model_id)
    if not base_model_success:
        print("\nError: Failed to download base model.")
        print("You can manually download it later by running:")
        print(f"  cd server && uv run python -c \"from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('{model_id}')\"")

    # Download ControlNet (optional)
    download_controlnet_models()

    # Setup RIFE directory
    download_rife_models()

    print("\n" + "=" * 50)
    if base_model_success:
        print("Model download complete!")
    else:
        print("Model download failed!")
    print("=" * 50)
    print("\nNote: Some models are downloaded on first use.")
    print("The first generation may take longer than usual.")

    # Exit with error if critical download failed
    if not base_model_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
