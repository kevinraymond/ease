#!/usr/bin/env python3
"""Diagnostic script to verify SOTA implementation components."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check all SOTA modules can be imported."""
    print("=" * 50)
    print("Checking module imports...")
    print("=" * 50)

    modules = [
        ("Config", "src.config", "settings"),
        ("Protocol", "src.server.protocol", "AudioMetrics"),
        ("Audio Mapper", "src.mapping.audio_mapper", "AudioMapper"),
        ("Prompt Modulator", "src.mapping.prompt_modulator", "PromptModulator"),
        ("Stream Diffusion", "src.generation.stream_diffusion", "StreamDiffusionWrapper"),
        ("TensorRT Compiler", "src.generation.tensorrt_compiler", "TensorRTCompiler"),
        ("Dual GPU", "src.generation.dual_gpu", "DualGPUParallel"),
        ("Latent Blending", "src.generation.latent_blending", "LatentBlender"),
        ("Keyframe Interpolator", "src.generation.keyframe_interpolator", "KeyframeInterpolationPipeline"),
        ("ControlNet Stack", "src.generation.controlnet_stack", "ControlNetStack"),
        ("IP-Adapter", "src.generation.ip_adapter", "IPAdapterFaceID"),
        ("Identity Pipeline", "src.generation.identity_pipeline", "IdentityPipeline"),
        ("Network Bending", "src.generation.network_bending", "NetworkBender"),
    ]

    results = []
    for name, module_path, class_name in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✓ {name}")
            results.append((name, True, None))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False, str(e)))

    return results


def check_config():
    """Display current configuration."""
    print("\n" + "=" * 50)
    print("Current Configuration")
    print("=" * 50)

    try:
        from src.config import settings

        config_items = [
            ("Resolution", f"{settings.width}x{settings.height}"),
            ("Use TAESD", settings.use_taesd),
            ("Use TensorRT", settings.use_tensorrt),
            ("Latent Blending", settings.latent_blending),
            ("Use ControlNet", settings.use_controlnet),
            ("Use IP-Adapter", settings.use_ip_adapter),
            ("Inference Steps", settings.steps),
            ("Stream Batch", settings.stream_batch),
            ("Residual CFG", settings.residual_cfg),
        ]

        for name, value in config_items:
            print(f"  {name}: {value}")

    except Exception as e:
        print(f"  Error loading config: {e}")


def check_torch():
    """Check PyTorch and CUDA availability."""
    print("\n" + "=" * 50)
    print("PyTorch & CUDA Status")
    print("=" * 50)

    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    GPU {i}: {name} ({mem:.1f} GB)")
    except ImportError:
        print("  ✗ PyTorch not installed")


def check_optional_deps():
    """Check optional dependencies."""
    print("\n" + "=" * 50)
    print("Optional Dependencies")
    print("=" * 50)

    deps = [
        ("diffusers", "Diffusers (Stable Diffusion)"),
        ("transformers", "Transformers (CLIP)"),
        ("accelerate", "Accelerate (distributed)"),
        ("tensorrt", "TensorRT"),
        ("cv2", "OpenCV"),
        ("insightface", "InsightFace (ArcFace)"),
    ]

    for module, name in deps:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
        except Exception as e:
            print(f"  ⚠ {name} (error: {type(e).__name__})")

    # Check controlnet_aux separately due to mediapipe issues
    try:
        # Try importing specific processors we need, avoiding mediapipe
        from controlnet_aux import OpenposeDetector, HEDdetector, LineartDetector
        print(f"  ✓ ControlNet Preprocessors")
    except ImportError:
        print(f"  ✗ ControlNet Preprocessors (not installed)")
    except AttributeError as e:
        if "mediapipe" in str(e):
            print(f"  ⚠ ControlNet Preprocessors (mediapipe issue - install mediapipe<0.10.10)")
        else:
            print(f"  ⚠ ControlNet Preprocessors (error: {e})")


def test_audio_mapping():
    """Test audio mapping with sample data."""
    print("\n" + "=" * 50)
    print("Audio Mapping Test")
    print("=" * 50)

    try:
        from src.mapping.audio_mapper import AudioMapper, get_preset
        from src.mapping.prompt_modulator import PromptModulator
        from src.server.protocol import AudioMetrics, OnsetInfo, ChromaFeatures, GenerationConfig

        # Create test metrics
        metrics = AudioMetrics(
            bass=0.8,
            mid=0.5,
            treble=0.3,
            rms=0.6,
            peak=0.9,
            raw_bass=0.7,
            raw_mid=0.4,
            raw_treble=0.2,
            bpm=120.0,
            is_beat=True,
            spectral_centroid=0.7,
            onset=OnsetInfo(is_onset=True, confidence=0.8, strength=0.7, spectral_flux=0.5),
            chroma=ChromaFeatures(
                bins=[0.1, 0.2, 0.8, 0.1, 0.3, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
                energy=0.5,
            ),
            dominant_chroma=2,
        )

        config = GenerationConfig(
            base_prompt="a dancer in neon light",
            negative_prompt="blurry, distorted",
            img2img_strength=0.5,
        )

        # Test mapper
        mapper = AudioMapper(get_preset("dancer"))
        params = mapper.map(metrics, config)

        print(f"  Base prompt: {config.base_prompt}")
        print(f"  Modulated prompt: {params.prompt[:80]}...")
        print(f"  Strength: {params.strength:.3f}")
        print(f"  Guidance scale: {params.guidance_scale:.3f}")
        print(f"  Color keywords: {params.color_keywords}")
        print(f"  Is onset: {params.is_onset}")
        print("  ✓ Audio mapping works!")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_prompt_modulation():
    """Test prompt modulation with CLIP weighting."""
    print("\n" + "=" * 50)
    print("Prompt Modulation Test")
    print("=" * 50)

    try:
        from src.mapping.prompt_modulator import PromptModulator, DynamicPromptBuilder
        from src.server.protocol import AudioMetrics, OnsetInfo, ChromaFeatures

        modulator = PromptModulator()

        # Test with high energy
        metrics_high = AudioMetrics(
            bass=0.9, mid=0.7, treble=0.8, rms=0.85, peak=0.95,
            raw_bass=0.85, raw_mid=0.65, raw_treble=0.75, bpm=140.0,
            is_beat=True,
            spectral_centroid=0.8,
            onset=OnsetInfo(is_onset=True, confidence=0.9, strength=0.8, spectral_flux=0.7),
            chroma=ChromaFeatures(bins=[0.9] + [0.1]*11, energy=0.8),
        )

        result_high = modulator.modulate("abstract visuals", metrics_high)
        print(f"  High energy: {result_high[:100]}...")

        # Test with low energy
        metrics_low = AudioMetrics(
            bass=0.1, mid=0.2, treble=0.1, rms=0.15, peak=0.2,
            raw_bass=0.1, raw_mid=0.15, raw_treble=0.1, bpm=80.0,
            is_beat=False,
            spectral_centroid=0.2,
        )

        result_low = modulator.modulate("abstract visuals", metrics_low)
        print(f"  Low energy: {result_low[:100]}...")

        # Test dynamic builder
        builder = DynamicPromptBuilder()
        builder.set_base("cyberpunk city").add("neon", 1.3).add("rain", 0.8)
        built = builder.build()
        print(f"  Dynamic builder: {built}")

        print("  ✓ Prompt modulation works!")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Run all diagnostics."""
    print("\n")
    print("╔" + "═" * 48 + "╗")
    print("║    EASE SOTA Implementation Diagnostic   ║")
    print("╚" + "═" * 48 + "╝")

    results = check_imports()
    check_config()
    check_torch()
    check_optional_deps()
    test_audio_mapping()
    test_prompt_modulation()

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  Module imports: {passed}/{total} passed")

    if passed == total:
        print("\n  ✓ All core modules loaded successfully!")
        print("  Ready for testing. Start the server with:")
        print("    uv run python -m src.server.main")
    else:
        print("\n  ⚠ Some modules failed to load.")
        print("  Check the errors above and install missing dependencies.")


if __name__ == "__main__":
    main()
