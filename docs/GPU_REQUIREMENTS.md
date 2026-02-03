# GPU Requirements

EASE requires an NVIDIA GPU with CUDA support. This document details VRAM usage for different configurations.

## Minimum Requirements

- **GPU**: NVIDIA with 6GB+ VRAM (RTX 3060, RTX 4060, or better)
- **CUDA**: Version 11.8 or higher
- **Driver**: 520.x or higher

## VRAM Usage by Configuration

**Note**: These are estimated values. Run actual measurements for your specific setup.

### Tiered Presets

| Preset | Est. VRAM | Resolution | StreamDiff | ControlNet | Lyrics |
|--------|-----------|------------|------------|------------|--------|
| minimal | ~4GB | 384x384 | Yes | No | No |
| **standard** | ~5.5GB | 512x512 | Yes | No | No |
| quality | ~7GB | 512x512 | Yes | Yes | No |
| full | ~12GB | 512x512 | Yes | Yes | Yes |

### Feature VRAM Breakdown

| Feature | Est. VRAM Delta | Notes |
|---------|-----------------|-------|
| Base SD model (fp16) | ~2.5GB | Loaded once |
| StreamDiffusion overhead | ~1.5GB | Batch processing buffers |
| ControlNet OpenPose | +~1GB | Adds pose guidance |
| Whisper (tiny) | +~1GB | Speech-to-text |
| Demucs (htdemucs) | +~1GB | Vocal separation |

### Resolution Impact

| Resolution | Relative VRAM | Notes |
|------------|---------------|-------|
| 384x384 | ~80% | Faster, less detail |
| 512x512 | 100% | Default |
| 768x768 | ~150% | More detail, slower |

## Configuration Examples

### RTX 3060 (12GB) - Recommended

```bash
# .env - Full features
EASE_MODEL=Lykon/dreamshaper-8
EASE_WIDTH=512
EASE_HEIGHT=512
EASE_USE_CONTROLNET=true
EASE_LYRICS=true
```

### RTX 4060 (8GB) - Standard

```bash
# .env - Most features
EASE_MODEL=Lykon/dreamshaper-8
EASE_WIDTH=512
EASE_HEIGHT=512
EASE_USE_CONTROLNET=true
EASE_LYRICS=false
```

### RTX 3060 (6GB) / GTX 1660 Super - Minimal

```bash
# .env - Core features only
EASE_MODEL=Lykon/dreamshaper-8
EASE_WIDTH=384
EASE_HEIGHT=384
EASE_USE_CONTROLNET=false
EASE_LYRICS=false
```

## Monitoring VRAM

Check current VRAM usage:

```bash
# One-time check
nvidia-smi

# Continuous monitoring
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce resolution:
   ```bash
   EASE_WIDTH=384
   EASE_HEIGHT=384
   ```

2. Disable features:
   ```bash
   EASE_USE_CONTROLNET=false
   EASE_LYRICS=false
   ```

3. Close other GPU applications (games, other ML models)

4. Restart the server to clear GPU memory

### Slow Generation

1. Check your generator backend - FLUX is higher quality but slower:
   ```bash
   # StreamDiffusion backend: ~20 FPS
   EASE_GENERATOR_BACKEND=stream_diffusion
   # FLUX backend: ~3 FPS (higher quality)
   EASE_GENERATOR_BACKEND=flux_klein
   ```

2. Reduce resolution for faster frames

3. Consider enabling TensorRT if you have time to compile:
   ```bash
   EASE_USE_TENSORRT=true
   ```
   Note: First run will be slow while compiling TRT engines.

## Measuring Actual VRAM

To get accurate measurements for your system:

```bash
# Terminal 1: Monitor VRAM
watch -n 1 nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Terminal 2: Start server with specific config
cd server
EASE_USE_CONTROLNET=false EASE_LYRICS=false uv run python -m src.main

# Note the peak VRAM after model loads
# Then enable features one at a time to measure deltas
```

Submit your measurements to help improve this documentation!
