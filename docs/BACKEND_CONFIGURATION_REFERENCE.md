# UI Configuration to Backend Usage Mapping

This document maps each UI configuration option to which backends use it and how.

## The Three Backends

| Backend | Description | FPS Range | Key Capabilities |
|---------|-------------|-----------|------------------|
| **Audio Reactive** | SD 1.5 + LCM optimized for audio reactivity | 8-12 FPS | TAESD, seed/strength control |
| **StreamDiffusion** | SD 1.5 real-time smooth streaming | 15-20 FPS | ControlNet, LoRA, temporal blending, acceleration |
| **FLUX Klein** | FLUX.2 [klein] 4B high-quality | 1-3 FPS | Prompt modulation, latent caching |

---

## Generation Mode vs Backend

**Important:** Generation Mode and Backend are separate concepts:

- **Generation Mode** (Live Feedback vs Pose Animation) determines the overall generation strategy
- **Backend** (Audio Reactive, StreamDiffusion, FLUX Klein) determines which diffusion pipeline is used

**Key behavior:**
- **Pose Animation mode requires StreamDiffusion backend**. This is because Pose Animation requires ControlNet support for pose guidance, which only StreamDiffusion provides.
- The UI will **disable the Pose Animation option** when Audio Reactive or FLUX Klein backends are selected.
- Your backend selection (Audio Reactive, StreamDiffusion, or FLUX Klein) only applies when using **Live Feedback** mode.

To use Pose Animation, first switch to the StreamDiffusion backend in the Connection section, then select Pose Animation mode.

---

## Configuration Usage by Section

### Prompts & Styles

| UI Input | Audio Reactive | StreamDiffusion | FLUX Klein |
|----------|----------------|-----------------|------------|
| **Base Prompt** | Used for img2img generation with LCM | Used for txt2img/img2img via StreamDiffusion | Used for txt2img/img2img generation |
| **Negative Prompt** | Used (passed to LCM pipeline) | Used (passed to pipeline) | Ignored (FLUX doesn't use negative prompts) |
| **Audio Mapping Preset** | Affects prompt modulation via AudioMapper | Affects prompt modulation via AudioMapper | Affects prompt modulation via AudioMapper |
| **Custom Mappings** | Parameters mapped to audio features | Parameters mapped to audio features | Parameters mapped to audio features |

### Generation Settings

| UI Input | Audio Reactive | StreamDiffusion | FLUX Klein |
|----------|----------------|-----------------|------------|
| **Generation Mode** | Feedback mode uses this backend; Pose Animation switches to StreamDiffusion | Used for both modes | Feedback mode uses this backend; Pose Animation switches to StreamDiffusion |
| **Transform Strength** | Directly controls img2img deviation (0.6-0.9 typical) | Controls img2img strength | Ignored (Klein uses conditioning, not denoising strength) |
| **Target FPS** | Controls frame rate target | Controls frame rate target | Controls frame rate target |
| **Smooth Transitions** | Ignored (no temporal blending) | Enables latent blending between frames | Ignored (no temporal coherence support) |
| **Beat Refresh** | Triggers new image on beat | Triggers new image on beat | Triggers new image on beat |
| **Keyframe Interval** | N/A (Pose Animation uses StreamDiffusion) | Used in Pose Animation mode | N/A (Pose Animation uses StreamDiffusion) |
| **Keyframe Strength** | N/A (Pose Animation uses StreamDiffusion) | Used in Pose Animation mode | N/A (Pose Animation uses StreamDiffusion) |

### Pose & ControlNet

| UI Input | Audio Reactive | StreamDiffusion | FLUX Klein |
|----------|----------------|-----------------|------------|
| **Pose Preservation (ControlNet)** | Ignored (no ControlNet support) | Enables OpenPose ControlNet for pose | Ignored (FLUX ControlNet not yet available) |
| **Pose Influence** | Ignored | Sets `controlnet_pose_weight` (0-1) | Ignored |
| **Lock Pose** | Ignored | Fixes pose from first extracted frame | Ignored |
| **Procedural Animation** | Ignored | Enables generated poses instead of extraction | Ignored |
| **Animation Style** | Ignored | Sets animation mode (gentle/idle/dancing/walking/waving) | Ignored |
| **Animation Speed** | Ignored | Controls procedural pose speed | Ignored |
| **Movement Intensity** | Ignored | Controls procedural pose intensity | Ignored |

### Audio-Reactive Effects

| UI Input | Audio Reactive | StreamDiffusion | FLUX Klein |
|----------|----------------|-----------------|------------|
| **Spectral Displacement** | Applied as post-processing | Applied as post-processing | Applied as post-processing |
| **Onset Glitch** | Applied as post-processing | Applied as post-processing | Applied as post-processing |
| **Treble Shimmer** | Applied as post-processing | Applied as post-processing | Applied as post-processing |
| **Wave Distortion** | Applied as post-processing | Applied as post-processing | Applied as post-processing |
| **Beat Flash** | Applied as post-processing | Applied as post-processing | Applied as post-processing |
| **Silence Degradation** | Applied as post-processing | Applied as post-processing | Applied as post-processing |

*Note: Audio-reactive effects are post-processing shaders applied after generation, so all backends support them equally.*

### Image Quality

| UI Input | Audio Reactive | StreamDiffusion | FLUX Klein |
|----------|----------------|-----------------|------------|
| **Bicubic Interpolation** | Applied during upscaling | Applied during upscaling | Applied during upscaling |
| **Audio-Reactive Sharpening** | Applied as post-processing | Applied as post-processing | Applied as post-processing |
| **Sharpen Strength** | Controls sharpening intensity | Controls sharpening intensity | Controls sharpening intensity |

*Note: Image quality options are post-processing applied after generation.*

### Lyrics

| UI Input | Audio Reactive | StreamDiffusion | FLUX Klein |
|----------|----------------|-----------------|------------|
| **Lyric Detection** | Keywords injected into prompt | Keywords injected into prompt | Keywords injected into prompt |
| **Lyric-Driven Mode** | Lyrics replace base prompt | Lyrics replace base prompt | Lyrics replace base prompt |
| **Show Subtitles** | Display only (not generation) | Display only (not generation) | Display only (not generation) |

### Advanced

| UI Input | Audio Reactive | StreamDiffusion | FLUX Klein |
|----------|----------------|-----------------|------------|
| **Acceleration Method** | Ignored (uses fixed LCM) | Selects LCM-LoRA, Hyper-SD, or None | Ignored (FLUX is pre-distilled) |
| **Hyper-SD Steps** | Ignored | Sets 1/2/4/8 steps for Hyper-SD | Ignored |
| **Server URL** | Connection endpoint | Connection endpoint | Connection endpoint |
| **Model ID** | Loads specified SD 1.5 model | Loads specified SD 1.5 model | Ignored (uses FLUX.2 Klein base) |
| **Resolution (Width/Height)** | Sets output dimensions | Sets output dimensions | Sets output dimensions |
| **Maintain Aspect Ratio** | Display letterboxing only | Display letterboxing only | Display letterboxing only |
| **Base Image** | Used as img2img input | Used as img2img input | Used as conditioning image |
| **Lock to Base Image** | Always starts from base | Always starts from base | Always conditions on base |
| **Custom LoRAs** | Ignored (no LoRA support) | Loads custom LoRAs with weights | Experimental (not fully supported) |

---

## Backend-Specific Settings (FLUX Only)

These settings only appear/apply when FLUX Klein backend is selected:

| UI Input | Description | Effect |
|----------|-------------|--------|
| **Precision** | bf16/fp8/nvfp4/auto | Auto-detects from VRAM, affects memory and quality |
| **CPU Offload** | Enable/disable | Auto-enabled on <12GB VRAM for memory efficiency |
| **Inference Steps** | Number of steps | Default 4 for Klein (guidance-distilled) |
| **Compile Transformer** | torch.compile | Auto-enabled on high-VRAM GPUs for 2-3x speedup |

---

## Capability Summary

| Capability | Audio Reactive | StreamDiffusion | FLUX Klein |
|------------|----------------|-----------------|------------|
| txt2img | Yes | Yes | Yes |
| img2img | Yes | Yes | Yes (conditioning-based) |
| Negative Prompt | Yes | Yes | No |
| ControlNet | No | Yes | No |
| Custom LoRA | No | Yes | Experimental |
| TAESD (fast VAE) | Yes | Yes | No (uses FLUX VAE) |
| Temporal Blending | No | Yes | No |
| LCM Acceleration | Fixed | Configurable | N/A (pre-distilled) |
| Hyper-SD | No | Yes | N/A |
| Seed Control | Yes | Yes | Yes |
| Strength Control | Yes (0-1) | Yes (0-1) | No (conditioning only) |

---

## Key Differences in Behavior

### Transform Strength
- **Audio Reactive**: Directly controls denoising strength (higher = more change from input)
- **StreamDiffusion**: Same as Audio Reactive
- **FLUX Klein**: Ignored - Klein uses image conditioning, not strength-based denoising. Variation comes from seed/prompt changes.

### Prompts
- **Audio Reactive/StreamDiffusion**: Both base and negative prompts used
- **FLUX Klein**: Only base prompt used (FLUX architecture doesn't support negative prompts)

### Model ID
- **Audio Reactive/StreamDiffusion**: Any SD 1.5 compatible model from HuggingFace
- **FLUX Klein**: Always uses `black-forest-labs/FLUX.2-klein-4B` (model ID setting ignored)

### ControlNet/Pose
- Only **StreamDiffusion** supports ControlNet and pose features
- When you select **Pose Animation** mode, the system automatically uses StreamDiffusion regardless of your backend setting
- In **Live Feedback** mode with Audio Reactive or FLUX Klein, pose settings are ignored

### Acceleration
- **Audio Reactive**: Fixed LCM (4 steps)
- **StreamDiffusion**: Configurable (LCM, Hyper-SD 1/2/4/8, or None)
- **FLUX Klein**: Pre-distilled (4 steps, guidance_scale=1.0)
