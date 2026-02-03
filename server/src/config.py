"""Configuration settings for the EASE AI generation server."""

from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """EASE server configuration settings.

    Default values are optimized for ~5.5GB VRAM (RTX 3060/4060).
    Enable additional features if you have more VRAM available.
    """

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8765
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Logging settings
    # Root log level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    # Module-specific overrides (set to DEBUG to troubleshoot specific areas)
    log_level_generation: str = "WARNING"  # Very verbose at DEBUG (timing per frame)
    log_level_pipeline: str = "INFO"  # High-level generation events
    log_level_lyrics: str = "INFO"  # Lyric detection/transcription
    log_level_server: str = "INFO"  # WebSocket handling

    # Model settings
    # For StreamDiffusion: use LCM model for best speed, or SD 1.5 + LCM-LoRA
    # Options: "SimianLuo/LCM_Dreamshaper_v7", "runwayml/stable-diffusion-v1-5", "Lykon/dreamshaper-8"
    model: str = "Lykon/dreamshaper-8"
    device: str = "cuda"
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"

    # Generation defaults
    width: int = 512
    height: int = 512
    steps: int = 4  # SD-Turbo uses 1-4 steps
    cfg_scale: float = 0.0  # SD-Turbo works with 0 guidance
    target_fps: int = 20

    # Performance settings
    use_tensorrt: bool = False
    compile_unet: bool = False  # torch.compile

    # Generator backend: "stream_diffusion", "audio_reactive" (default), "flux_klein"
    # - audio_reactive: SD 1.5 + LCM optimized for audio reactivity (~10-15 FPS)
    # - stream_diffusion: SD 1.5 + StreamDiffusion for smooth streaming (~15-20 FPS)
    # - flux_klein: FLUX.2 [klein] 4B for high quality (up to ~3 FPS with optimizations)
    generator_backend: str = "audio_reactive"

    # Acceleration method: "lcm", "hyper-sd", or "none"
    # - lcm: LCM-LoRA with LCMScheduler (default, proven stable)
    # - hyper-sd: ByteDance Hyper-SD with TCDScheduler (1-2 step generation)
    # - none: No acceleration LoRA, standard scheduler
    acceleration: Literal["lcm", "hyper-sd", "none"] = "lcm"
    # Hyper-SD specific settings
    hyper_sd_steps: Literal[1, 2, 4, 8] = 1  # Must match the LoRA variant
    hyper_sd_lora_scale: float = 0.125  # Recommended fusion scale (much lower than LCM)
    hyper_sd_eta: float = 1.0  # Stochasticity: 1.0 for 1-step, 0.3-0.5 for multi-step

    # FLUX.2 [klein] backend settings
    # Hardware tiers are auto-detected and configure optimal settings:
    # - High-end (24GB+): No offload, bf16, torch.compile max-autotune, TensorRT
    # - Mid-range (12-16GB): No offload, fp8, torch.compile reduce-overhead
    # - Entry (8-10GB): CPU offload, fp8, no compile
    # - Minimum (<8GB): Full offload, fp8
    #
    # Precision: auto, bf16, fp8, nvfp4
    # - auto: Auto-select based on hardware tier (recommended)
    # - bf16: Full precision bfloat16 (best quality, requires ~12GB VRAM)
    # - fp8: 8-bit float (requires optimum-quanto, ~50% memory savings)
    # - nvfp4: 4-bit float (requires Blackwell/RTX 50 series)
    flux_precision: Literal["auto", "bf16", "fp8", "nvfp4"] = "auto"
    flux_model_id: str = ""  # Empty = auto-select based on precision
    # CPU offload: None = auto-detect based on VRAM, True/False = manual override
    flux_cpu_offload: Optional[bool] = None  # None = auto-detect from hardware tier
    flux_inference_steps: int = 4  # FLUX.2 klein is optimized for 4 steps
    flux_guidance_scale: float = 1.0  # FLUX.2 Klein uses 1.0 guidance
    # torch.compile: None = auto-detect, True/False = manual override
    # Provides 1.5-3x speedup after initial compilation (~60-120s first run)
    flux_compile: Optional[bool] = None  # None = auto-detect from hardware tier
    # Compile mode: "max-autotune" (best perf, longer compile), "reduce-overhead" (faster compile)
    flux_compile_mode: str = ""  # Empty = auto-select based on tier
    # TensorRT: None = auto-detect, True/False = manual override
    # Provides additional 1.5-2.4x speedup, requires torch-tensorrt
    flux_use_tensorrt: Optional[bool] = None  # None = auto-detect from hardware tier
    flux_cache_prompt: bool = True  # Cache text encoder outputs (prompt rarely changes)
    # Force a specific hardware tier (for testing): high_end, mid_range, entry, minimum
    flux_force_hardware_tier: str = ""  # Empty = auto-detect
    # Latent caching: Reuse previous output when audio energy is low
    # Provides variable speedup during quiet sections (effectively infinite FPS)
    flux_latent_cache_enabled: bool = True
    flux_latent_cache_threshold: float = 0.1  # RMS energy below this triggers cache
    # TeaCache: Cache transformer outputs at early timesteps for similar prompts
    # Provides up to 2x speedup when prompt embedding is similar between frames
    flux_teacache_enabled: bool = True
    flux_teacache_threshold: float = 0.95  # Cosine similarity for cache hit

    # TAESD (Tiny AutoEncoder for Stable Diffusion) - 100x faster decode
    use_taesd: bool = False  # Disabled - TAESD scaling issues with newer diffusers
    taesd_model: str = "madebyollin/taesd"  # Default TAESD model
    taesd_sdxl_model: str = "madebyollin/taesdxl"  # TAESD for SDXL

    # StreamDiffusion optimizations
    stream_batch: bool = True  # Enable Stream Batch (1.5x speedup)
    residual_cfg: bool = True  # Enable Residual CFG (2x speedup)
    similarity_filter: bool = True  # Stochastic Similarity Filter (skip redundant frames)
    similarity_threshold: float = 0.98  # Threshold for skipping frames
    # t_index_list controls which denoising steps to use
    # [0, 16, 32, 45] works for both txt2img and img2img
    # Later steps like [32, 45] only work for img2img (not enough denoising for txt2img)
    t_index_list: list[int] = [0, 16, 32, 45]  # Works for both txt2img and img2img

    # TensorRT settings
    tensorrt_cache_dir: str = ".tensorrt_cache"
    tensorrt_max_batch_size: int = 2
    tensorrt_use_fp16: bool = True

    # Temporal coherence settings - balance between stability and variation
    # IMPORTANT: Latent blending is DISABLED because history is only tracked in
    # StreamDiffusion/TAESD paths. With ControlNet enabled (default), the diffusers
    # path doesn't update latent history, causing blending to always pull toward
    # the first frame. TODO: Fix latent tracking in all code paths.
    latent_blending: bool = False  # DISABLED - broken with ControlNet/diffusers path
    crossfeed_power: float = 0.3
    crossfeed_range: float = 0.4
    crossfeed_decay: float = 0.4

    # Identity preservation settings
    # NOTE: ControlNet requires SD 1.5 base model, not SD-Turbo
    # Set to False when using SD-Turbo, or switch model_id to SD 1.5
    # Default: False to minimize VRAM usage (~5.5GB baseline)
    use_controlnet: bool = False
    controlnet_pose_weight: float = 0.8
    controlnet_lineart_weight: float = 0.3
    use_ip_adapter: bool = False
    ip_adapter_scale: float = 0.6

    # LoRA settings
    lora_dir: str = "./loras"  # Base directory for LoRA files
    default_loras: list[dict] = []  # Default LoRAs: [{"path": "style.safetensors", "weight": 0.8}]

    # NSFW filtering - when enabled, uses safety checker and falls back to previous frame
    # instead of showing black. This keeps the visual flow while filtering inappropriate content.
    nsfw_filter: bool = False  # Enable to filter NSFW content (slight perf impact)

    # Frame encoding
    jpeg_quality: int = 85

    # Lyric detection settings
    # Default: False to minimize VRAM usage (adds ~2GB for Whisper + Demucs)
    lyrics: bool = False
    # Lyric provider: "whisper", "hybrid", or "none"
    # - whisper: Fast-whisper transcription (general ASR)
    # - hybrid: Fingerprint matching + whisper fallback
    # - none: Disable lyrics
    lyric_provider: Literal["whisper", "hybrid", "none"] = "whisper"
    # Recommended: large-v3-turbo (6x faster than large, similar accuracy, great balance)
    # Alternatives: large-v2 (best English), medium, small, tiny
    lyric_model_size: str = "large-v3-turbo"
    lyric_device: str = "cuda"  # cuda or cpu (use cpu to offload from GPU)
    lyric_compute_type: str = "float16"  # float16, int8, float32
    lyric_transcribe_interval: float = 1.0  # Seconds between transcription runs
    lyric_buffer_seconds: float = 5.0  # Rolling audio buffer (larger = more context, better accuracy)
    lyric_beam_size: int = 5  # Beam size for Whisper (1 = greedy, faster; 5 = beam search, better)
    lyric_initial_prompt: str = "Song lyrics, singing vocals, English"  # Hint context to Whisper
    # Filter out words that leak from the prompt or are common false positives
    lyric_filter_words: list[str] = [
        "lyrics", "singing", "music", "song", "vocal", "vocals",
        "english", "subscribe", "thank you for watching",
    ]

    # Audio fingerprinting settings
    fingerprint_enabled: bool = True  # Enable audio fingerprinting for known song detection
    fingerprint_duration_seconds: float = 15.0  # Duration to fingerprint at session start
    fingerprint_db_path: str = "./lyrics_db.sqlite"  # Path to lyrics database

    # Lyric scheduling settings
    lyric_beat_aligned: bool = True  # Align lyric changes to beat boundaries
    lyric_beat_delay: int = 2  # Delay lyric changes by N beats (perceptual masking)
    lyric_vad_filter: bool = True  # Enable VAD filtering (disable for music with heavy instrumentals)
    lyric_vad_threshold: float = 0.3  # VAD threshold (lower = more sensitive, good for music vocals)

    # Automatic song change detection (silence detection)
    silence_detection_enabled: bool = True  # Auto-detect song changes via silence gaps
    silence_threshold: float = 0.02  # RMS threshold below which audio is considered silence (0-1)
    silence_duration_seconds: float = 1.0  # How long silence must last to trigger song change
    silence_cooldown_seconds: float = 3.0  # Minimum time between auto-resets (prevent rapid triggers)

    # Vocal separation settings (Demucs)
    lyric_vocal_separation: bool = True  # Use Demucs to isolate vocals before transcription
    lyric_demucs_model: str = "htdemucs_ft"  # htdemucs (fast), htdemucs_ft (better quality), mdx_extra (best)
    lyric_demucs_device: str = "cuda"  # cuda or cpu (use cpu to avoid GPU contention)

    @field_validator("hyper_sd_steps", mode="before")
    @classmethod
    def coerce_hyper_sd_steps(cls, v):
        if isinstance(v, str):
            return int(v)
        return v

    class Config:
        env_prefix = "EASE_"
        env_file = ".env"


settings = Settings()
