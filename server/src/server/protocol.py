"""WebSocket protocol message schemas."""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


class GenerationMode(str, Enum):
    """Image generation mode.

    - FEEDBACK: Real-time live performance (default). First frame = txt2img,
                subsequent frames use previous frame as input (img2img loop).
    - KEYFRAME_RIFE: Real-time with pose control. Uses procedural poses with
                     ControlNet for keyframes, RIFE interpolation between.
    """

    FEEDBACK = "feedback"  # Real-time img2img loop (absorbs txt2img + img2img)
    KEYFRAME_RIFE = "keyframe_rife"  # Real-time + pose control


class AudioSource(str, Enum):
    """Audio sources for parameter mapping."""

    BASS = "bass"
    MID = "mid"
    TREBLE = "treble"
    RMS = "rms"
    PEAK = "peak"
    SPECTRAL_CENTROID = "spectral_centroid"
    BASS_MID = "bass_mid"
    BPM = "bpm"
    ONSET_STRENGTH = "onset_strength"
    FIXED = "fixed"


class CurveType(str, Enum):
    """Curve types for parameter mapping."""

    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EXPONENTIAL = "exponential"


class ParameterMappingConfig(BaseModel):
    """Configuration for a single parameter mapping."""

    id: str = Field(description="Unique identifier for this mapping")
    name: str = Field(description="Display name")
    source: AudioSource = Field(default=AudioSource.BASS, description="Audio source")
    curve: CurveType = Field(default=CurveType.LINEAR, description="Mapping curve")
    input_min: float = Field(default=0.0, ge=0, le=1)
    input_max: float = Field(default=1.0, ge=0, le=1)
    output_min: float = Field(default=0.0)
    output_max: float = Field(default=1.0)
    enabled: bool = Field(default=True)


class BeatTriggerConfig(BaseModel):
    """Configuration for beat-triggered actions."""

    seed_jump: bool = Field(default=True, description="Randomize seed on beat")
    strength_boost: float = Field(default=0.15, ge=0, le=0.5, description="Add to strength on beat")
    force_keyframe: bool = Field(default=False, description="Force keyframe generation on beat")


class OnsetTriggerConfig(BaseModel):
    """Configuration for onset-triggered actions."""

    seed_variation: int = Field(default=100, ge=0, le=1000, description="Seed variation range on onset")
    force_keyframe: bool = Field(default=False, description="Force keyframe generation on onset")


class TriggerConfig(BaseModel):
    """Configuration for audio-triggered actions."""

    on_beat: BeatTriggerConfig = Field(default_factory=BeatTriggerConfig)
    on_onset: OnsetTriggerConfig = Field(default_factory=OnsetTriggerConfig)
    chroma_threshold: float = Field(default=0.4, ge=0.1, le=0.8, description="Chroma intensity threshold for color detection (lower = more colors)")


class CrossfeedConfig(BaseModel):
    """Latent blending / crossfeed configuration for temporal coherence.

    NOTE: Latent blending is DISABLED by default because:
    1. The diffusers img2img path doesn't update latent history (without TAESD)
    2. StreamDiffusion path has separate issues
    This causes blending to always pull toward the first frame.
    """

    enabled: bool = Field(default=False, description="Enable latent blending between frames")
    power: float = Field(default=0.3, ge=0.0, le=1.0, description="Blend strength (0=none, 1=full previous)")
    range: float = Field(default=0.4, ge=0.0, le=1.0, description="Fraction of steps to apply blending")
    decay: float = Field(default=0.4, ge=0.0, le=1.0, description="Decay rate (higher = faster adaptation)")


class MappingConfig(BaseModel):
    """Full audio-to-parameter mapping configuration."""

    mappings: dict[str, ParameterMappingConfig] = Field(default_factory=dict)
    triggers: TriggerConfig = Field(default_factory=TriggerConfig)
    crossfeed: CrossfeedConfig = Field(default_factory=CrossfeedConfig)
    preset_name: str = Field(default="custom")
    beat_cooldown_ms: int = Field(default=300, ge=0, le=2000, description="Minimum ms between beat triggers")


class OnsetInfo(BaseModel):
    """Onset detection info - captures transients and note attacks."""

    is_onset: bool = Field(description="Whether this frame is an onset")
    confidence: float = Field(ge=0, le=1, description="Onset confidence level")
    strength: float = Field(ge=0, description="Onset strength (spectral flux)")
    spectral_flux: float = Field(ge=0, description="Total spectral change")


class ChromaFeatures(BaseModel):
    """Chroma features - 12-bin pitch class distribution."""

    bins: list[float] = Field(description="12-bin chroma values (C, C#, D, ..., B)")
    energy: float = Field(ge=0, le=1, description="Overall tonal content")


class LyricPipelineState(str, Enum):
    """State of the hybrid lyric pipeline."""

    INITIALIZING = "initializing"
    FINGERPRINTING = "fingerprinting"
    MATCHED = "matched"  # Song identified from database
    NOT_MATCHED = "not_matched"  # Using transcription fallback
    STOPPED = "stopped"


class LyricInfo(BaseModel):
    """Detected lyric information."""

    text: str = Field(description="Recent transcribed text")
    keywords: list[tuple[str, float]] = Field(description="Extracted keywords with weights")
    confidence: float = Field(ge=0, le=1, description="Transcription confidence")
    is_singing: bool = Field(default=False, description="Whether detected as singing vs spoken")
    language: str = Field(default="en", description="Detected language")

    # Hybrid pipeline status (optional, for enhanced UI)
    pipeline_state: Optional[LyricPipelineState] = Field(default=None, description="Lyric pipeline state")
    fingerprint_progress: Optional[float] = Field(default=None, ge=0, le=1, description="Fingerprint collection progress")
    matched_song_title: Optional[str] = Field(default=None, description="Title of matched song (if any)")
    matched_song_artist: Optional[str] = Field(default=None, description="Artist of matched song (if any)")


class AudioMetrics(BaseModel):
    """Audio analysis metrics from the frontend."""

    rms: float = Field(ge=0, le=1, description="Root mean square (volume)")
    peak: float = Field(ge=0, le=1, description="Peak amplitude")
    bass: float = Field(ge=0, le=1, description="Smoothed bass energy")
    mid: float = Field(ge=0, le=1, description="Smoothed mid energy")
    treble: float = Field(ge=0, le=1, description="Smoothed treble energy")
    raw_bass: float = Field(ge=0, le=1, description="Raw bass energy")
    raw_mid: float = Field(ge=0, le=1, description="Raw mid energy")
    raw_treble: float = Field(ge=0, le=1, description="Raw treble energy")
    bpm: float = Field(ge=0, description="Detected BPM")
    is_beat: bool = Field(description="Whether this frame is a beat")
    sample_rate: int = Field(default=48000)
    fft_size: int = Field(default=2048)

    # SOTA audio features for AI control
    spectral_centroid: float = Field(default=0.5, ge=0, le=1, description="Spectral centroid (brightness/timbre)")
    raw_spectral_centroid: float = Field(default=0.5, ge=0, le=1, description="Raw spectral centroid")
    onset: Optional[OnsetInfo] = Field(default=None, description="Onset detection info")
    chroma: Optional[ChromaFeatures] = Field(default=None, description="Chroma features")
    dominant_chroma: int = Field(default=0, ge=0, le=11, description="Dominant pitch class (0-11: C to B)")

    # Lyric-derived keywords (injected by backend)
    lyrics: Optional[LyricInfo] = Field(default=None, description="Detected lyrics info")


class MetricsMessage(BaseModel):
    """Incoming metrics message."""

    type: Literal["metrics"] = "metrics"
    metrics: AudioMetrics
    timestamp: float


class GenerationConfig(BaseModel):
    """Generation configuration."""

    generation_mode: GenerationMode = GenerationMode.FEEDBACK
    base_prompt: str = "photograph of a woman dancing, photorealistic, realistic skin texture, natural lighting, professional photography, detailed, 8k"
    negative_prompt: str = "cartoon, illustration, anime, drawing, painting, digital art, vector art, flat colors, stylized, blurry, text, watermark, low quality, distorted, deformed"
    model_id: str = "Lykon/dreamshaper-8"
    img2img_strength: float = Field(default=0.35, ge=0, le=1)
    mapping_preset: str = "reactive"
    target_fps: int = Field(default=20, ge=1, le=60)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    # Auto-reset feedback loop every N frames (0 = disabled)
    auto_reset_frames: int = Field(default=0, ge=0, le=100)

    # Acceleration method (LCM or Hyper-SD) - changing requires pipeline restart
    acceleration: str = Field(default="lcm", description="Acceleration: lcm, hyper-sd, or none")
    hyper_sd_steps: int = Field(default=1, ge=1, le=8, description="Hyper-SD step variant (1, 2, 4, or 8)")

    # Generator backend - changing requires pipeline restart
    generator_backend: str = Field(default="audio_reactive", description="Image generation backend")

    # SOTA settings from frontend
    use_taesd: bool = Field(default=False, description="Use TAESD for faster VAE decode")
    use_controlnet: bool = Field(default=False, description="Use ControlNet for pose preservation")
    controlnet_pose_weight: float = Field(default=0.8, ge=0, le=1)
    controlnet_pose_lock: bool = Field(default=True)  # True = extract once, False = re-extract for drift
    use_procedural_pose: bool = Field(default=False, description="Use procedural pose generation instead of extracting from images")
    pose_animation_mode: str = Field(default="gentle", description="Animation mode: idle, gentle, dancing, walking, waving")
    pose_animation_speed: float = Field(default=1.0, ge=0.1, le=5.0)
    pose_animation_intensity: float = Field(default=0.5, ge=0, le=1)
    pose_framing: str = Field(default="upper_body", description="Pose framing: full_body, upper_body, portrait")
    # Procedural pose txt2img settings (for better pose control)
    procedural_use_txt2img: bool = Field(default=True, description="Use txt2img + ControlNet for procedural poses (vs img2img)")
    procedural_fixed_seed: Optional[int] = Field(default=None, description="Fixed seed for character consistency (None = use 42)")
    procedural_blend_weight: float = Field(default=0.4, ge=0, le=1, description="Latent blend weight for frame smoothness")
    use_ip_adapter: bool = Field(default=False, description="Use IP-Adapter for identity")
    ip_adapter_scale: float = Field(default=0.6, ge=0, le=1)
    temporal_coherence: str = Field(default="blending", description="none or blending")

    # Keyframe + RIFE settings
    keyframe_interval: int = Field(default=4, ge=2, le=16, description="Generate keyframe every N frames")
    keyframe_strength: float = Field(default=0.6, ge=0.1, le=1.0, description="Denoising strength for keyframes (higher = more pose change)")
    rife_interpolation: bool = Field(default=True, description="Use RIFE for interpolation (vs blend)")

    # Periodic pose refresh - injects txt2img every 8 beats to break feedback convergence
    periodic_pose_refresh: bool = Field(default=False, description="Inject txt2img every 8 beats for fresh poses")

    # FLUX-specific settings
    flux_anchored_mode: bool = Field(
        default=True,
        description="Use first txt2img as stable reference (avoids feedback drift). Disable for abstract/chaotic mode."
    )

    # Audio mapping configuration
    mapping_config: Optional[MappingConfig] = Field(default=None, description="Custom audio-to-parameter mapping")

    # Lyric detection settings
    enable_lyrics: bool = Field(default=False, description="Enable lyric detection and keyword injection")
    lyric_driven_mode: bool = Field(default=False, description="Experimental: generate prompts entirely from lyrics")

    # Base image for img2img initialization
    base_image: Optional[str] = Field(default=None, description="Base64-encoded PNG/JPEG image for img2img starting point")
    lock_to_base_image: bool = Field(default=False, description="Always use base image as input (no feedback loop)")

    # Custom LoRA configuration (overrides global defaults if set)
    loras: Optional[list[dict]] = Field(
        default=None,
        description="Custom LoRAs: [{path: str, weight: float (0-1), name: str (optional)}]"
    )


class ConfigMessage(BaseModel):
    """Incoming configuration message."""

    type: Literal["config"] = "config"
    config: GenerationConfig


class StartMessage(BaseModel):
    """Start generation message."""

    type: Literal["start"] = "start"


class StopMessage(BaseModel):
    """Stop generation message."""

    type: Literal["stop"] = "stop"


class ResetMessage(BaseModel):
    """Reset feedback loop message."""

    type: Literal["reset"] = "reset"


class RefreshAnchorMessage(BaseModel):
    """Refresh FLUX anchor frame message.

    Clears the anchor frame so the next generation will be txt2img,
    creating a fresh visual reference. Useful for manually refreshing
    the visual style without a full reset.
    """

    type: Literal["refresh_anchor"] = "refresh_anchor"


class SwitchBackendMessage(BaseModel):
    """Switch image generation backend at runtime.

    This will:
    1. Stop generation if running
    2. Clean up current pipeline and free VRAM
    3. Initialize the new backend
    4. Resume generation if it was running
    """

    type: Literal["switch_backend"] = "switch_backend"
    backend_id: str = Field(description="Backend ID to switch to (e.g., 'stream_diffusion', 'flux_klein')")


class MappingMessage(BaseModel):
    """Incoming mapping configuration update message."""

    type: Literal["mapping"] = "mapping"
    mapping_config: MappingConfig


# Lyric detection messages
class RawAudioChunk(BaseModel):
    """Incoming raw audio data for lyric detection."""

    type: Literal["audio_chunk"] = "audio_chunk"
    audio_data: str = Field(description="Base64-encoded 16-bit PCM audio")
    sample_rate: int = Field(default=16000, description="Sample rate (16000 or 48000)")
    timestamp: float = Field(description="Timestamp in seconds")


class LyricUpdate(BaseModel):
    """Outgoing lyric detection update."""

    type: Literal["lyrics"] = "lyrics"
    lyrics: LyricInfo


class LearnFingerprintMessage(BaseModel):
    """Save the current audio fingerprint to the database for later LRC import."""

    type: Literal["learn_fingerprint"] = "learn_fingerprint"
    title: Optional[str] = Field(default=None, description="Song title (optional, can be set later)")
    artist: Optional[str] = Field(default=None, description="Artist name (optional)")


class LearnFingerprintResponse(BaseModel):
    """Response after saving a fingerprint."""

    type: Literal["fingerprint_saved"] = "fingerprint_saved"
    success: bool
    song_id: Optional[int] = Field(default=None, description="Database ID of the saved song")
    fingerprint_hash: Optional[str] = Field(default=None, description="First 32 chars of fingerprint")
    message: str


class ResetLyricsMessage(BaseModel):
    """Reset lyric detection pipeline for a new song."""

    type: Literal["reset_lyrics"] = "reset_lyrics"


# Story-driven prompt generation messages
class StorySceneConfig(BaseModel):
    """Scene definition for story messages (matches schema.SceneDefinition)."""

    id: str
    base_prompt: str
    negative_prompt: Optional[str] = None
    duration_frames: int = 120
    trigger: str = "time"  # time, beat_count, energy_drop, energy_peak
    trigger_value: float = 0
    energy_high_prompt: Optional[str] = None
    energy_low_prompt: Optional[str] = None
    energy_blend_range: tuple[float, float] = (0.3, 0.6)
    beat_prompt_modifier: Optional[str] = None
    transition: str = "crossfade"  # cut, crossfade, zoom_in, zoom_out
    transition_frames: int = 30


class StoryConfig(BaseModel):
    """Story configuration for loading via WebSocket."""

    name: str
    description: Optional[str] = None
    default_negative_prompt: str = "blurry, distorted, low quality, text, watermark"
    scenes: list[StorySceneConfig]
    loop: bool = False
    audio_reactive_keywords: bool = True
    base_seed: Optional[int] = None


class StoryLoadMessage(BaseModel):
    """Load a story for dynamic prompt generation."""

    type: Literal["story_load"] = "story_load"
    story: StoryConfig


class StoryLoadPresetMessage(BaseModel):
    """Load a preset story by name."""

    type: Literal["story_load_preset"] = "story_load_preset"
    preset_name: str  # skiing_adventure, dancing_figure, etc.


class StoryControlMessage(BaseModel):
    """Control story playback."""

    type: Literal["story_control"] = "story_control"
    action: Literal["play", "pause", "skip_next", "skip_prev", "restart", "stop"]


class StoryUnloadMessage(BaseModel):
    """Unload the current story."""

    type: Literal["story_unload"] = "story_unload"


class StoryStateResponse(BaseModel):
    """Story state update sent to client."""

    type: Literal["story_state"] = "story_state"
    story_name: str
    current_scene_idx: int
    current_scene_id: str
    frame_in_scene: int
    beat_count_in_scene: int
    is_transitioning: bool
    transition_progress: float
    is_playing: bool
    is_complete: bool
    total_scenes: int
    scene_progress: float


class FrameMetadata(BaseModel):
    """Metadata for a generated frame."""

    type: Literal["frame"] = "frame"
    frame_id: int
    timestamp: float
    width: int
    height: int
    format: Literal["jpeg"] = "jpeg"
    generation_params: dict
    system_stats: Optional["SystemStats"] = None


class BackendInfo(BaseModel):
    """Information about an available image generation backend."""

    id: str = Field(description="Backend identifier (e.g., 'stream_diffusion', 'flux_klein')")
    name: str = Field(description="Display name (e.g., 'StreamDiffusion', 'FLUX Klein')")
    description: str = Field(default="", description="Brief description of the backend")
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of capabilities: controlnet, lora, taesd, temporal_coherence, acceleration"
    )
    fps_range: tuple[float, float] = Field(
        default=(1.0, 30.0),
        description="Expected FPS range (min, max) for this backend"
    )


# Define available backends
AUDIO_REACTIVE_BACKEND = BackendInfo(
    id="audio_reactive",
    name="Audio Reactive",
    description="SD 1.5 + LCM optimized for audio reactivity (~8-12 FPS)",
    capabilities=["lora", "taesd", "seed_control", "strength_control"],
    fps_range=(6.0, 12.0),
)

STREAM_DIFFUSION_BACKEND = BackendInfo(
    id="stream_diffusion",
    name="StreamDiffusion",
    description="SD 1.5 real-time smooth streaming (~15-20 FPS)",
    capabilities=["controlnet", "lora", "taesd", "temporal_coherence", "acceleration"],
    fps_range=(5.0, 20.0),
)

FLUX_KLEIN_BACKEND = BackendInfo(
    id="flux_klein",
    name="FLUX Klein",
    description="FLUX.2 [klein] high-quality generation (~1-3 FPS)",
    capabilities=["prompt_modulation"],
    fps_range=(1.0, 3.0),
)

AVAILABLE_BACKENDS = [AUDIO_REACTIVE_BACKEND, STREAM_DIFFUSION_BACKEND, FLUX_KLEIN_BACKEND]


class ServerConfig(BaseModel):
    """Server-side configuration info (read-only from client perspective)."""

    acceleration: str = Field(default="lcm", description="Acceleration method: lcm, hyper-sd, or none")
    hyper_sd_steps: Optional[int] = Field(default=None, description="Hyper-SD step variant (1, 2, 4, or 8)")
    model: str = Field(default="", description="Base model ID")
    # Backend info
    current_backend: str = Field(default="audio_reactive", description="Currently active backend ID")
    available_backends: list[BackendInfo] = Field(
        default_factory=lambda: AVAILABLE_BACKENDS,
        description="List of available backends"
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities of the current backend"
    )


class SystemStats(BaseModel):
    """System resource usage stats."""

    cpu_percent: float = Field(description="CPU usage percentage (0-100)")
    ram_used_gb: float = Field(description="RAM used in GB")
    ram_total_gb: float = Field(description="Total RAM in GB")
    gpu_util: Optional[float] = Field(default=None, description="GPU utilization percentage (0-100)")
    vram_used_gb: Optional[float] = Field(default=None, description="VRAM used in GB")
    vram_total_gb: Optional[float] = Field(default=None, description="Total VRAM in GB")


class StatusMessage(BaseModel):
    """Status message from server."""

    type: Literal["status"] = "status"
    status: Literal["connected", "generating", "stopped", "error", "initializing", "switching"]
    message: Optional[str] = None
    fps: Optional[float] = None
    server_config: Optional[ServerConfig] = None
    system_stats: Optional[SystemStats] = None


class FpsMessage(BaseModel):
    """Lightweight FPS update message."""

    type: Literal["fps"] = "fps"
    fps: float


class ErrorMessage(BaseModel):
    """Error message from server."""

    type: Literal["error"] = "error"
    error: str
    code: Optional[str] = None


# Type union for incoming messages
IncomingMessage = (
    MetricsMessage | ConfigMessage | StartMessage | StopMessage | ResetMessage |
    MappingMessage | RawAudioChunk |
    StoryLoadMessage | StoryLoadPresetMessage | StoryControlMessage | StoryUnloadMessage
)

# Type union for outgoing messages
OutgoingMessage = FrameMetadata | StatusMessage | ErrorMessage | StoryStateResponse | LyricUpdate | FpsMessage
