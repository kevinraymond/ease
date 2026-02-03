"""Story and scene data models for dynamic prompt generation."""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


class SceneTrigger(str, Enum):
    """How a scene transitions to the next scene."""

    TIME = "time"          # After duration_frames
    BEAT_COUNT = "beat_count"  # After N beats
    ENERGY_DROP = "energy_drop"  # When energy drops below threshold
    ENERGY_PEAK = "energy_peak"  # When energy exceeds threshold


class SceneTransition(str, Enum):
    """Visual transition type between scenes."""

    CUT = "cut"            # Instant switch
    CROSSFADE = "crossfade"  # Blend prompts
    ZOOM_IN = "zoom_in"    # Add zoom-in camera keywords
    ZOOM_OUT = "zoom_out"  # Add zoom-out camera keywords


class SceneDefinition(BaseModel):
    """Defines a single scene in a story with audio-influenced variations."""

    id: str = Field(description="Unique scene identifier")
    base_prompt: str = Field(description="Core visual description for this scene")
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Scene-specific negative prompt (overrides story default)"
    )

    # Duration / Trigger
    duration_frames: int = Field(
        default=120,
        ge=1,
        description="Scene duration in frames (~4 seconds at 30fps)"
    )
    trigger: SceneTrigger = Field(
        default=SceneTrigger.TIME,
        description="What triggers transition to next scene"
    )
    trigger_value: float = Field(
        default=0,
        ge=0,
        description="Trigger threshold: beat count or energy level (0-1)"
    )

    # Audio-Influenced Variants (key feature!)
    energy_high_prompt: Optional[str] = Field(
        default=None,
        description="Prompt variant when audio energy is high (>0.6 RMS)"
    )
    energy_low_prompt: Optional[str] = Field(
        default=None,
        description="Prompt variant when audio energy is low (<0.3 RMS)"
    )
    energy_blend_range: tuple[float, float] = Field(
        default=(0.3, 0.6),
        description="Energy range (low_threshold, high_threshold) for prompt blending"
    )

    # Additional audio-reactive modifiers
    beat_prompt_modifier: Optional[str] = Field(
        default=None,
        description="Keywords to add on beat events"
    )

    # Transition to next scene
    transition: SceneTransition = Field(
        default=SceneTransition.CROSSFADE,
        description="How to transition to next scene"
    )
    transition_frames: int = Field(
        default=30,
        ge=1,
        description="Number of frames for transition"
    )


class StoryScript(BaseModel):
    """Complete story definition with multiple scenes."""

    name: str = Field(description="Story identifier")
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the story"
    )

    # Default prompts (can be overridden per-scene)
    default_negative_prompt: str = Field(
        default="blurry, distorted, low quality, text, watermark",
        description="Default negative prompt for all scenes"
    )

    # Scenes
    scenes: list[SceneDefinition] = Field(
        default_factory=list,
        description="Ordered list of scenes in the story"
    )

    # Playback options
    loop: bool = Field(
        default=False,
        description="Whether to loop back to first scene after last"
    )
    audio_reactive_keywords: bool = Field(
        default=True,
        description="Layer PromptModulator keywords on top of story prompts"
    )

    # Story-level settings
    base_seed: Optional[int] = Field(
        default=None,
        description="Fixed seed for the story (None = random)"
    )

    model_config = {"extra": "allow"}


class StoryState(BaseModel):
    """Current playback state of a story."""

    story_name: str
    current_scene_idx: int = 0
    current_scene_id: str = ""
    frame_in_scene: int = 0
    beat_count_in_scene: int = 0
    is_transitioning: bool = False
    transition_progress: float = 0.0  # 0-1
    is_playing: bool = True
    is_complete: bool = False

    # For UI display
    total_scenes: int = 0
    scene_progress: float = 0.0  # 0-1 progress through current scene


class StoryLoadRequest(BaseModel):
    """Request to load a story."""

    type: Literal["story_load"] = "story_load"
    story: StoryScript


class StoryControlRequest(BaseModel):
    """Request to control story playback."""

    type: Literal["story_control"] = "story_control"
    action: Literal["play", "pause", "skip_next", "skip_prev", "restart", "stop"]


class StoryStateMessage(BaseModel):
    """Story state update sent to client."""

    type: Literal["story_state"] = "story_state"
    state: StoryState
