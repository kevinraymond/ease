"""Story controller state machine for dynamic prompt generation."""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from .schema import (
    StoryScript,
    SceneDefinition,
    SceneTrigger,
    SceneTransition,
    StoryState,
)

# Import AudioMetrics - try relative first, fall back to absolute for testing
try:
    from ..server.protocol import AudioMetrics
except (ImportError, ValueError):
    from server.protocol import AudioMetrics  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass
class PromptOutput:
    """Output from the story controller for a single frame."""

    base_prompt: str
    negative_prompt: str
    # Metadata for debugging/UI
    scene_id: str = ""
    is_transitioning: bool = False
    transition_progress: float = 0.0
    energy_blend_factor: float = 0.0  # How much energy variant is blended in


class StoryController:
    """State machine for story-driven prompt generation.

    Manages scene progression, audio-influenced prompt selection,
    and smooth transitions between scenes.
    """

    def __init__(self, story: Optional[StoryScript] = None):
        self._story: Optional[StoryScript] = story
        self._current_scene_idx: int = 0
        self._frame_in_scene: int = 0
        self._beat_count_in_scene: int = 0
        self._is_transitioning: bool = False
        self._transition_progress: float = 0.0
        self._transition_from_prompt: str = ""
        self._transition_from_negative: str = ""
        self._is_playing: bool = True
        self._is_complete: bool = False

        # Smoothed energy for stable variant selection
        self._smoothed_energy: float = 0.5
        self._energy_smoothing: float = 0.1  # Lower = smoother

    @property
    def current_scene(self) -> Optional[SceneDefinition]:
        """Get the current scene, or None if no story loaded."""
        if not self._story or not self._story.scenes:
            return None
        if self._current_scene_idx >= len(self._story.scenes):
            return None
        return self._story.scenes[self._current_scene_idx]

    @property
    def next_scene(self) -> Optional[SceneDefinition]:
        """Get the next scene (for transitions), or None if at end."""
        if not self._story or not self._story.scenes:
            return None
        next_idx = self._current_scene_idx + 1
        if next_idx >= len(self._story.scenes):
            if self._story.loop:
                return self._story.scenes[0]
            return None
        return self._story.scenes[next_idx]

    @property
    def has_story(self) -> bool:
        """Check if a story is loaded."""
        return self._story is not None and len(self._story.scenes) > 0

    @property
    def audio_reactive_keywords_enabled(self) -> bool:
        """Check if PromptModulator should layer keywords on top."""
        if not self._story:
            return True  # Default behavior when no story
        return self._story.audio_reactive_keywords

    def load_story(self, story: StoryScript) -> None:
        """Load a new story and reset state."""
        self._story = story
        self.reset()
        logger.info(f"Loaded story '{story.name}' with {len(story.scenes)} scenes")

    def unload_story(self) -> None:
        """Unload the current story."""
        self._story = None
        self.reset()
        logger.info("Story unloaded")

    def reset(self) -> None:
        """Reset to the beginning of the story."""
        self._current_scene_idx = 0
        self._frame_in_scene = 0
        self._beat_count_in_scene = 0
        self._is_transitioning = False
        self._transition_progress = 0.0
        self._transition_from_prompt = ""
        self._transition_from_negative = ""
        self._is_playing = True
        self._is_complete = False
        self._smoothed_energy = 0.5
        logger.debug("Story controller reset")

    def play(self) -> None:
        """Resume playback."""
        self._is_playing = True
        logger.debug("Story playback resumed")

    def pause(self) -> None:
        """Pause playback (freeze at current frame)."""
        self._is_playing = False
        logger.debug("Story playback paused")

    def skip_to_next_scene(self) -> None:
        """Skip to the next scene immediately."""
        if not self._story:
            return

        next_idx = self._current_scene_idx + 1
        if next_idx >= len(self._story.scenes):
            if self._story.loop:
                next_idx = 0
            else:
                self._is_complete = True
                logger.info("Story complete (no more scenes)")
                return

        self._current_scene_idx = next_idx
        self._frame_in_scene = 0
        self._beat_count_in_scene = 0
        self._is_transitioning = False
        self._transition_progress = 0.0
        scene = self.current_scene
        logger.info(f"Skipped to scene {next_idx}: {scene.id if scene else 'none'}")

    def skip_to_prev_scene(self) -> None:
        """Skip to the previous scene immediately."""
        if not self._story:
            return

        prev_idx = max(0, self._current_scene_idx - 1)
        self._current_scene_idx = prev_idx
        self._frame_in_scene = 0
        self._beat_count_in_scene = 0
        self._is_transitioning = False
        self._transition_progress = 0.0
        self._is_complete = False  # Un-complete if we were done
        scene = self.current_scene
        logger.info(f"Skipped to scene {prev_idx}: {scene.id if scene else 'none'}")

    def get_prompt(self, metrics: AudioMetrics) -> PromptOutput:
        """Get the prompt for the current frame based on story state and audio.

        This is the main entry point called each frame to get the dynamic prompt.

        Args:
            metrics: Current audio analysis metrics

        Returns:
            PromptOutput with base_prompt, negative_prompt, and metadata
        """
        # No story loaded - return empty (caller should use config.base_prompt)
        if not self._story or not self.current_scene:
            return PromptOutput(
                base_prompt="",
                negative_prompt="",
            )

        scene = self.current_scene

        # Update smoothed energy
        current_energy = metrics.rms
        self._smoothed_energy = (
            self._smoothed_energy * (1 - self._energy_smoothing) +
            current_energy * self._energy_smoothing
        )

        # Select prompt based on audio energy
        prompt, energy_factor = self._select_energy_variant(scene, self._smoothed_energy)

        # Add beat modifier if applicable
        if metrics.is_beat and scene.beat_prompt_modifier:
            prompt = f"{prompt}, {scene.beat_prompt_modifier}"

        # Add camera keywords during zoom transitions
        if self._is_transitioning and scene.transition in (
            SceneTransition.ZOOM_IN, SceneTransition.ZOOM_OUT
        ):
            prompt = self._add_camera_keywords(prompt, scene.transition, self._transition_progress)

        # Handle crossfade transition blending
        if self._is_transitioning and scene.transition == SceneTransition.CROSSFADE:
            next_scene = self.next_scene
            if next_scene:
                next_prompt, _ = self._select_energy_variant(next_scene, self._smoothed_energy)
                prompt = self._blend_prompts(prompt, next_prompt, self._transition_progress)

        # Get negative prompt
        negative = scene.negative_prompt or self._story.default_negative_prompt

        return PromptOutput(
            base_prompt=prompt,
            negative_prompt=negative,
            scene_id=scene.id,
            is_transitioning=self._is_transitioning,
            transition_progress=self._transition_progress,
            energy_blend_factor=energy_factor,
        )

    def advance(self, metrics: AudioMetrics) -> None:
        """Advance the story state by one frame.

        Call this after get_prompt() each frame to progress the story.

        Args:
            metrics: Current audio analysis metrics
        """
        if not self._is_playing or not self._story or self._is_complete:
            return

        scene = self.current_scene
        if not scene:
            return

        # Increment frame counter
        self._frame_in_scene += 1

        # Track beats
        if metrics.is_beat:
            self._beat_count_in_scene += 1

        # Handle ongoing transition
        if self._is_transitioning:
            self._transition_progress += 1.0 / scene.transition_frames
            if self._transition_progress >= 1.0:
                self._complete_transition()
            return

        # Check if we should start a transition
        if self._should_transition(scene, metrics):
            self._start_transition(scene)

    def get_state(self) -> StoryState:
        """Get current story state for UI updates."""
        if not self._story:
            return StoryState(
                story_name="",
                is_playing=False,
            )

        scene = self.current_scene
        scene_progress = 0.0
        if scene and scene.duration_frames > 0:
            scene_progress = min(1.0, self._frame_in_scene / scene.duration_frames)

        return StoryState(
            story_name=self._story.name,
            current_scene_idx=self._current_scene_idx,
            current_scene_id=scene.id if scene else "",
            frame_in_scene=self._frame_in_scene,
            beat_count_in_scene=self._beat_count_in_scene,
            is_transitioning=self._is_transitioning,
            transition_progress=self._transition_progress,
            is_playing=self._is_playing,
            is_complete=self._is_complete,
            total_scenes=len(self._story.scenes),
            scene_progress=scene_progress,
        )

    def _select_energy_variant(
        self,
        scene: SceneDefinition,
        energy: float,
    ) -> Tuple[str, float]:
        """Select prompt variant based on audio energy with smooth blending.

        Returns:
            Tuple of (selected_prompt, blend_factor)
            blend_factor: 0 = base, -1 = low variant, +1 = high variant
        """
        low_thresh, high_thresh = scene.energy_blend_range

        # High energy zone
        if energy > high_thresh and scene.energy_high_prompt:
            blend = min(1.0, (energy - high_thresh) / (1.0 - high_thresh))
            prompt = self._blend_prompts(
                scene.base_prompt,
                scene.energy_high_prompt,
                blend,
            )
            return prompt, blend

        # Low energy zone
        if energy < low_thresh and scene.energy_low_prompt:
            blend = min(1.0, (low_thresh - energy) / low_thresh)
            prompt = self._blend_prompts(
                scene.base_prompt,
                scene.energy_low_prompt,
                blend,
            )
            return prompt, -blend

        # Neutral zone - use base prompt
        return scene.base_prompt, 0.0

    def _blend_prompts(self, prompt_a: str, prompt_b: str, blend: float) -> str:
        """Blend two prompts using CLIP weighting syntax.

        At blend=0, returns prompt_a. At blend=1, returns prompt_b.
        In between, weights both with (prompt:weight) syntax.
        """
        if blend <= 0.05:
            return prompt_a
        if blend >= 0.95:
            return prompt_b

        # Use CLIP weighting to blend
        weight_a = 1.0 - blend
        weight_b = blend

        # Wrap prompts with weights (only if significantly different from 1.0)
        if weight_a > 0.1:
            part_a = f"({prompt_a}:{weight_a:.2f})"
        else:
            part_a = ""

        if weight_b > 0.1:
            part_b = f"({prompt_b}:{weight_b:.2f})"
        else:
            part_b = ""

        # Combine
        if part_a and part_b:
            return f"{part_a}, {part_b}"
        return part_a or part_b or prompt_a

    def _add_camera_keywords(
        self,
        prompt: str,
        transition: SceneTransition,
        progress: float,
    ) -> str:
        """Add camera movement keywords during zoom transitions."""
        if transition == SceneTransition.ZOOM_IN:
            if progress < 0.3:
                return f"{prompt}, zooming in, camera pushing forward"
            elif progress < 0.7:
                return f"{prompt}, close-up, detailed view"
            else:
                return f"{prompt}, extreme close-up, intimate framing"

        elif transition == SceneTransition.ZOOM_OUT:
            if progress < 0.3:
                return f"{prompt}, zooming out, camera pulling back"
            elif progress < 0.7:
                return f"{prompt}, wide shot, establishing view"
            else:
                return f"{prompt}, panoramic view, vast landscape"

        return prompt

    def _should_transition(self, scene: SceneDefinition, metrics: AudioMetrics) -> bool:
        """Check if current scene should start transitioning to next."""
        trigger = scene.trigger

        if trigger == SceneTrigger.TIME:
            return self._frame_in_scene >= scene.duration_frames

        elif trigger == SceneTrigger.BEAT_COUNT:
            return self._beat_count_in_scene >= scene.trigger_value

        elif trigger == SceneTrigger.ENERGY_DROP:
            return self._smoothed_energy < scene.trigger_value

        elif trigger == SceneTrigger.ENERGY_PEAK:
            return self._smoothed_energy > scene.trigger_value

        return False

    def _start_transition(self, scene: SceneDefinition) -> None:
        """Start a transition to the next scene."""
        # Store current prompt for blending
        self._transition_from_prompt = scene.base_prompt
        self._transition_from_negative = scene.negative_prompt or ""

        self._is_transitioning = True
        self._transition_progress = 0.0

        # For cut transitions, complete immediately
        if scene.transition == SceneTransition.CUT:
            self._complete_transition()
        else:
            logger.info(
                f"Starting {scene.transition.value} transition from scene "
                f"'{scene.id}' ({scene.transition_frames} frames)"
            )

    def _complete_transition(self) -> None:
        """Complete the transition and move to next scene."""
        if not self._story:
            return

        next_idx = self._current_scene_idx + 1

        if next_idx >= len(self._story.scenes):
            if self._story.loop:
                next_idx = 0
                logger.info("Story looping back to first scene")
            else:
                self._is_complete = True
                logger.info("Story complete")
                return

        self._current_scene_idx = next_idx
        self._frame_in_scene = 0
        self._beat_count_in_scene = 0
        self._is_transitioning = False
        self._transition_progress = 0.0
        self._transition_from_prompt = ""
        self._transition_from_negative = ""

        scene = self.current_scene
        logger.info(f"Transitioned to scene {next_idx}: '{scene.id if scene else 'none'}'")
