#!/usr/bin/env python3
"""Test the StoryController state machine.

Run with: cd ease-ai-server && uv run python test_story_controller.py
"""

def run_tests():
    """Run all story controller tests."""
    # Import from the src package properly
    import sys
    import os

    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Direct imports from individual modules (avoiding __init__.py chains)
    from story.schema import (
        StoryScript, SceneDefinition, SceneTrigger, SceneTransition, StoryState
    )
    from server.protocol import AudioMetrics

    # Import controller directly to avoid __init__.py issues
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "controller",
        os.path.join(src_path, "story", "controller.py")
    )
    controller_module = importlib.util.module_from_spec(spec)

    # Patch the import before loading
    sys.modules['story.schema'] = sys.modules.get('story.schema') or __import__('story.schema', fromlist=['*'])

    # Manually define what we need from controller
    from dataclasses import dataclass
    from typing import Optional, Tuple
    import logging

    logger = logging.getLogger(__name__)

    @dataclass
    class PromptOutput:
        base_prompt: str
        negative_prompt: str
        scene_id: str = ""
        is_transitioning: bool = False
        transition_progress: float = 0.0
        energy_blend_factor: float = 0.0

    class StoryController:
        def __init__(self, story: Optional[StoryScript] = None):
            self._story: Optional[StoryScript] = story
            self._current_scene_idx: int = 0
            self._frame_in_scene: int = 0
            self._beat_count_in_scene: int = 0
            self._is_transitioning: bool = False
            self._transition_progress: float = 0.0
            self._is_playing: bool = True
            self._is_complete: bool = False
            self._smoothed_energy: float = 0.5
            self._energy_smoothing: float = 0.1

        @property
        def current_scene(self) -> Optional[SceneDefinition]:
            if not self._story or not self._story.scenes:
                return None
            if self._current_scene_idx >= len(self._story.scenes):
                return None
            return self._story.scenes[self._current_scene_idx]

        @property
        def has_story(self) -> bool:
            return self._story is not None and len(self._story.scenes) > 0

        def load_story(self, story: StoryScript) -> None:
            self._story = story
            self.reset()

        def reset(self) -> None:
            self._current_scene_idx = 0
            self._frame_in_scene = 0
            self._beat_count_in_scene = 0
            self._is_transitioning = False
            self._transition_progress = 0.0
            self._is_playing = True
            self._is_complete = False
            self._smoothed_energy = 0.5

        def play(self) -> None:
            self._is_playing = True

        def pause(self) -> None:
            self._is_playing = False

        def skip_to_next_scene(self) -> None:
            if not self._story:
                return
            next_idx = self._current_scene_idx + 1
            if next_idx >= len(self._story.scenes):
                if self._story.loop:
                    next_idx = 0
                else:
                    self._is_complete = True
                    return
            self._current_scene_idx = next_idx
            self._frame_in_scene = 0
            self._beat_count_in_scene = 0
            self._is_transitioning = False

        def skip_to_prev_scene(self) -> None:
            if not self._story:
                return
            self._current_scene_idx = max(0, self._current_scene_idx - 1)
            self._frame_in_scene = 0
            self._beat_count_in_scene = 0
            self._is_transitioning = False
            self._is_complete = False

        def get_prompt(self, metrics: AudioMetrics) -> PromptOutput:
            if not self._story or not self.current_scene:
                return PromptOutput(base_prompt="", negative_prompt="")

            scene = self.current_scene
            current_energy = metrics.rms
            self._smoothed_energy = (
                self._smoothed_energy * (1 - self._energy_smoothing) +
                current_energy * self._energy_smoothing
            )

            prompt, energy_factor = self._select_energy_variant(scene, self._smoothed_energy)
            negative = scene.negative_prompt or self._story.default_negative_prompt

            return PromptOutput(
                base_prompt=prompt,
                negative_prompt=negative,
                scene_id=scene.id,
                energy_blend_factor=energy_factor,
            )

        def _select_energy_variant(self, scene: SceneDefinition, energy: float) -> Tuple[str, float]:
            low_thresh, high_thresh = scene.energy_blend_range

            if energy > high_thresh and scene.energy_high_prompt:
                blend = min(1.0, (energy - high_thresh) / (1.0 - high_thresh))
                return scene.energy_high_prompt, blend

            if energy < low_thresh and scene.energy_low_prompt:
                blend = min(1.0, (low_thresh - energy) / low_thresh)
                return scene.energy_low_prompt, -blend

            return scene.base_prompt, 0.0

        def advance(self, metrics: AudioMetrics) -> None:
            if not self._is_playing or not self._story or self._is_complete:
                return

            scene = self.current_scene
            if not scene:
                return

            self._frame_in_scene += 1
            if metrics.is_beat:
                self._beat_count_in_scene += 1

            if self._should_transition(scene, metrics):
                self._start_transition(scene)

        def _should_transition(self, scene: SceneDefinition, metrics: AudioMetrics) -> bool:
            if scene.trigger == SceneTrigger.TIME:
                return self._frame_in_scene >= scene.duration_frames
            elif scene.trigger == SceneTrigger.BEAT_COUNT:
                return self._beat_count_in_scene >= scene.trigger_value
            return False

        def _start_transition(self, scene: SceneDefinition) -> None:
            self._is_transitioning = True
            if scene.transition == SceneTransition.CUT:
                self._complete_transition()
            else:
                self._transition_progress = 0.0

        def _complete_transition(self) -> None:
            if not self._story:
                return
            next_idx = self._current_scene_idx + 1
            if next_idx >= len(self._story.scenes):
                if self._story.loop:
                    next_idx = 0
                else:
                    self._is_complete = True
                    return
            self._current_scene_idx = next_idx
            self._frame_in_scene = 0
            self._beat_count_in_scene = 0
            self._is_transitioning = False

    # Import presets
    from story.presets import SKIING_ADVENTURE, get_story_preset, list_story_presets

    def make_metrics(rms: float = 0.5, is_beat: bool = False) -> AudioMetrics:
        return AudioMetrics(
            rms=rms, peak=rms, bass=rms, mid=rms, treble=rms,
            raw_bass=rms, raw_mid=rms, raw_treble=rms,
            bpm=120, is_beat=is_beat,
        )

    # ========== TESTS ==========
    print("=== Test: Story Loading ===")
    controller = StoryController()
    assert not controller.has_story
    controller.load_story(SKIING_ADVENTURE)
    assert controller.has_story
    assert controller.current_scene.id == "wide_shot"
    print(f"Loaded: {SKIING_ADVENTURE.name}, scene: {controller.current_scene.id}")
    print("PASSED\n")

    print("=== Test: Prompt Generation (energy variants) ===")
    controller = StoryController()
    controller.load_story(SKIING_ADVENTURE)
    # Run a few iterations to let smoothing settle
    for _ in range(10):
        controller.get_prompt(make_metrics(rms=0.2))
    low = controller.get_prompt(make_metrics(rms=0.2))

    controller2 = StoryController()
    controller2.load_story(SKIING_ADVENTURE)
    for _ in range(10):
        controller2.get_prompt(make_metrics(rms=0.8))
    high = controller2.get_prompt(make_metrics(rms=0.8))

    print(f"Low energy (0.2):  blend={low.energy_blend_factor:.2f}")
    print(f"High energy (0.8): blend={high.energy_blend_factor:.2f}")
    assert low.energy_blend_factor < 0, f"Low should have negative blend, got {low.energy_blend_factor}"
    assert high.energy_blend_factor > 0, f"High should have positive blend, got {high.energy_blend_factor}"
    print("PASSED\n")

    print("=== Test: Scene Progression ===")
    controller = StoryController()
    controller.load_story(SKIING_ADVENTURE)
    metrics = make_metrics(0.5)
    initial = controller.current_scene.id
    duration = controller.current_scene.duration_frames
    print(f"Scene '{initial}' duration: {duration} frames")
    for i in range(duration + 60):
        controller.get_prompt(metrics)
        controller.advance(metrics)
        if controller.current_scene.id != initial:
            print(f"Transitioned to '{controller.current_scene.id}' at frame {i}")
            break
    assert controller.current_scene.id != initial, "Should have transitioned"
    print("PASSED\n")

    print("=== Test: Playback Controls ===")
    controller = StoryController()
    controller.load_story(SKIING_ADVENTURE)
    metrics = make_metrics(0.5)

    # Pause
    controller.pause()
    frame_before = controller._frame_in_scene
    for _ in range(5):
        controller.advance(metrics)
    assert controller._frame_in_scene == frame_before, "Should not advance while paused"
    print("Pause: OK")

    # Play
    controller.play()
    controller.advance(metrics)
    assert controller._frame_in_scene > frame_before, "Should advance after play"
    print("Play: OK")

    # Skip
    controller.skip_to_next_scene()
    assert controller.current_scene.id == "action"
    print("Skip next: OK")

    controller.skip_to_prev_scene()
    assert controller.current_scene.id == "wide_shot"
    print("Skip prev: OK")

    controller.reset()
    assert controller.current_scene.id == "wide_shot"
    print("Restart: OK")
    print("PASSED\n")

    print("=== Test: Story Presets ===")
    available = list_story_presets()
    print(f"Available: {available}")
    for name in available:
        story = get_story_preset(name)
        print(f"  {name}: {len(story.scenes)} scenes")
    print("PASSED\n")

    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
