"""Story-driven prompt generation system."""

from .schema import (
    SceneDefinition,
    SceneTrigger,
    SceneTransition,
    StoryScript,
    StoryState,
    StoryLoadRequest,
    StoryControlRequest,
    StoryStateMessage,
)
from .controller import StoryController, PromptOutput
from .presets import (
    get_story_preset,
    list_story_presets,
    STORY_PRESETS,
    SKIING_ADVENTURE,
    DANCING_FIGURE,
    ABSTRACT_LANDSCAPE,
    MINIMAL_PORTRAIT,
)

__all__ = [
    # Schema
    "SceneDefinition",
    "SceneTrigger",
    "SceneTransition",
    "StoryScript",
    "StoryState",
    "StoryLoadRequest",
    "StoryControlRequest",
    "StoryStateMessage",
    # Controller
    "StoryController",
    "PromptOutput",
    # Presets
    "get_story_preset",
    "list_story_presets",
    "STORY_PRESETS",
    "SKIING_ADVENTURE",
    "DANCING_FIGURE",
    "ABSTRACT_LANDSCAPE",
    "MINIMAL_PORTRAIT",
]
