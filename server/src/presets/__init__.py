"""Presets module - Mapping configuration presets."""

from .defaults import (
    DEFAULT_PRESETS,
    REACTIVE,
    VJ_INTENSE,
    DREAMSCAPE,
    DANCER,
    PULSING,  # Alias for REACTIVE (backwards compatibility)
    get_preset,
)
from .types import MappingPreset

__all__ = [
    "DEFAULT_PRESETS",
    "REACTIVE",
    "VJ_INTENSE",
    "DREAMSCAPE",
    "DANCER",
    "PULSING",
    "get_preset",
    "MappingPreset",
]
