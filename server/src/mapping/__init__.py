"""Mapping module - Audio features to generation parameters."""

from .audio_mapper import AudioMapper, MappingPreset, ParameterMapping, GenerationParams
from .prompt_modulator import PromptModulator, ModulationConfig, FluxPromptModulator, FluxModulationConfig
from .color_organ_modulator import ColorOrganModulator, ColorOrganConfig, get_color_organ_preset, COLOR_ORGAN_PRESETS
from .parameter_curves import CurveType, map_range

__all__ = [
    "AudioMapper",
    "MappingPreset",
    "ParameterMapping",
    "GenerationParams",
    "PromptModulator",
    "ModulationConfig",
    "FluxPromptModulator",
    "FluxModulationConfig",
    "ColorOrganModulator",
    "ColorOrganConfig",
    "get_color_organ_preset",
    "COLOR_ORGAN_PRESETS",
    "CurveType",
    "map_range",
]
