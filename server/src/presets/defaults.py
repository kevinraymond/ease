"""Default mapping presets."""

from ..mapping.audio_mapper import MappingPreset, ParameterMapping
from ..mapping.parameter_curves import CurveType
from ..mapping.prompt_modulator import ModulationConfig


# Note: REACTIVE_ABSTRACT removed in favor of REACTIVE (renamed from PULSING)

VJ_INTENSE = MappingPreset(
    name="vj_intense",
    description="High-energy visuals with dramatic beat synchronization",
    strength_mapping=ParameterMapping(
        audio_feature="raw_bass",
        output_min=0.2,
        output_max=0.45,  # Still higher than others for more energy
        curve=CurveType.EXPONENTIAL,
    ),
    modulation_config=ModulationConfig(
        bass_threshold=0.5,
        mid_threshold=0.4,
        treble_threshold=0.4,
        loud_threshold=0.6,
        quiet_threshold=0.2,
        bass_keywords=["explosive", "pulsing", "powerful", "intense"],
        beat_keywords=["flash", "burst", "impact", "strobe"],
        enable_motion=True,
    ),
    beat_seed_jump=True,  # New subject on beats for VJ style
    beat_strength_boost=0.1,
    beat_cooldown_ms=200,  # Fast reactivity for VJ
)

DREAMSCAPE = MappingPreset(
    name="dreamscape",
    description="Smooth, ethereal visuals with no beat synchronization",
    strength_mapping=ParameterMapping(
        audio_feature="rms",
        output_min=0.15,  # Minimum safe for Hyper-SD acceleration
        output_max=0.25,  # Very gentle
        curve=CurveType.LOGARITHMIC,
    ),
    modulation_config=ModulationConfig(
        bass_threshold=0.8,
        mid_threshold=0.7,
        treble_threshold=0.7,
        loud_threshold=0.85,
        quiet_threshold=0.1,
        bass_keywords=["deep", "resonant"],
        mid_keywords=["flowing", "organic"],
        treble_keywords=["shimmering", "ethereal"],
        beat_keywords=[],
        loud_keywords=["luminous"],
        quiet_keywords=["dreamy", "soft", "misty"],
        enable_motion=False,
    ),
    beat_seed_jump=False,
    beat_strength_boost=0.0,
    beat_cooldown_ms=500,  # Slower for dreamy visuals
)

# Note: MINIMAL_RESPONSE removed - too subtle for practical use

DANCER = MappingPreset(
    name="dancer",
    description="Optimized for making subjects dance to the music",
    strength_mapping=ParameterMapping(
        audio_feature="bass",
        output_min=0.25,
        output_max=0.631,
        curve=CurveType.EASE_IN_OUT,
    ),
    modulation_config=ModulationConfig(
        bass_threshold=0.45,  # Lowered for more responsive triggering
        mid_threshold=0.6,
        treble_threshold=0.6,
        loud_threshold=0.65,
        quiet_threshold=0.2,
        beat_keywords=["shifting", "transforming", "dynamic motion"],
        motion_keywords=["flowing", "in motion", "fluid", "dynamic"],
        bass_keywords=["shift", "pulse", "change"],
        enable_motion=True,
    ),
    beat_seed_jump=False,
    beat_strength_boost=0.15,
    txt2img_on_beat_interval=0,  # Controlled by UI toggle (periodic_pose_refresh)
    beat_cooldown_ms=300,
)

REACTIVE = MappingPreset(
    name="reactive",
    description="Balanced audio response - the default preset",
    strength_mapping=ParameterMapping(
        audio_feature="rms",  # Uses volume for smooth response
        output_min=0.20,
        output_max=0.45,
        curve=CurveType.EASE_IN_OUT,
    ),
    modulation_config=ModulationConfig(
        bass_threshold=0.55,
        mid_threshold=0.55,
        treble_threshold=0.6,
        loud_threshold=0.7,
        quiet_threshold=0.15,
        beat_keywords=["pulsing", "breathing", "rhythmic"],
        enable_motion=False,
    ),
    beat_seed_jump=False,  # Keep subject consistent
    beat_strength_boost=0.18,  # Strong beat boost for visible pulse
    beat_cooldown_ms=300,  # Balanced cooldown
)

# Alias for backwards compatibility
PULSING = REACTIVE


DEFAULT_PRESETS: dict[str, MappingPreset] = {
    "reactive": REACTIVE,
    "dancer": DANCER,
    "vj_intense": VJ_INTENSE,
    "dreamscape": DREAMSCAPE,
    # Backwards compatibility aliases
    "pulsing": REACTIVE,  # Renamed to reactive
    "reactive_abstract": REACTIVE,  # Removed, maps to reactive
    "minimal": DREAMSCAPE,  # Removed, maps to closest alternative
}


def get_preset(name: str) -> MappingPreset:
    """Get a preset by name, returns reactive if not found."""
    return DEFAULT_PRESETS.get(name, REACTIVE)
