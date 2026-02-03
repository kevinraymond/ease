"""Color Organ prompt modulation - classic frequency-to-color mapping for AI generation.

Maps frequency bands directly to color words with intensity-based repetition:
- Bass → Red
- Mid → Green
- Treble → Blue

Higher intensity = more repetitions ("red red red" vs "red"), which increases
the model's attention to that color token.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

from ..server.protocol import AudioMetrics

logger = logging.getLogger(__name__)


@dataclass
class ColorOrganConfig:
    """Configuration for color organ prompt modulation."""

    # Color words for each frequency band (red/yellow/blue for colorblind accessibility)
    bass_color: str = "red"
    mid_color: str = "yellow"
    treble_color: str = "blue"

    # Alternative color sets for variety
    bass_colors: List[str] = field(default_factory=lambda: ["red", "crimson", "scarlet"])
    mid_colors: List[str] = field(default_factory=lambda: ["yellow", "gold", "amber"])
    treble_colors: List[str] = field(default_factory=lambda: ["blue", "cyan", "azure"])

    # Thresholds for activation and repetition
    activation_threshold: float = 0.25  # Minimum level to include color
    repetition_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    max_repetitions: int = 3

    # Intensity modifiers (added when bands are very strong)
    intensity_words: List[str] = field(default_factory=lambda: [
        "glowing", "vibrant", "luminous", "radiant"
    ])
    intensity_threshold: float = 0.8  # Add intensity word when any band exceeds this

    # Additive color mixing - when multiple bands are strong
    # Updated for red/yellow/blue color scheme
    mix_colors: dict = field(default_factory=lambda: {
        ("red", "yellow"): "orange",      # R+Y = Orange
        ("red", "blue"): "purple",        # R+B = Purple
        ("yellow", "blue"): "green",      # Y+B = Green
        ("red", "yellow", "blue"): "white"  # R+Y+B = White
    })
    use_color_mixing: bool = True
    mix_threshold: float = 0.6  # Both colors must exceed this for mix

    # Style suffixes based on overall energy
    low_energy_style: str = "soft glow"
    medium_energy_style: str = "bright lights"
    high_energy_style: str = "blazing neon"

    # Whether to include "light", "lighting", "color" context words
    include_context_words: bool = True
    context_words: List[str] = field(default_factory=lambda: [
        "light", "lighting", "colored light", "neon"
    ])

    # Beat flash - add "flash" or "burst" on beats
    beat_flash_words: List[str] = field(default_factory=lambda: [
        "flash", "burst", "pulse", "strobe"
    ])
    enable_beat_flash: bool = True

    # Smoothing factor for stability (0-1, higher = more smoothing)
    smoothing: float = 0.3


class ColorOrganModulator:
    """Transforms prompts using classic color organ frequency-to-color mapping.

    Creates prompts like:
    - Low bass only: "red light, soft glow"
    - Heavy bass: "red red red, blazing neon, vibrant"
    - Bass + treble: "red red blue, magenta light"
    - All bands high: "red green blue, white light, blazing neon"
    """

    def __init__(self, config: Optional[ColorOrganConfig] = None):
        self.config = config or ColorOrganConfig()

        # Smoothed values for stability
        self._smoothed_bass = 0.0
        self._smoothed_mid = 0.0
        self._smoothed_treble = 0.0

        # Beat tracking
        self._last_beat = False
        self._color_cycle = 0  # For variety in color word selection

    def modulate(self, base_prompt: str, metrics: AudioMetrics) -> str:
        """Generate color organ prompt based on audio metrics.

        Args:
            base_prompt: User's base prompt (will be prepended)
            metrics: Current audio metrics

        Returns:
            Modulated prompt with color organ keywords
        """
        cfg = self.config

        # Smooth the frequency values
        alpha = 1.0 - cfg.smoothing
        self._smoothed_bass = cfg.smoothing * self._smoothed_bass + alpha * metrics.bass
        self._smoothed_mid = cfg.smoothing * self._smoothed_mid + alpha * metrics.mid
        self._smoothed_treble = cfg.smoothing * self._smoothed_treble + alpha * metrics.treble

        bass = self._smoothed_bass
        mid = self._smoothed_mid
        treble = self._smoothed_treble

        parts = []

        # Start with base prompt if provided
        if base_prompt and base_prompt.strip():
            parts.append(base_prompt.strip())

        # Build color words with repetition based on intensity
        color_parts = []
        active_colors = []

        bass_str = self._get_color_repetitions(bass, cfg.bass_color)
        if bass_str:
            color_parts.append(bass_str)
            active_colors.append(cfg.bass_color)

        mid_str = self._get_color_repetitions(mid, cfg.mid_color)
        if mid_str:
            color_parts.append(mid_str)
            active_colors.append(cfg.mid_color)

        treble_str = self._get_color_repetitions(treble, cfg.treble_color)
        if treble_str:
            color_parts.append(treble_str)
            active_colors.append(cfg.treble_color)

        # Add color words
        if color_parts:
            parts.append(" ".join(color_parts))

        # Add color mixing results
        if cfg.use_color_mixing and len(active_colors) >= 2:
            mix_color = self._get_mix_color(bass, mid, treble, active_colors)
            if mix_color:
                parts.append(f"{mix_color} light")

        # Add context words
        if cfg.include_context_words and color_parts:
            context = cfg.context_words[self._color_cycle % len(cfg.context_words)]
            parts.append(context)

        # Add intensity modifier for very strong signals
        max_intensity = max(bass, mid, treble)
        if max_intensity > cfg.intensity_threshold:
            intensity_word = cfg.intensity_words[self._color_cycle % len(cfg.intensity_words)]
            parts.append(intensity_word)

        # Add energy-based style
        overall_energy = (bass + mid + treble) / 3
        if overall_energy < 0.3:
            parts.append(cfg.low_energy_style)
        elif overall_energy < 0.6:
            parts.append(cfg.medium_energy_style)
        else:
            parts.append(cfg.high_energy_style)

        # Beat flash
        is_new_beat = metrics.is_beat and not self._last_beat
        self._last_beat = metrics.is_beat

        if cfg.enable_beat_flash and is_new_beat:
            flash_word = cfg.beat_flash_words[self._color_cycle % len(cfg.beat_flash_words)]
            parts.append(flash_word)
            self._color_cycle += 1

        return ", ".join(parts)

    def _get_color_repetitions(self, intensity: float, color: str) -> str:
        """Get color word(s) based on intensity.

        Higher intensity = more repetitions for stronger model attention.
        """
        cfg = self.config

        if intensity < cfg.activation_threshold:
            return ""

        # Count how many thresholds we exceed
        count = 0
        for threshold in cfg.repetition_thresholds:
            if intensity >= threshold:
                count += 1
            else:
                break

        count = min(count, cfg.max_repetitions)
        if count == 0:
            return ""

        return " ".join([color] * count)

    def _get_mix_color(
        self,
        bass: float,
        mid: float,
        treble: float,
        active_colors: List[str]
    ) -> Optional[str]:
        """Get additive color mix result if applicable."""
        cfg = self.config

        # Check for RGB mix (white)
        if (bass > cfg.mix_threshold and
            mid > cfg.mix_threshold and
            treble > cfg.mix_threshold):
            key = (cfg.bass_color, cfg.mid_color, cfg.treble_color)
            return cfg.mix_colors.get(key, "white")

        # Check for two-color mixes
        if bass > cfg.mix_threshold and mid > cfg.mix_threshold:
            key = (cfg.bass_color, cfg.mid_color)
            return cfg.mix_colors.get(key)

        if bass > cfg.mix_threshold and treble > cfg.mix_threshold:
            key = (cfg.bass_color, cfg.treble_color)
            return cfg.mix_colors.get(key)

        if mid > cfg.mix_threshold and treble > cfg.mix_threshold:
            key = (cfg.mid_color, cfg.treble_color)
            return cfg.mix_colors.get(key)

        return None

    def set_config(self, config: ColorOrganConfig) -> None:
        """Update the configuration."""
        self.config = config

    def set_colors(self, bass: str, mid: str, treble: str) -> None:
        """Set custom colors for each frequency band."""
        self.config.bass_color = bass
        self.config.mid_color = mid
        self.config.treble_color = treble
        logger.info(f"Color organ colors set: bass={bass}, mid={mid}, treble={treble}")

    def reset(self) -> None:
        """Reset smoothed values."""
        self._smoothed_bass = 0.0
        self._smoothed_mid = 0.0
        self._smoothed_treble = 0.0
        self._color_cycle = 0

    def set_chroma_threshold(self, threshold: float) -> None:
        """Stub for compatibility - color organ doesn't use chroma threshold."""
        pass


# Preset configurations for different color organ styles
COLOR_ORGAN_PRESETS = {
    "classic": ColorOrganConfig(
        bass_color="red",
        mid_color="green",
        treble_color="blue",
        use_color_mixing=True,
        include_context_words=True,
    ),
    "warm": ColorOrganConfig(
        bass_color="deep red",
        mid_color="orange",
        treble_color="yellow",
        use_color_mixing=False,
        intensity_words=["warm", "glowing", "fiery", "golden"],
    ),
    "cool": ColorOrganConfig(
        bass_color="purple",
        mid_color="blue",
        treble_color="cyan",
        use_color_mixing=False,
        intensity_words=["cool", "icy", "electric", "neon"],
    ),
    "neon": ColorOrganConfig(
        bass_color="hot pink",
        mid_color="electric green",
        treble_color="electric blue",
        use_color_mixing=True,
        include_context_words=True,
        context_words=["neon", "neon lights", "synthwave", "cyberpunk"],
        intensity_words=["blazing", "electric", "fluorescent", "vivid"],
    ),
    "pastel": ColorOrganConfig(
        bass_color="coral",
        mid_color="mint",
        treble_color="lavender",
        use_color_mixing=False,
        intensity_words=["soft", "gentle", "dreamy", "ethereal"],
        low_energy_style="soft pastel glow",
        medium_energy_style="gentle colors",
        high_energy_style="bright pastels",
    ),
}


def get_color_organ_preset(name: str) -> ColorOrganConfig:
    """Get a color organ preset by name."""
    return COLOR_ORGAN_PRESETS.get(name, COLOR_ORGAN_PRESETS["classic"])
