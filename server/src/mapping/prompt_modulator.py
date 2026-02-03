"""Advanced prompt modulation with CLIP token weighting based on audio features."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import re
import logging
from ..server.protocol import AudioMetrics

logger = logging.getLogger(__name__)


@dataclass
class PromptModulation:
    """A conditional prompt modification."""

    keywords: list[str]
    audio_feature: str  # bass, mid, treble, rms, is_beat, spectral_centroid
    threshold: float = 0.5
    weight: float = 1.0


@dataclass
class WeightedKeyword:
    """A keyword with dynamic CLIP weight."""

    keyword: str
    base_weight: float = 1.0
    audio_feature: str = "rms"  # Feature that modulates weight
    weight_min: float = 0.5
    weight_max: float = 1.5


@dataclass
class ChromaColorWord:
    """Color word associated with chroma pitch class."""

    pitch_class: int  # 0-11 (C, C#, D, ..., B)
    color_words: List[str]
    intensity_threshold: float = 0.3


@dataclass
class ModulationConfig:
    """Configuration for prompt modulation."""

    # Keywords to add when audio features exceed thresholds
    bass_keywords: list[str] = field(
        default_factory=lambda: ["pulsing", "deep", "powerful", "bass-heavy"]
    )
    bass_threshold: float = 0.6

    mid_keywords: list[str] = field(
        default_factory=lambda: ["melodic", "harmonic", "rich", "textured"]
    )
    mid_threshold: float = 0.5

    treble_keywords: list[str] = field(
        default_factory=lambda: ["crystalline", "sharp", "bright", "sparkling"]
    )
    treble_threshold: float = 0.5

    beat_keywords: list[str] = field(
        default_factory=lambda: ["dynamic pose", "in motion", "movement", "action"]
    )

    loud_keywords: list[str] = field(
        default_factory=lambda: ["intense", "energetic", "vibrant"]
    )
    loud_threshold: float = 0.7

    quiet_keywords: list[str] = field(
        default_factory=lambda: ["subtle", "calm", "ambient", "ethereal"]
    )
    quiet_threshold: float = 0.2  # Add when rms BELOW this

    # Motion keywords for continuous movement
    motion_keywords: list[str] = field(
        default_factory=lambda: ["flowing", "graceful movement", "dance pose", "dynamic"]
    )
    enable_motion: bool = True

    # SOTA: Spectral centroid keywords (brightness/timbre)
    bright_keywords: list[str] = field(
        default_factory=lambda: ["bright colors", "sharp details", "vivid", "crisp"]
    )
    bright_threshold: float = 0.65  # High centroid

    warm_keywords: list[str] = field(
        default_factory=lambda: ["warm tones", "soft atmosphere", "muted", "dreamy"]
    )
    warm_threshold: float = 0.35  # Low centroid (below this)

    # SOTA: Onset keywords for transients
    onset_keywords: list[str] = field(
        default_factory=lambda: ["sudden movement", "dramatic", "sharp transition"]
    )
    onset_confidence_threshold: float = 0.6

    # SOTA: Dynamic keyword weighting using (keyword:weight) syntax
    use_dynamic_weighting: bool = True

    # Chroma-to-color mapping
    chroma_colors: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["red"],           # C
        1: ["orange-red"],    # C#
        2: ["orange"],        # D
        3: ["yellow-orange"], # D#
        4: ["yellow"],        # E
        5: ["yellow-green"],  # F
        6: ["green"],         # F#
        7: ["cyan"],          # G
        8: ["blue"],          # G#
        9: ["indigo"],        # A
        10: ["violet"],       # A#
        11: ["magenta"],      # B
    })
    chroma_intensity_threshold: float = 0.4

    # Token budget - limit total additions to avoid CLIP truncation
    max_keyword_additions: int = 6  # Limit to ~6 keywords to stay under 77 tokens

    # Lyric-derived keyword settings
    enable_lyrics: bool = True
    lyric_weight: float = 1.15  # Slightly emphasize lyrics
    max_lyric_keywords: int = 4
    lyric_confidence_threshold: float = 0.6


class PromptModulator:
    """Modulates prompts based on real-time audio features with SOTA capabilities."""

    def __init__(self, config: Optional[ModulationConfig] = None):
        self.config = config or ModulationConfig()

        # Smoothed values for stability
        self._smoothed_centroid = 0.5
        self._smoothed_chroma = [0.0] * 12

        # Lazy-loaded CLIP tokenizer for token counting
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy load CLIP tokenizer for token counting."""
        if self._tokenizer is None:
            try:
                from transformers import CLIPTokenizer
                self._tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("CLIP tokenizer loaded for prompt trimming")
            except Exception as e:
                logger.warning(f"Failed to load CLIP tokenizer: {e}. Falling back to keyword count limit.")
                self._tokenizer = False  # Mark as failed, use fallback
        return self._tokenizer if self._tokenizer else None

    def modulate(self, base_prompt: str, metrics: AudioMetrics) -> str:
        """Add keywords to prompt based on current audio metrics.

        Uses CLIP token weighting syntax: (keyword:weight) for dynamic emphasis.
        """
        additions: list[str] = []
        weighted_additions: list[Tuple[str, float]] = []

        # Motion keywords based on overall energy
        if self.config.enable_motion:
            energy = (metrics.bass + metrics.mid + metrics.rms) / 3
            if energy > 0.3:
                num_motion = 1 if energy < 0.5 else 2
                motion_keywords = self.config.motion_keywords[:num_motion]
                if self.config.use_dynamic_weighting:
                    # Weight motion keywords by energy
                    weight = 0.8 + energy * 0.4  # 0.8 to 1.2
                    weighted_additions.extend([(kw, weight) for kw in motion_keywords])
                else:
                    additions.extend(motion_keywords)

        # Beat modulation - stronger effect for pose changes
        if metrics.is_beat:
            if self.config.use_dynamic_weighting:
                weighted_additions.extend([
                    (kw, 1.3) for kw in self.config.beat_keywords[:2]
                ])
            else:
                additions.extend(self.config.beat_keywords[:2])

        # Onset modulation for transients
        if metrics.onset and metrics.onset.is_onset:
            if metrics.onset.confidence > self.config.onset_confidence_threshold:
                if self.config.use_dynamic_weighting:
                    weight = 1.0 + metrics.onset.confidence * 0.3
                    weighted_additions.extend([
                        (kw, weight) for kw in self.config.onset_keywords[:1]
                    ])
                else:
                    additions.extend(self.config.onset_keywords[:1])

        # Lyric modulation - inject detected keywords from lyrics
        if self.config.enable_lyrics and metrics.lyrics:
            if metrics.lyrics.confidence > self.config.lyric_confidence_threshold:
                for word, weight in metrics.lyrics.keywords[:self.config.max_lyric_keywords]:
                    weighted_additions.append((word, weight * self.config.lyric_weight))

        # Bass modulation
        if metrics.bass > self.config.bass_threshold:
            num_keywords = 1 if metrics.bass < 0.8 else 2
            if self.config.use_dynamic_weighting:
                weight = 0.9 + metrics.bass * 0.3
                weighted_additions.extend([
                    (kw, weight) for kw in self.config.bass_keywords[:num_keywords]
                ])
            else:
                additions.extend(self.config.bass_keywords[:num_keywords])

        # Mid modulation
        if metrics.mid > self.config.mid_threshold:
            num_keywords = 1 if metrics.mid < 0.7 else 2
            if self.config.use_dynamic_weighting:
                weight = 0.9 + metrics.mid * 0.2
                weighted_additions.extend([
                    (kw, weight) for kw in self.config.mid_keywords[:num_keywords]
                ])
            else:
                additions.extend(self.config.mid_keywords[:num_keywords])

        # Treble modulation
        if metrics.treble > self.config.treble_threshold:
            num_keywords = 1 if metrics.treble < 0.7 else 2
            if self.config.use_dynamic_weighting:
                weight = 0.9 + metrics.treble * 0.2
                weighted_additions.extend([
                    (kw, weight) for kw in self.config.treble_keywords[:num_keywords]
                ])
            else:
                additions.extend(self.config.treble_keywords[:num_keywords])

        # Volume modulation
        if metrics.rms > self.config.loud_threshold:
            if self.config.use_dynamic_weighting:
                weight = 1.0 + (metrics.rms - self.config.loud_threshold) * 0.5
                weighted_additions.extend([
                    (kw, weight) for kw in self.config.loud_keywords[:2]
                ])
            else:
                additions.extend(self.config.loud_keywords[:2])
        elif metrics.rms < self.config.quiet_threshold:
            if self.config.use_dynamic_weighting:
                weight = 1.0 + (self.config.quiet_threshold - metrics.rms)
                weighted_additions.extend([
                    (kw, weight) for kw in self.config.quiet_keywords[:2]
                ])
            else:
                additions.extend(self.config.quiet_keywords[:2])

        # SOTA: Spectral centroid for brightness/warmth
        if metrics.spectral_centroid is not None:
            # Smooth the centroid
            self._smoothed_centroid = 0.8 * self._smoothed_centroid + 0.2 * metrics.spectral_centroid

            if self._smoothed_centroid > self.config.bright_threshold:
                if self.config.use_dynamic_weighting:
                    weight = 1.0 + (self._smoothed_centroid - self.config.bright_threshold)
                    weighted_additions.extend([
                        (kw, weight) for kw in self.config.bright_keywords[:2]
                    ])
                else:
                    additions.extend(self.config.bright_keywords[:2])
            elif self._smoothed_centroid < self.config.warm_threshold:
                if self.config.use_dynamic_weighting:
                    weight = 1.0 + (self.config.warm_threshold - self._smoothed_centroid)
                    weighted_additions.extend([
                        (kw, weight) for kw in self.config.warm_keywords[:2]
                    ])
                else:
                    additions.extend(self.config.warm_keywords[:2])

        # SOTA: Chroma-based color keywords
        if metrics.chroma:
            color_keywords = self._get_chroma_colors(metrics.chroma.bins)
            if color_keywords:
                if self.config.use_dynamic_weighting:
                    weighted_additions.extend(color_keywords)
                else:
                    additions.extend([kw for kw, _ in color_keywords])

        # Build final prompt
        return self._build_prompt(base_prompt, additions, weighted_additions)

    def _get_chroma_colors(
        self,
        chroma_bins: List[float],
    ) -> List[Tuple[str, float]]:
        """Get color keywords from chroma features with weights."""
        # Smooth chroma bins
        for i in range(min(12, len(chroma_bins))):
            self._smoothed_chroma[i] = 0.7 * self._smoothed_chroma[i] + 0.3 * chroma_bins[i]

        # Find dominant pitch classes
        colors = []
        for i, intensity in enumerate(self._smoothed_chroma):
            if intensity > self.config.chroma_intensity_threshold:
                color_words = self.config.chroma_colors.get(i, [])
                if color_words:
                    # Weight by intensity
                    weight = 0.8 + intensity * 0.4
                    colors.append((color_words[0], weight, intensity))

        # Sort by intensity and take top 2
        colors.sort(key=lambda x: x[2], reverse=True)
        return [(c[0], c[1]) for c in colors[:2]]

    def _build_prompt(
        self,
        base_prompt: str,
        additions: List[str],
        weighted_additions: List[Tuple[str, float]],
    ) -> str:
        """Build the final prompt with optional CLIP weighting syntax.

        Uses CLIP tokenizer to count tokens and ensure prompt stays under 77 token limit.
        Prioritizes weighted additions by weight value.
        """
        tokenizer = self._get_tokenizer()
        max_tokens = 75  # Leave 2 for BOS/EOS tokens

        if tokenizer:
            # Token-aware building: count actual CLIP tokens
            base_tokens = len(tokenizer.encode(base_prompt))
            current_tokens = base_tokens
            final_keywords = []

            # Sort weighted additions by weight (highest first)
            sorted_weighted = sorted(weighted_additions, key=lambda x: x[1], reverse=True)

            # Add weighted keywords first, checking token count
            for keyword, weight in sorted_weighted:
                keyword_str = f"({keyword}:{weight:.2f})" if abs(weight - 1.0) > 0.05 else keyword
                # Count tokens for ", keyword" (the comma and space before it)
                keyword_tokens = len(tokenizer.encode(", " + keyword_str)) - 1  # Subtract BOS overlap

                if current_tokens + keyword_tokens > max_tokens:
                    break

                final_keywords.append(keyword_str)
                current_tokens += keyword_tokens

            # Add unweighted keywords with remaining budget
            for keyword in additions:
                keyword_tokens = len(tokenizer.encode(", " + keyword)) - 1
                if current_tokens + keyword_tokens > max_tokens:
                    break
                final_keywords.append(keyword)
                current_tokens += keyword_tokens

            if final_keywords:
                return f"{base_prompt}, {', '.join(final_keywords)}"
            return base_prompt

        else:
            # Fallback: use keyword count limit if tokenizer unavailable
            max_additions = self.config.max_keyword_additions

            sorted_weighted = sorted(weighted_additions, key=lambda x: x[1], reverse=True)
            all_keywords: list[str] = []

            for keyword, weight in sorted_weighted:
                if len(all_keywords) >= max_additions:
                    break
                if abs(weight - 1.0) > 0.05:
                    all_keywords.append(f"({keyword}:{weight:.2f})")
                else:
                    all_keywords.append(keyword)

            for keyword in additions:
                if len(all_keywords) >= max_additions:
                    break
                all_keywords.append(keyword)

            if all_keywords:
                return f"{base_prompt}, {', '.join(all_keywords)}"
            return base_prompt

    def modulate_with_weighting(
        self,
        base_prompt: str,
        metrics: AudioMetrics,
    ) -> Tuple[str, Dict[str, float]]:
        """Modulate prompt and return separate keyword weights.

        Useful for pipelines that support explicit CLIP token weighting.

        Returns:
            Tuple of (modulated_prompt, keyword_weights_dict)
        """
        # Get the modulated prompt
        prompt = self.modulate(base_prompt, metrics)

        # Extract weights from (keyword:weight) syntax
        weights = {}
        pattern = r'\(([^:]+):([0-9.]+)\)'
        for match in re.finditer(pattern, prompt):
            keyword = match.group(1)
            weight = float(match.group(2))
            weights[keyword] = weight

        return prompt, weights

    def apply_emphasis(
        self,
        prompt: str,
        keywords: List[str],
        emphasis: float = 1.2,
    ) -> str:
        """Apply emphasis to specific keywords in a prompt.

        Uses (keyword:emphasis) syntax supported by many diffusers.
        """
        result = prompt
        for keyword in keywords:
            # Find keyword and wrap with emphasis
            pattern = rf'\b{re.escape(keyword)}\b'
            if re.search(pattern, result, re.IGNORECASE):
                # Already has weighting syntax?
                if f"({keyword}:" not in result.lower():
                    result = re.sub(
                        pattern,
                        f"({keyword}:{emphasis:.2f})",
                        result,
                        flags=re.IGNORECASE,
                    )
        return result

    def de_emphasize(
        self,
        prompt: str,
        keywords: List[str],
        factor: float = 0.7,
    ) -> str:
        """Reduce emphasis on specific keywords."""
        return self.apply_emphasis(prompt, keywords, factor)

    def set_config(self, config: ModulationConfig) -> None:
        """Update the modulation configuration."""
        self.config = config

    def set_chroma_threshold(self, threshold: float) -> None:
        """Update just the chroma intensity threshold (for dynamic UI control)."""
        self.config.chroma_intensity_threshold = max(0.1, min(0.8, threshold))
        logger.debug(f"Chroma threshold updated to {self.config.chroma_intensity_threshold:.2f}")


@dataclass
class FluxModulationConfig:
    """Configuration for FLUX-specific prompt modulation.

    These are atmosphere/style modifiers that wrap around the user's subject.
    The user provides the subject (e.g., "magical orb over a lake"),
    and these modifiers add intensity, motion, and style based on audio.
    """

    # Intensity prefixes - prepended based on energy level
    # These describe the MOOD, not the subject
    calm_prefixes: List[str] = field(default_factory=lambda: [
        "serene", "peaceful", "gentle", "soft", "quiet", "still",
        "tranquil", "calm", "subtle", "delicate"
    ])

    medium_prefixes: List[str] = field(default_factory=lambda: [
        "dynamic", "flowing", "moving", "active", "animated",
        "rhythmic", "pulsing", "breathing", "shifting"
    ])

    intense_prefixes: List[str] = field(default_factory=lambda: [
        "explosive", "intense", "powerful", "dramatic", "epic",
        "thundering", "blazing", "surging", "overwhelming"
    ])

    chaotic_prefixes: List[str] = field(default_factory=lambda: [
        "chaotic", "wild", "frenzied", "turbulent", "violent",
        "shattered", "fractured", "disintegrating", "exploding"
    ])

    # Motion descriptors - describe HOW things move, not WHAT moves
    motion_calm: List[str] = field(default_factory=lambda: [
        "floating gently", "drifting slowly", "hovering peacefully",
        "suspended in stillness", "barely moving"
    ])

    motion_active: List[str] = field(default_factory=lambda: [
        "swirling", "spinning slowly", "dancing", "flowing",
        "undulating", "rippling", "oscillating"
    ])

    motion_intense: List[str] = field(default_factory=lambda: [
        "exploding outward", "shattering", "bursting", "erupting",
        "violently spinning", "rapidly transforming", "fragmenting"
    ])

    # Style/atmosphere suffixes based on spectral characteristics
    warm_styles: List[str] = field(default_factory=lambda: [
        "warm golden light", "soft amber glow", "sunset colors",
        "cozy atmosphere", "rich earth tones", "deep shadows"
    ])

    bright_styles: List[str] = field(default_factory=lambda: [
        "vivid neon colors", "sharp crystalline details", "electric highlights",
        "brilliant white light", "prismatic reflections", "glowing edges"
    ])

    # Beat/transient emphasis phrases
    beat_phrases: List[str] = field(default_factory=lambda: [
        "dramatic moment", "peak intensity", "climactic instant",
        "powerful impact", "striking moment"
    ])

    onset_phrases: List[str] = field(default_factory=lambda: [
        "sudden burst of energy", "flash of light", "sharp transition",
        "instant transformation", "explosive change"
    ])

    # Energy thresholds for intensity levels
    calm_threshold: float = 0.2
    medium_threshold: float = 0.5
    intense_threshold: float = 0.8

    # Whether to emphasize words in base prompt at high energy
    emphasize_at_high_energy: bool = True

    # Smoothing factor (0-1, higher = more smoothing)
    energy_smoothing: float = 0.7
    centroid_smoothing: float = 0.8


class FluxPromptModulator:
    """Aggressive prompt modulation for FLUX/Klein models.

    Unlike StreamDiffusion where strength controls variation, FLUX Klein relies
    on prompt changes to create visual variation. This modulator wraps the user's
    subject with atmosphere/style modifiers based on audio.

    Structure: [intensity] [user's subject] [motion] [style] [transient effects]

    Example:
        User prompt: "magical orb floating over a lake"
        Low energy:  "serene magical orb floating over a lake, hovering peacefully, warm golden light"
        High energy: "explosive magical orb floating over a lake, bursting, vivid neon colors, dramatic moment"
    """

    def __init__(self, config: Optional[FluxModulationConfig] = None):
        self.config = config or FluxModulationConfig()
        self._beat_count = 0
        self._last_beat = False
        self._smoothed_energy = 0.0
        self._smoothed_centroid = 0.5
        self._style_index = 0  # Cycles through styles on beats

    def modulate(self, base_prompt: str, metrics: AudioMetrics) -> str:
        """Transform prompt based on audio metrics.

        Wraps the user's subject with audio-reactive atmosphere and style.
        """
        cfg = self.config

        # Calculate overall energy (0-1), weight bass slightly higher
        energy = (metrics.bass * 1.2 + metrics.mid + metrics.rms) / 3.2
        energy = min(1.0, max(0.0, energy))

        # Smooth energy to prevent flickering
        self._smoothed_energy = (
            cfg.energy_smoothing * self._smoothed_energy +
            (1 - cfg.energy_smoothing) * energy
        )

        # Smooth spectral centroid
        if metrics.spectral_centroid is not None:
            self._smoothed_centroid = (
                cfg.centroid_smoothing * self._smoothed_centroid +
                (1 - cfg.centroid_smoothing) * metrics.spectral_centroid
            )

        # Track beats for style cycling
        is_new_beat = metrics.is_beat and not self._last_beat
        self._last_beat = metrics.is_beat
        if is_new_beat:
            self._beat_count += 1
            self._style_index = (self._style_index + 1) % 5

        # Build transformed prompt
        parts = []

        # 1. Intensity prefix (mood/atmosphere)
        prefix = self._get_intensity_prefix()
        if prefix:
            parts.append(prefix)

        # 2. User's subject (potentially with emphasis at high energy)
        subject = self._process_subject(base_prompt)
        parts.append(subject)

        # 3. Motion descriptor
        motion = self._get_motion_descriptor()
        if motion:
            parts.append(motion)

        # 4. Style/lighting based on spectral centroid
        style = self._get_style()
        if style:
            parts.append(style)

        # 5. Beat emphasis (only on actual beats with sufficient energy)
        if is_new_beat and self._smoothed_energy > cfg.medium_threshold:
            beat_phrase = cfg.beat_phrases[self._beat_count % len(cfg.beat_phrases)]
            parts.append(beat_phrase)

        # 6. Onset/transient emphasis
        if metrics.onset and metrics.onset.is_onset and metrics.onset.confidence > 0.7:
            onset_phrase = cfg.onset_phrases[self._style_index % len(cfg.onset_phrases)]
            parts.append(onset_phrase)

        # 7. Lyric keywords (these work great with Klein!)
        if metrics.lyrics and metrics.lyrics.confidence > 0.5:
            for word, weight in metrics.lyrics.keywords[:3]:
                if weight > 0.8:
                    parts.append(word)

        return ", ".join(parts)

    def _get_intensity_prefix(self) -> str:
        """Get intensity prefix based on smoothed energy level."""
        cfg = self.config
        energy = self._smoothed_energy

        if energy < cfg.calm_threshold:
            prefixes = cfg.calm_prefixes
        elif energy < cfg.medium_threshold:
            prefixes = cfg.medium_prefixes
        elif energy < cfg.intense_threshold:
            prefixes = cfg.intense_prefixes
        else:
            prefixes = cfg.chaotic_prefixes

        if not prefixes:
            return ""

        # Use beat count for deterministic but varying selection
        idx = (self._beat_count + self._style_index) % len(prefixes)
        return prefixes[idx]

    def _process_subject(self, base_prompt: str) -> str:
        """Process user's subject, optionally adding emphasis at high energy."""
        cfg = self.config

        if not cfg.emphasize_at_high_energy or self._smoothed_energy < cfg.intense_threshold:
            return base_prompt

        # At high energy, emphasize key words (longer words likely nouns/adjectives)
        words = base_prompt.split()
        modified_words = []
        for word in words:
            # Emphasize words > 4 chars that are alphabetic
            if len(word) > 4 and word.isalpha():
                modified_words.append(f"({word}:1.2)")
            else:
                modified_words.append(word)
        return " ".join(modified_words)

    def _get_motion_descriptor(self) -> str:
        """Get motion descriptor based on energy."""
        cfg = self.config
        energy = self._smoothed_energy

        if energy < cfg.calm_threshold:
            motions = cfg.motion_calm
        elif energy < cfg.medium_threshold:
            motions = cfg.motion_active
        else:
            motions = cfg.motion_intense

        if not motions:
            return ""

        idx = self._beat_count % len(motions)
        return motions[idx]

    def _get_style(self) -> str:
        """Get style/lighting based on spectral centroid (brightness)."""
        cfg = self.config

        if self._smoothed_centroid < 0.4:
            styles = cfg.warm_styles
        else:
            styles = cfg.bright_styles

        if not styles:
            return ""

        idx = self._style_index % len(styles)
        return styles[idx]

    def set_config(self, config: FluxModulationConfig) -> None:
        """Update the modulation configuration."""
        self.config = config


class DynamicPromptBuilder:
    """Builds prompts with more sophisticated weighting control."""

    def __init__(self):
        self._base_prompt = ""
        self._keywords: List[Tuple[str, float]] = []
        self._negative_keywords: List[Tuple[str, float]] = []

    def set_base(self, prompt: str) -> 'DynamicPromptBuilder':
        """Set the base prompt."""
        self._base_prompt = prompt
        return self

    def add(self, keyword: str, weight: float = 1.0) -> 'DynamicPromptBuilder':
        """Add a keyword with weight."""
        self._keywords.append((keyword, weight))
        return self

    def add_negative(self, keyword: str, weight: float = 1.0) -> 'DynamicPromptBuilder':
        """Add a negative keyword."""
        self._negative_keywords.append((keyword, weight))
        return self

    def add_from_audio(
        self,
        metrics: AudioMetrics,
        keyword_map: Dict[str, Tuple[str, str]],  # feature -> (keyword, scale_type)
    ) -> 'DynamicPromptBuilder':
        """Add keywords based on audio metrics.

        Args:
            metrics: Audio metrics
            keyword_map: Maps feature name to (keyword, scale_type)
                         scale_type: "linear", "threshold", "inverse"
        """
        feature_values = {
            "bass": metrics.bass,
            "mid": metrics.mid,
            "treble": metrics.treble,
            "rms": metrics.rms,
            "spectral_centroid": metrics.spectral_centroid or 0.5,
        }

        for feature, (keyword, scale_type) in keyword_map.items():
            value = feature_values.get(feature, 0.0)

            if scale_type == "linear":
                weight = 0.5 + value  # 0.5 to 1.5
            elif scale_type == "threshold":
                weight = 1.2 if value > 0.5 else 0.0
            elif scale_type == "inverse":
                weight = 1.5 - value  # 1.5 to 0.5
            else:
                weight = 1.0

            if weight > 0.1:
                self._keywords.append((keyword, weight))

        return self

    def build(self) -> str:
        """Build the final prompt string."""
        parts = [self._base_prompt] if self._base_prompt else []

        for keyword, weight in self._keywords:
            if abs(weight - 1.0) > 0.05:
                parts.append(f"({keyword}:{weight:.2f})")
            else:
                parts.append(keyword)

        return ", ".join(parts)

    def build_negative(self) -> str:
        """Build the negative prompt string."""
        parts = []
        for keyword, weight in self._negative_keywords:
            if abs(weight - 1.0) > 0.05:
                parts.append(f"({keyword}:{weight:.2f})")
            else:
                parts.append(keyword)
        return ", ".join(parts)

    def clear(self) -> 'DynamicPromptBuilder':
        """Clear all keywords."""
        self._keywords.clear()
        self._negative_keywords.clear()
        return self
