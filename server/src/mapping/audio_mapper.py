"""Enhanced audio feature mapping for SOTA AI generation control."""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List

from .parameter_curves import CurveType, map_range
from .prompt_modulator import PromptModulator, ModulationConfig, FluxPromptModulator
from .color_organ_modulator import ColorOrganModulator
from ..server.protocol import AudioMetrics, GenerationConfig, MappingConfig, AudioSource, CurveType as ProtocolCurveType

logger = logging.getLogger(__name__)


@dataclass
class ParameterMapping:
    """Defines how an audio feature maps to a generation parameter."""

    audio_feature: str  # bass, mid, treble, rms, peak, spectral_centroid, onset, etc.
    output_min: float
    output_max: float
    curve: CurveType = CurveType.LINEAR
    input_min: float = 0.0
    input_max: float = 1.0
    enabled: bool = True


@dataclass
class ChromaColorMapping:
    """Maps chroma features to color keywords for prompt modulation."""

    # Map each pitch class to color associations
    chroma_colors: dict = field(default_factory=lambda: {
        0: ["red", "warm"],           # C
        1: ["red-orange", "fiery"],    # C#
        2: ["orange", "golden"],       # D
        3: ["yellow-orange", "amber"], # D#
        4: ["yellow", "bright"],       # E
        5: ["yellow-green", "lime"],   # F
        6: ["green", "natural"],       # F#
        7: ["cyan", "aqua"],           # G
        8: ["blue", "cool"],           # G#
        9: ["indigo", "deep"],         # A
        10: ["violet", "purple"],      # A#
        11: ["magenta", "pink"],       # B
    })

    # Intensity threshold for including color
    intensity_threshold: float = 0.3


@dataclass
class MappingPreset:
    """Complete mapping configuration preset with SOTA features."""

    name: str
    description: str

    # Denoising strength mapping (for img2img/feedback)
    strength_mapping: ParameterMapping = field(
        default_factory=lambda: ParameterMapping(
            audio_feature="bass",
            output_min=0.3,
            output_max=0.7,
            curve=CurveType.EASE_IN_OUT,
        )
    )

    # CFG scale mapping - spectral centroid for dreamier/sharper control
    cfg_mapping: Optional[ParameterMapping] = field(
        default_factory=lambda: ParameterMapping(
            audio_feature="spectral_centroid",
            output_min=0.5,    # Low centroid = dreamier (lower guidance)
            output_max=2.0,    # High centroid = sharper (higher guidance)
            curve=CurveType.LINEAR,
        )
    )

    # Seed variation on onset
    onset_seed_variation: bool = True
    onset_seed_range: int = 100  # Range for seed offset on onset

    # Beat sync options
    beat_seed_jump: bool = False  # Randomize seed completely on beat
    beat_strength_boost: float = 0.0  # Add to strength on beat
    txt2img_on_beat_interval: int = 0  # Do txt2img every N beats (0 = disabled)
    beat_cooldown_ms: int = 300  # Minimum ms between beat triggers (prevents false positives)

    # Continuous variation options (for flowing visuals)
    continuous_seed_variation: bool = True  # Vary seed continuously based on audio
    continuous_variation_scale: float = 1000.0  # How much seed varies with energy (10x for more variety)

    # Chroma-based color mapping
    use_chroma_colors: bool = True
    chroma_color_mapping: ChromaColorMapping = field(default_factory=ChromaColorMapping)

    # Prompt modulation settings
    modulation_config: ModulationConfig = field(default_factory=ModulationConfig)


@dataclass
class GenerationParams:
    """Parameters for a single generation step."""

    prompt: str
    negative_prompt: str
    strength: float  # Denoising strength for img2img
    guidance_scale: float
    seed: Optional[int] = None
    # Additional SOTA params
    color_keywords: List[str] = field(default_factory=list)
    is_onset: bool = False
    onset_confidence: float = 0.0
    force_txt2img: bool = False  # Force txt2img this frame (for periodic pose refresh)
    is_beat_seed_jump: bool = False  # Beat triggered a seed jump - signals dramatic change


class AudioMapper:
    """Maps real-time audio metrics to generation parameters with SOTA features."""

    def __init__(self, preset: Optional[MappingPreset] = None, use_flux_modulation: bool = False, use_color_organ: bool = False):
        self.preset = preset or self._get_default_preset()
        self._use_flux_modulation = use_flux_modulation
        self._use_color_organ = use_color_organ

        # Use appropriate modulator based on mode
        if use_color_organ:
            self.prompt_modulator = ColorOrganModulator()
            logger.info("Using ColorOrganModulator for frequency-to-color mapping")
        elif use_flux_modulation:
            self.prompt_modulator = FluxPromptModulator()
            logger.info("Using FluxPromptModulator for aggressive prompt transformation")
        else:
            self.prompt_modulator = PromptModulator(self.preset.modulation_config)

        self._current_seed = random.randint(0, 2**32 - 1)
        self._last_beat = False
        self._last_onset = False
        self._last_beat_time: float = 0.0  # Timestamp of last accepted beat
        self._beat_count = 0  # Track beats for periodic txt2img
        self._beat_cooldown_ms: int = 300  # Can be overridden at runtime
        self._periodic_pose_refresh_override: Optional[int] = None  # None = use preset, 0 = disabled, N = every N beats

        # Smoothed values for stability
        self._smoothed_centroid = 0.5
        self._smoothed_chroma = [0.0] * 12

        # Dynamic config from frontend (overrides preset when set)
        self._dynamic_config: Optional[MappingConfig] = None

    def set_flux_mode(self, enabled: bool) -> None:
        """Enable/disable FLUX-specific aggressive prompt modulation.

        FLUX/Klein models rely on prompt changes for visual variation (not strength),
        so this mode uses more dramatic prompt transformations.
        """
        if enabled and not self._use_flux_modulation:
            self.prompt_modulator = FluxPromptModulator()
            self._use_flux_modulation = True
            self._use_color_organ = False
            logger.info("Switched to FluxPromptModulator")
        elif not enabled and self._use_flux_modulation:
            self.prompt_modulator = PromptModulator(self.preset.modulation_config)
            self._use_flux_modulation = False
            logger.info("Switched to standard PromptModulator")

    def set_color_organ_mode(self, enabled: bool, preset_name: str = "classic") -> None:
        """Enable/disable color organ mode for frequency-to-color prompt generation.

        Color organ maps frequency bands directly to colors:
        - Bass → Red (repeated based on intensity)
        - Mid → Green
        - Treble → Blue

        Args:
            enabled: Whether to enable color organ mode
            preset_name: Color organ preset ('classic', 'warm', 'cool', 'neon', 'pastel')
        """
        from .color_organ_modulator import get_color_organ_preset

        if enabled and not self._use_color_organ:
            config = get_color_organ_preset(preset_name)
            self.prompt_modulator = ColorOrganModulator(config)
            self._use_color_organ = True
            self._use_flux_modulation = False
            logger.info(f"Switched to ColorOrganModulator (preset: {preset_name})")
        elif not enabled and self._use_color_organ:
            self.prompt_modulator = PromptModulator(self.preset.modulation_config)
            self._use_color_organ = False
            logger.info("Switched to standard PromptModulator")

    def set_beat_cooldown(self, cooldown_ms: int) -> None:
        """Set the minimum time between beat triggers.

        Args:
            cooldown_ms: Minimum milliseconds between beats (0 = no cooldown)
        """
        self._beat_cooldown_ms = max(0, cooldown_ms)
        logger.info(f"Beat cooldown set to {self._beat_cooldown_ms}ms")

    def _get_default_preset(self) -> MappingPreset:
        """Get the default reactive preset with SOTA features."""
        return MappingPreset(
            name="reactive_sota",
            description="SOTA audio-reactive visuals with chroma and onset control",
            strength_mapping=ParameterMapping(
                audio_feature="bass",
                output_min=0.35,
                output_max=0.65,
                curve=CurveType.EASE_IN_OUT,
            ),
            cfg_mapping=ParameterMapping(
                audio_feature="spectral_centroid",
                output_min=0.5,
                output_max=2.0,
                curve=CurveType.LINEAR,
            ),
            onset_seed_variation=True,
            onset_seed_range=200,  # More variation on audio transients
            beat_seed_jump=True,   # Full random seed reset on beats
            beat_strength_boost=0.1,
            use_chroma_colors=True,
        )

    def map(self, metrics: AudioMetrics, config: GenerationConfig) -> GenerationParams:
        """Map audio metrics to generation parameters with SOTA features."""
        preset = self.preset
        dyn = self._dynamic_config

        # Map denoising strength - use dynamic mapping if available
        if dyn and "transformStrength" in dyn.mappings:
            strength = self._apply_dynamic_mapping(metrics, "transformStrength", config.img2img_strength)
        else:
            strength = self._map_parameter(metrics, preset.strength_mapping, config.img2img_strength)

        # Apply beat boost - use dynamic trigger config if available
        beat_strength_boost = dyn.triggers.on_beat.strength_boost if dyn else preset.beat_strength_boost
        if metrics.is_beat and beat_strength_boost > 0:
            strength = min(1.0, strength + beat_strength_boost)

        # Map CFG scale - use dynamic mapping if available
        if dyn and "guidanceScale" in dyn.mappings:
            guidance_scale = self._apply_dynamic_mapping(metrics, "guidanceScale", 0.0)
        elif preset.cfg_mapping and metrics.spectral_centroid is not None:
            self._smoothed_centroid = 0.8 * self._smoothed_centroid + 0.2 * metrics.spectral_centroid
            guidance_scale = self._map_spectral_centroid(
                self._smoothed_centroid,
                preset.cfg_mapping,
            )
        else:
            guidance_scale = 1.5  # LCM/Turbo appropriate default for CFG

        # Handle seed with onset detection
        seed = self._current_seed
        is_onset = False
        onset_confidence = 0.0

        if metrics.onset:
            is_onset = metrics.onset.is_onset
            onset_confidence = metrics.onset.confidence

            # Use dynamic onset config if available
            onset_seed_variation = dyn.triggers.on_onset.seed_variation if dyn else (preset.onset_seed_range if preset.onset_seed_variation else 0)
            if is_onset and not self._last_onset and onset_seed_variation > 0:
                variation = int(onset_confidence * onset_seed_variation)
                old_seed = seed
                seed = self._current_seed + random.randint(1, max(1, variation))
                logger.info(f"Onset seed variation: {old_seed} -> {seed} (confidence={onset_confidence:.2f}, range={onset_seed_variation})")

        # Handle beat seed jump - use dynamic trigger config if available
        beat_seed_jump = dyn.triggers.on_beat.seed_jump if dyn else preset.beat_seed_jump

        # Track beats and check for periodic txt2img
        force_txt2img = False
        current_time = time.time()

        # Check beat cooldown - get from preset or instance override
        cooldown_ms = getattr(preset, 'beat_cooldown_ms', self._beat_cooldown_ms)
        cooldown_elapsed = (current_time - self._last_beat_time) * 1000  # Convert to ms
        cooldown_ok = cooldown_elapsed >= cooldown_ms

        # Only accept beat if it's a new beat AND cooldown has elapsed
        is_new_beat = metrics.is_beat and not self._last_beat and cooldown_ok

        # Debug: log when beat is detected
        if metrics.is_beat:
            if not cooldown_ok:
                logger.debug(f"Beat ignored: cooldown ({cooldown_elapsed:.0f}ms < {cooldown_ms}ms)")
            else:
                logger.info(f"Beat in metrics: beat_seed_jump={beat_seed_jump}, last_beat={self._last_beat}, energy={metrics.rms:.2f}")

        if is_new_beat:
            self._last_beat_time = current_time  # Update cooldown timer
            self._beat_count += 1
            # Check for periodic txt2img (fresh pose on every N beats)
            # Use override if set, otherwise fall back to preset
            if self._periodic_pose_refresh_override is not None:
                txt2img_interval = self._periodic_pose_refresh_override
            else:
                txt2img_interval = preset.txt2img_on_beat_interval
            if txt2img_interval > 0:
                if self._beat_count % txt2img_interval == 0:
                    force_txt2img = True
                    logger.info(f"Periodic txt2img triggered (beat {self._beat_count}, every {txt2img_interval} beats)")
                else:
                    logger.debug(f"Beat {self._beat_count}, next refresh at beat {((self._beat_count // txt2img_interval) + 1) * txt2img_interval}")

        is_beat_seed_jump = False
        if beat_seed_jump and is_new_beat:
            old_seed = self._current_seed
            self._current_seed = random.randint(0, 2**32 - 1)
            seed = self._current_seed
            is_beat_seed_jump = True  # Signal to pipeline for txt2img injection
            logger.info(f"Beat seed jump: {old_seed} -> {seed}")
        elif is_new_beat:
            # Small seed offset on beat for pose variation
            old_seed = seed
            seed = self._current_seed + random.randint(1, 100)
            logger.info(f"Beat seed offset: {old_seed} -> {seed}")

        # Continuous seed variation based on audio energy (creates flowing visuals)
        # Only apply when there's meaningful audio (above 5% energy threshold)
        # to prevent drift from ambient noise when music is paused
        if preset.continuous_seed_variation:
            energy = (metrics.bass + metrics.mid + metrics.rms) / 3
            if energy > 0.05:  # Minimum threshold to ignore ambient noise
                noise = int(energy * preset.continuous_variation_scale * random.random())
                seed = seed + noise

        self._last_beat = metrics.is_beat
        self._last_onset = is_onset if metrics.onset else False

        # Get chroma-based color keywords
        color_keywords = []
        if preset.use_chroma_colors and metrics.chroma:
            color_keywords = self._map_chroma_to_colors(
                metrics.chroma.bins,
                preset.chroma_color_mapping,
            )

        # Modulate prompt with all audio features (includes chroma colors)
        prompt = self.prompt_modulator.modulate(config.base_prompt, metrics)

        return GenerationParams(
            prompt=prompt,
            negative_prompt=config.negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            seed=seed,
            color_keywords=color_keywords,
            is_onset=is_onset,
            onset_confidence=onset_confidence,
            force_txt2img=force_txt2img,
            is_beat_seed_jump=is_beat_seed_jump,
        )

    def _map_parameter(
        self,
        metrics: AudioMetrics,
        mapping: ParameterMapping,
        default: float,
    ) -> float:
        """Map a single parameter based on audio feature."""
        if not mapping.enabled:
            return default

        # Get the audio feature value
        feature_map = {
            "bass": metrics.bass,
            "mid": metrics.mid,
            "treble": metrics.treble,
            "rms": metrics.rms,
            "peak": metrics.peak,
            "raw_bass": metrics.raw_bass,
            "raw_mid": metrics.raw_mid,
            "raw_treble": metrics.raw_treble,
            "spectral_centroid": metrics.spectral_centroid if metrics.spectral_centroid else 0.5,
        }

        value = feature_map.get(mapping.audio_feature, 0.0)

        return map_range(
            value,
            in_min=mapping.input_min,
            in_max=mapping.input_max,
            out_min=mapping.output_min,
            out_max=mapping.output_max,
            curve=mapping.curve,
        )

    def _map_spectral_centroid(
        self,
        centroid: float,
        mapping: ParameterMapping,
    ) -> float:
        """Map spectral centroid to CFG scale.

        Low centroid (dark/warm sounds) -> lower guidance (dreamier)
        High centroid (bright/sharp sounds) -> higher guidance (sharper)
        """
        return map_range(
            centroid,
            in_min=mapping.input_min,
            in_max=mapping.input_max,
            out_min=mapping.output_min,
            out_max=mapping.output_max,
            curve=mapping.curve,
        )

    def _map_chroma_to_colors(
        self,
        chroma_bins: List[float],
        mapping: ChromaColorMapping,
    ) -> List[str]:
        """Map chroma features to color keywords.

        Returns list of color keywords based on dominant pitch classes.
        """
        # Update smoothed chroma
        for i in range(min(12, len(chroma_bins))):
            self._smoothed_chroma[i] = 0.7 * self._smoothed_chroma[i] + 0.3 * chroma_bins[i]

        # Find dominant pitch classes above threshold
        colors = []
        for i, intensity in enumerate(self._smoothed_chroma):
            if intensity > mapping.intensity_threshold:
                color_words = mapping.chroma_colors.get(i, [])
                if color_words:
                    # Weight by intensity
                    colors.append((color_words[0], intensity))

        # Sort by intensity and take top 2
        colors.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in colors[:2]]

    def get_dominant_color_hue(self, metrics: AudioMetrics) -> float:
        """Get dominant color hue (0-1) from chroma features for shader use."""
        if not metrics.chroma:
            return 0.0

        # Map dominant chroma (0-11) to hue (0-1)
        dominant = metrics.dominant_chroma
        return dominant / 12.0

    def set_preset(self, preset: MappingPreset) -> None:
        """Update the mapping preset."""
        self.preset = preset
        # Only update modulation config if using standard PromptModulator
        # ColorOrganModulator uses its own config type
        if not self._use_color_organ and hasattr(self.prompt_modulator, 'set_config'):
            if isinstance(self.prompt_modulator, PromptModulator):
                self.prompt_modulator.set_config(preset.modulation_config)

    def set_periodic_pose_refresh(self, enabled: bool) -> None:
        """Enable/disable periodic txt2img injection (every 8 beats).

        When enabled, overrides preset's txt2img_on_beat_interval to 8.
        When disabled, sets interval to 0 (no periodic txt2img).
        """
        self._periodic_pose_refresh_override = 8 if enabled else 0
        logger.info(f"Periodic pose refresh: {'enabled (every 8 beats)' if enabled else 'disabled'}")

    def set_dynamic_config(self, config: MappingConfig) -> None:
        """Set dynamic mapping configuration from frontend."""
        self._dynamic_config = config

        # Update prompt modulator with chroma threshold if provided
        if config.triggers and config.triggers.chroma_threshold:
            self.prompt_modulator.set_chroma_threshold(config.triggers.chroma_threshold)

        # Update beat cooldown if provided
        if hasattr(config, 'beat_cooldown_ms'):
            self.set_beat_cooldown(config.beat_cooldown_ms)

        logger.info(f"Dynamic mapping config set: {len(config.mappings)} mappings, preset={config.preset_name}, beat_cooldown={self._beat_cooldown_ms}ms")

    def _get_audio_value_by_source(self, metrics: AudioMetrics, source: AudioSource) -> float:
        """Get audio value for a given source enum."""
        source_map = {
            AudioSource.BASS: metrics.bass,
            AudioSource.MID: metrics.mid,
            AudioSource.TREBLE: metrics.treble,
            AudioSource.RMS: metrics.rms,
            AudioSource.PEAK: metrics.peak,
            AudioSource.SPECTRAL_CENTROID: metrics.spectral_centroid if metrics.spectral_centroid else 0.5,
            AudioSource.BASS_MID: (metrics.bass + metrics.mid) / 2,
            AudioSource.BPM: min(1.0, metrics.bpm / 200),  # Normalize BPM to 0-1
            AudioSource.ONSET_STRENGTH: metrics.onset.strength if metrics.onset else 0.0,
            AudioSource.FIXED: 1.0,
        }
        return source_map.get(source, 0.0)

    def _convert_curve_type(self, curve: ProtocolCurveType) -> CurveType:
        """Convert protocol curve type to internal curve type."""
        curve_map = {
            ProtocolCurveType.LINEAR: CurveType.LINEAR,
            ProtocolCurveType.EASE_IN: CurveType.EASE_IN,
            ProtocolCurveType.EASE_OUT: CurveType.EASE_OUT,
            ProtocolCurveType.EASE_IN_OUT: CurveType.EASE_IN_OUT,
            ProtocolCurveType.EXPONENTIAL: CurveType.EXPONENTIAL,
        }
        return curve_map.get(curve, CurveType.LINEAR)

    def _apply_dynamic_mapping(self, metrics: AudioMetrics, mapping_id: str, default: float) -> float:
        """Apply a dynamic mapping if configured, otherwise return default."""
        if not self._dynamic_config or mapping_id not in self._dynamic_config.mappings:
            return default

        mapping = self._dynamic_config.mappings[mapping_id]
        if not mapping.enabled:
            return default

        # Get audio value
        audio_value = self._get_audio_value_by_source(metrics, mapping.source)

        # Apply mapping
        curve = self._convert_curve_type(mapping.curve)
        return map_range(
            audio_value,
            in_min=mapping.input_min,
            in_max=mapping.input_max,
            out_min=mapping.output_min,
            out_max=mapping.output_max,
            curve=curve,
        )

    def reset_seed(self) -> None:
        """Reset to a new random seed."""
        old_seed = self._current_seed
        self._current_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Seed reset: {old_seed} -> {self._current_seed}")


# Preset definitions
PRESETS = {
    "reactive_abstract": MappingPreset(
        name="reactive_abstract",
        description="Subtle, flowing visuals that breathe with the music",
        strength_mapping=ParameterMapping(
            audio_feature="bass",
            output_min=0.26,  # Safe minimum for 4-step inference
            output_max=0.40,
            curve=CurveType.EASE_IN_OUT,
        ),
        onset_seed_variation=False,  # No seed changes - smooth morphing
        beat_seed_jump=False,
        beat_strength_boost=0.05,
        continuous_seed_variation=False,  # Stable seed for coherent evolution
        use_chroma_colors=True,
    ),
    "vj_intense": MappingPreset(
        name="vj_intense",
        description="High-energy, dramatic visuals for live performance",
        strength_mapping=ParameterMapping(
            audio_feature="rms",
            output_min=0.50,
            output_max=0.80,  # High strength for dramatic changes
            curve=CurveType.EASE_IN,
        ),
        cfg_mapping=ParameterMapping(
            audio_feature="spectral_centroid",
            output_min=1.0,
            output_max=3.0,
            curve=CurveType.EASE_IN,
        ),
        onset_seed_variation=True,
        onset_seed_range=1000,  # Big seed jumps on transients
        beat_seed_jump=True,  # Random seed every beat
        beat_strength_boost=0.2,
        continuous_seed_variation=False,  # Only change on beats/onsets
        use_chroma_colors=True,
    ),
    "dreamscape": MappingPreset(
        name="dreamscape",
        description="Smooth, ethereal visuals with gentle transitions",
        strength_mapping=ParameterMapping(
            audio_feature="mid",
            output_min=0.26,  # Safe minimum for 4-step inference
            output_max=0.35,  # Gentle changes
            curve=CurveType.EASE_OUT,
        ),
        cfg_mapping=ParameterMapping(
            audio_feature="spectral_centroid",
            output_min=0.0,
            output_max=1.0,  # Low CFG for dreamy look
            curve=CurveType.EASE_OUT,
        ),
        onset_seed_variation=False,
        beat_seed_jump=False,  # Never jump seed - keep dreamy coherence
        beat_strength_boost=0.0,
        continuous_seed_variation=False,
        use_chroma_colors=True,
    ),
    "minimal": MappingPreset(
        name="minimal",
        description="Very subtle response for calm ambient content",
        strength_mapping=ParameterMapping(
            audio_feature="rms",
            output_min=0.26,  # Safe minimum for 4-step inference
            output_max=0.30,
            curve=CurveType.LINEAR,
        ),
        onset_seed_variation=False,
        beat_seed_jump=False,
        beat_strength_boost=0.0,
        continuous_seed_variation=False,
        use_chroma_colors=False,
    ),
    "dancer": MappingPreset(
        name="dancer",
        description="Makes subjects dance to music with pose variations on beats",
        strength_mapping=ParameterMapping(
            audio_feature="bass",
            output_min=0.35,  # Higher base for more visible changes
            output_max=0.65,  # Higher max for dramatic pose shifts
            curve=CurveType.EASE_OUT,
        ),
        cfg_mapping=ParameterMapping(
            audio_feature="spectral_centroid",
            output_min=0.5,
            output_max=2.0,
            curve=CurveType.LINEAR,
        ),
        onset_seed_variation=True,
        onset_seed_range=200,
        beat_seed_jump=True,  # Pose changes on beats
        beat_strength_boost=0.25,  # Cranked up for dramatic pose changes
        txt2img_on_beat_interval=0,  # Controlled by UI toggle (periodic_pose_refresh)
        continuous_seed_variation=False,  # Only change on beats, not continuously
        use_chroma_colors=True,
        modulation_config=ModulationConfig(
            enable_motion=True,
            motion_keywords=["dancing", "dynamic pose", "graceful movement", "flowing motion"],
            beat_keywords=["dramatic pose", "action", "movement", "dynamic"],
            bass_threshold=0.45,  # More responsive
        ),
    ),
    "pulsing": MappingPreset(
        name="pulsing",
        description="Rhythmic visual breathing effect synced to volume",
        strength_mapping=ParameterMapping(
            audio_feature="rms",  # Uses volume for smooth pulsing
            output_min=0.26,  # Safe minimum for 4-step inference
            output_max=0.45,
            curve=CurveType.EASE_IN_OUT,
        ),
        cfg_mapping=ParameterMapping(
            audio_feature="spectral_centroid",
            output_min=0.5,
            output_max=1.5,
            curve=CurveType.LINEAR,
        ),
        onset_seed_variation=False,
        beat_seed_jump=False,  # Keep subject consistent
        beat_strength_boost=0.18,  # Strong beat boost for visible pulse
        continuous_seed_variation=False,
        use_chroma_colors=True,
        modulation_config=ModulationConfig(
            enable_motion=False,
            beat_keywords=["pulsing", "breathing", "rhythmic"],
        ),
    ),
    "color_organ": MappingPreset(
        name="color_organ",
        description="Classic color organ - bass=red, mid=green, treble=blue with AI generation",
        strength_mapping=ParameterMapping(
            audio_feature="rms",
            output_min=0.30,
            output_max=0.55,
            curve=CurveType.EASE_IN_OUT,
        ),
        cfg_mapping=ParameterMapping(
            audio_feature="spectral_centroid",
            output_min=0.5,
            output_max=2.0,
            curve=CurveType.LINEAR,
        ),
        onset_seed_variation=False,
        beat_seed_jump=True,  # Jump seed on beat for variety
        beat_strength_boost=0.15,
        continuous_seed_variation=False,
        use_chroma_colors=False,  # Disable chroma - color organ handles colors
        # Note: When using this preset, swap prompt_modulator for ColorOrganModulator
        modulation_config=ModulationConfig(
            enable_motion=False,
            enable_lyrics=False,
            # Disable standard keyword modulation - color organ handles it
            bass_threshold=2.0,  # Effectively disabled
            mid_threshold=2.0,
            treble_threshold=2.0,
            loud_threshold=2.0,
            quiet_threshold=-1.0,
        ),
    ),
}

# Add 'reactive' as an alias for 'reactive_abstract' to match frontend preset names
PRESETS["reactive"] = PRESETS["reactive_abstract"]


def get_preset(name: str) -> MappingPreset:
    """Get a mapping preset by name."""
    return PRESETS.get(name, PRESETS["reactive_abstract"])
