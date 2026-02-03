"""Hybrid Lyric Detection Pipeline - Fingerprint-first with transcription fallback.

This is the main entry point for lyric detection. It combines:
1. Audio fingerprinting for known songs (instant, perfect lyrics)
2. Fast Whisper transcription for unknown songs (low latency fallback)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Audio Input                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
  [Fingerprint]                [Fast Transcribe]
  (Chromaprint)                (Whisper tiny, 2s)
        │                             │
   Match found?                  Keywords only
        │                             │
   ┌────┴────┐                       │
   │Yes      │No                     │
   ▼         ▼                       │
[Local DB] [Fallback]◄───────────────┘
  lyrics   to fast
           transcribe
        │
        ▼
  [Keyword Output]
```

## Usage

```python
pipeline = HybridLyricPipeline()
await pipeline.start()

# In your audio processing loop:
pipeline.add_audio_chunk(audio_samples, sample_rate)

# Get current keywords (call this periodically or on demand)
keywords = pipeline.get_current_keywords()

# Track playback position for database lyrics
pipeline.update_playback_position(position_ms)
```

## State Machine

```
INITIALIZING → FINGERPRINTING → MATCHED (DB lyrics)
                            ↘ NOT_MATCHED (transcription fallback)
```
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from .detector import LyricDetector
from .fingerprinter import Fingerprinter
from .keywords import KeywordExtractor
from .lyric_database import LyricDatabase, LyricLine, Song

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """State of the hybrid pipeline."""
    INITIALIZING = auto()    # Setting up, not ready
    FINGERPRINTING = auto()  # Collecting audio for fingerprint
    MATCHED = auto()         # Song identified, using database lyrics
    NOT_MATCHED = auto()     # Unknown song, using transcription
    STOPPED = auto()         # Pipeline stopped


@dataclass
class PipelineStatus:
    """Current status of the pipeline."""
    state: PipelineState
    song: Optional[Song]  # If matched
    fingerprint_progress: float  # 0-1 during fingerprinting
    current_keywords: list[str]
    current_lyric_text: str
    playback_position_ms: int
    transcription_latency_ms: float  # Last transcription latency


@dataclass
class HybridPipelineConfig:
    """Configuration for the hybrid pipeline."""
    # Fingerprinting
    fingerprint_enabled: bool = True
    fingerprint_duration_seconds: float = 15.0

    # Database
    db_path: str = "./lyrics_db.sqlite"

    # Transcription fallback
    whisper_model_size: str = "tiny"
    whisper_device: str = "cuda"
    whisper_buffer_seconds: float = 2.0
    whisper_beam_size: int = 1
    transcribe_interval_seconds: float = 0.75

    # Vocal separation
    vocal_separation: bool = True
    demucs_model: str = "htdemucs"
    demucs_device: str = "cuda"

    # Keyword extraction
    max_keywords: int = 8

    # Lookahead for database lyrics (anticipate visuals)
    lookahead_ms: int = 500

    # Silence detection for automatic song change
    silence_detection_enabled: bool = True
    silence_threshold: float = 0.01  # RMS below this = silence
    silence_duration_seconds: float = 1.5  # Silence must last this long
    silence_cooldown_seconds: float = 5.0  # Min time between auto-resets


class HybridLyricPipeline:
    """Hybrid lyric detection with fingerprint-first strategy.

    Attempts to identify the song via fingerprinting first. If successful,
    uses pre-timed lyrics from the database. Otherwise, falls back to
    real-time transcription with Whisper.
    """

    def __init__(self, config: Optional[HybridPipelineConfig] = None):
        """Initialize the hybrid pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or HybridPipelineConfig()

        # State
        self._state = PipelineState.INITIALIZING
        self._matched_song: Optional[Song] = None
        self._db_lyrics: list[LyricLine] = []
        self._playback_position_ms: int = 0

        # Audio collection for fingerprinting
        self._fingerprint_buffer: list[float] = []
        self._fingerprint_sample_rate: int = 44100
        self._fingerprint_complete = False
        self._last_fingerprint_hash: Optional[str] = None  # Store hash for "learn" feature
        self._last_fingerprint_raw: Optional[list[int]] = None  # Store raw for fuzzy matching

        # Silence detection state
        self._silence_start_time: Optional[float] = None  # When silence started
        self._last_reset_time: float = 0.0  # Last auto-reset timestamp
        self._was_silent: bool = False  # Track silence→audio transition
        self._schedule_reset: bool = False  # Flag to trigger reset outside lock

        # Current output
        self._current_keywords: list[str] = []
        self._current_lyric_text: str = ""
        self._last_lyric_time_ms: int = -1

        # Components (lazy loaded)
        self._fingerprinter: Optional[Fingerprinter] = None
        self._database: Optional[LyricDatabase] = None
        self._detector: Optional[LyricDetector] = None
        self._keyword_extractor: Optional[KeywordExtractor] = None

        # Threading
        self._lock = threading.Lock()
        self._transcribe_thread: Optional[threading.Thread] = None
        self._running = False

        # Callbacks
        self._on_keywords_change: Optional[Callable[[list[str]], None]] = None
        self._on_state_change: Optional[Callable[[PipelineState], None]] = None
        self._on_song_change: Optional[Callable[[], None]] = None  # Called on auto-reset

        # Metrics
        self._last_transcription_latency_ms: float = 0.0

    def _ensure_components(self) -> None:
        """Lazy-load pipeline components."""
        if self._fingerprinter is None and self.config.fingerprint_enabled:
            self._fingerprinter = Fingerprinter(
                duration_seconds=self.config.fingerprint_duration_seconds,
            )

        if self._database is None:
            self._database = LyricDatabase(self.config.db_path)

        if self._detector is None:
            self._detector = LyricDetector(
                model_size=self.config.whisper_model_size,
                device=self.config.whisper_device,
                buffer_seconds=self.config.whisper_buffer_seconds,
                beam_size=self.config.whisper_beam_size,
                vocal_separation=self.config.vocal_separation,
                demucs_model=self.config.demucs_model,
                demucs_device=self.config.demucs_device,
            )

        if self._keyword_extractor is None:
            self._keyword_extractor = KeywordExtractor()

    def start(self) -> None:
        """Start the pipeline.

        Begins fingerprint collection and (in parallel) transcription.
        """
        self._ensure_components()
        self._running = True

        if self.config.fingerprint_enabled:
            self._set_state(PipelineState.FINGERPRINTING)
        else:
            self._set_state(PipelineState.NOT_MATCHED)

        # Start transcription thread
        self._transcribe_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True,
        )
        self._transcribe_thread.start()

        logger.info("Hybrid lyric pipeline started")

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        self._set_state(PipelineState.STOPPED)

        if self._transcribe_thread:
            self._transcribe_thread.join(timeout=2.0)
            self._transcribe_thread = None

        logger.info("Hybrid lyric pipeline stopped")

    def _set_state(self, new_state: PipelineState) -> None:
        """Update pipeline state and notify callback."""
        old_state = self._state
        self._state = new_state

        if old_state != new_state:
            if self._on_state_change:
                self._on_state_change(new_state)

    def add_audio_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Add audio for processing.

        This should be called continuously with incoming audio chunks.

        Args:
            audio: Audio samples (float32, -1 to 1 range preferred)
            sample_rate: Sample rate of the audio
        """
        should_reset = False

        with self._lock:
            # Check for silence/song change
            if self.config.silence_detection_enabled:
                self._check_silence(audio)

            # Check if reset was scheduled (by silence detection)
            if self._schedule_reset:
                self._schedule_reset = False
                should_reset = True

            # Always add to transcription detector
            if self._detector:
                self._detector.add_audio_chunk(audio, sample_rate)

            # Collect for fingerprinting if still in that phase
            if (
                self._state == PipelineState.FINGERPRINTING
                and not self._fingerprint_complete
            ):
                self._add_to_fingerprint_buffer(audio, sample_rate)

        # Perform reset outside lock to avoid deadlock
        if should_reset:
            self.reset()
            # Notify listener of song change
            if self._on_song_change:
                self._on_song_change()

    def _add_to_fingerprint_buffer(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Add audio to fingerprint buffer and check if ready."""
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Store sample rate for first chunk
        if not self._fingerprint_buffer:
            self._fingerprint_sample_rate = sample_rate

        # Resample if different rate
        if sample_rate != self._fingerprint_sample_rate:
            ratio = self._fingerprint_sample_rate / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        self._fingerprint_buffer.extend(audio.tolist())

        # Check if we have enough for fingerprint
        target_samples = int(
            self.config.fingerprint_duration_seconds * self._fingerprint_sample_rate
        )
        current_samples = len(self._fingerprint_buffer)

        if current_samples >= target_samples:
            self._fingerprint_complete = True
            self._try_fingerprint_match()

    def _check_silence(self, audio: np.ndarray) -> None:
        """Check for silence and trigger auto-reset on song change.

        Detects silence gaps between songs. When silence ends (audio returns),
        triggers a pipeline reset to re-fingerprint the new song.

        Args:
            audio: Audio samples to check
        """
        # Calculate RMS of audio chunk
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        rms = np.sqrt(np.mean(audio ** 2))

        current_time = time.time()
        is_silent = rms < self.config.silence_threshold

        if is_silent:
            # Audio is silent
            if self._silence_start_time is None:
                self._silence_start_time = current_time
            self._was_silent = True
        else:
            # Audio is playing
            if self._was_silent and self._silence_start_time is not None:
                silence_duration = current_time - self._silence_start_time
                time_since_last_reset = current_time - self._last_reset_time

                # Check if silence was long enough and we're past cooldown
                if (
                    silence_duration >= self.config.silence_duration_seconds
                    and time_since_last_reset >= self.config.silence_cooldown_seconds
                    and self._state in (PipelineState.MATCHED, PipelineState.NOT_MATCHED)
                ):
                    logger.info(
                        f"SONG CHANGE: {silence_duration:.1f}s silence gap detected. "
                        f"Auto-resetting pipeline."
                    )
                    self._last_reset_time = current_time
                    self._schedule_reset = True

            # Reset silence tracking
            self._silence_start_time = None
            self._was_silent = False

    def _try_fingerprint_match(self) -> None:
        """Attempt to match fingerprint against database."""
        if not self._fingerprinter or not self._fingerprinter.is_available():
            logger.info("Fingerprinting not available, falling back to transcription")
            self._set_state(PipelineState.NOT_MATCHED)
            return

        # Generate fingerprints (both encoded hash and raw)
        audio = np.array(self._fingerprint_buffer, dtype=np.float32)
        fp_hash = self._fingerprinter.fingerprint(audio, self._fingerprint_sample_rate)
        fp_raw = self._fingerprinter.fingerprint_raw(audio, self._fingerprint_sample_rate)

        if not fp_hash:
            logger.warning("Fingerprint generation failed, falling back to transcription")
            self._set_state(PipelineState.NOT_MATCHED)
            return

        logger.info(f"Fingerprint generated ({len(fp_raw) if fp_raw else 0} ints)")

        # Store fingerprints for potential "learn" feature
        self._last_fingerprint_hash = fp_hash
        self._last_fingerprint_raw = fp_raw

        # Exact match only - fuzzy matching causes too many false positives
        song = self._database.find_song_by_fingerprint(fp_hash)

        if song:
            self._matched_song = song
            self._db_lyrics = self._database.get_lyrics(song.id)
            logger.info(f"MATCHED: {song.title} by {song.artist} ({len(self._db_lyrics)} lyrics)")
            self._set_state(PipelineState.MATCHED)
        else:
            logger.info("No match - using transcription fallback")
            self._set_state(PipelineState.NOT_MATCHED)

        # Clear fingerprint buffer to free memory
        self._fingerprint_buffer.clear()

    def _transcription_loop(self) -> None:
        """Background thread for continuous transcription."""
        while self._running:
            # Only transcribe if not matched (or during fingerprinting as fallback)
            if self._state in (PipelineState.NOT_MATCHED, PipelineState.FINGERPRINTING):
                self._do_transcription()

            time.sleep(self.config.transcribe_interval_seconds)

    def _do_transcription(self) -> None:
        """Perform one transcription cycle."""
        if not self._detector:
            return

        try:
            start_time = time.time()
            result = self._detector.transcribe()
            self._last_transcription_latency_ms = (time.time() - start_time) * 1000

            if result and result.text.strip():
                # Extract keywords from transcription
                keywords = self._keyword_extractor.extract(result.text)
                keyword_list = [word for word, weight in keywords[:self.config.max_keywords]]

                self._update_keywords(keyword_list, result.text)

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def update_playback_position(self, position_ms: int) -> None:
        """Update current playback position for database lyrics.

        This should be called continuously with the current playback time
        when playing a matched song.

        Args:
            position_ms: Current playback position in milliseconds.
        """
        with self._lock:
            self._playback_position_ms = position_ms

            if self._state == PipelineState.MATCHED and self._db_lyrics:
                self._update_lyrics_from_db(position_ms)

    def _update_lyrics_from_db(self, position_ms: int) -> None:
        """Get and update keywords from database lyrics for current position."""
        # Add lookahead to anticipate visuals
        target_time = position_ms + self.config.lookahead_ms

        # Find the active lyric line
        # Binary search would be faster for large lyric lists
        active_lyric: Optional[LyricLine] = None
        for lyric in self._db_lyrics:
            if lyric.time_ms <= target_time:
                active_lyric = lyric
            else:
                break

        if active_lyric and active_lyric.time_ms != self._last_lyric_time_ms:
            self._last_lyric_time_ms = active_lyric.time_ms
            self._update_keywords(active_lyric.keywords, active_lyric.text)

    def _update_keywords(self, keywords: list[str], text: str) -> None:
        """Update current keywords and notify callback."""
        if keywords != self._current_keywords:
            self._current_keywords = keywords
            self._current_lyric_text = text

            logger.debug(f"Keywords updated: {keywords} from '{text[:50]}...'")

            if self._on_keywords_change:
                self._on_keywords_change(keywords)

    def get_current_keywords(self) -> list[str]:
        """Get the current visual keywords.

        Returns:
            List of keywords suitable for prompt modulation.
        """
        with self._lock:
            return self._current_keywords.copy()

    def get_status(self) -> PipelineStatus:
        """Get current pipeline status.

        Returns:
            PipelineStatus with current state and metrics.
        """
        with self._lock:
            # Calculate fingerprint progress
            if self._state == PipelineState.FINGERPRINTING:
                target = self.config.fingerprint_duration_seconds * self._fingerprint_sample_rate
                current = len(self._fingerprint_buffer)
                progress = min(1.0, current / target) if target > 0 else 0.0
            else:
                progress = 1.0 if self._fingerprint_complete else 0.0

            return PipelineStatus(
                state=self._state,
                song=self._matched_song,
                fingerprint_progress=progress,
                current_keywords=self._current_keywords.copy(),
                current_lyric_text=self._current_lyric_text,
                playback_position_ms=self._playback_position_ms,
                transcription_latency_ms=self._last_transcription_latency_ms,
            )

    def set_on_keywords_change(
        self,
        callback: Callable[[list[str]], None],
    ) -> None:
        """Set callback for when keywords change.

        Args:
            callback: Function called with new keyword list.
        """
        self._on_keywords_change = callback

    def set_on_state_change(
        self,
        callback: Callable[[PipelineState], None],
    ) -> None:
        """Set callback for when pipeline state changes.

        Args:
            callback: Function called with new state.
        """
        self._on_state_change = callback

    def set_on_song_change(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Set callback for when a song change is detected (auto-reset).

        This is called when silence detection triggers an automatic pipeline reset.
        Use this to reset any external state like playback position tracking.

        Args:
            callback: Function called when song change is detected.
        """
        self._on_song_change = callback

    def reset(self) -> None:
        """Reset the pipeline for a new song/session."""
        with self._lock:
            self._matched_song = None
            self._db_lyrics.clear()
            self._fingerprint_buffer.clear()
            self._fingerprint_complete = False
            self._last_fingerprint_hash = None  # Clear stored hash
            self._last_fingerprint_raw = None   # Clear stored raw fingerprint
            self._current_keywords.clear()
            self._current_lyric_text = ""
            self._last_lyric_time_ms = -1
            self._playback_position_ms = 0

            # Reset silence detection state
            self._silence_start_time = None
            self._was_silent = False
            self._schedule_reset = False

            if self._detector:
                self._detector.clear_buffer()

        if self.config.fingerprint_enabled:
            self._set_state(PipelineState.FINGERPRINTING)
        else:
            self._set_state(PipelineState.NOT_MATCHED)

        logger.info("Pipeline reset")

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        return self._state

    @property
    def is_matched(self) -> bool:
        """Whether a song was matched from the database."""
        return self._state == PipelineState.MATCHED

    @property
    def matched_song(self) -> Optional[Song]:
        """The matched song, if any."""
        return self._matched_song

    @property
    def last_fingerprint_hash(self) -> Optional[str]:
        """The fingerprint hash from the last fingerprinting attempt.

        This is stored so it can be used with the "learn" feature to save
        unmatched songs to the database.
        """
        return self._last_fingerprint_hash

    @property
    def last_fingerprint_raw(self) -> Optional[list[int]]:
        """The raw fingerprint integers from the last fingerprinting attempt.

        This is stored for the "learn" feature to enable fuzzy matching
        when the song is played from different sources.
        """
        return self._last_fingerprint_raw


# Convenience function to create pipeline from config
def create_pipeline_from_settings() -> HybridLyricPipeline:
    """Create a HybridLyricPipeline from the global settings.

    Returns:
        Configured pipeline instance.
    """
    from ..config import settings

    config = HybridPipelineConfig(
        fingerprint_enabled=settings.fingerprint_enabled,
        fingerprint_duration_seconds=settings.fingerprint_duration_seconds,
        db_path=settings.fingerprint_db_path,
        whisper_model_size=settings.lyric_model_size,
        whisper_device=settings.lyric_device,
        whisper_buffer_seconds=settings.lyric_buffer_seconds,
        whisper_beam_size=settings.lyric_beam_size,
        transcribe_interval_seconds=settings.lyric_transcribe_interval,
        vocal_separation=settings.lyric_vocal_separation,
        demucs_model=settings.lyric_demucs_model,
        demucs_device=settings.lyric_demucs_device,
        silence_detection_enabled=settings.silence_detection_enabled,
        silence_threshold=settings.silence_threshold,
        silence_duration_seconds=settings.silence_duration_seconds,
        silence_cooldown_seconds=settings.silence_cooldown_seconds,
    )

    return HybridLyricPipeline(config)
