"""Hybrid lyric provider using fingerprint-first with transcription fallback.

This provider wraps the existing HybridLyricPipeline to implement the
LyricProvider interface.
"""

import logging
from typing import Callable, Optional

import numpy as np

from ..provider import LyricProvider, LyricProviderState, LyricResult
from ..hybrid_pipeline import (
    HybridLyricPipeline,
    HybridPipelineConfig,
    PipelineState,
)

logger = logging.getLogger(__name__)


# Map internal pipeline states to provider states
_STATE_MAP = {
    PipelineState.INITIALIZING: LyricProviderState.INITIALIZING,
    PipelineState.FINGERPRINTING: LyricProviderState.FINGERPRINTING,
    PipelineState.MATCHED: LyricProviderState.MATCHED,
    PipelineState.NOT_MATCHED: LyricProviderState.PROCESSING,
    PipelineState.STOPPED: LyricProviderState.STOPPED,
}


class HybridProvider(LyricProvider):
    """Lyric provider using fingerprint matching with transcription fallback.

    Attempts to identify songs via audio fingerprinting first. If successful,
    uses pre-timed lyrics from the database. Otherwise, falls back to
    real-time Whisper transcription.
    """

    def __init__(
        self,
        config: Optional[HybridPipelineConfig] = None,
        # Convenience params that override config if provided
        fingerprint_enabled: bool = True,
        fingerprint_duration_seconds: float = 15.0,
        db_path: str = "./lyrics_db.sqlite",
        whisper_model_size: str = "tiny",
        whisper_device: str = "cuda",
        whisper_buffer_seconds: float = 2.0,
        whisper_beam_size: int = 1,
        transcribe_interval_seconds: float = 0.75,
        vocal_separation: bool = True,
        demucs_model: str = "htdemucs",
        demucs_device: str = "cuda",
        max_keywords: int = 8,
        silence_detection_enabled: bool = True,
        silence_threshold: float = 0.01,
        silence_duration_seconds: float = 1.5,
        silence_cooldown_seconds: float = 5.0,
    ):
        """Initialize the hybrid provider.

        Args:
            config: Full configuration (overrides other params if provided)
            fingerprint_enabled: Enable audio fingerprinting
            fingerprint_duration_seconds: Duration to collect for fingerprint
            db_path: Path to lyrics database
            whisper_model_size: Whisper model size
            whisper_device: Device for Whisper
            whisper_buffer_seconds: Rolling audio buffer duration
            whisper_beam_size: Beam size for decoding
            transcribe_interval_seconds: Seconds between transcription runs
            vocal_separation: Use Demucs for vocal separation
            demucs_model: Demucs model name
            demucs_device: Device for Demucs
            max_keywords: Maximum keywords to return
            silence_detection_enabled: Enable automatic song change detection
            silence_threshold: RMS threshold for silence detection
            silence_duration_seconds: How long silence must last
            silence_cooldown_seconds: Minimum time between auto-resets
        """
        if config is None:
            config = HybridPipelineConfig(
                fingerprint_enabled=fingerprint_enabled,
                fingerprint_duration_seconds=fingerprint_duration_seconds,
                db_path=db_path,
                whisper_model_size=whisper_model_size,
                whisper_device=whisper_device,
                whisper_buffer_seconds=whisper_buffer_seconds,
                whisper_beam_size=whisper_beam_size,
                transcribe_interval_seconds=transcribe_interval_seconds,
                vocal_separation=vocal_separation,
                demucs_model=demucs_model,
                demucs_device=demucs_device,
                max_keywords=max_keywords,
                silence_detection_enabled=silence_detection_enabled,
                silence_threshold=silence_threshold,
                silence_duration_seconds=silence_duration_seconds,
                silence_cooldown_seconds=silence_cooldown_seconds,
            )

        self._config = config
        self._pipeline: Optional[HybridLyricPipeline] = None
        self._on_song_change_callback: Optional[Callable[[], None]] = None

    def _ensure_pipeline(self) -> HybridLyricPipeline:
        """Lazy-load the pipeline."""
        if self._pipeline is None:
            self._pipeline = HybridLyricPipeline(self._config)

            # Forward song change events
            if self._on_song_change_callback:
                self._pipeline.set_on_song_change(self._on_song_change_callback)

        return self._pipeline

    def start(self) -> None:
        """Start the provider."""
        pipeline = self._ensure_pipeline()
        pipeline.start()
        logger.info("HybridProvider started")

    def stop(self) -> None:
        """Stop the provider."""
        if self._pipeline:
            self._pipeline.stop()
        logger.info("HybridProvider stopped")

    def add_audio_chunk(self, audio: np.ndarray, sample_rate: int) -> None:
        """Add audio for processing."""
        if self._pipeline:
            self._pipeline.add_audio_chunk(audio, sample_rate)

    def get_current_keywords(self) -> list[tuple[str, float]]:
        """Get current keywords for prompt modulation."""
        if self._pipeline:
            keywords = self._pipeline.get_current_keywords()
            # Convert from list[str] to list[tuple[str, float]] with decreasing weights
            return [(kw, 1.0 - i * 0.1) for i, kw in enumerate(keywords)]
        return []

    def get_result(self) -> Optional[LyricResult]:
        """Get the current lyric result."""
        if not self._pipeline:
            return None

        status = self._pipeline.get_status()

        # Build result from status
        matched_title = None
        matched_artist = None
        if status.song:
            matched_title = status.song.title
            matched_artist = status.song.artist

        return LyricResult(
            text=status.current_lyric_text,
            keywords=self.get_current_keywords(),
            confidence=0.9 if status.song else 0.7,
            is_singing=bool(status.current_keywords),
            language="en",
            matched_song_title=matched_title,
            matched_song_artist=matched_artist,
        )

    def reset(self) -> None:
        """Reset for a new song/session."""
        if self._pipeline:
            self._pipeline.reset()
        logger.info("HybridProvider reset")

    @property
    def state(self) -> LyricProviderState:
        """Current state of the provider."""
        if not self._pipeline:
            return LyricProviderState.INITIALIZING

        return _STATE_MAP.get(self._pipeline.state, LyricProviderState.INITIALIZING)

    @property
    def is_matched(self) -> bool:
        """Whether a song was matched from the database."""
        if self._pipeline:
            return self._pipeline.is_matched
        return False

    def update_playback_position(self, position_ms: int) -> None:
        """Update playback position for time-synced lyrics."""
        if self._pipeline:
            self._pipeline.update_playback_position(position_ms)

    def get_fingerprint_progress(self) -> float:
        """Get fingerprinting progress (0-1)."""
        if self._pipeline:
            status = self._pipeline.get_status()
            return status.fingerprint_progress
        return 0.0

    def get_transcription_latency_ms(self) -> float:
        """Get the latency of the last transcription."""
        if self._pipeline:
            status = self._pipeline.get_status()
            return status.transcription_latency_ms
        return 0.0

    def set_on_song_change(self, callback: Callable[[], None]) -> None:
        """Set callback for when song change is detected.

        Args:
            callback: Function to call when song change detected via silence.
        """
        self._on_song_change_callback = callback
        if self._pipeline:
            self._pipeline.set_on_song_change(callback)

    @property
    def last_fingerprint_hash(self) -> Optional[str]:
        """Get the last fingerprint hash (for learn feature)."""
        if self._pipeline:
            return self._pipeline.last_fingerprint_hash
        return None

    @property
    def last_fingerprint_raw(self) -> Optional[list[int]]:
        """Get the last raw fingerprint (for learn feature)."""
        if self._pipeline:
            return self._pipeline.last_fingerprint_raw
        return None

    @property
    def matched_song(self):
        """Get the matched song if any."""
        if self._pipeline:
            return self._pipeline.matched_song
        return None
