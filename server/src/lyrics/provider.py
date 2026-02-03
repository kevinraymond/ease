"""Abstract interface for lyric detection providers.

This module defines the plug-and-play interface for lyric detection systems.
Different providers can implement this interface to provide lyrics through
various methods (Whisper transcription, fingerprint matching, external APIs, etc.).

Example usage:
    from lyrics import create_lyric_provider

    provider = create_lyric_provider("whisper")
    provider.start()

    # In your audio processing loop:
    provider.add_audio_chunk(audio, sample_rate)

    # Get keywords for visual prompts
    keywords = provider.get_current_keywords()

    # Check full result when needed
    result = provider.get_result()
    if result and result.is_singing:
        print(f"Lyrics: {result.text}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class LyricProviderState(Enum):
    """State of the lyric provider."""

    INITIALIZING = auto()  # Setting up, not ready
    READY = auto()  # Ready to receive audio
    PROCESSING = auto()  # Actively processing audio
    FINGERPRINTING = auto()  # Collecting audio for fingerprint (hybrid mode)
    MATCHED = auto()  # Song identified from database (hybrid mode)
    STOPPED = auto()  # Provider stopped


@dataclass
class LyricResult:
    """Result from lyric detection.

    Attributes:
        text: The detected/transcribed text
        keywords: List of (keyword, weight) tuples for prompt modulation
        confidence: Overall confidence score (0-1)
        is_singing: Whether detected as singing vs spoken word
        language: Detected language code
        matched_song_title: Title if song was matched from database
        matched_song_artist: Artist if song was matched from database
    """

    text: str
    keywords: list[tuple[str, float]]
    confidence: float
    is_singing: bool
    language: str = "en"
    matched_song_title: Optional[str] = None
    matched_song_artist: Optional[str] = None


class LyricProvider(ABC):
    """Abstract base class for lyric detection providers.

    Providers must implement this interface to be usable with the EASE system.
    The interface supports both synchronous (Whisper) and asynchronous (fingerprint)
    lyric detection approaches.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the provider.

        This should initialize any models, threads, or connections needed
        for lyric detection. May be called multiple times (should be idempotent).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the provider.

        Clean up any resources, stop threads, close connections.
        Should be safe to call multiple times.
        """
        pass

    @abstractmethod
    def add_audio_chunk(self, audio: np.ndarray, sample_rate: int) -> None:
        """Add an audio chunk for processing.

        Args:
            audio: Audio samples as float32 array (-1 to 1 range)
            sample_rate: Sample rate of the audio (typically 16000 or 48000)
        """
        pass

    @abstractmethod
    def get_current_keywords(self) -> list[tuple[str, float]]:
        """Get the current keywords for prompt modulation.

        Returns:
            List of (keyword, weight) tuples, sorted by relevance.
            Empty list if no keywords available.
        """
        pass

    @abstractmethod
    def get_result(self) -> Optional[LyricResult]:
        """Get the current lyric detection result.

        Returns:
            LyricResult with current detection state, or None if not available.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the provider for a new song/session.

        Clears all buffers, cached results, and resets state.
        Called when user triggers a reset or song change is detected.
        """
        pass

    @property
    @abstractmethod
    def state(self) -> LyricProviderState:
        """Current state of the provider."""
        pass

    @property
    def is_ready(self) -> bool:
        """Whether the provider is ready to process audio."""
        return self.state in (
            LyricProviderState.READY,
            LyricProviderState.PROCESSING,
            LyricProviderState.FINGERPRINTING,
            LyricProviderState.MATCHED,
        )

    @property
    def is_matched(self) -> bool:
        """Whether a song was matched from a database (for hybrid providers)."""
        return self.state == LyricProviderState.MATCHED

    def update_playback_position(self, position_ms: int) -> None:
        """Update playback position for time-synced lyrics.

        Override in providers that support timed lyrics (e.g., from database).

        Args:
            position_ms: Current playback position in milliseconds
        """
        pass

    def get_fingerprint_progress(self) -> float:
        """Get fingerprinting progress (0-1) for providers that use fingerprinting.

        Override in providers that support audio fingerprinting.

        Returns:
            Progress value between 0 and 1, or 1.0 if not fingerprinting.
        """
        return 1.0

    def get_transcription_latency_ms(self) -> float:
        """Get the latency of the last transcription in milliseconds.

        Override in providers that track transcription latency.

        Returns:
            Latency in milliseconds, or 0.0 if not tracked.
        """
        return 0.0
