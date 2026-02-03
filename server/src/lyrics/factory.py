"""Factory for creating lyric providers.

This module provides a factory function to create lyric providers based on
configuration settings, enabling plug-and-play switching between different
lyric detection implementations.
"""

import logging
from typing import Literal

from .provider import LyricProvider
from .providers.whisper_provider import WhisperProvider
from .providers.hybrid_provider import HybridProvider

logger = logging.getLogger(__name__)

# Type alias for provider names
LyricProviderType = Literal["whisper", "hybrid", "none"]


class NullLyricProvider(LyricProvider):
    """A no-op lyric provider that does nothing.

    Used when lyrics are disabled but the interface is still needed.
    """

    def __init__(self):
        from .provider import LyricProviderState
        self._state = LyricProviderState.STOPPED

    def start(self) -> None:
        from .provider import LyricProviderState
        self._state = LyricProviderState.READY

    def stop(self) -> None:
        from .provider import LyricProviderState
        self._state = LyricProviderState.STOPPED

    def add_audio_chunk(self, audio, sample_rate: int) -> None:
        pass

    def get_current_keywords(self) -> list[tuple[str, float]]:
        return []

    def get_result(self):
        return None

    def reset(self) -> None:
        pass

    @property
    def state(self):
        return self._state


def create_lyric_provider(
    provider_type: LyricProviderType = "whisper",
    # Whisper settings
    model_size: str = "tiny",
    device: str = "cuda",
    compute_type: str = "float16",
    buffer_seconds: float = 2.0,
    beam_size: int = 1,
    transcribe_interval: float = 0.75,
    # Vocal separation
    vocal_separation: bool = True,
    demucs_model: str = "htdemucs",
    demucs_device: str = "cuda",
    # Hybrid/fingerprint settings
    fingerprint_enabled: bool = True,
    fingerprint_duration_seconds: float = 15.0,
    db_path: str = "./lyrics_db.sqlite",
    # Silence detection
    silence_detection_enabled: bool = True,
    silence_threshold: float = 0.01,
    silence_duration_seconds: float = 1.5,
    silence_cooldown_seconds: float = 5.0,
    # Keywords
    max_keywords: int = 8,
    keyword_decay_seconds: float = 10.0,
    filter_words: list[str] | None = None,
) -> LyricProvider:
    """Create a lyric provider based on type.

    Args:
        provider_type: Type of provider ("whisper", "hybrid", or "none")
        model_size: Whisper model size
        device: Device for Whisper
        compute_type: Compute type for Whisper
        buffer_seconds: Rolling audio buffer duration
        beam_size: Beam size for decoding
        transcribe_interval: Seconds between transcription runs
        vocal_separation: Use Demucs for vocal separation
        demucs_model: Demucs model name
        demucs_device: Device for Demucs
        fingerprint_enabled: Enable audio fingerprinting (hybrid only)
        fingerprint_duration_seconds: Duration for fingerprinting (hybrid only)
        db_path: Path to lyrics database (hybrid only)
        silence_detection_enabled: Enable automatic song change detection
        silence_threshold: RMS threshold for silence detection
        silence_duration_seconds: How long silence must last
        silence_cooldown_seconds: Minimum time between auto-resets
        max_keywords: Maximum keywords to return
        keyword_decay_seconds: How long keywords stay active
        filter_words: Words to filter out (prompt leakage, false positives)

    Returns:
        Configured LyricProvider instance
    """
    if provider_type == "none":
        logger.info("Creating NullLyricProvider (lyrics disabled)")
        return NullLyricProvider()

    elif provider_type == "whisper":
        logger.info(f"Creating WhisperProvider: model={model_size}, device={device}")
        return WhisperProvider(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            buffer_seconds=buffer_seconds,
            beam_size=beam_size,
            vocal_separation=vocal_separation,
            demucs_model=demucs_model,
            demucs_device=demucs_device,
            transcribe_interval=transcribe_interval,
            max_keywords=max_keywords,
            keyword_decay_seconds=keyword_decay_seconds,
            filter_words=filter_words,
        )

    elif provider_type == "hybrid":
        logger.info(f"Creating HybridProvider: fingerprint={fingerprint_enabled}, model={model_size}")
        return HybridProvider(
            fingerprint_enabled=fingerprint_enabled,
            fingerprint_duration_seconds=fingerprint_duration_seconds,
            db_path=db_path,
            whisper_model_size=model_size,
            whisper_device=device,
            whisper_buffer_seconds=buffer_seconds,
            whisper_beam_size=beam_size,
            transcribe_interval_seconds=transcribe_interval,
            vocal_separation=vocal_separation,
            demucs_model=demucs_model,
            demucs_device=demucs_device,
            max_keywords=max_keywords,
            silence_detection_enabled=silence_detection_enabled,
            silence_threshold=silence_threshold,
            silence_duration_seconds=silence_duration_seconds,
            silence_cooldown_seconds=silence_cooldown_seconds,
        )

    else:
        raise ValueError(f"Unknown lyric provider type: {provider_type}")


def create_lyric_provider_from_settings() -> LyricProvider:
    """Create a lyric provider from global settings.

    Reads configuration from the global Settings object and creates
    the appropriate provider.

    Returns:
        Configured LyricProvider instance
    """
    from ..config import settings

    provider_type = getattr(settings, "lyric_provider", "whisper")

    return create_lyric_provider(
        provider_type=provider_type,
        model_size=settings.lyric_model_size,
        device=settings.lyric_device,
        compute_type=settings.lyric_compute_type,
        buffer_seconds=settings.lyric_buffer_seconds,
        beam_size=settings.lyric_beam_size,
        transcribe_interval=settings.lyric_transcribe_interval,
        vocal_separation=settings.lyric_vocal_separation,
        demucs_model=settings.lyric_demucs_model,
        demucs_device=settings.lyric_demucs_device,
        fingerprint_enabled=settings.fingerprint_enabled,
        fingerprint_duration_seconds=settings.fingerprint_duration_seconds,
        db_path=settings.fingerprint_db_path,
        silence_detection_enabled=settings.silence_detection_enabled,
        silence_threshold=settings.silence_threshold,
        silence_duration_seconds=settings.silence_duration_seconds,
        silence_cooldown_seconds=settings.silence_cooldown_seconds,
        filter_words=getattr(settings, "lyric_filter_words", None),
    )
