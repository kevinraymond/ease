"""Lyric detection provider implementations."""

from .whisper_provider import WhisperProvider
from .hybrid_provider import HybridProvider

__all__ = [
    "WhisperProvider",
    "HybridProvider",
]
