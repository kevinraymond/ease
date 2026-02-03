"""Lyric detection and keyword extraction module.

Provides real-time lyric detection using faster-whisper and
keyword extraction for injecting detected lyrics into prompts.
Optionally uses Demucs for vocal isolation to improve accuracy.

## Plug-and-Play Provider Interface (Recommended)

Use the LyricProvider interface for plug-and-play lyric detection:

```python
from lyrics import create_lyric_provider, create_lyric_provider_from_settings

# Create from global settings (recommended)
provider = create_lyric_provider_from_settings()
provider.start()

# Or create with specific type
provider = create_lyric_provider("hybrid")
provider.start()

# Feed audio continuously
provider.add_audio_chunk(audio, sample_rate)

# Get keywords for visual prompts
keywords = provider.get_current_keywords()

# Get full result
result = provider.get_result()
if result and result.is_singing:
    print(f"Lyrics: {result.text}")
```

## Available Providers

- `whisper`: Real-time Whisper transcription (default)
- `hybrid`: Fingerprint-first with transcription fallback
- `none`: No-op provider for when lyrics are disabled

## Import Lyrics

Use the import_lrc CLI to add songs to the database:

```bash
uv run python -m lyrics.import_lrc song.mp3 lyrics.lrc
```
"""

# Provider interface (recommended entry point)
from .provider import LyricProvider, LyricProviderState, LyricResult
from .factory import create_lyric_provider, create_lyric_provider_from_settings

# Provider implementations
from .providers import WhisperProvider, HybridProvider

# Legacy/advanced components
from .detector import LyricDetector, TranscriptionResult, TranscriptionSegment
from .keywords import KeywordExtractor, KeywordExtractionConfig, VISUAL_WORDS, EMOTION_WORDS
from .buffer import LyricBuffer, LyricBufferConfig, WordEntry
from .vocal_separator import VocalSeparator
from .fingerprinter import Fingerprinter, FingerprintMatch, get_fingerprinter
from .lyric_database import (
    LyricDatabase,
    Song,
    LyricLine,
    get_database,
)
from .hybrid_pipeline import (
    HybridLyricPipeline,
    HybridPipelineConfig,
    PipelineState,
    PipelineStatus,
    create_pipeline_from_settings,
)

__all__ = [
    # Provider interface (recommended entry point)
    "LyricProvider",
    "LyricProviderState",
    "LyricResult",
    "create_lyric_provider",
    "create_lyric_provider_from_settings",
    # Provider implementations
    "WhisperProvider",
    "HybridProvider",
    # Hybrid pipeline (legacy, use HybridProvider instead)
    "HybridLyricPipeline",
    "HybridPipelineConfig",
    "PipelineState",
    "PipelineStatus",
    "create_pipeline_from_settings",
    # Detector (for fallback/advanced use)
    "LyricDetector",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Vocal separation
    "VocalSeparator",
    # Keywords
    "KeywordExtractor",
    "KeywordExtractionConfig",
    "VISUAL_WORDS",
    "EMOTION_WORDS",
    # Buffer
    "LyricBuffer",
    "LyricBufferConfig",
    "WordEntry",
    # Fingerprinting
    "Fingerprinter",
    "FingerprintMatch",
    "get_fingerprinter",
    # Database
    "LyricDatabase",
    "Song",
    "LyricLine",
    "get_database",
]
