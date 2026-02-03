"""Whisper-based lyric provider using real-time transcription.

This provider wraps the existing LyricDetector, KeywordExtractor, and LyricBuffer
components to implement the LyricProvider interface.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from ..provider import LyricProvider, LyricProviderState, LyricResult
from ..detector import LyricDetector
from ..keywords import KeywordExtractor
from ..buffer import LyricBuffer

logger = logging.getLogger(__name__)


class WhisperProvider(LyricProvider):
    """Lyric provider using Whisper transcription.

    Uses faster-whisper for real-time transcription with optional Demucs
    vocal separation for improved accuracy in music.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cuda",
        compute_type: str = "float16",
        buffer_seconds: float = 2.0,
        beam_size: int = 1,
        vocal_separation: bool = True,
        demucs_model: str = "htdemucs",
        demucs_device: str = "cuda",
        transcribe_interval: float = 0.75,
        max_keywords: int = 8,
        keyword_decay_seconds: float = 10.0,
        filter_words: Optional[list[str]] = None,
    ):
        """Initialize the Whisper provider.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device for Whisper (cuda or cpu)
            compute_type: Compute type (float16, int8, float32)
            buffer_seconds: Rolling audio buffer duration
            beam_size: Beam size for decoding (1 = greedy/fast)
            vocal_separation: Use Demucs to isolate vocals
            demucs_model: Demucs model name
            demucs_device: Device for Demucs
            transcribe_interval: Seconds between transcription runs
            max_keywords: Maximum keywords to return
            keyword_decay_seconds: How long keywords stay active
            filter_words: Words to filter out (prompt leakage, false positives)
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._buffer_seconds = buffer_seconds
        self._beam_size = beam_size
        self._vocal_separation = vocal_separation
        self._demucs_model = demucs_model
        self._demucs_device = demucs_device
        self._transcribe_interval = transcribe_interval
        self._max_keywords = max_keywords
        self._keyword_decay_seconds = keyword_decay_seconds
        self._filter_words = frozenset(w.lower() for w in (filter_words or []))

        # Components (lazy loaded)
        self._detector: Optional[LyricDetector] = None
        self._keyword_extractor: Optional[KeywordExtractor] = None
        self._lyric_buffer: Optional[LyricBuffer] = None

        # State
        self._state = LyricProviderState.INITIALIZING
        self._running = False
        self._lock = threading.Lock()

        # Current result
        self._current_result: Optional[LyricResult] = None
        self._current_keywords: list[tuple[str, float]] = []
        self._last_transcribe_time: float = 0.0
        self._last_transcription_latency_ms: float = 0.0

        # Transcription thread
        self._transcribe_thread: Optional[threading.Thread] = None

    def _ensure_components(self) -> None:
        """Lazy-load the transcription components."""
        if self._detector is None:
            self._detector = LyricDetector(
                model_size=self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                buffer_seconds=self._buffer_seconds,
                beam_size=self._beam_size,
                vocal_separation=self._vocal_separation,
                demucs_model=self._demucs_model,
                demucs_device=self._demucs_device,
            )
            logger.info(f"LyricDetector initialized: model={self._model_size}, device={self._device}")

        if self._keyword_extractor is None:
            from ..keywords import KeywordExtractionConfig
            config = KeywordExtractionConfig(filter_words=self._filter_words)
            self._keyword_extractor = KeywordExtractor(config=config)

        if self._lyric_buffer is None:
            self._lyric_buffer = LyricBuffer(decay_seconds=self._keyword_decay_seconds)

    def start(self) -> None:
        """Start the provider."""
        if self._running:
            return

        self._ensure_components()
        self._running = True
        self._state = LyricProviderState.READY

        # Start transcription thread
        self._transcribe_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True,
        )
        self._transcribe_thread.start()

        logger.info("WhisperProvider started")

    def stop(self) -> None:
        """Stop the provider."""
        self._running = False
        self._state = LyricProviderState.STOPPED

        if self._transcribe_thread:
            self._transcribe_thread.join(timeout=2.0)
            self._transcribe_thread = None

        logger.info("WhisperProvider stopped")

    def add_audio_chunk(self, audio: np.ndarray, sample_rate: int) -> None:
        """Add audio for processing."""
        if not self._running or self._detector is None:
            return

        with self._lock:
            self._detector.add_audio_chunk(audio, sample_rate)
            if self._state == LyricProviderState.READY:
                self._state = LyricProviderState.PROCESSING

    def _transcription_loop(self) -> None:
        """Background thread for continuous transcription."""
        while self._running:
            now = time.time()
            if now - self._last_transcribe_time >= self._transcribe_interval:
                self._last_transcribe_time = now
                self._do_transcription()

            # Sleep a short interval to avoid busy-waiting
            time.sleep(0.05)

    def _do_transcription(self) -> None:
        """Perform one transcription cycle."""
        if self._detector is None or self._keyword_extractor is None or self._lyric_buffer is None:
            return

        try:
            start_time = time.time()
            result = self._detector.transcribe()
            self._last_transcription_latency_ms = (time.time() - start_time) * 1000

            with self._lock:
                if result and result.text.strip():
                    # Extract keywords from transcription
                    keywords = self._keyword_extractor.extract(result.text)
                    keyword_weights = [(word, weight) for word, weight in keywords[:self._max_keywords]]

                    if keyword_weights:
                        self._lyric_buffer.add_words(keyword_weights)

                    # Update current result
                    active_keywords = self._lyric_buffer.get_active_keywords(max_count=self._max_keywords)
                    self._current_keywords = active_keywords

                    self._current_result = LyricResult(
                        text=result.text[-200:] if result.text else "",
                        keywords=active_keywords,
                        confidence=0.7,  # Could derive from result.language_probability
                        is_singing=result.is_singing,
                        language=result.language,
                    )
                else:
                    # No transcription, but still update keywords from buffer
                    active_keywords = self._lyric_buffer.get_active_keywords(max_count=self._max_keywords)
                    self._current_keywords = active_keywords

                    if self._current_result:
                        # Update keywords in existing result
                        self._current_result = LyricResult(
                            text=self._current_result.text,
                            keywords=active_keywords,
                            confidence=self._current_result.confidence,
                            is_singing=bool(active_keywords),
                            language=self._current_result.language,
                        )

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def get_current_keywords(self) -> list[tuple[str, float]]:
        """Get current keywords for prompt modulation."""
        with self._lock:
            return self._current_keywords.copy()

    def get_result(self) -> Optional[LyricResult]:
        """Get the current lyric result."""
        with self._lock:
            return self._current_result

    def reset(self) -> None:
        """Reset for a new song/session."""
        with self._lock:
            if self._detector:
                self._detector.clear_buffer()
            if self._lyric_buffer:
                self._lyric_buffer.clear()

            self._current_result = None
            self._current_keywords = []
            self._last_transcribe_time = 0.0

            if self._running:
                self._state = LyricProviderState.READY
            else:
                self._state = LyricProviderState.INITIALIZING

        logger.info("WhisperProvider reset")

    @property
    def state(self) -> LyricProviderState:
        """Current state of the provider."""
        return self._state

    def get_transcription_latency_ms(self) -> float:
        """Get the latency of the last transcription."""
        return self._last_transcription_latency_ms
