"""Real-time lyric detection using faster-whisper with optional vocal separation.

Optimized for lyrics accuracy with:
- large-v3-turbo model by default (6x faster than large, good accuracy)
- 2s buffer for low latency
- beam_size=1 (greedy decoding, faster than beam search)
- initial_prompt hint for singing/music context
- Optional Demucs vocal separation for noisy audio
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A single transcribed segment with timing info."""

    text: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionResult:
    """Result of a transcription run."""

    text: str
    segments: list[TranscriptionSegment]
    is_singing: bool  # True if detected as singing vs spoken word
    language: str
    language_probability: float


class LyricDetector:
    """Detects lyrics from audio using faster-whisper.

    Maintains a rolling buffer of audio and transcribes periodically.
    Designed for real-time lyric detection during live audio playback.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cuda",
        compute_type: str = "float16",
        buffer_seconds: float = 2.0,
        target_sample_rate: int = 16000,
        vad_filter: bool = True,
        vad_threshold: float = 0.3,
        vocal_separation: bool = False,
        demucs_model: str = "htdemucs",
        demucs_device: str = "cuda",
        beam_size: int = 1,
        initial_prompt: str = "lyrics, singing, music",
    ):
        """Initialize the lyric detector.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda or cpu)
            compute_type: Compute type (float16, int8, float32)
            buffer_seconds: Rolling buffer duration in seconds (default 2s for low latency)
            target_sample_rate: Target sample rate for Whisper (16kHz)
            vad_filter: Whether to use VAD filtering (disable for heavy instrumentals)
            vad_threshold: VAD threshold (lower = more sensitive)
            vocal_separation: Use Demucs to isolate vocals before transcription
            demucs_model: Demucs model name (htdemucs, htdemucs_ft, mdx_extra)
            demucs_device: Device for Demucs (cuda or cpu)
            beam_size: Beam size for decoding (1 = greedy/fast, 5 = beam search/better)
            initial_prompt: Initial prompt to hint transcription context
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.target_sample_rate = target_sample_rate
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.vocal_separation = vocal_separation
        self.demucs_model = demucs_model
        self.demucs_device = demucs_device
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt

        # Rolling audio buffer (maxlen in samples)
        buffer_samples = int(buffer_seconds * target_sample_rate)
        self._audio_buffer: deque[float] = deque(maxlen=buffer_samples)

        # Track what we've already transcribed to avoid duplicates
        self._last_transcription = ""
        self._last_transcription_time = 0.0

        # Model loaded lazily
        self._model = None
        self._model_loaded = False

        # Vocal separator loaded lazily
        self._vocal_separator = None
        self._vocal_separator_loaded = False

    def _ensure_model_loaded(self) -> bool:
        """Lazy load the Whisper model."""
        if self._model_loaded:
            return self._model is not None

        try:
            from faster_whisper import WhisperModel

            logger.info(
                f"Loading faster-whisper model: {self.model_size} on {self.device}"
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._model_loaded = True
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self._model_loaded = True  # Mark as attempted
            return False

    def _ensure_separator_loaded(self) -> bool:
        """Lazy load the Demucs vocal separator."""
        if not self.vocal_separation:
            return False

        if self._vocal_separator_loaded:
            return self._vocal_separator is not None

        try:
            from .vocal_separator import VocalSeparator

            logger.info(
                f"Loading Demucs model: {self.demucs_model} on {self.demucs_device}"
            )
            self._vocal_separator = VocalSeparator(
                model_name=self.demucs_model,
                device=self.demucs_device,
            )
            self._vocal_separator_loaded = True
            logger.info("Demucs model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            self._vocal_separator_loaded = True  # Mark as attempted
            return False

    def add_audio_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Add an audio chunk to the buffer.

        Resamples to 16kHz mono if needed.

        Args:
            audio: Audio samples as float32 array (-1 to 1 range)
            sample_rate: Sample rate of the input audio
        """
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            audio = self._resample(audio, sample_rate, self.target_sample_rate)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Add to buffer
        self._audio_buffer.extend(audio.tolist())

    def _resample(
        self,
        audio: np.ndarray,
        from_rate: int,
        to_rate: int,
    ) -> np.ndarray:
        """Simple linear resampling."""
        if from_rate == to_rate:
            return audio

        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)

        # Use simple linear interpolation
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def transcribe(self) -> Optional[TranscriptionResult]:
        """Run Whisper on the buffered audio.

        If vocal separation is enabled, uses Demucs to isolate vocals first.
        Returns new transcription result, or None if buffer is too small.

        Latency is logged for pipeline optimization.
        """
        transcribe_start_time = time.time()

        if not self._ensure_model_loaded():
            return None

        if len(self._audio_buffer) < self.target_sample_rate:  # At least 1 second
            return None

        # Convert buffer to numpy array
        audio = np.array(list(self._audio_buffer), dtype=np.float32)
        buffer_duration_ms = (len(audio) / self.target_sample_rate) * 1000

        try:
            # Optional: Separate vocals using Demucs
            separation_time_ms = 0.0
            if self._ensure_separator_loaded():
                sep_start = time.time()
                vocals = self._vocal_separator.separate_vocals(
                    audio, input_sample_rate=self.target_sample_rate
                )
                separation_time_ms = (time.time() - sep_start) * 1000
                if vocals is not None:
                    logger.debug(f"Vocal separation: {separation_time_ms:.0f}ms, "
                               f"input={len(audio)} samples, output={len(vocals)} samples")
                    audio = vocals
                else:
                    logger.warning("Vocal separation failed, using original audio")

            # Transcribe with Whisper
            # Optimized for low latency:
            # - beam_size=1 (greedy) is ~2x faster than beam_size=5
            # - initial_prompt hints music context for better accuracy
            # - VAD tuned for music vocals
            transcribe_kwargs = {
                "beam_size": self.beam_size,
                "vad_filter": self.vad_filter,
                "initial_prompt": self.initial_prompt,
            }
            if self.vad_filter:
                transcribe_kwargs["vad_parameters"] = dict(
                    threshold=self.vad_threshold,  # Lower = more sensitive (good for music vocals)
                    min_speech_duration_ms=150,  # Shorter bursts common in lyrics
                    min_silence_duration_ms=100,  # Faster transitions in music
                    speech_pad_ms=200,  # Add padding around detected speech
                )

            whisper_start = time.time()
            segments, info = self._model.transcribe(audio, **transcribe_kwargs)
            whisper_time_ms = (time.time() - whisper_start) * 1000

            # Collect segments
            result_segments = []
            full_text_parts = []

            for segment in segments:
                result_segments.append(
                    TranscriptionSegment(
                        text=segment.text.strip(),
                        start=segment.start,
                        end=segment.end,
                        confidence=segment.avg_logprob,
                    )
                )
                full_text_parts.append(segment.text.strip())

            full_text = " ".join(full_text_parts)

            # Detect if this is likely singing vs speaking
            # Singing tends to have longer segments and more repetition
            is_singing = self._detect_singing(result_segments, info)

            result = TranscriptionResult(
                text=full_text,
                segments=result_segments,
                is_singing=is_singing,
                language=info.language,
                language_probability=info.language_probability,
            )

            self._last_transcription = full_text

            # Log latency breakdown for pipeline optimization
            total_time_ms = (time.time() - transcribe_start_time) * 1000
            logger.info(
                f"Transcription latency: total={total_time_ms:.0f}ms "
                f"(buffer={buffer_duration_ms:.0f}ms, "
                f"separation={separation_time_ms:.0f}ms, "
                f"whisper={whisper_time_ms:.0f}ms) "
                f"text='{full_text[:50]}...'" if len(full_text) > 50 else f"text='{full_text}'"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _detect_singing(
        self,
        segments: list[TranscriptionSegment],
        info,
    ) -> bool:
        """Heuristic to detect if audio is singing vs speaking.

        Singing typically has:
        - More sustained sounds (longer segment durations)
        - More repetitive words
        - Music detected in background
        """
        if not segments:
            return False

        # Average segment duration
        if len(segments) > 0:
            avg_duration = sum(s.end - s.start for s in segments) / len(segments)
            # Singing tends to have longer sustained sounds
            if avg_duration > 2.0:
                return True

        # Check for word repetition (common in lyrics)
        all_words = " ".join(s.text for s in segments).lower().split()
        if len(all_words) > 5:
            unique_ratio = len(set(all_words)) / len(all_words)
            # Low unique ratio = more repetition = likely lyrics
            if unique_ratio < 0.6:
                return True

        return False

    def get_recent_words(self, seconds: float = 5.0) -> list[str]:
        """Get words from the most recent N seconds of transcription.

        Args:
            seconds: How far back to look

        Returns:
            List of words from recent transcription
        """
        if not self._last_transcription:
            return []

        # Simple approach: return last N words based on rough estimate
        # (Assumes ~3 words per second of speech)
        word_count = int(seconds * 3)
        words = self._last_transcription.split()
        return words[-word_count:] if len(words) > word_count else words

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self._audio_buffer) / self.target_sample_rate

    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        self._audio_buffer.clear()
        self._last_transcription = ""
