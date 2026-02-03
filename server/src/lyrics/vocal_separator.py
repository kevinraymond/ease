"""Vocal isolation using Demucs for improved lyric detection."""

import logging
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VocalSeparator:
    """Separates vocals from music using Demucs.

    This significantly improves Whisper transcription accuracy by
    removing instrumental interference before speech recognition.
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cuda",
        sample_rate: int = 44100,
    ):
        """Initialize the vocal separator.

        Args:
            model_name: Demucs model to use:
                - "htdemucs": Fast, good quality (default)
                - "htdemucs_ft": Fine-tuned, better quality, slower
                - "mdx_extra": Best quality, slowest
            device: Device to run on (cuda or cpu)
            sample_rate: Expected input sample rate (Demucs uses 44.1kHz internally)
        """
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self._model: Optional[Any] = None
        self._model_loaded = False

    def _ensure_model_loaded(self) -> bool:
        """Lazy load the Demucs model."""
        if self._model_loaded:
            return self._model is not None

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels

            logger.info(f"Loading Demucs model: {self.model_name} on {self.device}")

            self._model = get_model(self.model_name)
            if isinstance(self._model, BagOfModels):
                # Use first model from bag for speed
                self._model = self._model.models[0]

            self._model.to(self.device)
            self._model.eval()

            self._model_loaded = True
            logger.info("Demucs model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            self._model_loaded = True  # Mark as attempted
            return False

    def separate_vocals(
        self,
        audio: np.ndarray,
        input_sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        """Extract vocals from audio.

        Args:
            audio: Audio samples as float32 array (-1 to 1 range), mono or stereo
            input_sample_rate: Sample rate of input audio

        Returns:
            Vocals-only audio at 16kHz mono, or None on failure
        """
        if not self._ensure_model_loaded():
            return None

        try:
            import torchaudio

            # Convert to tensor
            if audio.ndim == 1:
                # Mono -> stereo (Demucs expects stereo)
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).repeat(2, 1)
            else:
                audio_tensor = torch.from_numpy(audio).float()

            # Resample to 44.1kHz if needed (Demucs native rate)
            if input_sample_rate != 44100:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, input_sample_rate, 44100
                )

            # Add batch dimension: (channels, samples) -> (1, channels, samples)
            audio_tensor = audio_tensor.unsqueeze(0).to(self.device)

            # Run separation
            with torch.no_grad():
                sources = self._apply_model(audio_tensor)

            # Extract vocals (index depends on model, typically index 3 for htdemucs)
            # htdemucs order: drums, bass, other, vocals
            vocal_idx = self._get_vocal_index()
            vocals = sources[:, vocal_idx]  # (1, 2, samples)

            # Convert to mono
            vocals_mono = vocals.mean(dim=1).squeeze(0)  # (samples,)

            # Resample to 16kHz for Whisper
            vocals_mono = torchaudio.functional.resample(
                vocals_mono.unsqueeze(0), 44100, 16000
            ).squeeze(0)

            return vocals_mono.cpu().numpy()

        except Exception as e:
            logger.error(f"Vocal separation error: {e}")
            return None

    def _apply_model(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply Demucs model to audio.

        Args:
            audio: Input tensor (batch, channels, samples)

        Returns:
            Separated sources (batch, sources, channels, samples)
        """
        from demucs.apply import apply_model

        # apply_model handles chunking internally for long audio
        return apply_model(
            self._model,
            audio,
            device=self.device,
            shifts=0,  # No random shifts (faster)
            overlap=0.25,
            progress=False,
        )

    def _get_vocal_index(self) -> int:
        """Get the index of vocals in the model's source order."""
        # Most Demucs models use: drums, bass, other, vocals
        if hasattr(self._model, 'sources'):
            sources = self._model.sources
            if 'vocals' in sources:
                return sources.index('vocals')
        # Default for htdemucs
        return 3
