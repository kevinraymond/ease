"""Audio fingerprinting using Chromaprint for song identification.

Uses pyacoustid (Chromaprint wrapper) to generate compact fingerprints from audio.
These fingerprints can be matched against a local database of known songs to enable
instant lyric lookup without transcription.

## How It Works

Chromaprint works by:
1. Computing a spectrogram of the audio
2. Extracting pitch-class chroma features
3. Comparing adjacent frames to detect changes
4. Encoding the pattern of changes as a compact binary string

The resulting fingerprint is robust to:
- Volume changes
- Compression artifacts
- Minor pitch shifts
- Background noise

## Usage

```python
fingerprinter = Fingerprinter()

# Fingerprint audio buffer
fp_hash = fingerprinter.fingerprint(audio_samples, sample_rate=44100)

# Compare against database
match = fingerprinter.match(fp_hash, database)
if match:
    print(f"Matched song: {match.title} by {match.artist}")
```

## Dependencies

Requires `pyacoustid` which wraps the Chromaprint library:
- Linux: `sudo apt install libchromaprint-tools` (or bundled with pyacoustid)
- macOS: `brew install chromaprint`
- Windows: Included with pyacoustid wheel
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import chromaprint
try:
    import chromaprint
    CHROMAPRINT_AVAILABLE = True
except ImportError:
    CHROMAPRINT_AVAILABLE = False
    logger.warning("chromaprint not available - fingerprinting disabled")


@dataclass
class FingerprintMatch:
    """Result of a fingerprint match against the database."""
    song_id: int
    fingerprint_hash: str
    title: str
    artist: str
    confidence: float  # 0-1, how well the fingerprints match


class Fingerprinter:
    """Generates and matches audio fingerprints using Chromaprint.

    Fingerprints are compact representations of audio that can be used
    to identify songs even with moderate audio quality degradation.
    """

    def __init__(
        self,
        duration_seconds: float = 15.0,
        target_sample_rate: int = 44100,
    ):
        """Initialize the fingerprinter.

        Args:
            duration_seconds: Amount of audio to use for fingerprinting.
                             More audio = more accurate but slower to match.
                             15s is a good balance for most music.
            target_sample_rate: Sample rate for fingerprint generation.
                               Chromaprint expects 44100 Hz.
        """
        self.duration_seconds = duration_seconds
        self.target_sample_rate = target_sample_rate
        self._fingerprint_cache: dict[str, str] = {}

    def is_available(self) -> bool:
        """Check if Chromaprint is available."""
        return CHROMAPRINT_AVAILABLE

    def fingerprint(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Optional[str]:
        """Generate a fingerprint hash from audio samples.

        Args:
            audio: Audio samples as numpy array (mono or stereo, any dtype).
                   Will be converted to 16-bit PCM internally.
            sample_rate: Sample rate of the input audio.

        Returns:
            Fingerprint hash string, or None if fingerprinting failed.
        """
        if not CHROMAPRINT_AVAILABLE:
            logger.warning("Chromaprint not available, skipping fingerprint")
            return None

        try:
            # Ensure we have enough audio
            min_samples = int(1.0 * sample_rate)  # At least 1 second
            if len(audio) < min_samples:
                logger.debug(f"Not enough audio for fingerprint: {len(audio)} < {min_samples}")
                return None

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                audio = self._resample(audio, sample_rate, self.target_sample_rate)

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Limit to duration_seconds
            max_samples = int(self.duration_seconds * self.target_sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Convert to 16-bit signed PCM (what Chromaprint expects)
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                # Assume float audio is in -1 to 1 range
                audio_int16 = (audio * 32767).astype(np.int16)
            elif audio.dtype != np.int16:
                audio_int16 = audio.astype(np.int16)
            else:
                audio_int16 = audio

            # Generate fingerprint using Fingerprinter class
            ctx = chromaprint.Fingerprinter()
            ctx.start(self.target_sample_rate, 1)  # 1 channel (mono)
            ctx.feed(audio_int16.tobytes())

            # finish() returns the raw fingerprint as bytes
            raw_fp = ctx.finish()
            if not raw_fp:
                logger.warning("Chromaprint returned empty fingerprint")
                return None

            # Encode to base64-like string (returns bytes, decode to str)
            fp_encoded = chromaprint.encode_fingerprint(raw_fp, algorithm=1)

            if fp_encoded:
                fp_hash = fp_encoded.decode('ascii') if isinstance(fp_encoded, bytes) else fp_encoded
                duration_sec = len(audio) / self.target_sample_rate
                logger.debug(f"Generated fingerprint: {fp_hash[:32]}... (duration={duration_sec:.1f}s)")
                return fp_hash

            return None

        except Exception as e:
            logger.error(f"Fingerprint generation failed: {e}")
            return None

    def fingerprint_raw(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Optional[list[int]]:
        """Generate raw fingerprint data (not encoded).

        This returns the raw integer array before base64 encoding,
        which is useful for fuzzy matching and partial comparisons.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate of the input audio.

        Returns:
            List of fingerprint integers, or None if failed.
        """
        if not CHROMAPRINT_AVAILABLE:
            return None

        try:
            # Ensure we have enough audio
            min_samples = int(1.0 * sample_rate)
            if len(audio) < min_samples:
                return None

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                audio = self._resample(audio, sample_rate, self.target_sample_rate)

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Limit to duration_seconds
            max_samples = int(self.duration_seconds * self.target_sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Convert to 16-bit signed PCM
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio_int16 = (audio * 32767).astype(np.int16)
            elif audio.dtype != np.int16:
                audio_int16 = audio.astype(np.int16)
            else:
                audio_int16 = audio

            # Generate raw fingerprint
            ctx = chromaprint.Fingerprinter()
            ctx.start(self.target_sample_rate, 1)  # mono
            ctx.feed(audio_int16.tobytes())

            # finish() returns the raw fingerprint as bytes
            raw_fp = ctx.finish()
            if not raw_fp:
                return None

            # Convert bytes to list of integers (each 4 bytes = 1 int32)
            import struct
            num_ints = len(raw_fp) // 4
            fp_ints = list(struct.unpack(f'{num_ints}I', raw_fp[:num_ints * 4]))
            logger.info(f"fingerprint_raw: generated {len(fp_ints)} integers from {len(raw_fp)} bytes")
            return fp_ints

        except Exception as e:
            logger.error(f"Raw fingerprint generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compare_fingerprints(
        self,
        fp1: str,
        fp2: str,
    ) -> float:
        """Compare two fingerprint hashes and return similarity score.

        Args:
            fp1: First fingerprint hash
            fp2: Second fingerprint hash

        Returns:
            Similarity score from 0.0 (no match) to 1.0 (identical).
            Typically, scores > 0.5 indicate a strong match.
        """
        if not CHROMAPRINT_AVAILABLE:
            return 0.0

        if not fp1 or not fp2:
            return 0.0

        try:
            # Decode fingerprints to raw integers
            # decode_fingerprint returns (fingerprint_list, algorithm)
            decoded1 = chromaprint.decode_fingerprint(fp1)
            decoded2 = chromaprint.decode_fingerprint(fp2)

            if not decoded1 or not decoded2:
                return 0.0

            fp1_ints, _ = decoded1
            fp2_ints, _ = decoded2

            if not fp1_ints or not fp2_ints:
                return 0.0

            # Calculate similarity using bit-level comparison
            # Count matching bits across aligned fingerprint segments
            min_len = min(len(fp1_ints), len(fp2_ints))
            if min_len == 0:
                return 0.0

            matching_bits = 0
            total_bits = min_len * 32  # 32 bits per integer

            for i in range(min_len):
                # XOR gives us differing bits, popcount gives us count
                diff = fp1_ints[i] ^ fp2_ints[i]
                # Count differing bits
                diff_bits = bin(diff & 0xFFFFFFFF).count('1')
                matching_bits += 32 - diff_bits

            similarity = matching_bits / total_bits
            return similarity

        except Exception as e:
            logger.error(f"Fingerprint comparison failed: {e}")
            return 0.0

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

        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


# Singleton instance for convenience
_fingerprinter: Optional[Fingerprinter] = None


def get_fingerprinter() -> Fingerprinter:
    """Get or create the global Fingerprinter instance."""
    global _fingerprinter
    if _fingerprinter is None:
        _fingerprinter = Fingerprinter()
    return _fingerprinter
