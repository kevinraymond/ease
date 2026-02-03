"""JPEG frame encoding for efficient transfer."""

import io
from typing import Optional

from PIL import Image
import numpy as np
from ..config import settings


class FrameEncoder:
    """Encodes PIL images to JPEG bytes for WebSocket transfer."""

    def __init__(self, quality: Optional[int] = None):
        self.quality = quality or settings.jpeg_quality

    def encode(self, image: Image.Image) -> bytes:
        """Encode a PIL image to JPEG bytes."""
        buffer = io.BytesIO()
        # Use JPEG for fast encoding and small size
        image.save(buffer, format="JPEG", quality=self.quality, optimize=False)
        return buffer.getvalue()

    def encode_numpy(self, array: np.ndarray) -> bytes:
        """Encode a numpy array (H, W, C) to JPEG bytes."""
        # Ensure uint8
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)

        image = Image.fromarray(array)
        return self.encode(image)

    def set_quality(self, quality: int) -> None:
        """Update JPEG quality (1-100)."""
        self.quality = max(1, min(100, quality))
