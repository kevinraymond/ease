"""Keyframe + Interpolation Hybrid for efficient high-quality generation.

Generates keyframes with full diffusion every N frames and uses
optical flow interpolation (RIFE) for intermediate frames.
This provides 4-8x compute reduction while maintaining quality.

RIFE (Real-time Intermediate Flow Estimation) loading priority:
1. hzwer/rife_model package (pip install rife)
2. torch.hub from hzwer/RIFE
3. Fallback to simple linear blend (no RIFE available)

For best results, install RIFE:
    pip install rife  # or: git clone https://github.com/hzwer/RIFE
"""

import torch
import logging
from typing import Any, Optional, List, Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class KeyframeConfig:
    """Configuration for keyframe interpolation."""

    keyframe_interval: int = 4          # Generate keyframe every N frames
    interpolation_mode: str = "rife"    # "rife", "optical_flow", "blend"
    rife_scale: float = 1.0             # RIFE model scale (0.5 for faster)
    motion_threshold: float = 0.3       # Skip interpolation if motion too high
    audio_adaptive: bool = True         # Adapt interval based on audio


class RIFEInterpolator:
    """RIFE (Real-time Intermediate Flow Estimation) for frame interpolation.

    RIFE is a lightweight neural network that generates intermediate frames
    using bidirectional optical flow.
    """

    def __init__(self, device: str = "cuda", scale: float = 1.0):
        self.device = device
        self.scale = scale
        self._model: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> None:
        """Load RIFE model."""
        if self._initialized:
            return

        try:
            # Try to load RIFE from various sources
            self._load_rife_model()
            self._initialized = True
            logger.info("RIFE interpolator initialized")

        except Exception as e:
            logger.warning(f"RIFE not available, falling back to blend: {e}")
            self._model = None

    def _load_rife_model(self) -> None:
        """Load RIFE model from available sources.

        Tries multiple loading methods in order of preference:
        1. hzwer/rife_model package (fastest, most reliable)
        2. torch.hub from hzwer/RIFE (requires network)
        3. Falls back to linear blend if both fail
        """
        # Method 1: Try hzwer's RIFE package
        try:
            from rife_model import Model as RIFE
            logger.info("Loading RIFE from rife_model package...")
            self._model = RIFE()
            self._model.load_model("rife-v4.6", -1)  # -1 for latest
            self._model.eval()
            self._model.device()
            logger.info("RIFE loaded successfully from rife_model package")
            return
        except ImportError:
            logger.debug("rife_model package not installed, trying torch.hub...")
        except Exception as e:
            logger.warning(f"rife_model loading failed: {e}")

        # Method 2: Try torch.hub
        try:
            logger.info("Loading RIFE from torch.hub...")
            self._model = torch.hub.load(
                "hzwer/RIFE",
                "rife",
                pretrained=True,
                trust_repo=True,
            ).to(self.device)
            self._model.eval()
            logger.info("RIFE loaded successfully from torch.hub")
            return
        except Exception as e:
            logger.warning(f"torch.hub RIFE loading failed: {e}")

        # Fallback: simple linear blend interpolation
        self._model = None
        logger.warning(
            "RIFE not available - using simple linear blend for interpolation. "
            "For better quality, install RIFE: pip install rife"
        )

    def interpolate(
        self,
        frame0: Image.Image,
        frame1: Image.Image,
        num_frames: int = 1,
    ) -> List[Image.Image]:
        """Interpolate frames between two keyframes.

        Args:
            frame0: First keyframe
            frame1: Second keyframe
            num_frames: Number of intermediate frames to generate

        Returns:
            List of interpolated frames (not including input frames)
        """
        if not self._initialized:
            self.initialize()

        if self._model is None:
            # Fallback to simple blend
            return self._blend_interpolate(frame0, frame1, num_frames)

        return self._rife_interpolate(frame0, frame1, num_frames)

    def _rife_interpolate(
        self,
        frame0: Image.Image,
        frame1: Image.Image,
        num_frames: int,
    ) -> List[Image.Image]:
        """Interpolate using RIFE model."""
        # Convert to tensors
        img0 = self._pil_to_tensor(frame0)
        img1 = self._pil_to_tensor(frame1)

        results = []
        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)

            with torch.no_grad():
                if hasattr(self._model, 'inference'):
                    # hzwer's RIFE
                    mid = self._model.inference(img0, img1, t, self.scale)
                else:
                    # torch hub RIFE
                    mid = self._model(img0, img1, t)

            results.append(self._tensor_to_pil(mid))

        return results

    def _blend_interpolate(
        self,
        frame0: Image.Image,
        frame1: Image.Image,
        num_frames: int,
    ) -> List[Image.Image]:
        """Simple linear blend interpolation (fallback)."""
        arr0 = np.array(frame0).astype(np.float32)
        arr1 = np.array(frame1).astype(np.float32)

        results = []
        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)
            blended = (1 - t) * arr0 + t * arr1
            results.append(Image.fromarray(blended.astype(np.uint8)))

        return results

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def cleanup(self) -> None:
        """Release resources."""
        self._model = None
        self._initialized = False


class KeyframeInterpolationPipeline:
    """Manages keyframe generation and interpolation for efficient generation."""

    def __init__(
        self,
        generator_fn,  # Function that generates a single frame
        config: Optional[KeyframeConfig] = None,
        device: str = "cuda",
    ):
        """
        Args:
            generator_fn: Callable that generates a frame given params
            config: Keyframe configuration
            device: CUDA device
        """
        self.generator_fn = generator_fn
        self.config = config or KeyframeConfig()
        self.device = device

        self._interpolator = RIFEInterpolator(device, self.config.rife_scale)
        self._keyframe_buffer: List[Image.Image] = []
        self._frame_count = 0
        self._pending_interpolated: List[Image.Image] = []

    def initialize(self) -> None:
        """Initialize the interpolation pipeline."""
        self._interpolator.initialize()

    def generate_frame(
        self,
        generation_params: dict,
        audio_energy: float = 0.5,
        is_beat: bool = False,
    ) -> Image.Image:
        """Generate or interpolate a frame.

        Args:
            generation_params: Parameters for the generator function
            audio_energy: Current audio energy level (0-1)
            is_beat: Whether this is a beat moment

        Returns:
            Generated or interpolated frame
        """
        # Determine if this should be a keyframe
        is_keyframe = self._should_generate_keyframe(audio_energy, is_beat)

        if is_keyframe:
            # Generate actual keyframe
            frame = self.generator_fn(**generation_params)
            self._add_keyframe(frame)
            self._frame_count += 1
            return frame
        else:
            # Return interpolated frame if available
            if self._pending_interpolated:
                return self._pending_interpolated.pop(0)
            else:
                # Need to generate a new keyframe
                frame = self.generator_fn(**generation_params)
                self._add_keyframe(frame)
                self._frame_count += 1
                return frame

    def _should_generate_keyframe(
        self,
        audio_energy: float,
        is_beat: bool,
    ) -> bool:
        """Determine if current frame should be a keyframe."""
        # Always keyframe if buffer is empty or has only one frame
        if len(self._keyframe_buffer) < 2:
            return True

        # Use adaptive interval based on audio
        if self.config.audio_adaptive:
            # More keyframes during high energy / beats
            if is_beat:
                interval = max(2, self.config.keyframe_interval // 2)
            elif audio_energy > 0.7:
                interval = max(2, self.config.keyframe_interval - 1)
            elif audio_energy < 0.3:
                interval = self.config.keyframe_interval + 2
            else:
                interval = self.config.keyframe_interval
        else:
            interval = self.config.keyframe_interval

        return self._frame_count % interval == 0

    def _add_keyframe(self, frame: Image.Image) -> None:
        """Add a keyframe and generate interpolated frames."""
        self._keyframe_buffer.append(frame)

        # Keep only last 2 keyframes
        if len(self._keyframe_buffer) > 2:
            self._keyframe_buffer.pop(0)

        # Generate interpolated frames between last two keyframes
        if len(self._keyframe_buffer) >= 2:
            num_interp = self.config.keyframe_interval - 1
            self._pending_interpolated = self._interpolator.interpolate(
                self._keyframe_buffer[-2],
                self._keyframe_buffer[-1],
                num_interp,
            )

    def get_compute_reduction(self) -> float:
        """Get the theoretical compute reduction factor."""
        return self.config.keyframe_interval

    def reset(self) -> None:
        """Reset the pipeline state."""
        self._keyframe_buffer.clear()
        self._pending_interpolated.clear()
        self._frame_count = 0

    def cleanup(self) -> None:
        """Release resources."""
        self._interpolator.cleanup()
        self.reset()


class MotionAwareInterpolator:
    """Motion-aware interpolation that adapts to scene changes."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._interpolator = RIFEInterpolator(device)

    def initialize(self) -> None:
        """Initialize the interpolator."""
        self._interpolator.initialize()

    def estimate_motion(
        self,
        frame0: Image.Image,
        frame1: Image.Image,
    ) -> float:
        """Estimate motion magnitude between frames.

        Returns:
            Motion score (0 = identical, 1 = completely different)
        """
        arr0 = np.array(frame0).astype(np.float32) / 255.0
        arr1 = np.array(frame1).astype(np.float32) / 255.0

        # Simple difference-based motion estimation
        diff = np.abs(arr1 - arr0).mean()

        # Normalize to 0-1 range (typical diff is 0-0.5)
        return min(1.0, diff * 2)

    def interpolate_adaptive(
        self,
        frame0: Image.Image,
        frame1: Image.Image,
        target_frames: int,
        motion_threshold: float = 0.3,
    ) -> Tuple[List[Image.Image], bool]:
        """Adaptively interpolate based on motion.

        If motion is too high, returns empty list and flag indicating
        full generation should be used instead.

        Args:
            frame0: First keyframe
            frame1: Second keyframe
            target_frames: Target number of intermediate frames
            motion_threshold: Skip interpolation if motion exceeds this

        Returns:
            Tuple of (interpolated frames, whether interpolation was used)
        """
        motion = self.estimate_motion(frame0, frame1)

        if motion > motion_threshold:
            # Motion too high - recommend full generation
            logger.debug(f"Motion {motion:.2f} > threshold, skipping interpolation")
            return [], False

        # Motion acceptable - interpolate
        frames = self._interpolator.interpolate(frame0, frame1, target_frames)
        return frames, True

    def cleanup(self) -> None:
        """Release resources."""
        self._interpolator.cleanup()


# Factory functions
def create_keyframe_pipeline(
    generator_fn,
    keyframe_interval: int = 4,
    device: str = "cuda",
) -> KeyframeInterpolationPipeline:
    """Create a keyframe interpolation pipeline."""
    config = KeyframeConfig(keyframe_interval=keyframe_interval)
    pipeline = KeyframeInterpolationPipeline(generator_fn, config, device)
    pipeline.initialize()
    return pipeline
