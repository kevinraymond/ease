"""Fast pose-based image warping using thin-plate splines.

Instead of regenerating the entire image with diffusion each frame,
we can WARP an existing reference image to match a new pose skeleton.
This is essentially instant compared to diffusion.

For subtle movements (breathing, swaying, small gestures), warping looks great.
For large pose changes, we regenerate with full diffusion.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PoseWarper:
    """Warps images based on skeleton keypoint movements."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

        # Reference image and its keypoints
        self._reference_image: Optional[np.ndarray] = None
        self._reference_keypoints: Optional[np.ndarray] = None

        # Warping parameters
        self._max_warp_distance = 0.15  # Max movement as fraction of image size before regenerating
        self._warp_count = 0
        self._max_warps_before_refresh = 30  # Regenerate after this many warps to prevent artifact buildup

    def set_reference(self, image: Image.Image, keypoints: list[tuple[float, float]]) -> None:
        """Set the reference image and its corresponding keypoints.

        Args:
            image: PIL Image to use as reference
            keypoints: List of (x, y) tuples in normalized coordinates (0-1)
        """
        # Convert to numpy array
        self._reference_image = np.array(image)

        # Convert normalized keypoints to pixel coordinates
        self._reference_keypoints = np.array([
            [x * self.width, y * self.height] for x, y in keypoints
        ], dtype=np.float32)

        self._warp_count = 0
        logger.info(f"Set reference image with {len(keypoints)} keypoints")

    def needs_refresh(self) -> bool:
        """Check if we should regenerate the reference image."""
        return (
            self._reference_image is None or
            self._warp_count >= self._max_warps_before_refresh
        )

    def warp_to_pose(self, target_keypoints: list[tuple[float, float]]) -> Optional[Image.Image]:
        """Warp the reference image to match new keypoints.

        Args:
            target_keypoints: Target keypoints in normalized coordinates (0-1)

        Returns:
            Warped image, or None if warping not possible/advisable
        """
        if self._reference_image is None or self._reference_keypoints is None:
            return None

        # Convert target keypoints to pixel coordinates
        target_pts = np.array([
            [x * self.width, y * self.height] for x, y in target_keypoints
        ], dtype=np.float32)

        # Check if movement is too large for warping
        max_movement = np.max(np.abs(target_pts - self._reference_keypoints))
        max_movement_normalized = max_movement / max(self.width, self.height)

        if max_movement_normalized > self._max_warp_distance:
            logger.info(f"Movement too large for warping ({max_movement_normalized:.3f} > {self._max_warp_distance}), need regeneration")
            return None

        # Perform thin-plate spline warping
        try:
            warped = self._tps_warp(self._reference_image, self._reference_keypoints, target_pts)
            self._warp_count += 1

            # Update reference keypoints to the new position for next frame
            # This allows cumulative small movements
            self._reference_keypoints = target_pts.copy()

            return Image.fromarray(warped)

        except Exception as e:
            logger.warning(f"Warping failed: {e}")
            return None

    def _tps_warp(
        self,
        image: np.ndarray,
        source_pts: np.ndarray,
        target_pts: np.ndarray
    ) -> np.ndarray:
        """Apply thin-plate spline warping.

        TPS creates smooth, natural-looking warps that handle rotation well.
        """
        # Add corner points to prevent edge artifacts
        corners = np.array([
            [0, 0],
            [self.width - 1, 0],
            [0, self.height - 1],
            [self.width - 1, self.height - 1],
            [self.width // 2, 0],
            [self.width // 2, self.height - 1],
            [0, self.height // 2],
            [self.width - 1, self.height // 2],
        ], dtype=np.float32)

        src_with_corners = np.vstack([source_pts, corners])
        tgt_with_corners = np.vstack([target_pts, corners])

        # Create TPS transformer
        # Note: cv2.createThinPlateSplineShapeTransformer requires matched points
        tps = cv2.createThinPlateSplineShapeTransformer()

        # Reshape for OpenCV (needs shape [1, n, 2])
        src_pts_cv = src_with_corners.reshape(1, -1, 2)
        tgt_pts_cv = tgt_with_corners.reshape(1, -1, 2)

        # Create matches (each source point matches to corresponding target point)
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_with_corners))]

        tps.estimateTransformation(tgt_pts_cv, src_pts_cv, matches)

        # Apply the transformation
        # We need to warp in reverse: for each target pixel, find source pixel
        warped = tps.warpImage(image)

        return warped

    def simple_warp(
        self,
        image: np.ndarray,
        source_pts: np.ndarray,
        target_pts: np.ndarray
    ) -> np.ndarray:
        """Simpler piecewise affine warping as fallback.

        Divides the image into triangles and warps each independently.
        Faster than TPS but can have visible seams.
        """
        # Delaunay triangulation on source points
        rect = (0, 0, self.width, self.height)
        subdiv = cv2.Subdiv2D(rect)

        for pt in source_pts:
            try:
                subdiv.insert((float(pt[0]), float(pt[1])))
            except cv2.error:
                pass  # Point outside bounds

        triangles = subdiv.getTriangleList()

        output = np.zeros_like(image)

        for t in triangles:
            # Get triangle vertices
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # Find corresponding target points
            # (simplified - in practice you'd track which keypoint each vertex corresponds to)
            src_tri = np.float32([pt1, pt2, pt3])

            # For now, just copy - full implementation would find corresponding target triangle
            # and warp the triangle region

        return output  # Placeholder - TPS is preferred

    def reset(self) -> None:
        """Clear the reference image."""
        self._reference_image = None
        self._reference_keypoints = None
        self._warp_count = 0


class HybridPoseAnimator:
    """Combines fast warping with periodic diffusion regeneration.

    Strategy:
    1. Generate initial reference with full diffusion
    2. For subtle movements, warp the reference (instant)
    3. When movement is large or quality degrades, regenerate with diffusion
    4. Audio reactivity can trigger regeneration on beats for style changes
    """

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self._warper = PoseWarper(width, height)

        # Callbacks for diffusion generation
        self._generate_callback = None

        # State
        self._last_keypoints: Optional[list[tuple[float, float]]] = None
        self._frames_since_generation = 0
        self._force_regenerate = False

    def set_generate_callback(self, callback) -> None:
        """Set callback for full diffusion generation.

        Callback signature: (keypoints: list[tuple[float, float]]) -> Image.Image
        """
        self._generate_callback = callback

    def process_frame(
        self,
        keypoints: list[tuple[float, float]],
        force_regenerate: bool = False,
        is_beat: bool = False,
    ) -> tuple[Image.Image, bool]:
        """Process a frame, using warping or generation as appropriate.

        Args:
            keypoints: Target pose keypoints (normalized 0-1)
            force_regenerate: Force full diffusion regeneration
            is_beat: Audio beat detected - could trigger style change

        Returns:
            Tuple of (output image, was_regenerated)
        """
        should_regenerate = (
            force_regenerate or
            self._force_regenerate or
            self._warper.needs_refresh() or
            self._generate_callback is None
        )

        if not should_regenerate:
            # Try warping first
            warped = self._warper.warp_to_pose(keypoints)
            if warped is not None:
                self._frames_since_generation += 1
                self._last_keypoints = keypoints
                return warped, False
            else:
                # Warping failed (movement too large), need to regenerate
                should_regenerate = True

        # Full regeneration needed
        if self._generate_callback is None:
            raise RuntimeError("No generation callback set")

        logger.info("Regenerating reference image with diffusion")
        image = self._generate_callback(keypoints)

        # Set as new reference
        self._warper.set_reference(image, keypoints)
        self._frames_since_generation = 0
        self._force_regenerate = False
        self._last_keypoints = keypoints

        return image, True

    def request_regenerate(self) -> None:
        """Request regeneration on next frame (e.g., for style change)."""
        self._force_regenerate = True

    def reset(self) -> None:
        """Reset state."""
        self._warper.reset()
        self._last_keypoints = None
        self._frames_since_generation = 0
        self._force_regenerate = False
