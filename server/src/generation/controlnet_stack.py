"""ControlNet Stack for pose and shape preservation.

Combines OpenPose for pose and Line Art/HED for body shape,
allowing audio-reactive transformations while maintaining subject identity.
"""

import torch
import logging
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image
from dataclasses import dataclass
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ControlNetConfig:
    """Configuration for ControlNet stack."""

    # OpenPose for pose preservation
    pose_model: str = "lllyasviel/control_v11p_sd15_openpose"
    pose_weight: float = 0.8

    # Line Art for body shape
    lineart_model: str = "lllyasviel/control_v11p_sd15_lineart"
    lineart_weight: float = 0.3

    # HED edge detection (alternative to lineart)
    hed_model: str = "lllyasviel/control_v11p_sd15_softedge"
    hed_weight: float = 0.3

    # Depth for spatial structure
    depth_model: str = "lllyasviel/control_v11f1p_sd15_depth"
    depth_weight: float = 0.4

    # Which controls to use
    use_pose: bool = True
    use_lineart: bool = True
    use_depth: bool = False


class PoseExtractor:
    """Extracts OpenPose keypoints from images."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._detector = None
        self._initialized = False

    def initialize(self) -> None:
        """Load OpenPose detector."""
        if self._initialized:
            return

        try:
            from controlnet_aux import OpenposeDetector

            self._detector = OpenposeDetector.from_pretrained(
                "lllyasviel/ControlNet"
            )
            self._initialized = True
            logger.info("OpenPose detector initialized")

        except ImportError as e:
            logger.warning(f"OpenPose not available: {e}")
            self._detector = None

    def extract(self, image: Image.Image) -> Optional[Image.Image]:
        """Extract pose from image.

        Returns:
            Pose visualization image or None if extraction fails
        """
        if not self._initialized:
            self.initialize()

        if self._detector is None:
            return None

        try:
            pose_image = self._detector(
                image,
                hand_and_face=False,  # Faster without hands/face
                output_type="pil",
            )
            return pose_image
        except Exception as e:
            logger.warning(f"Pose extraction failed: {e}")
            return None

    def cleanup(self) -> None:
        """Release resources."""
        self._detector = None
        self._initialized = False


class LineArtExtractor:
    """Extracts line art from images for shape preservation."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._detector = None
        self._initialized = False

    def initialize(self) -> None:
        """Load line art detector."""
        if self._initialized:
            return

        try:
            from controlnet_aux import LineartDetector

            self._detector = LineartDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self._initialized = True
            logger.info("Line art detector initialized")

        except ImportError as e:
            logger.warning(f"Line art detector not available: {e}")
            self._detector = None

    def extract(self, image: Image.Image) -> Optional[Image.Image]:
        """Extract line art from image."""
        if not self._initialized:
            self.initialize()

        if self._detector is None:
            return None

        try:
            lineart = self._detector(image)
            return lineart
        except Exception as e:
            logger.warning(f"Line art extraction failed: {e}")
            return None

    def cleanup(self) -> None:
        """Release resources."""
        self._detector = None
        self._initialized = False


class DepthExtractor:
    """Extracts depth maps from images."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._detector = None
        self._initialized = False

    def initialize(self) -> None:
        """Load depth estimator."""
        if self._initialized:
            return

        try:
            from controlnet_aux import MidasDetector

            self._detector = MidasDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self._initialized = True
            logger.info("Depth estimator initialized")

        except ImportError as e:
            logger.warning(f"Depth estimator not available: {e}")
            self._detector = None

    def extract(self, image: Image.Image) -> Optional[Image.Image]:
        """Extract depth map from image."""
        if not self._initialized:
            self.initialize()

        if self._detector is None:
            return None

        try:
            depth = self._detector(image)
            return depth
        except Exception as e:
            logger.warning(f"Depth extraction failed: {e}")
            return None

    def cleanup(self) -> None:
        """Release resources."""
        self._detector = None
        self._initialized = False


class ControlNetStack:
    """Manages multiple ControlNets for comprehensive identity preservation."""

    def __init__(
        self,
        config: Optional[ControlNetConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config or ControlNetConfig(
            pose_weight=settings.controlnet_pose_weight,
            lineart_weight=settings.controlnet_lineart_weight,
        )
        self.device = device
        self.dtype = dtype

        # Extractors
        self._pose_extractor = PoseExtractor(device)
        self._lineart_extractor = LineArtExtractor(device)
        self._depth_extractor = DepthExtractor(device)

        # ControlNet models
        self._controlnets: Dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all ControlNet components."""
        if self._initialized:
            return

        logger.info("Initializing ControlNet stack...")

        # Initialize extractors
        if self.config.use_pose:
            self._pose_extractor.initialize()
        if self.config.use_lineart:
            self._lineart_extractor.initialize()
        if self.config.use_depth:
            self._depth_extractor.initialize()

        # Load ControlNet models
        self._load_controlnets()

        self._initialized = True
        logger.info("ControlNet stack initialized")

    def _load_controlnets(self) -> None:
        """Load ControlNet models."""
        try:
            from diffusers import ControlNetModel

            if self.config.use_pose:
                logger.info(f"Loading pose ControlNet: {self.config.pose_model}")
                self._controlnets['pose'] = ControlNetModel.from_pretrained(
                    self.config.pose_model,
                    torch_dtype=self.dtype,
                ).to(self.device)

            if self.config.use_lineart:
                logger.info(f"Loading lineart ControlNet: {self.config.lineart_model}")
                self._controlnets['lineart'] = ControlNetModel.from_pretrained(
                    self.config.lineart_model,
                    torch_dtype=self.dtype,
                ).to(self.device)

            if self.config.use_depth:
                logger.info(f"Loading depth ControlNet: {self.config.depth_model}")
                self._controlnets['depth'] = ControlNetModel.from_pretrained(
                    self.config.depth_model,
                    torch_dtype=self.dtype,
                ).to(self.device)

        except Exception as e:
            logger.error(f"Failed to load ControlNets: {e}")

    def extract_conditions(
        self,
        image: Image.Image,
    ) -> Dict[str, Optional[Image.Image]]:
        """Extract all conditioning images from source image.

        Returns:
            Dictionary with pose, lineart, and depth images
        """
        if not self._initialized:
            self.initialize()

        conditions = {}

        if self.config.use_pose:
            conditions['pose'] = self._pose_extractor.extract(image)

        if self.config.use_lineart:
            conditions['lineart'] = self._lineart_extractor.extract(image)

        if self.config.use_depth:
            conditions['depth'] = self._depth_extractor.extract(image)

        return conditions

    def get_controlnets_and_images(
        self,
        conditions: Dict[str, Optional[Image.Image]],
    ) -> Tuple[List[Any], List[Image.Image], List[float]]:
        """Get ControlNet models and conditioning images for pipeline.

        Returns:
            Tuple of (controlnet_list, image_list, scale_list)
        """
        controlnets = []
        images = []
        scales = []

        if self.config.use_pose and 'pose' in self._controlnets and conditions.get('pose'):
            controlnets.append(self._controlnets['pose'])
            images.append(conditions['pose'])
            scales.append(self.config.pose_weight)

        if self.config.use_lineart and 'lineart' in self._controlnets and conditions.get('lineart'):
            controlnets.append(self._controlnets['lineart'])
            images.append(conditions['lineart'])
            scales.append(self.config.lineart_weight)

        if self.config.use_depth and 'depth' in self._controlnets and conditions.get('depth'):
            controlnets.append(self._controlnets['depth'])
            images.append(conditions['depth'])
            scales.append(self.config.depth_weight)

        return controlnets, images, scales

    def create_pipeline(self, base_pipe):
        """Create a ControlNet pipeline from a base pipeline.

        Args:
            base_pipe: Base diffusers pipeline (txt2img or img2img)

        Returns:
            ControlNet-enabled pipeline
        """
        if not self._initialized:
            self.initialize()

        if not self._controlnets:
            logger.warning("No ControlNets loaded, returning base pipeline")
            return base_pipe

        try:
            from diffusers import StableDiffusionControlNetPipeline, MultiControlNetModel

            # Combine multiple ControlNets
            controlnet_list = list(self._controlnets.values())

            if len(controlnet_list) == 1:
                controlnet = controlnet_list[0]
            else:
                controlnet = MultiControlNetModel(controlnet_list)

            # Create ControlNet pipeline
            pipe = StableDiffusionControlNetPipeline.from_pipe(
                base_pipe,
                controlnet=controlnet,
            )

            return pipe

        except Exception as e:
            logger.error(f"Failed to create ControlNet pipeline: {e}")
            return base_pipe

    def set_weights(
        self,
        pose_weight: Optional[float] = None,
        lineart_weight: Optional[float] = None,
        depth_weight: Optional[float] = None,
    ) -> None:
        """Update ControlNet weights dynamically."""
        if pose_weight is not None:
            self.config.pose_weight = pose_weight
        if lineart_weight is not None:
            self.config.lineart_weight = lineart_weight
        if depth_weight is not None:
            self.config.depth_weight = depth_weight

    def cleanup(self) -> None:
        """Release resources."""
        self._pose_extractor.cleanup()
        self._lineart_extractor.cleanup()
        self._depth_extractor.cleanup()

        for controlnet in self._controlnets.values():
            del controlnet
        self._controlnets.clear()

        self._initialized = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AudioReactiveControlNet(ControlNetStack):
    """ControlNet stack with audio-reactive weight modulation.

    Adjusts ControlNet weights based on audio features to allow
    more freedom during energetic moments while maintaining
    stability during quiet sections.
    """

    def __init__(
        self,
        config: Optional[ControlNetConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(config, device, dtype)

        # Base weights (restored during quiet moments)
        self._base_pose_weight = self.config.pose_weight
        self._base_lineart_weight = self.config.lineart_weight
        self._base_depth_weight = self.config.depth_weight

    def update_audio(
        self,
        energy: float,
        is_beat: bool,
        onset_strength: float = 0.0,
    ) -> None:
        """Update weights based on audio state.

        - Beats: Loosen pose constraints for more dramatic movements
        - High energy: Reduce all constraints for more reactivity
        - Low energy: Strengthen constraints for stability
        """
        # Calculate modulation factor
        if is_beat:
            # Significant loosening on beats
            modulation = 0.5
        elif onset_strength > 0.5:
            # Moderate loosening on transients
            modulation = 0.7
        elif energy > 0.7:
            # Slight loosening for high energy
            modulation = 0.8
        elif energy < 0.3:
            # Strengthen during quiet moments
            modulation = 1.1
        else:
            # Normal range
            modulation = 1.0

        # Apply modulation
        self.config.pose_weight = min(1.0, self._base_pose_weight * modulation)
        self.config.lineart_weight = min(1.0, self._base_lineart_weight * modulation)
        self.config.depth_weight = min(1.0, self._base_depth_weight * modulation)

    def reset_weights(self) -> None:
        """Reset to base weights."""
        self.config.pose_weight = self._base_pose_weight
        self.config.lineart_weight = self._base_lineart_weight
        self.config.depth_weight = self._base_depth_weight


# Factory functions
def create_controlnet_stack(
    device: str = "cuda",
    audio_reactive: bool = False,
) -> ControlNetStack:
    """Create a ControlNet stack instance."""
    config = ControlNetConfig(
        pose_weight=settings.controlnet_pose_weight,
        lineart_weight=settings.controlnet_lineart_weight,
        use_pose=settings.use_controlnet,
        use_lineart=settings.use_controlnet,
    )

    if audio_reactive:
        return AudioReactiveControlNet(config, device)
    return ControlNetStack(config, device)
