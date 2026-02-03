"""Combined Identity Pipeline for audio-reactive generation with identity preservation.

Orchestrates ControlNet + IP-Adapter + Latent Blending for maintaining
subject identity through audio-reactive transformations.
"""

import torch
import logging
from typing import Optional, List, Dict, Any, Callable
from PIL import Image
from dataclasses import dataclass
import numpy as np

from ..config import settings
from .controlnet_stack import ControlNetStack, AudioReactiveControlNet, ControlNetConfig
from .ip_adapter import IPAdapterFaceID, AudioReactiveIPAdapter, IPAdapterConfig
from .latent_blending import LatentBlender, AdaptiveLatentBlender, BlendingConfig

logger = logging.getLogger(__name__)


@dataclass
class IdentityConfig:
    """Configuration for identity preservation pipeline."""

    # ControlNet settings
    use_controlnet: bool = True
    pose_weight: float = 0.8
    lineart_weight: float = 0.3

    # IP-Adapter settings
    use_ip_adapter: bool = True
    ip_adapter_scale: float = 0.6

    # Latent blending settings
    use_latent_blending: bool = True
    crossfeed_power: float = 0.5

    # Reference management
    reference_update_interval: int = 0   # 0 = never auto-update
    face_similarity_threshold: float = 0.7

    # Audio reactivity
    audio_reactive: bool = True
    beat_pose_loosening: float = 0.5    # Reduce pose weight on beats


class IdentityPipeline:
    """Orchestrates identity preservation across all components."""

    def __init__(
        self,
        config: Optional[IdentityConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config or IdentityConfig(
            use_controlnet=settings.use_controlnet,
            pose_weight=settings.controlnet_pose_weight,
            lineart_weight=settings.controlnet_lineart_weight,
            use_ip_adapter=settings.use_ip_adapter,
            ip_adapter_scale=settings.ip_adapter_scale,
            use_latent_blending=settings.latent_blending,
            crossfeed_power=settings.crossfeed_power,
        )
        self.device = device
        self.dtype = dtype

        # Components
        self._controlnet: Optional[ControlNetStack] = None
        self._ip_adapter: Optional[IPAdapterFaceID] = None
        self._blender: Optional[LatentBlender] = None

        # Reference state
        self._reference_image: Optional[Image.Image] = None
        self._reference_conditions: Dict[str, Image.Image] = {}
        self._frame_count = 0

        # Audio state
        self._energy = 0.5
        self._is_beat = False
        self._onset_strength = 0.0
        self._spectral_centroid = 0.5

        self._initialized = False

    def initialize(self) -> None:
        """Initialize all identity preservation components."""
        if self._initialized:
            return

        logger.info("Initializing identity pipeline...")

        # Initialize ControlNet
        if self.config.use_controlnet:
            if self.config.audio_reactive:
                self._controlnet = AudioReactiveControlNet(
                    ControlNetConfig(
                        pose_weight=self.config.pose_weight,
                        lineart_weight=self.config.lineart_weight,
                    ),
                    self.device,
                    self.dtype,
                )
            else:
                self._controlnet = ControlNetStack(
                    ControlNetConfig(
                        pose_weight=self.config.pose_weight,
                        lineart_weight=self.config.lineart_weight,
                    ),
                    self.device,
                    self.dtype,
                )
            self._controlnet.initialize()

        # Initialize IP-Adapter
        if self.config.use_ip_adapter:
            if self.config.audio_reactive:
                self._ip_adapter = AudioReactiveIPAdapter(
                    IPAdapterConfig(scale=self.config.ip_adapter_scale),
                    self.device,
                    self.dtype,
                )
            else:
                self._ip_adapter = IPAdapterFaceID(
                    IPAdapterConfig(scale=self.config.ip_adapter_scale),
                    self.device,
                    self.dtype,
                )
            self._ip_adapter.initialize()

        # Initialize latent blender
        if self.config.use_latent_blending:
            if self.config.audio_reactive:
                self._blender = AdaptiveLatentBlender(
                    BlendingConfig(crossfeed_power=self.config.crossfeed_power)
                )
            else:
                self._blender = LatentBlender(
                    BlendingConfig(crossfeed_power=self.config.crossfeed_power)
                )

        self._initialized = True
        logger.info("Identity pipeline initialized")

    def set_reference(self, image: Image.Image) -> bool:
        """Set reference image for identity preservation.

        Args:
            image: Reference image containing the subject

        Returns:
            True if reference was successfully set
        """
        if not self._initialized:
            self.initialize()

        self._reference_image = image
        success = True

        # Extract ControlNet conditions
        if self._controlnet:
            self._reference_conditions = self._controlnet.extract_conditions(image)
            logger.info(f"Extracted conditions: {list(self._reference_conditions.keys())}")

        # Set IP-Adapter reference
        if self._ip_adapter:
            if not self._ip_adapter.set_reference_image(image):
                logger.warning("No face detected in reference image")
                success = False

        return success

    def update_audio(
        self,
        energy: float,
        is_beat: bool,
        onset_strength: float = 0.0,
        spectral_centroid: float = 0.5,
    ) -> None:
        """Update audio state for reactive components."""
        self._energy = energy
        self._is_beat = is_beat
        self._onset_strength = onset_strength
        self._spectral_centroid = spectral_centroid

        # Update components
        if isinstance(self._controlnet, AudioReactiveControlNet):
            self._controlnet.update_audio(energy, is_beat, onset_strength)

        if isinstance(self._ip_adapter, AudioReactiveIPAdapter):
            self._ip_adapter.update_audio(energy, is_beat, onset_strength)

        if isinstance(self._blender, AdaptiveLatentBlender):
            self._blender.update_audio(energy, is_beat, onset_strength)

    def get_pipeline_kwargs(
        self,
        current_image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Get kwargs for diffusers pipeline with all identity controls.

        Args:
            current_image: Current frame for condition extraction (uses reference if None)

        Returns:
            Dictionary with controlnet and ip_adapter kwargs
        """
        kwargs = {}

        # Determine source for conditions
        source_image = current_image or self._reference_image
        if source_image is None:
            return kwargs

        # ControlNet conditions
        if self._controlnet:
            if current_image:
                conditions = self._controlnet.extract_conditions(current_image)
            else:
                conditions = self._reference_conditions

            controlnets, images, scales = self._controlnet.get_controlnets_and_images(conditions)

            if controlnets:
                kwargs['controlnet'] = controlnets[0] if len(controlnets) == 1 else controlnets
                kwargs['image'] = images[0] if len(images) == 1 else images
                kwargs['controlnet_conditioning_scale'] = scales

        # IP-Adapter
        if self._ip_adapter:
            ip_kwargs = self._ip_adapter.get_ip_adapter_kwargs()
            kwargs.update(ip_kwargs)

        return kwargs

    def get_latent_callback(self, total_steps: int) -> Optional[Callable]:
        """Get latent blending callback for pipeline.

        Returns:
            Callback function for callback_on_step_end
        """
        if not self._blender:
            return None

        return self._blender.get_callback(total_steps)

    def update_latent_history(self, latent: torch.Tensor) -> None:
        """Update latent history after generation."""
        if self._blender:
            self._blender.update_history(latent)

    def check_identity(self, image: Image.Image) -> float:
        """Check how well identity is preserved in generated image.

        Returns:
            Similarity score (0-1)
        """
        if not self._ip_adapter:
            return 1.0

        return self._ip_adapter.compute_similarity(image)

    def should_reset(self, image: Image.Image) -> bool:
        """Check if identity has drifted too far and reset is needed.

        Returns:
            True if reset is recommended
        """
        similarity = self.check_identity(image)
        return similarity < self.config.face_similarity_threshold

    def reset(self) -> None:
        """Reset pipeline state (clear history, but keep reference)."""
        if self._blender:
            self._blender.clear_history()
        self._frame_count = 0

    def clear_reference(self) -> None:
        """Clear reference image and conditions."""
        self._reference_image = None
        self._reference_conditions.clear()
        self.reset()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of identity pipeline."""
        return {
            'has_reference': self._reference_image is not None,
            'frame_count': self._frame_count,
            'controlnet_enabled': self._controlnet is not None,
            'ip_adapter_enabled': self._ip_adapter is not None,
            'blending_enabled': self._blender is not None,
            'audio_state': {
                'energy': self._energy,
                'is_beat': self._is_beat,
                'onset_strength': self._onset_strength,
            }
        }

    def cleanup(self) -> None:
        """Release all resources."""
        if self._controlnet:
            self._controlnet.cleanup()
            self._controlnet = None

        if self._ip_adapter:
            self._ip_adapter.cleanup()
            self._ip_adapter = None

        if self._blender:
            self._blender.clear_history()
            self._blender = None

        self._reference_image = None
        self._reference_conditions.clear()
        self._initialized = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class IdentityAwarePipeline:
    """High-level wrapper that integrates identity pipeline with generation."""

    def __init__(
        self,
        generator_fn: Callable,
        config: Optional[IdentityConfig] = None,
        device: str = "cuda",
    ):
        """
        Args:
            generator_fn: Function that generates images (e.g., StreamDiffusionWrapper.generate_img2img)
            config: Identity configuration
            device: CUDA device
        """
        self.generator_fn = generator_fn
        self.device = device

        self._identity = IdentityPipeline(config, device)
        self._last_image: Optional[Image.Image] = None

    def initialize(self) -> None:
        """Initialize the pipeline."""
        self._identity.initialize()

    def set_reference(self, image: Image.Image) -> bool:
        """Set reference image."""
        return self._identity.set_reference(image)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.5,
        seed: Optional[int] = None,
        audio_metrics: Optional[dict] = None,
    ) -> Image.Image:
        """Generate image with identity preservation.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            strength: Denoising strength
            seed: Random seed
            audio_metrics: Audio features dict with energy, is_beat, etc.

        Returns:
            Generated image
        """
        # Update audio state
        if audio_metrics:
            self._identity.update_audio(
                energy=audio_metrics.get('rms', 0.5),
                is_beat=audio_metrics.get('is_beat', False),
                onset_strength=audio_metrics.get('onset_strength', 0.0),
                spectral_centroid=audio_metrics.get('spectral_centroid', 0.5),
            )

        # Get identity kwargs
        kwargs = self._identity.get_pipeline_kwargs(self._last_image)

        # Generate
        image = self.generator_fn(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=self._last_image,
            strength=strength,
            seed=seed,
            **kwargs
        )

        # Check identity drift
        if self._identity.should_reset(image):
            logger.warning("Identity drift detected, consider resetting")

        self._last_image = image
        self._identity._frame_count += 1

        return image

    def reset(self) -> None:
        """Reset to reference."""
        self._identity.reset()
        self._last_image = self._identity._reference_image

    def cleanup(self) -> None:
        """Release resources."""
        self._identity.cleanup()
        self._last_image = None


# Factory function
def create_identity_pipeline(
    device: str = "cuda",
    audio_reactive: bool = True,
) -> IdentityPipeline:
    """Create an identity preservation pipeline."""
    config = IdentityConfig(
        use_controlnet=settings.use_controlnet,
        pose_weight=settings.controlnet_pose_weight,
        lineart_weight=settings.controlnet_lineart_weight,
        use_ip_adapter=settings.use_ip_adapter,
        ip_adapter_scale=settings.ip_adapter_scale,
        use_latent_blending=settings.latent_blending,
        crossfeed_power=settings.crossfeed_power,
        audio_reactive=audio_reactive,
    )

    pipeline = IdentityPipeline(config, device)
    return pipeline
