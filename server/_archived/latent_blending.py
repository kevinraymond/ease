"""Latent blending for temporal coherence using lunarring approach.

This implements cross-feeding of latents between frames to maintain
subject identity and smooth transitions during iterative generation.
"""

import torch
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class BlendingConfig:
    """Configuration for latent blending."""

    crossfeed_power: float = 0.5      # Blend strength (0=none, 1=full previous)
    crossfeed_range: float = 0.7       # Fraction of steps to apply blending
    crossfeed_decay: float = 0.2       # Decay rate per step
    depth_power: float = 0.3           # Additional depth-based blending
    use_slerp: bool = True             # Use spherical interpolation
    num_history_frames: int = 3        # Number of frames to keep in history


class LatentBlender:
    """Manages latent cross-feeding for temporal coherence."""

    def __init__(self, config: Optional[BlendingConfig] = None):
        self.config = config or BlendingConfig(
            crossfeed_power=settings.crossfeed_power,
            crossfeed_range=settings.crossfeed_range,
            crossfeed_decay=settings.crossfeed_decay,
        )

        self._latent_history: list[torch.Tensor] = []
        self._depth_maps: list[torch.Tensor] = []

    def blend(
        self,
        current_latent: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Apply latent blending based on step position.

        Blending is strongest at early steps (high noise) and
        decays as denoising progresses.

        Args:
            current_latent: Current step's latent tensor
            step: Current denoising step (0 to total_steps-1)
            total_steps: Total number of denoising steps

        Returns:
            Blended latent tensor
        """
        if len(self._latent_history) == 0:
            return current_latent

        # Only blend during initial portion of steps
        blend_range = int(total_steps * self.config.crossfeed_range)
        if step >= blend_range:
            return current_latent

        # Calculate blend weight with decay
        progress = step / max(blend_range, 1)
        decay_factor = self.config.crossfeed_decay ** progress
        blend_weight = self.config.crossfeed_power * decay_factor

        # Get previous latent (most recent in history)
        prev_latent = self._latent_history[-1]

        # Ensure shapes match
        if prev_latent.shape != current_latent.shape:
            logger.warning("Latent shape mismatch, skipping blend")
            return current_latent

        # Apply blending
        if self.config.use_slerp:
            blended = self._slerp(prev_latent, current_latent, 1 - blend_weight)
        else:
            blended = (1 - blend_weight) * current_latent + blend_weight * prev_latent

        return blended

    def blend_multi_frame(
        self,
        current_latent: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Apply blending using multiple previous frames.

        Uses weighted average of history frames for smoother transitions.
        """
        if len(self._latent_history) == 0:
            return current_latent

        blend_range = int(total_steps * self.config.crossfeed_range)
        if step >= blend_range:
            return current_latent

        progress = step / max(blend_range, 1)
        base_weight = self.config.crossfeed_power * (self.config.crossfeed_decay ** progress)

        # Weight each history frame (recent frames get more weight)
        total_weight = 0.0
        blended = torch.zeros_like(current_latent)

        for i, hist_latent in enumerate(reversed(self._latent_history)):
            if hist_latent.shape != current_latent.shape:
                continue

            # Exponential decay for older frames
            frame_weight = base_weight * (0.5 ** i)
            if self.config.use_slerp and i == 0:
                # Use slerp for most recent frame only
                blended = self._slerp(hist_latent, current_latent, 1 - frame_weight)
                total_weight = 1.0
                break
            else:
                blended += frame_weight * hist_latent
                total_weight += frame_weight

        if total_weight > 0 and not self.config.use_slerp:
            remaining_weight = 1 - total_weight
            blended = remaining_weight * current_latent + blended

        return blended

    def blend_with_depth(
        self,
        current_latent: torch.Tensor,
        depth_map: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Apply depth-aware latent blending.

        Uses depth map to blend more heavily in background regions
        (maintains subject identity) while allowing more freedom
        in foreground.
        """
        if len(self._latent_history) == 0 or len(self._depth_maps) == 0:
            return current_latent

        base_blended = self.blend(current_latent, step, total_steps)

        # Get previous depth map
        prev_depth = self._depth_maps[-1]

        # Normalize depth map to latent resolution
        if depth_map.shape[-2:] != current_latent.shape[-2:]:
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(0).unsqueeze(0),
                size=current_latent.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze()

        # Create depth-based blend mask
        # Higher depth values (background) get more blending
        depth_weight = depth_map * self.config.depth_power

        # Apply depth-weighted blend
        final = base_blended * (1 + depth_weight.unsqueeze(0)) / (1 + self.config.depth_power)

        return final

    def _slerp(
        self,
        v0: torch.Tensor,
        v1: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """Spherical linear interpolation between two tensors.

        Better for interpolating in latent space than linear interpolation.
        """
        # Flatten for dot product
        v0_flat = v0.flatten()
        v1_flat = v1.flatten()

        # Compute angle between vectors
        dot = torch.dot(v0_flat, v1_flat) / (torch.norm(v0_flat) * torch.norm(v1_flat))
        dot = torch.clamp(dot, -1.0, 1.0)

        omega = torch.acos(dot)

        if omega.abs() < 1e-6:
            # Vectors are very close, use linear interpolation
            return (1 - t) * v0 + t * v1

        sin_omega = torch.sin(omega)
        s0 = torch.sin((1 - t) * omega) / sin_omega
        s1 = torch.sin(t * omega) / sin_omega

        return s0 * v0 + s1 * v1

    def update_history(
        self,
        latent: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a latent to the history buffer."""
        self._latent_history.append(latent.clone().detach())

        # Keep only recent history
        while len(self._latent_history) > self.config.num_history_frames:
            self._latent_history.pop(0)

        if depth_map is not None:
            self._depth_maps.append(depth_map.clone().detach())
            while len(self._depth_maps) > self.config.num_history_frames:
                self._depth_maps.pop(0)

    def clear_history(self) -> None:
        """Clear the latent history."""
        self._latent_history.clear()
        self._depth_maps.clear()

    def get_callback(self, total_steps: int):
        """Get a callback function for use with diffusers pipelines.

        Returns:
            Callback function compatible with callback_on_step_end
        """
        def callback(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            blended = self.blend(latents, step, total_steps)
            callback_kwargs["latents"] = blended
            return callback_kwargs

        return callback


class AdaptiveLatentBlender(LatentBlender):
    """Latent blender with audio-reactive adaptation.

    Adjusts blending parameters based on audio features for
    musical synchronization.
    """

    def __init__(self, config: Optional[BlendingConfig] = None):
        super().__init__(config)
        self._audio_energy = 0.0
        self._is_beat = False
        self._onset_strength = 0.0

    def update_audio(
        self,
        energy: float,
        is_beat: bool,
        onset_strength: float = 0.0,
    ) -> None:
        """Update audio state for adaptive blending."""
        self._audio_energy = energy
        self._is_beat = is_beat
        self._onset_strength = onset_strength

    def blend(
        self,
        current_latent: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Apply audio-adaptive latent blending.

        - On beats: reduce blending for more dramatic changes
        - High energy: reduce blending for more reactivity
        - Low energy: increase blending for stability
        """
        if len(self._latent_history) == 0:
            return current_latent

        # Calculate adaptive blend weight
        base_power = self.config.crossfeed_power

        # Reduce blending on beats (more dramatic changes)
        if self._is_beat:
            base_power *= 0.5

        # Reduce blending based on onset strength
        if self._onset_strength > 0.5:
            base_power *= (1 - self._onset_strength * 0.5)

        # Adjust based on energy (more reactive when loud)
        energy_factor = 1 - (self._audio_energy * 0.3)
        base_power *= energy_factor

        # Temporarily adjust config
        original_power = self.config.crossfeed_power
        self.config.crossfeed_power = max(0.1, min(0.8, base_power))

        # Apply blending with adjusted config
        result = super().blend(current_latent, step, total_steps)

        # Restore config
        self.config.crossfeed_power = original_power

        return result


# Factory function
def create_blender(adaptive: bool = False) -> LatentBlender:
    """Create a latent blender instance."""
    config = BlendingConfig(
        crossfeed_power=settings.crossfeed_power,
        crossfeed_range=settings.crossfeed_range,
        crossfeed_decay=settings.crossfeed_decay,
    )

    if adaptive:
        return AdaptiveLatentBlender(config)
    return LatentBlender(config)
