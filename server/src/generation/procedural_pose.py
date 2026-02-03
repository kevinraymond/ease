"""Procedural pose generator for ControlNet conditioning without camera input."""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)


class PoseAnimationMode(Enum):
    """Animation modes for procedural pose generation.

    Curated for visualization:
    - Human motion: gentle (low FPS), dancing (music-reactive)
    - Visualization: flowing (serpentine), pulsing (rhythmic), ethereal (floating)
    """
    # Human motion styles
    GENTLE = "gentle"       # Slow, graceful movements (good for low FPS)
    DANCING = "dancing"     # More dynamic dancing motion

    # Visualization-focused styles
    FLOWING = "flowing"     # Smooth serpentine motions - hypnotic wave patterns
    PULSING = "pulsing"     # Rhythmic expansion/contraction - syncs with beats
    ETHEREAL = "ethereal"   # Slow, weightless floating - dreamy atmosphere

    CUSTOM = "custom"       # User-defined keyframes


class PoseFraming(Enum):
    """Framing options for pose - controls what part of body is visible."""
    FULL_BODY = "full_body"      # Head to feet (may get cropped in square images)
    UPPER_BODY = "upper_body"    # Head to hips - good for 512x512
    PORTRAIT = "portrait"        # Head and shoulders only


@dataclass
class Keypoint:
    """A pose keypoint with normalized coordinates (0-1)."""
    x: float  # Normalized x (0-1)
    y: float  # Normalized y (0-1)
    score: float = 1.0
    id: int = -1


# OpenPose body keypoint indices (18 keypoints)
class BodyPart:
    NOSE = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_HIP = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_ANKLE = 13
    RIGHT_EYE = 14
    LEFT_EYE = 15
    RIGHT_EAR = 16
    LEFT_EAR = 17


# Limb connections (pairs of keypoint indices to connect)
# Note: OpenPose uses 1-indexed in their code, we use 0-indexed
LIMB_CONNECTIONS = [
    (BodyPart.NECK, BodyPart.RIGHT_SHOULDER),
    (BodyPart.NECK, BodyPart.LEFT_SHOULDER),
    (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
    (BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
    (BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
    (BodyPart.NECK, BodyPart.RIGHT_HIP),
    (BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
    (BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE),
    (BodyPart.NECK, BodyPart.LEFT_HIP),
    (BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
    (BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    (BodyPart.NECK, BodyPart.NOSE),
    (BodyPart.NOSE, BodyPart.RIGHT_EYE),
    (BodyPart.RIGHT_EYE, BodyPart.RIGHT_EAR),
    (BodyPart.NOSE, BodyPart.LEFT_EYE),
    (BodyPart.LEFT_EYE, BodyPart.LEFT_EAR),
]

# OpenPose rainbow color scheme (RGB)
POSE_COLORS = [
    [255, 0, 0],       # Red
    [255, 85, 0],      # Orange-red
    [255, 170, 0],     # Orange
    [255, 255, 0],     # Yellow
    [170, 255, 0],     # Yellow-green
    [85, 255, 0],      # Light green
    [0, 255, 0],       # Green
    [0, 255, 85],      # Cyan-green
    [0, 255, 170],     # Cyan
    [0, 255, 255],     # Bright cyan
    [0, 170, 255],     # Light blue
    [0, 85, 255],      # Blue
    [0, 0, 255],       # Dark blue
    [85, 0, 255],      # Purple
    [170, 0, 255],     # Magenta
    [255, 0, 255],     # Pink
    [255, 0, 170],     # Light pink
    [255, 0, 85],      # Red-pink
]


class ProceduralPoseGenerator:
    """Generates animated pose skeletons for ControlNet conditioning."""

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        animation_speed: float = 1.0,
        framing: PoseFraming = PoseFraming.UPPER_BODY,  # Default to upper body for 512x512
    ):
        self.width = width
        self.height = height
        self.animation_speed = animation_speed
        self._start_time = time.time()
        self._mode = PoseAnimationMode.DANCING
        self._framing = framing

        # Base pose (standing, arms slightly out) - normalized coordinates
        self._base_pose = self._create_base_pose()

        # Animation parameters
        self._dance_intensity: float = 0.5  # 0-1, how much movement
        self._audio_energy: float = 0.0     # 0-1, current audio energy for reactive poses

        # Frame-synced animation (for low FPS scenarios)
        self._frame_count: int = 0
        self._use_frame_sync: bool = False  # If True, animate based on frame count not time
        self._frames_per_cycle: int = 60    # How many frames for one animation cycle

    def _create_base_pose(self) -> list[Keypoint]:
        """Create a neutral standing pose as the base for animations."""
        # All coordinates are normalized (0-1)
        # X: 0 = left, 1 = right
        # Y: 0 = top, 1 = bottom
        return [
            # Head
            Keypoint(0.50, 0.15),   # 0: Nose
            Keypoint(0.50, 0.22),   # 1: Neck

            # Right arm (viewer's left)
            Keypoint(0.42, 0.25),   # 2: Right Shoulder
            Keypoint(0.35, 0.38),   # 3: Right Elbow
            Keypoint(0.30, 0.50),   # 4: Right Wrist

            # Left arm (viewer's right)
            Keypoint(0.58, 0.25),   # 5: Left Shoulder
            Keypoint(0.65, 0.38),   # 6: Left Elbow
            Keypoint(0.70, 0.50),   # 7: Left Wrist

            # Right leg
            Keypoint(0.45, 0.50),   # 8: Right Hip
            Keypoint(0.44, 0.70),   # 9: Right Knee
            Keypoint(0.43, 0.90),   # 10: Right Ankle

            # Left leg
            Keypoint(0.55, 0.50),   # 11: Left Hip
            Keypoint(0.56, 0.70),   # 12: Left Knee
            Keypoint(0.57, 0.90),   # 13: Left Ankle

            # Face
            Keypoint(0.47, 0.12),   # 14: Right Eye
            Keypoint(0.53, 0.12),   # 15: Left Eye
            Keypoint(0.40, 0.14),   # 16: Right Ear
            Keypoint(0.60, 0.14),   # 17: Left Ear
        ]

    def set_mode(self, mode: PoseAnimationMode) -> None:
        """Set the animation mode."""
        self._mode = mode
        logger.info(f"Pose animation mode set to: {mode.value}")

    def set_intensity(self, intensity: float) -> None:
        """Set dance intensity (0-1)."""
        self._dance_intensity = max(0.0, min(1.0, intensity))

    def set_audio_energy(self, energy: float) -> None:
        """Set current audio energy for reactive poses (0-1)."""
        self._audio_energy = max(0.0, min(1.0, energy))

    def set_animation_speed(self, speed: float) -> None:
        """Set animation speed multiplier."""
        self.animation_speed = speed

    def set_framing(self, framing: PoseFraming) -> None:
        """Set pose framing (full body, upper body, or portrait)."""
        self._framing = framing
        logger.info(f"Pose framing set to: {framing.value}")

    def _apply_framing(self, keypoints: list[Keypoint]) -> list[Keypoint]:
        """Transform keypoints based on framing setting.

        For upper_body/portrait, we scale and shift the pose to focus on
        the relevant body parts and fill the frame better.
        """
        if self._framing == PoseFraming.FULL_BODY:
            return keypoints  # No transform needed

        # Create copies to avoid modifying originals
        result = [Keypoint(kp.x, kp.y, kp.score, kp.id) for kp in keypoints]

        if self._framing == PoseFraming.UPPER_BODY:
            # Scale to show head-to-hips, filling more of the frame
            # Original: head at ~0.12, hips at ~0.50
            # Target: head at ~0.08, hips at ~0.92 (with some margin)
            scale_y = 2.0  # Vertical scale factor
            offset_y = -0.05  # Shift up slightly

            for kp in result:
                # Scale and shift Y coordinate
                kp.y = (kp.y * scale_y) + offset_y
                # Keep X centered but allow slight horizontal expansion
                kp.x = 0.5 + (kp.x - 0.5) * 1.1

        elif self._framing == PoseFraming.PORTRAIT:
            # Scale to show head and shoulders only
            # Original: head at ~0.12, shoulders at ~0.25
            # Target: head at ~0.15, shoulders at ~0.85
            scale_y = 4.0  # Larger scale for tighter framing
            offset_y = -0.35  # Shift up more

            for kp in result:
                kp.y = (kp.y * scale_y) + offset_y
                kp.x = 0.5 + (kp.x - 0.5) * 1.3  # Expand horizontally more

        return result

    def set_frame_sync(self, enabled: bool, frames_per_cycle: int = 60) -> None:
        """Enable frame-synced animation for low FPS scenarios.

        When enabled, animation progresses based on frame count rather than
        wall clock time. This ensures smooth perceived motion even at 1-2 FPS
        by making small, consistent pose changes between frames.

        Args:
            enabled: Whether to use frame-synced animation
            frames_per_cycle: Number of frames for one complete animation cycle
                             Higher = slower, smoother movement
                             At 1 FPS, 60 frames = 60 second cycle
                             At 2 FPS, 60 frames = 30 second cycle
        """
        self._use_frame_sync = enabled
        self._frames_per_cycle = max(10, frames_per_cycle)
        logger.info(f"Frame sync: enabled={enabled}, frames_per_cycle={frames_per_cycle}")

    def generate_pose(self, frame_time: Optional[float] = None) -> Image.Image:
        """Generate an animated pose skeleton image.

        Args:
            frame_time: Optional time in seconds. If None, uses wall clock time
                       or frame count depending on sync mode.

        Returns:
            PIL Image with the pose skeleton drawn on black background.
        """
        if self._use_frame_sync:
            # Frame-synced: use frame count for smooth low-FPS animation
            # One full cycle = frames_per_cycle frames
            # t goes from 0 to 2*pi over one cycle, scaled by animation_speed
            t = (self._frame_count / self._frames_per_cycle) * 2 * math.pi * self.animation_speed
            self._frame_count += 1
            logger.debug(f"Frame-sync pose: frame={self._frame_count}, t={t:.3f}, speed={self.animation_speed}")
        elif frame_time is None:
            frame_time = time.time() - self._start_time
            t = frame_time * self.animation_speed
        else:
            t = frame_time * self.animation_speed

        # Get animated keypoints based on mode
        if self._mode == PoseAnimationMode.GENTLE:
            keypoints = self._animate_gentle(t)
        elif self._mode == PoseAnimationMode.DANCING:
            keypoints = self._animate_dancing(t)
        elif self._mode == PoseAnimationMode.FLOWING:
            keypoints = self._animate_flowing(t)
        elif self._mode == PoseAnimationMode.PULSING:
            keypoints = self._animate_pulsing(t)
        elif self._mode == PoseAnimationMode.ETHEREAL:
            keypoints = self._animate_ethereal(t)
        else:
            keypoints = self._base_pose

        # Apply framing transform (scales/shifts for upper_body or portrait)
        keypoints = self._apply_framing(keypoints)

        # Draw the skeleton
        return self._draw_skeleton(keypoints)

    def _animate_gentle(self, t: float) -> list[Keypoint]:
        """Slow, graceful movement - ideal for low FPS generation.

        Uses very slow sine waves with small amplitudes so that
        consecutive frames have minimal pose differences, creating
        smooth perceived motion even at 1-2 FPS.
        """
        keypoints = [Keypoint(kp.x, kp.y) for kp in self._base_pose]
        intensity = self._dance_intensity * 0.5  # Halve intensity for gentler movement

        # Add audio reactivity (subtle)
        energy_boost = 1.0 + self._audio_energy * 0.3
        intensity *= energy_boost

        # Very slow body sway - like slow dancing
        # Using t directly (already scaled by frame sync if enabled)
        sway = math.sin(t) * 0.03 * intensity
        for i in range(len(keypoints)):
            keypoints[i].x += sway

        # Gentle up/down breathing motion
        breath = math.sin(t * 0.7) * 0.015 * intensity
        for i in range(len(keypoints)):
            keypoints[i].y += breath

        # Subtle shoulder roll (one side slightly up, other down)
        shoulder_roll = math.sin(t * 0.5) * 0.01 * intensity
        keypoints[BodyPart.RIGHT_SHOULDER].y -= shoulder_roll
        keypoints[BodyPart.LEFT_SHOULDER].y += shoulder_roll

        # Gentle arm float - arms drift slowly up and down
        arm_phase = t * 0.8
        right_arm_drift = math.sin(arm_phase) * 0.025 * intensity
        left_arm_drift = math.sin(arm_phase + math.pi * 0.5) * 0.025 * intensity

        keypoints[BodyPart.RIGHT_ELBOW].y += right_arm_drift
        keypoints[BodyPart.RIGHT_WRIST].y += right_arm_drift * 1.2
        keypoints[BodyPart.LEFT_ELBOW].y += left_arm_drift
        keypoints[BodyPart.LEFT_WRIST].y += left_arm_drift * 1.2

        # Very subtle hip sway
        hip_sway = math.sin(t * 0.6) * 0.015 * intensity
        keypoints[BodyPart.RIGHT_HIP].x += hip_sway
        keypoints[BodyPart.LEFT_HIP].x += hip_sway

        # Subtle weight shift (knees)
        weight = math.sin(t * 0.6) * 0.01 * intensity
        keypoints[BodyPart.RIGHT_KNEE].y += weight
        keypoints[BodyPart.LEFT_KNEE].y -= weight

        # Gentle head tilt
        head_tilt = math.sin(t * 0.4) * 0.008 * intensity
        for i in [BodyPart.NOSE, BodyPart.RIGHT_EYE, BodyPart.LEFT_EYE,
                  BodyPart.RIGHT_EAR, BodyPart.LEFT_EAR]:
            keypoints[i].x += head_tilt

        return keypoints

    def _animate_dancing(self, t: float) -> list[Keypoint]:
        """Dynamic dancing animation."""
        keypoints = [Keypoint(kp.x, kp.y) for kp in self._base_pose]
        intensity = self._dance_intensity

        # Add audio reactivity
        energy_boost = 1.0 + self._audio_energy * 0.5
        intensity *= energy_boost

        # Body bounce (up/down with beat)
        bounce = math.sin(t * 4) * 0.03 * intensity
        for i in range(len(keypoints)):
            keypoints[i].y += bounce

        # Hip sway (side to side)
        hip_sway = math.sin(t * 2) * 0.05 * intensity
        for i in [BodyPart.RIGHT_HIP, BodyPart.LEFT_HIP]:
            keypoints[i].x += hip_sway

        # Shoulder groove (opposite to hips)
        shoulder_sway = math.sin(t * 2 + math.pi) * 0.03 * intensity
        for i in [BodyPart.RIGHT_SHOULDER, BodyPart.LEFT_SHOULDER, BodyPart.NECK]:
            keypoints[i].x += shoulder_sway

        # Arm movements - right arm
        right_arm_phase = t * 2
        keypoints[BodyPart.RIGHT_ELBOW].x += math.sin(right_arm_phase) * 0.08 * intensity
        keypoints[BodyPart.RIGHT_ELBOW].y += math.cos(right_arm_phase) * 0.05 * intensity
        keypoints[BodyPart.RIGHT_WRIST].x += math.sin(right_arm_phase + 0.5) * 0.12 * intensity
        keypoints[BodyPart.RIGHT_WRIST].y += math.cos(right_arm_phase + 0.5) * 0.08 * intensity

        # Arm movements - left arm (offset phase)
        left_arm_phase = t * 2 + math.pi
        keypoints[BodyPart.LEFT_ELBOW].x += math.sin(left_arm_phase) * 0.08 * intensity
        keypoints[BodyPart.LEFT_ELBOW].y += math.cos(left_arm_phase) * 0.05 * intensity
        keypoints[BodyPart.LEFT_WRIST].x += math.sin(left_arm_phase + 0.5) * 0.12 * intensity
        keypoints[BodyPart.LEFT_WRIST].y += math.cos(left_arm_phase + 0.5) * 0.08 * intensity

        # Knee bend (weight shift)
        knee_bend = math.sin(t * 2) * 0.02 * intensity
        keypoints[BodyPart.RIGHT_KNEE].y += knee_bend
        keypoints[BodyPart.LEFT_KNEE].y -= knee_bend

        # Head bob
        head_bob = math.sin(t * 4) * 0.015 * intensity
        for i in [BodyPart.NOSE, BodyPart.RIGHT_EYE, BodyPart.LEFT_EYE,
                  BodyPart.RIGHT_EAR, BodyPart.LEFT_EAR]:
            keypoints[i].y += head_bob

        return keypoints

    def _animate_flowing(self, t: float) -> list[Keypoint]:
        """Smooth serpentine motions - hypnotic wave patterns.

        Creates wave-like motion that flows through the body from
        head to feet, like a ribbon dancer or underwater movement.
        """
        keypoints = [Keypoint(kp.x, kp.y) for kp in self._base_pose]
        intensity = self._dance_intensity

        # Add audio reactivity
        energy_boost = 1.0 + self._audio_energy * 0.4
        intensity *= energy_boost

        # Serpentine wave flowing down the body
        # Each body part is offset in phase to create a flowing wave
        wave_speed = 1.5
        wave_amp = 0.05 * intensity

        # Head leads the wave
        head_phase = t * wave_speed
        for i in [BodyPart.NOSE, BodyPart.RIGHT_EYE, BodyPart.LEFT_EYE,
                  BodyPart.RIGHT_EAR, BodyPart.LEFT_EAR]:
            keypoints[i].x += math.sin(head_phase) * wave_amp * 0.6

        # Neck and shoulders follow
        neck_phase = head_phase - 0.5
        keypoints[BodyPart.NECK].x += math.sin(neck_phase) * wave_amp * 0.8
        keypoints[BodyPart.RIGHT_SHOULDER].x += math.sin(neck_phase - 0.2) * wave_amp
        keypoints[BodyPart.LEFT_SHOULDER].x += math.sin(neck_phase - 0.2) * wave_amp

        # Arms flow with enhanced amplitude
        arm_phase = neck_phase - 1.0
        keypoints[BodyPart.RIGHT_ELBOW].x += math.sin(arm_phase) * wave_amp * 1.5
        keypoints[BodyPart.RIGHT_WRIST].x += math.sin(arm_phase - 0.5) * wave_amp * 2.0
        keypoints[BodyPart.LEFT_ELBOW].x += math.sin(arm_phase + math.pi) * wave_amp * 1.5
        keypoints[BodyPart.LEFT_WRIST].x += math.sin(arm_phase + math.pi - 0.5) * wave_amp * 2.0

        # Hips continue the wave
        hip_phase = neck_phase - 1.5
        keypoints[BodyPart.RIGHT_HIP].x += math.sin(hip_phase) * wave_amp
        keypoints[BodyPart.LEFT_HIP].x += math.sin(hip_phase) * wave_amp

        # Legs complete the wave
        leg_phase = hip_phase - 0.8
        keypoints[BodyPart.RIGHT_KNEE].x += math.sin(leg_phase) * wave_amp * 0.8
        keypoints[BodyPart.LEFT_KNEE].x += math.sin(leg_phase) * wave_amp * 0.8
        keypoints[BodyPart.RIGHT_ANKLE].x += math.sin(leg_phase - 0.4) * wave_amp * 0.6
        keypoints[BodyPart.LEFT_ANKLE].x += math.sin(leg_phase - 0.4) * wave_amp * 0.6

        # Gentle vertical undulation
        for i, kp in enumerate(keypoints):
            vert_phase = t * wave_speed * 0.7 + i * 0.2
            kp.y += math.sin(vert_phase) * wave_amp * 0.3

        return keypoints

    def _animate_pulsing(self, t: float) -> list[Keypoint]:
        """Rhythmic expansion/contraction - syncs well with beats.

        Creates a breathing/pulsing effect where the pose expands
        outward from center and contracts back, like a heartbeat.
        """
        keypoints = [Keypoint(kp.x, kp.y) for kp in self._base_pose]
        intensity = self._dance_intensity

        # Strong audio reactivity for beat sync
        energy_boost = 1.0 + self._audio_energy * 0.8
        intensity *= energy_boost

        # Pulsing expansion/contraction from center
        pulse_speed = 2.5  # Faster for rhythmic feel
        pulse = (math.sin(t * pulse_speed) + 1) / 2  # 0 to 1
        pulse_strength = pulse * 0.08 * intensity

        # Center point (roughly chest area)
        center_x = 0.5
        center_y = 0.35

        # Expand all points away from center
        for kp in keypoints:
            dx = kp.x - center_x
            dy = kp.y - center_y
            kp.x += dx * pulse_strength
            kp.y += dy * pulse_strength * 0.5  # Less vertical expansion

        # Arms have enhanced pulsing
        arm_pulse = pulse_strength * 1.5
        keypoints[BodyPart.RIGHT_WRIST].x -= arm_pulse
        keypoints[BodyPart.LEFT_WRIST].x += arm_pulse
        keypoints[BodyPart.RIGHT_ELBOW].x -= arm_pulse * 0.7
        keypoints[BodyPart.LEFT_ELBOW].x += arm_pulse * 0.7

        # Subtle vertical bounce on beats
        bounce = abs(math.sin(t * pulse_speed)) * 0.02 * intensity
        for kp in keypoints:
            kp.y += bounce

        # Arms raise slightly on expansion
        arm_lift = pulse * 0.03 * intensity
        keypoints[BodyPart.RIGHT_WRIST].y -= arm_lift
        keypoints[BodyPart.LEFT_WRIST].y -= arm_lift
        keypoints[BodyPart.RIGHT_ELBOW].y -= arm_lift * 0.5
        keypoints[BodyPart.LEFT_ELBOW].y -= arm_lift * 0.5

        return keypoints

    def _animate_ethereal(self, t: float) -> list[Keypoint]:
        """Slow, weightless floating - dreamy atmosphere.

        Creates the impression of floating in zero gravity or
        underwater, with slow drifting movements and subtle rotations.
        """
        keypoints = [Keypoint(kp.x, kp.y) for kp in self._base_pose]
        intensity = self._dance_intensity * 0.7  # Naturally gentler

        # Subtle audio reactivity
        energy_boost = 1.0 + self._audio_energy * 0.2
        intensity *= energy_boost

        # Very slow primary drift
        drift_x = math.sin(t * 0.3) * 0.04 * intensity
        drift_y = math.sin(t * 0.25) * 0.025 * intensity

        for kp in keypoints:
            kp.x += drift_x
            kp.y += drift_y

        # Arms float upward and drift independently
        # Right arm - slow circular float
        r_arm_phase = t * 0.4
        keypoints[BodyPart.RIGHT_ELBOW].x += math.sin(r_arm_phase) * 0.04 * intensity
        keypoints[BodyPart.RIGHT_ELBOW].y += math.cos(r_arm_phase) * 0.03 * intensity - 0.02 * intensity
        keypoints[BodyPart.RIGHT_WRIST].x += math.sin(r_arm_phase + 0.5) * 0.06 * intensity
        keypoints[BodyPart.RIGHT_WRIST].y += math.cos(r_arm_phase + 0.5) * 0.04 * intensity - 0.04 * intensity

        # Left arm - offset phase for asymmetry
        l_arm_phase = t * 0.35 + math.pi * 0.7
        keypoints[BodyPart.LEFT_ELBOW].x += math.sin(l_arm_phase) * 0.04 * intensity
        keypoints[BodyPart.LEFT_ELBOW].y += math.cos(l_arm_phase) * 0.03 * intensity - 0.02 * intensity
        keypoints[BodyPart.LEFT_WRIST].x += math.sin(l_arm_phase + 0.5) * 0.06 * intensity
        keypoints[BodyPart.LEFT_WRIST].y += math.cos(l_arm_phase + 0.5) * 0.04 * intensity - 0.04 * intensity

        # Head gentle tilt and nod
        head_tilt = math.sin(t * 0.2) * 0.015 * intensity
        head_nod = math.sin(t * 0.15) * 0.01 * intensity
        for i in [BodyPart.NOSE, BodyPart.RIGHT_EYE, BodyPart.LEFT_EYE,
                  BodyPart.RIGHT_EAR, BodyPart.LEFT_EAR]:
            keypoints[i].x += head_tilt
            keypoints[i].y += head_nod

        # Legs drift gently apart and together
        leg_drift = math.sin(t * 0.2) * 0.02 * intensity
        keypoints[BodyPart.RIGHT_KNEE].x -= leg_drift
        keypoints[BodyPart.LEFT_KNEE].x += leg_drift
        keypoints[BodyPart.RIGHT_ANKLE].x -= leg_drift * 1.2
        keypoints[BodyPart.LEFT_ANKLE].x += leg_drift * 1.2

        # Subtle rotation of whole body
        rotation = math.sin(t * 0.15) * 0.02 * intensity
        for kp in keypoints:
            # Rotate around center
            cx, cy = 0.5, 0.4
            dx, dy = kp.x - cx, kp.y - cy
            kp.x = cx + dx * math.cos(rotation) - dy * math.sin(rotation)
            kp.y = cy + dx * math.sin(rotation) + dy * math.cos(rotation)

        return keypoints

    def _draw_skeleton(self, keypoints: list[Keypoint]) -> Image.Image:
        """Draw the skeleton on a black canvas."""
        # Create black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        stickwidth = 4  # Thickness of limbs

        # Draw limbs first (so joints appear on top)
        for i, (k1_idx, k2_idx) in enumerate(LIMB_CONNECTIONS):
            kp1 = keypoints[k1_idx]
            kp2 = keypoints[k2_idx]

            # Convert normalized to pixel coordinates
            x1, y1 = int(kp1.x * self.width), int(kp1.y * self.height)
            x2, y2 = int(kp2.x * self.width), int(kp2.y * self.height)

            # Get color (darken by 60% for limbs, like OpenPose)
            # Convert RGB to BGR for OpenCV
            color_idx = i % len(POSE_COLORS)
            rgb_color = POSE_COLORS[color_idx]
            color = [int(c * 0.6) for c in reversed(rgb_color)]  # RGB -> BGR

            # Calculate limb parameters for ellipse drawing
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 1:
                continue

            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # Draw tapered limb as ellipse
            polygon = cv2.ellipse2Poly(
                (mid_x, mid_y),
                (int(length / 2), stickwidth),
                int(angle),
                0, 360, 1
            )
            cv2.fillConvexPoly(canvas, np.array(polygon, dtype=np.int32), color)  # type: ignore[call-overload]

        # Draw keypoints on top
        for i, kp in enumerate(keypoints):
            x = int(kp.x * self.width)
            y = int(kp.y * self.height)
            # Convert RGB to BGR for OpenCV
            rgb_color = POSE_COLORS[i % len(POSE_COLORS)]
            color = list(reversed(rgb_color))  # RGB -> BGR
            cv2.circle(canvas, (x, y), 4, color, thickness=-1)

        # Convert BGR (OpenCV) to RGB (PIL)
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(canvas_rgb)

    def reset_time(self) -> None:
        """Reset the animation start time."""
        self._start_time = time.time()
