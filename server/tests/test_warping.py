#!/usr/bin/env python3
"""Test script to evaluate pose-based image warping quality.

Run this to see if warping produces acceptable results before integrating.
"""

import sys
import os
import importlib.util

# Load modules directly to avoid package import issues
def load_module_direct(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

script_dir = os.path.dirname(os.path.abspath(__file__))
procedural_pose = load_module_direct("procedural_pose", os.path.join(script_dir, "src/generation/procedural_pose.py"))
pose_warping = load_module_direct("pose_warping", os.path.join(script_dir, "src/generation/pose_warping.py"))

ProceduralPoseGenerator = procedural_pose.ProceduralPoseGenerator
PoseAnimationMode = procedural_pose.PoseAnimationMode
PoseWarper = pose_warping.PoseWarper
BodyPart = procedural_pose.BodyPart

import numpy as np
from PIL import Image
import cv2


def detect_pose_keypoints(image: Image.Image) -> list[tuple[float, float]]:
    """Detect pose keypoints in an image using OpenPose detector.

    Returns normalized (0-1) keypoint coordinates.
    """
    try:
        from controlnet_aux import OpenposeDetector

        print("  Loading OpenPose detector...")
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        # Get pose detection result
        # The detector returns a PIL image, but we need the keypoints
        # We'll use the internal method to get raw keypoints
        print("  Detecting pose...")

        # Run detection
        import numpy as np
        from controlnet_aux.open_pose import Body

        # Access the body model directly
        body_estimation = detector.body_estimation

        # Convert image to numpy
        img_np = np.array(image)

        # Get candidate keypoints
        candidate, subset = body_estimation(img_np)

        if len(subset) == 0:
            print("  No pose detected in image!")
            return None

        # Get the first person's keypoints
        person = subset[0]

        width, height = image.size
        keypoints = []

        # OpenPose returns 18 keypoints, map to our format
        for i in range(18):
            idx = int(person[i])
            if idx == -1:
                # Keypoint not detected, use center as fallback
                keypoints.append((0.5, 0.5))
            else:
                x, y = candidate[idx][:2]
                keypoints.append((x / width, y / height))

        print(f"  Detected {sum(1 for kp in keypoints if kp != (0.5, 0.5))} keypoints")
        return keypoints

    except Exception as e:
        print(f"  Pose detection failed: {e}")
        return None


def generate_test_sequence():
    """Generate a warping test sequence."""

    print("=== Pose Warping Test ===\n")

    width, height = 512, 512

    # Check if we have a reference image to test with
    reference_path = "test_reference.png"
    if not os.path.exists(reference_path):
        print(f"No reference image found at '{reference_path}'")
        print("Please provide a test image (e.g., a generated dancer image)")
        print("\nTo create one, you can:")
        print("1. Run the main app and save a generated frame")
        print("2. Or provide any 512x512 image of a person")

        # Create a simple placeholder for testing the mechanics
        print("\nCreating a simple test pattern for now...")
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw a simple stick figure with colored body parts
        # Head
        cv2.circle(img, (256, 80), 30, (255, 200, 150), -1)
        # Body
        cv2.line(img, (256, 110), (256, 260), (100, 150, 200), 12)
        # Left arm
        cv2.line(img, (256, 140), (180, 200), (200, 100, 100), 10)
        cv2.line(img, (180, 200), (150, 260), (200, 100, 100), 8)
        # Right arm
        cv2.line(img, (256, 140), (332, 200), (100, 200, 100), 10)
        cv2.line(img, (332, 200), (362, 260), (100, 200, 100), 8)
        # Left leg
        cv2.line(img, (256, 260), (220, 350), (100, 100, 200), 10)
        cv2.line(img, (220, 350), (210, 450), (100, 100, 200), 8)
        # Right leg
        cv2.line(img, (256, 260), (292, 350), (200, 200, 100), 10)
        cv2.line(img, (292, 350), (302, 450), (200, 200, 100), 8)

        reference = Image.fromarray(img)
    else:
        reference = Image.open(reference_path).convert('RGB')
        reference = reference.resize((width, height))
        print(f"Loaded reference image from {reference_path}")

    # Initialize generators
    pose_gen = ProceduralPoseGenerator(width, height, animation_speed=1.0)
    pose_gen.set_mode(PoseAnimationMode.DANCING)
    pose_gen.set_intensity(0.5)

    warper = PoseWarper(width, height)
    # Increase max warp distance to allow more movement
    warper._max_warp_distance = 0.25

    # Detect the actual pose in the reference image
    print("\nDetecting pose in reference image...")
    detected_keypoints = detect_pose_keypoints(reference)

    if detected_keypoints is not None:
        initial_keypoints = detected_keypoints
        print("Using detected pose from reference image")
    else:
        # Fallback to base pose
        initial_keypoints = [(kp.x, kp.y) for kp in pose_gen._base_pose]
        print("Warning: Using base pose (may not match reference image)")

    warper.set_reference(reference, initial_keypoints)

    # Visualize detected keypoints on reference image
    print("\nSaving detected pose overlay...")
    ref_with_keypoints = np.array(reference).copy()
    for i, (x, y) in enumerate(initial_keypoints):
        px, py = int(x * width), int(y * height)
        # Draw keypoint
        cv2.circle(ref_with_keypoints, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(ref_with_keypoints, str(i), (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    output_dir = "warping_test_output"
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(ref_with_keypoints).save(f"{output_dir}/reference_with_keypoints.png")
    print(f"  Saved to {output_dir}/reference_with_keypoints.png")

    # Generate frames
    num_frames = 60
    frames = []
    pose_frames = []

    print(f"\nGenerating {num_frames} frames...")

    for i in range(num_frames):
        t = i / 30.0  # 30 fps simulation

        # Get animated pose image
        pose_image = pose_gen.generate_pose(frame_time=t)
        pose_frames.append(pose_image)

        # Get the animated keypoints by calling animation function directly
        anim_t = t * pose_gen.animation_speed
        if pose_gen._mode == PoseAnimationMode.DANCING:
            keypoints_obj = pose_gen._animate_dancing(anim_t)
        elif pose_gen._mode == PoseAnimationMode.IDLE:
            keypoints_obj = pose_gen._animate_idle(anim_t)
        elif pose_gen._mode == PoseAnimationMode.WALKING:
            keypoints_obj = pose_gen._animate_walking(anim_t)
        elif pose_gen._mode == PoseAnimationMode.WAVING:
            keypoints_obj = pose_gen._animate_waving(anim_t)
        else:
            keypoints_obj = pose_gen._base_pose

        current_keypoints = [(kp.x, kp.y) for kp in keypoints_obj]

        # Try to warp
        warped = warper.warp_to_pose(current_keypoints)

        if warped is None:
            print(f"  Frame {i}: Warping failed (movement too large), using reference")
            warped = reference
            # Reset reference with current keypoints for next attempt
            warper.set_reference(reference, current_keypoints)

        frames.append(warped)

        if i % 10 == 0:
            print(f"  Frame {i}/{num_frames}")

    # Save results
    output_dir = "warping_test_output"
    os.makedirs(output_dir, exist_ok=True)

    # Save as GIF
    print(f"\nSaving animation to {output_dir}/warped_animation.gif...")
    frames[0].save(
        f"{output_dir}/warped_animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=33,  # ~30fps
        loop=0
    )

    # Save pose animation too for comparison
    print(f"Saving pose reference to {output_dir}/pose_animation.gif...")
    pose_frames[0].save(
        f"{output_dir}/pose_animation.gif",
        save_all=True,
        append_images=pose_frames[1:],
        duration=33,
        loop=0
    )

    # Save side-by-side comparison
    print(f"Saving comparison frames to {output_dir}/...")
    for i in [0, 15, 30, 45]:
        if i < len(frames):
            # Create side-by-side: pose | warped
            pose_np = np.array(pose_frames[i])
            warped_np = np.array(frames[i])
            comparison = np.hstack([pose_np, warped_np])
            Image.fromarray(comparison).save(f"{output_dir}/comparison_frame_{i:03d}.png")

    print(f"\n✓ Done! Check {output_dir}/ for results")
    print("\nFiles created:")
    print(f"  - warped_animation.gif: The warped image sequence")
    print(f"  - pose_animation.gif: The procedural pose skeleton")
    print(f"  - comparison_frame_*.png: Side-by-side comparisons")
    print("\nIf warping looks good, we can integrate it for real-time use!")


if __name__ == "__main__":
    generate_test_sequence()
