"""Test Keyframe + RIFE interpolation mode with procedural poses."""

import time
from pathlib import Path
from PIL import Image

from src.server.protocol import AudioMetrics, GenerationConfig, GenerationMode
from src.generation.pipeline import GenerationPipeline


def create_test_metrics(bass: float = 0.5, is_beat: bool = False) -> AudioMetrics:
    """Create test audio metrics."""
    return AudioMetrics(
        rms=0.5,
        peak=0.6,
        bass=bass,
        mid=0.3,
        treble=0.2,
        raw_bass=bass,
        raw_mid=0.3,
        raw_treble=0.2,
        bpm=120,
        is_beat=is_beat,
    )


def main():
    print("=" * 60)
    print("KEYFRAME + RIFE MODE TEST")
    print("=" * 60)

    # Configure for keyframe + rife mode with procedural poses
    config = GenerationConfig(
        generation_mode=GenerationMode.KEYFRAME_RIFE,
        use_controlnet=True,
        use_procedural_pose=True,
        pose_animation_mode="gentle",
        pose_animation_intensity=0.5,
        pose_framing="upper_body",
        controlnet_pose_weight=0.8,
        keyframe_interval=4,  # Generate keyframe every 4 frames
        model_id="Lykon/dreamshaper-8",
        base_prompt="photograph of a woman dancing, photorealistic, natural lighting, professional photography",
        width=512,
        height=512,
        target_fps=20,
    )

    print(f"\nConfig:")
    print(f"  Mode: {config.generation_mode}")
    print(f"  Keyframe interval: {config.keyframe_interval}")
    print(f"  ControlNet: {config.use_controlnet}")
    print(f"  Procedural pose: {config.use_procedural_pose}")
    print(f"  Pose mode: {config.pose_animation_mode}")
    print(f"  Pose framing: {config.pose_framing}")

    # Create pipeline
    print("\nInitializing pipeline...")
    pipeline = GenerationPipeline(config)
    pipeline.initialize()
    print("Pipeline ready!")

    # Generate frames
    output_dir = Path("/tmp/keyframe_rife_test")
    output_dir.mkdir(exist_ok=True)

    num_frames = 12  # Should generate 3 keyframes + 9 interpolated
    frame_times = []

    print(f"\nGenerating {num_frames} frames...")
    print("-" * 60)

    for i in range(num_frames):
        # Simulate beat on every 4th frame
        is_beat = (i % 4 == 0)
        bass = 0.8 if is_beat else 0.4

        metrics = create_test_metrics(bass=bass, is_beat=is_beat)

        start = time.perf_counter()
        frame = pipeline.generate(metrics)
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)

        frame_type = "KEYFRAME" if (i % config.keyframe_interval == 0) else "interpolated"
        print(f"  Frame {i + 1:2d}: {elapsed * 1000:6.1f}ms ({frame_type}, beat={is_beat})")

        # Save frame
        frame.image.save(output_dir / f"frame_{i:03d}.png")

        # Save pose preview
        pose_bytes = pipeline.get_pose_preview()
        if pose_bytes:
            with open(output_dir / f"pose_{i:03d}.jpg", "wb") as f:
                f.write(pose_bytes)

    print("-" * 60)

    # Stats
    keyframe_times = [frame_times[i] for i in range(0, num_frames, config.keyframe_interval)]
    interp_times = [frame_times[i] for i in range(num_frames) if i % config.keyframe_interval != 0]

    avg_keyframe = sum(keyframe_times) / len(keyframe_times) * 1000 if keyframe_times else 0
    avg_interp = sum(interp_times) / len(interp_times) * 1000 if interp_times else 0
    avg_overall = sum(frame_times) / len(frame_times) * 1000
    effective_fps = 1000 / avg_overall if avg_overall > 0 else 0

    print(f"\nResults:")
    print(f"  Avg keyframe time:     {avg_keyframe:6.1f}ms")
    print(f"  Avg interpolated time: {avg_interp:6.1f}ms")
    print(f"  Avg overall time:      {avg_overall:6.1f}ms")
    print(f"  Effective FPS:         {effective_fps:6.1f}")
    print(f"  Compute reduction:     {avg_keyframe / avg_overall:.1f}x" if avg_overall > 0 else "")

    print(f"\nOutputs saved to: {output_dir}")
    print("  - frame_*.png: Generated/interpolated frames")
    print("  - pose_*.jpg: Pose skeletons")

    # Cleanup
    pipeline.cleanup()
    print("\nTest complete!")


if __name__ == "__main__":
    main()
