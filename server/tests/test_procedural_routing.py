"""Test procedural pose routing to txt2img pipeline with animation sequence."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "generation"))

import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from procedural_pose import ProceduralPoseGenerator, PoseAnimationMode

print("=== PROCEDURAL POSE ANIMATION TEST ===")
print("Testing that animated poses are followed across multiple frames\n")

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16
).to("cuda")

print("Loading SD pipeline with ControlNet...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.enable_attention_slicing()

# Load LCM-LoRA for fast 4-6 step generation
print("Loading LCM-LoRA for fast generation...")
from diffusers import LCMScheduler
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
print("LCM-LoRA loaded - using 4 steps instead of 15")

# Initialize procedural pose generator with frame sync for animation
pose_gen = ProceduralPoseGenerator(512, 512)
pose_gen.set_mode(PoseAnimationMode.DANCING)  # Dynamic movement
pose_gen.set_frame_sync(True, frames_per_cycle=30)  # 30 frames per animation cycle

# Simple latent blending for temporal coherence
last_latents = None
blend_weight = 0.35

print(f"\nGenerating 5 frames with {blend_weight:.0%} latent blending...")
print("Each frame should show a different pose from the animation\n")

for frame_num in range(5):
    # Generate animated pose
    pose_image = pose_gen.generate_pose()
    pose_image.save(f"/tmp/pose_frame_{frame_num}.png")

    print(f"Frame {frame_num}: Generating...")

    # Generate with ControlNet txt2img + LCM (4 steps, guidance 1.5)
    result = pipe(
        prompt="a robot dancing, full body, grey background, photorealistic",
        negative_prompt="multiple figures, blurry, low quality",
        image=pose_image,
        controlnet_conditioning_scale=0.9,
        num_inference_steps=4,  # LCM: 4 steps for max speed
        guidance_scale=1.5,      # LCM: 1.0-2.0 guidance
        generator=torch.Generator(device="cuda").manual_seed(42 + frame_num),
        output_type="latent",
    )

    latents = result.images

    # Apply latent blending for temporal coherence
    if last_latents is not None:
        latents = (1 - blend_weight) * latents + blend_weight * last_latents
        print(f"  Applied {blend_weight:.0%} blend with previous frame")

    last_latents = latents.clone()

    # Decode to image
    images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    image_np = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    result_image = Image.fromarray(image_np)

    result_image.save(f"/tmp/result_frame_{frame_num}.png")
    print(f"  Saved to /tmp/result_frame_{frame_num}.png")

print("\n=== VERIFICATION ===")
print("Compare pose skeletons with generated images:")
print("  /tmp/pose_frame_*.png  -> Animated skeleton poses")
print("  /tmp/result_frame_*.png -> Generated images")
print("\nEach result image should match its corresponding pose skeleton!")
print("The robot's pose should change across frames, following the animation.")
