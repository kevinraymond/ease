"""Isolation test: Verify ControlNet works with procedural poses."""
import sys
import os

# Add src to path for direct module import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "generation"))

import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from procedural_pose import ProceduralPoseGenerator, PoseAnimationMode

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

print("Generating procedural pose...")
pose_gen = ProceduralPoseGenerator(512, 512)
pose_gen.set_mode(PoseAnimationMode.GENTLE)
pose_image = pose_gen.generate_pose()
pose_image.save("/tmp/test_pose.png")
print(f"Saved pose skeleton to /tmp/test_pose.png")

print("Generating image with ControlNet txt2img...")
result = pipe(
    prompt="photograph of a woman dancing in a studio, photorealistic, high quality",
    negative_prompt="blurry, low quality, distorted",
    image=pose_image,
    controlnet_conditioning_scale=0.8,
    num_inference_steps=20,
    guidance_scale=7.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

result.save("/tmp/test_controlnet_result.png")
print(f"Saved result to /tmp/test_controlnet_result.png")
print("\n=== MANUAL VERIFICATION ===")
print("Compare /tmp/test_pose.png with /tmp/test_controlnet_result.png")
print("The generated person should match the skeleton pose!")
