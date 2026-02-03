"""Audio-reactive image generation pipeline optimized for real-time music visualization.

This pipeline is designed for audio-reactive use cases where:
- Seed changes should cause immediate visual changes
- Strength should control how much the image deviates from input
- Speed is critical (~60-100ms target per frame)

Key optimizations:
- LCM-LoRA for 1-4 step inference
- TAESD for fast VAE decode
- Cached prompt embeddings
- Dynamic timestep based on strength
- No denoising batch (single frame, no state)
"""

import gc
import logging
import os
from typing import Optional, Tuple

import torch
from PIL import Image

from ..config import settings

logger = logging.getLogger(__name__)


class AudioReactivePipeline:
    """Fast image generation optimized for audio-reactive visuals.

    Unlike StreamDiffusion which optimizes for temporal consistency,
    this pipeline optimizes for responsive variation based on audio.
    """

    def __init__(
        self,
        model_id: str = "Lykon/dreamshaper-8",
        width: int = 512,
        height: int = 512,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_inference_steps: int = 4,  # LCM works well with 2-4 steps
        use_taesd: bool = True,
    ):
        self.model_id = model_id
        self.width = width
        self.height = height
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.use_taesd = use_taesd

        # Pipeline components (initialized lazily)
        self._pipe = None
        self._taesd = None
        self._initialized = False

        # Caching
        self._cached_prompt: Optional[str] = None
        self._cached_prompt_embeds: Optional[torch.Tensor] = None
        self._cached_negative_embeds: Optional[torch.Tensor] = None

        # Precomputed timestep mapping for strength -> timestep
        # Lower timestep = more denoising = more change
        self._timesteps: Optional[torch.Tensor] = None

        # Custom LoRA tracking
        self._loaded_loras: dict[str, float] = {}  # adapter_name -> weight
        self._lora_dir: str = settings.lora_dir

    def initialize(self) -> None:
        """Load models and prepare for generation."""
        if self._initialized:
            return

        import time
        t_start = time.perf_counter()

        logger.info(f"Initializing AudioReactivePipeline: {self.model_id}")

        # Load base pipeline
        from diffusers import (
            AutoPipelineForImage2Image,
            LCMScheduler,
            AutoencoderTiny,
        )

        logger.info("  Loading base pipeline...")
        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # Replace scheduler with LCM
        logger.info("  Setting up LCM scheduler...")
        self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config)

        # Load and fuse LCM-LoRA for fast inference
        logger.info("  Loading LCM-LoRA...")
        self._pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self._pipe.fuse_lora()

        # Replace VAE with TAESD for fast decode
        if self.use_taesd:
            logger.info("  Loading TAESD for fast decode...")
            self._taesd = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=self.dtype,
            ).to(self.device)
            self._taesd.eval()

        # Enable memory optimizations
        self._pipe.enable_attention_slicing()

        # Precompute timesteps for the scheduler
        self._pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self._timesteps = self._pipe.scheduler.timesteps

        # Mark initialized BEFORE warmup to prevent recursion
        self._initialized = True

        # Warm up with a dummy generation
        logger.info("  Warming up...")
        dummy_image = Image.new("RGB", (self.width, self.height), color=(128, 128, 128))
        self.generate(
            image=dummy_image,
            prompt="warmup",
            seed=42,
            strength=0.5,
        )

        t_end = time.perf_counter()
        logger.info(f"AudioReactivePipeline initialized in {t_end - t_start:.2f}s")

    def _get_prompt_embeds(
        self,
        prompt: str,
        negative_prompt: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prompt embeddings, using cache if available."""
        cache_key = f"{prompt}||{negative_prompt}"

        if self._cached_prompt == cache_key and self._cached_prompt_embeds is not None:
            return self._cached_prompt_embeds, self._cached_negative_embeds

        # Encode prompts
        prompt_embeds, negative_embeds = self._pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        # Cache for reuse
        self._cached_prompt = cache_key
        self._cached_prompt_embeds = prompt_embeds
        self._cached_negative_embeds = negative_embeds

        return prompt_embeds, negative_embeds

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to latent space."""
        # Resize if needed
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)

        # Convert to tensor
        image_tensor = self._pipe.image_processor.preprocess(image)
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)

        # Encode to latents
        latents = self._pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * self._pipe.vae.config.scaling_factor

        return latents

    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents to image using TAESD if available."""
        if self.use_taesd and self._taesd is not None:
            # TAESD decode
            latents = latents / self._pipe.vae.config.scaling_factor
            image_tensor = self._taesd.decode(latents).sample
        else:
            # Standard VAE decode
            latents = latents / self._pipe.vae.config.scaling_factor
            image_tensor = self._pipe.vae.decode(latents, return_dict=False)[0]

        # Convert to PIL
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_tensor = image_tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = Image.fromarray((image_tensor[0] * 255).astype("uint8"))

        return image

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        seed: int,
        strength: float = 0.5,
        negative_prompt: str = "",
        guidance_scale: float = 1.5,  # LCM uses low guidance
        is_onset: bool = False,
        onset_confidence: float = 0.0,
        is_beat_seed_jump: bool = False,
    ) -> Image.Image:
        """Generate an audio-reactive image.

        Args:
            image: Input image (previous frame or reference)
            prompt: Text prompt for generation
            seed: Random seed - changes here cause visual variation
            strength: How much to change from input (0.0-1.0)
                     Maps to timestep: high strength = more change
            negative_prompt: What to avoid
            guidance_scale: CFG scale (1.0-2.0 for LCM)
            is_onset: Whether this frame is on an audio onset/transient
            onset_confidence: Confidence of onset detection (0-1)
            is_beat_seed_jump: Beat triggered a seed jump - use txt2img injection

        Returns:
            Generated PIL Image
        """
        import time
        t_start = time.perf_counter()

        if not self._initialized:
            self.initialize()

        # Use the beat_seed_jump flag from audio mapper (more reliable than local detection)
        is_seed_jump = is_beat_seed_jump
        if is_seed_jump:
            logger.info("Beat seed jump signaled: will inject txt2img")

        # Apply onset effects
        if is_onset and onset_confidence > 0.3:
            # Onset strength boost - increase deviation from input on transients
            onset_boost = onset_confidence * 0.15  # Up to +0.15 strength on strong onsets
            strength = min(0.95, strength + onset_boost)
            logger.debug(f"Onset detected (conf={onset_confidence:.2f}): strength boosted to {strength:.3f}")

        # Clamp strength to valid range
        # Ensure at least 1 denoising step: min_strength = 1.0 / num_inference_steps + epsilon
        # With 4 steps: min ~0.26, with 3 steps: min ~0.34
        min_strength = 1.0 / self.num_inference_steps + 0.01
        strength = max(min_strength, min(0.95, strength))

        # Set up generator with seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Resize image if needed
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)

        if is_seed_jump:
            # On beat, inject noise image for fresh generation (pseudo txt2img)
            import numpy as np
            np_rng = np.random.default_rng(seed)
            noise_array = np_rng.integers(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            image = Image.fromarray(noise_array)
            strength = 0.99  # Maximum strength to mostly ignore the noise pattern
            logger.info("Beat seed jump: injecting noise image for fresh generation")

        # Use the pipeline's built-in img2img - much simpler and reliable
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        output_image = result.images[0]

        t_end = time.perf_counter()
        logger.debug(f"AudioReactive: total={1000*(t_end-t_start):.1f}ms, strength={strength:.2f}")

        return output_image

    def load_custom_loras(self, loras: list[dict]) -> None:
        """Load custom LoRAs (including LyCORIS/LoKr formats).

        The LCM-LoRA remains fused for performance. Custom LoRAs are loaded
        as unfused adapters on top using diffusers' PEFT integration.

        Args:
            loras: List of LoRA configs, each with 'path' and 'weight' keys.
                   Paths are relative to the configured lora_dir.
        """
        if self._pipe is None:
            logger.warning("Cannot load LoRAs: pipeline not initialized")
            return

        # Build set of desired LoRAs
        desired_loras: dict[str, float] = {}
        for lora_config in loras:
            path = lora_config.get('path', '')
            weight = lora_config.get('weight', 1.0)
            if not path:
                continue

            # Use filename (without extension) as adapter name
            adapter_name = os.path.splitext(os.path.basename(path))[0]
            desired_loras[adapter_name] = weight

        # Find LoRAs to remove (loaded but not desired)
        to_remove = [name for name in self._loaded_loras if name not in desired_loras]
        if to_remove:
            try:
                self._pipe.delete_adapters(to_remove)
                for name in to_remove:
                    del self._loaded_loras[name]
                logger.info(f"Removed LoRAs: {to_remove}")
            except Exception as e:
                logger.warning(f"Failed to remove LoRAs {to_remove}: {e}")

        # Load new LoRAs (desired but not loaded)
        for lora_config in loras:
            path = lora_config.get('path', '')
            weight = lora_config.get('weight', 1.0)
            if not path:
                continue

            adapter_name = os.path.splitext(os.path.basename(path))[0]

            # Skip if already loaded with same weight
            if adapter_name in self._loaded_loras:
                if self._loaded_loras[adapter_name] == weight:
                    continue
                # Weight changed - will update in set_adapters below
                self._loaded_loras[adapter_name] = weight
                continue

            # Resolve path relative to lora_dir
            if not os.path.isabs(path):
                full_path = os.path.join(self._lora_dir, path)
            else:
                full_path = path

            if not os.path.exists(full_path):
                logger.warning(f"LoRA file not found: {full_path}")
                continue

            try:
                # load_lora_weights auto-detects LoRA format (standard, LyCORIS, LoKr, etc.)
                self._pipe.load_lora_weights(full_path, adapter_name=adapter_name)
                self._loaded_loras[adapter_name] = weight
                logger.info(f"Loaded LoRA: {adapter_name} (weight={weight}) from {full_path}")
            except Exception as e:
                logger.error(f"Failed to load LoRA {path}: {e}")

        # Set active adapters with their weights
        if self._loaded_loras:
            adapter_names = list(self._loaded_loras.keys())
            adapter_weights = list(self._loaded_loras.values())
            try:
                self._pipe.set_adapters(adapter_names, adapter_weights)
                logger.debug(f"Active LoRAs: {list(zip(adapter_names, adapter_weights))}")
            except Exception as e:
                logger.warning(f"Failed to set adapters: {e}")

    def cleanup(self) -> None:
        """Release GPU resources."""
        logger.info("Cleaning up AudioReactivePipeline...")

        if self._pipe is not None:
            del self._pipe
            self._pipe = None

        if self._taesd is not None:
            del self._taesd
            self._taesd = None

        self._cached_prompt = None
        self._cached_prompt_embeds = None
        self._cached_negative_embeds = None
        self._loaded_loras.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("AudioReactivePipeline cleanup complete")
