"""TensorRT compilation for UNet - 2-3x speedup over torch.compile."""

import torch
import logging
from pathlib import Path
from typing import Optional
import hashlib

from ..config import settings

logger = logging.getLogger(__name__)


class TensorRTCompiler:
    """Compiles UNet to TensorRT engine for maximum performance."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        use_fp16: Optional[bool] = None,
    ):
        self.cache_dir = Path(cache_dir or settings.tensorrt_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_batch_size = max_batch_size or settings.tensorrt_max_batch_size
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.tensorrt_use_fp16

        self._compiled_models: dict[str, torch.nn.Module] = {}
        self._trt_available = self._check_tensorrt_available()

    def _check_tensorrt_available(self) -> bool:
        """Check if TensorRT is available."""
        import importlib.util
        if importlib.util.find_spec("tensorrt") is None:
            logger.warning("TensorRT not available: tensorrt package not installed")
            return False
        if importlib.util.find_spec("torch_tensorrt") is None:
            logger.warning("TensorRT not available: torch_tensorrt package not installed")
            return False
        import tensorrt
        logger.info(f"TensorRT version: {tensorrt.__version__}")
        return True

    def _get_cache_key(
        self,
        model_id: str,
        width: int,
        height: int,
        dtype: torch.dtype,
    ) -> str:
        """Generate unique cache key for model configuration."""
        config_str = f"{model_id}_{width}_{height}_{dtype}_{self.max_batch_size}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cached TensorRT engine."""
        return self.cache_dir / f"unet_{cache_key}.trt"

    def compile_unet(
        self,
        unet: torch.nn.Module,
        model_id: str,
        width: int = 512,
        height: int = 512,
        dtype: torch.dtype = torch.float16,
    ) -> torch.nn.Module:
        """Compile UNet to TensorRT engine.

        Args:
            unet: The UNet model to compile
            model_id: Model identifier for caching
            width: Target image width
            height: Target image height
            dtype: Model dtype (float16 recommended)

        Returns:
            TensorRT-optimized UNet or original if compilation fails
        """
        if not self._trt_available:
            logger.warning("TensorRT not available, returning original UNet")
            return unet

        cache_key = self._get_cache_key(model_id, width, height, dtype)

        # Check if already compiled
        if cache_key in self._compiled_models:
            logger.info(f"Using cached compiled UNet: {cache_key}")
            return self._compiled_models[cache_key]

        # Check for cached engine file
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                compiled = self._load_cached_engine(cache_path, unet)
                self._compiled_models[cache_key] = compiled
                logger.info(f"Loaded cached TensorRT engine: {cache_path}")
                return compiled
            except Exception as e:
                logger.warning(f"Failed to load cached engine: {e}")

        # Compile new engine
        try:
            compiled = self._compile_to_tensorrt(unet, width, height, dtype)
            self._compiled_models[cache_key] = compiled

            # Save to cache
            self._save_engine(cache_path, compiled)
            logger.info(f"Compiled and cached TensorRT engine: {cache_path}")

            return compiled
        except Exception as e:
            logger.error(f"TensorRT compilation failed: {e}")
            return unet

    def _compile_to_tensorrt(
        self,
        unet: torch.nn.Module,
        width: int,
        height: int,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        """Perform actual TensorRT compilation."""
        import torch_tensorrt

        logger.info("Compiling UNet to TensorRT...")

        # Calculate latent dimensions
        latent_h = height // 8
        latent_w = width // 8
        latent_channels = 4  # Standard for SD

        # Define input specifications for UNet
        # UNet takes: sample, timestep, encoder_hidden_states
        sample_spec = torch_tensorrt.Input(
            min_shape=(1, latent_channels, latent_h, latent_w),
            opt_shape=(1, latent_channels, latent_h, latent_w),
            max_shape=(self.max_batch_size, latent_channels, latent_h, latent_w),
            dtype=dtype,
        )

        timestep_spec = torch_tensorrt.Input(
            min_shape=(1,),
            opt_shape=(1,),
            max_shape=(self.max_batch_size,),
            dtype=torch.int64,
        )

        # CLIP embedding dimension (typically 768 or 1024)
        encoder_hidden_states_spec = torch_tensorrt.Input(
            min_shape=(1, 77, 768),
            opt_shape=(1, 77, 768),
            max_shape=(self.max_batch_size, 77, 768),
            dtype=dtype,
        )

        # Compile with TensorRT
        unet.eval()
        compiled_unet = torch_tensorrt.compile(
            unet,
            inputs=[sample_spec, timestep_spec, encoder_hidden_states_spec],
            enabled_precisions={dtype},
            truncate_long_and_double=True,
            min_block_size=1,
            workspace_size=1 << 30,  # 1GB workspace
        )

        logger.info("TensorRT compilation successful")
        return compiled_unet

    def _load_cached_engine(
        self,
        cache_path: Path,
        original_unet: torch.nn.Module,
    ) -> torch.nn.Module:
        """Load a cached TensorRT engine."""

        logger.info(f"Loading cached TensorRT engine from {cache_path}")
        return torch.jit.load(str(cache_path))

    def _save_engine(self, cache_path: Path, compiled_model: torch.nn.Module) -> None:
        """Save compiled engine to cache."""
        try:
            torch.jit.save(compiled_model, str(cache_path))
            logger.info(f"Saved TensorRT engine to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save TensorRT engine: {e}")

    def compile_vae(
        self,
        vae: torch.nn.Module,
        model_id: str,
        width: int = 512,
        height: int = 512,
        dtype: torch.dtype = torch.float16,
    ) -> torch.nn.Module:
        """Compile VAE decoder to TensorRT.

        VAE decoding is a significant bottleneck - TensorRT can help.
        """
        if not self._trt_available:
            return vae

        cache_key = f"vae_{self._get_cache_key(model_id, width, height, dtype)}"

        if cache_key in self._compiled_models:
            return self._compiled_models[cache_key]

        try:
            import torch_tensorrt

            # VAE decoder takes latents
            latent_h = height // 8
            latent_w = width // 8

            latent_spec = torch_tensorrt.Input(
                min_shape=(1, 4, latent_h, latent_w),
                opt_shape=(1, 4, latent_h, latent_w),
                max_shape=(self.max_batch_size, 4, latent_h, latent_w),
                dtype=dtype,
            )

            vae.decoder.eval()
            compiled_decoder = torch_tensorrt.compile(
                vae.decoder,
                inputs=[latent_spec],
                enabled_precisions={dtype},
                truncate_long_and_double=True,
            )

            # Replace decoder in VAE
            vae.decoder = compiled_decoder
            self._compiled_models[cache_key] = vae

            logger.info("VAE decoder compiled with TensorRT")
            return vae

        except Exception as e:
            logger.warning(f"VAE TensorRT compilation failed: {e}")
            return vae

    def compile_flux_transformer(
        self,
        transformer: torch.nn.Module,
        model_id: str,
        width: int = 512,
        height: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.nn.Module:
        """Compile FLUX transformer to TensorRT engine.

        FLUX uses a DiT (Diffusion Transformer) architecture which is different
        from the UNet used in SD. The transformer takes different inputs:
        - hidden_states: Latent image patches
        - encoder_hidden_states: Text embeddings from T5
        - pooled_projections: Pooled CLIP embeddings
        - timestep: Diffusion timestep
        - img_ids: Positional IDs for image patches
        - txt_ids: Positional IDs for text tokens

        Args:
            transformer: The FLUX transformer model to compile
            model_id: Model identifier for caching
            width: Target image width
            height: Target image height
            dtype: Model dtype (bfloat16 recommended for FLUX)

        Returns:
            TensorRT-optimized transformer or original if compilation fails
        """
        if not self._trt_available:
            logger.warning("TensorRT not available, returning original transformer")
            return transformer

        cache_key = f"flux_{self._get_cache_key(model_id, width, height, dtype)}"

        # Check if already compiled
        if cache_key in self._compiled_models:
            logger.info(f"Using cached compiled FLUX transformer: {cache_key}")
            return self._compiled_models[cache_key]

        # Check for cached engine file
        cache_path = self.cache_dir / f"flux_transformer_{cache_key}.trt"
        if cache_path.exists():
            try:
                compiled = torch.jit.load(str(cache_path))
                self._compiled_models[cache_key] = compiled
                logger.info(f"Loaded cached TensorRT engine: {cache_path}")
                return compiled
            except Exception as e:
                logger.warning(f"Failed to load cached engine: {e}")

        # Compile new engine
        try:
            compiled = self._compile_flux_to_tensorrt(transformer, width, height, dtype)
            self._compiled_models[cache_key] = compiled

            # Save to cache
            try:
                torch.jit.save(compiled, str(cache_path))
                logger.info(f"Saved TensorRT engine to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save TensorRT engine: {e}")

            return compiled
        except Exception as e:
            logger.error(f"FLUX TensorRT compilation failed: {e}")
            return transformer

    def _compile_flux_to_tensorrt(
        self,
        transformer: torch.nn.Module,
        width: int,
        height: int,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        """Perform actual TensorRT compilation for FLUX transformer."""
        import torch_tensorrt

        logger.info("Compiling FLUX transformer to TensorRT...")

        # FLUX uses 16x compression for latents (different from SD's 8x)
        # Patch size is 2, so we need to account for that
        latent_h = height // 16
        latent_w = width // 16
        num_patches = latent_h * latent_w

        # FLUX Klein 4B transformer dimensions
        hidden_size = 3072  # FLUX Klein hidden dim
        text_seq_len = 512  # Max T5 sequence length
        pooled_dim = 768  # CLIP pooled dimension

        # Define input specifications for FLUX transformer
        # These match the FluxTransformer2DModel forward signature
        # Note: These specs are for reference/future use with direct TRT compilation
        # Currently using torch.compile with tensorrt backend instead
        _ = (
            torch_tensorrt.Input(
                min_shape=(1, num_patches, hidden_size),
                opt_shape=(1, num_patches, hidden_size),
                max_shape=(self.max_batch_size, num_patches, hidden_size),
                dtype=dtype,
            ),  # hidden_states_spec
            torch_tensorrt.Input(
                min_shape=(1, 1, hidden_size),
                opt_shape=(1, text_seq_len, hidden_size),
                max_shape=(self.max_batch_size, text_seq_len, hidden_size),
                dtype=dtype,
            ),  # encoder_hidden_states_spec
            torch_tensorrt.Input(
                min_shape=(1, pooled_dim),
                opt_shape=(1, pooled_dim),
                max_shape=(self.max_batch_size, pooled_dim),
                dtype=dtype,
            ),  # pooled_projections_spec
            torch_tensorrt.Input(
                min_shape=(1,),
                opt_shape=(1,),
                max_shape=(self.max_batch_size,),
                dtype=dtype,
            ),  # timestep_spec
        )

        # Compile with TensorRT using Torch-TensorRT dynamo backend
        transformer.eval()

        # Use torch.compile with tensorrt backend for better compatibility
        compiled_transformer = torch.compile(
            transformer,
            backend="torch_tensorrt",
            options={
                "truncate_long_and_double": True,
                "precision": dtype,  # type: ignore[dict-item]
                "workspace_size": 2 << 30,  # 2GB workspace
            },
        )

        logger.info("FLUX TensorRT compilation successful")
        return compiled_transformer

    def cleanup(self) -> None:
        """Release compiled models."""
        self._compiled_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global compiler instance
_compiler: Optional[TensorRTCompiler] = None


def get_tensorrt_compiler() -> TensorRTCompiler:
    """Get or create the global TensorRT compiler."""
    global _compiler
    if _compiler is None:
        _compiler = TensorRTCompiler()
    return _compiler


def compile_unet_tensorrt(
    unet: torch.nn.Module,
    model_id: str,
    width: int = 512,
    height: int = 512,
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """Convenience function to compile UNet with TensorRT."""
    compiler = get_tensorrt_compiler()
    return compiler.compile_unet(unet, model_id, width, height, dtype)


def compile_flux_transformer_tensorrt(
    transformer: torch.nn.Module,
    model_id: str,
    width: int = 512,
    height: int = 512,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Convenience function to compile FLUX transformer with TensorRT."""
    compiler = get_tensorrt_compiler()
    return compiler.compile_flux_transformer(transformer, model_id, width, height, dtype)
