"""FastAPI entry point for the AI generation server."""

import gc
import logging
import sys
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .server.websocket_handler import websocket_endpoint

# Configure logging from settings
def setup_logging():
    """Configure logging with sensible defaults.

    Log levels can be configured via environment variables:
      EASE_LOG_LEVEL=INFO               # Root level (DEBUG, INFO, WARNING, ERROR)
      EASE_LOG_LEVEL_GENERATION=WARNING # Generation module (very verbose at DEBUG)
      EASE_LOG_LEVEL_PIPELINE=INFO      # High-level pipeline events
      EASE_LOG_LEVEL_LYRICS=INFO        # Lyrics/transcription module
      EASE_LOG_LEVEL_SERVER=INFO        # WebSocket server
    """
    root_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=root_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Module-specific levels from settings
    module_levels = {
        "src.generation": settings.log_level_generation,
        "src.generation.pipeline": settings.log_level_pipeline,
        "src.lyrics": settings.log_level_lyrics,
        "src.server": settings.log_level_server,
        # Third-party libraries that can be noisy
        "httpx": "WARNING",
        "httpcore": "WARNING",
        "uvicorn.access": "WARNING",
    }

    for module, level_str in module_levels.items():
        level = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger(module).setLevel(level)


setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting EASE AI Server...")
    logger.info(f"Model: {settings.model}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Generator backend: {settings.generator_backend}")
    logger.info(f"TensorRT: {settings.use_tensorrt}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="EASE AI Server",
    description="Real-time AI image generation driven by audio",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "ease-ai-server",
        "model": settings.model,
    }


@app.get("/config")
async def get_config():
    """Get current server configuration."""
    return {
        "model_id": settings.model,
        "device": settings.device,
        "default_width": settings.width,
        "default_height": settings.height,
        "generator_backend": settings.generator_backend,
        "use_tensorrt": settings.use_tensorrt,
    }


@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """WebSocket endpoint for real-time generation."""
    await websocket_endpoint(websocket)


@app.get("/loras")
async def list_loras():
    """List available LoRA files in the configured directory."""
    from pathlib import Path

    lora_path = Path(settings.lora_dir)

    if not lora_path.exists() or not lora_path.is_dir():
        return {"loras": []}

    try:
        loras = sorted([
            f.name for f in lora_path.iterdir()
            if f.is_file() and f.suffix == ".safetensors"
        ])
        return {"loras": loras}
    except Exception as e:
        logger.error(f"Error listing LoRAs: {e}")
        return {"loras": []}


@app.post("/admin/clear-vram")
async def clear_vram():
    """Manually trigger VRAM cleanup.

    Useful for debugging memory issues or forcing cleanup between sessions.
    Runs gc.collect() + torch.cuda.synchronize() + torch.cuda.empty_cache().
    """
    if not torch.cuda.is_available():
        return {
            "status": "skipped",
            "message": "CUDA not available",
            "vram_before_gb": 0,
            "vram_after_gb": 0,
            "vram_freed_gb": 0,
        }

    vram_before = torch.cuda.memory_allocated() / 1024**3

    # Full cleanup sequence
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    vram_after = torch.cuda.memory_allocated() / 1024**3
    freed = vram_before - vram_after

    logger.info(f"Manual VRAM clear: {vram_before:.2f} GB -> {vram_after:.2f} GB (freed {freed:.2f} GB)")

    return {
        "status": "ok",
        "vram_before_gb": round(vram_before, 3),
        "vram_after_gb": round(vram_after, 3),
        "vram_freed_gb": round(freed, 3),
    }


def main():
    """Run the server."""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
