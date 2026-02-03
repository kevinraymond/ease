"""WebSocket endpoint and session management."""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Optional

import numpy as np
import psutil
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .protocol import (
    AudioMetrics,
    GenerationConfig,
    MappingConfig,
    FrameMetadata,
    StatusMessage,
    ServerConfig,
    SystemStats,
    ErrorMessage,
    FpsMessage,
    StoryConfig,
    StoryStateResponse,
    RawAudioChunk,
    LyricInfo,
    LyricUpdate,
    LyricPipelineState,
    AVAILABLE_BACKENDS,
    AUDIO_REACTIVE_BACKEND,
)
from ..story.schema import StoryScript, SceneDefinition, SceneTrigger, SceneTransition
from ..story.presets import get_story_preset, list_story_presets
from ..lyrics import (
    LyricProvider,
    create_lyric_provider_from_settings,
)
from ..config import settings

# Process-specific monitoring for accurate CPU/RAM stats
_PROCESS = psutil.Process(os.getpid())
_PROCESS.cpu_percent(interval=None)  # Prime first call (returns 0)

logger = logging.getLogger(__name__)


class GenerationSession:
    """Manages a single client's generation session."""

    def __init__(self, websocket: WebSocket):
        from ..generation.pipeline import GenerationPipeline  # Local import to avoid circular import
        self.websocket = websocket
        self.pipeline: Optional[GenerationPipeline] = None
        self.config = GenerationConfig()
        self.is_generating = False
        self._generation_task: Optional[asyncio.Task] = None
        self._latest_metrics: Optional[AudioMetrics] = None
        self._metrics_lock = asyncio.Lock()
        # Beat/onset persistence - these are momentary events that must persist
        # until the slow generation loop can consume them
        self._pending_beat: bool = False
        self._pending_onset: bool = False
        self._pending_onset_confidence: float = 0.0
        # Story to be loaded when pipeline is initialized
        self._pending_story: Optional[StoryScript] = None
        self._story_state_interval: int = 30  # Send story state every N frames

        # Lyric detection provider (plug-and-play interface)
        self._lyric_provider: Optional[LyricProvider] = None
        self._lyrics_enabled: bool = False
        self._current_lyrics: Optional[LyricInfo] = None

        # FPS update task
        self._fps_task: Optional[asyncio.Task] = None

    async def handle_message(self, data: str) -> None:
        """Process an incoming WebSocket message."""
        try:
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "config":
                await self._handle_config(msg)
            elif msg_type == "metrics":
                await self._handle_metrics(msg)
            elif msg_type == "mapping":
                await self._handle_mapping(msg)
            elif msg_type == "start":
                await self._handle_start()
            elif msg_type == "stop":
                await self._handle_stop()
            elif msg_type == "reset":
                await self._handle_reset()
            elif msg_type == "refresh_anchor":
                await self._handle_refresh_anchor()
            elif msg_type == "story_load":
                await self._handle_story_load(msg)
            elif msg_type == "story_load_preset":
                await self._handle_story_load_preset(msg)
            elif msg_type == "story_control":
                await self._handle_story_control(msg)
            elif msg_type == "story_unload":
                await self._handle_story_unload()
            elif msg_type == "audio_chunk":
                await self._handle_audio_chunk(msg)
            elif msg_type == "reset_lyrics":
                await self._handle_reset_lyrics()
            elif msg_type == "clear_base_image":
                await self._handle_clear_base_image()
            elif msg_type == "switch_backend":
                await self._handle_switch_backend(msg)
            else:
                await self._send_error(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            await self._send_error(f"Invalid JSON: {e}")
        except ValidationError as e:
            await self._send_error(f"Validation error: {e}")
        except Exception as e:
            logger.exception("Error handling message")
            await self._send_error(f"Internal error: {e}")

    async def _handle_config(self, msg: dict) -> None:
        """Handle configuration update."""
        config_data = msg.get("config", msg)
        # Remove 'type' if it exists in config_data
        config_data = {k: v for k, v in config_data.items() if k != "type"}

        # Merge partial config with existing config to preserve unchanged fields
        existing_config = self.config.model_dump() if self.config else {}
        merged_config = {**existing_config, **config_data}
        self.config = GenerationConfig(**merged_config)

        if self.pipeline:
            self.pipeline.update_config(self.config)

        # Handle base image if provided (only set, don't auto-clear on every config update)
        # Users can click "Reset Image" to go back to txt2img mode
        if self.config.base_image and self.pipeline:
            try:
                from io import BytesIO
                from PIL import Image
                image_data = base64.b64decode(self.config.base_image)
                image = Image.open(BytesIO(image_data)).convert("RGB")
                self.pipeline.set_source_image(image)
                logger.info(f"Base image set: {image.size}")
            except Exception as e:
                logger.warning(f"Failed to decode base image: {e}")

        # Update lyrics enabled state
        self._lyrics_enabled = getattr(self.config, 'enable_lyrics', False)
        if self._lyrics_enabled:
            logger.info("Lyric detection enabled")
        else:
            if self._lyric_provider:
                self._lyric_provider.reset()
            self._current_lyrics = None

        logger.info(f"Config updated: mode={self.config.generation_mode}, preset={self.config.mapping_preset}, acceleration={getattr(self.config, 'acceleration', 'lcm')}, lyrics={self._lyrics_enabled}")
        await self._send_status("generating" if self.is_generating else "connected", "Configuration updated", include_server_config=True)

    async def _handle_metrics(self, msg: dict) -> None:
        """Handle incoming audio metrics."""
        metrics_data = msg.get("metrics", {})
        # Debug: log incoming metrics periodically
        if hasattr(self, '_metrics_count'):
            self._metrics_count += 1
        else:
            self._metrics_count = 1
        if self._metrics_count % 100 == 0:
            logger.info(f"Received metrics #{self._metrics_count}: rms={metrics_data.get('rms', 0):.3f}, bass={metrics_data.get('bass', 0):.3f}")

        # Inject current lyrics into metrics if available
        if self._lyrics_enabled and self._current_lyrics:
            metrics_data["lyrics"] = self._current_lyrics.model_dump()

        metrics = AudioMetrics(**metrics_data)

        async with self._metrics_lock:
            self._latest_metrics = metrics

            # Persist beat/onset events - these are momentary but generation is slow
            # so we need to hold onto them until the generation loop consumes them
            if metrics.is_beat:
                self._pending_beat = True

            if metrics.onset and metrics.onset.is_onset:
                self._pending_onset = True
                self._pending_onset_confidence = max(self._pending_onset_confidence, metrics.onset.confidence)

    async def _handle_mapping(self, msg: dict) -> None:
        """Handle mapping configuration update."""
        mapping_data = msg.get("mapping_config", {})
        mapping_config = MappingConfig(**mapping_data)

        # Store in config for future use
        self.config.mapping_config = mapping_config

        # Update pipeline's audio mapper if running
        if self.pipeline:
            self.pipeline.update_mapping_config(mapping_config)

        logger.info(f"Mapping config updated: preset={mapping_config.preset_name}, mappings={len(mapping_config.mappings)}")
        await self._send_status("generating" if self.is_generating else "connected", "Mapping configuration updated")

    async def _handle_start(self) -> None:
        """Start generation loop."""
        if self.is_generating:
            return

        logger.info("Starting generation...")
        self.is_generating = True

        # Initialize pipeline if needed
        if not self.pipeline:
            from ..generation.pipeline import GenerationPipeline

            # Build initialization message with warnings about slow operations
            init_warnings = []
            if settings.compile_unet:
                init_warnings.append("torch.compile enabled - first frame will be slow (~30-60s)")
            if settings.generator_backend == "flux_klein" and settings.flux_compile:
                init_warnings.append("FLUX compile enabled - first frame will be slow (~60-120s)")

            init_msg = "Initializing pipeline..."
            if init_warnings:
                init_msg = f"Initializing: {'; '.join(init_warnings)}"

            await self._send_status("initializing", init_msg)

            self.pipeline = GenerationPipeline(self.config)
            # Run initialization in thread pool to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.pipeline.initialize)

            # Apply mapping config if it was received before pipeline creation
            if hasattr(self.config, 'mapping_config') and self.config.mapping_config:
                self.pipeline.update_mapping_config(self.config.mapping_config)
                logger.info(f"Applied pre-existing mapping config: triggers={self.config.mapping_config.triggers}")

            # Load pending story if one was received before pipeline creation
            if self._pending_story:
                self.pipeline.load_story(self._pending_story)
                logger.info(f"Applied pending story: {self._pending_story.name}")
                self._pending_story = None

        await self._send_status("generating", "Generation started", include_server_config=True)

        # Start generation loop
        self._generation_task = asyncio.create_task(self._generation_loop())

        # Start FPS update task
        self._fps_task = asyncio.create_task(self._fps_update_loop())

    async def _handle_stop(self) -> None:
        """Stop generation loop."""
        logger.info("Stopping generation...")
        self.is_generating = False

        if self._generation_task:
            self._generation_task.cancel()
            try:
                await self._generation_task
            except asyncio.CancelledError:
                pass
            self._generation_task = None

        if self._fps_task:
            self._fps_task.cancel()
            try:
                await self._fps_task
            except asyncio.CancelledError:
                pass
            self._fps_task = None

        if self.pipeline:
            self.pipeline.reset()

        await self._send_status("stopped", "Generation stopped")

    async def _handle_reset(self) -> None:
        """Reset feedback loop - next frame will be txt2img."""
        logger.info("Resetting feedback loop...")
        if self.pipeline:
            self.pipeline.reset()
        await self._send_status("generating" if self.is_generating else "connected", "Feedback loop reset")

    async def _handle_clear_base_image(self) -> None:
        """Clear the persistent base image - reset will now use txt2img."""
        logger.info("Clearing base image...")
        if self.pipeline:
            self.pipeline.clear_base_image()
        await self._send_status("generating" if self.is_generating else "connected", "Base image cleared")

    async def _handle_refresh_anchor(self) -> None:
        """Refresh the FLUX anchor frame - next frame will be fresh txt2img."""
        logger.info("Refreshing FLUX anchor frame...")
        if self.pipeline:
            self.pipeline.refresh_anchor()
        await self._send_status("generating" if self.is_generating else "connected", "Anchor frame refreshed")

    async def _handle_switch_backend(self, msg: dict) -> None:
        """Handle runtime backend switching."""
        backend_id = msg.get("backend_id", "")

        # Validate backend ID
        valid_backends = {b.id for b in AVAILABLE_BACKENDS}
        if backend_id not in valid_backends:
            await self._send_error(f"Unknown backend: {backend_id}. Available: {', '.join(valid_backends)}")
            return

        # Check if already on this backend
        current_backend = getattr(self.config, 'generator_backend', 'audio_reactive')
        if backend_id == current_backend:
            await self._send_status("generating" if self.is_generating else "connected",
                                   f"Already using {backend_id}")
            return

        logger.info(f"Switching backend from {current_backend} to {backend_id}...")

        # Send switching status immediately for UI feedback
        await self._send_status("switching", f"Switching to {backend_id}...", include_server_config=True)

        # Remember if we were generating
        was_generating = self.is_generating

        # Stop generation if running
        if self.is_generating:
            self.is_generating = False
            if self._generation_task:
                self._generation_task.cancel()
                try:
                    await self._generation_task
                except asyncio.CancelledError:
                    pass
                self._generation_task = None

        # Clean up current pipeline
        if self.pipeline:
            self.pipeline.cleanup()
            self.pipeline = None

        # Force VRAM cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"VRAM cleanup failed: {e}")

        # Update config with new backend
        self.config = GenerationConfig(
            **{**self.config.model_dump(), "generator_backend": backend_id}
        )

        # Send status update with new backend info
        await self._send_status("connected", f"Switched to {backend_id}", include_server_config=True)

        # Restart generation if it was running
        if was_generating:
            await self._handle_start()

    async def _handle_story_load(self, msg: dict) -> None:
        """Handle loading a story from JSON definition."""
        story_data = msg.get("story", {})
        story_config = StoryConfig(**story_data)

        # Convert protocol StoryConfig to internal StoryScript
        story = self._convert_story_config(story_config)

        if self.pipeline:
            self.pipeline.load_story(story)
            await self._send_story_state()
        else:
            # Store for later when pipeline is initialized
            self._pending_story = story

        logger.info(f"Story loaded: {story.name} with {len(story.scenes)} scenes")
        await self._send_status("generating" if self.is_generating else "connected", f"Story '{story.name}' loaded")

    async def _handle_story_load_preset(self, msg: dict) -> None:
        """Handle loading a preset story by name."""
        preset_name = msg.get("preset_name", "")
        available = list_story_presets()

        if preset_name not in available:
            await self._send_error(f"Unknown story preset: {preset_name}. Available: {', '.join(available)}")
            return

        story = get_story_preset(preset_name)

        if self.pipeline:
            self.pipeline.load_story(story)
            await self._send_story_state()
        else:
            self._pending_story = story

        logger.info(f"Story preset loaded: {preset_name}")
        await self._send_status("generating" if self.is_generating else "connected", f"Story preset '{preset_name}' loaded")

    async def _handle_story_control(self, msg: dict) -> None:
        """Handle story playback control commands."""
        action = msg.get("action", "")

        if not self.pipeline:
            await self._send_error("Pipeline not initialized")
            return

        if not self.pipeline.has_story():
            await self._send_error("No story loaded")
            return

        if action == "play":
            self.pipeline.story_play()
            logger.info("Story playback resumed")
        elif action == "pause":
            self.pipeline.story_pause()
            logger.info("Story playback paused")
        elif action == "skip_next":
            self.pipeline.story_skip_next()
            logger.info("Skipped to next scene")
        elif action == "skip_prev":
            self.pipeline.story_skip_prev()
            logger.info("Skipped to previous scene")
        elif action == "restart":
            self.pipeline.story_restart()
            logger.info("Story restarted")
        elif action == "stop":
            self.pipeline.unload_story()
            logger.info("Story stopped and unloaded")
        else:
            await self._send_error(f"Unknown story control action: {action}")
            return

        await self._send_story_state()
        await self._send_status("generating" if self.is_generating else "connected", f"Story {action}")

    async def _handle_story_unload(self) -> None:
        """Handle unloading the current story."""
        if self.pipeline:
            self.pipeline.unload_story()

        logger.info("Story unloaded")
        await self._send_status("generating" if self.is_generating else "connected", "Story unloaded")

    async def _handle_audio_chunk(self, msg: dict) -> None:
        """Process incoming raw audio for lyric detection."""
        if not self._lyrics_enabled:
            return

        try:
            chunk = RawAudioChunk(**msg)

            # Decode base64 audio data to numpy array
            audio_bytes = base64.b64decode(chunk.audio_data)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Initialize lyric provider lazily
            if self._lyric_provider is None:
                self._lyric_provider = create_lyric_provider_from_settings()
                self._lyric_provider.start()
                logger.info(f"Lyric provider initialized: {type(self._lyric_provider).__name__}")

            # Add audio to provider
            self._lyric_provider.add_audio_chunk(audio_array, chunk.sample_rate)

            # Get result and update current lyrics
            await self._update_lyrics_from_provider()

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    async def _handle_reset_lyrics(self) -> None:
        """Reset the lyric detector for a new song."""
        self._current_lyrics = None

        if self._lyric_provider is not None:
            self._lyric_provider.reset()
            logger.info("Lyric provider reset")

        # Send acknowledgment
        await self.websocket.send_text(json.dumps({"type": "reset_lyrics", "success": True}))

    async def _update_lyrics_from_provider(self) -> None:
        """Update current lyrics from the provider."""
        if self._lyric_provider is None:
            return

        try:
            result = self._lyric_provider.get_result()
            keywords = self._lyric_provider.get_current_keywords()

            # Convert to list of tuples with max 4 keywords
            active_keywords = keywords[:4] if keywords else []

            # Create lyric info from provider result
            self._current_lyrics = LyricInfo(
                text=result.text[-200:] if result and result.text else "",
                keywords=active_keywords,
                confidence=result.confidence if result else 0.0,
                is_singing=result.is_singing if result else bool(active_keywords),
                language=result.language if result else "en",
                pipeline_state=self._map_provider_state(),
                fingerprint_progress=self._lyric_provider.get_fingerprint_progress(),
                matched_song_title=result.matched_song_title if result else None,
                matched_song_artist=result.matched_song_artist if result else None,
            )

            # Send update to frontend
            await self._send_lyric_update()

        except Exception as e:
            logger.error(f"Error updating lyrics from provider: {e}")

    def _map_provider_state(self) -> Optional[LyricPipelineState]:
        """Map provider state to protocol state."""
        if self._lyric_provider is None:
            return None

        from ..lyrics import LyricProviderState
        state_map = {
            LyricProviderState.INITIALIZING: LyricPipelineState.INITIALIZING,
            LyricProviderState.READY: LyricPipelineState.NOT_MATCHED,
            LyricProviderState.PROCESSING: LyricPipelineState.NOT_MATCHED,
            LyricProviderState.FINGERPRINTING: LyricPipelineState.FINGERPRINTING,
            LyricProviderState.MATCHED: LyricPipelineState.MATCHED,
            LyricProviderState.STOPPED: LyricPipelineState.STOPPED,
        }
        return state_map.get(self._lyric_provider.state)

    async def _send_lyric_update(self) -> None:
        """Send lyric update to frontend."""
        if self._current_lyrics:
            update = LyricUpdate(lyrics=self._current_lyrics)
            await self.websocket.send_text(update.model_dump_json())

    def _convert_story_config(self, config: StoryConfig) -> StoryScript:
        """Convert protocol StoryConfig to internal StoryScript."""
        scenes = []
        for scene_config in config.scenes:
            scene = SceneDefinition(
                id=scene_config.id,
                base_prompt=scene_config.base_prompt,
                negative_prompt=scene_config.negative_prompt,
                duration_frames=scene_config.duration_frames,
                trigger=SceneTrigger(scene_config.trigger),
                trigger_value=scene_config.trigger_value,
                energy_high_prompt=scene_config.energy_high_prompt,
                energy_low_prompt=scene_config.energy_low_prompt,
                energy_blend_range=scene_config.energy_blend_range,
                beat_prompt_modifier=scene_config.beat_prompt_modifier,
                transition=SceneTransition(scene_config.transition),
                transition_frames=scene_config.transition_frames,
            )
            scenes.append(scene)

        return StoryScript(
            name=config.name,
            description=config.description,
            default_negative_prompt=config.default_negative_prompt,
            scenes=scenes,
            loop=config.loop,
            audio_reactive_keywords=config.audio_reactive_keywords,
            base_seed=config.base_seed,
        )

    async def _send_story_state(self) -> None:
        """Send current story state to client."""
        if not self.pipeline or not self.pipeline.has_story():
            return

        state = self.pipeline.get_story_state()
        if state:
            response = StoryStateResponse(
                story_name=state.story_name,
                current_scene_idx=state.current_scene_idx,
                current_scene_id=state.current_scene_id,
                frame_in_scene=state.frame_in_scene,
                beat_count_in_scene=state.beat_count_in_scene,
                is_transitioning=state.is_transitioning,
                transition_progress=state.transition_progress,
                is_playing=state.is_playing,
                is_complete=state.is_complete,
                total_scenes=state.total_scenes,
                scene_progress=state.scene_progress,
            )
            await self.websocket.send_text(response.model_dump_json())

    async def _generation_loop(self) -> None:
        """Main generation loop - runs continuously while generating."""
        loop = asyncio.get_event_loop()

        while self.is_generating:
            # Re-read target FPS each frame so it can be changed dynamically
            target_frame_time = 1.0 / self.config.target_fps
            frame_start = time.perf_counter()

            # Get latest metrics and consume any pending beat/onset events
            async with self._metrics_lock:
                metrics = self._latest_metrics

                # Consume pending beat - these persist until we process them
                pending_beat = self._pending_beat
                self._pending_beat = False

                # Consume pending onset
                pending_onset = self._pending_onset
                pending_onset_confidence = self._pending_onset_confidence
                self._pending_onset = False
                self._pending_onset_confidence = 0.0

            if metrics is None:
                # No metrics yet, wait and retry
                await asyncio.sleep(0.01)
                continue

            # Apply pending events to metrics (override momentary values)
            if pending_beat and not metrics.is_beat:
                # Create a copy with the beat flag set
                metrics = AudioMetrics(
                    **{**metrics.model_dump(), "is_beat": True}
                )

            if pending_onset and metrics.onset and not metrics.onset.is_onset:
                onset_data = metrics.onset.model_dump()
                onset_data["is_onset"] = True
                onset_data["confidence"] = max(onset_data["confidence"], pending_onset_confidence)
                metrics = AudioMetrics(
                    **{**metrics.model_dump(), "onset": onset_data}
                )

            # Skip generation when audio is silent (music paused/stopped)
            # This saves GPU resources when there's no audio to react to
            silence_threshold = 0.01  # RMS below this is considered silence
            if metrics.rms < silence_threshold:
                await asyncio.sleep(0.1)  # Throttle when silent
                continue

            try:
                # Generate frame in thread pool
                frame = await loop.run_in_executor(
                    None,
                    self.pipeline.generate,
                    metrics,
                )

                # Send frame metadata (include system stats every 10 frames)
                include_stats = frame.frame_id % 10 == 0
                metadata = FrameMetadata(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    width=self.config.width,
                    height=self.config.height,
                    generation_params={
                        "prompt": frame.params.prompt,
                        "strength": frame.params.strength,
                        "guidance_scale": frame.params.guidance_scale,
                        "seed": frame.params.seed,
                        "is_onset": frame.params.is_onset,
                        "onset_confidence": frame.params.onset_confidence,
                        "color_keywords": frame.params.color_keywords,
                    },
                    system_stats=self._get_system_stats() if include_stats else None,
                )
                await self.websocket.send_text(metadata.model_dump_json())

                # Send binary frame data
                await self.websocket.send_bytes(frame.jpeg_bytes)

                # Send pose preview every frame for real-time feedback
                pose_bytes = await loop.run_in_executor(
                    None,
                    self.pipeline.get_pose_preview,
                )
                if pose_bytes:
                    # Send pose preview as separate message
                    await self.websocket.send_text('{"type":"pose_preview"}')
                    await self.websocket.send_bytes(pose_bytes)

                # Send story state updates periodically
                if self.pipeline.has_story() and frame.frame_id % self._story_state_interval == 0:
                    await self._send_story_state()

            except Exception as e:
                logger.exception("Error in generation loop")
                await self._send_error(f"Generation error: {e}")
                await asyncio.sleep(0.1)
                continue

            # Frame rate limiting
            elapsed = time.perf_counter() - frame_start
            if elapsed < target_frame_time:
                await asyncio.sleep(target_frame_time - elapsed)

    async def _fps_update_loop(self) -> None:
        """Periodically send FPS updates to the client."""
        while self.is_generating:
            try:
                if self.pipeline:
                    fps = self.pipeline.get_fps()
                    msg = FpsMessage(fps=fps)
                    await self.websocket.send_text(msg.model_dump_json())
            except Exception:
                pass  # Skip failed updates silently
            await asyncio.sleep(0.5)

    def _get_system_stats(self) -> SystemStats:
        """Collect system resource usage stats (process-specific CPU/RAM)."""
        # Process-specific CPU and RAM for accurate app stats
        cpu_percent = _PROCESS.cpu_percent(interval=None)
        process_mem = _PROCESS.memory_info()
        ram_used_gb = process_mem.rss / (1024**3)  # RSS = actual memory used
        ram_total_gb = psutil.virtual_memory().total / (1024**3)  # Keep system total for context

        # GPU stats (if available)
        gpu_util = None
        vram_used_gb = None
        vram_total_gb = None

        try:
            import torch
            if torch.cuda.is_available():
                # Get memory info for the first GPU
                vram_reserved = torch.cuda.memory_reserved(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory
                vram_used_gb = vram_reserved / (1024**3)  # Use reserved as it's more accurate
                vram_total_gb = vram_total / (1024**3)

                # Try to get GPU utilization via pynvml
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except Exception:
                    pass  # pynvml not available or failed
        except Exception:
            pass  # torch not available or no CUDA

        return SystemStats(
            cpu_percent=round(cpu_percent, 1),
            ram_used_gb=round(ram_used_gb, 1),
            ram_total_gb=round(ram_total_gb, 1),
            gpu_util=round(gpu_util, 1) if gpu_util is not None else None,
            vram_used_gb=round(vram_used_gb, 1) if vram_used_gb is not None else None,
            vram_total_gb=round(vram_total_gb, 1) if vram_total_gb is not None else None,
        )

    async def _send_status(
        self,
        status: str,
        message: Optional[str] = None,
        include_server_config: bool = False,
        include_system_stats: bool = False,
    ) -> None:
        """Send status message to client."""
        fps = self.pipeline.get_fps() if self.pipeline else None

        # Include server config on initial connection or when requested
        server_config = None
        if include_server_config:
            # Use session config if available, otherwise fall back to server defaults
            accel = getattr(self.config, 'acceleration', settings.acceleration)
            hyper_steps = getattr(self.config, 'hyper_sd_steps', settings.hyper_sd_steps)
            model = getattr(self.config, 'model_id', settings.model)
            current_backend = getattr(self.config, 'generator_backend', settings.generator_backend)

            # Get capabilities for current backend
            backend_info = next(
                (b for b in AVAILABLE_BACKENDS if b.id == current_backend),
                AUDIO_REACTIVE_BACKEND
            )

            server_config = ServerConfig(
                acceleration=accel,
                hyper_sd_steps=hyper_steps if accel == "hyper-sd" else None,
                model=model,
                current_backend=current_backend,
                available_backends=AVAILABLE_BACKENDS,
                capabilities=backend_info.capabilities,
            )

        # Include system stats when generating
        system_stats = None
        if include_system_stats:
            system_stats = self._get_system_stats()

        msg = StatusMessage(
            status=status,
            message=message,
            fps=fps,
            server_config=server_config,
            system_stats=system_stats,
        )
        await self.websocket.send_text(msg.model_dump_json())

    async def _send_error(self, error: str, code: Optional[str] = None) -> None:
        """Send error message to client."""
        msg = ErrorMessage(error=error, code=code)
        await self.websocket.send_text(msg.model_dump_json())

    async def cleanup(self) -> None:
        """Clean up session resources."""
        self.is_generating = False

        if self._generation_task:
            self._generation_task.cancel()
            try:
                await self._generation_task
            except asyncio.CancelledError:
                pass

        if self._fps_task:
            self._fps_task.cancel()
            try:
                await self._fps_task
            except asyncio.CancelledError:
                pass

        if self.pipeline:
            self.pipeline.cleanup()

        if self._lyric_provider:
            self._lyric_provider.stop()
            self._lyric_provider = None


async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint handler."""
    await websocket.accept()
    logger.info(f"Client connected: {websocket.client}")

    session = GenerationSession(websocket)

    # Send initial status with server config
    await session._send_status("connected", "Ready for configuration", include_server_config=True)

    try:
        while True:
            data = await websocket.receive_text()
            await session.handle_message(data)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        await session.cleanup()
