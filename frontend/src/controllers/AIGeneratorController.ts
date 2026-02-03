/**
 * AIGeneratorController - Manages WebSocket connection and AI frame generation.
 * Replaces the useAIGenerator React hook.
 */

import { EventBus, eventBus } from '../core/EventBus';
import {
  AIGeneratorConfig,
  AIGeneratorState,
  LyricInfo,
  SystemStats,
  MappingConfigPayload,
} from '../core/types';
import { AudioMetrics } from '../audio/types';

// Frame metadata from server
interface FrameMetadata {
  type: 'frame';
  frame_id: number;
  timestamp: number;
  width: number;
  height: number;
  format: 'jpeg';
  generation_params: {
    prompt: string;
    strength: number;
    guidance_scale: number;
    seed?: number;
    is_onset?: boolean;
    onset_confidence?: number;
    color_keywords?: string[];
  };
  system_stats?: SystemStats;
}

const DEFAULT_CONFIG: AIGeneratorConfig = {
  serverUrl: 'ws://localhost:8765/ws',
  generationMode: 'feedback',
  basePrompt: 'prismatic light figure emerging from shattered pixel grid, RGB streaks forming humanoid silhouette, digital chrysalis breaking open, volumetric rays, cinematic lighting',
  negativePrompt: 'nsfw, text, watermark, low quality, blurry',
  modelId: 'Lykon/dreamshaper-8',
  img2imgStrength: 0.35,
  mappingPreset: 'reactive',
  targetFps: 20,
  width: 256,
  height: 256,
  maintainAspectRatio: false,
  acceleration: 'lcm',
  hyperSdSteps: 1,
  useTaesd: false,
  useControlNet: false,
  controlNetPoseWeight: 0.8,
  controlNetPoseLock: true,
  useProceduralPose: false,
  poseAnimationMode: 'gentle',
  poseAnimationSpeed: 1.0,
  poseAnimationIntensity: 0.5,
  proceduralUseTxt2img: true,
  proceduralFixedSeed: null,
  proceduralBlendWeight: 0.4,
  useIPAdapter: false,
  ipAdapterScale: 0.6,
  temporalCoherence: 'blending',
  keyframeInterval: 4,
  keyframeStrength: 0.6,
  poseFraming: 'upper_body',
  enableShaderEffects: false,
  enableFlash: false,
  enableSpectralDisplacement: false,
  enableGlitchBlocks: false,
  enableTrebleGrain: false,
  enableBicubic: true,
  enableSharpening: false,
  sharpenStrength: 0.5,
  enableFSR: false,
  fsrSharpness: 0.8,
  enableLyrics: false,
  lyricDrivenMode: false,
  showLyricSubtitles: false,
  baseImage: null,
  lockToBaseImage: false,
  periodicPoseRefresh: false,
  loras: [],
  enableSilenceDegradation: true,
  silenceThreshold: 0.03,
  degradationRate: 0.5,
  recoveryRate: 2.0,
};

const DEFAULT_STATE: AIGeneratorState = {
  isConnected: false,
  isConnecting: false,
  isGenerating: false,
  isInitializing: false,
  isSwitchingBackend: false,
  pendingBackendId: null,
  statusMessage: null,
  fps: 0,
  currentFrame: null,
  frameId: 0,
  error: null,
  lastParams: null,
  posePreview: null,
  lyrics: null,
  serverConfig: null,
  systemStats: null,
};

export class AIGeneratorController {
  // State
  private _config: AIGeneratorConfig;
  private _state: AIGeneratorState;

  // WebSocket
  private ws: WebSocket | null = null;
  private pendingFrameMetadata: FrameMetadata | null = null;
  private pendingPosePreview = false;
  private frameImage: HTMLImageElement | null = null;
  private poseImage: HTMLImageElement | null = null;

  // Story message callback
  private storyMessageCallback: ((msg: any) => void) | null = null;

  // Local event bus
  private localBus = new EventBus();

  constructor(initialConfig?: Partial<AIGeneratorConfig>) {
    this._config = { ...DEFAULT_CONFIG, ...initialConfig };
    this._state = { ...DEFAULT_STATE };
  }

  // === Getters ===

  get config(): AIGeneratorConfig {
    return this._config;
  }

  get state(): AIGeneratorState {
    return this._state;
  }

  get ws_ref(): WebSocket | null {
    return this.ws;
  }

  // === Event subscription ===

  on(event: 'stateChange' | 'configChange' | 'frameReceived' | 'storyMessage', callback: (data: any) => void): () => void {
    return this.localBus.on(event, callback);
  }

  // Register story message handler
  onStoryMessage(callback: (msg: any) => void): void {
    this.storyMessageCallback = callback;
  }

  // === Actions ===

  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.updateState({ isConnecting: true, error: null });

    return new Promise((resolve, reject) => {
      try {
        const ws = new WebSocket(this._config.serverUrl);
        ws.binaryType = 'blob';

        ws.onopen = () => {
          this.ws = ws;
          this.updateState({
            isConnected: true,
            isConnecting: false,
          });

          // Send initial configuration
          this.sendConfig();
          eventBus.emit('ai:connected');
          resolve();
        };

        ws.onmessage = (event) => this.handleMessage(event);

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.updateState({
            isConnecting: false,
            error: 'Connection failed',
          });
          reject(error);
        };

        ws.onclose = () => {
          this.ws = null;
          this.updateState({
            isConnected: false,
            isConnecting: false,
            isGenerating: false,
          });
          eventBus.emit('ai:disconnected');
        };
      } catch (error) {
        this.updateState({
          isConnecting: false,
          error: String(error),
        });
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.updateState({ ...DEFAULT_STATE });
  }

  startGeneration(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('Cannot start generation: not connected');
      return;
    }

    this.ws.send(JSON.stringify({ type: 'start' }));
    this.updateState({ isGenerating: true });
  }

  stopGeneration(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.ws.send(JSON.stringify({ type: 'stop' }));
    this.updateState({ isGenerating: false });
  }

  resetFeedback(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    this.ws.send(JSON.stringify({ type: 'reset' }));
  }

  clearBaseImage(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.ws.send(JSON.stringify({ type: 'clear_base_image' }));
    this.updateConfig({ baseImage: null });
  }

  async clearVram(): Promise<void> {
    try {
      const httpUrl = this._config.serverUrl.replace(/^ws/, 'http').replace(/\/ws$/, '');
      const response = await fetch(`${httpUrl}/admin/clear-vram`, { method: 'POST' });
      const result = await response.json();
      console.log('VRAM cleared:', result);
    } catch (error) {
      console.error('Failed to clear VRAM:', error);
    }
  }

  switchBackend(backendId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('Cannot switch backend: not connected');
      return;
    }

    // Auto-switch to feedback mode if current mode requires StreamDiffusion
    if (backendId !== 'stream_diffusion' && this._config.generationMode === 'keyframe_rife') {
      console.log('Auto-switching to feedback mode (Pose Animation requires StreamDiffusion)');
      this.updateConfig({ generationMode: 'feedback' });
    }

    this.updateState({
      isSwitchingBackend: true,
      pendingBackendId: backendId,
    });

    this.ws.send(JSON.stringify({
      type: 'switch_backend',
      backend_id: backendId,
    }));
  }

  updateConfig(newConfig: Partial<AIGeneratorConfig>): void {
    this._config = { ...this._config, ...newConfig };

    // Send to server if connected
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.sendConfig();
    }

    this.localBus.emit('configChange', this._config);
    eventBus.emit('ai:configChanged', newConfig);
  }

  sendMetrics(metrics: AudioMetrics): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const msg = {
      type: 'metrics',
      metrics: {
        rms: metrics.rms,
        peak: metrics.peak,
        bass: metrics.bass,
        mid: metrics.mid,
        treble: metrics.treble,
        raw_bass: metrics.rawBass,
        raw_mid: metrics.rawMid,
        raw_treble: metrics.rawTreble,
        bpm: metrics.bpm,
        is_beat: metrics.isBeat,
        sample_rate: metrics.sampleRate,
        fft_size: metrics.fftSize,
        spectral_centroid: metrics.spectralCentroid,
        raw_spectral_centroid: metrics.rawSpectralCentroid,
        onset: {
          is_onset: metrics.onset.isOnset,
          confidence: metrics.onset.confidence,
          strength: metrics.onset.strength,
          spectral_flux: metrics.onset.spectralFlux,
        },
        chroma: {
          bins: Array.from(metrics.chroma.bins),
          energy: metrics.chroma.energy,
        },
        dominant_chroma: metrics.dominantChroma,
      },
      timestamp: Date.now() / 1000,
    };

    this.ws.send(JSON.stringify(msg));
  }

  sendMappingConfig(mappingConfig: MappingConfigPayload): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const msg = {
      type: 'mapping',
      mapping_config: {
        mappings: Object.fromEntries(
          Object.entries(mappingConfig.mappings).map(([key, m]) => [
            key,
            {
              id: m.id,
              name: m.name,
              source: m.source,
              curve: m.curve,
              input_min: m.inputMin,
              input_max: m.inputMax,
              output_min: m.outputMin,
              output_max: m.outputMax,
              enabled: m.enabled,
            },
          ])
        ),
        triggers: {
          on_beat: {
            seed_jump: mappingConfig.triggers.onBeat.seedJump,
            strength_boost: mappingConfig.triggers.onBeat.strengthBoost,
            force_keyframe: mappingConfig.triggers.onBeat.forceKeyframe,
          },
          on_onset: {
            seed_variation: mappingConfig.triggers.onOnset.seedVariation,
            force_keyframe: mappingConfig.triggers.onOnset.forceKeyframe,
          },
          chroma_threshold: mappingConfig.triggers.chromaThreshold,
        },
        preset_name: mappingConfig.presetName,
      },
    };

    this.ws.send(JSON.stringify(msg));
  }

  sendAudioChunk(audioData: ArrayBuffer, sampleRate: number): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const bytes = new Uint8Array(audioData);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = btoa(binary);

    const msg = {
      type: 'audio_chunk',
      audio_data: base64,
      sample_rate: sampleRate,
      timestamp: Date.now() / 1000,
    };
    this.ws.send(JSON.stringify(msg));
  }

  resetLyrics(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.updateState({ lyrics: null });
    this.ws.send(JSON.stringify({ type: 'reset_lyrics' }));
  }

  // === Internal methods ===

  private sendConfig(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    const configMsg = {
      type: 'config',
      config: {
        generation_mode: this._config.generationMode,
        base_prompt: this._config.basePrompt,
        negative_prompt: this._config.negativePrompt,
        model_id: this._config.modelId,
        img2img_strength: this._config.img2imgStrength,
        mapping_preset: this._config.mappingPreset,
        target_fps: this._config.targetFps,
        width: this._config.width,
        height: this._config.height,
        acceleration: this._config.acceleration,
        hyper_sd_steps: this._config.hyperSdSteps,
        use_taesd: this._config.useTaesd,
        use_controlnet: this._config.useControlNet,
        controlnet_pose_weight: this._config.controlNetPoseWeight,
        controlnet_pose_lock: this._config.controlNetPoseLock,
        use_procedural_pose: this._config.useProceduralPose,
        pose_animation_mode: this._config.poseAnimationMode,
        pose_animation_speed: this._config.poseAnimationSpeed,
        pose_animation_intensity: this._config.poseAnimationIntensity,
        procedural_use_txt2img: this._config.proceduralUseTxt2img,
        procedural_fixed_seed: this._config.proceduralFixedSeed,
        procedural_blend_weight: this._config.proceduralBlendWeight,
        use_ip_adapter: this._config.useIPAdapter,
        ip_adapter_scale: this._config.ipAdapterScale,
        temporal_coherence: this._config.temporalCoherence,
        keyframe_interval: this._config.keyframeInterval,
        keyframe_strength: this._config.keyframeStrength,
        pose_framing: this._config.poseFraming,
        enable_lyrics: this._config.enableLyrics,
        lyric_driven_mode: this._config.lyricDrivenMode,
        base_image: this._config.baseImage,
        lock_to_base_image: this._config.lockToBaseImage,
        periodic_pose_refresh: this._config.periodicPoseRefresh,
        loras: this._config.loras.length > 0 ? this._config.loras : null,
      },
    };
    this.ws.send(JSON.stringify(configMsg));
  }

  private handleMessage(event: MessageEvent): void {
    if (event.data instanceof Blob) {
      this.handleBinaryMessage(event.data);
    } else {
      this.handleTextMessage(event.data);
    }
  }

  private handleBinaryMessage(blob: Blob): void {
    const url = URL.createObjectURL(blob);

    // Check if this is a pose preview
    if (this.pendingPosePreview) {
      this.pendingPosePreview = false;

      if (!this.poseImage) {
        this.poseImage = new Image();
      }

      const img = this.poseImage;
      img.onload = () => {
        this.updateState({ posePreview: img });
        setTimeout(() => URL.revokeObjectURL(url), 100);
      };
      img.src = url;
      return;
    }

    // Regular frame data
    const metadata = this.pendingFrameMetadata;
    if (!metadata) {
      console.warn('Received frame data without metadata');
      return;
    }

    if (!this.frameImage) {
      this.frameImage = new Image();
    }

    const img = this.frameImage;
    img.onload = () => {
      this.updateState({
        currentFrame: img,
        frameId: metadata.frame_id,
        lastParams: {
          prompt: metadata.generation_params.prompt,
          strength: metadata.generation_params.strength,
          guidance_scale: metadata.generation_params.guidance_scale,
          seed: metadata.generation_params.seed ?? null,
          is_onset: metadata.generation_params.is_onset ?? false,
          onset_confidence: metadata.generation_params.onset_confidence ?? 0,
          color_keywords: metadata.generation_params.color_keywords ?? [],
        },
        ...(metadata.system_stats ? { systemStats: metadata.system_stats } : {}),
      });

      this.localBus.emit('frameReceived', { frameId: metadata.frame_id, frame: img });
      eventBus.emit('ai:frameReceived', { frameId: metadata.frame_id, frame: img });
      eventBus.emit('ai:frame', { frame: img, frameId: metadata.frame_id });

      setTimeout(() => URL.revokeObjectURL(url), 100);
    };
    img.src = url;

    this.pendingFrameMetadata = null;
  }

  private handleTextMessage(data: string): void {
    try {
      const msg = JSON.parse(data);

      // Handle pose preview marker
      if (msg.type === 'pose_preview') {
        this.pendingPosePreview = true;
        return;
      }

      switch (msg.type) {
        case 'frame':
          this.pendingFrameMetadata = msg;
          break;

        case 'status': {
          const wasSwitching = this._state.isSwitchingBackend;
          const isSwitching = msg.status === 'switching';
          const switchingCleared = wasSwitching && !isSwitching;

          this.updateState({
            isConnected: msg.status !== 'error',
            isGenerating: msg.status === 'generating',
            isInitializing: msg.status === 'initializing',
            isSwitchingBackend: isSwitching,
            pendingBackendId: switchingCleared ? null : this._state.pendingBackendId,
            statusMessage: msg.message ?? null,
            fps: msg.fps ?? this._state.fps,
            error: msg.status === 'error' ? msg.message ?? 'Unknown error' : null,
            serverConfig: msg.server_config ?? this._state.serverConfig,
          });
          break;
        }

        case 'error':
          console.error('AI Generator error:', msg.error);
          this.updateState({ error: msg.error });
          break;

        case 'story_state':
          if (this.storyMessageCallback) {
            this.storyMessageCallback(msg);
          }
          this.localBus.emit('storyMessage', msg);
          eventBus.emit('ai:storyMessage', msg);
          break;

        case 'lyrics':
          if (msg.lyrics) {
            this.updateState({ lyrics: msg.lyrics as LyricInfo });
          }
          break;

        case 'fps':
          this.updateState({ fps: msg.fps ?? this._state.fps });
          break;

        case 'reset_lyrics':
          this.updateState({ lyrics: null });
          break;
      }
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e);
    }
  }

  private updateState(partial: Partial<AIGeneratorState>): void {
    this._state = { ...this._state, ...partial };
    this.localBus.emit('stateChange', this._state);
    eventBus.emit('ai:stateChanged', this._state);
  }

  // === Cleanup ===

  dispose(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.localBus.clear();
  }
}
