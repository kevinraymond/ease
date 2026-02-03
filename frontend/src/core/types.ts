/**
 * Shared types for the core framework and controllers.
 */

// Re-export commonly used types from other modules
export type { AudioMetrics, BeatDebugInfo } from '../audio/types';
export type { VisualizerConfig, RendererBackend } from '../visualizer/types';

// Generation modes
export type GenerationMode = 'feedback' | 'keyframe_rife';

// Mapping presets available from backend
export type MappingPreset = 'reactive' | 'dancer' | 'vj_intense' | 'dreamscape' | 'color_organ';

// Temporal coherence options
export type TemporalCoherence = 'none' | 'blending';

// Acceleration method for fast generation
export type AccelerationMethod = 'lcm' | 'hyper-sd' | 'none';

// Lyric detection info from backend
export interface LyricInfo {
  text: string;
  keywords: [string, number][];
  confidence: number;
  is_singing: boolean;
  language: string;
}

// Backend information
export interface BackendInfo {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  fps_range: [number, number];
}

// Server-side configuration (read-only)
export interface ServerConfig {
  acceleration: 'lcm' | 'hyper-sd' | 'none';
  hyper_sd_steps: number | null;
  model: string;
  current_backend: string;
  available_backends: BackendInfo[];
  capabilities: string[];
}

// System resource stats from server
export interface SystemStats {
  cpu_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  gpu_util: number | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
}

// LoRA configuration
export interface LoraConfig {
  path: string;
  weight: number;
  name?: string;
}

// Generation parameters from server
export interface GenerationParams {
  prompt: string;
  strength: number;
  guidance_scale: number;
  seed: number | null;
  is_onset: boolean;
  onset_confidence: number;
  color_keywords: string[];
}

// Configuration for AI generation
export interface AIGeneratorConfig {
  serverUrl: string;
  generationMode: GenerationMode;
  basePrompt: string;
  negativePrompt: string;
  modelId: string;
  img2imgStrength: number;
  mappingPreset: MappingPreset;
  targetFps: number;
  width: number;
  height: number;
  maintainAspectRatio: boolean;
  // Acceleration method
  acceleration: AccelerationMethod;
  hyperSdSteps: 1 | 2 | 4 | 8;
  // SOTA settings
  useTaesd: boolean;
  useControlNet: boolean;
  controlNetPoseWeight: number;
  controlNetPoseLock: boolean;
  useProceduralPose: boolean;
  poseAnimationMode: string;
  poseAnimationSpeed: number;
  poseAnimationIntensity: number;
  proceduralUseTxt2img: boolean;
  proceduralFixedSeed: number | null;
  proceduralBlendWeight: number;
  useIPAdapter: boolean;
  ipAdapterScale: number;
  temporalCoherence: TemporalCoherence;
  keyframeInterval: number;
  keyframeStrength: number;
  poseFraming: 'full_body' | 'upper_body' | 'portrait';
  // Visual effects
  enableShaderEffects: boolean;
  enableFlash: boolean;
  enableSpectralDisplacement: boolean;
  enableGlitchBlocks: boolean;
  enableTrebleGrain: boolean;
  // Upscaling
  enableBicubic: boolean;
  enableSharpening: boolean;
  sharpenStrength: number;
  enableFSR: boolean;
  fsrSharpness: number;
  // Lyric detection
  enableLyrics: boolean;
  lyricDrivenMode: boolean;
  showLyricSubtitles: boolean;
  // Base image
  baseImage: string | null;
  lockToBaseImage: boolean;
  periodicPoseRefresh: boolean;
  // LoRA
  loras: LoraConfig[];
  // Silence degradation
  enableSilenceDegradation: boolean;
  silenceThreshold: number;
  degradationRate: number;
  recoveryRate: number;
}

// Connection and generation state
export interface AIGeneratorState {
  isConnected: boolean;
  isConnecting: boolean;
  isGenerating: boolean;
  isInitializing: boolean;
  isSwitchingBackend: boolean;
  pendingBackendId: string | null;
  statusMessage: string | null;
  fps: number;
  currentFrame: HTMLImageElement | null;
  frameId: number;
  error: string | null;
  lastParams: GenerationParams | null;
  posePreview: HTMLImageElement | null;
  lyrics: LyricInfo | null;
  serverConfig: ServerConfig | null;
  systemStats: SystemStats | null;
}

// Mapping config payload type
export interface MappingConfigPayload {
  mappings: Record<string, {
    id: string;
    name: string;
    source: string;
    curve: string;
    inputMin: number;
    inputMax: number;
    outputMin: number;
    outputMax: number;
    enabled: boolean;
  }>;
  triggers: {
    onBeat: {
      seedJump: boolean;
      strengthBoost: number;
      forceKeyframe: boolean;
    };
    onOnset: {
      seedVariation: number;
      forceKeyframe: boolean;
    };
    chromaThreshold?: number;
  };
  presetName: string;
}

// Story types
export type StoryPresetName =
  | 'skiing_adventure'
  | 'dancing_figure'
  | 'abstract_landscape'
  | 'minimal_portrait';

export type SceneTrigger = 'time' | 'beat_count' | 'energy_drop' | 'energy_peak';

export type SceneTransition = 'cut' | 'crossfade' | 'zoom_in' | 'zoom_out';

export interface SceneDefinition {
  id: string;
  basePrompt: string;
  negativePrompt?: string;
  durationFrames: number;
  trigger: SceneTrigger;
  triggerValue: number;
  energyHighPrompt?: string;
  energyLowPrompt?: string;
  energyBlendRange: [number, number];
  beatPromptModifier?: string;
  transition: SceneTransition;
  transitionFrames: number;
}

export interface StoryConfig {
  name: string;
  description?: string;
  defaultNegativePrompt: string;
  scenes: SceneDefinition[];
  loop: boolean;
  audioReactiveKeywords: boolean;
  baseSeed?: number;
}

export interface StoryState {
  storyName: string;
  currentSceneIdx: number;
  currentSceneId: string;
  frameInScene: number;
  beatCountInScene: number;
  isTransitioning: boolean;
  transitionProgress: number;
  isPlaying: boolean;
  isComplete: boolean;
  totalScenes: number;
  sceneProgress: number;
}
