import { AudioMetrics } from '../audio/types';

export type VisualizationMode = 'ai';

export type RendererBackend = 'auto' | 'webgpu' | 'webgl2';

export interface VisualizerConfig {
  mode: VisualizationMode;
  backend: RendererBackend;
  intensity: number; // 0-1
  colorPalette: ColorPalette;
  accessibility: AccessibilitySettings;
  showFrequencyDebug: boolean;
  modeSettings: ModeSettings;
}

export interface ColorPalette {
  name: string;
  primary: string;
  secondary: string;
  tertiary: string;
  background: string;
  accent: string;
}

export interface AccessibilitySettings {
  maxFlashRate: number; // Hz, default 3
  reducedMotion: boolean;
  intensityCap: number; // 0-1, default 1
}

export const DEFAULT_PALETTES: ColorPalette[] = [
  {
    name: 'Cyber',
    primary: '#00ffff',
    secondary: '#ff00ff',
    tertiary: '#ffff00',
    background: '#0a0a1a',
    accent: '#ffffff',
  },
  {
    name: 'Sunset',
    primary: '#ff6b6b',
    secondary: '#feca57',
    tertiary: '#ff9ff3',
    background: '#1a0a0a',
    accent: '#ffffff',
  },
  {
    name: 'Forest',
    primary: '#00d26a',
    secondary: '#00b894',
    tertiary: '#55efc4',
    background: '#0a1a0a',
    accent: '#ffffff',
  },
  {
    name: 'Ocean',
    primary: '#0984e3',
    secondary: '#00cec9',
    tertiary: '#74b9ff',
    background: '#0a0a1a',
    accent: '#ffffff',
  },
  {
    name: 'Monochrome',
    primary: '#ffffff',
    secondary: '#aaaaaa',
    tertiary: '#666666',
    background: '#000000',
    accent: '#ffffff',
  },
];

export const DEFAULT_ACCESSIBILITY: AccessibilitySettings = {
  maxFlashRate: 3,
  reducedMotion: false,
  intensityCap: 1,
};

// Empty interface since AI mode has its own panel
export interface ModeSettings {}

export const DEFAULT_MODE_SETTINGS: ModeSettings = {};

export const DEFAULT_VISUALIZER_CONFIG: VisualizerConfig = {
  mode: 'ai',
  backend: 'auto',  // Auto-detect best backend (WebGPU if available, fallback to WebGL2)
  intensity: 0.7,
  colorPalette: DEFAULT_PALETTES[0],
  accessibility: DEFAULT_ACCESSIBILITY,
  showFrequencyDebug: false,
  modeSettings: DEFAULT_MODE_SETTINGS,
};

export interface VisualizationModeInterface {
  name: VisualizationMode;
  init(canvas: HTMLCanvasElement, config: VisualizerConfig): void;
  render(metrics: AudioMetrics, time: number, deltaTime: number): void;
  resize(width: number, height: number): void;
  setConfig(config: Partial<VisualizerConfig>): void;
  dispose(): void;
}
