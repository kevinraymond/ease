/**
 * VisualizerController - Manages the visualizer engine and rendering.
 * Replaces the useVisualizerEngine React hook.
 */

import { EventBus, eventBus } from '../core/EventBus';
import { VisualizerEngine, ModeRenderer } from '../visualizer/VisualizerEngine';
import {
  VisualizerConfig,
  AccessibilitySettings,
  RendererBackend,
  DEFAULT_VISUALIZER_CONFIG,
} from '../visualizer/types';
import { AudioMetrics } from '../audio/types';

export class VisualizerController {
  // State
  private _config: VisualizerConfig;
  private _backend: RendererBackend;
  private _activeBackend: 'webgpu' | 'webgl2' = 'webgl2';

  // Canvas and engine
  private canvas: HTMLCanvasElement | null = null;
  private engine: VisualizerEngine | null = null;
  private currentEngineBackend: RendererBackend | null = null;
  private initId = 0;

  // Local event bus
  private localBus = new EventBus();
  private disposed = false;

  constructor(initialConfig?: Partial<VisualizerConfig>) {
    this._config = {
      ...DEFAULT_VISUALIZER_CONFIG,
      ...initialConfig,
    };
    this._backend = this._config.backend;
  }

  /**
   * Initialize with a canvas element. Call this when the canvas is ready.
   */
  initialize(canvas: HTMLCanvasElement, accessibility?: AccessibilitySettings): void {
    this.canvas = canvas;
    if (accessibility) {
      this._config = { ...this._config, accessibility };
    }
    this.initializeEngine();
  }

  // === Getters ===

  get config(): VisualizerConfig {
    return this._config;
  }

  get backend(): RendererBackend {
    return this._backend;
  }

  get activeBackend(): 'webgpu' | 'webgl2' {
    return this._activeBackend;
  }

  getCanvas(): HTMLCanvasElement | null {
    return this.canvas;
  }

  // === Event subscription ===

  on(event: 'configChange' | 'backendChange', callback: (data: any) => void): () => void {
    return this.localBus.on(event, callback);
  }

  // === Actions ===

  setBackend(newBackend: RendererBackend): void {
    if (newBackend !== this._backend) {
      console.log(`Backend change requested: ${this._backend} -> ${newBackend}`);
      this._backend = newBackend;
      this._config = { ...this._config, backend: newBackend };
      this.reinitializeEngine();
    }
  }

  setAccessibility(settings: AccessibilitySettings): void {
    this._config = { ...this._config, accessibility: settings };
    this.engine?.setConfig(this._config);
    this.localBus.emit('configChange', this._config);
  }

  setShowFrequencyDebug(show: boolean): void {
    this._config = { ...this._config, showFrequencyDebug: show };
    this.engine?.setConfig(this._config);
    this.localBus.emit('configChange', this._config);
  }

  setConfig(config: Partial<VisualizerConfig>): void {
    this._config = { ...this._config, ...config };
    this.engine?.setConfig(this._config);
    this.localBus.emit('configChange', this._config);
  }

  render(metrics: AudioMetrics): void {
    this.engine?.render(metrics);
  }

  getCurrentMode(): ModeRenderer | null {
    return this.engine?.getCurrentMode() ?? null;
  }

  getRenderer(): unknown {
    return this.engine?.getRenderer() ?? null;
  }

  // FSR methods
  setFSREnabled(enabled: boolean, sourceWidth = 256, sourceHeight = 256): void {
    this.engine?.setFSREnabled(enabled, sourceWidth, sourceHeight);
  }

  setFSRSharpness(sharpness: number): void {
    this.engine?.setFSRSharpness(sharpness);
  }

  isFSREnabled(): boolean {
    return this.engine?.isFSREnabled() ?? false;
  }

  // === Internal methods ===

  private initializeEngine(): void {
    if (this.disposed || !this.canvas) return;

    // Skip if engine already exists with the same backend AND same canvas
    const engineCanvas = this.engine?.getCanvas();
    if (this.engine && this.currentEngineBackend === this._backend && engineCanvas === this.canvas) {
      return;
    }

    // Increment init ID to invalidate pending async callbacks
    const currentInitId = ++this.initId;

    // Dispose existing engine if reinitializing
    if (this.engine) {
      console.log(`Switching backend from ${this.currentEngineBackend} to ${this._backend}`);
      this.engine.dispose();
      this.engine = null;
    }

    // Replace canvas to allow context type change
    if (this.currentEngineBackend !== null) {
      const parent = this.canvas.parentElement;
      if (parent) {
        const newCanvas = document.createElement('canvas');
        newCanvas.className = this.canvas.className;
        newCanvas.style.cssText = this.canvas.style.cssText;
        parent.replaceChild(newCanvas, this.canvas);
        this.canvas = newCanvas;
        console.log('Created new canvas for backend switch');
      }
    }

    // Create engine
    this.currentEngineBackend = this._backend;
    this.engine = new VisualizerEngine(this.canvas, this._config);

    // Wait for async init
    this.engine.waitForInit().then(() => {
      if (this.initId !== currentInitId || this.disposed) return;

      const actualBackend = this.engine!.getBackend();
      this._activeBackend = actualBackend === 'webgpu' ? 'webgpu' : 'webgl2';
      console.log(`Engine initialized with ${actualBackend} backend`);

      this.localBus.emit('backendChange', this._activeBackend);
      eventBus.emit('visualizer:backendChanged', this._activeBackend);
    });
  }

  private reinitializeEngine(): void {
    this.initializeEngine();
  }

  // === Cleanup ===

  dispose(): void {
    this.disposed = true;
    this.engine?.dispose();
    this.engine = null;
    this.localBus.clear();
  }
}
