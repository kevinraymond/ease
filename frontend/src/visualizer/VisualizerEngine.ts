import * as THREE from 'three';
import { AudioMetrics } from '../audio/types';
import {
  VisualizerConfig,
  DEFAULT_VISUALIZER_CONFIG,
  RendererBackend,
} from './types';
import { AIGeneratedMode } from './modes/AIGeneratedMode';
import { AIGeneratedModeWebGPU } from './modes/AIGeneratedModeWebGPU';
import { FSRUpscaler } from './shaders/FSR';
import {
  IRenderer,
  RendererBackendType,
  createRenderer,
  getRendererCapabilities,
} from './renderer';
import { WebGLRenderer as AbstractWebGLRenderer } from './renderer/webgl/WebGLRenderer';

export type ModeRenderer = {
  render(metrics: AudioMetrics, time: number, deltaTime: number): void;
  resize(width: number, height: number): void;
  setConfig(config: Partial<VisualizerConfig>): void;
  dispose(): void;
  /** Optional: return mode-specific camera (e.g., orthographic for 2D modes) */
  getCamera?(): THREE.Camera;
};

export class VisualizerEngine {
  private canvas: HTMLCanvasElement;
  // Legacy Three.js renderer (for backward compatibility)
  private renderer: THREE.WebGLRenderer | null = null;
  // New abstracted renderer
  private abstractRenderer: IRenderer | null = null;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
  private config: VisualizerConfig;
  private currentMode: ModeRenderer | null = null;
  private animationId: number | null = null;
  private lastTime = 0;
  private backend: RendererBackend = 'webgl2';
  private initPromise: Promise<void> | null = null;
  private initialized = false;

  // Flash rate limiting
  private lastFlashTime = 0;
  private flashCount = 0;

  // FSR upscaling
  private fsrUpscaler: FSRUpscaler | null = null;
  private fsrEnabled = false;
  private fsrRenderTarget: THREE.WebGLRenderTarget | null = null;
  private fsrSourceWidth = 256;
  private fsrSourceHeight = 256;

  constructor(canvas: HTMLCanvasElement, config?: Partial<VisualizerConfig>) {
    this.canvas = canvas;
    this.config = { ...DEFAULT_VISUALIZER_CONFIG, ...config };
    this.scene = new THREE.Scene();

    // Initialize camera (will be adjusted per mode)
    this.camera = new THREE.PerspectiveCamera(
      75,
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      1000
    );
    this.camera.position.z = 5;

    // Start async initialization
    this.initPromise = this.initRenderer();
  }

  /**
   * Wait for the renderer to be fully initialized.
   * Call this before using the engine if you need to ensure initialization is complete.
   */
  public async waitForInit(): Promise<void> {
    if (this.initPromise) {
      await this.initPromise;
    }
  }

  private async initRenderer(): Promise<void> {
    try {
      // Get renderer capabilities
      const capabilities = await getRendererCapabilities();
      console.log('Renderer capabilities:', capabilities);

      // Determine preferred backend from config
      // 'auto' or undefined triggers auto-detection (prefers WebGPU if available)
      const preferredBackend: RendererBackendType | undefined =
        this.config.backend === 'webgpu' ? 'webgpu' :
        this.config.backend === 'webgl2' ? 'webgl2' :
        undefined; // 'auto' triggers auto-detect

      console.log(`Backend config: ${this.config.backend}, resolved to: ${preferredBackend ?? 'auto-detect'}`);
      console.log(`Recommended: ${capabilities.recommended}`);

      // Create the abstract renderer
      this.abstractRenderer = await createRenderer({
        canvas: this.canvas,
        preferredBackend,
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
        powerPreference: 'high-performance',
        pixelRatio: Math.min(window.devicePixelRatio, 2),
      });

      this.backend = this.abstractRenderer.backend as RendererBackend;

      // For WebGL2 backend, extract the Three.js renderer for compatibility
      if (this.abstractRenderer instanceof AbstractWebGLRenderer) {
        this.renderer = this.abstractRenderer.getThreeRenderer();
        if (this.renderer) {
          this.renderer.setClearColor(this.config.colorPalette.background);
        }
      }

      // Handle resize
      window.addEventListener('resize', this.handleResize);

      // Initialize the mode
      this.initMode();
      this.initialized = true;

      console.log(`VisualizerEngine initialized with ${this.backend} backend`);
    } catch (error) {
      console.error('Failed to initialize renderer:', error);
      // Fallback to legacy initialization
      await this.initLegacyRenderer();
    }
  }

  /**
   * Legacy renderer initialization (fallback)
   */
  private async initLegacyRenderer(): Promise<void> {
    // Check for WebGPU support
    if (this.config.backend === 'webgpu' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.backend = 'webgpu';
          console.log('WebGPU initialized (legacy path)');
          // Three.js WebGPU renderer would go here when fully supported
          // For now, fall back to WebGL2
        }
      } catch (e) {
        console.log('WebGPU not available, falling back to WebGL2');
      }
    }

    // Use WebGL2 (current implementation)
    this.backend = 'webgl2';
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });

    this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(this.config.colorPalette.background);

    // Handle resize
    window.addEventListener('resize', this.handleResize);

    // Initialize the mode
    this.initMode();
    this.initialized = true;
  }

  private handleResize = (): void => {
    const parent = this.canvas.parentElement;
    if (!parent) return;

    const width = parent.clientWidth;
    const height = parent.clientHeight;

    if (this.camera instanceof THREE.PerspectiveCamera) {
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }

    // Resize abstract renderer if available
    if (this.abstractRenderer) {
      this.abstractRenderer.resize(width, height);
    } else if (this.renderer) {
      this.renderer.setSize(width, height);
    }

    this.currentMode?.resize(width, height);

    // Update FSR output dimensions if enabled
    if (this.fsrEnabled && this.fsrUpscaler) {
      this.fsrUpscaler.resize({
        inputWidth: this.fsrSourceWidth,
        inputHeight: this.fsrSourceHeight,
        outputWidth: width,
        outputHeight: height,
      });
    }
  };

  private initMode(): void {
    this.currentMode?.dispose();

    // Select mode based on backend
    if (this.backend === 'webgpu' && this.abstractRenderer) {
      // Use WebGPU-native mode
      this.currentMode = new AIGeneratedModeWebGPU(this.abstractRenderer, this.config);
      console.log('Using AIGeneratedModeWebGPU for native WebGPU rendering');
    } else {
      // Use Three.js-based mode for WebGL2
      this.currentMode = new AIGeneratedMode(this.scene, this.camera, this.config);
      console.log('Using AIGeneratedMode for WebGL2 rendering');
    }

    const { width, height } = this.canvas.getBoundingClientRect();
    this.currentMode.resize(width, height);
  }

  private renderCount = 0;
  public render(metrics: AudioMetrics): void {
    // Skip if not initialized
    if (!this.initialized) {
      // DEBUG: log if skipping due to not initialized
      this.renderCount++;
      if (this.renderCount % 300 === 0) {
        console.log(`[VisualizerEngine] Skipping render - not initialized (count=${this.renderCount})`);
      }
      return;
    }

    const now = performance.now();
    const deltaTime = (now - this.lastTime) / 1000;
    this.lastTime = now;

    // Apply accessibility constraints
    const processedMetrics = this.applyAccessibilityConstraints(metrics, now);

    // Render current mode
    this.currentMode?.render(processedMetrics, now / 1000, deltaTime);

    // Render scene (with optional FSR post-processing)
    // Note: AIGeneratedMode requires Three.js WebGLRenderer for scene rendering
    if (this.renderer) {
      // Use mode's camera if available, otherwise use engine's camera
      const renderCamera = this.currentMode?.getCamera?.() ?? this.camera;

      if (this.fsrEnabled && this.fsrUpscaler && this.fsrRenderTarget) {
        // Render to intermediate target at source resolution
        this.renderer.setRenderTarget(this.fsrRenderTarget);
        this.renderer.setViewport(0, 0, this.fsrSourceWidth, this.fsrSourceHeight);
        this.renderer.render(this.scene, renderCamera);

        // Apply FSR upscaling to screen
        this.fsrUpscaler.upscale(this.fsrRenderTarget.texture, null);

        // Reset viewport to full canvas size
        this.renderer.setViewport(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
      } else {
        // Direct render to screen
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, renderCamera);
      }
    }
  }

  private applyAccessibilityConstraints(metrics: AudioMetrics, now: number): AudioMetrics {
    const { accessibility, intensity } = this.config;

    // Apply intensity cap
    const cappedIntensity = intensity * accessibility.intensityCap;

    // Flash rate limiting
    if (metrics.isBeat) {
      const timeSinceFlash = now - this.lastFlashTime;
      const minFlashInterval = 1000 / accessibility.maxFlashRate;

      if (timeSinceFlash < minFlashInterval) {
        // Suppress this flash
        metrics = { ...metrics, isBeat: false };
      } else {
        this.lastFlashTime = now;
        this.flashCount++;
      }
    }

    // Apply reduced motion if enabled
    if (accessibility.reducedMotion) {
      return {
        ...metrics,
        isBeat: false,
        bass: metrics.bass * 0.5,
        mid: metrics.mid * 0.5,
        treble: metrics.treble * 0.5,
      };
    }

    // Scale metrics by intensity
    return {
      ...metrics,
      bass: metrics.bass * cappedIntensity,
      mid: metrics.mid * cappedIntensity,
      treble: metrics.treble * cappedIntensity,
      rms: metrics.rms * cappedIntensity,
      peak: metrics.peak * cappedIntensity,
    };
  }

  public setConfig(config: Partial<VisualizerConfig>): void {
    this.config = { ...this.config, ...config };

    if (config.colorPalette?.background) {
      if (this.abstractRenderer) {
        // Parse hex color to RGB
        const hex = config.colorPalette.background.replace('#', '');
        const r = parseInt(hex.substring(0, 2), 16) / 255;
        const g = parseInt(hex.substring(2, 4), 16) / 255;
        const b = parseInt(hex.substring(4, 6), 16) / 255;
        this.abstractRenderer.setClearColor(r, g, b);
      } else if (this.renderer) {
        this.renderer.setClearColor(config.colorPalette.background);
      }
    }

    this.currentMode?.setConfig(this.config);
  }

  public getConfig(): VisualizerConfig {
    return { ...this.config };
  }

  public getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  public getBackend(): RendererBackend {
    return this.backend;
  }

  /**
   * Get the Three.js renderer (for backward compatibility).
   * Returns null if using a non-WebGL backend.
   */
  public getRenderer(): THREE.WebGLRenderer | null {
    return this.renderer;
  }

  /**
   * Get the abstract renderer interface.
   */
  public getAbstractRenderer(): IRenderer | null {
    return this.abstractRenderer;
  }

  /**
   * Check if the engine has finished initializing.
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  public getCurrentMode(): ModeRenderer | null {
    return this.currentMode;
  }

  /**
   * Get performance metrics from the renderer.
   */
  public getPerformanceMetrics(): {
    backend: string;
    frameTime?: number;
    drawCalls?: number;
  } {
    if (this.abstractRenderer) {
      const metrics = this.abstractRenderer.getMetrics();
      return {
        backend: metrics.backend,
        frameTime: metrics.frameTime,
        drawCalls: metrics.drawCalls,
      };
    }
    return { backend: this.backend };
  }

  // === FSR Upscaling ===

  /**
   * Enable or disable FSR upscaling for AI mode.
   * FSR renders the scene at source resolution then upscales to display.
   */
  public setFSREnabled(enabled: boolean, sourceWidth = 256, sourceHeight = 256): void {
    this.fsrEnabled = enabled;
    this.fsrSourceWidth = sourceWidth;
    this.fsrSourceHeight = sourceHeight;

    if (enabled && this.renderer) {
      this.initFSR();
      console.log(`FSR enabled: ${sourceWidth}x${sourceHeight} -> ${this.canvas.clientWidth}x${this.canvas.clientHeight}`);
    } else if (!enabled) {
      this.disposeFSR();
      console.log('FSR disabled');
    }
  }

  /**
   * Set FSR sharpness (0.0 - 2.0)
   */
  public setFSRSharpness(sharpness: number): void {
    if (this.fsrUpscaler) {
      this.fsrUpscaler.setSharpness(sharpness);
    }
  }

  /**
   * Update FSR source dimensions (AI frame size)
   */
  public setFSRSourceSize(width: number, height: number): void {
    this.fsrSourceWidth = width;
    this.fsrSourceHeight = height;
    if (this.fsrEnabled && this.renderer) {
      this.initFSR();
    }
  }

  /**
   * Check if FSR is currently active
   */
  public isFSREnabled(): boolean {
    return this.fsrEnabled && this.fsrUpscaler !== null;
  }

  private initFSR(): void {
    if (!this.renderer) return;

    // Clean up existing
    this.disposeFSR();

    const displayWidth = this.canvas.clientWidth;
    const displayHeight = this.canvas.clientHeight;

    // Create render target for source resolution
    this.fsrRenderTarget = new THREE.WebGLRenderTarget(
      this.fsrSourceWidth,
      this.fsrSourceHeight,
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
      }
    );

    // Create FSR upscaler with noticeable sharpness
    this.fsrUpscaler = new FSRUpscaler(this.renderer, {
      inputWidth: this.fsrSourceWidth,
      inputHeight: this.fsrSourceHeight,
      outputWidth: displayWidth,
      outputHeight: displayHeight,
      sharpness: 0.8, // Higher default for visible effect
    });
  }

  private disposeFSR(): void {
    if (this.fsrUpscaler) {
      this.fsrUpscaler.dispose();
      this.fsrUpscaler = null;
    }
    if (this.fsrRenderTarget) {
      this.fsrRenderTarget.dispose();
      this.fsrRenderTarget = null;
    }
  }

  public dispose(): void {
    window.removeEventListener('resize', this.handleResize);

    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }

    this.currentMode?.dispose();
    this.disposeFSR();

    // Dispose abstract renderer
    if (this.abstractRenderer) {
      this.abstractRenderer.dispose();
      this.abstractRenderer = null;
    }

    // Dispose legacy renderer
    if (this.renderer && !this.abstractRenderer) {
      this.renderer.dispose();
    }
    this.renderer = null;

    // Clear scene
    while (this.scene.children.length > 0) {
      const obj = this.scene.children[0];
      this.scene.remove(obj);
    }
  }
}
