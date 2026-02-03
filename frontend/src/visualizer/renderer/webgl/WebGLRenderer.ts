/**
 * WebGL2 Renderer Implementation
 *
 * Wraps Three.js WebGLRenderer with the IRenderer interface.
 * Maintains full compatibility with existing Three.js-based code.
 */

import * as THREE from 'three';
import {
  IRenderer,
  ITexture,
  IRenderTarget,
  IRenderPipeline,
  RendererBackendType,
  RendererConfig,
  TextureConfig,
  RenderTargetConfig,
  PipelineConfig,
  RendererFeature,
  RendererMetrics,
} from '../types';
import { WebGLTexture } from './WebGLTexture';
import { WebGLRenderTarget } from './WebGLRenderTarget';
import { WebGLPipeline } from './WebGLPipeline';

/**
 * WebGL2 renderer implementation using Three.js.
 */
export class WebGLRenderer implements IRenderer {
  readonly backend: RendererBackendType = 'webgl2';
  readonly canvas: HTMLCanvasElement;

  private renderer: THREE.WebGLRenderer | null = null;
  private config: RendererConfig;
  private _width: number;
  private _height: number;
  private _pixelRatio: number;
  private clearColor: THREE.Color;
  private clearAlpha: number;

  // For fullscreen quad rendering
  private fullscreenScene: THREE.Scene | null = null;
  private fullscreenCamera: THREE.OrthographicCamera | null = null;
  private fullscreenQuad: THREE.Mesh | null = null;

  // Metrics
  private frameStartTime = 0;
  private lastFrameTime = 0;
  private drawCallCount = 0;


  constructor(canvas: HTMLCanvasElement, config: RendererConfig) {
    this.canvas = canvas;
    this.config = config;
    this._width = canvas.clientWidth || 800;
    this._height = canvas.clientHeight || 600;
    this._pixelRatio = config.pixelRatio ?? Math.min(window.devicePixelRatio, 2);
    this.clearColor = new THREE.Color(0x000000);
    this.clearAlpha = config.alpha ? 0 : 1;
  }

  get width(): number {
    return this._width;
  }

  get height(): number {
    return this._height;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  async initialize(): Promise<void> {
    // Create Three.js renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: this.config.antialias ?? true,
      alpha: this.config.alpha ?? true,
      preserveDrawingBuffer: this.config.preserveDrawingBuffer ?? true,
      powerPreference: this.config.powerPreference ?? 'high-performance',
    });

    this.renderer.setSize(this._width, this._height);
    this.renderer.setPixelRatio(this._pixelRatio);
    this.renderer.setClearColor(this.clearColor, this.clearAlpha);

    // Verify WebGL2 context
    const gl = this.renderer.getContext();
    if (!(gl instanceof WebGL2RenderingContext)) {
      throw new Error('WebGL2 not available');
    }

    // Set up fullscreen quad for post-processing
    this.setupFullscreenQuad();

    console.log('WebGL2 renderer initialized');
  }

  private setupFullscreenQuad(): void {
    this.fullscreenScene = new THREE.Scene();
    this.fullscreenCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    const geometry = new THREE.PlaneGeometry(2, 2);
    this.fullscreenQuad = new THREE.Mesh(geometry);
    this.fullscreenScene.add(this.fullscreenQuad);
  }

  dispose(): void {
    if (this.fullscreenQuad) {
      this.fullscreenQuad.geometry.dispose();
      if (this.fullscreenQuad.material instanceof THREE.Material) {
        this.fullscreenQuad.material.dispose();
      }
    }

    this.renderer?.dispose();
    this.renderer = null;
    this.fullscreenScene = null;
    this.fullscreenCamera = null;
    this.fullscreenQuad = null;
  }

  // -------------------------------------------------------------------------
  // Frame Management
  // -------------------------------------------------------------------------

  beginFrame(): void {
    this.frameStartTime = performance.now();
    this.drawCallCount = 0;
  }

  endFrame(): void {
    this.lastFrameTime = performance.now() - this.frameStartTime;
  }

  clear(color?: [number, number, number, number], depth?: number): void {
    if (!this.renderer) return;

    if (color) {
      this.renderer.setClearColor(
        new THREE.Color(color[0], color[1], color[2]),
        color[3]
      );
    }

    let clearFlags = true; // Clear color
    if (depth !== undefined) {
      this.renderer.clearDepth();
    }

    this.renderer.clear(clearFlags, depth !== undefined, false);
  }

  // -------------------------------------------------------------------------
  // Resource Creation
  // -------------------------------------------------------------------------

  createTexture(config: TextureConfig): ITexture {
    return new WebGLTexture(config);
  }

  createRenderTarget(config: RenderTargetConfig): IRenderTarget {
    return new WebGLRenderTarget(config);
  }

  createPipeline(config: PipelineConfig): IRenderPipeline {
    return new WebGLPipeline(config);
  }

  // -------------------------------------------------------------------------
  // Render State
  // -------------------------------------------------------------------------

  setViewport(x: number, y: number, width: number, height: number): void {
    this.renderer?.setViewport(x, y, width, height);
  }

  setScissor(x: number, y: number, width: number, height: number): void {
    if (this.renderer) {
      this.renderer.setScissorTest(true);
      this.renderer.setScissor(x, y, width, height);
    }
  }

  setRenderTarget(target: IRenderTarget | null): void {
    if (!this.renderer) return;

    if (target === null) {
      this.renderer.setRenderTarget(null);
    } else if (target instanceof WebGLRenderTarget) {
      this.renderer.setRenderTarget(target.getThreeRenderTarget());
    }
  }

  // -------------------------------------------------------------------------
  // Drawing
  // -------------------------------------------------------------------------

  drawFullscreenQuad(pipeline: IRenderPipeline): void {
    if (!this.renderer || !this.fullscreenScene || !this.fullscreenCamera || !this.fullscreenQuad) {
      return;
    }

    // Get the material from the pipeline
    const material = (pipeline as WebGLPipeline).getMaterial();
    this.fullscreenQuad.material = material;

    // Render
    this.renderer.render(this.fullscreenScene, this.fullscreenCamera);
    this.drawCallCount++;
  }

  draw(pipeline: IRenderPipeline, vertexCount: number, _instanceCount?: number): void {
    // For simple vertex count drawing, we use the fullscreen quad approach
    // More complex geometry would require additional setup
    if (vertexCount === 6 || vertexCount === 4) {
      this.drawFullscreenQuad(pipeline);
    }
    this.drawCallCount++;
  }

  // -------------------------------------------------------------------------
  // Utility
  // -------------------------------------------------------------------------

  resize(width: number, height: number): void {
    this._width = width;
    this._height = height;
    this.renderer?.setSize(width, height);
  }

  setPixelRatio(ratio: number): void {
    this._pixelRatio = ratio;
    this.renderer?.setPixelRatio(ratio);
  }

  getPixelRatio(): number {
    return this._pixelRatio;
  }

  setClearColor(r: number, g: number, b: number, a = 1): void {
    this.clearColor.setRGB(r, g, b);
    this.clearAlpha = a;
    this.renderer?.setClearColor(this.clearColor, a);
  }

  supportsFeature(feature: RendererFeature): boolean {
    if (!this.renderer) return false;

    const gl = this.renderer.getContext();

    switch (feature) {
      case 'float-textures':
        return gl.getExtension('OES_texture_float') !== null;
      case 'half-float-textures':
        return gl.getExtension('EXT_color_buffer_half_float') !== null;
      case 'depth-textures':
        return true; // WebGL2 supports depth textures
      case 'anisotropic-filtering':
        return gl.getExtension('EXT_texture_filter_anisotropic') !== null;
      case 'msaa':
        return true; // WebGL2 supports MSAA
      case 'compute-shaders':
        return false; // WebGL2 does not support compute shaders
      case 'storage-textures':
        return false; // WebGL2 does not support storage textures
      case 'timestamp-queries':
        return gl.getExtension('EXT_disjoint_timer_query_webgl2') !== null;
      default:
        return false;
    }
  }

  getMetrics(): RendererMetrics {
    const info = this.renderer?.info;

    return {
      backend: this.backend,
      frameTime: this.lastFrameTime,
      drawCalls: this.drawCallCount,
      triangles: info?.render?.triangles ?? 0,
      textureMemory: info?.memory?.textures ?? 0,
      bufferMemory: info?.memory?.geometries ?? 0,
    };
  }

  // -------------------------------------------------------------------------
  // Three.js-specific Methods (for compatibility)
  // -------------------------------------------------------------------------

  /**
   * Get the underlying Three.js renderer.
   * Use with caution - breaks abstraction but needed for compatibility.
   */
  getThreeRenderer(): THREE.WebGLRenderer | null {
    return this.renderer;
  }

  /**
   * Render a Three.js scene with a camera.
   * Compatibility method for existing code.
   */
  renderScene(scene: THREE.Scene, camera: THREE.Camera): void {
    if (this.renderer) {
      this.renderer.render(scene, camera);
      this.drawCallCount++;
    }
  }

  /**
   * Get the fullscreen quad scene for advanced usage.
   */
  getFullscreenScene(): THREE.Scene | null {
    return this.fullscreenScene;
  }

  /**
   * Get the fullscreen camera for advanced usage.
   */
  getFullscreenCamera(): THREE.OrthographicCamera | null {
    return this.fullscreenCamera;
  }
}
