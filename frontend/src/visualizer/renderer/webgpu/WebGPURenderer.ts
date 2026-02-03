/**
 * WebGPU Renderer Implementation
 *
 * Native WebGPU renderer providing high-performance graphics.
 */

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
import { WebGPUTexture } from './WebGPUTexture';
import { WebGPURenderTarget } from './WebGPURenderTarget';
import { WebGPUPipeline } from './WebGPUPipeline';

/**
 * WebGPU renderer implementation.
 */
export class WebGPURenderer implements IRenderer {
  readonly backend: RendererBackendType = 'webgpu';
  readonly canvas: HTMLCanvasElement;

  private config: RendererConfig;
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private canvasFormat: GPUTextureFormat = 'rgba8unorm';
  private _width: number;
  private _height: number;
  private _pixelRatio: number;
  private clearColor: [number, number, number, number] = [0, 0, 0, 1];

  // Current render state
  private currentRenderTarget: WebGPURenderTarget | null = null;
  private commandEncoder: GPUCommandEncoder | null = null;
  private renderPassEncoder: GPURenderPassEncoder | null = null;

  // Default resources
  private defaultTexture: WebGPUTexture | null = null;

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
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }

    // Request adapter
    const powerPref = this.config.powerPreference === 'default' ? undefined : this.config.powerPreference;
    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: powerPref ?? 'high-performance',
    });

    if (!this.adapter) {
      throw new Error('Failed to get GPU adapter');
    }

    // Request device
    this.device = await this.adapter.requestDevice({
      requiredFeatures: [],
      requiredLimits: {},
    });

    // Handle device loss
    this.device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message);
    });

    // Configure canvas context
    this.context = this.canvas.getContext('webgpu');
    if (!this.context) {
      throw new Error('Failed to get WebGPU context');
    }

    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: this.config.alpha ? 'premultiplied' : 'opaque',
    });

    // Create default 1x1 white texture for missing texture bindings
    this.defaultTexture = new WebGPUTexture(this.device, {
      width: 1,
      height: 1,
      format: 'rgba8',
    });
    const whitePixel = new Uint8Array([255, 255, 255, 255]);
    this.defaultTexture.updateData(whitePixel, 1, 1);

    // Set initial size
    this.resize(this._width, this._height);

    console.log('WebGPU renderer initialized');
    console.log('  Adapter:', this.adapter.info);
    console.log('  Format:', this.canvasFormat);
  }

  dispose(): void {
    this.defaultTexture?.dispose();
    this.device?.destroy();
    this.device = null;
    this.adapter = null;
    this.context = null;
  }

  // -------------------------------------------------------------------------
  // Frame Management
  // -------------------------------------------------------------------------

  beginFrame(): void {
    if (!this.device) return;

    this.frameStartTime = performance.now();
    this.drawCallCount = 0;

    // Create command encoder for this frame
    this.commandEncoder = this.device.createCommandEncoder();
  }

  endFrame(): void {
    if (!this.device || !this.commandEncoder) return;

    // End any active render pass
    this.endRenderPass();

    // Submit commands
    this.device.queue.submit([this.commandEncoder.finish()]);
    this.commandEncoder = null;

    this.lastFrameTime = performance.now() - this.frameStartTime;
  }

  private beginRenderPass(clearColor?: [number, number, number, number]): void {
    if (!this.device || !this.commandEncoder || !this.context) return;

    // End previous pass if any
    this.endRenderPass();

    let colorAttachment: GPURenderPassColorAttachment;
    let depthAttachment: GPURenderPassDepthStencilAttachment | undefined;

    if (this.currentRenderTarget) {
      colorAttachment = this.currentRenderTarget.createColorAttachment(clearColor);
      depthAttachment = this.currentRenderTarget.createDepthAttachment();
    } else {
      // Render to canvas
      const view = this.context.getCurrentTexture().createView();
      colorAttachment = {
        view,
        loadOp: clearColor ? 'clear' : 'load',
        storeOp: 'store',
        clearValue: clearColor
          ? { r: clearColor[0], g: clearColor[1], b: clearColor[2], a: clearColor[3] }
          : undefined,
      };
    }

    this.renderPassEncoder = this.commandEncoder.beginRenderPass({
      colorAttachments: [colorAttachment],
      depthStencilAttachment: depthAttachment,
    });
  }

  private endRenderPass(): void {
    if (this.renderPassEncoder) {
      this.renderPassEncoder.end();
      this.renderPassEncoder = null;
    }
  }

  clear(color?: [number, number, number, number], _depth?: number): void {
    const clearColor = color ?? this.clearColor;
    this.beginRenderPass(clearColor);
    // The pass itself clears; we just need to end it if we're only clearing
  }

  // -------------------------------------------------------------------------
  // Resource Creation
  // -------------------------------------------------------------------------

  createTexture(config: TextureConfig): ITexture {
    if (!this.device) {
      throw new Error('Device not initialized');
    }
    return new WebGPUTexture(this.device, config);
  }

  createRenderTarget(config: RenderTargetConfig): IRenderTarget {
    if (!this.device) {
      throw new Error('Device not initialized');
    }
    return new WebGPURenderTarget(this.device, config);
  }

  createPipeline(config: PipelineConfig): IRenderPipeline {
    if (!this.device) {
      throw new Error('Device not initialized');
    }
    return new WebGPUPipeline(this.device, config, this.canvasFormat);
  }

  // -------------------------------------------------------------------------
  // Render State
  // -------------------------------------------------------------------------

  setViewport(x: number, y: number, width: number, height: number): void {
    if (this.renderPassEncoder) {
      this.renderPassEncoder.setViewport(x, y, width, height, 0, 1);
    }
  }

  setScissor(x: number, y: number, width: number, height: number): void {
    if (this.renderPassEncoder) {
      this.renderPassEncoder.setScissorRect(x, y, width, height);
    }
  }

  setRenderTarget(target: IRenderTarget | null): void {
    // End current pass to switch targets
    this.endRenderPass();

    if (target === null) {
      this.currentRenderTarget = null;
    } else if (target instanceof WebGPURenderTarget) {
      this.currentRenderTarget = target;
    }
  }

  // -------------------------------------------------------------------------
  // Drawing
  // -------------------------------------------------------------------------

  drawFullscreenQuad(pipeline: IRenderPipeline): void {
    if (!this.device || !this.commandEncoder || !this.defaultTexture) return;

    // Ensure render pass is active
    if (!this.renderPassEncoder) {
      this.beginRenderPass();
    }

    if (!this.renderPassEncoder) return;

    const gpuPipeline = pipeline as WebGPUPipeline;

    // Set pipeline and bind group
    this.renderPassEncoder.setPipeline(gpuPipeline.getGPUPipeline());
    this.renderPassEncoder.setBindGroup(0, gpuPipeline.getBindGroup(this.defaultTexture));

    // Draw fullscreen triangle (6 vertices for 2 triangles)
    this.renderPassEncoder.draw(6);
    this.drawCallCount++;
  }

  draw(pipeline: IRenderPipeline, vertexCount: number, instanceCount = 1): void {
    if (!this.device || !this.commandEncoder || !this.defaultTexture) return;

    // Ensure render pass is active
    if (!this.renderPassEncoder) {
      this.beginRenderPass();
    }

    if (!this.renderPassEncoder) return;

    const gpuPipeline = pipeline as WebGPUPipeline;

    // Set pipeline and bind group
    this.renderPassEncoder.setPipeline(gpuPipeline.getGPUPipeline());
    this.renderPassEncoder.setBindGroup(0, gpuPipeline.getBindGroup(this.defaultTexture));

    // Draw
    this.renderPassEncoder.draw(vertexCount, instanceCount);
    this.drawCallCount++;
  }

  // -------------------------------------------------------------------------
  // Utility
  // -------------------------------------------------------------------------

  resize(width: number, height: number): void {
    this._width = width;
    this._height = height;

    // Update canvas size
    const pixelWidth = Math.floor(width * this._pixelRatio);
    const pixelHeight = Math.floor(height * this._pixelRatio);
    this.canvas.width = pixelWidth;
    this.canvas.height = pixelHeight;

    // Reconfigure context
    if (this.device && this.context) {
      this.context.configure({
        device: this.device,
        format: this.canvasFormat,
        alphaMode: this.config.alpha ? 'premultiplied' : 'opaque',
      });
    }
  }

  setPixelRatio(ratio: number): void {
    this._pixelRatio = ratio;
    this.resize(this._width, this._height);
  }

  getPixelRatio(): number {
    return this._pixelRatio;
  }

  setClearColor(r: number, g: number, b: number, a = 1): void {
    this.clearColor = [r, g, b, a];
  }

  supportsFeature(feature: RendererFeature): boolean {
    if (!this.device) return false;

    switch (feature) {
      case 'float-textures':
        return true;
      case 'half-float-textures':
        return true;
      case 'depth-textures':
        return true;
      case 'anisotropic-filtering':
        return true;
      case 'msaa':
        return true;
      case 'compute-shaders':
        return true;
      case 'storage-textures':
        return true;
      case 'timestamp-queries':
        return this.device.features.has('timestamp-query');
      default:
        return false;
    }
  }

  getMetrics(): RendererMetrics {
    return {
      backend: this.backend,
      frameTime: this.lastFrameTime,
      drawCalls: this.drawCallCount,
      triangles: this.drawCallCount * 2, // Approximate for fullscreen quads
      textureMemory: 0, // WebGPU doesn't expose this directly
      bufferMemory: 0,
    };
  }

  // -------------------------------------------------------------------------
  // WebGPU-specific Methods
  // -------------------------------------------------------------------------

  /**
   * Get the underlying GPU device.
   * Use with caution - breaks abstraction.
   */
  getDevice(): GPUDevice | null {
    return this.device;
  }

  /**
   * Get the canvas context.
   */
  getContext(): GPUCanvasContext | null {
    return this.context;
  }

  /**
   * Get the preferred canvas format.
   */
  getCanvasFormat(): GPUTextureFormat {
    return this.canvasFormat;
  }

  /**
   * Get the current command encoder for advanced usage.
   */
  getCommandEncoder(): GPUCommandEncoder | null {
    return this.commandEncoder;
  }
}
