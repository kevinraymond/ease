/**
 * WebGPU Render Target Implementation
 *
 * Wraps GPUTexture as a render target with the IRenderTarget interface.
 */

import { IRenderTarget, ITexture, RenderTargetConfig, TextureFormat } from '../types';
import { WebGPUTexture } from './WebGPUTexture';

/**
 * Convert our texture format to WebGPU format.
 */
function toGPUTextureFormat(format: TextureFormat): GPUTextureFormat {
  switch (format) {
    case 'rgba8':
      return 'rgba8unorm';
    case 'rgba16f':
      return 'rgba16float';
    case 'rgba32f':
      return 'rgba32float';
    default:
      return 'rgba8unorm';
  }
}

/**
 * WebGPU render target implementation.
 */
export class WebGPURenderTarget implements IRenderTarget {
  private device: GPUDevice;
  private colorTexture: GPUTexture;
  private depthTexture: GPUTexture | null = null;
  private _texture: WebGPUTexture;
  private _width: number;
  private _height: number;
  private _hasDepth: boolean;
  private config: RenderTargetConfig;
  private disposed = false;

  constructor(device: GPUDevice, config: RenderTargetConfig) {
    this.device = device;
    this.config = { ...config };
    this._width = config.width;
    this._height = config.height;
    this._hasDepth = config.depthBuffer ?? true;

    const format = toGPUTextureFormat(config.format ?? 'rgba8');

    // Create color texture
    this.colorTexture = device.createTexture({
      size: { width: config.width, height: config.height },
      format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
      sampleCount: config.samples ?? 1,
    });

    // Create depth texture if needed
    if (this._hasDepth) {
      this.depthTexture = device.createTexture({
        size: { width: config.width, height: config.height },
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        sampleCount: config.samples ?? 1,
      });
    }

    // Create wrapper texture for the color attachment
    this._texture = new WebGPUTexture(device, {
      width: config.width,
      height: config.height,
      format: config.format ?? 'rgba8',
    });

    // Replace the wrapper's internal texture with our render attachment
    (this._texture as unknown as { texture: GPUTexture }).texture = this.colorTexture;
  }

  get width(): number {
    return this._width;
  }

  get height(): number {
    return this._height;
  }

  get texture(): ITexture {
    return this._texture;
  }

  get hasDepth(): boolean {
    return this._hasDepth;
  }

  resize(width: number, height: number): void {
    if (this.disposed) {
      console.warn('Attempting to resize disposed render target');
      return;
    }

    // Destroy old textures
    this.colorTexture.destroy();
    if (this.depthTexture) {
      this.depthTexture.destroy();
    }

    this._width = width;
    this._height = height;

    const format = toGPUTextureFormat(this.config.format ?? 'rgba8');

    // Recreate color texture
    this.colorTexture = this.device.createTexture({
      size: { width, height },
      format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
      sampleCount: this.config.samples ?? 1,
    });

    // Recreate depth texture if needed
    if (this._hasDepth) {
      this.depthTexture = this.device.createTexture({
        size: { width, height },
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        sampleCount: this.config.samples ?? 1,
      });
    }

    // Update wrapper texture reference
    (this._texture as unknown as { texture: GPUTexture }).texture = this.colorTexture;
    (this._texture as unknown as { _width: number })._width = width;
    (this._texture as unknown as { _height: number })._height = height;
  }

  dispose(): void {
    if (!this.disposed) {
      this.colorTexture.destroy();
      if (this.depthTexture) {
        this.depthTexture.destroy();
      }
      this._texture.dispose();
      this.disposed = true;
    }
  }

  /**
   * Get the color texture for creating render pass attachments.
   */
  getColorTexture(): GPUTexture {
    return this.colorTexture;
  }

  /**
   * Get the depth texture for creating render pass attachments.
   */
  getDepthTexture(): GPUTexture | null {
    return this.depthTexture;
  }

  /**
   * Create a color attachment descriptor for render pass.
   */
  createColorAttachment(clearColor?: [number, number, number, number]): GPURenderPassColorAttachment {
    return {
      view: this.colorTexture.createView(),
      loadOp: clearColor ? 'clear' : 'load',
      storeOp: 'store',
      clearValue: clearColor
        ? { r: clearColor[0], g: clearColor[1], b: clearColor[2], a: clearColor[3] }
        : undefined,
    };
  }

  /**
   * Create a depth attachment descriptor for render pass.
   */
  createDepthAttachment(clearDepth = 1.0): GPURenderPassDepthStencilAttachment | undefined {
    if (!this.depthTexture) return undefined;

    return {
      view: this.depthTexture.createView(),
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
      depthClearValue: clearDepth,
    };
  }
}
