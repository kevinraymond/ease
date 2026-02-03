/**
 * WebGPU Texture Implementation
 *
 * Wraps GPUTexture with the ITexture interface.
 */

import { ITexture, TextureConfig, TextureFormat, TextureFilter, TextureWrap } from '../types';

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
    case 'depth24':
      return 'depth24plus';
    case 'depth32f':
      return 'depth32float';
    default:
      return 'rgba8unorm';
  }
}

/**
 * Convert our filter mode to WebGPU filter.
 */
function toGPUFilter(filter: TextureFilter): GPUFilterMode {
  return filter === 'nearest' ? 'nearest' : 'linear';
}

/**
 * Convert our wrap mode to WebGPU address mode.
 */
function toGPUAddressMode(wrap: TextureWrap): GPUAddressMode {
  switch (wrap) {
    case 'clamp':
      return 'clamp-to-edge';
    case 'repeat':
      return 'repeat';
    case 'mirror':
      return 'mirror-repeat';
    default:
      return 'clamp-to-edge';
  }
}

/**
 * WebGPU texture implementation.
 */
export class WebGPUTexture implements ITexture {
  private device: GPUDevice;
  private texture: GPUTexture;
  private _sampler: GPUSampler;
  private _width: number;
  private _height: number;
  private _format: TextureFormat;
  private gpuFormat: GPUTextureFormat;
  private disposed = false;

  constructor(device: GPUDevice, config: TextureConfig) {
    this.device = device;
    this._width = config.width;
    this._height = config.height;
    this._format = config.format ?? 'rgba8';
    this.gpuFormat = toGPUTextureFormat(this._format);

    // Create texture
    this.texture = device.createTexture({
      size: { width: config.width, height: config.height },
      format: this.gpuFormat,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      mipLevelCount: config.generateMipmaps ? Math.floor(Math.log2(Math.max(config.width, config.height))) + 1 : 1,
    });

    // Create sampler
    this._sampler = device.createSampler({
      minFilter: toGPUFilter(config.minFilter ?? 'linear'),
      magFilter: toGPUFilter(config.magFilter ?? 'linear'),
      addressModeU: toGPUAddressMode(config.wrapS ?? 'clamp'),
      addressModeV: toGPUAddressMode(config.wrapT ?? 'clamp'),
      mipmapFilter: config.generateMipmaps ? 'linear' : 'nearest',
    });

    // Upload initial data if provided
    if (config.data) {
      this.update(config.data);
    }
  }

  get width(): number {
    return this._width;
  }

  get height(): number {
    return this._height;
  }

  get format(): TextureFormat {
    return this._format;
  }

  get sampler(): GPUSampler {
    return this._sampler;
  }

  update(source: ImageBitmap | HTMLImageElement | ImageData | HTMLCanvasElement): void {
    if (this.disposed) {
      console.warn('Attempting to update disposed texture');
      return;
    }

    // For ImageData, we need to use copyExternalImageToTexture with a canvas
    if (source instanceof ImageData) {
      const canvas = new OffscreenCanvas(source.width, source.height);
      const ctx = canvas.getContext('2d')!;
      ctx.putImageData(source, 0, 0);
      source = canvas as unknown as HTMLCanvasElement;
    }

    // Resize if necessary
    if (source.width !== this._width || source.height !== this._height) {
      this.resize(source.width, source.height);
    }

    // Copy image to texture
    this.device.queue.copyExternalImageToTexture(
      { source: source as ImageBitmap },
      { texture: this.texture },
      { width: this._width, height: this._height }
    );
  }

  updateData(data: ArrayBufferView, width: number, height: number): void {
    if (this.disposed) {
      console.warn('Attempting to update disposed texture');
      return;
    }

    // Resize if necessary
    if (width !== this._width || height !== this._height) {
      this.resize(width, height);
    }

    // Determine bytes per pixel based on format
    let bytesPerPixel = 4; // RGBA8
    if (this._format === 'rgba16f') bytesPerPixel = 8;
    if (this._format === 'rgba32f') bytesPerPixel = 16;

    // Write data to texture
    this.device.queue.writeTexture(
      { texture: this.texture },
      data instanceof ArrayBuffer ? data : data.buffer,
      {
        bytesPerRow: width * bytesPerPixel,
        rowsPerImage: height,
      },
      { width, height }
    );
  }

  private resize(width: number, height: number): void {
    // Destroy old texture
    this.texture.destroy();

    // Create new texture with new size
    this._width = width;
    this._height = height;
    this.texture = this.device.createTexture({
      size: { width, height },
      format: this.gpuFormat,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  dispose(): void {
    if (!this.disposed) {
      this.texture.destroy();
      this.disposed = true;
    }
  }

  getNativeTexture(): GPUTexture {
    return this.texture;
  }

  /**
   * Get a texture view for binding.
   */
  createView(): GPUTextureView {
    return this.texture.createView();
  }
}
