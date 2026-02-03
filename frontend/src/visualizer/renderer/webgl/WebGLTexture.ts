/**
 * WebGL2 Texture Implementation
 *
 * Wraps Three.js textures with the ITexture interface.
 */

import * as THREE from 'three';
import { ITexture, TextureConfig, TextureFormat, TextureFilter, TextureWrap } from '../types';

/**
 * Convert our texture format to Three.js format.
 */
function toThreeFormat(format: TextureFormat): THREE.PixelFormat {
  switch (format) {
    case 'rgba8':
    case 'rgba16f':
    case 'rgba32f':
      return THREE.RGBAFormat;
    case 'depth24':
    case 'depth32f':
      return THREE.DepthFormat;
    default:
      return THREE.RGBAFormat;
  }
}

/**
 * Convert our texture format to Three.js type.
 */
function toThreeType(format: TextureFormat): THREE.TextureDataType {
  switch (format) {
    case 'rgba8':
      return THREE.UnsignedByteType;
    case 'rgba16f':
      return THREE.HalfFloatType;
    case 'rgba32f':
      return THREE.FloatType;
    case 'depth24':
      return THREE.UnsignedIntType;
    case 'depth32f':
      return THREE.FloatType;
    default:
      return THREE.UnsignedByteType;
  }
}

/**
 * Convert our filter mode to Three.js filter.
 */
function toThreeFilter(filter: TextureFilter): THREE.TextureFilter {
  return filter === 'nearest' ? THREE.NearestFilter : THREE.LinearFilter;
}

/**
 * Convert our wrap mode to Three.js wrapping.
 */
function toThreeWrap(wrap: TextureWrap): THREE.Wrapping {
  switch (wrap) {
    case 'clamp':
      return THREE.ClampToEdgeWrapping;
    case 'repeat':
      return THREE.RepeatWrapping;
    case 'mirror':
      return THREE.MirroredRepeatWrapping;
    default:
      return THREE.ClampToEdgeWrapping;
  }
}

/**
 * WebGL2 texture implementation using Three.js.
 */
export class WebGLTexture implements ITexture {
  private texture: THREE.Texture | THREE.DataTexture;
  private _width: number;
  private _height: number;
  private _format: TextureFormat;
  private disposed = false;

  constructor(config: TextureConfig) {
    this._width = config.width;
    this._height = config.height;
    this._format = config.format ?? 'rgba8';

    if (config.data) {
      // Create texture from image source
      if (config.data instanceof ImageData) {
        this.texture = new THREE.DataTexture(
          config.data.data,
          config.width,
          config.height,
          toThreeFormat(this._format),
          toThreeType(this._format)
        );
      } else {
        // HTMLImageElement, ImageBitmap, or HTMLCanvasElement
        this.texture = new THREE.Texture(config.data as HTMLImageElement);
      }
    } else {
      // Create empty data texture
      const size = config.width * config.height * 4;
      const data = new Uint8Array(size);
      this.texture = new THREE.DataTexture(
        data,
        config.width,
        config.height,
        toThreeFormat(this._format),
        toThreeType(this._format)
      );
    }

    // Apply texture settings
    this.texture.minFilter = toThreeFilter(config.minFilter ?? 'linear');
    this.texture.magFilter = toThreeFilter(config.magFilter ?? 'linear') as THREE.MagnificationTextureFilter;
    this.texture.wrapS = toThreeWrap(config.wrapS ?? 'clamp');
    this.texture.wrapT = toThreeWrap(config.wrapT ?? 'clamp');
    this.texture.generateMipmaps = config.generateMipmaps ?? false;
    this.texture.needsUpdate = true;
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

  update(source: ImageBitmap | HTMLImageElement | ImageData | HTMLCanvasElement): void {
    if (this.disposed) {
      console.warn('Attempting to update disposed texture');
      return;
    }

    if (source instanceof ImageData) {
      // For ImageData, we need to create a new DataTexture or update existing
      if (this.texture instanceof THREE.DataTexture) {
        this.texture.image.data.set(source.data);
        this._width = source.width;
        this._height = source.height;
        this.texture.image.width = source.width;
        this.texture.image.height = source.height;
      } else {
        // Convert to canvas first
        const canvas = document.createElement('canvas');
        canvas.width = source.width;
        canvas.height = source.height;
        const ctx = canvas.getContext('2d')!;
        ctx.putImageData(source, 0, 0);
        this.texture.image = canvas;
        this._width = source.width;
        this._height = source.height;
      }
    } else {
      // HTMLImageElement, ImageBitmap, or HTMLCanvasElement
      this.texture.image = source;
      if ('width' in source) {
        this._width = source.width;
        this._height = source.height;
      }
    }

    this.texture.needsUpdate = true;
  }

  updateData(data: ArrayBufferView, width: number, height: number): void {
    if (this.disposed) {
      console.warn('Attempting to update disposed texture');
      return;
    }

    if (this.texture instanceof THREE.DataTexture) {
      // Update the data in place if sizes match
      if (width === this._width && height === this._height) {
        if (data instanceof Uint8Array) {
          (this.texture.image.data as Uint8Array).set(data);
        }
      } else {
        // Need to resize - create new texture data
        this._width = width;
        this._height = height;
        this.texture.image = {
          data: data instanceof Uint8Array ? data : new Uint8Array(data.buffer),
          width,
          height,
        };
      }
      this.texture.needsUpdate = true;
    }
  }

  dispose(): void {
    if (!this.disposed) {
      this.texture.dispose();
      this.disposed = true;
    }
  }

  getNativeTexture(): THREE.Texture {
    return this.texture;
  }

  /**
   * Get the underlying Three.js texture for direct manipulation.
   * Internal use only.
   */
  getThreeTexture(): THREE.Texture {
    return this.texture;
  }
}

/**
 * Create a WebGLTexture from an existing Three.js texture.
 * Useful for wrapping textures created elsewhere.
 */
export function wrapThreeTexture(texture: THREE.Texture): WebGLTexture {
  const config: TextureConfig = {
    width: texture.image?.width ?? 256,
    height: texture.image?.height ?? 256,
    format: 'rgba8',
  };

  const wrapper = new WebGLTexture(config);
  // Replace internal texture with the provided one
  (wrapper as unknown as { texture: THREE.Texture }).texture = texture;
  return wrapper;
}
