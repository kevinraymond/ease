/**
 * WebGL2 Render Target Implementation
 *
 * Wraps Three.js WebGLRenderTarget with the IRenderTarget interface.
 */

import * as THREE from 'three';
import { IRenderTarget, ITexture, RenderTargetConfig, TextureFormat } from '../types';
import { WebGLTexture, wrapThreeTexture } from './WebGLTexture';

/**
 * Convert our texture format to Three.js format and type.
 */
function toThreeRenderTargetConfig(format: TextureFormat): {
  format: THREE.PixelFormat;
  type: THREE.TextureDataType;
} {
  switch (format) {
    case 'rgba8':
      return { format: THREE.RGBAFormat, type: THREE.UnsignedByteType };
    case 'rgba16f':
      return { format: THREE.RGBAFormat, type: THREE.HalfFloatType };
    case 'rgba32f':
      return { format: THREE.RGBAFormat, type: THREE.FloatType };
    default:
      return { format: THREE.RGBAFormat, type: THREE.UnsignedByteType };
  }
}

/**
 * WebGL2 render target implementation using Three.js.
 */
export class WebGLRenderTarget implements IRenderTarget {
  private target: THREE.WebGLRenderTarget;
  private _texture: WebGLTexture;
  private disposed = false;

  constructor(config: RenderTargetConfig) {
    const { format, type } = toThreeRenderTargetConfig(config.format ?? 'rgba8');

    this.target = new THREE.WebGLRenderTarget(config.width, config.height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format,
      type,
      depthBuffer: config.depthBuffer ?? true,
      stencilBuffer: config.stencilBuffer ?? false,
      samples: config.samples ?? 1,
    });

    // Wrap the render target's texture
    this._texture = wrapThreeTexture(this.target.texture);
  }

  get width(): number {
    return this.target.width;
  }

  get height(): number {
    return this.target.height;
  }

  get texture(): ITexture {
    return this._texture;
  }

  get hasDepth(): boolean {
    return this.target.depthBuffer;
  }

  resize(width: number, height: number): void {
    if (this.disposed) {
      console.warn('Attempting to resize disposed render target');
      return;
    }

    this.target.setSize(width, height);

    // Update wrapped texture reference
    this._texture = wrapThreeTexture(this.target.texture);
  }

  dispose(): void {
    if (!this.disposed) {
      this._texture.dispose();
      this.target.dispose();
      this.disposed = true;
    }
  }

  /**
   * Get the underlying Three.js render target.
   * Internal use only.
   */
  getThreeRenderTarget(): THREE.WebGLRenderTarget {
    return this.target;
  }
}

/**
 * Create a WebGLRenderTarget from an existing Three.js render target.
 */
export function wrapThreeRenderTarget(target: THREE.WebGLRenderTarget): WebGLRenderTarget {
  const config: RenderTargetConfig = {
    width: target.width,
    height: target.height,
    depthBuffer: target.depthBuffer,
    stencilBuffer: target.stencilBuffer,
  };

  const wrapper = new WebGLRenderTarget(config);
  // Replace internal target with the provided one
  (wrapper as unknown as { target: THREE.WebGLRenderTarget }).target = target;
  (wrapper as unknown as { _texture: WebGLTexture })._texture = wrapThreeTexture(target.texture);
  return wrapper;
}
