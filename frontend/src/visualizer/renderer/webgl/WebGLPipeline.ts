/**
 * WebGL2 Render Pipeline Implementation
 *
 * Wraps Three.js ShaderMaterial with the IRenderPipeline interface.
 */

import * as THREE from 'three';
import {
  IRenderPipeline,
  ITexture,
  PipelineConfig,
  UniformValue,
  UniformDescriptor,
  BlendMode,
} from '../types';
import { WebGLTexture } from './WebGLTexture';

/**
 * Convert our blend mode to Three.js blending settings.
 */
function toThreeBlending(mode: BlendMode): {
  blending: THREE.Blending;
  transparent: boolean;
} {
  switch (mode) {
    case 'none':
      return { blending: THREE.NoBlending, transparent: false };
    case 'alpha':
      return { blending: THREE.NormalBlending, transparent: true };
    case 'additive':
      return { blending: THREE.AdditiveBlending, transparent: true };
    case 'multiply':
      return { blending: THREE.MultiplyBlending, transparent: true };
    default:
      return { blending: THREE.NormalBlending, transparent: false };
  }
}

/**
 * Convert uniform descriptor to Three.js uniform value.
 */
function createThreeUniform(descriptor: UniformDescriptor): THREE.IUniform {
  const value = descriptor.value;

  switch (descriptor.type) {
    case 'float':
    case 'int':
      return { value: typeof value === 'number' ? value : 0 };
    case 'bool':
      return { value: typeof value === 'boolean' ? value : false };
    case 'vec2':
      if (Array.isArray(value) && value.length === 2) {
        return { value: new THREE.Vector2(value[0], value[1]) };
      }
      return { value: new THREE.Vector2(0, 0) };
    case 'vec3':
      if (Array.isArray(value) && value.length === 3) {
        return { value: new THREE.Vector3(value[0], value[1], value[2]) };
      }
      return { value: new THREE.Vector3(0, 0, 0) };
    case 'vec4':
      if (Array.isArray(value) && value.length === 4) {
        return { value: new THREE.Vector4(value[0], value[1], value[2], value[3]) };
      }
      return { value: new THREE.Vector4(0, 0, 0, 0) };
    case 'mat3':
      return { value: new THREE.Matrix3() };
    case 'mat4':
      return { value: new THREE.Matrix4() };
    default:
      return { value: 0 };
  }
}

/**
 * WebGL2 render pipeline implementation using Three.js ShaderMaterial.
 */
export class WebGLPipeline implements IRenderPipeline {
  private material: THREE.ShaderMaterial;
  private _id: string;
  private uniformTypes: Map<string, UniformDescriptor['type']>;
  private disposed = false;

  constructor(config: PipelineConfig) {
    this._id = config.id ?? `pipeline-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    this.uniformTypes = new Map();

    // Validate shader source
    if (!config.vertexShader.glsl) {
      throw new Error('WebGL pipeline requires GLSL vertex shader');
    }
    if (!config.fragmentShader.glsl) {
      throw new Error('WebGL pipeline requires GLSL fragment shader');
    }

    // Convert uniforms to Three.js format
    const threeUniforms: { [key: string]: THREE.IUniform } = {};
    for (const [name, descriptor] of Object.entries(config.uniforms)) {
      threeUniforms[name] = createThreeUniform(descriptor);
      this.uniformTypes.set(name, descriptor.type);
    }

    // Get blending settings
    const { blending, transparent } = toThreeBlending(config.blend?.mode ?? 'none');

    // Create shader material
    this.material = new THREE.ShaderMaterial({
      vertexShader: config.vertexShader.glsl,
      fragmentShader: config.fragmentShader.glsl,
      uniforms: threeUniforms,
      blending,
      transparent,
      depthTest: config.depthTest ?? false,
      depthWrite: config.depthWrite ?? false,
      side: config.cullMode === 'front' ? THREE.BackSide :
            config.cullMode === 'back' ? THREE.FrontSide :
            THREE.DoubleSide,
    });
  }

  get id(): string {
    return this._id;
  }

  setUniform(name: string, value: UniformValue): void {
    if (this.disposed) {
      console.warn('Attempting to set uniform on disposed pipeline');
      return;
    }

    const uniform = this.material.uniforms[name];
    if (!uniform) {
      // Silently ignore unknown uniforms - they may be conditionally used
      return;
    }

    const type = this.uniformTypes.get(name);

    // Convert value based on type
    if (type === 'vec2' && Array.isArray(value) && value.length === 2) {
      if (uniform.value instanceof THREE.Vector2) {
        uniform.value.set(value[0], value[1]);
      } else {
        uniform.value = new THREE.Vector2(value[0], value[1]);
      }
    } else if (type === 'vec3' && Array.isArray(value) && value.length === 3) {
      if (uniform.value instanceof THREE.Vector3) {
        uniform.value.set(value[0], value[1], value[2]);
      } else {
        uniform.value = new THREE.Vector3(value[0], value[1], value[2]);
      }
    } else if (type === 'vec4' && Array.isArray(value) && value.length === 4) {
      if (uniform.value instanceof THREE.Vector4) {
        uniform.value.set(value[0], value[1], value[2], value[3]);
      } else {
        uniform.value = new THREE.Vector4(value[0], value[1], value[2], value[3]);
      }
    } else if (value instanceof Float32Array || value instanceof Int32Array) {
      // Handle typed arrays (for matrices or arrays)
      if (type === 'mat3' && value.length === 9) {
        if (uniform.value instanceof THREE.Matrix3) {
          uniform.value.fromArray(Array.from(value));
        }
      } else if (type === 'mat4' && value.length === 16) {
        if (uniform.value instanceof THREE.Matrix4) {
          uniform.value.fromArray(Array.from(value));
        }
      } else {
        uniform.value = value;
      }
    } else {
      uniform.value = value;
    }
  }

  setUniforms(uniforms: Record<string, UniformValue>): void {
    for (const [name, value] of Object.entries(uniforms)) {
      this.setUniform(name, value);
    }
  }

  setTexture(_slot: number, texture: ITexture, samplerName?: string): void {
    if (this.disposed) {
      console.warn('Attempting to set texture on disposed pipeline');
      return;
    }

    // Get the native Three.js texture
    let threeTexture: THREE.Texture;
    if (texture instanceof WebGLTexture) {
      threeTexture = texture.getThreeTexture();
    } else {
      threeTexture = (texture as { getNativeTexture(): THREE.Texture }).getNativeTexture() as THREE.Texture;
    }

    // Find the uniform for this texture slot
    // In Three.js, we use the sampler name directly
    if (samplerName && this.material.uniforms[samplerName]) {
      this.material.uniforms[samplerName].value = threeTexture;
    }
  }

  dispose(): void {
    if (!this.disposed) {
      this.material.dispose();
      this.disposed = true;
    }
  }

  getNativePipeline(): THREE.ShaderMaterial {
    return this.material;
  }

  /**
   * Get the underlying Three.js material for direct manipulation.
   * Internal use only.
   */
  getMaterial(): THREE.ShaderMaterial {
    return this.material;
  }

  /**
   * Add a texture uniform to the material.
   * Used when textures are set by slot number.
   */
  addTextureUniform(name: string): void {
    if (!this.material.uniforms[name]) {
      this.material.uniforms[name] = { value: null };
      this.uniformTypes.set(name, 'float'); // Sampler types don't need special handling
    }
  }
}
