/**
 * WebGPU Render Pipeline Implementation
 *
 * Wraps GPURenderPipeline with the IRenderPipeline interface.
 */

import {
  IRenderPipeline,
  ITexture,
  PipelineConfig,
  UniformValue,
  UniformDescriptor,
  BlendMode,
} from '../types';
import { WebGPUTexture } from './WebGPUTexture';

/**
 * Convert our blend mode to WebGPU blend state.
 */
function toGPUBlendState(mode: BlendMode): GPUBlendState | undefined {
  switch (mode) {
    case 'none':
      return undefined;
    case 'alpha':
      return {
        color: {
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
          operation: 'add',
        },
        alpha: {
          srcFactor: 'one',
          dstFactor: 'one-minus-src-alpha',
          operation: 'add',
        },
      };
    case 'additive':
      return {
        color: {
          srcFactor: 'src-alpha',
          dstFactor: 'one',
          operation: 'add',
        },
        alpha: {
          srcFactor: 'one',
          dstFactor: 'one',
          operation: 'add',
        },
      };
    case 'multiply':
      return {
        color: {
          srcFactor: 'dst' as GPUBlendFactor,
          dstFactor: 'zero',
          operation: 'add',
        },
        alpha: {
          srcFactor: 'dst-alpha',
          dstFactor: 'zero',
          operation: 'add',
        },
      };
    default:
      return undefined;
  }
}

/**
 * Calculate the byte size of a uniform type.
 */
function getUniformSize(type: UniformDescriptor['type']): number {
  switch (type) {
    case 'float':
    case 'int':
    case 'bool':
      return 4;
    case 'vec2':
      return 8;
    case 'vec3':
      return 12; // Note: WGSL aligns vec3 to 16 bytes
    case 'vec4':
      return 16;
    case 'mat3':
      return 48; // 3 * vec4 (padded)
    case 'mat4':
      return 64;
    default:
      return 4;
  }
}

/**
 * Calculate aligned offset for uniform.
 */
function alignOffset(offset: number, alignment: number): number {
  return Math.ceil(offset / alignment) * alignment;
}

/**
 * Information about a uniform in the buffer.
 */
interface UniformInfo {
  type: UniformDescriptor['type'];
  offset: number;
  size: number;
}

/**
 * WebGPU render pipeline implementation.
 */
export class WebGPUPipeline implements IRenderPipeline {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private uniformBuffer: GPUBuffer;
  private uniformData: Float32Array;
  private uniformInfos: Map<string, UniformInfo>;
  private _bindGroup: GPUBindGroup | null = null;
  private bindGroupLayout: GPUBindGroupLayout;
  private textureBindings: Map<number, { texture: WebGPUTexture; samplerName?: string }>;
  private _id: string;
  private disposed = false;
  private needsBindGroupUpdate = true;

  constructor(device: GPUDevice, config: PipelineConfig, colorFormat: GPUTextureFormat = 'rgba8unorm') {
    this.device = device;
    this._id = config.id ?? `pipeline-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    this.uniformInfos = new Map();
    this.textureBindings = new Map();

    // Validate shader source
    if (!config.vertexShader.wgsl) {
      throw new Error('WebGPU pipeline requires WGSL vertex shader');
    }
    if (!config.fragmentShader.wgsl) {
      throw new Error('WebGPU pipeline requires WGSL fragment shader');
    }

    // Calculate uniform buffer layout
    let offset = 0;
    for (const [name, descriptor] of Object.entries(config.uniforms)) {
      const size = getUniformSize(descriptor.type);
      const alignment = size >= 16 ? 16 : size >= 8 ? 8 : 4;
      offset = alignOffset(offset, alignment);

      this.uniformInfos.set(name, {
        type: descriptor.type,
        offset,
        size,
      });

      offset += size;
    }

    // Align total size to 16 bytes (required for uniform buffers)
    const bufferSize = alignOffset(offset, 16);
    this.uniformData = new Float32Array(bufferSize / 4);

    // Create uniform buffer
    this.uniformBuffer = device.createBuffer({
      size: Math.max(bufferSize, 16), // Minimum 16 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Initialize uniform values
    for (const [name, descriptor] of Object.entries(config.uniforms)) {
      if (descriptor.value !== undefined) {
        this.setUniform(name, descriptor.value);
      }
    }

    // Create bind group layout
    // Entry 0 is always the uniform buffer
    // Entries 1+ are texture/sampler pairs
    const bindGroupEntries: GPUBindGroupLayoutEntry[] = [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ];

    // Add texture/sampler entries (up to 8 texture slots)
    for (let i = 0; i < 8; i++) {
      // Sampler
      bindGroupEntries.push({
        binding: 1 + i * 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      });
      // Texture
      bindGroupEntries.push({
        binding: 2 + i * 2,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {},
      });
    }

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: bindGroupEntries,
    });

    // Create shader modules
    const vertexModule = device.createShaderModule({
      code: config.vertexShader.wgsl,
    });

    const fragmentModule = device.createShaderModule({
      code: config.fragmentShader.wgsl,
    });

    // Create pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    // Create render pipeline
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: vertexModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: fragmentModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: colorFormat,
            blend: toGPUBlendState(config.blend?.mode ?? 'none'),
          },
        ],
      },
      primitive: {
        topology: config.topology ?? 'triangle-list',
        cullMode: config.cullMode ?? 'none',
      },
      depthStencil: config.depthTest
        ? {
            format: 'depth24plus',
            depthWriteEnabled: config.depthWrite ?? false,
            depthCompare: 'less',
          }
        : undefined,
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

    const info = this.uniformInfos.get(name);
    if (!info) {
      // Silently ignore unknown uniforms
      return;
    }

    const offset = info.offset / 4; // Convert to float index

    if (typeof value === 'number') {
      this.uniformData[offset] = value;
    } else if (typeof value === 'boolean') {
      this.uniformData[offset] = value ? 1.0 : 0.0;
    } else if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) {
        this.uniformData[offset + i] = value[i];
      }
    } else if (value instanceof Float32Array || value instanceof Int32Array) {
      this.uniformData.set(value, offset);
    }
  }

  setUniforms(uniforms: Record<string, UniformValue>): void {
    for (const [name, value] of Object.entries(uniforms)) {
      this.setUniform(name, value);
    }
  }

  setTexture(slot: number, texture: ITexture, samplerName?: string): void {
    if (this.disposed) {
      console.warn('Attempting to set texture on disposed pipeline');
      return;
    }

    if (texture instanceof WebGPUTexture) {
      this.textureBindings.set(slot, { texture, samplerName });
      this.needsBindGroupUpdate = true;
    }
  }

  /**
   * Update the uniform buffer with current data.
   */
  updateUniformBuffer(): void {
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData.buffer);
  }

  /**
   * Get or create the bind group for rendering.
   */
  getBindGroup(defaultTexture: WebGPUTexture): GPUBindGroup {
    // Always update uniforms
    this.updateUniformBuffer();

    // Recreate bind group if textures changed
    if (this.needsBindGroupUpdate || !this._bindGroup) {
      const entries: GPUBindGroupEntry[] = [
        {
          binding: 0,
          resource: { buffer: this.uniformBuffer },
        },
      ];

      // Add texture/sampler entries
      for (let i = 0; i < 8; i++) {
        const binding = this.textureBindings.get(i);
        const tex = binding?.texture ?? defaultTexture;

        // Sampler
        entries.push({
          binding: 1 + i * 2,
          resource: tex.sampler,
        });

        // Texture view
        entries.push({
          binding: 2 + i * 2,
          resource: tex.createView(),
        });
      }

      this._bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries,
      });

      this.needsBindGroupUpdate = false;
    }

    return this._bindGroup;
  }

  dispose(): void {
    if (!this.disposed) {
      this.uniformBuffer.destroy();
      this.disposed = true;
    }
  }

  getNativePipeline(): GPURenderPipeline {
    return this.pipeline;
  }

  /**
   * Get the GPU render pipeline for encoding.
   */
  getGPUPipeline(): GPURenderPipeline {
    return this.pipeline;
  }
}
