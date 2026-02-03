/**
 * Renderer Abstraction Layer Types
 *
 * These interfaces define a backend-agnostic rendering API that supports
 * both WebGL2 (via Three.js) and native WebGPU implementations.
 */

// =============================================================================
// Core Types
// =============================================================================

export type RendererBackendType = 'webgl2' | 'webgpu';

export type UniformValue =
  | number
  | boolean
  | Float32Array
  | Int32Array
  | [number, number]
  | [number, number, number]
  | [number, number, number, number];

export type TextureFormat = 'rgba8' | 'rgba16f' | 'rgba32f' | 'depth24' | 'depth32f';

export type TextureFilter = 'nearest' | 'linear';

export type TextureWrap = 'clamp' | 'repeat' | 'mirror';

export type BlendMode = 'none' | 'alpha' | 'additive' | 'multiply';

export type PrimitiveTopology = 'triangle-list' | 'triangle-strip' | 'line-list' | 'point-list';

// =============================================================================
// Configuration Interfaces
// =============================================================================

export interface TextureConfig {
  width: number;
  height: number;
  format?: TextureFormat;
  minFilter?: TextureFilter;
  magFilter?: TextureFilter;
  wrapS?: TextureWrap;
  wrapT?: TextureWrap;
  generateMipmaps?: boolean;
  /** Initial data (optional) */
  data?: ImageBitmap | HTMLImageElement | ImageData | HTMLCanvasElement | null;
}

export interface RenderTargetConfig {
  width: number;
  height: number;
  format?: TextureFormat;
  depthBuffer?: boolean;
  stencilBuffer?: boolean;
  samples?: number; // MSAA samples (1 = no MSAA)
}

export interface BlendConfig {
  mode: BlendMode;
  srcFactor?: GPUBlendFactor;
  dstFactor?: GPUBlendFactor;
}

export interface UniformDescriptor {
  type: 'float' | 'int' | 'bool' | 'vec2' | 'vec3' | 'vec4' | 'mat3' | 'mat4';
  value?: UniformValue;
}

export interface UniformLayout {
  [name: string]: UniformDescriptor;
}

/**
 * Shader source supporting multiple backends.
 * Provide GLSL for WebGL2 and WGSL for WebGPU.
 */
export interface ShaderSource {
  glsl?: string;
  wgsl?: string;
}

export interface PipelineConfig {
  /** Unique identifier for caching */
  id?: string;
  vertexShader: ShaderSource;
  fragmentShader: ShaderSource;
  uniforms: UniformLayout;
  blend?: BlendConfig;
  topology?: PrimitiveTopology;
  /** Depth testing */
  depthTest?: boolean;
  depthWrite?: boolean;
  /** Face culling */
  cullMode?: 'none' | 'front' | 'back';
}

export interface RendererConfig {
  canvas: HTMLCanvasElement;
  preferredBackend?: RendererBackendType;
  antialias?: boolean;
  alpha?: boolean;
  preserveDrawingBuffer?: boolean;
  powerPreference?: 'default' | 'high-performance' | 'low-power';
  pixelRatio?: number;
}

// =============================================================================
// Resource Interfaces
// =============================================================================

/**
 * Abstract texture handle.
 * Wraps WebGL textures or GPUTexture objects.
 */
export interface ITexture {
  readonly width: number;
  readonly height: number;
  readonly format: TextureFormat;

  /**
   * Update texture contents from an image source.
   */
  update(source: ImageBitmap | HTMLImageElement | ImageData | HTMLCanvasElement): void;

  /**
   * Update texture contents from raw data.
   */
  updateData(data: ArrayBufferView, width: number, height: number): void;

  /**
   * Release GPU resources.
   */
  dispose(): void;

  /**
   * Get the native texture object (WebGLTexture or GPUTexture).
   * Use with caution - breaks abstraction.
   */
  getNativeTexture(): unknown;
}

/**
 * Abstract render target (framebuffer).
 * Can be rendered to and sampled from.
 */
export interface IRenderTarget {
  readonly width: number;
  readonly height: number;
  readonly texture: ITexture;
  readonly hasDepth: boolean;

  /**
   * Resize the render target.
   * Recreates internal resources.
   */
  resize(width: number, height: number): void;

  /**
   * Release GPU resources.
   */
  dispose(): void;
}

/**
 * Abstract render pipeline (shader program + state).
 * Encapsulates vertex/fragment shaders and render state.
 */
export interface IRenderPipeline {
  readonly id: string;

  /**
   * Set a single uniform value.
   */
  setUniform(name: string, value: UniformValue): void;

  /**
   * Set multiple uniforms at once.
   */
  setUniforms(uniforms: Record<string, UniformValue>): void;

  /**
   * Bind a texture to a sampler slot.
   * @param slot - Texture unit index (0-15 typically)
   * @param texture - Texture to bind
   * @param samplerName - Uniform name for the sampler (optional, for WebGPU)
   */
  setTexture(slot: number, texture: ITexture, samplerName?: string): void;

  /**
   * Release GPU resources.
   */
  dispose(): void;

  /**
   * Get the native pipeline object.
   * Use with caution - breaks abstraction.
   */
  getNativePipeline(): unknown;
}

// =============================================================================
// Renderer Interface
// =============================================================================

/**
 * Main renderer interface.
 * Abstracts WebGL2 and WebGPU rendering APIs.
 */
export interface IRenderer {
  readonly backend: RendererBackendType;
  readonly canvas: HTMLCanvasElement;
  readonly width: number;
  readonly height: number;

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Initialize the renderer.
   * Must be called before any other operations.
   * WebGPU requires async initialization.
   */
  initialize(): Promise<void>;

  /**
   * Release all GPU resources.
   */
  dispose(): void;

  // -------------------------------------------------------------------------
  // Frame Management
  // -------------------------------------------------------------------------

  /**
   * Begin a new frame.
   * Call before any rendering operations.
   */
  beginFrame(): void;

  /**
   * End the current frame.
   * Submits all queued commands for WebGPU.
   */
  endFrame(): void;

  /**
   * Clear the current render target.
   */
  clear(color?: [number, number, number, number], depth?: number): void;

  // -------------------------------------------------------------------------
  // Resource Creation
  // -------------------------------------------------------------------------

  /**
   * Create a texture.
   */
  createTexture(config: TextureConfig): ITexture;

  /**
   * Create a render target (framebuffer).
   */
  createRenderTarget(config: RenderTargetConfig): IRenderTarget;

  /**
   * Create a render pipeline (shader program).
   */
  createPipeline(config: PipelineConfig): IRenderPipeline;

  // -------------------------------------------------------------------------
  // Render State
  // -------------------------------------------------------------------------

  /**
   * Set the viewport.
   */
  setViewport(x: number, y: number, width: number, height: number): void;

  /**
   * Set the scissor rectangle.
   */
  setScissor(x: number, y: number, width: number, height: number): void;

  /**
   * Set the current render target.
   * Pass null to render to the canvas.
   */
  setRenderTarget(target: IRenderTarget | null): void;

  // -------------------------------------------------------------------------
  // Drawing
  // -------------------------------------------------------------------------

  /**
   * Draw a fullscreen quad using the given pipeline.
   * Commonly used for post-processing effects.
   */
  drawFullscreenQuad(pipeline: IRenderPipeline): void;

  /**
   * Draw with explicit vertex count.
   */
  draw(pipeline: IRenderPipeline, vertexCount: number, instanceCount?: number): void;

  // -------------------------------------------------------------------------
  // Utility
  // -------------------------------------------------------------------------

  /**
   * Resize the renderer to match canvas size.
   */
  resize(width: number, height: number): void;

  /**
   * Set the pixel ratio.
   */
  setPixelRatio(ratio: number): void;

  /**
   * Get current pixel ratio.
   */
  getPixelRatio(): number;

  /**
   * Set the clear color.
   */
  setClearColor(r: number, g: number, b: number, a?: number): void;

  /**
   * Check if a feature is supported.
   */
  supportsFeature(feature: RendererFeature): boolean;

  /**
   * Get performance metrics.
   */
  getMetrics(): RendererMetrics;
}

// =============================================================================
// Feature Detection
// =============================================================================

export type RendererFeature =
  | 'float-textures'
  | 'half-float-textures'
  | 'depth-textures'
  | 'anisotropic-filtering'
  | 'msaa'
  | 'compute-shaders'
  | 'storage-textures'
  | 'timestamp-queries';

export interface RendererMetrics {
  backend: RendererBackendType;
  frameTime: number;
  gpuTime?: number; // Only available with timestamp queries
  drawCalls: number;
  triangles: number;
  textureMemory: number;
  bufferMemory: number;
}

// =============================================================================
// Convenience Types for Shader Uniforms
// =============================================================================

/**
 * Common uniforms used across audio-reactive shaders.
 */
export interface AudioUniforms {
  // Basic frequency bands
  bass: number;
  mid: number;
  treble: number;
  rms: number;

  // Beat/onset detection
  beatFlash: number;
  onsetPulse: number;
  onsetStrength: number;

  // Spectral features
  spectralCentroid: number;
  dominantHue: number;
  chromaEnergy: number;

  // Effect toggles (as numbers for GPU: 0.0 or 1.0)
  enableDistortion: number;
  enableFlash: number;
  enableSpectralDisplacement: number;
  enableGlitchBlocks: number;
  enableTrebleGrain: number;

  // Effect parameters
  glitchSeed: number;
  glitchIntensity: number;
  smoothedTreble: number;

  // Upscaling
  enableBicubic: number;
  enableSharpening: number;
  sharpenStrength: number;
  textureSize: [number, number];

  // Silence degradation
  silenceLevel: number;
  degradationSeed: number;

  // General
  time: number;
  intensity: number;
  blendFactor: number;
  resolution: [number, number];
}

/**
 * Convert boolean to GPU-friendly number.
 */
export function boolToFloat(value: boolean): number {
  return value ? 1.0 : 0.0;
}

/**
 * Create default audio uniforms.
 */
export function createDefaultAudioUniforms(): AudioUniforms {
  return {
    bass: 0,
    mid: 0,
    treble: 0,
    rms: 0,
    beatFlash: 0,
    onsetPulse: 0,
    onsetStrength: 0,
    spectralCentroid: 0.5,
    dominantHue: 0,
    chromaEnergy: 0,
    enableDistortion: 0,
    enableFlash: 1,
    enableSpectralDisplacement: 0,
    enableGlitchBlocks: 0,
    enableTrebleGrain: 0,
    glitchSeed: 0,
    glitchIntensity: 0,
    smoothedTreble: 0,
    enableBicubic: 1,
    enableSharpening: 1,
    sharpenStrength: 0.5,
    textureSize: [256, 256],
    silenceLevel: 0,
    degradationSeed: 0,
    time: 0,
    intensity: 0.7,
    blendFactor: 1,
    resolution: [1920, 1080],
  };
}
