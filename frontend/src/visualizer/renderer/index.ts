/**
 * Renderer Abstraction Layer
 *
 * Provides a unified rendering API supporting both WebGL2 and WebGPU backends.
 *
 * Usage:
 * ```typescript
 * import { createRenderer, IRenderer } from './renderer';
 *
 * const renderer = await createRenderer({
 *   canvas: myCanvas,
 *   preferredBackend: 'auto', // or 'webgl2' or 'webgpu'
 * });
 *
 * // Create resources
 * const texture = renderer.createTexture({ width: 256, height: 256 });
 * const pipeline = renderer.createPipeline({ ... });
 *
 * // Render loop
 * renderer.beginFrame();
 * renderer.clear([0, 0, 0, 1]);
 * renderer.drawFullscreenQuad(pipeline);
 * renderer.endFrame();
 * ```
 */

// Core types
export * from './types';

// Factory
export {
  createRenderer,
  isWebGPUAvailable,
  isWebGL2Available,
  detectBestBackend,
  getRendererCapabilities,
} from './RendererFactory';

// WebGL2 backend (can be imported directly if needed)
export { WebGLRenderer } from './webgl/WebGLRenderer';
export { WebGLTexture, wrapThreeTexture } from './webgl/WebGLTexture';
export { WebGLRenderTarget, wrapThreeRenderTarget } from './webgl/WebGLRenderTarget';
export { WebGLPipeline } from './webgl/WebGLPipeline';

// WebGPU backend (can be imported directly if needed)
export { WebGPURenderer } from './webgpu/WebGPURenderer';
export { WebGPUTexture } from './webgpu/WebGPUTexture';
export { WebGPURenderTarget } from './webgpu/WebGPURenderTarget';
export { WebGPUPipeline } from './webgpu/WebGPUPipeline';
