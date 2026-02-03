/**
 * Renderer Factory
 *
 * Creates the appropriate renderer backend based on browser capabilities
 * and user preferences. Supports automatic fallback from WebGPU to WebGL2.
 */

import { IRenderer, RendererBackendType, RendererConfig } from './types';

// Lazy imports to avoid loading both backends when only one is needed
let WebGLRendererClass: typeof import('./webgl/WebGLRenderer').WebGLRenderer | null = null;
let WebGPURendererClass: typeof import('./webgpu/WebGPURenderer').WebGPURenderer | null = null;

/**
 * Check if WebGPU is available in the current browser.
 */
export async function isWebGPUAvailable(): Promise<boolean> {
  if (!navigator.gpu) {
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

/**
 * Check if WebGL2 is available in the current browser.
 */
export function isWebGL2Available(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    return gl !== null;
  } catch {
    return false;
  }
}

/**
 * Detect the best available renderer backend.
 */
export async function detectBestBackend(): Promise<RendererBackendType> {
  // WebGL2 is the stable default
  if (isWebGL2Available()) {
    return 'webgl2';
  }

  // Fall back to WebGPU only if WebGL2 unavailable
  if (await isWebGPUAvailable()) {
    return 'webgpu';
  }

  // Default to WebGL2 even if detection fails
  // (the actual renderer will handle the error)
  return 'webgl2';
}

/**
 * Create a renderer instance.
 *
 * @param config - Renderer configuration
 * @returns Promise resolving to an initialized renderer
 * @throws Error if no supported backend is available
 */
export async function createRenderer(config: RendererConfig): Promise<IRenderer> {
  const { canvas, preferredBackend } = config;

  // Determine which backend to use
  let backend: RendererBackendType;

  if (preferredBackend === 'webgpu') {
    // User explicitly wants WebGPU
    if (await isWebGPUAvailable()) {
      backend = 'webgpu';
    } else {
      console.warn('WebGPU requested but not available, falling back to WebGL2');
      backend = 'webgl2';
    }
  } else if (preferredBackend === 'webgl2') {
    // User explicitly wants WebGL2
    backend = 'webgl2';
  } else {
    // Auto-detect best backend
    backend = await detectBestBackend();
  }

  console.log(`Creating renderer with backend: ${backend}`);

  // Create the appropriate renderer
  let renderer: IRenderer;

  if (backend === 'webgpu') {
    // Dynamically import WebGPU renderer
    if (!WebGPURendererClass) {
      const module = await import('./webgpu/WebGPURenderer');
      WebGPURendererClass = module.WebGPURenderer;
    }
    renderer = new WebGPURendererClass(canvas, config);
  } else {
    // Dynamically import WebGL renderer
    if (!WebGLRendererClass) {
      const module = await import('./webgl/WebGLRenderer');
      WebGLRendererClass = module.WebGLRenderer;
    }
    renderer = new WebGLRendererClass(canvas, config);
  }

  // Initialize the renderer
  try {
    await renderer.initialize();
    console.log(`${backend} renderer initialized successfully`);
  } catch (error) {
    // If WebGPU fails, try falling back to WebGL2
    if (backend === 'webgpu') {
      console.warn('WebGPU initialization failed, falling back to WebGL2:', error);
      renderer.dispose();

      if (!WebGLRendererClass) {
        const module = await import('./webgl/WebGLRenderer');
        WebGLRendererClass = module.WebGLRenderer;
      }
      renderer = new WebGLRendererClass(canvas, config);
      await renderer.initialize();
      console.log('WebGL2 renderer initialized as fallback');
    } else {
      throw error;
    }
  }

  return renderer;
}

/**
 * Get information about available renderer backends.
 */
export async function getRendererCapabilities(): Promise<{
  webgpu: boolean;
  webgl2: boolean;
  recommended: RendererBackendType;
}> {
  const [webgpu, webgl2] = await Promise.all([
    isWebGPUAvailable(),
    Promise.resolve(isWebGL2Available()),
  ]);

  return {
    webgpu,
    webgl2,
    recommended: 'webgl2',
  };
}

// Re-export types for convenience
export type { IRenderer, RendererConfig, RendererBackendType };
