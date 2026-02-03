/**
 * AI Generated Mode (WebGPU Native)
 *
 * WebGPU-native implementation of the AI-generated visualization mode.
 * Uses the abstract renderer interface for GPU-agnostic rendering.
 *
 * Features:
 * - Fullscreen display of AI frames as texture
 * - Smooth blending between frames
 * - SOTA audio-reactive shader effects (same as WebGL2 version)
 * - Native WebGPU rendering with WGSL shaders
 */

import { AudioMetrics } from '../../audio/types';
import { VisualizerConfig } from '../types';
import { ModeRenderer } from '../VisualizerEngine';
import {
  IRenderer,
  ITexture,
  IRenderPipeline,
  UniformLayout,
  boolToFloat,
} from '../renderer/types';

// Import WGSL shader source
import aiGeneratedWGSL from '../shaders/ai-generated.wgsl?raw';

/**
 * WebGPU-native AI Generated Mode
 */
export class AIGeneratedModeWebGPU implements ModeRenderer {
  private renderer: IRenderer;
  private config: VisualizerConfig;

  // Pipeline and resources
  private pipeline: IRenderPipeline | null = null;
  private currentTexture: ITexture | null = null;
  private previousTexture: ITexture | null = null;
  private blendFactor = 1.0;

  // External frame source
  private lastFrameId = -1;

  // Effect state
  private beatFlash = 0;
  private bassDistortion = 0;
  private onsetPulse = 0;
  private smoothedCentroid = 0.5;
  private dominantHue = 0;

  // New effect state
  private glitchSeed = 0;
  private glitchIntensity = 0;
  private smoothedTreble = 0;

  // Silence degradation state
  private silenceLevel = 0;
  private silenceThreshold = 0.03;
  private degradationRate = 0.5;
  private recoveryRate = 2.0;
  private maxDegradation = 0.85;
  private silenceDegradationEnabled = true;

  // Effect toggles
  private enableDistortion = false;
  private enableFlash = true;
  private enableSpectralDisplacement = false;
  private enableGlitchBlocks = false;
  private enableTrebleGrain = false;
  private enableBicubic = true;
  private enableSharpening = true;
  private sharpenStrength = 0.5;

  // Texture dimensions
  private sourceWidth = 256;
  private sourceHeight = 256;
  private displayWidth = 1920;
  private displayHeight = 1080;

  constructor(renderer: IRenderer, config: VisualizerConfig) {
    this.renderer = renderer;
    this.config = config;

    this.createPipeline();
  }

  private createPipeline(): void {
    // Define uniform layout matching the WGSL struct
    const uniforms: UniformLayout = {
      // Texture/frame parameters
      blendFactor: { type: 'float', value: 1.0 },
      hasTexture: { type: 'float', value: 0.0 },

      // Audio metrics
      bass: { type: 'float', value: 0.0 },
      mid: { type: 'float', value: 0.0 },
      treble: { type: 'float', value: 0.0 },
      rms: { type: 'float', value: 0.0 },
      beatFlash: { type: 'float', value: 0.0 },
      onsetPulse: { type: 'float', value: 0.0 },
      onsetStrength: { type: 'float', value: 0.0 },
      spectralCentroid: { type: 'float', value: 0.5 },
      dominantHue: { type: 'float', value: 0.0 },
      chromaEnergy: { type: 'float', value: 0.0 },

      // Effect toggles
      enableDistortion: { type: 'float', value: 0.0 },
      enableFlash: { type: 'float', value: 1.0 },
      enableSpectralDisplacement: { type: 'float', value: 0.0 },
      enableGlitchBlocks: { type: 'float', value: 0.0 },
      enableTrebleGrain: { type: 'float', value: 0.0 },

      // Effect parameters
      glitchSeed: { type: 'float', value: 0.0 },
      glitchIntensity: { type: 'float', value: 0.0 },
      smoothedTreble: { type: 'float', value: 0.0 },

      // Upscaling
      enableBicubic: { type: 'float', value: 1.0 },
      enableSharpening: { type: 'float', value: 1.0 },
      sharpenStrength: { type: 'float', value: 0.5 },
      textureWidth: { type: 'float', value: 256.0 },
      textureHeight: { type: 'float', value: 256.0 },

      // Silence degradation
      silenceLevel: { type: 'float', value: 0.0 },
      degradationSeed: { type: 'float', value: 0.0 },

      // General
      time: { type: 'float', value: 0.0 },
      intensity: { type: 'float', value: this.config.intensity },
    };

    // Create pipeline with WGSL shaders
    this.pipeline = this.renderer.createPipeline({
      id: 'ai-generated-mode',
      vertexShader: { wgsl: aiGeneratedWGSL },
      fragmentShader: { wgsl: aiGeneratedWGSL },
      uniforms,
      blend: { mode: 'none' },
    });
  }

  /**
   * Set the current frame image from external source.
   * Call this when a new AI-generated frame is received.
   */
  public setFrame(image: HTMLImageElement, frameId: number): void {
    if (frameId === this.lastFrameId) {
      return; // Same frame, skip
    }

    // DEBUG: log frame reception
    if (frameId % 30 === 0) {
      console.log(`[AIGeneratedModeWebGPU.setFrame] frameId=${frameId}, imageSize=${image.width}x${image.height}`);
    }

    this.lastFrameId = frameId;

    // Swap textures for blending
    if (this.currentTexture) {
      if (this.previousTexture) {
        this.previousTexture.dispose();
      }
      this.previousTexture = this.currentTexture;
    }

    // Create new texture from image
    this.currentTexture = this.renderer.createTexture({
      width: image.width,
      height: image.height,
      format: 'rgba8',
      minFilter: 'linear',
      magFilter: 'linear',
    });
    this.currentTexture.update(image);

    // Update source dimensions
    this.sourceWidth = image.width;
    this.sourceHeight = image.height;

    // Reset blend factor for transition
    this.blendFactor = 0;
  }

  public render(metrics: AudioMetrics, time: number, deltaTime: number): void {
    if (!this.pipeline) return;

    const { bass, mid, treble, isBeat, onset, spectralCentroid, chroma, dominantChroma, rms } = metrics;
    const { intensity } = this.config;

    // Update beat flash
    if (isBeat) {
      this.beatFlash = 1.0;
    } else {
      this.beatFlash *= 0.85;
    }

    // Update onset pulse
    if (onset?.isOnset) {
      this.onsetPulse = onset.confidence;
      this.glitchSeed = Math.random() * 1000;
      this.glitchIntensity = onset.confidence;
    } else {
      this.onsetPulse *= 0.9;
      this.glitchIntensity *= 0.85;
    }

    // Update bass distortion (smoothed)
    this.bassDistortion += (bass - this.bassDistortion) * 0.3;

    // Update smoothed treble
    this.smoothedTreble += (treble - this.smoothedTreble) * 0.3;

    // Update smoothed spectral centroid
    if (spectralCentroid !== undefined) {
      this.smoothedCentroid += (spectralCentroid - this.smoothedCentroid) * 0.2;
    }

    // Update dominant hue from chroma
    if (dominantChroma !== undefined) {
      this.dominantHue = dominantChroma / 12.0;
    }

    // Update silence degradation level
    if (this.silenceDegradationEnabled) {
      if (rms < this.silenceThreshold) {
        this.silenceLevel = Math.min(
          this.maxDegradation,
          this.silenceLevel + this.degradationRate * deltaTime
        );
      } else {
        this.silenceLevel = Math.max(
          0,
          this.silenceLevel - this.recoveryRate * deltaTime
        );
      }
    } else {
      this.silenceLevel = 0;
    }

    // Animate blend factor
    this.blendFactor = Math.min(1.0, this.blendFactor + deltaTime * 8.0);

    // Update uniforms
    this.pipeline.setUniforms({
      blendFactor: this.blendFactor,
      hasTexture: boolToFloat(this.currentTexture !== null),
      bass: this.bassDistortion,
      mid,
      treble,
      rms,
      beatFlash: this.beatFlash,
      onsetPulse: this.onsetPulse,
      onsetStrength: onset?.strength ?? 0,
      spectralCentroid: this.smoothedCentroid,
      dominantHue: this.dominantHue,
      chromaEnergy: chroma?.energy ?? 0,
      enableDistortion: boolToFloat(this.enableDistortion),
      enableFlash: boolToFloat(this.enableFlash),
      enableSpectralDisplacement: boolToFloat(this.enableSpectralDisplacement),
      enableGlitchBlocks: boolToFloat(this.enableGlitchBlocks),
      enableTrebleGrain: boolToFloat(this.enableTrebleGrain),
      glitchSeed: this.glitchSeed,
      glitchIntensity: this.glitchIntensity,
      smoothedTreble: this.smoothedTreble,
      enableBicubic: boolToFloat(this.enableBicubic),
      enableSharpening: boolToFloat(this.enableSharpening),
      sharpenStrength: this.sharpenStrength,
      textureWidth: this.sourceWidth,
      textureHeight: this.sourceHeight,
      silenceLevel: this.silenceLevel,
      degradationSeed: this.silenceLevel > 0.01 && Math.random() < deltaTime * 5
        ? Math.random() * 1000
        : 0,
      time,
      intensity,
    });

    // Bind textures
    if (this.currentTexture) {
      this.pipeline.setTexture(0, this.currentTexture, 'currentFrame');
    }
    if (this.previousTexture) {
      this.pipeline.setTexture(1, this.previousTexture, 'previousFrame');
    } else if (this.currentTexture) {
      // If no previous frame, use current for both
      this.pipeline.setTexture(1, this.currentTexture, 'previousFrame');
    }

    // Begin frame, clear, draw, end frame
    this.renderer.beginFrame();
    this.renderer.clear([0, 0, 0, 1]);
    this.renderer.drawFullscreenQuad(this.pipeline);
    this.renderer.endFrame();
  }

  public resize(width: number, height: number): void {
    this.displayWidth = width;
    this.displayHeight = height;
    this.renderer.resize(width, height);
  }

  public setConfig(config: Partial<VisualizerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  // === Effect control methods (matching AIGeneratedMode API) ===

  public setShaderEffects(enabled: boolean): void {
    this.enableDistortion = enabled;
  }

  public setFlashEnabled(enabled: boolean): void {
    this.enableFlash = enabled;
  }

  public setSpectralDisplacementEnabled(enabled: boolean): void {
    this.enableSpectralDisplacement = enabled;
  }

  public setGlitchBlocksEnabled(enabled: boolean): void {
    this.enableGlitchBlocks = enabled;
  }

  public setTrebleGrainEnabled(enabled: boolean): void {
    this.enableTrebleGrain = enabled;
  }

  public setBicubicEnabled(enabled: boolean): void {
    this.enableBicubic = enabled;
  }

  public setSharpeningEnabled(enabled: boolean): void {
    this.enableSharpening = enabled;
  }

  public setSharpenStrength(strength: number): void {
    this.sharpenStrength = Math.max(0, Math.min(1.5, strength));
  }

  public setTextureSize(width: number, height: number): void {
    this.sourceWidth = width;
    this.sourceHeight = height;
  }

  public setMaintainAspectRatio(_enabled: boolean): void {
    // Note: Aspect ratio is handled differently in WebGPU mode
    // For now, this is a no-op; could be implemented with viewport adjustments
  }

  public setSilenceDegradationEnabled(enabled: boolean): void {
    this.silenceDegradationEnabled = enabled;
    if (!enabled) {
      this.silenceLevel = 0;
    }
  }

  public setSilenceThreshold(threshold: number): void {
    this.silenceThreshold = Math.max(0.01, Math.min(0.2, threshold));
  }

  public setDegradationRate(rate: number): void {
    this.degradationRate = Math.max(0.1, Math.min(2.0, rate));
  }

  public setRecoveryRate(rate: number): void {
    this.recoveryRate = Math.max(0.5, Math.min(5.0, rate));
  }

  public dispose(): void {
    this.pipeline?.dispose();
    this.currentTexture?.dispose();
    this.previousTexture?.dispose();
    this.pipeline = null;
    this.currentTexture = null;
    this.previousTexture = null;
  }
}
