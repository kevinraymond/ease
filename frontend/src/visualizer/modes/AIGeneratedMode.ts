import * as THREE from 'three';
import { AudioMetrics } from '../../audio/types';
import { VisualizerConfig } from '../types';
import { ModeRenderer } from '../VisualizerEngine';
import { FSRUpscaler, FSRConfig } from '../shaders/FSR';

/**
 * AI Generated Mode - Displays AI-generated frames with SOTA audio-reactive effects.
 *
 * Features:
 * - Fullscreen display of AI frames as texture
 * - Smooth blending between frames
 * - SOTA audio-reactive shader effects:
 *   - Spectral centroid-driven chromatic aberration
 *   - Onset pulse effect (brightness spike on transients)
 *   - Chroma-based color shifting
 *   - Bass distortion and beat flash
 *   - Vignette and saturation boost
 *   - Spectral displacement (bass pushes out, treble pulls in)
 *   - Onset glitch blocks (random rectangular offsets on transients)
 *   - Treble grain/shimmer (high-frequency noise tracking treble)
 * - High-quality upscaling options:
 *   - Bicubic interpolation (fast, good quality)
 *   - FSR 1.0 (AMD FidelityFX Super Resolution, excellent quality, 2-5ms)
 */
export class AIGeneratedMode implements ModeRenderer {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private config: VisualizerConfig;

  // Display mesh
  private planeMesh: THREE.Mesh | null = null;
  private material: THREE.ShaderMaterial | null = null;

  // Textures for blending
  private currentTexture: THREE.Texture | null = null;
  private previousTexture: THREE.Texture | null = null;
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
  private silenceLevel = 0;  // Current degradation amount (0-1)
  private silenceThreshold = 0.03;  // RMS threshold
  private degradationRate = 0.5;  // How fast degradation increases
  private recoveryRate = 2.0;  // How fast image recovers
  private maxDegradation = 0.85;  // Max degradation (keeps image barely recognizable)
  private silenceDegradationEnabled = true;  // Match config default

  // FSR upscaler (optional, for high-quality 256->1080p upscaling)
  private fsrUpscaler: FSRUpscaler | null = null;
  private fsrEnabled = false;
  private renderer: THREE.WebGLRenderer | null = null;
  private displayWidth = 1920;
  private displayHeight = 1080;
  private sourceWidth = 256;
  private sourceHeight = 256;

  // Aspect ratio constraint
  private maintainAspectRatio = false;

  constructor(
    scene: THREE.Scene,
    _camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
    config: VisualizerConfig
  ) {
    this.scene = scene;
    this.config = config;

    // Create orthographic camera for 2D display
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    this.camera.position.z = 1;

    this.createPlane();
  }

  private createPlane(): void {
    // Clean up existing
    if (this.planeMesh) {
      this.scene.remove(this.planeMesh);
      this.planeMesh.geometry.dispose();
      this.material?.dispose();
    }

    // Fullscreen quad geometry
    const geometry = new THREE.PlaneGeometry(2, 2);

    // SOTA shader material with enhanced audio-reactive effects
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uCurrentFrame: { value: null },
        uPreviousFrame: { value: null },
        uBlendFactor: { value: 1.0 },
        uBass: { value: 0.0 },
        uMid: { value: 0.0 },
        uTreble: { value: 0.0 },
        uBeatFlash: { value: 0.0 },
        uOnsetPulse: { value: 0.0 },
        uTime: { value: 0.0 },
        uIntensity: { value: this.config.intensity },
        uHasTexture: { value: false },
        // SOTA uniforms
        uSpectralCentroid: { value: 0.5 },
        uOnsetStrength: { value: 0.0 },
        uDominantHue: { value: 0.0 },
        uChromaEnergy: { value: 0.0 },
        uEnableDistortion: { value: false },
        uEnableFlash: { value: true },
        // New effect uniforms
        uEnableSpectralDisplacement: { value: false },
        uEnableGlitchBlocks: { value: false },
        uEnableTrebleGrain: { value: false },
        uGlitchSeed: { value: 0.0 },
        uGlitchIntensity: { value: 0.0 },
        uSmoothedTreble: { value: 0.0 },
        // Upscaling uniforms
        uEnableBicubic: { value: true },
        uEnableSharpening: { value: true },
        uSharpenStrength: { value: 0.5 },
        uTextureSize: { value: new THREE.Vector2(256, 256) },
        // Silence degradation uniforms
        uSilenceLevel: { value: 0.0 },
        uDegradationSeed: { value: 0.0 },
        // RMS for early-out optimization
        uRMS: { value: 0.0 },
      },
      vertexShader: `
        varying vec2 vUv;
        // Precomputed values for fragment shader optimization
        varying vec2 vNormalizedUv;  // UV in -1 to 1 range
        varying float vVignetteBase; // Distance from center for vignette
        varying vec2 vCenterDir;     // Direction from center (normalized)

        void main() {
          vUv = uv;
          // Precompute frequently used values
          vNormalizedUv = uv * 2.0 - 1.0;
          vec2 centerOffset = uv - 0.5;
          vVignetteBase = length(centerOffset) * 2.0;
          vCenterDir = normalize(centerOffset + 0.001);
          // Apply model matrix to respect mesh scale/position transforms
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D uCurrentFrame;
        uniform sampler2D uPreviousFrame;
        uniform float uBlendFactor;
        uniform float uBass;
        uniform float uMid;
        uniform float uTreble;
        uniform float uBeatFlash;
        uniform float uOnsetPulse;
        uniform float uTime;
        uniform float uIntensity;
        uniform bool uHasTexture;
        // SOTA uniforms
        uniform float uSpectralCentroid;
        uniform float uOnsetStrength;
        uniform float uDominantHue;
        uniform float uChromaEnergy;
        uniform bool uEnableDistortion;
        uniform bool uEnableFlash;
        // New effect uniforms
        uniform bool uEnableSpectralDisplacement;
        uniform bool uEnableGlitchBlocks;
        uniform bool uEnableTrebleGrain;
        uniform float uGlitchSeed;
        uniform float uGlitchIntensity;
        uniform float uSmoothedTreble;
        // Upscaling uniforms
        uniform bool uEnableBicubic;
        uniform bool uEnableSharpening;
        uniform float uSharpenStrength;
        uniform vec2 uTextureSize;
        // Silence degradation uniforms
        uniform float uSilenceLevel;
        uniform float uDegradationSeed;
        // RMS for early-out optimization
        uniform float uRMS;

        varying vec2 vUv;
        // Precomputed values from vertex shader
        varying vec2 vNormalizedUv;
        varying float vVignetteBase;
        varying vec2 vCenterDir;

        // Convert HSV to RGB
        vec3 hsv2rgb(vec3 c) {
          vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
          vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
          return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }

        // Hash function for pseudo-random values
        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        // 2D noise for grain effect
        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          float a = hash(i);
          float b = hash(i + vec2(1.0, 0.0));
          float c = hash(i + vec2(0.0, 1.0));
          float d = hash(i + vec2(1.0, 1.0));
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }

        // Cubic interpolation weight function (Catmull-Rom spline)
        vec4 cubic(float v) {
          vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
          vec4 s = n * n * n;
          float x = s.x;
          float y = s.y - 4.0 * s.x;
          float z = s.z - 4.0 * s.y + 6.0 * s.x;
          float w = 6.0 - x - y - z;
          return vec4(x, y, z, w) * (1.0 / 6.0);
        }

        // Bicubic texture sampling - provides sharper edges than bilinear
        vec4 textureBicubic(sampler2D tex, vec2 texCoords, vec2 texSize) {
          vec2 invTexSize = 1.0 / texSize;
          texCoords = texCoords * texSize - 0.5;

          vec2 fxy = fract(texCoords);
          texCoords -= fxy;

          vec4 xcubic = cubic(fxy.x);
          vec4 ycubic = cubic(fxy.y);

          vec4 c = texCoords.xxyy + vec2(-0.5, 1.5).xyxy;

          vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
          vec4 offset = c + vec4(xcubic.yw, ycubic.yw) / s;

          offset *= invTexSize.xxyy;

          vec4 sample0 = texture2D(tex, offset.xz);
          vec4 sample1 = texture2D(tex, offset.yz);
          vec4 sample2 = texture2D(tex, offset.xw);
          vec4 sample3 = texture2D(tex, offset.yw);

          float sx = s.x / (s.x + s.y);
          float sy = s.z / (s.z + s.w);

          return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
        }

        // Unsharp mask sharpening - enhances edges
        // Takes pre-sampled center color to work with bicubic
        vec3 sharpenWithCenter(vec3 center, sampler2D tex, vec2 uv, vec2 texSize, float strength) {
          vec2 texel = 1.0 / texSize;

          // Sample neighbors (bilinear is fine for edge detection)
          vec3 top = texture2D(tex, uv + vec2(0.0, texel.y)).rgb;
          vec3 bottom = texture2D(tex, uv - vec2(0.0, texel.y)).rgb;
          vec3 left = texture2D(tex, uv - vec2(texel.x, 0.0)).rgb;
          vec3 right = texture2D(tex, uv + vec2(texel.x, 0.0)).rgb;

          // Laplacian edge detection
          vec3 laplacian = 4.0 * center - top - bottom - left - right;

          // Add edge enhancement
          return center + laplacian * strength;
        }

        // Legacy sharpen function for backwards compatibility
        vec3 sharpen(sampler2D tex, vec2 uv, vec2 texSize, float strength) {
          vec3 center = texture2D(tex, uv).rgb;
          return sharpenWithCenter(center, tex, uv, texSize, strength);
        }

        // Bicubic sampling with chromatic aberration
        vec3 sampleBicubicChroma(sampler2D tex, vec2 uv, vec2 texSize, float aberration) {
          vec2 offset = (uv - 0.5) * aberration * 0.02;
          float r = textureBicubic(tex, uv + offset, texSize).r;
          float g = textureBicubic(tex, uv, texSize).g;
          float b = textureBicubic(tex, uv - offset, texSize).b;
          return vec3(r, g, b);
        }

        // Chromatic aberration based on spectral centroid
        vec3 chromaticAberration(sampler2D tex, vec2 uv, float strength) {
          vec2 offset = (uv - 0.5) * strength * uSpectralCentroid * 0.02;
          float r = texture2D(tex, uv + offset).r;
          float g = texture2D(tex, uv).g;
          float b = texture2D(tex, uv - offset).b;
          return vec3(r, g, b);
        }

        void main() {
          // === EARLY-OUT FOR SILENCE ===
          // When completely silent, skip all effects for performance
          if (uRMS < 0.001 && uSilenceLevel > 0.99 && uHasTexture) {
            gl_FragColor = texture2D(uCurrentFrame, vUv);
            return;
          }

          if (!uHasTexture) {
            // No frame yet - show animated placeholder gradient
            vec3 color1 = vec3(0.05, 0.05, 0.15);
            vec3 color2 = vec3(0.1, 0.05, 0.2);
            float t = sin(uTime * 0.5 + vUv.x * 3.14159) * 0.5 + 0.5;

            // Add chroma-based color accent
            vec3 chromaColor = hsv2rgb(vec3(uDominantHue, 0.5, 0.3));
            vec3 baseColor = mix(color1, color2, t);
            gl_FragColor = vec4(mix(baseColor, chromaColor, uChromaEnergy * 0.3), 1.0);
            return;
          }

          vec2 uv = vUv;
          // Use precomputed vCenterDir from vertex shader
          float dist = length(vUv - 0.5);

          // === SPECTRAL DISPLACEMENT (Branchless) ===
          // Bass pushes outward, treble pulls inward
          float displacement = (uBass * 0.03 - uSmoothedTreble * 0.02) * uIntensity;
          vec2 displacedUv = uv + vCenterDir * displacement * dist;
          // Branchless conditional: mix between original and displaced based on enable flag
          uv = mix(uv, displacedUv, float(uEnableSpectralDisplacement));

          // === ONSET GLITCH BLOCKS (Branchless) ===
          float blockSize = 0.1 + hash(vec2(uGlitchSeed, 0.0)) * 0.15;
          vec2 blockCoord = floor(uv / blockSize);
          float blockRand = hash(blockCoord + uGlitchSeed);
          // Branchless: compute offset always, multiply by enable conditions
          vec2 glitchOffset = vec2(
            (hash(blockCoord + uGlitchSeed + 1.0) - 0.5) * 0.1,
            (hash(blockCoord + uGlitchSeed + 2.0) - 0.5) * 0.1
          ) * uGlitchIntensity;
          // Apply only if enabled, intensity > threshold, and blockRand > 0.7
          float glitchEnable = float(uEnableGlitchBlocks) * step(0.05, uGlitchIntensity) * step(0.7, blockRand);
          uv += glitchOffset * glitchEnable;

          // === DISTORTION EFFECTS (Branchless) ===
          float distortionStrength = uBass * uIntensity * 0.02;
          float distortX = sin(vUv.y * 10.0 + uTime * 2.0) * distortionStrength;
          float distortY = cos(vUv.x * 10.0 + uTime * 2.0) * distortionStrength;
          vec2 distortedUv = uv + vec2(distortX, distortY);

          // Onset pulse - radial distortion on transients
          float pulseDist = length(distortedUv - 0.5);
          float pulseWave = sin(pulseDist * 20.0 - uTime * 15.0) * uOnsetPulse * 0.01;
          vec2 pulseDir = normalize(distortedUv - 0.5 + 0.001);
          distortedUv += pulseDir * pulseWave * step(0.01, uOnsetPulse);

          // Branchless mix for distortion enable
          uv = mix(uv, distortedUv, float(uEnableDistortion));

          // Clamp UVs to prevent wrapping
          uv = clamp(uv, 0.001, 0.999);

          // === TEXTURE SAMPLING (Branchless bicubic/bilinear selection) ===
          float aberrationStrength = uSpectralCentroid * uIntensity;

          // Always compute both paths, then select (avoids divergent branching)
          vec3 bicubicCurrent = sampleBicubicChroma(uCurrentFrame, uv, uTextureSize, aberrationStrength);
          vec3 bicubicPrevious = sampleBicubicChroma(uPreviousFrame, uv, uTextureSize, aberrationStrength * 0.5);
          vec3 bilinearCurrent = chromaticAberration(uCurrentFrame, uv, aberrationStrength);
          vec3 bilinearPrevious = chromaticAberration(uPreviousFrame, uv, aberrationStrength * 0.5);

          // Branchless selection
          float bicubicWeight = float(uEnableBicubic);
          vec3 currentColor = mix(bilinearCurrent, bicubicCurrent, bicubicWeight);
          vec3 previousColor = mix(bilinearPrevious, bicubicPrevious, bicubicWeight);

          // Blend between frames
          vec3 color = mix(previousColor, currentColor, uBlendFactor);

          // === SHARPENING (Branchless) ===
          float dynamicSharpen = uSharpenStrength + uOnsetPulse * 0.3 + uBeatFlash * 0.2;
          dynamicSharpen = clamp(dynamicSharpen, 0.0, 1.5);
          vec3 sharpCurrent = sharpenWithCenter(currentColor, uCurrentFrame, uv, uTextureSize, dynamicSharpen * uIntensity);
          vec3 sharpPrevious = sharpenWithCenter(previousColor, uPreviousFrame, uv, uTextureSize, dynamicSharpen * uIntensity * 0.5);
          vec3 sharpenedColor = mix(sharpPrevious, sharpCurrent, uBlendFactor);
          // Branchless selection
          color = mix(color, sharpenedColor, float(uEnableSharpening));

          // === SILENCE DEGRADATION EFFECTS ===
          // When audio goes silent, gradually degrade the image with VHS-like artifacts
          if (uSilenceLevel > 0.01) {
            float sl = uSilenceLevel;

            // 1. Block noise (JPEG-artifact-like rectangular blocks)
            float blockSize = 8.0 + sl * 24.0;  // Larger blocks as degradation increases
            vec2 blockCoord = floor(vUv * uTextureSize / blockSize);
            float blockNoise = hash(blockCoord + uDegradationSeed);
            if (blockNoise > (1.0 - sl * 0.4) && sl > 0.3) {
              // Shift this block's color slightly
              vec3 blockShift = vec3(
                hash(blockCoord + uDegradationSeed + 1.0) - 0.5,
                hash(blockCoord + uDegradationSeed + 2.0) - 0.5,
                hash(blockCoord + uDegradationSeed + 3.0) - 0.5
              ) * sl * 0.3;
              color += blockShift;
            }

            // 2. Scanline distortion (VHS aesthetic)
            float scanlinePhase = vUv.y * uTextureSize.y + uTime * 20.0 + uDegradationSeed * 100.0;
            float scanlineNoise = sin(scanlinePhase * 3.14159) * 0.5 + 0.5;
            scanlineNoise = pow(scanlineNoise, 4.0);  // Sharper lines
            color = mix(color, color * (1.0 - scanlineNoise * 0.3), sl * 0.5);

            // 3. Static/grain overlay (increasing intensity)
            float staticNoise = noise(vUv * 800.0 + uTime * 100.0 + uDegradationSeed * 50.0);
            staticNoise = (staticNoise - 0.5) * 2.0;  // Center around 0
            color += vec3(staticNoise * sl * 0.15);

            // 4. Color banding (posterization effect)
            if (sl > 0.2) {
              float levels = 32.0 - sl * 24.0;  // Fewer levels = more banding
              levels = max(levels, 4.0);
              color = floor(color * levels) / levels;
            }

            // 5. Enhanced vignette (darkness closes in from edges)
            float vignetteRadius = 1.5 - sl * 0.8;  // Shrinks with degradation
            float vignetteDeg = 1.0 - smoothstep(0.3, vignetteRadius, length(vUv - 0.5) * 2.0);
            color *= mix(1.0, vignetteDeg, sl * 0.6);

            // 6. Occasional glitch lines (horizontal displacement)
            if (sl > 0.4) {
              float glitchLine = hash(vec2(floor(vUv.y * 50.0), uDegradationSeed));
              if (glitchLine > (1.0 - sl * 0.15)) {
                float lineOffset = (hash(vec2(glitchLine, uDegradationSeed)) - 0.5) * 0.1 * sl;
                vec2 glitchUv = vec2(vUv.x + lineOffset, vUv.y);
                glitchUv = clamp(glitchUv, 0.001, 0.999);
                vec3 glitchColor = texture2D(uCurrentFrame, glitchUv).rgb;
                color = mix(color, glitchColor, 0.7);
              }
            }

            // 7. Color channel separation (chromatic aberration increases)
            if (sl > 0.25) {
              float chromaSep = sl * 0.015;
              color.r = texture2D(uCurrentFrame, vUv + vec2(chromaSep, 0.0)).r;
              color.b = texture2D(uCurrentFrame, vUv - vec2(chromaSep, 0.0)).b;
            }

            // 8. Overall desaturation (colors fade)
            vec3 gray = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
            color = mix(color, gray, sl * 0.4);
          }

          // === BEAT FLASH (Branchless) ===
          // Add brightness when flash is enabled
          color += vec3(uBeatFlash * uIntensity * 0.3 * float(uEnableFlash));

          // Onset pulse brightness (always active)
          color += vec3(uOnsetPulse * uIntensity * 0.15);

          // === TREBLE GRAIN/SHIMMER (Branchless) ===
          float grainScale = 400.0;
          float grain = noise(vUv * grainScale + uTime * 50.0);
          grain = (grain - 0.5) * 2.0;
          float grainStrength = uSmoothedTreble * uIntensity * 0.08;
          // Branchless: multiply by enable flag and threshold step
          color += vec3(grain * grainStrength * float(uEnableTrebleGrain) * step(0.1, uSmoothedTreble));

          // === VIGNETTE (using precomputed vVignetteBase) ===
          float vignetteStrength = 0.3 + (1.0 - uBass) * 0.2;
          float vignette = 1.0 - smoothstep(0.5, 1.5, vVignetteBase);
          color *= mix(1.0, vignette, vignetteStrength);

          // === SATURATION BOOST (always active) ===
          float saturationBoost = 1.0 + uBass * uIntensity * 0.3;
          vec3 grayColor = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
          color = mix(grayColor, color, saturationBoost);

          // === CHROMA TINTING (Branchless) ===
          vec3 chromaTint = hsv2rgb(vec3(uDominantHue, 0.3, 1.0));
          float luminance = dot(color, vec3(0.299, 0.587, 0.114));
          vec3 tintedColor = mix(color, color * chromaTint, uChromaEnergy * 0.15);
          // Branchless: apply tint only when chroma > 0.1 and luminance > 0.5
          color = mix(color, tintedColor, step(0.1, uChromaEnergy) * step(0.5, luminance));

          // === SPECTRAL CENTROID CONTRAST (always active) ===
          float contrast = 0.9 + uSpectralCentroid * 0.2;
          color = (color - 0.5) * contrast + 0.5;

          // Final clamp
          color = clamp(color, 0.0, 1.0);

          gl_FragColor = vec4(color, 1.0);
        }
      `,
    });

    this.planeMesh = new THREE.Mesh(geometry, this.material);
    this.scene.add(this.planeMesh);
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
      console.log(`[AIGeneratedMode.setFrame] frameId=${frameId}, imageSize=${image.width}x${image.height}, hasTexture=${!!this.currentTexture}`);
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
    this.currentTexture = new THREE.Texture(image);
    this.currentTexture.needsUpdate = true;
    this.currentTexture.minFilter = THREE.LinearFilter;
    this.currentTexture.magFilter = THREE.LinearFilter;

    // Reset blend factor for transition
    this.blendFactor = 0;

    // Update uniforms
    if (this.material) {
      this.material.uniforms.uCurrentFrame.value = this.currentTexture;
      if (this.previousTexture) {
        this.material.uniforms.uPreviousFrame.value = this.previousTexture;
      } else {
        this.material.uniforms.uPreviousFrame.value = this.currentTexture;
      }
      this.material.uniforms.uHasTexture.value = true;
    }
  }

  public render(metrics: AudioMetrics, time: number, deltaTime: number): void {
    if (!this.material) return;

    const { bass, mid, treble, isBeat, onset, spectralCentroid, chroma, dominantChroma, rms } = metrics;
    const { intensity } = this.config;

    // Update beat flash
    if (isBeat) {
      this.beatFlash = 1.0;
    } else {
      this.beatFlash *= 0.85; // Decay
    }

    // Update onset pulse (faster decay for transients)
    if (onset?.isOnset) {
      this.onsetPulse = onset.confidence;
      // Trigger glitch on onset
      this.glitchSeed = Math.random() * 1000;
      this.glitchIntensity = onset.confidence;
    } else {
      this.onsetPulse *= 0.9;
      // Decay glitch intensity
      this.glitchIntensity *= 0.85;
    }

    // Update bass distortion (smoothed)
    this.bassDistortion += (bass - this.bassDistortion) * 0.3;

    // Update smoothed treble for grain effect
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
        // Silence detected - increase degradation
        this.silenceLevel = Math.min(
          this.maxDegradation,
          this.silenceLevel + this.degradationRate * deltaTime
        );
      } else {
        // Sound detected - recover
        this.silenceLevel = Math.max(
          0,
          this.silenceLevel - this.recoveryRate * deltaTime
        );
      }
    } else {
      // Feature disabled - ensure no degradation
      this.silenceLevel = 0;
    }

    // Animate blend factor (frame transition)
    this.blendFactor = Math.min(1.0, this.blendFactor + deltaTime * 8.0);

    // Update shader uniforms
    this.material.uniforms.uBlendFactor.value = this.blendFactor;
    this.material.uniforms.uBass.value = this.bassDistortion;
    this.material.uniforms.uMid.value = mid;
    this.material.uniforms.uTreble.value = treble;
    this.material.uniforms.uBeatFlash.value = this.beatFlash;
    this.material.uniforms.uOnsetPulse.value = this.onsetPulse;
    this.material.uniforms.uTime.value = time;
    this.material.uniforms.uIntensity.value = intensity;

    // SOTA uniforms
    this.material.uniforms.uSpectralCentroid.value = this.smoothedCentroid;
    this.material.uniforms.uOnsetStrength.value = onset?.strength ?? 0;
    this.material.uniforms.uDominantHue.value = this.dominantHue;
    this.material.uniforms.uChromaEnergy.value = chroma?.energy ?? 0;

    // New effect uniforms
    this.material.uniforms.uGlitchSeed.value = this.glitchSeed;
    this.material.uniforms.uGlitchIntensity.value = this.glitchIntensity;
    this.material.uniforms.uSmoothedTreble.value = this.smoothedTreble;

    // Silence degradation uniforms
    this.material.uniforms.uSilenceLevel.value = this.silenceLevel;
    // Update seed periodically for variation in degradation patterns
    if (this.silenceLevel > 0.01 && Math.random() < deltaTime * 5) {
      this.material.uniforms.uDegradationSeed.value = Math.random() * 1000;
    }

    // RMS for early-out optimization
    this.material.uniforms.uRMS.value = rms;
  }

  public resize(width: number, height: number): void {
    // Update FSR display size when canvas resizes
    this.displayWidth = width;
    this.displayHeight = height;
    if (this.fsrUpscaler) {
      this.fsrUpscaler.resize({ outputWidth: width, outputHeight: height });
    }
    // Update plane scale for aspect ratio constraint
    this.updatePlaneScale();
  }

  /**
   * Update plane scale based on aspect ratio settings.
   * If maintainAspectRatio is true, scales to fit with letterbox/pillarbox.
   * If false, stretches to fill the display.
   */
  private updatePlaneScale(): void {
    if (!this.planeMesh) return;

    if (!this.maintainAspectRatio) {
      // Stretch to fill - reset to default scale and center position
      this.planeMesh.scale.set(1, 1, 1);
      this.planeMesh.position.set(0, 0, 0);
      return;
    }

    // Calculate aspect ratios
    const sourceAspect = this.sourceWidth / this.sourceHeight;
    const displayAspect = this.displayWidth / this.displayHeight;

    let scaleX = 1;
    let scaleY = 1;

    if (sourceAspect > displayAspect) {
      // Source is wider - pillarbox (bars on top/bottom)
      scaleY = displayAspect / sourceAspect;
    } else {
      // Source is taller - letterbox (bars on sides)
      scaleX = sourceAspect / displayAspect;
    }

    this.planeMesh.scale.set(scaleX, scaleY, 1);

    // Position at top-left with 20px margin
    // Convert pixel margin to NDC space (NDC goes from -1 to 1, so total is 2)
    const marginPx = 20;
    const marginX = (marginPx / this.displayWidth) * 2;
    const marginY = (marginPx / this.displayHeight) * 2;

    // Plane spans from -scale to +scale when centered at origin
    // To position top-left corner at (-1 + margin, 1 - margin):
    // centerX = -1 + margin + scaleX
    // centerY = 1 - margin - scaleY
    const posX = -1 + marginX + scaleX;
    const posY = 1 - marginY - scaleY;

    this.planeMesh.position.set(posX, posY, 0);
  }

  public setConfig(config: Partial<VisualizerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Enable or disable aspect ratio constraint.
   * When enabled, adds letterbox/pillarbox to maintain source proportions.
   * When disabled, stretches to fill the screen.
   */
  public setMaintainAspectRatio(enabled: boolean): void {
    this.maintainAspectRatio = enabled;
    this.updatePlaneScale();
  }

  /**
   * Returns the orthographic camera used by this mode.
   * VisualizerEngine uses this for rendering instead of its default perspective camera.
   */
  public getCamera(): THREE.Camera {
    return this.camera;
  }

  /**
   * Enable or disable shader distortion effects (wavy/pulse).
   */
  public setShaderEffects(enabled: boolean): void {
    if (this.material) {
      this.material.uniforms.uEnableDistortion.value = enabled;
    }
  }

  /**
   * Enable or disable the beat flash effect.
   * Disabling can help with video tearing issues.
   */
  public setFlashEnabled(enabled: boolean): void {
    if (this.material) {
      this.material.uniforms.uEnableFlash.value = enabled;
    }
  }

  /**
   * Enable or disable spectral displacement effect.
   * Bass pushes pixels outward, treble pulls inward.
   */
  public setSpectralDisplacementEnabled(enabled: boolean): void {
    if (this.material) {
      this.material.uniforms.uEnableSpectralDisplacement.value = enabled;
    }
  }

  /**
   * Enable or disable onset glitch blocks effect.
   * Random rectangular regions offset on audio transients.
   */
  public setGlitchBlocksEnabled(enabled: boolean): void {
    if (this.material) {
      this.material.uniforms.uEnableGlitchBlocks.value = enabled;
    }
  }

  /**
   * Enable or disable treble grain/shimmer effect.
   * High-frequency visual noise that tracks treble energy.
   */
  public setTrebleGrainEnabled(enabled: boolean): void {
    if (this.material) {
      this.material.uniforms.uEnableTrebleGrain.value = enabled;
    }
  }

  /**
   * Enable or disable bicubic interpolation for upscaling.
   * Provides sharper edges compared to default bilinear filtering.
   * Cost: ~0.5-1ms additional per frame.
   */
  public setBicubicEnabled(enabled: boolean): void {
    if (this.material) {
      this.material.uniforms.uEnableBicubic.value = enabled;
    }
  }

  /**
   * Enable or disable unsharp mask sharpening.
   * Enhances edges for crisper appearance when upscaling.
   * Strength is automatically boosted on beats/onsets for punch.
   */
  public setSharpeningEnabled(enabled: boolean): void {
    if (this.material) {
      this.material.uniforms.uEnableSharpening.value = enabled;
    }
  }

  /**
   * Set the base sharpening strength (0.0 - 1.5).
   * This is the baseline; actual strength varies with audio.
   * Default: 0.5
   */
  public setSharpenStrength(strength: number): void {
    if (this.material) {
      this.material.uniforms.uSharpenStrength.value = Math.max(0, Math.min(1.5, strength));
    }
  }

  // === Silence Degradation Methods ===

  /**
   * Enable or disable silence degradation effect.
   * When enabled, the image gradually degrades with VHS-like artifacts
   * during periods of silence (low RMS), then recovers when sound returns.
   */
  public setSilenceDegradationEnabled(enabled: boolean): void {
    this.silenceDegradationEnabled = enabled;
    if (!enabled) {
      this.silenceLevel = 0;  // Reset degradation when disabled
      if (this.material) {
        this.material.uniforms.uSilenceLevel.value = 0;
      }
    }
  }

  /**
   * Set the RMS threshold below which silence is detected.
   * @param threshold - RMS value (0.0-0.2), default: 0.08
   */
  public setSilenceThreshold(threshold: number): void {
    this.silenceThreshold = Math.max(0.01, Math.min(0.2, threshold));
  }

  /**
   * Set how fast degradation increases during silence.
   * @param rate - Degradation speed (0.1-2.0), default: 0.5
   */
  public setDegradationRate(rate: number): void {
    this.degradationRate = Math.max(0.1, Math.min(2.0, rate));
  }

  /**
   * Set how fast the image recovers when sound returns.
   * @param rate - Recovery speed (0.5-5.0), default: 2.0
   */
  public setRecoveryRate(rate: number): void {
    this.recoveryRate = Math.max(0.5, Math.min(5.0, rate));
  }

  /**
   * Get the current silence level for debugging/display.
   * @returns Current degradation level (0-1)
   */
  public getSilenceLevel(): number {
    return this.silenceLevel;
  }

  /**
   * Update the source texture size for proper bicubic sampling.
   * Call this when the AI generation resolution changes.
   */
  public setTextureSize(width: number, height: number): void {
    this.sourceWidth = width;
    this.sourceHeight = height;
    if (this.material) {
      this.material.uniforms.uTextureSize.value.set(width, height);
    }
    // Update FSR input size if enabled
    if (this.fsrUpscaler) {
      this.fsrUpscaler.resize({ inputWidth: width, inputHeight: height });
    }
    // Update plane scale for aspect ratio constraint
    this.updatePlaneScale();
  }

  // === FSR 1.0 Upscaling ===

  /**
   * Initialize FSR upscaler with a WebGL renderer.
   * Must be called before enabling FSR.
   *
   * @param renderer - Three.js WebGL renderer
   */
  public initFSR(renderer: THREE.WebGLRenderer): void {
    this.renderer = renderer;
    // Get current canvas size for display dimensions
    const canvas = renderer.domElement;
    this.displayWidth = canvas.clientWidth || 1920;
    this.displayHeight = canvas.clientHeight || 1080;
  }

  /**
   * Enable or disable FSR 1.0 upscaling.
   *
   * FSR provides high-quality upscaling from low-res AI frames (e.g., 256x256)
   * to display resolution (e.g., 1080p) with minimal latency (2-5ms).
   *
   * Note: Requires initFSR() to be called first with a renderer.
   *
   * @param enabled - Whether to enable FSR
   */
  public setFSREnabled(enabled: boolean): void {
    if (enabled && !this.renderer) {
      console.warn('FSR: Cannot enable without renderer. Call initFSR() first.');
      return;
    }

    this.fsrEnabled = enabled;

    if (enabled && !this.fsrUpscaler && this.renderer) {
      // Create FSR upscaler
      const config: FSRConfig = {
        inputWidth: this.sourceWidth,
        inputHeight: this.sourceHeight,
        outputWidth: this.displayWidth,
        outputHeight: this.displayHeight,
        sharpness: 0.2,
      };
      this.fsrUpscaler = new FSRUpscaler(this.renderer, config);
      console.log(`FSR enabled: ${this.sourceWidth}x${this.sourceHeight} -> ${this.displayWidth}x${this.displayHeight}`);
    } else if (!enabled && this.fsrUpscaler) {
      this.fsrUpscaler.dispose();
      this.fsrUpscaler = null;
      console.log('FSR disabled');
    }
  }

  /**
   * Set FSR sharpness level.
   *
   * @param sharpness - Sharpness value (0.0-2.0). Default: 0.2
   *                    Higher values produce sharper but potentially more artifacts.
   */
  public setFSRSharpness(sharpness: number): void {
    if (this.fsrUpscaler) {
      this.fsrUpscaler.setSharpness(sharpness);
    }
  }

  /**
   * Update FSR display resolution.
   * Call this when the canvas/display size changes.
   *
   * @param width - Display width
   * @param height - Display height
   */
  public setFSRDisplaySize(width: number, height: number): void {
    this.displayWidth = width;
    this.displayHeight = height;
    if (this.fsrUpscaler) {
      this.fsrUpscaler.resize({ outputWidth: width, outputHeight: height });
    }
  }

  /**
   * Get the FSR upscaler instance for advanced usage.
   * Returns null if FSR is not enabled.
   */
  public getFSRUpscaler(): FSRUpscaler | null {
    return this.fsrUpscaler;
  }

  /**
   * Check if FSR is currently enabled.
   */
  public isFSREnabled(): boolean {
    return this.fsrEnabled && this.fsrUpscaler !== null;
  }

  public dispose(): void {
    if (this.planeMesh) {
      this.scene.remove(this.planeMesh);
      this.planeMesh.geometry.dispose();
    }
    if (this.material) {
      this.material.dispose();
    }
    if (this.currentTexture) {
      this.currentTexture.dispose();
    }
    if (this.previousTexture) {
      this.previousTexture.dispose();
    }
    if (this.fsrUpscaler) {
      this.fsrUpscaler.dispose();
      this.fsrUpscaler = null;
    }
    this.planeMesh = null;
    this.material = null;
    this.currentTexture = null;
    this.previousTexture = null;
    this.renderer = null;
  }
}
