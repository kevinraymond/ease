/**
 * Audio-Reactive Effects Shader (WGSL)
 *
 * Port of the AIGeneratedMode GLSL shader to WGSL for WebGPU.
 * Features all the same audio-reactive visual effects:
 * - Spectral displacement
 * - Chromatic aberration
 * - Onset glitch blocks
 * - Treble grain/shimmer
 * - Beat flash and onset pulse
 * - Silence degradation effects
 * - Bicubic upscaling with sharpening
 */

// =============================================================================
// Uniforms
// =============================================================================

struct AudioUniforms {
  // Basic frequency bands
  bass: f32,
  mid: f32,
  treble: f32,
  rms: f32,

  // Beat/onset detection
  beatFlash: f32,
  onsetPulse: f32,
  onsetStrength: f32,
  _pad0: f32,  // Padding for alignment

  // Spectral features
  spectralCentroid: f32,
  dominantHue: f32,
  chromaEnergy: f32,
  _pad1: f32,

  // Effect toggles (0.0 or 1.0)
  enableDistortion: f32,
  enableFlash: f32,
  enableSpectralDisplacement: f32,
  enableGlitchBlocks: f32,

  enableTrebleGrain: f32,
  enableBicubic: f32,
  enableSharpening: f32,
  sharpenStrength: f32,

  // Effect parameters
  glitchSeed: f32,
  glitchIntensity: f32,
  smoothedTreble: f32,
  blendFactor: f32,

  // Silence degradation
  silenceLevel: f32,
  degradationSeed: f32,
  time: f32,
  intensity: f32,

  // Texture/resolution info
  textureSize: vec2f,
  resolution: vec2f,

  // Frame state
  hasTexture: f32,
  _pad2: vec3f,
}

@group(0) @binding(0) var<uniform> u: AudioUniforms;

// Texture bindings
@group(0) @binding(1) var texSampler: sampler;
@group(0) @binding(2) var currentFrame: texture_2d<f32>;
@group(0) @binding(3) var prevSampler: sampler;
@group(0) @binding(4) var previousFrame: texture_2d<f32>;

// =============================================================================
// Vertex Shader
// =============================================================================

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;

  // Fullscreen quad (6 vertices)
  var positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0)
  );

  var uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(0.0, 0.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0)
  );

  let pos = positions[vertexIndex];
  output.position = vec4f(pos, 0.0, 1.0);
  output.uv = uvs[vertexIndex];

  return output;
}

// =============================================================================
// Utility Functions
// =============================================================================

fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let a = hash(i);
  let b = hash(i + vec2f(1.0, 0.0));
  let c = hash(i + vec2f(0.0, 1.0));
  let d = hash(i + vec2f(1.0, 1.0));
  let uu = f * f * (3.0 - 2.0 * f);
  return mix(a, b, uu.x) + (c - a) * uu.y * (1.0 - uu.x) + (d - b) * uu.x * uu.y;
}

fn hsv2rgb(c: vec3f) -> vec3f {
  let K = vec4f(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

// Cubic weights for bicubic interpolation
fn cubic(v: f32) -> vec4f {
  let n = vec4f(1.0, 2.0, 3.0, 4.0) - v;
  let s = n * n * n;
  let x = s.x;
  let y = s.y - 4.0 * s.x;
  let z = s.z - 4.0 * s.y + 6.0 * s.x;
  let w = 6.0 - x - y - z;
  return vec4f(x, y, z, w) * (1.0 / 6.0);
}

// Bicubic texture sampling
fn textureBicubic(tex: texture_2d<f32>, samp: sampler, texCoords: vec2f, texSize: vec2f) -> vec4f {
  let invTexSize = 1.0 / texSize;
  var tc = texCoords * texSize - 0.5;

  let fxy = fract(tc);
  tc = tc - fxy;

  let xcubic = cubic(fxy.x);
  let ycubic = cubic(fxy.y);

  let c = tc.xxyy + vec4f(-0.5, 1.5, -0.5, 1.5);

  let s = vec4f(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x + ycubic.y, ycubic.z + ycubic.w);
  var offset = c + vec4f(xcubic.y, xcubic.w, ycubic.y, ycubic.w) / s;

  offset = offset * invTexSize.xxyy;

  let sample0 = textureSample(tex, samp, offset.xz);
  let sample1 = textureSample(tex, samp, offset.yz);
  let sample2 = textureSample(tex, samp, offset.xw);
  let sample3 = textureSample(tex, samp, offset.yw);

  let sx = s.x / (s.x + s.y);
  let sy = s.z / (s.z + s.w);

  return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

// Chromatic aberration with bicubic sampling
fn sampleBicubicChroma(tex: texture_2d<f32>, samp: sampler, uv: vec2f, texSize: vec2f, aberration: f32) -> vec3f {
  let offset = (uv - 0.5) * aberration * 0.02;
  let r = textureBicubic(tex, samp, uv + offset, texSize).r;
  let g = textureBicubic(tex, samp, uv, texSize).g;
  let b = textureBicubic(tex, samp, uv - offset, texSize).b;
  return vec3f(r, g, b);
}

// Standard chromatic aberration
fn chromaticAberration(tex: texture_2d<f32>, samp: sampler, uv: vec2f, strength: f32) -> vec3f {
  let offset = (uv - 0.5) * strength * u.spectralCentroid * 0.02;
  let r = textureSample(tex, samp, uv + offset).r;
  let g = textureSample(tex, samp, uv).g;
  let b = textureSample(tex, samp, uv - offset).b;
  return vec3f(r, g, b);
}

// Unsharp mask sharpening
fn sharpenWithCenter(center: vec3f, tex: texture_2d<f32>, samp: sampler, uv: vec2f, texSize: vec2f, strength: f32) -> vec3f {
  let texel = 1.0 / texSize;

  let top = textureSample(tex, samp, uv + vec2f(0.0, texel.y)).rgb;
  let bottom = textureSample(tex, samp, uv - vec2f(0.0, texel.y)).rgb;
  let left = textureSample(tex, samp, uv - vec2f(texel.x, 0.0)).rgb;
  let right = textureSample(tex, samp, uv + vec2f(texel.x, 0.0)).rgb;

  // Laplacian edge detection
  let laplacian = 4.0 * center - top - bottom - left - right;

  return center + laplacian * strength;
}

// =============================================================================
// Fragment Shader
// =============================================================================

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // Early out for silence (branchless optimization path)
  // Note: WGSL doesn't have early returns work the same way, so this is for reference
  // if (u.rms < 0.001 && u.silenceLevel > 0.99) {
  //   return textureSample(currentFrame, texSampler, input.uv);
  // }

  // No texture yet - show placeholder
  if (u.hasTexture < 0.5) {
    let color1 = vec3f(0.05, 0.05, 0.15);
    let color2 = vec3f(0.1, 0.05, 0.2);
    let t = sin(u.time * 0.5 + input.uv.x * 3.14159) * 0.5 + 0.5;
    let chromaColor = hsv2rgb(vec3f(u.dominantHue, 0.5, 0.3));
    let baseColor = mix(color1, color2, t);
    return vec4f(mix(baseColor, chromaColor, u.chromaEnergy * 0.3), 1.0);
  }

  var uv = input.uv;

  // === Spectral Displacement ===
  // Bass pushes outward, treble pulls inward (branchless)
  let center = vec2f(0.5, 0.5);
  let dir = uv - center;
  let dist = length(dir);
  let displacement = (u.bass * 0.03 - u.smoothedTreble * 0.02) * u.intensity;
  let displacedUv = uv + normalize(dir + 0.001) * displacement * dist;
  uv = mix(uv, displacedUv, u.enableSpectralDisplacement);

  // === Onset Glitch Blocks ===
  // Branchless glitch calculation
  let blockSize = 0.1 + hash(vec2f(u.glitchSeed, 0.0)) * 0.15;
  let blockCoord = floor(uv / blockSize);
  let blockRand = hash(blockCoord + u.glitchSeed);
  let glitchActive = step(0.7, blockRand) * step(0.05, u.glitchIntensity);
  let glitchOffset = vec2f(
    (hash(blockCoord + u.glitchSeed + 1.0) - 0.5) * 0.1,
    (hash(blockCoord + u.glitchSeed + 2.0) - 0.5) * 0.1
  ) * u.glitchIntensity;
  uv = mix(uv, uv + glitchOffset, glitchActive * u.enableGlitchBlocks);

  // === Original Distortion Effects ===
  let distortionStrength = u.bass * u.intensity * 0.02;
  let distortX = sin(input.uv.y * 10.0 + u.time * 2.0) * distortionStrength;
  let distortY = cos(input.uv.x * 10.0 + u.time * 2.0) * distortionStrength;
  let distortedUv = uv + vec2f(distortX, distortY);

  // Onset pulse radial distortion
  let pulseDir = uv - center;
  let pulseDist = length(pulseDir);
  let pulseWave = sin(pulseDist * 20.0 - u.time * 15.0) * u.onsetPulse * 0.01;
  let pulsedUv = distortedUv + normalize(pulseDir + 0.001) * pulseWave * step(0.01, u.onsetPulse);

  uv = mix(uv, pulsedUv, u.enableDistortion);

  // Clamp UVs
  uv = clamp(uv, vec2f(0.001), vec2f(0.999));

  // === Sample Texture ===
  let aberrationStrength = u.spectralCentroid * u.intensity;
  var currentColor: vec3f;
  var previousColor: vec3f;

  // Bicubic vs bilinear (branchless mix)
  let bicubicCurrent = sampleBicubicChroma(currentFrame, texSampler, uv, u.textureSize, aberrationStrength);
  let bicubicPrevious = sampleBicubicChroma(previousFrame, prevSampler, uv, u.textureSize, aberrationStrength * 0.5);
  let bilinearCurrent = chromaticAberration(currentFrame, texSampler, uv, aberrationStrength);
  let bilinearPrevious = chromaticAberration(previousFrame, prevSampler, uv, aberrationStrength * 0.5);

  currentColor = mix(bilinearCurrent, bicubicCurrent, u.enableBicubic);
  previousColor = mix(bilinearPrevious, bicubicPrevious, u.enableBicubic);

  // Blend frames
  var color = mix(previousColor, currentColor, u.blendFactor);

  // === Sharpening ===
  let dynamicSharpen = clamp(u.sharpenStrength + u.onsetPulse * 0.3 + u.beatFlash * 0.2, 0.0, 1.5);
  let sharpCurrent = sharpenWithCenter(currentColor, currentFrame, texSampler, uv, u.textureSize, dynamicSharpen * u.intensity);
  let sharpPrevious = sharpenWithCenter(previousColor, previousFrame, prevSampler, uv, u.textureSize, dynamicSharpen * u.intensity * 0.5);
  let sharpenedColor = mix(sharpPrevious, sharpCurrent, u.blendFactor);
  color = mix(color, sharpenedColor, u.enableSharpening);

  // === Silence Degradation ===
  let sl = u.silenceLevel;
  if (sl > 0.01) {
    // Block noise
    let blockSizeDeg = 8.0 + sl * 24.0;
    let blockCoordDeg = floor(input.uv * u.textureSize / blockSizeDeg);
    let blockNoiseDeg = hash(blockCoordDeg + u.degradationSeed);
    let blockShift = vec3f(
      hash(blockCoordDeg + u.degradationSeed + 1.0) - 0.5,
      hash(blockCoordDeg + u.degradationSeed + 2.0) - 0.5,
      hash(blockCoordDeg + u.degradationSeed + 3.0) - 0.5
    ) * sl * 0.3;
    color = mix(color, color + blockShift, step(1.0 - sl * 0.4, blockNoiseDeg) * step(0.3, sl));

    // Scanline distortion
    let scanlinePhase = input.uv.y * u.textureSize.y + u.time * 20.0 + u.degradationSeed * 100.0;
    var scanlineNoise = sin(scanlinePhase * 3.14159) * 0.5 + 0.5;
    scanlineNoise = pow(scanlineNoise, 4.0);
    color = mix(color, color * (1.0 - scanlineNoise * 0.3), sl * 0.5);

    // Static grain
    var staticNoise = noise(input.uv * 800.0 + u.time * 100.0 + u.degradationSeed * 50.0);
    staticNoise = (staticNoise - 0.5) * 2.0;
    color = color + vec3f(staticNoise * sl * 0.15);

    // Color banding (posterization)
    let levels = max(32.0 - sl * 24.0, 4.0);
    let bandedColor = floor(color * levels) / levels;
    color = mix(color, bandedColor, step(0.2, sl));

    // Enhanced vignette
    let vignetteRadius = 1.5 - sl * 0.8;
    let vignetteDeg = 1.0 - smoothstep(0.3, vignetteRadius, length(input.uv - 0.5) * 2.0);
    color = color * mix(1.0, vignetteDeg, sl * 0.6);

    // Glitch lines
    let glitchLineDeg = hash(vec2f(floor(input.uv.y * 50.0), u.degradationSeed));
    let lineOffset = (hash(vec2f(glitchLineDeg, u.degradationSeed)) - 0.5) * 0.1 * sl;
    let glitchUvDeg = clamp(vec2f(input.uv.x + lineOffset, input.uv.y), vec2f(0.001), vec2f(0.999));
    let glitchColorDeg = textureSample(currentFrame, texSampler, glitchUvDeg).rgb;
    color = mix(color, mix(color, glitchColorDeg, 0.7), step(1.0 - sl * 0.15, glitchLineDeg) * step(0.4, sl));

    // Chromatic separation
    let chromaSep = sl * 0.015;
    let sepR = textureSample(currentFrame, texSampler, input.uv + vec2f(chromaSep, 0.0)).r;
    let sepB = textureSample(currentFrame, texSampler, input.uv - vec2f(chromaSep, 0.0)).b;
    color = mix(color, vec3f(sepR, color.g, sepB), step(0.25, sl));

    // Desaturation
    let gray = vec3f(dot(color, vec3f(0.299, 0.587, 0.114)));
    color = mix(color, gray, sl * 0.4);
  }

  // === Beat Flash ===
  color = color + vec3f(u.beatFlash * u.intensity * 0.3 * u.enableFlash);

  // === Onset Pulse Brightness ===
  color = color + vec3f(u.onsetPulse * u.intensity * 0.15);

  // === Treble Grain ===
  let grainScale = 400.0;
  var grain = noise(input.uv * grainScale + u.time * 50.0);
  grain = (grain - 0.5) * 2.0;
  let grainStrength = u.smoothedTreble * u.intensity * 0.08;
  color = color + vec3f(grain * grainStrength * u.enableTrebleGrain * step(0.1, u.smoothedTreble));

  // === Vignette ===
  let vignetteStrength = 0.3 + (1.0 - u.bass) * 0.2;
  let vignette = 1.0 - smoothstep(0.5, 1.5, length(input.uv - 0.5) * 2.0);
  color = color * mix(1.0, vignette, vignetteStrength);

  // === Color Boost (Saturation) ===
  let saturationBoost = 1.0 + u.bass * u.intensity * 0.3;
  let grayVal = vec3f(dot(color, vec3f(0.299, 0.587, 0.114)));
  color = mix(grayVal, color, saturationBoost);

  // === Chroma Tinting ===
  let chromaTint = hsv2rgb(vec3f(u.dominantHue, 0.3, 1.0));
  let luminanceVal = dot(color, vec3f(0.299, 0.587, 0.114));
  let tintedColor = mix(color, color * chromaTint, u.chromaEnergy * 0.15);
  color = mix(color, tintedColor, step(0.5, luminanceVal) * step(0.1, u.chromaEnergy));

  // === Spectral Centroid Contrast ===
  let contrast = 0.9 + u.spectralCentroid * 0.2;
  color = (color - 0.5) * contrast + 0.5;

  // Final clamp
  color = clamp(color, vec3f(0.0), vec3f(1.0));

  return vec4f(color, 1.0);
}
