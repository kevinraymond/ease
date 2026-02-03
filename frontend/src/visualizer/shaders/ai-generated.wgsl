/**
 * AI Generated Mode Shader (WebGPU/WGSL)
 *
 * Fullscreen quad shader with SOTA audio-reactive effects.
 * Ported from the Three.js GLSL shader for native WebGPU rendering.
 */

// =============================================================================
// Uniform Buffer
// =============================================================================

struct Uniforms {
  // Texture/frame parameters
  blendFactor: f32,
  hasTexture: f32,  // bool as f32

  // Audio metrics
  bass: f32,
  mid: f32,
  treble: f32,
  rms: f32,
  beatFlash: f32,
  onsetPulse: f32,
  onsetStrength: f32,
  spectralCentroid: f32,
  dominantHue: f32,
  chromaEnergy: f32,

  // Effect toggles (as f32: 0.0 or 1.0)
  enableDistortion: f32,
  enableFlash: f32,
  enableSpectralDisplacement: f32,
  enableGlitchBlocks: f32,
  enableTrebleGrain: f32,

  // Effect parameters
  glitchSeed: f32,
  glitchIntensity: f32,
  smoothedTreble: f32,

  // Upscaling
  enableBicubic: f32,
  enableSharpening: f32,
  sharpenStrength: f32,
  textureWidth: f32,
  textureHeight: f32,

  // Silence degradation
  silenceLevel: f32,
  degradationSeed: f32,

  // General
  time: f32,
  intensity: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// Texture bindings (slot 0 = current frame, slot 1 = previous frame)
@group(0) @binding(1) var currentFrameSampler: sampler;
@group(0) @binding(2) var currentFrame: texture_2d<f32>;
@group(0) @binding(3) var previousFrameSampler: sampler;
@group(0) @binding(4) var previousFrame: texture_2d<f32>;

// =============================================================================
// Vertex Shader
// =============================================================================

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) normalizedUv: vec2<f32>,  // UV in -1 to 1 range
  @location(2) vignetteBase: f32,        // Distance from center for vignette
  @location(3) centerDir: vec2<f32>,     // Direction from center (normalized)
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  // Fullscreen quad vertices (2 triangles)
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0)
  );

  var uvs = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0)
  );

  var output: VertexOutput;
  output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
  output.uv = uvs[vertexIndex];

  // Precompute frequently used values
  output.normalizedUv = output.uv * 2.0 - 1.0;
  let centerOffset = output.uv - 0.5;
  output.vignetteBase = length(centerOffset) * 2.0;
  output.centerDir = normalize(centerOffset + 0.001);

  return output;
}

// =============================================================================
// Fragment Shader Helper Functions
// =============================================================================

// Convert HSV to RGB
fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
  let K = vec4<f32>(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  let p = abs(fract(vec3<f32>(c.x, c.x, c.x) + vec3<f32>(K.x, K.y, K.z)) * 6.0 - vec3<f32>(K.w, K.w, K.w));
  return c.z * mix(vec3<f32>(K.x, K.x, K.x), clamp(p - vec3<f32>(K.x, K.x, K.x), vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

// Hash function for pseudo-random values
fn hash(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// 2D noise for grain effect
fn noise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let a = hash(i);
  let b = hash(i + vec2<f32>(1.0, 0.0));
  let c = hash(i + vec2<f32>(0.0, 1.0));
  let d = hash(i + vec2<f32>(1.0, 1.0));
  let uu = f * f * (3.0 - 2.0 * f);
  return mix(a, b, uu.x) + (c - a) * uu.y * (1.0 - uu.x) + (d - b) * uu.x * uu.y;
}

// Cubic interpolation weight function (Catmull-Rom spline)
fn cubic(v: f32) -> vec4<f32> {
  let n = vec4<f32>(1.0, 2.0, 3.0, 4.0) - v;
  let s = n * n * n;
  let x = s.x;
  let y = s.y - 4.0 * s.x;
  let z = s.z - 4.0 * s.y + 6.0 * s.x;
  let w = 6.0 - x - y - z;
  return vec4<f32>(x, y, z, w) * (1.0 / 6.0);
}

// Get texture size as vec2
fn getTextureSize() -> vec2<f32> {
  return vec2<f32>(u.textureWidth, u.textureHeight);
}

// Bicubic texture sampling - provides sharper edges than bilinear
fn textureBicubic(tex: texture_2d<f32>, texSampler: sampler, texCoords: vec2<f32>, texSize: vec2<f32>) -> vec4<f32> {
  let invTexSize = 1.0 / texSize;
  var coords = texCoords * texSize - 0.5;

  let fxy = fract(coords);
  coords = coords - fxy;

  let xcubic = cubic(fxy.x);
  let ycubic = cubic(fxy.y);

  let c = vec4<f32>(coords.x - 0.5, coords.x + 1.5, coords.y - 0.5, coords.y + 1.5);

  let s = vec4<f32>(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x + ycubic.y, ycubic.z + ycubic.w);
  var offset = c + vec4<f32>(xcubic.y, xcubic.w, ycubic.y, ycubic.w) / s;

  offset = offset * vec4<f32>(invTexSize.x, invTexSize.x, invTexSize.y, invTexSize.y);

  let sample0 = textureSample(tex, texSampler, vec2<f32>(offset.x, offset.z));
  let sample1 = textureSample(tex, texSampler, vec2<f32>(offset.y, offset.z));
  let sample2 = textureSample(tex, texSampler, vec2<f32>(offset.x, offset.w));
  let sample3 = textureSample(tex, texSampler, vec2<f32>(offset.y, offset.w));

  let sx = s.x / (s.x + s.y);
  let sy = s.z / (s.z + s.w);

  return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

// Bicubic sampling with chromatic aberration
fn sampleBicubicChroma(tex: texture_2d<f32>, texSampler: sampler, uv: vec2<f32>, texSize: vec2<f32>, aberration: f32) -> vec3<f32> {
  let offset = (uv - 0.5) * aberration * 0.02;
  let r = textureBicubic(tex, texSampler, uv + offset, texSize).r;
  let g = textureBicubic(tex, texSampler, uv, texSize).g;
  let b = textureBicubic(tex, texSampler, uv - offset, texSize).b;
  return vec3<f32>(r, g, b);
}

// Chromatic aberration based on spectral centroid (bilinear sampling)
fn chromaticAberration(tex: texture_2d<f32>, texSampler: sampler, uv: vec2<f32>, strength: f32) -> vec3<f32> {
  let offset = (uv - 0.5) * strength * u.spectralCentroid * 0.02;
  let r = textureSample(tex, texSampler, uv + offset).r;
  let g = textureSample(tex, texSampler, uv).g;
  let b = textureSample(tex, texSampler, uv - offset).b;
  return vec3<f32>(r, g, b);
}

// Unsharp mask sharpening - enhances edges
// Takes pre-sampled center color to work with bicubic
fn sharpenWithCenter(center: vec3<f32>, tex: texture_2d<f32>, texSampler: sampler, uv: vec2<f32>, texSize: vec2<f32>, strength: f32) -> vec3<f32> {
  let texel = 1.0 / texSize;

  // Sample neighbors (bilinear is fine for edge detection)
  let top = textureSample(tex, texSampler, uv + vec2<f32>(0.0, texel.y)).rgb;
  let bottom = textureSample(tex, texSampler, uv - vec2<f32>(0.0, texel.y)).rgb;
  let left = textureSample(tex, texSampler, uv - vec2<f32>(texel.x, 0.0)).rgb;
  let right = textureSample(tex, texSampler, uv + vec2<f32>(texel.x, 0.0)).rgb;

  // Laplacian edge detection
  let laplacian = 4.0 * center - top - bottom - left - right;

  // Add edge enhancement
  return center + laplacian * strength;
}

// =============================================================================
// Fragment Shader
// =============================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let textureSize = getTextureSize();

  // === EARLY-OUT FOR SILENCE ===
  // When completely silent, skip all effects for performance
  if (u.rms < 0.001 && u.silenceLevel > 0.99 && u.hasTexture > 0.5) {
    return textureSample(currentFrame, currentFrameSampler, in.uv);
  }

  // === NO TEXTURE - PLACEHOLDER GRADIENT ===
  if (u.hasTexture < 0.5) {
    let color1 = vec3<f32>(0.05, 0.05, 0.15);
    let color2 = vec3<f32>(0.1, 0.05, 0.2);
    let t = sin(u.time * 0.5 + in.uv.x * 3.14159) * 0.5 + 0.5;

    // Add chroma-based color accent
    let chromaColor = hsv2rgb(vec3<f32>(u.dominantHue, 0.5, 0.3));
    let baseColor = mix(color1, color2, t);
    return vec4<f32>(mix(baseColor, chromaColor, u.chromaEnergy * 0.3), 1.0);
  }

  var uv = in.uv;
  let dist = length(in.uv - 0.5);

  // === SPECTRAL DISPLACEMENT (Branchless) ===
  // Bass pushes outward, treble pulls inward
  let displacement = (u.bass * 0.03 - u.smoothedTreble * 0.02) * u.intensity;
  let displacedUv = uv + in.centerDir * displacement * dist;
  // Branchless conditional: mix between original and displaced based on enable flag
  uv = mix(uv, displacedUv, u.enableSpectralDisplacement);

  // === ONSET GLITCH BLOCKS (Branchless) ===
  let blockSize = 0.1 + hash(vec2<f32>(u.glitchSeed, 0.0)) * 0.15;
  let blockCoord = floor(uv / blockSize);
  let blockRand = hash(blockCoord + u.glitchSeed);
  // Branchless: compute offset always, multiply by enable conditions
  let glitchOffset = vec2<f32>(
    (hash(blockCoord + u.glitchSeed + 1.0) - 0.5) * 0.1,
    (hash(blockCoord + u.glitchSeed + 2.0) - 0.5) * 0.1
  ) * u.glitchIntensity;
  // Apply only if enabled, intensity > threshold, and blockRand > 0.7
  let glitchEnable = u.enableGlitchBlocks * step(0.05, u.glitchIntensity) * step(0.7, blockRand);
  uv = uv + glitchOffset * glitchEnable;

  // === DISTORTION EFFECTS (Branchless) ===
  let distortionStrength = u.bass * u.intensity * 0.02;
  let distortX = sin(in.uv.y * 10.0 + u.time * 2.0) * distortionStrength;
  let distortY = cos(in.uv.x * 10.0 + u.time * 2.0) * distortionStrength;
  var distortedUv = uv + vec2<f32>(distortX, distortY);

  // Onset pulse - radial distortion on transients
  let pulseDist = length(distortedUv - 0.5);
  let pulseWave = sin(pulseDist * 20.0 - u.time * 15.0) * u.onsetPulse * 0.01;
  let pulseDir = normalize(distortedUv - 0.5 + 0.001);
  distortedUv = distortedUv + pulseDir * pulseWave * step(0.01, u.onsetPulse);

  // Branchless mix for distortion enable
  uv = mix(uv, distortedUv, u.enableDistortion);

  // Clamp UVs to prevent wrapping
  uv = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));

  // === TEXTURE SAMPLING (Uniform branch on enableBicubic - this is allowed!) ===
  let aberrationStrength = u.spectralCentroid * u.intensity;
  var currentColor: vec3<f32>;
  var previousColor: vec3<f32>;

  // Branch on uniform is allowed in WGSL - only per-fragment branches are forbidden
  if (u.enableBicubic > 0.5) {
    currentColor = sampleBicubicChroma(currentFrame, currentFrameSampler, uv, textureSize, aberrationStrength);
    previousColor = sampleBicubicChroma(previousFrame, previousFrameSampler, uv, textureSize, aberrationStrength * 0.5);
  } else {
    currentColor = chromaticAberration(currentFrame, currentFrameSampler, uv, aberrationStrength);
    previousColor = chromaticAberration(previousFrame, previousFrameSampler, uv, aberrationStrength * 0.5);
  }

  // Blend between frames
  var color = mix(previousColor, currentColor, u.blendFactor);

  // === SHARPENING (Uniform branch - allowed!) ===
  if (u.enableSharpening > 0.5) {
    let dynamicSharpen = clamp(u.sharpenStrength + u.onsetPulse * 0.3 + u.beatFlash * 0.2, 0.0, 1.5);
    let sharpCurrent = sharpenWithCenter(currentColor, currentFrame, currentFrameSampler, uv, textureSize, dynamicSharpen * u.intensity);
    let sharpPrevious = sharpenWithCenter(previousColor, previousFrame, previousFrameSampler, uv, textureSize, dynamicSharpen * u.intensity * 0.5);
    color = mix(sharpPrevious, sharpCurrent, u.blendFactor);
  }

  // === SILENCE DEGRADATION EFFECTS ===
  // Uniform branch on silenceLevel - this IS allowed in WGSL!
  // Only per-fragment branches (like on UV or hash results) are forbidden for textureSample
  if (u.silenceLevel > 0.01) {
    let sl = u.silenceLevel;

    // Pre-sample textures for glitch effects (must be outside per-fragment branches)
    let glitchLine = hash(vec2<f32>(floor(in.uv.y * 50.0), u.degradationSeed));
    let lineOffset = (hash(vec2<f32>(glitchLine, u.degradationSeed)) - 0.5) * 0.1 * sl;
    let glitchUv = clamp(vec2<f32>(in.uv.x + lineOffset, in.uv.y), vec2<f32>(0.001), vec2<f32>(0.999));
    let glitchColor = textureSample(currentFrame, currentFrameSampler, glitchUv).rgb;

    // Pre-sample for chromatic aberration
    let chromaSep = sl * 0.015;
    let chromaR = textureSample(currentFrame, currentFrameSampler, in.uv + vec2<f32>(chromaSep, 0.0)).r;
    let chromaB = textureSample(currentFrame, currentFrameSampler, in.uv - vec2<f32>(chromaSep, 0.0)).b;

    // 1. Block noise (JPEG-artifact-like rectangular blocks) - Branchless per-fragment
    let blockSizeDeg = 8.0 + sl * 24.0;
    let blockCoordDeg = floor(in.uv * textureSize / blockSizeDeg);
    let blockNoise = hash(blockCoordDeg + u.degradationSeed);
    let blockShift = vec3<f32>(
      hash(blockCoordDeg + u.degradationSeed + 1.0) - 0.5,
      hash(blockCoordDeg + u.degradationSeed + 2.0) - 0.5,
      hash(blockCoordDeg + u.degradationSeed + 3.0) - 0.5
    ) * sl * 0.3;
    let blockApply = step(0.3, sl) * step(1.0 - sl * 0.4, blockNoise);
    color = color + blockShift * blockApply;

    // 2. Scanline distortion (VHS aesthetic)
    let scanlinePhase = in.uv.y * textureSize.y + u.time * 20.0 + u.degradationSeed * 100.0;
    var scanlineNoise = sin(scanlinePhase * 3.14159) * 0.5 + 0.5;
    scanlineNoise = pow(scanlineNoise, 4.0);
    color = mix(color, color * (1.0 - scanlineNoise * 0.3), sl * 0.5);

    // 3. Static/grain overlay
    var staticNoise = noise(in.uv * 800.0 + u.time * 100.0 + u.degradationSeed * 50.0);
    staticNoise = (staticNoise - 0.5) * 2.0;
    color = color + vec3<f32>(staticNoise * sl * 0.15);

    // 4. Color banding (posterization effect) - Uniform branch OK
    if (sl > 0.2) {
      var levels = 32.0 - sl * 24.0;
      levels = max(levels, 4.0);
      color = floor(color * levels) / levels;
    }

    // 5. Enhanced vignette
    let vignetteRadius = 1.5 - sl * 0.8;
    let vignetteDeg = 1.0 - smoothstep(0.3, vignetteRadius, length(in.uv - 0.5) * 2.0);
    color = color * mix(1.0, vignetteDeg, sl * 0.6);

    // 6. Occasional glitch lines - Branchless per-fragment (texture already sampled above)
    let glitchApply = step(0.4, sl) * step(1.0 - sl * 0.15, glitchLine);
    color = mix(color, glitchColor, glitchApply * 0.7);

    // 7. Color channel separation (textures already sampled above) - Uniform branch OK
    if (sl > 0.25) {
      color.r = chromaR;
      color.b = chromaB;
    }

    // 8. Overall desaturation
    let gray = vec3<f32>(dot(color, vec3<f32>(0.299, 0.587, 0.114)));
    color = mix(color, gray, sl * 0.4);
  } // End silence degradation

  // === BEAT FLASH (Branchless) ===
  // Add brightness when flash is enabled
  color = color + vec3<f32>(u.beatFlash * u.intensity * 0.3 * u.enableFlash);

  // Onset pulse brightness (always active)
  color = color + vec3<f32>(u.onsetPulse * u.intensity * 0.15);

  // === TREBLE GRAIN/SHIMMER (Branchless) ===
  let grainScale = 400.0;
  var grain = noise(in.uv * grainScale + u.time * 50.0);
  grain = (grain - 0.5) * 2.0;
  let grainStrength = u.smoothedTreble * u.intensity * 0.08;
  // Branchless: multiply by enable flag and threshold step
  color = color + vec3<f32>(grain * grainStrength * u.enableTrebleGrain * step(0.1, u.smoothedTreble));

  // === VIGNETTE (using precomputed vignetteBase) ===
  let vignetteStrength = 0.3 + (1.0 - u.bass) * 0.2;
  let vignette = 1.0 - smoothstep(0.5, 1.5, in.vignetteBase);
  color = color * mix(1.0, vignette, vignetteStrength);

  // === SATURATION BOOST (always active) ===
  let saturationBoost = 1.0 + u.bass * u.intensity * 0.3;
  let grayColor = vec3<f32>(dot(color, vec3<f32>(0.299, 0.587, 0.114)));
  color = mix(grayColor, color, saturationBoost);

  // === CHROMA TINTING (Branchless) ===
  let chromaTint = hsv2rgb(vec3<f32>(u.dominantHue, 0.3, 1.0));
  let luminance = dot(color, vec3<f32>(0.299, 0.587, 0.114));
  let tintedColor = mix(color, color * chromaTint, u.chromaEnergy * 0.15);
  // Branchless: apply tint only when chroma > 0.1 and luminance > 0.5
  color = mix(color, tintedColor, step(0.1, u.chromaEnergy) * step(0.5, luminance));

  // === SPECTRAL CENTROID CONTRAST (always active) ===
  let contrast = 0.9 + u.spectralCentroid * 0.2;
  color = (color - 0.5) * contrast + 0.5;

  // Final clamp
  color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

  return vec4<f32>(color, 1.0);
}
