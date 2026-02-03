/**
 * AMD FidelityFX Super Resolution 1.0 (FSR) - WGSL Port
 *
 * Two-pass spatial upscaling algorithm:
 * 1. EASU (Edge-Adaptive Spatial Upsampling) - Main upscaling with edge detection
 * 2. RCAS (Robust Contrast Adaptive Sharpening) - Sharpening pass
 */

// =============================================================================
// EASU Pass - Edge-Adaptive Spatial Upsampling
// =============================================================================

struct EASUUniforms {
  inputSize: vec2f,
  outputSize: vec2f,
}

@group(0) @binding(0) var<uniform> easuUniforms: EASUUniforms;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

// Vertex shader for fullscreen quad
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;

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

// Lanczos-2 kernel for sharp upscaling
fn lanczos2(x: f32) -> f32 {
  if (x == 0.0) { return 1.0; }
  if (abs(x) >= 2.0) { return 0.0; }
  let pi = 3.14159265359;
  let pix = pi * x;
  return sin(pix) * sin(pix * 0.5) / (pix * pix * 0.5);
}

// Sample with Lanczos-2 filter
fn sampleLanczos(tex: texture_2d<f32>, samp: sampler, uv: vec2f, texelSize: vec2f, inputSize: vec2f) -> vec4f {
  // Convert UV to pixel coordinates
  let pixelCoord = uv * inputSize - 0.5;
  let centerFloor = floor(pixelCoord);
  let f = pixelCoord - centerFloor;

  var color = vec4f(0.0);
  var totalWeight = 0.0;

  // 4x4 kernel
  for (var y = -1; y <= 2; y++) {
    for (var x = -1; x <= 2; x++) {
      let offset = vec2f(f32(x), f32(y));
      var samplePos = (centerFloor + offset + 0.5) * texelSize;
      samplePos = clamp(samplePos, vec2f(0.0), vec2f(1.0));

      let wx = lanczos2(f32(x) - f.x);
      let wy = lanczos2(f32(y) - f.y);
      let weight = wx * wy;

      color = color + textureSample(tex, samp, samplePos) * weight;
      totalWeight = totalWeight + weight;
    }
  }

  return color / totalWeight;
}

// Sobel edge detection
fn detectEdge(tex: texture_2d<f32>, samp: sampler, uv: vec2f, texelSize: vec2f) -> f32 {
  let tl = dot(textureSample(tex, samp, uv + vec2f(-1.0, -1.0) * texelSize).rgb, vec3f(0.299, 0.587, 0.114));
  let t  = dot(textureSample(tex, samp, uv + vec2f( 0.0, -1.0) * texelSize).rgb, vec3f(0.299, 0.587, 0.114));
  let tr = dot(textureSample(tex, samp, uv + vec2f( 1.0, -1.0) * texelSize).rgb, vec3f(0.299, 0.587, 0.114));
  let l  = dot(textureSample(tex, samp, uv + vec2f(-1.0,  0.0) * texelSize).rgb, vec3f(0.299, 0.587, 0.114));
  let r  = dot(textureSample(tex, samp, uv + vec2f( 1.0,  0.0) * texelSize).rgb, vec3f(0.299, 0.587, 0.114));
  let bl = dot(textureSample(tex, samp, uv + vec2f(-1.0,  1.0) * texelSize).rgb, vec3f(0.299, 0.587, 0.114));
  let b  = dot(textureSample(tex, samp, uv + vec2f( 0.0,  1.0) * texelSize).rgb, vec3f(0.299, 0.587, 0.114));
  let br = dot(textureSample(tex, samp, uv + vec2f( 1.0,  1.0) * texelSize).rgb, vec3f(0.299, 0.587, 0.114));

  let gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
  let gy = -tl - 2.0*t - tr + bl + 2.0*b + br;

  return sqrt(gx*gx + gy*gy);
}

// EASU fragment shader
@fragment
fn fs_easu(input: VertexOutput) -> @location(0) vec4f {
  let texelSize = 1.0 / easuUniforms.inputSize;

  // Detect edges
  let edge = detectEdge(inputTexture, inputSampler, input.uv, texelSize);

  // Use Lanczos for edges, bilinear for smooth areas
  let lanczosColor = sampleLanczos(inputTexture, inputSampler, input.uv, texelSize, easuUniforms.inputSize);
  let bilinearColor = textureSample(inputTexture, inputSampler, input.uv);

  // Blend based on edge strength
  let edgeWeight = smoothstep(0.05, 0.2, edge);
  let color = mix(bilinearColor, lanczosColor, edgeWeight);

  return color;
}

// =============================================================================
// RCAS Pass - Robust Contrast Adaptive Sharpening
// =============================================================================

struct RCASUniforms {
  textureSize: vec2f,
  sharpness: f32,
  _pad: f32,
}

@group(0) @binding(0) var<uniform> rcasUniforms: RCASUniforms;
// Bindings 1-2 are same as EASU (sampler and texture)

// RCAS fragment shader
@fragment
fn fs_rcas(input: VertexOutput) -> @location(0) vec4f {
  let texelSize = 1.0 / rcasUniforms.textureSize;

  // Sample cross pattern
  let n = textureSample(inputTexture, inputSampler, input.uv + vec2f(0.0, -texelSize.y)).rgb;
  let w = textureSample(inputTexture, inputSampler, input.uv + vec2f(-texelSize.x, 0.0)).rgb;
  let c = textureSample(inputTexture, inputSampler, input.uv).rgb;
  let e = textureSample(inputTexture, inputSampler, input.uv + vec2f(texelSize.x, 0.0)).rgb;
  let s = textureSample(inputTexture, inputSampler, input.uv + vec2f(0.0, texelSize.y)).rgb;

  // Compute local contrast
  let minNeighbor = min(min(min(n, w), min(e, s)), c);
  let maxNeighbor = max(max(max(n, w), max(e, s)), c);

  // Soft min/max for smoother results
  let softMin = (minNeighbor + c) * 0.5;
  let softMax = (maxNeighbor + c) * 0.5;

  // Sharpening weight based on local contrast
  let contrast = maxNeighbor - minNeighbor;
  let sharpWeight = clamp(contrast * rcasUniforms.sharpness, vec3f(0.0), vec3f(1.0));

  // Apply sharpening
  let neighborAvg = (n + w + e + s) * 0.25;
  var sharpened = c + (c - neighborAvg) * sharpWeight * rcasUniforms.sharpness;

  // Clamp to prevent halo artifacts
  sharpened = clamp(sharpened, softMin, softMax);

  return vec4f(sharpened, 1.0);
}

// =============================================================================
// Combined Single-Pass FSR (Optional - for simpler pipelines)
// =============================================================================

struct FSRUniforms {
  inputSize: vec2f,
  outputSize: vec2f,
  sharpness: f32,
  _pad: vec3f,
}

@group(0) @binding(0) var<uniform> fsrUniforms: FSRUniforms;

// Combined FSR (EASU + RCAS in one pass - slightly lower quality but faster)
@fragment
fn fs_fsr_combined(input: VertexOutput) -> @location(0) vec4f {
  let texelSize = 1.0 / fsrUniforms.inputSize;

  // EASU: Edge-adaptive upsampling
  let edge = detectEdge(inputTexture, inputSampler, input.uv, texelSize);
  let lanczosColor = sampleLanczos(inputTexture, inputSampler, input.uv, texelSize, fsrUniforms.inputSize);
  let bilinearColor = textureSample(inputTexture, inputSampler, input.uv);
  let edgeWeight = smoothstep(0.05, 0.2, edge);
  let upscaled = mix(bilinearColor.rgb, lanczosColor.rgb, edgeWeight);

  // RCAS: Adaptive sharpening
  let n = textureSample(inputTexture, inputSampler, input.uv + vec2f(0.0, -texelSize.y)).rgb;
  let w = textureSample(inputTexture, inputSampler, input.uv + vec2f(-texelSize.x, 0.0)).rgb;
  let e = textureSample(inputTexture, inputSampler, input.uv + vec2f(texelSize.x, 0.0)).rgb;
  let s = textureSample(inputTexture, inputSampler, input.uv + vec2f(0.0, texelSize.y)).rgb;

  let minN = min(min(min(n, w), min(e, s)), upscaled);
  let maxN = max(max(max(n, w), max(e, s)), upscaled);
  let softMin = (minN + upscaled) * 0.5;
  let softMax = (maxN + upscaled) * 0.5;

  let contrast = maxN - minN;
  let sharpWeight = clamp(contrast * fsrUniforms.sharpness, vec3f(0.0), vec3f(1.0));
  let neighborAvg = (n + w + e + s) * 0.25;
  var sharpened = upscaled + (upscaled - neighborAvg) * sharpWeight * fsrUniforms.sharpness;
  sharpened = clamp(sharpened, softMin, softMax);

  return vec4f(sharpened, 1.0);
}
