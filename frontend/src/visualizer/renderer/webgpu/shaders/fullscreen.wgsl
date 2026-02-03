/**
 * Fullscreen Quad Shader
 *
 * Provides vertex shader for fullscreen rendering and common utilities.
 * Used as a base for post-processing effects.
 */

// Vertex output / Fragment input
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

// Fullscreen triangle vertex shader
// Uses vertex ID to generate positions for a fullscreen triangle
// No vertex buffer needed - more efficient than quad
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;

  // Generate fullscreen triangle positions
  // Triangle covers: (-1,-1), (3,-1), (-1,3)
  // This covers the entire clip space with a single triangle
  let x = f32((vertexIndex & 1u) << 2u) - 1.0;
  let y = f32((vertexIndex & 2u) << 1u) - 1.0;

  output.position = vec4f(x, y, 0.0, 1.0);

  // UV coordinates: (0,1), (2,1), (0,-1) -> clipped to (0,0)-(1,1)
  output.uv = vec2f(
    (x + 1.0) * 0.5,
    (1.0 - y) * 0.5  // Flip Y for texture coordinates
  );

  return output;
}

// Alternative: Fullscreen quad vertex shader (6 vertices, 2 triangles)
@vertex
fn vs_quad(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;

  // Quad vertices
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
// Common Utility Functions
// =============================================================================

// Hash function for pseudo-random values
fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

// 2D noise
fn noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let a = hash(i);
  let b = hash(i + vec2f(1.0, 0.0));
  let c = hash(i + vec2f(0.0, 1.0));
  let d = hash(i + vec2f(1.0, 1.0));
  let u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// HSV to RGB conversion
fn hsv2rgb(c: vec3f) -> vec3f {
  let K = vec4f(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

// RGB to HSV conversion
fn rgb2hsv(c: vec3f) -> vec3f {
  let K = vec4f(0.0, -1.0/3.0, 2.0/3.0, -1.0);
  let p = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), step(c.b, c.g));
  let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), step(p.x, c.r));
  let d = q.x - min(q.w, q.y);
  let e = 1.0e-10;
  return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Cubic interpolation weights (Catmull-Rom)
fn cubic(v: f32) -> vec4f {
  let n = vec4f(1.0, 2.0, 3.0, 4.0) - v;
  let s = n * n * n;
  let x = s.x;
  let y = s.y - 4.0 * s.x;
  let z = s.z - 4.0 * s.y + 6.0 * s.x;
  let w = 6.0 - x - y - z;
  return vec4f(x, y, z, w) * (1.0 / 6.0);
}

// Lanczos-2 weight for sharp upscaling
fn lanczos2(x: f32) -> f32 {
  if (x == 0.0) { return 1.0; }
  if (abs(x) >= 2.0) { return 0.0; }
  let pi = 3.14159265359;
  let pix = pi * x;
  return sin(pix) * sin(pix * 0.5) / (pix * pix * 0.5);
}

// Smoothstep
fn smoothstep_f(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Luminance
fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

// Simple passthrough fragment shader (for testing)
@fragment
fn fs_passthrough(input: VertexOutput) -> @location(0) vec4f {
  return vec4f(input.uv, 0.0, 1.0);
}
