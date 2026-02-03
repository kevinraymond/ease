export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function map(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function normalizeArray(arr: Uint8Array | Float32Array): number[] {
  const max = Math.max(...arr);
  if (max === 0) return Array.from(arr).map(() => 0);
  return Array.from(arr).map((v) => v / max);
}

export function downsample(arr: Uint8Array | Float32Array, targetLength: number): number[] {
  const result: number[] = [];
  const step = arr.length / targetLength;

  for (let i = 0; i < targetLength; i++) {
    const start = Math.floor(i * step);
    const end = Math.floor((i + 1) * step);

    let sum = 0;
    for (let j = start; j < end; j++) {
      sum += arr[j];
    }
    result.push(sum / (end - start));
  }

  return result;
}
