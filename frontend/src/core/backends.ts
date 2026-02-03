/**
 * Backend definitions and capability management.
 * Allows configuration when disconnected by providing default backend info.
 */

import { BackendInfo } from './types';

/**
 * Default backends matching server capabilities.
 * This allows the UI to show backend-specific sections even when disconnected.
 */
export const DEFAULT_BACKENDS: BackendInfo[] = [
  {
    id: 'audio_reactive',
    name: 'Audio Reactive',
    description: 'Optimized for real-time audio-reactive generation',
    capabilities: ['taesd', 'seed_control', 'strength_control'],
    fps_range: [6, 12],
  },
  {
    id: 'stream_diffusion',
    name: 'StreamDiffusion',
    description: 'High-performance streaming with advanced features',
    capabilities: [
      'controlnet',
      'lora',
      'taesd',
      'temporal_coherence',
      'acceleration',
    ],
    fps_range: [5, 20],
  },
  {
    id: 'flux_klein',
    name: 'FLUX Klein',
    description: 'FLUX model for high-quality generation',
    capabilities: ['prompt_modulation'],
    fps_range: [1, 3],
  },
];

const STORAGE_KEY = 'ease-selected-backend';

/**
 * Get backend info by ID from the default list.
 */
export function getDefaultBackendById(id: string): BackendInfo | undefined {
  return DEFAULT_BACKENDS.find((b) => b.id === id);
}

/**
 * Get the stored selected backend ID, or return the first default.
 */
export function getStoredBackendId(): string {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored && DEFAULT_BACKENDS.some((b) => b.id === stored)) {
    return stored;
  }
  return DEFAULT_BACKENDS[0].id;
}

/**
 * Store the selected backend ID.
 */
export function setStoredBackendId(id: string): void {
  localStorage.setItem(STORAGE_KEY, id);
}

/**
 * Get effective capabilities based on connection state.
 * When connected, uses server-provided capabilities.
 * When disconnected, uses the selected backend's default capabilities.
 */
export function getEffectiveCapabilities(
  isConnected: boolean,
  serverCapabilities: string[] | null | undefined,
  selectedBackendId: string
): string[] {
  if (isConnected && serverCapabilities) {
    return serverCapabilities;
  }

  const backend = getDefaultBackendById(selectedBackendId);
  return backend?.capabilities ?? [];
}

/**
 * Check if a capability is supported.
 */
export function isCapabilitySupported(
  capability: string,
  capabilities: string[]
): boolean {
  return capabilities.includes(capability);
}

/**
 * Get backends to display in the selector.
 * Uses server-provided list when connected, defaults when disconnected.
 */
export function getAvailableBackends(
  isConnected: boolean,
  serverBackends: BackendInfo[] | null | undefined
): BackendInfo[] {
  if (isConnected && serverBackends && serverBackends.length > 0) {
    return serverBackends;
  }
  return DEFAULT_BACKENDS;
}
