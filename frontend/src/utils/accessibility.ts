import { AccessibilitySettings, DEFAULT_ACCESSIBILITY } from '../visualizer/types';

export function detectReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function setupReducedMotionListener(
  callback: (reducedMotion: boolean) => void
): () => void {
  if (typeof window === 'undefined') return () => {};

  const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

  const handler = (e: MediaQueryListEvent) => {
    callback(e.matches);
  };

  mediaQuery.addEventListener('change', handler);

  return () => {
    mediaQuery.removeEventListener('change', handler);
  };
}

export function getAccessibilitySettings(): AccessibilitySettings {
  const reducedMotion = detectReducedMotion();

  return {
    ...DEFAULT_ACCESSIBILITY,
    reducedMotion,
    // If reduced motion is preferred, also cap flash rate
    maxFlashRate: reducedMotion ? 1 : DEFAULT_ACCESSIBILITY.maxFlashRate,
  };
}

export function validateFlashRate(rate: number): number {
  // WCAG 2.1 recommends max 3 flashes per second for seizure safety
  return Math.min(Math.max(rate, 0), 3);
}

export function validateIntensityCap(cap: number): number {
  return Math.min(Math.max(cap, 0), 1);
}
