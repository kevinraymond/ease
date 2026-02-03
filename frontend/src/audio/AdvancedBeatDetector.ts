/**
 * AdvancedBeatDetector - Multi-stage Beat Detection System
 *
 * This is the main coordinator class that combines three specialized components
 * into a robust beat detection pipeline:
 *
 * ```
 * Audio Input (FFT data)
 *        ↓
 * ┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
 * │  OnsetDetector  │────▶│  TempoEstimator  │────▶│ BeatScheduler │
 * │  (Multi-band    │     │  (Autocorrelation│     │ (Prediction + │
 * │   Spectral Flux)│     │   tempo tracking)│     │  Confirmation)│
 * └─────────────────┘     └──────────────────┘     └───────────────┘
 *                                                          ↓
 *                                                    BeatInfo Output
 * ```
 *
 * ## Why This Architecture?
 *
 * Simple energy-based beat detection suffers from several issues:
 * - Misses beats during quiet sections
 * - Double-triggers on complex transients
 * - BPM jitter from syncopated rhythms
 * - Half/double tempo errors
 *
 * This multi-stage approach addresses each issue:
 * - OnsetDetector: Multi-band analysis catches different instruments
 * - TempoEstimator: Autocorrelation provides stable BPM even with offbeats
 * - BeatScheduler: Prediction fills gaps, confirmation validates onsets
 *
 * ## Usage
 *
 * ```ts
 * const detector = new AdvancedBeatDetector();
 *
 * // In your audio processing loop:
 * const result = detector.detect(frequencyData, sampleRate, fftSize);
 *
 * if (result.isBeat) {
 *   // Trigger visual effects
 * }
 *
 * // Use result.bpm, result.confidence, result.phase for more control
 * ```
 *
 * ## Backward Compatibility
 *
 * The BeatInfo interface is compatible with the original BeatDetector,
 * making this a drop-in replacement. Additional information (phase,
 * scheduler state, etc.) is available in the full BeatDecision return type.
 */

import { OnsetDetector } from './OnsetDetector';
import { TempoEstimator } from './TempoEstimator';
import { BeatScheduler } from './BeatScheduler';
import {
  AdvancedBeatDetectorConfig,
  BeatDecision,
  BeatInfo,
  OnsetResult,
  TempoResult,
} from './types';

/**
 * Default configuration values.
 *
 * These defaults work well for most electronic and popular music.
 * For other genres, consider adjusting:
 * - Jazz/Classical: Higher thresholdMultiplier, lower bass weight
 * - EDM: Lower thresholdMultiplier, higher subBass weight
 */
const DEFAULT_CONFIG: AdvancedBeatDetectorConfig = {
  // Band weights emphasizing low frequencies for beat detection
  bandWeights: {
    subBass: 0.4,  // Kick drums dominate beat
    bass: 0.3,     // Bass guitar/synth
    mid: 0.2,      // Snares, vocals
    highMid: 0.1,  // Hi-hats (timing reference, not primary beat)
  },

  // Onset detection parameters
  thresholdMultiplier: 2.0,  // For adaptive threshold (median + k*MAD)
  onsetHistorySize: 22,      // ~0.5s at 43fps
  silenceThreshold: 0.002,   // Below this, skip detection (lowered from 0.005 to catch quieter beats)

  // Tempo estimation parameters
  tempoHistorySeconds: 4,    // Autocorrelation window
  bpmRange: [60, 200],       // Valid BPM range
  bpmSmoothingFactor: 0.05,  // Smoothing for BPM display (lower = more stable)

  // Beat scheduling parameters
  beatWindowMs: 80,          // ±ms around prediction
  refractoryPeriodMs: 150,   // Min time between beats
  minTempoConfidence: 0.4,   // Below this, onset-only mode
  phaseCorrectionRate: 0.3,  // Phase adjustment speed

  // Threshold ceiling parameters (prevents drift during loud sections)
  longTermHistorySize: 172,  // ~4s at 43fps for baseline reference
  thresholdCeiling: 0.5,     // Max threshold = 50% of long-term max

  // Recovery parameters
  maxConsecutiveMisses: 4,   // Decay confidence after 4 missed predictions
  beatTimeoutSeconds: 3,     // Force onset-only mode after 3s without beats
  debugLogging: import.meta.env.DEV, // Auto-enable in dev mode, disable in production
};

export class AdvancedBeatDetector {
  // Sub-components
  private onsetDetector: OnsetDetector;
  private tempoEstimator: TempoEstimator;
  private beatScheduler: BeatScheduler;

  // Configuration
  private config: AdvancedBeatDetectorConfig;

  // Cached results for debugging/visualization
  private lastOnsetResult: OnsetResult | null = null;
  private lastTempoResult: TempoResult | null = null;
  private lastBeatDecision: BeatDecision | null = null;

  /**
   * Creates a new AdvancedBeatDetector.
   *
   * @param config - Partial configuration (defaults used for unspecified values)
   */
  constructor(config: Partial<AdvancedBeatDetectorConfig> = {}) {
    // Merge provided config with defaults
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Initialize sub-components with shared config
    this.onsetDetector = new OnsetDetector(this.config);
    this.tempoEstimator = new TempoEstimator(this.config);
    this.beatScheduler = new BeatScheduler(this.config);
  }

  /**
   * Main detection method - processes one frame of audio data.
   *
   * This is the primary entry point. Call once per animation frame
   * with the current FFT data from AnalyserNode.getByteFrequencyData().
   *
   * @param frequencyData - Uint8Array from AnalyserNode.getByteFrequencyData()
   * @param sampleRate - Audio sample rate in Hz (e.g., 44100, 48000)
   * @param fftSize - FFT size used (e.g., 2048)
   * @returns BeatInfo for backward compatibility with original BeatDetector
   */
  detect(frequencyData: Uint8Array, sampleRate: number, fftSize: number): BeatInfo {
    const decision = this.detectAdvanced(frequencyData, sampleRate, fftSize);

    // Return BeatInfo format for backward compatibility
    return {
      isBeat: decision.isBeat,
      bpm: decision.bpm,
      confidence: decision.confidence,
      strength: decision.strength,
    };
  }

  /**
   * Advanced detection method - returns full BeatDecision with all details.
   *
   * Use this when you need access to:
   * - Beat phase (for smooth pulsing effects)
   * - Next beat prediction (for anticipatory effects)
   * - Scheduler state (for debugging)
   *
   * @param frequencyData - Uint8Array from AnalyserNode.getByteFrequencyData()
   * @param sampleRate - Audio sample rate in Hz
   * @param fftSize - FFT size used
   * @returns Full BeatDecision with all tracking information
   */
  detectAdvanced(
    frequencyData: Uint8Array,
    sampleRate: number,
    fftSize: number
  ): BeatDecision {
    const timestamp = performance.now();

    // Step 1: Onset Detection
    // Analyze spectral flux in multiple frequency bands
    const onsetResult = this.onsetDetector.detect(frequencyData, sampleRate, fftSize);
    this.lastOnsetResult = onsetResult;

    // Step 2: Tempo Estimation
    // Feed onset detection function into autocorrelation-based tempo tracker
    const tempoResult = this.tempoEstimator.update(onsetResult.strength);
    this.lastTempoResult = tempoResult;

    // Step 3: Beat Scheduling
    // Update scheduler with current tempo estimate
    this.beatScheduler.updateTempo(
      tempoResult.bpm,
      tempoResult.periodMs,
      tempoResult.confidence
    );

    // Process onset through scheduler to get final beat decision
    const beatDecision = this.beatScheduler.process(
      onsetResult.isOnset,
      onsetResult.strength,
      timestamp,
      onsetResult.isSustainedSilence ?? false
    );
    this.lastBeatDecision = beatDecision;

    return beatDecision;
  }

  /**
   * Legacy detection method for backward compatibility.
   *
   * This accepts the same parameters as the original BeatDetector.detect().
   * Internally, it creates synthetic frequency data for the onset detector.
   *
   * For best results, use detect() with actual frequencyData instead.
   *
   * @param bassEnergy - Bass energy level (0-1)
   * @param rms - RMS energy level (0-1)
   * @returns BeatInfo compatible with original BeatDetector
   */
  detectLegacy(bassEnergy: number, rms: number): BeatInfo {
    // This is a fallback that doesn't use the full multi-band analysis.
    // It provides basic compatibility but won't be as accurate.
    // For proper operation, use detect() with frequencyData.

    const timestamp = performance.now();

    // Create a simple synthetic onset value from bass energy
    const onsetValue = bassEnergy * 0.8 + rms * 0.2;

    // Feed through tempo estimator
    const tempoResult = this.tempoEstimator.update(onsetValue);

    // Simple onset detection (threshold-based)
    const isOnset = onsetValue > 0.1; // Basic threshold

    // Update scheduler
    this.beatScheduler.updateTempo(
      tempoResult.bpm,
      tempoResult.periodMs,
      tempoResult.confidence
    );

    // Process through scheduler
    const beatDecision = this.beatScheduler.process(isOnset, onsetValue, timestamp);

    return {
      isBeat: beatDecision.isBeat,
      bpm: beatDecision.bpm,
      confidence: beatDecision.confidence,
      strength: beatDecision.strength,
    };
  }

  // ============================================
  // Accessors for Debugging and Visualization
  // ============================================

  /**
   * Returns the most recent onset detection result.
   * Useful for visualizing per-band spectral flux.
   */
  getLastOnsetResult(): OnsetResult | null {
    return this.lastOnsetResult;
  }

  /**
   * Returns the most recent tempo estimation result.
   * Useful for displaying BPM and confidence.
   */
  getLastTempoResult(): TempoResult | null {
    return this.lastTempoResult;
  }

  /**
   * Returns the most recent beat decision.
   * Useful for accessing phase and scheduler state.
   */
  getLastBeatDecision(): BeatDecision | null {
    return this.lastBeatDecision;
  }

  /**
   * Returns the current beat phase (0-1).
   * Phase = 0 at the beat, increases toward 1, then wraps.
   * Useful for creating smooth pulsing effects.
   */
  getPhase(): number {
    return this.lastBeatDecision?.phase ?? 0;
  }

  /**
   * Returns the current BPM estimate.
   */
  getBPM(): number {
    return this.lastBeatDecision?.bpm ?? 0;
  }

  /**
   * Returns the current confidence level (0-1).
   */
  getConfidence(): number {
    return this.lastBeatDecision?.confidence ?? 0;
  }

  /**
   * Returns the predicted time of the next beat (ms since page load).
   */
  getNextBeatTime(): number {
    return this.lastBeatDecision?.nextBeatTime ?? 0;
  }

  // ============================================
  // Configuration and Control
  // ============================================

  /**
   * Updates configuration parameters.
   *
   * @param config - Partial config with values to update
   */
  configure(config: Partial<AdvancedBeatDetectorConfig>): void {
    this.config = { ...this.config, ...config };

    // Propagate to sub-components
    this.onsetDetector.configure(config);
    this.tempoEstimator.configure(config);
    this.beatScheduler.configure(config);
  }

  /**
   * Returns the current configuration.
   */
  getConfig(): AdvancedBeatDetectorConfig {
    return { ...this.config };
  }

  /**
   * Resets all internal state.
   * Call when audio source changes or playback stops.
   */
  reset(): void {
    this.onsetDetector.reset();
    this.tempoEstimator.reset();
    this.beatScheduler.reset();
    this.lastOnsetResult = null;
    this.lastTempoResult = null;
    this.lastBeatDecision = null;
  }

  // ============================================
  // Preset Configurations
  // ============================================

  /**
   * Applies preset configuration for electronic dance music.
   * Emphasizes sub-bass and bass for strong kick detection.
   */
  applyEDMPreset(): void {
    this.configure({
      bandWeights: {
        subBass: 0.5,
        bass: 0.3,
        mid: 0.15,
        highMid: 0.05,
      },
      thresholdMultiplier: 1.8, // More sensitive
      bpmRange: [100, 180],     // Typical EDM range
    });
  }

  /**
   * Applies preset configuration for rock/pop music.
   * Balanced weights for kick, snare, and vocals.
   */
  applyRockPreset(): void {
    this.configure({
      bandWeights: {
        subBass: 0.3,
        bass: 0.3,
        mid: 0.3,   // More snare emphasis
        highMid: 0.1,
      },
      thresholdMultiplier: 2.0,
      bpmRange: [80, 160],
    });
  }

  /**
   * Applies preset configuration for hip-hop music.
   * Strong bass emphasis with wider tempo range.
   */
  applyHipHopPreset(): void {
    this.configure({
      bandWeights: {
        subBass: 0.45,
        bass: 0.35,
        mid: 0.15,
        highMid: 0.05,
      },
      thresholdMultiplier: 1.9,
      bpmRange: [60, 140], // Hip-hop can be slower
    });
  }

  /**
   * Applies preset for more aggressive/sensitive detection.
   * Use when beats are being missed.
   */
  applySensitivePreset(): void {
    this.configure({
      thresholdMultiplier: 1.5,
      minTempoConfidence: 0.3,
      beatWindowMs: 100,
    });
  }

  /**
   * Applies preset for more conservative detection.
   * Use when getting too many false positives.
   */
  applyConservativePreset(): void {
    this.configure({
      thresholdMultiplier: 2.5,
      minTempoConfidence: 0.5,
      beatWindowMs: 60,
    });
  }
}

// Re-export types for convenience
export type { BeatInfo, BeatDecision, OnsetResult, TempoResult, AdvancedBeatDetectorConfig } from './types';
