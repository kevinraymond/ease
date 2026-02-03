/**
 * TempoEstimator - Autocorrelation-based Tempo Detection
 *
 * This module estimates the tempo (BPM) of music using autocorrelation of
 * the onset detection function. Autocorrelation reveals periodicity by
 * measuring how similar a signal is to delayed versions of itself.
 *
 * ## Algorithm Overview
 *
 * 1. **Onset Function History**: Maintain several seconds of onset detection
 *    function values (spectral flux peaks over time).
 *
 * 2. **Autocorrelation**: Compute R(lag) = sum(ODF(t) * ODF(t + lag))
 *    Peaks in R(lag) indicate periodicities at those lag times.
 *
 * 3. **Harmonic Enhancement**: To avoid half/double tempo errors:
 *    EnhancedR(lag) = R(lag) + 0.5*R(2*lag) + 0.33*R(3*lag) + 0.25*R(4*lag)
 *    This boosts the correct tempo while suppressing harmonically-related errors.
 *
 * 4. **BPM Conversion**: Convert the best lag to BPM:
 *    BPM = 60 / (lag * frameTime)
 *
 * 5. **Smoothing**: Exponential smoothing prevents jittery BPM display.
 *
 * ## Challenges Addressed
 *
 * - **Half/Double Tempo**: Harmonic enhancement disambiguates these
 * - **Varying Dynamics**: Onset function normalizes for volume changes
 * - **Tempo Changes**: Windowed analysis allows adaptation
 * - **Syncopation**: Autocorrelation finds periodicity even with offbeats
 *
 * ## References
 *
 * - Scheirer, E. "Tempo and Beat Analysis of Acoustic Musical Signals" (1998)
 * - Ellis, D. "Beat Tracking by Dynamic Programming" (2007)
 */

import { CircularBuffer } from './utils/CircularBuffer';
import { TempoResult, AdvancedBeatDetectorConfig } from './types';

/**
 * Assumed frame rate for converting between lags and time.
 * Typical browser animation frame rate.
 */
const DEFAULT_FRAME_RATE = 60;

export class TempoEstimator {
  // Configuration
  private historySeconds: number;
  private bpmRange: [number, number];
  private smoothingFactor: number;
  private debugLogging: boolean = false;

  // State
  private onsetHistory: CircularBuffer;
  private frameRate: number = DEFAULT_FRAME_RATE;
  private frameTime: number = 1000 / DEFAULT_FRAME_RATE; // ms per frame
  private lastUpdateTime: number = 0;
  private frameCount: number = 0;

  // Current tempo estimate
  private currentBpm: number = 0;
  private currentConfidence: number = 0;
  private currentPeriodFrames: number = 0;

  // Stable tempo (locked when confidence is high, held when confidence drops)
  private stableBpm: number = 0;
  private stablePeriodFrames: number = 0;
  private stabilityCounter: number = 0; // Frames of consistent tempo

  // For tracking actual frame rate
  private frameTimeHistory: CircularBuffer;

  // Debug stats
  private debugStats = {
    lastLogTime: 0,
    rawConfidenceSum: 0,
    rawConfidenceCount: 0,
    minRawConfidence: 1,
    maxRawConfidence: 0,
    bpmChanges: 0,
    lastBpm: 0,
  };

  /**
   * Creates a new TempoEstimator.
   *
   * @param config - Partial configuration
   */
  constructor(config: Partial<AdvancedBeatDetectorConfig> = {}) {
    this.historySeconds = config.tempoHistorySeconds ?? 4;
    this.bpmRange = config.bpmRange ?? [60, 200];
    this.smoothingFactor = config.bpmSmoothingFactor ?? 0.1;
    this.debugLogging = config.debugLogging ?? false;

    // Calculate history size based on seconds and frame rate
    // Start with default, will be adjusted as actual frame rate is measured
    const historySize = Math.ceil(this.historySeconds * DEFAULT_FRAME_RATE);
    this.onsetHistory = new CircularBuffer(historySize);

    // Track frame timing to measure actual frame rate
    this.frameTimeHistory = new CircularBuffer(30);

    this.lastUpdateTime = performance.now();
  }

  /**
   * Updates the tempo estimate with a new onset detection function value.
   *
   * @param onsetValue - Current onset detection function value
   * @returns TempoResult with BPM estimate and confidence
   */
  update(onsetValue: number): TempoResult {
    const now = performance.now();

    // Track frame timing for accurate lag-to-BPM conversion
    if (this.lastUpdateTime > 0) {
      const deltaMs = now - this.lastUpdateTime;
      if (deltaMs > 0 && deltaMs < 100) { // Reject outliers
        this.frameTimeHistory.push(deltaMs);

        // Update frame rate estimate (smoothed)
        if (this.frameTimeHistory.length >= 10) {
          const avgFrameTime = this.frameTimeHistory.mean();
          this.frameTime = avgFrameTime;
          this.frameRate = 1000 / avgFrameTime;
        }
      }
    }
    this.lastUpdateTime = now;

    // Add onset value to history
    this.onsetHistory.push(onsetValue);
    this.frameCount++;

    // Need enough history for meaningful autocorrelation
    // At least 2 seconds of data for decent tempo estimation
    const minFrames = Math.ceil(2 * this.frameRate);
    if (this.onsetHistory.length < minFrames) {
      return {
        bpm: 0,
        confidence: 0,
        periodFrames: 0,
        periodMs: 0,
      };
    }

    // Compute autocorrelation every few frames (not every frame for performance)
    // Update at ~10Hz is sufficient for tempo tracking
    if (this.frameCount % 6 !== 0) {
      return {
        bpm: Math.round(this.currentBpm),
        confidence: this.currentConfidence,
        periodFrames: this.currentPeriodFrames,
        periodMs: this.currentPeriodFrames * this.frameTime,
      };
    }

    // Compute tempo via autocorrelation
    const { bpm, confidence, periodFrames } = this.computeTempo();

    // Apply smoothing to prevent jittery display
    if (this.currentBpm > 0 && bpm > 0) {
      // Only smooth if we already have an estimate
      this.currentBpm = this.smoothBpm(this.currentBpm, bpm, confidence);
    } else if (bpm > 0) {
      this.currentBpm = bpm;
    }

    this.currentConfidence = confidence;
    this.currentPeriodFrames = periodFrames;

    // Track tempo stability - lock in tempo when it's consistent
    if (confidence > 0.5 && this.stableBpm > 0) {
      const bpmDiff = Math.abs(this.currentBpm - this.stableBpm) / this.stableBpm;
      if (bpmDiff < 0.08) { // Within 8% of stable (tighter than before)
        this.stabilityCounter++;
      } else if (bpmDiff > 0.3 && this.stabilityCounter < 60) { // 30% different AND not well-established
        // Tempo changed significantly - update stable tempo only if not locked
        this.stableBpm = this.currentBpm;
        this.stablePeriodFrames = periodFrames;
        this.stabilityCounter = 0;
      }
      // If stabilityCounter >= 60 (~1s), we're locked - require very high confidence to change
    } else if (confidence > 0.5 && this.currentBpm > 0) {
      // First time locking stable tempo
      this.stableBpm = this.currentBpm;
      this.stablePeriodFrames = periodFrames;
      this.stabilityCounter = 1;
    }

    // If we had a stable tempo and current confidence is low or BPM is jumping,
    // use the stable tempo instead of the potentially erratic current estimate
    const isJumping = this.stableBpm > 0 && Math.abs(this.currentBpm - this.stableBpm) / this.stableBpm > 0.15;
    if ((confidence < 0.5 || isJumping) && this.stableBpm > 0 && this.stabilityCounter > 60) {
      // Return stable tempo when confidence drops or BPM is erratic but we had good tracking
      this.currentBpm = this.stableBpm;
      this.currentPeriodFrames = this.stablePeriodFrames;
    }

    // Track BPM changes for debug
    if (this.debugLogging) {
      const roundedBpm = Math.round(this.currentBpm);
      if (roundedBpm !== this.debugStats.lastBpm && this.debugStats.lastBpm > 0) {
        this.debugStats.bpmChanges++;
      }
      this.debugStats.lastBpm = roundedBpm;
    }

    // Log periodic summary
    this.logPeriodicSummary(now);

    return {
      bpm: Math.round(this.currentBpm),
      confidence: this.currentConfidence,
      periodFrames: this.currentPeriodFrames,
      periodMs: this.currentPeriodFrames * this.frameTime,
    };
  }

  /**
   * Computes tempo using autocorrelation with harmonic enhancement.
   *
   * The autocorrelation R(lag) measures how similar the onset function is
   * to a version of itself delayed by 'lag' frames. Peaks in R indicate
   * periodicities, and the strongest peak in the valid BPM range gives us
   * the tempo.
   *
   * @returns Object with bpm, confidence, and periodFrames
   */
  private computeTempo(): { bpm: number; confidence: number; periodFrames: number } {
    const history = this.onsetHistory.toArray();
    const n = history.length;

    // Convert BPM range to lag range (in frames)
    // lag = 60 / (bpm * frameTime/1000) = 60000 / (bpm * frameTime)
    const maxLag = Math.min(
      Math.floor(60000 / (this.bpmRange[0] * this.frameTime)),
      Math.floor(n / 2)
    );
    const minLag = Math.max(
      Math.ceil(60000 / (this.bpmRange[1] * this.frameTime)),
      1
    );

    if (maxLag <= minLag) {
      return { bpm: 0, confidence: 0, periodFrames: 0 };
    }

    // Compute autocorrelation for each lag in range
    const autocorr = new Float32Array(maxLag + 1);

    // Zero lag (self-correlation) for normalization
    let energy = 0;
    for (let i = 0; i < n; i++) {
      energy += history[i] * history[i];
    }
    autocorr[0] = energy;

    // Compute R(lag) for each valid lag
    for (let lag = minLag; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < n - lag; i++) {
        sum += history[i] * history[i + lag];
      }
      autocorr[lag] = sum;
    }

    // Apply harmonic enhancement to reject half/double tempo errors
    // EnhancedR(lag) = R(lag) + 0.5*R(2*lag) + 0.33*R(3*lag) + 0.25*R(4*lag)
    const enhanced = new Float32Array(maxLag + 1);
    for (let lag = minLag; lag <= maxLag; lag++) {
      let value = autocorr[lag];

      // Add harmonics (if they exist in the buffer)
      if (lag * 2 <= maxLag) value += 0.5 * autocorr[lag * 2];
      if (lag * 3 <= maxLag) value += 0.33 * autocorr[lag * 3];
      if (lag * 4 <= maxLag) value += 0.25 * autocorr[lag * 4];

      enhanced[lag] = value;
    }

    // Find the peak in the enhanced autocorrelation
    let bestLag = minLag;
    let bestValue = enhanced[minLag];
    for (let lag = minLag + 1; lag <= maxLag; lag++) {
      if (enhanced[lag] > bestValue) {
        bestValue = enhanced[lag];
        bestLag = lag;
      }
    }

    // Convert lag to BPM
    // period_ms = lag * frameTime
    // bpm = 60000 / period_ms
    const periodMs = bestLag * this.frameTime;
    const bpm = 60000 / periodMs;

    // Calculate confidence as ratio of peak to energy
    // Higher values indicate stronger periodicity
    // Note: Using full energy (not * 0.5) to avoid confidence collapsing too easily
    // Also apply a floor to prevent total confidence collapse during complex sections
    const rawConfidence = energy > 0 ? Math.min(1, bestValue / energy) : 0;
    // Floor at 0.15 when we have some periodicity, prevents oscillating between modes
    const confidence = rawConfidence > 0.05 ? Math.max(0.15, rawConfidence) : rawConfidence;

    // Track raw confidence stats for debugging
    if (this.debugLogging) {
      this.debugStats.rawConfidenceSum += rawConfidence;
      this.debugStats.rawConfidenceCount++;
      this.debugStats.minRawConfidence = Math.min(this.debugStats.minRawConfidence, rawConfidence);
      this.debugStats.maxRawConfidence = Math.max(this.debugStats.maxRawConfidence, rawConfidence);
    }

    // Clamp BPM to valid range
    if (bpm < this.bpmRange[0] || bpm > this.bpmRange[1]) {
      return { bpm: 0, confidence: 0, periodFrames: 0 };
    }

    return {
      bpm: Math.round(bpm),
      confidence,
      periodFrames: bestLag,
    };
  }

  /**
   * Smooths BPM changes to prevent jittery display.
   *
   * Uses exponential smoothing with confidence-based adaptation:
   * - High confidence changes are adopted faster
   * - Low confidence changes are smoothed more heavily
   *
   * Also handles octave jumps (half/double tempo) by snapping
   * if the new value is close to a harmonic of the old value.
   *
   * @param currentBpm - Current smoothed BPM
   * @param newBpm - New raw BPM estimate
   * @param confidence - Confidence in new estimate
   * @returns Smoothed BPM value
   */
  private smoothBpm(currentBpm: number, newBpm: number, confidence: number): number {
    // Check for octave relationship (half or double tempo)
    const ratio = newBpm / currentBpm;
    const isHalfTempo = Math.abs(ratio - 0.5) < 0.15;
    const isDoubleTempo = Math.abs(ratio - 2.0) < 0.15;

    if (isHalfTempo || isDoubleTempo) {
      // Likely an octave error - stick with current unless confidence is very high
      // AND we've been seeing this consistently (stability helps here too)
      if (confidence < 0.8) {
        return currentBpm;
      }
    }

    // Reject large jumps unless confidence is very high
    const changeMagnitude = Math.abs(newBpm - currentBpm) / currentBpm;
    if (changeMagnitude > 0.25 && confidence < 0.7) {
      // Jump of >25% with low confidence - reject entirely
      return currentBpm;
    }

    // Adaptive smoothing based on confidence and change magnitude
    // Large changes with low confidence get heavy smoothing
    // Small changes or high confidence changes get light smoothing
    let effectiveSmoothing = this.smoothingFactor;
    if (changeMagnitude > 0.1 && confidence < 0.5) {
      effectiveSmoothing *= 0.3; // Much slower adaptation for suspicious changes
    } else if (confidence > 0.7 && changeMagnitude < 0.05) {
      effectiveSmoothing *= 1.5; // Slightly faster for small confident changes
    }

    // Exponential smoothing
    return currentBpm + effectiveSmoothing * (newBpm - currentBpm);
  }

  /**
   * Returns the current period in milliseconds.
   * Useful for beat prediction.
   */
  getPeriodMs(): number {
    return this.currentPeriodFrames * this.frameTime;
  }

  /**
   * Returns the current estimated BPM.
   */
  getBpm(): number {
    return Math.round(this.currentBpm);
  }

  /**
   * Returns the current confidence level (0-1).
   */
  getConfidence(): number {
    return this.currentConfidence;
  }

  /**
   * Returns the current frame time in milliseconds.
   */
  getFrameTime(): number {
    return this.frameTime;
  }

  /**
   * Resets all internal state.
   */
  reset(): void {
    this.onsetHistory.clear();
    this.frameTimeHistory.clear();
    this.currentBpm = 0;
    this.currentConfidence = 0;
    this.currentPeriodFrames = 0;
    this.frameCount = 0;
    this.lastUpdateTime = performance.now();
    this.stableBpm = 0;
    this.stablePeriodFrames = 0;
    this.stabilityCounter = 0;
  }

  /**
   * Updates configuration.
   *
   * @param config - Partial config with values to update
   */
  configure(config: Partial<AdvancedBeatDetectorConfig>): void {
    if (config.bpmRange) {
      this.bpmRange = config.bpmRange;
    }
    if (config.bpmSmoothingFactor !== undefined) {
      this.smoothingFactor = config.bpmSmoothingFactor;
    }
    if (config.debugLogging !== undefined) {
      this.debugLogging = config.debugLogging;
    }
  }

  /**
   * Logs periodic summary of tempo estimation stats (compact format).
   */
  private logPeriodicSummary(timestamp: number): void {
    if (!this.debugLogging) return;

    // Log summary every 5 seconds
    if (timestamp - this.debugStats.lastLogTime < 5000) return;
    this.debugStats.lastLogTime = timestamp;

    const avgRawConfidence = this.debugStats.rawConfidenceCount > 0
      ? this.debugStats.rawConfidenceSum / this.debugStats.rawConfidenceCount
      : 0;

    console.log(
      `[TEMPO] bpm:${Math.round(this.currentBpm)} period:${(this.currentPeriodFrames * this.frameTime).toFixed(0)}ms | ` +
      `conf:${(this.currentConfidence * 100).toFixed(0)}% rawAvg:${(avgRawConfidence * 100).toFixed(0)}% ` +
      `rawMin:${(this.debugStats.minRawConfidence * 100).toFixed(0)}% rawMax:${(this.debugStats.maxRawConfidence * 100).toFixed(0)}%`
    );

    // Reset counters
    this.debugStats.rawConfidenceSum = 0;
    this.debugStats.rawConfidenceCount = 0;
    this.debugStats.minRawConfidence = 1;
    this.debugStats.maxRawConfidence = 0;
    this.debugStats.bpmChanges = 0;
  }
}
