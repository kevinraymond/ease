/**
 * BeatScheduler - Predictive Beat Tracking with State Machine
 *
 * This module combines onset detection with tempo prediction to produce
 * robust beat tracking that handles:
 * - Weak or missing onsets (uses prediction to fill gaps)
 * - Syncopation (allows off-beat hits within tolerance)
 * - Tempo changes (adapts phase while maintaining stable tracking)
 *
 * ## State Machine
 *
 * ```
 *                          onset detected
 *    ┌─────────────────┐  in window     ┌────────────────┐
 *    │     WAITING     │ ───────────────│   CONFIRMED    │──┐
 *    │ (no tempo lock) │                │  (beat fired)  │  │
 *    └────────┬────────┘                └────────────────┘  │
 *             │ tempo locked                                │
 *             ↓                                             │
 *    ┌─────────────────┐     window expired  ┌──────────────┴─┐
 *    │    EXPECTING    │ ────────────────────│    MISSED      │
 *    │ (beat predicted)│                     │ (fallback fire)│
 *    └─────────────────┘                     └────────────────┘
 * ```
 *
 * ## Algorithm Overview
 *
 * 1. **Phase Tracking**: Maintain a phase (0-1) that increments based on tempo.
 *    Phase = 0 at beat time, wraps back to 0 each beat.
 *
 * 2. **Beat Prediction**: Using tempo estimate, predict when next beat should occur.
 *    nextBeatTime = lastBeatTime + (60000 / bpm)
 *
 * 3. **Confirmation Window**: Beat is confirmed if an onset falls within
 *    ±beatWindow ms of the predicted time.
 *
 * 4. **Phase Correction**: When onset deviates from prediction, gradually
 *    adjust phase to align with actual rhythm.
 *
 * 5. **Fallback**: If tempo confidence is low, use onset-only detection.
 *
 * ## Key Parameters
 *
 * - beatWindow: Tolerance for onset-prediction alignment (±80ms default)
 * - refractoryPeriod: Minimum time between beats (150ms = 400 BPM max)
 * - phaseCorrectionRate: How fast to adjust phase toward onsets (0.3 default)
 * - minTempoConfidence: Below this, fall back to onset-only mode (0.4 default)
 */

import { BeatSchedulerState, BeatDecision, AdvancedBeatDetectorConfig } from './types';

export class BeatScheduler {
  // Configuration
  private beatWindowMs: number;
  private refractoryPeriodMs: number;
  private minTempoConfidence: number;
  private phaseCorrectionRate: number;
  private maxConsecutiveMisses: number;
  private beatTimeoutMs: number;
  private debugLogging: boolean;

  // State machine
  private state: BeatSchedulerState = 'WAITING';

  // Timing state
  private lastBeatTime: number = 0;
  private nextPredictedBeatTime: number = 0;
  private currentPhase: number = 0; // 0-1, where 0 = on beat

  // Current tempo info (from TempoEstimator)
  private currentBpm: number = 0;
  private currentPeriodMs: number = 0;
  private tempoConfidence: number = 0;

  // Beat tracking
  private beatStrength: number = 0;

  // Confidence accumulator for overall beat tracking
  private trackingConfidence: number = 0;

  // Recovery state
  private consecutiveMissedBeats: number = 0;
  private lastBeatFiredTime: number = 0;

  // Stored tempo for prediction backup (persists when confidence drops)
  private storedPeriodMs: number = 0;

  // Debug stats for logging
  private debugStats = {
    lastBeatLogTime: 0,
    lastGapWarningTime: 0,
    beatNumber: 0,
    sessionStartTime: 0,
  };

  /**
   * Creates a new BeatScheduler.
   *
   * @param config - Partial configuration
   */
  constructor(config: Partial<AdvancedBeatDetectorConfig> = {}) {
    this.beatWindowMs = config.beatWindowMs ?? 80;
    this.refractoryPeriodMs = config.refractoryPeriodMs ?? 150;
    this.minTempoConfidence = config.minTempoConfidence ?? 0.4;
    this.phaseCorrectionRate = config.phaseCorrectionRate ?? 0.3;
    this.maxConsecutiveMisses = config.maxConsecutiveMisses ?? 4;
    this.beatTimeoutMs = (config.beatTimeoutSeconds ?? 3) * 1000;
    this.debugLogging = config.debugLogging ?? false;
  }

  /**
   * Updates the scheduler with current tempo estimate.
   * Call this with the output from TempoEstimator.
   *
   * @param bpm - Current BPM estimate
   * @param periodMs - Beat period in milliseconds
   * @param confidence - Tempo confidence (0-1)
   */
  updateTempo(bpm: number, periodMs: number, confidence: number): void {
    this.currentBpm = bpm;
    this.currentPeriodMs = periodMs;
    this.tempoConfidence = confidence;

    // Store tempo when we have reasonable confidence for prediction backup
    if (confidence >= this.minTempoConfidence && periodMs > 0) {
      this.storedPeriodMs = periodMs;
    }
  }

  /**
   * Main processing method. Call once per audio frame.
   *
   * @param isOnset - Whether an onset was detected this frame
   * @param onsetStrength - Strength/intensity of the onset
   * @param timestamp - Current time in milliseconds (performance.now())
   * @param isSustainedSilence - Whether audio has been silent for extended period
   * @returns BeatDecision with final beat determination
   */
  process(isOnset: boolean, onsetStrength: number, timestamp: number, isSustainedSilence: boolean = false): BeatDecision {
    // Time since last beat
    const timeSinceLastBeat = this.lastBeatTime > 0 ? timestamp - this.lastBeatTime : Infinity;

    // Check refractory period (prevent double-triggers)
    const inRefractoryPeriod = timeSinceLastBeat < this.refractoryPeriodMs;

    // Track onset strength for beat output
    if (isOnset) {
      this.beatStrength = onsetStrength;
    }

    // Update phase if we have tempo, and get count of missed predicted beats
    let missedPredictedBeats = 0;
    if (this.currentPeriodMs > 0) {
      missedPredictedBeats = this.updatePhase(timestamp);
    }

    // Check for beat timeout recovery
    const timeSinceLastFiredBeat = this.lastBeatFiredTime > 0
      ? timestamp - this.lastBeatFiredTime
      : 0;
    if (timeSinceLastFiredBeat > this.beatTimeoutMs && this.lastBeatFiredTime > 0) {
      // Gently reduce confidence to allow onset-only mode to kick in
      // Don't fully zero - preserve some tracking state for faster recovery
      this.logDebug('Beat timeout triggered - reducing confidence for recovery');
      this.trackingConfidence *= 0.3;
      // Don't zero tempo confidence - let TempoEstimator handle its own state
      this.consecutiveMissedBeats = 0;
      this.state = 'WAITING';
      // Reset lastBeatFiredTime to prevent repeated triggers
      this.lastBeatFiredTime = timestamp;
    }

    // Determine if we should fire a beat
    let isBeat = false;
    let beatSource = ''; // For debug tracking

    // Don't fire any beats during sustained silence (song ended/paused)
    if (isSustainedSilence) {
      // Reset state so we're ready when audio returns
      if (this.state !== 'WAITING') {
        this.logDebug('Sustained silence detected - pausing beat detection');
        this.state = 'WAITING';
        this.consecutiveMissedBeats = 0;
      }
      return {
        isBeat: false,
        bpm: this.currentBpm,
        confidence: 0,
        strength: 0,
        schedulerState: this.state,
        nextBeatTime: 0,
        phase: 0,
      };
    }

    // CRITICAL FIX: If we have high tempo confidence and missed predicted beats,
    // fire them even if onset detection is failing
    if (missedPredictedBeats > 0 && this.tempoConfidence >= this.minTempoConfidence) {
      isBeat = true;
      beatSource = 'predicted';
      this.beatStrength = 0.5; // Moderate strength for predicted beats
      this.state = 'MISSED';
    } else if (this.tempoConfidence < this.minTempoConfidence || this.currentPeriodMs === 0) {
      // Low confidence mode: use onset-only detection
      // This is the fallback when tempo can't be reliably determined
      isBeat = this.processOnsetOnly(isOnset, inRefractoryPeriod);
      if (isBeat) beatSource = 'onset';

      // Prediction backup: if onset-only hasn't fired in a while but we have stored tempo,
      // use prediction to maintain beat continuity
      if (!isBeat && this.storedPeriodMs > 0 && this.lastBeatFiredTime > 0) {
        const timeSinceBeat = timestamp - this.lastBeatFiredTime;
        // Fire if we've waited longer than the stored period (with some tolerance)
        if (timeSinceBeat >= this.storedPeriodMs * 0.9 && !inRefractoryPeriod) {
          isBeat = true;
          beatSource = 'backup';
          this.beatStrength = 0.5; // Moderate strength for predicted beats
        }
      }
    } else {
      // High confidence mode: use prediction + confirmation
      isBeat = this.processPredictive(isOnset, timestamp, inRefractoryPeriod);
      if (isBeat) {
        beatSource = this.state === 'CONFIRMED' ? 'confirmed' : 'predicted';
      }
    }

    // Fire beat and update state
    if (isBeat) {
      this.lastBeatTime = timestamp;
      this.lastBeatFiredTime = timestamp;
      this.currentPhase = 0;
      this.consecutiveMissedBeats = 0; // Reset miss counter on beat

      // Update prediction for next beat
      if (this.currentPeriodMs > 0) {
        this.nextPredictedBeatTime = timestamp + this.currentPeriodMs;
      }

      // Update tracking confidence - faster recovery (was 0.1)
      this.trackingConfidence = Math.min(1, this.trackingConfidence + 0.15);

      // Log beat
      this.logBeat(timestamp, beatSource);
    } else {
      // Decay tracking confidence very slowly
      this.trackingConfidence = Math.max(0, this.trackingConfidence - 0.0005);

      // Check for gaps and log warnings
      this.checkAndLogGap(timestamp, isOnset, onsetStrength);
    }

    // Calculate overall confidence
    const confidence = this.calculateOverallConfidence();

    return {
      isBeat,
      bpm: this.currentBpm,
      confidence,
      strength: isBeat ? this.beatStrength : 0,
      schedulerState: this.state,
      nextBeatTime: this.nextPredictedBeatTime,
      phase: this.currentPhase,
    };
  }

  /**
   * Processes in onset-only mode (fallback when tempo confidence is low).
   *
   * Simply triggers on detected onsets, respecting refractory period.
   *
   * @param isOnset - Whether an onset was detected
   * @param inRefractoryPeriod - Whether we're in refractory period
   * @returns Whether to fire a beat
   */
  private processOnsetOnly(isOnset: boolean, inRefractoryPeriod: boolean): boolean {
    this.state = 'WAITING';

    if (isOnset && !inRefractoryPeriod) {
      this.state = 'CONFIRMED';
      return true;
    }

    return false;
  }

  /**
   * Processes in predictive mode (when tempo is confidently known).
   *
   * Uses a prediction window approach:
   * 1. Predict when next beat should occur based on tempo
   * 2. If onset falls within ±beatWindow of prediction, confirm beat
   * 3. If window expires without onset, fire beat anyway (prediction)
   * 4. Apply phase correction when onsets deviate from prediction
   *
   * @param isOnset - Whether an onset was detected
   * @param timestamp - Current time
   * @param inRefractoryPeriod - Whether we're in refractory period
   * @returns Whether to fire a beat
   */
  private processPredictive(
    isOnset: boolean,
    timestamp: number,
    inRefractoryPeriod: boolean
  ): boolean {
    // Calculate distance to predicted beat
    const distanceToPrediction = this.nextPredictedBeatTime > 0
      ? timestamp - this.nextPredictedBeatTime
      : Infinity;

    // Check if we're within the beat window
    const inBeatWindow = Math.abs(distanceToPrediction) <= this.beatWindowMs;

    // Check if prediction window has expired (we've passed the beat time)
    const windowExpired = distanceToPrediction > this.beatWindowMs;

    // State machine transitions
    switch (this.state) {
      case 'WAITING':
        // Waiting for first beat or tempo lock
        if (this.nextPredictedBeatTime === 0) {
          // No prediction yet - wait for onset to start
          if (isOnset && !inRefractoryPeriod) {
            this.nextPredictedBeatTime = timestamp + this.currentPeriodMs;
            this.state = 'CONFIRMED';
            return true;
          }
        } else {
          // Have prediction - transition to expecting
          this.state = 'EXPECTING';
        }
        return false;

      case 'EXPECTING':
        // Expecting a beat within the window
        if (isOnset && inBeatWindow && !inRefractoryPeriod) {
          // Onset confirmed the prediction
          this.state = 'CONFIRMED';

          // Apply phase correction toward the onset
          // If onset was early, phase should be adjusted down
          // If onset was late, phase should be adjusted up
          if (this.currentPeriodMs > 0) {
            const phaseError = distanceToPrediction / this.currentPeriodMs;
            this.applyPhaseCorrection(phaseError);
          }

          return true;
        } else if (windowExpired) {
          // Window expired without onset - fire beat anyway based on prediction
          // NOTE: Removed !inRefractoryPeriod check here. Predicted beats are tempo-based,
          // not onset-based. Refractory period prevents double-triggers from onsets,
          // but predicted beats should still fire to maintain tempo tracking.
          this.state = 'MISSED';
          this.logDebug('Window expired - firing predicted beat');
          return true;
        }
        return false;

      case 'CONFIRMED':
      case 'MISSED':
        // Beat was just fired - transition to expecting next beat
        this.state = 'EXPECTING';

        // If an onset happens right after (and we're not in refractory),
        // it might be a syncopated rhythm - handle gracefully
        if (isOnset && inBeatWindow && !inRefractoryPeriod) {
          this.state = 'CONFIRMED';
          return true;
        }
        return false;

      default:
        return false;
    }
  }

  /**
   * Updates the beat phase based on elapsed time.
   *
   * Phase is a value from 0-1 representing position within the beat cycle.
   * Phase = 0 at the beat, increases toward 1, then wraps back to 0.
   *
   * Returns the number of predicted beats that should fire (when we've passed
   * the prediction window without an onset confirming).
   *
   * @param timestamp - Current time in milliseconds
   * @returns Number of predicted beats to fire
   */
  private updatePhase(timestamp: number): number {
    if (this.lastBeatTime === 0 || this.currentPeriodMs === 0) {
      return 0;
    }

    const timeSinceLastBeat = timestamp - this.lastBeatTime;
    this.currentPhase = (timeSinceLastBeat / this.currentPeriodMs) % 1;

    // Count how many predicted beats we've passed
    let missedBeatsToFire = 0;

    // Update prediction if needed, tracking missed beats
    while (this.nextPredictedBeatTime > 0 && timestamp > this.nextPredictedBeatTime + this.beatWindowMs) {
      // We've passed the prediction window without firing a beat
      this.consecutiveMissedBeats++;
      missedBeatsToFire++;
      this.nextPredictedBeatTime += this.currentPeriodMs;

      // Decay confidence after reaching miss threshold (only once per threshold crossing)
      if (this.consecutiveMissedBeats === this.maxConsecutiveMisses) {
        this.logDebug(`Reached ${this.maxConsecutiveMisses} consecutive misses - decaying confidence`);
        this.trackingConfidence *= 0.7;
      }
    }

    return missedBeatsToFire;
  }

  /**
   * Applies phase correction to align with detected onsets.
   *
   * This is crucial for maintaining sync with the actual music rhythm.
   * We don't snap immediately (which would cause jumpy visuals) but
   * gradually adjust the phase over several beats.
   *
   * @param phaseError - Normalized error (-1 to 1, where 0 = perfect alignment)
   */
  private applyPhaseCorrection(phaseError: number): void {
    // Clamp error to prevent wild corrections
    const clampedError = Math.max(-0.5, Math.min(0.5, phaseError));

    // Adjust next prediction based on error
    // If error is positive (onset was late), push next prediction later
    // If error is negative (onset was early), pull next prediction earlier
    const correctionMs = clampedError * this.currentPeriodMs * this.phaseCorrectionRate;
    this.nextPredictedBeatTime += correctionMs;
  }

  /**
   * Calculates overall confidence in beat tracking.
   *
   * Combines tempo confidence with tracking stability.
   * Tracking confidence is weighted more heavily to prevent tempo estimation
   * fluctuations from killing beat detection.
   *
   * @returns Overall confidence (0-1)
   */
  private calculateOverallConfidence(): number {
    // Balance tempo and tracking confidence more evenly
    // This prevents tempo confidence dips from collapsing overall confidence
    return this.tempoConfidence * 0.5 + this.trackingConfidence * 0.5;
  }

  /**
   * Returns the current phase (0-1).
   * Useful for creating effects that pulse with the beat.
   */
  getPhase(): number {
    return this.currentPhase;
  }

  /**
   * Returns the predicted time of the next beat.
   */
  getNextBeatTime(): number {
    return this.nextPredictedBeatTime;
  }

  /**
   * Returns the current scheduler state.
   */
  getState(): BeatSchedulerState {
    return this.state;
  }

  /**
   * Resets all internal state.
   */
  reset(): void {
    this.state = 'WAITING';
    this.lastBeatTime = 0;
    this.nextPredictedBeatTime = 0;
    this.currentPhase = 0;
    this.currentBpm = 0;
    this.currentPeriodMs = 0;
    this.tempoConfidence = 0;
    this.beatStrength = 0;
    this.trackingConfidence = 0;
    this.consecutiveMissedBeats = 0;
    this.lastBeatFiredTime = 0;
    this.storedPeriodMs = 0;
  }

  /**
   * Updates configuration.
   *
   * @param config - Partial config with values to update
   */
  configure(config: Partial<AdvancedBeatDetectorConfig>): void {
    if (config.beatWindowMs !== undefined) {
      this.beatWindowMs = config.beatWindowMs;
    }
    if (config.refractoryPeriodMs !== undefined) {
      this.refractoryPeriodMs = config.refractoryPeriodMs;
    }
    if (config.minTempoConfidence !== undefined) {
      this.minTempoConfidence = config.minTempoConfidence;
    }
    if (config.phaseCorrectionRate !== undefined) {
      this.phaseCorrectionRate = config.phaseCorrectionRate;
    }
    if (config.maxConsecutiveMisses !== undefined) {
      this.maxConsecutiveMisses = config.maxConsecutiveMisses;
    }
    if (config.beatTimeoutSeconds !== undefined) {
      this.beatTimeoutMs = config.beatTimeoutSeconds * 1000;
    }
    if (config.debugLogging !== undefined) {
      this.debugLogging = config.debugLogging;
    }
  }

  /**
   * Simple debug log helper.
   */
  private logDebug(msg: string): void {
    if (this.debugLogging) console.log(`[BEAT] ${msg}`);
  }

  /**
   * Logs a beat event in compact format for diagnosis.
   * Format: BEAT #N | T:12.34s | gap:456ms | src:confirmed | bpm:128 | tConf:45% | trkConf:80% | state:EXPECTING
   */
  private logBeat(timestamp: number, source: string): void {
    if (!this.debugLogging) return;

    if (this.debugStats.sessionStartTime === 0) {
      this.debugStats.sessionStartTime = timestamp;
    }

    this.debugStats.beatNumber++;
    const sessionTime = ((timestamp - this.debugStats.sessionStartTime) / 1000).toFixed(2);
    const gap = this.debugStats.lastBeatLogTime > 0
      ? (timestamp - this.debugStats.lastBeatLogTime).toFixed(0)
      : '---';

    console.log(
      `BEAT #${this.debugStats.beatNumber.toString().padStart(3)} | ` +
      `T:${sessionTime.padStart(6)}s | ` +
      `gap:${gap.padStart(4)}ms | ` +
      `src:${source.padEnd(9)} | ` +
      `bpm:${this.currentBpm.toString().padStart(3)} | ` +
      `tConf:${(this.tempoConfidence * 100).toFixed(0).padStart(2)}% | ` +
      `trkConf:${(this.trackingConfidence * 100).toFixed(0).padStart(2)}% | ` +
      `state:${this.state}`
    );

    this.debugStats.lastBeatLogTime = timestamp;
  }

  /**
   * Logs a warning when no beats have fired for too long.
   * This helps identify where beat drops occur.
   */
  private checkAndLogGap(timestamp: number, isOnset: boolean, onsetStrength: number): void {
    if (!this.debugLogging) return;
    if (this.debugStats.lastBeatLogTime === 0) return;

    const gap = timestamp - this.debugStats.lastBeatLogTime;
    const expectedPeriod = this.storedPeriodMs || this.currentPeriodMs || 500;

    // Warn if gap exceeds 2x expected period (missing beats)
    if (gap > expectedPeriod * 2 && timestamp - this.debugStats.lastGapWarningTime > 500) {
      this.debugStats.lastGapWarningTime = timestamp;
      const sessionTime = ((timestamp - this.debugStats.sessionStartTime) / 1000).toFixed(2);

      console.log(
        `⚠️ GAP ${gap.toFixed(0)}ms | ` +
        `T:${sessionTime.padStart(6)}s | ` +
        `expected:${expectedPeriod.toFixed(0)}ms | ` +
        `onset:${isOnset ? 'YES' : 'no '} str:${onsetStrength.toFixed(3)} | ` +
        `tConf:${(this.tempoConfidence * 100).toFixed(0)}% | ` +
        `trkConf:${(this.trackingConfidence * 100).toFixed(0)}% | ` +
        `misses:${this.consecutiveMissedBeats} | ` +
        `state:${this.state}`
      );
    }
  }
}
