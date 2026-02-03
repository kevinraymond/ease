/**
 * Basic beat detection result.
 * Backward-compatible interface used by AdvancedBeatDetector.
 */
export interface BeatInfo {
  isBeat: boolean;
  bpm: number;
  confidence: number;
  strength: number;
}

/**
 * Onset detection info - captures transients and note attacks.
 * More granular than beat detection.
 */
export interface OnsetInfo {
  isOnset: boolean;       // Whether this frame is an onset
  confidence: number;     // 0-1 confidence level
  strength: number;       // Onset strength (spectral flux)
  spectralFlux: number;   // Total spectral change
}

/**
 * Chroma features - 12-bin pitch class distribution.
 * Maps frequency content to musical notes (C, C#, D, ..., B).
 */
export interface ChromaFeatures {
  bins: [number, number, number, number, number, number, number, number, number, number, number, number];
  energy: number;         // Overall tonal content
  noteNames: readonly string[];  // ['C', 'C#', 'D', ...]
}

export interface AudioMetrics {
  rms: number;
  peak: number;
  bass: number;
  mid: number;
  treble: number;
  // Raw (unsmoothed) values for beat detection
  rawBass: number;
  rawMid: number;
  rawTreble: number;

  // SOTA audio features for AI control
  /** Spectral centroid - brightness/timbre (0-1, smoothed). Maps to CFG scale. */
  spectralCentroid: number;
  /** Raw spectral centroid (unsmoothed) */
  rawSpectralCentroid: number;
  /** Onset detection with confidence - catches transients */
  onset: OnsetInfo;
  /** Chroma features - 12-bin pitch class distribution */
  chroma: ChromaFeatures;
  /** Dominant pitch class index (0-11: C, C#, D, ..., B) */
  dominantChroma: number;

  frequencyData: Uint8Array;
  timeData: Uint8Array;
  sampleRate: number;
  fftSize: number;
  bpm: number;
  isBeat: boolean;
}

export interface AudioConfig {
  fftSize: 256 | 512 | 1024 | 2048 | 4096 | 8192;
  smoothingTimeConstant: number;
  // Frequency band boundaries in Hz
  bassRange: [number, number];
  midRange: [number, number];
  trebleRange: [number, number];
}

export const DEFAULT_AUDIO_CONFIG: AudioConfig = {
  fftSize: 2048,
  smoothingTimeConstant: 0.8,
  bassRange: [20, 250],
  midRange: [250, 4000],
  trebleRange: [4000, 20000],
};

export type SupportedAudioFormat = 'audio/mpeg' | 'audio/wav' | 'audio/flac' | 'audio/ogg' | 'audio/mp4';

export const SUPPORTED_EXTENSIONS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'];

export interface AudioFileInfo {
  name: string;
  sampleRate: number;
  channels: number;
  duration: number;
}

// ============================================
// Advanced Beat Detection Types
// ============================================

/**
 * Frequency band names for multi-band spectral flux onset detection.
 *
 * Each band targets different percussive elements:
 * - subBass: Kick drums, sub-bass synths (20-80 Hz)
 * - bass: Bass guitar, bass synths (80-250 Hz)
 * - mid: Snares, vocals, melodic content (500-2000 Hz)
 * - highMid: Hi-hats, cymbals, transients (2000-4000 Hz)
 */
export type BandName = 'subBass' | 'bass' | 'mid' | 'highMid';

/**
 * Result from the OnsetDetector.
 *
 * Onsets are detected by measuring spectral flux (rate of spectral change)
 * in multiple frequency bands, then combining them with perceptual weights.
 */
export interface OnsetResult {
  /** True if an onset is detected in the current frame */
  isOnset: boolean;
  /** Overall onset strength after weighted combination of bands (0 to ~1+) */
  strength: number;
  /** Per-band spectral flux values (useful for debugging/visualization) */
  bandFlux: Record<BandName, number>;
  /** Adaptive threshold that was used for this detection */
  threshold: number;
  /** Time (ms since start) when this onset occurred */
  timestamp: number;
  /** True if audio has been silent for an extended period (song ended/paused) */
  isSustainedSilence?: boolean;
}

/**
 * Result from the TempoEstimator.
 *
 * Tempo is estimated using autocorrelation of the onset detection function,
 * with harmonic enhancement to reject half/double tempo errors.
 */
export interface TempoResult {
  /** Estimated beats per minute (60-200 range) */
  bpm: number;
  /**
   * Confidence in the tempo estimate (0-1).
   * Higher values mean stronger periodicity in the signal.
   * Below minTempoConfidence, the scheduler falls back to onset-only detection.
   */
  confidence: number;
  /** The autocorrelation lag (in frames) corresponding to the detected tempo */
  periodFrames: number;
  /** Period in milliseconds between beats */
  periodMs: number;
}

/**
 * State machine states for the BeatScheduler.
 *
 * The scheduler uses tempo prediction to anticipate beats and confirm them
 * with actual onset detections.
 */
export type BeatSchedulerState =
  /** Waiting for first onset or tempo lock */
  | 'WAITING'
  /** Expecting a beat within the prediction window */
  | 'EXPECTING'
  /** Beat was confirmed by onset detection */
  | 'CONFIRMED'
  /** Beat was missed (no onset in prediction window) */
  | 'MISSED';

/**
 * Output from the BeatScheduler representing the final beat decision.
 *
 * This combines onset detection with tempo prediction for robust beat tracking
 * that works even when some onsets are weak or obscured.
 */
export interface BeatDecision {
  /** True if this frame should be treated as a beat */
  isBeat: boolean;
  /** Current estimated BPM */
  bpm: number;
  /** Overall confidence in beat detection (0-1) */
  confidence: number;
  /** Beat strength/intensity for visual scaling */
  strength: number;
  /** Current state of the beat scheduler state machine */
  schedulerState: BeatSchedulerState;
  /** Predicted time (ms) of the next beat */
  nextBeatTime: number;
  /** Current phase in the beat cycle (0-1, where 0 = on beat) */
  phase: number;
}

/**
 * Configuration for the AdvancedBeatDetector system.
 *
 * These parameters can be tuned for different music genres:
 * - EDM/Dance: Lower thresholds, higher sub-bass weight
 * - Rock/Pop: Balanced weights, standard thresholds
 * - Jazz/Classical: Higher thresholds, more mid-range focus
 */
/**
 * Debug information from the advanced beat detector.
 * Used for visualization in the frequency debug overlay.
 */
export interface BeatDebugInfo {
  /** Per-band spectral flux values */
  bandFlux: Record<BandName, number>;
  /** Combined onset strength (weighted sum of bands) */
  onsetStrength: number;
  /** Adaptive threshold used for onset detection */
  onsetThreshold: number;
  /** Whether an onset was detected this frame */
  isOnset: boolean;
  /** Current BPM estimate */
  bpm: number;
  /** Tempo detection confidence (0-1) */
  tempoConfidence: number;
  /** Beat scheduler state machine state */
  schedulerState: BeatSchedulerState;
  /** Current phase in beat cycle (0-1, 0 = on beat) */
  phase: number;
  /** Predicted time of next beat (ms) */
  nextBeatTime: number;
  /** Whether a beat was fired this frame */
  isBeat: boolean;
  /** Beat strength when beat fires */
  beatStrength: number;
}

export interface AdvancedBeatDetectorConfig {
  /**
   * Weights for combining spectral flux from different bands.
   * Higher weights mean that band has more influence on onset detection.
   * Default: subBass=0.4, bass=0.3, mid=0.2, highMid=0.1
   */
  bandWeights: Record<BandName, number>;

  /**
   * Multiplier for adaptive threshold calculation.
   * threshold = median + (thresholdMultiplier * MAD)
   * Higher values = less sensitive, fewer false positives.
   * Default: 2.0
   */
  thresholdMultiplier: number;

  /**
   * Number of frames of onset history for adaptive threshold.
   * ~22 frames at 43fps = ~0.5 seconds.
   * Longer history = more stable threshold but slower adaptation.
   * Default: 22
   */
  onsetHistorySize: number;

  /**
   * Seconds of onset function history for tempo autocorrelation.
   * Longer = more accurate tempo, but slower to detect changes.
   * Default: 4 seconds
   */
  tempoHistorySeconds: number;

  /**
   * Valid BPM range for tempo detection.
   * Helps reject spurious tempo estimates outside musical range.
   * Default: [60, 200]
   */
  bpmRange: [number, number];

  /**
   * Smoothing factor for BPM changes (0-1).
   * Higher = smoother but slower to adapt. Lower = responsive but jittery.
   * Default: 0.1
   */
  bpmSmoothingFactor: number;

  /**
   * Beat prediction window in milliseconds.
   * Onsets within ±beatWindow ms of predicted time confirm the beat.
   * Default: 80ms
   */
  beatWindowMs: number;

  /**
   * Minimum time (ms) between detected beats.
   * Prevents double-triggers from complex transients.
   * Default: 150ms (400 BPM max)
   */
  refractoryPeriodMs: number;

  /**
   * Minimum tempo confidence to use prediction.
   * Below this, falls back to onset-only detection.
   * Default: 0.4
   */
  minTempoConfidence: number;

  /**
   * Rate at which beat phase is corrected toward detected onsets (0-1).
   * Higher = snaps to onsets faster. Lower = more stable phase.
   * Default: 0.3
   */
  phaseCorrectionRate: number;

  /**
   * Minimum energy threshold below which beat detection is paused.
   * Prevents false triggers during quiet sections.
   * Default: 0.005
   */
  silenceThreshold: number;

  /**
   * Long-term history window (frames) for baseline reference.
   * Used to cap adaptive threshold and prevent drift during loud sections.
   * Default: 172 (~4s at 43fps)
   */
  longTermHistorySize: number;

  /**
   * Max threshold as proportion of long-term max onset value.
   * Prevents threshold from rising too high during sustained loud sections.
   * Default: 0.5
   */
  thresholdCeiling: number;

  /**
   * Consecutive missed predictions before confidence decay kicks in.
   * Helps system recover when tracking drifts out of sync.
   * Default: 4
   */
  maxConsecutiveMisses: number;

  /**
   * Seconds without any beats firing before forcing onset-only mode.
   * Acts as a recovery mechanism when tracking gets stuck.
   * Default: 3
   */
  beatTimeoutSeconds: number;

  /**
   * Enable debug logging for state transitions and recovery events.
   * Default: false
   */
  debugLogging: boolean;
}
