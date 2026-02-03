/**
 * OnsetDetector - Multi-band Spectral Flux Onset Detection
 *
 * This module detects musical onsets (note attacks, drum hits) using spectral flux
 * analysis across multiple frequency bands. Spectral flux measures the rate of
 * change in the frequency spectrum, which spikes during transients.
 *
 * ## Algorithm Overview
 *
 * 1. **Multi-band Analysis**: Split the spectrum into perceptually relevant bands:
 *    - Sub-bass (20-80 Hz): Kick drums, sub-bass synths
 *    - Bass (80-250 Hz): Bass guitar, bass synths
 *    - Mid (500-2000 Hz): Snares, vocals
 *    - High-mid (2000-4000 Hz): Hi-hats, cymbals
 *
 * 2. **Spectral Flux**: For each band, compute half-wave rectified difference:
 *    SF(t) = sum(max(0, current[k] - previous[k])) / numBins
 *    Half-wave rectification (only positive changes) focuses on energy increases.
 *
 * 3. **Adaptive Threshold**: Use median + k*MAD for robust threshold:
 *    - Median is resistant to outliers (unlike mean)
 *    - MAD (Median Absolute Deviation) measures spread robustly
 *    - This adapts to varying dynamics without manual threshold tuning
 *
 * 4. **Weighted Combination**: Combine band fluxes with perceptual weights
 *    emphasizing low frequencies (kick/bass are primary beat carriers).
 *
 * ## References
 *
 * - Bello et al. "A Tutorial on Onset Detection in Music Signals" (2005)
 * - Dixon, S. "Onset Detection Revisited" (2006)
 */

import { CircularBuffer } from './utils/CircularBuffer';
import { BandName, OnsetResult, AdvancedBeatDetectorConfig } from './types';

/**
 * Frequency band definition with Hz range.
 */
interface FrequencyBand {
  name: BandName;
  lowHz: number;
  highHz: number;
}

/**
 * Default frequency bands targeting percussive elements.
 *
 * These ranges are based on common musical instrument frequency content:
 * - Kick drum fundamental: 60-100 Hz
 * - Bass guitar: 40-400 Hz
 * - Snare drum: 150-250 Hz (body), 3-10 kHz (snares)
 * - Hi-hat: 300 Hz - 10 kHz
 */
const FREQUENCY_BANDS: FrequencyBand[] = [
  { name: 'subBass', lowHz: 20, highHz: 80 },
  { name: 'bass', lowHz: 80, highHz: 250 },
  { name: 'mid', lowHz: 500, highHz: 2000 },
  { name: 'highMid', lowHz: 2000, highHz: 4000 },
];

/**
 * Default weights for combining bands.
 * Emphasizes low frequencies where beat information is strongest.
 */
const DEFAULT_BAND_WEIGHTS: Record<BandName, number> = {
  subBass: 0.4,
  bass: 0.3,
  mid: 0.2,
  highMid: 0.1,
};

export class OnsetDetector {
  // Configuration
  private bandWeights: Record<BandName, number>;
  private thresholdMultiplier: number;
  private historySize: number;
  private silenceThreshold: number;
  private longTermHistorySize: number;
  private thresholdCeiling: number;
  private debugLogging: boolean = false;

  // Debug stats
  private debugStats = {
    lastLogTime: 0,
    onsetsDetected: 0,
    silenceFrames: 0,
    thresholdCaps: 0,
    totalFrames: 0,
  };

  // State for each frequency band
  private previousBandMagnitudes: Map<BandName, Float32Array> = new Map();
  private bandFluxHistory: Map<BandName, CircularBuffer> = new Map();

  // Combined onset detection function history (for tempo estimation)
  private onsetFunctionHistory: CircularBuffer;

  // Long-term onset history for threshold ceiling calculation
  private longTermOnsetHistory: CircularBuffer;

  // FFT parameters (set on first frame)
  private sampleRate: number = 44100;
  private fftSize: number = 2048;
  private initialized: boolean = false;

  // Timing
  private startTime: number = 0;

  // Silence tracking
  private consecutiveSilentFrames: number = 0;
  private readonly silentFrameThreshold: number = 30; // ~0.5s at 60fps = sustained silence

  /**
   * Creates a new OnsetDetector.
   *
   * @param config - Partial configuration (unspecified values use defaults)
   */
  constructor(config: Partial<AdvancedBeatDetectorConfig> = {}) {
    this.bandWeights = config.bandWeights ?? DEFAULT_BAND_WEIGHTS;
    this.thresholdMultiplier = config.thresholdMultiplier ?? 2.0;
    this.historySize = config.onsetHistorySize ?? 22;
    this.silenceThreshold = config.silenceThreshold ?? 0.005;
    this.longTermHistorySize = config.longTermHistorySize ?? 172; // ~4s at 43fps
    this.thresholdCeiling = config.thresholdCeiling ?? 0.5;
    this.debugLogging = config.debugLogging ?? false;

    // Initialize history buffers for each band
    for (const band of FREQUENCY_BANDS) {
      this.bandFluxHistory.set(band.name, new CircularBuffer(this.historySize));
    }

    // Combined onset function history (used by TempoEstimator)
    this.onsetFunctionHistory = new CircularBuffer(this.historySize);

    // Long-term history for threshold ceiling calculation
    this.longTermOnsetHistory = new CircularBuffer(this.longTermHistorySize);

    this.startTime = performance.now();
  }

  /**
   * Initialize FFT bin mappings based on audio parameters.
   *
   * @param sampleRate - Audio sample rate in Hz (e.g., 44100, 48000)
   * @param fftSize - FFT size (e.g., 2048)
   */
  private initialize(sampleRate: number, fftSize: number): void {
    this.sampleRate = sampleRate;
    this.fftSize = fftSize;
    this.initialized = true;

    // Pre-allocate arrays for previous magnitudes
    // Each band only needs to store magnitudes for its frequency range
    for (const band of FREQUENCY_BANDS) {
      const { lowBin, highBin } = this.getBinRange(band);
      const numBins = highBin - lowBin + 1;
      this.previousBandMagnitudes.set(band.name, new Float32Array(numBins));
    }
  }

  /**
   * Converts a frequency in Hz to an FFT bin index.
   *
   * The relationship is: frequency = bin * (sampleRate / fftSize)
   * Therefore: bin = frequency * (fftSize / sampleRate)
   *
   * @param hz - Frequency in Hz
   * @returns FFT bin index (clamped to valid range)
   */
  private hzToBin(hz: number): number {
    const binWidth = this.sampleRate / this.fftSize;
    const bin = Math.round(hz / binWidth);
    // frequencyData only has fftSize/2 bins (Nyquist)
    return Math.max(0, Math.min(this.fftSize / 2 - 1, bin));
  }

  /**
   * Gets the FFT bin range for a frequency band.
   *
   * @param band - Frequency band definition
   * @returns Object with lowBin and highBin indices
   */
  private getBinRange(band: FrequencyBand): { lowBin: number; highBin: number } {
    return {
      lowBin: this.hzToBin(band.lowHz),
      highBin: this.hzToBin(band.highHz),
    };
  }

  /**
   * Computes spectral flux for a single frequency band.
   *
   * Spectral flux measures spectral change between frames:
   *   SF = sum(HWR(current[k] - previous[k])) / N
   *
   * Where HWR (Half-Wave Rectification) = max(0, x) keeps only positive changes.
   * This focuses on energy increases (attacks) while ignoring decreases (decays).
   *
   * The magnitude values from frequencyData are in dB (0-255 scale from Web Audio API).
   * We convert to linear scale for more meaningful flux computation.
   *
   * @param frequencyData - Uint8Array from AnalyserNode.getByteFrequencyData()
   * @param band - Frequency band to analyze
   * @returns Spectral flux value (normalized to 0-1 range approximately)
   */
  private computeBandFlux(frequencyData: Uint8Array, band: FrequencyBand): number {
    const { lowBin, highBin } = this.getBinRange(band);
    const previousMags = this.previousBandMagnitudes.get(band.name)!;
    const numBins = highBin - lowBin + 1;

    let flux = 0;

    for (let i = 0; i < numBins; i++) {
      const binIndex = lowBin + i;

      // Convert from dB scale (0-255) to linear (0-1)
      // Web Audio getByteFrequencyData uses: value = (dB + 100) * 255 / 140
      // where dB is in range [-100, 40]. So linear ≈ 10^((value * 140/255 - 100) / 20)
      // Simplified: treat as normalized magnitude for relative comparison
      const currentMag = frequencyData[binIndex] / 255;
      const previousMag = previousMags[i];

      // Half-wave rectified difference (only positive changes matter)
      const diff = currentMag - previousMag;
      if (diff > 0) {
        flux += diff;
      }

      // Update previous magnitude for next frame
      previousMags[i] = currentMag;
    }

    // Normalize by number of bins to make comparable across bands
    return numBins > 0 ? flux / numBins : 0;
  }

  /**
   * Main detection method. Call once per audio frame.
   *
   * @param frequencyData - Uint8Array from AnalyserNode.getByteFrequencyData()
   * @param sampleRate - Audio sample rate in Hz
   * @param fftSize - FFT size used
   * @returns OnsetResult with detection decision and debug info
   */
  detect(frequencyData: Uint8Array, sampleRate: number, fftSize: number): OnsetResult {
    const timestamp = performance.now() - this.startTime;

    // Initialize on first call (or if parameters changed)
    if (!this.initialized || this.sampleRate !== sampleRate || this.fftSize !== fftSize) {
      this.initialize(sampleRate, fftSize);
    }

    // Track frame for debugging
    this.debugStats.totalFrames++;

    // Check for silence (skip processing during quiet sections)
    const overallEnergy = this.computeOverallEnergy(frequencyData);
    if (overallEnergy < this.silenceThreshold) {
      this.consecutiveSilentFrames++;
      this.debugStats.silenceFrames++;
      this.logPeriodicSummary(timestamp);
      return {
        isOnset: false,
        strength: 0,
        bandFlux: { subBass: 0, bass: 0, mid: 0, highMid: 0 },
        threshold: 0,
        timestamp,
        isSustainedSilence: this.consecutiveSilentFrames >= this.silentFrameThreshold,
      };
    }

    // Audio is present - reset silence counter
    this.consecutiveSilentFrames = 0;

    // Compute spectral flux for each band
    const bandFlux: Record<BandName, number> = {
      subBass: 0,
      bass: 0,
      mid: 0,
      highMid: 0,
    };

    for (const band of FREQUENCY_BANDS) {
      const flux = this.computeBandFlux(frequencyData, band);
      bandFlux[band.name] = flux;

      // Add to band's history for adaptive threshold
      this.bandFluxHistory.get(band.name)!.push(flux);
    }

    // Weighted combination of band fluxes
    let combinedFlux = 0;
    let totalWeight = 0;
    for (const band of FREQUENCY_BANDS) {
      const weight = this.bandWeights[band.name];
      combinedFlux += bandFlux[band.name] * weight;
      totalWeight += weight;
    }
    if (totalWeight > 0) {
      combinedFlux /= totalWeight;
    }

    // Update combined onset function history
    this.onsetFunctionHistory.push(combinedFlux);

    // Update long-term history for threshold ceiling
    this.longTermOnsetHistory.push(combinedFlux);

    // Compute adaptive threshold using median + k * MAD
    // This is more robust than mean + k * stdDev for non-Gaussian distributions
    const threshold = this.computeAdaptiveThreshold();

    // Onset detected if combined flux exceeds adaptive threshold
    const isOnset = combinedFlux > threshold;

    // Track for debugging
    if (isOnset) {
      this.debugStats.onsetsDetected++;
    }
    this.logPeriodicSummary(timestamp);

    return {
      isOnset,
      strength: combinedFlux,
      bandFlux,
      threshold,
      timestamp,
      isSustainedSilence: false,
    };
  }

  /**
   * Computes the adaptive threshold using median + k * MAD.
   *
   * The Median Absolute Deviation (MAD) approach is preferred over
   * mean + k * stdDev because:
   * 1. Median is resistant to outliers (transients don't skew baseline)
   * 2. MAD measures spread robustly without assuming normal distribution
   * 3. Works well with the typically skewed distribution of onset values
   *
   * Additionally applies a threshold ceiling based on long-term history
   * to prevent threshold drift during sustained loud sections.
   *
   * @returns Adaptive threshold value
   */
  private computeAdaptiveThreshold(): number {
    const median = this.onsetFunctionHistory.median();
    const mad = this.onsetFunctionHistory.mad();
    const baseThreshold = median + this.thresholdMultiplier * mad;

    // VERY LOW minimum threshold - let the adaptive system do its job
    // Old value of 0.01 was actually blocking detection when flux dropped below it
    const minThreshold = 0.001;

    // KEY FIX: Cap threshold at proportion of long-term max
    // This prevents threshold from chasing loud sections until no peaks exceed it
    let maxThreshold = Infinity;
    if (this.longTermOnsetHistory.length > this.historySize) {
      const longTermMax = this.longTermOnsetHistory.max();
      maxThreshold = longTermMax * this.thresholdCeiling;

      // Track when ceiling is applied
      if (baseThreshold > maxThreshold) {
        this.debugStats.thresholdCaps++;
      }
    }

    // ADDITIONAL FIX: Threshold should never exceed recent max flux
    // This ensures we can always detect peaks relative to recent activity
    const recentMax = this.onsetFunctionHistory.max();
    const recentCeiling = recentMax * 0.8; // Threshold can't exceed 80% of recent max

    const cappedThreshold = Math.min(baseThreshold, maxThreshold, recentCeiling);

    return Math.max(minThreshold, cappedThreshold);
  }

  /**
   * Computes overall energy for silence detection.
   *
   * @param frequencyData - FFT magnitude data
   * @returns Normalized energy (0-1)
   */
  private computeOverallEnergy(frequencyData: Uint8Array): number {
    let sum = 0;
    for (let i = 0; i < frequencyData.length; i++) {
      sum += frequencyData[i];
    }
    return sum / (frequencyData.length * 255);
  }

  /**
   * Returns the onset detection function history.
   * Used by TempoEstimator for autocorrelation-based tempo detection.
   *
   * @returns Array of recent onset function values (oldest to newest)
   */
  getOnsetFunctionHistory(): number[] {
    return this.onsetFunctionHistory.toArray();
  }

  /**
   * Returns the current onset function value (most recent combined flux).
   */
  getCurrentOnsetValue(): number {
    return this.onsetFunctionHistory.getLast();
  }

  /**
   * Resets all internal state. Call when audio source changes.
   */
  reset(): void {
    for (const band of FREQUENCY_BANDS) {
      this.bandFluxHistory.get(band.name)?.clear();
      const prev = this.previousBandMagnitudes.get(band.name);
      if (prev) prev.fill(0);
    }
    this.onsetFunctionHistory.clear();
    this.longTermOnsetHistory.clear();
    this.consecutiveSilentFrames = 0;
    this.startTime = performance.now();
  }

  /**
   * Updates configuration parameters.
   *
   * @param config - Partial config with values to update
   */
  configure(config: Partial<AdvancedBeatDetectorConfig>): void {
    if (config.bandWeights) {
      this.bandWeights = config.bandWeights;
    }
    if (config.thresholdMultiplier !== undefined) {
      this.thresholdMultiplier = config.thresholdMultiplier;
    }
    if (config.silenceThreshold !== undefined) {
      this.silenceThreshold = config.silenceThreshold;
    }
    if (config.longTermHistorySize !== undefined) {
      this.longTermHistorySize = config.longTermHistorySize;
      // Resize the buffer if needed
      this.longTermOnsetHistory = new CircularBuffer(this.longTermHistorySize);
    }
    if (config.thresholdCeiling !== undefined) {
      this.thresholdCeiling = config.thresholdCeiling;
    }
    if (config.debugLogging !== undefined) {
      this.debugLogging = config.debugLogging;
    }
  }

  /**
   * Logs periodic summary of onset detection stats (compact format).
   */
  private logPeriodicSummary(timestamp: number): void {
    if (!this.debugLogging) return;

    // Log summary every 5 seconds
    if (timestamp - this.debugStats.lastLogTime < 5000) return;
    this.debugStats.lastLogTime = timestamp;

    const currentFlux = this.onsetFunctionHistory.getLast();
    const threshold = this.computeAdaptiveThreshold();
    const longTermMax = this.longTermOnsetHistory.length > 0 ? this.longTermOnsetHistory.max() : 0;

    console.log(
      `[ONSET] flux:${currentFlux.toFixed(3)} thresh:${threshold.toFixed(3)} ` +
      `ltMax:${longTermMax.toFixed(3)} | ` +
      `onsets:${this.debugStats.onsetsDetected} silence:${this.debugStats.silenceFrames} caps:${this.debugStats.thresholdCaps}`
    );

    // Reset counters
    this.debugStats.totalFrames = 0;
    this.debugStats.onsetsDetected = 0;
    this.debugStats.silenceFrames = 0;
    this.debugStats.thresholdCaps = 0;
  }
}
