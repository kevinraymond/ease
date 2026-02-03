import { AudioMetrics, AudioConfig, DEFAULT_AUDIO_CONFIG, OnsetInfo, ChromaFeatures } from './types';

// Note frequencies for chroma mapping (C4 = 261.63 Hz)
const NOTE_FREQUENCIES: number[] = [];
for (let octave = 0; octave < 10; octave++) {
  // C, C#, D, D#, E, F, F#, G, G#, A, A#, B
  const baseFreqs = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87];
  for (const freq of baseFreqs) {
    NOTE_FREQUENCIES.push(freq * Math.pow(2, octave));
  }
}

export class AudioAnalyzer {
  private config: AudioConfig;
  private frequencyData: Uint8Array;
  private timeData: Uint8Array;
  private smoothedBass = 0;
  private smoothedMid = 0;
  private smoothedTreble = 0;
  private smoothingFactor = 0.5;

  // For onset detection
  private previousFrequencyData: Uint8Array;
  private onsetThreshold = 0.15; // Spectral flux threshold
  private onsetDecay = 0.95; // Adaptive threshold decay
  private adaptiveThreshold = 0.15;

  // Smoothed spectral centroid
  private smoothedCentroid = 0.5;

  constructor(config: Partial<AudioConfig> = {}) {
    this.config = { ...DEFAULT_AUDIO_CONFIG, ...config };
    const bufferLength = this.config.fftSize / 2;
    this.frequencyData = new Uint8Array(bufferLength);
    this.timeData = new Uint8Array(this.config.fftSize);
    this.previousFrequencyData = new Uint8Array(bufferLength);
  }

  public analyze(analyser: AnalyserNode, sampleRate: number): AudioMetrics {
    // Get frequency data (0-255)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    analyser.getByteFrequencyData(this.frequencyData as any);
    // Get time domain data (0-255, centered at 128)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    analyser.getByteTimeDomainData(this.timeData as any);

    // Calculate RMS (root mean square) - overall volume level
    const rms = this.calculateRMS(this.timeData);

    // Calculate peak amplitude
    const peak = this.calculatePeak(this.timeData);

    // Calculate frequency bands
    const binWidth = sampleRate / (2 * this.frequencyData.length);
    const bass = this.calculateBandEnergy(
      this.frequencyData,
      this.config.bassRange[0],
      this.config.bassRange[1],
      binWidth
    );
    const mid = this.calculateBandEnergy(
      this.frequencyData,
      this.config.midRange[0],
      this.config.midRange[1],
      binWidth
    );
    const treble = this.calculateBandEnergy(
      this.frequencyData,
      this.config.trebleRange[0],
      this.config.trebleRange[1],
      binWidth
    );

    // Apply smoothing for visual continuity
    this.smoothedBass = this.lerp(this.smoothedBass, bass, this.smoothingFactor);
    this.smoothedMid = this.lerp(this.smoothedMid, mid, this.smoothingFactor);
    this.smoothedTreble = this.lerp(this.smoothedTreble, treble, this.smoothingFactor);

    // Calculate spectral centroid (brightness indicator)
    const centroid = this.calculateSpectralCentroid(this.frequencyData, binWidth);
    this.smoothedCentroid = this.lerp(this.smoothedCentroid, centroid, this.smoothingFactor);

    // Calculate onset detection with confidence
    const onset = this.detectOnset(this.frequencyData);

    // Calculate chroma features (12-bin pitch class distribution)
    const chroma = this.calculateChroma(this.frequencyData, binWidth);

    // Store current frequency data for next frame's onset detection
    this.previousFrequencyData.set(this.frequencyData);

    return {
      rms,
      peak,
      bass: this.smoothedBass,
      mid: this.smoothedMid,
      treble: this.smoothedTreble,
      // Raw values for beat detection (no smoothing)
      rawBass: bass,
      rawMid: mid,
      rawTreble: treble,
      // New SOTA features
      spectralCentroid: this.smoothedCentroid,
      rawSpectralCentroid: centroid,
      onset,
      chroma,
      // Dominant pitch class (0-11: C, C#, D, ..., B)
      dominantChroma: chroma.bins.indexOf(Math.max(...chroma.bins)),
      frequencyData: this.frequencyData.slice() as Uint8Array,
      timeData: this.timeData.slice() as Uint8Array,
      sampleRate,
      fftSize: this.config.fftSize,
      bpm: 0, // Set by BeatDetector
      isBeat: false, // Set by BeatDetector
    };
  }

  private calculateRMS(timeData: Uint8Array): number {
    let sum = 0;
    for (let i = 0; i < timeData.length; i++) {
      // Normalize from 0-255 to -1 to 1
      const normalized = (timeData[i] - 128) / 128;
      sum += normalized * normalized;
    }
    return Math.sqrt(sum / timeData.length);
  }

  private calculatePeak(timeData: Uint8Array): number {
    let peak = 0;
    for (let i = 0; i < timeData.length; i++) {
      const normalized = Math.abs((timeData[i] - 128) / 128);
      if (normalized > peak) {
        peak = normalized;
      }
    }
    return peak;
  }

  private calculateBandEnergy(
    frequencyData: Uint8Array,
    minHz: number,
    maxHz: number,
    binWidth: number
  ): number {
    const startBin = Math.floor(minHz / binWidth);
    const endBin = Math.min(Math.floor(maxHz / binWidth), frequencyData.length);

    if (startBin >= endBin) return 0;

    let sum = 0;
    for (let i = startBin; i < endBin; i++) {
      sum += frequencyData[i];
    }

    // Normalize to 0-1 range
    return sum / ((endBin - startBin) * 255);
  }

  private lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
  }

  /**
   * Calculate spectral centroid - "brightness" or "timbre" indicator.
   * Higher values = brighter/sharper sound, lower = darker/warmer.
   * Formula: sum(freq * magnitude) / sum(magnitude)
   * Normalized to 0-1 range based on frequency range.
   */
  private calculateSpectralCentroid(frequencyData: Uint8Array, binWidth: number): number {
    let weightedSum = 0;
    let magnitudeSum = 0;

    for (let i = 0; i < frequencyData.length; i++) {
      const magnitude = frequencyData[i] / 255; // Normalize to 0-1
      const frequency = i * binWidth;
      weightedSum += frequency * magnitude;
      magnitudeSum += magnitude;
    }

    if (magnitudeSum === 0) return 0.5; // Default to middle

    const centroidHz = weightedSum / magnitudeSum;
    // Normalize to 0-1 based on typical hearing range (20Hz - 20kHz)
    // Use logarithmic scale for perceptual accuracy
    const minFreq = 20;
    const maxFreq = 20000;
    const logMin = Math.log(minFreq);
    const logMax = Math.log(maxFreq);
    const logCentroid = Math.log(Math.max(centroidHz, minFreq));

    return Math.max(0, Math.min(1, (logCentroid - logMin) / (logMax - logMin)));
  }

  /**
   * Detect audio onsets (transients) using spectral flux.
   * More granular than simple beat detection - catches drum hits, note attacks, etc.
   */
  private detectOnset(frequencyData: Uint8Array): OnsetInfo {
    // Calculate spectral flux: sum of positive differences between frames
    let flux = 0;
    let totalDiff = 0;

    for (let i = 0; i < frequencyData.length; i++) {
      const current = frequencyData[i] / 255;
      const previous = this.previousFrequencyData[i] / 255;
      const diff = current - previous;

      // Only count positive changes (onset detection)
      if (diff > 0) {
        flux += diff;
      }
      totalDiff += Math.abs(diff);
    }

    // Normalize by number of bins
    flux /= frequencyData.length;
    totalDiff /= frequencyData.length;

    // Update adaptive threshold
    this.adaptiveThreshold = Math.max(
      this.onsetThreshold,
      this.adaptiveThreshold * this.onsetDecay
    );

    const isOnset = flux > this.adaptiveThreshold;

    // Calculate confidence (how much above threshold)
    const confidence = isOnset
      ? Math.min(1, (flux - this.adaptiveThreshold) / this.adaptiveThreshold + 0.5)
      : flux / this.adaptiveThreshold;

    // Raise adaptive threshold on onset
    if (isOnset) {
      this.adaptiveThreshold = flux * 1.5;
    }

    return {
      isOnset,
      confidence,
      strength: flux,
      spectralFlux: totalDiff,
    };
  }

  /**
   * Calculate chroma features - 12-bin pitch class distribution.
   * Maps frequency content to musical notes (C, C#, D, ..., B).
   * Useful for tonal-to-visual color correspondence.
   */
  private calculateChroma(frequencyData: Uint8Array, binWidth: number): ChromaFeatures {
    const chromaBins = new Array(12).fill(0);
    const chromaCounts = new Array(12).fill(0);

    for (let i = 1; i < frequencyData.length; i++) {
      const frequency = i * binWidth;
      if (frequency < 20 || frequency > 5000) continue; // Focus on musical range

      const magnitude = frequencyData[i] / 255;
      if (magnitude < 0.01) continue; // Skip very quiet bins

      // Find the closest musical note
      const noteNumber = 12 * Math.log2(frequency / 440) + 69; // MIDI note number
      const pitchClass = Math.round(noteNumber) % 12;
      const normalizedPitchClass = pitchClass < 0 ? pitchClass + 12 : pitchClass;

      chromaBins[normalizedPitchClass] += magnitude;
      chromaCounts[normalizedPitchClass]++;
    }

    // Normalize chroma bins to 0-1
    const maxChroma = Math.max(...chromaBins, 0.001);
    const normalizedBins = chromaBins.map((v) => v / maxChroma);

    // Calculate chroma energy (overall tonal content)
    const totalChroma = chromaBins.reduce((a, b) => a + b, 0);
    const chromaEnergy = Math.min(1, totalChroma / frequencyData.length);

    return {
      bins: normalizedBins as [number, number, number, number, number, number, number, number, number, number, number, number],
      energy: chromaEnergy,
      noteNames: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
    };
  }

  public setSmoothingFactor(factor: number): void {
    this.smoothingFactor = Math.max(0, Math.min(1, factor));
  }

  public getFrequencyData(): Uint8Array {
    return this.frequencyData;
  }

  public getTimeData(): Uint8Array {
    return this.timeData;
  }

  public getConfig(): AudioConfig {
    return { ...this.config };
  }

  public updateConfig(config: Partial<AudioConfig>): void {
    this.config = { ...this.config, ...config };
    if (config.fftSize) {
      this.frequencyData = new Uint8Array(config.fftSize / 2);
      this.timeData = new Uint8Array(config.fftSize);
      this.previousFrequencyData = new Uint8Array(config.fftSize / 2);
    }
  }
}
