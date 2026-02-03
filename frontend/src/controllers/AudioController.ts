/**
 * AudioController - Manages audio input, analysis, and beat detection.
 * Replaces the useAudioAnalyzer React hook.
 */

import { EventBus, eventBus } from '../core/EventBus';
import { AudioInputManager, PlaybackState, AudioSource, AudioChunkCallback } from '../audio/AudioInputManager';
import { AudioAnalyzer } from '../audio/AudioAnalyzer';
import { AdvancedBeatDetector, BeatInfo } from '../audio/AdvancedBeatDetector';
import { AudioMetrics, AudioFileInfo, BeatDebugInfo } from '../audio/types';

const defaultMetrics: AudioMetrics = {
  rms: 0,
  peak: 0,
  bass: 0,
  mid: 0,
  treble: 0,
  rawBass: 0,
  rawMid: 0,
  rawTreble: 0,
  spectralCentroid: 0.5,
  rawSpectralCentroid: 0.5,
  onset: {
    isOnset: false,
    confidence: 0,
    strength: 0,
    spectralFlux: 0,
  },
  chroma: {
    bins: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    energy: 0,
    noteNames: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
  },
  dominantChroma: 0,
  frequencyData: new Uint8Array(1024),
  timeData: new Uint8Array(2048),
  sampleRate: 48000,
  fftSize: 2048,
  bpm: 0,
  isBeat: false,
};

export class AudioController {
  // State
  private _fileInfo: AudioFileInfo | null = null;
  private _playbackState: PlaybackState = 'idle';
  private _audioSource: AudioSource = 'file';
  private _currentTime = 0;
  private _duration = 0;
  private _metrics: AudioMetrics = defaultMetrics;
  private _beatDebugInfo: BeatDebugInfo | null = null;

  // Internal references
  private inputManager: AudioInputManager;
  private analyzer: AudioAnalyzer;
  private beatDetector: AdvancedBeatDetector;
  private rafId: number | null = null;
  private disposed = false;

  // Local event bus for state change notifications
  private localBus = new EventBus();

  constructor() {
    this.inputManager = new AudioInputManager();
    this.analyzer = new AudioAnalyzer();
    this.beatDetector = new AdvancedBeatDetector();

    this.setupEventListeners();
    this.startAnalysisLoop();
  }

  // === Getters ===

  get fileInfo(): AudioFileInfo | null {
    return this._fileInfo;
  }

  get playbackState(): PlaybackState {
    return this._playbackState;
  }

  get audioSource(): AudioSource {
    return this._audioSource;
  }

  get currentTime(): number {
    return this._currentTime;
  }

  get duration(): number {
    return this._duration;
  }

  get metrics(): AudioMetrics {
    return this._metrics;
  }

  get beatDebugInfo(): BeatDebugInfo | null {
    return this._beatDebugInfo;
  }

  // === Event subscription ===

  /**
   * Subscribe to state changes
   */
  on(event: 'fileLoaded' | 'playbackStateChange' | 'timeUpdate' | 'metricsUpdate' | 'beatDebugUpdate', callback: (data: any) => void): () => void {
    return this.localBus.on(event, callback);
  }

  // === Actions ===

  async loadFile(file: File): Promise<void> {
    this.inputManager.stopStream();
    await this.inputManager.loadFile(file);
  }

  async captureSystemAudio(): Promise<void> {
    await this.inputManager.captureSystemAudio();
  }

  async captureMicrophone(): Promise<void> {
    await this.inputManager.captureMicrophone();
  }

  stopCapture(): void {
    this.inputManager.stopStream();
    this.beatDetector.reset();
    this._metrics = defaultMetrics;
    this._fileInfo = null;
    this._playbackState = 'idle';
    this._audioSource = 'file';
    this.emitStateChange();
  }

  play(): void {
    this.inputManager.play();
  }

  pause(): void {
    this.inputManager.pause();
  }

  stop(): void {
    this.inputManager.stop();
    this.beatDetector.reset();
    this._metrics = defaultMetrics;
    this.localBus.emit('metricsUpdate', this._metrics);
    eventBus.emit('audio:metrics', this._metrics);
  }

  seek(time: number): void {
    this.inputManager.seek(time);
  }

  setVolume(volume: number): void {
    this.inputManager.setVolume(volume);
  }

  // Audio capture for lyric detection
  startAudioCapture(onChunk: AudioChunkCallback): void {
    this.inputManager.startAudioCapture(onChunk);
  }

  stopAudioCapture(): void {
    this.inputManager.stopAudioCapture();
  }

  // Audio export stream
  getAudioStreamForExport(): MediaStream | null {
    return this.inputManager.getAudioStreamForExport();
  }

  stopAudioStreamExport(): void {
    this.inputManager.stopAudioStreamExport();
  }

  // === Internal methods ===

  private setupEventListeners(): void {
    this.inputManager.on('onFileLoaded', (info: AudioFileInfo) => {
      this._fileInfo = info;
      this._duration = info.duration;
      this.localBus.emit('fileLoaded', info);
      eventBus.emit('audio:loaded', { name: info.name, duration: info.duration });
    });

    this.inputManager.on('onPlaybackStateChange', (state: PlaybackState) => {
      this._playbackState = state;
      this._audioSource = this.inputManager.getAudioSource();

      if (state === 'idle') {
        this.beatDetector.reset();
      }

      this.localBus.emit('playbackStateChange', { state, source: this._audioSource });

      if (state === 'playing') {
        eventBus.emit('audio:play');
      } else if (state === 'paused') {
        eventBus.emit('audio:pause');
      } else if (state === 'idle') {
        eventBus.emit('audio:stop');
      }
    });

    this.inputManager.on('onTimeUpdate', (time: number, dur: number) => {
      this._currentTime = time;
      this._duration = dur;
      this.localBus.emit('timeUpdate', { currentTime: time, duration: dur });
      eventBus.emit('audio:timeUpdate', { currentTime: time, duration: dur });
    });
  }

  private startAnalysisLoop(): void {
    const analyze = () => {
      if (this.disposed) return;

      const analyserNode = this.inputManager.getAnalyserNode();
      const audioContext = this.inputManager.getAudioContext();

      if (analyserNode && audioContext && this.inputManager.getState() === 'playing') {
        const analysisResult = this.analyzer.analyze(analyserNode, audioContext.sampleRate);

        // Advanced beat detection
        const beatInfo: BeatInfo = this.beatDetector.detect(
          analysisResult.frequencyData,
          audioContext.sampleRate,
          analyserNode.fftSize
        );

        // Extract debug info
        const onsetResult = this.beatDetector.getLastOnsetResult();
        const tempoResult = this.beatDetector.getLastTempoResult();
        const beatDecision = this.beatDetector.getLastBeatDecision();

        if (onsetResult && beatDecision) {
          this._beatDebugInfo = {
            bandFlux: onsetResult.bandFlux,
            onsetStrength: onsetResult.strength,
            onsetThreshold: onsetResult.threshold,
            isOnset: onsetResult.isOnset,
            bpm: beatDecision.bpm,
            tempoConfidence: tempoResult?.confidence ?? 0,
            schedulerState: beatDecision.schedulerState,
            phase: beatDecision.phase,
            nextBeatTime: beatDecision.nextBeatTime,
            isBeat: beatDecision.isBeat,
            beatStrength: beatDecision.strength,
          };
          this.localBus.emit('beatDebugUpdate', this._beatDebugInfo);
          eventBus.emit('audio:beatDebug', this._beatDebugInfo);
        }

        this._metrics = {
          ...analysisResult,
          bpm: beatInfo.bpm,
          isBeat: beatInfo.isBeat,
        };
      } else {
        this._metrics = defaultMetrics;
        this._beatDebugInfo = null;
      }

      this.localBus.emit('metricsUpdate', this._metrics);
      eventBus.emit('audio:metrics', this._metrics);

      this.rafId = requestAnimationFrame(analyze);
    };

    this.rafId = requestAnimationFrame(analyze);
  }

  private emitStateChange(): void {
    this.localBus.emit('playbackStateChange', {
      state: this._playbackState,
      source: this._audioSource,
    });
    this.localBus.emit('fileLoaded', this._fileInfo);
    this.localBus.emit('metricsUpdate', this._metrics);
  }

  // === Cleanup ===

  dispose(): void {
    this.disposed = true;

    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
    }

    this.inputManager.dispose();
    this.localBus.clear();
  }
}
