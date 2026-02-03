import { SUPPORTED_EXTENSIONS, AudioFileInfo } from './types';

export type PlaybackState = 'idle' | 'playing' | 'paused' | 'loading';
export type AudioSource = 'file' | 'system' | 'microphone';

export interface AudioInputEvents {
  onFileLoaded: (info: AudioFileInfo) => void;
  onPlaybackStateChange: (state: PlaybackState) => void;
  onTimeUpdate: (currentTime: number, duration: number) => void;
  onError: (error: Error) => void;
}

// Callback for raw audio chunks (for lyric detection)
export type AudioChunkCallback = (audioData: ArrayBuffer, sampleRate: number) => void;

export class AudioInputManager {
  private audioContext: AudioContext | null = null;
  private audioBuffer: AudioBuffer | null = null;
  private sourceNode: AudioBufferSourceNode | null = null;
  private mediaStreamSource: MediaStreamAudioSourceNode | null = null;
  private mediaStream: MediaStream | null = null;
  private gainNode: GainNode | null = null;
  private analyserNode: AnalyserNode | null = null;

  private state: PlaybackState = 'idle';
  private audioSource: AudioSource = 'file';
  private startTime = 0;
  private pauseTime = 0;
  private rafId: number | null = null;

  private events: Partial<AudioInputEvents> = {};

  // Raw audio capture for lyric detection
  private scriptProcessor: ScriptProcessorNode | null = null;
  private audioChunkBuffer: Float32Array[] = [];
  private audioChunkCallback: AudioChunkCallback | null = null;
  private chunkAccumulatedSamples = 0;
  private readonly TARGET_SAMPLE_RATE = 16000;  // Whisper expects 16kHz
  private readonly CHUNK_DURATION_SEC = 0.5;  // Send chunks every 0.5 seconds

  // Audio export stream
  private exportDestination: MediaStreamAudioDestinationNode | null = null;

  constructor() {
    this.initializeContext();
  }

  private initializeContext(): void {
    // AudioContext is created on user interaction to comply with browser policies
  }

  public getAudioSource(): AudioSource {
    return this.audioSource;
  }

  private ensureContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    }
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    return this.audioContext;
  }

  public on<K extends keyof AudioInputEvents>(event: K, callback: AudioInputEvents[K]): void {
    this.events[event] = callback;
  }

  public isFormatSupported(filename: string): boolean {
    const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'));
    return SUPPORTED_EXTENSIONS.includes(ext);
  }

  public async loadFile(file: File): Promise<AudioFileInfo> {
    this.setState('loading');

    if (!this.isFormatSupported(file.name)) {
      const error = new Error(`Unsupported audio format: ${file.name}`);
      this.events.onError?.(error);
      this.setState('idle');
      throw error;
    }

    try {
      const ctx = this.ensureContext();
      const arrayBuffer = await file.arrayBuffer();
      this.audioBuffer = await ctx.decodeAudioData(arrayBuffer);

      const info: AudioFileInfo = {
        name: file.name,
        sampleRate: this.audioBuffer.sampleRate,
        channels: this.audioBuffer.numberOfChannels,
        duration: this.audioBuffer.duration,
      };

      this.pauseTime = 0;
      this.setState('idle');
      this.events.onFileLoaded?.(info);
      return info;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to load audio file');
      this.events.onError?.(error);
      this.setState('idle');
      throw error;
    }
  }

  public getAnalyserNode(): AnalyserNode | null {
    return this.analyserNode;
  }

  public getAudioContext(): AudioContext | null {
    return this.audioContext;
  }

  public play(): void {
    if (!this.audioBuffer || this.state === 'playing') return;

    const ctx = this.ensureContext();

    // Create fresh nodes
    this.sourceNode = ctx.createBufferSource();
    this.sourceNode.buffer = this.audioBuffer;

    this.gainNode = ctx.createGain();
    this.analyserNode = ctx.createAnalyser();
    this.analyserNode.fftSize = 2048;
    this.analyserNode.smoothingTimeConstant = 0.8;

    // Connect nodes: source -> gain -> analyser -> destination
    this.sourceNode.connect(this.gainNode);
    this.gainNode.connect(this.analyserNode);
    this.analyserNode.connect(ctx.destination);

    // Reconnect to export destination if active
    if (this.exportDestination) {
      this.gainNode.connect(this.exportDestination);
    }

    // Start playback
    this.startTime = ctx.currentTime - this.pauseTime;
    this.sourceNode.start(0, this.pauseTime);
    this.setState('playing');

    // Handle playback end
    this.sourceNode.onended = () => {
      if (this.state === 'playing') {
        this.pauseTime = 0;
        this.setState('idle');
      }
    };

    // Start time update loop
    this.startTimeUpdateLoop();
  }

  public pause(): void {
    if (this.state !== 'playing' || !this.audioContext) return;

    this.pauseTime = this.audioContext.currentTime - this.startTime;
    this.stopSource();
    this.setState('paused');
    this.stopTimeUpdateLoop();
  }

  public stop(): void {
    this.pauseTime = 0;
    this.stopSource();
    this.setState('idle');
    this.stopTimeUpdateLoop();
    this.events.onTimeUpdate?.(0, this.audioBuffer?.duration || 0);
  }

  public seek(time: number): void {
    if (!this.audioBuffer) return;

    const wasPlaying = this.state === 'playing';
    this.pauseTime = Math.max(0, Math.min(time, this.audioBuffer.duration));

    if (wasPlaying) {
      this.stopSource();
      this.play();
    } else {
      this.events.onTimeUpdate?.(this.pauseTime, this.audioBuffer.duration);
    }
  }

  public setVolume(volume: number): void {
    if (this.gainNode) {
      this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
    }
  }

  public getCurrentTime(): number {
    if (!this.audioContext) return 0;
    if (this.state === 'playing') {
      return this.audioContext.currentTime - this.startTime;
    }
    return this.pauseTime;
  }

  public getDuration(): number {
    return this.audioBuffer?.duration || 0;
  }

  public getState(): PlaybackState {
    return this.state;
  }

  public hasAudio(): boolean {
    return this.audioBuffer !== null || this.mediaStream !== null;
  }

  public isStreamActive(): boolean {
    return this.mediaStream !== null && this.state === 'playing';
  }

  // Capture system audio from a browser tab
  public async captureSystemAudio(): Promise<void> {
    this.stopStream();
    this.setState('loading');

    try {
      const ctx = this.ensureContext();

      // Request display media with audio
      // User will be prompted to select a tab/window to capture
      this.mediaStream = await navigator.mediaDevices.getDisplayMedia({
        video: true, // Required, but we only use audio
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      // Attempt to refocus the EASE tab after user completes selection
      // Note: browsers may ignore this due to focus-stealing prevention policies
      window.focus();

      // Check if audio track is available
      const audioTracks = this.mediaStream.getAudioTracks();
      if (audioTracks.length === 0) {
        throw new Error('No audio track available. Make sure to check "Share audio" when selecting the tab.');
      }

      // Stop video tracks - we only need audio
      this.mediaStream.getVideoTracks().forEach(track => track.stop());

      // Create audio source from stream
      this.mediaStreamSource = ctx.createMediaStreamSource(this.mediaStream);

      // Setup audio chain
      this.gainNode = ctx.createGain();
      this.analyserNode = ctx.createAnalyser();
      this.analyserNode.fftSize = 2048;
      this.analyserNode.smoothingTimeConstant = 0.8;

      // Connect: stream -> gain -> analyser (don't connect to destination to avoid feedback)
      this.mediaStreamSource.connect(this.gainNode);
      this.gainNode.connect(this.analyserNode);

      this.audioSource = 'system';
      this.audioBuffer = null;
      this.setState('playing');

      // Handle stream ending
      audioTracks[0].onended = () => {
        this.stopStream();
      };

      const info: AudioFileInfo = {
        name: 'System Audio',
        sampleRate: ctx.sampleRate,
        channels: 1,
        duration: Infinity,
      };
      this.events.onFileLoaded?.(info);

    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to capture system audio');
      this.events.onError?.(error);
      this.setState('idle');
      throw error;
    }
  }

  // Capture microphone input
  public async captureMicrophone(): Promise<void> {
    this.stopStream();
    this.setState('loading');

    try {
      const ctx = this.ensureContext();

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      this.mediaStreamSource = ctx.createMediaStreamSource(this.mediaStream);

      this.gainNode = ctx.createGain();
      this.analyserNode = ctx.createAnalyser();
      this.analyserNode.fftSize = 2048;
      this.analyserNode.smoothingTimeConstant = 0.8;

      this.mediaStreamSource.connect(this.gainNode);
      this.gainNode.connect(this.analyserNode);
      // Don't connect to destination to avoid feedback

      this.audioSource = 'microphone';
      this.audioBuffer = null;
      this.setState('playing');

      const info: AudioFileInfo = {
        name: 'Microphone',
        sampleRate: ctx.sampleRate,
        channels: 1,
        duration: Infinity,
      };
      this.events.onFileLoaded?.(info);

    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to capture microphone');
      this.events.onError?.(error);
      this.setState('idle');
      throw error;
    }
  }

  public stopStream(): void {
    if (this.mediaStreamSource) {
      this.mediaStreamSource.disconnect();
      this.mediaStreamSource = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.audioSource !== 'file') {
      this.setState('idle');
      this.audioSource = 'file';
    }
  }

  private stopSource(): void {
    if (this.sourceNode) {
      try {
        this.sourceNode.stop();
        this.sourceNode.disconnect();
      } catch {
        // Already stopped
      }
      this.sourceNode = null;
    }
  }

  private setState(state: PlaybackState): void {
    this.state = state;
    this.events.onPlaybackStateChange?.(state);
  }

  private startTimeUpdateLoop(): void {
    const update = () => {
      if (this.state === 'playing' && this.audioBuffer) {
        const currentTime = this.getCurrentTime();
        this.events.onTimeUpdate?.(currentTime, this.audioBuffer.duration);
        this.rafId = requestAnimationFrame(update);
      }
    };
    this.rafId = requestAnimationFrame(update);
  }

  private stopTimeUpdateLoop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  /**
   * Start capturing raw audio samples for lyric detection.
   * Audio is downsampled to 16kHz mono and sent in ~1 second chunks.
   *
   * @param onChunk Callback that receives ArrayBuffer of 16-bit PCM samples at 16kHz
   */
  public startAudioCapture(onChunk: AudioChunkCallback): void {
    if (this.scriptProcessor) {
      // Already capturing
      return;
    }

    const ctx = this.audioContext;
    if (!ctx) {
      console.warn('Cannot start audio capture: no audio context');
      return;
    }

    this.audioChunkCallback = onChunk;
    this.audioChunkBuffer = [];
    this.chunkAccumulatedSamples = 0;

    // Create ScriptProcessorNode to capture samples
    // Using 4096 buffer size for balance between latency and efficiency
    const bufferSize = 4096;
    this.scriptProcessor = ctx.createScriptProcessor(bufferSize, 1, 1);

    // Calculate samples needed per chunk (at source rate, will downsample later)
    const sourceSampleRate = ctx.sampleRate;
    const samplesPerChunk = Math.floor(sourceSampleRate * this.CHUNK_DURATION_SEC);

    this.scriptProcessor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);

      // Accumulate samples
      this.audioChunkBuffer.push(new Float32Array(inputData));
      this.chunkAccumulatedSamples += inputData.length;

      // When we have enough samples for a chunk, process and send
      if (this.chunkAccumulatedSamples >= samplesPerChunk) {
        this.processAndSendChunk(sourceSampleRate);
      }
    };

    // Connect script processor to the audio chain
    // It needs to be connected both as input and output (even if we don't use output)
    if (this.analyserNode) {
      console.log('Audio capture: connecting via analyserNode');
      this.analyserNode.connect(this.scriptProcessor);
      this.scriptProcessor.connect(ctx.destination);  // Required for it to work
    } else if (this.gainNode) {
      console.log('Audio capture: connecting via gainNode');
      this.gainNode.connect(this.scriptProcessor);
      this.scriptProcessor.connect(ctx.destination);
    } else {
      console.warn('Audio capture: no audio nodes available to connect!');
      this.scriptProcessor = null;
      return;
    }

    console.log('Audio capture started for lyric detection, sampleRate:', ctx.sampleRate);
  }

  /**
   * Stop capturing raw audio samples.
   */
  public stopAudioCapture(): void {
    if (!this.scriptProcessor) {
      return; // Nothing to stop
    }
    this.scriptProcessor.disconnect();
    this.scriptProcessor.onaudioprocess = null;
    this.scriptProcessor = null;
    this.audioChunkCallback = null;
    this.audioChunkBuffer = [];
    this.chunkAccumulatedSamples = 0;
    console.log('Audio capture stopped');
  }

  /**
   * Get a MediaStream containing the audio for export.
   * Creates a MediaStreamAudioDestinationNode and connects it to the gain node.
   * For file playback, this connection is re-established in play().
   */
  public getAudioStreamForExport(): MediaStream | null {
    const ctx = this.audioContext;
    if (!ctx || !this.gainNode) {
      return null;
    }

    // Create destination node if not exists
    if (!this.exportDestination) {
      this.exportDestination = ctx.createMediaStreamDestination();
    }

    // Connect gain node to export destination
    try {
      this.gainNode.connect(this.exportDestination);
    } catch {
      // Already connected
    }

    return this.exportDestination.stream;
  }

  /**
   * Stop the audio export stream and clean up.
   */
  public stopAudioStreamExport(): void {
    if (this.exportDestination && this.gainNode) {
      try {
        this.gainNode.disconnect(this.exportDestination);
      } catch {
        // Not connected
      }
    }
    this.exportDestination = null;
  }

  /**
   * Process accumulated audio chunks: merge, downsample to 16kHz, convert to 16-bit PCM.
   */
  private processAndSendChunk(sourceSampleRate: number): void {
    if (!this.audioChunkCallback || this.audioChunkBuffer.length === 0) {
      return;
    }
    console.log('Processing audio chunk, buffers:', this.audioChunkBuffer.length);

    // Merge all accumulated chunks
    const totalSamples = this.audioChunkBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
    const mergedAudio = new Float32Array(totalSamples);
    let offset = 0;
    for (const chunk of this.audioChunkBuffer) {
      mergedAudio.set(chunk, offset);
      offset += chunk.length;
    }

    // Clear buffer
    this.audioChunkBuffer = [];
    this.chunkAccumulatedSamples = 0;

    // Downsample to 16kHz using linear interpolation
    const resampleRatio = this.TARGET_SAMPLE_RATE / sourceSampleRate;
    const outputLength = Math.floor(mergedAudio.length * resampleRatio);
    const resampled = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i / resampleRatio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, mergedAudio.length - 1);
      const t = srcIndex - srcIndexFloor;
      resampled[i] = mergedAudio[srcIndexFloor] * (1 - t) + mergedAudio[srcIndexCeil] * t;
    }

    // Convert float32 [-1, 1] to int16 [-32768, 32767]
    const int16Audio = new Int16Array(resampled.length);
    for (let i = 0; i < resampled.length; i++) {
      const sample = Math.max(-1, Math.min(1, resampled[i]));
      int16Audio[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }

    // Send to callback
    this.audioChunkCallback(int16Audio.buffer, this.TARGET_SAMPLE_RATE);
  }

  public dispose(): void {
    this.stop();
    this.stopStream();
    this.stopAudioCapture();
    this.stopAudioStreamExport();
    this.audioBuffer = null;
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}
