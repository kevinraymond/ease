/**
 * FrequencyDebugOverlay - Read-only display of audio metrics and beat detection info.
 */

import { Component } from '../core/Component';
import { html } from '../core/html';
import { AudioMetrics, BeatDebugInfo } from '../audio/types';
import { AIGeneratorConfig, AIGeneratorState } from '../core/types';

interface FrequencyDebugOverlayState {
  aiConfig: AIGeneratorConfig | null;
  aiState: AIGeneratorState | null;
}

// Descriptions for generation modes
const MODE_DESCRIPTIONS: Record<string, { name: string; effect: string }> = {
  feedback: {
    name: 'Feedback Loop',
    effect: 'Real-time: first frame from prompt, then each frame evolves from previous',
  },
  keyframe_rife: {
    name: 'Keyframe + RIFE',
    effect: 'Real-time with pose control - generates keyframes with interpolation between',
  },
};

// Descriptions for mapping presets
const PRESET_DESCRIPTIONS: Record<string, { name: string; effects: string[] }> = {
  dancer: {
    name: 'Dancer',
    effects: [
      'Bass → Movement intensity & pose changes',
      'Mid → Color/style shifts',
      'Treble → Detail & texture variation',
      'Beat → Sharp transitions',
    ],
  },
  reactive_abstract: {
    name: 'Reactive Abstract',
    effects: [
      'Bass → Slow color morphing',
      'Mid → Shape complexity',
      'Treble → Fine detail shimmer',
      'Beat → Subtle pulse',
    ],
  },
  vj_intense: {
    name: 'VJ Intense',
    effects: [
      'Bass → Heavy color distortion',
      'Mid → Rapid style changes',
      'Treble → Glitch effects',
      'Beat → Dramatic transformation',
    ],
  },
  dreamscape: {
    name: 'Dreamscape',
    effects: [
      'Bass → Gentle drift',
      'Mid → Ethereal glow',
      'Treble → Sparkle effects',
      'Beat → Soft breathing',
    ],
  },
  minimal: {
    name: 'Minimal',
    effects: [
      'Bass → Barely perceptible movement',
      'Mid → Subtle hue shift',
      'Treble → Minor variation',
      'Beat → Gentle pulse',
    ],
  },
};

export class FrequencyDebugOverlay extends Component<FrequencyDebugOverlayState> {
  private metrics: AudioMetrics | null = null;
  private beatDebugInfo: BeatDebugInfo | null = null;

  constructor(container: HTMLElement) {
    super(container, {
      aiConfig: null,
      aiState: null,
    });
  }

  protected render(): void {
    const { aiConfig, aiState } = this.state;
    const metrics = this.metrics;
    const beatDebugInfo = this.beatDebugInfo;

    // Calculate raw band values from frequency data
    const { bass, mid, treble } = this.calculateBandEnergies(metrics);
    const spectrumBands = this.getSpectrumBands(metrics?.frequencyData, 32);

    const bassPercent = Math.round(bass * 100);
    const midPercent = Math.round(mid * 100);
    const treblePercent = Math.round(treble * 100);
    const bpm = metrics?.bpm || 0;
    const isBeat = metrics?.isBeat || false;

    this.el.className = 'frequency-debug-overlay';
    this.el.innerHTML = html`
      <div class="freq-band">
        <span class="freq-label">BASS</span>
        <div class="freq-bar-container">
          <div class="freq-bar freq-bar-bass" style="width: ${bassPercent}%"></div>
        </div>
        <span class="freq-value freq-value-bass">${bassPercent}%</span>
      </div>

      <div class="freq-band">
        <span class="freq-label">MID</span>
        <div class="freq-bar-container">
          <div class="freq-bar freq-bar-mid" style="width: ${midPercent}%"></div>
        </div>
        <span class="freq-value freq-value-mid">${midPercent}%</span>
      </div>

      <div class="freq-band">
        <span class="freq-label">TREBLE</span>
        <div class="freq-bar-container">
          <div class="freq-bar freq-bar-treble" style="width: ${treblePercent}%"></div>
        </div>
        <span class="freq-value freq-value-treble">${treblePercent}%</span>
      </div>

      <div class="spectrum-analyzer">
        ${spectrumBands.map((level, i) => html`
          <div class="spectrum-bar-container">
            <div
              class="spectrum-bar"
              data-index="${i}"
              style="height: ${level * 100}%; background-color: ${this.getSpectrumColor(i, spectrumBands.length)}"
            ></div>
          </div>
        `)}
      </div>

      <div class="freq-status">
        <span class="bpm-info">BPM: ${bpm}</span>
        <span class="beat-indicator ${isBeat ? 'active' : ''}">
          BEAT: ${isBeat ? '●' : '○'}
        </span>
      </div>
      <div class="freq-debug-extra">
        <span class="rms-value">RMS: ${((metrics?.rms || 0) * 100).toFixed(0)}%</span>
        <span class="peak-value">Peak: ${((metrics?.peak || 0) * 100).toFixed(0)}%</span>
      </div>
      <div class="freq-debug-extra freq-audio-info">
        <span class="sample-rate-value">${(metrics?.sampleRate || 0) / 1000}kHz</span>
        <span class="fft-value">FFT: ${metrics?.fftSize || 0}</span>
        <span class="bins-value">Bins: ${metrics?.frequencyData?.length || 0}</span>
      </div>
      <div class="freq-debug-extra">
        <span class="max-bin-value">Max bin: ${this.getMaxBinInfo(metrics?.frequencyData)}</span>
      </div>

      ${beatDebugInfo ? this.renderBeatDebugSection(beatDebugInfo) : ''}
      ${aiConfig && aiState ? this.renderAIInfoSection(aiConfig, aiState) : ''}
    `;
  }

  private renderBeatDebugSection(info: BeatDebugInfo): string {
    return html`
      <div class="beat-debug-section">
        <div class="beat-debug-header" title="Real-time audio analysis for beat detection and tempo estimation">
          Beat Detection
        </div>

        <div class="beat-bands" title="Spectral flux per frequency band">
          ${this.renderBeatBand('Sub', info.bandFlux.subBass, 'beat-band-subbass', 'Sub-bass (20-60 Hz)')}
          ${this.renderBeatBand('Bass', info.bandFlux.bass, 'beat-band-bass', 'Bass (60-250 Hz)')}
          ${this.renderBeatBand('Mid', info.bandFlux.mid, 'beat-band-mid', 'Mid (250-2000 Hz)')}
          ${this.renderBeatBand('High', info.bandFlux.highMid, 'beat-band-highmid', 'High-mid (2000-8000 Hz)')}
        </div>

        <div class="beat-onset-meter" title="Onset detection">
          <span class="beat-onset-label">Onset</span>
          <div class="beat-onset-bar-container">
            <div class="beat-onset-threshold" style="left: ${Math.min(100, info.onsetThreshold * 5000)}%"></div>
            <div class="beat-onset-bar ${info.isOnset ? 'onset-active' : ''}" style="width: ${Math.min(100, info.onsetStrength * 5000)}%"></div>
          </div>
          <span class="beat-onset-indicator ${info.isOnset ? 'active' : ''}">${info.isOnset ? '●' : '○'}</span>
        </div>

        <div class="beat-phase-container" title="Beat phase">
          <span class="beat-phase-label">Phase</span>
          <div class="beat-phase-track">
            <div class="beat-phase-marker ${info.isBeat ? 'beat-flash' : ''}" style="left: ${info.phase * 100}%"></div>
            <div class="beat-phase-beat-zone"></div>
          </div>
          <span class="beat-phase-value">${(info.phase * 100).toFixed(0)}%</span>
        </div>

        <div class="beat-info-row">
          <span class="beat-tempo" title="Estimated tempo">
            ${info.bpm > 0 ? `${info.bpm} BPM` : '-- BPM'}
          </span>
          <span class="beat-confidence ${this.getConfidenceClass(info.tempoConfidence)}" title="Tempo confidence">
            ${(info.tempoConfidence * 100).toFixed(0)}% conf
          </span>
        </div>
        <div class="beat-info-row">
          <span class="beat-scheduler-state ${info.schedulerState.toLowerCase()}" title="Scheduler state">
            ${info.schedulerState}
          </span>
          <span class="beat-indicator-large ${info.isBeat ? 'beat-active' : ''}">
            ${info.isBeat ? 'BEAT!' : ''}
          </span>
        </div>
      </div>
    `;
  }

  private renderBeatBand(label: string, value: number, className: string, title: string): string {
    const scaledValue = Math.min(100, value * 5000);
    return html`
      <div class="beat-band" title="${title}">
        <span class="beat-band-label">${label}</span>
        <div class="beat-band-bar-container">
          <div class="beat-band-bar ${className}" style="width: ${scaledValue}%"></div>
        </div>
        <span class="beat-band-value">${scaledValue.toFixed(0)}</span>
      </div>
    `;
  }

  private renderAIInfoSection(config: AIGeneratorConfig, state: AIGeneratorState): string {
    const modeInfo = MODE_DESCRIPTIONS[config.generationMode];
    const presetInfo = PRESET_DESCRIPTIONS[config.mappingPreset];

    return html`
      <div class="ai-info-overlay">
        <div class="ai-info-header">
          <span class="ai-info-title">AI Generation</span>
          <span class="ai-info-fps">${state.fps.toFixed(1)} FPS</span>
          <span class="ai-info-frame">Frame #${state.frameId}</span>
        </div>

        <div class="ai-info-section">
          <span class="ai-info-label">Mode:</span>
          <span class="ai-info-value">${modeInfo?.name || config.generationMode}</span>
        </div>
        <div class="ai-info-description">${modeInfo?.effect || ''}</div>

        <div class="ai-info-section">
          <span class="ai-info-label">Preset:</span>
          <span class="ai-info-value">${presetInfo?.name || config.mappingPreset}</span>
        </div>
        <div class="ai-info-effects">
          ${presetInfo?.effects.map(effect => html`<div class="ai-info-effect">${effect}</div>`) || ''}
        </div>

        ${state.lastParams ? this.renderGenerationParams(state.lastParams) : ''}

        <div class="ai-info-divider"></div>
        <div class="ai-info-sota">
          <span class="ai-sota-badge ${config.temporalCoherence === 'blending' ? 'active' : ''}">
            ${config.temporalCoherence === 'blending' ? 'Blending' : 'No Blend'}
          </span>
          <span class="ai-sota-badge ${config.useControlNet ? 'active' : ''}">
            ${config.useControlNet ? 'ControlNet' : 'No CN'}
          </span>
        </div>
      </div>
    `;
  }

  private renderGenerationParams(params: AIGeneratorState['lastParams']): string {
    if (!params) return '';

    return html`
      <div class="ai-info-divider"></div>
      <div class="ai-info-section">
        <span class="ai-info-label">Strength:</span>
        <span class="ai-info-value">${Math.round(params.strength * 100)}%</span>
        <span class="ai-info-label" style="margin-left: 8px">CFG:</span>
        <span class="ai-info-value">${params.guidance_scale.toFixed(2)}</span>
      </div>
      <div class="ai-info-section">
        <span class="ai-info-label">Seed:</span>
        <span class="ai-info-seed">${params.seed?.toString().slice(-6) || '-'}</span>
        ${params.is_onset ? html`<span class="ai-info-onset">ONSET</span>` : ''}
      </div>
      ${params.color_keywords.length > 0 ? html`
        <div class="ai-info-section">
          <span class="ai-info-label">Colors:</span>
          <span class="ai-info-colors">${params.color_keywords.join(', ')}</span>
        </div>
      ` : ''}
    `;
  }

  // === Surgical DOM update for high-frequency data ===

  private updateMetricsDOM(): void {
    if (!this.el.isConnected) return;

    const metrics = this.metrics;
    const beatDebugInfo = this.beatDebugInfo;

    // Update frequency bands
    const { bass, mid, treble } = this.calculateBandEnergies(metrics);
    const bassPercent = Math.round(bass * 100);
    const midPercent = Math.round(mid * 100);
    const treblePercent = Math.round(treble * 100);

    const bassBar = this.el.querySelector<HTMLElement>('.freq-bar-bass');
    const midBar = this.el.querySelector<HTMLElement>('.freq-bar-mid');
    const trebleBar = this.el.querySelector<HTMLElement>('.freq-bar-treble');
    const bassValue = this.el.querySelector<HTMLElement>('.freq-value-bass');
    const midValue = this.el.querySelector<HTMLElement>('.freq-value-mid');
    const trebleValue = this.el.querySelector<HTMLElement>('.freq-value-treble');

    if (bassBar) bassBar.style.width = `${bassPercent}%`;
    if (midBar) midBar.style.width = `${midPercent}%`;
    if (trebleBar) trebleBar.style.width = `${treblePercent}%`;
    if (bassValue) bassValue.textContent = `${bassPercent}%`;
    if (midValue) midValue.textContent = `${midPercent}%`;
    if (trebleValue) trebleValue.textContent = `${treblePercent}%`;

    // Update spectrum bars
    const spectrumBands = this.getSpectrumBands(metrics?.frequencyData, 32);
    const spectrumBars = this.el.querySelectorAll<HTMLElement>('.spectrum-bar');
    spectrumBars.forEach((bar, i) => {
      if (i < spectrumBands.length) {
        bar.style.height = `${spectrumBands[i] * 100}%`;
      }
    });

    // Update BPM and beat indicator
    const bpm = metrics?.bpm || 0;
    const isBeat = metrics?.isBeat || false;
    const bpmInfo = this.el.querySelector<HTMLElement>('.bpm-info');
    const beatIndicator = this.el.querySelector<HTMLElement>('.beat-indicator');
    if (bpmInfo) bpmInfo.textContent = `BPM: ${bpm}`;
    if (beatIndicator) {
      beatIndicator.textContent = `BEAT: ${isBeat ? '●' : '○'}`;
      beatIndicator.classList.toggle('active', isBeat);
    }

    // Update RMS and Peak
    const rmsValue = this.el.querySelector<HTMLElement>('.rms-value');
    const peakValue = this.el.querySelector<HTMLElement>('.peak-value');
    if (rmsValue) rmsValue.textContent = `RMS: ${((metrics?.rms || 0) * 100).toFixed(0)}%`;
    if (peakValue) peakValue.textContent = `Peak: ${((metrics?.peak || 0) * 100).toFixed(0)}%`;

    // Update audio info (these change rarely but include for completeness)
    const sampleRateValue = this.el.querySelector<HTMLElement>('.sample-rate-value');
    const fftValue = this.el.querySelector<HTMLElement>('.fft-value');
    const binsValue = this.el.querySelector<HTMLElement>('.bins-value');
    const maxBinValue = this.el.querySelector<HTMLElement>('.max-bin-value');
    if (sampleRateValue) sampleRateValue.textContent = `${(metrics?.sampleRate || 0) / 1000}kHz`;
    if (fftValue) fftValue.textContent = `FFT: ${metrics?.fftSize || 0}`;
    if (binsValue) binsValue.textContent = `Bins: ${metrics?.frequencyData?.length || 0}`;
    if (maxBinValue) maxBinValue.textContent = `Max bin: ${this.getMaxBinInfo(metrics?.frequencyData)}`;

    // Update beat debug section if present
    if (beatDebugInfo) {
      this.updateBeatDebugDOM(beatDebugInfo);
    }
  }

  private updateBeatDebugDOM(info: BeatDebugInfo): void {
    // Update band flux bars
    const subBassBar = this.el.querySelector<HTMLElement>('.beat-band-subbass');
    const bassBar = this.el.querySelector<HTMLElement>('.beat-band-bass');
    const midBar = this.el.querySelector<HTMLElement>('.beat-band-mid');
    const highMidBar = this.el.querySelector<HTMLElement>('.beat-band-highmid');

    if (subBassBar) subBassBar.style.width = `${Math.min(100, info.bandFlux.subBass * 5000)}%`;
    if (bassBar) bassBar.style.width = `${Math.min(100, info.bandFlux.bass * 5000)}%`;
    if (midBar) midBar.style.width = `${Math.min(100, info.bandFlux.mid * 5000)}%`;
    if (highMidBar) highMidBar.style.width = `${Math.min(100, info.bandFlux.highMid * 5000)}%`;

    // Update band values
    const bandValues = this.el.querySelectorAll<HTMLElement>('.beat-band-value');
    const fluxValues = [info.bandFlux.subBass, info.bandFlux.bass, info.bandFlux.mid, info.bandFlux.highMid];
    bandValues.forEach((el, i) => {
      if (i < fluxValues.length) {
        el.textContent = Math.min(100, fluxValues[i] * 5000).toFixed(0);
      }
    });

    // Update onset meter
    const onsetThreshold = this.el.querySelector<HTMLElement>('.beat-onset-threshold');
    const onsetBar = this.el.querySelector<HTMLElement>('.beat-onset-bar');
    const onsetIndicator = this.el.querySelector<HTMLElement>('.beat-onset-indicator');
    if (onsetThreshold) onsetThreshold.style.left = `${Math.min(100, info.onsetThreshold * 5000)}%`;
    if (onsetBar) {
      onsetBar.style.width = `${Math.min(100, info.onsetStrength * 5000)}%`;
      onsetBar.classList.toggle('onset-active', info.isOnset);
    }
    if (onsetIndicator) {
      onsetIndicator.textContent = info.isOnset ? '●' : '○';
      onsetIndicator.classList.toggle('active', info.isOnset);
    }

    // Update phase
    const phaseMarker = this.el.querySelector<HTMLElement>('.beat-phase-marker');
    const phaseValue = this.el.querySelector<HTMLElement>('.beat-phase-value');
    if (phaseMarker) {
      phaseMarker.style.left = `${info.phase * 100}%`;
      phaseMarker.classList.toggle('beat-flash', info.isBeat);
    }
    if (phaseValue) phaseValue.textContent = `${(info.phase * 100).toFixed(0)}%`;

    // Update tempo info
    const beatTempo = this.el.querySelector<HTMLElement>('.beat-tempo');
    const beatConfidence = this.el.querySelector<HTMLElement>('.beat-confidence');
    if (beatTempo) beatTempo.textContent = info.bpm > 0 ? `${info.bpm} BPM` : '-- BPM';
    if (beatConfidence) {
      beatConfidence.textContent = `${(info.tempoConfidence * 100).toFixed(0)}% conf`;
      beatConfidence.className = `beat-confidence ${this.getConfidenceClass(info.tempoConfidence)}`;
    }

    // Update scheduler state and beat indicator
    const schedulerState = this.el.querySelector<HTMLElement>('.beat-scheduler-state');
    const beatIndicatorLarge = this.el.querySelector<HTMLElement>('.beat-indicator-large');
    if (schedulerState) {
      schedulerState.textContent = info.schedulerState;
      schedulerState.className = `beat-scheduler-state ${info.schedulerState.toLowerCase()}`;
    }
    if (beatIndicatorLarge) {
      beatIndicatorLarge.textContent = info.isBeat ? 'BEAT!' : '';
      beatIndicatorLarge.classList.toggle('beat-active', info.isBeat);
    }
  }

  // === Helper methods ===

  private calculateBandEnergies(metrics: AudioMetrics | null): { bass: number; mid: number; treble: number } {
    if (!metrics?.frequencyData?.length) {
      return { bass: 0, mid: 0, treble: 0 };
    }

    const sampleRate = metrics.sampleRate || 48000;
    const binWidth = sampleRate / (2 * metrics.frequencyData.length);

    const bassEnd = Math.floor(250 / binWidth);
    const midEnd = Math.floor(4000 / binWidth);
    const trebleEnd = Math.floor(16000 / binWidth);

    return {
      bass: this.calculateBandEnergy(metrics.frequencyData, 1, bassEnd),
      mid: this.calculateBandEnergy(metrics.frequencyData, bassEnd, midEnd),
      treble: this.calculateBandEnergy(metrics.frequencyData, midEnd, trebleEnd),
    };
  }

  private calculateBandEnergy(frequencyData: Uint8Array, startBin: number, endBin: number): number {
    if (startBin >= endBin || !frequencyData.length) return 0;
    let sum = 0;
    const clampedEnd = Math.min(endBin, frequencyData.length);
    for (let i = startBin; i < clampedEnd; i++) {
      sum += frequencyData[i];
    }
    return sum / ((clampedEnd - startBin) * 255);
  }

  private getSpectrumBands(frequencyData: Uint8Array | undefined, bandCount: number = 32): number[] {
    if (!frequencyData?.length) return new Array(bandCount).fill(0);

    const bands: number[] = [];
    const maxBin = Math.min(frequencyData.length, 512);

    for (let i = 0; i < bandCount; i++) {
      const logMin = Math.log(1);
      const logMax = Math.log(maxBin);
      const startLog = logMin + (logMax - logMin) * (i / bandCount);
      const endLog = logMin + (logMax - logMin) * ((i + 1) / bandCount);

      const startBin = Math.floor(Math.exp(startLog));
      const endBin = Math.floor(Math.exp(endLog));

      if (startBin >= endBin) {
        bands.push(frequencyData[startBin] / 255);
      } else {
        let sum = 0;
        for (let j = startBin; j < endBin; j++) {
          sum += frequencyData[j];
        }
        bands.push(sum / ((endBin - startBin) * 255));
      }
    }

    return bands;
  }

  private getSpectrumColor(index: number, total: number): string {
    const ratio = index / total;

    if (ratio < 0.33) {
      const t = ratio / 0.33;
      return `rgb(255, ${Math.round(t * 200)}, 50)`;
    } else if (ratio < 0.66) {
      const t = (ratio - 0.33) / 0.33;
      return `rgb(${Math.round(255 - t * 155)}, ${Math.round(200 + t * 55)}, 50)`;
    } else {
      const t = (ratio - 0.66) / 0.34;
      return `rgb(${Math.round(100 - t * 100)}, ${Math.round(255 - t * 155)}, ${Math.round(50 + t * 205)})`;
    }
  }

  private getMaxBinInfo(frequencyData: Uint8Array | undefined): string {
    if (!frequencyData?.length) return '-';
    let maxVal = 0;
    let maxBin = 0;
    for (let i = 0; i < frequencyData.length; i++) {
      if (frequencyData[i] > maxVal) {
        maxVal = frequencyData[i];
        maxBin = i;
      }
    }
    const freq = Math.round(maxBin * 23.4);
    return `${maxBin} (~${freq}Hz)`;
  }

  private getConfidenceClass(confidence: number): string {
    if (confidence >= 0.7) return 'high';
    if (confidence >= 0.4) return 'medium';
    return 'low';
  }

  // === Public update methods ===

  update(data: Partial<FrequencyDebugOverlayState>): void {
    Object.assign(this.state, data);
  }

  setMetrics(metrics: AudioMetrics | null): void {
    this.metrics = metrics;
    this.updateMetricsDOM(); // Surgical update, no re-render
  }

  setBeatDebugInfo(info: BeatDebugInfo | null): void {
    const hadInfo = this.beatDebugInfo !== null;
    const hasInfo = info !== null;
    this.beatDebugInfo = info;

    // Only trigger full re-render when section appears/disappears
    if (hadInfo !== hasInfo) {
      this.scheduleRender();
    } else {
      this.updateMetricsDOM(); // Surgical update, no re-render
    }
  }

  setAIInfo(config: AIGeneratorConfig | null, state: AIGeneratorState | null): void {
    // AI info changes are infrequent - full re-render is appropriate
    this.state.aiConfig = config;
    this.state.aiState = state;
  }
}
