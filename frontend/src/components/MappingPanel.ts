/**
 * MappingPanel - Audio parameter mapping configuration panel.
 * This is a complex component with nested collapsible sections and sliders.
 */

import { Component } from '../core/Component';
import { html } from '../core/html';
import { AudioMetrics } from '../audio/types';

// Re-export types for external use
export type AudioSource =
  | 'bass' | 'mid' | 'treble' | 'rms' | 'peak'
  | 'spectral_centroid' | 'bass_mid' | 'bpm' | 'onset_strength' | 'fixed';

export type CurveType = 'linear' | 'ease_in' | 'ease_out' | 'ease_in_out' | 'exponential';

export interface ParameterMapping {
  id: string;
  name: string;
  source: AudioSource;
  curve: CurveType;
  inputMin: number;
  inputMax: number;
  outputMin: number;
  outputMax: number;
  enabled: boolean;
}

export interface TriggerConfig {
  onBeat: {
    seedJump: boolean;
    strengthBoost: number;
    forceKeyframe: boolean;
  };
  onOnset: {
    seedVariation: number;
    forceKeyframe: boolean;
  };
  chromaThreshold: number;
}

export interface CrossfeedConfig {
  enabled: boolean;
  power: number;
  range: number;
  decay: number;
}

export interface MappingConfig {
  mappings: Record<string, ParameterMapping>;
  triggers: TriggerConfig;
  crossfeed: CrossfeedConfig;
  presetName: string;
}

// Default configurations
const DEFAULT_MAPPINGS: Record<string, ParameterMapping> = {
  transformStrength: {
    id: 'transformStrength',
    name: 'Transform Strength',
    source: 'bass',
    curve: 'ease_in_out',
    inputMin: 0,
    inputMax: 1,
    outputMin: 0.3,
    outputMax: 0.7,
    enabled: true,
  },
  poseIntensity: {
    id: 'poseIntensity',
    name: 'Pose Intensity',
    source: 'bass_mid',
    curve: 'linear',
    inputMin: 0,
    inputMax: 1,
    outputMin: 0.2,
    outputMax: 1.0,
    enabled: true,
  },
  keyframeStrength: {
    id: 'keyframeStrength',
    name: 'Keyframe Strength',
    source: 'fixed',
    curve: 'linear',
    inputMin: 0,
    inputMax: 1,
    outputMin: 0.6,
    outputMax: 0.6,
    enabled: true,
  },
  animationSpeed: {
    id: 'animationSpeed',
    name: 'Animation Speed',
    source: 'fixed',
    curve: 'linear',
    inputMin: 0,
    inputMax: 1,
    outputMin: 1.0,
    outputMax: 1.0,
    enabled: true,
  },
  guidanceScale: {
    id: 'guidanceScale',
    name: 'Guidance Scale',
    source: 'spectral_centroid',
    curve: 'linear',
    inputMin: 0,
    inputMax: 1,
    outputMin: 0.5,
    outputMax: 3.0,
    enabled: true,
  },
};

const DEFAULT_TRIGGERS: TriggerConfig = {
  onBeat: {
    seedJump: false,  // Disabled for smooth morphing (Reactive preset)
    strengthBoost: 0.15,
    forceKeyframe: false,
  },
  onOnset: {
    seedVariation: 100,
    forceKeyframe: false,
  },
  chromaThreshold: 0.4,
};

const DEFAULT_CROSSFEED: CrossfeedConfig = {
  enabled: true,
  power: 0.5,
  range: 0.6,
  decay: 0.3,
};

const AUDIO_SOURCES: { value: AudioSource; label: string }[] = [
  { value: 'bass', label: 'Bass' },
  { value: 'mid', label: 'Mid' },
  { value: 'treble', label: 'Treble' },
  { value: 'rms', label: 'RMS (Volume)' },
  { value: 'peak', label: 'Peak' },
  { value: 'spectral_centroid', label: 'Spectral Centroid' },
  { value: 'bass_mid', label: 'Bass + Mid' },
  { value: 'bpm', label: 'BPM' },
  { value: 'onset_strength', label: 'Onset Strength' },
  { value: 'fixed', label: 'Fixed Value' },
];

const CURVE_TYPES: { value: CurveType; label: string }[] = [
  { value: 'linear', label: 'Linear' },
  { value: 'ease_in', label: 'Ease In' },
  { value: 'ease_out', label: 'Ease Out' },
  { value: 'ease_in_out', label: 'Ease In/Out' },
  { value: 'exponential', label: 'Exponential' },
];

// Preset configurations
const PRESETS: Record<string, MappingConfig> = {
  // Default balanced preset
  reactive: {
    presetName: 'reactive',
    mappings: {
      ...DEFAULT_MAPPINGS,
      transformStrength: {
        ...DEFAULT_MAPPINGS.transformStrength,
        source: 'rms',
        outputMin: 0.20,
        outputMax: 0.45,
      },
    },
    triggers: {
      ...DEFAULT_TRIGGERS,
      onBeat: {
        ...DEFAULT_TRIGGERS.onBeat,
        strengthBoost: 0.18,
      },
    },
    crossfeed: { ...DEFAULT_CROSSFEED },
  },
  dancer: {
    presetName: 'dancer',
    mappings: {
      ...DEFAULT_MAPPINGS,
      transformStrength: {
        ...DEFAULT_MAPPINGS.transformStrength,
        source: 'bass',
        curve: 'ease_in_out',
        outputMin: 0.25,
        outputMax: 0.631,
      },
      keyframeStrength: {
        ...DEFAULT_MAPPINGS.keyframeStrength,
        source: 'fixed',
        curve: 'linear',
        outputMin: 0.8,
        outputMax: 0.8,
        enabled: false,
      },
      animationSpeed: {
        ...DEFAULT_MAPPINGS.animationSpeed,
        source: 'rms',
        curve: 'linear',
        outputMin: 0.45,
        outputMax: 0.6,
      },
    },
    triggers: { ...DEFAULT_TRIGGERS },
    crossfeed: { ...DEFAULT_CROSSFEED },
  },
  vj_intense: {
    presetName: 'vj_intense',
    mappings: {
      ...DEFAULT_MAPPINGS,
      transformStrength: { ...DEFAULT_MAPPINGS.transformStrength, source: 'bass', curve: 'exponential', outputMin: 0.5, outputMax: 0.95 },
      keyframeStrength: { ...DEFAULT_MAPPINGS.keyframeStrength, source: 'bass_mid', outputMin: 0.5, outputMax: 0.8 },
    },
    triggers: { onBeat: { seedJump: true, strengthBoost: 0.35, forceKeyframe: true }, onOnset: { seedVariation: 300, forceKeyframe: true }, chromaThreshold: 0.2 },
    crossfeed: { enabled: true, power: 0.2, range: 0.4, decay: 0.6 },
  },
  dreamscape: {
    presetName: 'dreamscape',
    mappings: {
      ...DEFAULT_MAPPINGS,
      transformStrength: { ...DEFAULT_MAPPINGS.transformStrength, source: 'mid', curve: 'ease_in_out', outputMin: 0.2, outputMax: 0.5 },
      poseIntensity: { ...DEFAULT_MAPPINGS.poseIntensity, source: 'treble', outputMin: 0.3, outputMax: 0.7 },
      animationSpeed: { ...DEFAULT_MAPPINGS.animationSpeed, source: 'rms', outputMin: 0.5, outputMax: 1.2 },
    },
    triggers: { onBeat: { seedJump: false, strengthBoost: 0.1, forceKeyframe: false }, onOnset: { seedVariation: 50, forceKeyframe: false }, chromaThreshold: 0.5 },
    crossfeed: { enabled: true, power: 0.7, range: 0.7, decay: 0.2 },
  },
  color_organ: {
    presetName: 'color_organ',
    mappings: {
      ...DEFAULT_MAPPINGS,
      transformStrength: { ...DEFAULT_MAPPINGS.transformStrength, source: 'rms', outputMin: 0.3, outputMax: 0.7 },
      poseIntensity: { ...DEFAULT_MAPPINGS.poseIntensity, enabled: false },
      guidanceScale: { ...DEFAULT_MAPPINGS.guidanceScale, enabled: true, source: 'spectral_centroid', outputMin: 0.8, outputMax: 2.5 },
    },
    triggers: { onBeat: { seedJump: false, strengthBoost: 0.1, forceKeyframe: false }, onOnset: { seedVariation: 50, forceKeyframe: false }, chromaThreshold: 0.15 },
    crossfeed: { enabled: true, power: 0.4, range: 0.5, decay: 0.4 },
  },
};

/**
 * Get a preset configuration by name.
 * Returns a deep clone to prevent mutation.
 */
export function getMappingPresetConfig(presetName: string): MappingConfig | null {
  const preset = PRESETS[presetName];
  if (!preset) return null;
  return JSON.parse(JSON.stringify(preset));
}

interface MappingPanelState {
  config: MappingConfig;
  expandedMappings: Set<string>;
}

interface MappingPanelCallbacks {
  onConfigChange: (config: MappingConfig) => void;
}

export class MappingPanel extends Component<MappingPanelState> {
  private callbacks: MappingPanelCallbacks;
  private metrics: AudioMetrics | null = null;

  constructor(container: HTMLElement, callbacks: MappingPanelCallbacks, initialConfig?: MappingConfig) {
    super(container, {
      config: initialConfig || getDefaultMappingConfig(),
      expandedMappings: new Set<string>(),
    });
    this.callbacks = callbacks;
  }

  protected render(): void {
    const { config } = this.state;
    const metrics = this.metrics;

    this.el.className = 'mapping-panel';
    this.el.innerHTML = html`
      <!-- Audio Inputs Section -->
      <div class="mapping-section">
        <h3 class="mapping-section-title">Audio Inputs</h3>
        <div class="audio-meters">
          ${this.renderMeter('Bass', metrics?.bass ?? 0, '#ef4444')}
          ${this.renderMeter('Mid', metrics?.mid ?? 0, '#f59e0b')}
          ${this.renderMeter('Treble', metrics?.treble ?? 0, '#22c55e')}
          ${this.renderMeter('RMS', metrics?.rms ?? 0, '#3b82f6')}
          ${this.renderMeter('Centroid', metrics?.spectralCentroid ?? 0.5, '#8b5cf6')}
        </div>
        <div class="audio-indicators">
          <span class="bpm-display">${metrics?.bpm?.toFixed(0) ?? '--'} BPM</span>
        </div>
      </div>

      <!-- Parameter Mappings Section -->
      <div class="mapping-section">
        <h3 class="mapping-section-title">Parameter Mappings</h3>
        <div class="parameter-mappings">
          ${Object.values(config.mappings).map(m => this.renderMapping(m))}
        </div>
      </div>

      <!-- Triggers Section -->
      <div class="mapping-section">
        <h3 class="mapping-section-title">Triggers</h3>
        ${this.renderTriggers(config.triggers)}
      </div>

      <!-- Temporal Coherence Section -->
      <div class="mapping-section">
        <h3 class="mapping-section-title">Frame Blending</h3>
        <p class="mapping-section-hint">Controls how frames blend together. Lower values = more variation, higher = more stability.</p>
        ${this.renderCrossfeed(config.crossfeed)}
      </div>
    `;
  }

  private renderMeter(label: string, value: number, color: string): string {
    return html`
      <div class="audio-meter">
        <span class="audio-meter-label">${label}</span>
        <div class="audio-meter-bar">
          <div class="audio-meter-fill" style="width: ${Math.min(100, value * 100)}%; background-color: ${color}"></div>
        </div>
        <span class="audio-meter-value">${value.toFixed(3)}</span>
      </div>
    `;
  }

  private renderMapping(mapping: ParameterMapping): string {
    const metrics = this.metrics;
    const inputValue = this.getAudioValue(metrics, mapping.source);
    const outputValue = this.mapValue(metrics, mapping);
    const isExpanded = this.state.expandedMappings.has(mapping.id);
    const sourceName = AUDIO_SOURCES.find(s => s.value === mapping.source)?.label ?? mapping.source;

    return html`
      <div class="parameter-mapping ${mapping.enabled ? '' : 'disabled'}">
        <div class="parameter-mapping-header" data-action="toggle-mapping" data-id="${mapping.id}">
          <label class="parameter-mapping-checkbox">
            <input type="checkbox" ${mapping.enabled ? 'checked' : ''} data-action="toggle-enabled" data-id="${mapping.id}">
          </label>
          <span class="parameter-mapping-name">${mapping.name}</span>
          <div class="parameter-mapping-flow">
            <span class="parameter-mapping-source">${sourceName}</span>
            <span class="parameter-mapping-input-value">(${inputValue.toFixed(3)})</span>
            <span class="parameter-mapping-arrow">→</span>
          </div>
          <div class="parameter-mapping-bar">
            <div class="parameter-mapping-bar-fill" style="width: ${Math.max(0, Math.min(100, ((outputValue - mapping.outputMin) / (mapping.outputMax - mapping.outputMin)) * 100))}%"></div>
            <span class="parameter-mapping-bar-value">${outputValue.toFixed(3)}</span>
          </div>
          <span class="parameter-mapping-expand">${isExpanded ? '▼' : '▶'}</span>
        </div>

        ${isExpanded ? html`
          <div class="parameter-mapping-controls">
            <div class="mapping-row">
              <label id="source-label-${mapping.id}">Source:</label>
              <select data-action="mapping-source" data-id="${mapping.id}" aria-labelledby="source-label-${mapping.id}">
                ${AUDIO_SOURCES.map(s => html`<option value="${s.value}" ${mapping.source === s.value ? 'selected' : ''}>${s.label}</option>`)}
              </select>
            </div>
            <div class="mapping-row">
              <label id="curve-label-${mapping.id}">Curve:</label>
              <select data-action="mapping-curve" data-id="${mapping.id}" aria-labelledby="curve-label-${mapping.id}">
                ${CURVE_TYPES.map(c => html`<option value="${c.value}" ${mapping.curve === c.value ? 'selected' : ''}>${c.label}</option>`)}
              </select>
            </div>
            <div class="mapping-row">
              <label>Output Range:</label>
              <input type="number" step="0.001" value="${mapping.outputMin}" data-action="mapping-output-min" data-id="${mapping.id}" class="mapping-number-input" aria-label="Output minimum">
              <span>to</span>
              <input type="number" step="0.001" value="${mapping.outputMax}" data-action="mapping-output-max" data-id="${mapping.id}" class="mapping-number-input" aria-label="Output maximum">
            </div>
            ${mapping.source !== 'fixed' ? html`
              <div class="mapping-row">
                <label>Input Range:</label>
                <input type="number" step="0.001" value="${mapping.inputMin}" data-action="mapping-input-min" data-id="${mapping.id}" class="mapping-number-input" aria-label="Input minimum">
                <span>to</span>
                <input type="number" step="0.001" value="${mapping.inputMax}" data-action="mapping-input-max" data-id="${mapping.id}" class="mapping-number-input" aria-label="Input maximum">
              </div>
            ` : ''}
          </div>
        ` : ''}
      </div>
    `;
  }

  private renderTriggers(triggers: TriggerConfig): string {
    return html`
      <div class="trigger-group">
        <h4>On Beat</h4>
        <label class="trigger-checkbox">
          <input type="checkbox" ${triggers.onBeat.seedJump ? 'checked' : ''} data-action="trigger-beat-seed">
          Seed Jump (randomize on beat)
        </label>
        <label class="trigger-checkbox">
          <input type="checkbox" ${triggers.onBeat.forceKeyframe ? 'checked' : ''} data-action="trigger-beat-keyframe">
          Force Keyframe
        </label>
        <div class="trigger-slider">
          <label id="strength-boost-label">Strength Boost: ${triggers.onBeat.strengthBoost.toFixed(3)}</label>
          <input type="range" min="0" max="50" value="${triggers.onBeat.strengthBoost * 100}" data-action="trigger-beat-boost" aria-labelledby="strength-boost-label">
        </div>
      </div>

      <div class="trigger-group">
        <h4>On Onset (Transients)</h4>
        <label class="trigger-checkbox">
          <input type="checkbox" ${triggers.onOnset.forceKeyframe ? 'checked' : ''} data-action="trigger-onset-keyframe">
          Force Keyframe
        </label>
        <div class="trigger-slider">
          <label id="seed-variation-label">Seed Variation: ${triggers.onOnset.seedVariation}</label>
          <input type="range" min="0" max="500" value="${triggers.onOnset.seedVariation}" data-action="trigger-onset-variation" aria-labelledby="seed-variation-label">
        </div>
      </div>

      <div class="trigger-group">
        <h4>Color (Chroma)</h4>
        <div class="trigger-slider">
          <label id="color-sensitivity-label">
            Color Sensitivity: ${((1 - triggers.chromaThreshold) * 100).toFixed(0)}%
            <span class="trigger-hint">
              ${triggers.chromaThreshold <= 0.2 ? ' (very sensitive)' :
                triggers.chromaThreshold <= 0.4 ? ' (sensitive)' :
                triggers.chromaThreshold <= 0.6 ? ' (moderate)' : ' (dominant only)'}
            </span>
          </label>
          <input type="range" min="10" max="80" value="${triggers.chromaThreshold * 100}" data-action="trigger-chroma" aria-labelledby="color-sensitivity-label">
        </div>
      </div>
    `;
  }

  private renderCrossfeed(crossfeed: CrossfeedConfig): string {
    return html`
      <label class="trigger-checkbox">
        <input type="checkbox" ${crossfeed.enabled ? 'checked' : ''} data-action="crossfeed-enabled">
        Enable Latent Blending
      </label>

      ${crossfeed.enabled ? html`
        <div class="trigger-slider">
          <label id="stability-label">
            Stability: ${(crossfeed.power * 100).toFixed(0)}%
            <span class="trigger-hint">
              ${crossfeed.power <= 0.3 ? ' (chaotic)' :
                crossfeed.power <= 0.5 ? ' (dynamic)' :
                crossfeed.power <= 0.7 ? ' (balanced)' : ' (locked)'}
            </span>
          </label>
          <input type="range" min="0" max="100" value="${crossfeed.power * 100}" data-action="crossfeed-power" aria-labelledby="stability-label">
        </div>
        <div class="trigger-slider">
          <label id="blend-range-label">
            Blend Range: ${(crossfeed.range * 100).toFixed(0)}%
            <span class="trigger-hint">
              ${crossfeed.range <= 0.4 ? ' (local)' :
                crossfeed.range <= 0.7 ? ' (moderate)' : ' (global)'}
            </span>
          </label>
          <input type="range" min="0" max="100" value="${crossfeed.range * 100}" data-action="crossfeed-range" aria-labelledby="blend-range-label">
        </div>
        <div class="trigger-slider">
          <label id="adaptation-speed-label">
            Adaptation Speed: ${(crossfeed.decay * 100).toFixed(0)}%
            <span class="trigger-hint">
              ${crossfeed.decay <= 0.2 ? ' (slow)' :
                crossfeed.decay <= 0.5 ? ' (medium)' : ' (fast)'}
            </span>
          </label>
          <input type="range" min="0" max="100" value="${crossfeed.decay * 100}" data-action="crossfeed-decay" aria-labelledby="adaptation-speed-label">
        </div>
      ` : ''}
    `;
  }

  protected actions = {
    'toggle-mapping': (e: Event, target: HTMLElement) => {
      e.preventDefault();
      e.stopPropagation();
      const id = target.dataset.id!;
      const expanded = this.state.expandedMappings;
      if (expanded.has(id)) {
        expanded.delete(id);
      } else {
        expanded.add(id);
      }
      this.forceRender();
    },
    'toggle-enabled': (e: Event, target: HTMLElement) => {
      e.stopPropagation();
      const id = target.dataset.id!;
      const mapping = this.state.config.mappings[id];
      if (mapping) {
        mapping.enabled = (target as HTMLInputElement).checked;
        this.emitChange();
      }
    },
    'mapping-source': (e: Event, target: HTMLElement) => {
      const id = target.dataset.id!;
      this.state.config.mappings[id].source = (target as HTMLSelectElement).value as AudioSource;
      this.emitChange();
    },
    'mapping-curve': (e: Event, target: HTMLElement) => {
      const id = target.dataset.id!;
      this.state.config.mappings[id].curve = (target as HTMLSelectElement).value as CurveType;
      this.emitChange();
    },
    'mapping-output-min': (e: Event, target: HTMLElement) => {
      const id = target.dataset.id!;
      this.state.config.mappings[id].outputMin = parseFloat((target as HTMLInputElement).value) || 0;
      this.emitChange();
    },
    'mapping-output-max': (e: Event, target: HTMLElement) => {
      const id = target.dataset.id!;
      this.state.config.mappings[id].outputMax = parseFloat((target as HTMLInputElement).value) || 1;
      this.emitChange();
    },
    'mapping-input-min': (e: Event, target: HTMLElement) => {
      const id = target.dataset.id!;
      this.state.config.mappings[id].inputMin = parseFloat((target as HTMLInputElement).value) || 0;
      this.emitChange();
    },
    'mapping-input-max': (e: Event, target: HTMLElement) => {
      const id = target.dataset.id!;
      this.state.config.mappings[id].inputMax = parseFloat((target as HTMLInputElement).value) || 1;
      this.emitChange();
    },
    'trigger-beat-seed': (e: Event) => {
      this.state.config.triggers.onBeat.seedJump = (e.target as HTMLInputElement).checked;
      this.emitChange();
    },
    'trigger-beat-keyframe': (e: Event) => {
      this.state.config.triggers.onBeat.forceKeyframe = (e.target as HTMLInputElement).checked;
      this.emitChange();
    },
    'trigger-beat-boost': (e: Event) => {
      this.state.config.triggers.onBeat.strengthBoost = parseInt((e.target as HTMLInputElement).value) / 100;
      this.emitChange();
    },
    'trigger-onset-keyframe': (e: Event) => {
      this.state.config.triggers.onOnset.forceKeyframe = (e.target as HTMLInputElement).checked;
      this.emitChange();
    },
    'trigger-onset-variation': (e: Event) => {
      this.state.config.triggers.onOnset.seedVariation = parseInt((e.target as HTMLInputElement).value);
      this.emitChange();
    },
    'trigger-chroma': (e: Event) => {
      this.state.config.triggers.chromaThreshold = parseInt((e.target as HTMLInputElement).value) / 100;
      this.emitChange();
    },
    'crossfeed-enabled': (e: Event) => {
      this.state.config.crossfeed.enabled = (e.target as HTMLInputElement).checked;
      this.emitChange();
    },
    'crossfeed-power': (e: Event) => {
      this.state.config.crossfeed.power = parseInt((e.target as HTMLInputElement).value) / 100;
      this.emitChange();
    },
    'crossfeed-range': (e: Event) => {
      this.state.config.crossfeed.range = parseInt((e.target as HTMLInputElement).value) / 100;
      this.emitChange();
    },
    'crossfeed-decay': (e: Event) => {
      this.state.config.crossfeed.decay = parseInt((e.target as HTMLInputElement).value) / 100;
      this.emitChange();
    },
  };

  private emitChange(): void {
    this.callbacks.onConfigChange(this.state.config);
  }

  private getAudioValue(metrics: AudioMetrics | null, source: AudioSource): number {
    if (!metrics) return 0;
    switch (source) {
      case 'bass': return metrics.bass;
      case 'mid': return metrics.mid;
      case 'treble': return metrics.treble;
      case 'rms': return metrics.rms;
      case 'peak': return metrics.peak;
      case 'spectral_centroid': return metrics.spectralCentroid ?? 0.5;
      case 'bass_mid': return (metrics.bass + metrics.mid) / 2;
      case 'bpm': return Math.min(1, metrics.bpm / 200);
      case 'onset_strength': return metrics.onset?.strength ?? 0;
      case 'fixed': return 1;
      default: return 0;
    }
  }

  private applyCurve(value: number, curve: CurveType): number {
    const t = Math.max(0, Math.min(1, value));
    switch (curve) {
      case 'linear': return t;
      case 'ease_in': return t * t;
      case 'ease_out': return 1 - (1 - t) * (1 - t);
      case 'ease_in_out': return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      case 'exponential': return t === 0 ? 0 : Math.pow(2, 10 * t - 10);
      default: return t;
    }
  }

  private mapValue(metrics: AudioMetrics | null, mapping: ParameterMapping): number {
    if (!mapping.enabled) return mapping.outputMin;

    const rawValue = this.getAudioValue(metrics, mapping.source);
    const normalizedInput = (rawValue - mapping.inputMin) / (mapping.inputMax - mapping.inputMin);
    const curved = this.applyCurve(normalizedInput, mapping.curve);
    return mapping.outputMin + curved * (mapping.outputMax - mapping.outputMin);
  }

  // === Public methods ===

  setMetrics(metrics: AudioMetrics | null): void {
    this.metrics = metrics;
    this.updateMetersDOM(metrics);
  }

  /**
   * Update meter elements directly via DOM manipulation (avoids full re-render)
   */
  private updateMetersDOM(metrics: AudioMetrics | null): void {
    // Early return if component is not connected to DOM (e.g., parent re-rendered)
    if (!this.el.isConnected) {
      return;
    }

    // Update meter fills and values
    const meters = this.el.querySelectorAll('.audio-meter');
    const values = [
      metrics?.bass ?? 0,
      metrics?.mid ?? 0,
      metrics?.treble ?? 0,
      metrics?.rms ?? 0,
      metrics?.spectralCentroid ?? 0.5,
    ];

    meters.forEach((meter, i) => {
      const fill = meter.querySelector('.audio-meter-fill') as HTMLElement;
      const valueEl = meter.querySelector('.audio-meter-value');
      if (fill) fill.style.width = `${Math.min(100, values[i] * 100)}%`;
      if (valueEl) valueEl.textContent = values[i].toFixed(3);
    });

    // Update BPM display
    const bpmDisplay = this.el.querySelector('.bpm-display');
    if (bpmDisplay) {
      bpmDisplay.textContent = `${metrics?.bpm?.toFixed(0) ?? '--'} BPM`;
    }

    // Update parameter mapping values (input/output displays)
    const mappingFlows = this.el.querySelectorAll('.parameter-mapping');
    const { config } = this.state;
    const mappingsList = Object.values(config.mappings);

    mappingFlows.forEach((mappingEl, i) => {
      if (i >= mappingsList.length) return;
      const mapping = mappingsList[i];
      const inputValue = this.getAudioValue(metrics, mapping.source);
      const outputValue = this.mapValue(metrics, mapping);

      const inputEl = mappingEl.querySelector('.parameter-mapping-input-value');
      const outputEl = mappingEl.querySelector('.parameter-mapping-bar-value');
      const barFill = mappingEl.querySelector('.parameter-mapping-bar-fill') as HTMLElement;

      if (inputEl) inputEl.textContent = `(${inputValue.toFixed(3)})`;
      if (outputEl) outputEl.textContent = outputValue.toFixed(3);
      if (barFill) {
        const mapping = mappingsList[i];
        const percent = Math.max(0, Math.min(100, ((outputValue - mapping.outputMin) / (mapping.outputMax - mapping.outputMin)) * 100));
        barFill.style.width = `${percent}%`;
      }
    });
  }

  getConfig(): MappingConfig {
    return this.state.config;
  }

  setConfig(config: MappingConfig): void {
    this.state.config = config;
    this.forceRender();
  }
}

// Export defaults for initialization
export function getDefaultMappingConfig(): MappingConfig {
  return JSON.parse(JSON.stringify(PRESETS.reactive));
}
