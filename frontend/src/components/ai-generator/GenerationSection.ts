/**
 * GenerationSection - Generation mode, strength, FPS, temporal settings, and ControlNet subsection.
 */

import { html, raw } from '../../core/html';
import { GenerationMode } from '../../core/types';
import { GenerationSectionProps, shouldShowTemporalCoherence, shouldShowControlNet } from './types';
import { renderControlNetContent, getControlNetSectionActions } from './ControlNetSection';

const GENERATION_MODES: { value: GenerationMode; label: string; description: string; requiresBackend?: string }[] = [
  {
    value: 'feedback',
    label: 'Live Feedback',
    description: 'Real-time: each frame evolves from the previous (default)',
  },
  {
    value: 'keyframe_rife',
    label: 'Pose Animation',
    description: 'Real-time with pose control + interpolation (StreamDiffusion only)',
    requiresBackend: 'stream_diffusion',
  },
];

/**
 * Check if a generation mode is available for the current backend.
 */
function isModeAvailable(mode: typeof GENERATION_MODES[0], currentBackend: string | undefined): boolean {
  if (!mode.requiresBackend) return true;
  return currentBackend === mode.requiresBackend;
}

/**
 * Render the generation settings section content.
 */
export function renderGenerationSection(props: GenerationSectionProps): string {
  const { config, state, capabilities } = props;

  // Get current backend from server config or fall back to detecting from capabilities
  const currentBackend = state.serverConfig?.current_backend;
  const showControlNet = shouldShowControlNet(capabilities);

  return html`
    <div class="control-section">
      <label class="control-label" for="generation-mode-select">Generation Mode</label>
      <select id="generation-mode-select" class="mode-select" data-action="generation-mode">
        ${GENERATION_MODES.map((mode) => {
          const isAvailable = isModeAvailable(mode, currentBackend);
          const isActive = config.generationMode === mode.value;
          const label = isAvailable
            ? mode.label
            : `${mode.label} (StreamDiffusion only)`;

          return html`
            <option
              value="${mode.value}"
              ${isActive ? 'selected' : ''}
              ${!isAvailable ? 'disabled' : ''}
            >${label}</option>
          `;
        })}
      </select>
      ${currentBackend && currentBackend !== 'stream_diffusion' && config.generationMode === 'feedback'
        ? html`<p class="control-description" style="color: var(--text-muted);">Pose Animation requires StreamDiffusion backend</p>`
        : ''}
    </div>

    <div class="control-section">
      <label class="control-label" id="transform-strength-label">Transform Strength: ${Math.round(config.img2imgStrength * 100)}%</label>
      <input
        type="range"
        min="0"
        max="100"
        value="${config.img2imgStrength * 100}"
        data-action="img2img-strength"
        class="slider"
        aria-labelledby="transform-strength-label"
      />
      <p class="control-description">Higher = more transformation per frame</p>
    </div>

    <div class="control-section">
      <label class="control-label" id="target-fps-label">Target FPS: ${config.targetFps}</label>
      <input
        type="range"
        min="5"
        max="60"
        value="${config.targetFps}"
        data-action="target-fps"
        class="slider"
        aria-labelledby="target-fps-label"
      />
    </div>

    ${config.generationMode === 'feedback' && shouldShowTemporalCoherence(capabilities)
      ? html`
          <div class="control-section">
            <label class="control-label">
              <input
                type="checkbox"
                ${config.temporalCoherence === 'blending' ? 'checked' : ''}
                data-action="temporal-coherence"
              />
              Smooth Transitions (Latent Blending)
            </label>
            <p class="control-description">Blends frames together for smoother visuals</p>
          </div>
        `
      : ''}

    ${config.generationMode === 'feedback'
      ? html`
          <div class="control-section">
            <label class="control-label">
              <input
                type="checkbox"
                ${config.periodicPoseRefresh ? 'checked' : ''}
                data-action="periodic-refresh"
              />
              Beat Refresh
            </label>
            <p class="control-description">Generate fresh image every 8 beats</p>
          </div>
        `
      : ''}

    ${config.generationMode === 'keyframe_rife'
      ? html`
          <div class="control-section">
            <label class="control-label" id="keyframe-interval-label">Keyframe Interval: every ${config.keyframeInterval} frames</label>
            <input
              type="range"
              min="2"
              max="8"
              value="${config.keyframeInterval}"
              data-action="keyframe-interval"
              class="slider"
              aria-labelledby="keyframe-interval-label"
            />
          </div>
          <div class="control-section">
            <label class="control-label" id="keyframe-strength-label">Keyframe Strength: ${Math.round(config.keyframeStrength * 100)}%</label>
            <input
              type="range"
              min="10"
              max="100"
              value="${config.keyframeStrength * 100}"
              data-action="keyframe-strength"
              class="slider"
              aria-labelledby="keyframe-strength-label"
            />
          </div>
        `
      : ''}

    ${showControlNet
      ? html`
          <div class="controlnet-subsection">
            <div class="controlnet-subsection-header">
              <span class="controlnet-subsection-title">Pose & ControlNet</span>
              ${config.useControlNet ? html`<span class="controlnet-badge">ON</span>` : ''}
            </div>
            <div class="controlnet-subsection-content">
              ${raw(renderControlNetContent(props))}
            </div>
          </div>
        `
      : ''}
  `;
}

/**
 * Get the actions map for GenerationSection.
 * Note: ControlNet actions are handled separately in the parent component.
 */
export function getGenerationSectionActions(
  callbacks: GenerationSectionProps['callbacks'],
  updateSliderLabel: (input: HTMLInputElement, text: string) => void,
  forceRender: () => void
) {
  return {
    'generation-mode': (e: Event) => {
      callbacks.onConfigChange({
        generationMode: (e.target as HTMLSelectElement).value as GenerationMode,
      });
      forceRender();
    },
    'img2img-strength': (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value);
      callbacks.onConfigChange({ img2imgStrength: value / 100 });
      updateSliderLabel(
        e.target as HTMLInputElement,
        `Transform Strength: ${Math.round(value)}%`
      );
    },
    'target-fps': (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value);
      callbacks.onConfigChange({ targetFps: value });
      updateSliderLabel(e.target as HTMLInputElement, `Target FPS: ${value}`);
    },
    'temporal-coherence': (e: Event) => {
      callbacks.onConfigChange({
        temporalCoherence: (e.target as HTMLInputElement).checked ? 'blending' : 'none',
      });
    },
    'periodic-refresh': (e: Event) => {
      callbacks.onConfigChange({
        periodicPoseRefresh: (e.target as HTMLInputElement).checked,
      });
    },
    'keyframe-interval': (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value);
      callbacks.onConfigChange({ keyframeInterval: value });
      updateSliderLabel(
        e.target as HTMLInputElement,
        `Keyframe Interval: every ${value} frames`
      );
    },
    'keyframe-strength': (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value);
      callbacks.onConfigChange({ keyframeStrength: value / 100 });
      updateSliderLabel(
        e.target as HTMLInputElement,
        `Keyframe Strength: ${Math.round(value)}%`
      );
    },
  };
}
