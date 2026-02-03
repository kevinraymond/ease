/**
 * AudioEffectsSection - Audio-reactive visual effects settings.
 */

import { html } from '../../core/html';
import { AudioEffectsSectionProps } from './types';

/**
 * Render the audio effects section content.
 */
export function renderAudioEffectsSection(props: AudioEffectsSectionProps): string {
  const { config } = props;

  return html`
    <div class="sub-control">
      <label class="control-label-small">
        <input
          type="checkbox"
          ${config.enableSpectralDisplacement ? 'checked' : ''}
          data-action="spectral-displacement"
        />
        Spectral Displacement
      </label>
      <p class="control-description">Bass pushes pixels outward, treble pulls inward</p>
    </div>

    <div class="sub-control">
      <label class="control-label-small">
        <input
          type="checkbox"
          ${config.enableGlitchBlocks ? 'checked' : ''}
          data-action="glitch-blocks"
        />
        Onset Glitch
      </label>
      <p class="control-description">Random block offsets on audio transients</p>
    </div>

    <div class="sub-control">
      <label class="control-label-small">
        <input
          type="checkbox"
          ${config.enableTrebleGrain ? 'checked' : ''}
          data-action="treble-grain"
        />
        Treble Shimmer
      </label>
      <p class="control-description">High-frequency grain that tracks treble energy</p>
    </div>

    <div class="sub-control">
      <label class="control-label-small">
        <input
          type="checkbox"
          ${config.enableShaderEffects ? 'checked' : ''}
          data-action="shader-effects"
        />
        Wave Distortion
      </label>
      <p class="control-description">Wavy UV distortion on beat/bass</p>
    </div>

    <div class="sub-control">
      <label class="control-label-small">
        <input
          type="checkbox"
          ${config.enableFlash ? 'checked' : ''}
          data-action="flash"
        />
        Beat Flash
      </label>
      <p class="control-description">Brightness flash on beat</p>
    </div>

    <div class="sub-control silence-degradation-section">
      <label class="control-label-small">
        <input
          type="checkbox"
          ${config.enableSilenceDegradation ? 'checked' : ''}
          data-action="silence-degradation"
        />
        Silence Degradation
      </label>
      <p class="control-description">VHS-like degradation when audio goes silent</p>
    </div>

    ${config.enableSilenceDegradation
      ? html`
          <div class="sub-control">
            <label class="control-label-small">
              Silence Threshold: ${(config.silenceThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="1"
              max="20"
              value="${config.silenceThreshold * 100}"
              data-action="silence-threshold"
              class="slider"
            />
          </div>
          <div class="sub-control">
            <label class="control-label-small">
              Degradation Speed: ${config.degradationRate.toFixed(1)}x
            </label>
            <input
              type="range"
              min="10"
              max="200"
              value="${config.degradationRate * 100}"
              data-action="degradation-rate"
              class="slider"
            />
          </div>
          <div class="sub-control">
            <label class="control-label-small">
              Recovery Speed: ${config.recoveryRate.toFixed(1)}x
            </label>
            <input
              type="range"
              min="50"
              max="500"
              value="${config.recoveryRate * 100}"
              data-action="recovery-rate"
              class="slider"
            />
          </div>
        `
      : ''}
  `;
}

/**
 * Get badge text for collapsed state.
 */
export function getAudioEffectsBadge(
  config: AudioEffectsSectionProps['config']
): string | undefined {
  const active =
    config.enableSpectralDisplacement ||
    config.enableGlitchBlocks ||
    config.enableTrebleGrain ||
    config.enableShaderEffects ||
    config.enableFlash ||
    config.enableSilenceDegradation;
  return active ? 'ON' : undefined;
}
