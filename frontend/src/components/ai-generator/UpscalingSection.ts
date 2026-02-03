/**
 * UpscalingSection - Image quality and upscaling settings.
 */

import { html } from '../../core/html';
import { UpscalingSectionProps } from './types';

/**
 * Render the upscaling/image quality section content.
 */
export function renderUpscalingSection(props: UpscalingSectionProps): string {
  const { config } = props;

  return html`
    <div class="sub-control">
      <label class="control-label-small">
        <input
          type="checkbox"
          ${config.enableBicubic ? 'checked' : ''}
          data-action="bicubic"
        />
        Bicubic Interpolation
      </label>
      <p class="control-description">Sharper edges than bilinear (~0.5ms)</p>
    </div>

    <div class="sub-control">
      <label class="control-label-small">
        <input
          type="checkbox"
          ${config.enableSharpening ? 'checked' : ''}
          data-action="sharpening"
        />
        Audio-Reactive Sharpening
      </label>
      <p class="control-description">Increases sharpness on beats/onsets</p>
    </div>

    ${config.enableSharpening
      ? html`
          <div class="sub-control">
            <label class="control-label-small">
              Sharpen Strength: ${config.sharpenStrength.toFixed(2)}
            </label>
            <input
              type="range"
              class="control-slider"
              min="0"
              max="1.5"
              step="0.05"
              value="${config.sharpenStrength}"
              data-action="sharpen-strength"
            />
          </div>
        `
      : ''}
  `;
}

/**
 * Get badge text for collapsed state.
 */
export function getUpscalingBadge(
  config: UpscalingSectionProps['config']
): string | undefined {
  return config.enableBicubic || config.enableSharpening ? 'ON' : undefined;
}

/**
 * Get the actions map for UpscalingSection.
 */
export function getUpscalingSectionActions(
  callbacks: UpscalingSectionProps['callbacks'],
  updateSliderLabel: (input: HTMLInputElement, text: string) => void,
  forceRender: () => void
) {
  return {
    bicubic: (e: Event) => {
      callbacks.onConfigChange({
        enableBicubic: (e.target as HTMLInputElement).checked,
      });
    },
    sharpening: (e: Event) => {
      callbacks.onConfigChange({
        enableSharpening: (e.target as HTMLInputElement).checked,
      });
      forceRender();
    },
    'sharpen-strength': (e: Event) => {
      const value = parseFloat((e.target as HTMLInputElement).value);
      callbacks.onConfigChange({ sharpenStrength: value });
      updateSliderLabel(
        e.target as HTMLInputElement,
        `Sharpen Strength: ${value.toFixed(2)}`
      );
    },
  };
}
