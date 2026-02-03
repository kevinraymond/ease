/**
 * AdvancedSection - Advanced settings including acceleration, resolution, base image, LoRA.
 */

import { html } from '../../core/html';
import { AccelerationMethod, LoraConfig } from '../../core/types';
import { AdvancedSectionProps, shouldShowAcceleration, shouldShowLora } from './types';

const ACCELERATION_OPTIONS: { value: AccelerationMethod; label: string; description: string }[] = [
  { value: 'lcm', label: 'LCM', description: 'LCM-LoRA (4 steps, proven stable)' },
  { value: 'hyper-sd', label: 'Hyper-SD', description: 'ByteDance Hyper-SD (1-8 steps, state-of-art)' },
  { value: 'none', label: 'None', description: 'Standard scheduler (slower, ~20 steps)' },
];

const HYPER_SD_STEPS: { value: 1 | 2 | 4 | 8; label: string }[] = [
  { value: 1, label: '1 step' },
  { value: 2, label: '2 steps' },
  { value: 4, label: '4 steps' },
  { value: 8, label: '8 steps' },
];

/**
 * Render the advanced section content.
 */
export function renderAdvancedSection(props: AdvancedSectionProps): string {
  const { config, capabilities, newLoraPath, newLoraWeight } = props;

  return html`
    ${shouldShowAcceleration(capabilities)
      ? html`
          <div class="control-section">
            <label class="control-label">Acceleration Method</label>
            <div class="acceleration-buttons">
              ${ACCELERATION_OPTIONS.map(
                (opt) => html`
                  <button
                    class="acceleration-btn ${config.acceleration === opt.value ? 'active' : ''}"
                    data-action="acceleration"
                    data-method="${opt.value}"
                    title="${opt.description}"
                  >
                    ${opt.label}
                  </button>
                `
              )}
            </div>
            <p class="control-description">
              Changing this will reinitialize the pipeline (~3-5s pause)
            </p>

            ${config.acceleration === 'hyper-sd'
              ? html`
                  <div class="sub-control">
                    <label class="control-label-small">Hyper-SD Steps</label>
                    <div class="hyper-sd-steps-buttons">
                      ${HYPER_SD_STEPS.map(
                        (opt) => html`
                          <button
                            class="hyper-sd-step-btn ${config.hyperSdSteps === opt.value ? 'active' : ''}"
                            data-action="hyper-sd-steps"
                            data-steps="${opt.value}"
                          >
                            ${opt.label}
                          </button>
                        `
                      )}
                    </div>
                  </div>
                `
              : ''}
          </div>
        `
      : ''}

    <div class="control-section">
      <label class="control-label" for="server-url-input">Server URL</label>
      <input
        id="server-url-input"
        type="text"
        class="text-input"
        value="${config.serverUrl}"
        data-action="server-url"
      />
    </div>

    <div class="control-section">
      <label class="control-label" for="model-id-input">SD 1.5 Model ID</label>
      <input
        id="model-id-input"
        type="text"
        class="text-input"
        value="${config.modelId}"
        data-action="model-id"
      />
    </div>

    <div class="control-section">
      <label class="control-label">Resolution</label>
      <div class="resolution-inputs">
        <input
          type="number"
          class="number-input"
          value="${config.width}"
          min="256"
          max="1024"
          step="64"
          data-action="width"
          aria-label="Width"
        />
        <span>x</span>
        <input
          type="number"
          class="number-input"
          value="${config.height}"
          min="256"
          max="1024"
          step="64"
          data-action="height"
          aria-label="Height"
        />
      </div>
      <div class="sub-control" style="margin-top: 8px">
        <label class="control-label-small">
          <input
            type="checkbox"
            ${config.maintainAspectRatio ? 'checked' : ''}
            data-action="maintain-aspect"
          />
          Maintain Aspect Ratio
        </label>
        <p class="control-description">
          ${config.maintainAspectRatio
            ? 'Letterbox/pillarbox to preserve proportions'
            : 'Stretch to fill screen'}
        </p>
      </div>
    </div>

    <div class="control-section">
      <label class="control-label" id="base-image-label">Base Image (img2img starting point)</label>
      <p class="control-description">
        Upload an image to use as the starting point for generation
      </p>
      <div class="base-image-controls">
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          data-action="base-image"
          class="file-input"
          aria-labelledby="base-image-label"
        />
        ${config.baseImage
          ? html`
              <button
                class="clear-btn"
                data-action="clear-base-image"
                title="Clear base image"
              >
                Clear
              </button>
            `
          : ''}
      </div>
      ${config.baseImage
        ? html`
            <div class="base-image-preview">
              <img
                src="data:image/png;base64,${config.baseImage}"
                alt="Base image preview"
                style="max-width: 128px; max-height: 128px; margin-top: 8px"
              />
            </div>
            <div class="sub-control" style="margin-top: 0.75rem">
              <label class="control-label-small">
                <input
                  type="checkbox"
                  ${config.lockToBaseImage ? 'checked' : ''}
                  data-action="lock-base-image"
                />
                Lock to Base Image
              </label>
              <p class="control-description">
                Always use base image as input (disables feedback loop)
              </p>
            </div>
          `
        : ''}
    </div>

    ${shouldShowLora(capabilities)
      ? renderLoraSection(config, newLoraPath, newLoraWeight, props.availableLoras, props.availableLorasLoading)
      : ''}
  `;
}

function renderLoraSection(
  config: AdvancedSectionProps['config'],
  newLoraPath: string,
  newLoraWeight: number,
  availableLoras: string[],
  availableLorasLoading: boolean
): string {
  // Filter out already-added LoRAs
  const addedPaths = new Set(config.loras.map((l: LoraConfig) => l.path));
  const selectableLoras = availableLoras.filter((lora) => !addedPaths.has(lora));
  const hasSelectableLoras = selectableLoras.length > 0;

  return html`
    <div class="control-section">
      <label class="control-label">Custom LoRAs</label>
      <p class="control-description">
        Add LoRA models to modify the generation style. Select from available .safetensors files
        in the ./loras/ directory.
      </p>

      ${config.loras.length > 0
        ? html`
            <div class="lora-list">
              ${config.loras.map(
                (lora: LoraConfig, index: number) => html`
                  <div class="lora-item">
                    <span class="lora-path" title="${lora.path}">
                      ${lora.path.length > 30 ? '...' + lora.path.slice(-27) : lora.path}
                    </span>
                    <div class="lora-weight-control">
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value="${lora.weight * 100}"
                        data-action="lora-weight"
                        data-index="${index}"
                        class="slider lora-weight-slider"
                        aria-label="LoRA weight for ${lora.path}"
                      />
                      <span class="lora-weight-value">${Math.round(lora.weight * 100)}%</span>
                    </div>
                    <button
                      class="remove-lora-btn"
                      data-action="remove-lora"
                      data-index="${index}"
                      title="Remove LoRA"
                    >
                      x
                    </button>
                  </div>
                `
              )}
            </div>
          `
        : ''}

      <div class="add-lora-form">
        <div class="lora-select-row">
          <select
            class="text-input lora-select"
            data-action="new-lora-path"
            ${!hasSelectableLoras || availableLorasLoading ? 'disabled' : ''}
            aria-label="Select a LoRA"
          >
            <option value="">
              ${availableLorasLoading
                ? 'Loading...'
                : hasSelectableLoras
                  ? 'Select a LoRA...'
                  : availableLoras.length === 0
                    ? 'No LoRAs found'
                    : 'All LoRAs added'}
            </option>
            ${selectableLoras.map(
              (lora) => html`
                <option value="${lora}" ${newLoraPath === lora ? 'selected' : ''}>
                  ${lora}
                </option>
              `
            )}
          </select>
          <button
            class="refresh-btn"
            data-action="refresh-loras"
            title="Refresh list"
            ${availableLorasLoading ? 'disabled' : ''}
          >
            &#x21bb;
          </button>
        </div>
        <div class="add-lora-weight">
          <label id="new-lora-weight-label">Weight: ${Math.round(newLoraWeight * 100)}%</label>
          <input
            type="range"
            min="0"
            max="100"
            value="${newLoraWeight * 100}"
            data-action="new-lora-weight"
            class="slider"
            aria-labelledby="new-lora-weight-label"
          />
        </div>
        <button
          class="add-lora-btn"
          data-action="add-lora"
          ${!newLoraPath.trim() ? 'disabled' : ''}
        >
          Add LoRA
        </button>
      </div>
    </div>
  `;
}

/**
 * Get the actions map for AdvancedSection.
 */
export function getAdvancedSectionActions(
  props: AdvancedSectionProps,
  forceRender: () => void
) {
  const { callbacks, onLoraPathChange, onLoraWeightChange, onAddLora } = props;

  return {
    acceleration: (e: Event, target: HTMLElement) => {
      callbacks.onConfigChange({
        acceleration: target.dataset.method as AccelerationMethod,
      });
      forceRender();
    },
    'hyper-sd-steps': (e: Event, target: HTMLElement) => {
      callbacks.onConfigChange({
        hyperSdSteps: Number(target.dataset.steps) as 1 | 2 | 4 | 8,
      });
    },
    'server-url': (e: Event) => {
      callbacks.onConfigChange({
        serverUrl: (e.target as HTMLInputElement).value,
      });
    },
    'model-id': (e: Event) => {
      callbacks.onConfigChange({
        modelId: (e.target as HTMLInputElement).value,
      });
    },
    width: (e: Event) => {
      callbacks.onConfigChange({
        width: Number((e.target as HTMLInputElement).value),
      });
    },
    height: (e: Event) => {
      callbacks.onConfigChange({
        height: Number((e.target as HTMLInputElement).value),
      });
    },
    'maintain-aspect': (e: Event) => {
      callbacks.onConfigChange({
        maintainAspectRatio: (e.target as HTMLInputElement).checked,
      });
      forceRender();
    },
    'base-image': (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = (reader.result as string).split(',')[1];
        callbacks.onConfigChange({ baseImage: base64 });
        forceRender();
      };
      reader.readAsDataURL(file);
    },
    'clear-base-image': () => {
      callbacks.onClearBaseImage?.();
      callbacks.onConfigChange({ baseImage: null });
      forceRender();
    },
    'lock-base-image': (e: Event) => {
      callbacks.onConfigChange({
        lockToBaseImage: (e.target as HTMLInputElement).checked,
      });
      forceRender();
    },
    'new-lora-path': (e: Event) => {
      onLoraPathChange((e.target as HTMLSelectElement).value);
    },
    'new-lora-weight': (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value) / 100;
      onLoraWeightChange(value);
      const container = (e.target as HTMLInputElement).closest('.add-lora-weight');
      const label = container?.querySelector('label');
      if (label) label.textContent = `Weight: ${Math.round(value * 100)}%`;
    },
    'add-lora': () => {
      onAddLora();
    },
    'lora-weight': (e: Event, target: HTMLElement) => {
      const index = Number(target.dataset.index);
      const value = Number((e.target as HTMLInputElement).value) / 100;
      const newLoras = props.config.loras.map((lora, i) =>
        i === index ? { ...lora, weight: value } : lora
      );
      callbacks.onConfigChange({ loras: newLoras });
      const container = (e.target as HTMLInputElement).closest('.lora-weight-control');
      const span = container?.querySelector('.lora-weight-value');
      if (span) span.textContent = `${Math.round(value * 100)}%`;
    },
    'remove-lora': (e: Event, target: HTMLElement) => {
      const index = Number(target.dataset.index);
      callbacks.onConfigChange({
        loras: props.config.loras.filter((_, i) => i !== index),
      });
      forceRender();
    },
  };
}
