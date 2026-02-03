/**
 * ControlNetSection - Pose/ControlNet settings (only visible when backend supports it).
 */

import { html } from '../../core/html';
import { ControlNetSectionProps, shouldShowControlNet } from './types';

// Animation styles curated for visualization
// - Human motion: gentle (low FPS), dancing (music-reactive)
// - Visualization: flowing (serpentine), pulsing (rhythmic), ethereal (floating)
const POSE_MODES = [
  { id: 'gentle', label: 'Gentle', tooltip: 'Slow, graceful movements - ideal for low FPS generation' },
  { id: 'dancing', label: 'Dancing', tooltip: 'Dynamic dancing with bounce - great for music visualization' },
  { id: 'flowing', label: 'Flowing', tooltip: 'Smooth serpentine motions - hypnotic wave patterns' },
  { id: 'pulsing', label: 'Pulsing', tooltip: 'Rhythmic expansion/contraction - syncs well with beats' },
  { id: 'ethereal', label: 'Ethereal', tooltip: 'Slow, weightless floating - dreamy atmosphere' },
];

const POSE_FRAMINGS = [
  { id: 'full_body', label: 'Full Body', tooltip: 'Head to feet - may crop in square images' },
  { id: 'upper_body', label: 'Upper Body', tooltip: 'Head to hips - recommended for 512x512' },
  { id: 'portrait', label: 'Portrait', tooltip: 'Head and shoulders close-up' },
];

/**
 * Check if ControlNet section should be rendered.
 */
export function shouldRenderControlNetSection(capabilities: string[]): boolean {
  return shouldShowControlNet(capabilities);
}

/**
 * Render the inner ControlNet controls (shared between embedded and standalone).
 */
function renderControlNetInner(
  config: ControlNetSectionProps['config'],
  state: ControlNetSectionProps['state']
): string {
  return html`
    <label class="control-label">
      <input
        type="checkbox"
        ${config.useControlNet ? 'checked' : ''}
        data-action="use-controlnet"
        title="Enable ControlNet pose detection to maintain body position across frames"
      />
      Pose Preservation (ControlNet)
    </label>
    <p class="control-description">
      Extracts pose from each frame to maintain consistent body position
    </p>

    ${config.useControlNet
      ? html`
          <!-- ControlNet Settings -->
          <div class="sub-control">
            <label class="control-label-small" id="pose-influence-label">
              Pose Influence: ${Math.round(config.controlNetPoseWeight * 100)}%
            </label>
            <input
              type="range"
              min="5"
              max="100"
              value="${config.controlNetPoseWeight * 100}"
              data-action="controlnet-weight"
              class="slider"
              title="How strongly the pose guides generation - higher = stricter adherence"
              aria-labelledby="pose-influence-label"
            />
          </div>

          <div class="sub-control">
            <label class="control-label-small">
              <input
                type="checkbox"
                ${config.controlNetPoseLock ? 'checked' : ''}
                data-action="pose-lock"
                title="When locked, pose stays fixed. When unlocked, pose drifts naturally over time"
              />
              Lock to Initial Pose
            </label>
            <p class="control-description">
              Locked: pose stays fixed. Unlocked: pose drifts over time.
            </p>
          </div>

          <hr class="section-divider" />

          <!-- Animation Settings -->
          <div class="sub-control">
            <label class="control-label-small">
              <input
                type="checkbox"
                ${config.useProceduralPose ? 'checked' : ''}
                data-action="procedural-pose"
                title="Generate animated poses procedurally instead of extracting from images"
              />
              Procedural Animation
            </label>
            <p class="control-description">
              Generate animated poses instead of extracting from images
            </p>
          </div>

          ${config.useProceduralPose
            ? html`
                <div class="sub-control">
                  <label class="control-label-small">Animation Style</label>
                  <div class="pose-mode-buttons">
                    ${POSE_MODES.map(
                      (mode) => html`
                        <button
                          class="pose-mode-btn ${config.poseAnimationMode === mode.id ? 'active' : ''}"
                          data-action="pose-mode"
                          data-mode="${mode.id}"
                          title="${mode.tooltip}"
                        >
                          ${mode.label}
                        </button>
                      `
                    )}
                  </div>
                </div>

                <div class="sub-control">
                  <label class="control-label-small">Zoom Level</label>
                  <div class="pose-mode-buttons">
                    ${POSE_FRAMINGS.map(
                      (framing) => html`
                        <button
                          class="pose-mode-btn ${config.poseFraming === framing.id ? 'active' : ''}"
                          data-action="pose-framing"
                          data-framing="${framing.id}"
                          title="${framing.tooltip}"
                        >
                          ${framing.label}
                        </button>
                      `
                    )}
                  </div>
                </div>

                <div class="sub-control">
                  <label class="control-label-small" id="animation-speed-label">
                    Animation Speed: ${config.poseAnimationSpeed.toFixed(1)}x
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="300"
                    value="${config.poseAnimationSpeed * 100}"
                    data-action="pose-speed"
                    class="slider"
                    title="Speed multiplier for animation playback (0.1x = very slow, 3x = fast)"
                    aria-labelledby="animation-speed-label"
                  />
                </div>

                <div class="sub-control">
                  <label class="control-label-small" id="movement-intensity-label">
                    Movement Intensity: ${Math.round(config.poseAnimationIntensity * 100)}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value="${config.poseAnimationIntensity * 100}"
                    data-action="pose-intensity"
                    class="slider"
                    title="How dramatic the movements are - higher = larger motions"
                    aria-labelledby="movement-intensity-label"
                  />
                </div>
              `
            : ''}

          ${state.posePreview
            ? html`
                <div class="sub-control pose-preview-container">
                  <label class="control-label-small">
                    Pose Preview ${config.useProceduralPose ? '(Procedural)' : '(Extracted)'}
                  </label>
                  <div class="pose-preview-window">
                    <canvas id="pose-preview-canvas" width="128" height="128"></canvas>
                  </div>
                </div>
              `
            : ''}
        `
      : ''}
  `;
}

/**
 * Render the ControlNet content for embedding in GenerationSection.
 * This is the inner content without a wrapping control-section div.
 */
export function renderControlNetContent(props: ControlNetSectionProps): string {
  const { config, state } = props;
  return renderControlNetInner(config, state);
}

/**
 * Render the ControlNet section content (standalone section).
 */
export function renderControlNetSection(props: ControlNetSectionProps): string {
  const { config, state } = props;

  return html`
    <div class="control-section">
      ${renderControlNetInner(config, state)}
    </div>
  `;
}

/**
 * Get badge text for collapsed state.
 */
export function getControlNetBadge(config: ControlNetSectionProps['config']): string | undefined {
  return config.useControlNet ? 'ON' : undefined;
}

/**
 * Get the actions map for ControlNetSection.
 */
export function getControlNetSectionActions(
  callbacks: ControlNetSectionProps['callbacks'],
  updateSliderLabel: (input: HTMLInputElement, text: string) => void,
  forceRender: () => void
) {
  return {
    'use-controlnet': (e: Event) => {
      callbacks.onConfigChange({
        useControlNet: (e.target as HTMLInputElement).checked,
      });
      forceRender();
    },
    'controlnet-weight': (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value);
      callbacks.onConfigChange({ controlNetPoseWeight: value / 100 });
      updateSliderLabel(
        e.target as HTMLInputElement,
        `Pose Influence: ${Math.round(value)}%`
      );
    },
    'pose-lock': (e: Event) => {
      callbacks.onConfigChange({
        controlNetPoseLock: (e.target as HTMLInputElement).checked,
      });
    },
    'procedural-pose': (e: Event) => {
      callbacks.onConfigChange({
        useProceduralPose: (e.target as HTMLInputElement).checked,
      });
      forceRender();
    },
    'pose-mode': (e: Event, target: HTMLElement) => {
      callbacks.onConfigChange({
        poseAnimationMode: target.dataset.mode!,
      });
    },
    'pose-framing': (e: Event, target: HTMLElement) => {
      callbacks.onConfigChange({
        poseFraming: target.dataset.framing as 'full_body' | 'upper_body' | 'portrait',
      });
    },
    'pose-speed': (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value) / 100;
      callbacks.onConfigChange({ poseAnimationSpeed: value });
      updateSliderLabel(
        e.target as HTMLInputElement,
        `Animation Speed: ${value.toFixed(1)}x`
      );
    },
    'pose-intensity': (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value);
      callbacks.onConfigChange({ poseAnimationIntensity: value / 100 });
      updateSliderLabel(
        e.target as HTMLInputElement,
        `Movement Intensity: ${Math.round(value)}%`
      );
    },
  };
}
