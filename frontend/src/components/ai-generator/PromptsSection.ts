/**
 * PromptsSection - Base prompt and negative prompt controls only.
 */

import { html } from '../../core/html';
import { PromptsSectionProps } from './types';

// Preset prompts
export const PROMPT_PRESETS = {
  subject: 'prismatic light figure emerging from shattered pixel grid, RGB streaks forming humanoid silhouette, digital chrysalis breaking open, volumetric rays, cinematic lighting',
  pattern: 'fractured geometric matrix dissolving into aurora waves, glitch artifacts becoming organic flow, neon circuit lines melting into liquid light',
};

/**
 * Render the prompts section content.
 */
export function renderPromptsSection(props: PromptsSectionProps): string {
  const { config, state } = props;

  return html`
    <div class="control-section">
      <label class="control-label" for="base-prompt-input">Base Prompt</label>
      <div class="prompt-preset-buttons">
        <button class="prompt-preset-btn" data-action="preset-subject">Subject</button>
        <button class="prompt-preset-btn" data-action="preset-pattern">Pattern</button>
      </div>
      <textarea
        id="base-prompt-input"
        class="prompt-input"
        data-action="base-prompt"
        placeholder="Describe the visual style..."
        rows="6"
      >${config.basePrompt}</textarea>
    </div>

    <div class="control-section">
      <label class="control-label" for="negative-prompt-input">Negative Prompt</label>
      <textarea
        id="negative-prompt-input"
        class="prompt-input"
        data-action="negative-prompt"
        placeholder="Things to avoid..."
        rows="4"
      >${config.negativePrompt}</textarea>
    </div>

    <div class="control-section">
      <label class="control-label">Last Prompt Sent</label>
      <div class="prompt-input last-prompt-display" style="opacity: 0.7; cursor: default; user-select: text; min-height: 100px; overflow-y: auto;">
        ${state.lastParams?.prompt || 'No prompt sent yet'}
      </div>
    </div>
  `;
}

/**
 * Get the actions map for PromptsSection.
 */
export function getPromptsSectionActions(
  callbacks: PromptsSectionProps['callbacks']
) {
  return {
    'base-prompt': (e: Event) => {
      callbacks.onConfigChange({
        basePrompt: (e.target as HTMLTextAreaElement).value,
      });
    },
    'negative-prompt': (e: Event) => {
      callbacks.onConfigChange({
        negativePrompt: (e.target as HTMLTextAreaElement).value,
      });
    },
    'preset-subject': () => {
      callbacks.onConfigChange({
        basePrompt: PROMPT_PRESETS.subject,
      });
    },
    'preset-pattern': () => {
      callbacks.onConfigChange({
        basePrompt: PROMPT_PRESETS.pattern,
      });
    },
  };
}
