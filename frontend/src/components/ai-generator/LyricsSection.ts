/**
 * LyricsSection - Lyric detection and display settings.
 */

import { html } from '../../core/html';
import { LyricsSectionProps } from './types';

/**
 * Render the lyrics section content.
 */
export function renderLyricsSection(props: LyricsSectionProps): string {
  const { config, state, callbacks } = props;

  return html`
    <div class="control-section">
      <label class="control-label">
        <input
          type="checkbox"
          ${config.enableLyrics ? 'checked' : ''}
          data-action="enable-lyrics"
        />
        Lyric Detection
      </label>
      <p class="control-description">
        Transcribe lyrics and inject keywords into prompts
      </p>

      ${config.enableLyrics
        ? html`
            ${callbacks.onResetLyrics
              ? html`
                  <button
                    class="reset-lyrics-btn"
                    data-action="reset-lyrics"
                    title="Clear transcription buffer"
                  >
                    Reset
                  </button>
                `
              : ''}

            ${state.lyrics
              ? html`
                  <div class="lyric-display">
                    ${state.lyrics.text
                      ? html`<div class="lyric-text">${state.lyrics.text.slice(-100)}</div>`
                      : ''}
                    ${state.lyrics.keywords.length > 0
                      ? html`
                          <div class="lyric-keywords">
                            Keywords: ${state.lyrics.keywords
                              .map(([word]: [string, number]) => word)
                              .join(', ')}
                          </div>
                        `
                      : ''}
                  </div>
                `
              : ''}

            <div class="sub-control">
              <label class="control-label-small">
                <input
                  type="checkbox"
                  ${config.lyricDrivenMode ? 'checked' : ''}
                  data-action="lyric-driven"
                />
                Lyric-Driven Mode
              </label>
              <p class="control-description">Generate prompts from detected lyrics</p>
            </div>

            <div class="sub-control">
              <label class="control-label-small">
                <input
                  type="checkbox"
                  ${config.showLyricSubtitles ? 'checked' : ''}
                  data-action="show-subtitles"
                />
                Show Subtitles
              </label>
            </div>
          `
        : ''}
    </div>
  `;
}

/**
 * Get badge text for collapsed state.
 */
export function getLyricsBadge(
  config: LyricsSectionProps['config']
): string | undefined {
  return config.enableLyrics ? 'ON' : undefined;
}

/**
 * Get the actions map for LyricsSection.
 */
export function getLyricsSectionActions(
  callbacks: LyricsSectionProps['callbacks'],
  forceRender: () => void
) {
  return {
    'enable-lyrics': (e: Event) => {
      callbacks.onConfigChange({
        enableLyrics: (e.target as HTMLInputElement).checked,
      });
      forceRender();
    },
    'reset-lyrics': () => {
      callbacks.onResetLyrics?.();
    },
    'lyric-driven': (e: Event) => {
      callbacks.onConfigChange({
        lyricDrivenMode: (e.target as HTMLInputElement).checked,
      });
    },
    'show-subtitles': (e: Event) => {
      callbacks.onConfigChange({
        showLyricSubtitles: (e.target as HTMLInputElement).checked,
      });
    },
  };
}
