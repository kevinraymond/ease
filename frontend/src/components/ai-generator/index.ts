/**
 * AI Generator section components.
 * Re-exports all section render functions and helpers.
 */

// Types
export * from './types';

// Section render functions
export {
  renderConnectionSection,
  getConnectionSectionActions,
} from './ConnectionSection';

export {
  renderPromptsSection,
  getPromptsSectionActions,
  PROMPT_PRESETS,
} from './PromptsSection';

export {
  renderStylesSection,
  getStylesSectionActions,
  getStylesBadge,
} from './StylesSection';

export type { StylesSectionOptions, StylesSectionActionCallbacks } from './StylesSection';

export {
  renderGenerationSection,
  getGenerationSectionActions,
} from './GenerationSection';

export {
  renderControlNetSection,
  renderControlNetContent,
  getControlNetSectionActions,
  getControlNetBadge,
  shouldRenderControlNetSection,
} from './ControlNetSection';

export {
  renderAudioEffectsSection,
  getAudioEffectsBadge,
} from './AudioEffectsSection';

export {
  renderUpscalingSection,
  getUpscalingSectionActions,
  getUpscalingBadge,
} from './UpscalingSection';

export {
  renderLyricsSection,
  getLyricsSectionActions,
  getLyricsBadge,
} from './LyricsSection';

export {
  renderAdvancedSection,
  getAdvancedSectionActions,
} from './AdvancedSection';

export {
  renderSettingsSection,
  getSettingsSectionActions,
  getSettingsBadge,
} from './SettingsSection';

export type { SettingsSectionProps, SettingsSectionCallbacks } from './types';
