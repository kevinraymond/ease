/**
 * StylesSection - Mapping presets and customize mappings.
 * Split from PromptsSection for better organization.
 */

import { html } from '../../core/html';
import { MappingPreset } from '../../core/types';
import { StylesSectionProps } from './types';
import { SavedMappingPreset } from '../../ui/data/mappingPresetStorage';

export interface StylesSectionOptions {
  isMappingExpanded: boolean;
  userPresets?: SavedMappingPreset[];
  selectedUserPresetId?: string | null;
  currentPresetName?: string | null;
  hasMappingChanges?: boolean;
  isBuiltInPreset?: boolean;
}

export const MAPPING_PRESETS: { value: MappingPreset; label: string; description: string }[] = [
  { value: 'reactive', label: 'Reactive', description: 'Balanced audio response (default)' },
  { value: 'dancer', label: 'Dancer', description: 'Makes subjects dance to beats' },
  { value: 'vj_intense', label: 'VJ Mode', description: 'High-energy, dramatic beats' },
  { value: 'dreamscape', label: 'Dream', description: 'Smooth, ethereal, no beats' },
  { value: 'color_organ', label: 'Color Organ', description: 'Bass=red, mid=yellow, treble=blue' },
];

/**
 * Get the badge text for the Styles section when collapsed.
 * Shows the current preset name.
 */
export function getStylesBadge(
  config: { mappingPreset: MappingPreset },
  currentPresetName?: string | null
): string | undefined {
  if (currentPresetName) return currentPresetName;
  const preset = MAPPING_PRESETS.find(p => p.value === config.mappingPreset);
  return preset?.label;
}

/**
 * Render the styles section content.
 */
export function renderStylesSection(props: StylesSectionProps, options?: StylesSectionOptions): string {
  const { config } = props;
  const isMappingExpanded = options?.isMappingExpanded ?? false;
  const userPresets = options?.userPresets ?? [];
  const selectedUserPresetId = options?.selectedUserPresetId ?? null;
  const currentPresetName = options?.currentPresetName ?? null;
  const hasMappingChanges = options?.hasMappingChanges ?? false;
  const isBuiltInPreset = options?.isBuiltInPreset ?? true;

  // Determine if a user preset is selected (not a built-in preset)
  const isUserPresetSelected = selectedUserPresetId !== null;
  const builtInValue = isUserPresetSelected ? '' : config.mappingPreset;

  return html`
    <div class="control-section">
      <label class="control-label">Audio Mapping Preset</label>
      <div class="mapping-preset-selects">
        <div class="preset-controls-row">
          <label class="preset-row-label" for="builtin-preset-select">Built-in:</label>
          <select id="builtin-preset-select" class="mapping-preset-select" data-action="mapping-preset">
            ${isUserPresetSelected ? html`<option value="" selected>-- Select --</option>` : ''}
            ${MAPPING_PRESETS.map(
              (preset) => html`
                <option value="${preset.value}" ${builtInValue === preset.value ? 'selected' : ''} title="${preset.description}">
                  ${preset.label}
                </option>
              `
            )}
          </select>
        </div>
        <div class="preset-controls-row">
          <label class="preset-row-label" for="user-preset-select">My Saves:</label>
          <select id="user-preset-select" class="mapping-preset-select user-mapping-preset-select" data-action="user-mapping-preset">
            <option value="" ${!isUserPresetSelected ? 'selected' : ''}>
              ${userPresets.length === 0 ? 'No saved presets' : '-- Select --'}
            </option>
            ${userPresets.map(p => html`
              <option value="${p.id}" ${selectedUserPresetId === p.id ? 'selected' : ''}>${p.name}</option>
            `)}
          </select>
        </div>
        <div class="preset-action-buttons">
          <button class="preset-action-btn" data-action="save-new-preset">Save New</button>
          <button class="preset-action-btn" data-action="copy-preset" ${!isUserPresetSelected ? 'disabled' : ''}>Copy</button>
          <button class="preset-action-btn" data-action="rename-preset" ${!isUserPresetSelected ? 'disabled' : ''}>Rename</button>
          <button class="preset-action-btn" data-action="delete-preset" ${!isUserPresetSelected ? 'disabled' : ''}>Delete</button>
        </div>
      </div>
    </div>

    <div class="collapsible-subsection ${isMappingExpanded ? 'expanded' : ''}">
      <button
        class="collapsible-subheader ${isMappingExpanded ? 'expanded' : ''}"
        data-action="toggle-mapping"
      >
        <span class="collapse-arrow ${isMappingExpanded ? 'expanded' : ''}">&#9654;</span>
        <span class="collapsible-subtitle">Customize Mappings</span>
      </button>
      ${isMappingExpanded ? html`
        <div class="collapsible-subcontent">
          <div class="mapping-edit-header">
            <span class="mapping-edit-preset-name ${hasMappingChanges ? 'modified' : ''}">
              Editing: ${currentPresetName || 'Custom'}${hasMappingChanges ? ' (modified)' : ''}
            </span>
            <div class="mapping-edit-actions">
              <button
                class="preset-action-btn"
                data-action="reset-mapping"
                ${!hasMappingChanges ? 'disabled' : ''}
              >Reset</button>
              <button
                class="preset-action-btn"
                data-action="save-mapping"
                ${isBuiltInPreset || !hasMappingChanges ? 'disabled' : ''}
                title="${isBuiltInPreset ? 'Built-in presets are read-only. Use Save New to create a custom preset.' : ''}"
              >Save</button>
            </div>
          </div>
          <div id="mapping-panel-container"></div>
        </div>
      ` : ''}
    </div>
  `;
}

export interface StylesSectionActionCallbacks {
  onConfigChange: (config: Partial<{ mappingPreset: MappingPreset }>) => void;
  onUserPresetSelect?: (presetId: string | null, preset: SavedMappingPreset | null) => void;
}

/**
 * Get the actions map for StylesSection.
 */
export function getStylesSectionActions(
  callbacks: StylesSectionActionCallbacks,
  userPresets?: SavedMappingPreset[]
) {
  return {
    'mapping-preset': (e: Event) => {
      const value = (e.target as HTMLSelectElement).value as MappingPreset;
      callbacks.onConfigChange({
        mappingPreset: value,
      });
      // Clear user preset selection when selecting built-in
      if (callbacks.onUserPresetSelect) {
        callbacks.onUserPresetSelect(null, null);
      }
    },
    'user-mapping-preset': (e: Event) => {
      const presetId = (e.target as HTMLSelectElement).value;
      if (!presetId) {
        // Deselected user preset - do nothing, keep current built-in
        if (callbacks.onUserPresetSelect) {
          callbacks.onUserPresetSelect(null, null);
        }
        return;
      }
      const preset = userPresets?.find(p => p.id === presetId);
      if (preset && callbacks.onUserPresetSelect) {
        callbacks.onUserPresetSelect(presetId, preset);
      }
    },
  };
}
