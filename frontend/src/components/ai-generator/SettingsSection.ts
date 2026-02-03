/**
 * SettingsSection - Settings profile save/load UI.
 * Allows users to save, load, rename, delete, export, and import settings profiles.
 */

import { html } from '../../core/html';
import { SavedSettings } from '../../ui/data/settingsStorage';
import { SettingsSectionProps } from './types';

/**
 * Render the settings section content.
 * This is a stateless render function - state management is handled by the parent.
 */
export function renderSettingsSection(props: SettingsSectionProps): string {
  const {
    savedSettings,
    selectedSettingsId,
    isLoading,
    lastSessionRecord,
  } = props;

  const hasSelection = selectedSettingsId !== null;
  const isLastSessionSelected = selectedSettingsId === '__current_state__';
  const canModifySelected = hasSelection && !isLastSessionSelected;

  return html`
    <div class="settings-profiles-section">
      <div class="control-section">
        <label class="control-label" for="settings-select">Saved Profiles</label>
        <select
          id="settings-select"
          class="settings-select"
          data-action="select-settings"
          ${isLoading ? 'disabled' : ''}
        >
          <option value="">-- Select a profile --</option>
          ${lastSessionRecord ? html`
            <option value="__current_state__" ${isLastSessionSelected ? 'selected' : ''}>
              ${lastSessionRecord.name} (auto-saved)
            </option>
          ` : ''}
          ${savedSettings.length > 0 && lastSessionRecord ? html`
            <option disabled>──────────</option>
          ` : ''}
          ${savedSettings.map(s => html`
            <option value="${s.id}" ${s.id === selectedSettingsId ? 'selected' : ''}>
              ${s.name}
            </option>
          `)}
        </select>
      </div>

      <div class="settings-buttons-row">
        <button
          class="settings-btn"
          data-action="save-settings"
          title="Save current settings to selected profile"
          ${!canModifySelected ? 'disabled' : ''}
        >
          Save
        </button>
        <button
          class="settings-btn"
          data-action="save-settings-as"
          title="Save current settings as a new profile"
        >
          Save As
        </button>
        <button
          class="settings-btn"
          data-action="rename-settings"
          title="Rename the selected profile"
          ${!canModifySelected ? 'disabled' : ''}
        >
          Rename
        </button>
        <button
          class="settings-btn danger"
          data-action="delete-settings"
          title="Delete the selected profile"
          ${!canModifySelected ? 'disabled' : ''}
        >
          Delete
        </button>
      </div>

      <div class="settings-buttons-row import-export-row">
        <button
          class="settings-btn secondary"
          data-action="export-settings"
          title="Export current settings to a file"
        >
          Export
        </button>
        <label class="settings-btn secondary import-btn" title="Import settings from a file">
          Import
          <input
            type="file"
            accept=".json"
            data-action="import-settings"
            class="hidden-file-input"
          />
        </label>
      </div>
    </div>
  `;
}

/**
 * Get badge text for the settings section header.
 */
export function getSettingsBadge(
  selectedSettingsId: string | null,
  savedSettings: SavedSettings[]
): string | undefined {
  if (!selectedSettingsId) return undefined;
  const selected = savedSettings.find(s => s.id === selectedSettingsId);
  return selected ? selected.name : undefined;
}

/**
 * Get the actions map for SettingsSection.
 * The parent component should merge these into its actions.
 */
export function getSettingsSectionActions(
  callbacks: SettingsSectionProps['callbacks']
) {
  return {
    'select-settings': (e: Event) => {
      const id = (e.target as HTMLSelectElement).value;
      callbacks.onSelectSettings(id || null);
    },
    'save-settings': () => {
      callbacks.onSaveSettings();
    },
    'save-settings-as': () => {
      callbacks.onSaveSettingsAs();
    },
    'rename-settings': () => {
      callbacks.onRenameSettings();
    },
    'delete-settings': () => {
      callbacks.onDeleteSettings();
    },
    'export-settings': () => {
      callbacks.onExportSettings();
    },
    'import-settings': (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        callbacks.onImportSettings(file);
        // Reset input so same file can be selected again
        (e.target as HTMLInputElement).value = '';
      }
    },
  };
}
