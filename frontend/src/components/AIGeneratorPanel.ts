/**
 * AIGeneratorPanel - Main control panel for AI image generation.
 * Orchestrates section components for a modular, maintainable UI.
 */

import { Component } from '../core/Component';
import { html, raw } from '../core/html';
import {
  AIGeneratorConfig,
  AIGeneratorState,
  LoraConfig,
  BackendInfo,
} from '../core/types';
import { MappingPanel, MappingConfig, getMappingPresetConfig } from './MappingPanel';
import {
  SavedMappingPreset,
  getAllMappingPresets,
  saveMappingPreset,
  deleteMappingPreset,
} from '../ui/data/mappingPresetStorage';
import {
  SavedSettings,
  CombinedSettings,
  getAllSettings,
  saveSettings,
  deleteSettings,
  renameSettings,
  saveCurrentState,
  getCurrentStateRecord,
  exportSettingsToJson,
  parseImportedJson,
  CURRENT_STATE_ID_EXPORT,
} from '../ui/data/settingsStorage';
import { StoryControlPanel } from './StoryControlPanel';
import { StoryController } from '../controllers/StoryController';
import { AudioMetrics } from '../audio/types';
import {
  getEffectiveCapabilities,
  getAvailableBackends,
  getStoredBackendId,
  setStoredBackendId,
} from '../core/backends';

// Import section render functions
import {
  renderConnectionSection,
  renderPromptsSection,
  renderStylesSection,
  renderGenerationSection,
  renderControlNetSection,
  renderAudioEffectsSection,
  renderUpscalingSection,
  renderLyricsSection,
  renderAdvancedSection,
  renderSettingsSection,
  shouldRenderControlNetSection,
  getControlNetBadge,
  getAudioEffectsBadge,
  getUpscalingBadge,
  getLyricsBadge,
  getStylesBadge,
  getSettingsBadge,
  PROMPT_PRESETS,
  ConnectionSectionProps,
  PromptsSectionProps,
  StylesSectionProps,
  StylesSectionOptions,
  GenerationSectionProps,
  ControlNetSectionProps,
  AudioEffectsSectionProps,
  UpscalingSectionProps,
  LyricsSectionProps,
  AdvancedSectionProps,
  SettingsSectionProps,
} from './ai-generator';

interface AIGeneratorPanelState {
  isConnecting: boolean;
  newLoraPath: string;
  newLoraWeight: number;
  expandedSections: Set<string>;
  selectedBackendId: string;
  hasPendingChanges: boolean;
  userMappingPresets: SavedMappingPreset[];
  selectedUserMappingPresetId: string | null;
  availableLoras: string[];
  availableLorasLoading: boolean;
  // Settings save/load state
  savedSettings: SavedSettings[];
  selectedSettingsId: string | null;
  settingsLoading: boolean;
  lastSessionRecord: SavedSettings | null;
}

interface AIGeneratorPanelCallbacks {
  onConnect: () => Promise<void>;
  onDisconnect: () => void;
  onStartGeneration: () => void;
  onStopGeneration: () => void;
  onResetFeedback: () => void;
  onClearBaseImage?: () => void;
  onClearVram?: () => Promise<void>;
  onSwitchBackend?: (backendId: string) => void;
  onConfigChange: (config: Partial<AIGeneratorConfig>) => void;
  onResetLyrics?: () => void;
  onMappingConfigChange: (config: MappingConfig) => void;
}

export class AIGeneratorPanel extends Component<AIGeneratorPanelState> {
  private callbacks: AIGeneratorPanelCallbacks;
  private aiState: AIGeneratorState;
  private aiConfig: AIGeneratorConfig;
  private mappingConfig: MappingConfig;
  private currentMetrics: AudioMetrics | null = null;
  private storyController: StoryController;
  private pendingRender = false;
  private blurListenerAttached = false;
  private pendingHasPendingChanges = false;
  private autoSaveTimer: number | null = null;
  private autoSaveDelay = 2000; // 2 seconds debounce
  private boundBeforeUnload: (() => void) | null = null;
  private skipAutoSave = false; // Skip auto-save when loading profiles

  // Child components
  private mappingPanel: MappingPanel | null = null;
  private storyPanel: StoryControlPanel | null = null;

  // Mapping preset edit state
  private originalMappingConfig: MappingConfig | null = null;
  private hasMappingChanges: boolean = false;
  private currentPresetName: string | null = 'Reactive';
  private currentPresetIsBuiltIn: boolean = true;

  constructor(
    container: HTMLElement,
    callbacks: AIGeneratorPanelCallbacks,
    aiState: AIGeneratorState,
    aiConfig: AIGeneratorConfig,
    mappingConfig: MappingConfig,
    storyController: StoryController
  ) {
    super(container, {
      isConnecting: false,
      newLoraPath: '',
      newLoraWeight: 0.8,
      expandedSections: new Set(['connection', 'prompts']),
      selectedBackendId: getStoredBackendId(),
      hasPendingChanges: false,
      userMappingPresets: [],
      selectedUserMappingPresetId: null,
      availableLoras: [],
      availableLorasLoading: false,
      savedSettings: [],
      selectedSettingsId: null,
      settingsLoading: false,
      lastSessionRecord: null,
    });
    this.callbacks = callbacks;
    this.aiState = aiState;
    this.aiConfig = aiConfig;
    this.mappingConfig = mappingConfig;
    this.storyController = storyController;

    this.loadExpandedState();
    this.loadUserMappingPresets();
    this.loadSavedSettings();

    // Fetch LoRAs if Advanced section is already expanded
    if (this.state.expandedSections.has('advanced')) {
      this.fetchAvailableLoras();
    }

    // Save settings when page is about to close
    this.boundBeforeUnload = () => {
      // Cancel any pending debounced save
      if (this.autoSaveTimer !== null) {
        window.clearTimeout(this.autoSaveTimer);
        this.autoSaveTimer = null;
      }
      // Save immediately (synchronously if possible)
      const settings = this.gatherCurrentSettings();
      // Use sendBeacon for reliable save on page unload
      // Fall back to sync localStorage as IndexedDB is async
      try {
        localStorage.setItem('ease-settings-backup', JSON.stringify(settings));
      } catch (e) {
        console.error('Failed to backup settings on unload:', e);
      }
    };
    window.addEventListener('beforeunload', this.boundBeforeUnload);
  }

  private async loadSavedSettings(): Promise<void> {
    this.state.settingsLoading = true;
    try {
      const settings = await getAllSettings();
      this.state.savedSettings = settings;

      // Check for localStorage backup (from beforeunload) and migrate to IndexedDB
      const backup = localStorage.getItem('ease-settings-backup');
      if (backup) {
        try {
          const backupSettings = JSON.parse(backup) as CombinedSettings;
          // Save to IndexedDB as current state
          await saveCurrentState(backupSettings);
          // Clear the backup
          localStorage.removeItem('ease-settings-backup');
          console.log('Migrated settings backup from localStorage to IndexedDB');
        } catch (e) {
          console.error('Failed to migrate settings backup:', e);
        }
      }

      // Load the last session record for display
      const lastSession = await getCurrentStateRecord();
      this.state.lastSessionRecord = lastSession;
    } catch (error) {
      console.error('Failed to load saved settings:', error);
    } finally {
      this.state.settingsLoading = false;
      this.forceRender();
    }
  }

  private gatherCurrentSettings(): CombinedSettings {
    return {
      version: 1,
      aiConfig: { ...this.aiConfig },
      mappingConfig: { ...this.mappingConfig },
    };
  }

  private scheduleAutoSave(): void {
    // Skip auto-save if we're loading a profile (not making user changes)
    if (this.skipAutoSave) {
      return;
    }

    if (this.autoSaveTimer !== null) {
      window.clearTimeout(this.autoSaveTimer);
    }
    this.autoSaveTimer = window.setTimeout(async () => {
      this.autoSaveTimer = null;
      const settings = this.gatherCurrentSettings();
      try {
        await saveCurrentState(settings);
        console.log('[Auto-save] Settings saved to IndexedDB');
        // Update the last session record in state
        this.state.lastSessionRecord = await getCurrentStateRecord();
      } catch (err) {
        console.error('[Auto-save] Failed:', err);
      }
    }, this.autoSaveDelay);
  }

  private async handleLoadSettings(id: string): Promise<void> {
    let settings: SavedSettings | null | undefined;

    // Check if loading the last session
    if (id === CURRENT_STATE_ID_EXPORT) {
      settings = this.state.lastSessionRecord;
    } else {
      settings = this.state.savedSettings.find(s => s.id === id);
    }

    if (!settings) return;

    // Set flag to skip auto-save (we're loading, not making changes)
    this.skipAutoSave = true;

    // Apply AI config
    if (settings.settings.aiConfig) {
      this.aiConfig = { ...this.aiConfig, ...settings.settings.aiConfig };
      this.callbacks.onConfigChange(settings.settings.aiConfig);
    }

    // Apply mapping config
    if (settings.settings.mappingConfig) {
      this.mappingConfig = settings.settings.mappingConfig;
      this.callbacks.onMappingConfigChange(settings.settings.mappingConfig);
      if (this.mappingPanel) {
        this.mappingPanel.setConfig(settings.settings.mappingConfig);
      }
    }

    // Reset the skip flag after all callbacks have been processed
    this.skipAutoSave = false;

    this.state.selectedSettingsId = id;
    this.forceRender();
  }

  private async handleSaveSettings(name?: string): Promise<void> {
    const currentName = name || (
      this.state.selectedSettingsId
        ? this.state.savedSettings.find(s => s.id === this.state.selectedSettingsId)?.name
        : null
    );
    if (!currentName) return;

    try {
      const settings = this.gatherCurrentSettings();
      const saved = await saveSettings(
        currentName,
        settings,
        this.state.selectedSettingsId || undefined
      );

      // Update local state
      if (this.state.selectedSettingsId) {
        this.state.savedSettings = this.state.savedSettings.map(s =>
          s.id === this.state.selectedSettingsId ? saved : s
        );
      } else {
        this.state.savedSettings = [saved, ...this.state.savedSettings];
        this.state.selectedSettingsId = saved.id;
      }
      this.forceRender();
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert('Failed to save settings. Please try again.');
    }
  }

  private async handleSaveSettingsAs(): Promise<void> {
    const name = prompt('Enter a name for this settings profile:');
    if (!name || !name.trim()) return;

    try {
      const settings = this.gatherCurrentSettings();
      const saved = await saveSettings(name.trim(), settings);
      this.state.savedSettings = [saved, ...this.state.savedSettings];
      this.state.selectedSettingsId = saved.id;
      this.forceRender();
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert('Failed to save settings. Please try again.');
    }
  }

  private async handleRenameSettings(): Promise<void> {
    if (!this.state.selectedSettingsId) return;
    const settings = this.state.savedSettings.find(s => s.id === this.state.selectedSettingsId);
    if (!settings) return;

    const newName = prompt('Enter a new name:', settings.name);
    if (!newName || !newName.trim() || newName.trim() === settings.name) return;

    try {
      const renamed = await renameSettings(this.state.selectedSettingsId, newName.trim());
      if (renamed) {
        this.state.savedSettings = this.state.savedSettings.map(s =>
          s.id === this.state.selectedSettingsId ? renamed : s
        );
        this.forceRender();
      }
    } catch (error) {
      console.error('Failed to rename settings:', error);
      alert('Failed to rename settings. Please try again.');
    }
  }

  private async handleDeleteSettings(): Promise<void> {
    if (!this.state.selectedSettingsId) return;
    const settings = this.state.savedSettings.find(s => s.id === this.state.selectedSettingsId);
    if (!settings) return;

    if (!confirm(`Delete settings profile "${settings.name}"?`)) return;

    try {
      await deleteSettings(this.state.selectedSettingsId);
      this.state.savedSettings = this.state.savedSettings.filter(
        s => s.id !== this.state.selectedSettingsId
      );
      this.state.selectedSettingsId = null;
      this.forceRender();
    } catch (error) {
      console.error('Failed to delete settings:', error);
      alert('Failed to delete settings. Please try again.');
    }
  }

  private handleExportSettings(): void {
    const settings = this.gatherCurrentSettings();
    const json = exportSettingsToJson(settings);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `ease-settings-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  private async handleImportSettings(file: File): Promise<void> {
    try {
      const text = await file.text();
      const settings = parseImportedJson(text);

      if (!settings) {
        alert('Invalid settings file. Please select a valid EASE settings JSON file.');
        return;
      }

      // Set flag to skip auto-save (we're loading, not making changes)
      this.skipAutoSave = true;

      // Apply imported settings
      if (settings.aiConfig) {
        this.aiConfig = { ...this.aiConfig, ...settings.aiConfig };
        this.callbacks.onConfigChange(settings.aiConfig);
      }

      if (settings.mappingConfig) {
        this.mappingConfig = settings.mappingConfig;
        this.callbacks.onMappingConfigChange(settings.mappingConfig);
        if (this.mappingPanel) {
          this.mappingPanel.setConfig(settings.mappingConfig);
        }
      }

      // Reset the skip flag after all callbacks have been processed
      this.skipAutoSave = false;

      // Clear selection since we loaded from file, not a saved profile
      this.state.selectedSettingsId = null;
      this.forceRender();

      alert('Settings imported successfully!');
    } catch (error) {
      console.error('Failed to import settings:', error);
      alert('Failed to import settings. Please try again.');
    }
  }

  private async loadUserMappingPresets(): Promise<void> {
    try {
      const presets = await getAllMappingPresets();
      this.state.userMappingPresets = presets;
      this.forceRender();
    } catch (error) {
      console.error('Failed to load user mapping presets:', error);
    }
  }

  private async fetchAvailableLoras(): Promise<void> {
    this.state.availableLorasLoading = true;
    try {
      const httpUrl = this.aiConfig.serverUrl
        .replace(/^ws/, 'http')
        .replace(/\/ws$/, '');
      const response = await fetch(`${httpUrl}/loras`);
      const data = await response.json();
      this.state.availableLoras = data.loras || [];
    } catch (error) {
      console.error('Failed to fetch available LoRAs:', error);
      this.state.availableLoras = [];
    } finally {
      this.state.availableLorasLoading = false;
      this.forceRender();
    }
  }

  private loadExpandedState(): void {
    const sections = [
      'settings',
      'connection',
      'prompts',
      'styles',
      'generation',
      'audio-effects',
      'upscaling',
      'story',
      'lyrics',
      'advanced',
      'mapping',
    ];
    sections.forEach((key) => {
      const stored = localStorage.getItem(`ai-panel-section-${key}`);
      if (stored === 'true') {
        this.state.expandedSections.add(key);
      } else if (stored === 'false') {
        this.state.expandedSections.delete(key);
      }
    });
    // Ensure defaults
    if (!localStorage.getItem('ai-panel-section-connection')) {
      this.state.expandedSections.add('connection');
    }
    if (!localStorage.getItem('ai-panel-section-prompts')) {
      this.state.expandedSections.add('prompts');
    }
  }

  private getCapabilities(): string[] {
    return getEffectiveCapabilities(
      this.aiState.isConnected,
      this.aiState.serverConfig?.capabilities,
      this.state.selectedBackendId
    );
  }

  private getAvailableBackends(): BackendInfo[] {
    return getAvailableBackends(
      this.aiState.isConnected,
      this.aiState.serverConfig?.available_backends
    );
  }

  protected render(): void {
    const { expandedSections } = this.state;
    const state = this.aiState;
    const config = this.aiConfig;
    const isConnected = state.isConnected;
    const capabilities = this.getCapabilities();

    this.el.className = `control-panel ai-generator-panel ${!isConnected ? 'disconnected' : ''}`;

    // Build section HTML
    const sectionsHtml = [
      this.renderSection(
        'settings',
        'Settings Profiles',
        false,
        getSettingsBadge(this.state.selectedSettingsId, this.state.savedSettings),
        this.renderSettingsSectionContent(),
        false
      ),
      this.renderSection(
        'connection',
        'Server Connection',
        true,
        undefined,
        this.renderConnectionSectionContent(),
        false
      ),
      this.renderSection(
        'prompts',
        'Prompts',
        true,
        undefined,
        this.renderPromptsSectionContent(),
        false
      ),
      this.renderSection(
        'styles',
        'Styles',
        false,
        getStylesBadge(config, this.currentPresetName),
        this.renderStylesSectionContent(),
        false
      ),
      this.renderSection(
        'generation',
        'Generation',
        false,
        this.getGenerationBadge(),
        this.renderGenerationSectionContent(),
        false
      ),
      this.renderSection(
        'audio-effects',
        'Audio-Reactive Effects',
        false,
        getAudioEffectsBadge(config),
        this.renderAudioEffectsSectionContent(),
        false
      ),
      this.renderSection(
        'upscaling',
        'Image Quality',
        false,
        getUpscalingBadge(config),
        this.renderUpscalingSectionContent(),
        false
      ),
      this.renderSection(
        'story',
        'Story Mode',
        false,
        this.getStoryBadge(),
        html`<div id="story-panel-container"></div>`,
        false
      ),
      this.renderSection(
        'lyrics',
        'Lyrics',
        false,
        getLyricsBadge(config),
        this.renderLyricsSectionContent(),
        false
      ),
      this.renderSection(
        'advanced',
        'Advanced',
        false,
        undefined,
        this.renderAdvancedSectionContent(),
        false
      ),
    ];

    this.el.innerHTML = sectionsHtml.join('');

    // Mount child components
    this.mountChildComponents();
  }

  private renderSection(
    key: string,
    title: string,
    defaultExpanded: boolean,
    badge: string | undefined,
    content: string,
    disabled: boolean = false
  ): string {
    const isExpanded = this.state.expandedSections.has(key);

    return html`
      <div class="collapsible-section ${isExpanded ? 'expanded' : ''} ${disabled ? 'disabled' : ''}">
        <button
          class="collapsible-header ${isExpanded ? 'expanded' : ''}"
          data-action="toggle-section"
          data-section="${key}"
        >
          <span class="collapse-arrow ${isExpanded ? 'expanded' : ''}">&#9654;</span>
          <span class="collapsible-title">${title}</span>
          ${badge && !isExpanded ? html`<span class="collapsible-badge">${badge}</span>` : ''}
          ${disabled && !isExpanded
            ? html`<span class="collapsible-disabled-hint">Connect to enable</span>`
            : ''}
        </button>
        ${isExpanded
          ? html`<div class="collapsible-content ${disabled ? 'disabled' : ''}">${raw(content)}</div>`
          : ''}
      </div>
    `;
  }

  // === Section Content Renderers ===

  private renderSettingsSectionContent(): string {
    const props: SettingsSectionProps = {
      savedSettings: this.state.savedSettings,
      selectedSettingsId: this.state.selectedSettingsId,
      isLoading: this.state.settingsLoading,
      lastSessionRecord: this.state.lastSessionRecord,
      callbacks: {
        onSelectSettings: (id) => {
          if (id) {
            this.handleLoadSettings(id);
          } else {
            this.state.selectedSettingsId = null;
            this.forceRender();
          }
        },
        onSaveSettings: () => this.handleSaveSettings(),
        onSaveSettingsAs: () => this.handleSaveSettingsAs(),
        onRenameSettings: () => this.handleRenameSettings(),
        onDeleteSettings: () => this.handleDeleteSettings(),
        onExportSettings: () => this.handleExportSettings(),
        onImportSettings: (file) => this.handleImportSettings(file),
      },
    };
    return renderSettingsSection(props);
  }

  private renderConnectionSectionContent(): string {
    const props: ConnectionSectionProps = {
      config: this.aiConfig,
      state: this.aiState,
      capabilities: this.getCapabilities(),
      isConnected: this.aiState.isConnected,
      isConnecting: this.state.isConnecting || this.aiState.isConnecting,
      availableBackends: this.getAvailableBackends(),
      selectedBackendId: this.state.selectedBackendId,
      hasPendingChanges: this.state.hasPendingChanges,
      callbacks: {
        onConnect: this.callbacks.onConnect,
        onDisconnect: this.callbacks.onDisconnect,
        onStartGeneration: this.callbacks.onStartGeneration,
        onStopGeneration: this.callbacks.onStopGeneration,
        onResetFeedback: this.callbacks.onResetFeedback,
        onClearVram: this.callbacks.onClearVram,
        onSwitchBackend: this.callbacks.onSwitchBackend,
        onSelectBackend: (backendId: string) => {
          this.state.selectedBackendId = backendId;
          setStoredBackendId(backendId);
          // Auto-switch to feedback mode if current mode requires StreamDiffusion
          if (backendId !== 'stream_diffusion' && this.aiConfig.generationMode === 'keyframe_rife') {
            this.callbacks.onConfigChange({ generationMode: 'feedback' });
          }
          this.forceRender();
        },
        onConfigChange: this.callbacks.onConfigChange,
      },
    };
    return renderConnectionSection(props);
  }

  private renderPromptsSectionContent(): string {
    const props: PromptsSectionProps = {
      config: this.aiConfig,
      state: this.aiState,
      capabilities: this.getCapabilities(),
      isConnected: this.aiState.isConnected,
      mappingConfig: this.mappingConfig,
      currentMetrics: this.currentMetrics,
      callbacks: {
        onConfigChange: this.callbacks.onConfigChange,
        onMappingConfigChange: this.callbacks.onMappingConfigChange,
      },
    };
    return renderPromptsSection(props);
  }

  private renderStylesSectionContent(): string {
    const props: StylesSectionProps = {
      config: this.aiConfig,
      state: this.aiState,
      capabilities: this.getCapabilities(),
      isConnected: this.aiState.isConnected,
      mappingConfig: this.mappingConfig,
      currentMetrics: this.currentMetrics,
      callbacks: {
        onConfigChange: this.callbacks.onConfigChange,
        onMappingConfigChange: this.callbacks.onMappingConfigChange,
      },
    };
    const options: StylesSectionOptions = {
      isMappingExpanded: this.state.expandedSections.has('mapping'),
      userPresets: this.state.userMappingPresets,
      selectedUserPresetId: this.state.selectedUserMappingPresetId,
      currentPresetName: this.currentPresetName,
      hasMappingChanges: this.hasMappingChanges,
      isBuiltInPreset: this.currentPresetIsBuiltIn,
    };
    return renderStylesSection(props, options);
  }

  private renderGenerationSectionContent(): string {
    const props: GenerationSectionProps = {
      config: this.aiConfig,
      state: this.aiState,
      capabilities: this.getCapabilities(),
      isConnected: this.aiState.isConnected,
      callbacks: {
        onConfigChange: this.callbacks.onConfigChange,
      },
    };
    return renderGenerationSection(props);
  }

  private renderAudioEffectsSectionContent(): string {
    const props: AudioEffectsSectionProps = {
      config: this.aiConfig,
      state: this.aiState,
      capabilities: this.getCapabilities(),
      isConnected: this.aiState.isConnected,
      callbacks: {
        onConfigChange: this.callbacks.onConfigChange,
      },
    };
    return renderAudioEffectsSection(props);
  }

  private renderUpscalingSectionContent(): string {
    const props: UpscalingSectionProps = {
      config: this.aiConfig,
      state: this.aiState,
      capabilities: this.getCapabilities(),
      isConnected: this.aiState.isConnected,
      callbacks: {
        onConfigChange: this.callbacks.onConfigChange,
      },
    };
    return renderUpscalingSection(props);
  }

  private renderLyricsSectionContent(): string {
    const props: LyricsSectionProps = {
      config: this.aiConfig,
      state: this.aiState,
      capabilities: this.getCapabilities(),
      isConnected: this.aiState.isConnected,
      callbacks: {
        onConfigChange: this.callbacks.onConfigChange,
        onResetLyrics: this.callbacks.onResetLyrics,
      },
    };
    return renderLyricsSection(props);
  }

  private renderAdvancedSectionContent(): string {
    const props: AdvancedSectionProps = {
      config: this.aiConfig,
      state: this.aiState,
      capabilities: this.getCapabilities(),
      isConnected: this.aiState.isConnected,
      newLoraPath: this.state.newLoraPath,
      newLoraWeight: this.state.newLoraWeight,
      availableLoras: this.state.availableLoras,
      availableLorasLoading: this.state.availableLorasLoading,
      callbacks: {
        onConfigChange: this.callbacks.onConfigChange,
        onClearBaseImage: this.callbacks.onClearBaseImage,
      },
      onLoraPathChange: (path: string) => {
        this.state.newLoraPath = path;
      },
      onLoraWeightChange: (weight: number) => {
        this.state.newLoraWeight = weight;
      },
      onAddLora: () => {
        if (!this.state.newLoraPath.trim()) return;
        const newLora: LoraConfig = {
          path: this.state.newLoraPath.trim(),
          weight: this.state.newLoraWeight,
          name: `lora_${this.aiConfig.loras.length}`,
        };
        this.callbacks.onConfigChange({
          loras: [...this.aiConfig.loras, newLora],
        });
        this.state.newLoraPath = '';
        this.state.newLoraWeight = 0.8;
        this.forceRender();
      },
    };
    return renderAdvancedSection(props);
  }

  private getStoryBadge(): string | undefined {
    return this.storyController.hasStory ? 'ON' : undefined;
  }

  private getGenerationBadge(): string | undefined {
    // Show ControlNet status if enabled
    if (this.aiConfig.useControlNet) {
      return 'ControlNet ON';
    }
    return undefined;
  }

  private mountChildComponents(): void {
    // Mount MappingPanel (now in Styles section)
    const mappingContainer = this.$('#mapping-panel-container');
    if (mappingContainer && this.state.expandedSections.has('styles')) {
      // Remount if container changed (after re-render destroyed the old one)
      if (this.mappingPanel && !this.mappingPanel.isConnected) {
        this.mappingPanel.dispose();
        this.mappingPanel = null;
      }
      if (!this.mappingPanel) {
        this.mappingPanel = new MappingPanel(
          mappingContainer,
          {
            onConfigChange: (config) => {
              this.callbacks.onMappingConfigChange(config);
              // Track changes by comparing to original config
              this.checkMappingChanges(config);
            },
          },
          this.mappingConfig
        );
        this.mappingPanel.mount();
      }
      this.mappingPanel.setMetrics(this.currentMetrics);
    }

    // Mount StoryControlPanel
    const storyContainer = this.$('#story-panel-container');
    if (storyContainer && this.state.expandedSections.has('story')) {
      // Remount if container changed (after re-render destroyed the old one)
      if (this.storyPanel && !this.storyPanel.isConnected) {
        this.storyPanel.dispose();
        this.storyPanel = null;
      }
      if (!this.storyPanel) {
        this.storyPanel = new StoryControlPanel(
          storyContainer,
          this.storyController,
          false // No longer disabled when disconnected
        );
        this.storyPanel.mount();
      }
    }

    // Update pose preview canvas
    if (this.aiState.posePreview) {
      const canvas = this.$('#pose-preview-canvas') as HTMLCanvasElement;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(this.aiState.posePreview, 0, 0, 128, 128);
        }
      }
    }
  }

  // === Action Handlers ===

  protected actions: Record<string, (e: Event, target: HTMLElement) => void> = {
    'toggle-section': (e, target) => {
      const section = target.dataset.section!;
      const expanded = this.state.expandedSections;
      if (expanded.has(section)) {
        expanded.delete(section);
        localStorage.setItem(`ai-panel-section-${section}`, 'false');
      } else {
        expanded.add(section);
        localStorage.setItem(`ai-panel-section-${section}`, 'true');
      }
      // Fetch LoRAs when Advanced section is expanded
      if (section === 'advanced' && this.state.expandedSections.has('advanced')) {
        this.fetchAvailableLoras();
      }
      this.forceRender();
    },

    'refresh-loras': () => this.fetchAvailableLoras(),

    // Settings section actions
    'select-settings': (e) => {
      const id = (e.target as HTMLSelectElement).value;
      if (id) {
        this.handleLoadSettings(id);
      } else {
        this.state.selectedSettingsId = null;
        this.forceRender();
      }
    },
    'save-settings': () => this.handleSaveSettings(),
    'save-settings-as': () => this.handleSaveSettingsAs(),
    'rename-settings': () => this.handleRenameSettings(),
    'delete-settings': () => this.handleDeleteSettings(),
    'export-settings': () => this.handleExportSettings(),
    'import-settings': (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        this.handleImportSettings(file);
        // Reset input so same file can be selected again
        (e.target as HTMLInputElement).value = '';
      }
    },

    // Connection section actions
    connect: async () => {
      if (this.aiState.isConnected) {
        this.callbacks.onDisconnect();
      } else {
        this.state.isConnecting = true;
        try {
          await this.callbacks.onConnect();
          this.state.hasPendingChanges = false;

          // Sync server to user's selected backend after connect
          this.callbacks.onSwitchBackend?.(this.state.selectedBackendId);
        } catch (e) {
          console.error('Connection failed:', e);
        }
        this.state.isConnecting = false;
      }
    },
    'toggle-generation': () => {
      if (this.aiState.isGenerating) {
        this.callbacks.onStopGeneration();
      } else {
        this.callbacks.onStartGeneration();
      }
    },
    'reset-feedback': () => this.callbacks.onResetFeedback(),
    'switch-backend': (e) => {
      const backendId = (e.target as HTMLSelectElement).value;
      this.callbacks.onSwitchBackend?.(backendId);
    },
    'select-backend': (e) => {
      const backendId = (e.target as HTMLSelectElement).value;
      this.state.selectedBackendId = backendId;
      setStoredBackendId(backendId);
      // Auto-switch to feedback mode if current mode requires StreamDiffusion
      if (backendId !== 'stream_diffusion' && this.aiConfig.generationMode === 'keyframe_rife') {
        this.callbacks.onConfigChange({ generationMode: 'feedback' });
      }
      this.forceRender();
    },

    // Prompts section actions
    'base-prompt': (e) => {
      this.callbacks.onConfigChange({
        basePrompt: (e.target as HTMLTextAreaElement).value,
      });
      this.markPendingChanges();
    },
    'negative-prompt': (e) => {
      this.callbacks.onConfigChange({
        negativePrompt: (e.target as HTMLTextAreaElement).value,
      });
      this.markPendingChanges();
    },
    'preset-subject': () => {
      this.callbacks.onConfigChange({
        basePrompt: PROMPT_PRESETS.subject,
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'preset-pattern': () => {
      this.callbacks.onConfigChange({
        basePrompt: PROMPT_PRESETS.pattern,
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'mapping-preset': (e) => {
      const value = (e.target as HTMLSelectElement).value as AIGeneratorConfig['mappingPreset'];
      if (!value) return; // Ignore placeholder selection
      this.callbacks.onConfigChange({
        mappingPreset: value,
      });
      // Load the preset's mapping config
      const presetConfig = getMappingPresetConfig(value);
      if (presetConfig) {
        // Update local config so MappingPanel gets correct config on re-render
        this.mappingConfig = presetConfig;
        this.callbacks.onMappingConfigChange(presetConfig);
        // Store original config for reset functionality
        this.originalMappingConfig = JSON.parse(JSON.stringify(presetConfig));
        this.hasMappingChanges = false;
      }
      // Update preset tracking state
      const presetLabels: Record<string, string> = {
        reactive: 'Reactive',
        dancer: 'Dancer',
        vj_intense: 'VJ Mode',
        dreamscape: 'Dream',
        color_organ: 'Color Organ',
      };
      this.currentPresetName = presetLabels[value] || value;
      this.currentPresetIsBuiltIn = true;
      // Clear user preset selection when selecting built-in
      this.state.selectedUserMappingPresetId = null;
      this.markPendingChanges();
      this.forceRender();
    },
    'user-mapping-preset': (e) => {
      const presetId = (e.target as HTMLSelectElement).value;
      if (!presetId) {
        // Deselected user preset
        this.state.selectedUserMappingPresetId = null;
        this.forceRender();
        return;
      }
      const preset = this.state.userMappingPresets.find(p => p.id === presetId);
      if (preset) {
        this.state.selectedUserMappingPresetId = presetId;
        // Apply the user preset's mapping config
        const configCopy = JSON.parse(JSON.stringify(preset.config));
        // Update local config so MappingPanel gets correct config on re-render
        this.mappingConfig = configCopy;
        this.callbacks.onMappingConfigChange(configCopy);
        // Store original config for reset functionality
        this.originalMappingConfig = JSON.parse(JSON.stringify(preset.config));
        this.hasMappingChanges = false;
        // Update preset tracking state
        this.currentPresetName = preset.name;
        this.currentPresetIsBuiltIn = false;
        this.markPendingChanges();
        this.forceRender();
      }
    },
    'save-new-preset': async () => {
      const name = prompt('Enter a name for this preset:');
      if (!name || !name.trim()) return;
      try {
        const saved = await saveMappingPreset(name.trim(), this.mappingConfig);
        this.state.userMappingPresets = [saved, ...this.state.userMappingPresets];
        this.state.selectedUserMappingPresetId = saved.id;
        this.forceRender();
      } catch (error) {
        console.error('Failed to save preset:', error);
        alert('Failed to save preset. Please try again.');
      }
    },
    'copy-preset': async () => {
      const presetId = this.state.selectedUserMappingPresetId;
      if (!presetId) return;
      const preset = this.state.userMappingPresets.find(p => p.id === presetId);
      if (!preset) return;
      const name = prompt('Enter a name for the copy:', `${preset.name} (copy)`);
      if (!name || !name.trim()) return;
      try {
        const saved = await saveMappingPreset(name.trim(), this.mappingConfig);
        this.state.userMappingPresets = [saved, ...this.state.userMappingPresets];
        this.state.selectedUserMappingPresetId = saved.id;
        this.forceRender();
      } catch (error) {
        console.error('Failed to copy preset:', error);
        alert('Failed to copy preset. Please try again.');
      }
    },
    'rename-preset': async () => {
      const presetId = this.state.selectedUserMappingPresetId;
      if (!presetId) return;
      const preset = this.state.userMappingPresets.find(p => p.id === presetId);
      if (!preset) return;
      const name = prompt('Enter a new name:', preset.name);
      if (!name || !name.trim() || name.trim() === preset.name) return;
      try {
        const updated = await saveMappingPreset(name.trim(), preset.config, presetId);
        this.state.userMappingPresets = this.state.userMappingPresets.map(p =>
          p.id === presetId ? updated : p
        );
        this.forceRender();
      } catch (error) {
        console.error('Failed to rename preset:', error);
        alert('Failed to rename preset. Please try again.');
      }
    },
    'delete-preset': async () => {
      const presetId = this.state.selectedUserMappingPresetId;
      if (!presetId) return;
      const preset = this.state.userMappingPresets.find(p => p.id === presetId);
      if (!preset) return;
      if (!confirm(`Delete preset "${preset.name}"?`)) return;
      try {
        await deleteMappingPreset(presetId);
        this.state.userMappingPresets = this.state.userMappingPresets.filter(p => p.id !== presetId);
        this.state.selectedUserMappingPresetId = null;
        this.forceRender();
      } catch (error) {
        console.error('Failed to delete preset:', error);
        alert('Failed to delete preset. Please try again.');
      }
    },
    'toggle-mapping': () => {
      const expanded = this.state.expandedSections;
      if (expanded.has('mapping')) {
        expanded.delete('mapping');
        localStorage.setItem('ai-panel-section-mapping', 'false');
      } else {
        expanded.add('mapping');
        localStorage.setItem('ai-panel-section-mapping', 'true');
      }
      this.forceRender();
    },
    'reset-mapping': () => {
      if (!this.originalMappingConfig) return;
      // Restore the original config
      const configCopy = JSON.parse(JSON.stringify(this.originalMappingConfig));
      // Update local config so MappingPanel gets correct config on re-render
      this.mappingConfig = configCopy;
      this.callbacks.onMappingConfigChange(configCopy);
      this.hasMappingChanges = false;
      this.forceRender();
    },
    'save-mapping': async () => {
      if (this.currentPresetIsBuiltIn) {
        // Built-in presets are read-only - shouldn't reach here due to disabled button
        return;
      }
      const presetId = this.state.selectedUserMappingPresetId;
      if (!presetId) return;
      try {
        const updated = await saveMappingPreset(this.currentPresetName || 'Custom', this.mappingConfig, presetId);
        // Update local state
        this.state.userMappingPresets = this.state.userMappingPresets.map(p =>
          p.id === presetId ? updated : p
        );
        // Update original config to match saved
        this.originalMappingConfig = JSON.parse(JSON.stringify(this.mappingConfig));
        this.hasMappingChanges = false;
        this.forceRender();
      } catch (error) {
        console.error('Failed to save preset:', error);
        alert('Failed to save preset. Please try again.');
      }
    },

    // Generation section actions
    'generation-mode': (e) => {
      this.callbacks.onConfigChange({
        generationMode: (e.target as HTMLSelectElement).value as AIGeneratorConfig['generationMode'],
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'img2img-strength': (e) => {
      const value = Number((e.target as HTMLInputElement).value);
      this.callbacks.onConfigChange({ img2imgStrength: value / 100 });
      this.updateSliderLabel(e.target as HTMLInputElement, `Transform Strength: ${Math.round(value)}%`);
      this.markPendingChanges();
    },
    'target-fps': (e) => {
      const value = Number((e.target as HTMLInputElement).value);
      this.callbacks.onConfigChange({ targetFps: value });
      this.updateSliderLabel(e.target as HTMLInputElement, `Target FPS: ${value}`);
      this.markPendingChanges();
    },
    'temporal-coherence': (e) => {
      this.callbacks.onConfigChange({
        temporalCoherence: (e.target as HTMLInputElement).checked ? 'blending' : 'none',
      });
      this.markPendingChanges();
    },
    'periodic-refresh': (e) => {
      this.callbacks.onConfigChange({
        periodicPoseRefresh: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    'keyframe-interval': (e) => {
      const value = Number((e.target as HTMLInputElement).value);
      this.callbacks.onConfigChange({ keyframeInterval: value });
      this.updateSliderLabel(e.target as HTMLInputElement, `Keyframe Interval: every ${value} frames`);
      this.markPendingChanges();
    },
    'keyframe-strength': (e) => {
      const value = Number((e.target as HTMLInputElement).value);
      this.callbacks.onConfigChange({ keyframeStrength: value / 100 });
      this.updateSliderLabel(e.target as HTMLInputElement, `Keyframe Strength: ${Math.round(value)}%`);
      this.markPendingChanges();
    },

    // ControlNet section actions
    'use-controlnet': (e) => {
      this.callbacks.onConfigChange({
        useControlNet: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'controlnet-weight': (e) => {
      const value = Number((e.target as HTMLInputElement).value);
      this.callbacks.onConfigChange({ controlNetPoseWeight: value / 100 });
      this.updateSliderLabel(e.target as HTMLInputElement, `Pose Influence: ${Math.round(value)}%`);
      this.markPendingChanges();
    },
    'pose-lock': (e) => {
      this.callbacks.onConfigChange({
        controlNetPoseLock: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    'procedural-pose': (e) => {
      this.callbacks.onConfigChange({
        useProceduralPose: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'pose-mode': (e, target) => {
      this.callbacks.onConfigChange({
        poseAnimationMode: target.dataset.mode!,
      });
      this.markPendingChanges();
    },
    'pose-framing': (e, target) => {
      this.callbacks.onConfigChange({
        poseFraming: target.dataset.framing as 'full_body' | 'upper_body' | 'portrait',
      });
      this.markPendingChanges();
    },
    'pose-speed': (e) => {
      const value = Number((e.target as HTMLInputElement).value) / 100;
      this.callbacks.onConfigChange({ poseAnimationSpeed: value });
      this.updateSliderLabel(e.target as HTMLInputElement, `Animation Speed: ${value.toFixed(1)}x`);
      this.markPendingChanges();
    },
    'pose-intensity': (e) => {
      const value = Number((e.target as HTMLInputElement).value);
      this.callbacks.onConfigChange({ poseAnimationIntensity: value / 100 });
      this.updateSliderLabel(e.target as HTMLInputElement, `Movement Intensity: ${Math.round(value)}%`);
      this.markPendingChanges();
    },

    // Audio effects section actions
    'spectral-displacement': (e) => {
      this.callbacks.onConfigChange({
        enableSpectralDisplacement: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    'glitch-blocks': (e) => {
      this.callbacks.onConfigChange({
        enableGlitchBlocks: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    'treble-grain': (e) => {
      this.callbacks.onConfigChange({
        enableTrebleGrain: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    'shader-effects': (e) => {
      this.callbacks.onConfigChange({
        enableShaderEffects: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    flash: (e) => {
      this.callbacks.onConfigChange({
        enableFlash: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    'silence-degradation': (e) => {
      this.callbacks.onConfigChange({
        enableSilenceDegradation: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'silence-threshold': (e) => {
      const value = Number((e.target as HTMLInputElement).value);
      this.callbacks.onConfigChange({ silenceThreshold: value / 100 });
      this.updateSliderLabel(e.target as HTMLInputElement, `Silence Threshold: ${value.toFixed(0)}%`);
      this.markPendingChanges();
    },
    'degradation-rate': (e) => {
      const value = Number((e.target as HTMLInputElement).value) / 100;
      this.callbacks.onConfigChange({ degradationRate: value });
      this.updateSliderLabel(e.target as HTMLInputElement, `Degradation Speed: ${value.toFixed(1)}x`);
      this.markPendingChanges();
    },
    'recovery-rate': (e) => {
      const value = Number((e.target as HTMLInputElement).value) / 100;
      this.callbacks.onConfigChange({ recoveryRate: value });
      this.updateSliderLabel(e.target as HTMLInputElement, `Recovery Speed: ${value.toFixed(1)}x`);
      this.markPendingChanges();
    },

    // Upscaling section actions
    bicubic: (e) => {
      this.callbacks.onConfigChange({
        enableBicubic: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    sharpening: (e) => {
      this.callbacks.onConfigChange({
        enableSharpening: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'sharpen-strength': (e) => {
      const value = parseFloat((e.target as HTMLInputElement).value);
      this.callbacks.onConfigChange({ sharpenStrength: value });
      this.updateSliderLabel(e.target as HTMLInputElement, `Sharpen Strength: ${value.toFixed(2)}`);
      this.markPendingChanges();
    },

    // Lyrics section actions
    'enable-lyrics': (e) => {
      this.callbacks.onConfigChange({
        enableLyrics: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'reset-lyrics': () => this.callbacks.onResetLyrics?.(),
    'lyric-driven': (e) => {
      this.callbacks.onConfigChange({
        lyricDrivenMode: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    'show-subtitles': (e) => {
      this.callbacks.onConfigChange({
        showLyricSubtitles: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },

    // Advanced section actions
    acceleration: (e, target) => {
      this.callbacks.onConfigChange({
        acceleration: target.dataset.method as AIGeneratorConfig['acceleration'],
      });
      this.markPendingChanges();
      this.forceRender();
    },
    'hyper-sd-steps': (e, target) => {
      this.callbacks.onConfigChange({
        hyperSdSteps: Number(target.dataset.steps) as 1 | 2 | 4 | 8,
      });
      this.markPendingChanges();
    },
    'server-url': (e) => {
      this.callbacks.onConfigChange({
        serverUrl: (e.target as HTMLInputElement).value,
      });
      this.markPendingChanges();
    },
    'model-id': (e) => {
      this.callbacks.onConfigChange({
        modelId: (e.target as HTMLInputElement).value,
      });
      this.markPendingChanges();
    },
    width: (e) => {
      this.callbacks.onConfigChange({
        width: Number((e.target as HTMLInputElement).value),
      });
      this.markPendingChanges();
    },
    height: (e) => {
      this.callbacks.onConfigChange({
        height: Number((e.target as HTMLInputElement).value),
      });
      this.markPendingChanges();
    },
    'maintain-aspect': (e) => {
      const checked = (e.target as HTMLInputElement).checked;
      this.callbacks.onConfigChange({
        maintainAspectRatio: checked,
      });
      // Update helper text surgically (no markPendingChanges - checkboxes don't need defer-while-typing)
      const description = (e.target as HTMLElement).closest('.sub-control')?.querySelector('.control-description');
      if (description) {
        description.textContent = checked
          ? 'Letterbox/pillarbox to preserve proportions'
          : 'Stretch to fill screen';
      }
    },
    'base-image': (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = (reader.result as string).split(',')[1];
        this.callbacks.onConfigChange({ baseImage: base64 });
        this.markPendingChanges();
        this.forceRender();
      };
      reader.readAsDataURL(file);
    },
    'clear-base-image': () => {
      this.callbacks.onClearBaseImage?.();
      this.callbacks.onConfigChange({ baseImage: null });
      this.markPendingChanges();
      this.forceRender();
    },
    'lock-base-image': (e) => {
      this.callbacks.onConfigChange({
        lockToBaseImage: (e.target as HTMLInputElement).checked,
      });
      this.markPendingChanges();
    },
    'new-lora-path': (e) => {
      this.state.newLoraPath = (e.target as HTMLSelectElement).value;
    },
    'new-lora-weight': (e) => {
      const value = Number((e.target as HTMLInputElement).value) / 100;
      this.state.newLoraWeight = value;
      const container = (e.target as HTMLInputElement).closest('.add-lora-weight');
      const label = container?.querySelector('label');
      if (label) label.textContent = `Weight: ${Math.round(value * 100)}%`;
    },
    'add-lora': () => {
      if (!this.state.newLoraPath.trim()) return;
      const newLora: LoraConfig = {
        path: this.state.newLoraPath.trim(),
        weight: this.state.newLoraWeight,
        name: `lora_${this.aiConfig.loras.length}`,
      };
      this.callbacks.onConfigChange({
        loras: [...this.aiConfig.loras, newLora],
      });
      this.state.newLoraPath = '';
      this.state.newLoraWeight = 0.8;
      this.markPendingChanges();
      this.forceRender();
    },
    'lora-weight': (e, target) => {
      const index = Number(target.dataset.index);
      const value = Number((e.target as HTMLInputElement).value) / 100;
      const newLoras = this.aiConfig.loras.map((lora, i) =>
        i === index ? { ...lora, weight: value } : lora
      );
      this.callbacks.onConfigChange({ loras: newLoras });
      const container = (e.target as HTMLInputElement).closest('.lora-weight-control');
      const span = container?.querySelector('.lora-weight-value');
      if (span) span.textContent = `${Math.round(value * 100)}%`;
      this.markPendingChanges();
    },
    'remove-lora': (e, target) => {
      const index = Number(target.dataset.index);
      this.callbacks.onConfigChange({
        loras: this.aiConfig.loras.filter((_, i) => i !== index),
      });
      this.markPendingChanges();
      this.forceRender();
    },
  };

  // === Helper Methods ===

  private markPendingChanges(): void {
    if (!this.aiState.isConnected && !this.state.hasPendingChanges) {
      // Defer state change if user is actively typing to prevent focus loss
      if (this.isFormInputFocused()) {
        this.pendingRender = true;
        this.setupBlurListener();
        this.pendingHasPendingChanges = true;
      } else {
        this.state.hasPendingChanges = true;
      }
    }
  }

  private updateSliderLabel(input: HTMLInputElement, text: string): void {
    const section = input.closest('.control-section, .sub-control');
    const label = section?.querySelector('.control-label, .control-label-small');
    if (label) {
      label.textContent = text;
    }
  }

  private checkMappingChanges(config: MappingConfig): void {
    if (!this.originalMappingConfig) {
      // No original config stored yet - store current as original
      this.originalMappingConfig = JSON.parse(JSON.stringify(config));
      this.hasMappingChanges = false;
      return;
    }
    // Compare current config to original using JSON serialization
    const currentJson = JSON.stringify(config);
    const originalJson = JSON.stringify(this.originalMappingConfig);
    const hasChanges = currentJson !== originalJson;

    if (hasChanges !== this.hasMappingChanges) {
      this.hasMappingChanges = hasChanges;
      // Update the header display without full re-render
      const presetLabel = this.$('.mapping-edit-preset-name');
      const resetBtn = this.$('[data-action="reset-mapping"]') as HTMLButtonElement;
      const saveBtn = this.$('[data-action="save-mapping"]') as HTMLButtonElement;

      if (presetLabel) {
        presetLabel.textContent = `Editing: ${this.currentPresetName || 'Custom'}${hasChanges ? ' (modified)' : ''}`;
        presetLabel.classList.toggle('modified', hasChanges);
      }
      if (resetBtn) {
        resetBtn.disabled = !hasChanges;
      }
      if (saveBtn) {
        saveBtn.disabled = this.currentPresetIsBuiltIn || !hasChanges;
      }
    }
  }

  // === Public Update Methods ===

  private isFormInputFocused(): boolean {
    const active = document.activeElement;
    return !!(
      active &&
      this.el.contains(active) &&
      (active instanceof HTMLTextAreaElement ||
        active instanceof HTMLInputElement ||
        active instanceof HTMLSelectElement)
    );
  }

  private setupBlurListener(): void {
    if (this.blurListenerAttached) return;
    this.blurListenerAttached = true;

    this.el.addEventListener('focusout', () => {
      requestAnimationFrame(() => {
        if (!this.isFormInputFocused()) {
          // Apply any pending state changes
          if (this.pendingHasPendingChanges) {
            this.state.hasPendingChanges = true;
            this.pendingHasPendingChanges = false;
          }
          if (this.pendingRender) {
            this.pendingRender = false;
            this.forceRender();
          }
        }
      });
    });
  }

  updateState(state: AIGeneratorState): void {
    const prevState = this.aiState;
    this.aiState = state;

    const isInteractingWithInput = this.isFormInputFocused();

    const needsFullRender =
      prevState.isConnected !== state.isConnected ||
      prevState.isConnecting !== state.isConnecting ||
      prevState.isInitializing !== state.isInitializing ||
      prevState.error !== state.error ||
      prevState.serverConfig !== state.serverConfig ||
      prevState.isSwitchingBackend !== state.isSwitchingBackend;

    if (needsFullRender) {
      // Clear pending changes when we connect
      if (!prevState.isConnected && state.isConnected) {
        this.state.hasPendingChanges = false;
      }

      if (isInteractingWithInput) {
        this.pendingRender = true;
        this.setupBlurListener();
      } else {
        this.forceRender();
      }
      return;
    }

    // Surgical updates for frequently-changing display elements
    const frameCounter = this.$('.frame-counter');
    if (frameCounter && state.isGenerating) {
      frameCounter.textContent = `Frame ${state.frameId}`;
    }

    const genBtn = this.$('.generation-btn') as HTMLButtonElement | null;
    if (genBtn) {
      genBtn.textContent = state.isInitializing
        ? 'Init...'
        : state.isGenerating
          ? 'Stop'
          : 'Start';
      genBtn.className = `generation-btn ${state.isGenerating ? 'generating' : ''} ${state.isInitializing ? 'initializing' : ''}`;
    }

    // Update lyrics display
    if (state.lyrics) {
      const lyricText = this.$('.lyric-text');
      if (lyricText) {
        lyricText.textContent = state.lyrics.text?.slice(-100) || '';
      }
      const lyricKeywords = this.$('.lyric-keywords');
      if (lyricKeywords && state.lyrics.keywords.length > 0) {
        lyricKeywords.textContent = `Keywords: ${state.lyrics.keywords
          .map(([word]: [string, number]) => word)
          .join(', ')}`;
      }
    }

    // Update last prompt display
    const lastPromptDisplay = this.$('.last-prompt-display');
    if (lastPromptDisplay && state.lastParams) {
      lastPromptDisplay.textContent = state.lastParams.prompt;
    }

    // Update pose preview canvas
    if (state.posePreview) {
      const canvas = this.$('#pose-preview-canvas') as HTMLCanvasElement | null;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(state.posePreview, 0, 0, 128, 128);
        }
      }
    }
  }

  updateConfig(config: AIGeneratorConfig): void {
    this.aiConfig = config;
    this.scheduleAutoSave();
  }

  updateMappingConfig(config: MappingConfig): void {
    this.mappingConfig = config;
    if (this.mappingPanel) {
      this.mappingPanel.setConfig(config);
    }
    this.scheduleAutoSave();
  }

  updateMetrics(metrics: AudioMetrics | null): void {
    this.currentMetrics = metrics;
    if (this.mappingPanel) {
      this.mappingPanel.setMetrics(metrics);
    }
  }

  protected onDispose(): void {
    if (this.autoSaveTimer !== null) {
      window.clearTimeout(this.autoSaveTimer);
      this.autoSaveTimer = null;
    }
    if (this.boundBeforeUnload) {
      window.removeEventListener('beforeunload', this.boundBeforeUnload);
      this.boundBeforeUnload = null;
    }
    this.mappingPanel?.dispose();
    this.storyPanel?.dispose();
  }
}
