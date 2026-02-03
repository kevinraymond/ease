/**
 * Shared types for AI generator section components.
 */

import { AIGeneratorConfig, AIGeneratorState, BackendInfo, LoraConfig } from '../../core/types';
import { MappingConfig } from '../MappingPanel';
import { AudioMetrics } from '../../audio/types';
import { StoryController } from '../../controllers/StoryController';
import { SavedSettings } from '../../ui/data/settingsStorage';

/**
 * Common props passed to all section components.
 */
export interface SectionProps {
  config: AIGeneratorConfig;
  state: AIGeneratorState;
  capabilities: string[];
  isConnected: boolean;
}

/**
 * Callbacks for configuration changes.
 */
export interface SectionCallbacks {
  onConfigChange: (config: Partial<AIGeneratorConfig>) => void;
}

/**
 * Connection section specific callbacks.
 */
export interface ConnectionSectionCallbacks extends SectionCallbacks {
  onConnect: () => Promise<void>;
  onDisconnect: () => void;
  onStartGeneration: () => void;
  onStopGeneration: () => void;
  onResetFeedback: () => void;
  onClearVram?: () => Promise<void>;
  onSwitchBackend?: (backendId: string) => void;
  onSelectBackend: (backendId: string) => void; // For offline selection
}

/**
 * Prompts section specific callbacks.
 */
export interface PromptsSectionCallbacks extends SectionCallbacks {
  onMappingConfigChange: (config: MappingConfig) => void;
}

/**
 * Lyrics section specific callbacks.
 */
export interface LyricsSectionCallbacks extends SectionCallbacks {
  onResetLyrics?: () => void;
}

/**
 * Advanced section specific callbacks.
 */
export interface AdvancedSectionCallbacks extends SectionCallbacks {
  onClearBaseImage?: () => void;
}

/**
 * Props for ConnectionSection.
 */
export interface ConnectionSectionProps extends SectionProps {
  isConnecting: boolean;
  availableBackends: BackendInfo[];
  selectedBackendId: string;
  hasPendingChanges: boolean;
  callbacks: ConnectionSectionCallbacks;
}

/**
 * Props for PromptsSection.
 */
export interface PromptsSectionProps extends SectionProps {
  mappingConfig: MappingConfig;
  currentMetrics: AudioMetrics | null;
  callbacks: PromptsSectionCallbacks;
}

/**
 * Styles section specific callbacks.
 */
export interface StylesSectionCallbacks extends SectionCallbacks {
  onMappingConfigChange: (config: MappingConfig) => void;
}

/**
 * Props for StylesSection.
 */
export interface StylesSectionProps extends SectionProps {
  mappingConfig: MappingConfig;
  currentMetrics: AudioMetrics | null;
  callbacks: StylesSectionCallbacks;
}

/**
 * Props for GenerationSection.
 */
export interface GenerationSectionProps extends SectionProps {
  callbacks: SectionCallbacks;
}

/**
 * Props for ControlNetSection.
 */
export interface ControlNetSectionProps extends SectionProps {
  callbacks: SectionCallbacks;
}

/**
 * Props for AudioEffectsSection.
 */
export interface AudioEffectsSectionProps extends SectionProps {
  callbacks: SectionCallbacks;
}

/**
 * Props for UpscalingSection.
 */
export interface UpscalingSectionProps extends SectionProps {
  callbacks: SectionCallbacks;
}

/**
 * Props for LyricsSection.
 */
export interface LyricsSectionProps extends SectionProps {
  callbacks: LyricsSectionCallbacks;
}

/**
 * Props for AdvancedSection.
 */
export interface AdvancedSectionProps extends SectionProps {
  newLoraPath: string;
  newLoraWeight: number;
  availableLoras: string[];
  availableLorasLoading: boolean;
  callbacks: AdvancedSectionCallbacks;
  onLoraPathChange: (path: string) => void;
  onLoraWeightChange: (weight: number) => void;
  onAddLora: () => void;
}

/**
 * Capability constants for section visibility.
 */
export const CAPABILITIES = {
  CONTROLNET: 'controlnet',
  LORA: 'lora',
  TAESD: 'taesd',
  TEMPORAL_COHERENCE: 'temporal_coherence',
  ACCELERATION: 'acceleration',
  PROMPT_MODULATION: 'prompt_modulation',
  SEED_CONTROL: 'seed_control',
  STRENGTH_CONTROL: 'strength_control',
} as const;

/**
 * Check if a section should be visible based on capabilities.
 */
export function shouldShowControlNet(capabilities: string[]): boolean {
  return capabilities.includes(CAPABILITIES.CONTROLNET);
}

export function shouldShowLora(capabilities: string[]): boolean {
  return capabilities.includes(CAPABILITIES.LORA);
}

export function shouldShowAcceleration(capabilities: string[]): boolean {
  return capabilities.includes(CAPABILITIES.ACCELERATION);
}

export function shouldShowTemporalCoherence(capabilities: string[]): boolean {
  return capabilities.includes(CAPABILITIES.TEMPORAL_COHERENCE);
}

/**
 * Settings section callbacks.
 */
export interface SettingsSectionCallbacks {
  onSelectSettings: (id: string | null) => void;
  onSaveSettings: () => void;
  onSaveSettingsAs: () => void;
  onRenameSettings: () => void;
  onDeleteSettings: () => void;
  onExportSettings: () => void;
  onImportSettings: (file: File) => void;
}

/**
 * Props for SettingsSection.
 */
export interface SettingsSectionProps {
  savedSettings: SavedSettings[];
  selectedSettingsId: string | null;
  isLoading: boolean;
  lastSessionRecord: SavedSettings | null;
  callbacks: SettingsSectionCallbacks;
}
