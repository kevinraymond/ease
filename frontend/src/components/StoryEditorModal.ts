/**
 * StoryEditorModal - Modal for editing and creating story sequences.
 */

import { ModalComponent } from '../core/Component';
import { html } from '../core/html';
import {
  StoryConfig,
  SceneDefinition,
  SceneTrigger,
  SceneTransition,
  StoryPresetName,
} from '../core/types';
import { STORY_PRESETS, BLANK_STORY } from '../ui/data/storyPresets';
import {
  SavedStory,
  saveStory,
  getAllStories,
  deleteStory,
} from '../ui/data/storyStorage';

const PRESET_INFO: Record<StoryPresetName, { label: string; description: string }> = {
  skiing_adventure: {
    label: 'Skiing Adventure',
    description: 'Woman skiing with dynamic action/peaceful vistas based on audio energy',
  },
  dancing_figure: {
    label: 'Dancing Figure',
    description: 'Dancer responding to music with dramatic poses during energetic sections',
  },
  abstract_landscape: {
    label: 'Abstract Landscape',
    description: 'Evolving abstract landscapes that respond to music mood',
  },
  minimal_portrait: {
    label: 'Minimal Portrait',
    description: 'Simple portrait with audio-reactive expression changes',
  },
};

const TRIGGER_OPTIONS: { value: SceneTrigger; label: string }[] = [
  { value: 'time', label: 'Time (frames)' },
  { value: 'beat_count', label: 'Beat Count' },
  { value: 'energy_drop', label: 'Energy Drop' },
  { value: 'energy_peak', label: 'Energy Peak' },
];

const TRANSITION_OPTIONS: { value: SceneTransition; label: string }[] = [
  { value: 'cut', label: 'Cut' },
  { value: 'crossfade', label: 'Crossfade' },
  { value: 'zoom_in', label: 'Zoom In' },
  { value: 'zoom_out', label: 'Zoom Out' },
];

type ViewMode = 'picker' | 'editor';

interface StoryEditorModalCallbacks {
  onClose: () => void;
  onLoadStory: (story: StoryConfig) => void;
}

interface StoryEditorModalState {
  view: ViewMode;
  story: StoryConfig;
  expandedScenes: Set<number>;
  expandedAudioSections: Set<number>;
  savedStories: SavedStory[];
  currentSavedId: string | null;
  isSaving: boolean;
}

// Deep clone helper for story config
function cloneStory(story: StoryConfig): StoryConfig {
  return {
    ...story,
    scenes: story.scenes.map((scene) => ({
      ...scene,
      energyBlendRange: [...scene.energyBlendRange] as [number, number],
    })),
  };
}

// Create a new scene with defaults
function createNewScene(index: number): SceneDefinition {
  return {
    id: `scene_${index}`,
    basePrompt: '',
    durationFrames: 120,
    trigger: 'time',
    triggerValue: 0,
    energyBlendRange: [0.3, 0.6],
    transition: 'crossfade',
    transitionFrames: 30,
  };
}

// Format relative time
function formatRelativeTime(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString();
}

export class StoryEditorModal extends ModalComponent<StoryEditorModalState> {
  private callbacks: StoryEditorModalCallbacks;
  private presets: StoryPresetName[];

  constructor(
    container: HTMLElement,
    callbacks: StoryEditorModalCallbacks,
    presets: StoryPresetName[]
  ) {
    super(container, {
      view: 'picker',
      story: cloneStory(BLANK_STORY),
      expandedScenes: new Set([0]),
      expandedAudioSections: new Set(),
      savedStories: [],
      currentSavedId: null,
      isSaving: false,
    });
    this.callbacks = callbacks;
    this.presets = presets;
  }

  override show(): void {
    super.show();
    this.loadSavedStories();
  }

  override hide(): void {
    // Reset state when closing
    this.state.view = 'picker';
    this.state.story = cloneStory(BLANK_STORY);
    this.state.expandedScenes = new Set([0]);
    this.state.expandedAudioSections = new Set();
    this.state.currentSavedId = null;
    super.hide();
    this.callbacks.onClose();
  }

  private async loadSavedStories(): Promise<void> {
    try {
      const stories = await getAllStories();
      this.state.savedStories = stories;
    } catch (e) {
      console.error('Failed to load saved stories:', e);
    }
  }

  protected render(): void {
    const { view, story, savedStories, currentSavedId, isSaving } = this.state;

    this.el.className = 'dialog-overlay';
    this.el.innerHTML = html`
      <div class="dialog story-editor-modal" data-action="stop-propagation">
        ${view === 'picker' ? this.renderPickerView(savedStories) : this.renderEditorView(story, currentSavedId, isSaving)}
      </div>
    `;
  }

  private renderPickerView(savedStories: SavedStory[]): string {
    return html`
      <h3>Edit Story</h3>
      <p class="story-editor-subtitle">
        Select a preset to modify or start from scratch
      </p>

      <div class="story-editor-picker-container">
        ${savedStories.length > 0 ? html`
          <div class="story-editor-section-label">Saved Stories</div>
          <div class="saved-stories-list">
            ${savedStories.map((saved) => html`
              <div
                class="saved-story-item"
                data-action="load-saved"
                data-id="${saved.id}"
              >
                <div class="saved-story-info">
                  <span class="saved-story-name">${saved.name}</span>
                  <span class="saved-story-meta">
                    ${saved.story.scenes.length} scene${saved.story.scenes.length !== 1 ? 's' : ''} · ${formatRelativeTime(saved.updatedAt)}
                  </span>
                </div>
                <button
                  class="saved-story-delete"
                  data-action="delete-saved"
                  data-id="${saved.id}"
                  title="Delete"
                >
                  x
                </button>
              </div>
            `)}
          </div>
          <div class="story-editor-divider">
            <span>or start fresh</span>
          </div>
        ` : ''}

        <div class="story-editor-section-label">Presets</div>
        <div class="story-editor-picker">
          ${this.presets.map((presetName) => html`
            <button
              class="story-editor-preset-btn"
              data-action="select-preset"
              data-preset="${presetName}"
            >
              <span class="preset-label">
                ${PRESET_INFO[presetName]?.label || presetName}
              </span>
              <span class="preset-desc">
                ${PRESET_INFO[presetName]?.description || ''}
              </span>
            </button>
          `)}
          <button
            class="story-editor-preset-btn blank-template"
            data-action="select-preset"
            data-preset=""
          >
            <span class="preset-label">Blank Template</span>
            <span class="preset-desc">Start with a single empty scene</span>
          </button>
        </div>
      </div>

      <div class="dialog-buttons">
        <button data-action="close">Cancel</button>
      </div>
    `;
  }

  private renderEditorView(story: StoryConfig, currentSavedId: string | null, isSaving: boolean): string {
    const { expandedScenes, expandedAudioSections } = this.state;

    return html`
      <div class="story-editor-header">
        <button class="story-editor-back-btn" data-action="back">
          &larr; Back
        </button>
        <h3>Edit Story</h3>
        ${currentSavedId ? html`<span class="story-saved-indicator">Saved</span>` : ''}
      </div>

      <div class="story-editor-form">
        <!-- Story-Level Fields -->
        <div class="story-editor-section">
          <label class="story-field-label" for="story-name-input">Story Name</label>
          <input
            id="story-name-input"
            type="text"
            class="text-input"
            data-field="name"
            data-action="update-story"
            value="${story.name}"
            placeholder="my_story"
          />

          <label class="story-field-label" for="story-description-input">Description</label>
          <input
            id="story-description-input"
            type="text"
            class="text-input"
            data-field="description"
            data-action="update-story"
            value="${story.description || ''}"
            placeholder="Brief description of this story..."
          />

          <label class="story-field-label" for="story-negative-prompt">Default Negative Prompt</label>
          <textarea
            id="story-negative-prompt"
            class="prompt-input"
            data-field="defaultNegativePrompt"
            data-action="update-story"
            placeholder="blurry, low quality, distorted..."
            rows="2"
          >${story.defaultNegativePrompt}</textarea>

          <div class="story-checkboxes">
            <label class="checkbox-label">
              <input
                type="checkbox"
                class="checkbox"
                data-field="loop"
                data-action="update-story-checkbox"
                ${story.loop ? 'checked' : ''}
              />
              Loop story
            </label>
            <label class="checkbox-label">
              <input
                type="checkbox"
                class="checkbox"
                data-field="audioReactiveKeywords"
                data-action="update-story-checkbox"
                ${story.audioReactiveKeywords ? 'checked' : ''}
              />
              Audio-reactive keywords
            </label>
          </div>
        </div>

        <!-- Scenes Section -->
        <div class="story-editor-section">
          <div class="scenes-header">
            <label class="story-field-label">Scenes (${story.scenes.length})</label>
            <button class="add-scene-btn" data-action="add-scene">
              + Add Scene
            </button>
          </div>

          <div class="scene-list">
            ${story.scenes.map((scene, idx) => this.renderScene(scene, idx, expandedScenes.has(idx), expandedAudioSections.has(idx), story.scenes.length))}
          </div>
        </div>
      </div>

      <div class="dialog-buttons">
        <button data-action="close">Cancel</button>
        <button
          class="secondary"
          data-action="save"
          ${isSaving ? 'disabled' : ''}
        >
          ${isSaving ? 'Saving...' : 'Save'}
        </button>
        <button class="primary" data-action="submit">
          Load Story
        </button>
      </div>
    `;
  }

  private renderScene(
    scene: SceneDefinition,
    idx: number,
    expanded: boolean,
    audioExpanded: boolean,
    totalScenes: number
  ): string {
    return html`
      <div class="scene-item" data-scene-index="${idx}">
        <div
          class="scene-header"
          data-action="toggle-scene"
          data-index="${idx}"
        >
          <span class="collapse-arrow">${expanded ? '▼' : '▶'}</span>
          <span class="scene-title">
            Scene ${idx + 1}: ${scene.id || '(unnamed)'}
          </span>
          <button
            class="remove-scene-btn"
            data-action="remove-scene"
            data-index="${idx}"
            ${totalScenes <= 1 ? 'disabled' : ''}
            title="Remove scene"
          >
            x
          </button>
        </div>

        ${expanded ? html`
          <div class="scene-fields">
            <label class="scene-field-label" id="scene-id-label-${idx}">Scene ID</label>
            <input
              type="text"
              class="text-input"
              data-action="update-scene"
              data-index="${idx}"
              data-field="id"
              value="${scene.id}"
              placeholder="scene_id"
              aria-labelledby="scene-id-label-${idx}"
            />

            <label class="scene-field-label" id="base-prompt-label-${idx}">Base Prompt</label>
            <textarea
              class="prompt-input"
              data-action="update-scene"
              data-index="${idx}"
              data-field="basePrompt"
              placeholder="Describe the main visual for this scene..."
              rows="2"
              aria-labelledby="base-prompt-label-${idx}"
            >${scene.basePrompt}</textarea>

            <div class="scene-field-row">
              <div class="scene-field-half">
                <label class="scene-field-label" id="duration-label-${idx}">Duration (frames)</label>
                <input
                  type="number"
                  class="number-input"
                  data-action="update-scene"
                  data-index="${idx}"
                  data-field="durationFrames"
                  data-type="int"
                  value="${scene.durationFrames}"
                  min="1"
                  aria-labelledby="duration-label-${idx}"
                />
              </div>
              <div class="scene-field-half">
                <label class="scene-field-label" id="trigger-label-${idx}">Trigger</label>
                <select
                  class="mode-setting-select"
                  data-action="update-scene"
                  data-index="${idx}"
                  data-field="trigger"
                  aria-labelledby="trigger-label-${idx}"
                >
                  ${TRIGGER_OPTIONS.map((opt) => html`
                    <option value="${opt.value}" ${scene.trigger === opt.value ? 'selected' : ''}>
                      ${opt.label}
                    </option>
                  `)}
                </select>
              </div>
            </div>

            ${scene.trigger !== 'time' ? html`
              <div class="scene-field-row">
                <div class="scene-field-half">
                  <label class="scene-field-label" id="trigger-value-label-${idx}">
                    Trigger Value
                    ${scene.trigger === 'beat_count' ? ' (beats)' : ''}
                    ${(scene.trigger === 'energy_drop' || scene.trigger === 'energy_peak') ? ' (0-1)' : ''}
                  </label>
                  <input
                    type="number"
                    class="number-input"
                    data-action="update-scene"
                    data-index="${idx}"
                    data-field="triggerValue"
                    data-type="float"
                    value="${scene.triggerValue}"
                    min="0"
                    step="${scene.trigger === 'beat_count' ? '1' : '0.05'}"
                    ${scene.trigger === 'beat_count' ? '' : 'max="1"'}
                    aria-labelledby="trigger-value-label-${idx}"
                  />
                </div>
              </div>
            ` : ''}

            <div class="scene-field-row">
              <div class="scene-field-half">
                <label class="scene-field-label" id="transition-label-${idx}">Transition</label>
                <select
                  class="mode-setting-select"
                  data-action="update-scene"
                  data-index="${idx}"
                  data-field="transition"
                  aria-labelledby="transition-label-${idx}"
                >
                  ${TRANSITION_OPTIONS.map((opt) => html`
                    <option value="${opt.value}" ${scene.transition === opt.value ? 'selected' : ''}>
                      ${opt.label}
                    </option>
                  `)}
                </select>
              </div>
              <div class="scene-field-half">
                <label class="scene-field-label" id="transition-frames-label-${idx}">Transition Frames</label>
                <input
                  type="number"
                  class="number-input"
                  data-action="update-scene"
                  data-index="${idx}"
                  data-field="transitionFrames"
                  data-type="int"
                  value="${scene.transitionFrames}"
                  min="1"
                  aria-labelledby="transition-frames-label-${idx}"
                />
              </div>
            </div>

            <!-- Audio Reactive Section - Collapsible -->
            <div class="audio-reactive-section">
              <button
                class="audio-reactive-toggle"
                data-action="toggle-audio-section"
                data-index="${idx}"
              >
                <span class="collapse-arrow">
                  ${audioExpanded ? '▼' : '▶'}
                </span>
                Audio Reactive Options
              </button>

              ${audioExpanded ? html`
                <div class="audio-reactive-fields">
                  <label class="scene-field-label" id="high-energy-prompt-label-${idx}">High Energy Prompt</label>
                  <textarea
                    class="prompt-input"
                    data-action="update-scene"
                    data-index="${idx}"
                    data-field="energyHighPrompt"
                    placeholder="Prompt when audio energy is high..."
                    rows="2"
                    aria-labelledby="high-energy-prompt-label-${idx}"
                  >${scene.energyHighPrompt || ''}</textarea>

                  <label class="scene-field-label" id="low-energy-prompt-label-${idx}">Low Energy Prompt</label>
                  <textarea
                    class="prompt-input"
                    data-action="update-scene"
                    data-index="${idx}"
                    data-field="energyLowPrompt"
                    placeholder="Prompt when audio energy is low..."
                    rows="2"
                    aria-labelledby="low-energy-prompt-label-${idx}"
                  >${scene.energyLowPrompt || ''}</textarea>

                  <label class="scene-field-label" id="beat-prompt-modifier-label-${idx}">Beat Prompt Modifier</label>
                  <textarea
                    class="prompt-input"
                    data-action="update-scene"
                    data-index="${idx}"
                    data-field="beatPromptModifier"
                    placeholder="Keywords to add on beat events..."
                    rows="1"
                    aria-labelledby="beat-prompt-modifier-label-${idx}"
                  >${scene.beatPromptModifier || ''}</textarea>

                  <div class="scene-field-row">
                    <div class="scene-field-half">
                      <label class="scene-field-label" id="energy-low-threshold-label-${idx}">Energy Low Threshold</label>
                      <input
                        type="number"
                        class="number-input"
                        data-action="update-energy-range"
                        data-index="${idx}"
                        data-range-index="0"
                        value="${scene.energyBlendRange[0]}"
                        min="0"
                        max="1"
                        step="0.05"
                        aria-labelledby="energy-low-threshold-label-${idx}"
                      />
                    </div>
                    <div class="scene-field-half">
                      <label class="scene-field-label" id="energy-high-threshold-label-${idx}">Energy High Threshold</label>
                      <input
                        type="number"
                        class="number-input"
                        data-action="update-energy-range"
                        data-index="${idx}"
                        data-range-index="1"
                        value="${scene.energyBlendRange[1]}"
                        min="0"
                        max="1"
                        step="0.05"
                        aria-labelledby="energy-high-threshold-label-${idx}"
                      />
                    </div>
                  </div>
                </div>
              ` : ''}
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }

  protected actions = {
    'stop-propagation': (e: Event) => {
      e.stopPropagation();
    },

    'close': () => {
      this.hide();
    },

    'back': () => {
      this.state.view = 'picker';
    },

    'select-preset': (_e: Event, target: HTMLElement) => {
      const presetName = target.dataset.preset;
      if (presetName === '') {
        this.state.story = cloneStory(BLANK_STORY);
      } else if (presetName && STORY_PRESETS[presetName]) {
        this.state.story = cloneStory(STORY_PRESETS[presetName]);
      }
      this.state.currentSavedId = null;
      this.state.expandedScenes = new Set([0]);
      this.state.expandedAudioSections = new Set();
      this.state.view = 'editor';
    },

    'load-saved': (_e: Event, target: HTMLElement) => {
      const id = target.dataset.id;
      const saved = this.state.savedStories.find(s => s.id === id);
      if (saved) {
        this.state.story = cloneStory(saved.story);
        this.state.currentSavedId = saved.id;
        this.state.expandedScenes = new Set([0]);
        this.state.expandedAudioSections = new Set();
        this.state.view = 'editor';
      }
    },

    'delete-saved': async (e: Event, target: HTMLElement) => {
      e.stopPropagation();
      const id = target.dataset.id;
      if (!id || !confirm('Delete this saved story?')) return;

      try {
        await deleteStory(id);
        const updated = await getAllStories();
        this.state.savedStories = updated;
        if (this.state.currentSavedId === id) {
          this.state.currentSavedId = null;
        }
      } catch (err) {
        console.error('Failed to delete story:', err);
      }
    },

    'toggle-scene': (_e: Event, target: HTMLElement) => {
      const idx = parseInt(target.dataset.index || '0', 10);
      const expanded = new Set(this.state.expandedScenes);
      if (expanded.has(idx)) {
        expanded.delete(idx);
      } else {
        expanded.add(idx);
      }
      this.state.expandedScenes = expanded;
    },

    'toggle-audio-section': (_e: Event, target: HTMLElement) => {
      const idx = parseInt(target.dataset.index || '0', 10);
      const expanded = new Set(this.state.expandedAudioSections);
      if (expanded.has(idx)) {
        expanded.delete(idx);
      } else {
        expanded.add(idx);
      }
      this.state.expandedAudioSections = expanded;
    },

    'add-scene': () => {
      const newScene = createNewScene(this.state.story.scenes.length + 1);
      const story = { ...this.state.story };
      story.scenes = [...story.scenes, newScene];
      this.state.story = story;
      this.state.expandedScenes = new Set([...this.state.expandedScenes, story.scenes.length - 1]);
    },

    'remove-scene': (e: Event, target: HTMLElement) => {
      e.stopPropagation();
      const idx = parseInt(target.dataset.index || '0', 10);
      if (this.state.story.scenes.length <= 1) return;

      const story = { ...this.state.story };
      story.scenes = story.scenes.filter((_, i) => i !== idx);
      this.state.story = story;

      const expanded = new Set(this.state.expandedScenes);
      expanded.delete(idx);
      this.state.expandedScenes = expanded;
    },

    'update-story': (e: Event, target: HTMLElement) => {
      const field = target.dataset.field as keyof StoryConfig;
      const value = (target as HTMLInputElement | HTMLTextAreaElement).value;
      const story = { ...this.state.story, [field]: value || undefined };
      this.state.story = story;
    },

    'update-story-checkbox': (e: Event, target: HTMLElement) => {
      const field = target.dataset.field as keyof StoryConfig;
      const checked = (target as HTMLInputElement).checked;
      const story = { ...this.state.story, [field]: checked };
      this.state.story = story;
    },

    'update-scene': (e: Event, target: HTMLElement) => {
      const idx = parseInt(target.dataset.index || '0', 10);
      const field = target.dataset.field as keyof SceneDefinition;
      const dataType = target.dataset.type;
      let value: string | number = (target as HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement).value;

      if (dataType === 'int') {
        value = parseInt(value, 10) || 0;
      } else if (dataType === 'float') {
        value = parseFloat(value) || 0;
      }

      const story = { ...this.state.story };
      story.scenes = story.scenes.map((s, i) =>
        i === idx ? { ...s, [field]: value || undefined } : s
      );
      this.state.story = story;
    },

    'update-energy-range': (e: Event, target: HTMLElement) => {
      const idx = parseInt(target.dataset.index || '0', 10);
      const rangeIdx = parseInt(target.dataset.rangeIndex || '0', 10);
      const value = parseFloat((target as HTMLInputElement).value) || 0;

      const story = { ...this.state.story };
      const scene = story.scenes[idx];
      const newRange: [number, number] = [...scene.energyBlendRange] as [number, number];
      newRange[rangeIdx] = value;

      story.scenes = story.scenes.map((s, i) =>
        i === idx ? { ...s, energyBlendRange: newRange } : s
      );
      this.state.story = story;
    },

    'save': async () => {
      if (!this.state.story.name.trim()) {
        alert('Please enter a story name');
        return;
      }

      this.state.isSaving = true;
      try {
        const saved = await saveStory(this.state.story, this.state.currentSavedId || undefined);
        this.state.currentSavedId = saved.id;
        const updated = await getAllStories();
        this.state.savedStories = updated;
        console.log(`Saved story: ${saved.name} (${saved.id})`);
      } catch (err) {
        console.error('Failed to save story:', err);
        alert('Failed to save story');
      } finally {
        this.state.isSaving = false;
      }
    },

    'submit': () => {
      this.callbacks.onLoadStory(this.state.story);
      this.hide();
    },
  };

  protected onDispose(): void {
    // No additional cleanup needed
  }
}
