/**
 * StoryControlPanel - Controls for loading and playing story sequences.
 */

import { Component } from '../core/Component';
import { html } from '../core/html';
import { StoryController } from '../controllers/StoryController';
import { StoryPresetName } from '../core/types';
import { StoryEditorModal } from './StoryEditorModal';
import { getAllStories, SavedStory } from '../ui/data/storyStorage';

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

interface StoryControlPanelState {
  savedStories: SavedStory[];
  showEditor: boolean;
}

export class StoryControlPanel extends Component<StoryControlPanelState> {
  private controller: StoryController;
  private disabled: boolean;
  private editorModal: StoryEditorModal | null = null;

  constructor(container: HTMLElement, controller: StoryController, disabled: boolean = false) {
    super(container, {
      savedStories: [],
      showEditor: false,
    });
    this.controller = controller;
    this.disabled = disabled;
  }

  protected async onMount(): Promise<void> {
    // Load saved stories
    try {
      const stories = await getAllStories();
      this.state.savedStories = stories;
    } catch (e) {
      console.error('Failed to load saved stories:', e);
    }

    // Listen for story state changes
    this.controller.on('stateChange', () => {
      this.forceRender();
    });

    // Create editor modal
    const modalContainer = document.createElement('div');
    document.body.appendChild(modalContainer);
    this.editorModal = new StoryEditorModal(modalContainer, {
      onClose: () => {
        this.state.showEditor = false;
        this.refreshSavedStories();
      },
      onLoadStory: (story) => {
        this.controller.loadStory(story);
      },
    }, this.controller.availablePresets);
    this.editorModal.mount();
  }

  private async refreshSavedStories(): Promise<void> {
    try {
      const stories = await getAllStories();
      this.state.savedStories = stories;
    } catch (e) {
      console.error('Failed to refresh saved stories:', e);
    }
  }

  protected render(): void {
    const storyState = this.controller.storyState;
    const hasStory = storyState.hasStory;
    const state = storyState.state;
    const { savedStories } = this.state;

    this.el.className = 'control-panel story-control-panel';
    this.el.innerHTML = html`
      <div class="control-section">
        <p class="control-description">Load a story to have prompts evolve over time, influenced by audio</p>
      </div>

      ${!hasStory ? html`
        <div class="control-section">
          <label class="control-label">Select a Story</label>
          <div class="story-selector-scroll">
            ${savedStories.length > 0 ? html`
              <div class="story-section-label">Your Stories</div>
              <div class="story-preset-grid">
                ${savedStories.map(saved => html`
                  <button
                    class="story-preset-btn user-story"
                    data-action="load-saved"
                    data-id="${saved.id}"
                    ${this.disabled ? 'disabled' : ''}
                    title="${saved.story.description || `${saved.story.scenes.length} scenes`}"
                  >
                    <span class="preset-label">${saved.name}</span>
                    <span class="preset-desc">${saved.story.description || `${saved.story.scenes.length} scene${saved.story.scenes.length !== 1 ? 's' : ''}`}</span>
                  </button>
                `)}
              </div>
            ` : ''}

            <div class="story-section-label">Presets</div>
            <div class="story-preset-grid">
              ${storyState.availablePresets.map(preset => html`
                <button
                  class="story-preset-btn"
                  data-action="load-preset"
                  data-preset="${preset}"
                  ${this.disabled ? 'disabled' : ''}
                  title="${PRESET_INFO[preset].description}"
                >
                  <span class="preset-label">${PRESET_INFO[preset].label}</span>
                  <span class="preset-desc">${PRESET_INFO[preset].description}</span>
                </button>
              `)}
            </div>
          </div>
          <button class="story-edit-btn" data-action="show-editor">Edit Story</button>
        </div>
      ` : ''}

      ${hasStory && state ? html`
        <div class="control-section story-info">
          <div class="story-header">
            <span class="story-name">${state.storyName.replace(/_/g, ' ')}</span>
            <button
              class="story-unload-btn"
              data-action="unload"
              ${this.disabled ? 'disabled' : ''}
              title="Unload story and return to static prompts"
            >
              Unload
            </button>
          </div>
        </div>

        <div class="control-section">
          <div class="scene-progress">
            <div class="scene-info">
              <span class="scene-label">Scene ${state.currentSceneIdx + 1}/${state.totalScenes}</span>
              <span class="scene-id">${state.currentSceneId}</span>
            </div>
            <div class="progress-bar">
              <div class="progress-fill" style="width: ${state.sceneProgress * 100}%"></div>
              ${state.isTransitioning ? html`
                <div
                  class="transition-marker"
                  style="left: ${(1 - state.transitionProgress) * 100}%; width: ${state.transitionProgress * 100}%"
                ></div>
              ` : ''}
            </div>
            ${state.isTransitioning ? html`<span class="transition-label">Transitioning...</span>` : ''}
            ${state.isComplete ? html`<span class="complete-label">Story Complete</span>` : ''}
          </div>
        </div>

        <div class="control-section">
          <div class="playback-controls">
            <button
              class="playback-btn skip-btn"
              data-action="skip-prev"
              ${this.disabled || state.currentSceneIdx === 0 ? 'disabled' : ''}
              title="Previous scene"
            >
              ⏮
            </button>
            <button
              class="playback-btn play-pause-btn ${state.isPlaying ? 'playing' : 'paused'}"
              data-action="play-pause"
              ${this.disabled ? 'disabled' : ''}
              title="${state.isPlaying ? 'Pause' : 'Play'}"
            >
              ${state.isPlaying ? '⏸' : '▶'}
            </button>
            <button
              class="playback-btn skip-btn"
              data-action="skip-next"
              ${this.disabled ? 'disabled' : ''}
              title="Next scene"
            >
              ⏭
            </button>
            <button
              class="playback-btn restart-btn"
              data-action="restart"
              ${this.disabled ? 'disabled' : ''}
              title="Restart from beginning"
            >
              ↻
            </button>
          </div>
        </div>

        <div class="control-section story-stats">
          <div class="stat">
            <span class="stat-label">Scene Frame</span>
            <span class="stat-value">${state.frameInScene}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Beats in Scene</span>
            <span class="stat-value">${state.beatCountInScene}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Status</span>
            <span class="stat-value status-${state.isPlaying ? 'playing' : 'paused'}">${state.isPlaying ? 'Playing' : 'Paused'}</span>
          </div>
        </div>
      ` : ''}
    `;
  }

  protected actions = {
    'load-preset': (e: Event, target: HTMLElement) => {
      const preset = target.dataset.preset as StoryPresetName;
      this.controller.loadPreset(preset);
    },
    'load-saved': (e: Event, target: HTMLElement) => {
      const id = target.dataset.id!;
      const saved = this.state.savedStories.find(s => s.id === id);
      if (saved) {
        this.controller.loadStory(saved.story);
      }
    },
    'show-editor': () => {
      this.editorModal?.show();
    },
    'unload': () => {
      this.controller.unloadStory();
    },
    'play-pause': () => {
      const state = this.controller.storyState.state;
      if (state?.isPlaying) {
        this.controller.pause();
      } else {
        this.controller.play();
      }
    },
    'skip-prev': () => {
      this.controller.skipPrev();
    },
    'skip-next': () => {
      this.controller.skipNext();
    },
    'restart': () => {
      this.controller.restart();
    },
  };

  setDisabled(disabled: boolean): void {
    this.disabled = disabled;
    this.forceRender();
  }

  protected onDispose(): void {
    this.editorModal?.dispose();
  }
}
