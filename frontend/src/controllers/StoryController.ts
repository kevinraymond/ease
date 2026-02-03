/**
 * StoryController - Manages story loading, playback, and state.
 * Replaces the useStoryController React hook.
 */

import { EventBus } from '../core/EventBus';
import {
  StoryConfig,
  StoryState,
  StoryPresetName,
} from '../core/types';
import { AIGeneratorController } from './AIGeneratorController';

// Story preset names
const AVAILABLE_PRESETS: StoryPresetName[] = [
  'skiing_adventure',
  'dancing_figure',
  'abstract_landscape',
  'minimal_portrait',
];

export interface StoryControllerState {
  hasStory: boolean;
  state: StoryState | null;
  availablePresets: StoryPresetName[];
}

const DEFAULT_STATE: StoryControllerState = {
  hasStory: false,
  state: null,
  availablePresets: AVAILABLE_PRESETS,
};

export class StoryController {
  // State
  private _state: StoryControllerState = { ...DEFAULT_STATE };

  // AI Generator reference (for WebSocket)
  private aiController: AIGeneratorController;

  // Local event bus
  private localBus = new EventBus();

  constructor(aiController: AIGeneratorController) {
    this.aiController = aiController;

    // Listen for story messages from AI controller
    aiController.onStoryMessage((msg) => this.handleStoryMessage(msg));
  }

  // === Getters ===

  get storyState(): StoryControllerState {
    return this._state;
  }

  get hasStory(): boolean {
    return this._state.hasStory;
  }

  get state(): StoryState | null {
    return this._state.state;
  }

  get availablePresets(): StoryPresetName[] {
    return this._state.availablePresets;
  }

  // === Event subscription ===

  on(event: 'stateChange', callback: (data: StoryControllerState) => void): () => void {
    return this.localBus.on(event, callback);
  }

  // === Actions ===

  loadPreset(presetName: StoryPresetName): void {
    this.sendMessage({
      type: 'story_load_preset',
      preset_name: presetName,
    });
  }

  loadStory(story: StoryConfig): void {
    // Convert to snake_case for backend
    const msg = {
      type: 'story_load',
      story: {
        name: story.name,
        description: story.description,
        default_negative_prompt: story.defaultNegativePrompt,
        scenes: story.scenes.map((scene) => ({
          id: scene.id,
          base_prompt: scene.basePrompt,
          negative_prompt: scene.negativePrompt,
          duration_frames: scene.durationFrames,
          trigger: scene.trigger,
          trigger_value: scene.triggerValue,
          energy_high_prompt: scene.energyHighPrompt,
          energy_low_prompt: scene.energyLowPrompt,
          energy_blend_range: scene.energyBlendRange,
          beat_prompt_modifier: scene.beatPromptModifier,
          transition: scene.transition,
          transition_frames: scene.transitionFrames,
        })),
        loop: story.loop,
        audio_reactive_keywords: story.audioReactiveKeywords,
        base_seed: story.baseSeed,
      },
    };
    this.sendMessage(msg);
  }

  unloadStory(): void {
    this.sendMessage({ type: 'story_unload' });
    this.updateState(DEFAULT_STATE);
  }

  play(): void {
    this.sendMessage({ type: 'story_control', action: 'play' });
    // Optimistically update local state
    if (this._state.state) {
      this.updateState({
        ...this._state,
        state: { ...this._state.state, isPlaying: true },
      });
    }
  }

  pause(): void {
    this.sendMessage({ type: 'story_control', action: 'pause' });
    // Optimistically update local state
    if (this._state.state) {
      this.updateState({
        ...this._state,
        state: { ...this._state.state, isPlaying: false },
      });
    }
  }

  skipNext(): void {
    this.sendMessage({ type: 'story_control', action: 'skip_next' });
  }

  skipPrev(): void {
    this.sendMessage({ type: 'story_control', action: 'skip_prev' });
  }

  restart(): void {
    this.sendMessage({ type: 'story_control', action: 'restart' });
  }

  // === Internal methods ===

  handleStoryMessage(msg: any): void {
    if (msg.type === 'story_state') {
      this.updateState({
        ...this._state,
        hasStory: true,
        state: {
          storyName: msg.story_name,
          currentSceneIdx: msg.current_scene_idx,
          currentSceneId: msg.current_scene_id,
          frameInScene: msg.frame_in_scene,
          beatCountInScene: msg.beat_count_in_scene,
          isTransitioning: msg.is_transitioning,
          transitionProgress: msg.transition_progress,
          isPlaying: msg.is_playing,
          isComplete: msg.is_complete,
          totalScenes: msg.total_scenes,
          sceneProgress: msg.scene_progress,
        },
      });
    }
  }

  private sendMessage(msg: any): void {
    const ws = this.aiController.ws_ref;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }

  private updateState(newState: StoryControllerState): void {
    this._state = newState;
    this.localBus.emit('stateChange', this._state);
  }

  // === Cleanup ===

  dispose(): void {
    this.localBus.clear();
  }
}
