/**
 * Timeline - Transport controls, seek bar, and time display.
 */

import { Component } from '../core/Component';
import { html, icons } from '../core/html';
import { formatTime } from '../utils/mathUtils';
import { PlaybackState } from '../audio/AudioInputManager';

interface TimelineState {
  currentTime: number;
  duration: number;
  playbackState: PlaybackState;
  bpm: number;
  fps?: number;
  frameCount?: number;
  showInfo: boolean;
}

interface TimelineCallbacks {
  onSeek: (time: number) => void;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onInfoToggle?: (show: boolean) => void;
}

export class Timeline extends Component<TimelineState> {
  private callbacks: TimelineCallbacks;

  // Non-reactive time values (updated frequently without re-render)
  private _currentTime = 0;
  private _duration = 0;

  constructor(
    container: HTMLElement,
    callbacks: TimelineCallbacks,
    initialState?: Partial<TimelineState>
  ) {
    super(container, {
      currentTime: 0,
      duration: 0,
      playbackState: 'idle',
      bpm: 0,
      fps: undefined,
      frameCount: undefined,
      showInfo: false,
      ...initialState,
    });
    this.callbacks = callbacks;

    // Initialize non-reactive time values from initial state
    this._currentTime = initialState?.currentTime ?? 0;
    this._duration = initialState?.duration ?? 0;
  }

  protected render(): void {
    const { playbackState, bpm, fps, frameCount, showInfo } = this.state;
    // Use non-reactive time values for rendering
    const currentTime = this._currentTime;
    const duration = this._duration;
    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

    this.el.className = 'timeline';
    this.el.innerHTML = html`
      <div class="transport-controls">
        ${playbackState === 'playing'
          ? html`<button class="transport-button" data-action="pause" aria-label="Pause">
              ${icons.pause}
            </button>`
          : html`<button
              class="transport-button"
              data-action="play"
              ${duration === 0 ? 'disabled' : ''}
              aria-label="Play"
            >
              ${icons.play}
            </button>`
        }
        <button
          class="transport-button"
          data-action="stop"
          ${playbackState === 'idle' ? 'disabled' : ''}
          aria-label="Stop"
        >
          ${icons.stop}
        </button>
      </div>

      <div class="timeline-track" data-action="track-click">
        <div class="timeline-progress" style="width: ${progress}%"></div>
        <input
          type="range"
          min="0"
          max="100"
          value="${progress}"
          data-action="seek"
          class="timeline-slider"
          aria-label="Seek"
        />
      </div>

      <div class="timeline-info">
        <span class="time-display">
          ${formatTime(currentTime)} / ${formatTime(duration)}
        </span>
        ${bpm > 0 ? html`<span class="bpm-display">${bpm} BPM</span>` : ''}
        ${frameCount !== undefined ? html`<span class="frame-display">Frame ${frameCount}</span>` : ''}
        ${fps !== undefined && fps > 0 ? html`<span class="fps-display">${fps.toFixed(1)} FPS</span>` : ''}
        ${this.callbacks.onInfoToggle
          ? html`<button
              class="info-toggle-btn ${showInfo ? 'active' : ''}"
              data-action="toggle-info"
              title="Toggle frequency debug overlay"
            >
              Info
            </button>`
          : ''
        }
      </div>
    `;
  }

  protected onMount(): void {
    // Handle track click for seeking
    const track = this.$('.timeline-track');
    if (track) {
      this.listen(track, 'click', (e) => {
        // Ignore if clicking the slider itself
        if ((e.target as HTMLElement).classList.contains('timeline-slider')) return;

        const rect = track.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const percent = clickX / rect.width;
        this.callbacks.onSeek(percent * this._duration);
      });
    }
  }

  protected actions = {
    play: () => {
      this.callbacks.onPlay();
    },
    pause: () => {
      this.callbacks.onPause();
    },
    stop: () => {
      this.callbacks.onStop();
    },
    seek: (e: Event) => {
      const value = Number((e.target as HTMLInputElement).value);
      const time = (value / 100) * this._duration;
      this.callbacks.onSeek(time);
    },
    'toggle-info': () => {
      const newValue = !this.state.showInfo;
      this.state.showInfo = newValue;
      this.callbacks.onInfoToggle?.(newValue);
    },
  };

  // === Public update methods ===

  update(data: Partial<TimelineState>): void {
    Object.assign(this.state, data);
  }

  setTime(currentTime: number, duration: number): void {
    // Store in non-reactive properties (no re-render triggered)
    this._currentTime = currentTime;
    this._duration = duration;

    // Surgically update only the DOM elements that need to change
    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

    const progressEl = this.el.querySelector('.timeline-progress') as HTMLElement;
    if (progressEl) {
      progressEl.style.width = `${progress}%`;
    }

    const sliderEl = this.el.querySelector('.timeline-slider') as HTMLInputElement;
    if (sliderEl) {
      sliderEl.value = String(progress);
    }

    const timeEl = this.el.querySelector('.time-display');
    if (timeEl) {
      timeEl.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
    }
  }

  setPlaybackState(state: PlaybackState): void {
    this.state.playbackState = state;
  }

  setBpm(bpm: number): void {
    this.state.bpm = bpm;
  }

  setFps(fps: number | undefined): void {
    this.state.fps = fps;
  }

  setFrameCount(count: number | undefined): void {
    this.state.frameCount = count;
  }

  setShowInfo(show: boolean): void {
    this.state.showInfo = show;
  }
}
