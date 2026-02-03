/**
 * App - Main application orchestrator (vanilla TypeScript).
 * Coordinates controllers and UI components.
 */

import { eventBus } from './core/EventBus';
import { AudioController } from './controllers/AudioController';
import { VisualizerController } from './controllers/VisualizerController';
import { AIGeneratorController } from './controllers/AIGeneratorController';
import { StoryController } from './controllers/StoryController';
import {
  Timeline,
  AIGeneratorPanel,
  FrequencyDebugOverlay,
} from './components';
import { getDefaultMappingConfig, MappingConfig } from './components/MappingPanel';
import { getCurrentState } from './ui/data/settingsStorage';
import { getAccessibilitySettings, setupReducedMotionListener } from './utils/accessibility';
import { SUPPORTED_EXTENSIONS, AudioMetrics, BeatDebugInfo } from './audio/types';
import { AIGeneratedMode } from './visualizer/modes/AIGeneratedMode';
import { AIGeneratedModeWebGPU } from './visualizer/modes/AIGeneratedModeWebGPU';

import './styles.css';

export class App {
  // Controllers
  private audioController: AudioController;
  private visualizerController: VisualizerController;
  private aiController: AIGeneratorController;
  private storyController: StoryController;

  // UI Components
  private timeline: Timeline | null = null;
  private aiPanel: AIGeneratorPanel | null = null;
  private frequencyOverlay: FrequencyDebugOverlay | null = null;

  // State
  private mappingConfig: MappingConfig;
  private showAIPanel = true;
  private showFrequencyDebug = false;
  private isImmersiveMode = false;
  private showAIPanelBeforeImmersive = true;
  private isDragging = false;
  private rafId: number | null = null;

  // DOM elements
  private rootEl: HTMLElement;
  private visualizerContainer: HTMLElement | null = null;
  private canvas: HTMLCanvasElement | null = null;

  constructor(rootElement: HTMLElement) {
    this.rootEl = rootElement;
    this.mappingConfig = getDefaultMappingConfig();

    // Initialize controllers
    this.audioController = new AudioController();
    this.visualizerController = new VisualizerController();
    this.aiController = new AIGeneratorController();
    this.storyController = new StoryController(this.aiController);

    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    // Immersive mode keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === 'i' || e.key === 'I') {
        e.preventDefault();
        this.toggleImmersiveMode();
      }
      if (e.key === 'Escape' && this.isImmersiveMode) {
        e.preventDefault();
        this.toggleImmersiveMode();
      }
    });

    // Forward audio metrics to AI generator
    eventBus.on<AudioMetrics>('audio:metrics', (metrics) => {
      if (this.aiController.state.isGenerating && metrics) {
        this.aiController.sendMetrics(metrics);
      }
      // Update frequency overlay
      if (this.frequencyOverlay) {
        this.frequencyOverlay.setMetrics(metrics ?? null);
      }
      // Forward metrics to AI panel for mapping controls
      this.aiPanel?.updateMetrics(metrics ?? null);
    });

    // Forward beat debug info
    eventBus.on<BeatDebugInfo | null>('audio:beatDebug', (info) => {
      if (this.frequencyOverlay) {
        this.frequencyOverlay.setBeatDebugInfo(info ?? null);
      }
    });

    // Handle AI frames
    eventBus.on<{ frame: HTMLImageElement; frameId: number }>('ai:frame', ({ frame, frameId }) => {
      const mode = this.visualizerController.getCurrentMode();
      // DEBUG: trace frame reception
      if (frameId % 30 === 0) {
        console.log(`[AI Frame] frameId=${frameId}, mode=${mode?.constructor.name}`);
      }
      // Support both WebGL2 and WebGPU modes
      if (mode && mode instanceof AIGeneratedMode) {
        mode.setFrame(frame, frameId);
      } else if (mode && mode instanceof AIGeneratedModeWebGPU) {
        mode.setFrame(frame, frameId);
      }
    });

    // Handle story messages
    eventBus.on<unknown>('ai:storyMessage', (message) => {
      this.storyController.handleStoryMessage(message);
    });

    // Audio capture for lyrics
    eventBus.on('ai:configChanged', () => {
      this.updateAudioCapture();
    });

    eventBus.on('ai:stateChanged', () => {
      this.updateAudioCapture();
      this.updateFrequencyOverlayAIInfo();
      // Update timeline FPS (file mode)
      const state = this.aiController.state;
      this.timeline?.setFps(state.isConnected ? state.fps : undefined);
      // Update frame count in timeline
      this.timeline?.setFrameCount(state.isGenerating ? state.frameId : undefined);
      // Update stream info (stream mode)
      this.updateStreamInfo();
      // Update refresh header button visibility and label
      this.updateRefreshHeaderButton();
    });

    // Story state changes - update the panel
    eventBus.on('story:stateChanged', () => {
      // The panel listens internally via storyController events
    });

    // Setup reduced motion listener
    setupReducedMotionListener((reducedMotion) => {
      this.visualizerController.setAccessibility({
        ...this.visualizerController.config.accessibility,
        reducedMotion,
        maxFlashRate: reducedMotion ? 1 : this.visualizerController.config.accessibility.maxFlashRate,
      });
    });
  }

  private updateAudioCapture(): void {
    const shouldCapture =
      this.aiController.config.enableLyrics &&
      this.aiController.state.isConnected &&
      this.audioController.playbackState === 'playing';

    if (shouldCapture) {
      console.log('Starting audio capture for lyric detection');
      this.audioController.startAudioCapture((audioData, sampleRate) => {
        this.aiController.sendAudioChunk(audioData, sampleRate);
      });
    } else {
      this.audioController.stopAudioCapture();
    }
  }

  private updateFrequencyOverlayAIInfo(): void {
    if (this.frequencyOverlay) {
      this.frequencyOverlay.setAIInfo(
        this.aiController.state.isGenerating ? this.aiController.config : null,
        this.aiController.state.isGenerating ? this.aiController.state : null
      );
    }
  }

  private updateRefreshHeaderButton(): void {
    const refreshBtn = document.getElementById('refresh-header-btn');
    if (!refreshBtn) return;

    const { isConnected, isGenerating, isInitializing } = this.aiController.state;
    const { baseImage } = this.aiController.config;

    // Show only when connected and not initializing
    const shouldShow = isConnected && !isInitializing;
    refreshBtn.style.display = shouldShow ? '' : 'none';

    if (shouldShow) {
      // Update label based on whether base image is set
      const label = baseImage ? 'To Base' : 'New Seed';
      const title = baseImage
        ? 'Reset to uploaded base image'
        : 'Generate from new random seed';
      refreshBtn.textContent = label;
      refreshBtn.title = title;
    }
  }

  async mount(): Promise<void> {
    // Restore last session state from IndexedDB or localStorage backup
    try {
      // First check localStorage backup (saved on page unload)
      const backup = localStorage.getItem('ease-settings-backup');
      let restoredSettings = null;

      if (backup) {
        try {
          restoredSettings = JSON.parse(backup);
          console.log('Restoring settings from localStorage backup');
          // Clear the backup after reading
          localStorage.removeItem('ease-settings-backup');
        } catch (e) {
          console.error('Failed to parse settings backup:', e);
        }
      }

      // Fall back to IndexedDB if no localStorage backup
      if (!restoredSettings) {
        restoredSettings = await getCurrentState();
      }

      if (restoredSettings) {
        if (restoredSettings.aiConfig) {
          this.aiController.updateConfig(restoredSettings.aiConfig);
        }
        if (restoredSettings.mappingConfig) {
          this.mappingConfig = restoredSettings.mappingConfig;
        }
      }
    } catch (error) {
      console.error('Failed to restore session state:', error);
    }

    this.render();
    this.setupComponents();
    this.startRenderLoop();
  }

  private render(): void {
    const { fileInfo, audioSource } = this.audioController;

    this.rootEl.innerHTML = `
      <div class="app ${this.isImmersiveMode ? 'immersive-mode' : ''}">
        <!-- Main visualization area -->
        <div class="visualizer-container ${this.isDragging ? 'dragging' : ''}" id="visualizer-container">
          <canvas class="visualizer-canvas" id="main-canvas"></canvas>

          <!-- Lyric subtitle overlay -->
          <div class="lyric-subtitle-overlay" id="lyric-overlay" style="display: none;">
            <div class="lyric-subtitle-text" id="lyric-text"></div>
          </div>

          <!-- Frequency debug overlay container -->
          <div id="frequency-overlay-container"></div>

          <!-- Onboarding overlay -->
          ${!fileInfo ? this.renderDropZone() : ''}

          <!-- Drag overlay -->
          ${this.isDragging ? `
            <div class="drag-overlay">
              <div class="drag-overlay-content">Drop audio file here</div>
            </div>
          ` : ''}
        </div>

        <!-- Header -->
        <header class="header">
          <div class="header-left">
            <h1 class="logo" id="home-btn" style="cursor: pointer;" title="Return to home">EASE</h1>
            <button class="backend-badge" id="backend-btn" title="Backend: auto (using ${this.visualizerController.activeBackend}).${this.visualizerController.activeBackend === 'webgpu' ? ' WebGPU is experimental.' : ''} Click to switch.">
              ${this.visualizerController.activeBackend.toUpperCase()}
              ${this.visualizerController.backend === 'auto' ? '<span class="auto-indicator">*</span>' : ''}
            </button>
          </div>

          <div class="header-center">
            ${fileInfo ? `
              <span class="file-name">${fileInfo.name}</span>
              ${audioSource !== 'file' ? `
                <button class="stop-capture-btn" id="stop-capture-btn" title="Stop capture">Stop</button>
              ` : ''}
            ` : ''}
          </div>

          <div class="header-right">
            <button class="header-btn immersive-btn" id="immersive-btn" title="${this.isImmersiveMode ? 'Exit immersive mode (Esc)' : 'Enter immersive mode (I)'}">${this.isImmersiveMode ? 'Exit' : 'Immersive'}</button>
            <button class="header-btn ${this.showAIPanel ? 'active' : ''}" id="ai-panel-btn">Settings</button>
            <button class="header-btn refresh-header-btn" id="refresh-header-btn" style="display: none;" title="Generate from new random seed">New Seed</button>
          </div>
        </header>

        <!-- Side panel -->
        <aside class="side-panel" id="side-panel" ${!this.showAIPanel ? 'style="display: none;"' : ''}>
          <div id="ai-panel-container"></div>
        </aside>

        <!-- Footer -->
        <footer class="footer">
          <div id="timeline-container"></div>
          <!-- System Stats -->
          <div class="system-stats" id="system-stats" style="display: none;"></div>
        </footer>

        <!-- Immersive restore zone (only rendered in immersive mode) -->
        ${this.isImmersiveMode ? `
          <div class="immersive-restore-zone" id="immersive-restore-zone">
            <button class="immersive-exit-btn" id="immersive-exit-btn">Exit Immersive</button>
          </div>
        ` : ''}

      </div>
    `;

    // Store references to key elements
    this.visualizerContainer = document.getElementById('visualizer-container');
    this.canvas = document.getElementById('main-canvas') as HTMLCanvasElement;
  }

  private renderDropZone(): string {
    return `
      <div class="drop-zone">
        <div class="drop-zone-content">
          <h2>EASE</h2>
          <p class="tagline">Effortless Audio-Synesthesia Experience</p>
          <div class="drop-zone-box">
            <input type="file" accept="${SUPPORTED_EXTENSIONS.join(',')}" class="file-input" id="file-input" />
            <div class="drop-zone-box-content">
              <span class="drop-icon">+</span>
              <p>Drop audio file or click to browse</p>
              <p class="supported-formats">${SUPPORTED_EXTENSIONS.join(', ')}</p>
            </div>
          </div>
          <div class="capture-buttons">
            <button class="capture-btn" id="capture-tab-btn" title="Capture audio from a browser tab (e.g., YouTube, Spotify)">
              Capture Tab Audio
            </button>
            <button class="capture-btn" id="capture-mic-btn" title="Use microphone input">
              Use Microphone
            </button>
          </div>
        </div>
      </div>
    `;
  }

  private setupComponents(): void {
    // Initialize visualizer with canvas
    if (this.canvas) {
      this.visualizerController.initialize(this.canvas, getAccessibilitySettings());
    }

    // Setup drag and drop
    if (this.visualizerContainer) {
      this.visualizerContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!this.isDragging) {
          this.isDragging = true;
          this.updateVisualizerContainer();
        }
      });

      this.visualizerContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        this.isDragging = false;
        this.updateVisualizerContainer();
      });

      this.visualizerContainer.addEventListener('drop', async (e) => {
        e.preventDefault();
        this.isDragging = false;
        this.updateVisualizerContainer();

        const file = e.dataTransfer?.files[0];
        if (file) {
          await this.audioController.loadFile(file);
          this.render();
          this.setupComponents();
        }
      });
    }

    // Setup file input
    const fileInput = document.getElementById('file-input') as HTMLInputElement | null;
    if (fileInput) {
      fileInput.addEventListener('change', async (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
          await this.audioController.loadFile(file);
          this.render();
          this.setupComponents();
        }
      });
    }

    // Setup capture buttons
    const captureTabBtn = document.getElementById('capture-tab-btn');
    if (captureTabBtn) {
      captureTabBtn.addEventListener('click', async () => {
        await this.audioController.captureSystemAudio();
        this.render();
        this.setupComponents();
      });
    }

    const captureMicBtn = document.getElementById('capture-mic-btn');
    if (captureMicBtn) {
      captureMicBtn.addEventListener('click', async () => {
        await this.audioController.captureMicrophone();
        this.render();
        this.setupComponents();
      });
    }

    // Setup header buttons
    const homeBtn = document.getElementById('home-btn');
    homeBtn?.addEventListener('click', () => this.goHome());

    const backendBtn = document.getElementById('backend-btn');
    if (backendBtn) {
      backendBtn.addEventListener('click', () => {
        const current = this.visualizerController.backend;
        const next = current === 'auto' ? 'webgpu' : current === 'webgpu' ? 'webgl2' : 'auto';
        this.visualizerController.setBackend(next);
        // Note: updateBackendButton() is called via the visualizer:backendChanged event
        // when the backend actually changes (async), not synchronously here
      });
    }

    // Update backend button when backend actually changes (async)
    eventBus.on('visualizer:backendChanged', () => {
      this.updateBackendButton();
    });

    const aiPanelBtn = document.getElementById('ai-panel-btn');
    if (aiPanelBtn) {
      aiPanelBtn.addEventListener('click', () => {
        this.showAIPanel = !this.showAIPanel;
        aiPanelBtn.classList.toggle('active', this.showAIPanel);
        const sidePanel = document.getElementById('side-panel');
        if (sidePanel) {
          sidePanel.style.display = this.showAIPanel ? '' : 'none';
        }
      });
    }

    // Immersive mode toggle
    const immersiveBtn = document.getElementById('immersive-btn');
    if (immersiveBtn) {
      immersiveBtn.addEventListener('click', () => this.toggleImmersiveMode());
    }

    const immersiveExitBtn = document.getElementById('immersive-exit-btn');
    if (immersiveExitBtn) {
      immersiveExitBtn.addEventListener('click', () => this.toggleImmersiveMode());
    }

    // Setup refresh header button
    const refreshHeaderBtn = document.getElementById('refresh-header-btn');
    if (refreshHeaderBtn) {
      refreshHeaderBtn.addEventListener('click', () => {
        this.aiController.resetFeedback();
      });
    }
    this.updateRefreshHeaderButton();

    const stopCaptureBtn = document.getElementById('stop-capture-btn');
    if (stopCaptureBtn) {
      stopCaptureBtn.addEventListener('click', () => {
        this.audioController.stopCapture();
        this.render();
        this.setupComponents();
      });
    }

    // Setup AI Panel
    const aiPanelContainer = document.getElementById('ai-panel-container');
    if (aiPanelContainer) {
      this.aiPanel?.dispose();
      this.aiPanel = new AIGeneratorPanel(
        aiPanelContainer,
        {
          onConnect: () => this.aiController.connect(),
          onDisconnect: () => this.aiController.disconnect(),
          onStartGeneration: () => this.aiController.startGeneration(),
          onStopGeneration: () => this.aiController.stopGeneration(),
          onResetFeedback: () => this.aiController.resetFeedback(),
          onClearBaseImage: () => this.aiController.clearBaseImage(),
          onClearVram: () => this.aiController.clearVram(),
          onSwitchBackend: (backendId) => this.aiController.switchBackend(backendId),
          onConfigChange: (config) => this.aiController.updateConfig(config),
          onResetLyrics: () => this.aiController.resetLyrics(),
          onMappingConfigChange: (config) => {
            this.mappingConfig = config;
            if (this.aiController.state.isConnected) {
              this.aiController.sendMappingConfig(config);
            }
          },
        },
        this.aiController.state,
        this.aiController.config,
        this.mappingConfig,
        this.storyController
      );
      this.aiPanel.mount();

      // Listen for state/config changes to update panel
      this.aiController.on('stateChange', (state) => {
        if (this.aiPanel) {
          this.aiPanel.updateState(state);
        }
      });
      this.aiController.on('configChange', (config) => {
        if (this.aiPanel) {
          this.aiPanel.updateConfig(config);
        }
      });
    }

    // Setup Timeline or stream info
    const timelineContainer = document.getElementById('timeline-container');
    if (timelineContainer) {
      this.timeline?.dispose();

      if (this.audioController.fileInfo) {
        if (this.audioController.audioSource === 'file') {
          this.timeline = new Timeline(
            timelineContainer,
            {
              onSeek: (time) => this.audioController.seek(time),
              onPlay: () => this.audioController.play(),
              onPause: () => this.audioController.pause(),
              onStop: () => this.audioController.stop(),
              onInfoToggle: (show) => {
                this.showFrequencyDebug = show;
                this.visualizerController.setShowFrequencyDebug(show);
                this.updateFrequencyOverlay();
              },
            },
            {
              currentTime: this.audioController.currentTime,
              duration: this.audioController.duration,
              playbackState: this.audioController.playbackState,
              bpm: this.audioController.metrics?.bpm || 0,
              fps: this.aiController.state.isConnected ? this.aiController.state.fps : undefined,
              showInfo: this.showFrequencyDebug,
            }
          );
          this.timeline.mount();

          // Listen for audio state changes
          this.audioController.on('timeUpdate', ({ currentTime, duration }) => {
            this.timeline?.setTime(currentTime, duration);
          });
          this.audioController.on('playbackStateChange', ({ state }) => {
            this.timeline?.setPlaybackState(state);
          });
          this.audioController.on('metricsUpdate', (metrics) => {
            this.timeline?.setBpm(metrics?.bpm || 0);
          });
        } else {
          // Stream info mode
          timelineContainer.innerHTML = this.renderStreamInfo();
          this.setupStreamInfoListeners();
        }
      } else {
        // Minimal status bar
        timelineContainer.innerHTML = this.renderMinimalStatusBar();
        this.setupMinimalStatusListeners();
      }
    }

    // Setup Frequency Debug Overlay
    const freqOverlayContainer = document.getElementById('frequency-overlay-container');
    if (freqOverlayContainer) {
      this.frequencyOverlay?.dispose();
      if (this.showFrequencyDebug) {
        this.frequencyOverlay = new FrequencyDebugOverlay(freqOverlayContainer);
        this.frequencyOverlay.mount();
        this.updateFrequencyOverlayAIInfo();
      }
    }

    // Setup lyric overlay
    this.updateLyricOverlay();

    // Setup system stats updates
    this.setupSystemStatsUpdates();

    // Sync AI config to visualizer mode
    this.syncAIConfigToMode();
  }

  private renderStreamInfo(): string {
    const { audioSource } = this.audioController;
    const { metrics } = this.audioController;
    const { state } = this.aiController;

    return `
      <div class="stream-info">
        <span class="stream-indicator">
          <span class="stream-dot"></span>
          ${audioSource === 'system' ? 'Capturing Tab Audio' : 'Microphone Active'}
        </span>
        <div class="stream-metrics">
          <span class="bpm-display">${metrics?.bpm || 0} BPM</span>
          ${state.isConnected && state.fps > 0 ? `
            <span class="fps-display">${state.fps.toFixed(1)} FPS</span>
          ` : ''}
          <button class="info-toggle-btn ${this.showFrequencyDebug ? 'active' : ''}" id="info-toggle-btn" title="Toggle frequency debug overlay">
            Info
          </button>
        </div>
      </div>
    `;
  }

  private renderMinimalStatusBar(): string {
    return `
      <div class="status-bar-minimal">
        <button class="info-toggle-btn ${this.showFrequencyDebug ? 'active' : ''}" id="info-toggle-btn" title="Toggle frequency debug overlay">
          Info
        </button>
      </div>
    `;
  }

  private setupStreamInfoListeners(): void {
    const infoBtn = document.getElementById('info-toggle-btn');
    if (infoBtn) {
      infoBtn.addEventListener('click', () => {
        this.showFrequencyDebug = !this.showFrequencyDebug;
        this.visualizerController.setShowFrequencyDebug(this.showFrequencyDebug);
        this.updateFrequencyOverlay();
        infoBtn.classList.toggle('active', this.showFrequencyDebug);
      });
    }

    // Listen for metrics updates in stream mode
    this.audioController.on('metricsUpdate', () => {
      this.updateStreamInfo();
    });
  }

  /**
   * Surgically update stream info display (BPM/FPS) without full re-render
   */
  private updateStreamInfo(): void {
    // Only update in stream mode (not file mode)
    if (this.audioController.audioSource === 'file') return;

    const streamMetrics = document.querySelector('.stream-metrics');
    if (!streamMetrics) return;

    const { metrics } = this.audioController;
    const { state } = this.aiController;

    // Update BPM display
    const bpmEl = streamMetrics.querySelector('.bpm-display');
    if (bpmEl) {
      bpmEl.textContent = `${metrics?.bpm || 0} BPM`;
    }

    // Update FPS display (add/remove/update as needed)
    let fpsEl = streamMetrics.querySelector('.fps-display');
    if (state.isConnected && state.fps > 0) {
      if (fpsEl) {
        fpsEl.textContent = `${state.fps.toFixed(1)} FPS`;
      } else {
        // Insert FPS element before info button
        const infoBtn = streamMetrics.querySelector('.info-toggle-btn');
        fpsEl = document.createElement('span');
        fpsEl.className = 'fps-display';
        fpsEl.textContent = `${state.fps.toFixed(1)} FPS`;
        if (infoBtn) {
          streamMetrics.insertBefore(fpsEl, infoBtn);
        } else {
          streamMetrics.appendChild(fpsEl);
        }
      }
    } else if (fpsEl) {
      fpsEl.remove();
    }
  }

  private setupMinimalStatusListeners(): void {
    const infoBtn = document.getElementById('info-toggle-btn');
    if (infoBtn) {
      infoBtn.addEventListener('click', () => {
        this.showFrequencyDebug = !this.showFrequencyDebug;
        this.visualizerController.setShowFrequencyDebug(this.showFrequencyDebug);
        this.updateFrequencyOverlay();
        infoBtn.classList.toggle('active', this.showFrequencyDebug);
      });
    }
  }

  private updateVisualizerContainer(): void {
    if (this.visualizerContainer) {
      this.visualizerContainer.classList.toggle('dragging', this.isDragging);

      // Update drag overlay
      const dragOverlay = this.visualizerContainer.querySelector('.drag-overlay');
      if (this.isDragging && !dragOverlay) {
        const overlay = document.createElement('div');
        overlay.className = 'drag-overlay';
        overlay.innerHTML = '<div class="drag-overlay-content">Drop audio file here</div>';
        this.visualizerContainer.appendChild(overlay);
      } else if (!this.isDragging && dragOverlay) {
        dragOverlay.remove();
      }
    }
  }

  private updateBackendButton(): void {
    const btn = document.getElementById('backend-btn');
    if (btn) {
      btn.innerHTML = `
        ${this.visualizerController.activeBackend.toUpperCase()}
        ${this.visualizerController.backend === 'auto' ? '<span class="auto-indicator">*</span>' : ''}
      `;
      btn.title = `Backend: ${this.visualizerController.backend} (using ${this.visualizerController.activeBackend}).${
        this.visualizerController.activeBackend === 'webgpu' ? ' WebGPU is experimental.' : ''
      } Click to switch.`;
    }
  }

  private toggleImmersiveMode(): void {
    this.isImmersiveMode = !this.isImmersiveMode;

    const app = this.rootEl.querySelector('.app');
    if (app) {
      app.classList.toggle('immersive-mode', this.isImmersiveMode);
    }

    // Handle side panel: hide when entering immersive, restore when exiting
    const sidePanel = document.getElementById('side-panel');
    const aiPanelBtn = document.getElementById('ai-panel-btn');

    if (this.isImmersiveMode) {
      // Save current state and hide panel
      this.showAIPanelBeforeImmersive = this.showAIPanel;
      if (this.showAIPanel && sidePanel) {
        this.showAIPanel = false;
        sidePanel.style.display = 'none';
        aiPanelBtn?.classList.remove('active');
      }

      // Add restore zone
      if (app && !document.getElementById('immersive-restore-zone')) {
        const restoreZone = document.createElement('div');
        restoreZone.className = 'immersive-restore-zone';
        restoreZone.id = 'immersive-restore-zone';
        restoreZone.innerHTML = '<button class="immersive-exit-btn" id="immersive-exit-btn">Exit Immersive</button>';
        app.appendChild(restoreZone);

        // Add click handler
        const exitBtn = document.getElementById('immersive-exit-btn');
        exitBtn?.addEventListener('click', () => this.toggleImmersiveMode());
      }
    } else {
      // Restore panel state
      if (this.showAIPanelBeforeImmersive && sidePanel) {
        this.showAIPanel = true;
        sidePanel.style.display = '';
        aiPanelBtn?.classList.add('active');
      }

      // Remove restore zone
      const restoreZone = document.getElementById('immersive-restore-zone');
      restoreZone?.remove();
    }

    const btn = document.getElementById('immersive-btn');
    if (btn) {
      btn.textContent = this.isImmersiveMode ? 'Exit' : 'Immersive';
      btn.title = this.isImmersiveMode ? 'Exit immersive mode (Esc)' : 'Enter immersive mode (I)';
    }
  }

  private goHome(): void {
    // Stop any active generation
    if (this.aiController.state.isGenerating) {
      this.aiController.stopGeneration();
    }

    // Clear audio (stopCapture resets fileInfo to null and state to idle)
    this.audioController.stopCapture();

    // Re-render to show drop zone
    this.render();
    this.setupComponents();
  }

  private updateFrequencyOverlay(): void {
    const container = document.getElementById('frequency-overlay-container');
    if (!container) return;

    if (this.showFrequencyDebug && !this.frequencyOverlay) {
      this.frequencyOverlay = new FrequencyDebugOverlay(container);
      this.frequencyOverlay.mount();
      this.updateFrequencyOverlayAIInfo();
    } else if (!this.showFrequencyDebug && this.frequencyOverlay) {
      this.frequencyOverlay.dispose();
      this.frequencyOverlay = null;
    }
  }

  private updateLyricOverlay(): void {
    const overlay = document.getElementById('lyric-overlay');
    const textEl = document.getElementById('lyric-text');

    if (!overlay || !textEl) return;

    const { config, state } = this.aiController;
    const shouldShow = config.enableLyrics && config.showLyricSubtitles && state.lyrics;

    overlay.style.display = shouldShow ? '' : 'none';
    if (shouldShow && state.lyrics) {
      textEl.textContent = state.lyrics.text.slice(-150);
    }
  }

  private setupSystemStatsUpdates(): void {
    // Poll for state changes to update system stats
    const statsEl = document.getElementById('system-stats');
    if (!statsEl) return;

    // Update via eventBus
    eventBus.on('ai:stateChanged', () => {
      const stats = this.aiController.state.systemStats;
      if (stats) {
        statsEl.style.display = '';
        statsEl.innerHTML = `
          <span class="stat" title="CPU Usage">CPU ${stats.cpu_percent}%</span>
          <span class="stat" title="RAM Usage">RAM ${stats.ram_used_gb}/${stats.ram_total_gb}GB</span>
          ${stats.vram_used_gb !== null ? `
            <span class="stat" title="VRAM Usage">VRAM ${stats.vram_used_gb}/${stats.vram_total_gb}GB</span>
          ` : ''}
          ${stats.gpu_util !== null ? `
            <span class="stat" title="GPU Utilization">GPU ${stats.gpu_util}%</span>
          ` : ''}
        `;
      } else {
        statsEl.style.display = 'none';
      }

      // Also update lyric overlay
      this.updateLyricOverlay();
    });
  }

  private syncAIConfigToMode(): void {
    // Helper to sync config to mode (supports both WebGL2 and WebGPU modes)
    const syncToMode = () => {
      const mode = this.visualizerController.getCurrentMode();
      const config = this.aiController.config;

      // Support both AIGeneratedMode (WebGL2) and AIGeneratedModeWebGPU (WebGPU)
      if (mode instanceof AIGeneratedMode) {
        mode.setShaderEffects(config.enableShaderEffects);
        mode.setFlashEnabled(config.enableFlash);
        mode.setSpectralDisplacementEnabled(config.enableSpectralDisplacement);
        mode.setGlitchBlocksEnabled(config.enableGlitchBlocks);
        mode.setTrebleGrainEnabled(config.enableTrebleGrain);
        mode.setBicubicEnabled(config.enableBicubic);
        mode.setSharpeningEnabled(config.enableSharpening);
        mode.setSharpenStrength(config.sharpenStrength);
        mode.setTextureSize(config.width, config.height);
        mode.setMaintainAspectRatio(config.maintainAspectRatio);
        mode.setSilenceDegradationEnabled(config.enableSilenceDegradation);
        mode.setSilenceThreshold(config.silenceThreshold);
        mode.setDegradationRate(config.degradationRate);
        mode.setRecoveryRate(config.recoveryRate);
      } else if (mode instanceof AIGeneratedModeWebGPU) {
        mode.setShaderEffects(config.enableShaderEffects);
        mode.setFlashEnabled(config.enableFlash);
        mode.setSpectralDisplacementEnabled(config.enableSpectralDisplacement);
        mode.setGlitchBlocksEnabled(config.enableGlitchBlocks);
        mode.setTrebleGrainEnabled(config.enableTrebleGrain);
        mode.setBicubicEnabled(config.enableBicubic);
        mode.setSharpeningEnabled(config.enableSharpening);
        mode.setSharpenStrength(config.sharpenStrength);
        mode.setTextureSize(config.width, config.height);
        mode.setMaintainAspectRatio(config.maintainAspectRatio);
        mode.setSilenceDegradationEnabled(config.enableSilenceDegradation);
        mode.setSilenceThreshold(config.silenceThreshold);
        mode.setDegradationRate(config.degradationRate);
        mode.setRecoveryRate(config.recoveryRate);
      }
    };

    // Try initial sync (may fail if mode not ready)
    syncToMode();

    // Listen for future config changes (ALWAYS set up, even if mode not ready yet)
    eventBus.on('ai:configChanged', syncToMode);

    // Also sync after engine initializes (catches the race condition where mode wasn't ready)
    eventBus.on('visualizer:backendChanged', syncToMode);
  }

  private startRenderLoop(): void {
    let frameCount = 0;
    const animate = () => {
      const metrics = this.audioController.metrics;
      if (metrics) {
        this.visualizerController.render(metrics);
      }
      // DEBUG: log every 300 frames (~5 sec at 60fps)
      frameCount++;
      if (frameCount % 300 === 0) {
        const mode = this.visualizerController.getCurrentMode();
        console.log(`[Render Loop] frame=${frameCount}, hasMetrics=${!!metrics}, mode=${mode?.constructor.name}, canvas=${this.visualizerController.getCanvas()?.width}x${this.visualizerController.getCanvas()?.height}`);
      }
      this.rafId = requestAnimationFrame(animate);
    };

    this.rafId = requestAnimationFrame(animate);
  }

  dispose(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
    }

    this.timeline?.dispose();
    this.aiPanel?.dispose();
    this.frequencyOverlay?.dispose();
    this.audioController.dispose();
    this.visualizerController.dispose();
    this.aiController.dispose();
    this.storyController.dispose();
  }
}
