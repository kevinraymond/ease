# Architecture

EASE consists of two main components: a React frontend and a Python backend, communicating over WebSocket.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐ │
│  │ AudioInput   │───▶│ AudioAnalyzer │───▶│ VisualizerEngine     │ │
│  │ (File/Tab)   │    │ (FFT/Metrics) │    │ (7 modes + Three.js) │ │
│  └──────────────┘    └───────────────┘    └──────────────────────┘ │
│         │                    │                       │              │
│         │            ┌───────┴───────┐               │              │
│         │            ▼               ▼               │              │
│         │    ┌──────────────┐ ┌────────────┐        │              │
│         │    │ BeatDetector │ │ Onset      │        │              │
│         │    │ (BPM/Phase)  │ │ Detection  │        │              │
│         │    └──────────────┘ └────────────┘        │              │
│         │                                           │              │
│         └────────────────────┬──────────────────────┘              │
│                              │                                      │
│                              ▼                                      │
│                    ┌───────────────────┐                           │
│                    │ WebSocket Client  │                           │
│                    │ (Audio Metrics)   │                           │
│                    └─────────┬─────────┘                           │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               │ WebSocket (ws://localhost:8765)
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                              │                                      │
│                    ┌─────────▼─────────┐                           │
│                    │ WebSocket Handler │                           │
│                    └─────────┬─────────┘                           │
│                              │                                      │
│  ┌───────────────────────────┼───────────────────────────────────┐ │
│  │                           ▼                                    │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  │ │
│  │  │ Audio Mapper   │  │ Prompt         │  │ Story           │  │ │
│  │  │ (Metrics→Params│  │ Modulator      │  │ Controller      │  │ │
│  │  └───────┬────────┘  └───────┬────────┘  └────────┬────────┘  │ │
│  │          │                   │                    │           │ │
│  │          └───────────────────┴────────────────────┘           │ │
│  │                              │                                 │ │
│  │                              ▼                                 │ │
│  │                    ┌─────────────────┐                        │ │
│  │                    │ Generation      │                        │ │
│  │                    │ Pipeline        │                        │ │
│  │                    └────────┬────────┘                        │ │
│  │                             │                                  │ │
│  │  ┌──────────────────────────┼───────────────────────────────┐ │ │
│  │  │                          ▼                                │ │ │
│  │  │  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  │ │ │
│  │  │  │StreamDiffusion│  │ ControlNet   │  │ RIFE          │  │ │ │
│  │  │  │(Real-time SD) │  │ (Pose Guide) │  │ (Interpolate) │  │ │ │
│  │  │  └──────────────┘  └───────────────┘  └───────────────┘  │ │ │
│  │  │                          │                                │ │ │
│  │  │  Generation Backends ────┴────────────────────────────────│ │ │
│  │  └───────────────────────────────────────────────────────────┘ │ │
│  │                                                                │ │
│  │  MAPPING & GENERATION LAYER                                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │ │
│  │  │ Whisper        │  │ Demucs         │  │ Fingerprinting  │   │ │
│  │  │ (Transcribe)   │  │ (Vocal Split)  │  │ (Song ID)       │   │ │
│  │  └────────────────┘  └────────────────┘  └─────────────────┘   │ │
│  │                                                                 │ │
│  │  LYRIC DETECTION (Optional)                                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│                           SERVER (Python/FastAPI)                    │
└──────────────────────────────────────────────────────────────────────┘
```

## Frontend Components

### Audio Pipeline
- **AudioInputManager**: Handles file loading and tab audio capture
- **AudioAnalyzer**: FFT analysis, computes RMS, bass/mid/treble, spectral centroid
- **AdvancedBeatDetector**: Multi-stage beat detection with tempo tracking
  - OnsetDetector: Multi-band spectral flux for transient detection
  - TempoEstimator: Autocorrelation-based BPM calculation
  - BeatScheduler: Prediction + confirmation for robust beat tracking

### Visualization
- **VisualizerEngine**: Manages rendering modes and Three.js scene
- **Modes**: Bars, Waveform, Circular, Particles, Tunnel, Fluid, AIGenerated
- Each mode renders based on current audio metrics

### AI Integration
- **useAIGenerator**: React hook managing WebSocket connection
- **AIGeneratorPanel**: UI for prompts, settings, and connection status
- **AIGeneratedMode**: Displays frames received from server

## Server Components

### WebSocket Handler
- Receives audio metrics from frontend
- Sends generated frames back as base64 JPEG

### Mapping Layer
- **AudioMapper**: Converts audio metrics to generation parameters
- **PromptModulator**: Adjusts prompts based on energy/mood
- **StoryController**: Long-form narrative arc management

### Generation Pipeline
- **StreamDiffusion**: Real-time Stable Diffusion inference
- **ControlNet**: Optional pose guidance for consistency
- **KeyframeInterpolator**: RIFE-based frame interpolation

### Lyric Detection (Optional)
- **Whisper**: Speech-to-text for sung lyrics
- **Demucs**: Vocal separation for cleaner transcription
- **Fingerprinting**: Identify known songs for pre-loaded lyrics

## Data Flow

1. **Audio In**: User provides audio (file or tab capture)
2. **Analysis**: Frontend analyzes audio, computes metrics
3. **Send Metrics**: Metrics sent to server via WebSocket
4. **Map to Params**: Server maps audio features to generation parameters
5. **Generate**: StreamDiffusion produces frame
6. **Send Frame**: Frame sent back as base64 JPEG
7. **Display**: Frontend displays frame in AIGeneratedMode

## Configuration

All server settings use the `EASE_` prefix in environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `EASE_MODEL` | HuggingFace model ID | `Lykon/dreamshaper-8` |
| `EASE_WIDTH` | Output width | `512` |
| `EASE_HEIGHT` | Output height | `512` |
| `EASE_GENERATOR_BACKEND` | Backend: `stream_diffusion` or `flux_klein` | `stream_diffusion` |
| `EASE_USE_CONTROLNET` | Enable ControlNet | `false` |
| `EASE_LYRICS` | Enable lyric detection | `false` |

See `server/.env.example` for full list.
