# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-02

### Added

- **Real-time AI Visualization**: Generate visuals synchronized to audio using Audio Reactive (modded StreamDiffusion for reactivity), StreamDiffusion, or FLUX.2 Klein backends
- **Audio Mapping Presets**: 5 built-in presets (Reactive, Atmospheric, Psychedelic, Minimal, Cinematic) for different visual styles
- **Custom Preset Editor**: Create and save your own audio-to-prompt mappings
- **Beat Detection**: Real-time beat and onset detection for rhythm-synced effects
- **Lyric Detection**: Optional integration with faster-whisper for lyric-driven prompts
- **Docker Support**: Pre-configured containers for easy deployment with CUDA support
- **7 Visualization Modes**: Multiple rendering modes including AI-generated, reactive shapes, and frequency displays
- **Audio Input Flexibility**: Support for local files (MP3, WAV, FLAC, OGG) and browser tab capture
- **WebSocket Architecture**: Low-latency communication between frontend and AI server
- **Info Overlay**: Real-time visualization of audio frequency data and mappings
