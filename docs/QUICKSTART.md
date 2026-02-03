# Quick Start Guide

Get EASE running in 5 minutes.

## Quick Start with Docker (Recommended)

The easiest way to run EASE is with Docker:

```bash
# Clone the repository
git clone https://github.com/kevinraymond/ease
cd ease

# Start with Docker Compose
docker compose up
```

Open http://localhost:5173 and you're ready to go!

Requirements: Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

---

## Manual Installation

For development or if you prefer not to use Docker:

## Prerequisites

Before you begin, make sure you have:

- **Node.js 18+**: Download from [nodejs.org](https://nodejs.org)
- **Python 3.10+**: Download from [python.org](https://python.org)
- **NVIDIA GPU**: 6GB+ VRAM with CUDA 11.8+ drivers
- **Git**: For cloning the repository

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/kevinraymond/ease
cd ease

# Run the setup script
./scripts/setup.sh
```

The setup script will:
- Install Python dependencies using `uv`
- Install Node.js dependencies using `npm`
- Create a default `.env` configuration
- Download required AI models (~2-4 GB)

## Step 2: Start the Server

```bash
cd server
uv run python -m src.main
```

You should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Loading model: Lykon/dreamshaper-8
INFO:     Application startup complete.
INFO:     Uvicorn running on ws://0.0.0.0:8765
```

## Step 3: Start the Frontend

Open a new terminal:

```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
```

## Step 4: Open the App

1. Open http://localhost:5173 in your browser
2. You'll see the visualization canvas with controls on the left

## Step 5: Add Audio

Choose one of these methods:

**Option A: Drop an audio file**
- Drag and drop an MP3, WAV, FLAC, or OGG file onto the window

**Option B: Capture browser audio**
- Click "Capture Tab" button
- Select the browser tab playing audio
- Note: You need something playing audio in another tab

## Step 6: Enable AI Generation (Optional)

1. Expand the "AI Generator" panel on the left
2. Click "Connect" to connect to the server
3. Enter a prompt describing your desired visuals
4. Click "Start" to begin AI generation
5. Select "AI Generated" mode from the mode selector

## Troubleshooting

### Server won't start
- Check that CUDA is available: `nvidia-smi`
- Ensure Python 3.10+: `python3 --version`
- Try: `cd server && uv sync`

### No audio detected
- Make sure your audio file is a supported format
- For tab capture, ensure something is playing in another tab
- Check browser permissions for audio capture

### AI generation is slow
- Reduce resolution: set `EASE_WIDTH=384` and `EASE_HEIGHT=384` in `.env`
- Make sure you're using GPU (not CPU)
- Close other GPU-heavy applications

### Out of VRAM
- See [GPU Requirements](GPU_REQUIREMENTS.md) to reduce VRAM usage
- Disable features you don't need in `.env`

## Next Steps

- Explore the 7 different visualization modes
- Customize the AI prompts for your music style
- Read [Architecture](ARCHITECTURE.md) to understand the system
