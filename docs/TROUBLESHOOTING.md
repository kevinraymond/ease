# Troubleshooting

Common issues and their solutions.

## Installation Issues

### "Python not found" or wrong version

**Symptom**: Setup script fails with Python error

**Solution**:
```bash
# Check Python version
python3 --version  # Should be 3.10+

# On Ubuntu/Debian
sudo apt install python3.10 python3.10-venv

# On macOS with Homebrew
brew install python@3.10
```

### "uv: command not found"

**Solution**:
```bash
pip install uv
# Or
pip3 install uv
```

### "npm: command not found"

**Solution**: Install Node.js 18+ from [nodejs.org](https://nodejs.org)

### CUDA not detected

**Symptom**: Server starts but says "CUDA not available"

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall PyTorch with CUDA
cd server
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Runtime Issues

### Server won't start

**Symptom**: Error on `uv run python -m src.main`

**Check**:
1. Are you in the `server/` directory?
2. Did `uv sync` complete successfully?
3. Is another process using port 8765?

```bash
# Kill process on port 8765
lsof -ti:8765 | xargs kill -9

# Try again
uv run python -m src.main
```

### Frontend won't connect to server

**Symptom**: "Connection failed" in AI Generator panel

**Check**:
1. Is the server running?
2. Is it running on the expected port (8765)?
3. Check browser console for WebSocket errors

**Solution**:
- Ensure server shows "Uvicorn running on ws://0.0.0.0:8765"
- Try refreshing the page
- Check firewall isn't blocking localhost connections

### No audio detected

**Symptom**: Visualization doesn't respond to audio

**For file playback**:
- Ensure file is a supported format (MP3, WAV, FLAC, OGG, M4A)
- Try a different audio file
- Check browser console for decoding errors

**For tab capture**:
- Ensure audio is playing in another tab
- Select the correct tab when prompted
- Some sites may block audio capture (try YouTube or SoundCloud)

### Black screen in AI Generated mode

**Symptom**: Selected AI Generated mode but nothing appears

**Check**:
1. Is the server connected? (Check status in AI Generator panel)
2. Is generation started? (Click "Start" button)
3. Check server logs for errors

### Out of Memory (OOM)

**Symptom**: Server crashes with CUDA out of memory error

**Solution**:
```bash
# Edit server/.env
EASE_WIDTH=384
EASE_HEIGHT=384
EASE_USE_CONTROLNET=false
EASE_LYRICS=false
```

See [GPU Requirements](GPU_REQUIREMENTS.md) for detailed VRAM optimization.

## Performance Issues

### Low FPS / Slow generation

**Causes and solutions**:

1. **Resolution too high**: Reduce in `.env`
   ```bash
   EASE_WIDTH=384
   EASE_HEIGHT=384
   ```

2. **Using FLUX backend**: FLUX.2 Klein offers higher quality but is slower
   ```bash
   # For faster generation (~20 FPS), use StreamDiffusion backend:
   EASE_GENERATOR_BACKEND=stream_diffusion
   ```

3. **Other GPU processes**: Close games, other ML models

4. **TensorRT not compiled**: First-time TensorRT compilation is slow
   - Wait for compilation to complete
   - Or disable TensorRT: `EASE_USE_TENSORRT=false`

### Audio/visual sync issues

**Symptom**: Visuals don't match the music

**Solutions**:
- Reduce network latency by running server locally
- Increase `EASE_TARGET_FPS` if GPU can handle it

## Getting Help

### Collect debug info

When reporting issues, include:

```bash
# System info
nvidia-smi
python3 --version
node --version
uv --version

# Server logs
cd server && uv run python -m src.main 2>&1 | tee server.log
```

### Where to report

- GitHub Issues: [github.com/[user]/ease/issues](https://github.com/[user]/ease/issues)
- Include: Steps to reproduce, error messages, system info
