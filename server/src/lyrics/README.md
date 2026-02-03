# Lyric Detection System

A hybrid lyric detection pipeline that combines audio fingerprinting with real-time transcription for optimal latency and accuracy.

## Architecture

```
                     Audio Input
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   [Fingerprint]                  [Fast Transcribe]
   (Chromaprint)                  (Whisper large-v3-turbo)
          │                               │
     Match found?                   Keywords only
          │                               │
     ┌────┴────┐                         │
     │Yes      │No                       │
     ▼         ▼                         │
 [Local DB] [Fallback]◄──────────────────┘
   lyrics   to fast
            transcribe
          │
          ▼
   [Keyword Output]
          │
          ▼
   [Visual Prompts]
```

## Why Hybrid?

**Problem**: Whisper needs 2-5 seconds of audio context for accurate lyric transcription, creating unavoidable latency.

**Solution**:
- **For known songs**: Audio fingerprinting identifies the song in ~15 seconds. Pre-timed lyrics from the database provide perfect synchronization with zero transcription delay.
- **For unknown songs**: Optimized Whisper transcription (large-v3-turbo model with Demucs vocal separation) provides high-quality real-time fallback.

## Components

### HybridLyricPipeline (`hybrid_pipeline.py`)

The main entry point. Automatically handles:
1. Fingerprint collection during first 15 seconds
2. Database lookup for matched songs
3. Fallback to transcription if no match
4. Keyword extraction and delivery

```python
from lyrics import HybridLyricPipeline, create_pipeline_from_settings

# Create from config
pipeline = create_pipeline_from_settings()
pipeline.start()

# Feed audio continuously
pipeline.add_audio_chunk(audio_samples, sample_rate)

# Get keywords for visual prompts
keywords = pipeline.get_current_keywords()  # e.g., ['fire', 'night', 'dance']

# For matched songs, update playback position
if pipeline.is_matched:
    pipeline.update_playback_position(position_ms)
```

### Fingerprinter (`fingerprinter.py`)

Audio fingerprinting using Chromaprint:

```python
from lyrics import Fingerprinter

fp = Fingerprinter(duration_seconds=15.0)
hash = fp.fingerprint(audio_samples, sample_rate)
# hash = "AQAAxxxxxxx..." (compact fingerprint)
```

### LyricDatabase (`lyric_database.py`)

SQLite storage for songs and pre-timed lyrics:

```python
from lyrics import LyricDatabase

db = LyricDatabase("./lyrics_db.sqlite")

# Add a song
song_id = db.add_song(
    fingerprint_hash="AQAAxxxxxxx...",
    title="Bohemian Rhapsody",
    artist="Queen",
    duration_ms=354000,
)

# Add lyrics with timing
db.add_lyric(song_id, time_ms=0, text="Is this the real life?", keywords=["life", "dream"])

# Query at playback time
lyric = db.get_lyric_at_time(song_id, position_ms=3500, lookahead_ms=500)
```

### LyricDetector (`detector.py`)

Whisper-based transcription (fallback path):

```python
from lyrics import LyricDetector

detector = LyricDetector(
    model_size="large-v3-turbo",  # High-quality model (6x faster than large)
    buffer_seconds=5.0,           # Good context for accuracy
    beam_size=5,                  # Beam search for better accuracy
    vocal_separation=True,        # Use Demucs for clean vocals
)

detector.add_audio_chunk(audio, sample_rate)
result = detector.transcribe()
# result.text = "dancing through the fire..."
```

### KeywordExtractor (`keywords.py`)

Extracts visual keywords from lyric text:

```python
from lyrics import KeywordExtractor

extractor = KeywordExtractor()
keywords = extractor.extract("dancing through the fire tonight")
# [('fire', 1.0), ('dancing', 0.9), ('night', 0.8)]
```

## Importing Songs

### Method 1: From Audio File

Use the CLI tool to import songs with LRC lyrics when you have the audio file:

```bash
cd ease-ai-server

# Import a song (fingerprints the audio file)
uv run python -m src.lyrics.import_lrc /path/to/song.mp3 /path/to/lyrics.lrc

# With metadata override
uv run python -m src.lyrics.import_lrc song.mp3 lyrics.lrc \
    --title "Song Name" \
    --artist "Artist Name"

# Replace existing lyrics
uv run python -m src.lyrics.import_lrc song.mp3 lyrics.lrc --replace
```

### Method 2: From Tab Capture (YouTube, Spotify, etc.)

When playing audio from another browser tab:

1. **Enable lyrics** in the AI Gen panel
2. **Play the song** from the other tab (YouTube, Spotify, etc.)
3. **Wait 15 seconds** for fingerprinting to complete
4. **Click "Learn Fingerprint"** button (appears after fingerprinting)
5. **Note the song ID** from the notification
6. **Import LRC file** using the song ID:

```bash
# Attach lyrics to the learned fingerprint
uv run python -m src.lyrics.import_lrc --song-id 5 /path/to/lyrics.lrc
```

The next time that song plays, it will be recognized automatically!

### Management Commands

```bash
# List all songs in database
uv run python -m src.lyrics.import_lrc --list

# Show database stats
uv run python -m src.lyrics.import_lrc --stats
```

### LRC Format

Standard LRC format with timestamps:

```
[ti:Song Title]
[ar:Artist Name]
[00:12.34] First line of lyrics
[00:17.89] Second line of lyrics
[00:23.45] Third line of lyrics
```

## Configuration

All settings are in `config.py`:

```python
# Fingerprinting
fingerprint_enabled: bool = True
fingerprint_duration_seconds: float = 15.0
fingerprint_db_path: str = "./lyrics_db.sqlite"

# Transcription (fallback)
lyric_model_size: str = "large-v3-turbo"
lyric_buffer_seconds: float = 5.0
lyric_beam_size: int = 5
lyric_initial_prompt: str = "Song lyrics, singing vocals, English"

# Vocal separation
lyric_vocal_separation: bool = True
lyric_demucs_model: str = "htdemucs_ft"

# Beat alignment (frontend)
lyric_beat_aligned: bool = True
lyric_beat_delay: int = 2  # beats
```

## Frontend Integration

### LyricScheduler (`LyricScheduler.ts`)

Beat-aligned keyword delivery for perceptually synchronized visuals:

```typescript
import { LyricScheduler } from './audio/LyricScheduler';

const scheduler = new LyricScheduler({
  beatDelay: 2,           // Delay keywords by 2 beats
  keywordLifetimeBeats: 4 // Keywords fade after 4 beats
});

// When keywords arrive from WebSocket:
scheduler.enqueue(['fire', 'night', 'dance']);

// In render loop, on each beat:
if (beatInfo.isBeat) {
  scheduler.onBeat();
  const keywords = scheduler.getActiveKeywords();
  const freshness = scheduler.getKeywordFreshness(); // 0-1 for fading
}
```

### UI State

The `LyricInfo` interface includes pipeline status:

```typescript
interface LyricInfo {
  text: string;
  keywords: [string, number][];
  confidence: number;

  // Pipeline status
  pipeline_state?: 'fingerprinting' | 'matched' | 'not_matched';
  fingerprint_progress?: number;  // 0-1 during fingerprinting
  matched_song_title?: string;
  matched_song_artist?: string;
}
```

## Pipeline States

```
INITIALIZING → FINGERPRINTING → MATCHED (database lyrics)
                             ↘ NOT_MATCHED (transcription fallback)
```

| State | Description | Lyric Source |
|-------|-------------|--------------|
| `initializing` | Setting up components | None |
| `fingerprinting` | Collecting audio for fingerprint | Transcription (in parallel) |
| `matched` | Song found in database | Database lyrics |
| `not_matched` | Unknown song | Transcription |

## Performance Characteristics

| Scenario | Latency | Accuracy |
|----------|---------|----------|
| Known song (DB) | ~0ms (pre-timed) | Perfect |
| Unknown song | ~500-1000ms | Good (with Demucs) |
| Unknown + beat-aligned | Perceptually synced | Good |

## Dependencies

```toml
# pyproject.toml
pyacoustid = ">=1.3.0"  # Chromaprint wrapper
faster-whisper = ">=1.0.0"
demucs = ">=4.0.1"  # Vocal separation
```

System dependencies:
- **Linux**: `sudo apt install libchromaprint-tools`
- **macOS**: `brew install chromaprint`
- **Windows**: Included with pyacoustid

## Troubleshooting

### Fingerprint not matching

1. Ensure at least 15 seconds of audio has been processed
2. Check that `fingerprint_enabled = True` in config
3. Verify the song was imported: `uv run python -m src.lyrics.import_lrc --list`

### Transcription too slow

1. Use smaller model: `lyric_model_size = "tiny"` (default is `large-v3-turbo`)
2. Reduce buffer: `lyric_buffer_seconds = 2.0` (default is 5.0)
3. Use greedy decoding: `lyric_beam_size = 1` (default is 5)
4. Use faster Demucs: `lyric_demucs_model = "htdemucs"` (default is `htdemucs_ft`)
5. Consider disabling vocal separation if GPU is constrained

### Keywords not appearing

1. Check that lyrics are enabled in the UI
2. Verify audio is being captured (check browser permissions)
3. Look at server logs for transcription output

### Beat alignment feels off

Adjust `lyric_beat_delay` in config (default: 2 beats). Lower = more responsive, higher = more "on the beat".
