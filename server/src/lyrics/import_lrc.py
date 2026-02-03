#!/usr/bin/env python3
"""CLI tool for importing LRC lyric files into the database.

LRC (LyRiCs) is a standard format for synchronized lyrics. This tool:
1. Fingerprints the audio file
2. Parses the LRC file
3. Extracts visual keywords from each line
4. Stores everything in the local database

## Usage

```bash
# Import lyrics for a single song
uv run python -m lyrics.import_lrc song.mp3 lyrics.lrc

# Import with manual title/artist (if metadata detection fails)
uv run python -m lyrics.import_lrc song.mp3 lyrics.lrc --title "Bohemian Rhapsody" --artist "Queen"

# Update existing song's lyrics
uv run python -m lyrics.import_lrc song.mp3 lyrics.lrc --replace

# List all songs in database
uv run python -m lyrics.import_lrc --list

# Show database stats
uv run python -m lyrics.import_lrc --stats
```

## LRC Format

Standard LRC format with timestamps:
```
[00:12.34] First line of lyrics
[00:17.89] Second line of lyrics
```

Extended LRC with metadata:
```
[ti:Song Title]
[ar:Artist Name]
[al:Album Name]
[00:12.34] First line of lyrics
```
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_lrc(lrc_path: Path) -> tuple[dict[str, str], list[tuple[int, str]]]:
    """Parse an LRC file.

    Args:
        lrc_path: Path to the LRC file.

    Returns:
        Tuple of (metadata_dict, lyrics_list).
        metadata_dict contains: ti (title), ar (artist), al (album), etc.
        lyrics_list is a list of (time_ms, text) tuples.
    """
    metadata = {}
    lyrics = []

    # Regex patterns
    # Metadata: [ti:Song Title] or [ar:Artist]
    metadata_pattern = re.compile(r"\[(\w+):(.+?)\]")
    # Timestamp: [mm:ss.xx] or [mm:ss:xx] or [mm:ss]
    timestamp_pattern = re.compile(r"\[(\d{2}):(\d{2})(?:[.:](\d{2,3}))?\]")

    with open(lrc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for metadata
            meta_match = metadata_pattern.match(line)
            if meta_match and not timestamp_pattern.match(line):
                key = meta_match.group(1).lower()
                value = meta_match.group(2).strip()
                metadata[key] = value
                continue

            # Check for timestamp + lyrics
            ts_matches = list(timestamp_pattern.finditer(line))
            if ts_matches:
                # Get the text after all timestamps
                last_match = ts_matches[-1]
                text = line[last_match.end():].strip()

                if text:  # Only add if there's actual text
                    # Parse each timestamp (some LRC files have multiple per line)
                    for match in ts_matches:
                        minutes = int(match.group(1))
                        seconds = int(match.group(2))
                        centiseconds = int(match.group(3) or "0")

                        # Handle both .xx (centiseconds) and .xxx (milliseconds)
                        if len(match.group(3) or "") == 3:
                            ms = centiseconds
                        else:
                            ms = centiseconds * 10

                        time_ms = minutes * 60000 + seconds * 1000 + ms
                        lyrics.append((time_ms, text))

    # Sort lyrics by time
    lyrics.sort(key=lambda x: x[0])

    return metadata, lyrics


def extract_audio_duration(audio_path: Path) -> Optional[int]:
    """Extract duration from an audio file using soundfile if available.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in milliseconds, or None if extraction failed.
    """
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        return int(info.duration * 1000)
    except Exception:
        pass

    # Fallback: try using ffprobe if available
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return int(duration * 1000)
    except Exception:
        pass

    return None


def load_audio_for_fingerprint(audio_path: Path) -> Optional[tuple[np.ndarray, int]]:
    """Load audio file for fingerprinting.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Tuple of (audio_samples, sample_rate) or None if loading failed.
    """
    # Try soundfile first (supports many formats)
    try:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32")
        return audio, sr
    except Exception as e:
        logger.debug(f"soundfile failed: {e}")

    # Try librosa as fallback
    try:
        import librosa
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        return audio, sr
    except Exception as e:
        logger.debug(f"librosa failed: {e}")

    # Try pydub as last resort
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(audio_path))
        audio = np.array(seg.get_array_of_samples(), dtype=np.float32)
        audio /= 32768.0  # Normalize to -1, 1
        if seg.channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        return audio, seg.frame_rate
    except Exception as e:
        logger.debug(f"pydub failed: {e}")

    return None


def import_lrc_to_song(
    song_id: int,
    lrc_path: Path,
    replace: bool = False,
    db_path: str = "./lyrics_db.sqlite",
) -> bool:
    """Import an LRC file to an existing song in the database.

    This is used after "learning" a fingerprint from tab capture.

    Args:
        song_id: Database ID of the song to attach lyrics to.
        lrc_path: Path to the LRC lyrics file.
        replace: If True, replace existing lyrics for this song.
        db_path: Path to the database file.

    Returns:
        True if import was successful.
    """
    from .keywords import KeywordExtractor
    from .lyric_database import LyricDatabase

    if not lrc_path.exists():
        logger.error(f"LRC file not found: {lrc_path}")
        return False

    # Initialize database
    db = LyricDatabase(db_path)

    # Find the song
    song = db.find_song_by_id(song_id)
    if not song:
        logger.error(f"Song ID {song_id} not found in database")
        logger.error("Use --list to see available songs")
        return False

    logger.info(f"Attaching lyrics to: {song.title} by {song.artist} (ID: {song_id})")

    # Check for existing lyrics
    existing_lyrics = db.get_lyrics(song_id)
    if existing_lyrics:
        if replace:
            logger.info(f"Replacing {len(existing_lyrics)} existing lyrics")
            db.clear_lyrics(song_id)
        else:
            logger.error(f"Song already has {len(existing_lyrics)} lyrics")
            logger.error("Use --replace to update lyrics")
            return False

    # Parse LRC file
    logger.info(f"Parsing LRC file: {lrc_path}")
    metadata, lyrics = parse_lrc(lrc_path)

    if not lyrics:
        logger.error("No lyrics found in LRC file")
        return False

    logger.info(f"Found {len(lyrics)} lyric lines")

    # Update song metadata from LRC if available
    if metadata.get("ti") or metadata.get("ar"):
        # Could update title/artist here if desired
        pass

    # Extract keywords for each lyric line
    logger.info("Extracting visual keywords...")
    keyword_extractor = KeywordExtractor()

    lyrics_with_keywords = []
    for time_ms, text in lyrics:
        keywords = keyword_extractor.extract(text)
        keyword_list = [word for word, weight in keywords]
        lyrics_with_keywords.append((time_ms, text, keyword_list))

        if keyword_list:
            logger.debug(f"  [{time_ms}ms] {text} -> {keyword_list}")

    # Add lyrics to database
    count = db.add_lyrics_batch(song_id, lyrics_with_keywords)
    logger.info(f"Added {count} lyric lines to database")

    # Show summary
    stats = db.get_stats()
    logger.info(f"Database now contains {stats['song_count']} songs, {stats['lyric_count']} lyrics")

    return True


def import_lrc(
    audio_path: Path,
    lrc_path: Path,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    replace: bool = False,
    db_path: str = "./lyrics_db.sqlite",
) -> bool:
    """Import an LRC file for an audio file.

    Args:
        audio_path: Path to the audio file (mp3, wav, etc.)
        lrc_path: Path to the LRC lyrics file.
        title: Override title (uses LRC metadata or filename if not provided).
        artist: Override artist (uses LRC metadata if not provided).
        replace: If True, replace existing lyrics for this song.
        db_path: Path to the database file.

    Returns:
        True if import was successful.
    """
    from .fingerprinter import Fingerprinter
    from .keywords import KeywordExtractor
    from .lyric_database import LyricDatabase

    # Validate files exist
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return False

    if not lrc_path.exists():
        logger.error(f"LRC file not found: {lrc_path}")
        return False

    # Parse LRC file
    logger.info(f"Parsing LRC file: {lrc_path}")
    metadata, lyrics = parse_lrc(lrc_path)

    if not lyrics:
        logger.error("No lyrics found in LRC file")
        return False

    logger.info(f"Found {len(lyrics)} lyric lines")

    # Determine title and artist
    final_title = title or metadata.get("ti") or audio_path.stem
    final_artist = artist or metadata.get("ar") or "Unknown Artist"

    logger.info(f"Song: {final_title} by {final_artist}")

    # Load audio for fingerprinting
    logger.info(f"Loading audio: {audio_path}")
    audio_data = load_audio_for_fingerprint(audio_path)
    if audio_data is None:
        logger.error("Failed to load audio file. Install soundfile or librosa:")
        logger.error("  uv pip install soundfile")
        return False

    audio, sample_rate = audio_data
    logger.info(f"Audio loaded: {len(audio)/sample_rate:.1f}s at {sample_rate}Hz")

    # Generate fingerprint
    logger.info("Generating fingerprint...")
    fingerprinter = Fingerprinter()
    if not fingerprinter.is_available():
        logger.error("Chromaprint not available. Install pyacoustid:")
        logger.error("  uv pip install pyacoustid")
        logger.error("  Linux: sudo apt install libchromaprint-tools")
        logger.error("  macOS: brew install chromaprint")
        return False

    fp_hash = fingerprinter.fingerprint(audio, sample_rate)
    fp_raw = fingerprinter.fingerprint_raw(audio, sample_rate)
    if not fp_hash:
        logger.error("Failed to generate fingerprint")
        return False

    logger.info(f"Fingerprint: {fp_hash[:32]}...")
    logger.info(f"Raw fingerprint: {len(fp_raw) if fp_raw else 0} integers")

    # Get duration
    duration_ms = extract_audio_duration(audio_path) or int(len(audio) / sample_rate * 1000)

    # Initialize database
    db = LyricDatabase(db_path)

    # Check if song exists (exact match only)
    existing = db.find_song_by_fingerprint(fp_hash)

    if existing:
        if replace:
            logger.info(f"Replacing lyrics for existing song (id={existing.id})")
            db.clear_lyrics(existing.id)
            song_id = existing.id
        else:
            logger.error(f"Song already exists in database (id={existing.id})")
            logger.error("Use --replace to update lyrics")
            return False
    else:
        # Add new song
        song_id = db.add_song(
            fingerprint_hash=fp_hash,
            title=final_title,
            artist=final_artist,
            duration_ms=duration_ms,
            fingerprint_raw=fp_raw,
        )
        logger.info(f"Added new song (id={song_id})")

    # Extract keywords for each lyric line
    logger.info("Extracting visual keywords...")
    keyword_extractor = KeywordExtractor()

    lyrics_with_keywords = []
    for time_ms, text in lyrics:
        # Extract keywords from this line
        keywords = keyword_extractor.extract(text)
        keyword_list = [word for word, weight in keywords]
        lyrics_with_keywords.append((time_ms, text, keyword_list))

        if keyword_list:
            logger.debug(f"  [{time_ms}ms] {text} -> {keyword_list}")

    # Add lyrics to database
    count = db.add_lyrics_batch(song_id, lyrics_with_keywords)
    logger.info(f"Added {count} lyric lines to database")

    # Show summary
    stats = db.get_stats()
    logger.info(f"Database now contains {stats['song_count']} songs, {stats['lyric_count']} lyrics")

    return True


def list_songs(db_path: str = "./lyrics_db.sqlite", verbose: bool = False) -> None:
    """List all songs in the database."""
    from .lyric_database import LyricDatabase

    db = LyricDatabase(db_path)
    songs = db.list_songs()

    if not songs:
        print("No songs in database")
        return

    if verbose:
        print(f"\n{'ID':<6} {'Title':<35} {'Artist':<20} {'Lyrics':<8} {'FP Raw':<10}")
        print("-" * 85)
        for song in songs:
            lyrics = db.get_lyrics(song.id)
            has_raw = "YES" if song.fingerprint_raw else "NO"
            raw_len = len(song.fingerprint_raw) if song.fingerprint_raw else 0
            print(f"{song.id:<6} {song.title[:33]:<35} {song.artist[:18]:<20} {len(lyrics):<8} {has_raw} ({raw_len})")
    else:
        print(f"\n{'ID':<6} {'Title':<40} {'Artist':<25} {'Lyrics':<8}")
        print("-" * 80)
        for song in songs:
            lyrics = db.get_lyrics(song.id)
            print(f"{song.id:<6} {song.title[:38]:<40} {song.artist[:23]:<25} {len(lyrics):<8}")


def show_stats(db_path: str = "./lyrics_db.sqlite") -> None:
    """Show database statistics."""
    from .lyric_database import LyricDatabase

    db = LyricDatabase(db_path)
    stats = db.get_stats()

    print(f"\nDatabase: {stats['db_path']}")
    print(f"Songs: {stats['song_count']}")
    print(f"Lyrics: {stats['lyric_count']}")


def diagnose(audio_path: Path, db_path: str = "./lyrics_db.sqlite") -> None:
    """Diagnose fingerprinting issues."""
    from .fingerprinter import Fingerprinter
    from .lyric_database import LyricDatabase

    print(f"\n=== Fingerprint Diagnostic ===\n")

    # 1. Load audio
    print(f"1. Loading audio: {audio_path}")
    audio_data = load_audio_for_fingerprint(audio_path)
    if audio_data is None:
        print("   FAILED: Could not load audio")
        return
    audio, sample_rate = audio_data
    print(f"   OK: {len(audio)/sample_rate:.1f}s at {sample_rate}Hz")

    # 2. Create fingerprinter
    print(f"\n2. Initializing fingerprinter...")
    fp = Fingerprinter()
    if not fp.is_available():
        print("   FAILED: Chromaprint not available")
        return
    print("   OK: Chromaprint available")

    # 3. Generate hash fingerprint
    print(f"\n3. Generating encoded fingerprint...")
    fp_hash = fp.fingerprint(audio, sample_rate)
    if fp_hash:
        print(f"   OK: {fp_hash[:50]}...")
    else:
        print("   FAILED: No hash generated")

    # 4. Generate raw fingerprint
    print(f"\n4. Generating raw fingerprint...")
    fp_raw = fp.fingerprint_raw(audio, sample_rate)
    if fp_raw:
        print(f"   OK: {len(fp_raw)} integers")
        print(f"   First 5: {fp_raw[:5]}")
    else:
        print("   FAILED: No raw fingerprint generated")

    # 5. Check database
    print(f"\n5. Checking database: {db_path}")
    db = LyricDatabase(db_path)
    songs = db.list_songs()
    print(f"   Found {len(songs)} songs")

    # 6. Test exact match
    if fp_hash:
        print(f"\n6. Testing exact match...")
        exact = db.find_song_by_fingerprint(fp_hash)
        if exact:
            print(f"   MATCH: {exact.title} (id={exact.id})")
        else:
            print("   No exact match")

    # 7. Test fuzzy match
    if fp_raw and songs:
        print(f"\n7. Testing fuzzy match...")
        for song in songs:
            if song.fingerprint_raw:
                score = db._compare_fingerprints_raw(fp_raw, song.fingerprint_raw)
                status = "MATCH" if score >= 0.4 else "no match"
                print(f"   {song.title}: {score:.3f} ({status}, {len(song.fingerprint_raw)} ints)")
            else:
                print(f"   {song.title}: NO RAW DATA")

    print(f"\n=== Done ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Import LRC lyric files into the lyrics database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import with audio file (fingerprints the audio)
  %(prog)s song.mp3 lyrics.lrc
  %(prog)s song.mp3 lyrics.lrc --title "Song Name" --artist "Artist"

  # Import to existing song (after using "Learn" button in UI)
  %(prog)s --song-id 5 lyrics.lrc

  # Management commands
  %(prog)s --list
  %(prog)s --stats

Tab Capture Workflow:
  1. Enable lyrics in the UI and play audio from a browser tab
  2. Wait for fingerprinting to complete (15 seconds)
  3. Click "Learn Fingerprint" button in UI
  4. Note the song ID from the response
  5. Run: %(prog)s --song-id <ID> lyrics.lrc
        """,
    )

    parser.add_argument("audio_file", nargs="?", type=Path, help="Audio file (mp3, wav, etc.)")
    parser.add_argument("lrc_file", nargs="?", type=Path, help="LRC lyrics file")
    parser.add_argument("--song-id", type=int, help="Attach lyrics to existing song by ID (for tab capture)")
    parser.add_argument("--title", "-t", help="Override song title")
    parser.add_argument("--artist", "-a", help="Override artist name")
    parser.add_argument("--replace", "-r", action="store_true", help="Replace existing lyrics")
    parser.add_argument("--db", default="./lyrics_db.sqlite", help="Database path")
    parser.add_argument("--list", "-l", action="store_true", help="List all songs")
    parser.add_argument("--stats", "-s", action="store_true", help="Show database stats")
    parser.add_argument("--diagnose", "-d", action="store_true", help="Diagnose fingerprint issues with an audio file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle list/stats commands
    if args.list:
        list_songs(args.db, verbose=args.verbose)
        return

    if args.stats:
        show_stats(args.db)
        return

    if args.diagnose:
        if not args.audio_file:
            logger.error("Audio file required for --diagnose")
            logger.error("Usage: import_lrc --diagnose audio.mp3")
            sys.exit(1)
        diagnose(args.audio_file, args.db)
        return

    # Handle --song-id mode (attach lyrics to existing fingerprint)
    if args.song_id is not None:
        # In --song-id mode, the LRC path is the first positional arg (audio_file)
        lrc_path = args.audio_file
        if not lrc_path:
            logger.error("LRC file required with --song-id")
            logger.error("Usage: import_lrc --song-id <ID> lyrics.lrc")
            sys.exit(1)

        success = import_lrc_to_song(
            song_id=args.song_id,
            lrc_path=lrc_path,
            replace=args.replace,
            db_path=args.db,
        )
        sys.exit(0 if success else 1)

    # Standard mode: require audio and LRC files
    if not args.audio_file or not args.lrc_file:
        parser.print_help()
        sys.exit(1)

    success = import_lrc(
        audio_path=args.audio_file,
        lrc_path=args.lrc_file,
        title=args.title,
        artist=args.artist,
        replace=args.replace,
        db_path=args.db,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
