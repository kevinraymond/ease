"""Local SQLite database for storing pre-timed lyrics with audio fingerprints.

This enables instant lyric lookup for known songs, bypassing the latency
of real-time transcription. Songs are identified by their audio fingerprint
(from Chromaprint), and lyrics are stored with precise timing information.

## Database Schema

```sql
songs:
  id INTEGER PRIMARY KEY
  fingerprint_hash TEXT UNIQUE  -- Chromaprint fingerprint
  title TEXT
  artist TEXT
  duration_ms INTEGER
  created_at TIMESTAMP

lyrics:
  id INTEGER PRIMARY KEY
  song_id INTEGER REFERENCES songs(id)
  time_ms INTEGER              -- When this lyric line starts
  end_time_ms INTEGER          -- When this lyric line ends (optional)
  text TEXT                    -- The lyric text
  keywords TEXT                -- JSON array of visual keywords
```

## Usage

```python
db = LyricDatabase("./lyrics.sqlite")

# Add a song with fingerprint
song_id = db.add_song(
    fingerprint_hash="AQAAxxxxxxx...",
    title="Bohemian Rhapsody",
    artist="Queen",
    duration_ms=354000,
)

# Add lyrics with timing
db.add_lyric(song_id, time_ms=0, text="Is this the real life?", keywords=["life", "dream"])
db.add_lyric(song_id, time_ms=3500, text="Is this just fantasy?", keywords=["fantasy", "dream"])

# Look up song by fingerprint
match = db.find_song_by_fingerprint("AQAAxxxxxxx...")
if match:
    lyrics = db.get_lyrics(match.id)
    for lyric in lyrics:
        print(f"{lyric.time_ms}ms: {lyric.text}")
```
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Song:
    """A song entry in the database."""
    id: int
    fingerprint_hash: str
    fingerprint_raw: list[int]  # Raw fingerprint integers for fuzzy matching
    title: str
    artist: str
    duration_ms: int


@dataclass
class LyricLine:
    """A single lyric line with timing and keywords."""
    id: int
    song_id: int
    time_ms: int
    end_time_ms: Optional[int]
    text: str
    keywords: list[str]


class LyricDatabase:
    """SQLite database for storing songs and their pre-timed lyrics.

    Thread-safe through SQLite's built-in locking mechanisms.
    """

    def __init__(self, db_path: str = "./lyrics_db.sqlite"):
        """Initialize the database, creating tables if needed.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_tables(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Songs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint_hash TEXT UNIQUE,
                fingerprint_raw TEXT,
                title TEXT NOT NULL,
                artist TEXT,
                duration_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: add fingerprint_raw column if it doesn't exist
        cursor.execute("PRAGMA table_info(songs)")
        columns = [row[1] for row in cursor.fetchall()]
        if "fingerprint_raw" not in columns:
            cursor.execute("ALTER TABLE songs ADD COLUMN fingerprint_raw TEXT")
            logger.info("Added fingerprint_raw column to songs table")

        # Lyrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lyrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
                time_ms INTEGER NOT NULL,
                end_time_ms INTEGER,
                text TEXT NOT NULL,
                keywords TEXT DEFAULT '[]'
            )
        """)

        # Index for fast fingerprint lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_songs_fingerprint ON songs(fingerprint_hash)
        """)

        # Index for fast lyric retrieval by song and time
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lyrics_song_time ON lyrics(song_id, time_ms)
        """)

        conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def add_song(
        self,
        fingerprint_hash: str,
        title: str,
        artist: str = "",
        duration_ms: int = 0,
        fingerprint_raw: Optional[list[int]] = None,
    ) -> int:
        """Add a new song to the database.

        Args:
            fingerprint_hash: Chromaprint fingerprint hash (encoded).
            title: Song title.
            artist: Artist name.
            duration_ms: Song duration in milliseconds.
            fingerprint_raw: Raw fingerprint integers for fuzzy matching.

        Returns:
            The ID of the newly created song.

        Raises:
            sqlite3.IntegrityError: If fingerprint already exists.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        raw_json = json.dumps(fingerprint_raw) if fingerprint_raw else None

        cursor.execute("""
            INSERT INTO songs (fingerprint_hash, fingerprint_raw, title, artist, duration_ms)
            VALUES (?, ?, ?, ?, ?)
        """, (fingerprint_hash, raw_json, title, artist, duration_ms))

        conn.commit()
        song_id = cursor.lastrowid
        logger.info(f"Added song: {title} by {artist} (id={song_id})")
        return song_id

    def add_song_if_not_exists(
        self,
        fingerprint_hash: str,
        title: str,
        artist: str = "",
        duration_ms: int = 0,
        fingerprint_raw: Optional[list[int]] = None,
    ) -> tuple[int, bool]:
        """Add a song if it doesn't already exist.

        Args:
            fingerprint_hash: Chromaprint fingerprint hash.
            title: Song title.
            artist: Artist name.
            duration_ms: Song duration in milliseconds.
            fingerprint_raw: Raw fingerprint integers for fuzzy matching.

        Returns:
            Tuple of (song_id, was_created).
        """
        existing = self.find_song_by_fingerprint(fingerprint_hash)
        if existing:
            return existing.id, False

        try:
            song_id = self.add_song(fingerprint_hash, title, artist, duration_ms, fingerprint_raw)
            return song_id, True
        except sqlite3.IntegrityError:
            # Race condition - another thread added it
            existing = self.find_song_by_fingerprint(fingerprint_hash)
            return existing.id if existing else 0, False

    def find_song_by_fingerprint(self, fingerprint_hash: str) -> Optional[Song]:
        """Find a song by its fingerprint hash (exact match).

        Args:
            fingerprint_hash: Chromaprint fingerprint hash to search for.

        Returns:
            Song if found, None otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, fingerprint_hash, fingerprint_raw, title, artist, duration_ms
            FROM songs WHERE fingerprint_hash = ?
        """, (fingerprint_hash,))

        row = cursor.fetchone()
        if row:
            raw = json.loads(row["fingerprint_raw"]) if row["fingerprint_raw"] else []
            return Song(
                id=row["id"],
                fingerprint_hash=row["fingerprint_hash"],
                fingerprint_raw=raw,
                title=row["title"],
                artist=row["artist"],
                duration_ms=row["duration_ms"],
            )
        return None

    def find_song_by_fingerprint_fuzzy(
        self,
        fingerprint_raw: list[int],
        threshold: float = 0.4,
    ) -> Optional[tuple[Song, float]]:
        """Find a song by fuzzy fingerprint matching.

        Compares the input fingerprint against all songs in the database
        using bit-level similarity. This is more robust than exact matching
        when audio comes from different sources (e.g., YouTube vs MP3 file).

        Args:
            fingerprint_raw: Raw fingerprint integers to match.
            threshold: Minimum similarity score (0-1) to consider a match.
                       0.4 is a reasonable default for same-song matching.

        Returns:
            Tuple of (Song, similarity_score) if a match is found above
            the threshold, None otherwise.
        """
        if not fingerprint_raw:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, fingerprint_hash, fingerprint_raw, title, artist, duration_ms
            FROM songs WHERE fingerprint_raw IS NOT NULL
        """)

        best_match: Optional[Song] = None
        best_score: float = 0.0
        all_scores: list[tuple[str, float, bool]] = []  # (title, score, has_raw)

        for row in cursor.fetchall():
            stored_raw = json.loads(row["fingerprint_raw"]) if row["fingerprint_raw"] else []
            title = row["title"]

            if not stored_raw:
                all_scores.append((title, 0.0, False))
                continue

            # Calculate bit-level similarity
            similarity = self._compare_fingerprints_raw(fingerprint_raw, stored_raw)
            all_scores.append((title, similarity, True))

            if similarity > best_score:
                best_score = similarity
                best_match = Song(
                    id=row["id"],
                    fingerprint_hash=row["fingerprint_hash"],
                    fingerprint_raw=stored_raw,
                    title=row["title"],
                    artist=row["artist"],
                    duration_ms=row["duration_ms"],
                )

        if best_match and best_score >= threshold:
            logger.info(f"Fuzzy match: {best_match.title} (score={best_score:.3f})")
            return best_match, best_score

        # Only log details when no match found (for debugging)
        if all_scores:
            top_scores = sorted(all_scores, key=lambda x: -x[1])[:3]
            scores_str = ", ".join(
                f"{title}: {score:.2f}{'' if has_raw else ' (no raw)'}"
                for title, score, has_raw in top_scores
            )
            logger.info(f"No fuzzy match (best={best_score:.2f} < {threshold}). Top: {scores_str}")
        else:
            logger.info(f"No fuzzy match: database empty or no raw fingerprints")
        return None

    def _compare_fingerprints_raw(
        self,
        fp1: list[int],
        fp2: list[int],
    ) -> float:
        """Compare two raw fingerprints using bit-level similarity.

        Args:
            fp1: First fingerprint as list of integers.
            fp2: Second fingerprint as list of integers.

        Returns:
            Similarity score from 0.0 (no match) to 1.0 (identical).
        """
        if not fp1 or not fp2:
            return 0.0

        # Compare overlapping portion
        min_len = min(len(fp1), len(fp2))
        if min_len == 0:
            return 0.0

        matching_bits = 0
        total_bits = min_len * 32  # 32 bits per integer

        for i in range(min_len):
            # XOR gives differing bits, popcount gives count of 1s
            diff = fp1[i] ^ fp2[i]
            diff_bits = bin(diff & 0xFFFFFFFF).count('1')
            matching_bits += 32 - diff_bits

        return matching_bits / total_bits

    def find_song_by_id(self, song_id: int) -> Optional[Song]:
        """Find a song by its ID.

        Args:
            song_id: Song ID to search for.

        Returns:
            Song if found, None otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, fingerprint_hash, fingerprint_raw, title, artist, duration_ms
            FROM songs WHERE id = ?
        """, (song_id,))

        row = cursor.fetchone()
        if row:
            raw = json.loads(row["fingerprint_raw"]) if row["fingerprint_raw"] else []
            return Song(
                id=row["id"],
                fingerprint_hash=row["fingerprint_hash"],
                fingerprint_raw=raw,
                title=row["title"],
                artist=row["artist"],
                duration_ms=row["duration_ms"],
            )
        return None

    def add_lyric(
        self,
        song_id: int,
        time_ms: int,
        text: str,
        keywords: Optional[list[str]] = None,
        end_time_ms: Optional[int] = None,
    ) -> int:
        """Add a lyric line to a song.

        Args:
            song_id: ID of the song.
            time_ms: Start time of the lyric in milliseconds.
            text: The lyric text.
            keywords: List of visual keywords extracted from the text.
            end_time_ms: Optional end time of the lyric in milliseconds.

        Returns:
            The ID of the newly created lyric entry.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        keywords_json = json.dumps(keywords or [])

        cursor.execute("""
            INSERT INTO lyrics (song_id, time_ms, end_time_ms, text, keywords)
            VALUES (?, ?, ?, ?, ?)
        """, (song_id, time_ms, end_time_ms, text, keywords_json))

        conn.commit()
        return cursor.lastrowid

    def add_lyrics_batch(
        self,
        song_id: int,
        lyrics: list[tuple[int, str, list[str]]],
    ) -> int:
        """Add multiple lyric lines at once (more efficient).

        Args:
            song_id: ID of the song.
            lyrics: List of (time_ms, text, keywords) tuples.

        Returns:
            Number of lyrics added.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        data = [
            (song_id, time_ms, None, text, json.dumps(keywords))
            for time_ms, text, keywords in lyrics
        ]

        cursor.executemany("""
            INSERT INTO lyrics (song_id, time_ms, end_time_ms, text, keywords)
            VALUES (?, ?, ?, ?, ?)
        """, data)

        conn.commit()
        logger.info(f"Added {len(data)} lyrics to song {song_id}")
        return len(data)

    def get_lyrics(self, song_id: int) -> list[LyricLine]:
        """Get all lyrics for a song, ordered by time.

        Args:
            song_id: ID of the song.

        Returns:
            List of LyricLine objects ordered by time_ms.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, song_id, time_ms, end_time_ms, text, keywords
            FROM lyrics WHERE song_id = ?
            ORDER BY time_ms
        """, (song_id,))

        return [
            LyricLine(
                id=row["id"],
                song_id=row["song_id"],
                time_ms=row["time_ms"],
                end_time_ms=row["end_time_ms"],
                text=row["text"],
                keywords=json.loads(row["keywords"]),
            )
            for row in cursor.fetchall()
        ]

    def get_lyric_at_time(
        self,
        song_id: int,
        time_ms: int,
        lookahead_ms: int = 0,
    ) -> Optional[LyricLine]:
        """Get the lyric line at a specific time.

        Args:
            song_id: ID of the song.
            time_ms: Current playback time in milliseconds.
            lookahead_ms: Look ahead this many ms (for anticipation).

        Returns:
            The lyric line active at the given time, or None.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        target_time = time_ms + lookahead_ms

        cursor.execute("""
            SELECT id, song_id, time_ms, end_time_ms, text, keywords
            FROM lyrics
            WHERE song_id = ? AND time_ms <= ?
            ORDER BY time_ms DESC
            LIMIT 1
        """, (song_id, target_time))

        row = cursor.fetchone()
        if row:
            return LyricLine(
                id=row["id"],
                song_id=row["song_id"],
                time_ms=row["time_ms"],
                end_time_ms=row["end_time_ms"],
                text=row["text"],
                keywords=json.loads(row["keywords"]),
            )
        return None

    def get_upcoming_lyrics(
        self,
        song_id: int,
        time_ms: int,
        window_ms: int = 5000,
    ) -> list[LyricLine]:
        """Get lyrics in an upcoming time window.

        Args:
            song_id: ID of the song.
            time_ms: Current playback time in milliseconds.
            window_ms: Window size in milliseconds.

        Returns:
            List of lyrics in the window, ordered by time.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, song_id, time_ms, end_time_ms, text, keywords
            FROM lyrics
            WHERE song_id = ? AND time_ms >= ? AND time_ms < ?
            ORDER BY time_ms
        """, (song_id, time_ms, time_ms + window_ms))

        return [
            LyricLine(
                id=row["id"],
                song_id=row["song_id"],
                time_ms=row["time_ms"],
                end_time_ms=row["end_time_ms"],
                text=row["text"],
                keywords=json.loads(row["keywords"]),
            )
            for row in cursor.fetchall()
        ]

    def delete_song(self, song_id: int) -> bool:
        """Delete a song and all its lyrics.

        Args:
            song_id: ID of the song to delete.

        Returns:
            True if the song was deleted, False if it didn't exist.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete lyrics first (foreign key constraint)
        cursor.execute("DELETE FROM lyrics WHERE song_id = ?", (song_id,))
        cursor.execute("DELETE FROM songs WHERE id = ?", (song_id,))

        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted song {song_id} and its lyrics")
        return deleted

    def clear_lyrics(self, song_id: int) -> int:
        """Remove all lyrics for a song (keep the song entry).

        Args:
            song_id: ID of the song.

        Returns:
            Number of lyrics deleted.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM lyrics WHERE song_id = ?", (song_id,))
        conn.commit()

        return cursor.rowcount

    def list_songs(self, limit: int = 100) -> list[Song]:
        """List all songs in the database.

        Args:
            limit: Maximum number of songs to return.

        Returns:
            List of Song objects.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, fingerprint_hash, fingerprint_raw, title, artist, duration_ms
            FROM songs ORDER BY created_at DESC LIMIT ?
        """, (limit,))

        return [
            Song(
                id=row["id"],
                fingerprint_hash=row["fingerprint_hash"],
                fingerprint_raw=json.loads(row["fingerprint_raw"]) if row["fingerprint_raw"] else [],
                title=row["title"],
                artist=row["artist"],
                duration_ms=row["duration_ms"],
            )
            for row in cursor.fetchall()
        ]

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with song_count and lyric_count.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM songs")
        song_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM lyrics")
        lyric_count = cursor.fetchone()["count"]

        return {
            "song_count": song_count,
            "lyric_count": lyric_count,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Singleton instance
_database: Optional[LyricDatabase] = None


def get_database(db_path: str = "./lyrics_db.sqlite") -> LyricDatabase:
    """Get or create the global LyricDatabase instance."""
    global _database
    if _database is None:
        _database = LyricDatabase(db_path)
    return _database
