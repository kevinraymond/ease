"""Rolling lyric buffer with time-based decay for keyword persistence."""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WordEntry:
    """An entry in the lyric buffer."""

    weight: float
    timestamp: float
    count: int = 1  # How many times this word has been added


@dataclass
class LyricBufferConfig:
    """Configuration for the lyric buffer."""

    # How long before keywords decay (seconds)
    decay_seconds: float = 10.0

    # Boost factor for repeated words
    repetition_boost: float = 0.15

    # Maximum repetition boost (caps at this)
    max_repetition_boost: float = 0.5

    # Minimum weight to keep word in buffer
    min_weight_threshold: float = 0.5


class LyricBuffer:
    """Rolling buffer for lyric-derived keywords with time-based decay.

    Maintains a set of active keywords that fade out over time.
    Repeated words get boosted weights.
    """

    def __init__(
        self,
        decay_seconds: float = 10.0,
        config: Optional[LyricBufferConfig] = None,
    ):
        """Initialize the lyric buffer.

        Args:
            decay_seconds: How long before keywords fully decay
            config: Optional configuration object
        """
        self.config = config or LyricBufferConfig(decay_seconds=decay_seconds)
        self._words: dict[str, WordEntry] = {}

    def add_words(self, words: list[tuple[str, float]]) -> None:
        """Add words with weights, refreshing timestamps.

        Args:
            words: List of (word, weight) tuples
        """
        now = time.time()

        for word, weight in words:
            word_lower = word.lower()

            if word_lower in self._words:
                # Update existing entry
                entry = self._words[word_lower]
                entry.count += 1
                entry.timestamp = now

                # Boost weight for repetition (capped)
                repetition_boost = min(
                    entry.count * self.config.repetition_boost,
                    self.config.max_repetition_boost,
                )
                entry.weight = max(entry.weight, weight) + repetition_boost
            else:
                # Add new entry
                self._words[word_lower] = WordEntry(
                    weight=weight,
                    timestamp=now,
                    count=1,
                )

    def get_active_keywords(
        self,
        max_count: int = 4,
    ) -> list[tuple[str, float]]:
        """Get top keywords that haven't decayed, sorted by effective weight.

        Args:
            max_count: Maximum number of keywords to return

        Returns:
            List of (word, effective_weight) tuples
        """
        now = time.time()
        active: list[tuple[str, float]] = []

        # Calculate effective weights with decay
        for word, entry in self._words.items():
            age = now - entry.timestamp
            if age >= self.config.decay_seconds:
                continue

            # Linear decay based on age
            decay_factor = 1.0 - (age / self.config.decay_seconds)
            effective_weight = entry.weight * decay_factor

            if effective_weight >= self.config.min_weight_threshold:
                active.append((word, effective_weight))

        # Sort by effective weight descending
        active.sort(key=lambda x: x[1], reverse=True)

        return active[:max_count]

    def get_all_active(self) -> list[tuple[str, float, float]]:
        """Get all active keywords with their ages.

        Returns:
            List of (word, effective_weight, age_seconds) tuples
        """
        now = time.time()
        result = []

        for word, entry in self._words.items():
            age = now - entry.timestamp
            if age >= self.config.decay_seconds:
                continue

            decay_factor = 1.0 - (age / self.config.decay_seconds)
            effective_weight = entry.weight * decay_factor

            if effective_weight >= self.config.min_weight_threshold:
                result.append((word, effective_weight, age))

        return result

    def cleanup(self) -> int:
        """Remove fully decayed entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        to_remove = [
            word
            for word, entry in self._words.items()
            if now - entry.timestamp >= self.config.decay_seconds
        ]

        for word in to_remove:
            del self._words[word]

        return len(to_remove)

    def clear(self) -> None:
        """Clear all entries."""
        self._words.clear()

    def __len__(self) -> int:
        """Return number of entries (including decayed ones)."""
        return len(self._words)

    @property
    def active_count(self) -> int:
        """Return number of active (non-decayed) entries."""
        now = time.time()
        return sum(
            1
            for entry in self._words.values()
            if now - entry.timestamp < self.config.decay_seconds
        )
