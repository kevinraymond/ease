"""Keyword extraction from lyrics for visual prompt generation."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KeywordExtractionConfig:
    """Configuration for keyword extraction."""

    # Base weight for extracted keywords
    base_weight: float = 1.0

    # Weight boost for visual/evocative words
    visual_boost: float = 0.3

    # Weight boost for emotion words
    emotion_boost: float = 0.2

    # Minimum word length to consider
    min_word_length: int = 3

    # Maximum keywords to return
    max_keywords: int = 8

    # Words to filter out (e.g., prompt leakage, false positives)
    filter_words: frozenset[str] = field(default_factory=frozenset)


# Words to filter out - common stop words that don't add visual meaning
STOP_WORDS = frozenset({
    # Articles
    "a", "an", "the",
    # Pronouns
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what",
    "this", "that", "these", "those",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "down", "out", "off", "over", "under",
    "into", "onto", "upon", "about", "through", "during",
    "before", "after", "above", "below", "between", "among",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "for",
    "because", "although", "while", "if", "when", "where",
    # Common verbs (non-visual)
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "shall", "can", "need", "dare", "ought",
    "get", "got", "getting", "let", "make", "made",
    "go", "going", "gone", "come", "came", "coming",
    "say", "said", "says", "tell", "told", "ask", "asked",
    "know", "knew", "known", "think", "thought", "feel", "felt",
    "want", "wanted", "like", "liked", "need", "needed",
    # Common adverbs
    "not", "no", "yes", "very", "really", "just", "only",
    "also", "too", "more", "most", "less", "least",
    "now", "then", "here", "there", "when", "where",
    "how", "why", "all", "each", "every", "both",
    "few", "some", "any", "many", "much", "other",
    # Common contractions (expanded)
    "don", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
    "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
    "can", "ll", "ve", "re", "ain",
    # Filler words
    "oh", "ah", "uh", "um", "yeah", "yea", "hey", "ooh", "whoa",
    "na", "la", "da", "ba", "sha",
})

# Visual/evocative words that are particularly good for image generation
VISUAL_WORDS = frozenset({
    # Colors
    "red", "blue", "green", "yellow", "orange", "purple", "violet",
    "pink", "black", "white", "gold", "silver", "crimson", "azure",
    "emerald", "scarlet", "golden", "dark", "light", "bright", "dim",
    # Nature
    "sun", "moon", "star", "stars", "sky", "cloud", "clouds", "rain",
    "storm", "thunder", "lightning", "snow", "ice", "fire", "flame",
    "flames", "water", "ocean", "sea", "river", "lake", "mountain",
    "forest", "tree", "trees", "flower", "flowers", "rose", "roses",
    "garden", "earth", "wind", "air", "night", "day", "dawn", "dusk",
    "sunset", "sunrise", "shadow", "shadows", "light", "darkness",
    # Body/physical
    "eyes", "eye", "heart", "hearts", "soul", "body", "face", "hand",
    "hands", "arms", "wings", "blood", "tears", "smile", "kiss",
    # Movement/action (visual)
    "dance", "dancing", "fly", "flying", "fall", "falling", "run",
    "running", "jump", "spinning", "floating", "rising", "sinking",
    "burning", "shining", "glowing", "flowing", "breaking", "crashing",
    # Atmosphere
    "dream", "dreams", "dreaming", "magic", "magical", "mystical",
    "ethereal", "cosmic", "electric", "neon", "crystal", "glass",
    "mirror", "smoke", "mist", "fog", "haze", "glow", "shine",
    "sparkle", "glitter", "shimmer",
    # Abstract but visual
    "love", "hate", "pain", "joy", "hope", "fear", "rage", "peace",
    "chaos", "silence", "scream", "whisper", "echo", "memory",
    "time", "forever", "eternity", "infinity", "void", "abyss",
})

# Emotion words that add mood/atmosphere
EMOTION_WORDS = frozenset({
    "love", "loving", "loved", "hate", "hating", "hated",
    "happy", "happiness", "sad", "sadness", "sorrow",
    "angry", "anger", "rage", "fury", "furious",
    "fear", "afraid", "scared", "terrified", "terror",
    "joy", "joyful", "joyous", "bliss", "blissful",
    "pain", "painful", "hurt", "hurting", "ache", "aching",
    "hope", "hopeful", "hopeless", "despair", "desperate",
    "lonely", "loneliness", "alone", "lost",
    "free", "freedom", "wild", "crazy", "insane",
    "beautiful", "beauty", "ugly", "broken", "whole",
    "alive", "dead", "dying", "living", "life", "death",
    "strong", "strength", "weak", "weakness", "power", "powerful",
    "peace", "peaceful", "calm", "chaos", "chaotic",
    "desire", "passion", "passionate", "intense", "intensity",
})


class KeywordExtractor:
    """Extracts visual keywords from lyrics text.

    Filters out stop words and prioritizes visually evocative words
    that work well for image generation prompts.
    """

    def __init__(self, config: Optional[KeywordExtractionConfig] = None):
        self.config = config or KeywordExtractionConfig()

    def extract(self, text: str) -> list[tuple[str, float]]:
        """Extract visual keywords with weights from lyrics text.

        Args:
            text: Raw lyrics text

        Returns:
            List of (keyword, weight) tuples, sorted by weight descending
        """
        if not text:
            return []

        # Normalize text
        text = text.lower()
        # Remove punctuation but keep spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Split into words
        words = text.split()

        # Count word frequency
        word_counts: dict[str, int] = {}
        for word in words:
            if (
                len(word) >= self.config.min_word_length
                and word not in STOP_WORDS
                and word not in self.config.filter_words
            ):
                word_counts[word] = word_counts.get(word, 0) + 1

        # Calculate weights
        results: list[tuple[str, float]] = []

        for word, count in word_counts.items():
            # Base weight from frequency (normalized)
            weight = self.config.base_weight

            # Boost for frequency (repeated words are emphasized in lyrics)
            if count > 1:
                weight += 0.1 * min(count - 1, 3)  # Cap at +0.3 for 4+ occurrences

            # Boost for visual words
            if word in VISUAL_WORDS:
                weight += self.config.visual_boost

            # Boost for emotion words
            if word in EMOTION_WORDS:
                weight += self.config.emotion_boost

            results.append((word, weight))

        # Sort by weight descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top keywords
        return results[: self.config.max_keywords]

    def text_to_prompt(self, text: str) -> str:
        """Convert raw lyrics to a visual prompt (experimental lyric-driven mode).

        Creates a coherent prompt from the most evocative words in the lyrics.

        Args:
            text: Raw lyrics text

        Returns:
            A prompt string suitable for image generation
        """
        keywords = self.extract(text)

        if not keywords:
            return "abstract visual, ethereal atmosphere"

        # Build prompt from top keywords with weights
        prompt_parts = []
        for word, weight in keywords[:6]:  # Use top 6 keywords
            if weight > 1.1:
                # Add weight syntax for emphasized words
                prompt_parts.append(f"({word}:{weight:.2f})")
            else:
                prompt_parts.append(word)

        # Add some visual connectors
        base_prompt = ", ".join(prompt_parts)

        # Add artistic style suffix
        style_suffix = "artistic visualization, dreamlike atmosphere, cinematic"

        return f"{base_prompt}, {style_suffix}"

    def filter_for_visuals(self, words: list[str]) -> list[str]:
        """Filter a word list to only include visually relevant words.

        Args:
            words: List of words to filter

        Returns:
            List of visually relevant words
        """
        return [
            word
            for word in words
            if word.lower() in VISUAL_WORDS or word.lower() in EMOTION_WORDS
        ]
