/**
 * LyricScheduler - Beat-Aligned Lyric Prompt Injection
 *
 * This module queues lyric-derived keywords and applies them on beat boundaries
 * to create perceptually synchronized visual changes. The key insight is that
 * a 2-beat delay (masked by being on-beat) feels more "in time" than an
 * arbitrary sub-beat delay.
 *
 * ## How It Works
 *
 * ```
 * Lyric Detection (async)          Beat Detection (real-time)
 *        │                                   │
 *        ▼                                   ▼
 * ┌──────────────┐                  ┌────────────────┐
 * │ Keywords     │──enqueue──▶      │ BeatScheduler  │
 * │ (with delay) │                  │ (isBeat signal)│
 * └──────────────┘                  └────────────────┘
 *                                            │
 *                                   dequeue on beat
 *                                            ▼
 *                                   ┌────────────────┐
 *                                   │ Active Keywords│
 *                                   │ (for visuals)  │
 *                                   └────────────────┘
 * ```
 *
 * ## Usage
 *
 * ```ts
 * const scheduler = new LyricScheduler({ beatDelay: 2 });
 *
 * // When lyrics are detected (from WebSocket):
 * scheduler.enqueue(['fire', 'night', 'dance']);
 *
 * // In your render loop, on each beat:
 * if (beatInfo.isBeat) {
 *   scheduler.onBeat();
 *   const keywords = scheduler.getActiveKeywords();
 *   // Use keywords for prompt modulation
 * }
 * ```
 *
 * ## Why Beat-Aligned?
 *
 * Whisper needs 2-5 seconds of audio context for accurate transcription,
 * creating an inherent delay. Rather than showing lyrics "late" (which
 * feels out of sync), we delay to the next beat boundary. The human brain
 * perceives on-beat events as synchronized, even with modest delays.
 *
 * A 2-beat delay at 120 BPM = 1 second, which typically still feels "with"
 * the music rather than "behind" it.
 */

export interface LyricSchedulerConfig {
  /**
   * Number of beats to delay keyword application.
   * Higher values increase perceived sync at the cost of responsiveness.
   * Default: 2 beats
   */
  beatDelay: number;

  /**
   * Maximum number of queued keyword sets.
   * Older entries are dropped if queue overflows.
   * Default: 8
   */
  maxQueueSize: number;

  /**
   * How long keywords remain "active" after application (in beats).
   * Keywords fade out after this many beats without new input.
   * Default: 4 beats
   */
  keywordLifetimeBeats: number;

  /**
   * Whether to enable the scheduler.
   * When disabled, keywords are applied immediately without beat alignment.
   * Default: true
   */
  enabled: boolean;
}

interface QueuedKeywords {
  keywords: string[];
  enqueuedAtBeat: number;
  targetBeat: number;
}

const DEFAULT_CONFIG: LyricSchedulerConfig = {
  beatDelay: 2,
  maxQueueSize: 8,
  keywordLifetimeBeats: 4,
  enabled: true,
};

export class LyricScheduler {
  private config: LyricSchedulerConfig;

  // Queue of pending keyword sets
  private queue: QueuedKeywords[] = [];

  // Currently active keywords (applied on beat)
  private activeKeywords: string[] = [];

  // Beat counter for timing
  private beatCount: number = 0;

  // When current keywords were applied (beat number)
  private keywordsAppliedAtBeat: number = 0;

  // Callback when keywords change
  private onKeywordsChange?: (keywords: string[]) => void;

  /**
   * Creates a new LyricScheduler.
   *
   * @param config - Partial configuration (defaults used for unspecified values)
   */
  constructor(config: Partial<LyricSchedulerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Enqueues a set of keywords for beat-aligned application.
   *
   * Keywords will be applied after `beatDelay` beats from the next beat.
   *
   * @param keywords - Array of visual keywords extracted from lyrics
   */
  enqueue(keywords: string[]): void {
    if (!keywords || keywords.length === 0) {
      return;
    }

    // If scheduler is disabled, apply immediately
    if (!this.config.enabled) {
      this.applyKeywords(keywords);
      return;
    }

    const targetBeat = this.beatCount + this.config.beatDelay;

    // Add to queue
    this.queue.push({
      keywords,
      enqueuedAtBeat: this.beatCount,
      targetBeat,
    });

    // Trim queue if too large (drop oldest entries)
    while (this.queue.length > this.config.maxQueueSize) {
      this.queue.shift();
    }
  }

  /**
   * Called on each beat to process the queue.
   *
   * This should be called from your render loop whenever `beatInfo.isBeat` is true.
   */
  onBeat(): void {
    this.beatCount++;

    // Check for keywords ready to apply
    const readyIndex = this.queue.findIndex(
      (item) => this.beatCount >= item.targetBeat
    );

    if (readyIndex !== -1) {
      // Apply the most recent ready keywords (skip older ones)
      const ready = this.queue.splice(0, readyIndex + 1);
      const latest = ready[ready.length - 1];

      if (latest) {
        this.applyKeywords(latest.keywords);
      }
    }

    // Check for keyword expiration
    const beatsSinceApplied = this.beatCount - this.keywordsAppliedAtBeat;
    if (
      this.activeKeywords.length > 0 &&
      beatsSinceApplied > this.config.keywordLifetimeBeats &&
      this.queue.length === 0
    ) {
      // Keywords have expired and nothing new is coming
      // Keep them fading rather than clearing abruptly
      // (actual fade implementation is in the visualizer)
    }
  }

  /**
   * Applies keywords immediately (internal).
   */
  private applyKeywords(keywords: string[]): void {
    this.activeKeywords = keywords;
    this.keywordsAppliedAtBeat = this.beatCount;

    // Notify callback if set
    if (this.onKeywordsChange) {
      this.onKeywordsChange(keywords);
    }
  }

  /**
   * Returns the currently active keywords.
   *
   * These are the keywords that should be used for visual prompt modulation.
   */
  getActiveKeywords(): string[] {
    return this.activeKeywords;
  }

  /**
   * Returns the "freshness" of current keywords (0-1).
   *
   * 1.0 = just applied this beat
   * 0.0 = at or past keyword lifetime
   *
   * Use this for fading effects.
   */
  getKeywordFreshness(): number {
    if (this.activeKeywords.length === 0) {
      return 0;
    }

    const beatsSinceApplied = this.beatCount - this.keywordsAppliedAtBeat;
    const freshness = 1 - beatsSinceApplied / this.config.keywordLifetimeBeats;
    return Math.max(0, Math.min(1, freshness));
  }

  /**
   * Returns whether there are pending keywords in the queue.
   */
  hasPendingKeywords(): boolean {
    return this.queue.length > 0;
  }

  /**
   * Returns the number of beats until the next keyword application.
   * Returns -1 if queue is empty.
   */
  getBeatsUntilNextKeywords(): number {
    if (this.queue.length === 0) {
      return -1;
    }

    const next = this.queue[0];
    return Math.max(0, next.targetBeat - this.beatCount);
  }

  /**
   * Sets a callback for when keywords change.
   *
   * @param callback - Function called with new keywords when they're applied
   */
  setOnKeywordsChange(callback: (keywords: string[]) => void): void {
    this.onKeywordsChange = callback;
  }

  /**
   * Clears all pending keywords and resets state.
   */
  clear(): void {
    this.queue = [];
    this.activeKeywords = [];
    this.keywordsAppliedAtBeat = 0;
  }

  /**
   * Resets the scheduler completely.
   */
  reset(): void {
    this.clear();
    this.beatCount = 0;
  }

  /**
   * Updates configuration.
   *
   * @param config - Partial config with values to update
   */
  configure(config: Partial<LyricSchedulerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Returns the current configuration.
   */
  getConfig(): LyricSchedulerConfig {
    return { ...this.config };
  }

  /**
   * Returns the current beat count (for debugging).
   */
  getBeatCount(): number {
    return this.beatCount;
  }

  /**
   * Returns the queue length (for debugging).
   */
  getQueueLength(): number {
    return this.queue.length;
  }
}

export type { LyricSchedulerConfig as LyricSchedulerOptions };
