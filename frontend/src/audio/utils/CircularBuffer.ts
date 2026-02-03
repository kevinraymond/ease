/**
 * CircularBuffer - An efficient fixed-size ring buffer for audio signal processing.
 *
 * This data structure is optimized for scenarios where we need to maintain
 * a sliding window of recent values (e.g., energy history for adaptive thresholds).
 * Unlike arrays with shift(), this implementation has O(1) push and access operations.
 *
 * @example
 * ```ts
 * const buffer = new CircularBuffer(10);
 * buffer.push(1.0);
 * buffer.push(2.0);
 * console.log(buffer.median()); // Statistical median of values
 * ```
 */
export class CircularBuffer {
  /** Internal storage array (pre-allocated for performance) */
  private readonly buffer: number[];
  /** Maximum capacity of the buffer */
  private readonly capacity: number;
  /** Current write position (wraps around) */
  private writeIndex: number = 0;
  /** Number of values currently stored (0 to capacity) */
  private count: number = 0;

  /**
   * Creates a new CircularBuffer with the specified capacity.
   *
   * @param capacity - Maximum number of values to store. Once full,
   *                   new values overwrite the oldest values.
   */
  constructor(capacity: number) {
    if (capacity <= 0) {
      throw new Error('CircularBuffer capacity must be positive');
    }
    this.capacity = capacity;
    // Pre-allocate array filled with zeros for consistent memory layout
    this.buffer = new Array(capacity).fill(0);
  }

  /**
   * Adds a value to the buffer. If the buffer is full, the oldest value
   * is overwritten (FIFO behavior).
   *
   * Time complexity: O(1)
   *
   * @param value - The numeric value to add
   */
  push(value: number): void {
    this.buffer[this.writeIndex] = value;
    this.writeIndex = (this.writeIndex + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    }
  }

  /**
   * Retrieves a value by index, where 0 is the oldest value and
   * (length - 1) is the most recent.
   *
   * Time complexity: O(1)
   *
   * @param index - Index from 0 (oldest) to length-1 (newest)
   * @returns The value at that position, or 0 if index is out of bounds
   */
  get(index: number): number {
    if (index < 0 || index >= this.count) {
      return 0;
    }
    // Calculate actual position accounting for circular wrapping
    // readIndex points to the oldest element
    const readIndex = (this.writeIndex - this.count + this.capacity) % this.capacity;
    const actualIndex = (readIndex + index) % this.capacity;
    return this.buffer[actualIndex];
  }

  /**
   * Returns the most recently added value.
   *
   * Time complexity: O(1)
   *
   * @returns The newest value, or 0 if buffer is empty
   */
  getLast(): number {
    if (this.count === 0) return 0;
    // writeIndex points to next write position, so last written is one before
    const lastIndex = (this.writeIndex - 1 + this.capacity) % this.capacity;
    return this.buffer[lastIndex];
  }

  /**
   * Returns the value that was added `n` steps ago.
   *
   * Time complexity: O(1)
   *
   * @param n - Number of steps back (0 = most recent, 1 = second most recent, etc.)
   * @returns The value n steps back, or 0 if n exceeds available history
   */
  getFromEnd(n: number): number {
    if (n < 0 || n >= this.count) return 0;
    const index = (this.writeIndex - 1 - n + this.capacity * 2) % this.capacity;
    return this.buffer[index];
  }

  /**
   * Converts the buffer to a standard array, ordered from oldest to newest.
   *
   * Time complexity: O(n)
   *
   * @returns A new array containing all values in chronological order
   */
  toArray(): number[] {
    const result: number[] = new Array(this.count);
    for (let i = 0; i < this.count; i++) {
      result[i] = this.get(i);
    }
    return result;
  }

  /**
   * Returns the number of values currently stored.
   */
  get length(): number {
    return this.count;
  }

  /**
   * Returns true if the buffer has reached its maximum capacity.
   */
  isFull(): boolean {
    return this.count === this.capacity;
  }

  /**
   * Resets the buffer to empty state without reallocating memory.
   */
  clear(): void {
    this.writeIndex = 0;
    this.count = 0;
    // Optional: zero out the array for cleaner debugging
    this.buffer.fill(0);
  }

  // ============================================
  // Statistical Methods for Audio Signal Analysis
  // ============================================

  /**
   * Calculates the arithmetic mean of all values in the buffer.
   *
   * Used for baseline energy levels in beat detection.
   *
   * Time complexity: O(n)
   *
   * @returns The average value, or 0 if buffer is empty
   */
  mean(): number {
    if (this.count === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.count; i++) {
      sum += this.get(i);
    }
    return sum / this.count;
  }

  /**
   * Calculates the median (middle value) of all values in the buffer.
   *
   * The median is more robust to outliers than the mean, making it ideal
   * for adaptive thresholds in beat detection where occasional loud transients
   * shouldn't skew the baseline.
   *
   * Time complexity: O(n log n) due to sorting
   *
   * @returns The median value, or 0 if buffer is empty
   */
  median(): number {
    if (this.count === 0) return 0;

    // Create sorted copy (don't modify internal state)
    const sorted = this.toArray().sort((a, b) => a - b);

    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      // Even number of elements: average the two middle values
      return (sorted[mid - 1] + sorted[mid]) / 2;
    } else {
      // Odd number of elements: return the middle value
      return sorted[mid];
    }
  }

  /**
   * Calculates the Median Absolute Deviation (MAD).
   *
   * MAD is a robust measure of statistical dispersion. Unlike standard deviation,
   * it's not heavily influenced by outliers, making it perfect for adaptive
   * beat detection thresholds.
   *
   * Formula: MAD = median(|Xi - median(X)|)
   *
   * In beat detection, the adaptive threshold is typically:
   *   threshold = median + k * MAD
   * where k is a sensitivity multiplier (typically 2.0-3.0)
   *
   * Time complexity: O(n log n)
   *
   * @returns The median absolute deviation, or 0 if buffer is empty
   */
  mad(): number {
    if (this.count === 0) return 0;

    const med = this.median();
    const deviations: number[] = new Array(this.count);

    // Calculate absolute deviations from median
    for (let i = 0; i < this.count; i++) {
      deviations[i] = Math.abs(this.get(i) - med);
    }

    // Sort and find median of deviations
    deviations.sort((a, b) => a - b);
    const mid = Math.floor(deviations.length / 2);

    if (deviations.length % 2 === 0) {
      return (deviations[mid - 1] + deviations[mid]) / 2;
    } else {
      return deviations[mid];
    }
  }

  /**
   * Calculates the standard deviation of values in the buffer.
   *
   * While less robust than MAD, standard deviation is useful for
   * calculating confidence scores and analyzing beat interval consistency.
   *
   * Time complexity: O(n)
   *
   * @returns The population standard deviation, or 0 if buffer is empty
   */
  stdDev(): number {
    if (this.count === 0) return 0;

    const avg = this.mean();
    let sumSquaredDiff = 0;

    for (let i = 0; i < this.count; i++) {
      const diff = this.get(i) - avg;
      sumSquaredDiff += diff * diff;
    }

    return Math.sqrt(sumSquaredDiff / this.count);
  }

  /**
   * Returns the maximum value in the buffer.
   *
   * Time complexity: O(n)
   */
  max(): number {
    if (this.count === 0) return 0;
    let maxVal = this.get(0);
    for (let i = 1; i < this.count; i++) {
      const val = this.get(i);
      if (val > maxVal) maxVal = val;
    }
    return maxVal;
  }

  /**
   * Returns the minimum value in the buffer.
   *
   * Time complexity: O(n)
   */
  min(): number {
    if (this.count === 0) return 0;
    let minVal = this.get(0);
    for (let i = 1; i < this.count; i++) {
      const val = this.get(i);
      if (val < minVal) minVal = val;
    }
    return minVal;
  }

  /**
   * Calculates the sum of all values in the buffer.
   *
   * Time complexity: O(n)
   */
  sum(): number {
    let total = 0;
    for (let i = 0; i < this.count; i++) {
      total += this.get(i);
    }
    return total;
  }
}
