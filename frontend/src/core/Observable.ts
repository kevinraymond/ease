/**
 * Proxy-based observable state wrapper.
 * Triggers callbacks when state properties are changed.
 */

type ChangeCallback<T> = (newValue: T, oldValue: T, key: keyof T) => void;

/**
 * Create an observable state object that triggers callbacks on change.
 *
 * @param initial - Initial state object
 * @param onChange - Callback triggered when any property changes
 * @returns Proxy-wrapped state object
 */
export function observable<T extends object>(
  initial: T,
  onChange: ChangeCallback<T>
): T {
  return new Proxy(initial, {
    set(target, prop, value) {
      const key = prop as keyof T;
      const oldValue = { ...target };
      (target as any)[prop] = value;

      // Only trigger change if value actually changed
      if (oldValue[key] !== value) {
        onChange(target, oldValue, key);
      }

      return true;
    },
    get(target, prop) {
      return (target as any)[prop];
    },
  });
}

/**
 * Create a deep observable that handles nested objects.
 * Changes to nested properties also trigger the callback.
 */
export function deepObservable<T extends object>(
  initial: T,
  onChange: () => void
): T {
  const createProxy = <U extends object>(obj: U): U => {
    return new Proxy(obj, {
      set(target, prop, value) {
        const oldValue = (target as any)[prop];
        (target as any)[prop] = value;

        // Wrap nested objects
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          (target as any)[prop] = createProxy(value);
        }

        if (oldValue !== value) {
          onChange();
        }

        return true;
      },
      get(target, prop) {
        const value = (target as any)[prop];
        // Wrap nested objects on access
        if (value && typeof value === 'object' && !Array.isArray(value) && !(value instanceof Proxy)) {
          (target as any)[prop] = createProxy(value);
          return (target as any)[prop];
        }
        return value;
      },
    });
  };

  return createProxy(initial);
}

/**
 * Batch multiple state updates into a single render cycle.
 */
export class BatchUpdater {
  private scheduled = false;
  private callbacks: Set<() => void> = new Set();

  /**
   * Schedule a callback to run in the next microtask.
   * Multiple calls before the microtask runs will be batched.
   */
  schedule(callback: () => void): void {
    this.callbacks.add(callback);

    if (!this.scheduled) {
      this.scheduled = true;
      queueMicrotask(() => {
        this.scheduled = false;
        const toRun = [...this.callbacks];
        this.callbacks.clear();
        toRun.forEach((cb) => {
          try {
            cb();
          } catch (e) {
            console.error('Error in batched update:', e);
          }
        });
      });
    }
  }
}

// Global batch updater for coordinated renders
export const batchUpdater = new BatchUpdater();
