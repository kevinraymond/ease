/**
 * Typed Event Bus for cross-component communication.
 * Provides a simple pub/sub pattern for decoupled event handling.
 */

type EventCallback<T = unknown> = (data: T) => void;

export type { EventCallback };

export class EventBus {
  private listeners = new Map<string, Set<EventCallback>>();

  /**
   * Subscribe to an event
   * @returns Unsubscribe function
   */
  on<T>(event: string, callback: EventCallback<T>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback as EventCallback);

    // Return unsubscribe function
    return () => {
      this.off(event, callback);
    };
  }

  /**
   * Subscribe to an event once
   */
  once<T>(event: string, callback: EventCallback<T>): () => void {
    const wrapper = (data: T) => {
      this.off(event, wrapper as EventCallback);
      callback(data);
    };
    return this.on(event, wrapper);
  }

  /**
   * Unsubscribe from an event
   */
  off<T>(event: string, callback: EventCallback<T>): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.delete(callback as EventCallback);
      if (callbacks.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  /**
   * Emit an event with data
   */
  emit<T>(event: string, data?: T): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach((callback) => {
        try {
          callback(data);
        } catch (e) {
          console.error(`Error in event handler for "${event}":`, e);
        }
      });
    }
  }

  /**
   * Remove all listeners for an event (or all events if no event specified)
   */
  clear(event?: string): void {
    if (event) {
      this.listeners.delete(event);
    } else {
      this.listeners.clear();
    }
  }

  /**
   * Check if an event has any listeners
   */
  hasListeners(event: string): boolean {
    return (this.listeners.get(event)?.size ?? 0) > 0;
  }
}

// Global event bus instance for app-wide events
export const eventBus = new EventBus();

// Common event types for type safety
export interface AppEvents {
  // Audio events
  'audio:loaded': { name: string; duration: number };
  'audio:play': void;
  'audio:pause': void;
  'audio:stop': void;
  'audio:seek': number;
  'audio:timeUpdate': { currentTime: number; duration: number };
  'audio:metrics': import('../audio/types').AudioMetrics;
  'audio:beatDebug': import('../audio/types').BeatDebugInfo;

  // Visualizer events
  'visualizer:backendChanged': 'webgpu' | 'webgl2';

  // AI Generator events
  'ai:connected': void;
  'ai:disconnected': void;
  'ai:frameReceived': { frameId: number; frame: HTMLImageElement };
  'ai:stateChanged': import('./types').AIGeneratorState;
  'ai:configChanged': Partial<import('./types').AIGeneratorConfig>;

  // UI events
  'ui:togglePanel': string;
}
