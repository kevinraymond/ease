/**
 * Base Component class for vanilla TypeScript UI components.
 * Provides lifecycle management, event handling, and state management.
 */

import { EventBus, eventBus } from './EventBus';
import { batchUpdater } from './Observable';

// Function to cleanup subscription
type Unsubscribe = () => void;

export abstract class Component<TState extends object = Record<string, never>> {
  /** The root DOM element for this component */
  protected el: HTMLElement;

  /** Component state - changes trigger re-render */
  protected state: TState;

  /** Local event bus for component-specific events */
  protected localBus = new EventBus();

  /** Subscriptions to clean up on dispose */
  private subscriptions: Unsubscribe[] = [];

  /** Whether the component has been mounted */
  private mounted = false;

  /** Whether the component is currently rendering */
  private rendering = false;

  /** Flag to indicate a render is pending */
  private renderPending = false;

  constructor(container: HTMLElement | string, initialState: TState) {
    // Resolve container element
    if (typeof container === 'string') {
      const element = document.querySelector(container);
      if (!element) {
        throw new Error(`Container element not found: ${container}`);
      }
      this.el = element as HTMLElement;
    } else {
      this.el = container;
    }

    // Set up reactive state
    this.state = new Proxy(initialState, {
      set: (target, prop, value) => {
        const oldValue = (target as any)[prop];
        (target as any)[prop] = value;

        // Schedule re-render if value changed
        if (oldValue !== value && this.mounted) {
          this.scheduleRender();
        }

        return true;
      },
    });
  }

  /**
   * Initialize the component and mount it to the DOM.
   * Call this after construction to trigger initial render.
   */
  mount(): this {
    if (this.mounted) return this;

    this.mounted = true;
    this.render();
    this.setupEventDelegation();
    this.onMount?.();

    return this;
  }

  /**
   * Clean up the component and remove from DOM.
   */
  dispose(): void {
    if (!this.mounted) return;

    this.onDispose?.();

    // Cleanup all subscriptions
    this.subscriptions.forEach((unsub) => unsub());
    this.subscriptions = [];

    // Clear local event bus
    this.localBus.clear();

    // Clear element content
    this.el.innerHTML = '';

    this.mounted = false;
  }

  /**
   * Override to render component HTML.
   * Called on mount and whenever state changes.
   */
  protected abstract render(): void;

  /**
   * Override for setup after initial render.
   */
  protected onMount?(): void;

  /**
   * Override for cleanup before unmount.
   */
  protected onDispose?(): void;

  /**
   * Define action handlers for event delegation.
   * Keys are data-action values, values are handler functions.
   */
  protected actions?: Record<string, (e: Event, target: HTMLElement) => void>;

  /**
   * Schedule a batched render update.
   */
  protected scheduleRender(): void {
    if (this.renderPending) return;
    this.renderPending = true;

    batchUpdater.schedule(() => {
      this.renderPending = false;
      if (this.mounted && !this.rendering) {
        this.render();
      }
    });
  }

  /**
   * Force immediate re-render (bypasses batching).
   */
  protected forceRender(): void {
    if (this.mounted && !this.rendering) {
      this.render();
    }
  }

  /**
   * Subscribe to global event bus.
   * Subscription is automatically cleaned up on dispose.
   */
  protected on<T>(event: string, callback: (data: T) => void): Unsubscribe {
    const unsub = eventBus.on(event, callback);
    this.subscriptions.push(unsub);
    return unsub;
  }

  /**
   * Emit event to global event bus.
   */
  protected emit<T>(event: string, data?: T): void {
    eventBus.emit(event, data);
  }

  /**
   * Check if this component's element is still connected to the DOM.
   * Useful for detecting when a parent re-render has orphaned this component.
   */
  get isConnected(): boolean {
    return this.el.isConnected;
  }

  /**
   * Query a single element within this component.
   */
  protected $(selector: string): HTMLElement | null {
    return this.el.querySelector(selector);
  }

  /**
   * Query all elements within this component.
   */
  protected $$(selector: string): NodeListOf<HTMLElement> {
    return this.el.querySelectorAll(selector);
  }

  /**
   * Set up event delegation for action handlers.
   * Looks for elements with data-action attributes.
   */
  private setupEventDelegation(): void {
    if (!this.actions) return;

    // Click handler - only for non-form elements (buttons, divs, etc.)
    this.el.addEventListener('click', (e) => {
      const target = (e.target as HTMLElement).closest('[data-action]') as HTMLElement | null;
      if (target) {
        const tagName = target.tagName.toLowerCase();
        // Skip form inputs - they use input/change events instead
        if (tagName === 'input' || tagName === 'textarea' || tagName === 'select') {
          return;
        }
        const action = target.dataset.action;
        if (action && this.actions?.[action]) {
          e.preventDefault();
          this.actions[action](e, target);
        }
      }
    });

    // Change handler for inputs
    this.el.addEventListener('change', (e) => {
      const target = (e.target as HTMLElement).closest('[data-action]') as HTMLElement | null;
      if (target) {
        const action = target.dataset.action;
        if (action && this.actions?.[action]) {
          this.actions[action](e, target);
        }
      }
    });

    // Input handler for real-time updates
    this.el.addEventListener('input', (e) => {
      const target = (e.target as HTMLElement).closest('[data-action]') as HTMLElement | null;
      if (target) {
        const action = target.dataset.action;
        if (action && this.actions?.[action]) {
          this.actions[action](e, target);
        }
      }
    });
  }

  /**
   * Update a subset of state properties.
   */
  protected setState(partial: Partial<TState>): void {
    Object.assign(this.state, partial);
  }

  /**
   * Add a DOM event listener that's cleaned up on dispose.
   */
  protected listen<K extends keyof HTMLElementEventMap>(
    target: EventTarget,
    event: K,
    handler: (e: HTMLElementEventMap[K]) => void,
    options?: AddEventListenerOptions
  ): Unsubscribe {
    target.addEventListener(event, handler as EventListener, options);
    const unsub = () => target.removeEventListener(event, handler as EventListener, options);
    this.subscriptions.push(unsub);
    return unsub;
  }

  /**
   * Add a window event listener that's cleaned up on dispose.
   */
  protected listenWindow<K extends keyof WindowEventMap>(
    event: K,
    handler: (e: WindowEventMap[K]) => void,
    options?: AddEventListenerOptions
  ): Unsubscribe {
    window.addEventListener(event, handler as EventListener, options);
    const unsub = () => window.removeEventListener(event, handler as EventListener, options);
    this.subscriptions.push(unsub);
    return unsub;
  }
}

/**
 * Modal component base class with show/hide functionality.
 * Modals start hidden by default.
 */
export abstract class ModalComponent<TState extends object = Record<string, never>> extends Component<TState> {
  protected isOpen = false;

  /**
   * Override mount to ensure modal starts hidden.
   */
  override mount(): this {
    super.mount();
    // Ensure modal starts hidden
    this.el.style.display = 'none';
    return this;
  }

  show(): void {
    this.isOpen = true;
    this.el.classList.add('visible');
    this.el.style.display = '';
    document.body.style.overflow = 'hidden';
    this.forceRender(); // Re-render to update content
    this.onShow?.();
  }

  hide(): void {
    this.isOpen = false;
    this.el.classList.remove('visible');
    this.el.style.display = 'none';
    document.body.style.overflow = '';
    this.onHide?.();
  }

  toggle(): void {
    if (this.isOpen) {
      this.hide();
    } else {
      this.show();
    }
  }

  protected onShow?(): void;
  protected onHide?(): void;

  protected onMount(): void {
    // Close on overlay click
    this.listen(this.el, 'click', (e) => {
      if (e.target === this.el) {
        this.hide();
      }
    });

    // Close on Escape key
    this.listenWindow('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen) {
        this.hide();
      }
    });
  }
}
