/**
 * CollapsibleSection - Reusable collapsible section with localStorage persistence.
 */

import { Component } from '../core/Component';
import { html, raw } from '../core/html';

interface CollapsibleSectionState {
  isExpanded: boolean;
}

interface CollapsibleSectionOptions {
  title: string;
  storageKey: string;
  defaultExpanded?: boolean;
  badge?: string;
  content: string; // HTML content to render inside
  onToggle?: (isExpanded: boolean) => void;
}

export class CollapsibleSection extends Component<CollapsibleSectionState> {
  private options: CollapsibleSectionOptions;
  private localStorageKey: string;

  constructor(container: HTMLElement, options: CollapsibleSectionOptions) {
    const localStorageKey = `ai-panel-section-${options.storageKey}`;
    const stored = localStorage.getItem(localStorageKey);
    const isExpanded = stored !== null ? stored === 'true' : (options.defaultExpanded ?? false);

    super(container, { isExpanded });
    this.options = options;
    this.localStorageKey = localStorageKey;
  }

  protected render(): void {
    const { title, badge } = this.options;
    const { isExpanded } = this.state;

    this.el.className = `collapsible-section ${isExpanded ? 'expanded' : ''}`;
    this.el.innerHTML = html`
      <button
        class="collapsible-header ${isExpanded ? 'expanded' : ''}"
        data-action="toggle"
      >
        <span class="collapse-arrow ${isExpanded ? 'expanded' : ''}">▶</span>
        <span class="collapsible-title">${title}</span>
        ${badge && !isExpanded ? html`<span class="collapsible-badge">${badge}</span>` : ''}
      </button>
      ${isExpanded ? html`<div class="collapsible-content">${raw(this.options.content)}</div>` : ''}
    `;
  }

  protected actions = {
    toggle: () => {
      const newValue = !this.state.isExpanded;
      localStorage.setItem(this.localStorageKey, String(newValue));
      this.state.isExpanded = newValue;
      this.options.onToggle?.(newValue);
    },
  };

  // Update the content without re-rendering the whole section
  updateContent(content: string): void {
    this.options.content = content;
    if (this.state.isExpanded) {
      const contentEl = this.$('.collapsible-content');
      if (contentEl) {
        contentEl.innerHTML = content;
      }
    }
  }

  // Update the badge
  updateBadge(badge?: string): void {
    this.options.badge = badge;
    if (!this.state.isExpanded) {
      this.render();
    }
  }

  // Check if expanded
  isExpanded(): boolean {
    return this.state.isExpanded;
  }

  // Expand/collapse programmatically
  setExpanded(expanded: boolean): void {
    if (this.state.isExpanded !== expanded) {
      this.state.isExpanded = expanded;
      localStorage.setItem(this.localStorageKey, String(expanded));
    }
  }
}

/**
 * Helper to create a collapsible section as an HTML string for embedding.
 * Returns the outer container element ID so you can mount a CollapsibleSection to it.
 */
let sectionIdCounter = 0;

export function createCollapsibleSectionContainer(): { id: string; html: string } {
  const id = `collapsible-section-${++sectionIdCounter}`;
  return {
    id,
    html: `<div id="${id}"></div>`,
  };
}
