/**
 * Template literal helper for safe HTML rendering.
 * Automatically escapes interpolated values to prevent XSS.
 */

// Marker for trusted HTML that shouldn't be escaped
const TRUSTED_HTML = Symbol('trustedHTML');

interface TrustedHTML {
  [TRUSTED_HTML]: true;
  html: string;
}

/**
 * Mark a string as trusted HTML (won't be escaped).
 * Use with caution - only for HTML you control.
 */
export function raw(html: string): TrustedHTML {
  return { [TRUSTED_HTML]: true, html };
}

/**
 * Check if a value is trusted HTML
 */
function isTrustedHTML(value: unknown): value is TrustedHTML {
  return (
    typeof value === 'object' &&
    value !== null &&
    TRUSTED_HTML in value &&
    (value as TrustedHTML)[TRUSTED_HTML] === true
  );
}

/**
 * Escape HTML special characters to prevent XSS
 */
function escapeHTML(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Format a value for HTML output
 */
function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }
  if (isTrustedHTML(value)) {
    return value.html;
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : '';
  }
  if (Array.isArray(value)) {
    return value.map(formatValue).join('');
  }
  return escapeHTML(String(value));
}

// Create a TrustedHTML string class that works with innerHTML assignment
class TrustedHTMLString extends String implements TrustedHTML {
  [TRUSTED_HTML] = true as const;
  html: string;

  constructor(value: string) {
    super(value);
    this.html = value;
  }
}

/**
 * Tagged template literal for HTML with auto-escaping.
 * Returns a TrustedHTML object so nested html calls work correctly.
 *
 * @example
 * const name = '<script>alert("xss")</script>';
 * element.innerHTML = html`<div>Hello, ${name}!</div>`;
 * // Result: <div>Hello, &lt;script&gt;alert("xss")&lt;/script&gt;!</div>
 *
 * @example
 * // Use raw() for trusted HTML
 * const icon = raw('<svg>...</svg>');
 * element.innerHTML = html`<button>${icon}</button>`;
 *
 * @example
 * // Nested html calls work correctly
 * const inner = html`<span>inner</span>`;
 * element.innerHTML = html`<div>${inner}</div>`;
 */
export function html(strings: TemplateStringsArray, ...values: unknown[]): string {
  let result = strings[0];

  for (let i = 0; i < values.length; i++) {
    result += formatValue(values[i]);
    result += strings[i + 1];
  }

  // Return a TrustedHTMLString that works with innerHTML and won't be escaped when nested
  return new TrustedHTMLString(result) as unknown as string;
}

/**
 * Create an element from an HTML string.
 * Returns the first element in the string.
 */
export function createElement<T extends HTMLElement = HTMLElement>(htmlString: string): T {
  const template = document.createElement('template');
  template.innerHTML = htmlString.trim();
  return template.content.firstChild as T;
}

/**
 * Create multiple elements from an HTML string.
 */
export function createElements(htmlString: string): NodeList {
  const template = document.createElement('template');
  template.innerHTML = htmlString.trim();
  return template.content.childNodes;
}

/**
 * SVG icon helper - wraps SVG in trusted HTML
 */
export function icon(svg: string): TrustedHTML {
  return raw(svg);
}

// Common SVG icons
export const icons = {
  play: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M8 5v14l11-7z"/></svg>'),
  pause: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/></svg>'),
  stop: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M6 6h12v12H6z"/></svg>'),
  chevronRight: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z"/></svg>'),
  chevronDown: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6z"/></svg>'),
  close: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>'),
  skipPrev: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/></svg>'),
  skipNext: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/></svg>'),
  refresh: raw('<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>'),
};
