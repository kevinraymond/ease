/**
 * Main entry point for EASE (vanilla TypeScript).
 */

import { App } from './App';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element not found');
}

const app = new App(rootElement);
app.mount().catch(console.error);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  app.dispose();
});
