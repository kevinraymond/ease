/**
 * ConnectionSection - Backend selection, connection, and generation controls.
 */

import { html } from '../../core/html';
import { BackendInfo } from '../../core/types';
import { ConnectionSectionProps } from './types';

/**
 * Render the connection section content.
 * This is a stateless render function - state management is handled by the parent.
 */
export function renderConnectionSection(props: ConnectionSectionProps): string {
  const {
    config,
    state,
    isConnected,
    isConnecting,
    availableBackends,
    selectedBackendId,
    hasPendingChanges,
    callbacks,
  } = props;

  const currentBackendId = isConnected
    ? state.serverConfig?.current_backend
    : selectedBackendId;

  return html`
    ${renderBackendSelector(
      availableBackends,
      currentBackendId || selectedBackendId,
      isConnected,
      state.isSwitchingBackend,
      state.pendingBackendId,
      callbacks
    )}

    <div class="control-section">
      ${renderConnectionButtons(state, config, isConnected, isConnecting, callbacks)}
      ${hasPendingChanges && !isConnected
        ? html`<p class="pending-changes-hint">Changes will sync on connect</p>`
        : ''}
      ${state.isInitializing && state.statusMessage
        ? html`<div class="status-message warning">${state.statusMessage}</div>`
        : ''}
    </div>
  `;
}

function renderBackendSelector(
  backends: BackendInfo[],
  currentBackendId: string,
  isConnected: boolean,
  isSwitching: boolean,
  pendingBackendId: string | null,
  callbacks: ConnectionSectionProps['callbacks']
): string {
  if (backends.length === 0) return '';

  return html`
    <div class="control-section">
      <label class="control-label" for="backend-select">Backend</label>
      <div class="backend-select-wrapper ${isSwitching ? 'switching' : ''}">
        <select
          id="backend-select"
          class="backend-select"
          data-action="${isConnected ? 'switch-backend' : 'select-backend'}"
          ${isSwitching ? 'disabled' : ''}
        >
          ${backends.map((backend: BackendInfo) => {
            const isSelected = currentBackendId === backend.id;
            return html`
              <option
                value="${backend.id}"
                ${isSelected ? 'selected' : ''}
              >${backend.name} (~${backend.fps_range[0]}-${backend.fps_range[1]} FPS)</option>
            `;
          })}
        </select>
        ${isSwitching ? '<span class="backend-select-spinner"></span>' : ''}
      </div>
    </div>
  `;
}

/**
 * Render the connection and generation buttons side-by-side.
 */
function renderConnectionButtons(
  state: ConnectionSectionProps['state'],
  config: ConnectionSectionProps['config'],
  isConnected: boolean,
  isConnecting: boolean,
  callbacks: ConnectionSectionProps['callbacks']
): string {
  const hasError = !!state.error;
  const connecting = isConnecting || state.isConnecting;
  const isGenerating = state.isGenerating;
  const isInitializing = state.isInitializing;

  // Connection button label
  let connectLabel = 'Connect';
  if (hasError) {
    connectLabel = 'Retry';
  } else if (connecting) {
    connectLabel = 'Connecting...';
  } else if (isConnected) {
    connectLabel = 'Disconnect';
  }

  // Generation button label
  let genLabel = 'Start';
  if (isInitializing) {
    genLabel = 'Init...';
  } else if (isGenerating) {
    genLabel = 'Stop';
  }

  return html`
    <div class="connection-buttons-row">
      <button
        class="connection-btn-half ${isConnected ? 'connected' : ''} ${hasError ? 'error' : ''} ${connecting ? 'connecting' : ''}"
        data-action="connect"
        ${connecting ? 'disabled' : ''}
        title="${state.error || (isConnected ? 'Click to disconnect' : 'Click to connect')}"
      >
        ${connectLabel}
      </button>
      <button
        class="generation-btn-half ${isGenerating ? 'generating' : ''} ${isInitializing ? 'initializing' : ''}"
        data-action="toggle-generation"
        ${!isConnected ? 'disabled' : ''}
        title="${!isConnected ? 'Connect first to enable' : isGenerating ? 'Stop generation' : 'Start generation'}"
      >
        ${genLabel}
      </button>
    </div>
  `;
}

function getBackendDisplayName(state: ConnectionSectionProps['state']): string {
  const serverConfig = state.serverConfig;
  if (!serverConfig?.current_backend) return '';

  if (serverConfig.available_backends) {
    const backend = serverConfig.available_backends.find(
      (b) => b.id === serverConfig.current_backend
    );
    if (backend) return backend.name;
  }

  return serverConfig.current_backend;
}

/**
 * Get the actions map for ConnectionSection.
 * The parent component should merge these into its actions.
 */
export function getConnectionSectionActions(
  callbacks: ConnectionSectionProps['callbacks'],
  getState: () => { isConnected: boolean; isConnecting: boolean }
) {
  return {
    connect: async () => {
      const { isConnected } = getState();
      if (isConnected) {
        callbacks.onDisconnect();
      } else {
        await callbacks.onConnect();
      }
    },
    'toggle-generation': () => {
      // This will be handled by parent who has access to current state
    },
    'reset-feedback': () => {
      callbacks.onResetFeedback();
    },
    'switch-backend': (e: Event) => {
      const backendId = (e.target as HTMLSelectElement).value;
      callbacks.onSwitchBackend?.(backendId);
    },
    'select-backend': (e: Event) => {
      const backendId = (e.target as HTMLSelectElement).value;
      callbacks.onSelectBackend(backendId);
    },
  };
}
