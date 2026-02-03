/**
 * IndexedDB storage for saved settings configurations.
 * Provides persistent local storage for AI generator and mapping settings.
 */

import { AIGeneratorConfig } from '../../core/types';
import { MappingConfig } from '../../components/MappingPanel';

const DB_NAME = 'ease-settings';
const DB_VERSION = 1;
const STORE_NAME = 'settings';
const CURRENT_STATE_ID = '__current_state__';
const SETTINGS_VERSION = 1;

/**
 * Combined settings that includes both AI config and mapping config.
 */
export interface CombinedSettings {
  version: number;
  aiConfig: Partial<AIGeneratorConfig>;
  mappingConfig: MappingConfig;
}

/**
 * Saved settings record stored in IndexedDB.
 */
export interface SavedSettings {
  id: string;
  name: string;
  settings: CombinedSettings;
  createdAt: number;
  updatedAt: number;
}

// Open database connection
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('name', 'name', { unique: false });
        store.createIndex('updatedAt', 'updatedAt', { unique: false });
      }
    };
  });
}

// Generate a simple unique ID
function generateId(): string {
  return `settings_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Save settings to the database.
 * If existingId is provided, updates existing record. Otherwise creates new.
 */
export async function saveSettings(
  name: string,
  settings: CombinedSettings,
  existingId?: string
): Promise<SavedSettings> {
  const db = await openDB();
  const now = Date.now();

  // Deep clone the settings to ensure all data is captured
  const settingsClone: CombinedSettings = JSON.parse(JSON.stringify(settings));

  const savedSettings: SavedSettings = {
    id: existingId || generateId(),
    name,
    settings: settingsClone,
    createdAt: now,
    updatedAt: now,
  };

  // If updating, preserve original createdAt
  if (existingId) {
    const existing = await getSettings(existingId);
    if (existing) {
      savedSettings.createdAt = existing.createdAt;
    }
  }

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(savedSettings);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(savedSettings);
  });
}

/**
 * Get settings by ID.
 */
export async function getSettings(id: string): Promise<SavedSettings | null> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(id);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result || null);
  });
}

/**
 * Get all saved settings, sorted by most recently updated.
 * Excludes the special "current state" auto-save record.
 */
export async function getAllSettings(): Promise<SavedSettings[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const settings = (request.result as SavedSettings[])
        .filter(s => s.id !== CURRENT_STATE_ID);
      // Sort by most recently updated
      settings.sort((a, b) => b.updatedAt - a.updatedAt);
      resolve(settings);
    };
  });
}

/**
 * Delete settings by ID.
 */
export async function deleteSettings(id: string): Promise<void> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.delete(id);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

/**
 * Rename saved settings.
 */
export async function renameSettings(id: string, newName: string): Promise<SavedSettings | null> {
  const existing = await getSettings(id);
  if (!existing) return null;

  existing.name = newName;
  existing.updatedAt = Date.now();

  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(existing);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(existing);
  });
}

/**
 * Save current state for auto-restore on next session.
 * Uses a special reserved ID that won't appear in getAllSettings().
 */
export async function saveCurrentState(settings: CombinedSettings): Promise<void> {
  const db = await openDB();
  const now = Date.now();

  const record: SavedSettings = {
    id: CURRENT_STATE_ID,
    name: 'Current State',
    settings: JSON.parse(JSON.stringify(settings)),
    createdAt: now,
    updatedAt: now,
  };

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(record);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

/**
 * Load last auto-saved current state.
 */
export async function getCurrentState(): Promise<CombinedSettings | null> {
  const record = await getSettings(CURRENT_STATE_ID);
  return record ? record.settings : null;
}

/**
 * Get the current state record for display in UI.
 * Returns a SavedSettings object with a user-friendly name.
 */
export async function getCurrentStateRecord(): Promise<SavedSettings | null> {
  const record = await getSettings(CURRENT_STATE_ID);
  if (!record) return null;

  // Return with a user-friendly name
  return {
    ...record,
    name: 'Last Session',
  };
}

/** Export the current state ID for comparison */
export const CURRENT_STATE_ID_EXPORT = CURRENT_STATE_ID;

/**
 * Export settings to a JSON string for download.
 */
export function exportSettingsToJson(settings: CombinedSettings): string {
  return JSON.stringify({
    type: 'ease-settings',
    version: SETTINGS_VERSION,
    exportedAt: new Date().toISOString(),
    settings,
  }, null, 2);
}

/**
 * Parse imported JSON and extract settings.
 * Returns null if the JSON is invalid or incompatible.
 */
export function parseImportedJson(json: string): CombinedSettings | null {
  try {
    const data = JSON.parse(json);

    // Validate structure
    if (data.type !== 'ease-settings') {
      console.warn('Invalid settings file: wrong type');
      return null;
    }

    if (!data.settings || typeof data.settings !== 'object') {
      console.warn('Invalid settings file: missing settings');
      return null;
    }

    // Ensure version field exists
    const settings = data.settings as CombinedSettings;
    if (!settings.version) {
      settings.version = 1;
    }

    return settings;
  } catch (e) {
    console.error('Failed to parse settings JSON:', e);
    return null;
  }
}
