/**
 * IndexedDB storage for saved mapping presets.
 * Provides persistent local storage for user-created audio mapping configurations.
 */

import { MappingConfig } from '../../components/MappingPanel';

const DB_NAME = 'ease-mapping-presets';
const DB_VERSION = 1;
const STORE_NAME = 'presets';

export interface SavedMappingPreset {
  id: string;
  name: string;
  config: MappingConfig;
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
  return `preset_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Save a mapping preset to the database.
 * If id is provided, updates existing preset. Otherwise creates new.
 */
export async function saveMappingPreset(
  name: string,
  config: MappingConfig,
  existingId?: string
): Promise<SavedMappingPreset> {
  const db = await openDB();
  const now = Date.now();

  // Deep clone the config to ensure all data is captured
  const configClone: MappingConfig = JSON.parse(JSON.stringify(config));

  const savedPreset: SavedMappingPreset = {
    id: existingId || generateId(),
    name,
    config: configClone,
    createdAt: now, // Will be overwritten if updating
    updatedAt: now,
  };

  // If updating, preserve original createdAt
  if (existingId) {
    const existing = await getMappingPreset(existingId);
    if (existing) {
      savedPreset.createdAt = existing.createdAt;
    }
  }

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(savedPreset);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(savedPreset);
  });
}

/**
 * Get a preset by ID.
 */
export async function getMappingPreset(id: string): Promise<SavedMappingPreset | null> {
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
 * Get all saved presets, sorted by most recently updated.
 */
export async function getAllMappingPresets(): Promise<SavedMappingPreset[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const presets = request.result as SavedMappingPreset[];
      // Sort by most recently updated
      presets.sort((a, b) => b.updatedAt - a.updatedAt);
      resolve(presets);
    };
  });
}

/**
 * Delete a preset by ID.
 */
export async function deleteMappingPreset(id: string): Promise<void> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.delete(id);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}
