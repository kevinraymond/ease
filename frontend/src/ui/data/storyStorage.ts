/**
 * IndexedDB storage for saved story configurations.
 * Provides persistent local storage without requiring file management.
 */

import { StoryConfig } from '../../core/types';

const DB_NAME = 'ease-stories';
const DB_VERSION = 1;
const STORE_NAME = 'stories';

export interface SavedStory {
  id: string;
  name: string;
  story: StoryConfig;
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
  return `story_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Save a story to the database.
 * If id is provided, updates existing story. Otherwise creates new.
 */
export async function saveStory(story: StoryConfig, existingId?: string): Promise<SavedStory> {
  const db = await openDB();
  const now = Date.now();

  // Deep clone the story to ensure all data is captured
  const storyClone: StoryConfig = JSON.parse(JSON.stringify(story));

  const savedStory: SavedStory = {
    id: existingId || generateId(),
    name: storyClone.name,
    story: storyClone,
    createdAt: now, // Will be overwritten if updating
    updatedAt: now,
  };

  // If updating, preserve original createdAt
  if (existingId) {
    const existing = await getStory(existingId);
    if (existing) {
      savedStory.createdAt = existing.createdAt;
    }
  }

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(savedStory);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(savedStory);
  });
}

/**
 * Get a story by ID.
 */
export async function getStory(id: string): Promise<SavedStory | null> {
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
 * Get all saved stories, sorted by most recently updated.
 */
export async function getAllStories(): Promise<SavedStory[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const stories = request.result as SavedStory[];
      // Sort by most recently updated
      stories.sort((a, b) => b.updatedAt - a.updatedAt);
      resolve(stories);
    };
  });
}

/**
 * Delete a story by ID.
 */
export async function deleteStory(id: string): Promise<void> {
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
 * Rename a saved story.
 */
export async function renameStory(id: string, newName: string): Promise<SavedStory | null> {
  const existing = await getStory(id);
  if (!existing) return null;

  existing.name = newName;
  existing.story.name = newName;
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
