// Types for Gipformer (Vietnamese Zipformer transducer) integration

export interface GipformerModelInfo {
  name: string;
  size_mb: number;
  description: string;
  status: GipformerModelStatus;
}

export type GipformerModelStatus =
  | 'Available'
  | 'Missing'
  | { Downloading: { progress: number } }
  | { Error: string }
  | { Corrupted: { file_size: number; expected_min_size: number } };

export type GipformerCancelDownloadOutcome = 'cancelled' | 'pending';

// Event payloads (serde renames to camelCase via Tauri event serialization)
export interface GipformerDownloadProgressEvent {
  modelName: string;
  progress: number;
}

// User-friendly display info
export interface GipformerModelDisplayInfo {
  icon: string;
  tagline: string;
}

export const GIPFORMER_MODEL_DISPLAY_CONFIG: Record<string, GipformerModelDisplayInfo> = {
  'gipformer1.5-68M-rnnt': {
    icon: '🇻🇳',
    tagline: 'Best Vietnamese accuracy in a compact 68M model',
  },
};

export function getGipformerModelDisplayName(modelName: string): string {
  return modelName
    .replace('gipformer1.5-68M-rnnt', 'Gipformer 1.5 68M (Vietnamese)');
}

export function getGipformerModelDisplayInfo(modelName: string): GipformerModelDisplayInfo | null {
  return GIPFORMER_MODEL_DISPLAY_CONFIG[modelName] || null;
}

export function formatFileSize(sizeMb: number): string {
  if (sizeMb >= 1024) {
    return `${(sizeMb / 1024).toFixed(1)} GB`;
  }
  return `${Math.round(sizeMb)} MB`;
}

// Tauri command wrappers for Gipformer backend
import { invoke } from '@tauri-apps/api/core';

export class GipformerAPI {
  static async init(): Promise<void> {
    await invoke('gipformer_init');
  }

  static async getAvailableModels(): Promise<GipformerModelInfo[]> {
    return await invoke('gipformer_get_available_models');
  }

  static async loadModel(modelName: string): Promise<void> {
    await invoke('gipformer_load_model', { modelName });
  }

  static async getCurrentModel(): Promise<string | null> {
    return await invoke('gipformer_get_current_model');
  }

  static async isModelLoaded(): Promise<boolean> {
    return await invoke('gipformer_is_model_loaded');
  }

  static async getModelsDirectory(): Promise<string> {
    return await invoke('gipformer_get_models_directory');
  }

  static async downloadModel(modelName: string): Promise<void> {
    await invoke('gipformer_download_model', { modelName });
  }

  static async cancelDownload(modelName: string): Promise<GipformerCancelDownloadOutcome> {
    return await invoke('gipformer_cancel_download', { modelName });
  }

  static async deleteCorruptedModel(modelName: string): Promise<string> {
    return await invoke('gipformer_delete_corrupted_model', { modelName });
  }

  static async validateModelReady(): Promise<boolean> {
    return await invoke('gipformer_validate_model_ready_with_config', {
      config: { provider: 'gipformer' },
    });
  }
}
