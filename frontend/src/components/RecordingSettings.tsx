import React, { useState, useEffect } from 'react';
import { Switch } from '@/components/ui/switch';
import { FolderOpen } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { DeviceSelection, SelectedDevices } from '@/components/DeviceSelection';
import Analytics from '@/lib/analytics';
import { toast } from 'sonner';
import { useRecordingState } from '@/contexts/RecordingStateContext';
import { useConfig } from '@/contexts/ConfigContext';

interface DiarizationDownloadProgress {
  artifact: string;
  percent: number;
}
export interface RecordingPreferences {
  save_folder: string;
  auto_save: boolean;
  file_format: string;
  preferred_mic_device: string | null;
  preferred_system_device: string | null;
}

interface RecordingSettingsProps {
  onSave?: (preferences: RecordingPreferences) => void;
}

export function RecordingSettings({ onSave }: RecordingSettingsProps) {
  const [preferences, setPreferences] = useState<RecordingPreferences>({
    save_folder: '',
    auto_save: true,
    file_format: 'mp4',
    preferred_mic_device: null,
    preferred_system_device: null
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showRecordingNotification, setShowRecordingNotification] = useState(true);
  const [speakerAttribution, setSpeakerAttribution] = useState<'off' | 'local_remote' | 'multi_speaker'>('off');
  const [loadingAttribution, setLoadingAttribution] = useState(true);
  const { isRecording } = useRecordingState();
  const { setSelectedDevices } = useConfig();

  // Load recording preferences on component mount
  useEffect(() => {
    const loadPreferences = async () => {
      try {
        const prefs = await invoke<RecordingPreferences>('get_recording_preferences');
        setPreferences(prefs);
      } catch (error) {
        console.error('Failed to load recording preferences:', error);
        // If loading fails, get default folder path
        try {
          const defaultPath = await invoke<string>('get_default_recordings_folder_path');
          setPreferences(prev => ({ ...prev, save_folder: defaultPath }));
        } catch (defaultError) {
          console.error('Failed to get default folder path:', defaultError);
        }
      } finally {
        setLoading(false);
      }
    };

    loadPreferences();
  }, []);

  // Load recording notification preference
  useEffect(() => {
    const loadNotificationPref = async () => {
      try {
        const { Store } = await import('@tauri-apps/plugin-store');
        const store = await Store.load('preferences.json');
        const show = await store.get<boolean>('show_recording_notification') ?? true;
        setShowRecordingNotification(show);
      } catch (error) {
        console.error('Failed to load notification preference:', error);
      }
    };
    loadNotificationPref();
  }, []);

  // Load speaker attribution mode from backend settings
  useEffect(() => {
    const loadAttribution = async () => {
      try {
        const mode = await invoke<string>('api_get_speaker_attribution');
        if (mode === 'local_remote' || mode === 'multi_speaker' || mode === 'off') {
          setSpeakerAttribution(mode);
        }
      } catch (error) {
        console.error('Failed to load speaker attribution mode:', error);
      } finally {
        setLoadingAttribution(false);
      }
    };
    loadAttribution();
  }, []);

  const handleSpeakerAttributionChange = async (mode: 'off' | 'local_remote' | 'multi_speaker') => {
    const previous = speakerAttribution;
    setSpeakerAttribution(mode);
    try {
      await invoke('api_save_speaker_attribution', { mode });
      toast.success('Speaker attribution saved', {
        description: mode === 'off'
          ? 'Transcripts will not be tagged with speakers.'
          : mode === 'local_remote'
            ? 'Mic and system audio will be transcribed separately and tagged You / Remote.'
            : 'A diarization model will assign speakers after recording stops.'
      });
      await Analytics.track('speaker_attribution_changed', { mode });
    } catch (error) {
      setSpeakerAttribution(previous);
      console.error('Failed to save speaker attribution mode:', error);
      toast.error('Failed to save speaker attribution');
    }
  };

  const handleAutoSaveToggle = async (enabled: boolean) => {
    const newPreferences = { ...preferences, auto_save: enabled };
    setPreferences(newPreferences);
    await savePreferences(newPreferences);

    // Track auto-save setting change
    await Analytics.track('auto_save_recording_toggled', {
      enabled: enabled.toString()
    });
  };

  const handleDeviceChange = async (devices: SelectedDevices) => {
    const newPreferences = {
      ...preferences,
      preferred_mic_device: devices.micDevice,
      preferred_system_device: devices.systemDevice
    };
    setPreferences(newPreferences);
    // Sync the in-memory selection ConfigContext exposes to the start path.
    // Without this, the picker only writes to disk (loaded into context once at
    // app mount), so a newly-chosen mic isn't honored until the next launch —
    // start keeps sending the stale launch-time device.
    setSelectedDevices(devices);
    await savePreferences(newPreferences);

    // Track default device preference changes
    // Note: Individual device selection analytics are tracked in DeviceSelection component
    await Analytics.track('default_devices_changed', {
      has_preferred_microphone: (!!devices.micDevice).toString(),
      has_preferred_system_audio: (!!devices.systemDevice).toString()
    });
  };

  const handleOpenFolder = async () => {
    try {
      await invoke('open_recordings_folder');
    } catch (error) {
      console.error('Failed to open recordings folder:', error);
    }
  };

  const handleNotificationToggle = async (enabled: boolean) => {
    try {
      setShowRecordingNotification(enabled);
      const { Store } = await import('@tauri-apps/plugin-store');
      const store = await Store.load('preferences.json');
      await store.set('show_recording_notification', enabled);
      await store.save();
      toast.success('Preference saved');
      await Analytics.track('recording_notification_preference_changed', {
        enabled: enabled.toString()
      });
    } catch (error) {
      console.error('Failed to save notification preference:', error);
      toast.error('Failed to save preference');
    }
  };

  const savePreferences = async (prefs: RecordingPreferences) => {
    setSaving(true);
    try {
      await invoke('set_recording_preferences', { preferences: prefs });
      onSave?.(prefs);

      // Show success toast with device details
      const micDevice = prefs.preferred_mic_device || 'Default';
      const systemDevice = prefs.preferred_system_device || 'Default';
      toast.success("Device preferences saved", {
        description: `Microphone: ${micDevice}, System Audio: ${systemDevice}`
      });
    } catch (error) {
      console.error('Failed to save recording preferences:', error);
      toast.error("Failed to save device preferences", {
        description: error instanceof Error ? error.message : String(error)
      });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
        <div className="h-8 bg-gray-200 rounded mb-4"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-4">Recording Settings</h3>
        <p className="text-sm text-gray-600 mb-6">
          Configure how your audio recordings are saved during meetings.
        </p>
      </div>

      {/* Auto Save Toggle */}
      <div className="flex items-center justify-between p-4 border rounded-lg">
        <div className="flex-1">
          <div className="font-medium">Save Audio Recordings</div>
          <div className="text-sm text-gray-600">
            Automatically save audio files when recording stops
          </div>
        </div>
        <Switch
          checked={preferences.auto_save}
          onCheckedChange={handleAutoSaveToggle}
          disabled={saving}
        />
      </div>

      {/* Folder Location - Only shown when auto_save is enabled */}
      {preferences.auto_save && (
        <div className="space-y-4">
          <div className="p-4 border rounded-lg bg-gray-50">
            <div className="font-medium mb-2">Save Location</div>
            <div className="text-sm text-gray-600 mb-3 break-all">
              {preferences.save_folder || 'Default folder'}
            </div>
            <button
              onClick={handleOpenFolder}
              className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
            >
              <FolderOpen className="w-4 h-4" />
              Open Folder
            </button>
          </div>

          <div className="p-4 border rounded-lg bg-blue-50">
            <div className="text-sm text-blue-800">
              <strong>File Format:</strong> {preferences.file_format.toUpperCase()} files
            </div>
            <div className="text-xs text-blue-600 mt-1">
              Recordings are saved with timestamp: recording_YYYYMMDD_HHMMSS.{preferences.file_format}
            </div>
          </div>
        </div>
      )}

      {/* Info when auto_save is disabled */}
      {!preferences.auto_save && (
        <div className="p-4 border rounded-lg bg-yellow-50">
          <div className="text-sm text-yellow-800">
            Audio recording is disabled. Enable "Save Audio Recordings" to automatically save your meeting audio.
          </div>
        </div>
      )}

      {/* Recording Notification Toggle */}
      <div className="flex items-center justify-between p-4 border rounded-lg">
        <div className="flex-1">
          <div className="text-sm text-gray-600">
            Show reminder to inform participants when recording starts
          </div>
        </div>
        <Switch
          checked={showRecordingNotification}
          onCheckedChange={handleNotificationToggle}
        />
      </div>

      {/* Speaker Attribution */}
      <div className="p-4 border rounded-lg">
        <div className="font-medium mb-1">Speaker Attribution</div>
        <div className="text-sm text-gray-600 mb-4">
          Tag transcript segments with who is speaking. Requires system audio capture for remote speakers.
        </div>
        {loadingAttribution ? (
          <div className="h-9 bg-gray-100 rounded-md animate-pulse" />
        ) : (
          <div className="grid gap-2">
            {([
              { value: 'off', label: 'Off', description: 'Mixed audio, no speaker tags' },
              { value: 'local_remote', label: 'You / Remote', description: 'Transcribe mic and system audio separately (best for online meetings)' },
              { value: 'multi_speaker', label: 'Multi-speaker (beta)', description: 'Neural diarization after recording stops (handles many speakers)' },
            ] as const).map((option) => (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={speakerAttribution === option.value}
                onClick={() => handleSpeakerAttributionChange(option.value)}
                disabled={saving}
                className={`text-left px-4 py-3 border rounded-md transition-colors ${
                  speakerAttribution === option.value
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-300 hover:bg-gray-50'
                }`}
              >
                <div className="text-sm font-medium">{option.label}</div>
                <div className="text-xs text-gray-600">{option.description}</div>
              </button>
            ))}
          </div>
        )}

        {speakerAttribution === 'multi_speaker' && (
          <DiarizationModelStatus />
        )}
      </div>

      {/* Device Preferences */}
      <div className="space-y-4">
        <div className="border-t pt-6">
          <h4 className="text-base font-medium text-gray-900 mb-4">Default Audio Devices</h4>
          <p className="text-sm text-gray-600 mb-4">
            Set your preferred microphone and system audio devices for recording. These will be automatically selected when starting new recordings.
          </p>

          {isRecording && (
            <p
              role="status"
              aria-live="polite"
              className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-md p-2 mb-4"
            >
              Device selection is locked while a recording is in progress. Connecting a new device mid-recording will not switch to it. Stop the current meeting to change devices.
            </p>
          )}

          <div className="border rounded-lg p-4 bg-gray-50">
            <DeviceSelection
              selectedDevices={{
                micDevice: preferences.preferred_mic_device,
                systemDevice: preferences.preferred_system_device
              }}
              onDeviceChange={handleDeviceChange}
              disabled={saving || isRecording}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// Status row for the diarization model set: availability check, one-click
// download with progress, shown only when multi-speaker mode is selected.
function DiarizationModelStatus() {
  const [available, setAvailable] = useState<boolean | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [progressLabel, setProgressLabel] = useState('');

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;

    const check = async () => {
      try {
        const ok = await invoke<boolean>('diarization_models_available');
        if (!cancelled) setAvailable(ok);
      } catch (error) {
        console.warn('Failed to check diarization model availability:', error);
        if (!cancelled) setAvailable(false);
      }
    };
    void check();

    const listenToProgress = async () => {
      try {
        unlisten = await listen<DiarizationDownloadProgress>(
          'diarization-model-download-progress',
          (event) => {
            const { artifact, percent } = event.payload;
            setProgressLabel(
              percent > 0 ? `Downloading ${artifact}… ${percent}%` : `Downloading ${artifact}…`,
            );
          },
        );
      } catch (error) {
        console.warn('Failed to listen for diarization download progress:', error);
      }
    };
    void listenToProgress();

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const handleDownload = async () => {
    setDownloading(true);
    setProgressLabel('Preparing download…');
    try {
      await invoke('diarization_download_models');
      setAvailable(true);
      setProgressLabel('');
      toast.success('Diarization models downloaded');
    } catch (error) {
      console.error('Diarization model download failed:', error);
      toast.error('Model download failed', {
        description: error instanceof Error ? error.message : String(error),
      });
      setProgressLabel('');
    } finally {
      setDownloading(false);
    }
  };

  if (available === null) {
    return <div className="mt-4 text-xs text-gray-500">Checking diarization models…</div>;
  }
  if (available) {
    return (
      <div className="mt-4 text-xs text-green-700">
        Diarization models ready — speakers will be labeled after each recording stops.
      </div>
    );
  }
  return (
    <div className="mt-4 flex items-center gap-3">
      <button
        type="button"
        onClick={handleDownload}
        disabled={downloading}
        className="px-3 py-1.5 text-xs border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
      >
        {downloading ? 'Downloading…' : 'Download diarization models (~40 MB)'}
      </button>
      {progressLabel && <span className="text-xs text-gray-500">{progressLabel}</span>}
    </div>
  );
}