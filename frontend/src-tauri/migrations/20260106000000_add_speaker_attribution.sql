-- Migration: Add speaker attribution mode to transcript settings.
-- Values: 'off' (default), 'local_remote' (tag mic/system streams),
-- 'multi_speaker' (post-recording diarization with neural model).

ALTER TABLE transcript_settings ADD COLUMN speakerAttribution TEXT NOT NULL DEFAULT 'off';
