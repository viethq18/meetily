// diarization_engine/commands.rs
//
// Tauri command interface for the post-recording speaker diarization engine:
// model availability/download and a per-meeting diarization run that maps
// speaker labels onto existing transcript rows (transcripts.speaker).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context as _, Result};
use tauri::{command, AppHandle, Emitter, Manager, Runtime};
use tokio_util::sync::CancellationToken;

use crate::diarization_engine::engine::{DiarizationEngine, DiarizationResult};
use crate::diarization_engine::model;

/// Global diarization engine (loaded lazily after models are downloaded).
/// Inner mutex: ONNX sessions require mutable access to run.
static ENGINE: Mutex<Option<Arc<std::sync::Mutex<DiarizationEngine>>>> = Mutex::new(None);

/// Global models directory path (set during app initialization).
static MODELS_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Set the models directory from app_data_dir during app setup.
pub fn set_models_directory<R: Runtime>(app: &AppHandle<R>) {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .expect("Failed to get app data dir");
    let models_dir = app_data_dir.join("models");
    if !models_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&models_dir) {
            log::error!("Failed to create models directory: {}", e);
            return;
        }
    }
    log::info!("Diarization models directory set to: {}", models_dir.display());
    let mut guard = MODELS_DIR.lock().unwrap();
    *guard = Some(models_dir);
}

fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

fn load_engine() -> Result<Arc<std::sync::Mutex<DiarizationEngine>>> {
    let mut guard = ENGINE.lock().unwrap();
    if let Some(engine) = guard.as_ref() {
        return Ok(Arc::clone(engine));
    }
    let models_dir = get_models_directory().ok_or_else(|| anyhow!("models directory not set"))?;
    if !model::all_artifacts_present(&models_dir) {
        return Err(anyhow!(
            "diarization models not downloaded yet; call diarization_download_models first"
        ));
    }
    let engine = Arc::new(std::sync::Mutex::new(DiarizationEngine::load(&models_dir)?));
    *guard = Some(Arc::clone(&engine));
    Ok(engine)
}

#[command]
pub async fn diarization_models_available() -> Result<bool, String> {
    let Some(models_dir) = get_models_directory() else {
        return Ok(false);
    };
    Ok(model::all_artifacts_present(&models_dir))
}

#[command]
pub async fn diarization_download_models(app: AppHandle) -> Result<(), String> {
    let models_dir = get_models_directory().ok_or_else(|| "models directory not set".to_string())?;
    let cancellation = CancellationToken::new();

    let app_for_progress = app.clone();
    let result = model::download_models(&models_dir, &cancellation, &move |progress| {
        let percent = if progress.total_bytes > 0 {
            (progress.downloaded_bytes as f64 / progress.total_bytes as f64 * 100.0) as u32
        } else {
            0
        };
        let _ = app_for_progress.emit(
            "diarization-model-download-progress",
            &serde_json::json!({
                "artifact": progress.artifact,
                "percent": percent,
                "downloadedBytes": progress.downloaded_bytes,
                "totalBytes": progress.total_bytes,
            }),
        );
    })
    .await;

    match result {
        Ok(()) => {
            let _ = app.emit("diarization-model-download-complete", &serde_json::json!({}));
            Ok(())
        }
        Err(e) => {
            let _ = app.emit(
                "diarization-model-download-error",
                &serde_json::json!({ "error": e.to_string() }),
            );
            Err(e.to_string())
        }
    }
}

#[derive(serde::Serialize)]
pub struct DiarizationRunResult {
    pub meeting_id: String,
    pub num_speakers: usize,
    pub updated_transcripts: usize,
}

/// Run post-recording diarization for a meeting: decode the meeting's audio
/// file, diarize, then update `transcripts.speaker` for each row by maximal
/// overlap between the row's audio window and the diarization segments.
/// Emits `diarization-complete` when done (or `diarization-error`).
#[command]
pub async fn diarization_run_for_meeting(
    app: AppHandle,
    meeting_id: String,
    meeting_folder_path: String,
) -> Result<DiarizationRunResult, String> {
    let engine = load_engine().map_err(|e| e.to_string())?;

    let folder = PathBuf::from(&meeting_folder_path);
    let audio_path = find_audio_file(&folder).map_err(|e| e.to_string())?;

    // Decode + mono/16k conversion are CPU-heavy: run on the blocking pool.
    let decoded = tokio::task::spawn_blocking(move || {
        let decoded = crate::audio::decoder::decode_audio_file(&audio_path)?;
        let samples = decoded.to_whisper_format();
        Ok::<_, anyhow::Error>((samples, decoded.sample_rate))
    })
    .await
    .map_err(|e| format!("decode task panicked: {e}"))?
    .map_err(|e| e.to_string())?;

    let (samples, sample_rate) = decoded;
    let diar_result = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().unwrap();
        engine.process(&samples, sample_rate).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("diarization task panicked: {e}"))?;

    let diar = diar_result.map_err(|e| {
        let _ = app.emit(
            "diarization-error",
            &serde_json::json!({ "meetingId": meeting_id, "error": e }),
        );
        e
    })?;

    if diar.segments.is_empty() {
        let _ = app.emit(
            "diarization-error",
            &serde_json::json!({
                "meetingId": meeting_id,
                "error": "no speech found in meeting audio",
            }),
        );
        return Err("no speech found in meeting audio".to_string());
    }

    // Map transcript rows to speakers by maximal time overlap.
    let pool = app
        .state::<crate::state::AppState>()
        .db_manager
        .pool()
        .clone();

    let updated = update_transcript_speakers(&pool, &meeting_id, &diar)
        .await
        .map_err(|e| e.to_string())?;

    let result = DiarizationRunResult {
        meeting_id: meeting_id.clone(),
        num_speakers: diar.num_speakers,
        updated_transcripts: updated,
    };
    let _ = app.emit("diarization-complete", &result);
    Ok(result)
}

/// For each transcript row, pick the speaker whose segments overlap the row's
/// [audio_start_time, audio_end_time] window the most; write "speaker_N".
async fn update_transcript_speakers(
    pool: &sqlx::SqlitePool,
    meeting_id: &str,
    diar: &DiarizationResult,
) -> Result<usize> {
    let rows: Vec<(String, Option<f64>, Option<f64>)> = sqlx::query_as(
        "SELECT id, audio_start_time, audio_end_time FROM transcripts WHERE meeting_id = ?",
    )
    .bind(meeting_id)
    .fetch_all(pool)
    .await
    .context("loading transcripts for diarization")?;

    let mut updated = 0usize;
    for (id, start, end) in rows {
        let (Some(start), Some(end)) = (start, end) else {
            continue;
        };
        let best = best_speaker_for_window(&diar.segments, start, end);
        let Some(speaker) = best else {
            continue;
        };
        sqlx::query("UPDATE transcripts SET speaker = ? WHERE id = ?")
            .bind(format!("speaker_{}", speaker))
            .bind(&id)
            .execute(pool)
            .await
            .context("updating transcript speaker")?;
        updated += 1;
    }
    Ok(updated)
}

/// Speaker with maximal total overlap with [start, end] (seconds).
fn best_speaker_for_window(
    segments: &[crate::diarization_engine::engine::SpeakerSegment],
    start: f64,
    end: f64,
) -> Option<usize> {
    use std::collections::HashMap;
    let mut totals: HashMap<usize, f64> = HashMap::new();
    for seg in segments {
        let overlap =
            (seg.end as f64).min(end) - (seg.start as f64).max(start);
        if overlap > 0.0 {
            *totals.entry(seg.speaker).or_insert(0.0) += overlap;
        }
    }
    totals
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(speaker, _)| speaker)
}

/// Find the audio file inside a meeting folder (retranscription's heuristic).
fn find_audio_file(folder: &std::path::Path) -> Result<PathBuf> {
    let candidates = [
        "audio.mp4",
        "audio.m4a",
        "audio.wav",
        "audio.mp3",
        "audio.flac",
        "audio.ogg",
        "recording.mp4",
        "audio.mkv",
        "audio.webm",
        "audio.wma",
    ];
    for name in candidates {
        let path = folder.join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    let audio_extensions = [
        "mp4", "m4a", "wav", "mp3", "flac", "ogg", "mkv", "webm", "wma", "aac",
    ];
    if let Ok(entries) = std::fs::read_dir(folder) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if audio_extensions.contains(&ext.to_lowercase().as_str()) {
                        return Ok(path);
                    }
                }
            }
        }
    }
    Err(anyhow!("no audio file found in {}", folder.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarization_engine::engine::SpeakerSegment;

    #[test]
    fn best_speaker_picks_maximal_overlap() {
        let segments = vec![
            SpeakerSegment { start: 0.0, end: 4.0, speaker: 0 },
            SpeakerSegment { start: 5.0, end: 9.0, speaker: 1 },
            SpeakerSegment { start: 13.0, end: 17.0, speaker: 0 },
        ];
        // Window mostly inside speaker 1.
        assert_eq!(best_speaker_for_window(&segments, 5.0, 8.0), Some(1));
        // Window mostly inside speaker 0.
        assert_eq!(best_speaker_for_window(&segments, 1.0, 3.5), Some(0));
        assert_eq!(best_speaker_for_window(&segments, 14.0, 16.0), Some(0));
        // No overlap at all.
        assert_eq!(best_speaker_for_window(&segments, 9.5, 12.5), None);
    }

    #[test]
    fn find_audio_file_prefers_known_names() {
        let dir = std::env::temp_dir().join("meetily_diar_test_find_audio");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("audio.wav"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let found = find_audio_file(&dir).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.wav");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
