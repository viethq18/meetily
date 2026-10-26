// gipformer_engine/commands.rs
//
// Tauri command interface for the Gipformer engine (mirrors parakeet_engine).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::{command, AppHandle, Emitter, Manager, Runtime};

use crate::gipformer_engine::gipformer_engine::{
    is_download_cancelled, CancelDownloadOutcome, GipformerEngine, ModelInfo,
};

// Global Gipformer engine
pub static GIPFORMER_ENGINE: Mutex<Option<Arc<GipformerEngine>>> = Mutex::new(None);

// Global models directory path (set during app initialization)
static MODELS_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Initialize the models directory path using app_data_dir.
/// Called during app setup before gipformer_init.
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

    log::info!("Gipformer models directory set to: {}", models_dir.display());

    let mut guard = MODELS_DIR.lock().unwrap();
    *guard = Some(models_dir);
}

/// Get the configured models directory
fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

#[command]
pub async fn gipformer_init() -> Result<(), String> {
    let mut guard = GIPFORMER_ENGINE.lock().unwrap();
    if guard.is_some() {
        return Ok(());
    }

    let models_dir = get_models_directory();
    let engine = GipformerEngine::new_with_models_dir(models_dir)
        .map_err(|e| format!("Failed to initialize Gipformer engine: {}", e))?;
    *guard = Some(Arc::new(engine));
    Ok(())
}

fn engine() -> Result<Arc<GipformerEngine>, String> {
    GIPFORMER_ENGINE
        .lock()
        .unwrap()
        .as_ref()
        .cloned()
        .ok_or_else(|| "Gipformer engine not initialized".to_string())
}

#[command]
pub async fn gipformer_get_available_models() -> Result<Vec<ModelInfo>, String> {
    engine()?
        .discover_models()
        .await
        .map_err(|e| format!("Failed to discover Gipformer models: {}", e))
}

#[command]
pub async fn gipformer_load_model(model_name: String) -> Result<(), String> {
    engine()?
        .load_model(&model_name)
        .await
        .map_err(|e| format!("Failed to load Gipformer model {}: {}", model_name, e))
}

#[command]
pub async fn gipformer_unload_model() -> Result<bool, String> {
    Ok(engine()?.unload_model().await)
}

#[command]
pub async fn gipformer_get_current_model() -> Result<Option<String>, String> {
    Ok(engine()?.get_current_model().await)
}

#[command]
pub async fn gipformer_is_model_loaded() -> Result<bool, String> {
    Ok(engine()?.is_model_loaded().await)
}

#[command]
pub async fn gipformer_download_model(
    app: AppHandle,
    model_name: String,
) -> Result<(), String> {
    let engine = engine()?;
    let progress_app = app.clone();
    let progress_model_name = model_name.clone();

    let progress_callback: Option<Box<dyn Fn(crate::gipformer_engine::DownloadProgress) + Send>> =
        Some(Box::new(move |progress| {
            let _ = progress_app.emit(
                "gipformer-model-download-progress",
                &serde_json::json!({
                    "modelName": progress_model_name,
                    "progress": progress,
                }),
            );
        }));

    match engine
        .download_model_detailed(&model_name, progress_callback)
        .await
    {
        Ok(()) => {
            let _ = app.emit(
                "gipformer-model-download-complete",
                &serde_json::json!({ "modelName": model_name }),
            );
            Ok(())
        }
        Err(error) => {
            let _ = app.emit(
                "gipformer-model-download-error",
                &serde_json::json!({
                    "modelName": model_name,
                    "error": error.to_string(),
                }),
            );
            Err(error.to_string())
        }
    }
}

#[command]
pub async fn gipformer_cancel_download(model_name: String) -> Result<String, String> {
    match engine()?.cancel_download(&model_name).await {
        Ok(CancelDownloadOutcome::Cancelled) => {
            Ok(format!("Download cancelled for {}", model_name))
        }
        Ok(CancelDownloadOutcome::Pending) => Ok(format!(
            "Cancellation requested for {}; cleanup is still in progress",
            model_name
        )),
        Err(e) => Err(e.to_string()),
    }
}


#[command]
pub async fn gipformer_has_available_models() -> Result<bool, String> {
    let engine = {
        let guard = GIPFORMER_ENGINE.lock().unwrap();
        guard.as_ref().cloned()
    };

    if let Some(engine) = engine {
        let models = engine
            .discover_models()
            .await
            .map_err(|e| format!("Failed to discover Gipformer models: {}", e))?;

        // Check if at least one model is available
        Ok(models
            .iter()
            .any(|model| matches!(model.status, crate::gipformer_engine::ModelStatus::Available)))
    } else {
        Ok(false)
    }
}
#[command]
pub async fn gipformer_retry_download(app: AppHandle, model_name: String) -> Result<(), String> {
    gipformer_download_model(app, model_name).await
}

#[command]
pub async fn gipformer_delete_corrupted_model(model_name: String) -> Result<String, String> {
    engine()?
        .delete_model(&model_name)
        .await
        .map_err(|e| format!("Failed to delete Gipformer model {}: {}", model_name, e))
}

#[command]
pub async fn gipformer_get_models_directory() -> Result<String, String> {
    let path = engine()?
        .get_models_directory()
        .await;
    Ok(path.to_string_lossy().into_owned())
}

#[command]
pub async fn gipformer_validate_model_ready_with_config(
    config: serde_json::Value,
) -> Result<bool, String> {
    let provider = config
        .get("provider")
        .and_then(|p| p.as_str())
        .unwrap_or_default();

    if provider != "gipformer" {
        return Ok(false);
    }

    let engine = engine()?;
    if !engine.is_model_loaded().await {
        return Ok(false);
    }
    Ok(engine.get_current_model().await.is_some())
}

/// Direct transcription entrypoint used by the unified transcription dispatch.
/// Accepts 16kHz mono f32 samples; returns the transcript text.
pub async fn transcribe_with_gipformer(samples: Vec<f32>) -> Result<String, String> {
    engine()?
        .transcribe_audio(samples)
        .await
        .map_err(|e| format!("Gipformer transcription failed: {}", e))
}

/// Whether the last download error was a user cancellation (for UI messaging).
#[allow(dead_code)]
pub fn is_cancelled_error(error: &str) -> bool {
    is_download_cancelled(&anyhow::anyhow!("{}", error))
}
