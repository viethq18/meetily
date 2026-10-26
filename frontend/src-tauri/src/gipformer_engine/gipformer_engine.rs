// gipformer_engine/gipformer_engine.rs
//
// Gipformer engine: model catalog, model directory management, and resumable
// HuggingFace downloads for the Zipformer-RNNT Vietnamese model
// (g-group-ai-lab/gipformer1.5-68M-rnnt, sherpa-onnx format).

use crate::gipformer_engine::model::GipformerModel;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::{watch, Mutex, RwLock};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

/// Model status for Gipformer models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Missing,
    Downloading { progress: u8 },
    Error(String),
    Corrupted { file_size: u64, expected_min_size: u64 },
}

/// Detailed download progress info (MB-based with speed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub downloaded_mb: f64,
    pub total_mb: f64,
    pub speed_mbps: f64,
    pub percent: u8,
}

impl DownloadProgress {
    pub fn new(downloaded: u64, total: u64, speed_mbps: f64) -> Self {
        let percent = if total > 0 {
            ((downloaded as f64 / total as f64) * 100.0).min(100.0) as u8
        } else {
            0
        };
        Self {
            downloaded_bytes: downloaded,
            total_bytes: total,
            downloaded_mb: downloaded as f64 / (1024.0 * 1024.0),
            total_mb: total as f64 / (1024.0 * 1024.0),
            speed_mbps,
            percent,
        }
    }
}

/// Information about a Gipformer model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub status: ModelStatus,
    pub description: String,
    pub language: String,
}

#[derive(Clone, Copy)]
struct ArtifactSpec {
    filename: &'static str,
    exact_bytes: u64,
}

struct ModelSpec {
    name: &'static str,
    size_mb: u32,
    description: &'static str,
    source_base_url: &'static str,
    artifacts: &'static [ArtifactSpec],
}

impl ModelSpec {
    fn exact_bytes(&self) -> u64 {
        self.artifacts.iter().map(|a| a.exact_bytes).sum()
    }
}

// Exact sizes from https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt
// (int8 quantized artifacts, ~73MB total).
const GIPFORMER_15_ARTIFACTS: &[ArtifactSpec] = &[
    ArtifactSpec { filename: "encoder.int8.onnx", exact_bytes: 70_876_359 },
    ArtifactSpec { filename: "decoder.int8.onnx", exact_bytes: 1_308_690 },
    ArtifactSpec { filename: "joiner.int8.onnx", exact_bytes: 1_033_417 },
    ArtifactSpec { filename: "tokens.txt", exact_bytes: 25_847 },
];

const GIPFORMER_MODEL_SPECS: &[ModelSpec] = &[ModelSpec {
    name: "gipformer1.5-68M-rnnt",
    size_mb: 73,
    description: "Vietnamese speech recognition (Zipformer transducer, int8). Best Vietnamese WER in a 68M model.",
    source_base_url: "https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt/resolve/main",
    artifacts: GIPFORMER_15_ARTIFACTS,
}];

fn find_model_spec(model_name: &str) -> Option<&'static ModelSpec> {
    GIPFORMER_MODEL_SPECS.iter().find(|spec| spec.name == model_name)
}

struct ActiveDownload {
    cancellation: CancellationToken,
    completion: watch::Sender<bool>,
}

#[derive(Default)]
struct ActiveDownloadState {
    downloads: HashMap<String, Arc<ActiveDownload>>,
    revision: u64,
}

#[derive(Debug, thiserror::Error)]
#[error("Download cancelled by user")]
pub(crate) struct DownloadCancelled;

pub(crate) fn is_download_cancelled(error: &anyhow::Error) -> bool {
    error.downcast_ref::<DownloadCancelled>().is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CancelDownloadOutcome {
    Cancelled,
    Pending,
}

const CANCEL_DOWNLOAD_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

enum ContentRange {
    Range { start: u64, end: u64, total: u64 },
    Unsatisfied { total: u64 },
}

fn parse_content_range(value: &reqwest::header::HeaderValue) -> Result<ContentRange> {
    let value = value
        .to_str()
        .map_err(|e| anyhow!("Invalid Content-Range header encoding: {}", e))?;
    let value = value
        .strip_prefix("bytes ")
        .ok_or_else(|| anyhow!("Content-Range must use bytes: {}", value))?;

    if let Some(total) = value.strip_prefix("*/") {
        return total
            .parse()
            .map(|total| ContentRange::Unsatisfied { total })
            .map_err(|e| anyhow!("Invalid unsatisfied Content-Range total: {}", e));
    }

    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| anyhow!("Malformed Content-Range: {}", value))?;
    let (start, end) = range
        .split_once('-')
        .ok_or_else(|| anyhow!("Malformed Content-Range range: {}", range))?;
    let start = start
        .parse()
        .map_err(|e| anyhow!("Invalid Content-Range start: {}", e))?;
    let end = end
        .parse()
        .map_err(|e| anyhow!("Invalid Content-Range end: {}", e))?;
    let total = total
        .parse()
        .map_err(|e| anyhow!("Invalid Content-Range total: {}", e))?;
    if start > end {
        return Err(anyhow!("Content-Range start exceeds end: {}", value));
    }

    Ok(ContentRange::Range { start, end, total })
}

#[derive(Debug)]
pub enum GipformerEngineError {
    ModelNotLoaded,
    ModelNotFound(String),
    TranscriptionFailed(String),
    DownloadFailed(String),
    IoError(std::io::Error),
    Other(String),
}

impl std::fmt::Display for GipformerEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GipformerEngineError::ModelNotLoaded => write!(f, "No Gipformer model loaded"),
            GipformerEngineError::ModelNotFound(name) => write!(f, "Model '{}' not found", name),
            GipformerEngineError::TranscriptionFailed(err) => {
                write!(f, "Transcription failed: {}", err)
            }
            GipformerEngineError::DownloadFailed(err) => write!(f, "Download failed: {}", err),
            GipformerEngineError::IoError(err) => write!(f, "IO error: {}", err),
            GipformerEngineError::Other(err) => write!(f, "Error: {}", err),
        }
    }
}

impl std::error::Error for GipformerEngineError {}

impl From<std::io::Error> for GipformerEngineError {
    fn from(err: std::io::Error) -> Self {
        GipformerEngineError::IoError(err)
    }
}

pub struct GipformerEngine {
    models_dir: PathBuf,
    current_model: Arc<RwLock<Option<GipformerModel>>>,
    current_model_name: Arc<RwLock<Option<String>>>,
    model_lifecycle_lock: Mutex<()>,
    pub(crate) available_models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    active_downloads: Arc<Mutex<ActiveDownloadState>>,
}

impl GipformerEngine {
    /// Create a new Gipformer engine with optional custom models directory
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir.join("gipformer")
        } else {
            let current_dir = std::env::current_dir()
                .map_err(|e| anyhow!("Failed to get current directory: {}", e))?;

            if cfg!(debug_assertions) {
                current_dir.join("models").join("gipformer")
            } else {
                dirs::data_dir()
                    .or_else(|| dirs::home_dir())
                    .ok_or_else(|| anyhow!("Could not find system data directory"))?
                    .join("Meetily")
                    .join("models")
                    .join("gipformer")
            }
        };

        log::info!("GipformerEngine using models directory: {}", models_dir.display());

        if !models_dir.exists() {
            std::fs::create_dir_all(&models_dir)?;
        }

        Ok(Self {
            models_dir,
            current_model: Arc::new(RwLock::new(None)),
            current_model_name: Arc::new(RwLock::new(None)),
            model_lifecycle_lock: Mutex::new(()),
            available_models: Arc::new(RwLock::new(HashMap::new())),
            active_downloads: Arc::new(Mutex::new(ActiveDownloadState::default())),
        })
    }

    /// Discover available Gipformer models.
    pub async fn discover_models(&self) -> Result<Vec<ModelInfo>> {
        loop {
            let revision = self.active_downloads.lock().await.revision;
            let mut models = Vec::with_capacity(GIPFORMER_MODEL_SPECS.len());
            let mut validation_errors = Vec::new();

            for spec in GIPFORMER_MODEL_SPECS {
                let model_path = self.models_dir.join(spec.name);
                let status = if model_path.exists() {
                    match Self::validate_model_directory(&model_path, spec.artifacts) {
                        Ok(()) => ModelStatus::Available,
                        Err(error) => {
                            let file_size = spec
                                .artifacts
                                .iter()
                                .filter_map(|a| std::fs::metadata(model_path.join(a.filename)).ok())
                                .map(|m| m.len())
                                .sum();
                            validation_errors.push((spec.name, error));
                            ModelStatus::Corrupted {
                                file_size,
                                expected_min_size: spec.exact_bytes(),
                            }
                        }
                    }
                } else {
                    ModelStatus::Missing
                };

                models.push(ModelInfo {
                    name: spec.name.to_string(),
                    path: model_path,
                    size_mb: spec.size_mb,
                    status,
                    description: spec.description.to_string(),
                    language: "vi".to_string(),
                });
            }

            let active_downloads = self.active_downloads.lock().await;
            if active_downloads.revision != revision {
                continue;
            }

            validation_errors.retain(|(name, _)| !active_downloads.downloads.contains_key(*name));
            for model in &mut models {
                if active_downloads.downloads.contains_key(&model.name) {
                    model.status = ModelStatus::Downloading { progress: 0 };
                }
            }

            let mut available_models = self.available_models.write().await;
            available_models.clear();
            for model in &models {
                available_models.insert(model.name.clone(), model.clone());
            }
            drop(available_models);
            drop(active_downloads);

            for (name, error) in validation_errors {
                log::warn!("Gipformer model directory {} appears corrupted: {}", name, error);
            }
            return Ok(models);
        }
    }

    fn validate_model_directory(model_dir: &Path, artifacts: &[ArtifactSpec]) -> Result<()> {
        for artifact in artifacts {
            let path = model_dir.join(artifact.filename);
            let metadata = std::fs::metadata(&path)
                .map_err(|e| anyhow!("Failed to read {} metadata: {}", artifact.filename, e))?;
            if metadata.len() != artifact.exact_bytes {
                return Err(anyhow!(
                    "{} has {} bytes, expected exactly {} bytes",
                    artifact.filename,
                    metadata.len(),
                    artifact.exact_bytes
                ));
            }
        }
        Ok(())
    }

    /// Load a Gipformer model
    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        let model_info = {
            let models = self.available_models.read().await;
            models
                .get(model_name)
                .cloned()
                .ok_or_else(|| anyhow!("Model {} not found", model_name))?
        };

        match &model_info.status {
            ModelStatus::Available => {
                let _lifecycle_guard = self.model_lifecycle_lock.lock().await;
                let current_model = self.current_model_name.read().await.clone();
                if current_model.as_deref() == Some(model_name) {
                    log::info!("Gipformer model {} is already loaded, skipping reload", model_name);
                    return Ok(());
                }

                if let Some(current_model) = current_model {
                    log::info!(
                        "Unloading current Gipformer model '{}' before loading '{}'",
                        current_model,
                        model_name
                    );
                }
                self.unload_model_locked().await;

                log::info!("Loading Gipformer model: {}", model_name);
                let model_path = model_info.path.clone();
                let model = tokio::task::spawn_blocking(move || {
                    GipformerModel::new(&model_path, true).map_err(|e| e.to_string())
                })
                .await
                .map_err(|e| anyhow!("Gipformer model load task failed for {}: {}", model_name, e))?
                .map_err(|e| anyhow!("Failed to load Gipformer model {}: {}", model_name, e))?;

                *self.current_model.write().await = Some(model);
                *self.current_model_name.write().await = Some(model_name.to_string());

                log::info!("Successfully loaded Gipformer model: {}", model_name);
                Ok(())
            }
            ModelStatus::Missing => Err(anyhow!("Gipformer model {} is not downloaded", model_name)),
            ModelStatus::Downloading { .. } => {
                Err(anyhow!("Gipformer model {} is currently downloading", model_name))
            }
            ModelStatus::Error(err) => Err(anyhow!("Gipformer model {} has error: {}", model_name, err)),
            ModelStatus::Corrupted { .. } => {
                Err(anyhow!("Gipformer model {} is corrupted and cannot be loaded", model_name))
            }
        }
    }

    /// Unload the current model
    pub async fn unload_model(&self) -> bool {
        let _lifecycle_guard = self.model_lifecycle_lock.lock().await;
        self.unload_model_locked().await
    }

    async fn unload_model_locked(&self) -> bool {
        let unloaded = self.current_model.write().await.take().is_some();
        if unloaded {
            log::info!("Gipformer model unloaded");
        }
        self.current_model_name.write().await.take();
        unloaded
    }

    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model_name.read().await.clone()
    }

    pub async fn is_model_loaded(&self) -> bool {
        self.current_model.read().await.is_some()
    }

    /// Transcribe audio samples (16kHz mono f32) using the loaded model.
    pub async fn transcribe_audio(&self, audio_data: Vec<f32>) -> Result<String> {
        let mut model_guard = self.current_model.write().await;
        let model = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("No Gipformer model loaded. Please load a model first."))?;

        let duration_seconds = audio_data.len() as f64 / 16000.0;
        log::debug!(
            "Gipformer transcribing {} samples ({:.1}s duration)",
            audio_data.len(),
            duration_seconds
        );

        let result = model
            .transcribe_samples(&audio_data)
            .map_err(|e| anyhow!("Gipformer transcription failed: {}", e))?;

        log::debug!("Gipformer transcription result: '{}'", result.text);
        Ok(result.text)
    }

    pub async fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }

    /// Delete a corrupted model
    pub async fn delete_model(&self, model_name: &str) -> Result<String> {
        log::info!("Attempting to delete Gipformer model: {}", model_name);

        let model_info = {
            let models = self.available_models.read().await;
            models.get(model_name).cloned()
        };

        let model_info =
            model_info.ok_or_else(|| anyhow!("Gipformer model '{}' not found", model_name))?;

        match &model_info.status {
            ModelStatus::Corrupted { .. } | ModelStatus::Available => {
                if model_info.path.exists() {
                    fs::remove_dir_all(&model_info.path).await.map_err(|e| {
                        anyhow!("Failed to delete directory '{}': {}", model_info.path.display(), e)
                    })?;
                    log::info!(
                        "Successfully deleted Gipformer model directory: {}",
                        model_info.path.display()
                    );
                }

                {
                    let mut models = self.available_models.write().await;
                    if let Some(model) = models.get_mut(model_name) {
                        model.status = ModelStatus::Missing;
                    }
                }

                Ok(format!("Successfully deleted Gipformer model '{}'", model_name))
            }
            _ => Err(anyhow!(
                "Can only delete corrupted or available Gipformer models. Model '{}' has status: {:?}",
                model_name,
                model_info.status
            )),
        }
    }

    /// Download a catalogued Gipformer model with detailed progress.
    pub async fn download_model_detailed(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<()> {
        let model_info = self
            .available_models
            .read()
            .await
            .get(model_name)
            .cloned()
            .ok_or_else(|| anyhow!("Model {} not found", model_name))?;
        let spec = find_model_spec(model_name)
            .ok_or_else(|| anyhow!("Unsupported Gipformer model: {}", model_name))?;

        self.download_model_detailed_from_source(
            model_name,
            &model_info.path,
            spec.source_base_url,
            spec.artifacts,
            progress_callback,
        )
        .await
    }

    async fn reserve_active_download(&self, model_name: &str) -> Result<Arc<ActiveDownload>> {
        let mut active_downloads = self.active_downloads.lock().await;
        if active_downloads.downloads.contains_key(model_name) {
            return Err(anyhow!("Download already in progress for model: {}", model_name));
        }

        let (completion, _) = watch::channel(false);
        let active_download = Arc::new(ActiveDownload {
            cancellation: CancellationToken::new(),
            completion,
        });
        active_downloads
            .downloads
            .insert(model_name.to_string(), Arc::clone(&active_download));
        active_downloads.revision = active_downloads.revision.wrapping_add(1);
        Ok(active_download)
    }

    async fn set_downloading_status(&self, model_name: &str, progress: u8) {
        let mut models = self.available_models.write().await;
        if let Some(model) = models.get_mut(model_name) {
            model.status = ModelStatus::Downloading { progress };
        }
    }

    fn in_flight_progress(downloaded: u64, total: u64, speed_mbps: f64) -> DownloadProgress {
        let mut progress = DownloadProgress::new(downloaded, total, speed_mbps);
        progress.percent = progress.percent.min(99);
        progress
    }

    async fn send_download_request(
        &self,
        client: &reqwest::Client,
        file_url: &str,
        range_start: Option<u64>,
        active_download: &ActiveDownload,
    ) -> Result<reqwest::Response> {
        let mut request = client.get(file_url);
        if let Some(range_start) = range_start {
            request = request.header(reqwest::header::RANGE, format!("bytes={range_start}-"));
        }

        tokio::select! {
            biased;
            _ = active_download.cancellation.cancelled() => Err(DownloadCancelled.into()),
            response = request.send() => response
                .map_err(|error| anyhow!("Failed to start download for {}: {}", file_url, error)),
        }
    }

    fn declared_content_length(response: &reqwest::Response) -> Result<Option<u64>> {
        response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .map(|value| {
                value
                    .to_str()
                    .map_err(|e| anyhow!("Invalid Content-Length header encoding: {}", e))?
                    .parse()
                    .map_err(|e| anyhow!("Invalid Content-Length header: {}", e))
            })
            .transpose()
    }

    fn validate_full_response(response: &reqwest::Response, exact_bytes: u64) -> Result<()> {
        if response.status() != reqwest::StatusCode::OK {
            return Err(anyhow!(
                "Expected full 200 response, received {}",
                response.status()
            ));
        }
        // HuggingFace resolve URLs serve LFS content directly with exact lengths.
        if let Some(content_length) = Self::declared_content_length(response)? {
            if content_length != exact_bytes {
                return Err(anyhow!(
                    "Full response declared {} bytes, expected {}",
                    content_length,
                    exact_bytes
                ));
            }
        }
        Ok(())
    }

    fn validate_partial_response(
        response: &reqwest::Response,
        expected_start: u64,
        exact_bytes: u64,
    ) -> Result<()> {
        if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(anyhow!(
                "Expected partial 206 response, received {}",
                response.status()
            ));
        }
        let content_range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .ok_or_else(|| anyhow!("Partial response is missing Content-Range"))?;
        let ContentRange::Range { start, end, total } = parse_content_range(content_range)? else {
            return Err(anyhow!("Partial response has an unsatisfied Content-Range"));
        };
        if start != expected_start || total != exact_bytes {
            return Err(anyhow!(
                "Partial response range {}-{}/{} does not match {}-*/{}",
                start,
                end,
                total,
                expected_start,
                exact_bytes
            ));
        }
        Ok(())
    }

    async fn download_model_detailed_from_source(
        &self,
        model_name: &str,
        model_dir: &Path,
        base_url: &str,
        artifacts: &[ArtifactSpec],
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<()> {
        let active_download = self.reserve_active_download(model_name).await?;
        self.set_downloading_status(model_name, 0).await;

        let result = self
            .download_artifacts(model_name, model_dir, base_url, artifacts, &active_download, progress_callback)
            .await;
        let (result, progress_callback) = match result {
            Ok((progress, progress_callback)) => (Ok(progress), progress_callback),
            Err(error) => (Err(error), None),
        };

        self.finish_download(model_name, model_dir, artifacts, &active_download, result, progress_callback)
            .await
    }

    async fn download_artifacts(
        &self,
        model_name: &str,
        model_dir: &Path,
        base_url: &str,
        artifacts: &[ArtifactSpec],
        active_download: &ActiveDownload,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<(DownloadProgress, Option<Box<dyn Fn(DownloadProgress) + Send>>)> {
        if active_download.cancellation.is_cancelled() {
            return Err(DownloadCancelled.into());
        }
        fs::create_dir_all(model_dir)
            .await
            .map_err(|e| anyhow!("Failed to create model directory: {}", e))?;

        let client = reqwest::Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(1)
            .timeout(Duration::from_secs(3600))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;
        let total_bytes: u64 = artifacts.iter().map(|a| a.exact_bytes).sum();
        let download_started = Instant::now();
        let mut confirmed_bytes = 0u64;
        let mut streamed_bytes = 0u64;
        let mut bytes_since_report = 0u64;
        let mut last_report = Instant::now();
        let mut last_percent = 0u8;

        for artifact in artifacts {
            if active_download.cancellation.is_cancelled() {
                return Err(DownloadCancelled.into());
            }

            let file_path = model_dir.join(artifact.filename);
            let local_bytes = match fs::metadata(&file_path).await {
                Ok(metadata) => metadata.len(),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
                Err(e) => {
                    return Err(anyhow!("Failed to read {} metadata: {}", artifact.filename, e));
                }
            };

            if local_bytes == artifact.exact_bytes {
                confirmed_bytes = confirmed_bytes
                    .checked_add(artifact.exact_bytes)
                    .ok_or_else(|| anyhow!("Progress overflow while skipping {}", artifact.filename))?;
                let progress = Self::in_flight_progress(confirmed_bytes, total_bytes, 0.0);
                if let Some(callback) = &progress_callback {
                    callback(progress.clone());
                }
                self.set_downloading_status(model_name, progress.percent).await;
                last_percent = progress.percent;
                last_report = Instant::now();
                continue;
            }

            let file_url = format!("{}/{}", base_url.trim_end_matches('/'), artifact.filename);
            let range_start = (local_bytes > 0 && local_bytes < artifact.exact_bytes).then_some(local_bytes);
            let response = self
                .send_download_request(&client, &file_url, range_start, active_download)
                .await?;

            let (response, mut artifact_bytes, append) = match range_start {
                Some(range_start) => match response.status() {
                    reqwest::StatusCode::PARTIAL_CONTENT => {
                        Self::validate_partial_response(&response, range_start, artifact.exact_bytes)?;
                        confirmed_bytes = confirmed_bytes
                            .checked_add(range_start)
                            .ok_or_else(|| anyhow!("Progress overflow while resuming {}", artifact.filename))?;
                        let progress = Self::in_flight_progress(confirmed_bytes, total_bytes, 0.0);
                        if let Some(callback) = &progress_callback {
                            callback(progress.clone());
                        }
                        self.set_downloading_status(model_name, progress.percent).await;
                        last_percent = progress.percent;
                        last_report = Instant::now();
                        (response, range_start, true)
                    }
                    reqwest::StatusCode::OK => {
                        Self::validate_full_response(&response, artifact.exact_bytes)?;
                        (response, 0, false)
                    }
                    reqwest::StatusCode::RANGE_NOT_SATISFIABLE => {
                        // 416 with unsatisfiable range: file may be complete or size mismatch.
                        let retry = self.send_download_request(&client, &file_url, None, active_download).await?;
                        Self::validate_full_response(&retry, artifact.exact_bytes)?;
                        (retry, 0, false)
                    }
                    status => {
                        return Err(anyhow!(
                            "Download failed for {} with status {}",
                            artifact.filename,
                            status
                        ));
                    }
                },
                None => match response.status() {
                    reqwest::StatusCode::OK => {
                        Self::validate_full_response(&response, artifact.exact_bytes)?;
                        (response, 0, false)
                    }
                    reqwest::StatusCode::PARTIAL_CONTENT => {
                        Self::validate_partial_response(&response, 0, artifact.exact_bytes)?;
                        (response, 0, false)
                    }
                    status => {
                        return Err(anyhow!(
                            "Download failed for {} with status {}",
                            artifact.filename,
                            status
                        ));
                    }
                },
            };

            if active_download.cancellation.is_cancelled() {
                return Err(DownloadCancelled.into());
            }
            let file = if append {
                fs::OpenOptions::new()
                    .append(true)
                    .open(&file_path)
                    .await
                    .map_err(|e| anyhow!("Failed to open {} for resume: {}", artifact.filename, e))?
            } else {
                fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(&file_path)
                    .await
                    .map_err(|e| anyhow!("Failed to replace {}: {}", artifact.filename, e))?
            };
            let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
            use futures_util::StreamExt;
            let mut stream = response.bytes_stream();

            loop {
                let next_chunk = tokio::select! {
                    biased;
                    _ = active_download.cancellation.cancelled() => {
                        writer.flush().await.map_err(|e| {
                            anyhow!("Failed to preserve {} during cancellation: {}", artifact.filename, e)
                        })?;
                        return Err(DownloadCancelled.into());
                    }
                    chunk = timeout(Duration::from_secs(30), stream.next()) => chunk,
                };
                let chunk = match next_chunk {
                    Err(_) => {
                        writer.flush().await.map_err(|e| {
                            anyhow!("Failed to preserve {} after timeout: {}", artifact.filename, e)
                        })?;
                        return Err(anyhow!(
                            "Download timeout for {}: no data received for 30 seconds",
                            artifact.filename
                        ));
                    }
                    Ok(None) => break,
                    Ok(Some(Err(error))) => {
                        writer.flush().await.map_err(|flush_error| {
                            anyhow!(
                                "Failed to preserve {} after stream error: {}",
                                artifact.filename,
                                flush_error
                            )
                        })?;
                        return Err(anyhow!("Download stream failed for {}: {}", artifact.filename, error));
                    }
                    Ok(Some(Ok(chunk))) => chunk,
                };

                let chunk_bytes = chunk.len() as u64;
                let next_artifact_bytes = artifact_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| anyhow!("{} size overflow", artifact.filename))?;
                if next_artifact_bytes > artifact.exact_bytes {
                    writer.flush().await.map_err(|e| {
                        anyhow!("Failed to preserve {} after overlong response: {}", artifact.filename, e)
                    })?;
                    return Err(anyhow!(
                        "{} response exceeds its exact {} byte size",
                        artifact.filename,
                        artifact.exact_bytes
                    ));
                }
                let next_confirmed_bytes = confirmed_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| anyhow!("Progress overflow while downloading {}", artifact.filename))?;
                if next_confirmed_bytes > total_bytes {
                    return Err(anyhow!("Download progress exceeds the catalog total"));
                }

                writer
                    .write_all(&chunk)
                    .await
                    .map_err(|e| anyhow!("Failed to write {}: {}", artifact.filename, e))?;
                artifact_bytes = next_artifact_bytes;
                confirmed_bytes = next_confirmed_bytes;
                streamed_bytes = streamed_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| anyhow!("Streamed byte count overflow"))?;
                bytes_since_report = bytes_since_report
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| anyhow!("Progress byte count overflow"))?;

                let progress = Self::in_flight_progress(confirmed_bytes, total_bytes, 0.0);
                let elapsed = last_report.elapsed();
                if progress.percent > last_percent
                    || elapsed >= Duration::from_millis(500)
                    || artifact_bytes == artifact.exact_bytes
                {
                    let speed_mbps = if elapsed.as_secs_f64() > 0.0 {
                        bytes_since_report as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };
                    if let Some(callback) = &progress_callback {
                        callback(Self::in_flight_progress(confirmed_bytes, total_bytes, speed_mbps));
                    }
                    self.set_downloading_status(model_name, progress.percent).await;
                    last_percent = progress.percent;
                    last_report = Instant::now();
                    bytes_since_report = 0;
                }
            }

            writer
                .flush()
                .await
                .map_err(|e| anyhow!("Failed to flush {}: {}", artifact.filename, e))?;
            drop(writer);

            if active_download.cancellation.is_cancelled() {
                return Err(DownloadCancelled.into());
            }
            let stored_bytes = fs::metadata(&file_path)
                .await
                .map_err(|e| anyhow!("Failed to read {} after download: {}", artifact.filename, e))?
                .len();
            if stored_bytes != artifact.exact_bytes {
                return Err(anyhow!(
                    "{} stored {} bytes, expected exactly {} bytes",
                    artifact.filename,
                    stored_bytes,
                    artifact.exact_bytes
                ));
            }
        }

        if confirmed_bytes != total_bytes {
            return Err(anyhow!(
                "Download confirmed {} bytes, expected {} bytes",
                confirmed_bytes,
                total_bytes
            ));
        }
        let elapsed = download_started.elapsed().as_secs_f64();
        let speed_mbps = if elapsed > 0.0 {
            streamed_bytes as f64 / (1024.0 * 1024.0) / elapsed
        } else {
            0.0
        };
        Ok((DownloadProgress::new(total_bytes, total_bytes, speed_mbps), progress_callback))
    }

    async fn finish_download(
        &self,
        model_name: &str,
        model_dir: &Path,
        artifacts: &[ArtifactSpec],
        active_download: &Arc<ActiveDownload>,
        mut result: Result<DownloadProgress>,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<()> {
        if result.is_ok() && !active_download.cancellation.is_cancelled() {
            let final_progress = result.expect("successful transfer must carry final progress");
            result = Self::validate_model_directory(model_dir, artifacts).map(|()| final_progress);
        }

        let mut active_downloads = self.active_downloads.lock().await;
        let owner_matches = matches!(
            active_downloads.downloads.get(model_name),
            Some(owner) if Arc::ptr_eq(owner, active_download)
        );
        if !owner_matches {
            drop(active_downloads);
            active_download.completion.send_replace(true);
            return result.map(|_| ());
        }

        let cancellation_won = active_download.cancellation.is_cancelled()
            || result
                .as_ref()
                .err()
                .is_some_and(is_download_cancelled);
        let mut models = self.available_models.write().await;
        active_downloads.downloads.remove(model_name);
        if let Some(model) = models.get_mut(model_name) {
            if cancellation_won || result.is_err() {
                model.status = ModelStatus::Missing;
            } else {
                model.status = ModelStatus::Available;
                model.path = model_dir.to_path_buf();
            }
        }
        active_downloads.revision = active_downloads.revision.wrapping_add(1);
        drop(models);
        drop(active_downloads);
        active_download.completion.send_replace(true);

        if cancellation_won {
            return Err(DownloadCancelled.into());
        }
        let final_progress = result?;
        if let Some(callback) = progress_callback {
            callback(final_progress);
        }
        log::info!("Download completed for Gipformer model: {}", model_name);
        Ok(())
    }

    /// Cancel an ongoing model download.
    pub async fn cancel_download(&self, model_name: &str) -> Result<CancelDownloadOutcome> {
        self.cancel_download_with_timeout(model_name, CANCEL_DOWNLOAD_CLEANUP_TIMEOUT)
            .await
    }

    async fn cancel_download_with_timeout(
        &self,
        model_name: &str,
        cleanup_timeout: Duration,
    ) -> Result<CancelDownloadOutcome> {
        let active_download = {
            let active_downloads = self.active_downloads.lock().await;
            let Some(active_download) = active_downloads.downloads.get(model_name).cloned() else {
                return Ok(CancelDownloadOutcome::Cancelled);
            };
            active_download.cancellation.cancel();
            active_download
        };

        let mut completion = active_download.completion.subscribe();
        if !*completion.borrow() {
            match timeout(cleanup_timeout, completion.changed()).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => {
                    return Err(anyhow!(
                        "Download worker ended before completing cancellation cleanup"
                    ));
                }
                Err(_) => return Ok(CancelDownloadOutcome::Pending),
            }
        }

        Ok(CancelDownloadOutcome::Cancelled)
    }
}
