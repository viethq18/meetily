// diarization_engine/embedding.rs
//
// 3dspeaker eres2net speaker-embedding ONNX model (input [N, T, 80] fbank,
// output [N, 512]). Features reuse gipformer_engine's kaldi-style fbank
// (80 mel bins, 16 kHz, 25 ms window / 10 ms hop, snip_edges=false), matching
// the extractor config used by sherpa-onnx for these models (frame_shift 10 ms,
// frame_length 25 ms); normalize_samples=1 means waveforms stay in [-1, 1].
// feature_normalize_type=global-mean: subtract the per-bin mean over frames.

use anyhow::{anyhow, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
enum EmbeddingLoadError {
    #[error("ORT session builder error: {0}")]
    Builder(String),
    #[error("ORT error: {0}")]
    Ort(String),
}

impl From<ort::Error<ort::session::builder::SessionBuilder>> for EmbeddingLoadError {
    fn from(e: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        EmbeddingLoadError::Builder(e.to_string())
    }
}

impl From<ort::Error> for EmbeddingLoadError {
    fn from(e: ort::Error) -> Self {
        EmbeddingLoadError::Ort(e.to_string())
    }
}

pub struct SpeakerEmbeddingModel {
    session: Session,
    embedding_dim: usize,
}

impl SpeakerEmbeddingModel {
    pub fn load(model_path: &Path) -> Result<Self> {
        crate::ensure_onnx_runtime_available()?;
        let session: Session = (|| -> std::result::Result<Session, EmbeddingLoadError> {
            Ok(Session::builder(crate::ort_env())?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers(vec![ort::ep::CPU::default().build()])?
                .with_parallel_execution(true)?
                .commit_from_file(model_path)?)
        })()?;
        // Output shape is dynamic ([N, 512]); we learn the true dim at the
        // first inference call, defaulting to the known 512.
        let embedding_dim = 512usize;
        log::info!(
            "Loaded speaker embedding model from {} (dim {})",
            model_path.display(),
            embedding_dim
        );
        Ok(Self {
            session,
            embedding_dim,
        })
    }

    pub fn dim(&self) -> usize {
        self.embedding_dim
    }

    /// Compute the embedding for one speaker's audio span (already 16 kHz).
    /// Returns None when the span is too short for even one fbank frame.
    pub fn compute(&mut self, audio: &[f32]) -> Result<Option<Vec<f32>>> {
        let (features, num_frames) = crate::gipformer_engine::features::compute_fbank(audio);
        if num_frames == 0 {
            return Ok(None);
        }

        // global-mean normalization over frames (per mel bin).
        let feat_dim = features.len() / num_frames;
        let mut features = features;
        let mut mean = vec![0f32; feat_dim];
        for f in 0..num_frames {
            for b in 0..feat_dim {
                mean[b] += features[f * feat_dim + b];
            }
        }
        for m in mean.iter_mut() {
            *m /= num_frames as f32;
        }
        for f in 0..num_frames {
            for b in 0..feat_dim {
                features[f * feat_dim + b] -= mean[b];
            }
        }

        let x = ndarray::Array3::<f32>::from_shape_vec((1, num_frames, feat_dim), features)
            .map_err(|e| anyhow!("embedding feature reshape: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![
                "x" => TensorRef::from_array_view(x.view())?,
            ])
            .map_err(|e| anyhow!("embedding inference: {e}"))?;
        let tensor = outputs
            .get("embedding")
            .ok_or_else(|| anyhow!("embedding model returned no 'embedding' output"))?
            .try_extract_array()
            .map_err(|e| anyhow!("embedding output extraction: {e}"))?;

        let view = tensor.view();
        if view.ndim() != 2 {
            return Err(anyhow!(
                "embedding output ndim {} != 2",
                view.ndim()
            ));
        }
        let dim = view.shape()[1];
        let mut embedding = vec![0f32; dim];
        for (i, slot) in embedding.iter_mut().enumerate() {
            *slot = view[[0, i]];
        }

        // Skip NaN embeddings (the model can emit them for degenerate input);
        // the engine treats those chunks as invalid.
        if embedding.iter().any(|v| v.is_nan()) {
            return Ok(None);
        }
        Ok(Some(embedding))
    }
}
