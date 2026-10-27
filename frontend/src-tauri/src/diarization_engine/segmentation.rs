// diarization_engine/segmentation.rs
//
// pyannote segmentation-3.0 ONNX inference (window 160000 samples @ 16kHz,
// shift = 10% of window, output [frames, 7] powerset logits) plus the
// powerset -> per-speaker multilabel conversion used by sherpa-onnx.
//
// IO contract (verified against the sherpa-onnx export):
//   input  x: [1, 1, 160000] f32
//   output y: [1, T, 7] f32 powerset logits
//   per-frame stride: receptive_field_shift = 270 samples,
//   receptive_field_size = 991 samples (frame t covers samples
//   [t*270, t*270 + 991)).
//
// Powerset classes (num_speakers=3, powerset_max_classes=2):
//   0: {}            1: {spk0}   2: {spk1}   3: {spk2}
//   4: {spk0,spk1}   5: {spk0,spk2}   6: {spk1,spk2}

use anyhow::{anyhow, Result};
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
enum SegmentationLoadError {
    #[error("ORT session builder error: {0}")]
    Builder(String),
    #[error("ORT error: {0}")]
    Ort(String),
}

impl From<ort::Error<ort::session::builder::SessionBuilder>> for SegmentationLoadError {
    fn from(e: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        SegmentationLoadError::Builder(e.to_string())
    }
}

impl From<ort::Error> for SegmentationLoadError {
    fn from(e: ort::Error) -> Self {
        SegmentationLoadError::Ort(e.to_string())
    }
}

pub const NUM_SPEAKERS: usize = 3;
pub const NUM_CLASSES: usize = 7;
pub const WINDOW_SIZE: usize = 160_000;
pub const WINDOW_SHIFT_RATIO: f32 = 0.1;
pub const RECEPTIVE_FIELD_SIZE: usize = 991;
pub const RECEPTIVE_FIELD_SHIFT: usize = 270;

/// Powerset class -> per-speaker 0/1 mapping (row = class, col = speaker).
pub fn powerset_mapping() -> Array2<i32> {
    let mut m = Array2::<i32>::zeros((NUM_CLASSES, NUM_SPEAKERS));
    m[[1, 0]] = 1;
    m[[2, 1]] = 1;
    m[[3, 2]] = 1;
    m[[4, 0]] = 1;
    m[[4, 1]] = 1;
    m[[5, 0]] = 1;
    m[[5, 2]] = 1;
    m[[6, 1]] = 1;
    m[[6, 2]] = 1;
    m
}

pub struct PyannoteSegmentationModel {
    session: Session,
}

impl PyannoteSegmentationModel {
    pub fn load(model_path: &Path) -> Result<Self> {
        crate::ensure_onnx_runtime_available()?;
        let session: Session = (|| -> std::result::Result<Session, SegmentationLoadError> {
            Ok(Session::builder(crate::ort_env())?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers(vec![ort::ep::CPU::default().build()])?
                .with_parallel_execution(true)?
                .commit_from_file(model_path)?)
        })()?;
        log::info!(
            "Loaded pyannote segmentation model from {}",
            model_path.display()
        );
        Ok(Self { session })
    }

    /// Run one window through the model; returns [num_frames, NUM_CLASSES]
    /// (batch dimension removed).
    fn process_window(&mut self, window: &[f32]) -> Result<Array2<f32>> {
        let x = ndarray::Array3::<f32>::from_shape_vec((1, 1, window.len()), window.to_vec())
            .map_err(|e| anyhow!("window reshape: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "x" => TensorRef::from_array_view(x.view())?,
            ])
            .map_err(|e| anyhow!("segmentation inference: {e}"))?;

        let tensor = outputs
            .get("y")
            .ok_or_else(|| anyhow!("segmentation model returned no 'y' output"))?
            .try_extract_array()
            .map_err(|e| anyhow!("segmentation output extraction: {e}"))?;

        // [1, T, 7] -> [T, 7]
        let squeezed = tensor.to_owned().remove_axis(ndarray::Axis(0));
        let shape = squeezed.shape();
        let cols = shape[shape.len() - 1];
        if cols != NUM_CLASSES {
            return Err(anyhow!(
                "segmentation model output has {} classes, expected {}",
                cols,
                NUM_CLASSES
            ));
        }
        Ok(squeezed
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| anyhow!("output reshape: {e}"))?)
    }

    /// Run the segmentation model over the whole waveform with the sliding
    /// window used by sherpa-onnx (shift = ratio * window, zero-padded tail
    /// chunk). Returns one [frames, NUM_CLASSES] matrix per chunk.
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<Array2<f32>>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }
        let window_shift = (WINDOW_SIZE as f32 * WINDOW_SHIFT_RATIO) as usize;
        let mut chunks = Vec::new();

        if audio.len() <= WINDOW_SIZE {
            let mut window = vec![0f32; WINDOW_SIZE];
            window[..audio.len()].copy_from_slice(audio);
            chunks.push(self.process_window(&window)?);
            return Ok(chunks);
        }

        let num_chunks = (audio.len() - WINDOW_SIZE) / window_shift + 1;
        let has_last = ((audio.len() - WINDOW_SIZE) % window_shift) > 0;

        for i in 0..num_chunks {
            let start = i * window_shift;
            let window = &audio[start..start + WINDOW_SIZE];
            chunks.push(self.process_window(window)?);
        }
        if has_last {
            let start = num_chunks * window_shift;
            let mut window = vec![0f32; WINDOW_SIZE];
            let tail = &audio[start.min(audio.len())..];
            window[..tail.len()].copy_from_slice(tail);
            chunks.push(self.process_window(&window)?);
        }
        Ok(chunks)
    }
}

/// Convert powerset logits to per-speaker 0/1 labels: argmax over classes,
/// then map the winning class to its speaker set.
pub fn to_multi_label(powerset: &Array2<f32>, mapping: &Array2<i32>) -> Array2<i32> {
    let frames = powerset.nrows();
    let mut labels = Array2::<i32>::zeros((frames, mapping.ncols()));
    for f in 0..frames {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..powerset.ncols() {
            let v = powerset[[f, c]];
            if v > best_val {
                best_val = v;
                best = c;
            }
        }
        for s in 0..mapping.ncols() {
            labels[[f, s]] = mapping[[best, s]];
        }
    }
    labels
}
