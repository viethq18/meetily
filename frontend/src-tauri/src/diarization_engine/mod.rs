// diarization_engine/mod.rs
//
// Post-recording neural speaker diarization (pyannote segmentation-3.0 ONNX +
// 3dspeaker eres2net embedding + agglomerative clustering), following the
// sherpa-onnx offline-speaker-diarization pipeline (Apache-2.0).

pub mod clustering;
pub mod commands;
pub mod embedding;
pub mod engine;
pub mod model;
pub mod segmentation;

pub use engine::{DiarizationEngine, DiarizationResult, SpeakerSegment};
