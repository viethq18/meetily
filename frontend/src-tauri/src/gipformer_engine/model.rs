// gipformer_engine/model.rs
//
// Zipformer transducer ONNX model (sherpa-onnx format: encoder + decoder + joiner).
//
// IO contract (verified against g-group-ai-lab/gipformer1.5-68M-rnnt):
//   encoder:  x [N, T, 80] f32, x_lens [N] i64
//             -> encoder_out [N, T', 512] f32, encoder_out_lens [N] i64
//   decoder:  y [N, 2] i64 (token context) -> decoder_out [N, 512] f32 (fp32 out)
//   joiner:   encoder_out [N, 512], decoder_out [N, 512] -> logit [N, vocab] f32
//
// Decode: greedy RNNT with context size 2, max 1 token per frame, decoder
// outputs cached by token context (mirrors onnx-asr KaldiTransducer).

use ndarray::{Array1, Array2, Array3};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::gipformer_engine::features::compute_fbank;

const CONTEXT_SIZE: usize = 2;
const MAX_TOKENS_PER_STEP: usize = 1;

#[derive(thiserror::Error, Debug)]
pub enum GipformerModelError {
    #[error("ORT error")]
    Ort(#[from] ort::Error),
    #[error("ORT session builder error")]
    Builder(#[from] ort::Error<ort::session::builder::SessionBuilder>),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model input not found: {0}")]
    InputNotFound(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("Invalid vocabulary: {0}")]
    InvalidVocab(String),
    #[error("ONNX Runtime unavailable: {0}")]
    RuntimeUnavailable(String),
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f32>,
}

pub struct GipformerModel {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    vocab: Vec<String>,
    blank_idx: i64,
    vocab_size: usize,
}

impl Drop for GipformerModel {
    fn drop(&mut self) {
        log::debug!("Dropping GipformerModel with {} vocab tokens", self.vocab.len());
    }
}

impl GipformerModel {
    pub fn new<P: AsRef<Path>>(model_dir: P, allow_int8_fallback: bool) -> Result<Self, GipformerModelError> {
        let encoder = Self::init_session(&model_dir, "encoder", allow_int8_fallback)?;
        let decoder = Self::init_session(&model_dir, "decoder", allow_int8_fallback)?;
        let joiner = Self::init_session(&model_dir, "joiner", allow_int8_fallback)?;
        let (vocab, blank_idx) = Self::load_vocab(&model_dir)?;

        let vocab_size = vocab.len();
        log::info!(
            "Loaded Gipformer vocabulary with {} tokens, blank_idx={}",
            vocab_size,
            blank_idx
        );

        Ok(Self {
            encoder,
            decoder,
            joiner,
            vocab,
            blank_idx,
            vocab_size,
        })
    }

    fn init_session<P: AsRef<Path>>(
        model_dir: P,
        base_name: &str,
        allow_int8_fallback: bool,
    ) -> Result<Session, GipformerModelError> {
        crate::ensure_onnx_runtime_available()
            .map_err(|error| GipformerModelError::RuntimeUnavailable(error.to_string()))?;

        let providers = vec![ort::ep::CPU::default().build()];

        // Prefer full-precision fp32, fall back to int8 when absent.
        let candidates: Vec<String> = if allow_int8_fallback {
            vec![
                format!("{}.onnx", base_name),
                format!("{}.int8.onnx", base_name),
            ]
        } else {
            vec![format!("{}.onnx", base_name)]
        };

        let model_filename = candidates
            .iter()
            .find(|name| model_dir.as_ref().join(name).exists())
            .ok_or_else(|| {
                GipformerModelError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("No ONNX file found for '{}' (tried {:?})", base_name, candidates),
                ))
            })?
            .clone();
        log::info!("Loading Gipformer {} from {}...", base_name, model_filename);

        let session = Session::builder(crate::ort_env())?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?
            .commit_from_file(model_dir.as_ref().join(&model_filename))?;

        for input in session.inputs() {
            log::info!(
                "Gipformer '{}' input: name={}, type={:?}",
                model_filename,
                input.name(),
                input.dtype()
            );
        }

        Ok(session)
    }

    /// Load sherpa-onnx `tokens.txt`: lines of "<token> <id>", ids are 0-based
    /// and contiguous for gipformer (0..1999). \u2581 maps to space.
    fn load_vocab<P: AsRef<Path>>(model_dir: P) -> Result<(Vec<String>, i64), GipformerModelError> {
        let vocab_path = model_dir.as_ref().join("tokens.txt");
        let content = fs::read_to_string(vocab_path)?;

        let mut max_id: i64 = 0;
        let mut blank_idx: Option<i64> = None;
        let mut tokens: Vec<(String, i64)> = Vec::new();

        for line in content.lines() {
            let line = line.trim_end_matches('\r');
            if let Some((token, id)) = line.rsplit_once(' ') {
                if let Ok(id) = id.parse::<i64>() {
                    if token == "<blk>" {
                        blank_idx = Some(id);
                    }
                    tokens.push((token.to_string(), id));
                    max_id = max_id.max(id);
                }
            }
        }

        if tokens.is_empty() {
            return Err(GipformerModelError::InvalidVocab(
                "tokens.txt contains no parseable entries".to_string(),
            ));
        }

        let mut vocab = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in tokens {
            let idx = id as usize;
            if idx < vocab.len() {
                vocab[idx] = token.replace('\u{2581}', " ");
            }
        }

        let blank_idx = blank_idx.ok_or_else(|| {
            GipformerModelError::InvalidVocab("Missing <blk> token in vocabulary".to_string())
        })?;

        Ok((vocab, blank_idx))
    }

    /// Run the encoder over fbank features.
    ///
    /// `features` is a flat row-major [T, 80] buffer with `frames` valid rows.
    /// Returns encoder output [T', 512] (batch 1 squeezed) and the valid length.
    fn encode(
        &mut self,
        features: &[f32],
        frames: usize,
    ) -> Result<(Array2<f32>, usize), GipformerModelError> {
        let x = Array3::from_shape_vec((1, frames, 80), features.to_vec())?;
        let x_lens = Array1::from_vec(vec![frames as i64]);

        let outputs = self.encoder.run(inputs![
            "x" => TensorRef::from_array_view(x.view())?,
            "x_lens" => TensorRef::from_array_view(x_lens.view())?,
        ])?;

        let encoder_out = outputs
            .get("encoder_out")
            .ok_or_else(|| GipformerModelError::OutputNotFound("encoder_out".to_string()))?
            .try_extract_array()?;
        let encoder_out_lens = outputs
            .get("encoder_out_lens")
            .ok_or_else(|| GipformerModelError::OutputNotFound("encoder_out_lens".to_string()))?
            .try_extract_array()?;

        // [1, T', 512] -> [T', 512]
        let squeezed = encoder_out.to_owned().remove_axis(ndarray::Axis(0));
        let enc_len = encoder_out_lens
            .as_slice()
            .and_then(|s| s.first().copied())
            .unwrap_or(frames as i64) as usize;

        let two_d = squeezed.into_dimensionality::<ndarray::Ix2>()?;
        Ok((two_d, enc_len))
    }

    /// Run the decoder for a token context (cached). Returns decoder_out [512].
    fn decoder_out(
        &mut self,
        context: [i64; CONTEXT_SIZE],
        cache: &mut HashMap<[i64; CONTEXT_SIZE], Array1<f32>>,
    ) -> Result<Array1<f32>, GipformerModelError> {
        if let Some(cached) = cache.get(&context) {
            return Ok(cached.clone());
        }

        let y = Array2::from_shape_vec((1, CONTEXT_SIZE), vec![context[0], context[1]])?;
        let outputs = self.decoder.run(inputs![
            "y" => TensorRef::from_array_view(y.view())?,
        ])?;

        let decoder_out = outputs
            .get("decoder_out")
            .ok_or_else(|| GipformerModelError::OutputNotFound("decoder_out".to_string()))?
            .try_extract_array()?;

        // [1, 512] -> [512]
        let squeezed = decoder_out.to_owned().remove_axis(ndarray::Axis(0));
        let one_d = squeezed.into_dimensionality::<ndarray::Ix1>()?;
        cache.insert(context, one_d.clone());
        Ok(one_d)
    }

    /// Run the joiner for one encoder frame + decoder output. Returns logits [vocab].
    fn joiner_logits(
        &mut self,
        encoder_frame: ndarray::ArrayView1<f32>,
        decoder_out: &Array1<f32>,
    ) -> Result<Vec<f32>, GipformerModelError> {
        let enc = encoder_frame.to_owned().insert_axis(ndarray::Axis(0)); // [1, 512]
        let dec = decoder_out.to_owned().insert_axis(ndarray::Axis(0)); // [1, 512]

        let outputs = self.joiner.run(inputs![
            "encoder_out" => TensorRef::from_array_view(enc.view())?,
            "decoder_out" => TensorRef::from_array_view(dec.view())?,
        ])?;

        let logit = outputs
            .get("logit")
            .ok_or_else(|| GipformerModelError::OutputNotFound("logit".to_string()))?
            .try_extract_array()?;

        let squeezed = logit.to_owned().remove_axis(ndarray::Axis(0));
        let one_d = squeezed.into_dimensionality::<ndarray::Ix1>()?;
        Ok(one_d.to_vec())
    }

    fn argmax(logits: &[f32], fallback: i64) -> i64 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as i64)
            .unwrap_or(fallback)
    }

    /// Greedy RNNT decode over encoder output.
    fn decode_greedy(
        &mut self,
        encoder_out: &Array2<f32>,
        enc_len: usize,
    ) -> Result<(Vec<i64>, Vec<usize>), GipformerModelError> {
        let mut cache: HashMap<[i64; CONTEXT_SIZE], Array1<f32>> = HashMap::new();
        let mut tokens: Vec<i64> = Vec::new();
        let mut timestamps: Vec<usize> = Vec::new();

        let mut t = 0usize;
        let mut emitted = 0usize;

        while t < enc_len {
            let frame = encoder_out.slice(ndarray::s![t, ..]);
            // Context: last CONTEXT_SIZE tokens, padded with -1 and blank.
            let last = tokens.last().copied().unwrap_or(self.blank_idx);
            let second_last = if tokens.len() >= 2 {
                tokens[tokens.len() - 2]
            } else {
                -1
            };
            let context = [second_last, last];

            let decoder_out = self.decoder_out(context, &mut cache)?;
            let logits = self.joiner_logits(frame, &decoder_out)?;

            let token = if logits.len() <= self.vocab_size.max(1) {
                Self::argmax(&logits, self.blank_idx)
            } else {
                // Unexpected extra logits: argmax over vocab slice only.
                Self::argmax(&logits[..self.vocab_size], self.blank_idx)
            };

            if token != self.blank_idx {
                tokens.push(token);
                timestamps.push(t);
                emitted += 1;
            }

            // Advance one frame on blank or per-frame emission cap.
            if token == self.blank_idx || emitted >= MAX_TOKENS_PER_STEP {
                t += 1;
                emitted = 0;
            }
        }

        if tokens.is_empty() {
            log::debug!(
                "Gipformer decoded zero tokens for {} encoding timesteps - audio may be too short or low energy",
                enc_len
            );
        }

        Ok((tokens, timestamps))
    }

    /// Convert token ids to text using the same spacing cleanup as the
    /// parakeet engine (\u2581 already mapped to space at vocab load).
    fn tokens_to_text(&self, ids: &[i64]) -> (String, Vec<String>) {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| {
                let idx = id as usize;
                if idx < self.vocab.len() {
                    Some(self.vocab[idx].clone())
                } else {
                    None
                }
            })
            .collect();
        (tokens.join(""), tokens)
    }

    /// Transcribe a mono 16kHz waveform.
    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<TranscriptionResult, GipformerModelError> {
        let (features, frames) = compute_fbank(samples);
        if frames == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
            });
        }
        let (encoder_out, enc_len) = self.encode(&features, frames)?;
        let (ids, timestamps) = self.decode_greedy(&encoder_out, enc_len)?;
        let (text, tokens) = self.tokens_to_text(&ids);
        // First token usually carries a leading ▁ (space); onnx-asr strips it.
        let text = text.trim_start().to_string();

        // 10ms hop * subsampling; encoder emits one step per 4 frames
        // (zipformer subsampling factor 4), timestamps are in encoder steps.
        let float_timestamps: Vec<f32> = timestamps
            .iter()
            .map(|&t| 0.01 * 4.0 * t as f32)
            .collect();


        Ok(TranscriptionResult {
            text,
            tokens,
            timestamps: float_timestamps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_max_and_falls_back() {
        assert_eq!(GipformerModel::argmax(&[0.1, 3.0, 2.0], -1), 1);
        assert_eq!(GipformerModel::argmax(&[], 5), 5);
        // NaN-safe: falls back to comparison ordering, never panics
        let _ = GipformerModel::argmax(&[f32::NAN, 1.0], 0);
    }

    #[test]
    fn vocab_parsing_from_string() {
        // tokens.txt format: "<token> <id>" with \u2581 -> space
        let dir = std::env::temp_dir().join("gipformer_vocab_test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokens.txt"),
            "<blk> 0\n<sos/eos> 1\n\u{2581}XIN 2\nCH\u{00C0}O 3\n",
        )
        .unwrap();
        let (vocab, blank) = GipformerModel::load_vocab(&dir).unwrap();
        assert_eq!(blank, 0);
        assert_eq!(vocab.len(), 4);
        assert_eq!(vocab[0], "<blk>");
        assert_eq!(vocab[2], " XIN");
        assert_eq!(vocab[3], "CHÀO");
    }
}
