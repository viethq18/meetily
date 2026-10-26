use ndarray::{Array, Array1, Array2, Array3, ArrayD, ArrayViewD, IxDyn};
use once_cell::sync::Lazy;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use regex::Regex;

use std::fs;
use std::path::Path;

pub type DecoderState = (Array3<f32>, Array3<f32>);

const SUBSAMPLING_FACTOR: usize = 8;
const WINDOW_SIZE: f32 = 0.01;

/// Decode architecture of a Parakeet ONNX export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderKind {
    /// Transducer with TDT durations: encoder + decoder_joint + preprocessor ONNX.
    Tdt,
    /// CTC single ONNX: input [N, 80, T] NeMo-80 mel features, output logprobs.
    Ctc,
}

impl DecoderKind {
    /// Infer the decode architecture from a model name (catalogued specs).
    pub fn from_model_name(model_name: &str) -> Self {
        if model_name.contains("ctc") {
            Self::Ctc
        } else {
            Self::Tdt
        }
    }
}
const MAX_TOKENS_PER_STEP: usize = 3;
const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

static DECODE_SPACE_RE: Lazy<Result<Regex, regex::Error>> =
    Lazy::new(|| Regex::new(r"\A\s|\s\B|(\s)\b"));

#[derive(Debug, Clone)]
pub struct TimestampedResult {
    pub text: String,
    pub timestamps: Vec<f32>,
    pub tokens: Vec<String>,
}

#[derive(thiserror::Error, Debug)]
pub enum ParakeetError {
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
    #[error("Failed to get tensor shape for input: {0}")]
    TensorShape(String),
    #[error("ONNX Runtime unavailable: {0}")]
    RuntimeUnavailable(String),
    #[error("Engine error: {0}")]
    EngineFailed(String),
}

pub struct ParakeetModel {
    /// Transducer (TDT) sessions: encoder + decoder_joint + preprocessor.
    encoder: Option<Session>,
    decoder_joint: Option<Session>,
    preprocessor: Option<Session>,
    /// CTC session: single ONNX taking [N, 80, T] NeMo-80 mel features.
    ctc: Option<Session>,
    decode_kind: DecoderKind,
    vocab: Vec<String>,
    blank_idx: i32,
    vocab_size: usize,
}

impl Drop for ParakeetModel {
    fn drop(&mut self) {
        log::debug!("Dropping ParakeetModel with {} vocab tokens", self.vocab.len());
    }
}

impl ParakeetModel {
    pub fn new<P: AsRef<Path>>(model_dir: P, quantized: bool) -> Result<Self, ParakeetError> {
        Self::new_with_kind(model_dir, quantized, DecoderKind::Tdt)
    }

    pub fn new_with_kind<P: AsRef<Path>>(
        model_dir: P,
        quantized: bool,
        decode_kind: DecoderKind,
    ) -> Result<Self, ParakeetError> {
        let (encoder, decoder_joint, preprocessor, ctc) = match decode_kind {
            DecoderKind::Tdt => {
                let encoder = Self::init_session(&model_dir, "encoder-model", None, quantized)?;
                let decoder_joint =
                    Self::init_session(&model_dir, "decoder_joint-model", None, quantized)?;
                let preprocessor = Self::init_session(&model_dir, "nemo128", None, false)?;
                (Some(encoder), Some(decoder_joint), Some(preprocessor), None)
            }
            DecoderKind::Ctc => {
                let ctc = Self::init_session(&model_dir, "model", None, quantized)?;
                (None, None, None, Some(ctc))
            }
        };

        let (vocab, blank_idx) = Self::load_vocab(&model_dir)?;
        let vocab_size = vocab.len();

        log::info!(
            "Loaded Parakeet vocabulary with {} tokens, blank_idx={}, decoder={:?}",
            vocab_size,
            blank_idx,
            decode_kind
        );

        Ok(Self {
            encoder,
            decoder_joint,
            preprocessor,
            ctc,
            decode_kind,
            vocab,
            blank_idx,
            vocab_size,
        })
    }

    fn init_session<P: AsRef<Path>>(
        model_dir: P,
        model_name: &str,
        intra_threads: Option<usize>,
        try_quantized: bool,
    ) -> Result<Session, ParakeetError> {
        crate::ensure_onnx_runtime_available()
            .map_err(|error| ParakeetError::RuntimeUnavailable(error.to_string()))?;
        let providers = vec![ort::ep::CPU::default().build()];


        // Try quantized version first if requested, fallback to regular version
        let model_filename = if try_quantized {
            let quantized_name = format!("{}.int8.onnx", model_name);
            let quantized_path = model_dir.as_ref().join(&quantized_name);
            if quantized_path.exists() {
                log::info!("Loading quantized Parakeet model from {}...", quantized_name);
                quantized_name
            } else {
                let regular_name = format!("{}.onnx", model_name);
                log::info!(
                    "Quantized model not found, loading regular Parakeet model from {}...",
                    regular_name
                );
                regular_name
            }
        } else {
            let regular_name = format!("{}.onnx", model_name);
            log::info!("Loading Parakeet model from {}...", regular_name);
            regular_name
        };
        let mut builder = Session::builder(crate::ort_env())?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?;

        if let Some(threads) = intra_threads {
            builder = builder
                .with_intra_threads(threads)?
                .with_inter_threads(threads)?;
        }

        let session = builder.commit_from_file(model_dir.as_ref().join(&model_filename))?;

        for input in session.inputs() {
            log::info!(
                "Parakeet Model '{}' input: name={}, type={:?}",
                model_filename,
                input.name(),
                input.dtype()
            );
        }

        Ok(session)
    }

    fn load_vocab<P: AsRef<Path>>(model_dir: P) -> Result<(Vec<String>, i32), ParakeetError> {
        let vocab_path = model_dir.as_ref().join("vocab.txt");
        let content = fs::read_to_string(vocab_path)?;

        let mut max_id = 0;
        let mut tokens_with_ids: Vec<(String, usize)> = Vec::new();
        let mut blank_idx: Option<usize> = None;

        for line in content.lines() {
            let parts: Vec<&str> = line.trim_end().split(' ').collect();
            if parts.len() >= 2 {
                let token = parts[0].to_string();
                if let Ok(id) = parts[1].parse::<usize>() {
                    if token == "<blk>" {
                        blank_idx = Some(id);
                    }
                    tokens_with_ids.push((token, id));
                    max_id = max_id.max(id);
                }
            }
        }

        // Create vocab vector with \u2581 replaced with space
        let mut vocab = vec![String::new(); max_id + 1];
        for (token, id) in tokens_with_ids {
            vocab[id] = token.replace('\u{2581}', " ");
        }

        let blank_idx = blank_idx.ok_or_else(|| {
            ParakeetError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing <blk> token in vocabulary",
            ))
        })? as i32;

        Ok((vocab, blank_idx))
    }

    pub fn preprocess(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_lens: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), ParakeetError> {
        log::trace!("Running Parakeet preprocessor inference...");
        let inputs = inputs![
            "waveforms" => TensorRef::from_array_view(waveforms.view())?,
            "waveforms_lens" => TensorRef::from_array_view(waveforms_lens.view())?,
        ];
        let outputs = self.preprocessor.as_mut().ok_or_else(|| ParakeetError::EngineFailed("preprocessor session not loaded (CTC model)".into()))?.run(inputs)?;

        let features = outputs
            .get("features")
            .ok_or_else(|| ParakeetError::OutputNotFound("features".to_string()))?
            .try_extract_array()?;
        let features_lens = outputs
            .get("features_lens")
            .ok_or_else(|| ParakeetError::OutputNotFound("features_lens".to_string()))?
            .try_extract_array()?;

        Ok((features.to_owned(), features_lens.to_owned()))
    }

    pub fn encode(
        &mut self,
        audio_signal: &ArrayViewD<f32>,
        length: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), ParakeetError> {
        log::trace!("Running Parakeet encoder inference...");
        let inputs = inputs![
            "audio_signal" => TensorRef::from_array_view(audio_signal.view())?,
            "length" => TensorRef::from_array_view(length.view())?,
        ];
        let outputs = self.encoder.as_mut().ok_or_else(|| ParakeetError::EngineFailed("encoder session not loaded (CTC model)".into()))?.run(inputs)?;

        let encoder_output = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?;
        let encoded_lengths = outputs
            .get("encoded_lengths")
            .ok_or_else(|| ParakeetError::OutputNotFound("encoded_lengths".to_string()))?
            .try_extract_array()?;

        let encoder_output = encoder_output.permuted_axes(IxDyn(&[0, 2, 1]));

        Ok((encoder_output.to_owned(), encoded_lengths.to_owned()))
    }

    pub fn create_decoder_state(&self) -> Result<DecoderState, ParakeetError> {
        // Get input shapes from decoder model
        let inputs = self.decoder_joint.as_ref().ok_or_else(|| ParakeetError::EngineFailed("decoder_joint session not loaded (CTC model)".into()))?.inputs();

        let state1_shape = inputs
            .iter()
            .find(|input| input.name() == "input_states_1")
            .ok_or_else(|| ParakeetError::InputNotFound("input_states_1".to_string()))?
            .dtype()
            .tensor_shape()
            .ok_or_else(|| ParakeetError::TensorShape("input_states_1".to_string()))?;

        let state2_shape = inputs
            .iter()
            .find(|input| input.name() == "input_states_2")
            .ok_or_else(|| ParakeetError::InputNotFound("input_states_2".to_string()))?
            .dtype()
            .tensor_shape()
            .ok_or_else(|| ParakeetError::TensorShape("input_states_2".to_string()))?;

        // Create zero states with batch_size=1
        // Shape is [2, -1, 640] so we use [2, 1, 640] for batch_size=1
        let state1 = Array::zeros((
            state1_shape[0] as usize,
            1, // batch_size = 1
            state1_shape[2] as usize,
        ));

        let state2 = Array::zeros((
            state2_shape[0] as usize,
            1, // batch_size = 1
            state2_shape[2] as usize,
        ));

        Ok((state1, state2))
    }

    pub fn decode_step(
        &mut self,
        prev_tokens: &[i32],
        prev_state: &DecoderState,
        encoder_out: &ArrayViewD<f32>, // [time_steps, 1024]
    ) -> Result<(ArrayD<f32>, DecoderState), ParakeetError> {
        log::trace!("Running Parakeet decoder inference...");

        // Get last token or blank_idx if empty
        let target_token = prev_tokens.last().copied().unwrap_or(self.blank_idx);

        // Prepare inputs matching Python: encoder_out[None, :, None] -> [1, time_steps, 1]
        let encoder_outputs = encoder_out
            .to_owned()
            .insert_axis(ndarray::Axis(0))
            .insert_axis(ndarray::Axis(2));
        let targets = Array2::from_shape_vec((1, 1), vec![target_token])?;
        let target_length = Array1::from_vec(vec![1]);

        let inputs = inputs![
            "encoder_outputs" => TensorRef::from_array_view(encoder_outputs.view())?,
            "targets" => TensorRef::from_array_view(targets.view())?,
            "target_length" => TensorRef::from_array_view(target_length.view())?,
            "input_states_1" => TensorRef::from_array_view(prev_state.0.view())?,
            "input_states_2" => TensorRef::from_array_view(prev_state.1.view())?,
        ];

        let outputs = self.decoder_joint.as_mut().ok_or_else(|| ParakeetError::EngineFailed("decoder_joint session not loaded (CTC model)".into()))?.run(inputs)?;

        let logits = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?;
        log::trace!(
            "Parakeet Logits shape: {:?}, vocab_size: {}",
            logits.shape(),
            self.vocab_size
        );
        let state1 = outputs
            .get("output_states_1")
            .ok_or_else(|| ParakeetError::OutputNotFound("output_states_1".to_string()))?
            .try_extract_array()?;
        let state2 = outputs
            .get("output_states_2")
            .ok_or_else(|| ParakeetError::OutputNotFound("output_states_2".to_string()))?
            .try_extract_array()?;

        // Squeeze outputs like Python (remove batch dimension)
        let logits = logits.remove_axis(ndarray::Axis(0));

        // Convert ArrayD back to Array3 to match expected return type
        let state1_3d = state1.to_owned().into_dimensionality::<ndarray::Ix3>()?;
        let state2_3d = state2.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        Ok((logits.to_owned(), (state1_3d, state2_3d)))
    }

    pub fn recognize_batch(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_len: &ArrayViewD<i64>,
    ) -> Result<Vec<TimestampedResult>, ParakeetError> {
        // Preprocess and encode
        let (features, features_lens) = self.preprocess(waveforms, waveforms_len)?;
        let (encoder_out, encoder_out_lens) =
            self.encode(&features.view(), &features_lens.view())?;

        // Decode for each batch item
        let mut results = Vec::new();
        for (encodings, &encodings_len) in encoder_out.outer_iter().zip(encoder_out_lens.iter()) {
            let (tokens, timestamps) =
                self.decode_sequence(&encodings.view(), encodings_len as usize)?;
            let result = self.decode_tokens(tokens, timestamps);
            results.push(result);
        }

        Ok(results)
    }

    fn decode_sequence(
        &mut self,
        encodings: &ArrayViewD<f32>, // [time_steps, 1024]
        encodings_len: usize,
    ) -> Result<(Vec<i32>, Vec<usize>), ParakeetError> {
        let mut prev_state = self.create_decoder_state()?;
        let mut tokens = Vec::new();
        let mut timestamps = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;

        while t < encodings_len {
            let encoder_step = encodings.slice(ndarray::s![t, ..]);
            // Convert to dynamic dimension to match decode_step parameter type
            let encoder_step_dyn = encoder_step.to_owned().into_dyn();
            let (probs, new_state) =
                self.decode_step(&tokens, &prev_state, &encoder_step_dyn.view())?;

            // For TDT models, split output into vocab logits and duration logits
            // output[:vocab_size] = vocabulary logits
            // output[vocab_size:] = duration logits
            let vocab_logits_slice = probs.as_slice().ok_or_else(|| {
                ParakeetError::Shape(ndarray::ShapeError::from_kind(
                    ndarray::ErrorKind::IncompatibleShape,
                ))
            })?;

            let is_tdt = probs.len() > self.vocab_size;
            let (vocab_logits, duration_logits) = if is_tdt {
                let (v, d) = vocab_logits_slice.split_at(self.vocab_size);
                (v, Some(d))
            } else {
                (vocab_logits_slice, None)
            };

            // Get argmax token from vocabulary logits only
            let token = vocab_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i32)
                .unwrap_or(self.blank_idx);

            if token != self.blank_idx {
                prev_state = new_state;
                tokens.push(token);
                timestamps.push(t);
                emitted_tokens += 1;
            }

            if let Some(duration_logits) = duration_logits {
                // TDT: advance by the model's predicted duration (frames to skip).
                let dur_idx = duration_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let mut skip = TDT_DURATIONS.get(dur_idx).copied().unwrap_or(1);

                // Ensure forward progress on blank-with-zero-duration, and cap
                // same-frame emissions to avoid runaway repetition.
                if skip == 0 && (token == self.blank_idx || emitted_tokens >= MAX_TOKENS_PER_STEP) {
                    skip = 1;
                }
                if skip > 0 {
                    t += skip;
                    emitted_tokens = 0;
                }
            } else {
                // RNN-T greedy: advance one frame on blank or after emission cap.
                if token == self.blank_idx || emitted_tokens >= MAX_TOKENS_PER_STEP {
                    t += 1;
                    emitted_tokens = 0;
                }
            }
        }

        // NEW: Log if no tokens were decoded (helps debugging empty transcriptions)
        if tokens.is_empty() {
            log::debug!(
                "Parakeet decoded zero tokens (all blank) for audio with {} encoding timesteps - audio may be too short or low energy",
                encodings_len
            );
        }

        Ok((tokens, timestamps))
    }

    fn decode_tokens(&self, ids: Vec<i32>, timestamps: Vec<usize>) -> TimestampedResult {
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

        let text = match &*DECODE_SPACE_RE {
            Ok(regex) => regex
                .replace_all(&tokens.join(""), |caps: &regex::Captures| {
                    if caps.get(1).is_some() {
                        " "
                    } else {
                        ""
                    }
                })
                .to_string(),
            Err(_) => tokens.join(""), // Fallback if regex failed to compile
        };

        let float_timestamps: Vec<f32> = timestamps
            .iter()
            .map(|&t| WINDOW_SIZE * SUBSAMPLING_FACTOR as f32 * t as f32)
            .collect();

        TimestampedResult {
            text,
            timestamps: float_timestamps,
            tokens,
        }
    }

    pub fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<TimestampedResult, ParakeetError> {
        match self.decode_kind {
            DecoderKind::Ctc => self.transcribe_samples_ctc(samples),
            DecoderKind::Tdt => self.transcribe_samples_tdt(samples),
        }
    }

    fn transcribe_samples_tdt(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<TimestampedResult, ParakeetError> {
        let batch_size = 1;
        let samples_len = samples.len();

        // Create waveforms array [batch_size, samples_len]
        let waveforms = Array2::from_shape_vec((batch_size, samples_len), samples)?.into_dyn();

        // Create waveforms_lens array [batch_size] with the actual length
        let waveforms_lens = Array1::from_vec(vec![samples_len as i64]).into_dyn();

        // Run recognition to get detailed results
        let results = self.recognize_batch(&waveforms.view(), &waveforms_lens.view())?;

        // Extract the first (and only) result
        let timestamped_result = results.into_iter().next().ok_or_else(|| {
            ParakeetError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No transcription result returned",
            ))
        })?;

        Ok(timestamped_result)
    }

    /// CTC path: NeMo-80 mel features -> single ONNX -> greedy CTC decode.
    fn transcribe_samples_ctc(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<TimestampedResult, ParakeetError> {
        let blank_idx = self.blank_idx;
        let (features, frames) = crate::parakeet_engine::nemo80_features::compute_nemo80_features(&samples);
        if frames == 0 {
            return Ok(TimestampedResult {
                text: String::new(),
                timestamps: Vec::new(),
                tokens: Vec::new(),
            });
        }

        // Features [T, 80] -> model input [1, 80, T]
        let mut transposed = vec![0f32; frames * 80];
        for t in 0..frames {
            for b in 0..80 {
                transposed[b * frames + t] = features[t * 80 + b];
            }
        }
        let audio_signal = Array3::from_shape_vec((1, 80, frames), transposed)?;
        let length = Array1::from_vec(vec![frames as i64]);

        let outputs = {
            let ctc = self.ctc.as_mut().ok_or_else(|| {
                ParakeetError::EngineFailed("CTC session not loaded (TDT model)".to_string())
            })?;
            ctc.run(inputs![
                "audio_signal" => TensorRef::from_array_view(audio_signal.view())?,
                "length" => TensorRef::from_array_view(length.view())?,
            ])?
        };


        let logprobs = outputs
            .get("logprobs")
            .ok_or_else(|| ParakeetError::OutputNotFound("logprobs".to_string()))?
            .try_extract_array::<f32>()?;

        // [1, T', vocab] -> [T', vocab]
        let logprobs = logprobs.to_owned().remove_axis(ndarray::Axis(0));
        let logprobs = logprobs.into_dimensionality::<ndarray::Ix2>()?;

        // Greedy CTC: argmax per timestep, collapse repeats, strip blank.
        let mut ids: Vec<i32> = Vec::new();
        let mut timestamps: Vec<usize> = Vec::new();
        let mut prev: i32 = -1;
        for (t, row) in logprobs.outer_iter().enumerate() {
            let token = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i32)
                .unwrap_or(blank_idx);
            if token != prev && token != blank_idx {
                ids.push(token);
                timestamps.push(t);
            }
            prev = token;
        }

        if ids.is_empty() {
            log::debug!(
                "Parakeet CTC decoded zero tokens for {} timesteps - audio may be silence",
                logprobs.nrows()
            );
        }
        drop(outputs);

        // Reuse the TDT token-to-text path (same vocab format and spacing rules).
        let mut result = self.decode_tokens(ids, timestamps);
        // CTC emits one timestep per 8 frames (0.08s steps), same subsampling as TDT.
        result.timestamps = result
            .timestamps
            .iter()
            .map(|&t| WINDOW_SIZE * SUBSAMPLING_FACTOR as f32 * t as f32)
            .collect();
        Ok(result)
    }
}
