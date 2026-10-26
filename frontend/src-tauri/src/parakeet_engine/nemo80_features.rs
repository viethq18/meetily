// parakeet_engine/nemo80_features.rs
//
// NeMo 80-bin log-mel feature extraction with per-utterance CMVN, matching
// onnx-asr's NemoPreprocessorNumpy (Apache-2.0). Used by Parakeet CTC models
// exported in NeMo format (input [N, 80, T] mel features).
//
// Pipeline:
//   1. preemphasis on the raw waveform: w[n] - 0.97 * w[n-1] (edge-padded),
//      zeroed beyond the utterance length
//   2. zero-pad n_fft/2 (256) on both sides
//   3. frame into 512-sample windows every 160 samples (hop)
//   4. apply hann window (400) zero-padded to 512
//   5. 512-point rfft power spectrum (257 bins)
//   6. mel projection (80 bins), log with 2^-24 guard
//   7. per-utterance CMVN: (x - mean) / (sqrt(var) + 1e-5), computed over the
//      valid frames only; frames beyond features_len are zeroed
//
// features_len = num_samples / hop

use realfft::RealFftPlanner;

use crate::parakeet_engine::nemo80_mel::{NEMO80_MEL_LENS, NEMO80_MEL_STARTS, NEMO80_MEL_VALUES};

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 1.0 / (1 << 24) as f32; // 2^-24, NeMo log zero guard
const CMVN_EPS: f32 = 1e-5;
const NUM_MEL_BINS: usize = 80;
const NUM_SPECTRUM_BINS: usize = N_FFT / 2 + 1; // 257

/// Precomputed hann window (400 samples) zero-padded to 512, as a [N_FFT] frame mask.
static HANN_WINDOW: std::sync::LazyLock<[f32; N_FFT]> = std::sync::LazyLock::new(|| {
    let mut w = [0f32; N_FFT];
    // numpy.hanning(400) (symmetric, endpoints zero), centered in 512
    let pad = (N_FFT - WIN_LENGTH) / 2;
    let denom = (WIN_LENGTH - 1) as f32;
    for k in 0..WIN_LENGTH {
        w[pad + k] = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / denom).cos();
    }
    w
});

/// Number of valid feature frames for a waveform of `num_samples`.
pub fn num_frames(num_samples: usize) -> usize {
    num_samples / HOP_LENGTH
}

/// Compute NeMo 80-bin log-mel features with CMVN for a mono 16kHz waveform.
///
/// Returns a flat row-major `[frames * 80]` f32 buffer plus the valid frame count.
pub fn compute_nemo80_features(wave: &[f32]) -> (Vec<f32>, usize) {
    let frames_len = num_frames(wave.len());
    if frames_len == 0 {
        return (Vec::new(), 0);
    }
    // 1. Preemphasis on the raw waveform. numpy pads with zeros on the left
    //    (np.pad default constant), so preemph[0] = wave[0] - 0.97 * 0.
    let mut preemph: Vec<f32> = Vec::with_capacity(wave.len());
    for (n, &s) in wave.iter().enumerate() {
        let prev = if n == 0 { 0.0 } else { wave[n - 1] };
        preemph.push(s - PREEMPH * prev);
    }
    // 2. Zero-pad 256 both sides
    let pad = N_FFT / 2;
    let mut padded = vec![0f32; wave.len() + 2 * pad];
    padded[pad..pad + wave.len()].copy_from_slice(&preemph);

    // 3-6. Framed spectra -> log-mel energies per window
    let window = &*HANN_WINDOW;
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(N_FFT);
    let mut spectrum = r2c.make_output_vec();

    // windows = 1 + (padded_len - N_FFT) / HOP = 1 + wave.len()/HOP >= frames_len
    let num_windows = 1 + wave.len() / HOP_LENGTH;
    let mut logmel = vec![0f32; num_windows * NUM_MEL_BINS];
    let mut frame = vec![0f32; N_FFT];

    for w_i in 0..num_windows {
        let start = w_i * HOP_LENGTH;
        let end = start + N_FFT;
        if end > padded.len() {
            break;
        }
        for n in 0..N_FFT {
            frame[n] = padded[start + n] * window[n];
        }
        if r2c.process(&mut frame, &mut spectrum).is_err() {
            log::warn!("parakeet nemo80: fft failed on window {}", w_i);
            continue;
        }
        let power: Vec<f32> = spectrum
            .iter()
            .take(NUM_SPECTRUM_BINS)
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        let row = &mut logmel[w_i * NUM_MEL_BINS..w_i * NUM_MEL_BINS + NUM_MEL_BINS];
        let mut offset = 0usize;
        for b in 0..NUM_MEL_BINS {
            let start_row = NEMO80_MEL_STARTS[b] as usize;
            let len = NEMO80_MEL_LENS[b] as usize;
            let mut acc = 0f32;
            for j in 0..len {
                acc += power[start_row + j] * NEMO80_MEL_VALUES[offset + j];
            }
            row[b] = (acc + LOG_ZERO_GUARD).ln();
            offset += len;
        }
    }

    // 7. Per-utterance CMVN over the valid frames only
    let mut mean = [0f32; NUM_MEL_BINS];
    for f_i in 0..frames_len {
        for b in 0..NUM_MEL_BINS {
            mean[b] += logmel[f_i * NUM_MEL_BINS + b];
        }
    }
    for m in mean.iter_mut() {
        *m /= frames_len as f32;
    }
    let mut var = [0f32; NUM_MEL_BINS];
    for f_i in 0..frames_len {
        for b in 0..NUM_MEL_BINS {
            let d = logmel[f_i * NUM_MEL_BINS + b] - mean[b];
            var[b] += d * d;
        }
    }
    let denom_var = (frames_len as f32 - 1.0).max(1.0);
    for v in var.iter_mut() {
        *v /= denom_var;
    }

    let mut out = vec![0f32; frames_len * NUM_MEL_BINS];
    for f_i in 0..frames_len {
        for b in 0..NUM_MEL_BINS {
            let x = logmel[f_i * NUM_MEL_BINS + b];
            out[f_i * NUM_MEL_BINS + b] = (x - mean[b]) / (var[b].sqrt() + CMVN_EPS);
        }
    }

    (out, frames_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_frames_formula() {
        assert_eq!(num_frames(16000), 100);
        assert_eq!(num_frames(4000), 25);
        assert_eq!(num_frames(8000), 50);
        assert_eq!(num_frames(0), 0);
        assert_eq!(num_frames(159), 0);
        assert_eq!(num_frames(160), 1);
    }

    #[test]
    fn silence_features_match_reference() {
        // Silence: logmel is constant per bin, so CMVN reduces to float
        // rounding residue (numpy reference yields ~0.43 in some bins).
        // Assert we match the reference exactly instead of assuming zeros.
        let data = include_str!("../../nemo80_features_reference.json");
        let parsed: serde_json::Value = serde_json::from_str(data).unwrap();
        let expected: Vec<f32> = parsed["silence"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected_frames = parsed["silence"]["frames"].as_u64().unwrap() as usize;

        let wave = vec![0.0f32; 8000];
        let (feats, frames) = compute_nemo80_features(&wave);
        assert_eq!(frames, expected_frames);
        assert_eq!(feats.len(), expected.len());
        for (i, (a, b)) in feats.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "silence feature {} differs: got {}, want {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn matches_reference_features() {
        // Reference generated from onnx-asr's NemoPreprocessorNumpy with the
        // same deterministic LCG waves (nemo80_features_reference.json).
        let data = include_str!("../../nemo80_features_reference.json");
        let parsed: serde_json::Value = serde_json::from_str(data).unwrap();

        for name in ["noise1", "noise2"] {
            let wave: Vec<f32> = parsed[name]["wave"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let expected_frames = parsed[name]["frames"].as_u64().unwrap() as usize;
            let expected: Vec<f32> = parsed[name]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();

            let (feats, frames) = compute_nemo80_features(&wave);
            assert_eq!(frames, expected_frames, "{}: frame count", name);
            assert_eq!(feats.len(), expected.len(), "{}: buffer length", name);

            let mut max_diff = 0f32;
            for (i, (a, b)) in feats.iter().zip(expected.iter()).enumerate() {
                let diff = (a - b).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                assert!(
                    diff < 1e-3,
                    "{}: feature {} differs by {} (got {}, want {})",
                    name,
                    i,
                    diff,
                    a,
                    b
                );
            }
            // CMVN makes most features O(1); ensure the test is meaningful
            let spread = expected.iter().cloned().fold(f32::MIN, f32::max)
                - expected.iter().cloned().fold(f32::MAX, f32::min);
            assert!(spread > 1.0, "{}: reference features have no dynamic range", name);
        }
    }
}
