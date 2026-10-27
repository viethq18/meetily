// diarization_engine/engine.rs
//
// Post-recording speaker diarization orchestrator, a port of
// sherpa-onnx's OfflineSpeakerDiarization (Apache-2.0):
//
//   1. sliding-window pyannote segmentation (powerset -> multilabel)
//   2. exclude overlap frames; per (chunk, speaker) collect sample intervals
//      (>= 10 frames of speech in the chunk)
//   3. eres2net embedding per (chunk, speaker) span
//   4. cosine-dissimilarity complete-link agglomerative clustering
//   5. relabel frames by cluster, majority-vote finalize, emit segments
//
// Speaker labels are stable per run ("speaker_0", "speaker_1", ...).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;

use crate::diarization_engine::clustering;
use crate::diarization_engine::embedding::SpeakerEmbeddingModel;
use crate::diarization_engine::segmentation::{
    powerset_mapping, to_multi_label, PyannoteSegmentationModel, NUM_SPEAKERS, RECEPTIVE_FIELD_SIZE,
    RECEPTIVE_FIELD_SHIFT, WINDOW_SHIFT_RATIO, WINDOW_SIZE,
};

/// A final speaker segment (seconds from audio start).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: usize,
}

pub struct DiarizationResult {
    pub segments: Vec<SpeakerSegment>,
    pub num_speakers: usize,
}

pub struct DiarizationEngine {
    segmentation: PyannoteSegmentationModel,
    embedding: SpeakerEmbeddingModel,
    /// Complete-linkage merge threshold on cosine dissimilarity. 0.8
    /// empirically separates speakers with eres2net embeddings; sherpa-onnx's
    /// 0.5 default over-segments with this fbank front-end.
    pub clustering_threshold: f64,
}

const SAMPLE_RATE: usize = 16_000;

impl DiarizationEngine {
    pub fn load(models_dir: &Path) -> Result<Self> {
        let segmentation_path = models_dir
            .join("pyannote-segmentation-3.0")
            .join("model.int8.onnx");
        let embedding_path = models_dir
            .join("3dspeaker-eres2net")
            .join("eres2net.int8.onnx");
        let segmentation = PyannoteSegmentationModel::load(&segmentation_path)
            .with_context(|| format!("loading {}", segmentation_path.display()))?;
        let embedding = SpeakerEmbeddingModel::load(&embedding_path)
            .with_context(|| format!("loading {}", embedding_path.display()))?;
        Ok(Self {
            segmentation,
            embedding,
            clustering_threshold: 0.8,
        })
    }

    pub fn is_model_available(models_dir: &Path) -> bool {
        crate::diarization_engine::model::all_artifacts_present(models_dir)
    }

    /// Diarize a mono 16 kHz f32 waveform.
    pub fn process(&mut self, audio: &[f32], sample_rate: u32) -> Result<DiarizationResult> {
        if audio.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                num_speakers: 0,
            });
        }

        // Resample to 16 kHz when needed (simple linear interpolation; the
        // diarization models are robust to slight resampler differences).
        let audio: Vec<f32> = if sample_rate as usize != SAMPLE_RATE {
            linear_resample(audio, sample_rate as usize, SAMPLE_RATE)
        } else {
            audio.to_vec()
        };
        let n = audio.len();

        // 1. Segmentation: one [frames, NUM_SPEAKERS] 0/1 label matrix per chunk.
        let powerset_chunks = self.segmentation.run(&audio)?;
        if powerset_chunks.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                num_speakers: 0,
            });
        }
        let mapping = powerset_mapping();
        let labels: Vec<Array2<i32>> = powerset_chunks
            .iter()
            .map(|p| to_multi_label(p, &mapping))
            .collect();

        // Speakers per frame (fractional count across overlapping chunks,
        // rounded) — drives how many clusters to keep per frame at finalize.
        let speakers_per_frame = compute_speakers_per_frame(&labels);

        if speakers_per_frame.iter().all(|v| *v == 0) {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                num_speakers: 0,
            });
        }

        // 2. Overlap exclusion + per-(chunk, speaker) sample intervals.
        let (chunk_speaker_list, sample_intervals) = get_chunk_speaker_sample_indexes(&labels);

        // 3. Embeddings (skip NaN / too-short spans).
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        let mut valid_chunk_speaker: Vec<(usize, usize)> = Vec::new();
        let mut valid_intervals: Vec<Vec<(usize, usize)>> = Vec::new();
        for (k, intervals) in sample_intervals.iter().enumerate() {
            let mut span = Vec::new();
            for &(start, end) in intervals {
                let end = end.min(n);
                if end > start {
                    span.extend_from_slice(&audio[start..end]);
                }
            }
            let Some(mut embedding) = self.embedding.compute(&span)? else {
                log::debug!("diarization: skipping invalid embedding for chunk-speaker {k}");
                continue;
            };
            clustering::normalize(&mut embedding);
            embeddings.push(embedding);
            valid_chunk_speaker.push(chunk_speaker_list[k]);
            valid_intervals.push(sample_intervals[k].clone());
        }

        if embeddings.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                num_speakers: 0,
            });
        }

        // 4. Clustering.
        let cluster_labels = clustering::cluster(&embeddings, self.clustering_threshold);
        let max_cluster = *cluster_labels.iter().max().unwrap();

        // 5. Relabel + finalize.
        let chunk_speaker_to_cluster: HashMap<(usize, usize), i32> = valid_chunk_speaker
            .iter()
            .zip(cluster_labels.iter())
            .map(|(cs, l)| (*cs, *l))
            .collect();

        let new_labels = relabel(&labels, max_cluster, &chunk_speaker_to_cluster);
        let speaker_count = compute_speaker_count(&new_labels, n);
        let final_labels = finalize_labels(&speaker_count, &speakers_per_frame);

        let segments = compute_result(&final_labels);
        let num_speakers = final_labels.ncols();
        Ok(DiarizationResult {
            segments,
            num_speakers,
        })
    }
}

/// Linear resample by ratio (good enough for 44.1k/48k -> 16k diarization use).
fn linear_resample(input: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = input[idx.min(input.len() - 1)];
        let b = input[(idx + 1).min(input.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Fractional per-frame speaker count across chunks, rounded with +0.5.
fn compute_speakers_per_frame(labels: &[Array2<i32>]) -> Vec<i32> {
    let num_chunks = labels.len();
    let window_shift = (WINDOW_SIZE as f32 * WINDOW_SHIFT_RATIO) as usize;
    let num_frames = (WINDOW_SIZE + (num_chunks - 1) * window_shift) / RECEPTIVE_FIELD_SHIFT + 1;

    let mut count = vec![0f32; num_frames];
    let mut weight = vec![0f32; num_frames];
    for (i, label) in labels.iter().enumerate() {
        let start = (i as f32 * window_shift as f32 / RECEPTIVE_FIELD_SHIFT as f32 + 0.5) as usize;
        let rows = label.nrows();
        for k in 0..rows {
            let frame = start + k;
            if frame >= num_frames {
                break;
            }
            let sum: i32 = (0..label.ncols()).map(|c| label[[k, c]]).sum();
            count[frame] += sum as f32;
            weight[frame] += 1.0;
        }
    }
    count
        .iter()
        .zip(weight.iter())
        .map(|(c, w)| (c / (w + 1e-12) + 0.5) as i32)
        .collect()
}

/// Drop frames where more than one speaker is active (for embedding spans).
fn exclude_overlap(labels: &[Array2<i32>]) -> Vec<Array2<i32>> {
    labels
        .iter()
        .map(|label| {
            let mut out = Array2::<i32>::zeros((label.nrows(), label.ncols()));
            for r in 0..label.nrows() {
                let sum: i32 = (0..label.ncols()).map(|c| label[[r, c]]).sum();
                if sum < 2 {
                    for c in 0..label.ncols() {
                        out[[r, c]] = label[[r, c]];
                    }
                }
            }
            out
        })
        .collect()
}

/// Per-(chunk, speaker) sample intervals with >= 10 active frames.
pub(crate) fn get_chunk_speaker_sample_indexes(
    labels: &[Array2<i32>],
) -> (Vec<(usize, usize)>, Vec<Vec<(usize, usize)>>) {
    let excluded = exclude_overlap(labels);
    let window_shift = (WINDOW_SIZE as f32 * WINDOW_SHIFT_RATIO) as usize;

    let mut chunk_speaker_list = Vec::new();
    let mut sample_intervals = Vec::new();

    for (chunk_index, label) in excluded.iter().enumerate() {
        let num_frames = label.nrows();
        let sample_offset = chunk_index * window_shift;

        for speaker_index in 0..NUM_SPEAKERS {
            let frame_sum: i32 = (0..num_frames).map(|f| label[[f, speaker_index]]).sum();
            if frame_sum < 10 {
                continue; // skip segments shorter than 10 frames
            }

            let mut intervals: Vec<(usize, usize)> = Vec::new();
            let mut is_active = false;
            let mut start_index = 0usize;
            for k in 0..=num_frames {
                let active = k < num_frames && label[[k, speaker_index]] != 0;
                if active && !is_active {
                    is_active = true;
                    start_index = k;
                } else if !active && is_active {
                    is_active = false;
                    let end_k = k;
                    let start_samples =
                        (start_index as f32 / num_frames as f32 * WINDOW_SIZE as f32) as usize
                            + sample_offset;
                    let end_samples =
                        (end_k as f32 / num_frames as f32 * WINDOW_SIZE as f32) as usize
                            + sample_offset;
                    intervals.push((start_samples, end_samples));
                }
            }

            chunk_speaker_list.push((chunk_index, speaker_index));
            sample_intervals.push(intervals);
        }
    }
    (chunk_speaker_list, sample_intervals)
}

/// Replace per-chunk speaker columns with cluster columns.
fn relabel(
    labels: &[Array2<i32>],
    max_cluster: i32,
    chunk_speaker_to_cluster: &HashMap<(usize, usize), i32>,
) -> Vec<Array2<i32>> {
    let num_cols = max_cluster as usize + 1;
    labels
        .iter()
        .enumerate()
        .map(|(chunk_index, label)| {
            let mut new_label = Array2::<i32>::zeros((label.nrows(), num_cols));
            for speaker_index in 0..NUM_SPEAKERS {
                let Some(&cluster) = chunk_speaker_to_cluster.get(&(chunk_index, speaker_index))
                else {
                    continue;
                };
                if cluster < 0 || cluster as usize >= num_cols {
                    continue;
                }
                for f in 0..label.nrows() {
                    if label[[f, speaker_index]] == 1 {
                        new_label[[f, cluster as usize]] = 1;
                    }
                }
            }
            new_label
        })
        .collect()
}

/// Per-frame vote counts over cluster columns.
fn compute_speaker_count(labels: &[Array2<i32>], num_samples: usize) -> Array2<i32> {
    let window_shift = (WINDOW_SIZE as f32 * WINDOW_SHIFT_RATIO) as usize;
    let num_chunks = labels.len();
    let num_frames =
        (WINDOW_SIZE + (num_chunks - 1) * window_shift) / RECEPTIVE_FIELD_SHIFT + 1;
    let num_cols = labels[0].ncols();

    let mut count = Array2::<i32>::zeros((num_frames, num_cols));
    for (i, label) in labels.iter().enumerate() {
        let start = (i as f32 * window_shift as f32 / RECEPTIVE_FIELD_SHIFT as f32 + 0.5) as usize;
        let rows = label.nrows();
        for k in 0..rows {
            let frame = start + k;
            if frame >= num_frames {
                break;
            }
            for c in 0..num_cols {
                count[[frame, c]] += label[[k, c]];
            }
        }
    }

    let has_last_chunk = ((num_samples - WINDOW_SIZE) % window_shift) > 0;
    if !has_last_chunk {
        return count;
    }
    let last_frame = (num_samples / RECEPTIVE_FIELD_SHIFT).min(num_frames - 1);
    count.slice_axis(ndarray::Axis(0), ndarray::Slice::new(0, Some(last_frame as isize + 1), 1)).to_owned()
}

/// Keep the top-k voted clusters per frame (k = speakers_per_frame).
fn finalize_labels(count: &Array2<i32>, speakers_per_frame: &[i32]) -> Array2<i32> {
    let num_rows = count.nrows();
    let num_cols = count.ncols();
    let mut ans = Array2::<i32>::zeros((num_rows, num_cols));
    for i in 0..num_rows {
        let k = speakers_per_frame.get(i).copied().unwrap_or(0);
        if k == 0 {
            continue;
        }
        // top-k cluster columns by count (ties: lower column wins, matching
        // a stable argmax scan).
        let mut order: Vec<usize> = (0..num_cols).collect();
        order.sort_by(|a, b| count[[i, *b]].cmp(&count[[i, *a]]).then(a.cmp(b)));
        for &c in order.iter().take((k as usize).min(num_cols)) {
            if count[[i, c]] > 0 {
                ans[[i, c]] = 1;
            }
        }
    }
    ans
}

/// Frame runs -> time segments. Frame f covers samples
/// [f*shift, f*shift + receptive_field_size), i.e. seconds
/// f*shift/rate + 0.5*rf_size/rate .. (f+1)*shift/rate + 0.5*rf_size/rate.
fn compute_result(final_labels: &Array2<i32>) -> Vec<SpeakerSegment> {
    let num_frames = final_labels.nrows();
    let num_speakers = final_labels.ncols();
    let scale = RECEPTIVE_FIELD_SHIFT as f32 / SAMPLE_RATE as f32;
    let scale_offset = 0.5 * RECEPTIVE_FIELD_SIZE as f32 / SAMPLE_RATE as f32;

    let mut segments = Vec::new();
    for speaker in 0..num_speakers {
        let mut is_active = num_frames > 0 && final_labels[[0, speaker]] > 0;
        let mut start_index = if is_active { 0usize } else { 0 };
        for f in 1..num_frames {
            if is_active {
                if final_labels[[f, speaker]] == 0 {
                    segments.push(SpeakerSegment {
                        start: start_index as f32 * scale + scale_offset,
                        end: f as f32 * scale + scale_offset,
                        speaker,
                    });
                    is_active = false;
                }
            } else if final_labels[[f, speaker]] > 0 {
                is_active = true;
                start_index = f;
            }
        }
        if is_active {
            segments.push(SpeakerSegment {
                start: start_index as f32 * scale + scale_offset,
                end: num_frames as f32 * scale + scale_offset,
                speaker,
            });
        }
    }
    segments.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal PCM16 mono WAV reader for probe fixtures.
    fn load_probe_wav(path: &str) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        let data_pos = bytes.windows(4).position(|w| w == b"data")? + 8;
        let data = &bytes[data_pos..];
        Some(
            (0..data.len() / 2)
                .map(|i| {
                    i16::from_le_bytes([data[i * 2], data[i * 2 + 1]]) as f32 / 32768.0
                })
                .collect(),
        )
    }

    #[test]
    fn resample_changes_length() {
        let input = vec![0.0f32; 4800]; // 0.1 s @ 48 kHz
        let out = linear_resample(&input, 48_000, 16_000);
        assert_eq!(out.len(), 1600);
    }

    #[test]
    fn exclude_overlap_drops_multi_speaker_frames() {
        let mut label = Array2::<i32>::zeros((3, 3));
        label[[0, 0]] = 1;
        label[[1, 0]] = 1;
        label[[1, 1]] = 1; // overlap -> dropped
        label[[2, 2]] = 1;
        let out = &exclude_overlap(&[label])[0];
        assert_eq!(out[[0, 0]], 1);
        assert_eq!(out[[1, 0]], 0);
        assert_eq!(out[[1, 1]], 0);
        assert_eq!(out[[2, 2]], 1);
    }

    #[test]
    fn intervals_skip_short_spans() {
        // 20 frames of speaker 0 -> one interval; 5 frames of speaker 1 -> skipped.
        let mut label = Array2::<i32>::zeros((100, 3));
        for f in 10..30 {
            label[[f, 0]] = 1;
        }
        for f in 40..45 {
            label[[f, 1]] = 1;
        }
        let (cs, intervals) = get_chunk_speaker_sample_indexes(&[label]);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0], (0, 0));
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].len(), 1);
        let (s, e) = intervals[0][0];
        assert_eq!(s, 10 * WINDOW_SIZE / 100);
        assert_eq!(e, 30 * WINDOW_SIZE / 100);
    }

    #[test]
    fn finalize_keeps_top_k() {
        let mut count = Array2::<i32>::zeros((2, 2));
        count[[0, 0]] = 3;
        count[[0, 1]] = 1;
        count[[1, 0]] = 0;
        count[[1, 1]] = 2;
        let spf = vec![1, 1];
        let out = finalize_labels(&count, &spf);
        assert_eq!(out[[0, 0]], 1);
        assert_eq!(out[[0, 1]], 0);
        assert_eq!(out[[1, 1]], 1);
        assert_eq!(out[[1, 0]], 0);
    }

    #[test]
    fn compute_result_makes_time_segments() {
        let mut labels = Array2::<i32>::zeros((100, 2));
        for f in 10..50 {
            labels[[f, 0]] = 1;
        }
        let segments = compute_result(&labels);
        assert_eq!(segments.len(), 1);
        let seg = segments[0];
        assert_eq!(seg.speaker, 0);
        let scale = RECEPTIVE_FIELD_SHIFT as f32 / SAMPLE_RATE as f32;
        let offset = 0.5 * RECEPTIVE_FIELD_SIZE as f32 / SAMPLE_RATE as f32;
        assert!((seg.start - (10.0 * scale + offset)).abs() < 1e-6);
        assert!((seg.end - (50.0 * scale + offset)).abs() < 1e-6);
    }

    #[test]
    fn speakers_per_frame_counts_across_chunks() {
        // One chunk, 100 frames, speaker 0 active in all chunk frames.
        let mut label = Array2::<i32>::zeros((100, 3));
        for f in 0..100 {
            label[[f, 0]] = 1;
        }
        let spf = compute_speakers_per_frame(&[label]);
        // Covered frames count 1 speaker; frames beyond the chunk have no
        // coverage (weight 0) and stay 0, matching sherpa-onnx behavior.
        assert!(spf.iter().take(100).all(|v| *v == 1));
        assert!(spf.iter().skip(100).all(|v| *v == 0));
    }

    /// End-to-end ground-truth test over the real ONNX models: two synthetic
    /// "voices" (different formant stacks) alternating in 5-second turns must
    /// produce at least 2 speakers with voice-A turns mapping consistently.
    /// Ignored by default (needs the downloaded model set + CPU time).
    #[test]
    #[ignore = "requires diarization models in the app models dir"]
    fn two_alternating_voices_diarize_into_two_speakers() {
        let models_dir = match std::env::var_os("HOME") {
            Some(home) => std::path::PathBuf::from(home)
                .join("Library/Application Support/com.meetily.ai/models"),
            None => return,
        };
        if !crate::diarization_engine::model::all_artifacts_present(&models_dir) {
            eprintln!("diarization models not present; skipping");
            return;
        }

        let mut engine = DiarizationEngine::load(&models_dir).expect("engine load");

        // /tmp/two_speakers.wav: real speech (JFK sample) alternating with a
        // pitch-shifted copy — 6s turns with 0.5s gaps, 16 kHz mono PCM16.
        let Some(audio) = load_probe_wav("/tmp/two_speakers.wav") else {
            eprintln!("/tmp/two_speakers.wav not present; skipping");
            return;
        };

        let result = engine.process(&audio, 16_000).expect("diarization");
        assert!(
            result.num_speakers >= 2,
            "expected at least 2 speakers, got {} (segments: {:?})",
            result.num_speakers,
            result.segments
        );

        // Dominant speaker per turn window; adjacent turns must differ and
        // same-voice turns must agree.
        let dominant = |window: (f32, f32)| -> usize {
            let mut totals = vec![0f32; result.num_speakers];
            for seg in &result.segments {
                let overlap = seg.end.min(window.1) - seg.start.max(window.0);
                if overlap > 0.0 {
                    totals[seg.speaker] += overlap;
                }
            }
            totals
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        // Fixture layout: A 0-6s, B 6.5-12.5s, A 13-18.3s.
        let a = dominant((1.0, 4.0));
        let b = dominant((7.5, 10.5));
        let a2 = dominant((14.0, 17.0));
        assert_ne!(a, b, "voice A and voice B must be different speakers");
        assert_eq!(a, a2, "voice A turns must map to the same speaker");
    }
}
