# Vietnamese Transcription Providers — Design

Date: 2026-10-26
Status: Approved (in implementation)

## Goal

Add three new transcription providers to the Meetily Tauri app, focused on Vietnamese:

1. **Gipformer 1.5 (68M RNNT)** — local ONNX, Vietnamese-only, best WER on Vietnamese benchmarks (call-center, tech, finance, education domains). Model: `g-group-ai-lab/gipformer1.5-68M-rnnt` (sherpa-onnx format, int8, ~73MB).
2. **Parakeet CTC 0.6B Vietnamese** — local ONNX via existing parakeet engine, new model spec + CTC decode path. Model: `OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx` (653MB int8).
3. **OpenAI Compatible** — custom HTTP endpoint (`{base}/audio/transcriptions`, multipart WAV), realtime streaming AND retranscribe/import paths.

Out of scope: Cohere Transcribe (API cloud rejected by user; local infeasible — 4.1GB fp32 safetensors, custom Conformer code, no ONNX export). Self-hosted Cohere can be used via the OpenAI Compatible provider.

## Architecture

### Gipformer engine (`frontend/src-tauri/src/gipformer_engine/`)

Mirrors `parakeet_engine/` structure:

- `features.rs` — kaldi-style fbank (80 bins, 16kHz, 25ms frame / 10ms shift), pure Rust, log-mel + mean-variance normalization as used by sherpa-onnx zipformer.
- `model.rs` — three `ort` sessions (encoder / decoder / joiner int8), greedy RNNT decode loop (encoder once per chunk; decoder+joiner per emitted token), BPE token decoding from `tokens.txt` (2000 tokens) + `bpe.model` (sentencepiece) if needed for pieces.
- `gipformer_engine.rs` — model catalog (single model), model dir management, resumable HF download (reuse pattern from parakeet engine), `load_model`, `transcribe_audio`, `is_model_loaded`, model status reporting.
- `commands.rs` — Tauri commands: `gipformer_init`, `gipformer_get_available_models`, `gipformer_download_model` (+ retry/cancel), `gipformer_validate_model_ready_with_config`, progress events `gipformer-model-download-progress/complete/error`.

Language: fixed `vi`; language hints ignored.

### Parakeet CTC Vietnamese (extend `parakeet_engine/`)

- New `ArtifactSpec` for `parakeet-ctc-0.6b-vietnamese` (`model.int8.onnx` + `vocab.txt` from OpenVoiceOS; reuses `nemo128.onnx` preprocessor).
- New decode path in `model.rs`: CTC greedy (argmax → collapse repeats → remove blank), selected by model spec (decoder type enum: TDT vs CTC). Existing TDT path untouched.

### OpenAI Compatible provider (`frontend/src-tauri/src/audio/transcription/openai_compatible.rs`)

- `OpenAICompatibleProvider` implements `TranscriptionProvider`:
  - `transcribe(samples, language)`: encode 16kHz mono f32 → WAV bytes → multipart POST `{endpoint}/audio/transcriptions` with `model`, `file`, optional `language`; parse `{"text": ...}`.
  - Segments below minimum length return `TranscriptionError::AudioTooShort` (worker drops them as with Whisper).
- Config storage: JSON record in settings table (`transcriptOpenAICompatibleConfig`: `endpoint`, `api_key`, `model`), mirroring `CustomOpenAIConfig` pattern. Commands: `api_save_transcript_openai_config`, `api_get_transcript_openai_config`, `api_test_transcript_openai_connection`.
- Works in both paths: realtime streaming (worker's `TranscriptionEngine::Provider` arm) and retranscribe/import (whole-file upload).

### Unified dispatch

- `audio/transcription/engine.rs`: `validate_transcription_model_ready` and `get_or_init_transcription_engine` gain branches for `gipformer` and `openai-compatible` providers.
- `audio/retranscription.rs`: replaces hardcoded `use_parakeet` bool with provider-string dispatch returning `TranscriptionEngine`; whisper/parakeet behavior unchanged; new providers handled (`gipformer`, `parakeet` (existing models + new Vietnamese CTC), `openai-compatible`).

Provider strings (stored in `transcript_settings`): `localWhisper`, `parakeet`, `gipformer`, `openai-compatible`.

### Frontend

- `TranscriptSettings.tsx`: new select items — Gipformer (Vietnamese, recommended for vi), Parakeet CTC Vietnamese, OpenAI Compatible; OpenAI Compatible gets endpoint/apiKey/model form (saved via new commands).
- New `GipformerModelManager` component (mirrors `ParakeetModelManager`: list, download with progress events, auto-save config).
- Parakeet model manager lists the new Vietnamese model (from engine catalog).
- `useTranscriptionModels.ts` + Import/Retranscribe dialogs: include gipformer + parakeet-vi models.
- Language handling: gipformer and parakeet-ctc-vietnamese lock language to `vi` (same UI pattern as existing parakeet auto-detect lock).

## Error handling

- Model download failures: resumable, retry + cancel (existing pattern).
- HTTP provider: non-2xx → `EngineFailed` with status + body; connection errors → `EngineFailed`.
- No model loaded → `ModelNotLoaded` (worker skips chunk, logs).

## Testing

- Unit: fbank feature extraction against sherpa-onnx reference values; CTC greedy decode; RNNT decode-loop logic with fake logits; config round-trip (JSON record).
- HTTP provider: mock server tests (pattern exists in parakeet engine tests).
- Smoke: `cargo check`/`cargo test`, then app run — download gipformer, transcribe a Vietnamese WAV via each new provider on both paths.
