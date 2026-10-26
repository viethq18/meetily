// audio/transcription/openai_compatible.rs
//
// TranscriptionProvider backed by an OpenAI-compatible HTTP endpoint
// (`{base}/audio/transcriptions`). Samples are encoded to a 16kHz mono WAV
// in memory and uploaded as multipart form data. Works for both realtime
// streaming segments and retranscribe/import flows.

use super::provider::{TranscriptResult, TranscriptionError, TranscriptionProvider};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Minimum segment length we are willing to upload (~0.5s at 16kHz). Shorter
/// chunks are rejected with `AudioTooShort` so the worker drops them.
const MIN_SAMPLES: usize = 8000;

/// Stored configuration for the OpenAI-compatible transcription endpoint.
/// Mirrors the summary-side `CustomOpenAIConfig` JSON-column pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptOpenAIConfig {
    pub endpoint: String,
    #[serde(default)]
    pub api_key: Option<String>,
    pub model: String,
}

/// Encode f32 mono samples (assumed 16kHz) as a 16-bit PCM WAV file.
fn encode_wav(samples: &[f32]) -> Result<Vec<u8>, TranscriptionError> {
    let data_len = samples.len() * 2;
    let mut wav = Vec::with_capacity(44 + data_len);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        wav.extend_from_slice(&((clamped * 32767.0) as i16).to_le_bytes());
    }
    Ok(wav)
}

/// Build a multipart/form-data body with the WAV file and optional fields.
fn build_multipart(
    boundary: &str,
    wav: Vec<u8>,
    model: &str,
    language: Option<&str>,
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(&wav);
    body.extend_from_slice(format!("\r\n--{}\r\n", boundary).as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"model\"\r\n\r\n{}\r\n", model).as_bytes(),
    );
    if let Some(lang) = language {
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"language\"\r\n\r\n{}\r\n",
                lang
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());
    body
}

#[derive(Deserialize)]
struct TranscriptionResponse {
    #[serde(default)]
    text: String,
}

/// OpenAI-compatible transcription provider.
pub struct OpenAICompatibleProvider {
    client: reqwest::Client,
    config: std::sync::RwLock<Option<TranscriptOpenAIConfig>>,
}

impl OpenAICompatibleProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build reqwest client"),
            config: std::sync::RwLock::new(None),
        }
    }

    /// Load the stored configuration from the database.
    pub async fn load_config(&self, pool: &sqlx::SqlitePool) -> Result<(), String> {
        let config =
            crate::database::repositories::setting::SettingsRepository::get_transcript_openai_config(pool)
                .await
                .map_err(|e| e.to_string())?;
        *self.config.write().unwrap() = config;
        Ok(())
    }

    /// Update the in-memory configuration (used after save commands).
    pub fn set_config(&self, config: Option<TranscriptOpenAIConfig>) {
        *self.config.write().unwrap() = config;
    }

    fn current_config(&self) -> Result<TranscriptOpenAIConfig, TranscriptionError> {
        self.config
            .read()
            .unwrap()
            .clone()
            .ok_or(TranscriptionError::ModelNotLoaded)
    }
}

impl Default for OpenAICompatibleProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TranscriptionProvider for OpenAICompatibleProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> Result<TranscriptResult, TranscriptionError> {
        if audio.len() < MIN_SAMPLES {
            return Err(TranscriptionError::AudioTooShort {
                samples: audio.len(),
                minimum: MIN_SAMPLES,
            });
        }

        let config = self.current_config()?;
        let wav = encode_wav(&audio)?;
        let boundary = "meetily-transcribe-boundary-7f3a9c2d";
        let body = build_multipart(boundary, wav, &config.model, language.as_deref());

        let url = format!(
            "{}/audio/transcriptions",
            config.endpoint.trim_end_matches('/')
        );
        let mut request = self
            .client
            .post(&url)
            .header(
                reqwest::header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={}", boundary),
            )
            .body(body);
        if let Some(api_key) = config.api_key.as_deref().filter(|k| !k.is_empty()) {
            request = request.bearer_auth(api_key);
        }

        let response = request
            .send()
            .await
            .map_err(|e| TranscriptionError::EngineFailed(format!("request failed: {}", e)))?;

        let status = response.status();
        let payload = response
            .text()
            .await
            .map_err(|e| TranscriptionError::EngineFailed(format!("failed to read body: {}", e)))?;

        if !status.is_success() {
            return Err(TranscriptionError::EngineFailed(format!(
                "endpoint returned {}: {}",
                status,
                payload.chars().take(500).collect::<String>()
            )));
        }

        let parsed: TranscriptionResponse = serde_json::from_str(&payload)
            .map_err(|e| {
                TranscriptionError::EngineFailed(format!(
                    "failed to parse response: {}",
                    e
                ))
            })?;

        Ok(TranscriptResult {
            text: parsed.text.trim().to_string(),
            confidence: None,
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        self.config.read().unwrap().is_some()
    }

    async fn get_current_model(&self) -> Option<String> {
        self.config.read().unwrap().as_ref().map(|c| c.model.clone())
    }

    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_is_wellformed() {
        let wav = encode_wav(&[0.0; 1600]).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(wav.len(), 44 + 3200);
    }

    #[test]
    fn wav_encodes_clamped_samples() {
        let wav = encode_wav(&[0.5, -0.5, 2.0, -2.0]).unwrap();
        let data = &wav[44..];
        let a = i16::from_le_bytes([data[0], data[1]]);
        let b = i16::from_le_bytes([data[2], data[3]]);
        let c = i16::from_le_bytes([data[4], data[5]]);
        let d = i16::from_le_bytes([data[6], data[7]]);
        assert_eq!(a, (0.5 * 32767.0) as i16);
        assert_eq!(b, (-0.5 * 32767.0) as i16);
        assert_eq!(c, i16::MAX); // clamped
        assert_eq!(d, (-1.0 * 32767.0) as i16); // clamped
    }

    #[test]
    fn multipart_contains_model_and_language() {
        let body = build_multipart("B", vec![1, 2, 3], "whisper-1", Some("vi"));
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("name=\"model\"\r\n\r\nwhisper-1"));
        assert!(text.contains("name=\"language\"\r\n\r\nvi"));
        assert!(text.contains("name=\"file\"; filename=\"audio.wav\""));
        assert!(text.ends_with("--B--\r\n"));
    }

    #[test]
    fn short_audio_is_rejected() {
        let provider = OpenAICompatibleProvider::new();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(provider.transcribe(vec![0.0; 100], Some("vi".into())));
        match result {
            Err(TranscriptionError::AudioTooShort { samples, minimum }) => {
                assert_eq!(samples, 100);
                assert_eq!(minimum, MIN_SAMPLES);
            }
            other => panic!("expected AudioTooShort, got {:?}", other.map(|r| r.text)),
        }
    }

    #[test]
    fn not_configured_reports_model_not_loaded() {
        let provider = OpenAICompatibleProvider::new();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(provider.transcribe(vec![0.0; 16000], None));
        assert!(matches!(result, Err(TranscriptionError::ModelNotLoaded)));
    }
}
