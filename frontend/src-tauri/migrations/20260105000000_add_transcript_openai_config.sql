-- Migration: Add OpenAI-compatible transcription endpoint configuration.
-- Stores JSON: {endpoint, apiKey, model} for the `openai-compatible`
-- transcription provider (POST {endpoint}/audio/transcriptions).

ALTER TABLE transcript_settings ADD COLUMN transcriptOpenAIConfig TEXT;
