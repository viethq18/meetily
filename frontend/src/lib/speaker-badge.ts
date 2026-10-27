// Speaker badge presentation shared by TranscriptView and
// VirtualizedTranscriptView.
//
// Speaker values (transcripts.speaker):
//   "local"     — microphone stream (local/remote mode)
//   "remote"    — system-audio stream (local/remote mode)
//   "speaker_N" — diarization cluster N (multi-speaker mode)

export function speakerBadgeLabel(speaker: string): string {
  if (speaker === 'local') return 'You';
  if (speaker === 'remote') return 'Remote';
  const match = /^speaker_(\d+)$/.exec(speaker);
  if (match) {
    const index = Number.parseInt(match[1], 10);
    return Number.isNaN(index) ? speaker : `Speaker ${index + 1}`;
  }
  return speaker.charAt(0).toUpperCase() + speaker.slice(1);
}

// Distinct Tailwind chip colors cycling per speaker index.
const SPEAKER_COLORS = [
  'bg-purple-100 text-purple-700',
  'bg-emerald-100 text-emerald-700',
  'bg-amber-100 text-amber-700',
  'bg-pink-100 text-pink-700',
  'bg-cyan-100 text-cyan-700',
] as const;

export function speakerBadgeClass(speaker: string): string {
  if (speaker === 'local') return 'bg-blue-100 text-blue-700';
  if (speaker === 'remote') return 'bg-gray-200 text-gray-700';
  const match = /^speaker_(\d+)$/.exec(speaker);
  if (match) {
    const index = Number.parseInt(match[1], 10);
    if (!Number.isNaN(index) && index >= 0) {
      return SPEAKER_COLORS[index % SPEAKER_COLORS.length];
    }
  }
  return 'bg-gray-200 text-gray-700';
}
