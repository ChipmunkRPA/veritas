import type { Kind } from '../lib/api'
import type { AnalysisProfile } from '../lib/api/intake'

const MB = 1024 * 1024

/** Analysis profiles (ARCHITECTURE §2.7; sampling in `engine/profiles.py`). '' leaves the choice to the instance default. */
export const PROFILES: { id: AnalysisProfile | ''; label: string; detail: string }[] = [
  { id: '', label: 'Instance default', detail: 'Uses this instance’s configured choice. Select a profile to see which checks are available.' },
  { id: 'quick', label: 'Quick', detail: 'An initial review with reduced sampling and the checks enabled for quick. Portions of a long recording may not be scored.' },
  { id: 'full', label: 'Full', detail: 'Broader coverage with the checks enabled for full, more video samples and available face, voice and lip-sync analysis. Takes longer than quick.' },
  { id: 'deep', label: 'Deep', detail: 'Additional checks and a larger analysis budget where supported. The slowest profile; more checks do not guarantee a conclusive result.' },
]

interface TabSpec {
  kind: Kind; label: string; noun: string; accept: string; exts: string[]; mimePrefix?: string; maxBytes: number; formats: string
}

export const TABS: TabSpec[] = [
  {
    kind: 'image', label: 'Image', noun: 'an image', accept: 'image/*,.apng,.heic,.heif', exts: ['jpg', 'jpeg', 'png', 'apng', 'webp', 'heic', 'heif', 'bmp', 'tif', 'tiff', 'gif'], mimePrefix: 'image/', maxBytes: 60 * MB,
    formats: 'JPEG · PNG · APNG · WebP · HEIC · up to 60 MB',
  },
  {
    kind: 'video', label: 'Video', noun: 'a video', accept: 'video/*,.mkv', exts: ['mp4', 'mov', 'webm', 'mkv', 'm4v', 'avi'], mimePrefix: 'video/', maxBytes: 500 * MB,
    formats: 'MP4 · MOV · WebM · MKV · up to 500 MB',
  },
  {
    kind: 'audio', label: 'Audio', noun: 'a recording', accept: 'audio/*,.m4a,.opus,.flac', exts: ['wav', 'mp3', 'm4a', 'ogg', 'opus', 'flac'], mimePrefix: 'audio/', maxBytes: 150 * MB,
    formats: 'WAV · MP3 · M4A · OGG · OPUS · FLAC · up to 150 MB',
  },
  {
    kind: 'document', label: 'Document', noun: 'a document', accept: '.pdf,.docx,.txt,.md,.html', exts: ['pdf', 'docx', 'txt', 'md', 'html', 'htm'], maxBytes: 25 * MB,
    formats: 'PDF · DOCX · TXT · MD · HTML · up to 25 MB',
  },
]

export function validate(f: File, t: TabSpec): string | null {
  const ext = f.name.split('.').pop()?.toLowerCase() ?? ''
  const mimeOk = t.mimePrefix ? f.type.startsWith(t.mimePrefix) : false
  if (!mimeOk && !t.exts.includes(ext)) return `Unsupported type. Accepted: ${t.formats.split(' · ').slice(0, -1).join(', ')}`
  if (f.size > t.maxBytes) return `File exceeds ${Math.round(t.maxBytes / MB)} MB`
  if (f.size === 0) return 'File is empty'
  return null
}

export const intakeKind = (value: string | null): Kind => TABS.find((tab) => tab.kind === value)?.kind ?? 'image'

/** Why "Begin analysis" is switched off, in one sentence (N-19a; design review, P2 "screens" 10: the button was disabled
 *  without a reason before a file was chosen). Null when the form can be sent. */
export function beginBlocked(s: { kind: Kind; hasFile: boolean; words: number; minWords: number; batch: number; maxBatch: number; problem: string | null }): string | null {
  if (s.batch > 1) return s.batch > s.maxBatch ? `A batch holds at most ${s.maxBatch} files: remove ${s.batch - s.maxBatch} to submit it.` : null
  if (!s.hasFile) {
    if (s.kind !== 'document') return `Choose ${TABS.find((t) => t.kind === s.kind)!.noun} to begin.`
    if (s.words === 0) return `Choose a document or paste at least ${s.minWords} words to begin.`
    if (s.words < s.minWords) return `The pasted text has ${s.words} ${s.words === 1 ? 'word' : 'words'}: ${s.minWords} are needed to begin.`
  }
  return s.problem ? 'Correct the case and custody fields to begin.' : null
}
