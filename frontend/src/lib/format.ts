/** Formatting helpers shared across pages. Pure functions; no DOM access. */

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const pad = (n: number) => String(n).padStart(2, '0')

/** `22 Aug 2026 13:58 UTC`: one day-month-year format across the workspace, in UTC and saying so (L-7, design P2: screen tables
 *  showed local time without a zone beside a custody log and reports in UTC). */
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`
}

/** `1 Oct 2026` for an ISO date (a review record's date); the text as given when it is no date. */
export function fmtDay(iso: string): string {
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(iso) ? `${iso}T00:00:00Z` : iso)
  return Number.isNaN(d.getTime()) ? iso : `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`
}

/** A local date-time entry as the reader typed it: `2026-09-22 14:30` (also with a `T`), else null. The value is local time. */
export function parseLocalDateTime(text: string): Date | null {
  const m = /^\s*(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})\s*$/.exec(text)
  if (!m) return null
  const [y, mo, d, h, mi] = m.slice(1).map(Number)
  const date = new Date(y, mo - 1, d, h, mi)
  // a Date silently rolls 31 September into October: only a date that reads back the same is one
  return date.getFullYear() === y && date.getMonth() === mo - 1 && date.getDate() === d && date.getHours() === h && date.getMinutes() === mi ? date : null
}

export const fmtDateLong = fmtDate

/** `2.4 MB`, `812 KB`, `640 B`. */
export function fmtBytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

/** `3a9f0c…c21e` — first 6 and last 4 hex characters. */
export function shortHash(h: string | null | undefined, head = 6, tail = 4): string {
  if (!h) return '—'
  if (h.length <= head + tail + 1) return h
  return `${h.slice(0, head)}…${h.slice(-tail)}`
}

/** Seconds to `3.1 s` or `1 min 12 s`. */
export function fmtSeconds(s: number | null | undefined, d = 1): string {
  if (s == null || !Number.isFinite(s)) return '—'
  if (s < 60) return `${s.toFixed(d)} s`
  const m = Math.floor(s / 60)
  return `${m} min ${Math.round(s - m * 60)} s`
}

/** Elapsed `m:ss` from an ISO start. */
export function fmtElapsed(fromIso: string | null | undefined, now: number): string {
  if (!fromIso) return '0:00'
  const start = new Date(fromIso).getTime()
  if (Number.isNaN(start)) return '0:00'
  const s = Math.max(0, Math.floor((now - start) / 1000))
  return `${Math.floor(s / 60)}:${pad(s % 60)}`
}

/** The one rounding rule of a number that is printed: the nearest whole number, a half upwards (N-10b; code re-verification
 *  R2-06). The server uses the same expression (`app/core/rounding.py` `half_up`), where its own `round` went to the even
 *  neighbour: "score 96 / 100" in the report beside "Score 97 / 100" here. Both suites read `rounding_vectors.json`. */
export const halfUp = (v: number): number => Math.floor(v + 0.5)

/** Fixed-decimal probability: `0.97`, or em dash when null. */
export const prob = (x: number | null | undefined, d = 2) => (x == null || !Number.isFinite(x) ? '—' : x.toFixed(d))

/** Word count for the paste box. */
export const wordCount = (t: string) => (t.trim() ? t.trim().split(/\s+/).length : 0)

/** Human-readable kind word for list rows. */
export const kindWord = (k: string) => (k === 'document' ? 'doc' : k)

/** Two-decimal fusion weight for the ledger's WEIGHT column (`1.00`, `0.35`). */
export const fmtWeight = (w: number | null | undefined) => (w == null || !Number.isFinite(w) ? '—' : w.toFixed(2))

/** Reader-facing names for detector/score keys emitted by the engine. Unknown keys fall back to a tidied form. */
const DETECTOR_NAMES: Record<string, string> = {
  univfd: 'UnivFD (CLIP ViT-L/14)',
  dfdc_effnet_b7: 'DFDC EfficientNet-B7',
  deepfake_vs_real_image_detection: 'Deepfake vs real (ViT)',
  ViT_Deepfake_Detection: 'ViT deepfake',
  'Deep-Fake-Detector-v2-Model': 'Deep-Fake Detector v2',
  'ai-vs-human-image-detector': 'AI vs human (Ateeqq)',
  'AI-image-detector': 'AI image detector (Swin)',
  'sdxl-detector': 'SDXL detector',
  'siglip-ai-detector': 'SigLIP AI detector',
  classifier: 'Classifier',
  classifier_secondary: 'Classifier (secondary)',
  classifier_document: 'Classifier (document)',
  binoculars: 'Binoculars',
  binoculars_document: 'Binoculars (document)',
  binoculars_p: 'Binoculars p(AI)',
  burstiness: 'Burstiness',
  perplexity: 'Perplexity',
  stylometry: 'Stylometry',
  fused: 'Fused',
  p_frame: 'Fused (frame)',
  p_face: 'Face-swap',
  p_general: 'Whole-frame',
}
export function detectorName(key: string): string {
  if (DETECTOR_NAMES[key]) return DETECTOR_NAMES[key]
  const short = key.split('/').pop() ?? key
  if (DETECTOR_NAMES[short]) return DETECTOR_NAMES[short]
  return short.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim()
}

/** C2PA presence as a word: booleans and `{present}` objects become `present` / `absent`; strings pass through. */
export function c2paWord(v: unknown): string {
  if (typeof v === 'boolean') return v ? 'present' : 'absent'
  if (v && typeof v === 'object') {
    const o = v as { present?: unknown; validation_state?: unknown; claim_generator?: unknown; status?: unknown }
    if (typeof o.status === 'string') return o.status
    if (typeof o.present === 'boolean') {
      if (!o.present) return 'absent'
      const extra = [typeof o.validation_state === 'string' && o.validation_state !== 'unknown' ? o.validation_state : null, typeof o.claim_generator === 'string' && o.claim_generator ? o.claim_generator : null].filter(Boolean)
      return extra.length ? `present · ${extra.join(' · ')}` : 'present'
    }
  }
  if (v == null || v === '') return 'absent'
  return String(v)
}

/** EXIF `YYYY:MM:DD HH:MM:SS` (or ISO) to `22 Aug 2026 13:58`; unparseable input is returned as given. */
export function fmtExifDate(v: unknown): string {
  if (typeof v !== 'string' || !v.trim()) return '—'
  const m = v.match(/^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?/)
  const month = m ? MONTHS[Number(m[2]) - 1] : undefined
  // EXIF records the camera's wall clock without a zone: printed as written, and said to be so (L-7)
  return m && month ? `${Number(m[3])} ${month} ${m[1]} ${m[4]}:${m[5]} (zone not recorded)` : v
}
