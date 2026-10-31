/** The values of the analysis record in words (BUILD_SPEC N-19d; design review DS-19). The result keeps the recorded value
 *  ("mps", "commercial+allowlist", "tpl-2.11", "pts-plan-3", "legacy-fuser"); the page prints the words. Mirror of the
 *  functions of the report formatting (`device_words` …); both read the `record` vectors of
 *  `backend/tests/fixtures/slice_words_vectors.json`. */

const DEVICE: Record<string, string> = { mps: 'Apple GPU', cuda: 'NVIDIA GPU', cpu: 'CPU' }
const PRECISION: Record<string, string> = { fp16: '16-bit floats', float16: '16-bit floats', fp32: '32-bit floats', float32: '32-bit floats' }
const POLICY: Record<string, string> = {
  commercial: 'commercial licences only', research: 'research use, every licence class', custom: 'a custom licence policy',
  allowlist: 'an allow list', denylist: 'a deny list',
}
const VERSIONED = /^(?:tpl|pts-plan)-(\d+(?:\.\d+)*)$/
const FUSION: Record<string, string> = { fallback: 'no fusion model (hand-set weights, not calibrated)', artefact: 'calibrated fusion model' }
export const NO_FUSION_MODEL = 'hand-set weights (no fusion model)'
const METHOD: Record<string, string> = { 'legacy-heuristic': 'not calibrated', platt: 'Platt scaling', isotonic: 'isotonic regression' }
export const CONTEXT: Record<string, string> = { 'video T2V': 'text-to-video', 'portrait face': 'portrait', 'video talking head': 'talking-head video', 'audio lossy': 'lossy audio' }
const SAMPLING: Record<string, string> = {
  voice_window_s: 'voice window (s)', max_windows: 'windows at most', max_seconds: 'seconds analysed at most', sample_rate: 'sample rate (Hz)',
  plan_version: 'sampling plan', legacy_frames: 'frames sampled', tracked_frames: 'frames tracked', tracked_fps: 'tracking rate (frames per second)',
  frame_indices_sha256: 'frame indices SHA-256',
}

type Value = string | null | undefined

/** `mps`, `fp16` → "Apple GPU · 16-bit floats"; `cuda:1` → "NVIDIA GPU 1"; a value outside the list keeps its words. */
export function deviceWords(device: Value, precision?: Value): string | null {
  const parts: string[] = []
  if (device) {
    const colon = device.indexOf(':')
    const [head, index] = colon < 0 ? [device, ''] : [device.slice(0, colon), device.slice(colon + 1)]
    parts.push(head in DEVICE ? `${DEVICE[head]} ${index}`.trim() : device)
  }
  if (precision) parts.push(PRECISION[precision] ?? precision)
  return parts.join(' · ') || null
}

/** `commercial+allowlist` → "commercial licences only, with an allow list" (as a model card says it). */
export function policyWords(label: Value): string | null {
  if (!label) return null
  const [head, ...rest] = label.split('+').map((part) => POLICY[part] ?? part.replace(/[_-]/g, ' ').trim())
  return head + (rest.length ? `, with ${rest.join(' and ')}` : '')
}

/** The sentence templates of an analysis (`tpl-2.11`) and the sampling plan of a video (`pts-plan-3`) by their version. */
export function versionWords(value: Value): string | null {
  if (!value) return null
  const match = VERSIONED.exec(value)
  return match ? `version ${match[1]}` : value
}

export const fusionWords = (mode: Value): string | null => (mode ? FUSION[mode] ?? mode : null)

/** The subject of a calibration record; for the record of a result without a fusion model, what it is. */
export function calibrationSubject(subject: Value): string {
  const text = subject ?? ''
  return text === 'legacy-fuser' || text.startsWith('v1-fuse') ? NO_FUSION_MODEL : text || 'not recorded'
}

export const calibrationMethod = (method: Value): string => (method ? METHOD[method] ?? method : 'not recorded')

/** An error-rate context as the page says it: the key "video T2V" reads "text-to-video", "portrait face" reads "portrait". */
export const contextWords = (context: string): string => CONTEXT[context] ?? context

/** The sampling record of a manifest as the reports print it: "sampling plan version 3 · frames sampled 16". */
export function samplingWords(sampling: Record<string, unknown> | null | undefined): string | null {
  const rows = Object.entries(sampling ?? {}).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
  return rows.map(([k, v]) => `${SAMPLING[k] ?? k.replace(/_/g, ' ')} ${k === 'plan_version' ? versionWords(String(v)) : typeof v === 'object' ? JSON.stringify(v) : String(v)}`).join(' · ') || null
}
