/** Saved speech-reference contracts, mirrored by detectors/audio/reference.py. */
export const CURRENT_SPEECH_REFERENCE = '912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf'
export const COMBINED_SPEECH_REFERENCE = 'e7de202c42b80fc81d8ab7faa0ce3538c6af5a4c28a9830e4b9dfbcaa6b63641'
export const COMBINED_SPEECH_NOTE = 'A screening estimate under a balanced reference prior, not a measured accuracy rate for this recording. Validation used four-second English speech with controlled backgrounds. Strong noise, overlapping voices and unfamiliar generators can cause errors. This does not assess the authenticity of background sounds, establish a per-file confidence interval or provide a court conclusion.'
export type SpeechTendency = 'ai' | 'original' | 'inconclusive'

const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value)
const integer = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value)

export const knownSpeechReference = (value: unknown) => value === CURRENT_SPEECH_REFERENCE || value === COMBINED_SPEECH_REFERENCE

export function combinedSpeechTendency(value: unknown): [number | null, SpeechTendency] {
  if (!record(value)) return [null, 'inconclusive']
  const windows = value.checked_windows, ranges = value.checked_source_ranges
  const window = Array.isArray(windows) && windows.length === 1 && record(windows[0]) ? windows[0] : null
  const range = Array.isArray(ranges) && ranges.length === 1 && record(ranges[0]) ? ranges[0] : null
  const windowBound = window !== null && Object.keys(window).sort().join(',') === 'end_s,end_sample,start_s,start_sample'
    && integer(window.start_sample) && window.start_sample === 0 && integer(window.end_sample) && window.end_sample === 64000
    && window.start_s === 0 && window.end_s === 4
  const sourceBound = range !== null && integer(range.start_sample) && range.start_sample === 0
    && integer(range.end_sample) && range.end_sample === 64000 && range.time_base === '1/16000'
    && integer(range.source_start_pts) && integer(range.source_end_pts) && range.source_end_pts - range.source_start_pts === 64000
  const p = value.probability_ai
  if (value.calibration_sha256 !== COMBINED_SPEECH_REFERENCE || value.status !== 'ok' || value.scope_supported !== true
    || !windowBound || !sourceBound || typeof p !== 'number' || !Number.isFinite(p) || p < 0 || p > 1) return [null, 'inconclusive']
  return [p, value.decision === 'ai' && p > .5 ? 'ai' : value.decision === 'original' && p < .5 ? 'original' : 'inconclusive']
}
