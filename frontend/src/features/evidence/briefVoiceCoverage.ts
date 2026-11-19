import type { AudioEvidenceSet, BriefVoiceReview } from './multimediaPackets'

const rate = 16000
const scope = 192000
const width = 12000
const step = 8000
const count = 23
export type BriefCoverageStatus = 'checked' | 'insufficient_speech' | 'outside_schedule'
export interface BriefCoverageInterval { start_sample: number; end_sample: number; status: BriefCoverageStatus }
export interface BriefVoiceCoverage {
  checked_samples: number
  intervals: BriefCoverageInterval[]
  windows: { start_sample: number; end_sample: number; speech_samples: number; checked: boolean }[]
}
export const briefCoverageLabels: Record<BriefCoverageStatus, string> = {
  checked: 'At least one checked window',
  insufficient_speech: 'Not checked · limited or uncertain speech',
  outside_schedule: 'Outside this check’s schedule',
}

/** Project the signed admissions without changing their outcome or inferring missing checks. */
export function briefVoiceCoverage(review: BriefVoiceReview, source: AudioEvidenceSet['source']): BriefVoiceCoverage | undefined {
  if (!['chipmunk.brief-voice-review/1', 'chipmunk.brief-voice-review/2'].includes(review.schema)
    || review.status === 'skipped' || source.samples !== scope || source.sample_rate !== rate
    || source.timeline_time_base !== '1/16000' || !Array.isArray(source.timeline) || source.timeline.length !== 1 || !source.timeline[0]
    || source.timeline[0].start_sample !== 0 || source.timeline[0].end_sample !== scope || source.timeline[0].source_pts !== 0
    || review.scheduled_windows !== count || !Array.isArray(review.window_checks) || review.window_checks.length !== count) return undefined
  const windows = []
  for (let index = 0; index < count; index++) {
    const row = review.window_checks[index]
    if (!row || typeof row !== 'object' || row.start_sample !== index * step
      || !Number.isInteger(row.speech_samples) || row.speech_samples < 0 || row.speech_samples > width
      || !(row.elevated === null || typeof row.elevated === 'boolean')
      || (row.elevated !== null) !== (row.speech_samples * 2 >= width)) return undefined
    windows.push({ start_sample: row.start_sample, end_sample: row.start_sample + width,
      speech_samples: row.speech_samples, checked: row.elevated !== null })
  }
  if (windows.filter(row => row.checked).length !== review.speech_windows_checked) return undefined
  const boundaries = [...new Set([0, scope, ...windows.flatMap(row => [row.start_sample, row.end_sample])])].sort((a, b) => a - b)
  const intervals: BriefCoverageInterval[] = []
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i], end = boundaries[i + 1]
    const covering = windows.filter(row => row.start_sample <= start && end <= row.end_sample)
    const status = covering.some(row => row.checked) ? 'checked' : covering.length ? 'insufficient_speech' : 'outside_schedule'
    const previous = intervals[intervals.length - 1]
    if (previous?.status === status && previous.end_sample === start) previous.end_sample = end
    else intervals.push({ start_sample: start, end_sample: end, status })
  }
  return { checked_samples: intervals.filter(row => row.status === 'checked').reduce((n, row) => n + row.end_sample - row.start_sample, 0),
    intervals, windows }
}
