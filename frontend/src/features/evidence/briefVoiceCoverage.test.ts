import { describe, expect, it } from 'vitest'
import { briefVoiceCoverage } from './briefVoiceCoverage'
import type { AudioEvidenceSet, BriefVoiceReview } from './multimediaPackets'

function fixture(schema: BriefVoiceReview['schema'] = 'chipmunk.brief-voice-review/1') {
  const review: BriefVoiceReview = { schema, status: 'unassessed', reason: 'No adjacent checked pair.', limitation: 'Zero decision weight.',
    affects_overall_confidence: false, scheduled_windows: 23, speech_windows_checked: 1, windows: [],
    window_checks: Array.from({ length: 23 }, (_, i) => ({ start_sample: i * 8000,
      speech_samples: i === 1 ? 6000 : 5999, elevated: i === 1 ? false : null })) }
  const source: AudioEvidenceSet['source'] = { sha256: 'a'.repeat(64), samples: 192000, sample_rate: 16000,
    timeline_time_base: '1/16000', timeline: [{ start_sample: 0, end_sample: 192000, source_pts: 0 }] }
  return { review, source }
}

describe('brief recorded speech coverage', () => {
  it.each(['chipmunk.brief-voice-review/1', 'chipmunk.brief-voice-review/2'] as const)('maps original checked/gap/tail positions for %s', schema => {
    const { review, source } = fixture(schema)
    const before = JSON.stringify(review)
    const coverage = briefVoiceCoverage(review, source)!
    expect(coverage.checked_samples).toBe(12000)
    expect(coverage.intervals).toEqual([
      { start_sample: 0, end_sample: 8000, status: 'insufficient_speech' },
      { start_sample: 8000, end_sample: 20000, status: 'checked' },
      { start_sample: 20000, end_sample: 188000, status: 'insufficient_speech' },
      { start_sample: 188000, end_sample: 192000, status: 'outside_schedule' },
    ])
    expect(coverage.windows).toHaveLength(23)
    expect(JSON.stringify(review)).toBe(before)
    review.window_checks![2] = { start_sample: 16000, speech_samples: 12000, elevated: true }
    review.speech_windows_checked = 2
    expect(briefVoiceCoverage(review, source)!.checked_samples).toBe(20000)
  })

  it.each(['skipped', 'missing', 'partial', 'shifted', 'invalid_admission', 'wrong_count', 'source_clock', 'null_row', 'null_timeline', 'schema'])('declines %s without inventing checked or uncovered regions', mutation => {
    const { review, source } = fixture()
    if (mutation === 'skipped') review.status = 'skipped'
    else if (mutation === 'missing') delete review.window_checks
    else if (mutation === 'partial') review.window_checks!.pop()
    else if (mutation === 'shifted') review.window_checks![0].start_sample = 1
    else if (mutation === 'invalid_admission') review.window_checks![0].elevated = false
    else if (mutation === 'wrong_count') review.speech_windows_checked = 23
    else if (mutation === 'null_row') review.window_checks![0] = null as unknown as NonNullable<BriefVoiceReview['window_checks']>[number]
    else if (mutation === 'null_timeline') source.timeline![0] = null as unknown as NonNullable<AudioEvidenceSet['source']['timeline']>[number]
    else if (mutation === 'schema') review.schema = 'unknown' as BriefVoiceReview['schema']
    else source.timeline![0].source_pts = 1
    expect(briefVoiceCoverage(review, source)).toBeUndefined()
  })
})
