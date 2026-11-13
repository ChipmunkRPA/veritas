import { describe, expect, it } from 'vitest'
import type { Finding, SignalV2 } from '../../lib/api/queries'
import type { ImageEvidencePacket, ImageEvidenceSet } from '../evidence/imagePackets'
import type {
  AudioEvidencePacket, AudioEvidenceSet, DocumentEvidencePacket, DocumentEvidenceSet,
  VideoAVPacket, VideoEvidencePacket, VideoEvidenceSet,
} from '../evidence/multimediaPackets'
import { findingSupport } from './findingSupport'

const analysisId = 'analysis-1'
const finding = (value: Partial<Finding> = {}): Finding => ({ id: 'F1', family: 'F4_localised_manipulation',
  title: 'Recorded result', statement: 'Saved statement.', rank: 1, strength: 'weak', reference_range: '',
  review_priority: 'low', evidence_ids: ['signal:one'], ...value })
const signal = (value: Partial<SignalV2> = {}): SignalV2 => ({ id: 'signal:one', detector_id: 'private-engine',
  display_id: 'S01', status: 'ok', observed: 'A measured response was recorded.', ...value } as SignalV2)
const commonPacket = { id: 'E01', finding_ids: ['F1'], title: 'Internal engine name', observation: 'A region was selected for inspection.',
  interpretation: 'Inspect in context.', limitations: [], signal_refs: [{ id: 'signal:one', display_id: 'S01' }], artifacts: {} }
const commonSet = { status: 'ready' as const, id: 'saved-set', analysis_id: analysisId, revision: 1, sha256: 'set-hash',
  original_result_hash: 'result-hash', limitations: [] }
const imagePacket = (value: Partial<ImageEvidencePacket> = {}): ImageEvidencePacket => ({ ...commonPacket,
  region: { x: 12, y: 24, w: 40, h: 80 }, crop_region: { x: 2, y: 4, w: 80, h: 100 }, exhibit_refs: ['exhibit-1'],
  evidence_type: 'anomaly', semantics: 'residual', overlay_legend: '', measurements: [], ...value } as ImageEvidencePacket)
const image = (packets = [imagePacket()]): ImageEvidenceSet => ({ ...commonSet, schema: 'chipmunk.image-evidence/1',
  version: 'image-evidence-3', source: { sha256: 'original', width: 500, height: 500, coordinate_space: 'exif-oriented-original' },
  packets, omitted_regions: 0 })
const audioPacket = (value: Partial<AudioEvidencePacket> = {}): AudioEvidencePacket => ({ ...commonPacket, id: 'A01',
  interval: { start_sample: 32000, end_sample: 64000 }, context_interval: { start_sample: 0, end_sample: 96000 },
  start_s: 2, end_s: 4, observations: [], anchor: { sample_rate: 16000, source_timestamp_ranges: [] }, ...value })
const audio = (packets = [audioPacket()]): AudioEvidenceSet => ({ ...commonSet, schema: 'chipmunk.audio-evidence/1',
  version: 'audio-evidence-1', source: { sha256: 'original', sample_rate: 16000 }, packets, omitted_windows: 0 })
const videoPacket = (value: Partial<VideoEvidencePacket> = {}): VideoEvidencePacket => ({ ...commonPacket, id: 'V01',
  claim_scope: 'sampled_frames', measurements: [], frames: [{ id: 'frame-17', score: 0.8, is_peak: true, region_meaning: 'context',
    anchor: { frame_index: 17, pts: 2300, time_base: '1/1000', stream_time_s: 2.3, playback_time_s: 1.3 },
    artifact_refs: { original: 'frame', context: 'context' }, context_region: null }], ...value })
const video = (packets = [videoPacket()]): VideoEvidenceSet => ({ ...commonSet, schema: 'chipmunk.video-evidence/1',
  version: 'video-evidence-4', source: { sha256: 'original' }, packets, omitted_candidates: 0 })
const docPacket = (value: Partial<DocumentEvidencePacket> = {}): DocumentEvidencePacket => ({ ...commonPacket, id: 'T01',
  focus: { start: 12, end: 30, text: 'The selected text.' }, context: { start: 0, end: 50, text: 'Surrounding context.' },
  anchors: [{ kind: 'pdf', page: 3 }], measurements: [], supporting_windows: [], ...value })
const document = (packets = [docPacket()]): DocumentEvidenceSet => ({ ...commonSet, schema: 'chipmunk.document-evidence/1',
  version: 'document-evidence-1', source: { sha256: 'original', chars: 100, offset_unit: 'unicode_codepoint' }, packets, omitted_windows: 0 })
const resolve = (evidence = image(), selected = finding(), signals = [signal()]) => findingSupport({ analysisId, finding: selected, signals, evidence })

describe('findingSupport', () => {
  it('links the exact image packet, crop coordinates and saved observation without claiming edited pixels', () => {
    const item = resolve().packets[0]
    expect(item).toMatchObject({ id: 'E01', association: 'finding', scope: 'image_region',
      href: '/analyses/analysis-1/image-evidence?view=plain#image-evidence-E01', summary: commonPacket.observation })
    expect(item.label).toBe('E01 · Region x 12, y 24 · 40 × 80 px')
    expect(item.scopeNote).toContain('not proof')
    expect(item.label).not.toContain(commonPacket.title)
  })

  it('resolves exact exhibit references but labels shared-check association as inspection context', () => {
    const unrelated = imagePacket({ finding_ids: ['F99'] })
    expect(resolve(image([unrelated]), finding({ evidence_ids: ['exhibit-1'] }), []).packets[0].association).toBe('finding')
    expect(resolve(image([unrelated])).packets[0].association).toBe('shared_check')
    expect(resolve(image([unrelated]), finding({ evidence_ids: ['exhibit-10', 'S01'] }), []).packets).toEqual([])
  })

  it('never assigns an unrelated whole-image classifier to a localized region', () => {
    const selected = finding({ family: 'F3_whole_image_generation', evidence_ids: ['signal:global'] })
    const result = resolve(image([imagePacket({ finding_ids: ['F2'] })]), selected, [signal({ id: 'signal:global' })])
    expect(result.packets).toEqual([])
    expect(result.checks).toHaveLength(1)
  })

  it.each(['coverage', 'F9_distribution_conditions'] as const)('does not turn %s into a local evidence claim', (family) => {
    const result = findingSupport({ analysisId, finding: finding({ family }), signals: [signal()], evidence: audio() })
    expect(result.packets).toEqual([])
    expect(result.checks).toHaveLength(1)
  })

  it('creates fallback links only for real exact checks and uses their actual legacy ledger anchor', () => {
    const selected = finding({ evidence_ids: ['missing', 'S02', 'detector:private-engine', 'signal:one'] })
    const result = findingSupport({ analysisId: 'analysis / one', finding: selected,
      signals: [signal({ legacy: { v1_id: 'old/check#1' } as SignalV2['legacy'] }), signal({ id: 'unrelated', detector_id: 'private-engine-extra', display_id: 'S02' })] })
    expect(result.packets).toEqual([])
    expect(result.checks).toHaveLength(1)
    expect(result.checks[0]).toMatchObject({ label: 'S01 · Recorded check',
      href: '/analyses/analysis%20%2F%20one/ledger?view=expert#signal-old%2Fcheck%231' })
    expect(findingSupport({ analysisId, finding: finding({ evidence_ids: ['missing'] }), signals: [] }).checks).toEqual([])
  })

  it('does not use display identifiers as signal identity or invent fallback check links', () => {
    expect(resolve(image([imagePacket({ finding_ids: [] })]), finding({ evidence_ids: ['S01'] })).packets).toEqual([])
    expect(findingSupport({ analysisId, finding: finding({ evidence_ids: ['signal:missing'] }), signals: [signal()] }).checks).toEqual([])
    expect(resolve(image([imagePacket({ finding_ids: [] })]), finding({ evidence_ids: ['signal:one'] }), []).packets).toEqual([])
  })

  it('resolves evidence references attached to a recorded finding sentence', () => {
    const result = resolve(image([imagePacket({ finding_ids: [] })]), finding({ evidence_ids: [],
      sentences: [{ text: 'Saved observation.', evidence_ids: ['signal:one', 'exhibit-1'] }] }))
    expect(result.checks).toHaveLength(1)
    expect(result.packets[0].association).toBe('finding')
  })

  it('uses the actual not-run row anchor for an unavailable check with a legacy id', () => {
    const result = findingSupport({ analysisId, finding: finding(), signals: [signal({ status: 'skipped',
      legacy: { v1_id: 'old/check' } as SignalV2['legacy'] })] })
    expect(result.checks[0].href).toBe('/analyses/analysis-1/ledger?view=expert#signal-signal%3Aone')
  })

  it('retains contradictory document responses and exact Unicode offsets', () => {
    const observation = 'Two overlapping checks disagree: one supports and one opposes AI authorship.'
    const result = findingSupport({ analysisId, finding: finding(), signals: [signal()], evidence: document([docPacket({ observation })]) })
    expect(result.packets[0]).toMatchObject({ label: 'T01 · Page 3 · Characters [12, 30)', summary: observation,
      href: '/analyses/analysis-1/media-evidence?view=plain#media-evidence-T01', scope: 'text_passage' })
    expect(result.packets[0].scopeNote).toContain('not proof of authorship')
  })

  it('labels audio with decoded sample positions, not original stream timestamps', () => {
    const result = findingSupport({ analysisId, finding: finding(), signals: [signal()], evidence: audio() })
    expect(result.packets[0].label).toBe('A01 · Audio 2.000 s–4.000 s (decoded audio)')
    expect(result.packets[0].scopeNote).toContain('Samples [32000, 64000) at 16000 Hz')
    expect(result.packets[0].href).toContain('#media-evidence-A01')
  })

  it('links a recorded audio finding to its scored coverage when no inspection windows were selected', () => {
    const set = audio([])
    set.coverage = { decoded_samples: 96000, scored_samples: 96000, unscored_samples: 0, decode_limit_reached: false,
      selection_threshold: .5, windows: [
        { start_sample: 0, end_sample: 64000, score: .004, signal_refs: commonPacket.signal_refs },
        { start_sample: 32000, end_sample: 96000, score: .015, signal_refs: commonPacket.signal_refs },
      ] }
    const voice = finding({ family: 'A1_voice' })
    expect(findingSupport({ analysisId, finding: voice, signals: [signal()], evidence: set }).voiceCoverage).toMatchObject({
      href: '/analyses/analysis-1/media-evidence?view=plain#media-evidence-voice-coverage', scope: 'audio_coverage',
    })
    expect(findingSupport({ analysisId, finding: voice, signals: [signal({ status: 'skipped' })], evidence: set }).voiceCoverage).toBeUndefined()
    expect(findingSupport({ analysisId, finding: voice, signals: [signal()], evidence: { ...set, packets: [audioPacket()] } }).voiceCoverage).toBeUndefined()
    expect(findingSupport({ analysisId, finding: voice, signals: [signal()], evidence: audio([]) }).voiceCoverage).toBeUndefined()
  })

  it('uses actual video source timestamps, preserving sampled-window scope', () => {
    const first = findingSupport({ analysisId, finding: finding(), signals: [signal()], evidence: video() }).packets[0]
    expect(first.label).toBe('V01 · Frame 17 · 2.300 s (source clock)')
    const set = video([videoPacket({ claim_scope: 'sampled_window', window: { frame_indices: [17], start_pts: 2300, end_pts: 3100,
      time_base: '1/1000', start_stream_s: 2.3, end_stream_s: 3.1, start_playback_s: 1.3, end_playback_s: 2.1 } })])
    const item = findingSupport({ analysisId, finding: finding(), signals: [signal()], evidence: set }).packets[0]
    expect(item.label).toContain('Sampled window 2.300 s–3.100 s (source clock)')
    expect(item.scopeNote).toContain('does not identify which individual frames')
  })

  it('uses verified paired boundaries for a linked face finding', () => {
    const packet = videoPacket()
    packet.face_boundary = { method: 'face-appearance-paired-edge/1', start_stream_s: 2.35, end_stream_s: 2.56,
      frame_inputs: [{ frame_index: 17 }] }
    packet.face_boundary_frames = [{ ...packet.frames[0], id: 'V01.B1', interval_role: 'before' }]
    const set = { ...video([packet]), version: 'video-evidence-20' }
    const item = findingSupport({ analysisId, finding: finding(), signals: [signal()], evidence: set }).packets[0]
    expect(item.label).toBe('V01 · Refined face-change range 2.350 s–2.560 s (source clock)')
    expect(item.scopeNote).toContain('The end is exclusive.')
  })

  it('routes nested video voice and timing packets to their own real packet hashes', () => {
    const set = nestedVideo()
    const voice = findingSupport({ analysisId, finding: finding({ id: 'F4' }), signals: [], evidence: set })
    expect(voice.packets.map((item) => item.id)).toEqual(['A01'])
    expect(voice.packets[0].href).toContain('#media-evidence-A01')
    const timing = findingSupport({ analysisId, finding: finding({ id: 'F7' }), signals: [], evidence: set })
    expect(timing.packets.map((item) => item.id)).toEqual(['AV01'])
    expect(timing.packets[0]).toMatchObject({ scope: 'av_interval', href: '/analyses/analysis-1/media-evidence?view=plain#media-evidence-AV01' })
  })

  it('rejects nested packets when their source, renderable version or coverage is missing', () => {
    for (const mutate of [
      (set: VideoEvidenceSet) => { set.audio!.source.sha256 = 'different-source'; set.av!.source.sha256 = 'different-source' },
      (set: VideoEvidenceSet) => { set.version = 'video-evidence-2' },
      (set: VideoEvidenceSet) => { set.audio!.coverage = null; set.av!.coverage = null },
      (set: VideoEvidenceSet) => { set.audio!.status = 'unavailable'; set.av!.status = 'unavailable' },
    ]) {
      const set = nestedVideo(); mutate(set)
      for (const id of ['F4', 'F7']) expect(findingSupport({ analysisId, finding: finding({ id }), signals: [], evidence: set }).packets).toEqual([])
    }
  })

  it('fails closed on invalid source locations and unavailable or different-analysis records', () => {
    expect(resolve(image([imagePacket({ region: { x: 490, y: 1, w: 40, h: 10 } })])).packets).toEqual([])
    const otherAnalysis = image(); otherAnalysis.analysis_id = 'other-analysis'
    expect(resolve(otherAnalysis).packets).toEqual([])
    const unsaved = image(); unsaved.id = null
    expect(resolve(unsaved).packets).toEqual([])
    const unavailable = image(); unavailable.status = 'not_generated'
    expect(resolve(unavailable).packets).toEqual([])
    for (const evidence of [audio([audioPacket({ start_s: 9 })]), audio([audioPacket({ anchor: { sample_rate: 0, source_timestamp_ranges: [] } })]),
      document([docPacket({ focus: { start: 90, end: 110, text: 'outside source' } })]),
      video([videoPacket({ frames: [] })])]) {
      expect(findingSupport({ analysisId, finding: finding(), signals: [signal()], evidence }).packets).toEqual([])
    }
  })

  it('keeps direct associations ahead of related inspections and removes duplicate packet links', () => {
    const result = resolve(image([imagePacket({ id: 'E02', finding_ids: ['F2'] }), imagePacket(), imagePacket()]))
    expect(result.packets.map((item) => item.id)).toEqual(['E01', 'E02'])
  })
})

function nestedVideo(): VideoEvidenceSet {
  return { ...video([videoPacket({ finding_ids: ['F3'] })]),
    audio: { status: 'ready', reason: null, source: { sha256: 'original', sample_rate: 16000 }, limitations: [],
      packets: [audioPacket({ finding_ids: ['F4'] })], coverage: { decoded_samples: 96000, scored_samples: 32000, unscored_samples: 64000,
        windows: [], decode_limit_reached: false, selection_threshold: 0.5 } },
    av: { status: 'ready', reason: null, source: { sha256: 'original', sample_rate: 16000 }, limitations: [],
      packets: [{ ...commonPacket, id: 'AV01', finding_ids: ['F7'], interval: { start_sample: 32000, end_sample: 64000 },
        claim_scope: 'sampled_av_window', context_interval: { start_sample: 0, end_sample: 96000 },
        anchor: { sample_rate: 16000, source_timestamp_ranges: [] }, measurements: [], frames: [],
        window: { start_tick: 50, end_tick: 100, frame_indices: [], frame_inputs: [] },
        measurement: { offset_ticks: 2, offset_ms: 80, peak_correlation: 0.8, low_correlation: false, lag_profile: [], audio_curve: [],
          overlaps_offset_step: false, overlaps_correlation_collapse: false, baseline_offset_ticks: 2, collapse_threshold: 0.1,
          minimum_trusted_peak: 0.2, reading: 'Timing offset measured.' } } satisfies VideoAVPacket],
      coverage: { recorded_windows: 1, selected_windows: 1, omitted_windows: 0, selection: 'recorded', unsupported_checks: [] } } }
}
