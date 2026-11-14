// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { MultimediaEvidenceContent } from './MultimediaEvidenceCards'
import { evidenceAssetUrl, type AudioEvidenceSet, type DocumentEvidenceSet, type EvidenceAsset, type LongVoiceReview, type MultimediaEvidenceSet, type VideoEvidenceSet } from './multimediaPackets'

const asset = (hash: string, media_type: 'image/png' | 'audio/wav' = 'image/png'): EvidenceAsset => ({ sha256: hash.repeat(64), media_type,
  width: 300, height: 200, url: 'https://unrelated.invalid/collect-token', samples: 64000, sample_rate: 16000 })
const base = { status: 'ready' as const, id: 'set 1', version: 'evidence-1', sha256: 'a'.repeat(64), analysis_id: 'analysis-1', revision: 1,
  original_result_hash: 'b'.repeat(64), limitations: [], source: { sha256: 'c'.repeat(64) } }
const packet = { id: 'P01', title: 'Recorded inspection clue', observation: 'A recorded response selected this input.',
  interpretation: 'This response is not proof.', limitations: ['Ordinary conditions can cause a response.'], finding_ids: ['F5'], signal_refs: [{ id: 'check.private:global', display_id: 'S04' }] }
const audio: AudioEvidenceSet = { ...base, schema: 'chipmunk.audio-evidence/1', omitted_windows: 0,
  source: { ...base.source, samples: 160000, sample_rate: 16000, source_channels: 2 },
  coverage: { decoded_samples: 160000, scored_samples: 96000, unscored_samples: 64000, decode_limit_reached: true, selection_threshold: 0.5,
    windows: [{ start_sample: 32000, end_sample: 96000, score: 0.7, signal_refs: packet.signal_refs }] },
  overview: { waveform: asset('d'), scores: asset('e') },
  packets: [{ ...packet, interval: { start_sample: 32000, end_sample: 96000 }, context_interval: { start_sample: 0, end_sample: 128000 }, start_s: 2, end_s: 6,
    observations: [{ label: 'Window response', value: 0.7, unit: 'uncalibrated score', signal_refs: packet.signal_refs, unit_ref: 'window:0' }],
    anchor: { sample_rate: 16000, source_timestamp_ranges: [{ start_sample: 32000, end_sample: 96000, source_start_pts: 192000, source_end_pts: 256000, time_base: '1/16000' }] },
    artifacts: { selected_clip: asset('f', 'audio/wav'), context_clip: asset('1', 'audio/wav'), waveform: asset('2'), scores: asset('3') } }] }
const video: VideoEvidenceSet = { ...base, schema: 'chipmunk.video-evidence/1', omitted_candidates: 0,
  source: { ...base.source, timeline_origin_s: 10, time_base: '1/1000' },
  coverage: { frame_count: 100, scored_frame_count: 3, window_frame_indices: [], window_check_count: 0, decoded_frame_indices: [0, 3, 6], table_truncated: false, unmapped_observations: 1 },
  packets: [{ ...packet, claim_scope: 'sampled_frames', measurements: [{ label: 'Frame response', value: 0.7, unit: 'uncalibrated score' }],
    artifacts: { original: asset('d'), context: asset('e'), crop: asset('f') }, frames: [{ id: 'V01.1', score: 0.7, is_peak: true, region_meaning: 'detected_face_context',
      context_region: { x: 1, y: 2, w: 3, h: 4 }, artifact_refs: { original: 'original', context: 'context', crop: 'crop' },
      anchor: { frame_index: 6, pts: 12000, time_base: '1/1000', stream_time_s: 12, playback_time_s: 2 } }] }] }
const document: DocumentEvidenceSet = { ...base, schema: 'chipmunk.document-evidence/1', omitted_windows: 0,
  source: { ...base.source, offset_unit: 'unicode_codepoint', chars: 80 }, coverage: { scored_windows: 2, unmapped_pages: [3], email: true },
  packets: [{ ...packet, focus: { start: 8, end: 30, text: 'Quoted 😀 <script>text</script>' }, context: { start: 0, end: 60, text: 'Continuous source context with intervening layout.' },
    supporting_windows: [{ signal_ref: packet.signal_refs[0], score: 0.8, text: 'Exact scored context A.', input_sha256: 'f'.repeat(64), spans: [{ start: 8, end: 30 }], token_coverage: { tokens: 8, token_windows: 1 } },
      { signal_ref: { id: 'check.opaque:global', display_id: 'S05' }, score: 0.2, text: 'Different actual scored context B.', input_sha256: 'e'.repeat(64), spans: [{ start: 0, end: 30 }], token_coverage: {} }],
    measurements: [], anchors: [{ kind: 'email', quote_role: 'quoted', selected_start: 8, selected_end: 30 }, { kind: 'pdf', page: 2 }],
    artifacts: { page2_context: { ...asset('d'), page: 2, role: 'context' }, page2_crop: { ...asset('e'), page: 2, role: 'crop' }, page2_overlay: { ...asset('f'), page: 2, role: 'overlay' } } }] }
const show = (set: MultimediaEvidenceSet, token = 'signed token', mediaReady = true) => render(<MemoryRouter><MultimediaEvidenceContent set={set} token={token} mediaReady={mediaReady} /></MemoryRouter>)
afterEach(cleanup)

describe('ordinary multimedia evidence', () => {
  it('shows submitted Word revision wording and its native XML position', () => {
    const set: DocumentEvidenceSet = { ...document, packets: [], coverage: {
      word_tracked_changes: { status: 'partial', part: 'word/document.xml', paragraphs: 3, inserted: 1, deleted: 0,
        scope: 'Checks inline Track Changes.', limitation: 'Authorship remains undetermined.',
        unsupported_marks: 1, omitted_excerpts: 0, authorship_confidence_contribution: 0,
        rows: [{ kind: 'inserted', paragraph: 2, body_index: 1, inline_index: 4,
          text: '<b>Added text</b> 😀', characters: 19 }] },
    } }
    show(set)
    const section = within(screen.getByRole('region', { name: 'Word tracked text changes' }))
    expect(section.getByText('Inserted · Body paragraph 2')).toBeTruthy()
    expect(section.getByText(/paragraph child index 4/)).toBeTruthy()
    expect(section.getByText(/<b>Added text<\/b> 😀/).querySelector('b')).toBeNull()
    expect(section.getByText(/additional revision marks were outside scope/)).toBeTruthy()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it('links voice evidence to only sound categories overlapping its exact samples', () => {
    const set = structuredClone(audio)
    set.source.pcm_sha256 = 'a'.repeat(64)
    set.sound_events = {
      schema: 'chipmunk.sound-events/1', status: 'ready',
      source: { sha256: set.source.sha256, pcm_sha256: set.source.pcm_sha256, sample_rate: 16000, samples: 160000 },
      method: { window_samples: 160000, hop_samples: 160000, description: 'Sound categories' }, limitations: [],
      windows: [[0, 32000], [32000, 96000]].map(([start, end], index) => ({
        id: `C0${index + 1}`, start_sample: start, end_sample: end, start_s: start / 16000, end_s: end / 16000,
        labels: [{ label: 'Music', score: 0.7 }],
        source_timestamp_ranges: [{ start_sample: start, end_sample: end, source_start_pts: start, source_end_pts: end, time_base: '1/16000' }],
        artifacts: { selected_clip: asset('e', 'audio/wav'), waveform: asset('d') },
      })),
    }
    render(<MemoryRouter initialEntries={['/analyses/audio/media-evidence?view=plain']}>
      <MultimediaEvidenceContent set={set} token="signed token" mediaReady />
    </MemoryRouter>)
    const voice = within(screen.getByRole('article', { name: 'Selected audio evidence P01' }))
    expect(voice.queryByRole('link', { name: 'Sound categories C01' })).toBeNull()
    const contextLink = voice.getByRole('link', { name: 'Sound categories C02' })
    expect(contextLink.getAttribute('href')).toBe('/analyses/audio/media-evidence?view=plain#media-evidence-C02')
    fireEvent.click(contextLink)
    const categories = within(screen.getByRole('article', { name: 'Sound categories C02' }))
    const returnLink = categories.getByRole('link', { name: 'P01' })
    expect(returnLink.getAttribute('href')).toBe('/analyses/audio/media-evidence?view=plain#media-evidence-P01')
    fireEvent.click(returnLink)
    expect(screen.getByRole('article', { name: 'Selected audio evidence P01' })).toBeTruthy()
  })

  it('distinguishes second writing responses and links the matching source passage', () => {
    const set = structuredClone(document)
    set.coverage!.writing_second_check = {
      schema: 'document-writing-second-check/1', authorship_confidence_contribution: 0,
      ranges: [
        { start: 8, end: 30, spans: [{ start: 0, end: 6 }, { start: 8, end: 30 }], state: 'corroborated' },
        { start: 32, end: 45, spans: [{ start: 8, end: 30 }, { start: 32, end: 45 }], state: 'different' },
        { start: 47, end: 60, spans: [{ start: 32, end: 45 }, { start: 47, end: 60 }], state: 'unavailable' },
      ],
    }
    show(set)
    expect(screen.getByRole('heading', { name: 'Second writing check' })).toBeTruthy()
    expect(screen.getByText(/Both checks gave an AI-like response/)).toBeTruthy()
    expect(screen.getByText(/The checks gave different responses/)).toBeTruthy()
    expect(screen.getByText(/Second response unavailable/)).toBeTruthy()
    const passage = screen.getByRole('link', { name: 'Passage [8, 30)' })
    expect(passage.getAttribute('href')).toBe('#media-evidence-P01')
    fireEvent.click(passage)
    expect(screen.getByText('Quoted 😀 <script>text</script>')).toBeTruthy()
  })

  it('shows the original selectable page for a mixed-PDF writing response', () => {
    const set = structuredClone(document)
    set.coverage!.writing_second_check = {
      schema: 'document-writing-second-check/6', authorship_confidence_contribution: 0,
      pdf_spans: [{ start: 0, end: 60, page: 3 }],
      ranges: [{ start: 8, end: 30, spans: [{ start: 0, end: 6 }, { start: 8, end: 30 }], state: 'corroborated' }],
    }
    show(set)
    expect(screen.getByText(/uses selectable text on PDF page 3/)).toBeTruthy()
    expect(screen.getByText(/leaves OCR, partially checked pages, and contexts crossing pages outside/)).toBeTruthy()
    expect(screen.queryByText(/wholly inside the reply body/)).toBeNull()
    const link = screen.getByRole('link', { name: 'Page 3 · Passage [8, 30)' })
    expect(link.getAttribute('href')).toBe('#media-evidence-P01')
    fireEvent.click(link)
    expect(screen.getByText('Quoted 😀 <script>text</script>')).toBeTruthy()
  })

  it.each(['document-writing-second-check/2', 'document-writing-second-check/3', 'document-writing-second-check/4', 'document-writing-second-check/5'] as const)(
    'labels the second check on an email reply as body-only for %s', (schema) => {
    const set = structuredClone(document)
    set.coverage!.writing_second_check = {
      schema, authorship_confidence_contribution: 0,
      body_spans: [{ start: 0, end: 30 }], ranges: [],
    }
    show(set)
    expect(screen.getByText(/This check covers contexts wholly inside the reply body/)).toBeTruthy()
    expect(screen.getByText(/Text marked as a quote or signature and contexts that cross those boundaries are outside this check/)).toBeTruthy()
    expect(Boolean(screen.queryByText(/decoded HTML body of the selected email part/)))
      .toBe(schema === 'document-writing-second-check/4' || schema === 'document-writing-second-check/5')
  })

  it('shows a source-clock gap and its affected voice window in ordinary audio evidence', () => {
    const set = structuredClone(audio)
    set.coverage!.source_clock_context = { affects_overall_confidence: false, time_base: '1/16000',
      minimum_jump_samples: 320, checked_samples: 160000, total_discontinuities: 1, scored_windows_crossing: 1,
      breaks: [{ boundary_sample: 64000, jump_samples: 16000, scored_windows_crossing: 1,
        before: { start_sample: 0, end_sample: 64000, source_start_pts: 0, source_end_pts: 64000 },
        after: { start_sample: 64000, end_sample: 160000, source_start_pts: 80000, source_end_pts: 176000 } }] }
    show(set)
    const clock = within(screen.getByRole('region', { name: 'Audio source clock breaks' }))
    expect(clock.getByText('At 4.000 s of analyzed audio. Source timestamp gap of 1,000 ms.')).toBeTruthy()
    expect(clock.getByText('On the original timeline, the audio spans 0.000 s to 4.000 s before the break and 5.000 s to 11.000 s after it.')).toBeTruthy()
    expect(clock.getByText('1 recorded voice window crosses this break.')).toBeTruthy()
    expect(clock.getByText(/carries zero weight in the AI assessment/)).toBeTruthy()
    expect(clock.queryByText(/a recording-wide voice estimate is unavailable/)).toBeNull()
    const guarded = structuredClone(set)
    const guardedClock = guarded.coverage!.source_clock_context!
    guardedClock.recording_voice_estimate_withheld = true
    guardedClock.scored_windows_crossing = 0
    guardedClock.breaks[0].scored_windows_crossing = 0
    cleanup()
    show(guarded)
    expect(screen.getByText(/a recording-wide voice estimate is unavailable/)).toBeTruthy()
  })

  it('shows checked duration when no source-clock break is recorded', () => {
    const set = structuredClone(audio)
    set.coverage!.source_clock_context = { affects_overall_confidence: false, time_base: '1/16000',
      minimum_jump_samples: 320, checked_samples: 160000, total_discontinuities: 0, scored_windows_crossing: 0, breaks: [] }
    show(set)
    const clock = within(screen.getByRole('region', { name: 'Audio source clock breaks' }))
    expect(clock.getByText(/No source clock break of at least 20 ms was measured in 10\.000 s/)).toBeTruthy()
  })

  it('shows the flagged short speech window even when the primary voice review has no selected packet', () => {
    const set = { ...structuredClone(audio), packets: [] }
    const interval = { start_sample: 48000, end_sample: 80000 }
    set.short_voice_review = {
      schema: 'chipmunk.short-voice-review/1', status: 'flagged', title: 'Short speech-window review',
      reason: 'A two-second speech window returned an elevated response for review.',
      limitation: 'It carries no weight in the overall assessment.', affects_overall_confidence: false,
      checked_windows: [interval], interval, start_s: 3, end_s: 5,
      source_timestamp_ranges: [{ ...interval, source_start_pts: 48000, source_end_pts: 80000, time_base: '1/16000' }],
      artifacts: { selected_clip: asset('f', 'audio/wav'), waveform: asset('2') },
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Short speech-window review' }))
    expect(review.getByText(/3\.000 s–5\.000 s · samples \[48000, 80000\)/)).toBeTruthy()
    expect(review.getByText('Source PTS [48000, 80000) at 1/16000.')).toBeTruthy()
    expect(review.getByLabelText('Flagged two-second speech window').getAttribute('src')).toContain(`${'f'.repeat(64)}.wav`)
    expect(review.getByAltText('Waveform with the flagged two-second speech window marked')).toBeTruthy()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it('shows the exact background-reduced interval with distinct original and processed playback', () => {
    const set = { ...structuredClone(audio), packets: [] }
    const interval = { start_sample: 0, end_sample: 48000 }
    set.background_reduced_voice_review = {
      schema: 'chipmunk.background-reduced-voice-review/1', title: 'Speech review with reduced background sound',
      status: 'flagged', reason: 'The speech response leaned AI-generated after background sound was reduced.',
      limitation: 'This review carries no weight in the overall assessment.', affects_overall_confidence: false,
      interval, source_timestamp_ranges: [{ ...interval, source_start_pts: 0, source_end_pts: 48000, time_base: '1/16000' }],
      artifacts: { selected_clip: asset('f', 'audio/wav'), processed_clip: asset('e', 'audio/wav'), waveform: asset('2') },
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Speech review with reduced background sound' }))
    expect(review.getByText('Loud background sound can make genuine speech look AI-generated in this check.')).toBeTruthy()
    expect(review.getByText(/Original audio 0\.000 s–3\.000 s\. Source PTS \[0, 48000\) at 1\/16000/)).toBeTruthy()
    expect(review.getByAltText('Original waveform across the flagged three-second interval')).toBeTruthy()
    const original = review.getByLabelText('Original three-second interval')
    const processed = review.getByLabelText('Processed copy with reduced background sound')
    expect(original.getAttribute('src')).toContain(`${'f'.repeat(64)}.wav`)
    expect(processed.getAttribute('src')).toContain(`${'e'.repeat(64)}.wav`)
    const pause = vi.fn()
    Object.defineProperties(original, { paused: { value: false }, pause: { value: pause } })
    fireEvent.play(processed)
    expect(pause).toHaveBeenCalledOnce()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it('shows source ranges and plays only corroborating long speech windows', () => {
    const windows: LongVoiceReview['windows'] = [0, 56000, 112000].map((start_sample, index) => {
      const end_sample = start_sample + 48000
      const range = { start_sample, end_sample }
      return {
        ...range, pcm_sha256: 'b'.repeat(64), direction: index === 1 ? 'inconclusive' : 'ai',
        source_timestamp_ranges: [{ ...range, source_start_pts: start_sample, source_end_pts: end_sample, time_base: '1/16000' }],
        artifacts: { selected_clip: asset(index === 2 ? 'e' : 'f', 'audio/wav'), waveform: asset('2') },
      }
    })
    const set = { ...structuredClone(audio), packets: [] }
    set.long_voice_review = {
      schema: 'chipmunk.long-voice-review/1', title: 'Long speech-window review', status: 'corroborated',
      reason: 'At least two separate windows leaned AI-generated, and none leaned original.',
      limitation: 'The review carries no weight in the overall assessment.', affects_overall_confidence: false, windows,
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Long speech-window review' }))
    expect(review.getByRole('status').textContent).toContain('At least two separate windows')
    expect(review.getAllByText('Leans AI-generated in this window')).toHaveLength(2)
    expect(review.getByText('Inconclusive window')).toBeTruthy()
    expect(review.getByText(/7\.000 s–10\.000 s · samples \[112000, 160000\)/)).toBeTruthy()
    expect(review.getByText(/Original audio 7\.000 s–10\.000 s\. Source PTS \[112000, 160000\) at 1\/16000/)).toBeTruthy()
    expect(review.getAllByRole('img')).toHaveLength(2)
    expect(review.getByText(/carries no weight in the overall assessment/)).toBeTruthy()
    const first = review.getByLabelText('Corroborating speech window at 0.000 s–3.000 s')
    const last = review.getByLabelText('Corroborating speech window at 7.000 s–10.000 s')
    expect(first.getAttribute('src')).toContain(`${'f'.repeat(64)}.wav`)
    expect(last.getAttribute('src')).toContain(`${'e'.repeat(64)}.wav`)
    const pause = vi.fn()
    Object.defineProperties(first, { paused: { value: false }, pause: { value: pause } })
    fireEvent.play(last)
    expect(pause).toHaveBeenCalledOnce()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it.each(['no_corroboration', 'skipped'] as const)('keeps a %s long speech review visible without offering clips', (status) => {
    const set = { ...structuredClone(audio), packets: [] }
    const skipped = status === 'skipped'
    const reason = skipped ? 'The optional long-window review could not complete.'
      : 'The sampled windows did not produce corroborating AI-like responses.'
    set.long_voice_review = {
      schema: 'chipmunk.long-voice-review/1', title: 'Long speech-window review', status, reason,
      limitation: 'The review carries no weight in the overall assessment.', affects_overall_confidence: false,
      windows: skipped ? [] : [{
        start_sample: 0, end_sample: 48000, pcm_sha256: 'a'.repeat(64), direction: 'original',
        source_timestamp_ranges: [{ start_sample: 0, end_sample: 48000, source_start_pts: 0, source_end_pts: 48000, time_base: '1/16000' }],
        artifacts: {},
      }],
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Long speech-window review' }))
    expect(review.getByRole('status').textContent).toBe(reason)
    expect(Boolean(review.queryByText('Leans original in this window'))).toBe(!skipped)
    expect(review.queryByLabelText(/Corroborating speech window/)).toBeNull()
    expect(review.queryByRole('img')).toBeNull()
    expect(screen.getByTestId('multimedia-evidence-empty')).toBeTruthy()
  })

  it('shows selected speech playback separately from the windows that support it', () => {
    const set = { ...structuredClone(audio), packets: [] }
    const timestamps = (start_sample: number, end_sample: number) => [{ start_sample, end_sample, source_start_pts: start_sample,
      source_end_pts: end_sample, time_base: '1/16000' }]
    set.long_voice_review = {
      schema: 'chipmunk.long-voice-review/2', title: 'Long speech-window review', status: 'corroborated',
      reason: 'The sampled windows support the speech ranges shown below.', limitation: 'This review carries no weight in the overall assessment.',
      affects_overall_confidence: false,
      windows: [24000, 48000].map((start_sample) => ({ start_sample, end_sample: start_sample + 48000,
        pcm_sha256: 'a'.repeat(64), direction: 'ai', source_timestamp_ranges: timestamps(start_sample, start_sample + 48000), artifacts: {} })),
      regions: [{ start_sample: 48000, end_sample: 72000, pcm_sha256: 'b'.repeat(64), basis: 'overlap', supporting_windows: [0, 1],
        source_timestamp_ranges: timestamps(48000, 72000), artifacts: { selected_clip: asset('f', 'audio/wav'), waveform: asset('2') } }],
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Long speech-window review' }))
    expect(review.getByText('Overlap of two AI-leaning windows')).toBeTruthy()
    expect(review.getByText(/Selected range 3\.000 s–4\.500 s.*48000, 72000/)).toBeTruthy()
    expect(review.getAllByText('Leans AI-generated in this window')).toHaveLength(2)
    const inputs = within(review.getByRole('list', { name: 'Supporting speech readings' }))
    const firstInput = inputs.getByRole('link', { name: '1.500 s–4.500 s' })
    expect(firstInput.getAttribute('href')).toBe('#long-speech-window-0')
    fireEvent.click(firstInput)
    expect(firstInput.ownerDocument.activeElement?.id).toBe('long-speech-window-0')
    expect(review.getByLabelText('Selected speech range at 3.000 s–4.500 s').getAttribute('src')).toContain(`${'f'.repeat(64)}.wav`)
    expect(review.queryByLabelText(/Corroborating speech window/)).toBeNull()
  })

  it('explains when dense audio windows have no usable speech responses', () => {
    const set = { ...structuredClone(audio), packets: [] }
    set.long_voice_review = {
      schema: 'chipmunk.long-voice-review/2', title: 'Long speech-window review', status: 'no_corroboration',
      reason: 'The sampled windows did not produce corroborating AI-like responses.',
      limitation: 'This review carries no weight in the overall assessment.', affects_overall_confidence: false, regions: [],
      windows: [{ start_sample: 0, end_sample: 48000, pcm_sha256: 'a'.repeat(64), direction: 'unassessed', artifacts: {},
        source_timestamp_ranges: [{ start_sample: 0, end_sample: 48000, source_start_pts: 0, source_end_pts: 48000, time_base: '1/16000' }] }],
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Long speech-window review' }))
    expect(review.getByRole('status').textContent).toBe('None of the sampled windows produced a usable speech response.')
    expect(review.queryByRole('link')).toBeNull()
  })

  it('shows the whole-clip face review as six positions and three source examples', () => {
    const set = structuredClone(video)
    const item = set.packets[0]
    const samples = [6, 12, 18, 24, 30, 36].map((frame_index, i) => ({ frame_index, pts: 12000 + i * 2000,
      responses_hex: ['0x1.0p-1', '0x1.0p+0'] as [string, string] }))
    item.title = 'Sampled faces to review'
    item.claim_scope = 'sampled_window'
    item.whole_face_review = { flagged: true, time_base: '1/1000', samples }
    item.window = { frame_indices: samples.map((row) => row.frame_index), start_pts: 12000, end_pts: 22000,
      time_base: '1/1000', start_stream_s: 12, end_stream_s: 22, start_playback_s: 2, end_playback_s: 12 }
    item.frames = [samples[0], samples[3], samples[5]].map((row) => ({ ...item.frames[0], id: `V01.${row.frame_index}`,
      score: null, is_peak: false, anchor: { frame_index: row.frame_index, pts: row.pts, time_base: '1/1000',
        stream_time_s: row.pts / 1000, playback_time_s: row.pts / 1000 - 10 } }))
    show(set)
    expect(screen.getByRole('heading', { name: /Sampled faces to review/ })).toBeTruthy()
    expect(screen.getByText(/Planned frame indices: 6, 12, 18, 24, 30, 36/)).toBeTruthy()
    expect(screen.getAllByAltText(/exact original video frame/)).toHaveLength(3)
    expect(screen.getAllByAltText(/source face crop for context/)).toHaveLength(3)
    expect(screen.getByRole('link', { name: 'F5' })).toBeTruthy()
    expect(screen.queryByText(/response 0\.700|flagged range/)).toBeNull()
  })

  it('explains a withheld coarse range in source coverage without a new finding card', () => {
    const set = structuredClone(video)
    set.packets = []
    const observation = 'The sampled face change at 2.360–2.880 s weakened when the same frames were checked at lower detail. The interval was withheld from the review ranges.'
    set.coverage!.face_detail_repeat = { status: 'withheld', observation, start_pts: 59000, end_pts: 72000,
      time_base: '1/25000', start_stream_s: 2.36, end_stream_s: 2.88, affects_overall_confidence: false }
    show(set)
    expect(screen.getByText(observation)).toBeTruthy()
    expect(screen.queryByRole('article')).toBeNull()
  })

  it('distinguishes a measured brief face scan from an unassessed follow-up without a finding', () => {
    const set = { ...structuredClone(video), packets: [] }
    set.coverage!.brief_face_review = { affects_overall_confidence: false, observations: [
      'Initial scan. No distinct brief face-change range was found in this scan. Comparable face inputs: 15 of 16 scheduled pictures.',
      'Denser scan. More than one face was detected in a sampled picture. Comparable face inputs: 10 of 24 scheduled pictures.',
    ] }
    show(set)
    const coverage = within(screen.getByRole('region', { name: 'Brief face-change check' }))
    for (const text of set.coverage!.brief_face_review.observations) expect(coverage.getByText(text)).toBeTruthy()
    expect(coverage.getByText('This check carries zero AI confidence weight.')).toBeTruthy()
    expect(screen.queryByRole('article')).toBeNull()
  })

  it('distinguishes native motion inputs from face samples in the brief HEVC range', () => {
    const set = { ...structuredClone(video), version: 'video-evidence-21' }
    const packet = set.packets[0]
    packet.face_return = { method: 'face-short-return/9', start_stream_s: 2.066667, end_stream_s: 2.266667,
      frame_inputs: [{ frame_index: 61 }, { frame_index: 65 }, { frame_index: 68 }],
      scene_frame_inputs: Array.from({ length: 120 }, (_, frame_index) => ({ frame_index })) }
    packet.window = { frame_indices: [61, 65, 68], start_pts: 62000, end_pts: 68000, time_base: '1/30000',
      start_stream_s: 2.066667, end_stream_s: 2.266667, start_playback_s: 2.066667, end_playback_s: 2.266667 }
    show(set)
    fireEvent.click(screen.getByText('Exact window coverage'))
    expect(screen.getByText(/Nearby motion was reviewed in 120 native pictures. Face appearances were measured in 3 sampled pictures/)).toBeTruthy()
    expect(screen.getByText('2.067 s–2.267 s')).toBeTruthy()
    expect(screen.getByText(/with an exclusive end/)).toBeTruthy()
  })

  it.each(['video-evidence-11', 'video-evidence-20'])('shows measured and refined ranges with source crops and playback navigation in %s', (version) => {
    const set = { ...structuredClone(video), version }
    const item = set.packets[0]
    item.claim_scope = 'sampled_window'
    item.frames[0].score = null
    item.face_sequence = { start_stream_s: 12, end_stream_s: 12.52, response: .2,
      centers: { before: [1, 2], during: [6, 8, 10], after: [12, 14] }, frame_inputs: [] }
    item.face_boundary = { method: version === 'video-evidence-20' ? 'face-appearance-paired-edge/1' : 'face-appearance-boundary/1', start_stream_s: 11.84, end_stream_s: 12.56,
      frame_inputs: [{ frame_index: 4 }, { frame_index: 8 }, { frame_index: 11 }] }
    item.face_boundary_frames = (['before', 'during', 'after'] as const).map((role, i) => ({
      ...item.frames[0], id: `V01.B${i + 1}`, interval_role: role, is_peak: false,
      anchor: { frame_index: 4 + 3 * i, pts: 11760 + 400 * i, time_base: '1/1000',
        stream_time_s: 11.76 + .4 * i, playback_time_s: 1.76 + .4 * i },
    }))
    show(set)
    expect(screen.getByText('12.000 s–12.520 s')).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Refined boundary estimate' })).toBeTruthy()
    expect(screen.getByText('11.840 s–12.560 s')).toBeTruthy()
    expect(screen.getAllByAltText(/V01\.B\d: source face crop for context/)).toHaveLength(3)
    fireEvent.click(screen.getByRole('button', { name: 'Go to frame 4 in playback' }))
    expect(screen.getByText(/Selected frame 4 · source 11\.760 s · player 1\.760 s/)).toBeTruthy()
    const refinedNavigation = screen.queryByRole('button', { name: 'Inspect V01.B1 at source 11.760 s' })
    if (version === 'video-evidence-20') {
      expect(screen.getByText(/Two appearance transitions on the same tracked face/)).toBeTruthy()
      expect(refinedNavigation).toBeTruthy()
      fireEvent.click(refinedNavigation!)
      expect(refinedNavigation!.getAttribute('aria-pressed')).toBe('true')
    } else {
      expect(screen.getByText(/3 denser samples on the same tracked face/)).toBeTruthy()
      expect(refinedNavigation).toBeNull()
    }
  })

  it.each(['video-evidence-12', 'video-evidence-13', 'video-evidence-14', 'video-evidence-15', 'video-evidence-16', 'video-evidence-17'])('shows the distinct brief face-change interval in %s', (version) => {
    const set = { ...structuredClone(video), version }
    const item = set.packets[0]
    item.title = 'Brief face-change range to review'
    item.claim_scope = 'sampled_window'
    item.face_return = { start_stream_s: 12, end_stream_s: 12.36,
      frame_inputs: [5, 6, 7, 8, 9, 10, 11, 12, 13].map((frame_index) => ({ frame_index })) }
    item.frames = (['before', 'during', 'after'] as const).map((role, i) => ({
      ...item.frames[0], id: `V01.${i + 1}`, interval_role: role, score: null, is_peak: false,
      anchor: { frame_index: 5 + 4 * i, pts: 11960 + 200 * i, time_base: '1/1000',
        stream_time_s: 11.96 + .2 * i, playback_time_s: 1.96 + .2 * i },
    }))
    show(set)
    expect(screen.getByText('12.000 s–12.360 s')).toBeTruthy()
    expect(screen.getByText(/exclusive end.*9 sampled face views.*no weight/)).toBeTruthy()
    expect(screen.getAllByAltText(/V01\.\d: source face crop for context/)).toHaveLength(3)
    fireEvent.click(screen.getByRole('button', { name: 'Go to frame 9 in playback' }))
    expect(screen.getByText(/Selected frame 9 · source 12\.160 s · player 2\.160 s/)).toBeTruthy()
  })

  it('keeps the two brief face-change ranges and their crop anchors separate', () => {
    const set = { ...structuredClone(video), version: 'video-evidence-18' }
    const first = set.packets[0]
    set.packets = [12, 13].map((start, packetIndex) => {
      const id = `VF0${packetIndex + 1}`
      const index = 10 + packetIndex * 25
      return {
        ...first, id, title: 'Brief face-change range to review', claim_scope: 'sampled_window',
        face_return: { start_stream_s: start, end_stream_s: start + .28,
          frame_inputs: Array.from({ length: 8 }, (_, i) => ({ frame_index: index - 1 + i })) },
        frames: (['before', 'during', 'after'] as const).map((interval_role, i) => {
          const time = start - .04 + i * .16
          return { ...first.frames[0], id: `${id}.${i + 1}`, interval_role, score: null, is_peak: false,
            anchor: { frame_index: index - 1 + 4 * i, pts: Math.round(time * 1000), time_base: '1/1000',
              stream_time_s: time, playback_time_s: time - 10 } }
        }),
      }
    })
    show(set)
    expect(screen.getByText(/These counts cover visual checks only/)).toBeTruthy()
    const [early, late] = screen.getAllByRole('article').map((article) => within(article))
    expect(early.getByText('12.000 s–12.280 s')).toBeTruthy()
    expect(late.getByText('13.000 s–13.280 s')).toBeTruthy()
    expect(early.getAllByAltText(/VF01\.\d: source face crop for context/)).toHaveLength(3)
    expect(late.getAllByAltText(/VF02\.\d: source face crop for context/)).toHaveLength(3)
    expect(early.getByText(/exclusive end.*no weight/)).toBeTruthy()
    expect(late.getByText(/exclusive end.*no weight/)).toBeTruthy()
    fireEvent.click(late.getByRole('button', { name: 'Go to frame 38 in playback' }))
    expect(screen.getByText(/Selected frame 38 · source 13\.120 s · player 3\.120 s/)).toBeTruthy()
  })

  it('keeps v19 native 30 fps ranges and playback anchors separate without implementation names', () => {
    const set = { ...structuredClone(video), version: 'video-evidence-19' }
    set.source = { ...set.source, time_base: '1/15360', timeline_origin_s: 0, width: 1280, height: 720 }
    const item = set.packets[0]
    set.packets = [
      { id: 'VFR01', start: 64, end: 70, anchors: [62, 66, 70], method: 'face-short-return/6' },
      { id: 'VFR02', start: 88, end: 96, anchors: [86, 92, 96], method: 'face-short-return/8' },
    ].map(({ id, start, end, anchors, method }) => {
      const frame_indices = Array.from({ length: end - start + 3 }, (_, i) => start - 2 + i)
      const time = { start_stream_s: start / 30, end_stream_s: end / 30 }
      return {
        ...item, id, title: 'Brief face-change range to review', claim_scope: 'sampled_window',
        face_return: { ...time, method, frame_inputs: frame_indices.map((frame_index) => ({ frame_index })) },
        window: { ...time, frame_indices, start_pts: start * 512, end_pts: end * 512, time_base: '1/15360',
          start_playback_s: start / 30, end_playback_s: end / 30 },
        frames: (['before', 'during', 'after'] as const).map((interval_role, index) => {
          const frame_index = anchors[index]
          return { ...item.frames[0], id: `${id}.${index + 1}`, interval_role, score: null, is_peak: false,
            anchor: { frame_index, pts: frame_index * 512, time_base: '1/15360',
              stream_time_s: frame_index / 30, playback_time_s: frame_index / 30 } }
        }),
      }
    })
    const { container } = show(set)
    expect(screen.getByText(/These counts cover visual checks only/)).toBeTruthy()
    const [early, late] = screen.getAllByRole('article').map((article) => within(article))
    expect(early.getByText('2.133 s–2.333 s')).toBeTruthy()
    expect(late.getByText('2.933 s–3.200 s')).toBeTruthy()
    expect(early.getAllByAltText(/VFR01\.\d: source face crop for context/)).toHaveLength(3)
    expect(late.getAllByAltText(/VFR02\.\d: source face crop for context/)).toHaveLength(3)
    expect(late.getByText(/exclusive end.*no weight/)).toBeTruthy()
    fireEvent.click(late.getByRole('button', { name: 'Go to frame 92 in playback' }))
    expect(screen.getByText(/Selected frame 92 · source 3\.067 s · player 3\.067 s/)).toBeTruthy()
    expect(container.textContent).not.toMatch(/face-short-return|check\.private|GenD|SFace/i)
  })

  it('shows one exact picture per hold endpoint and locates it on the source clock', () => {
    const set = structuredClone(video)
    set.version = 'video-evidence-10'
    set.picture_hold = { status: 'matched', frames_scanned: 120, frame_limit: 120, last_frame_index: 119,
      last_stream_time_s: 14.76, last_pts: 14760, time_base: '1/1000', reason: null, packet_id: 'VPH01' }
    const item = set.packets[0]
    item.id = 'VPH01'
    item.claim_scope = 'consecutive_frames'
    item.picture_hold = { start_frame_index: 50, end_frame_index: 63, start_pts: 12000, end_pts: 12520,
      start_stream_s: 12, end_stream_s: 12.52, start_playback_s: 2, end_playback_s: 2.52, span_s: 0.52,
      time_base: '1/1000', frame_count: 14, decoded_rgb_sha256: 'b'.repeat(64), pts: [] }
    item.frames = [50, 63].map((index, endpoint) => ({ ...item.frames[0], id: `VPH01.${endpoint + 1}`, score: null, is_peak: false,
      context_region: null, artifact_refs: { original: 'original', context: 'original' },
      anchor: { ...item.frames[0].anchor, frame_index: index, pts: 12000 + 520 * endpoint, stream_time_s: 12 + .52 * endpoint, playback_time_s: 2 + .52 * endpoint } }))
    show(set)
    expect(screen.getByRole('link', { name: 'Inspect the recorded hold' }).getAttribute('href')).toBe('#media-evidence-VPH01')
    expect(screen.getByText(/14 consecutive decoded frames/)).toBeTruthy()
    expect(screen.getAllByAltText(/exact original video frame/)).toHaveLength(2)
    expect(screen.getByText(/PTS 12520 at 1\/1000/)).toBeTruthy()
  })

  it('reports the actual scanned prefix, clock breaks and uncovered tail for the extended hold check', () => {
    const set = structuredClone(video)
    set.version = 'video-evidence-10'
    set.coverage!.frame_count = 900
    set.picture_hold = { status: 'no_match', frames_scanned: 751, frame_limit: 751, last_frame_index: 750,
      last_stream_time_s: 30, last_pts: 30000, time_base: '1/1000', reason: null, packet_id: null,
      scan_scope: 'extended-native-prefix/1', continuity_breaks: 2 }
    show(set)
    expect(screen.getByText(/Checked 751 frames from the start through 30\.000 s on the source clock/)).toBeTruthy()
    expect(screen.getByText(/found 2 gaps or unverifiable transitions/)).toBeTruthy()
    expect(screen.getByText('149 later frames were outside this check.')).toBeTruthy()
    expect(screen.queryByRole('link', { name: 'Inspect the recorded hold' })).toBeNull()
  })

  it.each([undefined, 'source_pts'] as const)('shows exact crops and identifies the mouth-hold waveform clock (%s)', (timeAxis) => {
    const set = { ...structuredClone(video), version: 'video-evidence-17' }
    set.picture_hold = { status: 'no_match', frames_scanned: 50, frame_limit: 50, last_frame_index: 49,
      last_stream_time_s: 13.64, last_pts: 13640, time_base: '1/1000', reason: null, packet_id: null,
      mouth_local_hold: { status: 'matched', frames_scanned: 50, packet_id: 'VPH02', affects_overall_confidence: false } }
    const item = set.packets[0]
    item.id = 'VPH02'
    item.title = 'Mouth-region picture hold'
    item.claim_scope = 'sampled_window'
    item.mouth_local_hold = { frames: [8, 12], box: [110, 86, 24, 18] }
    item.artifacts.mouth_hold_waveform = { ...item.artifacts[item.frames[0].artifact_refs.original], time_axis: timeAxis }
    item.artifacts.mouth_hold_audio = { ...item.artifacts[item.frames[0].artifact_refs.original], media_type: 'audio/wav' }
    item.window = { frame_indices: [8, 9, 10, 11], start_pts: 12000, end_pts: 12160,
      time_base: '1/1000', start_stream_s: 12, end_stream_s: 12.16, start_playback_s: 2, end_playback_s: 2.16 }
    item.frames = (['before', 'during', 'after'] as const).map((interval_role, i) => {
      const frame_index = [7, 8, 12][i]
      const pts = 12000 + 40 * (frame_index - 8)
      return { ...item.frames[0], id: `VPH02.${i + 1}`, interval_role, score: null, is_peak: false,
        anchor: { frame_index, pts, time_base: '1/1000', stream_time_s: pts / 1000, playback_time_s: pts / 1000 - 10 } }
    })
    show(set)
    const coverage = within(screen.getByRole('region', { name: 'Mouth-region picture hold' }))
    expect(coverage.getByRole('link', { name: 'Inspect the measured mouth-region hold' }).getAttribute('href')).toBe('#media-evidence-VPH02')
    expect(coverage.getByText(/first 50 source frames.*no weight in overall AI confidence/)).toBeTruthy()
    const measured = within(screen.getByRole('article'))
    expect(measured.getByText('12.000 s–12.160 s')).toBeTruthy()
    expect(measured.getByText(/The end is exclusive\. Source frames 8–11/)).toBeTruthy()
    for (const role of ['before', 'during', 'after']) expect(measured.getByText(new RegExp(`${role} hold range`))).toBeTruthy()
    expect(measured.getAllByAltText(/exact original video frame/)).toHaveLength(3)
    expect(measured.getAllByAltText(/exact measured mouth region/)).toHaveLength(3)
    expect(measured.getByAltText('Waveform during the measured mouth-region hold')).toBeTruthy()
    expect(measured.getByText(timeAxis === 'source_pts'
      ? /same source clock as the pictures above/
      : /waveform axis starts from the decoded audio origin/)).toBeTruthy()
    expect(measured.queryByAltText(/source face crop for context/)).toBeNull()
    fireEvent.click(measured.getByRole('button', { name: 'Go to frame 8 in playback' }))
    expect(screen.getByText(/Selected frame 8 · source 12\.000 s · player 2\.000 s/)).toBeTruthy()
  })

  it.each([
    ['no_match', 'No qualifying mouth-region hold was found in the checked frames.'],
    ['no_stable_face', 'A stable mouth region could not be tracked in the checked frames.'],
    ['unavailable', 'A mouth-region hold could not be assessed in this video.'],
  ] as const)('explains mouth-region coverage with status %s and no evidence packet', (status, message) => {
    const set = { ...structuredClone(video), packets: [] }
    set.picture_hold = { status: 'no_match', frames_scanned: 50, frame_limit: 50, last_frame_index: 49,
      last_stream_time_s: 13.64, last_pts: 13640, time_base: '1/1000', reason: null, packet_id: null,
      mouth_local_hold: { status, frames_scanned: 50, packet_id: null, affects_overall_confidence: false } }
    show(set)
    const coverage = within(screen.getByRole('region', { name: 'Mouth-region picture hold' }))
    expect(coverage.getByText(message)).toBeTruthy()
    expect(coverage.getByText(/first 50 source frames/)).toBeTruthy()
    expect(coverage.queryByRole('link')).toBeNull()
    expect(screen.queryByRole('article')).toBeNull()
  })

  it('shows which mouth-region windows were measured within a longer HD video', () => {
    const set = { ...structuredClone(video), packets: [] }
    set.picture_hold = { status: 'no_match', frames_scanned: 240, frame_limit: 240, last_frame_index: 239,
      last_stream_time_s: 7.967, last_pts: 239000, time_base: '1/30000', reason: null, packet_id: null,
      mouth_local_hold: { status: 'no_match', frames_scanned: 120, packet_id: null, affects_overall_confidence: false,
        assessed_window_starts: [0, 60] } }
    show(set)
    const section = within(screen.getByRole('region', { name: 'Mouth-region picture hold' }))
    expect(section.getByText(/first 120 source frames/)).toBeTruthy()
    expect(section.getByText(/2 of 3 overlapping windows.*Source frames 0–59, 60–119/)).toBeTruthy()
    expect(section.queryByRole('link')).toBeNull()
  })

  it('shows two original time ranges and one identical recording in the video voice lane', () => {
    const set = { ...structuredClone(video), version: 'video-evidence-10' }
    const sample = (start_sample: number) => ({ start_sample, end_sample: start_sample + 4000,
      source_timestamp_ranges: [{ start_sample, end_sample: start_sample + 4000,
        source_start_pts: 176000 + start_sample, source_end_pts: 180000 + start_sample, time_base: '1/16000' }] })
    set.audio = { status: 'not_assessed', reason: 'Voice model skipped this short input.', packets: [], limitations: [],
      source: { ...set.source, pcm_sha256: 'a'.repeat(64), samples: 48000, sample_rate: 16000 },
      coverage: { decoded_samples: 48000, scored_samples: 0, unscored_samples: 48000, windows: [], decode_limit_reached: false, selection_threshold: .5,
        unscored_intervals: [{ start_sample: 0, end_sample: 48000, source_timestamp_ranges: [
          { start_sample: 0, end_sample: 48000, source_start_pts: 176000, source_end_pts: 224000, time_base: '1/16000' },
        ] }] },
      repetition_context: { schema: 'chipmunk.audio-exact-repetition/1', status: 'measured', reason: null, source_sha256: set.source.sha256,
        pcm_sha256: 'a'.repeat(64), sample_rate: 16000, input_samples: 48000, searched_interval: { start_sample: 0, end_sample: 48000 },
        pairs: [{ id: 'R01', first: sample(3200), second: sample(19200), sample_count: 4000, lag_samples: 16000,
          pcm_s16le_sha256: 'a'.repeat(64), artifacts: { first_clip: asset('9', 'audio/wav'), second_clip: asset('9', 'audio/wav') } }] } }
    render(<MemoryRouter initialEntries={['/#media-evidence-R01']}><MultimediaEvidenceContent set={set} token="signed token" mediaReady /></MemoryRouter>)
    expect(screen.getByRole('tab', { name: /Voice intervals/ }).getAttribute('aria-selected')).toBe('true')
    const first = screen.getByLabelText('R01 · First occurrence').getAttribute('src')
    expect(first).toContain(`/assets/${'9'.repeat(64)}.wav?token=signed%20token`)
    expect(screen.getByLabelText('R01 · Second occurrence').getAttribute('src')).toBe(first)
    expect(screen.getByText(/Video playback 1\.200 s–1\.450 s/)).toBeTruthy()
    expect(screen.getByText(/Video playback 2\.200 s–2\.450 s/)).toBeTruthy()
    expect(screen.getByText('Unscored voice intervals')).toBeTruthy()
    expect(screen.getByText(/Source PTS \[176000, 224000\).*Video playback 1\.000 s–4\.000 s/)).toBeTruthy()
  })

  it.each(['chipmunk.audio-repeat-similarity/1', 'chipmunk.audio-repeat-similarity/2'] as const)(
    'plays the two distinct originals for a similar-audio pair in %s', (schema) => {
    const set = { ...structuredClone(audio), packets: [], source: { ...audio.source, samples: 48000, pcm_sha256: 'a'.repeat(64) } }
    const span = (start_sample: number, sha: string) => ({ start_sample, end_sample: start_sample + 4000, pcm_s16le_sha256: sha.repeat(64),
      source_timestamp_ranges: [{ start_sample, end_sample: start_sample + 4000,
        source_start_pts: 16000 + start_sample, source_end_pts: 20000 + start_sample, time_base: '1/16000' }] })
    set.similarity_context = { schema, status: 'measured', reason: null,
      ...(schema.endsWith('/2') ? { frequency_band_search: { status: 'measured', reason: null } } : {}),
      source_sha256: set.source.sha256, pcm_sha256: set.source.pcm_sha256, sample_rate: 16000, input_samples: 48000,
      searched_interval: { start_sample: 0, end_sample: 48000 }, pairs: [{ id: 'RS01', first: span(1600, 'b'), second: span(17600, 'c'),
        sample_count: 4000, lag_samples: 16000, support_windows: 21,
        artifacts: { first_clip: asset('b', 'audio/wav'), second_clip: asset('c', 'audio/wav') } }] }
    show(set)
    expect(screen.getByRole('heading', { name: 'Similar repeated audio' })).toBeTruthy()
    expect(screen.getByText(schema.endsWith('/2') ? /compared waveform patterns/ : /across frequency bands/)).toBeTruthy()
    expect(screen.getByText(/Matches carry no AI confidence weight/)).toBeTruthy()
    expect(screen.getByLabelText('RS01 · First occurrence').getAttribute('src')).toContain(`${'b'.repeat(64)}.wav`)
    expect(screen.getByLabelText('RS01 · Second occurrence').getAttribute('src')).toContain(`${'c'.repeat(64)}.wav`)
    expect(screen.getByText(/Source PTS \[33600, 37600\) at 1\/16000/)).toBeTruthy()
    expect(screen.queryByText(/of identical samples/)).toBeNull()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it('shows exact repeated passages and original page locations without a writing score', () => {
    const set = { ...structuredClone(document), version: 'document-evidence-5', packets: [] }
    set.repeated_passages = { schema: 'document-repeated-passages/1', status: 'matched',
      coverage: { compared_blocks: 2, excluded_ocr_blocks: 1, excluded_quoted_or_signature_blocks: 0, omitted_groups: 0 },
      groups: [{ id: 'RP01', text: 'The same original words are printed on both pages.', text_sha256: 'a'.repeat(64), word_count: 31, total_occurrences: 2,
        occurrences: [{ start: 0, end: 200, anchor: { kind: 'pdf', page: 1, block: 0, bbox: [40, 48, 200, 96] } },
          { start: 202, end: 402, anchor: { kind: 'pdf', page: 2, block: 0 } }] }] }
    show(set)
    expect(screen.getByRole('heading', { name: 'Exact repeated passages' })).toBeTruthy()
    expect(screen.getByText(/Page 1, block 1 · Extracted characters \[0, 200\)/)).toBeTruthy()
    expect(screen.getByText(/Page 2, block 1 · Extracted characters \[202, 402\)/)).toBeTruthy()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it('shows local and missing coverage in a short email without a selected passage', () => {
    const set = { ...structuredClone(document), version: 'document-evidence-5', packets: [] }
    set.coverage!.local_passage_coverage = {
      schema: 'document-local-passage-coverage/1', count_unit: 'non_whitespace_unicode_codepoint', mapped_characters: 23,
      counts: { response_range: 7, context_only: 8, no_local_record: 8 }, omitted_ranges: 0,
      ranges: [
        { start: 0, end: 8, source_block: 1, mime_part: '1.1', quote_role: 'body', state: 'response_range', characters: 7 },
        { start: 9, end: 18, source_block: 1, mime_part: '1.1', quote_role: 'body', state: 'context_only', characters: 8 },
        { start: 20, end: 30, source_block: 2, mime_part: '1.1', quote_role: 'quoted', state: 'no_local_record', characters: 8 },
      ],
    }
    show(set)
    expect(screen.getByTestId('multimedia-evidence-empty')).toBeTruthy()
    const coverage = within(screen.getByRole('region', { name: 'Recorded passage coverage' }))
    fireEvent.click(coverage.getByText('Source ranges (3)'))
    expect(coverage.getByRole('row', { name: /\[0, 8\) Local response recorded/ })).toBeTruthy()
    expect(coverage.getByRole('row', { name: /\[9, 18\) Read as context only/ })).toBeTruthy()
    expect(coverage.getByRole('row', { name: /quoted \[20, 30\) No exact local record/ })).toBeTruthy()
  })

  it.each(['video-evidence-7', 'video-evidence-8', 'video-evidence-9'])('shows before/during/after review evidence in %s', (version) => {
    const set = structuredClone(video)
    set.version = version
    const item = set.packets[0]
    item.claim_scope = 'sampled_window'
    item.face_sequence = { start_stream_s: 12, end_stream_s: 13, response: .02,
      centers: { before: [1, 2], during: [3, 4, 5], after: [6, 7] }, frame_inputs: [] }
    item.frames = (['before', 'during', 'after'] as const).map((role, index) => ({ ...item.frames[0],
      id: `V01.${index+1}`, interval_role: role, score: null, is_peak: false }))
    item.measurements = version !== 'video-evidence-7' ? [
      { label: 'Median eye spacing before the range', value: 50, unit: '% of face-box width' },
      { label: 'Median eye spacing during the range', value: 16.67, unit: '% of face-box width' },
      { label: 'Median eye spacing after the range', value: 50, unit: '% of face-box width' },
    ] : []
    if (version === 'video-evidence-9') {
      item.measurements.push({ label: 'Scene outside the face brightness change from before', value: -2.42,
        unit: 'points on a 0–100 scale' })
    }
    show(set)
    expect(screen.getByText(/Three consecutive crops support each of 3 centers/)).toBeTruthy()
    expect(screen.getAllByText(/before flagged range/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/during flagged range/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/after flagged range/i).length).toBeGreaterThan(0)
    expect(screen.queryByText(/response 0\.700/)).toBeNull()
    expect(screen.queryByText(/peak sample/)).toBeNull()
    if (version !== 'video-evidence-7') {
      expect(screen.getByText('Median eye spacing during the range')).toBeTruthy()
      expect(screen.getByText(/16\.670? % of face-box width/)).toBeTruthy()
    } else {
      expect(screen.queryByText('Median eye spacing during the range')).toBeNull()
    }
    if (version === 'video-evidence-9') {
      expect(screen.getByText('Scene outside the face brightness change from before')).toBeTruthy()
    }
  })

  it('plays exact audio windows and context while showing sample coverage, source offsets and precision limits', () => {
    show(audio)
    expect(screen.getByLabelText('P01: exact scored window').getAttribute('src')).toContain(`/assets/${'f'.repeat(64)}.wav?token=signed%20token`)
    expect(screen.getByLabelText('P01: window with surrounding audio').getAttribute('src')).toContain(`${'1'.repeat(64)}.wav`)
    expect(screen.getByText(/samples \[32000, 96000\)/)).toBeTruthy()
    expect(screen.getByText(/normally four seconds/)).toBeTruthy()
    expect(screen.getByText(/6.000 s of the 10.000 s decoded/)).toBeTruthy()
    expect(screen.getByText(/12.000 s–16.000 s/)).toBeTruthy()
    expect(screen.getByText(/decoding limit was reached/)).toBeTruthy()
    expect(screen.getByRole('link', { name: 'F5' })).toBeTruthy()
    expect(screen.queryByText(/check.private/)).toBeNull()
  })

  it('seeks video with recorded player time while keeping the source clock and exact frame images visible', () => {
    show(video)
    const player = screen.getByLabelText('Original video evidence playback') as HTMLVideoElement
    Object.defineProperty(player, 'readyState', { value: 1 })
    fireEvent.click(screen.getByRole('button', { name: 'Go to frame 6 in playback' }))
    expect(player.currentTime).toBe(2)
    expect(screen.getByRole('status').textContent).toContain('source 12.000 s · player 2.000 s')
    expect(screen.getByAltText('V01.1: exact original video frame').getAttribute('src')).toContain(`${'d'.repeat(64)}.png`)
    expect(screen.getByText(/not an edit boundary or the exact aligned classifier input/)).toBeTruthy()
    expect(screen.getByText(/gaps between samples have not been cleared/)).toBeTruthy()
  })

  it('refuses navigation for an unknown playback clock and never invents per-frame window scores', () => {
    const set = structuredClone(video)
    set.packets[0].frames[0].anchor.playback_time_s = null
    set.packets[0].frames[0].score = null
    set.packets[0].frames[0].is_peak = false
    set.packets[0].claim_scope = 'sampled_window'
    set.packets[0].window = { frame_indices: [0, 3, 6], start_pts: 10000, end_pts: 12000, time_base: '1/1000', start_stream_s: 10, end_stream_s: 12, start_playback_s: null, end_playback_s: null }
    show(set)
    expect((screen.getByRole('button', { name: 'Go to frame 6 in playback' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText(/Representative frames do not have individual scores/)).toBeTruthy()
    expect(screen.getByText(/Input frame indices: 0, 3, 6/)).toBeTruthy()
    expect(screen.getByText(/Playback timing is not verified/)).toBeTruthy()
  })

  it('shows the interval clock and labels before, during and after images', () => {
    const set = structuredClone(video)
    const item = set.packets[0], frame = item.frames[0]
    item.claim_scope = 'sampled_window'
    const sample = { frame_index: 6, pts: 12000, score: .7 }
    item.face_interval = { start_stream_s: 12, end_stream_s: 13, lift: .28, samples: { before: [], during: [sample], after: [] } }
    item.frames = (['before', 'during', 'after'] as const).map((role, index) => ({ ...frame, id: `V01.${index+1}`, interval_role: role }))
    show(set)
    expect(screen.getByText('12.000 s–13.000 s')).toBeTruthy()
    for (const role of ['Before', 'During', 'After']) {
      expect(screen.getAllByText(new RegExp(`${role} flagged range`, 'i'))).toHaveLength(2)
    }
  })

  it('selects a visible video moment using its saved playback clock and shows its source inspection area', () => {
    show(video)
    const player = screen.getByLabelText('Original video evidence playback') as HTMLVideoElement
    Object.defineProperty(player, 'readyState', { value: 1 })
    fireEvent.click(screen.getByLabelText('Inspect V01.1 at source 12.000 s'))
    expect(player.currentTime).toBe(2)
    expect(screen.getByAltText('V01.1: selected frame with the inspection area marked').getAttribute('src')).toContain(`${'e'.repeat(64)}.png`)
    expect(screen.getByAltText('V01.1: selected original face crop').getAttribute('src')).toContain(`${'f'.repeat(64)}.png`)
    expect(screen.getByText(/outline locates the area inspected/)).toBeTruthy()
    expect(screen.getByText(/spaces between samples were not continuously assessed/)).toBeTruthy()
  })

  it('plays a silent derivative on its exact recorded clip clock and exposes omitted channels', () => {
    const set = structuredClone(video)
    set.coverage!.channels = { audio: { status: 'absent', reason: 'The source has no audio stream.', successful_check_count: 0 },
      av_sync: { status: 'evidence_unavailable', reason: 'No frozen synchronization windows are available.', successful_check_count: 1 } }
    set.packets[0].artifacts.context_clip = { ...asset('a'), media_type: 'video/mp4' }
    set.packets[0].clip = { status: 'ready', artifact_ref: 'context_clip', start_frame_index: 4, end_frame_index_exclusive: 8,
      source_pts: [11500, 11650, 12000, 12120], clip_pts: [0, 150, 500, 620], clip_time_base: '1/1000', start_stream_s: 11.5,
      last_stream_s: 12.12, start_playback_s: 1.5, sampled_frame_indices: [6],
      last_frame_duration: { basis: 'inferred_previous_source_interval', ticks: 120, time_base: '1/1000' }, limitations: [] }
    show(set)
    const player = screen.getByLabelText('P01: silent source-derived context clip') as HTMLVideoElement
    expect(player.getAttribute('src')).toContain(`${'a'.repeat(64)}.mp4?token=signed%20token`)
    expect(player.autoplay).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Inspect frame 6 in clip' }))
    expect(player.currentTime).toBe(0.5)
    expect(screen.getByRole('status').textContent).toContain('Frame 6 · clip 0.500 s')
    expect(screen.getByText(/final frame’s display duration is inferred/)).toBeTruthy()
    expect(screen.getByText(/Continuous playback does not establish continuous analysis coverage/)).toBeTruthy()
    expect(screen.getByText(/1 displayed evidence frame\(s\) occur in this clip/)).toBeTruthy()
    expect(screen.getByText(/No frozen synchronization windows/)).toBeTruthy()
    fireEvent.error(player)
    expect(screen.getByRole('alert').textContent).toContain('context clip could not be loaded')
  })

  it('shows a context export failure alongside the exact frames and waits for secure media access', () => {
    const set = structuredClone(video)
    set.packets[0].clip = { status: 'unavailable', reason: 'Source timestamps could not be preserved.' }
    show(set, '', false)
    expect(screen.getByText(/Context clip unavailable: Source timestamps/)).toBeTruthy()
    expect(globalThis.document.querySelectorAll('video[src], img[src]')).toHaveLength(0)
  })

  it('preserves exact text and all scored contexts, disagreement, quote ownership and source page crops', () => {
    show(document)
    expect(screen.getByText(document.packets[0].focus.text)).toBeTruthy()
    expect(globalThis.document.querySelector('script')).toBeNull()
    expect(screen.getByText('Exact scored context A.')).toBeTruthy()
    expect(screen.getByText('Different actual scored context B.')).toBeTruthy()
    expect(screen.getByText(/Disagreement is retained/)).toBeTruthy()
    expect(screen.getByText(/Do not attribute its wording to the email sender/)).toBeTruthy()
    expect(screen.getByText(/Pages without mapped text: 3/)).toBeTruthy()
    expect(screen.getByAltText('P01: page 2 excerpt highlight').getAttribute('src')).toContain(`${'f'.repeat(64)}.png`)
    expect(screen.getByText(/do not identify which words were generated/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Show original page crop' }))
    expect(screen.getByAltText('P01: page 2 original crop').getAttribute('src')).toContain(`${'e'.repeat(64)}.png`)
  })

  it('shows raw pattern ratios and their direction without an authorship confidence or cutoff', () => {
    const set = structuredClone(document)
    set.packets[0].limitations = ['Each check uses its own recorded operating threshold. Threshold direction is not a validated probability or proof of authorship.']
    set.packets[0].supporting_windows = [{ ...document.packets[0].supporting_windows[0], score: 1.2,
      score_kind: 'perplexity_ratio', threshold: 0.9, ai_when: 'lt', direction: 'opposes_ai',
      inference_contexts: [{ id: 'context-1', source_token_range: [0, 200], scored_token_range: [0, 200],
        spans: [{ start: 0, end: 30 }], text: 'Exact source context.', input_sha256: 'a'.repeat(64),
        token_ids_sha256: 'b'.repeat(64), anchor_token_count: 1, context_limit: 1024, stride: 768 }] }]
    show(set)
    expect(screen.getByText(/raw writing-pattern ratio/)).toBeTruthy()
    expect(screen.getByText('1.200')).toBeTruthy()
    expect(screen.getByText(/No AI-like flag for this input/)).toBeTruthy()
    expect(screen.getByText(/carries no passage-level authorship confidence/)).toBeTruthy()
    expect(screen.getByText(/Actual token context \[0, 200\)/)).toBeTruthy()
    expect(screen.queryByText(/Ratios below 0.9|recorded operating threshold|Disagreement is retained/)).toBeNull()
    expect(screen.queryByText(/120%/)).toBeNull()
  })

  it('distinguishes unavailable evidence from an empty prepared set without claiming a negative finding', () => {
    const { unmount } = show({ ...audio, status: 'not_generated', packets: [], id: null })
    expect(screen.getByRole('status').textContent).toContain('Unavailable evidence does not establish authenticity')
    expect(screen.queryAllByRole('img')).toHaveLength(0)
    unmount()
    show({ ...audio, packets: [] })
    expect(screen.getByTestId('multimedia-evidence-empty').textContent).toContain('does not establish authenticity')
    expect(screen.getByText(/6.000 s of the 10.000 s decoded/)).toBeTruthy()
  })

  it('does not claim OCR was used on text PDFs and labels rejected OCR pages as unchecked', () => {
    const set = structuredClone(document)
    set.coverage = { scored_windows: 2, ocr: { status: 'not_needed', pages: [], unchecked_pages: [] } }
    const { unmount } = show(set)
    expect(screen.queryByText(/Some text was recognized/)).toBeNull()
    unmount()
    set.coverage.ocr = { status: 'partial', unchecked_pages: [4], pages: [
      { page: 2, status: 'extracted', mean_recognition_confidence: 92 },
      { page: 4, status: 'unchecked', reason: 'Unsupported language; page not checked.', mean_recognition_confidence: 65 },
      { page: 5, status: 'unchecked', coverage_scope: 'image_region', reason: 'The image area could not be read.' },
    ] }
    set.packets[0].artifacts.page2_ocr_input = { ...asset('4'), page: 2, role: 'ocr_input' }
    show(set)
    expect(screen.getByText(/Page 4: not checked/)).toBeTruthy()
    expect(screen.getByText(/Page 5 image area: not checked/)).toBeTruthy()
    expect(screen.queryByText(/Page 5: not checked/)).toBeNull()
    expect(screen.getByText(/Pages without mapped text: 4/)).toBeTruthy()
    expect(screen.getByText(/92.0 out of 100; this is not an AI score/)).toBeTruthy()
    expect(screen.getByAltText('P01: page 2 exact frozen text-recognition input').getAttribute('src')).toContain(`${'4'.repeat(64)}.png`)
  })

  it('keeps assets on our origin, waits for media access, and recovers after token renewal', () => {
    expect(evidenceAssetUrl(audio, asset('d'), 'secret')).not.toContain('unrelated.invalid')
    expect(evidenceAssetUrl(audio, { ...asset('d'), sha256: '../escape' }, 'secret')).toBeNull()
    const { unmount } = show(audio, '', false)
    expect(globalThis.document.querySelectorAll('audio[src], img[src]')).toHaveLength(0)
    unmount()
    const { rerender } = show(audio)
    fireEvent.error(screen.getByLabelText('P01: exact scored window'))
    expect(screen.getByRole('alert').textContent).toContain('could not be loaded')
    rerender(<MemoryRouter><MultimediaEvidenceContent set={audio} token="renewed" mediaReady /></MemoryRouter>)
    expect(screen.getByLabelText('P01: exact scored window').getAttribute('src')).toContain('token=renewed')
    expect(screen.queryByRole('alert')).toBeNull()
  })
})
