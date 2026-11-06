// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import { MultimediaEvidenceContent } from './MultimediaEvidenceCards'
import { evidenceAssetUrl, type AudioEvidenceSet, type DocumentEvidenceSet, type EvidenceAsset, type MultimediaEvidenceSet, type VideoEvidenceSet } from './multimediaPackets'

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
  it('shows a sequence review range with before/during/after frames and no invented frame confidence', () => {
    const set = structuredClone(video)
    set.version = 'video-evidence-7'
    const item = set.packets[0]
    item.claim_scope = 'sampled_window'
    item.face_sequence = { start_stream_s: 12, end_stream_s: 13, response: .02,
      centers: { before: [1, 2], during: [3, 4, 5], after: [6, 7] }, frame_inputs: [] }
    item.frames = (['before', 'during', 'after'] as const).map((role, index) => ({ ...item.frames[0],
      id: `V01.${index+1}`, interval_role: role, score: null, is_peak: false }))
    show(set)
    expect(screen.getByText(/Three consecutive crops support each of 3 centers/)).toBeTruthy()
    expect(screen.getAllByText(/before flagged range/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/during flagged range/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/after flagged range/i).length).toBeGreaterThan(0)
    expect(screen.queryByText(/response 0\.700/)).toBeNull()
    expect(screen.queryByText(/peak sample/)).toBeNull()
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

  it('shows raw pattern ratios with their actual threshold direction and exact token context record', () => {
    const set = structuredClone(document)
    set.packets[0].supporting_windows = [{ ...document.packets[0].supporting_windows[0], score: 1.2,
      score_kind: 'perplexity_ratio', threshold: 0.9, ai_when: 'lt', direction: 'opposes_ai',
      inference_contexts: [{ id: 'context-1', source_token_range: [0, 200], scored_token_range: [0, 200],
        spans: [{ start: 0, end: 30 }], text: 'Exact source context.', input_sha256: 'a'.repeat(64),
        token_ids_sha256: 'b'.repeat(64), anchor_token_count: 1, context_limit: 1024, stride: 768 }] }]
    show(set)
    expect(screen.getByText(/raw writing-pattern ratio/)).toBeTruthy()
    expect(screen.getByText(/does not support the AI flag/)).toBeTruthy()
    expect(screen.getByText(/Ratios below 0.9/)).toBeTruthy()
    expect(screen.getByText(/Actual token context \[0, 200\)/)).toBeTruthy()
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
    ] }
    set.packets[0].artifacts.page2_ocr_input = { ...asset('4'), page: 2, role: 'ocr_input' }
    show(set)
    expect(screen.getByText(/Page 4: not checked/)).toBeTruthy()
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
