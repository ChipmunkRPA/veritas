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
  it('shows checked and insufficient speech intervals on the original audio clock', () => {
    show({ ...audio, packets: [], two_second_review: {
      schema: 'chipmunk.extended-short-voice-review/5', title: 'Two-second speech windows',
      status: 'no_flag', reason: 'No elevated window was confirmed.', limitation: 'The review carries zero weight.',
      affects_overall_confidence: false, regions: [], excerpts: [],
      sample_coverage: { scope_samples: 160000, checked_samples: 32000, scheduled_windows: 9, checked_windows: 1,
        intervals: [
          { start_sample: 0, end_sample: 32000, status: 'checked', source_timestamp_ranges: [],
            native_source_range: { start_pts: 0, end_pts: 96000, time_base: '1/48000' } },
          { start_sample: 32000, end_sample: 160000, status: 'insufficient_speech', source_timestamp_ranges: [],
            native_source_range: { start_pts: 96000, end_pts: 480000, time_base: '1/48000' } },
        ] },
    } })
    const region = within(screen.getByRole('region', { name: 'Two-second speech windows' }))
    expect(region.getByText(/1 of 9 distinct windows checked/)).toBeTruthy()
    const table = within(region.getByRole('table', { name: 'Two-second speech coverage intervals' }))
    const [, checked, insufficient] = table.getAllByRole('row')
    expect(within(checked).getByText('Checked')).toBeTruthy()
    expect(within(insufficient).getByText('Insufficient detected speech')).toBeTruthy()
    expect(within(insufficient).getByText(/Original ticks \[96000, 480000\) at 1\/48000/)).toBeTruthy()
  })

  it.each(['timing', 'articulation', 'movement'] as const)('links speech and mouth ranges to exact crops and audio (%s)', (method) => {
    const previous = video.packets[0]
    const mismatch = method !== 'timing'
    const title = method === 'movement' ? 'Local sound and mouth movement mismatch'
      : method === 'articulation' ? 'Local speech and mouth mismatch' : 'Local speech and mouth timing change'
    const schema = { timing: 'chipmunk.video-lip-timing/2', articulation: 'chipmunk.video-lip-timing/3', movement: 'chipmunk.video-lip-timing/6' }[method]
    show({ ...video, lip_timing_review: { schema,
      status: 'matched', observation: title, affects_overall_confidence: false },
      packets: [{ ...previous, id: 'VLT01', title, claim_scope: 'sampled_window',
        observation: 'Audio source time 0.680–1.400 s. Corresponding picture time 0.520–1.240 s.',
        lip_timing_review: { audio_source_pts: [10880, 22400], picture_source_samples: [8320, 19840], change_samples: mismatch ? 0 : 2560,
          ...(method === 'articulation' && { articulation: { mismatched_events: 3, audio_sample: 15680, picture_frame: 6, picture_pts: 12000 } }),
          ...(method === 'movement' && { motion_sound_agreement: { agreement_units: [4200000, 1200000, 3500000] } }) },
        frames: (['before', 'during', 'after'] as const).map(role => ({ ...previous.frames[0], id: `VLT01.${role}`, interval_role: role, score: null, is_peak: false })),
        artifacts: { ...previous.artifacts, lip_timing_waveform: asset('2'), lip_timing_audio: asset('3', 'audio/wav') },
      }],
    })
    const coverage = within(screen.getByRole('region', { name: mismatch ? 'Speech and mouth comparison' : 'Speech and mouth timing trial' }))
    const link = coverage.getByRole('link', { name: 'Inspect source frames and waveform' })
    expect(link.getAttribute('href')).toBe('#media-evidence-VLT01')
    const evidence = within(screen.getByRole('article'))
    expect(evidence.getByText(/Audio source time 0.680–1.400 s.*picture time 0.520–1.240 s/)).toBeTruthy()
    expect(evidence.getAllByAltText(/exact measured mouth region/)).toHaveLength(3)
    expect(evidence.getByText(mismatch ? /during mismatch range/i : /during timing range/i)).toBeTruthy()
    expect(evidence.getByAltText(mismatch ? /Waveform during the speech and mouth mismatch/ : /Waveform on the audio source clock/)).toBeTruthy()
    expect(evidence.getByLabelText(mismatch ? 'Listen to the soundtrack during the speech and mouth mismatch' :
      'Listen to the soundtrack during the measured timing change').getAttribute('src')).toContain('.wav')
  })

  it('shows the unassessed shot between two speech and mouth findings', () => {
    show({ ...video, lip_timing_review: {
      schema: 'chipmunk.video-lip-timing/7', status: 'matched', affects_overall_confidence: false,
      observation: 'Compared speech and mouth movement in 2 of 3 detected shots. Found 2 local mismatches.',
      shot_coverage: [
        { frames: [0, 50], audio_samples: [0, 32000], observation: 'Local mismatch found', exhibit: 'VLT01' },
        { frames: [50, 65], audio_samples: [32000, 41600], observation: 'Shot length outside the checked range', exhibit: null },
        { frames: [65, 125], audio_samples: [41600, 80000], observation: 'Local mismatch found', exhibit: 'VLT02' },
      ],
    } })
    const region = within(screen.getByRole('region', { name: 'Speech and mouth comparison' }))
    const shots = region.getAllByRole('listitem')
    expect(shots).toHaveLength(3)
    expect(within(shots[1]).getByText(/2\.000 s–2\.600 s · Shot length outside the checked range/)).toBeTruthy()
    expect(within(shots[1]).queryByRole('link')).toBeNull()
    expect(region.getAllByRole('link').map(link => link.getAttribute('href'))).toEqual(['#media-evidence-VLT01', '#media-evidence-VLT02'])
  })

  it('links reduced speech activity to source frames and playback using the same sample range', () => {
    const previous = video.packets[0]
    const title = 'Reduced speech activity during visible mouth movement'
    show({ ...video, lip_timing_review: {
      schema: 'chipmunk.video-lip-timing/8', status: 'matched', observation: title, affects_overall_confidence: false,
    }, packets: [{ ...previous, id: 'VLT01', title, claim_scope: 'sampled_window',
      observation: 'Audio source time 0.640–1.024 s. Corresponding picture time 0.640–1.024 s.',
      measurements: [{ label: 'Visible mouth frames', value: 8, unit: 'frames' }],
      lip_timing_review: { audio_source_pts: [10240, 16384], picture_source_samples: [10240, 16384],
        speech_activity_gap: { activity_millionths: [[900000, 880000], [150000, 130000], [860000, 840000]],
          visible_mouth_frames: 8, margin_millionths: 710000 },
      },
      frames: (['before', 'during', 'after'] as const).map(role => ({ ...previous.frames[0], id: `VLT01.${role}`, interval_role: role })),
      artifacts: { ...previous.artifacts, lip_timing_waveform: asset('2'), lip_timing_audio: asset('3', 'audio/wav') },
    }] })
    const region = within(screen.getByRole('region', { name: 'Speech activity and visible mouth movement' }))
    expect(region.getByRole('link', { name: 'Inspect source frames and waveform' }).getAttribute('href')).toBe('#media-evidence-VLT01')
    const evidence = within(screen.getByRole('article'))
    expect(evidence.getByText(/mix of the original channels.*\[10240, 16384\) at 16 kHz/)).toBeTruthy()
    for (const role of ['before', 'during', 'after']) {
      expect(evidence.getByText(new RegExp(`${role} reduced speech activity range`, 'i'))).toBeTruthy()
    }
    expect(evidence.getAllByAltText(/exact measured mouth region/)).toHaveLength(3)
    expect(evidence.getByAltText('Waveform around the reduced speech activity interval')).toBeTruthy()
    expect(evidence.getByLabelText('Listen to the original channels mixed during reduced speech activity').getAttribute('src')).toContain('.wav')
  })

  it('shows the exact language evidence when all authorship checks were outside scope', () => {
    show({ ...document, packets: [], coverage: { writing_check_coverage: {
      schema: 'document-writing-check-coverage/3',
      checks: { passage_classifier: 'not_applicable' },
      reasons: { passage_classifier: 'mixed_language' },
      language_witness: {
        sentence: 3, range: { start: 16, end: 46, sha256: 'a'.repeat(64) },
        context: { start: 8, end: 60, sha256: 'b'.repeat(64) },
      },
    } } })
    const section = within(screen.getByRole('region', { name: 'Writing check coverage' }))
    expect(section.getByText('The extracted text contained English and another language.')).toBeTruthy()
    expect(section.getByText('Two overlapping samples in extracted sentence 3 indicated another language.')).toBeTruthy()
    expect(section.getByText(/Marked characters \[16, 46\) and supporting context \[8, 60\)/)).toBeTruthy()
  })

  it('distinguishes displayed passages from the full list of writing candidates', () => {
    show({ ...document, omitted_windows: 6, coverage: {
      scored_windows: 24,
      candidate_spans: 12,
      balanced_passage_selection: { schema: 'document-balanced-passage-selection/1',
        candidate_blocks: 4, displayed_blocks: 3, displayed_passages: 6 },
    } })
    expect(screen.getByText(/Showing 6 of 12 candidate passages across 3 of 4 source blocks with flags/)).toBeTruthy()
    expect(screen.getByText(/The selection reserves space for disagreements/)).toBeTruthy()
  })

  it('shows a skipped writing check on a result that has no highlighted passages', () => {
    show({ ...document, packets: [], coverage: { writing_check_coverage: {
      schema: 'document-writing-check-coverage/1',
      checks: { passage_classifier: 'completed', writing_pattern_ratio: 'skipped' },
    } } })
    const section = within(screen.getByRole('region', { name: 'Writing check coverage' }))
    expect(section.getByText('Passage classifier · Completed')).toBeTruthy()
    expect(section.getByText('Writing-pattern ratio · Skipped')).toBeTruthy()
  })

  it('explains why a writing check was skipped without labeling a passage', () => {
    show({ ...document, packets: [], coverage: { writing_check_coverage: {
      schema: 'document-writing-check-coverage/2',
      checks: { passage_classifier: 'completed', writing_pattern_ratio: 'skipped' },
      reasons: { writing_pattern_ratio: 'memory_budget' },
    } } })
    const section = within(screen.getByRole('region', { name: 'Writing check coverage' }))
    expect(section.getByText('Available analysis memory was too low to run this check.')).toBeTruthy()
    expect(screen.queryByRole('region', { name: 'Passages to inspect' })).toBeNull()
  })

  it('distinguishes measured text with no selected passage from missing local evidence', () => {
    const { rerender } = show({ ...document, packets: [], coverage: { scored_windows: 2, candidate_spans: 0 } })
    expect(screen.getByText(/Recorded 2 scored contexts.*AI editing may still be present/)).toBeTruthy()
    rerender(<MemoryRouter><MultimediaEvidenceContent set={{
      ...document, packets: [], coverage: { scored_windows: 0, candidate_spans: 0 },
    }} token="signed token" mediaReady /></MemoryRouter>)
    expect(screen.getByText('No local writing responses were recorded.')).toBeTruthy()
    expect(screen.queryByText(/The writing checks selected no passages/)).toBeNull()
  })

  it('explains that ordinary editing can produce a flagged writing response', () => {
    show(document)
    expect(screen.getByText(/Ordinary editing and polished writing can produce a similar response/)).toBeTruthy()
  })

  it('links mismatched DOI evidence to both original Word ranges and the local registry scope', () => {
    const set = { ...document, packets: [], source: { ...document.source, chars: 250 }, coverage: {
      writing_bibliography: { schema: 'chipmunk.writing-bibliography/1', heading: 'References', entry_count: 3,
        section_start: 40, section_end: 220, first_location: { kind: 'docx', body_index: 5 },
        last_location: { kind: 'docx', body_index: 7 } },
    } } satisfies DocumentEvidenceSet
    const review = {
      schema: 'document-doi-reference/1', affects_overall_confidence: false,
      scope: 'Compares visible DOI references against a frozen local registry.',
      limitation: 'This reference observation carries zero AI confidence weight.',
      snapshot: { provider: 'Crossref', as_of: '2026-11-18', sha256: 'e'.repeat(64), records: 364 },
      references: 3, consistent: 1, different_work: 1, unassessed: 1,
      locations: [{
        start: 190, end: 207, title_start: 104, title_end: 138, text: '10.1234/example.1',
        cited_title: 'A long title in the written citation',
        anchor: { kind: 'docx', body_path: [6] },
        location: { ranges: [[94, 111]] }, title_location: { ranges: [[8, 42]] },
        printed: { doi: '10.1234/example.1', title: 'Different work found by the printed DOI' },
        cited: { doi: '10.1234/example.2', title: 'Publication matched by title and author' },
      }],
    } satisfies NonNullable<DocumentEvidenceSet['doi_reference_consistency']>
    const { rerender } = show({ ...set, doi_reference_consistency: review })
    const doi = within(screen.getByRole('region', { name: 'DOI reference check' }))
    expect(doi.getByText(/3 bibliography entries.*1 matched.*1 pointed to a different work.*1 was unassessed/)).toBeTruthy()
    expect(doi.getByText('10.1234/example.1').tagName).toBe('MARK')
    expect(doi.getByText('Different work found by the printed DOI')).toBeTruthy()
    expect(doi.getByText('10.1234/example.2')).toBeTruthy()
    expect(doi.getByText('Word body path 6')).toBeTruthy()
    expect(doi.getByText('Printed DOI · Extracted characters [190, 207) · Original characters [94, 111)')).toBeTruthy()
    expect(doi.getByText('Cited title · Extracted characters [104, 138) · Original characters [8, 42)')).toBeTruthy()
    expect(doi.getByText(/Local Crossref snapshot dated 2026-11-18 with 364 registry records.*zero AI confidence weight/)).toBeTruthy()
    expect(screen.getByRole('link', { name: 'See the DOI reference check.' }).getAttribute('href')).toBe('#document-doi-review')
    expect(screen.queryByText(/Citation accuracy was not assessed/)).toBeNull()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
    for (const [kind, label] of [
      ['odt_paragraph', 'Decoded OpenDocument paragraph 7'],
      ['rtf_paragraph', 'Decoded Rich Text block 7'],
    ] as const) {
      rerender(<MemoryRouter><MultimediaEvidenceContent set={{ ...set, doi_reference_consistency: {
        ...review, locations: [{ ...review.locations[0], anchor: { kind, paragraph_index: 6 } }],
      } }} token="signed token" mediaReady /></MemoryRouter>)
      expect(screen.getByText(label)).toBeTruthy()
    }
    rerender(<MemoryRouter><MultimediaEvidenceContent set={{ ...set, doi_reference_consistency: {
      ...review, consistent: 0, different_work: 0, unassessed: 3, locations: [],
    } }} token="signed token" mediaReady /></MemoryRouter>)
    expect(screen.getByText(/0 matched.*0 pointed to a different work.*3 were unassessed/)).toBeTruthy()
    expect(screen.queryByText('10.1234/example.1')).toBeNull()
  })

  it('shows separate writing ranges and keeps an empty review inconclusive', () => {
    const coverage = (status: 'flagged' | 'no_ranges'): DocumentEvidenceSet['coverage'] => ({
      writing_passage_intervals: {
        schema: 'document-writing-passage-intervals/1', status, authorship_confidence_contribution: 0,
        ranges: status === 'flagged' ? [
          { start: 11, end: 23, source_start: 12, source_end: 24, preview: 'First region' },
          { start: 41, end: 54, source_start: 45, source_end: 58, preview: 'Second region' },
        ] : [],
      },
    })
    const { rerender } = show({ ...document, packets: [], coverage: coverage('flagged') })
    const section = within(screen.getByRole('region', { name: 'Possible AI-like writing passages' }))
    expect(section.getByText('[12, 24)')).toBeTruthy()
    expect(section.getByText('[45, 58)')).toBeTruthy()
    expect(section.getByText('Second region')).toBeTruthy()
    rerender(<MemoryRouter><MultimediaEvidenceContent set={{
      ...document, packets: [], coverage: coverage('no_ranges'),
    }} token="signed token" mediaReady /></MemoryRouter>)
    expect(section.getByText(/Authorship remains undetermined/)).toBeTruthy()
    expect(section.queryByText('Second region')).toBeNull()
  })

  it('shows original PDF page numbers for a separate writing passage', () => {
    show({ ...document, source: { ...document.source, method: 'pymupdf' }, packets: [], coverage: {
      writing_passage_intervals: {
        schema: 'document-writing-passage-intervals/2', status: 'flagged', authorship_confidence_contribution: 0,
        ranges: [{ start: 0, end: 11, preview: 'Across pages', locations: [{ page: 1, boxes: [] }, { page: 2, boxes: [] }] }],
      },
    } })
    const section = within(screen.getByRole('region', { name: 'Possible AI-like writing passages' }))
    expect(section.getByText(/Original PDF pages 1, 2/)).toBeTruthy()
    expect(section.getByText('Across pages')).toBeTruthy()
  })

  it('moves exact dated wording out of style highlights while preserving a different flagged passage', () => {
    show({ ...document, packets: [], source: { ...document.source, method: 'plain' }, coverage: {
      writing_passage_intervals: {
        schema: 'document-writing-passage-intervals/1', status: 'flagged', authorship_confidence_contribution: 0,
        ranges: [
          { start: 10, end: 28, source_start: 10, source_end: 28, preview: 'Published wording.' },
          { start: 31, end: 49, source_start: 31, source_end: 49, preview: 'Another paragraph.' },
        ],
      },
      published_wording_review: {
        schema: 'document-published-wording-review/1', authorship_confidence_contribution: 0, archive_sha256: 'b'.repeat(64),
        reference: { id: 'GAO-16-123', published_date: '2016-01-01', version_url: 'https://huggingface.co/',
          text: '😀 Published wording.', source_record_sha256: 'd'.repeat(64), transcript_sha256: 'e'.repeat(64) },
        submitted_range: [0, 54], archive_range: [0, 20],
        published_style_matches: [{ start: 10, end: 28, archive_range: [2, 20] }],
        changes: [{ start: 31, end: 49, submitted: 'Another paragraph.', archive_range: [2, 11],
          locations: [{ start: 31, end: 49, ranges: [[31, 49]] }] }],
      },
    } })
    const style = within(screen.getByRole('region', { name: 'Possible AI-like writing passages' }))
    expect(style.queryByText('Published wording.')).toBeNull()
    expect(style.getByText('Another paragraph.')).toBeTruthy()
    expect(style.getByRole('link', { name: 'source comparison' }).getAttribute('href')).toBe('#document-published-wording')
    const source = within(screen.getByRole('region', { name: 'Earlier published wording' }))
    expect(source.getByText('Published wording.')).toBeTruthy()
    expect(source.getByText('Another paragraph.')).toBeTruthy()
    expect(source.getAllByText('[31, 49)')).toHaveLength(2)
    expect(source.getByRole('link', { name: 'U.S. GAO report GAO-16-123' }).getAttribute('href')).toBe('https://www.gao.gov/products/GAO-16-123')
  })

  it('labels the paragraph positions of OpenDocument wording differences', () => {
    show({ ...document, packets: [], source: { ...document.source, method: 'odt' }, coverage: {
      published_wording_review: {
        schema: 'document-published-wording-review/3', authorship_confidence_contribution: 0, archive_sha256: 'b'.repeat(64),
        reference: { id: 'GAO-16-123', published_date: '2016-01-01', version_url: 'https://huggingface.co/',
          text: 'Monday', source_record_sha256: 'd'.repeat(64), transcript_sha256: 'e'.repeat(64) },
        submitted_range: [0, 39], archive_range: [0, 6], published_style_matches: [],
        changes: [{ start: 31, end: 39, submitted: 'Thursday', archive_range: [0, 6], locations: [
          { start: 31, end: 39, paragraph: 2, ranges: [[12, 20]], coordinate_space: 'extracted-paragraph-codepoints' },
        ] }],
      },
    } })
    const section = within(screen.getByRole('region', { name: 'Earlier published wording' }))
    expect(section.getByText(/within extracted OpenDocument body paragraphs/)).toBeTruthy()
    expect(section.getByText(/Extracted OpenDocument paragraph 2 characters/)).toBeTruthy()
    expect(section.getByText('[12, 20)')).toBeTruthy()
    expect(section.getByText('Thursday')).toBeTruthy()
    expect(section.getByText('Monday')).toBeTruthy()
  })

  it('shows the cited federal release, pinned version and original Word location', () => {
    const url = 'https://www.bea.gov/news/2018/personal-income-and-outlays-july-2018'
    show({ ...document, packets: [], source: { ...document.source, method: 'python-docx' }, coverage: {
      published_wording_review: {
        schema: 'document-published-wording-review/4', authorship_confidence_contribution: 0, archive_sha256: 'b'.repeat(64),
        reference: { id: 'c'.repeat(64), publisher: 'bea', title: 'Personal Income and Outlays, July 2018', source_url: url,
          published_date: '2018-08-30', text: 'Monday', source_record_sha256: 'd'.repeat(64), transcript_sha256: 'e'.repeat(64) },
        submitted_range: [0, 39], archive_range: [0, 6], published_style_matches: [],
        changes: [{ start: 31, end: 39, submitted: 'Thursday', archive_range: [0, 6], locations: [
          { start: 31, end: 39, paragraph: 2, ranges: [[12, 20]] },
        ] }],
      },
    } })
    const section = within(screen.getByRole('region', { name: 'Earlier published wording' }))
    expect(section.getByRole('link', { name: 'Personal Income and Outlays, July 2018' }).getAttribute('href')).toBe(url)
    expect(section.getByText(/contributes zero weight/)).toBeTruthy()
    expect(section.getByText(/Word paragraph 2 characters/)).toBeTruthy()
    expect(section.getByText('d'.repeat(64))).toBeTruthy()
    expect(section.queryByRole('link', { name: 'GovReport transcript' })).toBeNull()
  })

  it('explains a passage found by comparing writing within the file', () => {
    show({ ...document, packets: [], coverage: {
      writing_passage_intervals: {
        schema: 'document-writing-passage-intervals/3', status: 'flagged', authorship_confidence_contribution: 0,
        ranges: [{ start: 5, end: 18, source_start: 7, source_end: 20, preview: 'Local passage', basis: 'document_baseline' }],
      },
    } })
    const section = within(screen.getByRole('region', { name: 'Possible AI-like writing passages' }))
    expect(section.getByText('Selected by comparing this passage with the writing patterns elsewhere in this file.')).toBeTruthy()
    expect(section.getByText('[7, 20)')).toBeTruthy()
    expect(section.getByText(/carries zero weight in the overall assessment/)).toBeTruthy()
  })

  it('shows decoded HTML positions in the mapped reply', () => {
    show({ ...document, source: { ...document.source, method: 'email' }, packets: [], coverage: {
      email: { body_content_type: 'text/html' }, writing_passage_intervals: {
        schema: 'document-writing-passage-intervals/2', status: 'flagged', authorship_confidence_contribution: 0,
        ranges: [{ start: 0, end: 11, preview: 'Reply clause', locations: [{ mime_part: '1.2', ranges: [[43, 54]] }] }],
      },
    } })
    const section = within(screen.getByRole('region', { name: 'Possible AI-like writing passages' }))
    expect(section.getByText('This check covers the mapped reply body.')).toBeTruthy()
    expect(section.getByText(/Decoded email HTML body part 1\.2/)).toBeTruthy()
    expect(section.getByText('[43, 54)')).toBeTruthy()
  })

  it.each([
    ['odt', 'Decoded OpenDocument paragraph'],
    ['rtf', 'Decoded Rich Text block'],
  ])('shows a passage in the original decoded %s blocks', (method, label) => {
    show({ ...document, source: { ...document.source, method }, packets: [], coverage: {
      writing_passage_intervals: {
        schema: 'document-writing-passage-intervals/3', status: 'flagged', authorship_confidence_contribution: 0,
        ranges: [{ start: 12, end: 34, preview: 'A selected passage.',
          locations: [{ paragraph: 2, ranges: [[12, 17]] }, { paragraph: 4, ranges: [[0, 15]] }] }],
      },
    } })
    const section = within(screen.getByRole('region', { name: 'Possible AI-like writing passages' }))
    expect(section.getByText(`${label} 2 characters`, { exact: false })).toBeTruthy()
    expect(section.getByText(`${label} 4 characters`, { exact: false })).toBeTruthy()
    expect(section.getByText('[12, 17)')).toBeTruthy()
    expect(section.getByText('[0, 15)')).toBeTruthy()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it.each([
    [{ kind: 'docx', paragraph_index: 0, body_path: [0] }, /Word body paragraph 1/],
    [{ kind: 'email', mime_part: '1.2' }, /decoded email MIME part 1\.2/],
  ] as const)('shows the original sentence location for %o', (locator, label) => {
    const source_locator = locator.kind === 'docx' ? { ...locator, body_path: [0] as [number] } : locator
    show({ ...document, packets: [], coverage: { passage_position_review: {
      schema: 'document-passage-position-review/3', status: 'flagged',
      focus_text: 'The middle sentence showed a stronger local response.',
      authorship_confidence_contribution: 0, signal_ref: packet.signal_refs[0], source_locator,
      sentences: [
        { start: 0, end: 10, source_start: 0, source_end: 10 },
        { start: 11, end: 64, source_start: 11, source_end: 66 },
        { start: 65, end: 80, source_start: 67, source_end: 82 },
      ],
    } } })
    const section = within(screen.getByRole('region', { name: 'Three-sentence writing check' }))
    expect(section.getByText(label)).toBeTruthy()
    expect(section.getByText('[11, 66)')).toBeTruthy()
    expect(section.getByText('The middle sentence showed a stronger local response.')).toBeTruthy()
  })

  it('shows the missing center in a partial longer-passage check', () => {
    const context = (paragraph: number, base: number) => ({ paragraph, sentences: [0, 10, 20].map(offset => ({
      start: base + offset, end: base + offset + 8, source_start: base + offset + 2, source_end: base + offset + 10,
    })) })
    const reading = { input_sha256: 'a'.repeat(64), response_hex: '0x1.0000000000000p+0', focus_tokens: 12, neighbor_tokens: 20 }
    show({ ...document, packets: [], coverage: { passage_position_review: {
      schema: 'document-passage-position-review/4', status: 'partial', paragraphs: 3, sentence_count: 9,
      contexts: [{ ...context(0, 0), readings: [reading, reading, reading] }, context(1, 31)],
      selected: null, authorship_confidence_contribution: 0, signal_ref: packet.signal_refs[0],
    } } })
    const section = within(screen.getByRole('region', { name: 'Longer-passage writing check' }))
    expect(section.getByText(/Completed 1 of 2 eligible centers/)).toBeTruthy()
    expect(section.getByText('[43, 51)')).toBeTruthy()
    expect(section.queryByText(/Authorship remains undetermined/)).toBeNull()
  })

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

  it('shows the accepted paragraph range beside the original Word revision position', () => {
    show({ ...document, packets: [], coverage: { word_tracked_changes: {
      status: 'found', part: 'word/document.xml', paragraphs: 2, inserted: 1, deleted: 0,
      writing_input: 'accepted_inline_changes', scope: 'Checks inline changes.', limitation: 'Authorship remains undetermined.',
      unsupported_marks: 0, omitted_excerpts: 0, authorship_confidence_contribution: 0,
      rows: [{ kind: 'inserted', paragraph: 2, body_index: 1, inline_index: 2, text: 'Insert 😀',
        characters: 8, writing_input_range: [17, 25] }],
    } } })
    const section = within(screen.getByRole('region', { name: 'Word tracked text changes' }))
    expect(section.getByText(/insertions included and deletions excluded/)).toBeTruthy()
    expect(section.getByText(/paragraph child index 2/)).toBeTruthy()
    expect(section.getByText('Accepted paragraph characters [17, 25).')).toBeTruthy()
  })

  it('shows tracked paragraph breaks even when no passage is highlighted', () => {
    show({ ...document, packets: [], coverage: { word_tracked_changes: {
      status: 'found', part: 'word/document.xml', paragraphs: 3, inserted: 0, deleted: 0,
      writing_input: 'accepted_paragraph_changes', scope: 'Checks paragraph breaks.', limitation: 'Authorship remains undetermined.',
      unsupported_marks: 0, omitted_excerpts: 0, authorship_confidence_contribution: 0,
      rows: [], paragraph_marks: [{ kind: 'deleted', paragraph: 2, body_index: 1, child_path: [0, 0, 0] }],
    } } })
    const section = within(screen.getByRole('region', { name: 'Word tracked text changes' }))
    expect(section.getByText(/accepted wording and paragraph breaks/)).toBeTruthy()
    expect(section.getByText('Deleted paragraph break · Body paragraph 2')).toBeTruthy()
    expect(section.getByText(/XML body index 1, child path 0 \/ 0 \/ 0/)).toBeTruthy()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it('explains why writing was skipped when the unsupported revision has no excerpt', () => {
    show({ ...document, packets: [], coverage: { word_tracked_changes: {
      status: 'partial', part: 'word/document.xml', paragraphs: 3, inserted: 0, deleted: 0,
      writing_input: 'unsupported_tracked_changes', scope: 'Checks paragraph breaks.', limitation: 'Authorship remains undetermined.',
      unsupported_marks: 1, omitted_excerpts: 0, authorship_confidence_contribution: 0,
      rows: [], paragraph_marks: [],
    } } })
    expect(screen.getByText(/Automated writing checks were skipped/)).toBeTruthy()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it('shows both chat markers with their original character positions', () => {
    show({ ...document, packets: [], coverage: { chat_format_markers: {
      schema: 'document-chat-format-markers/1', scope: 'Maps literal source text.',
      limitation: 'Anyone can type or copy these markers.', checked_chars: 80, turns: 1,
      authorship_confidence_contribution: 0, locations: [{
        opener: { literal: '<|im_start|>assistant', start: 8, end: 29, ranges: [[12, 41]], anchor: { kind: 'email', mime_part: '1.1' } },
        closer: { literal: '<|im_end|>', start: 48, end: 58, ranges: [[62, 80]], anchor: { kind: 'email', mime_part: '1.1' } },
      }],
    } } })
    const section = within(screen.getByRole('region', { name: 'Chat formatting markers' }))
    expect(section.getByText('<|im_start|>assistant').tagName).toBe('CODE')
    expect(section.getByText('<|im_end|>')).toBeTruthy()
    expect(section.getByText(/Decoded email part 1.1 · characters \[12, 41\)/)).toBeTruthy()
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
    expect(review.queryByText('Both recorded channels contain the same samples.')).toBeNull()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
    cleanup()
    show({ ...set, source: { ...set.source, source_channels: 2, identical_stereo_channel_pcm_sha256: '2'.repeat(64) } })
    expect(screen.getByText('Both recorded channels contain the same samples.')).toBeTruthy()
  })

  it('shows timing-sensitive short speech evidence with the original alert and nearby source positions', () => {
    const set = { ...structuredClone(audio), packets: [], source: { ...audio.source, source_sample_rate: 48000 } }
    const interval = { start_sample: 64000, end_sample: 96000 }
    const shifted = { start_sample: 62720, end_sample: 94720 }
    set.short_voice_review = {
      schema: 'chipmunk.short-voice-review/3', status: 'flagged', title: 'Short speech-window review',
      reason: 'A two-second speech window returned an elevated response for review.',
      limitation: 'It carries no weight in the overall assessment.', affects_overall_confidence: false,
      checked_windows: [interval], interval, start_s: 4, end_s: 6,
      source_timestamp_ranges: [{ ...interval, source_start_pts: 64000, source_end_pts: 96000, time_base: '1/16000' }],
      artifacts: { selected_clip: asset('f', 'audio/wav') },
      window_coverage: [{ ...interval, speech_samples: 24000, status: 'checked', source_timestamp_ranges: [
        { ...interval, source_start_pts: 64000, source_end_pts: 96000, time_base: '1/16000' },
      ] }],
      nearby_window_review: { status: 'varied', windows: [{ ...shifted, speech_samples: 22720, status: 'below_review_point',
        pcm_sha256: '7'.repeat(64), source_timestamp_ranges: [
          { ...shifted, source_start_pts: 62720, source_end_pts: 94720, time_base: '1/16000' },
        ] }] },
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Short speech-window review' }))
    expect(review.getByText('This check uses 16 kHz mono audio decoded from the 48 kHz source.')).toBeTruthy()
    expect(review.getByText(/Small timing changes affected this finding/)).toBeTruthy()
    expect(review.getByText('3.920 s–5.920 s')).toBeTruthy()
    expect(review.getByText('Source PTS [62720, 94720) at 1/16000.')).toBeTruthy()
    expect(review.getByLabelText('Flagged two-second speech window')).toBeTruthy()
  })

  it('shows measured short-window coverage when the speech review has no eligible window', () => {
    const set = { ...structuredClone(audio), packets: [] }
    set.short_voice_review = {
      schema: 'chipmunk.short-voice-review/2', status: 'skipped', title: 'Short speech-window review',
      reason: 'No two-second interval contained enough detected speech.', limitation: 'For review.', affects_overall_confidence: false,
      checked_windows: [], interval: null, start_s: null, end_s: null, source_timestamp_ranges: [], artifacts: {},
      window_coverage: Array.from({ length: 5 }, (_, i) => {
        const interval = { start_sample: i * 16000, end_sample: (i + 2) * 16000 }
        return { ...interval, speech_samples: i === 2 ? 8000 : 0, status: 'below_speech_minimum',
          source_timestamp_ranges: [{ ...interval, source_start_pts: 1105 + interval.start_sample,
            source_end_pts: 1105 + interval.end_sample, time_base: '1/16000' }] }
      }),
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Short speech-window review' }))
    expect(review.getByText('Two-second window coverage')).toBeTruthy()
    expect(review.getByText(/2\.000 s–4\.000 s · Below the speech minimum/)).toBeTruthy()
    expect(review.getByText('Detected speech 0.500 s.')).toBeTruthy()
    expect(review.getByText('Source PTS [33105, 65105) at 1/16000.')).toBeTruthy()
    expect(review.queryByLabelText('Flagged two-second speech window')).toBeNull()
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
    expect(review.getByText(/Loud background sound can make genuine speech look AI-generated in this check/)).toBeTruthy()
    expect(review.getByText(/cannot identify which speaker drove the result/)).toBeTruthy()
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

  it.each(['chipmunk.long-voice-review/2', 'chipmunk.long-voice-review/3'] as const)(
    'shows selected speech playback and supporting windows for %s', (schema) => {
    const set = { ...structuredClone(audio), packets: [] }
    const timestamps = (start_sample: number, end_sample: number) => [{ start_sample, end_sample, source_start_pts: start_sample,
      source_end_pts: end_sample, time_base: '1/16000' }]
    set.long_voice_review = {
      schema, title: 'Long speech-window review', status: 'corroborated',
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
      sample_coverage: { samples_by_status: { directional: 0, inconclusive: 0, below_speech_minimum: 48000, outside_schedule: 0 },
        intervals: [{ start_sample: 0, end_sample: 48000, status: 'below_speech_minimum',
          source_timestamp_ranges: [{ start_sample: 0, end_sample: 48000, source_start_pts: 0, source_end_pts: 48000, time_base: '1/16000' }] }] },
      windows: [{ start_sample: 0, end_sample: 48000, pcm_sha256: 'a'.repeat(64), direction: 'unassessed', speech_samples: 16000, artifacts: {},
        source_timestamp_ranges: [{ start_sample: 0, end_sample: 48000, source_start_pts: 0, source_end_pts: 48000, time_base: '1/16000' }] }],
    }
    show(set)
    const review = within(screen.getByRole('region', { name: 'Long speech-window review' }))
    expect(review.getByRole('status').textContent).toBe('None of the sampled windows produced a usable speech response.')
    expect(review.getByText(/Detected speech 1\.000 s of 3\.000 s. Below the 1\.500 s speech minimum./)).toBeTruthy()
    expect(review.getByRole('row', { name: /0\.000 s–3\.000 s All covering windows were below the speech minimum/ })).toBeTruthy()
    expect(review.queryByRole('link')).toBeNull()
  })

  it.each([undefined, 'same-frame-face-samples/1'] as const)('shows six planned positions and three source examples (%s)', (schema) => {
    const set = structuredClone(video)
    const item = set.packets[0]
    const samples = [6, 12, 18, 24, 30, 36].map((frame_index, i) => ({ frame_index, pts: 12000 + i * 2000,
      responses_hex: ['0x1.0p-1', '0x1.0p+0'] as [string, string] }))
    item.title = 'Sampled faces to review'
    item.claim_scope = 'sampled_window'
    item.whole_face_review = { schema, flagged: true, time_base: '1/1000', samples }
    item.window = { frame_indices: samples.map((row) => row.frame_index), start_pts: 12000, end_pts: 22000,
      time_base: '1/1000', start_stream_s: 12, end_stream_s: 22, start_playback_s: 2, end_playback_s: 12 }
    item.frames = [samples[0], samples[3], samples[5]].map((row) => ({ ...item.frames[0], id: `V01.${row.frame_index}`,
      score: null, is_peak: false, anchor: { frame_index: row.frame_index, pts: row.pts, time_base: '1/1000',
        stream_time_s: row.pts / 1000, playback_time_s: row.pts / 1000 - 10 } }))
    show(set)
    expect(screen.getByRole('heading', { name: /Sampled faces to review/ })).toBeTruthy()
    expect(screen.getByText(schema ? /combined at each jointly measured face/ : /compared faces at six sampled positions/)).toBeTruthy()
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

  it.each([['video-evidence-21', 'face-short-return/9'], ['video-evidence-28', 'face-short-return/15'],
    ['video-evidence-29', 'face-short-return/16'], ['video-evidence-30', 'face-short-return/17'],
    ['video-evidence-31', 'face-short-return/18'], ['video-evidence-32', 'face-short-return/19'],
    ['video-evidence-33', 'face-short-return/20'], ['video-evidence-34', 'face-short-return/21'],
    ['video-evidence-35', 'face-short-return/22'], ['video-evidence-36', 'face-short-return/23'],
    ['video-evidence-37', 'face-short-return/24'], ['video-evidence-38', 'face-short-return/25'],
    ['video-evidence-39', 'face-short-return/26'], ['video-evidence-40', 'face-short-return/27']])(
    'distinguishes native motion inputs from face samples in %s', (version, method) => {
    const set = { ...structuredClone(video), version }
    const packet = set.packets[0]
    packet.face_return = { method, start_stream_s: 2.066667, end_stream_s: 2.266667,
      frame_inputs: Array.from({ length: 12 }, (_, i) => ({ frame_index: 59 + i })),
      scene_frame_inputs: Array.from({ length: 120 }, (_, frame_index) => ({ frame_index })),
      edges: { entry: { frame_indices: [59, 60, 61, 62, 63, 64] }, exit: { frame_indices: [65, 66, 67, 68, 69, 70] } } }
    packet.window = { frame_indices: [61, 65, 68], start_pts: 62000, end_pts: 68000, time_base: '1/30000',
      start_stream_s: 2.066667, end_stream_s: 2.266667, start_playback_s: 2.066667, end_playback_s: 2.266667 }
    show(set)
    fireEvent.click(screen.getByText('Exact window coverage'))
    expect(screen.getByText(/Nearby motion was reviewed in 120 native pictures. Face appearances were measured in 12 sampled pictures/)).toBeTruthy()
    expect(screen.getByText('2.067 s–2.267 s')).toBeTruthy()
    expect(screen.getByText(/with an exclusive end/)).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Source frames compared at each boundary' })).toBeTruthy()
    expect(screen.getByText('Start of change · frames 59, 60, 61, 62, 63, 64')).toBeTruthy()
    expect(screen.getByText('End of change · frames 65, 66, 67, 68, 69, 70')).toBeTruthy()
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

  it.each(['VPR03', 'VPR04'])('links %s reversed sequence to its source ranges and four endpoint frames', packetId => {
    const set = structuredClone(video)
    set.picture_hold = { status: 'no_match', frames_scanned: 180, frame_limit: 180, last_frame_index: 179,
      last_stream_time_s: 18.95, last_pts: 18950, time_base: '1/1000', reason: null, packet_id: null,
      sequence_review: { status: 'no_match' }, reverse_sequence_review: { status: packetId === 'VPR03' ? 'matched' : 'no_match' } }
    if (packetId === 'VPR04') {
      set.picture_hold.reverse_motion_sequence_review = { status: 'matched', frames_scanned: 180, reason: null }
    }
    const item = set.packets[0]
    item.id = packetId
    item.claim_scope = 'paired_frame_sequences'
    item.picture_sequence = { order: 'reverse', frame_count: 22, span_s: 1.05,
      first: { start_frame_index: 20, end_frame_index: 41, start_stream_s: 11, end_stream_s: 12.05 },
      second: { start_frame_index: 130, end_frame_index: 151, start_stream_s: 16.5, end_stream_s: 17.55 } }
    item.frames = [20, 41, 130, 151].map((frameIndex, index) => ({ ...item.frames[0], id: `${packetId}.${index + 1}`,
      score: null, is_peak: false, context_region: null, artifact_refs: { original: 'original', context: 'original' },
      anchor: { frame_index: frameIndex, pts: 10000 + frameIndex * 50, time_base: '1/1000',
        stream_time_s: 10 + frameIndex / 20, playback_time_s: frameIndex / 20 } }))
    const { container } = show(set)
    const link = screen.getByRole('link', { name: 'Inspect both source ranges and their endpoint frames' })
    expect(link.getAttribute('href')).toBe(`#media-evidence-${packetId}`)
    const packet = within(container.querySelector<HTMLElement>(`#media-evidence-${packetId}`)!)
    expect(packet.getByText('11.000 s–12.050 s')).toBeTruthy()
    expect(packet.getByText('16.500 s–17.550 s')).toBeTruthy()
    expect(packet.getAllByAltText(/exact original video frame/)).toHaveLength(4)
    fireEvent.click(packet.getByRole('button', { name: 'Go to frame 130 in playback' }))
    expect(screen.getByText(/Selected frame 130 · source 16\.500 s · player 6\.500 s/)).toBeTruthy()
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
    item.measurements = [{ label: 'Audio blocks with voice-like periodicity', value: '0 of 7', unit: '40 ms blocks' },
      { label: 'Mixed soundtrack level during the hold', value: 'Digital silence', unit: '' }]
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
    expect(measured.getByText('0 of 7 40 ms blocks')).toBeTruthy()
    expect(measured.getByText('Digital silence')).toBeTruthy()
    expect(measured.getByText(timeAxis === 'source_pts'
      ? /same source clock as the pictures above/
      : /waveform axis starts from the decoded audio origin/)).toBeTruthy()
    expect(measured.queryByAltText(/source face crop for context/)).toBeNull()
    fireEvent.click(measured.getByRole('button', { name: 'Go to frame 8 in playback' }))
    expect(screen.getByText(/Selected frame 8 · source 12\.000 s · player 2\.000 s/)).toBeTruthy()
  })

  it('links the measured eye region to source frames, crops, and the original clock', () => {
    const set = structuredClone(video)
    set.picture_hold = { status: 'no_match', frames_scanned: 60, frame_limit: 60, last_frame_index: 59,
      last_stream_time_s: 59 / 30, last_pts: 59, time_base: '1/30', reason: null, packet_id: null,
      eye_local_hold: { schema: 'chipmunk.video-eye-local-hold/1', status: 'matched', frames_scanned: 60,
        packet_id: 'VPH03', affects_overall_confidence: false } }
    const item = set.packets[0]
    item.id = 'VPH03'
    item.title = 'Eye-region picture hold'
    item.claim_scope = 'sampled_window'
    item.eye_local_hold = { frames: [16, 32], anchors: [15, 24, 32], box: [120, 96, 44, 28] }
    item.window = { frame_indices: Array.from({ length: 16 }, (_, i) => 16 + i), start_pts: 16, end_pts: 32,
      time_base: '1/30', start_stream_s: 16 / 30, end_stream_s: 32 / 30, start_playback_s: 16 / 30, end_playback_s: 32 / 30 }
    item.frames = (['before', 'during', 'after'] as const).map((interval_role, i) => {
      const frame_index = [15, 24, 32][i]
      return { ...item.frames[0], id: `VPH03.${i + 1}`, interval_role, score: null, is_peak: false,
        anchor: { frame_index, pts: frame_index, time_base: '1/30', stream_time_s: frame_index / 30, playback_time_s: frame_index / 30 } }
    })
    show(set)
    const overview = within(screen.getByRole('region', { name: 'Eye-region picture hold' }))
    expect(overview.getByRole('link', { name: 'Inspect the measured eye-region hold' }).getAttribute('href')).toBe('#media-evidence-VPH03')
    expect(overview.getByText(/Checked 60 verified source frames.*zero AI confidence weight/)).toBeTruthy()
    const detail = within(screen.getByRole('article'))
    expect(detail.getByText('0.533 s–1.067 s')).toBeTruthy()
    expect(detail.getByText(/The end is exclusive\. Source frames 16–31/)).toBeTruthy()
    expect(detail.getByText('Source PTS [16, 32) at 1/30. Measured crop: x 120, y 96, 44 × 28 pixels.')).toBeTruthy()
    for (const role of ['before', 'during', 'after']) expect(detail.getByText(new RegExp(`${role} hold range`))).toBeTruthy()
    expect(detail.getAllByAltText(/exact original video frame/)).toHaveLength(3)
    expect(detail.getAllByAltText(/exact measured eye region/)).toHaveLength(3)
    expect(detail.queryByAltText(/source face crop for context/)).toBeNull()
  })

  it.each([
    ['outside_scope', 0, 'This upload falls outside the supported video profile for this check.'],
    ['no_stable_face', 60, 'The check could not track a stable eye region in the checked frames.'],
    ['no_match', 60, 'No qualifying eye-region hold was found in the checked frames.'],
  ] as const)('states eye-region coverage for %s without offering a finding', (status, frames_scanned, message) => {
    const set = structuredClone(video)
    set.picture_hold = { status: 'no_match', frames_scanned: 60, frame_limit: 60, last_frame_index: 59,
      last_stream_time_s: 59 / 30, last_pts: 59, time_base: '1/30', reason: null, packet_id: null,
      eye_local_hold: { schema: 'chipmunk.video-eye-local-hold/1', status, frames_scanned, packet_id: null, affects_overall_confidence: false } }
    show(set)
    const overview = within(screen.getByRole('region', { name: 'Eye-region picture hold' }))
    expect(overview.getByText(message)).toBeTruthy()
    expect(overview.getByText(/480 × 360 H\.264 MP4 at 30 fps/)).toBeTruthy()
    expect(overview.queryByRole('link')).toBeNull()
    expect(Boolean(overview.queryByText(/Checked 60 verified source frames/))).toBe(frames_scanned > 0)
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

  it.each(['video-evidence-10', 'video-evidence-35', 'video-evidence-36'])('shows both original audio ranges in the video voice lane for %s', (version) => {
    const set = { ...structuredClone(video), version }
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
    fireEvent.error(screen.getByLabelText('Original video evidence playback'))
    expect(screen.getByRole('link', { name: 'Open the playable context clip' }).getAttribute('href')).toBe('#media-evidence-P01')
    expect((screen.getByRole('button', { name: 'Go to frame 6 in playback' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByAltText('V01.1: exact original video frame')).toBeTruthy()
    fireEvent.error(player)
    expect(screen.getByText(/The context clip could not be loaded/)).toBeTruthy()
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
      mean_negative_log_likelihood: 2.4, mean_cross_entropy: 2,
      inference_contexts: [{ id: 'context-1', source_token_range: [0, 200], scored_token_range: [0, 200],
        spans: [{ start: 0, end: 30 }], text: 'Exact source context.', input_sha256: 'a'.repeat(64),
        token_ids_sha256: 'b'.repeat(64), anchor_token_count: 1, context_limit: 1024, stride: 768 }] }]
    show(set)
    expect(screen.getByText(/raw writing-pattern ratio/)).toBeTruthy()
    expect(screen.getByText('1.200')).toBeTruthy()
    expect(screen.getByText(/No AI-like flag for this input/)).toBeTruthy()
    expect(screen.getByText(/carries no passage-level authorship confidence/)).toBeTruthy()
    expect(screen.getByText(/Mean surprise for the observed tokens/)).toBeTruthy()
    expect(screen.getByText('2.4000')).toBeTruthy()
    expect(screen.getByText('2.0000')).toBeTruthy()
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
