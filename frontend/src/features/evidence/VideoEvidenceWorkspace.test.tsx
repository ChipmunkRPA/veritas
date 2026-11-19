// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it } from 'vitest'
import VideoEvidenceWorkspace, { videoLaneForHash } from './VideoEvidenceWorkspace'
import type { VideoEvidenceFrame, VideoEvidencePacket, VideoEvidenceSet } from './multimediaPackets'

const set: VideoEvidenceSet = { schema: 'chipmunk.video-evidence/1', version: 'video-evidence-3', status: 'ready', id: 's',
  analysis_id: 'analysis', revision: 1, sha256: 'a'.repeat(64), original_result_hash: 'b'.repeat(64), source: { sha256: 'c'.repeat(64) },
  packets: [], omitted_candidates: 0, limitations: [],
  audio: { status: 'absent', reason: 'No sound track in the source.', source: { sha256: 'c'.repeat(64) }, packets: [], limitations: [] },
  av: { status: 'unavailable', reason: 'No timing inputs retained.', source: { sha256: 'c'.repeat(64) }, packets: [], limitations: [] } }
const show = (hash = '', evidence = set) => render(<MemoryRouter initialEntries={[`/analyses/analysis/media-evidence?view=plain${hash}`]}>
  <VideoEvidenceWorkspace set={evidence} token="" mediaReady />
</MemoryRouter>)
afterEach(cleanup)
it('puts video, voice and timing behind visible channel selectors and mounts only the selected evidence', () => {
  show()
  expect(screen.getByRole('tab', { name: /Video frames/ }).getAttribute('aria-selected')).toBe('true')
  expect(screen.queryByRole('region', { name: 'Voice evidence from this video' })).toBeNull()
  fireEvent.click(screen.getByRole('tab', { name: /Voice intervals/ }))
  expect(screen.getByText(/No sound track in the source/)).toBeTruthy()
  expect(screen.queryByTestId('video-evidence-content')).toBeNull()
  fireEvent.click(screen.getByRole('tab', { name: /Audio\/picture timing/ }))
  expect(screen.getByText(/No timing inputs retained/)).toBeTruthy()
  expect(screen.queryByRole('region', { name: 'Voice evidence from this video' })).toBeNull()
  fireEvent.click(screen.getByRole('tab', { name: /All evidence/ }))
  expect(screen.getByRole('region', { name: 'Voice evidence from this video' })).toBeTruthy()
  expect(screen.getByTestId('video-evidence-content')).toBeTruthy()
})
it('opens the correct channel for a deep-linked audio or alignment packet', () => {
  expect(videoLaneForHash('#media-evidence-AV03')).toBe('timing')
  expect(videoLaneForHash('#media-evidence-VLT01')).toBe('timing')
  expect(videoLaneForHash('#media-evidence-VFB01')).toBe('visual')
  expect(videoLaneForHash('#media-evidence-VPR07')).toBe('visual')
  expect(videoLaneForHash('#media-evidence-VSC01')).toBe('visual')
  expect(videoLaneForHash('#media-evidence-A02')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-C02')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-sounds')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-difference-repetition')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-DR01')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-AR01')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-aligned-repetition')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-rate-repetition')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-SR01')).toBe('voice')
  expect(videoLaneForHash('#finding-F2')).toBeNull()
  show('#media-evidence-A02')
  expect(screen.getByRole('tab', { name: /Voice intervals/ }).getAttribute('aria-selected')).toBe('true')
  expect(screen.getByRole('region', { name: 'Voice evidence from this video' })).toBeTruthy()
})
it('exposes recorded local lip evidence in timing even when correlation windows were not assessed', () => {
  const roles = ['before', 'during', 'after'] as const
  const frameIndices = [24, 33, 41]
  const artifact = { sha256: 'd'.repeat(64), media_type: 'image/png' as const, width: 960, height: 280, url: '/evidence/frame.png' }
  const local: VideoEvidencePacket = { id: 'VLT01', title: 'Reduced speech activity during visible mouth movement',
    observation: 'The sound and the mouth were checked between 0.832 and 1.344 seconds.', interpretation: 'Review the local interval.',
    limitations: [], finding_ids: [], signal_refs: [], measurements: [], claim_scope: 'sampled_window',
    lip_timing_review: { audio_source_pts: [13312, 21504], picture_source_samples: [13312, 21504], change_samples: 0 },
    artifacts: { source: artifact, crop: { ...artifact, sha256: 'e'.repeat(64) }, lip_timing_waveform: { ...artifact, sha256: 'f'.repeat(64) },
      lip_timing_audio: { sha256: '1'.repeat(64), media_type: 'audio/wav', samples: 8192, sample_rate: 16000, url: '/evidence/sound.wav' } },
    frames: roles.map((role, i) => ({ id: `VLT01.${role}`, score: null, is_peak: false, interval_role: role,
      region_meaning: 'measured_mouth_region', context_region: { x: 20, y: 30, w: 40, h: 30 },
      artifact_refs: { original: 'source', context: 'source', crop: 'crop' },
      anchor: { frame_index: frameIndices[i], pts: frameIndices[i] * 1000, time_base: '1/30000',
        stream_time_s: frameIndices[i] / 30, playback_time_s: frameIndices[i] / 30 } })),
  }
  const visual: VideoEvidencePacket = { ...local, id: 'V01', title: 'A separate face check', lip_timing_review: undefined, frames: [], artifacts: {} }
  show('#media-evidence-VLT01', { ...set, packets: [visual, local], lip_timing_review: { schema: 'chipmunk.video-lip-timing/9',
    status: 'matched', observation: 'Found a local interval.', affects_overall_confidence: false } })
  const timing = screen.getByRole('tab', { name: /Audio\/picture timing.*1 selected window/ })
  expect(timing.getAttribute('aria-selected')).toBe('true')
  const card = within(screen.getByRole('article'))
  expect(card.getAllByRole('img', { name: /exact measured mouth region/ })).toHaveLength(3)
  expect(card.getByRole('img', { name: /Waveform on the audio source clock/ })).toBeTruthy()
  expect(card.getByLabelText('Listen to the soundtrack during the measured timing change')).toBeTruthy()
  expect(screen.getByRole('heading', { name: 'Speech activity and visible mouth movement' })).toBeTruthy()
  expect(screen.getByRole('heading', { name: 'Audio and picture correlation windows' })).toBeTruthy()
  const destination = document.getElementById('media-evidence-timing')
  expect(destination?.compareDocumentPosition(card.getByRole('heading'))).toBe(Node.DOCUMENT_POSITION_FOLLOWING)
  expect(screen.getByText(/No timing inputs retained/)).toBeTruthy()
  fireEvent.click(screen.getByRole('tab', { name: /Video frames/ }))
  expect(screen.queryByText('VLT01')).toBeNull()
  expect(screen.getByRole('heading', { name: /A separate face check/ })).toBeTruthy()
  fireEvent.click(screen.getByRole('tab', { name: /All evidence/ }))
  expect(screen.getAllByRole('heading', { name: /VLT01\s*Reduced speech activity/ })).toHaveLength(1)
})
it('retains local timing coverage when no qualifying interval was found', () => {
  show('#media-evidence-timing', { ...set, lip_timing_review: { schema: 'chipmunk.video-lip-timing/9', status: 'no_match',
    observation: 'No qualifying short decline in speech activity was measured.', affects_overall_confidence: false } })
  expect(screen.getByRole('tab', { name: /Audio\/picture timing.*No qualifying local interval/ })).toBeTruthy()
  expect(screen.getByText('No qualifying short decline in speech activity was measured.')).toBeTruthy()
  expect(screen.queryByRole('link', { name: 'Inspect source frames and waveform' })).toBeNull()
})
it('supports arrow-key channel switching and moves focus with the selected tab', () => {
  show()
  fireEvent.keyDown(screen.getByRole('tab', { name: /Video frames/ }), { key: 'ArrowRight' })
  const voice = screen.getByRole('tab', { name: /Voice intervals/ })
  expect(voice.getAttribute('aria-selected')).toBe('true')
  expect(document.activeElement).toBe(voice)
})
it.each(['picture-change-return/1', 'picture-change-return/2'] as const)(
  'marks recorded %s tiles only on a separate copy of the during frame', (schema) => {
  const roles = ['before', 'during', 'after'] as const
  const frames: VideoEvidenceFrame[] = roles.map((role, i) => ({
    id: `VSC01.${role}`, score: null, is_peak: false, interval_role: role, context_region: null, region_meaning: 'whole_frame_context',
    anchor: { frame_index: 30 + i * 4, pts: 1000 + i * 100, time_base: '1/3000', stream_time_s: i * .13, playback_time_s: i * .13 },
    artifact_refs: { original: role, context: role },
  }))
  const tiles = [0, 9, 10, 31, 59]
  const packet: VideoEvidencePacket = {
    id: 'VSC01', title: 'Short picture change and return', observation: 'A short picture change was measured.',
    interpretation: 'An ordinary edit can produce this pattern.', limitations: [], finding_ids: [], signal_refs: [], measurements: [],
    artifacts: Object.fromEntries(roles.map((role, i) => [role, { sha256: String(i + 1).repeat(64), media_type: 'image/png', width: 1280, height: 720,
      url: `/api/v2/analyses/analysis/evidence-assets/${role}` }])),
    claim_scope: 'sampled_window', frames, scene_return: { schema, start_stream_s: .1, end_stream_s: .2, tiles },
  }
  show('#media-evidence-VSC01', { ...set, packets: [packet] })
  const originals = screen.getAllByRole('img', { name: /exact original video frame/ })
  expect(originals).toHaveLength(3)
  const marked = screen.getByRole('img', { name: 'VSC01.during: source frame with picture-change support tiles' })
  expect(marked.getAttribute('src')).toBe(originals[1].getAttribute('src'))
  expect(originals[1].parentElement?.querySelector('svg')).toBeNull()
  const grid = marked.parentElement?.querySelector('svg')
  expect(grid?.getAttribute('viewBox')).toBe('0 0 100 60')
  expect(Array.from(grid?.querySelectorAll('rect[fill-opacity]') ?? [], rect =>
    ['x', 'y', 'width', 'height'].map(attr => Number(rect.getAttribute(attr))))).toEqual([
    [0, 0, 10, 10], [90, 0, 10, 10], [0, 10, 10, 10], [10, 30, 10, 10], [90, 50, 10, 10],
  ])
  expect(screen.getByText('The 5 outlined tiles show texture measurements that support the start and end times. Ordinary cuts can produce this pattern.')).toBeTruthy()
})
