// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
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
  expect(videoLaneForHash('#media-evidence-A02')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-C02')).toBe('voice')
  expect(videoLaneForHash('#media-evidence-sounds')).toBe('voice')
  expect(videoLaneForHash('#finding-F2')).toBeNull()
  show('#media-evidence-A02')
  expect(screen.getByRole('tab', { name: /Voice intervals/ }).getAttribute('aria-selected')).toBe('true')
  expect(screen.getByRole('region', { name: 'Voice evidence from this video' })).toBeTruthy()
})
it('supports arrow-key channel switching and moves focus with the selected tab', () => {
  show()
  fireEvent.keyDown(screen.getByRole('tab', { name: /Video frames/ }), { key: 'ArrowRight' })
  const voice = screen.getByRole('tab', { name: /Voice intervals/ })
  expect(voice.getAttribute('aria-selected')).toBe('true')
  expect(document.activeElement).toBe(voice)
})
it('marks the recorded picture-change tiles only on a separate copy of the during frame', () => {
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
    claim_scope: 'sampled_window', frames, scene_return: { schema: 'picture-change-return/1', start_stream_s: .1, end_stream_s: .2, tiles },
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
