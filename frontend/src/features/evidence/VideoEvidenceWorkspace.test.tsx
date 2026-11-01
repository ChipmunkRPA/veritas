// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it } from 'vitest'
import VideoEvidenceWorkspace, { videoLaneForHash } from './VideoEvidenceWorkspace'
import type { VideoEvidenceSet } from './multimediaPackets'

const set: VideoEvidenceSet = { schema: 'chipmunk.video-evidence/1', version: 'video-evidence-3', status: 'ready', id: 's',
  analysis_id: 'analysis', revision: 1, sha256: 'a'.repeat(64), original_result_hash: 'b'.repeat(64), source: { sha256: 'c'.repeat(64) },
  packets: [], omitted_candidates: 0, limitations: [],
  audio: { status: 'absent', reason: 'No sound track in the source.', source: { sha256: 'c'.repeat(64) }, packets: [], limitations: [] },
  av: { status: 'unavailable', reason: 'No timing inputs retained.', source: { sha256: 'c'.repeat(64) }, packets: [], limitations: [] } }
const show = (hash = '') => render(<MemoryRouter initialEntries={[`/analyses/analysis/media-evidence?view=plain${hash}`]}>
  <VideoEvidenceWorkspace set={set} token="" mediaReady />
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
