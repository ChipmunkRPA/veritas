// @vitest-environment jsdom
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it } from 'vitest'
import VideoEvidenceContent from './VideoEvidenceContent'
import type { VideoEvidenceSet } from './multimediaPackets'

afterEach(cleanup)

function show(boundary_frames: number[]) {
  const withCut = boundary_frames.length > 0
  const set: VideoEvidenceSet = {
    schema: 'chipmunk.video-evidence/1', id: 'video-native25', version: 'video-evidence-3', analysis_id: 'local-video',
    status: 'ready', sha256: 'a'.repeat(64), original_result_hash: 'b'.repeat(64), revision: 1, limitations: [], omitted_candidates: 0,
    source: { sha256: 'c'.repeat(64), time_base: '1/90000', timeline_origin_s: 0, width: 360, height: 288 }, packets: [],
    lip_timing_review: {
      schema: 'chipmunk.video-lip-timing/12', status: withCut ? 'shot_change' : 'no_match', affects_overall_confidence: false,
      observation: withCut ? 'The speech and mouth comparison was withheld because the pictures showed an abrupt change.'
        : 'No qualifying local speech and mouth difference was found in the measured ranges.',
      shot_guard: { method: 'native25-persistent-shot/1', frame_count: 75, boundary_frames },
    },
  }
  render(<MemoryRouter><VideoEvidenceContent set={set} mediaReady token="fixture" /></MemoryRouter>)
  return within(screen.getByRole('region', { name: 'Speech and mouth comparison' }))
}

it('reports the source frame and time of an abrupt picture change without a timing finding', () => {
  const region = show([37])
  expect(region.getByText(/comparison was withheld/i)).toBeTruthy()
  expect(region.getByText(/1\.480 s \(frame 37\)/i)).toBeTruthy()
  expect(region.queryByRole('link', { name: /source frames and waveform/i })).toBeNull()
})

it('describes completed coverage and limitations on an assessed clip', () => {
  const region = show([])
  expect(region.getByText(/75 source frames.*small jumps.*zero AI confidence weight/i)).toBeTruthy()
  expect(region.getByText(/no qualifying local/i)).toBeTruthy()
})
