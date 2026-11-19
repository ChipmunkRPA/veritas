// @vitest-environment jsdom
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import VideoEvidenceContent from './VideoEvidenceContent'
import type { EvidenceAsset, VideoEvidencePacket, VideoEvidenceSet } from './multimediaPackets'

function fixture(method: 'single-face-boundary/5' | 'single-face-boundary/6' = 'single-face-boundary/6'): VideoEvidenceSet {
  const native720p = method === 'single-face-boundary/6'
  const [width, height] = native720p ? [1280, 720] : [640, 360]
  const [ticks, rate] = native720p ? [1000, 30000] : [2002, 60000]
  const time_base = `1/${rate}`
  const artifacts: Record<string, EvidenceAsset> = {}
  const roles = ['before', 'transition', 'after'] as const
  const frames = roles.map((role, i) => {
    const index = [41, 42, 53][i]
    const context_region = { x: 210 + i * 4, y: 124, w: 208, h: 264 }
    const refs = { original: `${role}_original`, context: `${role}_context`, crop: `${role}_crop` }
    for (const [j, name] of Object.values(refs).entries()) {
      artifacts[name] = { media_type: 'image/png', sha256: (i * 3 + j + 1).toString(16).repeat(64),
        width: name === refs.crop ? context_region.w : width, height: name === refs.crop ? context_region.h : height,
        url: 'https://unrelated.invalid/image' }
    }
    return { id: `VFB01.${role}`, interval_role: role, score: null, is_peak: false, region_meaning: 'detected_face_context',
      context_region, artifact_refs: refs, anchor: { frame_index: index, pts: index * ticks, time_base,
        stream_time_s: index * ticks / rate, playback_time_s: index * ticks / rate } }
  })
  const packet: VideoEvidencePacket = {
    id: 'VFB01', title: 'Face appearance transition', claim_scope: 'sampled_window',
    observation: 'The face looked consistently different across this frame.', interpretation: 'The cause remains undetermined.',
    measurements: [], limitations: [], finding_ids: [], signal_refs: [], frames, artifacts,
    face_single_boundary: { method, boundary: 42, pts: 42 * ticks, time_base, time_s: 42 * ticks / rate,
      affects_overall_confidence: false, ...(native720p && { measurement_raster: { width: 640, height: 360, transform: 'rgb-box-2x2-half-up' } }) },
  }
  return { schema: 'chipmunk.video-evidence/1', id: 'face witnesses', version: 'video-evidence-3', analysis_id: 'video-a',
    status: 'ready', sha256: 'a'.repeat(64), original_result_hash: 'b'.repeat(64), revision: 1, limitations: [], omitted_candidates: 0,
    source: { sha256: 'c'.repeat(64), time_base, width, height, timeline_origin_s: 0 }, packets: [packet],
    coverage: { frame_count: 120, scored_frame_count: 0, decoded_frame_indices: [41, 42, 53], window_check_count: 0,
      window_frame_indices: [], table_truncated: false, unmapped_observations: 0,
      face_single_boundary_coverage: { method, status: 'localized', reason: null, frames_scanned: 120, measured_face_frames: 98 } },
  }
}

const show = (set: VideoEvidenceSet) => render(<MemoryRouter>
  <VideoEvidenceContent set={set} mediaReady token="verified token" />
</MemoryRouter>)

afterEach(cleanup)

describe('720p source face transition', () => {
  it('shows reduced measurement resolution and exact native witnesses with their source clock', () => {
    const set = fixture()
    show(set)
    const coverage = within(screen.getByRole('region', { name: 'Face transition check' }))
    expect(coverage.getByText(/98 usable face views.*30 frames per second/)).toBeTruthy()
    expect(coverage.getByText(/averaged 2 × 2 groups of source pixels to measure 640 × 360 views from 1280 × 720 frames/)).toBeTruthy()
    expect(coverage.getByText(/ordinary focus change can trigger/i)).toBeTruthy()
    expect(coverage.getByText(/zero AI confidence weight/)).toBeTruthy()
    expect(coverage.getByRole('link', { name: 'Inspect the transition' }).getAttribute('href')).toBe('#media-evidence-VFB01')

    const article = within(screen.getByRole('article'))
    expect(article.getByText(/Last frame before transition · Frame 41/)).toBeTruthy()
    expect(article.getByText(/First frame at transition · Frame 42/)).toBeTruthy()
    expect(article.getByText(/Later supporting frame · Frame 53/)).toBeTruthy()
    expect(article.getByText(/Source 1.400 s · PTS 42000 at 1\/30000/)).toBeTruthy()
    expect(article.getByText('Exact source face crop · x 214, y 124 · 208 × 264 pixels.')).toBeTruthy()

    const original = article.getByRole('img', { name: 'VFB01.transition: exact original video frame' })
    const crop = article.getByRole('img', { name: 'VFB01.transition: source face crop for context' })
    const path = '/api/v2/analyses/video-a/evidence-packets/face%20witnesses/assets/'
    expect(original.getAttribute('src')).toBe(`${path}${'4'.repeat(64)}.png?token=verified%20token`)
    expect([original.getAttribute('width'), original.getAttribute('height')]).toEqual(['1280', '720'])
    expect(crop.getAttribute('src')).toBe(`${path}${'6'.repeat(64)}.png?token=verified%20token`)
    expect([crop.getAttribute('width'), crop.getAttribute('height')]).toEqual(['208', '264'])
  })

  it('shows the 720p scope when the file cannot be checked and displays no transition', () => {
    const set = fixture()
    set.packets = []
    set.coverage!.face_single_boundary_coverage = { method: 'single-face-boundary/6', status: 'unavailable', reason: 'profile',
      frames_scanned: 0, measured_face_frames: 0 }
    show(set)
    const coverage = within(screen.getByRole('region', { name: 'Face transition check' }))
    expect(coverage.getByText(/four-second 1280 × 720 H.264 MP4 \(avc1\) at exactly 30 fps starting at source time zero/)).toBeTruthy()
    expect(coverage.queryByRole('link', { name: 'Inspect the transition' })).toBeNull()
    expect(coverage.queryByText(/averaged 2 × 2/)).toBeNull()
  })

  it('describes the sampled views on a supported clip with no localized transition', () => {
    const set = fixture()
    set.packets = []
    set.coverage!.face_single_boundary_coverage!.status = 'no_match'
    show(set)
    const coverage = within(screen.getByRole('region', { name: 'Face transition check' }))
    expect(coverage.getByText(/No qualifying face appearance transition was found/)).toBeTruthy()
    expect(coverage.getByText(/640 × 360 views from 1280 × 720 frames/)).toBeTruthy()
    expect(coverage.queryByText(/displayed frames and face crops/)).toBeNull()
    expect(coverage.queryByRole('link', { name: 'Inspect the transition' })).toBeNull()
  })

  it('keeps the historical 29.97 fps clock and original caption for native-size measurements', () => {
    show(fixture('single-face-boundary/5'))
    const coverage = within(screen.getByRole('region', { name: 'Face transition check' }))
    expect(coverage.getByText(/29.97 frames per second/)).toBeTruthy()
    expect(coverage.queryByText(/averaged 2 × 2/)).toBeNull()
    expect(within(screen.getByRole('article')).getAllByText('Exact crop of the face from this source frame.')).toHaveLength(3)
  })
})
