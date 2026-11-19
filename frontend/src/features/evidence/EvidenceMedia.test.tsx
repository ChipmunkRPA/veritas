// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import { EvidenceAudio } from './EvidenceMedia'
import type { AudioEvidenceSet, EvidenceAsset } from './multimediaPackets'

afterEach(cleanup)

const set: AudioEvidenceSet = {
  schema: 'chipmunk.audio-evidence/1', status: 'ready', id: 'set', version: 'audio-evidence-3',
  sha256: 'a'.repeat(64), original_result_hash: 'b'.repeat(64), analysis_id: 'audio', revision: 1,
  source: { sha256: 'c'.repeat(64), source_channels: 1, source_stream_index: 0, sample_rate: 16000, samples: 16000 },
  packets: [], omitted_windows: 0, limitations: [],
}
const artifact: EvidenceAsset = { sha256: 'd'.repeat(64), media_type: 'audio/wav', url: '', samples: 3345, sample_rate: 16000 }

it('describes a subsecond clip from its measured sample count', () => {
  const { container } = render(<EvidenceAudio set={set} artifact={artifact} token="" ready alt="Play selected interval" />)
  const length = screen.getByText('Clip length 209 ms')
  const player = container.querySelector('audio')
  expect(player?.getAttribute('aria-label')).toBe('Play selected interval')
  expect(player?.getAttribute('aria-describedby')).toBe(length.id)
})

it.each([16000, undefined])('uses the normal player for %s samples', samples => {
  const { container } = render(<EvidenceAudio set={set} artifact={{ ...artifact, samples }} token="" ready alt="Play selected interval" />)
  const player = container.querySelector('audio')
  expect(player).not.toBeNull()
  expect(player?.hasAttribute('aria-describedby')).toBe(false)
  expect(screen.queryByText(/Clip length/)).toBeNull()
})
