// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import AudioSoundEvents from './AudioSoundEvents'
import type { AudioEvidenceSet } from './multimediaPackets'

afterEach(cleanup)
const fixture = (): AudioEvidenceSet => ({ schema: 'chipmunk.audio-evidence/1', version: 'audio-evidence-2', status: 'ready',
  id: 'set', analysis_id: 'recording', revision: 1, sha256: 'a'.repeat(64), original_result_hash: 'b'.repeat(64),
  source: { sha256: 'c'.repeat(64), pcm_sha256: 'd'.repeat(64), sample_rate: 16000, samples: 320000 },
  packets: [], omitted_windows: 0, limitations: [], sound_events: { schema: 'chipmunk.sound-events/1', status: 'ready',
    source: { sha256: 'c'.repeat(64), pcm_sha256: 'd'.repeat(64), sample_rate: 16000, samples: 320000 },
    method: { window_samples: 160000, hop_samples: 160000, description: 'Mixed track' }, limitations: [],
    windows: [0, 1].map((i) => ({ id: `C0${i + 1}`, start_sample: i * 160000, end_sample: (i + 1) * 160000,
      start_s: i * 10, end_s: (i + 1) * 10, labels: [{ label: i ? 'Music' : 'Speech', score: .8 }],
      source_timestamp_ranges: [{ start_sample: i * 160000, end_sample: (i + 1) * 160000,
        source_start_pts: 32000 + i * 160000, source_end_pts: 32000 + (i + 1) * 160000, time_base: '1/16000' }],
      artifacts: { selected_clip: { sha256: (i ? 'f' : 'e').repeat(64), media_type: 'audio/wav', url: 'https://untrusted.invalid/' },
        waveform: { sha256: (i ? 'a' : 'b').repeat(64), media_type: 'image/png', url: 'https://untrusted.invalid/' } },
    })) } })

it('deep-links and switches sound intervals, with source-bound media and no AI-confidence claim', () => {
  const pause = vi.fn(), value = fixture(), original = structuredClone(value)
  render(<MemoryRouter initialEntries={['/media-evidence#media-evidence-C02']}>
    <AudioSoundEvents set={value} token="secret" mediaReady pauseAudio={pause} />
  </MemoryRouter>)
  expect(screen.getByRole('article', { name: 'Sound categories C02' })).toBeTruthy()
  expect(screen.getByText(/Music \(score 0.800\)/)).toBeTruthy()
  expect(screen.getAllByText(/not AI confidence percentages/)).toHaveLength(2)
  expect(screen.getByLabelText('C02: exact sound-category interval').getAttribute('src')).toContain('/api/v2/analyses/recording/evidence-packets/set/assets/')
  fireEvent.click(screen.getByRole('button', { name: /C01/ }))
  expect(screen.getByRole('article', { name: 'Sound categories C01' })).toBeTruthy()
  expect(pause).toHaveBeenCalled()
  expect(value).toEqual(original)
})

it('does not display clips or waveforms that belong to a different source', () => {
  const value = fixture(); value.sound_events!.source.samples += 1
  render(<MemoryRouter><AudioSoundEvents set={value} token="" mediaReady pauseAudio={() => {}} /></MemoryRouter>)
  expect(screen.getByRole('alert')).toBeTruthy()
  expect(screen.queryByRole('img')).toBeNull()
  expect(screen.queryByLabelText('C01: exact sound-category interval')).toBeNull()
})
