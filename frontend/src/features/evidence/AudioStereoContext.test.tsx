// @vitest-environment jsdom
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it } from 'vitest'
import AudioEvidenceContent from './AudioEvidenceContent'
import type { AudioEvidenceSet } from './multimediaPackets'

afterEach(cleanup)

function fixture(): AudioEvidenceSet {
  return {
    schema: 'chipmunk.audio-evidence/1', status: 'ready', id: 'set', version: 'audio-evidence-3',
    sha256: 'a'.repeat(64), original_result_hash: 'b'.repeat(64), analysis_id: 'stereo-audio', revision: 1,
    source: { sha256: 'c'.repeat(64), source_channels: 2, source_stream_index: 0, sample_rate: 16000, samples: 16000 },
    packets: [], omitted_windows: 0, limitations: [],
    stereo_context: {
      schema: 'chipmunk.stereo-channel-context/1', status: 'measured', source_sha256: 'c'.repeat(64),
      evidence_type: 'context', affects_overall_confidence: false,
      source_stream_index: 0, source_channel_count: 2, sample_rate: 16000, max_seconds: 30,
      quiet_floor_dbfs: -50, strong_level_gap_db: 12, silence_floor_dbfs: -120,
      windows: [{
        start_sample: 0, end_sample: 16000, status: 'strong_imbalance',
        left_rms_dbfs: -12, right_rms_dbfs: -36, level_gap_db: 24, louder_channel: 'left', correlation: 1,
        source_timestamp_ranges: [{
          start_sample: 0, end_sample: 16000, source_start_pts: 32000, source_end_pts: 48000, time_base: '1/16000',
        }],
      }],
    },
  }
}

function renderAudio(set: AudioEvidenceSet) {
  return render(<MemoryRouter><AudioEvidenceContent set={set} token="" mediaReady={false} /></MemoryRouter>)
}

it('shows source-bound channel differences inside standalone audio evidence', () => {
  renderAudio(fixture())
  const section = within(screen.getByRole('region', { name: 'Stereo channel comparison' }))
  expect(section.getByText(/1 window.*strong channel-level difference/)).toBeTruthy()
  expect(section.getByText('[32000, 48000) at 1/16000')).toBeTruthy()
  expect(section.getByText('24')).toBeTruthy()
  expect(section.getByText(/do not contribute to the AI assessment/)).toBeTruthy()
})

it('reports mono and suppresses mismatched or historical context', () => {
  const set = fixture()
  set.source.source_channels = 1
  set.stereo_context = { ...set.stereo_context!, source_channel_count: 1, status: 'mono', windows: [] }
  const { rerender } = renderAudio(set)
  expect(screen.getByText(/first audio stream is mono/)).toBeTruthy()
  const changed = { ...set, source: { ...set.source, sha256: 'different' } }
  rerender(<MemoryRouter><AudioEvidenceContent set={changed} token="" mediaReady={false} /></MemoryRouter>)
  expect(screen.getByRole('alert').textContent).toMatch(/do not match this analyzed audio input/)
  delete changed.stereo_context
  rerender(<MemoryRouter><AudioEvidenceContent set={changed} token="" mediaReady={false} /></MemoryRouter>)
  expect(screen.queryByRole('region', { name: 'Stereo channel comparison' })).toBeNull()
})


it('shows mono level loss at the matching source interval without marking it as AI evidence', () => {
  const set = fixture()
  set.source.pcm_sha256 = 'd'.repeat(64)
  const original = set.stereo_context!
  set.stereo_context = {
    ...original,
    schema: 'chipmunk.stereo-channel-context/2',
    silence_floor_dbfs: -140,
    mono_pcm_sha256: set.source.pcm_sha256,
    strong_downmix_attenuation_db: 12,
    windows: [{
      ...original.windows[0],
      status: 'measured',
      right_rms_dbfs: -12,
      level_gap_db: 0,
      louder_channel: null,
      correlation: -1,
      downmix: { mono_rms_dbfs: -140, attenuation_db: null, strong_attenuation: true },
    }],
  }
  const { rerender } = renderAudio(set)
  const section = within(screen.getByRole('region', { name: 'Stereo channel comparison' }))
  expect(section.getByText(/Strong level drop in the analyzed mono mix/)).toBeTruthy()
  expect(section.getByText('[32000, 48000) at 1/16000')).toBeTruthy()
  expect(section.getByText('Mono is digitally silent')).toBeTruthy()
  expect(section.getByText(/do not contribute to the AI assessment/)).toBeTruthy()

  const changed = { ...set, source: { ...set.source, pcm_sha256: 'e'.repeat(64) } }
  rerender(<MemoryRouter><AudioEvidenceContent set={changed} token="" mediaReady={false} /></MemoryRouter>)
  expect(section.getByRole('alert')).toBeTruthy()
  expect(section.queryByText('Mono is digitally silent')).toBeNull()
})
