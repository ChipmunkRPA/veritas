// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import AudioEvidenceContent from './AudioEvidenceContent'
import type { AudioEvidenceSet, EvidenceAsset } from './multimediaPackets'

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

it('plays only the committed comparison and anchors all four assets to its sample and source times', () => {
  const set = fixture()
  set.source.pcm_sha256 = 'd'.repeat(64)
  set.source.samples = 32000
  const original = set.stereo_context!
  const first = { ...original.windows[0], downmix: { mono_rms_dbfs: -18, attenuation_db: 6, strong_attenuation: false } }
  set.stereo_context = {
    ...original, schema: 'chipmunk.stereo-channel-context/2', mono_pcm_sha256: set.source.pcm_sha256,
    strong_downmix_attenuation_db: 12,
    windows: [first, { ...first, start_sample: 16000, end_sample: 32000, source_timestamp_ranges: [{
      start_sample: 16000, end_sample: 32000, source_start_pts: 48000, source_end_pts: 64000, time_base: '1/16000',
    }] }],
  }
  const clip = (hash: string): EvidenceAsset => ({
    sha256: hash.repeat(64), media_type: 'audio/wav', url: 'https://foreign.invalid', samples: 16000,
    channels: 1, sample_rate: 16000,
  })
  set.stereo_listening = { schema: 'chipmunk.stereo-listening/1', affects_overall_confidence: false, windows: [{
    stereo_window_index: 1, artifacts: { left: clip('1'), right: clip('2'), mono: clip('3'),
      waveform: { sha256: '4'.repeat(64), media_type: 'image/png', url: 'https://foreign.invalid', width: 960, height: 290 } },
  }] }
  const { rerender } = render(<MemoryRouter><AudioEvidenceContent set={set} token="signed token" mediaReady /></MemoryRouter>)
  const region = within(screen.getByRole('region', { name: 'Stereo channel comparison' }))
  const comparison = within(region.getByText('Compare stereo and analyzed mono').closest('details')!)
  expect(comparison.getAllByText(/Samples \[/)).toHaveLength(1)
  expect(comparison.getByText('Samples [16000, 32000) at 16000 Hz.')).toBeTruthy()
  expect(comparison.getByText('Source PTS [48000, 64000) at 1/16000.')).toBeTruthy()
  expect(comparison.getByText(/only the selected interval.*carry no weight in the AI assessment/)).toBeTruthy()
  for (const [label, hash] of [['Decoded left channel', '1'], ['Decoded right channel', '2'], ['Exact analyzed mono', '3']]) {
    expect(comparison.getByLabelText(label).getAttribute('src')).toBe(
      `/api/v2/analyses/stereo-audio/evidence-packets/set/assets/${hash.repeat(64)}.wav?token=signed%20token`,
    )
  }
  const image = comparison.getByAltText(/exact analyzed mono waveforms, 1\.000 s–2\.000 s/)
  expect(image.getAttribute('src')).toContain(`/${'4'.repeat(64)}.png?token=signed%20token`)
  expect(image.getAttribute('width')).toBe('960')
  expect(image.getAttribute('height')).toBe('290')
  const left = comparison.getByLabelText('Decoded left channel') as HTMLAudioElement
  Object.defineProperty(left, 'paused', { configurable: true, value: false })
  const stopLeft = vi.spyOn(left, 'pause').mockImplementation(() => {
    Object.defineProperty(left, 'paused', { configurable: true, value: true })
  })
  fireEvent.play(comparison.getByLabelText('Exact analyzed mono'))
  expect(stopLeft).toHaveBeenCalledOnce()

  const changed = { ...set, source: { ...set.source, pcm_sha256: 'e'.repeat(64) } }
  rerender(<MemoryRouter><AudioEvidenceContent set={changed} token="signed token" mediaReady /></MemoryRouter>)
  expect(region.queryByText('Compare stereo and analyzed mono')).toBeNull()
  expect(region.queryByLabelText('Exact analyzed mono')).toBeNull()
})

it('shows the short channel-silence interval and contextual playback on the corresponding video clock', () => {
  const set = fixture()
  const context = set.stereo_context!
  set.source.pcm_sha256 = 'd'.repeat(64)
  set.stereo_context = {
    ...context, schema: 'chipmunk.stereo-channel-context/2', mono_pcm_sha256: set.source.pcm_sha256,
    channel_silence: {
      schema: 'chipmunk.stereo-channel-silence/1', affects_overall_confidence: false,
      checked_samples: 16000, qualifying_intervals: 1,
      selected: [{
        start_sample: 4800, end_sample: 8000, silent_channel: 'right',
        flank_rms_dbfs: [-20, -19], minimum_other_rms_dbfs: -24,
        source_timestamp_ranges: [{
          start_sample: 4800, end_sample: 8000, source_start_pts: 36800, source_end_pts: 40000, time_base: '1/16000',
        }],
      }],
    },
  }
  const clip = (hash: string): EvidenceAsset => ({
    sha256: hash.repeat(64), media_type: 'audio/wav', url: '', samples: 6400, channels: 1, sample_rate: 16000,
  })
  set.stereo_listening = {
    schema: 'chipmunk.stereo-listening/1', affects_overall_confidence: false, windows: [],
    channel_silence: [{ artifacts: { left: clip('1'), right: clip('2'), mono: clip('3'),
      waveform: { sha256: '4'.repeat(64), media_type: 'image/png', url: '', width: 960, height: 290 } } }],
  }
  render(<MemoryRouter><AudioEvidenceContent set={set} token="signed token" mediaReady videoOriginS={1.5} /></MemoryRouter>)
  const item = within(screen.getByText(/channel silent, decoded audio/).closest('article')!)
  expect(item.getByText(/Samples \[4800, 8000\)/)).toBeTruthy()
  expect(item.getByText(/Video playback 0\.800 s–1\.000 s/)).toBeTruthy()
  expect(item.getByAltText(/right channel silence highlighted/)).toBeTruthy()
  expect(item.getByLabelText('Decoded right channel').getAttribute('src')).toContain(`/${'2'.repeat(64)}.wav?`)
  expect(screen.getByText(/carries no weight in the AI assessment/)).toBeTruthy()
})
