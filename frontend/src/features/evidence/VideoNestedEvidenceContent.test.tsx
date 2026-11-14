// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import { MultimediaEvidenceContent } from './MultimediaEvidenceCards'
import { videoAudioView } from './VideoNestedEvidenceContent'
import type { EvidenceAsset, VideoAVPacket, VideoEvidenceSet } from './multimediaPackets'

function fixture(): VideoEvidenceSet {
  let counter = 1
  const asset = (type: EvidenceAsset['media_type'] = 'image/png'): EvidenceAsset => ({
    sha256: (counter++).toString(16).padStart(64, '0'), media_type: type, url: 'https://foreign.invalid/token',
    width: 96, height: 64, samples: 16000, sample_rate: 16000, channels: 1,
  })
  const source = { sha256: 'c'.repeat(64), sample_rate: 16000, samples: 57600, source_channels: 1,
    timeline_time_base: '1/16000', timeline: [{ start_sample: 0, end_sample: 57600, source_pts: 86400 }] }
  const common = { title: 'Measured alignment example', observation: 'The selected audio lag is +200 ms.',
    interpretation: 'This is an informational measured example.', limitations: ['A high correlation does not establish authenticity.'],
    finding_ids: ['F5'], signal_refs: [{ id: 'check.private:track:0', display_id: 'S02' }] }
  const artifacts: Record<string, EvidenceAsset> = {
    selected_clip: asset('audio/wav'), context_clip: asset('audio/wav'), context_clip_video: asset('video/mp4'),
    lag_profile: asset(), alignment: asset(), waveform: asset(),
  }
  const inputs = Array.from({ length: 25 }, (_, n) => {
    const tick = 25 + n, frame_index = 35 + n
    artifacts[`mouth_${tick}`] = { ...asset(), width: 32, height: 24 }
    return { tick, frame_index, frame_pts: 5000 + frame_index * 40, source_audio_pts: 86400 + tick * 640,
      source_audio_time_base: '1/16000', decoded_rgb_sha256: 'd'.repeat(64), mouth_patch_sha256: 'e'.repeat(64),
      mouth_region: { x: 24, y: 26, w: 48, h: 32 }, mouth_value: .15 + n / 100, artifact_ref: `mouth_${tick}` }
  })
  const frames = [inputs[0], inputs[12], inputs[24]].map((input, index) => {
    artifacts[`frame_${index}`] = asset(); artifacts[`context_${index}`] = asset(); artifacts[`crop_${index}`] = asset()
    return { id: `AV01.${index + 1}`, av_tick: input.tick, score: null, is_peak: false, region_meaning: 'mouth_measurement_input',
      context_region: input.mouth_region, artifact_refs: { original: `frame_${index}`, context: `context_${index}`,
        crop: `crop_${index}`, measured_patch: input.artifact_ref },
      anchor: { frame_index: input.frame_index, pts: input.frame_pts, time_base: '1/1000',
        stream_time_s: input.frame_pts / 1000, playback_time_s: input.frame_pts / 1000 - 5 } }
  })
  const av: VideoAVPacket = { ...common, id: 'AV01', claim_scope: 'sampled_av_window',
    interval: { start_sample: 16000, end_sample: 32000 }, context_interval: { start_sample: 6400, end_sample: 41600 },
    anchor: { sample_rate: 16000, source_timestamp_ranges: [{ start_sample: 16000, end_sample: 32000,
      source_start_pts: 102400, source_end_pts: 118400, time_base: '1/16000' }] },
    window: { start_tick: 25, end_tick: 50, frame_indices: inputs.map((input) => input.frame_index), frame_inputs: inputs },
    measurement: { offset_ticks: 5, offset_ms: 200, peak_correlation: .95, low_correlation: false,
      baseline_offset_ticks: 5, collapse_threshold: null, minimum_trusted_peak: .4, reading: 'constant_offset',
      overlaps_correlation_collapse: false, overlaps_offset_step: false,
      lag_profile: Array.from({ length: 31 }, (_, index) => ({ offset_ticks: index - 15, correlation: index === 20 ? .95 : -.1 })),
      audio_curve: Array.from({ length: 55 }, (_, n) => ({ tick: n + 10, sample_start: (n + 10) * 640,
        sample_end: (n + 11) * 640, rms: .2, speech: true })) },
    frames, artifacts, measurements: [{ label: 'Selected audio lag', value: 200, unit: 'ms; positive is audio later' }],
    clip: { status: 'ready', artifact_ref: 'context_clip_video', start_frame_index: 0, end_frame_index_exclusive: 100,
      source_pts: Array.from({ length: 100 }, (_, n) => 5000 + n * 40), clip_pts: Array.from({ length: 100 }, (_, n) => n * 40),
      clip_time_base: '1/1000', start_stream_s: 5, last_stream_s: 8.96, start_playback_s: 0,
      sampled_frame_indices: inputs.map((input) => input.frame_index),
      last_frame_duration: { basis: 'inferred_previous_source_interval', ticks: 40, time_base: '1/1000' }, limitations: [] },
  }
  return { schema: 'chipmunk.video-evidence/1', version: 'video-evidence-3', status: 'ready', id: 'parent set',
    analysis_id: 'video-1', revision: 3, sha256: 'a'.repeat(64), original_result_hash: 'b'.repeat(64),
    source: { sha256: source.sha256, timeline_origin_s: 5, time_base: '1/1000' }, limitations: [], omitted_candidates: 0, packets: [],
    coverage: { frame_count: 100, scored_frame_count: 0, window_check_count: 0, window_frame_indices: [], decoded_frame_indices: inputs.map((r) => r.frame_index),
      unmapped_observations: 0, table_truncated: false, channels: { audio: { status: 'ready', reason: 'Retained voice windows.' }, av: { status: 'ready', reason: 'Retained timing windows.' } } },
    audio: { status: 'ready', reason: null, source, omitted_windows: 2, limitations: ['Unscored audio is not a negative finding.'],
      coverage: { decoded_samples: 57600, scored_samples: 57600, unscored_samples: 0, decode_limit_reached: false,
        selection_threshold: .5, windows: [{ start_sample: 0, end_sample: 57600, score: .8, signal_refs: common.signal_refs }] },
      packets: [{ ...common, id: 'A01', title: 'Voice inspection window', interval: { start_sample: 0, end_sample: 57600 },
        context_interval: { start_sample: 0, end_sample: 57600 }, start_s: 0, end_s: 3.6, observations: [],
        anchor: { sample_rate: 16000, source_timestamp_ranges: [{ start_sample: 0, end_sample: 57600,
          source_start_pts: 86400, source_end_pts: 144000, time_base: '1/16000' }] },
        artifacts: { selected_clip: asset('audio/wav'), context_clip: asset('audio/wav'), waveform: asset(), scores: asset() } }] },
    av: { status: 'ready', reason: null, source: { ...source, tick_rate: 25, samples_per_tick: 640, video_time_base: '1/1000' },
      coverage: { recorded_windows: 3, selected_windows: 1, omitted_windows: 2, selection: 'flagged first', unsupported_checks: [] },
      packets: [av], limitations: ['Misalignment is not proof of AI generation or a deepfake.'] },
  }
}
const show = (set: VideoEvidenceSet, token = 'signed token', mediaReady = true) => {
  const rendered = render(<MemoryRouter><MultimediaEvidenceContent set={set} token={token} mediaReady={mediaReady} /></MemoryRouter>)
  const all = screen.queryByRole('tab', { name: /All evidence/ })
  if (all) fireEvent.click(all)
  return rendered
}
afterEach(cleanup)

describe('video v3 nested source-bound evidence', () => {
  it.each(['not_assessed', 'unavailable'] as const)('shows source ranges and reasons when alignment is %s', (status) => {
    const set = fixture()
    const av = set.av!
    av.status = status
    av.packets = []
    av.coverage = { ...av.coverage!, recorded_windows: 0, selected_windows: 0, omitted_windows: 0,
      timing_windows: { windows: [{
        status: 'no_face_track', speech_ticks: 16,
        interval: { start_sample: 16000, end_sample: 32000, source_start_pts: 102400, source_end_pts: 118400, time_base: '1/16000' },
      }] },
    }
    show(set)
    const coverage = screen.getByRole('region', { name: 'Speech and mouth timing coverage' })
    expect(within(coverage).getByText(/0 of 1 scheduled one-second windows/)).toBeTruthy()
    expect(within(coverage).getByText('6.400 s–7.400 s')).toBeTruthy()
    expect(within(coverage).getByText(/No tracked face was available/)).toBeTruthy()
  })

  it('shares parent media authority and preserves differing source and clip clocks', () => {
    const set = fixture()
    Object.assign(set.audio!, { id: 'foreign-child', analysis_id: 'other-analysis', sha256: 'f'.repeat(64) })
    const view = videoAudioView(set, set.audio!)
    expect([view.id, view.analysis_id, view.sha256]).toEqual([set.id, set.analysis_id, set.sha256])
    show(set)
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
    expect(screen.getByText(/5.400 s–9.000 s/)).toBeTruthy()
    expect(screen.getByText(/Selected source interval: 6.400 s–7.400 s/)).toBeTruthy()
    expect(screen.getByText(/Analyzed audio samples \[16000, 32000\)/)).toBeTruthy()
    expect(screen.getByText(/1 of 3 recorded alignment windows/)).toBeTruthy()
    expect(screen.getByText(/These counts cover visual checks only/)).toBeTruthy()
    expect(screen.getByText(/2 additional voice window/)).toBeTruthy()
    for (const element of document.querySelectorAll('img[src], audio[src], video[src]')) {
      const url = element.getAttribute('src')!
      expect(url).toMatch(/^\/api\/v2\/analyses\/video-1\/evidence-packets\/parent%20set\/assets\/[a-f0-9]{64}\.(png|wav|mp4)\?token=signed%20token$/)
      expect(url).not.toContain('foreign')
    }
    const video = screen.getByLabelText('AV01: silent source-derived context clip') as HTMLVideoElement
    fireEvent.click(screen.getByRole('button', { name: 'Inspect frame 35 in clip' }))
    expect(video.currentTime).toBe(1.4) // Source 6.4s minus video origin 5s, not audio origin 5.4s.
    expect(video.muted).toBe(true)
    expect(video.autoplay).toBe(false)
    expect(screen.getByText(/Audio is not shifted to the selected lag/)).toBeTruthy()
    expect(screen.queryByText(/check.private|track:0|constant_offset/)).toBeNull()
  })

  it('retains alternatives and exact measured frame inputs without inventing AI scores or speaker identity', () => {
    show(fixture())
    expect(screen.getByText('All 31 measured lag candidates')).toBeTruthy()
    const tables = screen.getAllByRole('table', { hidden: true })
    expect(within(tables[0]).getAllByRole('row', { hidden: true })).toHaveLength(32)
    expect(screen.getByText('-600 ms')).toBeTruthy()
    expect(screen.getByText('+600 ms')).toBeTruthy()
    expect(screen.getByText('Selected maximum')).toBeTruthy()
    expect(screen.getByText('All 25 measured mouth inputs and exact clocks')).toBeTruthy()
    expect(screen.getByText('All 55 measured audio-energy blocks')).toBeTruthy()
    expect(screen.getAllByAltText(/measured mouth input at tick/)).toHaveLength(25)
    expect(screen.getByAltText('AV01.1: exact source mouth crop')).toBeTruthy()
    expect(screen.getByAltText('AV01.1: exact measured grayscale mouth patch')).toBeTruthy()
    expect(screen.getAllByText(/Representative frames have no individual AI score/)).toHaveLength(3)
    expect(screen.getByText(/does not estimate the probability of manipulation/)).toBeTruthy()
    expect(screen.getByText(/do not prove AI generation, editing, or who is speaking/)).toBeTruthy()
    expect(screen.getByText(/Informational measured example/)).toBeTruthy()
    expect(screen.queryByText(/95%/)).toBeNull()
  })

  it('links a flagged timing change to its source interval and warns when the measured match is weak', () => {
    const set = fixture()
    set.av!.packets[0].measurement.overlaps_offset_step = true
    set.av!.packets[0].measurement.peak_correlation = .39
    show(set)
    const index = within(screen.getByRole('navigation', { name: 'Flagged timing intervals' }))
    const link = index.getByRole('link', { name: 'AV01 · 6.400 s–7.400 s' })
    expect(link.getAttribute('href')).toBe('#media-evidence-AV01')
    expect(document.getElementById('media-evidence-AV01')).toBeTruthy()
    expect(screen.getByText(/This window cannot support a reliable timing estimate/)).toBeTruthy()
  })

  it('distinguishes missing audio and unsupported alignment from negative findings', () => {
    const set = fixture()
    set.audio = { status: 'absent', reason: 'The source has no audio stream.', source: set.audio!.source, packets: [], limitations: [] }
    set.av = { status: 'unavailable', reason: 'The audio clock has gaps.', source: set.av!.source, packets: [], limitations: [] }
    show(set)
    expect(screen.getByText(/The source has no audio stream/)).toBeTruthy()
    expect(screen.getByText(/The audio clock has gaps/)).toBeTruthy()
    expect(screen.getAllByText(/Missing measurements do not establish authenticity or synchronization/)).toHaveLength(2)
    expect(document.querySelectorAll('img[src], audio[src], video[src]')).toHaveLength(0)
  })

  it('retains decoded context, zero scored coverage, and source clocks when channels were not assessed', () => {
    const set = fixture()
    set.audio = { ...set.audio!, status: 'not_assessed', reason: 'No speech windows could be scored.', packets: [], omitted_windows: 0,
      overview: { waveform: set.audio!.packets[0].artifacts.waveform, scores: set.audio!.packets[0].artifacts.scores },
      coverage: { ...set.audio!.coverage!, scored_samples: 0, unscored_samples: 57600, windows: [] } }
    set.av = { ...set.av!, status: 'not_assessed', reason: 'No usable mouth track was recorded.', packets: [],
      coverage: { recorded_windows: 0, selected_windows: 0, omitted_windows: 0, selection: 'flagged first', unsupported_checks: [] } }
    show(set)
    const voice = within(screen.getByRole('region', { name: 'Voice evidence from this video' }))
    expect(voice.getByText(/No speech windows could be scored/)).toBeTruthy()
    expect(voice.getByText(/Voice checks cover 0.000 s of the 3.600 s decoded input/)).toBeTruthy()
    expect(voice.getByText(/0 scored windows/)).toBeTruthy()
    expect(voice.getByText(/does not establish a natural voice/)).toBeTruthy()
    expect(voice.getByText(/5.400 s–9.000 s/)).toBeTruthy()
    expect(voice.getByAltText('Waveform of the analyzed audio input')).toBeTruthy()
    expect(voice.getByAltText('Recorded voice-check window responses over the analyzed input')).toBeTruthy()
    const alignment = within(screen.getByRole('region', { name: 'Audio and picture timing evidence' }))
    expect(alignment.getByText(/No usable mouth track was recorded/)).toBeTruthy()
    expect(alignment.getByText(/0 of 0 recorded alignment windows are shown/)).toBeTruthy()
    expect(alignment.getByText(/No measured alignment window is available/)).toBeTruthy()
    expect(alignment.getByText(/5.400 s–9.000 s/)).toBeTruthy()
    expect(alignment.getByText(/decoded context, not scored coverage/)).toBeTruthy()
    expect(document.querySelectorAll('audio[src], video[src]')).toHaveLength(0)
    expect(document.querySelectorAll('img[src]')).toHaveLength(2)
  })

  it('does not expose assets or invent counts for unassessed channels without valid coverage', () => {
    const set = fixture()
    set.audio = { ...set.audio!, status: 'not_assessed', reason: 'No frozen coverage record.', coverage: null }
    set.av = { ...set.av!, status: 'not_assessed', reason: 'No valid timing coverage record.',
      coverage: { ...set.av!.coverage!, recorded_windows: Number.NaN } }
    show(set)
    expect(screen.getByText(/No frozen coverage record/)).toBeTruthy()
    expect(screen.getByText(/No valid timing coverage record/)).toBeTruthy()
    expect(screen.queryByTestId('audio-evidence-content')).toBeNull()
    expect(screen.queryByTestId('video-av-evidence')).toBeNull()
    expect(document.querySelectorAll('img[src], audio[src], video[src]')).toHaveLength(0)
  })

  it('refuses nested media from another source and leaves historical v1/v2 rendering unchanged', () => {
    const set = fixture()
    set.audio!.source = { ...set.audio!.source, sha256: 'f'.repeat(64) }
    set.av!.source = { ...set.av!.source, sha256: 'f'.repeat(64) }
    const { unmount } = show(set)
    expect(screen.getByText(/audio record does not match this original video/)).toBeTruthy()
    expect(screen.getByText(/alignment record does not match this original video/)).toBeTruthy()
    expect(document.querySelectorAll('audio[src], video[src], img[src]')).toHaveLength(0)
    unmount()
    show({ ...fixture(), version: 'video-evidence-2' })
    expect(screen.queryByTestId('video-nested-evidence')).toBeNull()
    expect(screen.queryByLabelText(/exact unshifted source audio/)).toBeNull()
  })

  it('waits for media credentials, restores players after token renewal, and keeps exact frames when a clip fails', () => {
    const set = fixture()
    const { rerender } = show(set, '', false)
    expect(document.querySelectorAll('audio[src], video[src], img[src]')).toHaveLength(0)
    rerender(<MemoryRouter><MultimediaEvidenceContent set={set} token="first" mediaReady /></MemoryRouter>)
    fireEvent.error(screen.getByLabelText('AV01: exact unshifted source audio window'))
    fireEvent.error(screen.getByLabelText('AV01: silent source-derived context clip'))
    expect(screen.getAllByRole('alert')).toHaveLength(2)
    expect(screen.getByAltText('AV01.1: exact original video frame')).toBeTruthy()
    rerender(<MemoryRouter><MultimediaEvidenceContent set={set} token="renewed" mediaReady /></MemoryRouter>)
    expect(screen.getByLabelText('AV01: exact unshifted source audio window').getAttribute('src')).toContain('token=renewed')
    expect(screen.getByLabelText('AV01: silent source-derived context clip').getAttribute('src')).toContain('token=renewed')
    expect(screen.queryAllByRole('alert')).toHaveLength(0)
  })
})


it('shows soundtrack level loss in the voice lane with the original video clock', () => {
  const set = fixture()
  const audio = set.audio!
  audio.status = 'not_assessed'
  audio.reason = 'No voice-check windows were scored.'
  audio.coverage = { ...audio.coverage!, scored_samples: 0, unscored_samples: audio.source.samples!, windows: [] }
  audio.coverage.stereo_attenuation_overlaps = [{ start_sample: 34000, end_sample: 48000,
    unscored_interval_index: 0, stereo_window_index: 0,
    source_timestamp_ranges: [{ start_sample: 34000, end_sample: 48000, source_start_pts: 120400,
      source_end_pts: 134400, time_base: '1/16000' }],
  }]
  audio.packets = []
  audio.source = { ...audio.source, source_channels: 2, source_stream_index: 1, audio_stream_indices: [1, 2],
    pcm_sha256: 'e'.repeat(64) }
  audio.stereo_context = {
    schema: 'chipmunk.stereo-channel-context/2', status: 'measured', evidence_type: 'context', affects_overall_confidence: false,
    source_sha256: set.source.sha256, source_channel_count: 2, source_stream_index: 1,
    sample_rate: 16000, max_seconds: 30, quiet_floor_dbfs: -50, strong_level_gap_db: 12, silence_floor_dbfs: -140,
    mono_pcm_sha256: audio.source.pcm_sha256, strong_downmix_attenuation_db: 12,
    windows: [{
      start_sample: 32000, end_sample: 48000, status: 'measured', left_rms_dbfs: -12, right_rms_dbfs: -12,
      level_gap_db: 0, correlation: -1, louder_channel: null,
      downmix: { mono_rms_dbfs: -140, attenuation_db: null, strong_attenuation: true },
      source_timestamp_ranges: [{ start_sample: 32000, end_sample: 48000,
        source_start_pts: 118400, source_end_pts: 134400, time_base: '1/16000' }],
    }],
  }
  const clip = (hash: string): EvidenceAsset => ({
    sha256: hash.repeat(64), media_type: 'audio/wav', url: 'https://foreign.invalid', samples: 16000,
    sample_rate: 16000, channels: 1,
  })
  audio.stereo_listening = { schema: 'chipmunk.stereo-listening/1', affects_overall_confidence: false, windows: [{
    stereo_window_index: 0, artifacts: { left: clip('1'), right: clip('2'), mono: clip('3'),
      waveform: { sha256: '4'.repeat(64), media_type: 'image/png', url: 'https://foreign.invalid', width: 960, height: 290 } },
  }] }
  render(<MemoryRouter initialEntries={['/media-evidence#media-evidence-stereo']}>
    <MultimediaEvidenceContent set={set} token="secure" mediaReady />
  </MemoryRouter>)
  expect(screen.getByRole('tab', { name: /Voice intervals.*1 mono level drop/ }).getAttribute('aria-selected')).toBe('true')
  expect(screen.getByRole('region', { name: 'Audio track coverage' }).textContent).toContain(
    'The file contains 2 audio tracks. Audio checks used the first track (source stream index 1). The second track was unassessed.',
  )
  const section = within(screen.getByRole('region', { name: 'Stereo channel comparison' }))
  expect(section.getByText(/Strong level drop in the analyzed mono mix/).parentElement?.textContent).toContain(
    'Decoded audio 2.000 s–3.000 s.',
  )
  expect(section.getByText('7.400 s–8.400 s')).toBeTruthy()
  expect(section.getByText('Video playback 2.400 s–3.400 s')).toBeTruthy()
  expect(section.getByText('[118400, 134400) at 1/16000')).toBeTruthy()
  expect(section.getByText(/do not contribute to the AI assessment/)).toBeTruthy()
  const comparison = within(section.getByText('Compare stereo and analyzed mono').closest('details')!)
  expect(comparison.getByText(/Source PTS \[118400, 134400\).*Video playback 2\.400 s–3\.400 s/)).toBeTruthy()
  expect(comparison.getByLabelText('Exact analyzed mono').getAttribute('src')).toBe(
    `/api/v2/analyses/video-1/evidence-packets/parent%20set/assets/${'3'.repeat(64)}.wav?token=secure`,
  )
  const overlap = within(screen.getByRole('region', { name: 'Unscored voice during a mono level drop' }))
  expect(overlap.getByText(/Overlap 2\.125 s–3\.000 s/)).toBeTruthy()
  expect(overlap.getByText(/Source PTS \[120400, 134400\).*Video playback 2\.525 s–3\.400 s/)).toBeTruthy()
  expect(overlap.getByText(/cause of the missing voice score is unknown/)).toBeTruthy()
  expect(overlap.getByRole('link', { name: 'stereo measurements' }).getAttribute('href')).toBe('#media-evidence-stereo')
})
