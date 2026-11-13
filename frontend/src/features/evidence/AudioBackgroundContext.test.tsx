// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { Link, MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AudioEvidenceContent from './AudioEvidenceContent'
import { MultimediaEvidenceContent } from './MultimediaEvidenceCards'
import type { AcousticContextEvent, AudioEvidenceSet, EvidenceAsset, VideoEvidenceSet } from './multimediaPackets'

function fixture(): AudioEvidenceSet {
  let i = 1
  const asset = (media_type: EvidenceAsset['media_type'] = 'image/png'): EvidenceAsset => ({
    sha256: (i++).toString(16).padStart(64, '0'), media_type, url: 'https://foreign.invalid/asset',
  })
  const event = (id: string, start: number, category: AcousticContextEvent['category']): AcousticContextEvent => ({
    id, category, interval: { start_sample: start, end_sample: start + 16000 }, start_s: start / 16000, end_s: start / 16000 + 1,
    context_interval: { start_sample: 0, end_sample: 64000 }, observation: 'Observed in the mixed input.',
    interpretation: 'A recording change can have several causes.', limitations: ['This observation does not establish AI use.'],
    source_timestamp_ranges: [{ start_sample: start, end_sample: start + 16000, source_start_pts: start + 80000,
      source_end_pts: start + 96000, time_base: '1/16000' }],
    observations: [{ label: 'RMS level', value: -62, unit: 'dBFS' }], related_voice_packet_ids: [],
    artifacts: { selected_clip: asset('audio/wav'), context_clip: asset('audio/wav'), waveform: asset(), spectrogram: asset() },
  })
  const source = { sha256: 'a'.repeat(64), pcm_sha256: 'd'.repeat(64), sample_rate: 16000, samples: 64000, source_channels: 2 }
  return { schema: 'chipmunk.audio-evidence/1', status: 'ready', version: 'audio-evidence-2', id: 'parent set',
    analysis_id: 'analysis-a', revision: 1, original_result_hash: 'b'.repeat(64), sha256: 'c'.repeat(64), source,
    limitations: [], packets: [], omitted_windows: 0,
    coverage: { decoded_samples: 64000, scored_samples: 0, unscored_samples: 64000, decode_limit_reached: false,
      selection_threshold: .5, windows: [] },
    acoustic_context: { schema: 'chipmunk.acoustic-context/1', status: 'ready',
      source: { ...source, pcm_sha256: 'd'.repeat(64), coordinate_space: 'analysed-mono-pcm' },
      method: { version: 'mixed-acoustics-1', frame_samples: 8000, description: 'Measured mixed-track acoustics.', thresholds: {} },
      overview: { energy: asset(), spectrogram: asset() }, bins: [],
      events: [event('B01', 32000, 'energy_change'), event('B02', 0, 'near_silence')],
      limitations: ['Voice and background sound were not separated.'], omitted_events: 1,
    },
  }
}
const show = (set: AudioEvidenceSet, entry = '/analysis') => render(<MemoryRouter initialEntries={[entry]}>
  <Link to="#media-evidence-B01">Open B01</Link>
  <AudioEvidenceContent set={set} token="secure token" mediaReady />
</MemoryRouter>)
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('background and recording context', () => {
  it('shows the upper-frequency observation with the exact playable range', () => {
    const set = fixture()
    const context = set.acoustic_context!
    context.schema = 'chipmunk.acoustic-context/3'
    context.upper_frequency_drop = { assessed_ranges: 3, status: 'measured_drop' }
    context.events[0].category = 'upper_frequency_drop'
    context.events[0].interpretation = 'Ordinary sound changes or filtering can produce this pattern. This observation carries zero weight in the AI estimate.'
    show(set)
    expect(screen.getByRole('button', { name: 'Inspect B01, Temporary upper-frequency drop, 2.000 s–3.000 s' })).toBeTruthy()
    expect(screen.getByLabelText('B01: exact acoustic interval').getAttribute('src')).toContain(context.events[0].artifacts.selected_clip.sha256)
    expect(screen.getByText(/carries zero weight in the AI estimate/)).toBeTruthy()
  })

  it('shows that the frequency comparison was unassessed when either flank lacks support', () => {
    const set = fixture()
    const context = set.acoustic_context!
    context.schema = 'chipmunk.acoustic-context/3'
    context.upper_frequency_drop = { assessed_ranges: 0, status: 'unassessed' }
    show(set)
    expect(screen.getByText('The surrounding sound did not support a temporary upper-frequency comparison.')).toBeTruthy()
    expect(screen.queryByText('Temporary upper-frequency drop')).toBeNull()
  })

  it('shows a new continuity lead with the same source-bound media and ordinary-music limitation', () => {
    const set = fixture()
    set.version = 'audio-evidence-3'
    set.acoustic_context!.schema = 'chipmunk.acoustic-context/2'
    set.acoustic_context!.events[0].category = 'spectral_transition'
    set.acoustic_context!.events[0].interpretation = 'Ordinary music can cause this pattern; this is a listening lead, not evidence of AI origin.'
    set.acoustic_context!.omitted_continuity_windows = 2
    show(set)
    expect(screen.getByRole('button', { name: 'Inspect B01, Possible sound-content transition, 2.000 s–3.000 s' })).toBeTruthy()
    expect(screen.getByText(/Ordinary music can cause/)).toBeTruthy()
    expect(screen.getByText(/2 overlapping or lower-ranked frequency comparisons/)).toBeTruthy()
    expect(screen.getByLabelText('B01: exact acoustic interval').getAttribute('src')).toContain(set.acoustic_context!.events[0].artifacts.selected_clip.sha256)
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('shows real acoustic assets and intervals even with no voice scoring, without treating quiet sound as synthesis', () => {
    const set = fixture(), original = structuredClone(set)
    show(set)
    const section = within(screen.getByRole('region', { name: 'Background and recording context' }))
    expect(section.getByAltText('Mixed-track frequency content over the analyzed input').getAttribute('src')).toBe(
      `/api/v2/analyses/analysis-a/evidence-packets/parent%20set/assets/${set.acoustic_context!.overview.spectrogram.sha256}.png?token=secure%20token`)
    expect(section.getByText(/not an isolated background track or a verdict about AI use/)).toBeTruthy()
    expect(section.getByText(/Voice and background sound were not separated/)).toBeTruthy()
    expect(section.getByText(/1 further acoustic observations/)).toBeTruthy()
    expect(screen.queryByRole('region', { name: 'Audio interval navigator' })).toBeNull()
    const buttons = within(section.getByRole('group', { name: 'Choose a background or recording observation' })).getAllByRole('button')
    expect(buttons.map((button) => button.getAttribute('aria-label'))).toEqual([
      'Inspect B02, Very quiet interval, 0.000 s–1.000 s', 'Inspect B01, Sound level change, 2.000 s–3.000 s',
    ])
    expect(set).toEqual(original)
  })

  it('selects the exact interval, shows original source clock, and stops the prior clip before changing selection', async () => {
    const set = fixture()
    show(set)
    const previous = screen.getByLabelText('B01: exact acoustic interval') as HTMLAudioElement
    Object.defineProperty(previous, 'paused', { configurable: true, value: false })
    const pause = vi.spyOn(previous, 'pause').mockImplementation(() => {})
    fireEvent.click(screen.getByRole('button', { name: 'Inspect B02, Very quiet interval, 0.000 s–1.000 s' }))
    expect(pause).toHaveBeenCalledOnce()
    const article = screen.getByRole('article', { name: 'Selected acoustic evidence B02' })
    expect(screen.queryByRole('article', { name: 'Selected acoustic evidence B01' })).toBeNull()
    expect(within(article).getByLabelText('B02: exact acoustic interval').getAttribute('src')).toContain(set.acoustic_context!.events[1].artifacts.selected_clip.sha256)
    expect(within(article).getByText(/samples \[0, 16000\)/)).toBeTruthy()
    expect(within(article).getByText(/5.000 s–6.000 s/)).toBeTruthy()
    expect(within(article).getByText(/This observation does not establish AI use/)).toBeTruthy()
    await waitFor(() => expect(document.activeElement).toBe(article))
  })

  it('opens a saved B interval by hash and supports subsequent reference navigation', () => {
    show(fixture(), '/analysis#media-evidence-B02')
    expect(screen.getByRole('article', { name: 'Selected acoustic evidence B02' })).toBeTruthy()
    fireEvent.click(screen.getByRole('link', { name: 'Open B01' }))
    expect(screen.getByRole('article', { name: 'Selected acoustic evidence B01' })).toBeTruthy()
  })

  it('withholds charts and clips if their source clock does not match the analyzed input', () => {
    const set = fixture()
    set.acoustic_context!.source.samples += 1
    show(set)
    const section = within(screen.getByRole('region', { name: 'Background and recording context' }))
    expect(section.getByRole('alert').textContent).toContain('does not match this analyzed audio input')
    expect(section.queryByRole('img')).toBeNull()
    expect(document.querySelectorAll('audio')).toHaveLength(0)
  })

  it('distinguishes a legacy unassessed result from a measured result with no selected acoustic event', () => {
    const set = fixture()
    set.acoustic_context!.events = []
    const view = show(set)
    expect(screen.getByText(/No acoustic interval met the saved selection rules/)).toBeTruthy()
    expect(screen.getByAltText('Mixed-track sound level over the analyzed input')).toBeTruthy()
    view.unmount()
    delete set.acoustic_context
    show(set)
    expect(screen.getByText(/measurements were not captured for this saved result/)).toBeTruthy()
    expect(document.querySelectorAll('img, audio')).toHaveLength(0)
  })

  it('routes a video B reference to the shared audio view and retains parent asset authority with zero voice windows', () => {
    const audio = fixture()
    const video: VideoEvidenceSet = { schema: 'chipmunk.video-evidence/1', version: 'video-evidence-4', status: 'ready',
      id: 'video parent', analysis_id: 'video-analysis', revision: 3, sha256: 'e'.repeat(64), original_result_hash: 'f'.repeat(64),
      limitations: [], source: { sha256: audio.source.sha256 }, packets: [], omitted_candidates: 0,
      audio: { status: 'not_assessed', reason: 'No successful voice windows.', source: audio.source, coverage: audio.coverage,
        packets: [], limitations: [], acoustic_context: audio.acoustic_context },
    }
    render(<MemoryRouter initialEntries={['/analysis#media-evidence-B02']}><MultimediaEvidenceContent set={video} token="video token" mediaReady /></MemoryRouter>)
    expect(screen.getByRole('tab', { name: /Voice intervals/ }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByRole('article', { name: 'Selected acoustic evidence B02' })).toBeTruthy()
    expect(screen.getByLabelText('B02: exact acoustic interval').getAttribute('src')).toContain('/analyses/video-analysis/evidence-packets/video%20parent/assets/')
    expect(screen.getByText('0 voice intervals · 2 acoustic observations')).toBeTruthy()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })
})
