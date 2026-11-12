// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { Link, MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AudioEvidenceContent from './AudioEvidenceContent'
import type { AudioEvidencePacket, AudioEvidenceSet, EvidenceAsset } from './multimediaPackets'

function fixture(): AudioEvidenceSet {
  let counter = 1
  const asset = (media_type: EvidenceAsset['media_type'] = 'image/png'): EvidenceAsset => ({
    sha256: (counter++).toString(16).padStart(64, '0'), media_type, url: 'https://foreign.invalid/private',
    width: 800, height: 240, samples: 4000, sample_rate: 1000, channels: 1,
  })
  const ref = { id: 'check.opaque:global', display_id: 'S01' }
  const packet = (id: string, start: number, score: number): AudioEvidencePacket => ({
    id, title: 'Synthetic-voice inspection window', observation: `Window response ${score}.`,
    interpretation: 'A recorded response reaches the inspection boundary.', limitations: [], finding_ids: ['F1'], signal_refs: [ref],
    start_s: start / 1000, end_s: (start + 4000) / 1000,
    interval: { start_sample: start, end_sample: start + 4000 },
    context_interval: { start_sample: start - 1000, end_sample: start + 5000 },
    anchor: { sample_rate: 1000, source_timestamp_ranges: [{ start_sample: start, end_sample: start + 4000,
      source_start_pts: start + 10000, source_end_pts: start + 14000, time_base: '1/1000' }] },
    observations: [{ label: 'Window response', value: score, unit: 'uncalibrated score', signal_refs: [ref], unit_ref: `window:${id}` }],
    artifacts: { waveform: asset(), scores: asset(), selected_clip: asset('audio/wav'), context_clip: asset('audio/wav') },
  })
  const packets = [packet('A01', 8000, .9), packet('A02', 2000, .7)]
  return { schema: 'chipmunk.audio-evidence/1', status: 'ready', version: 'audio-evidence-1', id: 'parent set',
    sha256: 'a'.repeat(64), analysis_id: 'analysis-1', revision: 2, original_result_hash: 'b'.repeat(64), limitations: [], omitted_windows: 0,
    source: { sha256: 'c'.repeat(64), samples: 20000, sample_rate: 1000, source_channels: 2 },
    overview: { waveform: asset(), scores: asset() }, packets,
    coverage: { decoded_samples: 20000, scored_samples: 10000, unscored_samples: 10000, decode_limit_reached: false,
      selection_threshold: .5, windows: [
        { start_sample: 2000, end_sample: 6000, score: .7, signal_refs: [ref] },
        { start_sample: 4000, end_sample: 8000, score: .2, signal_refs: [ref] },
        { start_sample: 8000, end_sample: 12000, score: .9, signal_refs: [ref] },
      ] },
  }
}
const show = (set: AudioEvidenceSet, entry = '/analysis', ready = true) => render(<MemoryRouter initialEntries={[entry]}>
  <Link to="#media-evidence-A01">Open referenced A01</Link>
  <AudioEvidenceContent set={set} token="media token" mediaReady={ready} />
</MemoryRouter>)
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('audio interval inspection', () => {
  it('lists the exact checked intervals from an overview-only coverage link and keeps it scoped to standalone audio', () => {
    const set = fixture()
    set.packets = []
    const { rerender } = show(set, '/analysis#media-evidence-voice-coverage')
    expect(screen.getByText('Scored voice intervals').closest('details')?.open).toBe(true)
    expect(screen.getByText('2.000 s–6.000 s · samples [2000, 6000)')).toBeTruthy()
    expect(screen.getByText('4.000 s–8.000 s · samples [4000, 8000)')).toBeTruthy()
    expect(screen.queryByRole('region', { name: 'Audio interval navigator' })).toBeNull()
    rerender(<MemoryRouter><AudioEvidenceContent set={set} token="" mediaReady={false} standalone={false} /></MemoryRouter>)
    expect(screen.queryByText('Scored voice intervals')).toBeNull()
    expect(document.getElementById('media-evidence-voice-coverage')).toBeNull()
  })

  it('shows source-derived waveform and chronological saved intervals without turning gaps into negative findings', () => {
    const set = fixture(), original = structuredClone(set)
    show(set)
    const navigator = within(screen.getByRole('region', { name: 'Audio interval navigator' }))
    expect(navigator.getByRole('img', { name: 'Waveform of the analyzed audio input' }).getAttribute('src')).toContain(set.overview!.waveform.sha256)
    const buttons = within(navigator.getByRole('group', { name: 'Choose an audio inspection interval' })).getAllByRole('button')
    expect(buttons.map((button) => button.getAttribute('aria-label'))).toEqual([
      'Inspect A02, 2.000 s–6.000 s', 'Inspect A01, 8.000 s–12.000 s',
    ])
    expect(buttons[1].getAttribute('aria-pressed')).toBe('true')
    const coverage = navigator.getByRole('img', { name: /Scored audio coverage/ })
    expect(coverage.children).toHaveLength(3)
    expect((coverage.children[0] as HTMLElement).style.left).toBe('10%')
    expect((coverage.children[0] as HTMLElement).style.width).toBe('20%')
    expect(navigator.getByText(/Gaps have no score and have not been cleared/)).toBeTruthy()
    expect(navigator.getByText(/not probabilities that a voice was replaced/)).toBeTruthy()
    expect(set).toEqual(original)
  })

  it('selects the exact saved WAV and highlighted waveform, pauses previous audio and moves focus to the chosen evidence', async () => {
    const set = fixture()
    show(set)
    const oldClip = screen.getByLabelText('A01: exact scored window') as HTMLAudioElement
    Object.defineProperty(oldClip, 'paused', { configurable: true, value: false })
    const pause = vi.spyOn(oldClip, 'pause').mockImplementation(() => {})
    fireEvent.click(screen.getByRole('button', { name: 'Inspect A02, 2.000 s–6.000 s' }))
    expect(pause).toHaveBeenCalledOnce()
    const selected = screen.getByRole('article', { name: 'Selected audio evidence A02' })
    expect(screen.queryByRole('article', { name: 'Selected audio evidence A01' })).toBeNull()
    const clip = within(selected).getByLabelText('A02: exact scored window') as HTMLAudioElement
    expect(clip.getAttribute('src')).toBe(`/api/v2/analyses/analysis-1/evidence-packets/parent%20set/assets/${set.packets[1].artifacts.selected_clip.sha256}.wav?token=media%20token`)
    expect(clip.autoplay).toBe(false)
    expect(within(selected).getByAltText('A02: waveform with the selected window marked').getAttribute('src')).toContain(set.packets[1].artifacts.waveform.sha256)
    expect(within(selected).getByText(/12.000 s–16.000 s/)).toBeTruthy()
    await waitFor(() => expect(document.activeElement).toBe(selected))
    fireEvent.click(within(selected).getByRole('button', { name: 'Choose another interval' }))
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Inspect A02, 2.000 s–6.000 s' }))
  })

  it('opens the referenced window from a finding hash, including subsequent in-app navigation', () => {
    show(fixture(), '/analysis#media-evidence-A02')
    expect(screen.getByRole('article', { name: 'Selected audio evidence A02' })).toBeTruthy()
    fireEvent.click(screen.getByRole('link', { name: 'Open referenced A01' }))
    expect(screen.getByRole('article', { name: 'Selected audio evidence A01' })).toBeTruthy()
  })

  it('pauses every playing clip when its channel unmounts, including players created after credentials load', () => {
    const set = fixture()
    const { rerender, unmount } = render(<MemoryRouter><AudioEvidenceContent set={set} token="" mediaReady={false} /></MemoryRouter>)
    expect(document.querySelectorAll('audio')).toHaveLength(0)
    rerender(<MemoryRouter><AudioEvidenceContent set={set} token="renewed" mediaReady /></MemoryRouter>)
    const players = [screen.getByLabelText('A01: exact scored window'), screen.getByLabelText('A01: window with surrounding audio')]
    const pauses = players.map((player) => {
      Object.defineProperty(player, 'paused', { configurable: true, value: false })
      return vi.spyOn(player as HTMLAudioElement, 'pause').mockImplementation(() => {})
    })
    unmount()
    for (const pause of pauses) expect(pause).toHaveBeenCalledOnce()
  })

  it('withholds interval geometry without a valid source clock and never invents a waveform', () => {
    const set = fixture()
    set.source.sample_rate = undefined
    set.overview = undefined
    show(set, '/analysis', false)
    const navigator = within(screen.getByRole('region', { name: 'Audio interval navigator' }))
    expect(navigator.getByText(/Exact interval placement cannot be shown/)).toBeTruthy()
    expect(navigator.queryByRole('img')).toBeNull()
    expect(document.querySelectorAll('audio[src], img[src]')).toHaveLength(0)
  })

  it('shows zero scored coverage and context, with no selectable windows when none were selected', () => {
    const set = fixture()
    set.packets = []
    set.coverage = { ...set.coverage!, windows: [], scored_samples: 0, unscored_samples: 20000 }
    show(set)
    expect(screen.queryByRole('region', { name: 'Audio interval navigator' })).toBeNull()
    expect(screen.getByText(/Voice checks cover 0.000 s of the 20.000 s decoded input/)).toBeTruthy()
    expect(screen.getByAltText('Waveform of the analyzed audio input')).toBeTruthy()
    expect(document.querySelectorAll('audio[src]')).toHaveLength(0)
  })
})
