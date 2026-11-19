// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AudioEvidenceContent from './AudioEvidenceContent'
import { MultimediaEvidenceContent } from './MultimediaEvidenceCards'
import type { AudioEvidenceSet, BriefVoiceReview, EvidenceAsset } from './multimediaPackets'

const clip = (key: string): EvidenceAsset => ({
  sha256: key.repeat(64), media_type: 'audio/wav', samples: 12000, sample_rate: 16000, url: 'https://foreign.invalid/clip',
})
const waveform = (key: string): EvidenceAsset => ({
  sha256: key.repeat(64), media_type: 'image/png', url: 'https://foreign.invalid/waveform',
})
const briefWindow = (id: string, start: number, clipKey: string, waveKey: string): BriefVoiceReview['windows'][number] => ({
  id, start_sample: start, end_sample: start + 12000, start_s: start / 16000, end_s: (start + 12000) / 16000,
  source_timestamp_ranges: [{ start_sample: start, end_sample: start + 12000,
    source_start_pts: start, source_end_pts: start + 12000, time_base: '1/16000' }],
  artifacts: { selected_clip: clip(clipKey), waveform: waveform(waveKey) },
})

function fixture(): AudioEvidenceSet {
  return {
    schema: 'chipmunk.audio-evidence/1', status: 'ready', version: 'audio-evidence-2', id: 'audio set',
    analysis_id: 'audio-analysis', revision: 1, original_result_hash: 'd'.repeat(64), sha256: 'e'.repeat(64),
    source: { sha256: 'f'.repeat(64), sample_rate: 16000, samples: 192000 }, limitations: [], packets: [], omitted_windows: 0,
    brief_voice_review: {
      schema: 'chipmunk.brief-voice-review/1', status: 'flagged', reason: 'Adjacent brief speech windows returned elevated responses.',
      limitation: 'This review carries zero weight in the overall AI estimate.', affects_overall_confidence: false,
      scheduled_windows: 23, speech_windows_checked: 9,
      windows: [briefWindow('B750-02', 8000, '1', '2'), briefWindow('B750-03', 16000, '3', '4')],
    },
  }
}

function page(set: AudioEvidenceSet, mediaReady = true, standalone = true) {
  return <MemoryRouter initialEntries={['/analysis?view=plain#media-evidence-brief-voice-review']}>
    <AudioEvidenceContent set={set} token="signed token" mediaReady={mediaReady} standalone={standalone} />
  </MemoryRouter>
}
const section = () => within(screen.getByRole('region', { name: 'Brief speech windows' }))

afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('brief source-linked speech windows', () => {
  it('shows the two signed waveform and playback links with exact sample and source clock ranges', () => {
    const set = fixture()
    render(page(set))
    expect(screen.getByRole('region', { name: 'Brief speech windows' }).id).toBe('media-evidence-brief-voice-review')
    const review = section()
    expect(review.getByRole('status').textContent).toBe(set.brief_voice_review!.reason)
    expect(review.getByText('9 of 23 scheduled 750 ms windows checked.')).toBeTruthy()
    expect(review.getByText('Each marked excerpt appears once, even when it supports two pairs.')).toBeTruthy()
    expect(review.getByText(/zero weight in the overall AI estimate/)).toBeTruthy()
    expect(review.getAllByRole('listitem').map(item => item.id)).toEqual(['media-evidence-B750-02', 'media-evidence-B750-03'])

    const uri = '/api/v2/analyses/audio-analysis/evidence-packets/audio%20set/assets/'
    for (const [item, witness] of review.getAllByRole('listitem').map((element, i) => [within(element), set.brief_voice_review!.windows[i]] as const)) {
      const { id, start_sample, end_sample, source_timestamp_ranges: [original], artifacts } = witness
      expect(item.getByText(`${id} · Analyzed samples [${start_sample}, ${end_sample})`)).toBeTruthy()
      expect(item.getByText(new RegExp(`Source PTS \\[${original.source_start_pts}, ${original.source_end_pts}\\) at 1/16000`))).toBeTruthy()
      expect(item.getByRole('img', { name: new RegExp(`Waveform for brief speech window ${id} at`) }).getAttribute('src'))
        .toBe(`${uri}${artifacts.waveform.sha256}.png?token=signed%20token`)
      expect(item.getByLabelText(new RegExp(`Brief speech window ${id} at`)).getAttribute('src'))
        .toBe(`${uri}${artifacts.selected_clip.sha256}.wav?token=signed%20token`)
      expect(item.getByText('Clip length 750 ms')).toBeTruthy()
    }
  })

  it('pauses the other clip when its neighbor starts and loads no media before it is available', () => {
    const set = fixture()
    const { rerender } = render(page(set, false))
    expect(section().getAllByText('Preparing secure media links…')).toHaveLength(4)
    expect(document.querySelectorAll('audio[src], img[src]')).toHaveLength(0)
    rerender(page(set))
    const first = section().getByLabelText(/Brief speech window B750-02 at/)
    const second = section().getByLabelText(/Brief speech window B750-03 at/) as HTMLAudioElement
    Object.defineProperty(second, 'paused', { configurable: true, value: false })
    const pause = vi.spyOn(second, 'pause').mockImplementation(() => {})
    fireEvent.play(first)
    expect(pause).toHaveBeenCalledOnce()
  })

  it.each(['no_flag', 'unassessed', 'skipped'] as const)('shows the recorded %s outcome without displaying an asserted clip', status => {
    const set = fixture()
    set.brief_voice_review!.status = status
    set.brief_voice_review!.reason = `Review status: ${status}`
    render(page(set))
    const review = section()
    expect(review.getByRole('status').textContent).toBe(set.brief_voice_review!.reason)
    expect(review.queryByRole('img')).toBeNull()
    expect(review.queryByLabelText(/Brief speech window B750/)).toBeNull()
    expect(review.queryByText(/Leans original/i)).toBeNull()
  })

  it('leaves nested video audio and older audio reports unchanged', () => {
    const set = fixture()
    const { rerender } = render(page(set, true, false))
    expect(screen.queryByRole('region', { name: 'Brief speech windows' })).toBeNull()
    rerender(page({ ...set, brief_voice_review: undefined }))
    expect(screen.queryByRole('region', { name: 'Brief speech windows' })).toBeNull()
  })

  it('counts the paired review as selected evidence in the overall evidence view', () => {
    render(<MemoryRouter><MultimediaEvidenceContent set={fixture()} token="signed token" mediaReady /></MemoryRouter>)
    expect(section().getAllByRole('listitem')).toHaveLength(2)
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })
})
