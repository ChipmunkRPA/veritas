// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { saveBlob } from '../../lib/api/reports'
import MediaReferenceComparison from './MediaReferenceComparison'

vi.mock('../../lib/api/reports', () => ({ saveBlob: vi.fn() }))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.resetAllMocks()
})

function show(kind: 'audio' | 'video' = 'video') {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MediaReferenceComparison analysisId="test/video" kind={kind} />
  </QueryClientProvider>)
}

it('shows matching source times and waveform previews and exports their JSON and PDF', async () => {
  const picture = { file: 'earlier.png', media_type: 'image/png', sha256: 'a'.repeat(64) }
  const sound = { file: 'earlier.wav', media_type: 'audio/wav', sha256: 'b'.repeat(64) }
  const waveform = { low: [-0.4], high: [0.2] }
  const assets = { 'earlier.png': 'iVBORw0KGgo=', 'earlier.wav': 'UklGRg==' }
  const comparison = {
    schema: 'chipmunk.video-reference-comparison/1', status: 'measured', interpretation: 'Decoded source differences.',
    submitted: { width: 64, height: 48 },
    picture: { matching_frames: 42, changed_frames: 8, ranges: [{
      id: 'V01', start_s: .48, end_s: .8,
      strongest_frame: { changed_pixels: 264, difference_bounds: { x: 10, y: 7, w: 22, h: 12 } },
      previews: { during: { time_s: .48, assets: { earlier: picture, submitted: picture } } },
    }] },
    audio: { status: 'measured', changed_samples: 2880, ranges: [{
      id: 'A01', start_s: .24, end_s: .42, changed_samples: 2880,
      preview: { end_s: .42, waveforms: { earlier: waveform, submitted: waveform }, assets: { earlier: sound, submitted: sound } },
    }] },
  }
  const fetch = vi.fn().mockResolvedValueOnce(Response.json({ available: true }))
    .mockResolvedValueOnce(Response.json({ comparison, assets, pdf: 'JVBERi0xLjQK' }))
  const url = new URL('https://example.com/')
  const createObjectURL = vi.fn().mockReturnValue('blob:earlier-video')
  const revokeObjectURL = vi.fn()
  vi.stubGlobal('URL', Object.assign(function URL() { return url }, { createObjectURL, revokeObjectURL }))
  vi.stubGlobal('fetch', fetch)
  const page = show()
  fireEvent.change(await screen.findByLabelText('Earlier video'), { target: { files: [new File(['movie'], 'earlier.mov')] } })
  expect(screen.queryByLabelText('Earlier mouth closure (seconds)')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Compare videos' }))
  expect(await screen.findByText('V01 · 0.480–0.800 s')).toBeTruthy()
  expect(screen.getByText('Local soundtrack timing was skipped because the decoded pictures differ.')).toBeTruthy()
  expect(screen.getByText('A01 · 0.240–0.420 s')).toBeTruthy()
  expect(screen.getByAltText('Earlier clip at 0.480 seconds').getAttribute('src')).toBe('data:image/png;base64,iVBORw0KGgo=')
  expect(screen.getByRole('img', { name: 'Analyzed clip waveform' })).toBeTruthy()
  expect(screen.getByLabelText('Play analyzed clip').getAttribute('src')).toBe('blob:earlier-video')
  expect(fetch.mock.calls[1][0]).toBe('/api/v2/analyses/test%2Fvideo/video-reference-comparison')
  expect((fetch.mock.calls[1][1].body as FormData).get('include_pdf')).toBe('true')
  fireEvent.click(screen.getByRole('button', { name: 'Download JSON evidence' }))
  fireEvent.click(screen.getByRole('button', { name: 'Download PDF' }))
  expect(vi.mocked(saveBlob).mock.calls.map(([blob]) => blob.type)).toEqual(['application/json', 'application/pdf'])
  page.unmount()
  expect(revokeObjectURL).toHaveBeenCalledTimes(2)
})

it('renders the signed local soundtrack shift and exact native picture from the current video export', async () => {
  const picture = { file: 'native.png', media_type: 'image/png', sha256: 'a'.repeat(64) }
  const audio = { file: 'matched.wav', media_type: 'audio/wav', sha256: 'b'.repeat(64) }
  const waveform = { low: [-0.4], high: [0.2] }
  const summary = 'Matching soundtrack audio moved 200.0 ms later relative to the same pictures.'
  const comparison = {
    schema: 'chipmunk.video-reference-comparison/5', status: 'measured', interpretation: 'Decoded source differences.',
    submitted: { width: 64, height: 48 },
    soundtrack_alignment: { summary, scope: 'The pictures matched.', range: {
      earlier: { start_s: .56, end_s: 1.16, audio, waveform }, submitted: { start_s: .76, end_s: 1.36, audio, waveform },
      picture: { frame_index: 31, time_s: 1.034, asset: picture },
    } },
  }
  const fetch = vi.fn().mockResolvedValueOnce(Response.json({ available: true }))
    .mockResolvedValueOnce(Response.json({ comparison, assets: { 'native.png': 'iVBORw0KGgo=', 'matched.wav': 'UklGRg==' } }))
  vi.stubGlobal('fetch', fetch)
  show()
  fireEvent.change(await screen.findByLabelText('Earlier video'), { target: { files: [new File(['movie'], 'earlier.mp4')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare videos' }))
  expect(await screen.findByText(summary)).toBeTruthy()
  expect(screen.getByText('Earlier clip · 0.560–1.160 s')).toBeTruthy()
  expect(screen.getByText('Analyzed clip · 0.760–1.360 s')).toBeTruthy()
  expect(screen.getByAltText('Analyzed clip picture 31 at 1.034 seconds').getAttribute('src')).toBe('data:image/png;base64,iVBORw0KGgo=')
  expect(screen.getAllByRole('img', { name: /matched soundtrack waveform/ })).toHaveLength(2)
})

it('shows and exports the standalone audio difference waveform on the recording timeline', async () => {
  const clip = { file: 'earlier.wav', media_type: 'audio/wav', sha256: 'c'.repeat(64) }
  const waveform = { low: [-0.4], high: [0.2] }
  const comparison = {
    schema: 'chipmunk.audio-reference-comparison/1', status: 'measured', interpretation: 'Decoded audio differences.',
    audio: { status: 'measured', changed_samples: 140, submitted: { samples: 32000 }, ranges: [{
      id: 'A01', start_s: .2, end_s: .24, changed_samples: 140, difference: { peak: 1.5, rms: .3 },
      preview: { end_s: .24, waveforms: { earlier: waveform, submitted: waveform },
        difference_waveform: { low: [-1.5], high: [.1] }, assets: { earlier: clip, submitted: clip } },
    }] },
  }
  const fetch = vi.fn().mockResolvedValueOnce(Response.json({ available: true }))
    .mockResolvedValueOnce(Response.json({ comparison, assets: {}, pdf: 'JVBERi0xLjQK' }))
  vi.stubGlobal('fetch', fetch)
  show('audio')
  fireEvent.change(await screen.findByLabelText('Earlier audio'), { target: { files: [new File(['audio'], 'earlier.wav')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare recordings' }))
  expect(await screen.findByRole('img', { name: 'Difference waveform' })).toBeTruthy()
  expect(screen.getByText('A01 · 0.200–0.240 s')).toBeTruthy()
  expect(screen.getByText('Full range difference: peak 1.500000, RMS 0.300000 of full scale.')).toBeTruthy()
  expect(fetch.mock.calls[1][0]).toBe('/api/v2/analyses/test%2Fvideo/audio-reference-comparison')
  fireEvent.click(screen.getByRole('button', { name: 'Download JSON evidence' }))
  fireEvent.click(screen.getByRole('button', { name: 'Download PDF' }))
  expect(vi.mocked(saveBlob).mock.calls.map(([, name]) => name)).toEqual([
    'chipmunk-verify-audio-comparison.json', 'chipmunk-verify-audio-comparison.pdf',
  ])
})

it('shows the localized audio difference after volume adjustment alongside the raw comparison', async () => {
  const waveform = { low: [-0.4], high: [0.2] }
  const clip = { file: 'earlier-volume.wav', media_type: 'audio/wav', sha256: 'b'.repeat(64) }
  const comparison = {
    schema: 'chipmunk.audio-reference-comparison/2', status: 'measured', interpretation: 'Decoded audio differences.',
    audio: { status: 'measured', changed_samples: 31200, submitted: { samples: 32000 }, ranges: [{
      id: 'A01', start_s: 0, end_s: 2, start_sample: 0, end_sample: 32000, changed_samples: 31200,
    }] },
    level_adjusted: { gain_db: -6.021, changed_samples: 2700, interpretation: 'Both end sections support the volume adjustment.', ranges: [{
      id: 'V01', start_s: .76, end_s: .94, changed_samples: 2700, difference: { peak: .25, rms: .1 },
      preview: { end_s: .94, waveforms: { earlier: waveform, submitted: waveform },
        difference_waveform: waveform, assets: { earlier: clip, submitted: clip } },
    }] },
  }
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(Response.json({ available: true }))
    .mockResolvedValueOnce(Response.json({ comparison, assets: {}, pdf: 'JVBERi0xLjQK' })))
  show('audio')
  fireEvent.change(await screen.findByLabelText('Earlier audio'), { target: { files: [new File(['audio'], 'earlier.wav')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare recordings' }))
  expect(await screen.findByRole('region', { name: 'Differences after a volume adjustment' })).toBeTruthy()
  expect(screen.getByText('The analyzed recording is 6.021 dB quieter in the matching sections at both ends.')).toBeTruthy()
  expect(screen.getByText('V01 · 0.760–0.940 s')).toBeTruthy()
  expect(screen.getByRole('img', { name: 'Volume-adjusted difference waveform' })).toBeTruthy()
  expect(screen.getByText('A01 · 0.000–2.000 s')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Download JSON evidence' }))
  fireEvent.click(screen.getByRole('button', { name: 'Download PDF' }))
  expect(vi.mocked(saveBlob).mock.calls.map(([, name]) => name)).toEqual([
    'chipmunk-verify-audio-comparison.json', 'chipmunk-verify-audio-comparison.pdf',
  ])
})

it('shows both timestamps and waveforms for a moved excerpt in two recordings', async () => {
  const audio = { file: 'excerpt.wav', media_type: 'audio/wav', sha256: 'c'.repeat(64) }
  const waveform = { low: [-.4], high: [.2] }
  const comparison = {
    schema: 'chipmunk.audio-reference-comparison/6', status: 'measured', interpretation: 'Decoded audio differences.',
    local_timing: {
      summary: 'The matching excerpt occurs 200.0 ms later in the analyzed recording.',
      scope: 'Copying or ordinary editing can produce this pattern.',
      range: {
        earlier: { start_s: .8, end_s: 1.44, audio, waveform },
        submitted: { start_s: 1, end_s: 1.64, audio, waveform },
      },
    },
  }
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(Response.json({ available: true }))
    .mockResolvedValueOnce(Response.json({ comparison, assets: {}, pdf: 'JVBERi0xLjQK' })))
  show('audio')
  fireEvent.change(await screen.findByLabelText('Earlier audio'), { target: { files: [new File(['audio'], 'earlier.wav')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare recordings' }))
  expect(await screen.findByRole('region', { name: 'Local audio timing' })).toBeTruthy()
  expect(screen.getByText('Earlier recording · 0.800–1.440 s')).toBeTruthy()
  expect(screen.getByText('Analyzed recording · 1.000–1.640 s')).toBeTruthy()
  expect(screen.getAllByRole('img', { name: /matched excerpt waveform/ })).toHaveLength(2)
  expect(screen.getByRole('button', { name: 'Download PDF' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Download JSON evidence' })).toBeTruthy()
})

it('accepts the scoped video comparison and shows native crops linked to the review interval', async () => {
  const picture = { file: 'picture.png', media_type: 'image/png', sha256: 'c'.repeat(64) }
  const crop = { file: 'crop.png', media_type: 'image/png', sha256: 'd'.repeat(64) }
  const sample = (time_s: number) => ({ time_s, assets: { earlier: picture, submitted: picture }, crop_assets: { earlier: crop, submitted: crop } })
  const comparison = {
    schema: 'chipmunk.video-reference-comparison/2', status: 'measured', interpretation: 'Decoded source differences.',
    submitted: { width: 360, height: 288 },
    picture: { matching_frames: 0, changed_frames: 75, ranges: [], local_comparison: {
      status: 'local_difference', scope: 'Ordinary edits may produce this evidence.', summary: 'A local picture difference persisted.',
      range: { id: 'L01', start_s: 1.08, end_s: 1.88, tile: { x: 126, y: 200, w: 29, h: 20 },
        previews: { before: sample(1.04), during: sample(1.36), after: sample(1.88) } },
    } },
  }
  const fetch = vi.fn().mockResolvedValueOnce(Response.json({ available: true }))
    .mockResolvedValueOnce(Response.json({ comparison, assets: { 'picture.png': 'iVBORw0KGgo=', 'crop.png': 'iVBORw0KGgo=' } }))
  vi.stubGlobal('fetch', fetch)
  show()
  fireEvent.change(await screen.findByLabelText('Earlier video'), { target: { files: [new File(['movie'], 'earlier.mkv')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare videos' }))
  expect(await screen.findByText('L01 · 1.080–1.880 s')).toBeTruthy()
  expect(screen.getByAltText('Earlier clip comparison crop at 1.040 seconds')).toBeTruthy()
  expect(screen.getByAltText('Analyzed clip comparison crop at 1.360 seconds')).toBeTruthy()
  expect(screen.getByAltText('Earlier clip comparison crop at 1.880 seconds')).toBeTruthy()
  expect(screen.getByText('Ordinary edits may produce this evidence.')).toBeTruthy()
})

it.each([3, 4, 7])('renders exact timing evidence from marked comparison format %i', async (version) => {
  const crop = { file: 'mouth.png', media_type: 'image/png', sha256: 'd'.repeat(64) }
  const speech = { file: 'speech.wav', media_type: 'audio/wav', sha256: 'e'.repeat(64) }
  const evidence = (frame_index: number, picture_time_s: number, pcm_start_s: number, pcm_end_s: number) => ({
    frame_index, picture_time_s, pcm_start_s, pcm_end_s, crop, speech, waveform: { low: [-.4], high: [.2] },
  })
  const comparison = {
    schema: 'chipmunk.video-reference-comparison/1', status: 'measured', interpretation: 'Decoded source differences.',
    submitted: { width: 360, height: 288 }, picture: { matching_frames: 57, changed_frames: 18, ranges: [] },
  }
  const speechStart = version === 4 ? .72 : .7600625
  const speechEnd = version === 4 ? .78 : .9200625
  const localSound = version === 7
  const marked = { ...comparison, schema: `chipmunk.video-reference-comparison/${version}`,
    ...(localSound ? { soundtrack_alignment: { summary: 'No qualifying local soundtrack shift was found.' } } : {}),
    timing: {
    status: 'timing_difference', summary: 'The picture event moved 320 ms later relative to speech.',
    scope: version === 4 ? 'The timing check used a shorter speech excerpt beginning at the marked frame.'
      : 'The measured timing difference can come from ordinary editing.', offset_ms: 320, event_range_s: [.72, 1.04],
    ...(localSound ? { changed_audio_samples: [9600, 19200] } : {}),
    earlier: evidence(18, .72, speechStart, speechEnd), submitted: evidence(26, 1.04, speechStart, speechEnd),
  } }
  const fetch = vi.fn().mockResolvedValueOnce(Response.json({ available: true, timing_available: true }))
    .mockResolvedValueOnce(Response.json({ comparison, assets: {} }))
    .mockResolvedValueOnce(Response.json({ comparison: marked, assets: { 'mouth.png': 'iVBORw0KGgo=', 'speech.wav': 'UklGRg==' } }))
  const createObjectURL = vi.fn().mockReturnValue('blob:timing-speech')
  const revokeObjectURL = vi.fn()
  const url = new URL('https://example.com/')
  vi.stubGlobal('URL', Object.assign(function URL() { return url }, { createObjectURL, revokeObjectURL }))
  vi.stubGlobal('fetch', fetch)
  const page = show()
  fireEvent.change(await screen.findByLabelText('Earlier video'), { target: { files: [new File(['movie'], 'earlier.mpg')] } })
  const marker = screen.getByRole('spinbutton', { name: 'Earlier mouth closure (seconds)' })
  expect(marker.getAttribute('step')).toBe('0.001')
  fireEvent.click(screen.getByRole('button', { name: 'Compare videos' }))
  expect(await screen.findByText('57 exact matches and 18 changed decoded pictures.')).toBeTruthy()
  expect((fetch.mock.calls[1][1].body as FormData).has('timing_event_s')).toBe(false)
  expect(screen.queryByRole('region', { name: 'Mouth and speech timing' })).toBeNull()
  fireEvent.change(marker, { target: { value: '0.720' } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare videos' }))
  expect(await screen.findByText(marked.timing.summary)).toBeTruthy()
  expect((fetch.mock.calls[2][1].body as FormData).get('timing_event_s')).toBe('0.720')
  expect(screen.getByText(localSound ? 'Changed sound in the analyzed clip 0.600–1.200 s.'
    : 'Compared event times 0.720–1.040 s.')).toBeTruthy()
  expect(screen.queryByText('No qualifying local soundtrack shift was found.')).toBeNull()
  expect(screen.getByAltText('Earlier clip mouth crop at 0.720 seconds').getAttribute('src')).toBe('data:image/png;base64,iVBORw0KGgo=')
  expect(screen.getByAltText('Analyzed clip mouth crop at 1.040 seconds')).toBeTruthy()
  expect(screen.getByText('Frame 26 at 1.040 s')).toBeTruthy()
  expect(screen.getAllByText(version === 4 ? 'Speech 0.720–0.780 s' : 'Speech 0.7600625–0.9200625 s')).toHaveLength(2)
  expect(screen.getByRole('img', { name: 'Earlier clip timing speech waveform' })).toBeTruthy()
  expect(screen.getByLabelText('Play analyzed clip timing speech').getAttribute('src')).toBe('blob:timing-speech')
  expect(screen.getByText(marked.timing.scope)).toBeTruthy()
  page.unmount()
  expect(revokeObjectURL).toHaveBeenCalledTimes(2)
})
