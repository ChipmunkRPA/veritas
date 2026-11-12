// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { saveBlob } from '../../lib/api/reports'
import VideoReferenceComparison from './VideoReferenceComparison'

vi.mock('../../lib/api/reports', () => ({ saveBlob: vi.fn() }))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.resetAllMocks()
})

function show() {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <VideoReferenceComparison analysisId="test/video" />
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
  fireEvent.click(screen.getByRole('button', { name: 'Compare videos' }))
  expect(await screen.findByText('V01 · 0.480–0.800 s')).toBeTruthy()
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
