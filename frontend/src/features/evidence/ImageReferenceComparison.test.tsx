// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { saveBlob } from '../../lib/api/reports'
import ImageReferenceComparison from './ImageReferenceComparison'
import { imageReferenceExport, referencePngSrc, type ImageReferenceResult } from './imageReferenceApi'

vi.mock('../../lib/api/reports', () => ({ saveBlob: vi.fn() }))

const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/XocAAAAASUVORK5CYII='
const pdf = 'JVBERi0xLjQK'
const hash = 'a'.repeat(64)
const artifact = (file: string, width = 90, height = 70) => ({ file, sha256: hash, width, height })
const comparison: ImageReferenceResult = {
  comparison: {
    schema: 'chipmunk.image-reference-comparison/2',
    status: 'comparable',
    reference: { sha256: 'b'.repeat(64), width: 384, height: 256 },
    submitted: { sha256: 'c'.repeat(64), width: 356, height: 230 },
    regions: [{
      x: 118, y: 65, w: 90, h: 70,
      reference_region: { x: 134, y: 76, w: 90, h: 70 },
      reference_crop: artifact('reference-01.png'),
      submitted_crop: artifact('submitted-01.png'),
      submitted_overlay: artifact('submitted-overlay-01.png'),
    }],
    mask: artifact('difference-mask.png', 356, 230),
    alignment: {
      submitted_overlap: { x: 0, y: 0, w: 356, h: 230 },
      reference_overlap: { x: 16, y: 11, w: 356, h: 230 },
    },
  },
  pngs: { 'reference-01.png': png, 'submitted-01.png': png, 'submitted-overlay-01.png': png, 'difference-mask.png': png },
  pdf,
}

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status })
const file = (name: string) => new File(['earlier image'], name, { type: 'image/png' })
const heading = 'Compare with an earlier copy'

function show(id = 'analysis-1', query = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  return render(<QueryClientProvider client={query}><ImageReferenceComparison analysisId={id} /></QueryClientProvider>)
}

async function choose(name = 'earlier.png') {
  const input = await screen.findByLabelText('Earlier copy')
  fireEvent.change(input, { target: { files: [file(name)] } })
}

function blobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  localStorage.clear()
  document.cookie = 'veritas_csrf=; max-age=0; path=/'
})

describe('earlier-copy comparison', () => {
  it('checks fresh availability before displaying the form, even with a previously available cached value', async () => {
    const query = new QueryClient()
    query.setQueryData(['analysis', 'analysis-1', 'image-reference-comparison'], true)
    let finish!: (value: Response) => void
    const fetch = vi.fn(() => new Promise<Response>(resolve => { finish = resolve }))
    vi.stubGlobal('fetch', fetch)
    show('analysis-1', query)
    expect(screen.queryByRole('heading', { name: heading })).toBeNull()
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))
    await act(async () => finish(json({ available: false })))
    expect(screen.queryByRole('heading', { name: heading })).toBeNull()
  })

  it('posts the earlier copy and CSRF, shows the exact positions, and downloads separate PDF and JSON evidence', async () => {
    localStorage.setItem('veritas_api_key', 'test-key')
    document.cookie = 'veritas_csrf=test-csrf; path=/'
    const fetch = vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(comparison) : json({ available: true }),
    ))
    vi.stubGlobal('fetch', fetch)
    show('analysis/1')
    await choose()
    expect(screen.getByLabelText('Earlier copy').getAttribute('accept')).toBe('image/png,image/jpeg,.png,.jpg,.jpeg')
    expect(screen.getByText('PNG or JPEG, up to 4 megapixels.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    await screen.findByText('1 difference region found')
    expect(fetch.mock.calls[0][0]).toBe('/api/v2/analyses/analysis%2F1/image-reference-comparison')
    expect(fetch.mock.calls[0][1].headers).toEqual({ 'X-API-Key': 'test-key' })
    const request = fetch.mock.calls[1][1]
    expect(request.headers).toEqual({ 'X-API-Key': 'test-key', 'X-CSRF-Token': 'test-csrf' })
    expect(((request.body as FormData).get('reference') as File).name).toBe('earlier.png')
    expect((request.body as FormData).get('include_pdf')).toBe('true')
    for (const label of ['Earlier copy crop for region 1', 'Submitted image crop for region 1', 'Difference mask in submitted image coordinates']) {
      expect(screen.getByRole('img', { name: label }).getAttribute('src')).toBe(`data:image/png;base64,${png}`)
    }
    expect(screen.getByText('x 134, y 76 · 90 × 70 px')).toBeTruthy()
    expect(screen.getByText('x 118, y 65 · 90 × 70 px')).toBeTruthy()
    expect(screen.getByText('x 16, y 11 · 356 × 230 px')).toBeTruthy()
    expect(screen.getByText(/Pixels outside the overlap were unassessed/)).toBeTruthy()
    const toggle = screen.getByRole('button', { name: 'Highlight difference pixels' })
    expect(toggle.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-pressed')).toBe('true')
    const submitted = screen.getByRole('img', { name: 'Submitted image crop for region 1' })
    expect(submitted.nextElementSibling?.getAttribute('src')).toBe(`data:image/png;base64,${png}`)
    expect(screen.getByText('Amber marks the selected pixels in this crop.')).toBeTruthy()
    fireEvent.click(toggle)
    expect(submitted.nextElementSibling).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Download comparison PDF' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
    const [pdfCall, jsonCall] = vi.mocked(saveBlob).mock.calls
    expect(pdfCall[0].type).toBe('application/pdf')
    expect(pdfCall[1]).toBe('chipmunk-verify-image-comparison.pdf')
    expect(await blobText(pdfCall[0])).toMatch(/^%PDF-/)
    expect(jsonCall[0].type).toBe('application/json')
    expect(JSON.parse(await blobText(jsonCall[0]))).toEqual({ comparison: comparison.comparison, pngs: comparison.pngs })
  })

  it('distinguishes a measured empty result from an unassessed result', async () => {
    const fetch = vi.fn((_url: string, request: RequestInit) => {
      if (request.method === 'GET') return Promise.resolve(json({ available: true }))
      const name = ((request.body as FormData).get('reference') as File).name
      return Promise.resolve(json({
        comparison: { ...comparison.comparison, status: name === 'shifted.png' ? 'unaligned' : 'comparable', regions: [] },
        pngs: {}, pdf: 'javascript:alert(1)',
      }))
    })
    vi.stubGlobal('fetch', fetch)
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText('No localized differences found')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Download comparison PDF' })).toBeNull()
    await choose('shifted.png')
    expect(screen.queryByText('No localized differences found')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText('Comparison unassessed')).toBeTruthy()
    expect(screen.getByText('The copies could not be aligned reliably.')).toBeTruthy()
  })

  it('explains resize comparison and preserves both native coordinates in the JSON download', async () => {
    const result = structuredClone(comparison)
    result.comparison.schema = 'chipmunk.image-reference-comparison/3'
    result.comparison.submitted_over_reference_xy = [[315, 384], [210, 256]]
    delete result.comparison.alignment
    result.comparison.regions[0].reference_region = { x: 143, y: 79, w: 111, h: 86 }
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(result) : json({ available: true }),
    )))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText(/matching framing at different pixel dimensions/)).toBeTruthy()
    expect(screen.getByText('x 143, y 79 · 111 × 86 px')).toBeTruthy()
    expect(screen.getByText('x 118, y 65 · 90 × 70 px')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
    const [[download]] = vi.mocked(saveBlob).mock.calls
    expect(JSON.parse(await blobText(download))).toEqual({ comparison: result.comparison, pngs: result.pngs })
  })

  it('shows the exact earlier area used when the submitted copy is a resized crop', async () => {
    const result = structuredClone(comparison)
    result.comparison.schema = 'chipmunk.image-reference-comparison/4'
    result.comparison.reference_crop_bounds = { x: 24, y: 14, w: 328, h: 227 }
    delete result.comparison.alignment
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(result) : json({ available: true }),
    )))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText('x 24, y 14 · 328 × 227 px')).toBeTruthy()
    expect(screen.getByText(/Pixels outside the earlier area were unassessed/)).toBeTruthy()
  })

  it('lets the reviewer retry a failed upload without choosing the same file again', async () => {
    let attempted = false
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => {
      if (request.method === 'GET') return Promise.resolve(json({ available: true }))
      if (attempted) return Promise.resolve(json(comparison))
      attempted = true
      return Promise.resolve(json({ detail: 'The earlier file could not be decoded.' }, 422))
    }))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect((await screen.findByRole('alert')).textContent).toContain('The earlier file could not be decoded.')
    fireEvent.click(screen.getByRole('button', { name: 'Retry comparison' }))
    expect(await screen.findByText('1 difference region found')).toBeTruthy()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('ignores a superseded response when the reviewer chooses another file during comparison', async () => {
    let finish!: (value: Response) => void
    let firstSignal: AbortSignal | null | undefined
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => {
      if (request.method === 'GET') return Promise.resolve(json({ available: true }))
      if (((request.body as FormData).get('reference') as File).name === 'earlier.png') {
        firstSignal = request.signal
        return new Promise<Response>(resolve => { finish = resolve })
      }
      return Promise.resolve(json(comparison))
    }))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(screen.getByText('Measuring pixel differences…')).toBeTruthy()
    await choose('another.png')
    expect(firstSignal?.aborted).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    await screen.findByText('1 difference region found')
    await act(async () => finish(json({ detail: 'Stale error' }, 422)))
    expect(screen.queryByText(/Stale error/)).toBeNull()
    expect(screen.getByText('1 difference region found')).toBeTruthy()
  })

  it('cancels the old analysis request and checks the new analysis independently', async () => {
    let previousSignal: AbortSignal | null | undefined
    let finishPrevious!: (value: Response) => void
    let finishAvailability!: (value: Response) => void
    vi.stubGlobal('fetch', vi.fn((url: string, request: RequestInit) => {
      if (request.method === 'POST') {
        previousSignal = request.signal
        return new Promise<Response>(resolve => { finishPrevious = resolve })
      }
      if (url.includes('/analysis-2/')) return new Promise<Response>(resolve => { finishAvailability = resolve })
      return Promise.resolve(json({ available: true }))
    }))
    const query = new QueryClient()
    query.setQueryData(['analysis', 'analysis-2', 'image-reference-comparison'], true)
    const view = show('analysis-1', query)
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    view.rerender(<QueryClientProvider client={query}><ImageReferenceComparison analysisId="analysis-2" /></QueryClientProvider>)
    expect(previousSignal?.aborted).toBe(true)
    expect(screen.queryByLabelText('Earlier copy')).toBeNull()
    await act(async () => {
      finishPrevious(json(comparison))
      finishAvailability(json({ available: false }))
    })
    expect(screen.queryByRole('heading', { name: heading })).toBeNull()
    expect(screen.queryByText('1 difference region found')).toBeNull()
  })

  it('allows only fixed-PNG data from files referenced by the comparison record', () => {
    const wrong = { ...comparison.pngs, 'reference-01.png': '<svg onload="alert(1)"></svg>' }
    const source = comparison.comparison.regions[0].reference_crop
    expect(referencePngSrc(wrong, source)).toBeNull()
    expect(referencePngSrc({ '../unexpected.png': png }, artifact('../unexpected.png'))).toBeNull()
    expect(referencePngSrc({ 'unused.png': png }, source)).toBeNull()
    const exported = imageReferenceExport({ ...comparison, pngs: { ...wrong, 'unused.png': png } })
    expect(Object.keys(exported.pngs).sort()).toEqual(['difference-mask.png', 'submitted-01.png', 'submitted-overlay-01.png'])
  })
})
