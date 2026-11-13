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

  it('explains a proved orientation and keeps the earlier crop in its own source coordinates', async () => {
    const result = structuredClone(comparison)
    result.comparison.schema = 'chipmunk.image-reference-comparison/6'
    result.comparison.exact_orientation = {
      transform: 'rotate_90_clockwise', matched_pixels: 75_000, matched_tiles: [[1, 1]], rgb_sha256: hash,
    }
    delete result.comparison.alignment
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(result) : json({ available: true }),
    )))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText(/90-degree clockwise rotation/)).toBeTruthy()
    expect(screen.getByText(/75,000 submitted pixels match exactly/)).toBeTruthy()
    expect(screen.getByText('x 134, y 76 · 90 × 70 px')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
    const [[download]] = vi.mocked(saveBlob).mock.calls
    expect(JSON.parse(await blobText(download))).toEqual({ comparison: result.comparison, pngs: result.pngs })
  })

  it('explains the JPEG to PNG comparison and rejects a crop outside the earlier area', async () => {
    const result = structuredClone(comparison)
    result.comparison.schema = 'chipmunk.image-reference-comparison/10'
    result.comparison.reference_crop_bounds = { x: 16, y: 11, w: 356, h: 230 }
    delete result.comparison.alignment
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(result) : json({ available: true }),
    )))
    show()
    await choose('earlier.jpg')
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText(/Ordinary processing can still cause highlights/)).toBeTruthy()
    expect(screen.getByText(/outside the expected delivery range/)).toBeTruthy()
    expect(screen.getByText('x 134, y 76 · 90 × 70 px')).toBeTruthy()
    result.comparison.regions[0].reference_region!.x = 15
    await choose('different.jpg')
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect((await screen.findByRole('alert')).textContent).toContain('The server returned an incomplete JPEG to PNG comparison.')
  })

  it('shows the earlier-image overlap for a cropped reflection and exports its proof', async () => {
    const result = structuredClone(comparison)
    result.comparison.schema = 'chipmunk.image-reference-comparison/7'
    result.comparison.exact_oriented_crop = {
      transform: 'mirror_horizontal', matched_pixels: 75_000, matched_tiles: [[1, 1]],
      native_crop: { x: 16, y: 10, w: 356, h: 230 }, projected_crop: { x: 12, y: 10, w: 356, h: 230 },
      projected_rgb_sha256: hash, difference_mask_sha256: hash,
    }
    delete result.comparison.alignment
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(result) : json({ available: true }),
    )))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText(/An earlier crop aligned using left-to-right reflection/)).toBeTruthy()
    expect(screen.getByText('x 16, y 10 · 356 × 230 px')).toBeTruthy()
    expect(screen.getByText(/Pixels outside this earlier area were unassessed/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
    const [[download]] = vi.mocked(saveBlob).mock.calls
    expect(JSON.parse(await blobText(download))).toEqual({ comparison: result.comparison, pngs: result.pngs })
  })

  it('shows an earlier crop matched after rotation and resize with native coordinates', async () => {
    const result = structuredClone(comparison)
    result.comparison.schema = 'chipmunk.image-reference-comparison/8'
    result.comparison.submitted.width = 201
    result.comparison.submitted.height = 297
    result.comparison.exact_oriented_resize = {
      transform: 'rotate_90_clockwise', matched_pixels: 58_000, matched_tiles: [[1, 1]],
      native_crop: { x: 19, y: 19, w: 330, h: 223 }, projected_crop: { x: 14, y: 19, w: 223, h: 330 },
      projected_rgb_sha256: hash, difference_mask_sha256: hash, kernels: ['bicubic'],
    }
    delete result.comparison.alignment
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(result) : json({ available: true }),
    )))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText(/90-degree clockwise rotation and bicubic resizing/)).toBeTruthy()
    expect(screen.getByText('x 19, y 19 · 330 × 223 px')).toBeTruthy()
    expect(screen.getByText(/White marks retained differences from the reproduced resize/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
    const [[download]] = vi.mocked(saveBlob).mock.calls
    expect(JSON.parse(await blobText(download))).toEqual({ comparison: result.comparison, pngs: result.pngs })
  })

  it('shows exactly which photo area in a screenshot was compared and exports the native proof', async () => {
    const result = structuredClone(comparison)
    result.comparison.schema = 'chipmunk.image-reference-comparison/9'
    result.comparison.submitted.width = 600
    result.comparison.submitted.height = 440
    result.comparison.embedded_native_photo = {
      reference_bounds: { x: 96, y: 31, w: 250, h: 195 },
      submitted_bounds: { x: 80, y: 20, w: 250, h: 195 },
      exact_pixels: 42_450, native_difference_sha256: hash,
      witness_cells: Array.from({ length: 12 }, (_, index) => [Math.floor(index / 4), index % 4, 3]),
    }
    delete result.comparison.alignment
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(result) : json({ available: true }),
    )))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText(/earlier photo matched at native scale/)).toBeTruthy()
    expect(screen.getByText('x 96, y 31 · 250 × 195 px')).toBeTruthy()
    expect(screen.getByText('x 80, y 20 · 250 × 195 px')).toBeTruthy()
    expect(screen.getByText(/AI origin remains unassessed/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
    const [[download]] = vi.mocked(saveBlob).mock.calls
    expect(JSON.parse(await blobText(download))).toEqual({ comparison: result.comparison, pngs: result.pngs })
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

  it('shows pixel-exact PNG support and downloads the exact projection proof', async () => {
    const result = structuredClone(comparison)
    result.comparison.schema = 'chipmunk.image-reference-comparison/5'
    result.comparison.reference_crop_bounds = { x: 24, y: 14, w: 328, h: 227 }
    result.comparison.supported_resize_kernels = ['opencv_area']
    result.comparison.exact_projection = { matched_pixels: 46_816, matched_tiles: [[0, 1], [3, 4]], rgb_sha256: 'e'.repeat(64) }
    delete result.comparison.alignment
    vi.stubGlobal('fetch', vi.fn((_url: string, request: RequestInit) => Promise.resolve(
      request.method === 'POST' ? json(result) : json({ available: true }),
    )))
    show()
    await choose()
    fireEvent.click(screen.getByRole('button', { name: 'Compare images' }))
    expect(await screen.findByText(/matches 46,816 submitted pixels exactly/)).toBeTruthy()
    expect(screen.getByText(/2 textured tiles using area resizing/)).toBeTruthy()
    expect(screen.getByText(/White marks retained differences from the reproduced resize/)).toBeTruthy()
    expect(screen.queryByText(/at least two tested resizes/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
    const [[download]] = vi.mocked(saveBlob).mock.calls
    expect(JSON.parse(await blobText(download))).toEqual({ comparison: result.comparison, pngs: result.pngs })
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
