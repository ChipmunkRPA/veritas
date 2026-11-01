// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { Link, MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import SectionRoute from '../analysis/SectionRoute'
import DocumentEvidenceContent from './DocumentEvidenceContent'
import { ImageEvidenceContent } from './ImageEvidenceCards'
import type { ImageEvidenceSet } from './imagePackets'
import type { DocumentEvidenceSet } from './multimediaPackets'

const hash = 'a'.repeat(64)
const artifact = { sha256: hash, media_type: 'image/png' as const, width: 100, height: 100, url: '' }
const images: ImageEvidenceSet = {
  schema: 'chipmunk.image-evidence/1', status: 'ready', id: 'image-set', version: 'image-evidence-3',
  sha256: hash, analysis_id: 'analysis-1', revision: 1, original_result_hash: hash,
  source: { sha256: hash, width: 100, height: 100, coordinate_space: 'exif-oriented-original' },
  limitations: [], omitted_regions: 0,
  packets: ['I01', 'I02', 'I03', 'I04'].map((id) => ({
    id, title: `Region ${id}`, finding_ids: [], signal_refs: [], exhibit_refs: [], measurements: [],
    evidence_type: 'anomaly', semantics: 'error_level', region: { x: 1, y: 2, w: 10, h: 20 },
    crop_region: { x: 0, y: 0, w: 20, h: 30 }, observation: `Recorded difference ${id}`,
    interpretation: 'Inspect the recorded source region.', overlay_legend: 'Relative diagnostic response.', limitations: [],
    artifacts: { context: artifact, crop: artifact, overlay: artifact },
  })),
}
const passages: DocumentEvidenceSet = {
  schema: 'chipmunk.document-evidence/1', status: 'ready', id: 'text-set', version: 'document-evidence-1',
  sha256: hash, analysis_id: 'analysis-1', revision: 1, original_result_hash: hash,
  source: { sha256: hash, chars: 10, offset_unit: 'unicode_codepoint' }, limitations: [], omitted_windows: 0,
  packets: ['T01', 'T02'].map((id, index) => ({
    id, title: `Passage ${id}`, finding_ids: [], signal_refs: [], measurements: [], artifacts: {},
    observation: `Recorded passage ${id}`, interpretation: 'Inspect this recorded passage.', limitations: [],
    focus: { start: 0, end: 5, text: 'Hello' }, context: { start: 0, end: 10, text: 'Hello text' },
    anchors: [{ kind: 'text', paragraph: index + 1 }], supporting_windows: [],
  })),
}
const scroll = vi.fn()
beforeEach(() => {
  scroll.mockClear()
  Element.prototype.scrollIntoView = scroll
})
afterEach(cleanup)

describe('finding support destinations', () => {
  it('opens and focuses the hidden fourth image region on a cold deep link, then allows collapse', () => {
    render(<MemoryRouter initialEntries={['/analyses/analysis-1/image-evidence?view=plain#image-evidence-I04']}>
      <ImageEvidenceContent set={images} token="" mediaReady={false} />
    </MemoryRouter>)
    const target = document.getElementById('image-evidence-I04')
    expect(target).not.toBeNull()
    expect(document.activeElement).toBe(target)
    expect(scroll.mock.contexts.at(-1)).toBe(target)
    expect(screen.getAllByTestId('image-evidence-card')).toHaveLength(4)
    fireEvent.click(screen.getByRole('button', { name: 'Show first 3 regions' }))
    expect(screen.getAllByTestId('image-evidence-card')).toHaveLength(3)
  })

  it('follows a new image hash and ignores a missing image packet', () => {
    render(<MemoryRouter initialEntries={['/analyses/analysis-1/image-evidence?view=plain']}>
      <ImageEvidenceContent set={images} token="" mediaReady={false} />
      <Link to="#image-evidence-I99">Missing region</Link>
      <Link to="#image-evidence-I04">Fourth region</Link>
    </MemoryRouter>)
    fireEvent.click(screen.getByRole('link', { name: 'Missing region' }))
    expect(screen.getAllByTestId('image-evidence-card')).toHaveLength(3)
    expect(scroll).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('link', { name: 'Fourth region' }))
    expect(screen.getAllByTestId('image-evidence-card')).toHaveLength(4)
    expect(document.activeElement).toBe(document.getElementById('image-evidence-I04'))
  })

  it('keeps the workspace scroll target on the exact image region', () => {
    render(<MemoryRouter initialEntries={['/analyses/analysis-1/image-evidence?view=plain#image-evidence-I04']}>
      <Routes><Route path="/analyses/:analysisId" element={<>
        <section id="section-image-evidence"><article id="image-evidence-I04">Recorded region</article></section>
        <Outlet context={{ ready: true }} />
      </>}><Route path="image-evidence" element={<SectionRoute section="image-evidence" />} /></Route></Routes>
    </MemoryRouter>)
    expect(scroll.mock.contexts.at(-1)).toBe(document.getElementById('image-evidence-I04'))
  })

  it('selects and focuses the linked document passage, preserving later manual choices across data refresh', () => {
    const view = (set: DocumentEvidenceSet) => <MemoryRouter initialEntries={['/analyses/analysis-1/media-evidence?view=plain#media-evidence-T02']}>
      <DocumentEvidenceContent set={set} token="" mediaReady={false} />
      <Link to="#media-evidence-T02">Linked passage</Link>
    </MemoryRouter>
    const { rerender } = render(view(passages))
    const first = screen.getByRole('button', { name: 'Inspect passage T01: Paragraph 1' })
    const second = screen.getByRole('button', { name: 'Inspect passage T02: Paragraph 2' })
    expect(second.getAttribute('aria-pressed')).toBe('true')
    expect(document.activeElement).toBe(document.getElementById('media-evidence-T02'))
    expect(scroll.mock.contexts.at(-1)).toBe(document.getElementById('media-evidence-T02'))
    fireEvent.click(first)
    rerender(view({ ...passages, packets: [...passages.packets] }))
    expect(first.getAttribute('aria-pressed')).toBe('true')
    expect(second.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(screen.getByRole('link', { name: 'Linked passage' }))
    expect(second.getAttribute('aria-pressed')).toBe('true')
  })

  it('does not change document selection or scroll for a missing packet', () => {
    render(<MemoryRouter initialEntries={['/analyses/analysis-1/media-evidence?view=plain#media-evidence-T99']}>
      <DocumentEvidenceContent set={passages} token="" mediaReady={false} />
      <Link to="#media-evidence-T99">Missing passage</Link>
    </MemoryRouter>)
    expect(screen.getByRole('button', { name: 'Inspect passage T01: Paragraph 1' }).getAttribute('aria-pressed')).toBe('true')
    expect(scroll).not.toHaveBeenCalled()
    const second = screen.getByRole('button', { name: 'Inspect passage T02: Paragraph 2' })
    fireEvent.click(second)
    scroll.mockClear()
    fireEvent.click(screen.getByRole('link', { name: 'Missing passage' }))
    expect(second.getAttribute('aria-pressed')).toBe('true')
    expect(scroll).not.toHaveBeenCalled()
  })
})
