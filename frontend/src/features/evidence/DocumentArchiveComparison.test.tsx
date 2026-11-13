// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import DocumentArchiveComparison from './DocumentArchiveComparison'
import { archiveAvailability, compareArchive, type ArchiveComparison } from './documentArchiveApi'
import { saveBlob } from '../../lib/api/reports'

vi.mock('./documentArchiveApi', () => ({ archiveAvailability: vi.fn(), compareArchive: vi.fn() }))
vi.mock('../../lib/api/reports', () => ({ saveBlob: vi.fn() }))

const edited: ArchiveComparison = {
  pdf: 'JVBERi0=',
  comparison: {
    schema: 'chipmunk.document-archive-comparison/1',
    status: 'reference_bound',
    coordinate_space: 'extracted_text_unicode_codepoint',
    layout: 'recovered_archive_boundaries',
    scope: 'Compares aligned archive passages.',
    interpretation: 'This comparison carries zero AI-assessment weight.',
    authorship_confidence_contribution: 0,
    submitted_sha256: 'a'.repeat(64),
    extracted_text_sha256: 'b'.repeat(64),
    library: {
      title: 'Foreign Relations of the United States',
      publisher: 'Office of the Historian, U.S. Department of State',
      url: 'https://history.state.gov/historicaldocuments',
      license: 'CC0-1.0',
      license_url: 'https://github.com/HistoryAtState/frus/blob/master/LICENSE.md',
      revision: 'a'.repeat(40),
      sha256: 'c'.repeat(64),
      indexed_excerpts: 118,
    },
    coverage: { body_paragraphs: 3, body_words: 210, matched_words: 202 },
    reference: {
      id: 'frus1964-68v24/d585',
      url: 'https://history.state.gov/historicaldocuments/frus1964-68v24/d585',
      version_url: 'https://github.com/HistoryAtState/frus/blob/revision/volumes/frus1964-68v24.xml',
      volume: 'frus1964-68v24',
      document: 'd585',
    },
    aligned_paragraphs: [{ submitted_paragraph: 1, submitted_range: [10, 16], archive_paragraph: 4, alignment: 'fingerprint_anchor' }],
    changes: [{
      submitted_paragraph: 1, archive_paragraph: 4, alignment: 'fingerprint_anchor',
      archive: { start: 2, end: 5, text: 'tea', changed_tokens: [[2, 5]] },
      submitted: { start: 10, end: 16, text: '🧪 café', changed_tokens: [[12, 16]] },
    }],
  },
}

function show() {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <DocumentArchiveComparison analysisId="archive-fixture" />
  </QueryClientProvider>)
}

describe('Public archive comparison', () => {
  afterEach(() => {
    cleanup()
    vi.resetAllMocks()
  })

  it('loads on demand, highlights exact Unicode ranges and exports both formats', async () => {
    vi.mocked(archiveAvailability).mockResolvedValue({ gaoAvailable: false })
    vi.mocked(compareArchive).mockResolvedValue(edited)
    show()
    const start = await screen.findByRole('button', { name: 'Compare with public archive' })
    expect(compareArchive).not.toHaveBeenCalled()
    fireEvent.click(start)
    expect(await screen.findByText('Archive passage matched')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Source frus1964-68v24/d585' })).toHaveAttribute('href', edited.comparison.reference!.url)
    expect(screen.getByText('café').tagName).toBe('MARK')
    expect(screen.getByText('[10, 16)')).toBeVisible()
    expect(screen.getByText(/recovered their boundaries from the archive/)).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Passage in submitted paragraph 1' })).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Download archive JSON' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download archive PDF' }))
    expect(saveBlob).toHaveBeenNthCalledWith(1, expect.any(Blob), 'chipmunk-verify-archive-comparison.json')
    expect(saveBlob).toHaveBeenNthCalledWith(2, expect.any(Blob), 'chipmunk-verify-archive-comparison.pdf')
  })

  it('presents an archive no-match without a source attribution', async () => {
    vi.mocked(archiveAvailability).mockResolvedValue({ gaoAvailable: false })
    vi.mocked(compareArchive).mockResolvedValue({
      pdf: null,
      comparison: { ...edited.comparison, status: 'no_qualified_reference', reference: null, changes: [], aligned_paragraphs: [] },
    })
    show()
    fireEvent.click(await screen.findByRole('button', { name: 'Compare with public archive' }))
    expect(await screen.findByText('No matching passage in this archive library')).toBeVisible()
    expect(screen.queryByRole('heading', { name: /submitted paragraph 1/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /Source frus/ })).not.toBeInTheDocument()
  })

  it('shows the GAO transcript and changed sentence positions separately', async () => {
    const result = structuredClone(edited)
    result.comparison.schema = 'chipmunk.document-archive-comparison/2'
    result.comparison.layout = 'submitted_paragraphs'
    result.comparison.reference = {
      id: 'GAO-16-442', url: 'https://www.gao.gov/products/GAO-16-442', version_url: 'https://huggingface.co/datasets/launch/gov_report',
      volume: 'GAO highlights', document: 'GAO-16-442', published_date: '2016-04-18',
      submitted_range: [0, 180], transcript_range: [0, 176],
    }
    result.comparison.changes[0].archive_paragraph = null
    result.comparison.submitted_sentence_ranges = [[6, 41]]
    result.comparison.aligned_paragraphs[0].docx_location = {
      paragraph: 2, body_index: 1, paragraph_range: [2, 176], changed_tokens: [], runs_complete: true,
    }
    vi.mocked(archiveAvailability).mockResolvedValue({ gaoAvailable: true })
    vi.mocked(compareArchive).mockResolvedValue(result)
    show()
    const collection = await screen.findByLabelText('Archive collection')
    expect(collection).toHaveValue('frus')
    fireEvent.change(collection, { target: { value: 'gao' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compare with public archive' }))
    expect(await screen.findByRole('heading', { name: 'Published highlights transcript' })).toBeVisible()
    expect(screen.getByText(/Matched submitted range \[0, 180\).*Transcript range \[0, 176\)/)).toBeVisible()
    expect(screen.getByText(/Submitted sentences containing differences \[6, 41\)/)).toBeVisible()
    expect(screen.getByRole('list', { name: 'Matched Word paragraphs' })).toHaveTextContent('Word body paragraph 2, characters [2, 176).')
    expect(screen.getByRole('link', { name: 'Source GAO-16-442' })).toHaveAttribute('href', result.comparison.reference.url)
    expect(screen.getByText('café').tagName).toBe('MARK')
    expect(compareArchive).toHaveBeenCalledWith('archive-fixture', expect.any(AbortSignal), 'gao')
  })

  it('shows the changed Word paragraph and its original text range', async () => {
    const result = structuredClone(edited)
    result.comparison.layout = 'submitted_paragraphs'
    result.comparison.changes[0].submitted.docx_location = {
      paragraph: 3,
      body_index: 2,
      paragraph_range: [20, 26],
      runs_complete: true,
      changed_tokens: [{ extracted_range: [12, 16], paragraph_range: [22, 26],
        runs: [{ run: 1, range: [2, 4] }, { run: 2, range: [0, 2] }] }],
    }
    vi.mocked(archiveAvailability).mockResolvedValue({ gaoAvailable: false })
    vi.mocked(compareArchive).mockResolvedValue(result)
    show()
    fireEvent.click(await screen.findByRole('button', { name: 'Compare with public archive' }))
    expect(await screen.findByRole('heading', { name: 'Word body paragraph 3' })).toBeVisible()
    expect(screen.getByText('Word paragraph characters [20, 26)')).toBeVisible()
    expect(screen.getByText('café').tagName).toBe('MARK')
  })

  it('shows both native positions when an archive change spans Word paragraphs', async () => {
    const result = structuredClone(edited)
    result.comparison.schema = 'chipmunk.document-archive-comparison/2'
    result.comparison.layout = 'submitted_paragraphs'
    result.comparison.changes[0].submitted = {
      start: 10, end: 21, text: 'left\n\nright', changed_tokens: [[10, 14], [16, 21]],
      docx_locations: [
        { paragraph: 3, body_index: 2, paragraph_range: [8, 12], runs_complete: true, changed_tokens: [] },
        { paragraph: 4, body_index: 3, paragraph_range: [0, 5], runs_complete: true, changed_tokens: [] },
      ],
    }
    vi.mocked(archiveAvailability).mockResolvedValue({ gaoAvailable: true })
    vi.mocked(compareArchive).mockResolvedValue(result)
    show()
    const collection = await screen.findByLabelText('Archive collection')
    fireEvent.change(collection, { target: { value: 'gao' } })
    fireEvent.click(screen.getByRole('button', { name: 'Compare with public archive' }))
    expect(await screen.findByRole('heading', { name: 'Change across Word paragraphs' })).toBeVisible()
    const positions = screen.getByRole('list', { name: 'Changed Word positions' })
    expect(positions).toHaveTextContent('Word body paragraph 3, characters [8, 12).')
    expect(positions).toHaveTextContent('Word body paragraph 4, characters [0, 5).')
    expect(screen.getByText('left').tagName).toBe('MARK')
    expect(screen.getByText('right').tagName).toBe('MARK')
  })

  it('shows the matched PDF page and the exact original and highlighted crops', async () => {
    const result = structuredClone(edited)
    result.comparison.aligned_paragraphs[0].pdf_pages = [{ page: 2, extracted_range: [10, 16] }]
    result.comparison.changes[0].submitted.pdf_pages = [{
      page: 2, page_width: 612, page_height: 792, page_rotation: 0,
      word_regions: [{ start: 12, end: 16, bbox: [10, 20, 40, 30] }],
      preview: { source_bbox: [4, 14, 46, 36], render_scale: 1, source_crop_pixels: [4, 14, 46, 36],
        crop_png: 'b3JpZ2luYWw=', crop_sha256: 'a'.repeat(64), highlighted_png: 'aGlnaGxpZ2h0ZWQ=', highlighted_sha256: 'b'.repeat(64) },
    }]
    vi.mocked(archiveAvailability).mockResolvedValue({ gaoAvailable: true })
    vi.mocked(compareArchive).mockResolvedValue(result)
    show()
    fireEvent.click(await screen.findByRole('button', { name: 'Compare with public archive' }))
    expect(await screen.findByRole('list', { name: 'Matched PDF pages' })).toHaveTextContent('Page 2, extracted characters [10, 16).')
    expect(screen.getByRole('heading', { name: 'Submitted PDF · page 2' })).toBeVisible()
    expect(screen.getByRole('img', { name: 'Page 2 with changed wording highlighted' })).toHaveAttribute('src', 'data:image/png;base64,aGlnaGxpZ2h0ZWQ=')
    const original = screen.getByText('View original crop')
    fireEvent.click(original)
    expect(screen.getByAltText('Page 2 original crop')).toHaveAttribute('src', 'data:image/png;base64,b3JpZ2luYWw=')
  })

  it('hides the action when the retained owner is ineligible', async () => {
    vi.mocked(archiveAvailability).mockResolvedValue(null)
    show()
    await waitFor(() => expect(archiveAvailability).toHaveBeenCalledOnce())
    expect(screen.queryByRole('button', { name: 'Compare with public archive' })).not.toBeInTheDocument()
    expect(compareArchive).not.toHaveBeenCalled()
  })
})
