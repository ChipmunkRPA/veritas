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
    vi.mocked(archiveAvailability).mockResolvedValue(true)
    vi.mocked(compareArchive).mockResolvedValue(edited)
    show()
    const start = await screen.findByRole('button', { name: 'Compare with public archive' })
    expect(compareArchive).not.toHaveBeenCalled()
    fireEvent.click(start)
    expect(await screen.findByText('Archive passage matched')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Source frus1964-68v24/d585' })).toHaveAttribute('href', edited.comparison.reference!.url)
    expect(screen.getByText('café').tagName).toBe('MARK')
    expect(screen.getByText('[10, 16)')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Download archive JSON' }))
    fireEvent.click(screen.getByRole('button', { name: 'Download archive PDF' }))
    expect(saveBlob).toHaveBeenNthCalledWith(1, expect.any(Blob), 'chipmunk-verify-archive-comparison.json')
    expect(saveBlob).toHaveBeenNthCalledWith(2, expect.any(Blob), 'chipmunk-verify-archive-comparison.pdf')
  })

  it('presents an archive no-match without a source attribution', async () => {
    vi.mocked(archiveAvailability).mockResolvedValue(true)
    vi.mocked(compareArchive).mockResolvedValue({
      pdf: null,
      comparison: { ...edited.comparison, status: 'no_qualified_reference', reference: null, changes: [], aligned_paragraphs: [] },
    })
    show()
    fireEvent.click(await screen.findByRole('button', { name: 'Compare with public archive' }))
    expect(await screen.findByText('No matching passage in this archive library')).toBeVisible()
    expect(screen.queryByText('Submitted paragraph 1')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /Source frus/ })).not.toBeInTheDocument()
  })

  it('hides the action when the retained owner is ineligible', async () => {
    vi.mocked(archiveAvailability).mockResolvedValue(false)
    show()
    await waitFor(() => expect(archiveAvailability).toHaveBeenCalledOnce())
    expect(screen.queryByRole('button', { name: 'Compare with public archive' })).not.toBeInTheDocument()
    expect(compareArchive).not.toHaveBeenCalled()
  })
})
