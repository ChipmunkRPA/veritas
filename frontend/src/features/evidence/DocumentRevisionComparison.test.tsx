// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { saveBlob } from '../../lib/api/reports'
import DocumentRevisionComparison from './DocumentRevisionComparison'
import { compareDraft, draftAvailability, type DraftComparison } from './documentRevisionApi'

vi.mock('./documentRevisionApi', () => ({ draftAvailability: vi.fn(), compareDraft: vi.fn() }))
vi.mock('../../lib/api/reports', () => ({ saveBlob: vi.fn() }))

afterEach(() => {
  cleanup()
  vi.resetAllMocks()
})

function show() {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <DocumentRevisionComparison analysisId="draft-analysis" />
  </QueryClientProvider>)
}

it('keeps the form hidden when the authenticated comparison is unavailable', async () => {
  vi.mocked(draftAvailability).mockResolvedValue(null)
  show()
  await waitFor(() => expect(draftAvailability).toHaveBeenCalledTimes(1))
  expect(screen.queryByText('Compare with an earlier draft')).toBeNull()
})

it('displays the exact server excerpts and Unicode ranges and downloads both reports', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 20, decoded_text_sha256: '1'.repeat(64), decoded_characters: 40, mime_part: '1.1' }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/1', format: 'email', coordinate_space: 'decoded_source_unicode_codepoint',
    earlier: source, later: source, scope: 'Words and punctuation.', interpretation: 'The revision sequence requires independent evidence.',
    changes: [
      { id: 'C01', role: 'body', earlier: { start: 20, end: 29, text: '🚀 Déjà vu', changed_tokens: [[20, 21], [27, 29]] },
        later: { start: 20, end: 33, text: '🙂 Déjà brûlée', changed_tokens: [[20, 21], [27, 33]] } },
      { id: 'C02', role: 'quoted', earlier: null, later: { start: 35, end: 40, text: 'quote', changed_tokens: [[35, 40]] } },
    ],
  }
  vi.mocked(draftAvailability).mockResolvedValue('.eml')
  vi.mocked(compareDraft).mockResolvedValue({ comparison, pdf: 'JVBERi0xLjQK' })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  const file = new File(['earlier'], 'earlier.eml')
  fireEvent.change(input, { target: { files: [file] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByText('2 changes found')
  expect(compareDraft).toHaveBeenCalledWith('draft-analysis', file, expect.any(AbortSignal))
  for (const exact of ['[20, 29)', '[20, 33)', 'C01 · Message body', 'C02 · Quoted text']) {
    expect(screen.getByText(exact)).toBeTruthy()
  }
  const earlier = screen.getByText('🚀').closest('pre')
  const later = screen.getByText('🙂').closest('pre')
  expect(earlier?.textContent).toBe('🚀 Déjà vu')
  expect(later?.textContent).toBe('🙂 Déjà brûlée')
  expect([...earlier!.querySelectorAll('mark')].map(mark => mark.textContent)).toEqual(['🚀', 'vu'])
  expect([...later!.querySelectorAll('mark')].map(mark => mark.textContent)).toEqual(['🙂', 'brûlée'])
  fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
  fireEvent.click(screen.getByRole('button', { name: 'Download comparison PDF' }))
  const [[json, jsonName], [pdf, pdfName]] = vi.mocked(saveBlob).mock.calls
  expect([jsonName, pdfName]).toEqual(['chipmunk-verify-draft-comparison.json', 'chipmunk-verify-draft-comparison.pdf'])
  expect(pdf.type).toBe('application/pdf')
  const exported = await new Promise<string>(resolve => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.readAsText(json)
  })
  expect(JSON.parse(exported)).toEqual(comparison)
})
