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

it('shows the page and both views of the source PDF crop', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 20, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/1', format: 'pdf', coordinate_space: 'pdf_native_text_unicode_codepoint',
    earlier: source, later: source, scope: 'Words and punctuation.', interpretation: 'Source comparison only.',
    changes: [{ id: 'C01', role: 'body', earlier: null,
      later: { start: 4, end: 9, text: 'green', changed_tokens: [[4, 9]], pdf_pages: [{
        page: 2, page_width: 612, page_height: 792, page_rotation: 90,
        word_regions: [{ start: 4, end: 9, bbox: [20, 40, 72, 60] }], preview: {
          source_bbox: [2, 22, 90, 78], source_crop_pixels: [4, 44, 180, 156], render_scale: 2,
          crop_png: 'YQ==', highlighted_png: 'Yg==', crop_sha256: 'a'.repeat(64), highlighted_sha256: 'b'.repeat(64),
        },
      }] } }],
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.pdf', pairedCheckAvailable: false })
  vi.mocked(compareDraft).mockResolvedValue({ comparison })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  expect(input.getAttribute('accept')).toBe('.pdf')
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier.pdf')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByText('C01 · PDF text')
  expect(screen.getByText('Page 2 · highlighted source crop')).toBeTruthy()
  expect(screen.getByAltText('Analyzed version, page 2, with changed text highlighted').getAttribute('src')).toBe('data:image/png;base64,Yg==')
  expect(screen.getByAltText('Analyzed version, page 2, original crop').getAttribute('src')).toBe('data:image/png;base64,YQ==')
})

it('shows exact changes in Word table cells and explains skipped table coverage', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 30, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/5', format: 'docx', coordinate_space: 'docx_body_unicode_codepoint',
    earlier: source, later: source, scope: 'Matching Word tables.', interpretation: 'Source comparison only.', changes: [],
    word_tables: { coordinate_space: 'docx_table_cell_unicode_codepoint', omitted_positions: 0, compared: [{
      earlier_body_index: 1, later_body_index: 3, match: 'caption',
      unchecked_rows: { earlier: 1, later: 0 }, unchecked_columns: { earlier: 0, later: 0 },
      earlier_table: 1, later_table: 1, rows: 2, columns: 2,
      changes: [{ id: 'T01', row: 2, column: 2, earlier_row: 3, earlier_column: 1,
        earlier: { start: 1, end: 5, text: '🚀red', changed_tokens: [[2, 5]],
          paragraphs: [{ paragraph: 1, cell_index: 1, start: 1, end: 5 }] },
        later: { start: 1, end: 7, text: '🚀green', changed_tokens: [[2, 7]],
          paragraphs: [{ paragraph: 1, cell_index: 1, start: 1, end: 7 }] },
      }],
    }], skipped: [{ body_index: 3, earlier_table: 2, later_table: 2, reason: 'unsupported_layout' }] },
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.docx', pairedCheckAvailable: false })
  vi.mocked(compareDraft).mockResolvedValue({ comparison, pdf: 'JVBERi0xLjQK' })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier.docx')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByText('1 change found')
  expect(screen.getByText('T01 · Row 2, column 2')).toBeTruthy()
  expect(screen.getByText('Earlier cell · Row 3, column 1')).toBeTruthy()
  expect(screen.getByText('Analyzed cell · Row 2, column 2')).toBeTruthy()
  expect(screen.getByText('Unchecked rows. 1 earlier, 0 analyzed.')).toBeTruthy()
  expect(screen.getByText('Paragraph 1 [1, 7)')).toBeTruthy()
  expect(screen.getByText('green').tagName).toBe('MARK')
  expect(screen.getByText('green').closest('pre')?.textContent).toBe('🚀green')
  expect(screen.getByText(/Analyzed table 2.*Outside the supported table layout/)).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Download comparison JSON' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Download comparison PDF' })).toBeTruthy()
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
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.eml', pairedCheckAvailable: false })
  vi.mocked(compareDraft).mockResolvedValue({ comparison, pdf: 'JVBERi0xLjQK' })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  const file = new File(['earlier'], 'earlier.eml')
  fireEvent.change(input, { target: { files: [file] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByText('2 changes found')
  expect(compareDraft).toHaveBeenCalledWith('draft-analysis', file, expect.any(AbortSignal), false)
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

it('offers Word uploads and shows the server paragraph and body offsets', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 30, decoded_text_sha256: '1'.repeat(64), decoded_characters: 40, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/1', format: 'docx', coordinate_space: 'docx_body_unicode_codepoint',
    earlier: source, later: source, scope: 'Top-level body paragraphs.', interpretation: 'Compare the wording.',
    changes: [{ id: 'C01', role: 'body', earlier: { start: 20, end: 24, text: 'soup', changed_tokens: [[20, 24]],
      paragraphs: [{ paragraph: 2, body_index: 3, start: 1, end: 5 }] },
    later: { start: 20, end: 24, text: 'café', changed_tokens: [[20, 24]],
      paragraphs: [{ paragraph: 2, body_index: 3, start: 1, end: 5 }] } }],
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.docx', pairedCheckAvailable: false })
  vi.mocked(compareDraft).mockResolvedValue({ comparison })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  expect(input.getAttribute('accept')).toBe('.docx')
  const file = new File(['earlier'], 'earlier.DOCX')
  fireEvent.change(input, { target: { files: [file] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByText('C01 · Word body text')
  expect(compareDraft).toHaveBeenCalledWith('draft-analysis', file, expect.any(AbortSignal), false)
  expect(screen.getAllByText('Paragraph 2 [1, 5)')).toHaveLength(2)
  expect(screen.getAllByText('[20, 24)')).toHaveLength(2)
  expect(screen.getByText('café').tagName).toBe('MARK')
})

it('shows separate Word and PDF locations for an earlier draft in another format', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 20, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/3', format: 'mixed', coordinate_space: 'per_source_unicode_codepoint',
    earlier: { ...source, format: 'docx', coordinate_space: 'docx_body_unicode_codepoint' },
    later: { ...source, format: 'pdf', coordinate_space: 'pdf_native_text_unicode_codepoint' },
    scope: 'Words and punctuation.', interpretation: 'Source comparison only.',
    changes: [{ id: 'C01', role: 'body',
      earlier: { start: 4, end: 9, text: 'amber', changed_tokens: [[4, 9]], paragraphs: [{ paragraph: 2, body_index: 1, start: 0, end: 5 }] },
      later: { start: 4, end: 15, text: 'green\n\nblue', changed_tokens: [[4, 9], [11, 15]], pdf_pages: [
        { page: 1, page_width: 612, page_height: 792, page_rotation: 0, word_regions: [{ start: 4, end: 9, bbox: [20, 40, 72, 60] }] },
        { page: 2, page_width: 612, page_height: 792, page_rotation: 0, word_regions: [{ start: 11, end: 15, bbox: [20, 40, 68, 60] }] },
      ] } }],
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.pdf', earlierExtensions: ['.txt', '.docx', '.pdf'], pairedCheckAvailable: true })
  vi.mocked(compareDraft).mockResolvedValue({ comparison })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  expect(input.getAttribute('accept')).toBe('.txt,.docx,.pdf')
  fireEvent.click(screen.getByRole('checkbox', { name: /Include experimental paired/ }))
  const file = new File(['word'], 'earlier.docx')
  fireEvent.change(input, { target: { files: [file] } })
  expect(screen.queryByRole('checkbox', { name: /Include experimental paired/ })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  expect(await screen.findByText('C01 · Wording change')).toBeTruthy()
  expect(screen.getByText('Paragraph 2 [0, 5)')).toBeTruthy()
  expect(screen.getByText('Page 1')).toBeTruthy()
  expect(screen.getByText('Page 2')).toBeTruthy()
  expect(screen.getByText('Earlier draft · Word body text. Analyzed version · PDF text layer.')).toBeTruthy()
  expect(compareDraft).toHaveBeenCalledWith('draft-analysis', file, expect.any(AbortSignal), false)
})

it('links highlighted HTML reply words to their decoded MIME positions', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 400, decoded_text_sha256: '1'.repeat(64), decoded_characters: 80,
    decoded_html_sha256: '2'.repeat(64), decoded_html_characters: 140, mime_part: '1.1' }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/2', format: 'email', coordinate_space: 'email_html_extracted_text_unicode_codepoint',
    earlier: source, later: source, scope: 'Checks the reply body.', interpretation: 'Shows wording changes.',
    changes: [{ id: 'C01', role: 'body', earlier: null, later: {
      start: 12, end: 16, text: 'café', changed_tokens: [[12, 16]],
      html_tokens: [{ start: 12, end: 16, source_ranges: [[29, 31], [37, 38], [45, 51]] }],
    } }],
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.eml', pairedCheckAvailable: false })
  vi.mocked(compareDraft).mockResolvedValue({ comparison })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier.eml')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  expect((await screen.findByText('café')).tagName).toBe('MARK')
  expect(screen.getByText('Earlier MIME part 1.1 · Analyzed MIME part 1.1')).toBeTruthy()
  fireEvent.click(screen.getByText('Decoded HTML positions'))
  expect(screen.getByText('[12, 16) → [29, 31), [37, 38), [45, 51)')).toBeTruthy()
})

it.each([false, true])('opts in to the paired response and links the sentence locations, wide=%s', async (wide) => {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 100, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/1', format: 'plain_text', coordinate_space: 'decoded_source_unicode_codepoint',
    earlier: source, later: source, scope: 'Words and punctuation.', interpretation: 'Compare the wording.',
    changes: [{ id: 'C01', role: 'body', earlier: { start: 20, end: 24, text: 'here', changed_tokens: [[20, 24]] },
      later: { start: 20, end: 25, text: 'there', changed_tokens: [[20, 25]] } }],
    paired_check: { schema: wide ? 'chipmunk.paired-writing-review/2' : 'chipmunk.paired-writing-review/1', status: 'joint_increase', scope: 'Zero authorship confidence weight.',
      summary: 'Both writing checks gave a stronger AI-like response.', authorship_confidence_contribution: 0, change_ids: ['C01'],
      earlier: { start: 20, end: wide ? 31 : 25, text: wide ? 'Here. Then.' : 'Here.',
        context_spans: wide ? [[0, 19], [20, 25], [26, 31], [32, 46]] : [[0, 19], [20, 25], [26, 40]],
        input_sha256: 'a'.repeat(64) },
      later: { start: 20, end: wide ? 33 : 26, text: wide ? 'There. Later.' : 'There.',
        context_spans: wide ? [[0, 19], [20, 26], [27, 33], [34, 48]] : [[0, 19], [20, 26], [27, 41]],
        input_sha256: 'b'.repeat(64) },
      responses: [{ earlier: .1, later: .4 }, { earlier: .3, later: .6 }] },
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.txt', pairedCheckAvailable: true })
  vi.mocked(compareDraft).mockResolvedValue({ comparison })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  const option = screen.getByRole('checkbox', { name: 'Include experimental paired writing check' }) as HTMLInputElement
  expect(option.checked).toBe(false)
  fireEvent.click(option)
  const file = new File(['earlier'], 'earlier.txt')
  fireEvent.change(input, { target: { files: [file] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  expect(await screen.findByRole('region', { name: 'Experimental paired writing check' })).toBeTruthy()
  expect(compareDraft).toHaveBeenCalledWith('draft-analysis', file, expect.any(AbortSignal), true)
  const change = screen.getByRole('heading', { name: 'C01 · Text' })
  expect(screen.getByRole('link', { name: 'C01' }).getAttribute('href')).toBe(`#${change.id}`)
  expect(screen.getByText(wide ? 'Analyzed sentences [20, 33)' : 'Analyzed sentence [20, 26)')).toBeTruthy()
  if (wide) expect(screen.getByText('There. Later.')).toBeTruthy()
  expect(screen.queryByText(/ModernBERT|Tropa/)).toBeNull()
  expect(screen.getByText(/Check 2 · 0.3000 → 0.6000/)).toBeTruthy()
})
