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

it('shows only paired value changes and marks the source figure in each draft', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 41, mime_part: null }
  const excerpt = (value: string) => ({ start: 10, end: 30 + value.length,
    text: `budget for work was ${value}`, changed_tokens: [[30, 30 + value.length]] as [number, number][] })
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/1', format: 'plain_text',
    coordinate_space: 'decoded_source_unicode_codepoint',
    earlier: source, later: source, changes: [], scope: 'Body text.', interpretation: 'Source comparison.',
    numeric_comparison: {
      schema: 'chipmunk.draft-value-comparison/1', scope: 'Matched wording.', affects_overall_confidence: false,
      compared: 2, differences: 1, changes: [{ kind: 'currency', earlier: excerpt('$800'), later: excerpt('$920'),
        earlier_value: '$800', later_value: '$920', earlier_range: [30, 34], later_range: [30, 34] }],
    },
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.txt', pairedCheckAvailable: false })
  vi.mocked(compareDraft).mockResolvedValue({ comparison })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier.txt')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByText('Amount changed · $800 → $920')
  expect(screen.getByText('2 matched · 1 changed')).toBeTruthy()
  expect(screen.getByText('$800').tagName).toBe('MARK')
  expect(screen.getByText('$920').tagName).toBe('MARK')
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

it('shows the OpenDocument paragraph and retains exact offsets when comparing it with Word', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 33, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/3', format: 'mixed', coordinate_space: 'per_source_unicode_codepoint',
    earlier: { ...source, format: 'odt', coordinate_space: 'odt_body_unicode_codepoint' },
    later: { ...source, format: 'docx', coordinate_space: 'docx_body_unicode_codepoint' },
    changes: [{ id: 'C01', role: 'body',
      earlier: { start: 20, end: 26, text: 'Monday', changed_tokens: [[20, 26]],
        paragraphs: [{ paragraph: 2, body_index: 1, start: 5, end: 11 }] },
      later: { start: 20, end: 28, text: 'Thursday', changed_tokens: [[20, 28]],
        paragraphs: [{ paragraph: 2, body_index: 1, start: 5, end: 13 }] } }],
    scope: 'Body wording.', interpretation: 'Source comparison.',
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.docx', earlierExtensions: ['.txt', '.docx', '.pdf', '.odt'],
    pairedCheckAvailable: true, pairedMixedFormatsAvailable: true })
  vi.mocked(compareDraft).mockResolvedValue({ comparison, pdf: 'JVBERi0xLjQK' })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  expect(input.getAttribute('accept')).toContain('.odt')
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier.odt')] } })
  expect(screen.queryByRole('checkbox')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByText(/Earlier draft · OpenDocument body text/)
  expect(screen.getByText('Paragraph 2 [5, 11)')).toBeTruthy()
  expect(screen.getByText('Monday').tagName).toBe('MARK')
  expect(screen.getByText('Thursday').tagName).toBe('MARK')
  expect(vi.mocked(compareDraft).mock.calls[0][3]).toBe(false)
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

it('shows both locations of a reordered body passage and exports the exact comparison', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 588, decoded_text_sha256: '1'.repeat(64), decoded_characters: 240, mime_part: null }
  const text = 'The archivist retained café notes 🦊 for the annual public records review.'
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/6', format: 'docx', coordinate_space: 'docx_body_unicode_codepoint',
    earlier: source, later: source, scope: 'The passage changed its relative position.', interpretation: 'Source comparison only.',
    changes: [], relocated_passages: [{
      earlier: { start: 23, end: 95, text, changed_tokens: [], paragraphs: [{ paragraph: 2, body_index: 1, start: 0, end: 72 }] },
      later: { start: 116, end: 188, text, changed_tokens: [], paragraphs: [{ paragraph: 4, body_index: 3, start: 0, end: 72 }] },
    }],
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.docx', pairedCheckAvailable: false })
  vi.mocked(compareDraft).mockResolvedValue({ comparison, pdf: 'JVBERi0xLjQK' })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier.docx')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByRole('heading', { name: 'Reordered body passage' })
  expect(screen.getByRole('heading', { name: 'Earlier draft' })).toBeTruthy()
  expect(screen.getByRole('heading', { name: 'Analyzed version' })).toBeTruthy()
  expect(screen.getByText('[23, 95)')).toBeTruthy()
  expect(screen.getByText('[116, 188)')).toBeTruthy()
  expect(screen.getByText('Paragraph 2 [0, 72)')).toBeTruthy()
  expect(screen.getByText('Paragraph 4 [0, 72)')).toBeTruthy()
  expect(screen.getAllByText(text)).toHaveLength(2)
  expect(screen.getAllByText(text).every(excerpt => !excerpt.querySelector('mark'))).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Download comparison JSON' }))
  const file = vi.mocked(saveBlob).mock.calls[0][0]
  const exported = await new Promise<string>((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.readAsText(file)
  })
  expect(JSON.parse(exported)).toEqual(comparison)
})

it('highlights wording changes within a reordered body passage', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 100, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/8', format: 'plain_text', coordinate_space: 'decoded_source_unicode_codepoint',
    earlier: source, later: source, scope: 'Compare wording within the reordered passage.', interpretation: 'Source comparison only.',
    changes: [], relocated_passages: [{
      earlier: { start: 24, end: 41, text: 'Notes recorded 6.', changed_tokens: [[39, 40]] },
      later: { start: 64, end: 81, text: 'Notes recorded 8.', changed_tokens: [[79, 80]] },
    }],
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.txt', pairedCheckAvailable: false })
  vi.mocked(compareDraft).mockResolvedValue({ comparison, pdf: 'JVBERi0xLjQK' })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier.txt')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByRole('heading', { name: 'Reordered and edited body passage' })
  expect(screen.getByRole('status').textContent).toContain('One body passage was reordered and edited.')
  expect(screen.getByText('6').tagName).toBe('MARK')
  expect(screen.getByText('8').tagName).toBe('MARK')
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

it('requests the paired writing check across formats when the server enables it', async () => {
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64),
    decoded_characters: 20, mime_part: null }
  vi.mocked(draftAvailability).mockResolvedValue({
    extension: '.pdf', earlierExtensions: ['.txt', '.docx', '.pdf'],
    pairedCheckAvailable: true, pairedMixedFormatsAvailable: true,
  })
  vi.mocked(compareDraft).mockResolvedValue({ comparison: {
    schema: 'chipmunk.document-revision-comparison/3', format: 'mixed',
    coordinate_space: 'per_source_unicode_codepoint',
    earlier: { ...source, format: 'docx' }, later: { ...source, format: 'pdf' },
    changes: [], scope: 'Words and punctuation.', interpretation: 'Source comparison only.',
    paired_check: { schema: 'chipmunk.paired-writing-review/11', status: 'outside_scope',
      summary: 'No changed sentence could be matched in both drafts.', scope: 'Only verified source positions are checked.',
      authorship_confidence_contribution: 0 },
  } })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  const file = new File(['word'], 'earlier.DOCX')
  fireEvent.change(input, { target: { files: [file] } })
  fireEvent.click(screen.getByRole('checkbox', { name: /Include experimental paired/ }))
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  await screen.findByText('No changed sentence could be matched in both drafts.')
  expect(compareDraft).toHaveBeenCalledWith('draft-analysis', file, expect.any(AbortSignal), true)
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

it.each(['single', 'wide', 'interior'] as const)('opts in to the %s paired response and links the sentence locations', async (mode) => {
  const wide = mode === 'wide'
  const interior = mode === 'interior'
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 100, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/1', format: 'plain_text', coordinate_space: 'decoded_source_unicode_codepoint',
    earlier: source, later: source, scope: 'Words and punctuation.', interpretation: 'Compare the wording.',
    changes: [{ id: 'C01', role: 'body', earlier: { start: 20, end: 24, text: 'here', changed_tokens: [[20, 24]] },
      later: { start: 20, end: 25, text: 'there', changed_tokens: [[20, 25]] } }],
    paired_check: { schema: interior ? 'chipmunk.paired-writing-review/5' : wide ? 'chipmunk.paired-writing-review/2' : 'chipmunk.paired-writing-review/1', status: 'joint_increase', scope: 'Zero authorship confidence weight.',
      summary: 'Both writing checks gave a stronger AI-like response.', authorship_confidence_contribution: 0, change_ids: ['C01'],
      earlier: { start: 20, end: wide ? 31 : 25, text: wide ? 'Here. Then.' : 'Here.',
        context_spans: wide ? [[0, 19], [20, 25], [26, 31], [32, 46]] : [[0, 19], [20, 25], [26, 40]],
        input_sha256: 'a'.repeat(64) },
      later: { start: 20, end: wide ? 33 : 26, text: wide ? 'There. Later.' : 'There.',
        context_spans: wide ? [[0, 19], [20, 26], [27, 33], [34, 48]] : [[0, 19], [20, 26], [27, 41]],
        input_sha256: 'b'.repeat(64) },
      responses: [{ earlier: .1, later: .4 }, { earlier: .3, later: .6 }],
      ...(interior ? { neighboring_contexts: [{ status: 'joint_increase', responses: [{ earlier: .1, later: .7 }, { earlier: .2, later: .6 }],
        earlier: { start: 20, end: 61, text: 'Here. Next sentence. Last sentence.',
          context_spans: [[20, 25], [26, 40], [41, 61]], input_sha256: 'c'.repeat(64) },
        later: { start: 20, end: 62, text: 'There. Next sentence. Last sentence.',
          context_spans: [[20, 26], [27, 41], [42, 62]], input_sha256: 'd'.repeat(64) } }] } : {}) },
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
  if (interior) {
    expect(screen.getByRole('heading', { name: 'Additional scored contexts' })).toBeTruthy()
    expect(screen.getByText('Analyzed scored context [20, 62)')).toBeTruthy()
    expect(screen.getByText('There. Next sentence. Last sentence.')).toBeTruthy()
  }
  expect(screen.queryByText(/ModernBERT|Tropa/)).toBeNull()
  expect(screen.getByText(interior ? /Check 2 · Centered context 0.3000 → 0.6000/ : /Check 2 · 0.3000 → 0.6000/)).toBeTruthy()
})

it('shows separate revision outcomes with links to the correct wording changes', async () => {
  type Revision = NonNullable<NonNullable<DraftComparison['paired_check']>['revisions']>[number]
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 100, mime_part: null }
  const first: Revision['later'] = {
    start: 20, end: 26, text: 'First.', context_spans: [[0, 19], [20, 26], [27, 40]], input_sha256: 'a'.repeat(64),
  }
  const second: Revision['later'] = {
    start: 60, end: 67, text: 'Second.', context_spans: [[42, 59], [60, 67], [68, 90]], input_sha256: 'b'.repeat(64),
  }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/1', format: 'plain_text', coordinate_space: 'decoded_source_unicode_codepoint',
    earlier: source, later: source, scope: 'Words and punctuation.', interpretation: 'Shows wording changes.',
    changes: [{ id: 'C01', role: 'body', earlier: null, later: { ...first, changed_tokens: [[20, 25]] } },
      { id: 'C02', role: 'body', earlier: null, later: { ...second, changed_tokens: [[60, 66]] } }],
    paired_check: {
      schema: 'chipmunk.paired-writing-review/6', status: 'joint_increase', scope: 'Zero authorship confidence weight.',
      summary: 'Checked 1 of 2 mapped revisions.', authorship_confidence_contribution: 0,
      revisions: [{ earlier: first, later: first, status: 'joint_increase', change_ids: ['C01'],
        responses: [{ earlier: .1, later: .4 }, { earlier: .3, later: .6 }] },
      { earlier: second, later: second, status: 'unavailable', change_ids: ['C02'] }],
    },
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension: '.txt', pairedCheckAvailable: true })
  vi.mocked(compareDraft).mockResolvedValue({ comparison })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  fireEvent.click(screen.getByRole('checkbox', { name: 'Include experimental paired writing check' }))
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier.txt')] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  expect(await screen.findByText('Analyzed sentence [20, 26)')).toBeTruthy()
  expect(screen.getByText('Analyzed sentence [60, 67)')).toBeTruthy()
  expect(screen.getByText('Both writing checks gave a stronger AI-like response in the analyzed version.')).toBeTruthy()
  expect(screen.getByText('This mapped revision could not be scored.')).toBeTruthy()
  for (const id of ['C01', 'C02']) {
    const change = screen.getByRole('heading', { name: `${id} · Text` })
    expect(screen.getByRole('link', { name: id }).getAttribute('href')).toBe(`#${change.id}`)
  }
})

it.each([
  ['plain text', '.txt', false, false], ['Word', '.docx', false, false],
  ['PDF', '.pdf', false, false], ['split PDF', '.pdf', true, false],
  ['sentence split in Word', '.docx', false, true],
] as const)('shows wording changes and native scope for %s', async (_layout, extension, continued, split) => {
  const word = extension === '.docx'
  const pdf = extension === '.pdf'
  const page = (number: number) => ({ page: number, page_width: 612, page_height: 792, page_rotation: 0, word_regions: [] })
  const source = { sha256: '0'.repeat(64), bytes: 100, decoded_text_sha256: '1'.repeat(64), decoded_characters: 100, mime_part: null }
  const comparison: DraftComparison = {
    schema: 'chipmunk.document-revision-comparison/1', format: pdf ? 'pdf' : word ? 'docx' : 'plain_text',
    coordinate_space: pdf ? 'pdf_native_text_unicode_codepoint' : word ? 'docx_body_unicode_codepoint' : 'decoded_source_unicode_codepoint',
    earlier: source, later: source, scope: 'Compares the wording.', interpretation: 'Shows wording changes.', changes: [],
    paired_check: {
      schema: split ? 'chipmunk.paired-writing-review/12'
        : continued ? 'chipmunk.paired-writing-review/10' : pdf ? 'chipmunk.paired-writing-review/9'
        : word ? 'chipmunk.paired-writing-review/8' : 'chipmunk.paired-writing-review/7',
      status: 'joint_increase', authorship_confidence_contribution: 0,
      summary: 'Checked 1 of 1 mapped revisions.', scope: 'Other edits may remain unchecked.',
      revisions: [{
        earlier: { start: 20, end: 31, text: 'Early rain.', context_spans: [[0, 19], [20, 31], [32, 40]],
          input_sha256: 'a'.repeat(64),
          ...(pdf && { pdf_pages: [page(1)] }),
          ...(word && { context_paragraphs: [[0, 19], [20, 31], [32, 40]].map(([start, end]) => ({
            paragraph: 1, body_index: 0, start, end,
          })) }) },
        later: { start: 23, end: split ? 41 : 33, text: split ? 'Heavy fog. It held.' : 'Heavy fog.',
          context_spans: split ? [[3, 22], [23, 33], [34, 41], [42, 50]] : [[3, 22], [23, 33], [34, 42]],
          input_sha256: 'b'.repeat(64),
          ...(pdf && { pdf_pages: continued ? [page(2), page(3)] : [page(2)] }),
          ...(word && { context_paragraphs: (split ? [[0, 19], [20, 30], [31, 38], [39, 47]]
            : [[0, 19], [20, 30], [31, 39]]).map(([start, end]) => ({
            paragraph: 2, body_index: 1, start, end,
          })) }) },
        wording_changes: [{ earlier: [20, 30], later: [23, 32],
          earlier_changed_tokens: [[20, 25], [26, 30]], later_changed_tokens: [[23, 28], [29, 32]] }],
        responses: [{ earlier: .2, later: .5 }, { earlier: .3, later: .8 }], status: 'joint_increase',
      }],
    },
  }
  vi.mocked(draftAvailability).mockResolvedValue({ extension, pairedCheckAvailable: true })
  vi.mocked(compareDraft).mockResolvedValue({ comparison })
  show()
  const input = await screen.findByLabelText('Earlier draft')
  fireEvent.click(screen.getByRole('checkbox', { name: 'Include experimental paired writing check' }))
  fireEvent.change(input, { target: { files: [new File(['earlier'], 'earlier' + extension)] } })
  fireEvent.click(screen.getByRole('button', { name: 'Compare drafts' }))
  expect(await screen.findByText('Other edits may remain unchecked.')).toBeTruthy()
  expect(screen.getByText(split ? 'Analyzed sentences [23, 41)' : 'Analyzed sentence [23, 33)')).toBeTruthy()
  if (split) expect(screen.getByRole('heading', { name: 'Changed passage 1' })).toBeTruthy()
  if (word) {
    expect(screen.getByText('Paragraph 1 [20, 31)')).toBeTruthy()
    expect(screen.getByText(split ? 'Paragraph 2 [20, 38)' : 'Paragraph 2 [20, 30)')).toBeTruthy()
  }
  if (pdf) {
    expect(screen.getByText('PDF page 1')).toBeTruthy()
    expect(screen.getByText('PDF page 2')).toBeTruthy()
  }
  if (continued) expect(screen.getByText('PDF page 3')).toBeTruthy()
  for (const word of ['Early', 'rain', 'Heavy', 'fog']) {
    expect(screen.getByText(word).tagName).toBe('MARK')
  }
})
