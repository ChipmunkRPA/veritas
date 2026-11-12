// @vitest-environment jsdom
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import DocumentPassageCoverage from './DocumentPassageCoverage'
import type { DocumentPassageCoverage as Coverage } from './multimediaPackets'

describe('PDF page text coverage', () => {
  it('keeps unmapped pages and unchecked image areas distinct from local writing records', () => {
    const coverage: Coverage = {
      schema: 'document-local-passage-coverage/1',
      count_unit: 'non_whitespace_unicode_codepoint',
      mapped_characters: 42,
      counts: { response_range: 30, context_only: 4, no_local_record: 8 },
      ranges: [],
      omitted_ranges: 0,
      pdf_pages: {
        schema: 'document-pdf-page-coverage/1',
        page_count: 5,
        mapped_pages: [
          { page: 1, extraction: 'mixed', unchecked_image_areas: false,
            counts: { response_range: 30, context_only: 0, no_local_record: 0 } },
          { page: 4, extraction: 'text_layer', unchecked_image_areas: true,
            counts: { response_range: 0, context_only: 4, no_local_record: 8 } },
        ],
        unmapped_page_ranges: [[2, 3], [5, 5]],
      },
    }
    render(<DocumentPassageCoverage coverage={coverage} />)
    expect(screen.getByText('Text coverage by PDF page')).toBeTruthy()
    const page1 = screen.getByRole('row', { name: /Page 1/ })
    expect(within(page1).getByText('Selectable and recognized text')).toBeTruthy()
    expect(within(page1).getByText('30')).toBeTruthy()
    const gap = screen.getByRole('row', { name: /Pages 2–3/ })
    expect(within(gap).getByText('No mapped text')).toBeTruthy()
    expect(within(gap).getAllByText('—')).toHaveLength(3)
    const page4 = screen.getByRole('row', { name: /Page 4/ })
    expect(within(page4).getByText('Unchecked image areas')).toBeTruthy()
    expect(within(page4).getByText('0')).toBeTruthy()
  })
})
