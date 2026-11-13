// @vitest-environment jsdom

import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import DocumentUnselectedResponses from './DocumentUnselectedResponses'
import type { DocumentUnselectedResponses as Responses } from './multimediaPackets'

describe('additional recorded writing responses', () => {
  it('shows the measured input and its selectable and recognized source ranges', () => {
    const responses: Responses = {
      schema: 'document-unselected-passage-responses/1', affects_overall_confidence: false,
      eligible_windows: 2, omitted_windows: 1,
      rows: [{ id: 'U01', focus: { start: 6, end: 9 }, spans: [{ start: 0, end: 4 }, { start: 6, end: 9 }],
        sources: [{ kind: 'pdf', page: 1, extraction: 'text_layer' }, { kind: 'pdf', page: 4, extraction: 'ocr' }],
        check: 'S01', input_sha256: 'a'.repeat(64), direction: 'opposes_ai', response: .123, text: 'Word end' }],
    }
    const { rerender } = render(<DocumentUnselectedResponses responses={responses} />)
    const region = screen.getByRole('region', { name: 'Additional recorded passage responses' })
    const response = within(region).getByText('Response leaned away from AI in this window', { exact: false })
    fireEvent.click(response.closest('summary')!)
    expect(within(region).getByText('Word end')).toBeTruthy()
    expect(within(region).getByText(/Page 1 · selectable text/)).toBeTruthy()
    expect(within(region).getByText(/Page 4 · recognized text/)).toBeTruthy()
    expect(within(region).getByText('[6, 9)')).toBeTruthy()
    expect(within(region).getByText(/adds no weight/)).toBeTruthy()
    expect(within(region).getByText(/omitted 1 additional eligible responses/)).toBeTruthy()
    rerender(<DocumentUnselectedResponses responses={undefined} />)
    expect(screen.queryByRole('region', { name: 'Additional recorded passage responses' })).toBeNull()
  })
})
