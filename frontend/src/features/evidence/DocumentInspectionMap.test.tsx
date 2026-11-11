// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import DocumentEvidenceContent from './DocumentEvidenceContent'
import { RecordedPassage, sourceHighlight } from './DocumentInspectionMap'
import type { DocumentEvidencePacket, DocumentEvidenceSet, EvidenceAsset } from './multimediaPackets'

const source = '🧪 lead. Same sentence. Same sentence. end.'
const packet: DocumentEvidencePacket = {
  id: 'T01', title: 'Writing-pattern inspection clue', observation: 'Recorded responses disagree about this passage.',
  interpretation: 'Inspect the selected passage with its context.', limitations: [], finding_ids: [], signal_refs: [],
  focus: { start: 123, end: 137, text: 'Same sentence.' }, context: { start: 100, end: 142, text: source },
  anchors: [{ kind: 'pdf', page: 2 }], measurements: [], artifacts: {},
  supporting_windows: [
    { signal_ref: { id: 'check.1', display_id: 'S01' }, text: source, input_sha256: 'a'.repeat(64),
      spans: [{ start: 100, end: 142 }], score: 0.91, threshold: 0.5, direction: 'supports_ai', token_coverage: {} },
    { signal_ref: { id: 'check.2', display_id: 'S02' }, text: source, input_sha256: 'b'.repeat(64),
      spans: [{ start: 100, end: 142 }], score: 1.1, threshold: 0.9, ai_when: 'lt', score_kind: 'perplexity_ratio',
      direction: 'opposes_ai', token_coverage: {} },
  ],
}
const asset = (hash: string, role: string): EvidenceAsset => ({ sha256: hash.repeat(64), media_type: 'image/png',
  width: 600, height: 800, url: '/unused', page: 2, role })
const evidence = (packets = [packet]): DocumentEvidenceSet => ({ status: 'ready', id: 'set-1', version: 'document-evidence-2',
  schema: 'chipmunk.document-evidence/1', sha256: 'c'.repeat(64), original_result_hash: 'd'.repeat(64),
  analysis_id: 'analysis-1', revision: 1, limitations: [], source: { sha256: 'e'.repeat(64), chars: 500, offset_unit: 'unicode_codepoint' },
  packets, omitted_windows: 0, coverage: { scored_windows: 2, unmapped_pages: [3] } })
const show = (set: DocumentEvidenceSet) => render(<MemoryRouter><DocumentEvidenceContent set={set} token="signed-token" mediaReady /></MemoryRouter>)
afterEach(cleanup)

describe('ordinary document passage inspection', () => {
  it('marks all three measured sentences and leaves the navigation focus separate', () => {
    const item: DocumentEvidencePacket = { ...packet, inspection_scope: { basis: 'scored_context',
      spans: [{ start: 108, end: 122 }, { start: 123, end: 137 }, { start: 138, end: 142 }] } }
    render(<RecordedPassage packet={item} />)
    const marks = screen.getAllByTestId('passage-highlight-T01')
    expect(marks.map((mark) => mark.textContent)).toEqual(['Same sentence.', 'Same sentence.', 'end.'])
    expect(marks[0].previousSibling?.textContent).toBe('🧪 lead. ')
    expect(screen.getByText(/navigation sentence gives you a place to start reading/)).toBeTruthy()
  })

  it('keeps the separate measured ratio reduction narrow', () => {
    const item: DocumentEvidencePacket = { ...packet, inspection_scope: { basis: 'ratio_reduction',
      spans: [{ start: packet.focus.start, end: packet.focus.end }] } }
    render(<RecordedPassage packet={item} />)
    expect(screen.getByTestId('passage-highlight-T01').textContent).toBe('Same sentence.')
    expect(screen.getByText('Measured writing-pattern range')).toBeTruthy()
  })

  it('highlights the recorded second occurrence after an emoji using nonzero source coordinates', () => {
    render(<RecordedPassage packet={packet} />)
    const highlight = screen.getByTestId('passage-highlight-T01')
    expect(highlight.textContent).toBe('Same sentence.')
    expect(highlight.previousSibling?.textContent).toBe('🧪 lead. Same sentence. ')
    expect(highlight.closest('blockquote')?.textContent).toBe(source)
    expect(screen.getByText(/Unhighlighted text is not cleared/)).toBeTruthy()
  })

  it('refuses an incorrect offset or source envelope instead of locating a matching substring elsewhere', () => {
    const shifted = { ...packet, focus: { ...packet.focus, start: 124, end: 138 } }
    expect(sourceHighlight(shifted)).toBeNull()
    expect(sourceHighlight({ ...packet, context: { ...packet.context, end: 143 } })).toBeNull()
    render(<RecordedPassage packet={shifted} />)
    expect(screen.queryByTestId('passage-highlight-T01')).toBeNull()
    expect(screen.getByText('Same sentence.')).toBeTruthy()
    expect(screen.getByText(/no highlight position is inferred/)).toBeTruthy()
  })

  it('keeps source quotations literal and expands the saved surrounding text without changing the selection', () => {
    const quote = '“DeBERTa” <script>🧪</script>'
    const before = '一'.repeat(240), after = '二'.repeat(240)
    const focus = { start: 740, end: 740 + Array.from(quote).length, text: quote }
    const item = { ...packet, focus, context: { start: 500, end: focus.end + 240, text: before + quote + after } }
    render(<RecordedPassage packet={item} />)
    expect(screen.getByTestId('passage-highlight-T01').textContent).toBe(quote)
    expect(document.querySelector('script')).toBeNull()
    expect(screen.getByTestId('passage-highlight-T01').closest('blockquote')?.textContent).not.toBe(item.context.text)
    fireEvent.click(screen.getByRole('button', { name: 'Show full saved context' }))
    expect(screen.getByTestId('passage-highlight-T01').closest('blockquote')?.textContent).toBe(item.context.text)
    expect(screen.getByTestId('passage-highlight-T01').textContent).toBe(quote)
    fireEvent.click(screen.getByRole('button', { name: 'Show nearby context' }))
    expect(screen.getByTestId('passage-highlight-T01').closest('blockquote')?.textContent).not.toBe(item.context.text)
  })

  it('makes source locations and disagreement visible and focuses the selected passage reasons', () => {
    const second = { ...packet, id: 'T02', anchors: [{ kind: 'docx', paragraph_index: 1, body_path: [1] }] }
    show(evidence([packet, second]))
    const overview = screen.getByRole('region', { name: 'Passages to inspect' })
    expect(within(overview).getAllByText('Responses disagree')).toHaveLength(2)
    const next = within(overview).getByRole('button', { name: 'Inspect passage T02: Paragraph 2' })
    fireEvent.click(next)
    const article = screen.getByRole('article', { name: 'T02: source passage and reasons' })
    expect(document.activeElement).toBe(article)
    expect(next.getAttribute('aria-pressed')).toBe('true')
    expect(article.getAttribute('data-selected')).toBe('true')
    expect(within(article).getByText(/Disagreement is retained/)).toBeTruthy()
    expect(screen.getByText(/Pages without mapped text: 3/)).toBeTruthy()
  })

  it('shows partially checked pages and their frozen image bounds separately from unmapped pages', () => {
    const set = evidence()
    show({ ...set, coverage: { ...set.coverage, partially_checked_pages: [
      { page: 2, reason: 'Recognition quality was too low.', coordinate_space: 'unrotated-page-points', image_boxes: [[40, 150, 570, 735]] },
      { page: 5, reason: 'The image overlaps mapped text.', coordinate_space: 'unrotated-page-points', image_boxes: [[15, 20, 500, 680]] },
    ] } })
    expect(screen.getByText('Pages 2, 5 have mapped text and unchecked image areas.')).toBeTruthy()
    expect(screen.getByText(/Pages without mapped text: 3/)).toBeTruthy()
    const details = screen.getByText('Unchecked image areas').closest('details')!
    expect(within(details).getByText(/Recognition quality was too low/)).toBeTruthy()
    expect(within(details).getByText('Bounds in unrotated PDF points [40, 150, 570, 735].')).toBeTruthy()
    expect(within(details).getByText('Bounds in unrotated PDF points [15, 20, 500, 680].')).toBeTruthy()
  })

  it('shows the frozen PDF highlight first and switches to its unmodified original crop', () => {
    const item = { ...packet, artifacts: { context: asset('1', 'context'), crop: asset('2', 'crop'), overlay: asset('3', 'overlay') } }
    show(evidence([item]))
    expect(screen.getByAltText('T01: page 2 excerpt highlight').getAttribute('src')).toContain(`${'3'.repeat(64)}.png`)
    fireEvent.click(screen.getByRole('button', { name: 'Show original page crop' }))
    expect(screen.getByAltText('T01: page 2 original crop').getAttribute('src')).toContain(`${'2'.repeat(64)}.png`)
    expect(screen.getByAltText('T01: page 2 with the excerpt position marked')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Show excerpt highlight' }))
    expect(screen.getByAltText('T01: page 2 excerpt highlight')).toBeTruthy()
  })

  it('keeps the original crop when no frozen highlight is available and labels email quote ownership', () => {
    const item = { ...packet, anchors: [{ kind: 'email', mime_part: '1.1', quote_role: 'quoted' }, ...packet.anchors],
      artifacts: { context: asset('1', 'context'), crop: asset('2', 'crop') } }
    show(evidence([item]))
    expect(screen.getByAltText('T01: page 2 original crop')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Show excerpt highlight' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Inspect passage T01: Email part 1.1 · quoted; Page 2' })).toBeTruthy()
    expect(screen.getByText(/Do not attribute its wording to the email sender/)).toBeTruthy()
  })
})
