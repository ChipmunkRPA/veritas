// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import DocumentContextComparison from './DocumentContextComparison'
import type { DocumentContextComparison as Comparison } from './multimediaPackets'

const focus = { start: 3, end: 8, text: 'focus' }
function fixture(): Comparison {
  return { schema: 'chipmunk.document-context-comparison/1', status: 'recorded', evidence_type: 'context',
    authorship_confidence_pct: null, affects_overall_confidence: false, focus: { start: 3, end: 8 },
    note: 'Recorded context information; it does not change overall confidence.', reason: null, omitted_checks: 0,
    checks: [{ signal_ref: { id: 'opaque', display_id: 'S01' }, score_kind: 'classifier_probability',
      distinct_inputs: 3, omitted_inputs: 1, lowest_response: .1, highest_response: .9, response_spread: .8,
      inputs: [{ spans: [{ start: 0, end: 8 }], text: '😀x focus', input_sha256: 'a'.repeat(64), score: .1, token_coverage: {} },
        { spans: [{ start: 0, end: 8 }, { start: 10, end: 15 }], text: '😀x focus after', input_sha256: 'b'.repeat(64), score: .9, token_coverage: {} }] }] }
}
afterEach(cleanup)
describe('recorded document context comparison', () => {
  it('shows bounded recorded responses and source excerpts with Unicode codepoint highlights', () => {
    const data = fixture(), before = structuredClone(data)
    render(<DocumentContextComparison comparison={data} focus={focus} />)
    expect(screen.getByText(/response range 0.100–0.900 across 3/)).toBeTruthy()
    expect(screen.getByText(/No authorship confidence is assigned/)).toBeTruthy()
    expect([...document.querySelectorAll('mark')].map(x => x.textContent)).toEqual(['focus', 'focus'])
    expect(screen.getByText(/1 other saved input/)).toBeTruthy()
    expect(data).toEqual(before)
    expect(screen.queryByText(/Leaning towards original/)).toBeNull()
  })
  it('does not label equal responses as proof of originality', () => {
    const data = fixture(), check = data.checks[0]
    check.lowest_response = check.highest_response = .4
    check.response_spread = 0
    check.inputs.forEach(x => { x.score = .4 })
    render(<DocumentContextComparison comparison={data} focus={focus} />)
    expect(screen.getByText(/Agreement on these inputs does not establish originality/)).toBeTruthy()
  })
  it('keeps unavailable comparison distinct from a negative finding', () => {
    const data = fixture()
    data.status = 'not_assessed'; data.checks = []; data.reason = 'Fewer than two distinct compatible inputs.'
    render(<DocumentContextComparison comparison={data} focus={focus} />)
    expect(screen.getByText(/Context dependence could not be assessed/)).toBeTruthy()
    expect(document.querySelectorAll('blockquote')).toHaveLength(0)
  })
  it('shows both actual inputs when the paragraph response loses support', () => {
    const data = fixture(), check = data.checks[0]
    data.isolated_paragraph = { signal_ref: check.signal_ref, context: check.inputs[1], paragraph: check.inputs[0],
      supports_inspection: false, note: 'Both responses share the same classifier and do not change overall confidence.' }
    render(<DocumentContextComparison comparison={data} focus={focus} />)
    expect(screen.getByText('Paragraph read on its own')).toBeTruthy()
    expect(screen.getByText(/AI-like response did not persist/)).toBeTruthy()
    expect(screen.getByText(/Sentence in context · 0.900 raw response · source characters/)).toBeTruthy()
    expect(screen.getByText(/Paragraph alone · 0.100 raw response · source characters/)).toBeTruthy()
    expect(screen.queryByText(/Leaning towards original/)).toBeNull()
  })
  it('keeps historical packets without this measurement unchanged', () => {
    const { container } = render(<DocumentContextComparison focus={focus} />)
    expect(container.textContent).toBe('')
  })
})
