import type { DocumentContextComparison as Comparison, TextSpan } from './multimediaPackets'

function Excerpt({ input, focus }: { input: Comparison['checks'][number]['inputs'][number]; focus: TextSpan }) {
  const text = Array.from(input.text)
  let cursor = 0, start: number | null = null
  for (const span of input.spans) {
    if (span.start <= focus.start && focus.end <= span.end) start = cursor + focus.start - span.start
    cursor += span.end - span.start + 1
  }
  const end = start !== null ? start + focus.end - focus.start : null
  const valid = start !== null && end !== null && start >= 0 && end <= text.length
    && text.slice(start, end).join('') === focus.text
  return <blockquote className="m-0 mt-2 pl-3 border-l border-border whitespace-pre-wrap break-words text-small">
    {valid ? <>{text.slice(0, start!).join('')}<mark className="bg-amber-100 text-text">{text.slice(start!, end!).join('')}</mark>{text.slice(end!).join('')}</> : input.text}
  </blockquote>
}

export default function DocumentContextComparison({ comparison, focus }: { comparison?: Comparison; focus: TextSpan }) {
  if (!comparison) return null
  return <section aria-label="Recorded context comparison" className="mt-5 rounded-md border border-border bg-surface-2 p-4">
    <h4 className="m-0 text-small font-medium">How surrounding text affects this check</h4>
    {comparison.status === 'not_assessed' ? <p className="mt-2 mb-0 text-small text-text-muted">{comparison.reason} Context dependence could not be assessed; this does not establish human authorship.</p> : <>
      <p className="mt-2 mb-0 text-small text-text-muted">{comparison.note} Displayed responses are rounded; JSON retains the recorded values.</p>
      <div className="space-y-4 mt-4">{comparison.checks.map((check) => <div key={check.signal_ref.id}>
        <p className="m-0 text-small font-medium">{check.signal_ref.display_id ?? 'Recorded check'} · response range {check.lowest_response.toFixed(3)}–{check.highest_response.toFixed(3)} across {check.distinct_inputs} distinct saved inputs</p>
        <p className="mt-1 mb-0 text-caption text-text-muted">Spread {check.response_spread.toFixed(3)} on this check’s response scale. No authorship confidence is assigned to this comparison.
          {check.response_spread === 0 && ' Agreement on these inputs does not establish originality.'}</p>
        <div className="grid grid-cols-1 gap-3 mt-3 lg:grid-cols-2">{check.inputs.map((input, index) => <div key={input.input_sha256} className="min-w-0 rounded-sm border border-border bg-surface p-3">
          <p className="m-0 text-caption font-medium">{index === 0 ? 'Lowest recorded response' : 'Highest recorded response'} · {input.score.toFixed(3)}</p>
          <Excerpt input={input} focus={focus} />
          <p className="mt-2 mb-0 text-caption text-text-muted">Source characters {input.spans.map((span) => `[${span.start}, ${span.end})`).join(', ')}. The selected passage is marked within the actual input.</p>
          <details className="mt-2 text-caption text-text-muted"><summary className="cursor-pointer">Input commitment</summary><p className="mb-0 font-mono break-all">SHA-256: {input.input_sha256}</p></details>
        </div>)}</div>
        {check.omitted_inputs > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">{check.omitted_inputs} other saved input(s) are not repeated here. The two endpoints show the full recorded response range for this passage.</p>}
      </div>)}</div>
      {comparison.omitted_checks > 0 && <p className="mt-3 mb-0 text-caption text-text-muted">{comparison.omitted_checks} additional check comparison(s) are outside this bounded display.</p>}
    </>}
  </section>
}
