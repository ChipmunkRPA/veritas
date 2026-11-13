import type { DocumentUnselectedResponses as Responses } from './multimediaPackets'

type Source = Responses['rows'][number]['sources'][number]

function location(source: Source) {
  if (source.kind === 'pdf') return `Page ${source.page} · ${source.extraction === 'ocr' ? 'recognized text' : 'selectable text'}`
  if (source.kind === 'email') return `Email part ${source.mime_part} · ${source.quote_role}`
  return `Paragraph ${source.paragraph}`
}

export default function DocumentUnselectedResponses({ responses }: { responses: Responses | undefined }) {
  if (!responses) return null
  return <section className="mt-4 rounded-md border border-border p-4" aria-label="Additional recorded passage responses">
    <h3 className="m-0 text-body font-medium">Additional recorded passage responses</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">
      The listed ranges show what each check read together. Displaying these responses adds no weight to the overall assessment.
    </p>
    <ul className="m-0 mt-3 list-none p-0 space-y-2">{responses.rows.map((row) => <li key={row.id}>
      <details id={`document-response-${row.id}`} className="rounded-md border border-border p-3 text-small">
        <summary className="cursor-pointer">
          <span className="font-mono text-accent">{row.id} · {row.check}</span>
          {' · '}{row.direction === 'supports_ai' ? 'AI-like response' : 'Response leaned away from AI in this window'}
          {' · '}<span className="font-mono">{row.response.toFixed(3)}</span> raw
        </summary>
        <blockquote className="m-0 mt-3 border-l border-border pl-3 whitespace-pre-wrap break-words">{row.text}</blockquote>
        <ul className="mt-3 pl-4 list-disc text-caption text-text-muted">{row.spans.map((span, index) => <li key={span.start}>
          {location(row.sources[index])} · <span className="font-mono">[{span.start}, {span.end})</span>
        </li>)}</ul>
        <p className="mt-2 mb-0 text-caption text-text-muted">The response measures the writing pattern of the full input. Offsets count extracted Unicode characters.</p>
      </details>
    </li>)}</ul>
    {responses.omitted_windows > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">
      The display limit omitted {responses.omitted_windows.toLocaleString()} additional eligible responses.
    </p>}
  </section>
}
