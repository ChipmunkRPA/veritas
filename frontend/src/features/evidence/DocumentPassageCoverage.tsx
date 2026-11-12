import type { DocumentCoverageState, DocumentPassageCoverage as Coverage } from './multimediaPackets'

const states = ['response_range', 'context_only', 'no_local_record'] as const
const labels: Record<DocumentCoverageState, string> = {
  response_range: 'Local response recorded',
  context_only: 'Read as context only',
  no_local_record: 'No exact local record',
}

function location(row: Coverage['ranges'][number]) {
  const place = row.page != null ? `Page ${row.page}`
    : row.paragraph != null ? `Paragraph ${row.paragraph}` : `Source block ${row.source_block}`
  return [place, row.mime_part && `Email part ${row.mime_part}`, row.quote_role].filter(Boolean).join(' · ')
}

export default function DocumentPassageCoverage({ coverage }: { coverage: Coverage | undefined }) {
  if (!coverage) return null
  return <section className="mt-4 rounded-md border border-border p-4" aria-label="Recorded passage coverage">
    <h3 className="m-0 text-body font-medium">Recorded passage coverage</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">
      See where this analysis has exact local records. Counts use extracted characters excluding whitespace.
    </p>
    <dl className="mt-3 mb-0 grid gap-3 sm:grid-cols-3">
      {states.map((state) => <div key={state}>
        <dt className="text-small">{labels[state]}</dt>
        <dd className="m-0 font-mono text-body">{coverage.counts[state].toLocaleString()}</dd>
      </div>)}
    </dl>
    <p className="mt-3 mb-0 text-caption text-text-muted">
      A local response applies to its recorded input window. Context-only text was read as part of another response.
      Text without an exact local record has no passage-level conclusion here.
    </p>
    {coverage.ranges.length > 0 && <details className="mt-3 text-small">
      <summary className="cursor-pointer">Source ranges ({coverage.ranges.length.toLocaleString()})</summary>
      <p className="mt-2 mb-0 text-caption text-text-muted">Offsets use extracted Unicode characters. Each range includes its start and excludes its end.</p>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-left text-small">
          <thead><tr><th scope="col" className="py-1 pr-3">Source</th><th scope="col" className="py-1 pr-3">Characters</th>
            <th scope="col" className="py-1">Local coverage</th></tr></thead>
          <tbody>{coverage.ranges.map((row) => <tr key={row.start} className="border-t border-border">
            <td className="py-1 pr-3">{location(row)}</td><td className="py-1 pr-3 font-mono">[{row.start}, {row.end})</td>
            <td className="py-1">{labels[row.state]}</td>
          </tr>)}</tbody>
        </table>
      </div>
      {coverage.omitted_ranges > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">
        The summary includes {coverage.omitted_ranges.toLocaleString()} additional ranges beyond the retained list.
      </p>}
    </details>}
  </section>
}
