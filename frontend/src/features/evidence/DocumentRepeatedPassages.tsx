import type { DocumentEvidenceSet } from './multimediaPackets'

export default function DocumentRepeatedPassages({ result }: { result?: DocumentEvidenceSet['repeated_passages'] }) {
  if (!result) return null
  const coverage = result.coverage
  return <section aria-labelledby="repeated-passages-heading" className="mt-5 rounded-md border border-border p-4">
    <h3 id="repeated-passages-heading" className="m-0 text-body font-medium">Exact repeated passages</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">Compared {coverage.compared_blocks} mapped text blocks of at least 30 words and at most 1,600 characters. Quotes, templates, and editorial duplication can repeat wording. Matches carry no AI confidence weight.</p>
    {coverage.excluded_ocr_blocks > 0 && <p className="mt-2 mb-0 text-small">Excluded {coverage.excluded_ocr_blocks} blocks read from page images.</p>}
    {coverage.excluded_quoted_or_signature_blocks > 0 && <p className="mt-2 mb-0 text-small">Excluded {coverage.excluded_quoted_or_signature_blocks} blocks marked as email quotations or signatures.</p>}
    {result.status === 'not_assessed' && <p className="mt-2 mb-0 text-small">There were too few eligible blocks to compare.</p>}
    {result.status === 'no_match' && <p className="mt-2 mb-0 text-small">No exact repeat was found among the compared blocks.</p>}
    {result.groups.map((group) => <article key={group.id} id={`media-evidence-${group.id}`} className="mt-4 border-t border-border pt-3 scroll-mt-20">
      <h4 className="m-0 text-small font-medium">{group.id} · {group.word_count} words repeated at {group.total_occurrences} locations</h4>
      <blockquote className="m-0 mt-2 border-l border-border pl-3 text-small whitespace-pre-wrap break-words">{group.text}</blockquote>
      <ul className="mb-0 pl-4 list-disc text-small text-text-muted">{group.occurrences.map(({ start, end, anchor }) => <li key={start}>
        {anchor.kind === 'pdf' && `Page ${anchor.page}${anchor.block != null ? `, block ${anchor.block + 1}` : ''} · `}
        {anchor.kind === 'docx' && `Paragraph ${(anchor.paragraph_index ?? 0) + 1} · `}
        {anchor.kind === 'email' && 'Email body · '}
        Extracted characters [{start}, {end}){anchor.source_start != null && anchor.source_end != null && ` · Source characters [${anchor.source_start}, ${anchor.source_end})`}
        {anchor.bbox && ` · Bounds [${anchor.bbox.map((value) => value.toFixed(1)).join(', ')}] on the unrotated page`}
      </li>)}</ul>
      {group.total_occurrences > group.occurrences.length && <p className="mt-2 mb-0 text-caption text-text-muted">Showing the first {group.occurrences.length} locations.</p>}
    </article>)}
    {coverage.omitted_groups > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">{coverage.omitted_groups} additional repeat groups are outside the evidence display limit.</p>}
  </section>
}
