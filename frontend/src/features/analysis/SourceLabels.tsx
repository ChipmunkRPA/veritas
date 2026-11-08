type Label = { id: string; declaration: string; location: string; byte_offset: number | null;
  claimed_provider: string | null; payload_sha256: string }
type Packet = { status: string; source_sha256: string; records: Label[]; checked_metadata: string[]; issues: string[]; notice: string }

function packet(value: unknown, source: string | undefined): Packet | null {
  if (!value || typeof value !== 'object' || !source) return null
  const data = value as Record<string, unknown>
  if (data.version !== 'source-labels/1' || data.source_sha256 !== source || !Array.isArray(data.records)
    || typeof data.status !== 'string' || typeof data.notice !== 'string') return null
  const records = data.records.filter((r): r is Label => !!r && typeof r === 'object'
    && typeof r.id === 'string' && typeof r.declaration === 'string' && typeof r.location === 'string'
    && typeof r.payload_sha256 === 'string' && (r.claimed_provider === null || typeof r.claimed_provider === 'string'))
  return { status: data.status, source_sha256: source, records, notice: data.notice,
    checked_metadata: Array.isArray(data.checked_metadata) ? data.checked_metadata.filter((x): x is string => typeof x === 'string') : [],
    issues: Array.isArray(data.issues) ? data.issues.filter((x): x is string => typeof x === 'string') : [] }
}

export default function SourceLabels({ value, source }: { value?: unknown; source?: string }) {
  const data = packet(value, source)
  return <section id="source-labels" aria-label="Source labeling" className="mt-6 rounded-md border border-border px-4 py-4 text-small">
    <h3 className="m-0 text-body font-semibold">Source labeling</h3>
    {!data ? <p className="mt-2 mb-0 text-text-muted">A structured source-label check was not recorded for this analysis. Missing labels do not establish originality.</p>
      : <>
        <p className="mt-2 mb-0 text-text-muted">{data.notice}</p>
        <p className="mt-2 mb-0 text-text-muted">{data.checked_metadata.length ? `Inspected: ${data.checked_metadata.join('; ')}.`
          : 'No supported metadata fields were assessed.'} {data.status === 'partial' && 'Coverage is incomplete.'}</p>
        {data.records.length ? <ol className="mt-3 mb-0 list-none p-0 space-y-3">{data.records.map((r) => <li key={r.id} id={`source-label-${r.id}`}>
          <p className="m-0"><strong>{r.id} — {r.declaration}</strong></p>
          <p className="mt-1 mb-0">Claimed provider: {r.claimed_provider ?? 'not identified; unrecognized names and codes are withheld'}. Provider identity is not authenticated.</p>
          <p className="mt-1 mb-0 text-text-muted">{r.location}{typeof r.byte_offset === 'number' ? `, container byte offset ${r.byte_offset}` : ''}. No confidence percentage is assigned.</p>
          <details className="mt-1 text-text-muted"><summary className="cursor-pointer">Source reference</summary>
            <p className="mt-1 mb-0 break-all font-mono text-micro">Extracted label-field SHA-256: {r.payload_sha256}</p>
            <p className="mt-1 mb-0 break-all font-mono text-micro">Bound source SHA-256: {data.source_sha256}</p>
            <p className="mt-1 mb-0">The same evidence identifier appears in the PDF and JSON export.</p>
          </details>
        </li>)}</ol> : <p className="mt-2 mb-0">{data.status === 'checked'
          ? 'No recognized label was found in the inspected fields. This does not establish originality.'
          : 'This check is incomplete or unavailable. No conclusion follows from missing labels.'}</p>}
        {data.issues.map((issue, i) => <p key={i} className="mt-2 mb-0 text-text-muted">{issue}</p>)}
      </>}
  </section>
}
