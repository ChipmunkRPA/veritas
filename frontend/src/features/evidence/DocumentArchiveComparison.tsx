import { useEffect, useId, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button } from '../../components/ui'
import { saveBlob } from '../../lib/api/reports'
import { archiveAvailability, compareArchive, type ArchiveComparison, type ArchiveExcerpt } from './documentArchiveApi'

const statusLabels = {
  reference_bound: 'Archive passage matched',
  no_qualified_reference: 'No matching passage in this archive library',
  ambiguous_reference: 'More than one archive passage could match',
  out_of_scope: 'Text is outside the comparison scope',
}

function Excerpt({ label, excerpt }: { label: string; excerpt: ArchiveExcerpt | null }) {
  const points = Array.from(excerpt?.text ?? '')
  let cursor = 0
  const marked = excerpt?.changed_tokens.map(([a, b]) => {
    const start = a - excerpt.start
    const end = b - excerpt.start
    const before = points.slice(cursor, start).join('')
    cursor = end
    return <span key={a}>{before}<mark className="rounded-xs border-b border-accent bg-surface-3 text-text">{points.slice(start, end).join('')}</mark></span>
  })
  return <div className="min-w-0">
    <h4 className="m-0 text-small font-medium">{label}</h4>
    {excerpt && <p className="mt-1 mb-0 font-mono text-caption text-text-muted">[{excerpt.start}, {excerpt.end})</p>}
    {excerpt?.text
      ? <p className="mt-2 mb-0 whitespace-pre-wrap break-words text-small">{marked}{points.slice(cursor).join('')}</p>
      : <p className="mt-2 mb-0 text-small text-text-muted">{excerpt ? 'No text at this position.' : 'No corresponding text.'}</p>}
  </div>
}

function ArchiveResult({ result }: { result: ArchiveComparison }) {
  const { comparison, pdf } = result
  const { library, reference, coverage, changes } = comparison
  const downloadJson = () => saveBlob(new Blob([JSON.stringify(comparison, null, 2) + '\n'], { type: 'application/json' }),
    'chipmunk-verify-archive-comparison.json')
  const downloadPdf = (body: string) => {
    const bytes = Uint8Array.from(atob(body), value => value.charCodeAt(0))
    saveBlob(new Blob([bytes], { type: 'application/pdf' }), 'chipmunk-verify-archive-comparison.pdf')
  }
  return <div className="mt-4 border-t border-border pt-4">
    <div className="flex flex-wrap items-baseline justify-between gap-3">
      <p className="m-0 text-small font-medium" role="status">{statusLabels[comparison.status]}</p>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {pdf?.startsWith('JVBERi0') && <Button variant="text" onClick={() => downloadPdf(pdf)}>Download archive PDF</Button>}
        <Button variant="text" onClick={downloadJson}>Download archive JSON</Button>
      </div>
    </div>
    <p className="mt-2 mb-0 text-small text-text-muted">{comparison.scope} {comparison.interpretation}</p>
    <p className="mt-2 mb-0 text-small">
      <a href={library.url} className="underline">{library.title}</a> · {library.publisher}
    </p>
    {reference && <>
      <p className="mt-2 mb-0 text-small">
        <a href={reference.url} className="underline">Source {reference.id}</a> ·{' '}
        <a href={reference.version_url} className="underline">Pinned source version</a>
      </p>
      <p className="mt-2 mb-0 text-small text-text-muted">
        {coverage.matched_words} of {coverage.body_words} body words support the source match.
      </p>
      {changes.length === 0 && <p className="mt-2 mb-0 text-small">No word or punctuation differences in the compared passages.</p>}
    </>}
    {changes.length > 0 && <>
      <p className="mt-2 mb-0 text-caption text-text-muted">Submitted positions count Unicode code points in the extracted text. Archive positions start within the cited paragraph. Range ends are excluded.</p>
      <ol className="mt-3 mb-0 list-none space-y-3 p-0">
        {changes.map((change, i) => <li key={i} className="rounded-md border border-border p-3">
          <h3 className="m-0 text-small font-medium">Submitted paragraph {change.submitted_paragraph}</h3>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <Excerpt label={change.archive_paragraph == null ? 'Archive version' : `Archive body paragraph ${change.archive_paragraph}`}
              excerpt={change.archive} />
            <Excerpt label="Submitted text" excerpt={change.submitted} />
          </div>
        </li>)}
      </ol>
    </>}
  </div>
}

function ArchiveAction({ analysisId }: { analysisId: string }) {
  const heading = useId()
  const request = useRef<AbortController | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ArchiveComparison | null>(null)
  useEffect(() => () => request.current?.abort(), [])

  async function run() {
    if (request.current) return
    const active = new AbortController()
    request.current = active
    setPending(true)
    setError(null)
    setResult(null)
    try {
      const response = await compareArchive(analysisId, active.signal)
      if (!active.signal.aborted) setResult(response)
    } catch (reason) {
      if (!active.signal.aborted) setError(reason instanceof Error ? reason.message : 'Please try again.')
    } finally {
      if (!active.signal.aborted) {
        request.current = null
        setPending(false)
      }
    }
  }

  return <section aria-labelledby={heading} className="rounded-md border border-border p-4">
    <h2 id={heading} className="m-0 text-body font-medium">Public archive comparison</h2>
    <p className="mt-2 mb-3 text-small text-text-muted">Compare the retained text with a local collection of U.S. diplomatic records. See exact differences in matched passages.</p>
    <Button variant="primary" disabled={pending} onClick={() => void run()}>
      {pending ? 'Comparing…' : 'Compare with public archive'}
    </Button>
    {pending && <p role="status" className="mt-3 mb-0 text-small text-text-muted">Comparing archived text…</p>}
    {error && <p role="alert" className="mt-3 mb-0 text-small text-danger">{error}</p>}
    {result && <ArchiveResult result={result} />}
  </section>
}

export default function DocumentArchiveComparison({ analysisId }: { analysisId: string }) {
  const availability = useQuery({ queryKey: ['analysis', analysisId, 'document-archive-comparison'],
    queryFn: ({ signal }) => archiveAvailability(analysisId, signal), refetchOnMount: 'always',
    refetchOnWindowFocus: false, retry: false, gcTime: 0 })
  return availability.isFetchedAfterMount && availability.data && !availability.isError
    ? <ArchiveAction key={analysisId} analysisId={analysisId} /> : null
}
