import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button } from '../../components/ui'
import { saveBlob } from '../../lib/api/reports'
import { compareDraft, draftAvailability, type DraftExcerpt, type DraftExtension, type DocumentRevisionResponse } from './documentRevisionApi'

const labels = { body: 'Message body', quoted: 'Quoted text', signature: 'Signature' }

function MarkedExcerpt({ excerpt }: { excerpt: DraftExcerpt }) {
  const points = Array.from(excerpt.text)
  let cursor = 0
  const changes = excerpt.changed_tokens.map(([from, to]) => {
    const start = from - excerpt.start
    const end = to - excerpt.start
    const context = points.slice(cursor, start).join('')
    cursor = end
    return <span key={from}>{context}<mark className="rounded-xs border-b border-accent bg-surface-3 text-text">{points.slice(start, end).join('')}</mark></span>
  })
  return <>{changes}{points.slice(cursor).join('')}</>
}

function Excerpt({ label, excerpt }: { label: string; excerpt: DraftExcerpt | null }) {
  return <div className="min-w-0">
    <h4 className="m-0 text-small font-medium">{label}</h4>
    {excerpt && <p className="mt-1 mb-0 font-mono text-caption text-text-muted">[{excerpt.start}, {excerpt.end})</p>}
    {excerpt?.paragraphs?.map(anchor => <p key={anchor.body_index} className="mt-1 mb-0 font-mono text-caption text-text-muted">
      Paragraph {anchor.paragraph} [{anchor.start}, {anchor.end})
    </p>)}
    {excerpt?.text ? <pre className="mt-2 mb-0 whitespace-pre-wrap break-words font-sans text-small"><MarkedExcerpt excerpt={excerpt} /></pre>
      : <p className="mt-2 mb-0 text-small text-text-muted">{excerpt ? 'No text at this location.' : 'No corresponding section.'}</p>}
  </div>
}

function DraftResult({ result }: { result: DocumentRevisionResponse }) {
  const { comparison, pdf } = result
  const docx = comparison.format === 'docx'
  const formatLabel = docx ? 'Word body text' : comparison.format === 'plain_text' ? 'Text' : null
  const downloadJson = () => saveBlob(new Blob([JSON.stringify(comparison, null, 2) + '\n'], { type: 'application/json' }),
    'chipmunk-verify-draft-comparison.json')
  const downloadPdf = (body: string) => {
    const bytes = Uint8Array.from(atob(body), (value) => value.charCodeAt(0))
    saveBlob(new Blob([bytes], { type: 'application/pdf' }), 'chipmunk-verify-draft-comparison.pdf')
  }
  return <div className="mt-4 border-t border-border pt-4">
    <div className="flex flex-wrap items-baseline justify-between gap-3">
      <p className="m-0 text-small font-medium" role="status">
        {comparison.changes.length === 0 ? 'No word or punctuation changes found'
          : `${comparison.changes.length} ${comparison.changes.length === 1 ? 'change' : 'changes'} found`}
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {pdf?.startsWith('JVBERi0') && <Button variant="text" onClick={() => downloadPdf(pdf)}>Download comparison PDF</Button>}
        <Button variant="text" onClick={downloadJson}>Download comparison JSON</Button>
      </div>
    </div>
    <p className="mt-2 mb-0 text-small text-text-muted">{comparison.scope} {comparison.interpretation}</p>
    {!!comparison.changes.length && <>
      <p className="mt-2 mb-0 text-caption text-text-muted">Highlights mark changed words and punctuation. {docx
        ? 'Word offsets count Unicode code points in the body text. Two line breaks separate included paragraphs. Paragraph ranges start within each paragraph.'
        : 'Offsets count decoded Unicode code points.'} The end of each range is excluded.</p>
      <ol className="mt-3 mb-0 list-none space-y-3 p-0">
        {comparison.changes.map(change => <li key={change.id} className="rounded-md border border-border p-3">
          <h3 className="m-0 text-small font-medium">{change.id} · {formatLabel ?? labels[change.role]}</h3>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <Excerpt label="Earlier draft" excerpt={change.earlier} />
            <Excerpt label="Analyzed version" excerpt={change.later} />
          </div>
        </li>)}
      </ol>
    </>}
  </div>
}

function DraftForm({ analysisId, extension }: { analysisId: string; extension: DraftExtension }) {
  const headingId = useId()
  const inputId = useId()
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<DocumentRevisionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const active = useRef<AbortController | null>(null)
  useEffect(() => () => active.current?.abort(), [])

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!file || active.current) return
    const request = new AbortController()
    active.current = request
    setPending(true)
    setError(null)
    setResult(null)
    try {
      const comparison = await compareDraft(analysisId, file, request.signal)
      if (!request.signal.aborted) setResult(comparison)
    } catch (error) {
      if (!request.signal.aborted) setError(error instanceof Error ? error.message : 'Please try again.')
    } finally {
      if (active.current === request && !request.signal.aborted) {
        active.current = null
        setPending(false)
      }
    }
  }

  return <section aria-labelledby={headingId} className="rounded-md border border-border p-4">
    <h2 id={headingId} className="m-0 text-body font-medium">Compare with an earlier draft</h2>
    <p className="mt-2 mb-0 text-small text-text-muted">Choose an earlier {extension} file, up to 1 MiB, to see where the wording changed.</p>
    {extension === '.docx' && <p className="mt-2 mb-0 text-small text-text-muted">Checks wording and punctuation in body paragraphs.</p>}
    <form onSubmit={event => void submit(event)} className="mt-4 flex flex-wrap items-end gap-3">
      <div className="min-w-0">
        <label htmlFor={inputId} className="mb-1 block text-small font-medium">Earlier draft</label>
        <input id={inputId} type="file" accept={extension} className="block max-w-full text-small"
          onChange={event => {
            active.current?.abort()
            active.current = null
            setFile(event.currentTarget.files?.[0] ?? null)
            setResult(null)
            setError(null)
            setPending(false)
          }} />
      </div>
      <Button type="submit" variant="primary" disabled={!file || pending}>{pending ? 'Comparing…' : 'Compare drafts'}</Button>
    </form>
    {pending && <p role="status" className="mt-3 mb-0 text-small text-text-muted">Comparing draft text…</p>}
    {error && <p role="alert" className="mt-3 mb-0 text-small text-danger">{error}</p>}
    {result && <DraftResult result={result} />}
  </section>
}

export default function DocumentRevisionComparison({ analysisId }: { analysisId: string }) {
  const availability = useQuery({ queryKey: ['analysis', analysisId, 'document-revision-comparison'],
    queryFn: ({ signal }) => draftAvailability(analysisId, signal), refetchOnMount: 'always', refetchOnWindowFocus: false, retry: false })
  return availability.isFetchedAfterMount && availability.data && !availability.isError
    ? <DraftForm key={analysisId} analysisId={analysisId} extension={availability.data} /> : null
}
