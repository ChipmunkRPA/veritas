import { useEffect, useId, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button } from '../../components/ui'
import { saveBlob } from '../../lib/api/reports'
import { archiveAvailability, compareArchive, type ArchiveCollection, type ArchiveComparison, type ArchiveExcerpt } from './documentArchiveApi'

const statusLabels = {
  reference_bound: 'Archive passage matched',
  no_qualified_reference: 'No matching passage in this archive library',
  ambiguous_reference: 'More than one archive passage could match',
  out_of_scope: 'Text is outside the comparison scope',
}

function changeHeading(excerpt: ArchiveExcerpt, paragraph: number, joined: boolean) {
  if (excerpt.docx_location) return `Word body paragraph ${excerpt.docx_location.paragraph}`
  if (excerpt.docx_locations?.length) {
    return excerpt.start === excerpt.end ? 'Archive wording missing between Word paragraphs' : 'Change across Word paragraphs'
  }
  if (excerpt.pdf_pages?.length) return `Submitted PDF · ${excerpt.pdf_pages.map(page => `page ${page.page}`).join(', ')}`
  return `${joined ? 'Passage in submitted paragraph' : 'Submitted paragraph'} ${paragraph}`
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
    {excerpt?.docx_location && <p className="mt-1 mb-0 text-caption text-text-muted">
      Word paragraph characters [{excerpt.docx_location.paragraph_range.join(', ')})
    </p>}
    {!!excerpt?.docx_locations?.length && <ul aria-label="Changed Word positions" className="mt-1 mb-0 text-caption text-text-muted">
      {excerpt.docx_locations.map(location => <li key={location.body_index}>
        Word body paragraph {location.paragraph}, {location.paragraph_range[0] === location.paragraph_range[1]
          ? `boundary at character ${location.paragraph_range[0]}.`
          : `characters [${location.paragraph_range.join(', ')}).`}
      </li>)}
    </ul>}
    {excerpt?.text
      ? <p className="mt-2 mb-0 whitespace-pre-wrap break-words text-small">{marked}{points.slice(cursor).join('')}</p>
      : <p className="mt-2 mb-0 text-small text-text-muted">{excerpt ? 'No text at this position.' : 'No corresponding text.'}</p>}
    {excerpt?.pdf_pages?.map(page => <figure key={page.page} className="mx-0 mt-3 mb-0">
      <figcaption className="text-caption text-text-muted">Page {page.page}{page.preview ? ' · highlighted source crop' : ''}</figcaption>
      {page.preview && <>
        <img alt={`Page ${page.page} with changed wording highlighted`} className="mt-1 max-h-56 max-w-full rounded-sm border border-border"
          src={`data:image/png;base64,${page.preview.highlighted_png}`} />
        <details className="mt-1 text-caption text-text-muted">
          <summary className="cursor-pointer">View original crop</summary>
          <img alt={`Page ${page.page} original crop`} className="mt-1 max-h-56 max-w-full rounded-sm border border-border"
            src={`data:image/png;base64,${page.preview.crop_png}`} />
        </details>
      </>}
    </figure>)}
  </div>
}

function ArchiveResult({ result }: { result: ArchiveComparison }) {
  const { comparison, pdf } = result
  const { library, reference, coverage, changes } = comparison
  const joined = comparison.layout === 'recovered_archive_boundaries'
  const gao = comparison.schema === 'chipmunk.document-archive-comparison/2'
  const wordPassages = comparison.aligned_paragraphs.flatMap(row => row.docx_location ? [row.docx_location] : [])
  const pdfPassages = comparison.aligned_paragraphs.flatMap(row => row.pdf_pages ?? [])
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
        {coverage.matched_words} of {coverage.body_words} {gao ? 'body tokens' : 'body words'} support the source match.
      </p>
      {gao && reference.submitted_range && reference.transcript_range && <p className="mt-2 mb-0 text-caption text-text-muted">
        Published {reference.published_date}. Matched submitted range [{reference.submitted_range.join(', ')}).
        {' '}Transcript range [{reference.transcript_range.join(', ')}).
      </p>}
      {wordPassages.length > 0 && <ul aria-label="Matched Word paragraphs" className="mt-2 mb-0 text-caption text-text-muted">
        {wordPassages.map(location => <li key={location.body_index}>
          Word body paragraph {location.paragraph}, characters [{location.paragraph_range.join(', ')}).
        </li>)}
      </ul>}
      {pdfPassages.length > 0 && <ul aria-label="Matched PDF pages" className="mt-2 mb-0 text-caption text-text-muted">
        {pdfPassages.map(({ page, extracted_range: range }) => <li key={`${page}/${range[0]}`}>
          Page {page}, extracted characters [{range.join(', ')}).
        </li>)}
      </ul>}
      {joined && <p className="mt-2 mb-0 text-small text-text-muted">The submitted text joins several source paragraphs. The comparison recovered their boundaries from the archive.</p>}
      {changes.length === 0 && <p className="mt-2 mb-0 text-small">{gao ? 'No normalized wording differences in the compared passage.'
        : 'No word or punctuation differences in the compared passages.'}</p>}
    </>}
    {changes.length > 0 && <>
      <p className="mt-2 mb-0 text-caption text-text-muted">Submitted positions count Unicode code points in the extracted text.
        {' '}{gao ? 'Archive positions start within the full highlights transcript.' : 'Archive positions start within the cited paragraph.'} Range ends are excluded.</p>
      {!!comparison.submitted_sentence_ranges?.length && <p className="mt-2 mb-0 text-caption text-text-muted">
        Submitted sentences containing differences {comparison.submitted_sentence_ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')}.
      </p>}
      <ol className="mt-3 mb-0 list-none space-y-3 p-0">
        {changes.map((change, i) => <li key={i} className="rounded-md border border-border p-3">
          <h3 className="m-0 text-small font-medium">{changeHeading(change.submitted, change.submitted_paragraph, joined)}</h3>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <Excerpt label={gao ? 'Published highlights transcript' : change.archive_paragraph == null ? 'Archive version' : `Archive body paragraph ${change.archive_paragraph}`}
              excerpt={change.archive} />
            <Excerpt label="Submitted text" excerpt={change.submitted} />
          </div>
        </li>)}
      </ol>
    </>}
  </div>
}

function ArchiveAction({ analysisId, gaoAvailable }: { analysisId: string; gaoAvailable: boolean }) {
  const heading = useId()
  const request = useRef<AbortController | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ArchiveComparison | null>(null)
  const [collection, setCollection] = useState<ArchiveCollection>('frus')
  useEffect(() => () => request.current?.abort(), [])

  async function run() {
    if (request.current) return
    const active = new AbortController()
    request.current = active
    setPending(true)
    setError(null)
    setResult(null)
    try {
      const response = await compareArchive(analysisId, active.signal, collection)
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
    <p className="mt-2 mb-3 text-small text-text-muted">{collection === 'gao'
      ? 'Compare the retained text with local GovReport transcripts of U.S. GAO report highlights.'
      : 'Compare the retained text with a local collection of U.S. diplomatic records.'} See exact differences in matched passages.</p>
    {gaoAvailable && <label className="mb-3 block text-small">
      Archive collection
      <select className="ml-2 rounded border border-border bg-surface-1 p-1" value={collection} disabled={pending}
        onChange={event => {
          setCollection(event.currentTarget.value === 'gao' ? 'gao' : 'frus')
          setResult(null)
          setError(null)
        }}>
        <option value="frus">U.S. diplomatic records</option>
        <option value="gao">U.S. GAO report highlights</option>
      </select>
    </label>}
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
    ? <ArchiveAction key={analysisId} analysisId={analysisId} gaoAvailable={availability.data.gaoAvailable} /> : null
}
