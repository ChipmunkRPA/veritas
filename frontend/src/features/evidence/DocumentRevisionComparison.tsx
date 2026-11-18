import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button } from '../../components/ui'
import { saveBlob } from '../../lib/api/reports'
import { compareDraft, draftAvailability, type DraftComparison, type DraftExcerpt, type DraftExtension, type DocumentRevisionResponse } from './documentRevisionApi'

const labels = { body: 'Message body', quoted: 'Quoted text', signature: 'Signature' }
const sourceFormats = { plain_text: 'Plain text', docx: 'Word body text', odt: 'OpenDocument body text', pdf: 'PDF text layer', rtf: 'Rich Text passage' }
const changeLabels = { ...sourceFormats, email: null, plain_text: 'Text', pdf: 'PDF text', mixed: 'Wording change' }
const pairedMixedFormats: DraftExtension[] = ['.txt', '.docx', '.pdf']
const contextLabels = ['Previous sentence', 'Changed sentence', 'Next sentence']
const tableSkips = {
  no_matching_position: 'No table at the same body position.',
  unsupported_layout: 'Outside the supported table layout.',
  changed_structure: 'The table shape, headings or row labels changed.',
  comparison_limit: 'The table exceeded the comparison limit.',
  caption_unmatched: 'The table caption has no unique match in both drafts.',
}
type WordTables = NonNullable<DraftComparison['word_tables']>
type CellExcerpt = WordTables['compared'][number]['changes'][number]['earlier']

function MarkedExcerpt({ excerpt }: { excerpt: DraftExcerpt | CellExcerpt }) {
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

function WordTableResults({ coverage }: { coverage: WordTables }) {
  return <section aria-label="Word table wording changes" className="mt-4">
    <h3 className="m-0 text-small font-medium">Word table wording changes</h3>
    <p className="mt-2 mb-0 text-caption text-text-muted">Table, row and column numbers start at one. Character positions start within each cell.</p>
    {coverage.compared.map(table => <div key={table.earlier_body_index ?? table.body_index} className="mt-3">
      <h4 className="m-0 text-small font-medium">Earlier table {table.earlier_table} · Analyzed table {table.later_table}</h4>
      {table.match === 'caption' && <p className="mt-2 mb-0 text-small">Matched by a unique Word caption.</p>}
      {(['rows', 'columns'] as const).map(label => {
        const count = label === 'rows' ? table.unchecked_rows : table.unchecked_columns
        return count && (count.earlier > 0 || count.later > 0) ? <p key={label} className="mt-2 mb-0 text-small">
          Unchecked {label}. {count.earlier} earlier, {count.later} analyzed.
        </p> : null
      })}
      {!table.changes.length && <p className="mt-2 mb-0 text-small">No wording changes in the compared cells.</p>}
      {!!table.changes.length && <ol className="mt-2 mb-0 list-none space-y-3 p-0">
        {table.changes.map(change => <li key={change.id} className="rounded-md border border-border p-3">
          <h5 className="m-0 text-small font-medium">{change.id} · Row {change.row}, column {change.column}</h5>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            {(['earlier', 'later'] as const).map(side => {
              const excerpt = change[side]
              return <div key={side} className="min-w-0">
                <p className="m-0 text-small font-medium">{side === 'earlier' ? 'Earlier cell' : 'Analyzed cell'}
                  {change.earlier_row != null && ` · Row ${side === 'earlier' ? change.earlier_row : change.row}, column ${side === 'earlier' ? change.earlier_column : change.column}`}
                </p>
                <p className="mt-1 mb-0 font-mono text-caption text-text-muted">[{excerpt.start}, {excerpt.end})</p>
                {excerpt.paragraphs.map(anchor => <p key={anchor.cell_index} className="mt-1 mb-0 font-mono text-caption text-text-muted">
                  Paragraph {anchor.paragraph} [{anchor.start}, {anchor.end})
                </p>)}
                {excerpt.text ? <pre className="mt-2 mb-0 whitespace-pre-wrap break-words font-sans text-small"><MarkedExcerpt excerpt={excerpt} /></pre>
                  : <p className="mt-2 mb-0 text-small text-text-muted">No text at this location.</p>}
              </div>
            })}
          </div>
        </li>)}
      </ol>}
    </div>)}
    {coverage.skipped.map(table => <p key={`${table.earlier_body_index ?? table.body_index}:${table.later_body_index}`} className="mt-2 mb-0 text-small text-text-muted">
      {[table.earlier_table != null && `Earlier table ${table.earlier_table}`, table.later_table != null && `Analyzed table ${table.later_table}`]
        .filter(Boolean).join(' / ')}. {tableSkips[table.reason]}
    </p>)}
    {!!coverage.omitted_positions && <p className="mt-2 mb-0 text-small text-text-muted">
      {coverage.omitted_positions} further table positions exceed the reporting limit.
    </p>}
  </section>
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
    {!!excerpt?.html_tokens?.length && <details className="mt-1 text-caption text-text-muted">
      <summary className="cursor-pointer">Decoded HTML positions</summary>
      <ul className="mt-1 mb-0 list-none p-0 font-mono">
        {excerpt.html_tokens.map(token => <li key={token.start}>
          [{token.start}, {token.end}) → {token.source_ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')}
        </li>)}
      </ul>
    </details>}
    {excerpt?.pdf_pages?.map(page => <div key={page.page} className="mt-3">
      <p className="m-0 text-caption text-text-muted">Page {page.page}{page.preview ? ' · highlighted source crop' : ''}</p>
      {page.preview && <>
        <img alt={`${label}, page ${page.page}, with changed text highlighted`} className="mt-1 max-h-56 max-w-full rounded-sm border border-border object-contain object-left"
          src={`data:image/png;base64,${page.preview.highlighted_png}`} />
        <details className="mt-1 text-caption text-text-muted">
          <summary className="cursor-pointer">View source crop without highlight</summary>
          <img alt={`${label}, page ${page.page}, original crop`} className="mt-1 max-h-56 max-w-full rounded-sm border border-border object-contain object-left"
            src={`data:image/png;base64,${page.preview.crop_png}`} />
        </details>
      </>}
    </div>)}
  </div>
}

const figureLabels = {
  number: 'Number changed',
  date: 'Date changed',
  currency: 'Amount changed',
  percentage: 'Percentage changed',
  percentage_points: 'Percentage points changed',
}

function FigureResults({ comparison }: { comparison: NonNullable<DraftComparison['numeric_comparison']> }) {
  return <section aria-label="Figures and dates" className="mt-4 rounded-md border border-border p-3">
    <h3 className="m-0 text-small font-medium">Figures and dates</h3>
    <p className="mt-1 mb-0 text-caption text-text-muted">
      {comparison.compared} matched · {comparison.differences} changed
      {comparison.changes.length < comparison.differences && ` · ${comparison.changes.length} shown`}
    </p>
    <p className="mt-1 mb-0 text-caption text-text-muted">{comparison.scope}</p>
    {comparison.changes.map(change => (
      <div key={change.later_range[0]} className="mt-3 border-t border-border pt-3">
        <p className="m-0 text-small font-medium">
          {figureLabels[change.kind]} · {change.earlier_value} → {change.later_value}
        </p>
        <div className="mt-2 grid gap-4 sm:grid-cols-2">
          <Excerpt label="Earlier draft" excerpt={change.earlier} />
          <Excerpt label="Analyzed version" excerpt={change.later} />
        </div>
      </div>
    ))}
  </section>
}

function DraftResult({ result }: { result: DocumentRevisionResponse }) {
  const anchorPrefix = useId()
  const { comparison, pdf } = result
  const paired = comparison.paired_check
  const mixed = comparison.format === 'mixed'
  const docx = comparison.format === 'docx'
  const pdfText = comparison.format === 'pdf'
  const html = comparison.coordinate_space === 'email_html_extracted_text_unicode_codepoint'
  const wordTables = comparison.word_tables
  const relocated = comparison.relocated_passages ?? []
  const editedRelocation = comparison.schema === 'chipmunk.document-revision-comparison/8'
  const changeCount = comparison.changes.length + (wordTables?.compared.reduce((sum, table) => sum + table.changes.length, 0) ?? 0)
  const formatLabel = changeLabels[comparison.format]
  const downloadJson = () => saveBlob(new Blob([JSON.stringify(comparison, null, 2) + '\n'], { type: 'application/json' }),
    'chipmunk-verify-draft-comparison.json')
  const downloadPdf = (body: string) => {
    const bytes = Uint8Array.from(atob(body), (value) => value.charCodeAt(0))
    saveBlob(new Blob([bytes], { type: 'application/pdf' }), 'chipmunk-verify-draft-comparison.pdf')
  }
  return <div className="mt-4 border-t border-border pt-4">
    <div className="flex flex-wrap items-baseline justify-between gap-3">
      <p className="m-0 text-small font-medium" role="status">
        {editedRelocation && 'One body passage was reordered and edited. '}
        {changeCount === 0 ? `${editedRelocation ? 'No other' : 'No'} word or punctuation changes found${html ? ' in the reply body' : wordTables ? ' in the compared Word text' : ''}`
          : `${changeCount} ${changeCount === 1 ? 'change' : 'changes'} found`}
        {!editedRelocation && relocated.length > 0 && '. One reordered body passage found.'}
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {pdf?.startsWith('JVBERi0') && <Button variant="text" onClick={() => downloadPdf(pdf)}>Download comparison PDF</Button>}
        <Button variant="text" onClick={downloadJson}>Download comparison JSON</Button>
      </div>
    </div>
    <p className="mt-2 mb-0 text-small text-text-muted">{comparison.scope} {comparison.interpretation}</p>
    {mixed && comparison.earlier.format && comparison.later.format && <p className="mt-2 mb-0 text-caption text-text-muted">
      Earlier draft · {sourceFormats[comparison.earlier.format]}. Analyzed version · {sourceFormats[comparison.later.format]}.
    </p>}
    {comparison.format === 'email' && <p className="mt-2 mb-0 text-caption text-text-muted">
      Earlier MIME part {comparison.earlier.mime_part} · Analyzed MIME part {comparison.later.mime_part}
    </p>}
    {comparison.numeric_comparison && <FigureResults comparison={comparison.numeric_comparison} />}
    {paired && <section aria-label="Experimental paired writing check" className="mt-4 rounded-md border border-border p-3">
      <h3 className="m-0 text-small font-medium">Experimental paired writing check</h3>
      <p className="mt-2 mb-0 text-small">{paired.summary}</p>
      <p className="mt-1 mb-0 text-caption text-text-muted">{paired.scope}</p>
      {paired.revisions?.map((revision, index) => <article key={revision.later.start} className="mt-3 border-t border-border pt-3">
        <h4 className="m-0 text-small font-medium">{paired.schema === 'chipmunk.paired-writing-review/12'
          ? 'Changed passage' : 'Changed sentence'} {index + 1}</h4>
        <p className="mt-1 mb-0 text-small">{revision.status === 'joint_increase'
          ? 'Both writing checks gave a stronger AI-like response in the analyzed version.'
          : revision.status === 'unavailable' ? 'This mapped revision could not be scored.'
            : 'The writing responses did not show a shared increase at the review level.'}</p>
        {!!revision.change_ids?.length && <p className="mt-2 mb-0 text-caption text-text-muted">Linked wording changes{' '}
          {revision.change_ids.map((id, position) => <span key={id}>{position > 0 && ', '}
            <a className="underline" href={`#${anchorPrefix}-${id}`}>{id}</a>
          </span>)}
        </p>}
        <div className="mt-2 grid gap-4 sm:grid-cols-2">
          {(['earlier', 'later'] as const).map(side => {
            const sentence = revision[side]
            const location = sentence.context_paragraphs?.[1]
            const lastLocation = sentence.context_paragraphs?.at(-2)
            return <div key={side}>
              <p className="m-0 font-mono text-caption text-text-muted">
                {side === 'earlier' ? 'Earlier' : 'Analyzed'} {sentence.context_spans.length === 4 ? 'sentences' : 'sentence'} [{sentence.start}, {sentence.end})
              </p>
              {location && <p className="mt-1 mb-0 font-mono text-caption text-text-muted">
                Paragraph {location.paragraph} [{location.start}, {lastLocation?.end ?? location.end})
              </p>}
              {sentence.pdf_pages?.map(page => <p key={page.page} className="mt-1 mb-0 font-mono text-caption text-text-muted">
                PDF page {page.page}
              </p>)}
              <p className="mt-1 mb-0 whitespace-pre-wrap break-words text-small">
                <MarkedExcerpt excerpt={{ ...sentence,
                  changed_tokens: revision.wording_changes?.flatMap(change => change[`${side}_changed_tokens`]) ?? [] }} />
              </p>
            </div>
          })}
        </div>
        {revision.responses?.map((response, i) => <p key={i} className="mt-1 mb-0 text-caption">
          Check {i + 1} · {response.earlier.toPrecision(4)} → {response.later.toPrecision(4)}
        </p>)}
      </article>)}
      {paired.earlier && paired.later && paired.responses?.length === 2 && <>
        <p className="mt-2 mb-0 text-caption text-text-muted">Linked wording changes{' '}
          {paired.change_ids?.map((id, index) => <span key={id}>{index > 0 && ', '}
            <a className="underline" href={`#${anchorPrefix}-${id}`}>{id}</a>
          </span>)}
        </p>
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          <div>
            <p className="m-0 text-caption font-medium">
              Earlier {paired.earlier.context_spans.length === 4 ? 'sentences' : 'sentence'} [{paired.earlier.start}, {paired.earlier.end})
            </p>
            {paired.earlier.context_paragraphs?.map((location, index) => <p key={index} className="mt-1 mb-0 font-mono text-caption text-text-muted">
              {contextLabels[index]} · Paragraph {location.paragraph} [{location.start}, {location.end})
            </p>)}
            <p className="mt-1 mb-0 whitespace-pre-wrap break-words text-small">{paired.earlier.text}</p>
          </div>
          <div>
            <p className="m-0 text-caption font-medium">
              Analyzed {paired.later.context_spans.length === 4 ? 'sentences' : 'sentence'} [{paired.later.start}, {paired.later.end})
            </p>
            {paired.later.context_paragraphs?.map((location, index) => <p key={index} className="mt-1 mb-0 font-mono text-caption text-text-muted">
              {contextLabels[index]} · Paragraph {location.paragraph} [{location.start}, {location.end})
            </p>)}
            <p className="mt-1 mb-0 whitespace-pre-wrap break-words text-small">{paired.later.text}</p>
          </div>
        </div>
        <ul aria-label="Uncalibrated writing responses from earlier to analyzed" className="mt-3 mb-0 list-none p-0 text-caption">
          {paired.responses.map((response, index) => {
            const center = paired.center_responses?.[index]
            return <li key={index}>
              Check {index + 1} · {center ? 'With neighbors ' : paired.neighboring_contexts?.length ? 'Centered context ' : ''}
              {response.earlier.toPrecision(4)} → {response.later.toPrecision(4)}
              {center && <>. Changed sentence alone {center.earlier.toPrecision(4)} → {center.later.toPrecision(4)}.</>}
            </li>
          })}
        </ul>
      </>}
      {!!paired.neighboring_contexts?.length && <div className="mt-3">
        <h4 className="m-0 text-caption font-medium">Additional scored contexts</h4>
        {paired.neighboring_contexts.map(context => <div key={context.later.start} className="mt-3 border-t border-border pt-2">
          <p className="m-0 text-caption">{context.status === 'joint_increase' ? 'Both responses increased enough for review.'
            : 'The responses did not show a shared increase at the review level.'}</p>
          <div className="mt-2 grid gap-3 sm:grid-cols-2">
            <div>
              <p className="m-0 font-mono text-caption text-text-muted">Earlier scored context [{context.earlier.start}, {context.earlier.end})</p>
              <p className="mt-1 mb-0 whitespace-pre-wrap break-words text-small">{context.earlier.text}</p>
            </div>
            <div>
              <p className="m-0 font-mono text-caption text-text-muted">Analyzed scored context [{context.later.start}, {context.later.end})</p>
              <p className="mt-1 mb-0 whitespace-pre-wrap break-words text-small">{context.later.text}</p>
            </div>
          </div>
          <ul aria-label="Uncalibrated responses for this scored context" className="mt-2 mb-0 list-none p-0 text-caption">
            {context.responses.map((response, index) => <li key={index}>
              Check {index + 1} · {response.earlier.toPrecision(4)} → {response.later.toPrecision(4)}
            </li>)}
          </ul>
        </div>)}
      </div>}
    </section>}
    {relocated.map(passage => <section key={passage.later.start} className="mt-3 rounded-md border border-border p-3">
      <h3 className="m-0 text-small font-medium">{editedRelocation ? 'Reordered and edited body passage' : 'Reordered body passage'}</h3>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <Excerpt label="Earlier draft" excerpt={passage.earlier} />
        <Excerpt label="Analyzed version" excerpt={passage.later} />
      </div>
    </section>)}
    {!!comparison.changes.length && <>
      <p className="mt-2 mb-0 text-caption text-text-muted">Highlights mark changed words and punctuation. {docx
        ? 'Word offsets count Unicode code points in the body text. Two line breaks separate included paragraphs. Paragraph ranges start within each paragraph.'
        : comparison.format === 'odt' ? 'OpenDocument offsets count Unicode code points in the extracted body and within each listed paragraph.'
        : html ? 'The first range gives the position in the extracted reply. The ranges after the arrow give the positions in the decoded HTML MIME part. Both count Unicode code points.'
        : pdfText ? 'Offsets count Unicode code points in the extracted text. Page previews cover selected regions.'
        : mixed ? 'Each side uses its own Unicode character offsets and source positions.'
        : 'Offsets count decoded Unicode code points.'} The end of each range is excluded.</p>
      <ol className="mt-3 mb-0 list-none space-y-3 p-0">
        {comparison.changes.map(change => <li key={change.id} className="rounded-md border border-border p-3">
          <h3 id={`${anchorPrefix}-${change.id}`} tabIndex={-1} className="m-0 text-small font-medium">{change.id} · {formatLabel ?? labels[change.role]}</h3>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <Excerpt label="Earlier draft" excerpt={change.earlier} />
            <Excerpt label="Analyzed version" excerpt={change.later} />
          </div>
        </li>)}
      </ol>
    </>}
    {wordTables && <WordTableResults coverage={wordTables} />}
  </div>
}

function DraftForm({ analysisId, extension, earlierExtensions = [extension], pairedCheckAvailable, pairedMixedFormatsAvailable = false }: {
  analysisId: string; extension: DraftExtension; earlierExtensions?: DraftExtension[];
  pairedCheckAvailable: boolean; pairedMixedFormatsAvailable?: boolean
}) {
  const headingId = useId()
  const inputId = useId()
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<DocumentRevisionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const [includePairedCheck, setIncludePairedCheck] = useState(false)
  const earlierName = file?.name.toLowerCase()
  const sameFormat = !earlierName || earlierName.endsWith(extension)
  const canRunPairedCheck = pairedCheckAvailable && (sameFormat || (pairedMixedFormatsAvailable
    && pairedMixedFormats.some(suffix => earlierName?.endsWith(suffix))))
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
      const comparison = await compareDraft(analysisId, file, request.signal, canRunPairedCheck && includePairedCheck)
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
    <p className="mt-2 mb-0 text-small text-text-muted">Choose an earlier draft up to 1 MiB. Supported formats are {earlierExtensions.map(suffix => suffix.slice(1).toUpperCase()).join(', ')}.</p>
    {extension === '.docx' && <p className="mt-2 mb-0 text-small text-text-muted">Checks body wording and matches simple Word tables by exact row and column labels. A moved table needs the same unique Word caption.</p>}
    {extension === '.odt' && <p className="mt-2 mb-0 text-small text-text-muted">Checks wording in extracted OpenDocument body paragraphs, headings, lists and tables.</p>}
    {extension === '.rtf' && <p className="mt-2 mb-0 text-small text-text-muted">Rich Text positions refer to paragraphs in the extracted text.</p>}
    {(extension === '.txt' || extension === '.docx') && sameFormat && <p className="mt-2 mb-0 text-small text-text-muted">Matching file formats can also show where an unchanged body paragraph moved.</p>}
    {extension === '.eml' && <p className="mt-2 mb-0 text-small text-text-muted">Use two plain-text emails or two HTML-only replies with two, three, five or six simple body paragraphs and a trailing blockquote.</p>}
    {extension === '.pdf' && <p className="mt-2 mb-0 text-small text-text-muted">Use unlocked PDFs with selectable text on every page, up to 12 pages each.</p>}
    {earlierExtensions.length > 1 && <p className="mt-2 mb-0 text-small text-text-muted">For drafts in different formats, use an upright PDF with selectable text. Multi-page PDFs need a single column per page and can have up to 12 pages.</p>}
    {canRunPairedCheck && (extension === '.pdf' || earlierName?.endsWith('.pdf')) &&
      <p className="mt-2 mb-0 text-small text-text-muted">The paired writing check needs visible, selectable PDF text in a clear reading order and the same neighboring sentences in both drafts.</p>}
    <form onSubmit={event => void submit(event)} className="mt-4 flex flex-wrap items-end gap-3">
      <div className="min-w-0">
        <label htmlFor={inputId} className="mb-1 block text-small font-medium">Earlier draft</label>
        <input id={inputId} type="file" accept={earlierExtensions.join(',')} className="block max-w-full text-small"
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
      {canRunPairedCheck && <label className="flex w-full items-center gap-2 text-small text-text-muted">
        <input type="checkbox" checked={includePairedCheck} disabled={pending}
          onChange={event => setIncludePairedCheck(event.currentTarget.checked)} />
        Include experimental paired writing check
      </label>}
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
    ? <DraftForm key={analysisId} analysisId={analysisId} {...availability.data} /> : null
}
