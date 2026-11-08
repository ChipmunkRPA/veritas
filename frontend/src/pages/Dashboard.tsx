import { useMemo, useState, type KeyboardEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FilePlus2, MoreHorizontal, Search } from 'lucide-react'
import { api } from '../lib/api'
import { useMe } from '../lib/api/queries'
import { deletable, mayDelete, mayStartWork } from '../lib/api/roles'
import { fmtBytes, fmtDate, kindWord, shortHash } from '../lib/format'
import { cx } from '../lib/cx'
import { Button, ButtonLink, Menu, ReviewStateMark, ScaleGlyph, Skeleton } from '../components/ui'
import { REVIEW_STATE } from '../components/ui/Marks'
import { examinerPosition, numeral } from '../features/analysis/answer'
import { MediaIcon } from '../components/MediaTypes'
import { useDemo } from '../demo/DemoProvider'
import StandardScreeningNotice from '../features/analysis/StandardScreeningNotice'
import { libraryDecision, libraryNumber, libraryPollInterval, loadLibrary, matchesLibrary, sortLibrary, type LibraryAnalysis, type Sort, type StatusFilter, type TypeFilter, type VerdictFilter } from './analysisLibrary'

const PAGE = 50
const TYPE_LABEL: Record<TypeFilter, string> = { all: 'All types', image: 'Image', video: 'Video', audio: 'Audio', document: 'Document' }
const STATUS_LABEL: Record<StatusFilter, string> = { all: 'All statuses', done: 'Done', running: 'Running or queued', failed: 'Failed', cancelled: 'Cancelled' }
const VERDICT_LABEL: Record<VerdictFilter, string> = { all: 'All verdicts', 'no-indications': 'No indications or capture record', inconclusive: 'Inconclusive', indications: 'Indications', 'not-assessable': 'Not assessable' }
const SORT_LABEL: Record<Sort, string> = { newest: 'Newest', oldest: 'Oldest', 'indications-desc': 'Indications high to low', 'indications-asc': 'Indications low to high', name: 'Filename' }
/** The review state in the list's column of 240 px (N-19b): the longest state, "Disputed · peer reviewer does not concur",
 *  needs 260 px, and printed over the date. The whole wording is the cell's hover text and stands on the analysis. */
const LIST_REVIEW_WORD: Partial<Record<keyof typeof REVIEW_STATE, string>> = { machine: 'Not reviewed', disputed: 'Disputed by the peer reviewer' }
const menuItems = <T extends string>(m: Record<T, string>) => (Object.keys(m) as T[]).map((id) => ({ id, label: m[id] }))

/** VERDICT cell while there is no verdict: the status takes its place (the STATUS column was pure noise for done rows). */
function PendingVerdict({ a }: { a: LibraryAnalysis }) {
  if (a.status === 'failed') {
    return <span className="inline-flex items-center gap-2.5 text-small text-danger" title={a.error ?? undefined}><span aria-hidden className="font-mono">●</span>failed</span>
  }
  if (a.status === 'running') {
    return <span className="inline-flex items-center gap-2.5 text-small text-text"><span aria-hidden className="font-mono text-accent animate-pulse-dot">◌</span>running</span>
  }
  if (a.status === 'queued') {
    return <span className="inline-flex items-center gap-2.5 text-small text-text-muted"><span aria-hidden className="font-mono">○</span>queued</span>
  }
  return <span className="text-small text-text-muted">{a.status === 'done' ? 'Decision not recorded' : a.status === 'cancelled' ? 'Cancelled' : '—'}</span>
}

/** ESTIMATE cell (D-68): a calibrated estimate as a number, the fallback's score in words, and nothing where the row cannot say
 *  what its number is (an inconclusive result, a result decided by a provenance record). */
function EstimateCell({ a }: { a: LibraryAnalysis }) {
  const n = libraryNumber(a)
  if (n.kind === 'calibrated') return <span className="font-mono text-mono text-text">{numeral(n.estimate)} %</span>
  if (n.kind === 'uncalibrated') {
    // two short lines, broken after the comma, so the column stays as narrow as its heading allows
    const [score, rest] = n.text.split(', ')
    return <span className="inline-block text-caption leading-4 text-text-muted whitespace-nowrap"><span className="block">{score}, </span><span className="block">{rest}</span></span>
  }
  return <span className="text-text-muted" title={n.why ?? undefined}><span aria-hidden>—</span><span className="sr-only">{n.why ?? 'No estimate'}</span></span>
}

export default function Dashboard() {
  const qc = useQueryClient()
  const nav = useNavigate()
  const demo = useDemo()
  const q = useQuery({ queryKey: ['analyses', 'v2'], queryFn: loadLibrary, refetchInterval: (q) => libraryPollInterval(q.state.data?.items) })
  const del = useMutation({ mutationFn: api.remove, onSuccess: () => { setConfirmId(null); qc.invalidateQueries({ queryKey: ['analyses'] }) } })
  const me = useMe()
  const deletes = mayDelete(me.data)  // L-7: deleting is an administrator action; the menu offers it to no one else
  const starts = mayStartWork(me.data)  // N-19a: a viewer or a reviewer is not offered what starts work
  const [search, setSearch] = useState('')
  const [type, setType] = useState<TypeFilter>('all')
  const [status, setStatus] = useState<StatusFilter>('all')
  const [verdict, setVerdict] = useState<VerdictFilter>('all')
  const [sort, setSort] = useState<Sort>('newest')
  const [page, setPage] = useState(1)
  const [confirmId, setConfirmId] = useState<string | null>(null)

  const all = q.data?.items ?? []
  const filtered = useMemo(() => sortLibrary(all.filter((a) => matchesLibrary(a, search.trim(), type, status, verdict)), sort), [all, search, type, status, verdict, sort])
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE))
  const cur = Math.min(page, pages)
  const rows = filtered.slice((cur - 1) * PAGE, cur * PAGE)
  const running = all.filter((a) => a.status === 'running' || a.status === 'queued').length
  const lastDone = all.filter((a) => a.status === 'done' && a.finished_at).map((a) => a.finished_at as string).sort().at(-1)

  const facts = [`${all.length} ${all.length === 1 ? 'analysis' : 'analyses'}`, running ? `${running} running` : null, lastDone ? `last completed ${fmtDate(lastDone)}` : null].filter(Boolean).join(' · ')

  const onRowKey = (e: KeyboardEvent<HTMLTableRowElement>, id: string) => {
    if (e.target !== e.currentTarget) return
    if (e.key === 'Enter') nav(`/analyses/${id}`)
  }

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <h1 className="display text-h1 font-normal text-text m-0">Analyses</h1>
        {starts && <ButtonLink href="/upload" variant="primary" onClick={(e) => { e.preventDefault(); nav('/upload') }}><FilePlus2 size={16} strokeWidth={1.5} aria-hidden />New analysis</ButtonLink>}
      </div>
      <p className="text-small text-text-muted mt-1 num">{q.isLoading ? <Skeleton w={200} /> : facts}</p>
      <StandardScreeningNotice />

      {q.data?.hasMore && <p className="mt-2 text-small text-text-muted">The newest {all.length.toLocaleString()} analyses are loaded. Older analyses are not included in search or filters.</p>}

      {q.error && (
        <p className="mt-8 text-small text-danger">
          {(q.error as Error).message} <Button variant="text" onClick={() => q.refetch()}>Retry</Button>
        </p>
      )}

      {!q.isLoading && !q.error && all.length === 0 && (
        <div className="pt-24">
          <h2 className="display text-h1 font-normal text-text">No analyses yet.</h2>
          {starts ? <>
            <p className="text-body text-text-muted mt-2">Submit an image, video, audio recording or document to begin — or <Button variant="text" className="text-body" onClick={demo.start}>run the demo</Button> on five prepared samples.</p>
            <Button variant="primary" className="mt-6" onClick={() => nav('/upload')}>New analysis</Button>
          </> : <p className="text-body text-text-muted mt-2">No analysis has been shared with this account yet.</p>}
        </div>
      )}

      {(q.isLoading || all.length > 0) && (
        <>
          <div className="workspace-toolbar mt-8 flex flex-wrap items-center gap-x-5 gap-y-2">
            <div className="relative max-w-full grow min-w-0 sm:grow-0">
            <Search size={16} strokeWidth={1.5} aria-hidden className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input type="search" value={search} onChange={(e) => { setSearch(e.target.value); setPage(1) }} placeholder="filename or sha256" aria-label="Search by filename or sha256"
              className="h-11 w-64 max-w-full pl-9 pr-2.5 rounded-sm border border-graphic bg-surface text-small placeholder:font-mono placeholder:text-text-muted focus-visible:border-accent" />
            </div>
            <Menu label={type === 'all' ? 'Type' : `Type: ${TYPE_LABEL[type]}`} onSelect={(v) => { setType(v); setPage(1) }} items={menuItems(TYPE_LABEL)} />
            <Menu label={status === 'all' ? 'Status' : `Status: ${STATUS_LABEL[status]}`} onSelect={(v) => { setStatus(v); setPage(1) }} items={menuItems(STATUS_LABEL)} />
            <Menu label={verdict === 'all' ? 'Verdict' : `Verdict: ${VERDICT_LABEL[verdict]}`} onSelect={(v) => { setVerdict(v); setPage(1) }} items={menuItems(VERDICT_LABEL)} />
            <div className="ml-auto">
              <Menu align="right" label={`Sort: ${SORT_LABEL[sort]}`} onSelect={setSort} items={menuItems(SORT_LABEL)} />
            </div>
          </div>

          <div className="relative mt-4 overflow-x-auto workspace-table" data-tour="analyses">
            {/* N-19b (design review DS-20, DS-13): a fixed layout. The file column takes what the fixed columns leave, so the
                table is as wide as its card: the Submitted column and the row menu are on screen at every width. Below 960 px
                the Type and Estimate columns and the hash line go first, and the fixed columns are narrower, so that the file
                column keeps room for a name on two lines; below 640 px each analysis is one stacked block (name, verdict,
                review state, date, menu). A row is as high as the others unless its name takes two lines (N-19k). */}
            <table role="table" className="stack-sm w-full table-fixed border-collapse" data-testid="analyses">
              <thead role="rowgroup">
                <tr role="row" className="border-b border-border-strong bg-surface-2">
                  {/* N-19a (design review DS-12): where a revision belongs stands under its file name */}
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pl-4 pr-4">File · Case · Ex.</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-[84px] max-[959px]:hidden">Type</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-[200px] max-[959px]:w-[188px]">Verdict</th>
                  <th role="columnheader" scope="col" className="eyebrow text-right font-medium py-3 pr-4 w-24 max-[959px]:hidden">Estimate</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-[240px] max-[959px]:w-[228px]">Review</th>
                  <th role="columnheader" scope="col" className="eyebrow text-right font-medium py-3 pr-4 w-[104px] max-[959px]:w-[88px]">Submitted</th>
                  <th role="columnheader" scope="col" className="w-11"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody role="rowgroup">
                {q.isLoading && Array.from({ length: 6 }).map((_, i) => (
                  <tr role="row" key={i} className="border-b border-border h-[88px]">
                    <td role="cell" className="pl-4 pr-4"><Skeleton w={160} /><br /><Skeleton w={120} h={8} className="mt-1" /></td>
                    <td role="cell" className="pr-4 max-[959px]:hidden"><Skeleton w={40} /></td>
                    <td role="cell" className="pr-4"><Skeleton w={140} /></td>
                    <td role="cell" className="pr-4 text-right max-[959px]:hidden"><Skeleton w={36} /></td>
                    <td role="cell" className="pr-4"><Skeleton w={100} /></td>
                    <td role="cell" className="pr-4 text-right"><Skeleton w={72} /></td>
                    <td role="cell" />
                  </tr>
                ))}
                {rows.map((a) => {
                  const confirming = confirmId === a.id
                  const decision = libraryDecision(a)
                  const position = a.examiner_position ? examinerPosition(a.examiner_position, a.examiner_departs, true, a.examiner_reference) : null
                  return (
                    <tr role="row" key={a.id} tabIndex={0} data-testid="analysis-row" onClick={() => !confirming && nav(`/analyses/${a.id}`)} onKeyDown={(e) => onRowKey(e, a.id)}
                      className={cx('group border-b border-border h-[88px] cursor-pointer transition-colors duration-fast hover:bg-surface-3 focus-visible:bg-surface-3',
                        'max-[639px]:relative max-[639px]:h-auto max-[639px]:px-4 max-[639px]:pr-14', confirming && 'bg-surface-2')}>
                      {/* N-19a: the confirmation takes the row's width, so that its sentence stays on one line where there is room */}
                      <td role="cell" colSpan={confirming ? 6 : 1} className="pl-4 pr-4 py-3 min-w-0">
                        {confirming ? (
                          <span className="flex items-center flex-wrap gap-x-3 gap-y-2 text-small" onClick={(e) => e.stopPropagation()}>
                            <span className="text-text">Delete <span className="font-medium [overflow-wrap:anywhere]">{a.filename}</span> and its files? This cannot be undone.</span>
                            <Button variant="danger" className="h-7 px-2.5" onClick={() => del.mutate(a.id)} disabled={del.isPending}>{del.isPending ? 'Deleting…' : 'Delete'}</Button>
                            <Button variant="text" onClick={() => setConfirmId(null)}>Keep</Button>
                            {del.error && <span className="text-danger text-caption">{(del.error as Error).message}</span>}
                          </span>
                        ) : (
                          <>
                            {/* N-19k (design re-review D2-04): the name wraps to two lines, so that a name of 45 characters stands whole; a
                                longer one is cut there with the whole name as its hover text. Below 640 px it wraps in its block */}
                            <Link to={`/analyses/${a.id}`} title={a.filename} onClick={(e) => e.stopPropagation()} data-testid="row-name"
                              className="max-w-full text-body font-medium text-text leading-5 line-clamp-2 [overflow-wrap:anywhere] max-[639px]:line-clamp-none hover:underline underline-offset-[3px]">{a.filename}</Link>
                            <span className="flex items-center gap-x-2 text-caption text-text-muted mt-1 overflow-hidden">
                              <code data-testid="row-hash" className="font-mono text-caption leading-[14px] bg-surface-2 border border-border px-1 rounded-sm whitespace-nowrap max-[959px]:hidden">{shortHash(a.sha256)}</code>
                              <span className="num whitespace-nowrap max-[959px]:hidden">{fmtBytes(a.size_bytes)}</span>
                              <span data-testid="row-revision" className="font-mono text-caption leading-[14px] border border-border px-1 rounded-sm num whitespace-nowrap">rev {a.revision}</span>
                            </span>
                            {a.case_reference && a.case_id && (
                              <Link to={`/cases/${a.case_id}`} onClick={(e) => e.stopPropagation()} data-testid="row-case"
                                className="block max-w-full mt-1 text-caption text-text-muted truncate hover:text-text hover:underline underline-offset-[3px]">
                                <span className="font-mono">{a.case_reference}</span>{a.exhibit_label ? ` · Ex. ${a.exhibit_label}` : ''}
                              </Link>
                            )}
                          </>
                        )}
                      </td>
                      {!confirming && <>
                      <td role="cell" className="pr-4 text-small text-text-muted max-[959px]:hidden"><span className="inline-flex items-center gap-2"><MediaIcon kind={a.kind} size={16} />{kindWord(a.kind)}</span></td>
                      <td role="cell" className="pr-4">
                        {decision ? (
                          <span className="inline-flex items-center gap-2.5 text-small text-text" title={decision.legacy ? 'Legacy result: label reflects its stored band.' : undefined}>
                            <ScaleGlyph step={decision.step} label={decision.label} className="shrink-0" />
                            {decision.label}
                          </span>
                        ) : <PendingVerdict a={a} />}
                      </td>
                      <td role="cell" data-testid="row-estimate" className="pr-4 text-right num max-[959px]:hidden"><EstimateCell a={a} /></td>
                      <td role="cell" className="pr-4 py-3">
                        {/* the state word stays on one line; the examiner's position stands whole under it, on two lines where it needs them
                            (N-19k, D2-04: it was cut before "agrees" or "differs") */}
                        {/* a state word is one line inside its column: the list's words are short, and a word that still did not fit
                            would be cut with its whole text as the hover text, never printed over the date */}
                        <span data-testid="row-review" className="block max-w-full truncate" title={a.review_state && a.status !== 'cancelled' && a.status !== 'failed' ? REVIEW_STATE[a.review_state].word : undefined}>
                          {a.status === 'cancelled' || a.status === 'failed' ? <span data-testid="row-no-review" className="text-small text-text-muted" title="This run published no result to review">—</span> : a.review_state ? <ReviewStateMark state={a.review_state} outstanding={a.awaits_second_review}>{a.awaits_second_review ? 'Awaiting second review' : LIST_REVIEW_WORD[a.review_state] ?? REVIEW_STATE[a.review_state].word}</ReviewStateMark> : <span className="text-small text-text-muted">Review not recorded</span>}
                        </span>
                        {position && <span data-testid="row-position" title={position} className="block max-w-full mt-0.5 text-caption leading-4 text-text-muted">{position}</span>}
                      </td>
                      <td role="cell" data-testid="row-submitted" className="pr-4 text-right max-[639px]:text-left text-small text-text-muted num">
                        {/* the date and the time on two lines, so the column is as narrow as a date */}
                        {a.created_at ? <><span className="block whitespace-nowrap max-[639px]:inline">{fmtDate(a.created_at).slice(0, -10)}</span><span className="block whitespace-nowrap max-[639px]:inline"> {fmtDate(a.created_at).slice(-9)}</span></> : '—'}
                      </td>
                      </>}
                      <td role="cell" className="text-right max-[639px]:!absolute max-[639px]:top-1 max-[639px]:right-1 max-[639px]:!p-0" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
                        <Menu align="right" ariaLabel={`Actions for ${a.filename}`} onSelect={(id) => { if (id === 'delete') setConfirmId(a.id); if (id === 'open') nav(`/analyses/${a.id}`) }}
                          triggerClassName="opacity-70 hover:opacity-100 focus-visible:opacity-100 h-11 w-11 justify-center"
                          trigger={<MoreHorizontal size={16} strokeWidth={1.5} aria-hidden />}
                          items={[{ id: 'open', label: 'Open' }, ...(deletes && deletable(a.review_state, a.locked_at) ? [{ id: 'delete', label: 'Delete', danger: true }] : [])]} />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {!q.isLoading && filtered.length === 0 && all.length > 0 && (
            <p className="mt-6 text-small text-text-muted">No analyses match the current filters. <Button variant="text" onClick={() => { setSearch(''); setType('all'); setStatus('all'); setVerdict('all') }}>Clear filters</Button></p>
          )}

          {filtered.length > 0 && (
            <div className="mt-4 flex items-center justify-between text-small text-text-muted num">
              <span>Showing {(cur - 1) * PAGE + 1}–{Math.min(cur * PAGE, filtered.length)} of {filtered.length}</span>
              {pages > 1 && (
                <nav aria-label="Pagination" className="flex items-center gap-3">
                  <button type="button" onClick={() => setPage(Math.max(1, cur - 1))} disabled={cur === 1} aria-label="Previous page" className="min-w-8 h-8 disabled:opacity-40 hover:text-text">‹</button>
                  {Array.from({ length: pages }, (_, i) => i + 1).map((p) => (
                    <button key={p} type="button" onClick={() => setPage(p)} aria-current={p === cur ? 'page' : undefined}
                      className={cx('min-w-8 h-8 hover:text-text', p === cur && 'text-text underline underline-offset-[3px]')}>{p}</button>
                  ))}
                  <button type="button" onClick={() => setPage(Math.min(pages, cur + 1))} disabled={cur === pages} aria-label="Next page" className="min-w-8 h-8 disabled:opacity-40 hover:text-text">›</button>
                </nav>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
