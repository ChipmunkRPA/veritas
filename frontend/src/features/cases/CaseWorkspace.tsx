import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ApiError } from '../../lib/api/client'
import { useAnalysisDetails, useCase, useCaseAnalyses, useCaseItems, useMe, useMembers, type AnalysisDetail, type CaseAnalysis } from '../../lib/api/queries'
import { fmtBytes, fmtDate, fmtDateLong, shortHash } from '../../lib/format'
import EditCaseDialog, { retentionDay } from './EditCaseDialog'
import { Button, CodeChip, Disclosure, EmptyState, ErrorState, Eyebrow, ReviewStateMark, ScaleGlyph, Skeleton } from '../../components/ui'
import { AWAITING_SECOND_REVIEW, SCALE_WORD, examinerPosition, type ScalePosition } from '../review/review'
import { CAPTURE_HEADLINE, DECLARATION_HEADLINE, captureRow, declarationRow } from '../analysis/record'
import { chainsByMedia, type Chain } from './chains'
import NotFound from '../../components/NotFound'
import CustodyLog from './CustodyLog'
import CaseReports from './CaseReports'
import { NOTES_INLINE, fmtUtcMinutes, hashComparison } from './intakeRecord'
import MembersPanel from './MembersPanel'
import { CASE_ROLE_WORD, CASE_STATUS_WORD, CLASSIFICATION_WORD } from './labels'

const STEP: Record<string, 1 | 2 | 3 | 4 | 5> = { ai_supported_strong: 5, ai_supported: 4, inconclusive: 3, no_ai_indicators: 2, no_ai_indicators_strong: 1 }
const STATUS_WORD: Record<string, string> = { queued: 'Queued', running: 'Running', failed: 'Failed', cancelled: 'Cancelled' }

/** The assessment of the latest revision: the decision code in UX §7.1 wording with its scale glyph, or the pipeline status. */
export function Assessment({ a }: { a: CaseAnalysis }) {
  if (a.status !== 'done') return <span className={a.status === 'failed' ? 'text-small text-danger' : 'text-small text-text-muted'}>{STATUS_WORD[a.status] ?? a.status}</span>
  const code = a.decision_code
  if (captureRow(a)) {  // N-19m (D3-01): a clean code that a provenance record decides is a capture's, headed as its page is
    return <span className="inline-flex items-baseline gap-2 text-small text-text"><ScaleGlyph step={1} className="shrink-0 self-center" />{CAPTURE_HEADLINE}</span>
  }
  if (declarationRow(a)) {  // N-19p (D6-01): a trusted declaration on the fallback is headed as its page is, on the declaration's step
    return <span className="inline-flex items-baseline gap-2 text-small text-text"><ScaleGlyph step={5} className="shrink-0 self-center" />{DECLARATION_HEADLINE}</span>
  }
  if (code && STEP[code]) {
    return <span className="inline-flex items-baseline gap-2 text-small text-text"><ScaleGlyph step={STEP[code]} className="shrink-0 self-center" />{SCALE_WORD[code as ScalePosition]}</span>
  }
  return <span className="text-small text-text">{code === 'not_assessable' ? 'Could not be assessed reliably' : 'No v2 decision (v1 result)'}</span>
}

/** UX §5.12 case table words: the glyph and a short state; the full sentence is on the analysis page. */
const REVIEW_SHORT: Record<string, string> = {
  machine: 'not reviewed', in_review: 'in review', reviewed: 'reviewed', approved: 'reviewed · second review',
  disputed: 'disputed · peer does not concur', superseded: 'superseded', withdrawn: 'withdrawn',
}

function ReviewCell({ chain, details }: { chain: Chain | undefined; details: ReadonlyMap<string, AnalysisDetail> }) {
  const review = chain ? details.get(chain.latest.id)?.review : undefined
  if (!review) return <span className="text-small text-text-muted">—</span>
  const position = examinerPosition(review.examiner_position, review.examiner_departs, true, review.examiner_reference)
  const outstanding = review.state === 'reviewed' && review.requires_second_review  // signed, not issuable until the peer review
  return (
    <>
      <ReviewStateMark state={review.state} outstanding={outstanding}>{outstanding ? AWAITING_SECOND_REVIEW : REVIEW_SHORT[review.state] ?? review.state}</ReviewStateMark>
      {position && <span className="block mt-0.5 text-caption text-text-muted" data-testid="examiner-position">{position}</span>}
    </>
  )
}

function AnalysisLinks({ chains }: { chains: Chain[] }) {
  if (chains.length === 0) return <span className="text-small text-text-muted">Not analysed</span>
  return (
    <ul className="m-0 p-0 list-none flex flex-col gap-1">
      {chains.map((ch) => (
        <li key={ch.latest.id} data-testid="chain" data-superseded={ch.superseded.length}>
          <Link to={`/analyses/${ch.latest.id}`} className="text-small text-accent underline underline-offset-[3px] decoration-1 hover:decoration-2">
            Analysis <span className="font-mono">{ch.latest.id.slice(0, 8)}</span>
          </Link>
          <span className="text-caption text-text-muted num"> · revision {ch.latest.revision}</span>
          {ch.unpublished.length > 0 && (
            <ul className="m-0 mt-1 p-0 list-none flex flex-col gap-1" data-testid="chain-unpublished">
              {ch.unpublished.map((u) => (
                <li key={u.id} className="text-small text-text-muted">
                  <Link to={`/analyses/${u.id}`} className="text-accent underline underline-offset-[3px] decoration-1 hover:decoration-2">Analysis <span className="font-mono">{u.id.slice(0, 8)}</span></Link>
                  <span className="num"> · revision {u.revision} · {STATUS_WORD[u.status] ?? u.status} before it published, so it replaced nothing</span>
                </li>
              ))}
            </ul>
          )}
          {ch.superseded.length > 0 && (
            <Disclosure label={`${ch.superseded.length} superseded`}>
              <ol className="m-0 p-0 list-none flex flex-col gap-1">
                {ch.superseded.map((s) => (
                  <li key={s.id} className="text-small text-text-muted">
                    <span aria-hidden className="inline-block w-3 text-center mr-1">⊘</span>
                    <Link to={`/analyses/${s.id}`} className="text-accent underline underline-offset-[3px] decoration-1 hover:decoration-2">Analysis <span className="font-mono">{s.id.slice(0, 8)}</span></Link>
                    <span className="num"> · revision {s.revision}{s.pipeline_version ? ` · ${s.pipeline_version}` : ''}</span>
                  </li>
                ))}
              </ol>
            </Disclosure>
          )}
        </li>
      ))}
    </ul>
  )
}

/** Case workspace (ARCHITECTURE §7.2, UX §5.12): case facts; the exhibits with the assessment and review state of each latest
 *  revision and its supersede chain; the reports, issued and drafts; the custody log with verification; members. */
export default function CaseWorkspace() {
  const { caseId = '' } = useParams()
  const nav = useNavigate()
  const c = useCase(caseId)
  const items = useCaseItems(caseId)
  const analyses = useCaseAnalyses(caseId)
  const members = useMembers(caseId)
  const me = useMe()
  const [editing, setEditing] = useState(false)
  const chains = useMemo(() => chainsByMedia(analyses.data ?? []), [analyses.data])
  const doneLatest = useMemo(() => [...chains.values()].flat().filter((ch) => ch.latest.status === 'done').map((ch) => ch.latest.id), [chains])
  const details = useAnalysisDetails(doneLatest)
  const names = useMemo(() => {
    const m = new Map<string, string>()
    for (const x of members.data ?? []) m.set(x.principal_id, x.display_name)
    if (me.data) m.set(me.data.id, me.data.display_name)
    return m
  }, [members.data, me.data])
  const labels = useMemo(() => {
    const media = new Map((items.data ?? []).map((it) => [it.media.id, it.label]))
    const byAnalysis = new Map((analyses.data ?? []).flatMap((a) => (a.media_id && media.has(a.media_id) ? [[a.id, media.get(a.media_id)!] as const] : [])))
    return { media, analyses: byAnalysis }
  }, [items.data, analyses.data])

  if (c.isPending) {
    return (
      <div aria-busy="true">
        <Skeleton w={80} h={12} /><div className="mt-6"><Skeleton w={120} h={12} /></div><div className="mt-3"><Skeleton w={420} h={32} /></div>
      </div>
    )
  }
  if (c.error instanceof ApiError && c.error.status === 404) return <NotFound what="case" />
  if (c.error || !c.data) {
    return (
      <ErrorState onRetry={() => void c.refetch()}>
        {c.error?.message ?? 'Not found'} <span className="text-text-muted mx-1">·</span> <Link to="/" className="text-accent underline underline-offset-[3px]">Back to cases</Link>
      </ErrorState>
    )
  }

  const k = c.data
  const isAdmin = me.data?.roles.includes('administrator') ?? false
  const canAdd = (isAdmin || k.my_role === 'owner' || k.my_role === 'examiner') && k.status !== 'closed' && k.status !== 'signed_off'
  const manages = isAdmin || k.my_role === 'owner'  // D-67, D-70: the owner's case
  const facts: [string, string][] = [
    ['Status', CASE_STATUS_WORD[k.status] ?? k.status],
    ['Classification', CLASSIFICATION_WORD[k.classification] ?? k.classification],
    ...(k.matter_reference ? [['Client / matter', k.matter_reference] as [string, string]] : []),
    ['Opened', fmtDateLong(k.created_at)],
    ['Your role', k.my_role ? CASE_ROLE_WORD[k.my_role] ?? k.my_role : 'Administrator (not a member)'],
    ...(k.requires_second_review ? [['Sign-off', 'needs a second reviewer'] as [string, string]] : []),
    ...(k.legal_hold ? [['Legal hold', 'on'] as [string, string]] : []),
    ...(k.retention_until ? [['Retained until', retentionDay(k.retention_until)] as [string, string]] : []),
  ]
  const rows = items.data ?? []

  return (
    <article>
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <Link to="/" className="text-small text-text-muted hover:text-text inline-flex items-center min-h-8">‹ Cases</Link>
        {manages && !k.implicit && <Button variant="ghost" onClick={() => setEditing(true)}>Edit case…</Button>}
      </div>
      {/* mounted while it is open: a closed form is not part of the page */}
      {manages && !k.implicit && editing && <EditCaseDialog open case={k} isAdmin={isAdmin} onClose={() => setEditing(false)} />}
      <Eyebrow className="mt-6">{k.implicit ? 'Unfiled case' : 'Case'} · <span className="font-mono normal-case tracking-normal">{k.reference}</span></Eyebrow>
      <h1 className="display text-h1 font-normal text-text mt-1 break-words">{k.title}</h1>
      <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-small" data-testid="case-facts">
        {facts.map(([label, value]) => (
          <div key={label} className="flex gap-2"><dt className="text-text-muted">{label}</dt><dd className="m-0 text-text">{value}</dd></div>
        ))}
      </dl>
      {k.description && <p className="mt-4 text-body text-text max-w-[72ch]">{k.description}</p>}

      <section className="mt-12" aria-labelledby="exhibits-heading">
        <div className="flex items-center justify-between gap-4 border-b border-border-strong pb-2">
          <h2 id="exhibits-heading" className="eyebrow m-0">Exhibits</h2>
          {canAdd && <Button variant="ghost" className="h-8 px-3" onClick={() => nav(`/cases/${k.id}/upload`)}>Add exhibit</Button>}
        </div>
        {items.error && <ErrorState className="mt-4" onRetry={() => void items.refetch()}>{items.error.message}</ErrorState>}
        {analyses.error && <ErrorState className="mt-4" onRetry={() => void analyses.refetch()}>{analyses.error.message}</ErrorState>}
        {items.isSuccess && rows.length === 0 && (
          <EmptyState className="mt-4" action={canAdd ? 'Add exhibit' : undefined} onAction={() => nav(`/cases/${k.id}/upload`)}>No exhibits have been taken into this case.</EmptyState>
        )}
        {(items.isPending || rows.length > 0) && (
          <div className="overflow-x-auto">
            {/* below 640 px each exhibit is one stacked block, so the assessment and the review state stay on screen (L-7) */}
            <table role="table" className="stack-sm w-full border-collapse" data-testid="case-items">
              <thead role="rowgroup">
                <tr role="row" className="border-b border-border">
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-12">Ex.</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4">File</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-64">Assessment</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-36">Review</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 w-48">Analyses</th>
                </tr>
              </thead>
              <tbody role="rowgroup">
                {items.isPending && (
                  <tr role="row" className="border-b border-border h-12"><td role="cell" colSpan={5}><Skeleton w={360} /></td></tr>
                )}
                {rows.map((it) => {
                  const itemChains = chains.get(it.media.id) ?? []
                  const intake = it.intakes[it.intakes.length - 1]
                  return (
                    <tr role="row" key={it.media.id} data-testid="case-item" data-label={it.label} className="border-b border-border align-baseline">
                      <td role="cell" data-th="Exhibit" className="py-3 pr-4 font-mono text-mono text-text">{it.label}</td>
                      <td role="cell" className="py-3 pr-4 min-w-0 max-w-[360px]">
                        <span className="block text-body font-medium text-text [overflow-wrap:anywhere]" data-testid="exhibit-name" title={it.media.original_filename ?? undefined}>{it.media.original_filename ?? 'Unnamed file'}</span>
                        <span className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-0.5 text-caption text-text-muted num">
                          <CodeChip copy={it.media.sha256} title="Copy full sha256">{shortHash(it.media.sha256)}</CodeChip>
                          {it.media.md5 && <CodeChip copy={it.media.md5} title="Copy full MD5">{`MD5 ${shortHash(it.media.md5)}`}</CodeChip>}
                          <span>{[it.media.mime_sniffed, fmtBytes(it.media.size_bytes), `taken in ${fmtDate(it.added_at)}`].filter(Boolean).join(' · ')}</span>
                        </span>
                        {intake?.supplied_by && <span className="block mt-0.5 text-caption text-text-muted truncate">custodian {intake.supplied_by}</span>}
                        {/* N-19 (browser QA P1-8): what the custodian stated at intake: the date acquired, the hash comparison, the notes */}
                        {intake?.date_acquired && <span className="block mt-0.5 text-caption text-text-muted num" data-testid="date-acquired">acquired (as stated) {fmtUtcMinutes(intake.date_acquired)}</span>}
                        {hashComparison(it.intakes) === 'mismatch' && (
                          <span className="block mt-0.5 text-caption text-danger" data-testid="custodian-mismatch">Custodian hash differs from the received file</span>
                        )}
                        {hashComparison(it.intakes) === 'match' && (
                          <span className="block mt-0.5 text-caption text-text-muted" data-testid="custodian-match">Custodian hash matches the received file</span>
                        )}
                        {intake?.notes && (intake.notes.length <= NOTES_INLINE
                          ? <span className="block mt-0.5 text-caption text-text-muted whitespace-pre-line break-words" data-testid="intake-notes">Intake notes: {intake.notes}</span>
                          : <Disclosure label="Intake notes" testId="intake-notes"><p className="m-0 text-caption text-text-muted whitespace-pre-line break-words max-w-[72ch]">{intake.notes}</p></Disclosure>)}
                      </td>
                      <td role="cell" className="py-3 pr-4">{itemChains[0] ? <Assessment a={itemChains[0].latest} /> : analyses.isPending ? <Skeleton w={120} /> : <span className="text-small text-text-muted">—</span>}</td>
                      <td role="cell" className="py-3 pr-4"><ReviewCell chain={itemChains[0]} details={details} /></td>
                      <td role="cell" className="py-3">{analyses.isPending ? <Skeleton w={100} /> : <AnalysisLinks chains={itemChains} />}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <CaseReports analysisIds={(analyses.data ?? []).map((x) => x.id)} exhibit={(aid: string) => labels.analyses.get(aid)} pending={analyses.isPending} />

      <CustodyLog caseId={k.id} reference={k.reference} myRole={k.my_role} labels={labels} names={names} />
      <MembersPanel caseId={k.id} canManage={manages} />
    </article>
  )
}
