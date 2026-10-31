import { Link } from 'react-router-dom'
import { REVIEW_STATE, type ReviewState } from '../../components/ui/Marks'
import type { Case } from '../../lib/api/queries'
import { AWAITING_SECOND_REVIEW } from '../review/review'

/** Review states in the order an analysis moves through them; the words are the case table's. */
export const REVIEW_ORDER: readonly ReviewState[] = ['machine', 'in_review', 'reviewed', 'approved', 'disputed', 'superseded', 'withdrawn']
export const REVIEW_LABEL: Record<ReviewState, string> = {
  machine: 'not reviewed', in_review: 'in review', reviewed: 'reviewed', approved: 'second review complete',
  disputed: 'disputed by the peer reviewer', superseded: 'superseded', withdrawn: 'withdrawn',
}
// N-19b (design review, P2 "screens" 1): no two states share a fill, and the legend carries each fill as a swatch, so the
// bar can be read without its order. The inks step from the hollow "not reviewed" to the full "reviewed"; the second review
// has the accent, a dispute the warning step of the scale, a withdrawal the danger colour; a superseded revision is hollow.
const FILL: Record<ReviewState, string> = {
  machine: 'bg-border-strong', in_review: 'bg-graphic', reviewed: 'bg-text', approved: 'bg-accent', disputed: 'bg-verdict-4',
  superseded: 'review-fill-superseded', withdrawn: 'bg-danger',
}
const AWAITING_FILL = 'bg-text-muted'
const fillOf = (s: StateCount) => (s.label === AWAITING_SECOND_REVIEW ? AWAITING_FILL : FILL[s.state])
const OPEN_STATES: readonly ReviewState[] = ['machine', 'in_review']
const WORKLIST_ROWS = 5

type CaseCount = Pick<Case, 'id' | 'reference' | 'title' | 'implicit' | 'analysis_count' | 'review_counts' | 'updated_at'> & { demo?: boolean }
interface StateCount { state: ReviewState; label: string; count: number; glyph: string }
/** A revision that awaits its second review (N-19a): what the queue shows of it. */
export interface Awaiting { id: string; filename: string; case_id: string | null; revision: number }

/** What waits for an examiner across the cases the caller can see (L-7, design P2: a worklist instead of headline totals and a
 *  one-colour status bar). Counts are analysis revisions by review state; a revision without a recorded state is counted apart. */
export function reviewOverview(all: readonly CaseCount[], awaiting = 0) {
  // N-19d (browser QA "Demo tour"): a case that holds only samples of the guided demo is no work for an examiner; it is
  // left out of the counts and of "Needs review" (it stays in the library below)
  const cases = all.filter((c) => !c.demo)
  // N-19a (P1-12): the count of `reviewed` holds the revisions that await their second review; they are counted apart, with
  // the half glyph, and never as reviewed
  const states = REVIEW_ORDER.flatMap((state): StateCount[] => {
    const count = cases.reduce((sum, c) => sum + (c.review_counts?.[state] ?? 0), 0)
    if (state !== 'reviewed') return [{ state, label: REVIEW_LABEL[state], count, glyph: REVIEW_STATE[state].glyph }]
    const waiting = Math.min(awaiting, count)
    const reviewed = { state, label: REVIEW_LABEL[state], count: count - waiting, glyph: REVIEW_STATE[state].glyph }
    return waiting > 0 ? [{ state, label: AWAITING_SECOND_REVIEW, count: waiting, glyph: REVIEW_STATE.in_review.glyph }, reviewed] : [reviewed]
  })
  const total = cases.reduce((sum, c) => sum + c.analysis_count, 0)
  const recorded = states.reduce((sum, s) => sum + s.count, 0)
  const waiting = cases.map((c) => ({ case: c, machine: c.review_counts?.machine ?? 0, inReview: c.review_counts?.in_review ?? 0 }))
    .filter((w) => w.machine + w.inReview > 0)
    .sort((a, b) => b.inReview - a.inReview || b.machine - a.machine || Date.parse(b.case.updated_at) - Date.parse(a.case.updated_at))
  return { states, total, unrecorded: Math.max(0, total - recorded), waiting, open: states.filter((s) => OPEN_STATES.includes(s.state)).reduce((sum, s) => sum + s.count, 0) }
}

export default function CaseOverview({ cases, awaiting = [] }: { cases: readonly CaseCount[]; awaiting?: readonly Awaiting[] }) {
  const summary = reviewOverview(cases, awaiting.length)
  const caseOf = new Map(cases.map((c) => [c.id, c]))
  if (summary.total === 0) return null
  const shown = summary.waiting.slice(0, WORKLIST_ROWS)
  return (
    <section aria-labelledby="review-overview-heading" className="mt-8 grid gap-x-12 gap-y-6 min-[960px]:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]" data-testid="workspace-overview">
      <figure className="m-0 min-w-0">
        <figcaption id="review-overview-heading" className="eyebrow">Review state</figcaption>
        <p className="mt-2 mb-0 text-small text-text num">
          {summary.open === 0 ? (awaiting.length > 0 ? 'Nothing waits for an examiner.' : 'Nothing waits for review.') : `${summary.open} of ${summary.total} analysis ${summary.total === 1 ? 'revision waits' : 'revisions wait'} for review.`}
          {awaiting.length > 0 && ` ${awaiting.length} ${awaiting.length === 1 ? 'awaits' : 'await'} a second review.`}
        </p>
        <div className="mt-3 flex gap-px h-2 overflow-hidden rounded-xs bg-surface-3" aria-hidden data-testid="review-bar">
          {summary.states.filter((s) => s.count > 0).map((s) => <span key={s.label} data-fill={fillOf(s)} className={fillOf(s)} style={{ width: `${100 * s.count / summary.total}%` }} />)}
        </div>
        <ul className="m-0 mt-3 p-0 list-none grid grid-cols-2 gap-x-6 gap-y-1 text-caption max-w-[420px]">
          {summary.states.filter((s) => s.count > 0).map((s) => (
            <li key={s.label} className="flex items-baseline gap-2 text-text-muted">
              <span aria-hidden data-testid="review-swatch" data-fill={fillOf(s)} className={`inline-block w-2.5 h-2.5 shrink-0 self-center rounded-[1px] ${fillOf(s)}`} />
              <span aria-hidden className="inline-block w-3 text-center text-text">{s.glyph}</span>
              <span>{s.label}</span><span className="ml-auto font-mono text-text num">{s.count}</span>
            </li>
          ))}
          {summary.unrecorded > 0 && <li className="flex items-baseline gap-2 text-text-muted"><span aria-hidden className="inline-block w-3" /><span>review not recorded</span><span className="ml-auto font-mono text-text num">{summary.unrecorded}</span></li>}
        </ul>
      </figure>
      <div className="min-w-0">
        <h2 className="m-0 eyebrow">Needs review</h2>
        {shown.length === 0 ? <p className="mt-2 mb-0 text-small text-text-muted">No case has an analysis waiting for an examiner.</p> : (
          <ul className="m-0 mt-2 p-0 list-none border-t border-border" data-testid="needs-review">
            {shown.map(({ case: c, machine, inReview }) => (
              <li key={c.id} className="flex items-baseline gap-x-4 gap-y-0.5 flex-wrap border-b border-border py-2 text-small">
                {/* N-19b: a long title takes two lines here too; the whole title is its hover text */}
                <Link to={`/cases/${c.id}`} title={c.title} className="min-w-0 flex-1 basis-48 line-clamp-2 [overflow-wrap:anywhere] text-text hover:underline underline-offset-[3px]">
                  <span className="font-mono text-mono text-text-muted mr-2">{c.reference}</span>{c.implicit ? `Unfiled · ${c.title}` : c.title}
                </Link>
                <span className="text-text-muted num whitespace-nowrap">{[inReview ? `${inReview} in review` : null, machine ? `${machine} not reviewed` : null].filter(Boolean).join(' · ')}</span>
              </li>
            ))}
          </ul>
        )}
        {summary.waiting.length > shown.length && <p className="mt-2 mb-0 text-caption text-text-muted num">{summary.waiting.length - shown.length} more {summary.waiting.length - shown.length === 1 ? 'case is' : 'cases are'} in the library below.</p>}
        {awaiting.length > 0 && (
          // N-19a (browser QA P1-12; design review DS-09): what waits for a second reviewer, each a link to its review
          <div className="mt-6" data-testid="awaiting-second-review">
            <h2 className="m-0 eyebrow">Awaiting second review</h2>
            <ul className="m-0 mt-2 p-0 list-none border-t border-border">
              {awaiting.slice(0, WORKLIST_ROWS).map((a) => {
                const c = a.case_id ? caseOf.get(a.case_id) : undefined
                return (
                  <li key={a.id} className="flex items-baseline gap-x-4 gap-y-0.5 flex-wrap border-b border-border py-2 text-small">
                    <Link to={`/analyses/${a.id}/review`} title={a.filename} className="min-w-0 flex-1 basis-48 line-clamp-2 [overflow-wrap:anywhere] text-text hover:underline underline-offset-[3px]">
                      {c && <span className="font-mono text-mono text-text-muted mr-2">{c.reference}</span>}{a.filename}
                    </Link>
                    <span className="text-text-muted num whitespace-nowrap"><span aria-hidden className="inline-block w-3 text-center text-text mr-1">{REVIEW_STATE.in_review.glyph}</span>signed · revision {a.revision}</span>
                  </li>
                )
              })}
            </ul>
            {awaiting.length > WORKLIST_ROWS && <p className="mt-2 mb-0 text-caption text-text-muted num">{awaiting.length - WORKLIST_ROWS} more in the analyses list.</p>}
          </div>
        )}
      </div>
    </section>
  )
}
