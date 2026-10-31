import { useEffect, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Lock } from 'lucide-react'
import { qk, useAnnotations, useConclusion, useMe, useMembers, useSignoffs, type AnalysisDetail, type Finding } from '../../lib/api/queries'
import { fmtDateLong } from '../../lib/format'
import { ErrorState, SkeletonRows } from '../../components/ui'
import ReviewStateLine from './ReviewStateLine'
import ConclusionBlock from './ConclusionBlock'
import Dispositions from './Dispositions'
import NotesThread from './NotesThread'
import SignoffBlock from './SignoffBlock'
import { mayExamine, namedAccount } from '../../lib/api/roles'
import { reviewGate } from './review'
import { findingsFailed } from '../analysis/vocabulary'

/** After a write to the examiner layer: the review block of the detail, the annotations, conclusion and sign-offs, the findings
 *  (their joined examiner layer) and the case views that show the review state and the custody log. */
export function useReviewRefresh(id: string, caseId: string | null | undefined) {
  const qc = useQueryClient()
  return () => {
    void qc.invalidateQueries({ queryKey: qk.analysis(id), exact: true })
    for (const key of [qk.annotations(id), qk.conclusion(id), qk.signoffs(id), qk.findings(id)]) void qc.invalidateQueries({ queryKey: key })
    if (caseId) for (const key of [qk.custody(caseId), qk.caseAnalyses(caseId)]) void qc.invalidateQueries({ queryKey: key })
  }
}

/** Locked banner (UX §5.14, QA C "Superseded / locked"): shown once the examiner sign-off has frozen the revision. */
export function LockedBanner({ analysis }: { analysis: AnalysisDetail }) {
  const locked = analysis.review?.locked_at
  if (!locked) return null
  return (
    <p role="status" data-testid="locked-banner" className="mt-4 mb-0 flex items-baseline gap-2.5 border-y border-border py-3 text-small text-text max-w-[88ch]">
      <Lock size={14} strokeWidth={1.5} aria-hidden className="shrink-0 self-center text-text-muted" />
      <span><span className="font-medium">Signed off and locked</span> <span className="text-text-muted num">· {fmtDateLong(locked)}</span>. This revision is read-only: statuses, notes and the conclusion cannot change. A later change needs a new revision.</span>
    </p>
  )
}

/** Review section of the analysis workspace (ARCHITECTURE §5.4, §7.2 `review`; PRD RW-01…RW-04; UX §5.13, §5.14): examiner status of
 *  each finding, the notes thread, the conclusion and the sign-off. The examiner layer is set apart by the accent rule and N-chips
 *  and never alters the machine assessment (RW-01). */
export default function ReviewPanel({ analysis, findings, caseRole, plain = false }: {
  analysis: AnalysisDetail; findings: readonly Finding[]
  /** The caller's role on the analysis's case (null: not a member; undefined: not loaded yet). */
  caseRole?: string | null
  /** Plain view, which holds no evidence ledger (N-19n: where the findings could not be written, it names the expert view) */
  plain?: boolean
}) {
  const id = analysis.id
  const annotations = useAnnotations(id)
  // The review block names the current conclusion, so the route (404 before one exists) is read only once there is one.
  const conclusion = useConclusion(id, analysis.review?.conclusion_id != null)
  const signoffs = useSignoffs(id)
  const me = useMe()
  const members = useMembers(analysis.case_id ?? '')
  const refresh = useReviewRefresh(id, analysis.case_id)
  const state = reviewGate(analysis)
  // L-7: statuses, notes and the conclusion are offered only to a caller the server lets write them
  const examines = mayExamine(me.data, caseRole)
  const gate = state.writable && !examines
    ? { writable: false, reason: me.isSuccess && caseRole !== undefined ? 'You can read this review. Statuses, notes and the conclusion are recorded by the examiners of the case.' : null }
    : state
  // N-9b: a conclusion and a sign-off carry a named person; the local examiner of a single-user instance, the bootstrap key and
  // an API client are refused by the API, so they are offered "Sign in" and no form
  const unnamed = me.isSuccess && !namedAccount(me.data)
  const names = useMemo(() => {
    const m = new Map<string, string>()
    for (const x of members.data ?? []) m.set(x.principal_id, x.display_name)
    if (me.data) m.set(me.data.id, me.data.display_name)
    return m
  }, [members.data, me.data])
  const loading = annotations.isPending || conclusion.isLoading || signoffs.isPending
  const error = annotations.error ?? conclusion.error ?? signoffs.error
  // N-19n (browser QA Q2-09): once the examiner layer of this analysis was drawn, it stays drawn whatever the page loads
  // afterwards. A first conclusion saved from a tab that another tab had overtaken is answered 409; the page then reads
  // the analysis again (`queryClient.ts`), its review block names a conclusion for the first time, the query of the
  // conclusion starts, and the skeleton rows took the place of the blocks meanwhile: the form was unmounted with what
  // was typed in it, and the sentence of N-19j was never shown. The state is the analysis's: another analysis in this
  // panel is loaded behind the skeleton, as before
  const [drawnFor, setDrawnFor] = useState<string | null>(null)
  useEffect(() => { if (!loading && !error) setDrawnFor(id) }, [loading, error, id])
  const drawn = drawnFor === id
  const pending = loading && !drawn
  const rows = annotations.data ?? []

  return (
    <section id="section-review" data-testid="review" aria-labelledby="review-heading" className="scroll-mt-14 mt-12 border-t border-border-strong pt-8">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h2 id="review-heading" className="m-0 eyebrow">Review</h2>
        <ReviewStateLine analysis={analysis} />
      </div>
      <p className="mt-2 mb-0 text-small text-text-muted max-w-[72ch]">The examiner layer. Statuses, notes and the conclusion sit beside the machine assessment and never change it; a report prints both in separate blocks.</p>
      {!gate.writable && gate.reason && !analysis.review?.locked_at && <p role="status" className="mt-4 mb-0 text-small text-text">{gate.reason}</p>}
      {error && (
        <ErrorState className="mt-6" onRetry={() => { void annotations.refetch(); void conclusion.refetch(); void signoffs.refetch() }}>{error.message}</ErrorState>
      )}
      {pending && !error && <SkeletonRows className="mt-6" widths={['70%', '64%', '58%', '52%']} />}
      {!pending && (!error || drawn) && (
        <>
          <Dispositions analysisId={id} findings={findings} annotations={rows} writable={gate.writable} onChanged={refresh} notWritten={findingsFailed(analysis.caveats) ? (plain ? 'expert' : 'ledger') : undefined} />
          <NotesThread analysisId={id} annotations={rows} names={names} meId={me.data?.id ?? null} writable={gate.writable} onChanged={refresh} />
          <ConclusionBlock analysis={analysis} conclusion={conclusion.data ?? null} names={names} writable={gate.writable && !unnamed}
            needsAccount={gate.writable && unnamed} onChanged={refresh} />
          <SignoffBlock analysis={analysis} findings={findings} conclusion={conclusion.data ?? null} records={signoffs.data ?? []} annotations={rows}
            me={me.data ?? null} names={names} onChanged={refresh} />
        </>
      )}
    </section>
  )
}
