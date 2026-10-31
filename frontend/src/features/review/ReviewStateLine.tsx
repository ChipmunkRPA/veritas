import { Link } from 'react-router-dom'
import { useCaseAnalyses, useMembers, useSignoffs, type AnalysisDetail } from '../../lib/api/queries'
import { fmtDateLong } from '../../lib/format'
import { ReviewStateMark } from '../../components/ui'
import { reviewLine, reviewLineKind, signerName } from '../analysis/answer'
import { supersedes } from '../cases/chains'

/** Shared review line for the workspace, review panel and export dialog. Only recorded sign-offs and direct revision links
 * supply details. Missing/failed queries do not downgrade the authoritative review state or invent an assignee. */
export default function ReviewStateLine({ analysis, enabled = true }: { analysis: AnalysisDetail; enabled?: boolean }) {
  const state = analysis.review?.state ?? 'machine'
  const signed = state === 'reviewed' || state === 'approved' || state === 'disputed'
  const records = useSignoffs(analysis.id, enabled && signed)
  const members = useMembers(enabled && signed ? analysis.case_id ?? '' : '')
  const revisions = useCaseAnalyses(enabled && state === 'superseded' ? analysis.case_id ?? '' : '')
  const kind = reviewLineKind(state)
  const signoff = records.data?.find((s) => s.kind === kind && s.analysis_id === analysis.id && s.analysis_revision === analysis.revision)
  // the name the sign-off carries: an administrator who signs on a single-user instance is no member of the case (N-10b)
  const name = signoff ? signerName(signoff, members.data?.find((m) => m.principal_id === signoff.principal_id)?.display_name) : undefined
  // The server permits one direct successor that is queued, running or published; a cancelled or failed one replaced nothing
  // (K-2) and is not it. Do not guess if incomplete/ambiguous evidence reaches the client.
  const successors = revisions.data?.filter((a) => a.supersedes_id === analysis.id && a.case_id === analysis.case_id && supersedes(a.status)) ?? []
  const successorId = successors.length === 1 ? successors[0].id : null
  // N-19a: once the peer review is recorded the line names the examiner who signed as well
  const first = kind === 'peer_review'
    ? records.data?.find((s) => s.kind === 'examiner' && s.analysis_id === analysis.id && s.analysis_revision === analysis.revision) : undefined
  const examiner = first ? { principal_id: first.principal_id, signed_at: first.signed_at,
    name: signerName(first, members.data?.find((m) => m.principal_id === first.principal_id)?.display_name) } : null
  const line = reviewLine(analysis.review, fmtDateLong, { signoff, reviewerName: name, examiner, successorId, loading: records.isLoading })

  return (
    <span className="inline-flex items-baseline gap-x-2 gap-y-1 flex-wrap">
      <ReviewStateMark state={line.state} outstanding={state === 'reviewed' && !!analysis.review?.requires_second_review}>{line.text}</ReviewStateMark>
      {state === 'superseded' && successorId && (
        <Link to={`/analyses/${successorId}/revision?view=expert`} className="text-small text-accent underline underline-offset-[3px]"
          aria-label={`Compare this analysis with ${successorId.slice(0, 8)}`}>Compare ▸</Link>
      )}
    </span>
  )
}
