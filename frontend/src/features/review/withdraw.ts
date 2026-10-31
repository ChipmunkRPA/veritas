import { useCase, useConclusion, useMe, useSignoffs, type AnalysisDetail, type Conclusion, type Me, type Signoff } from '../../lib/api/queries'
import { EXAMINER_ROLES, namedAccount } from '../../lib/api/roles'

/** Whether "Withdraw…" is offered on this revision (D-70; `app/api/v2_review.py` `revision_custodian`): the revision is signed
 *  and not withdrawn yet, and the reader is a named account with an examiner's instance role who is an administrator, an owner
 *  of the case, the examiner who signed the revision or the author of its conclusion. The server applies the same rule. */
export function mayWithdraw(a: Pick<AnalysisDetail, 'status' | 'review'>, me: Me | null | undefined, caseRole: string | null | undefined,
  conclusion: Pick<Conclusion, 'author_id'> | null | undefined, records: readonly Pick<Signoff, 'kind' | 'principal_id'>[] | undefined): boolean {
  if (a.status !== 'done' || !a.review?.locked_at || a.review.state === 'withdrawn') return false
  if (!me || !namedAccount(me) || !(me.roles ?? []).some((r) => EXAMINER_ROLES.has(r))) return false
  if (me.roles.includes('administrator') || caseRole === 'owner') return true
  return conclusion?.author_id === me.id || (records ?? []).some((s) => s.kind === 'examiner' && s.principal_id === me.id)
}

/** `mayWithdraw` for the page: the sign-off records and the conclusion are read for a signed revision only. */
export function useMayWithdraw(analysis: AnalysisDetail | undefined): boolean {
  const signed = !!analysis && analysis.status === 'done' && !!analysis.review?.locked_at && analysis.review.state !== 'withdrawn'
  const me = useMe()
  const kase = useCase(signed ? analysis?.case_id ?? '' : '')
  const records = useSignoffs(analysis?.id ?? '', signed)
  const conclusion = useConclusion(analysis?.id ?? '', signed)
  return !!analysis && signed && mayWithdraw(analysis, me.data, kase.data?.my_role, conclusion.data, records.data)
}
