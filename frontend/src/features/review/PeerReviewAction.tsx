import { useNavigate } from 'react-router-dom'
import { useCase, useConclusion, useMe, useSignoffs, type AnalysisDetail } from '../../lib/api/queries'
import { namedAccount } from '../../lib/api/roles'
import { Button } from '../../components/ui'
import { signoffAction } from './review'

/** "Record peer review…" in the header of an analysis that awaits its second review (N-19a; browser QA P1-12; design review
 *  DS-09): offered to the reader who may record it, by the rule of the sign-off block at the foot of the page
 *  (`signoffAction`). It opens that block's dialog. */
export default function PeerReviewAction({ analysis }: { analysis: AnalysisDetail }) {
  const nav = useNavigate()
  const waits = !!analysis.awaits_second_review
  const me = useMe()
  const kase = useCase(waits ? analysis.case_id ?? '' : '')
  const records = useSignoffs(analysis.id, waits)
  const conclusion = useConclusion(analysis.id, waits)
  if (!waits || !me.data || !namedAccount(me.data) || !kase.data || !records.data || conclusion.data === undefined) return null
  const action = signoffAction(analysis, conclusion.data, records.data, me.data.id, { roles: me.data.roles, caseRole: kase.data.my_role })
  if (action !== 'peer_review') return null
  return <Button variant="primary" data-testid="header-peer-review" onClick={() => nav(`/analyses/${analysis.id}/review?view=expert&signoff=peer`)}>Record peer review…</Button>
}
