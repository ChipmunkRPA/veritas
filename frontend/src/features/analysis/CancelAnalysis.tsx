import { useId, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Button } from '../../components/ui'
import { requestCancellation } from '../../lib/api/cancel'
import { EXAMINER_ROLES } from '../../lib/api/roles'
import { qk, type AnalysisDetail } from '../../lib/api/queries'
import { inputClass } from '../cases/NewCaseDialog'

/** Pending cancellation is durable in the detail response; acceptance does not mean a native model call has stopped. */
export default function CancelAnalysis({ analysis, role, caseRole }: {
  analysis: AnalysisDetail; role: string | null | undefined; caseRole: string | null | undefined
}) {
  const qc = useQueryClient()
  const refresh = () => {
    // A terminal detail can arrive before its SSE event; refresh completed checks as well.
    void qc.invalidateQueries({ queryKey: qk.analysis(analysis.id) })
    // N-9b: a cancelled re-run supersedes nothing, so the revision it would have replaced is current again
    if (analysis.supersedes_id) void qc.invalidateQueries({ queryKey: qk.analysis(analysis.supersedes_id), exact: true })
    void qc.invalidateQueries({ queryKey: qk.analyses() })
    void qc.invalidateQueries({ queryKey: qk.cases() })
    if (analysis.case_id) {
      void qc.invalidateQueries({ queryKey: qk.caseAnalyses(analysis.case_id) })
      void qc.invalidateQueries({ queryKey: qk.custody(analysis.case_id) })
    }
  }
  const [note, setNote] = useState('')
  const noteId = useId()
  const cancel = useMutation({ mutationFn: () => requestCancellation(analysis.id, note), onSettled: refresh })
  const active = analysis.status === 'queued' || analysis.status === 'running'
  if (!active) return null
  const writable = !analysis.locked_at && !analysis.review?.locked_at
    && analysis.review?.state !== 'superseded' && analysis.review?.state !== 'withdrawn'
  const allowed = writable && !!role && EXAMINER_ROLES.has(role)
    && (role === 'administrator' || (!!analysis.case_id && (caseRole === 'owner' || caseRole === 'examiner')))
  const pending = analysis.cancel_requested || cancel.isSuccess
  if (!allowed && !pending) return null

  return (
    <div data-testid="analysis-cancellation" className="mt-4 text-small">
      {pending
        ? <p role="status" className="m-0 text-text">{analysis.cancel_requested || cancel.data?.cancel_requested
          ? 'Cancellation requested; waiting for the current check to stop.'
          : 'Cancellation accepted; updating the analysis status.'}</p>
        : (
          <>
            {/* N-19a (design review DS-12): the page of a cancelled run says who cancelled it, when, and this note */}
            <label htmlFor={noteId} className="block text-text-muted">Note <span className="text-caption">(optional; recorded with the cancellation and shown on this analysis)</span></label>
            <input id={noteId} type="text" value={note} maxLength={1000} onChange={(e) => setNote(e.target.value)} disabled={cancel.isPending}
              className={`${inputClass} mt-1 mb-3 max-w-[56ch]`} />
            <Button variant="ghost" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
              {cancel.isPending ? 'Requesting cancellation…' : 'Cancel analysis'}
            </Button>
          </>
        )}
      <p className="mt-2 mb-0 text-text-muted">The source and completed checks are retained.</p>
      {cancel.error && <p role="alert" className="mt-2 mb-0 text-danger">{cancel.error.message}</p>}
    </div>
  )
}
