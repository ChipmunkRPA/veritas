import { useEffect, useId, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '../../lib/api/client'
import { qk, type AnalysisDetail, type Me } from '../../lib/api/queries'
import { requestRerun, rerunNeedsReason, rerunRefusal, unpublishedRevision } from '../../lib/api/rerun'
import { namedAccount } from '../../lib/api/roles'
import { useBilling } from '../../lib/api/billing'
import { Button } from '../../components/ui'
import { textareaClass } from '../review/Dispositions'

/** Re-run with the pinned manifest (PRD CC-03; ARCHITECTURE §5.2): revision n+1 of this file with the same profile, detector versions
 *  and weights, fuser artefact and policy thresholds, after the original is re-hashed; the note goes to the custody log. The server
 *  refuses with 409 and names every pin that is no longer available, and the dialog shows its reasons. On success the new revision
 *  opens on its delta table. */
export default function RerunDialog({ open, analysis, me, onClose, onRefused }: {
  open: boolean
  analysis: AnalysisDetail
  me: Me | null | undefined
  onClose: () => void
  /** The server refused the request: the page then reads the analysis again (N-10b), and the dialog stays to say why. */
  onRefused?: () => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  const nav = useNavigate()
  const qc = useQueryClient()
  const billing = useBilling()
  const paid = billing.data?.enabled === true
  const [note, setNote] = useState('')
  const retry = unpublishedRevision(analysis.status, analysis.supersedes_id)
  const signed = !retry && rerunNeedsReason(analysis.review?.locked_at)
  const missing = signed && note.trim() === ''
  // N-9b: the re-run of a signed revision carries a named person, as its sign-off did; no form the server refuses is offered
  const refusal = rerunRefusal(analysis.status, analysis.review?.state, me?.role, analysis.supersedes_id)
    ?? (signed && me != null && !namedAccount(me) ? 'Re-running a signed-off revision needs a named account. Sign in, then re-run it.' : null)
  const run = useMutation({
    mutationFn: () => requestRerun(analysis.id, note),
    onSuccess: (accepted) => {
      void qc.invalidateQueries({ queryKey: qk.analysis(analysis.id) })
      // N-9b: the retry of an unpublished revision is a new revision of the one that revision would have replaced
      if (analysis.supersedes_id) void qc.invalidateQueries({ queryKey: qk.analysis(analysis.supersedes_id), exact: true })
      if (analysis.case_id) void qc.invalidateQueries({ queryKey: qk.caseAnalyses(analysis.case_id) })
      void qc.invalidateQueries({ queryKey: qk.cases() })
      onClose()
      nav(`/analyses/${accepted.analysis_id}/revision`)
    },
    onError: () => onRefused?.(),
  })

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) {
      run.reset()
      setNote('')
      d.showModal()
    } else if (!open && d.open) {
      d.close()
    }
    // Reset only when the dialog opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const busy = run.isPending
  const error = run.error
  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="rerun-dialog" onCancel={(e) => { e.preventDefault(); if (!busy) onClose() }}
      className="m-auto w-[560px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <h2 id={`${uid}-title`} className="m-0 px-6 pt-6 pb-4 text-[17px] leading-6 font-medium border-b border-border">
        Re-run <span className="font-mono text-mono text-text-muted">{analysis.id.slice(0, 8)}</span>
      </h2>
      <div className="px-6 py-5">
        {paid ? <p className="m-0 text-small text-text">Each new analysis costs {billing.data?.price_display} USD. Upload the source again and complete checkout to start another analysis. Your existing result stays available.</p> : <>{retry ? (
          <p className="m-0 text-small text-text">
            Revision {analysis.revision} was {analysis.status} before it published, so the revision it would have replaced is still
            current. A new revision of that one analyses this file again with the pins revision {analysis.revision} was queued with. The
            original is re-hashed first. Chipmunk Verify refuses when a pinned weight or detector is no longer available, and names it.
          </p>
        ) : <p className="m-0 text-small text-text">
          Revision {analysis.revision + 1} analyses this file again with the pinned manifest of revision {analysis.revision}: the same
          profile, detector versions and weights, fusion artefact and policy thresholds. The original is re-hashed first. Chipmunk Verify refuses
          when a pinned weight or detector is no longer available, and names it.
        </p>}
        {signed && (
          <p data-testid="rerun-signed" className="mt-3 mb-0 text-small text-text">
            This revision is signed off. The new revision supersedes it, so nothing more is issued from it. Its signer, the author of its
            conclusion, a case owner or an administrator can re-run it, and the reason is required.
          </p>
        )}
        <label className="mt-4 flex flex-col gap-1.5 text-small">
          <span className="text-text">Why re-run <span className="text-text-muted">({signed ? 'required; ' : ''}recorded in the custody log)</span></span>
          <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={3} maxLength={4000} disabled={busy} required={signed} className={textareaClass} />
        </label></>}
      </div>
      <div className="dialog-actions px-6 py-4 border-t border-border">
        {(refusal || error) && (
          <p role="alert" data-testid="rerun-refusal" className="m-0 mb-3 text-small text-danger">
            {/* the server's own refusal first: after it the page reads the analysis again, and the state it then holds gives a
                refusal of its own for the same reason (N-10b) */}
            {error instanceof ApiError ? <><span className="font-medium">{error.title}.</span> {error.message}</> : refusal ?? error?.message}
          </p>
        )}
        <div className="flex items-center justify-end gap-4">
          <Button variant="text" onClick={onClose} disabled={busy}>Close</Button>
          <Button variant="primary" onClick={() => {
            if (paid) { onClose(); nav(`/upload?kind=${analysis.kind}`) }
            else run.mutate()
          }} disabled={busy || billing.isPending || billing.isError || (!paid && (!!refusal || missing))}>{paid ? 'New paid analysis' : busy ? 'Queuing…' : 'Re-run'}</Button>
          {billing.isError && <span role="alert" className="text-small text-danger">Pricing is unavailable. Reopen this dialog to try again.</span>}
        </div>
      </div>
    </dialog>
  )
}
