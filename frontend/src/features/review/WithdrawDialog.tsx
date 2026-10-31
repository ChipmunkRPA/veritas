import { useEffect, useId, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ApiError, apiClient, unwrap } from '../../lib/api/client'
import { qk, type AnalysisDetail } from '../../lib/api/queries'
import { Button } from '../../components/ui'
import { textareaClass } from './Dispositions'

/** Withdraw a signed revision (D-70; design review DS-14; `POST /analyses/{id}/withdraw`): a required reason, what the
 *  withdrawal does, and the destructive button. The reason is recorded in the custody log (`status_changed`) and printed with
 *  the state wherever the revision is shown. */
export default function WithdrawDialog({ open, analysis, onClose }: { open: boolean; analysis: AnalysisDetail; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  const qc = useQueryClient()
  const [reason, setReason] = useState('')
  const withdraw = useMutation({
    mutationFn: () => unwrap(apiClient.POST('/api/v2/analyses/{aid}/withdraw', { params: { path: { aid: analysis.id } }, body: { reason: reason.trim() } })),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.analysis(analysis.id) })
      void qc.invalidateQueries({ queryKey: qk.analyses() })
      void qc.invalidateQueries({ queryKey: qk.cases() })
      if (analysis.case_id) for (const key of [qk.custody(analysis.case_id), qk.caseAnalyses(analysis.case_id)]) void qc.invalidateQueries({ queryKey: key })
      onClose()
    },
  })

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) {
      withdraw.reset()
      setReason('')
      d.showModal()
    } else if (!open && d.open) {
      d.close()
    }
    // Reset only when the dialog opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const busy = withdraw.isPending
  const error = withdraw.error
  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="withdraw-dialog" onCancel={(e) => { e.preventDefault(); if (!busy) onClose() }}
      className="m-auto w-[520px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <h2 id={`${uid}-title`} className="m-0 px-6 pt-6 pb-4 text-[17px] leading-6 font-medium border-b border-border">
        Withdraw revision {analysis.revision} <span className="font-mono text-mono text-text-muted">{analysis.id.slice(0, 8)}</span>
      </h2>
      <div className="px-6 py-5">
        <p className="m-0 text-small text-text">This revision will read Withdrawn everywhere and no report can be generated from it.</p>
        <p className="mt-2 mb-0 text-small text-text-muted">Reports that were issued before stay in the record of the case. A withdrawal cannot be taken back, and a withdrawn revision cannot be re-run: to analyse the file anew, take it into the case again.</p>
        <label className="mt-4 flex flex-col gap-1.5 text-small">
          <span className="text-text">Reason <span className="text-text-muted">(required; recorded in the custody log and shown with the revision)</span></span>
          <textarea value={reason} onChange={(e) => setReason(e.target.value)} rows={3} maxLength={4000} disabled={busy} required aria-required="true" className={textareaClass} />
        </label>
      </div>
      <div className="dialog-actions px-6 py-4 border-t border-border">
        {error && (
          <p role="alert" className="m-0 mb-3 text-small text-danger">
            {error instanceof ApiError ? <><span className="font-medium">{error.title}.</span> {error.message}</> : error.message}
          </p>
        )}
        <div className="flex items-center justify-end gap-4">
          <Button variant="text" onClick={onClose} disabled={busy}>Keep the revision</Button>
          <Button variant="danger" onClick={() => withdraw.mutate()} disabled={busy || reason.trim() === ''}>{busy ? 'Withdrawing…' : 'Withdraw revision'}</Button>
        </div>
      </div>
    </dialog>
  )
}
