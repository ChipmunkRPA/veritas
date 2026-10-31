import { useEffect, useId, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import type { AnalysisDetail, Me } from '../../lib/api/queries'
import {
  EXPORT_CHOICES, PROFILE_WORD, downloadPackage, downloadReport, exportActions, exportRefusal, generatePackage, generateReport, reportsKey,
  saveInstanceKey, useInstanceKey, useReports, type ExportChoice, type Report,
} from '../../lib/api/reports'
import { fmtBytes, fmtDateLong } from '../../lib/format'
import { EXAMINER_ROLES } from '../../lib/api/roles'
import { Button } from '../../components/ui'
import ReviewStateLine from '../review/ReviewStateLine'

const JOB_WORD: Record<string, string> = { queued: 'queued', running: 'generating', done: 'done', failed: 'failed' }

function note(state: string, requiresSecondReview: boolean, signaturesValid: boolean): string {
  if (!signaturesValid && state !== 'superseded' && state !== 'withdrawn') return 'A review record of this revision does not verify: the signature of its conclusion or of a sign-off does not match its content. Nothing is issued; reports are drafts that say so.'
  if (state === 'superseded') return 'This revision is superseded. Generate reports from the newer revision.'
  if (state === 'withdrawn') return 'This revision was withdrawn. New reports cannot be generated.'
  if (state === 'in_review') return 'In review. This revision has not been signed off; reports are drafts.'
  if (state === 'disputed') return 'The peer reviewer does not concur, so nothing is issued from this revision. Reports are drafts that print the peer decision; the way forward is a new revision.'
  if (state === 'reviewed' || state === 'approved') {
    return requiresSecondReview && state === 'reviewed'
      ? 'Signed off by the examiner. This case needs the peer review before anything is issued; until then reports are drafts that say the second review is outstanding.'
      : 'Signed off. Reports print the review status line and an issued id; the bundle can be issued.'
  }
  return 'Not reviewed. Every page will carry “Machine assessment — not reviewed by an examiner”, and the report id carries DRAFT.'
}

/** Share and export (UX §5.16; PRD RP-03, RP-05; D-10, D-25, D-27): one of the four report profiles or the evidence bundle, exported
 *  as a draft or issued once signed off, then downloaded; the analysis's earlier reports below. Veritas is offline: "share" means
 *  download. Paper size and the include options of the UX sketch have no API field and are not offered. The last block names
 *  the instance key: a recipient verifies a bundle against that key, which must reach them apart from the bundle. */
/** What the dialog says once a report is saved. N-19d (browser QA "Copy"): a one-page report read "(1 pages)". */
export const downloadedLine = (report: { display_id: string; page_count?: number | null }): string =>
  `Downloaded ${report.display_id} (${report.page_count ?? '?'} ${report.page_count === 1 ? 'page' : 'pages'}).`

export default function ExportDialog({ open, analysis, me, onClose }: {
  open: boolean
  analysis: AnalysisDetail
  me: Me | null | undefined
  onClose: () => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  const qc = useQueryClient()
  const [choice, setChoice] = useState<ExportChoice>('forensic')
  const [original, setOriginal] = useState(false)
  const [progress, setProgress] = useState<string | null>(null)
  const reports = useReports(analysis.id, open)
  const signingKey = useInstanceKey(open)
  const state = analysis.review?.state ?? 'machine'
  const second = !!analysis.review?.requires_second_review
  const refusal = exportRefusal(analysis.status, state)
    ?? (me && !EXAMINER_ROLES.has(me.role) ? 'Exporting needs the examiner role; the reports below can be downloaded.' : null)
  const valid = analysis.review?.signatures_valid !== false
  const actions = exportActions(choice, state, second, valid)

  const run = useMutation({
    mutationFn: async (kind: 'draft' | 'issue') => {
      const label = choice === 'bundle' ? 'Forensic report for the bundle' : PROFILE_WORD[choice]
      setProgress(`${label}: requested`)
      // a bundle takes the forensic report that already holds the record as it stands, when there is one (N-19)
      const report = await generateReport(analysis.id, choice === 'bundle' ? 'forensic' : choice, (s) => setProgress(`${label}: ${JOB_WORD[s] ?? s}`), choice === 'bundle')
      void qc.invalidateQueries({ queryKey: reportsKey(analysis.id) })
      if (choice !== 'bundle') {
        await downloadReport(report)
        return downloadedLine(report)
      }
      setProgress('Evidence bundle: requested')
      const pkg = await generatePackage(analysis.id, report.id, kind, original, (s) => setProgress(`Evidence bundle: ${JOB_WORD[s] ?? s}`))
      await downloadPackage(pkg)
      return `Downloaded ${pkg.filename ?? 'the evidence bundle'} (${pkg.file_count ?? '?'} files, ${kind === 'issue' ? 'issued' : 'draft'}).`
    },
    onSettled: () => setProgress(null),
  })
  const download = useMutation({ mutationFn: (r: Report) => downloadReport(r) })

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) {
      run.reset()
      download.reset()
      d.showModal()
    } else if (!open && d.open) {
      d.close()
    }
    // Reset only when the dialog opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const busy = run.isPending
  const items = reports.data ?? []

  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="export-dialog" onCancel={(e) => { e.preventDefault(); if (!busy) onClose() }}
      className="m-auto w-[720px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <h2 id={`${uid}-title`} className="m-0 px-6 pt-6 pb-4 text-[17px] leading-6 font-medium border-b border-border">
        Export <span className="font-mono text-mono text-text-muted">{analysis.id.slice(0, 8)}</span>
      </h2>
      <fieldset data-testid="export-controls" disabled={busy || !!refusal} className="m-0 min-w-0 border-0 px-6 py-5">
        <div role="radiogroup" aria-labelledby={`${uid}-what`} className="flex flex-col">
          <span id={`${uid}-what`} className="sr-only">What to export</span>
          {EXPORT_CHOICES.map((c) => (
            <label key={c.id} className="grid grid-cols-[20px_190px_minmax(0,1fr)] items-baseline gap-x-2 min-h-9 py-1 text-small max-[639px]:grid-cols-[20px_minmax(0,1fr)]">
              <input type="radio" name={`${uid}-choice`} value={c.id} checked={choice === c.id} onChange={() => setChoice(c.id)} disabled={busy} className="accent-text translate-y-px" />
              <span className="text-text font-medium">{c.label}</span>
              <span className="text-text-muted max-[639px]:col-start-2">{c.description}</span>
            </label>
          ))}
        </div>
        {choice === 'bundle' && (
          <label className="mt-3 ml-7 inline-flex items-center gap-2.5 min-h-8 text-small text-text">
            <input type="checkbox" checked={original} onChange={(e) => setOriginal(e.target.checked)} disabled={busy} className="accent-text" />
            Original file in the bundle <span className="text-text-muted num">({fmtBytes(analysis.size_bytes)}; needed to reproduce the analysis offline)</span>
          </label>
        )}
      </fieldset>
      <div className="dialog-actions px-6 py-4 border-t border-border">
        <p data-testid="export-review-line" data-state={state} className="m-0 text-small text-text">
          <ReviewStateLine analysis={analysis} enabled={open} />
        </p>
        <p className="mt-1 mb-0 text-small text-text-muted">{note(state, second, valid)}</p>
        {refusal && <p role="alert" className="mt-2 mb-0 text-small text-danger">{refusal}</p>}
        <p aria-live="polite" data-testid="export-progress" className="mt-2 mb-0 min-h-5 text-small text-text">
          {progress ?? (run.data && !run.error ? run.data : '')}
        </p>
        {run.error && <p role="alert" className="mt-1 mb-0 text-small text-danger">{run.error.message}</p>}
        <div className="mt-3 flex items-center justify-end gap-4 flex-wrap">
          <Button variant="text" onClick={onClose} disabled={busy}>Close</Button>
          {actions.draft && <Button variant="ghost" onClick={() => run.mutate('draft')} disabled={busy || !!refusal}>{busy ? 'Exporting…' : 'Download draft'}</Button>}
          <Button variant="primary" onClick={() => run.mutate('issue')} disabled={busy || !!refusal || !actions.issue}
            aria-describedby={actions.issue || refusal ? undefined : `${uid}-issue`}>Issue</Button>
        </div>
        {actions.issue && !refusal && choice !== 'bundle' && (
          // N-19a ("Actions by role"): what a second use of "Issue" does
          <p data-testid="issue-note" className="mt-2 mb-0 text-caption text-text-muted text-right">
            Each use of Issue makes another numbered report. A report that is already issued is in the list below and downloads without a new number.
          </p>
        )}
        {!actions.issue && !refusal && (
          <p id={`${uid}-issue`} className="mt-2 mb-0 text-caption text-text-muted text-right">
            {!valid ? 'Issue is not available: a review record does not verify.'
              : state === 'disputed' ? 'Issue is not available: the peer reviewer does not concur.'
              : 'Issue needs the examiner sign-off, and a concurring peer review when the case requires one; until then a report is a draft.'}
          </p>
        )}
      </div>
      <div className="px-6 py-4 border-t border-border" data-testid="export-signing-key">
        <h3 className="m-0 text-small font-medium text-text">Signing key of this instance</h3>
        {signingKey.data ? (
          <>
            <p className="mt-2 mb-0 flex items-baseline gap-x-4 gap-y-1 flex-wrap text-small">
              <span className="font-mono text-mono text-text break-all" data-testid="export-key-id">{signingKey.data.key_id}</span>
              <Button variant="text" onClick={() => saveInstanceKey(signingKey.data)}>Save the key file</Button>
            </p>
            <p className="mt-1 mb-0 text-small text-text-muted">
              Whoever receives a bundle checks it against this key. Give them the key id or the key file yourself, apart from the
              bundle: the copy inside a bundle does not show who signed it, and without the key the offline check answers
              “signer not authenticated”.
            </p>
          </>
        ) : signingKey.error ? <p role="alert" className="mt-2 mb-0 text-small text-danger">The signing key could not be read: {signingKey.error.message}</p>
          : <p className="mt-2 mb-0 text-small text-text-muted">Loading…</p>}
      </div>
      <div className="px-6 pt-4 pb-6 border-t border-border" data-testid="export-reports">
        <h3 className="m-0 text-small font-medium text-text">Reports of this analysis</h3>
        {reports.isPending ? <p className="mt-2 mb-0 text-small text-text-muted">Loading…</p>
          : reports.error ? <p role="alert" className="mt-2 mb-0 text-small text-danger">{reports.error.message}</p>
            : items.length === 0 ? <p className="mt-2 mb-0 text-small text-text-muted">No report yet.</p> : (
              <table className="mt-2 w-full border-collapse text-small">
                <thead>
                  <tr className="border-b border-border text-left text-text-muted">
                    <th className="py-1.5 pr-4 font-normal">Report</th>
                    <th className="py-1.5 pr-4 font-normal">Profile</th>
                    <th className="py-1.5 pr-4 font-normal">Requested</th>
                    <th className="py-1.5 pr-4 font-normal">Signed</th>
                    <th className="py-1.5 font-normal"><span className="sr-only">Download</span></th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((r) => (
                    <tr key={r.id} data-testid="export-report" data-profile={r.profile} data-status={r.status} className="border-b border-border align-baseline">
                      <td className="py-2 pr-4 font-mono text-mono">{r.display_id}</td>
                      <td className="py-2 pr-4 text-text">{PROFILE_WORD[r.profile]}{r.redacted ? <span className="text-text-muted"> · redacted</span> : null}</td>
                      <td className="py-2 pr-4 text-text-muted num">{fmtDateLong(r.requested_at)}</td>
                      <td className="py-2 pr-4 font-mono text-mono text-text-muted">{r.signing_key_id ? r.signing_key_id.replace(/^ed25519:/, '').slice(0, 8) : r.status === 'done' ? 'unsigned' : JOB_WORD[r.status]}</td>
                      <td className="py-2 text-right">
                        {r.status === 'done' && <Button variant="text" onClick={() => download.mutate(r)} disabled={download.isPending} aria-label={`Download ${r.display_id}`}>Download</Button>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
        {download.error && <p role="alert" className="mt-2 mb-0 text-small text-danger">{download.error.message}</p>}
      </div>
    </dialog>
  )
}
