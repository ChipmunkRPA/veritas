import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ApiError, apiClient, authHeaders, unwrap } from '../../lib/api/client'
import { qk, useCustody, useMe, type AuditEvent } from '../../lib/api/queries'
import { mayExportCustody, mayReadCustody } from '../../lib/api/roles'
import { cx } from '../../lib/cx'
import { Button, EmptyState, ErrorState, SkeletonRows } from '../../components/ui'
import { actorName, eventDetail, eventHash, eventWord, exhibitOf, fmtUtc, isAlarm, type ExhibitLabels } from './custody'

/** Custody log (UX §5.12; PRD CC-01, CC-02): the case's audit events in seq order, never editable, with the verification of the
 *  hash chain and of every original (`GET /cases/{id}/custody/verify`). Verifying re-hashes the originals, so it runs on request.
 *  An original that verifies leaves no event; one that does not joins the log as `hash_mismatch`, once (N-10). The log itself is
 *  open to the owner of the case, to reviewers and to administrators (`GET /cases/{id}/custody`; D-67); its export is the
 *  export of the audit log (reviewers and administrators). */
export default function CustodyLog({ caseId, reference, myRole, labels, names }: {
  caseId: string
  reference: string
  /** The reader's role on the case: its owner reads the log. */
  myRole?: string | null
  labels: ExhibitLabels
  names: ReadonlyMap<string, string>
}) {
  const qc = useQueryClient()
  // The log is open to the owner, to reviewers and to administrators; for anyone else the page says so without asking the
  // server (L-7).
  const me = useMe()
  const readable = mayReadCustody(me.data, myRole)
  const log = useCustody(caseId, readable)
  const forbidden = (me.isSuccess && !readable) || (log.error instanceof ApiError && log.error.status === 403)
  const verify = useMutation({
    mutationFn: () => unwrap(apiClient.GET('/api/v2/cases/{case_id}/custody/verify', { params: { path: { case_id: caseId } } })),
    onSettled: () => { void qc.invalidateQueries({ queryKey: qk.custody(caseId) }) },
  })
  const [exporting, setExporting] = useState<string | null>(null)

  const exportCsv = async () => {
    setExporting('Preparing…')
    try {
      const r = await fetch(`/api/v2/audit/export?format=csv&case_id=${encodeURIComponent(caseId)}`, { headers: authHeaders() })
      if (!r.ok) throw new Error(r.statusText || `HTTP ${r.status}`)
      const url = URL.createObjectURL(await r.blob())
      const a = document.createElement('a')
      a.href = url
      a.download = `custody-${reference}.csv`
      a.click()
      URL.revokeObjectURL(url)
      setExporting(null)
    } catch (e) {
      setExporting(`Export failed: ${(e as Error).message}`)
    }
  }

  const v = verify.data
  const brokenOriginals = v ? v.originals.filter((o) => o.status !== 'verified') : []
  // a report of an analysis that was deleted or purged is listed as removed under that event: it is not a failure (N-9a)
  const removedReports = v ? v.reports.filter((o) => o.status === 'removed') : []
  const keptReports = v ? v.reports.filter((o) => o.status !== 'removed') : []
  const brokenReports = keptReports.filter((o) => o.status !== 'verified')
  const events: AuditEvent[] = log.data ?? []

  return (
    <section className="mt-12" aria-labelledby="custody-heading" data-testid="custody">
      <div className="flex items-center justify-between gap-4 flex-wrap border-b border-border-strong pb-2">
        <h2 id="custody-heading" className="eyebrow m-0">Custody log</h2>
        <div className="flex items-center gap-4">
          {log.isSuccess && mayExportCustody(me.data) && <Button variant="text" onClick={() => void exportCsv()}>Export CSV</Button>}
          <Button variant="ghost" className="h-8 px-3" onClick={() => verify.mutate()} disabled={verify.isPending}>
            {verify.isPending ? 'Verifying…' : 'Verify chain and originals'}
          </Button>
        </div>
      </div>
      <p className="mt-2 mb-0 text-caption text-text-muted max-w-[72ch]">Append-only. Times are UTC. Verifying re-hashes every original of the case and records the result here.</p>
      {exporting && <p className="mt-2 mb-0 text-small text-text-muted" role="status">{exporting}</p>}

      <div aria-live="polite" data-testid="custody-verify" data-state={verify.isPending ? 'pending' : v ? (v.valid ? 'valid' : 'invalid') : verify.error ? 'error' : 'idle'}>
        {v && (
          <p className={cx('mt-4 mb-0 flex flex-wrap items-baseline gap-x-2 text-small', v.valid ? 'text-text' : 'text-danger')}>
            <span aria-hidden className={cx('inline-block w-3 text-center', v.valid ? 'text-ok' : 'text-danger')}>{v.valid ? '●' : '✕'}</span>
            <span className="font-medium">{v.valid ? 'Verified' : 'Verification failed'}</span>
            <span className="num">
              {v.chain.valid ? `chain of ${v.chain.checked} events intact` : `chain broken at event ${v.chain.first_broken_seq ?? '?'}${v.chain.reason ? ` (${v.chain.reason})` : ''}`}
              {` · ${v.originals.length - brokenOriginals.length} of ${v.originals.length} ${v.originals.length === 1 ? 'original' : 'originals'} re-hashed unchanged`}
              {keptReports.length > 0 && ` · ${keptReports.length - brokenReports.length} of ${keptReports.length} reports unchanged`}
              {removedReports.length > 0 && ` · ${removedReports.length} ${removedReports.length === 1 ? 'report' : 'reports'} removed with ${removedReports.length === 1 ? 'its' : 'their'} analysis`}
            </span>
            {brokenOriginals.map((o) => <span key={o.media_id} className="num">· Ex. {o.label} {o.status}</span>)}
          </p>
        )}
        {verify.error && <ErrorState className="mt-4" onRetry={() => verify.mutate()}>{verify.error.message}</ErrorState>}
      </div>

      {readable && log.isPending && <SkeletonRows className="mt-4" widths={['88%', '80%', '84%', '76%']} />}
      {forbidden && <EmptyState className="mt-4">The custody log is open to the owner of the case, to reviewers and to administrators. Verification is open to every member of the case.</EmptyState>}
      {log.error && !forbidden && <ErrorState className="mt-4" onRetry={() => void log.refetch()}>{log.error.message}</ErrorState>}
      {log.isSuccess && events.length === 0 && <EmptyState className="mt-4">No events recorded for this case.</EmptyState>}
      {events.length > 0 && (
        <div className="mt-3 overflow-x-auto">
          {/* below 640 px each event is one stacked block: event, time, who, details, hash (L-7) */}
          {/* N-19b (browser QA "Layout"): a fixed layout. The details take what the other columns leave, so the Hash column is
              never cut at the right edge; a long event or detail wraps inside its cell */}
          <table role="table" className="stack-sm w-full table-fixed border-collapse" data-testid="custody-events">
            <thead role="rowgroup">
              <tr role="row" className="border-b border-border">
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-[212px]">Time (UTC)</th>
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-[208px]">Event</th>
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-40 max-[959px]:hidden">By</th>
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4">Details</th>
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 w-[112px]">Hash</th>
              </tr>
            </thead>
            <tbody role="rowgroup">
              {events.map((e) => {
                const exhibit = exhibitOf(e, labels)
                const hash = eventHash(e)
                const actor = actorName(e.actor_id, names, e.actor_name)
                const detail = eventDetail(e, names)
                return (
                  <tr role="row" key={e.seq} data-testid="custody-row" data-action={e.action} className="border-b border-border align-baseline">
                    <td role="cell" className="py-2 pr-4 font-mono text-mono text-text-muted num whitespace-nowrap">{fmtUtc(e.ts)}</td>
                    <td role="cell" className={cx('py-2 pr-4 text-small', isAlarm(e) ? 'text-danger font-medium' : 'text-text')}>
                      <span>{exhibit && <span className="font-mono text-mono text-text-muted mr-1.5 whitespace-nowrap">{exhibit}</span>}{eventWord(e)}</span>
                      <span className="block text-caption text-text-muted font-normal min-[960px]:hidden">by {actor}</span>
                    </td>
                    <td role="cell" className="py-2 pr-4 text-small text-text-muted break-words max-[959px]:hidden">{actor}</td>
                    <td role="cell" data-testid="custody-details" className={cx('py-2 pr-4 text-small text-text-muted [overflow-wrap:anywhere]', !detail && 'max-[639px]:hidden')}>{detail || '—'}</td>
                    <td role="cell" data-th={hash ? 'hash' : undefined} data-testid="custody-hash" className="py-2 font-mono text-mono text-text-muted num whitespace-nowrap">{hash ?? ''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
