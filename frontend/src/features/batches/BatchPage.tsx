import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Button, ErrorState, Eyebrow, Skeleton, StatusDot, type Status } from '../../components/ui'
import { downloadSummary, useBatch, type BatchItem, type BatchItemStatus } from '../../lib/api/batches'
import { fmtBytes, fmtDate } from '../../lib/format'
import { CAPTURE_ROW, DECLARATION_ROW, captureRow, declarationRow } from '../analysis/record'

const ITEM_WORD: Record<BatchItemStatus, string> = {
  staged: 'Waiting', rejected: 'Rejected', queued: 'Queued', running: 'Running', done: 'Analysed', failed: 'Failed', cancelled: 'Cancelled', deleted: 'Deleted',
}
const ITEM_DOT: Record<BatchItemStatus, Status> = {
  staged: 'queued', rejected: 'failed', queued: 'queued', running: 'running', done: 'done', failed: 'failed', cancelled: 'offline', deleted: 'offline',
}
const STATE_WORD = { expanding: 'Taking in files', running: 'Running', done: 'Complete', failed: 'Failed' } as const
const SOURCE_WORD = { upload: 'uploaded files', archive: 'a zip archive', local_path: 'a server directory' } as const
/** UX §7.1 list / chip wording, keyed by decision code (D-16). */
const DECISION_SHORT: Record<string, string> = {
  ai_supported_strong: 'Strong indications', ai_supported: 'Indications', inconclusive: 'Inconclusive',
  no_ai_indicators: 'No significant indications', no_ai_indicators_strong: 'No indications found', not_assessable: 'Not assessable',
}
const link = 'text-accent underline underline-offset-[3px] decoration-1 hover:text-accent-hover hover:decoration-2 text-small font-medium'

/** Batch progress (ARCHITECTURE §7.2 `/batches/:id`; PRD API-01): every item with its intake outcome or analysis status, polled
 *  while items are taken in or running, the rejected files with their reasons, and the CSV / JSON summaries. */
export default function BatchPage() {
  const { batchId = '' } = useParams()
  const q = useBatch(batchId)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const b = q.data
  const n = (k: BatchItemStatus) => b?.counts[k] ?? 0
  const waiting = n('staged') + n('queued') + n('running')
  const settled = b ? b.item_count - waiting : 0
  const download = (format: 'csv' | 'json') => {
    setDownloadError(null)
    downloadSummary(batchId, format).catch((e: Error) => setDownloadError(e.message))
  }

  return (
    <div className="max-w-[1080px]">
      <Eyebrow>Batch</Eyebrow>
      <h1 className="display text-h1 font-normal text-text mt-1">
        {b ? `${b.item_count} ${b.item_count === 1 ? 'file' : 'files'}` : <Skeleton w={160} />}
      </h1>
      {q.error && <ErrorState className="mt-8" onRetry={() => void q.refetch()}>{q.error.message}</ErrorState>}
      {b && (
        <>
          <p className="text-small text-text-muted mt-1 num">
            From {SOURCE_WORD[b.source]} · submitted {fmtDate(b.created_at)}{b.profile ? ` · ${b.profile} profile` : ''}
          </p>

          <div className="mt-8 flex flex-wrap items-baseline gap-x-8 gap-y-2">
            <span data-testid="batch-state" className="text-body font-medium text-text" aria-live="polite">{STATE_WORD[b.status]}</span>
            <span data-testid="batch-counts" className="text-small text-text-muted num">
              {n('done')} analysed · {n('rejected')} rejected · {n('failed')} failed{waiting ? ` · ${waiting} waiting or running` : ''}
            </span>
            <div className="ml-auto flex flex-wrap items-center gap-x-5 gap-y-1">
              <Button variant="text" onClick={() => download('csv')}>Summary CSV</Button>
              <Button variant="text" onClick={() => download('json')}>Summary JSON</Button>
              <Link to={`/cases/${b.case_id}`} className={link}>Open the case</Link>
            </div>
          </div>
          <div aria-hidden className="mt-3 h-0.5 bg-border overflow-hidden">
            <span className="block h-full bg-text transition-[width] duration-fast" style={{ width: `${b.item_count ? Math.round((100 * settled) / b.item_count) : 0}%` }} />
          </div>
          {b.error && <p className="mt-2 mb-0 text-caption text-danger" role="alert">{b.error}</p>}
          {downloadError && <p className="mt-2 mb-0 text-caption text-danger" role="alert">{downloadError}</p>}

          <div className="mt-6 overflow-x-auto">
            {/* N-19b (design review DS-13): below 640 px one stacked block for each file */}
            <table role="table" className="stack-sm w-full border-collapse" data-testid="batch-items">
              <thead role="rowgroup">
                <tr role="row" className="border-b border-border-strong">
                  <th role="columnheader" scope="col" className="eyebrow text-right font-medium py-2 pr-4 w-12">#</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4">File</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-24 max-[959px]:hidden">Kind</th>
                  <th role="columnheader" scope="col" className="eyebrow text-right font-medium py-2 pr-4 w-24 max-[959px]:hidden">Size</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-32">Status</th>
                  <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2">Outcome</th>
                </tr>
              </thead>
              <tbody role="rowgroup">
                {b.items.map((item) => <ItemRow key={item.seq} item={item} />)}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

function ItemRow({ item }: { item: BatchItem }) {
  const outcome = item.analysis_id
    ? (
      <Link to={`/analyses/${item.analysis_id}`} className="text-text hover:underline underline-offset-[3px]">
        {/* N-19m (D3-01): a clean code that a provenance record decides is a capture's row, not "No indications found";
            N-19p (D6-01): a trusted declaration on the fallback is the declaration's, not "Indications" */}
        {item.status === 'done' ? (captureRow(item) ? CAPTURE_ROW : declarationRow(item) ? DECLARATION_ROW : DECISION_SHORT[item.decision_code ?? ''] ?? 'Open the analysis') : 'Open the analysis'}
      </Link>
    )
    : null
  return (
    <tr role="row" data-testid="batch-item" className="border-b border-border h-11 max-[639px]:h-auto">
      <td role="cell" data-th="file" className="pr-4 text-right max-[639px]:text-left text-small text-text-muted num">{item.seq + 1}</td>
      <td role="cell" className="py-2 pr-4 min-w-0 max-w-[420px]">
        <span className="block text-small text-text truncate max-[639px]:whitespace-normal max-[639px]:[overflow-wrap:anywhere]" title={item.source_name}>{item.source_name}</span>
      </td>
      <td role="cell" className="pr-4 text-small text-text-muted max-[959px]:hidden">{item.kind ?? '–'}</td>
      <td role="cell" className="pr-4 text-right font-mono text-mono text-text-muted num max-[959px]:hidden">{item.size_bytes == null ? '–' : fmtBytes(item.size_bytes)}</td>
      <td role="cell" className="pr-4"><StatusDot status={ITEM_DOT[item.status]} word={ITEM_WORD[item.status]} /></td>
      <td role="cell" className="py-2 text-small min-w-0">
        {outcome}
        {item.error && <span className="block text-caption text-text-muted">{item.error}</span>}
      </td>
    </tr>
  )
}
