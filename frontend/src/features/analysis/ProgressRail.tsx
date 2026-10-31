import { useEffect, useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { isTerminal, useDetectorRuns, useStages, type AnalysisDetail, type DetectorRun, type Stage } from '../../lib/api/queries'
import type { EventsState } from '../../lib/api/sse'
import { fmtElapsed, fmtSeconds } from '../../lib/format'
import { cx } from '../../lib/cx'
import { notRunOnServer, ofTheServer, ON_THIS_SERVER } from './answer'

/** Stage ids of the image, document, video and audio DAGs (backend `app/engine/pipelines.py`). */
const STAGE_LABEL: Record<string, string> = {
  ingest: 'Intake & hashing', probe: 'Container probe', provenance: 'Provenance & metadata', decode: 'Decode',
  shots: 'Shot boundaries', tracks: 'Face tracks', faces: 'Faces', condition: 'Recording conditions', detect: 'Detectors',
  temporal: 'Temporal checks', second_pass: 'Second pass', localise: 'Localisation', fuse: 'Fusion & calibration',
  synthesise: 'Findings', render: 'Figures', publish: 'Publish result',
}

interface Glyph { glyph: string; word: string; className: string }

/** Glyph idiom of UX §5.2: ● done, ◌ running, ○ queued, ⊘ skipped, ✕ failed. */
const GLYPH: Record<string, Glyph> = {
  running: { glyph: '◌', word: 'running', className: 'text-accent animate-pulse-dot' },
  done: { glyph: '●', word: 'done', className: 'text-text' },
  skipped: { glyph: '⊘', word: 'skipped', className: 'text-text-muted' },
  cancelled: { glyph: '⊘', word: 'cancelled', className: 'text-text-muted' },
  failed: { glyph: '✕', word: 'failed', className: 'text-danger' },
  pending: { glyph: '○', word: 'queued', className: 'text-text-muted' },
}

/** Detector run statuses (ARCHITECTURE §2.6) in the same idiom. */
const RUN_GLYPH: Record<string, Glyph> = {
  ok: GLYPH.done,
  skipped: GLYPH.skipped,
  not_applicable: { ...GLYPH.skipped, word: 'not applicable' },
  cancelled: GLYPH.cancelled,
  failed: GLYPH.failed,
  timeout: { ...GLYPH.failed, word: 'timed out' },
}

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [active])
  return now
}

const ROW = 'grid grid-cols-[20px_minmax(0,240px)_112px_minmax(0,1fr)] items-baseline gap-x-3 py-1.5 border-b border-border text-small max-[639px]:grid-cols-[20px_minmax(0,1fr)_96px]'

function StageRow({ s, now }: { s: Stage; now: number }) {
  const g = GLYPH[s.status] ?? GLYPH.pending
  const end = s.finished_at ? Date.parse(s.finished_at) : now
  const secs = s.started_at ? (end - Date.parse(s.started_at)) / 1000 : null
  return (
    <li className={ROW}>
      <span aria-hidden className={cx('font-mono', g.className)}>{g.glyph}</span>
      <span className="text-text truncate">{STAGE_LABEL[s.stage] ?? s.stage}</span>
      <span className="text-text-muted">{g.word}</span>
      <span className="font-mono text-mono text-text-muted num max-[639px]:hidden">{secs != null && secs >= 0 ? fmtSeconds(secs, 1) : ''}</span>
      {s.error && <span className={cx('col-start-2 col-span-3 font-mono text-caption break-words', s.status === 'failed' ? 'text-danger' : 'text-text-muted')}>{s.error}</span>}
    </li>
  )
}

/** N-19k (D2-05): what a list of checks says of one whose name neither the result nor the registry holds; never its id. */
export const UNNAMED_CHECK = 'a check without a recorded name'
/** A check as the ledger names it (the API's `name` of the run), with its id and version for the tooltip. */
export const checkName = (r: Pick<DetectorRun, 'name'>): string => r.name ?? UNNAMED_CHECK

/** One detector that finished: glyph, the check's name as the ledger has it (N-19k; it read the detector id: "img.dda"),
 *  status, then its duration or the reason it did not run (P9). */
function CheckRow({ r }: { r: DetectorRun }) {
  const g = RUN_GLYPH[r.status] ?? { glyph: '○', word: r.status.replace(/_/g, ' '), className: 'text-text-muted' }
  const detail = r.status === 'ok' ? (r.duration_ms != null ? fmtSeconds(r.duration_ms / 1000, 1) : '') : (r.reason ?? '')
  return (
    <li className={ROW} data-status={r.status}>
      <span aria-hidden className={cx('font-mono', g.className)}>{g.glyph}</span>
      <span data-check className="text-small text-text truncate" title={`${r.detector_id} ${r.detector_version}`}>{checkName(r)}</span>
      <span className="text-text-muted">{g.word}</span>
      <span className={cx('text-text-muted break-words max-[639px]:col-start-2 max-[639px]:col-span-2', r.status === 'ok' && 'font-mono text-mono num')}>{detail}</span>
    </li>
  )
}

/** The checks line. While the analysis runs it counts the runs listed so far. For a finished analysis it is given the API's
 *  coverage, which the coverage panel prints too: that count also holds a check whose run could not be stored, which has no
 *  row in the run list (N-10a; code re-verification R2-03). */
/** What a count of checks says when there is none to count (a file no check applies to). */
export const NO_CHECK_APPLIED = 'no check was applied to this file'

export function coverageLine(runs: readonly DetectorRun[], counted?: AnalysisDetail['coverage'] | null): string {
  const by = (status: string) => counted ? (counted.by_status[status] ?? 0) : runs.filter((r) => r.status === status).length
  const total = counted ? counted.detectors_total : runs.length
  // N-19d (browser QA "Copy"): never "0 of 0 checks": while an analysis is queued or has just begun no check is listed yet
  if (total === 0) return counted ? NO_CHECK_APPLIED : 'no check has started yet'
  const failed = by('failed') + by('timeout')
  const notRun = by('skipped') + by('not_applicable')
  // L-7 (QA P2 "15 checks finished" beside "14 of 15"): the count is of checks that produced a result, as the coverage panel says it
  const ran = total - failed - notRun
  // N-14a: checks skipped for the server's memory budget or for model files that are not installed are counted as the server's
  const server = Math.min(notRun, counted ? notRunOnServer(counted).total : runs.filter((r) => ofTheServer(r.status, r.reason)).length)
  return [`${ran} of ${total} ${total === 1 ? 'check' : 'checks'} produced a result`, failed ? `${failed} failed` : null,
    server ? `${server} ${ON_THIS_SERVER}` : null, notRun - server ? `${notRun - server} not run` : null]
    .filter(Boolean).join(' · ')
}

function Checks({ runs }: { runs: readonly DetectorRun[] }) {
  return (
    <>
      <h3 className="mt-6 eyebrow">Checks</h3>
      {runs.length > 0
        ? <ol data-testid="coverage" className="mt-2 border-t border-border">{runs.map((r) => <CheckRow key={r.id} r={r} />)}</ol>
        : <p data-testid="coverage" className="mt-2 text-small text-text-muted">No check has finished yet.</p>}
    </>
  )
}

/** Analysis progress (UX §5.2; ARCHITECTURE §3.3): stages and the coverage list as detectors finish, with no provisional estimate.
 *  Once the analysis is terminal the rail collapses into the analysis record. */
export default function ProgressRail({ analysis, events }: { analysis: AnalysisDetail; events: EventsState }) {
  const terminal = isTerminal(analysis.status)
  const stages = useStages(analysis.id)
  const runs = useDetectorRuns(analysis.id)
  const now = useNow(!terminal)
  const [open, setOpen] = useState(false)
  const rows = stages.data ?? []
  const checks = runs.data ?? []
  const attrs = {
    'data-testid': 'progress-rail', 'data-state': terminal ? analysis.status : 'running',
    'data-transport': events.transport, 'data-last-seq': events.lastSeq,
  }
  const list = <ol className="border-t border-border">{rows.map((s) => <StageRow key={s.stage} s={s} now={now} />)}</ol>

  if (!terminal) {
    return (
      <section {...attrs} aria-live="polite" aria-labelledby="progress-heading" className="mt-12 relative border-t border-border-strong border-b border-b-border py-8">
        <div className="flex items-baseline justify-between gap-4">
          <h2 id="progress-heading" className="eyebrow">Analysis {analysis.status}</h2>
          <span className="font-mono text-h2 text-text num" aria-label="Elapsed time">{fmtElapsed(analysis.started_at ?? analysis.created_at, now)}</span>
        </div>
        <div className="mt-4">{rows.length > 0 ? list : <p className="text-small text-text-muted">Waiting for a worker to start the analysis.</p>}</div>
        <Checks runs={checks} />
        {/* N-19k (D2-05): one thing of a queue: with no check listed the block above says so, and no count says it again
            ("No check has finished yet." stood over "no check has started yet") */}
        {(checks.length > 0 || events.transport === 'polling') && (
          <p data-testid="checks-line" className="mt-4 text-small text-text num">
            {checks.length > 0 && coverageLine(checks)}
            {events.transport === 'polling' && <span className="text-text-muted">{checks.length > 0 ? ' · ' : ''}live updates unavailable, checking every 2 s</span>}
          </p>
        )}
        <p className="mt-2 text-small text-text-muted max-w-[64ch]">
          No estimate is shown until every check has finished and the results are fused. Models load on first use and can take about a minute; later runs take seconds.
        </p>
        <span aria-hidden className="absolute left-0 right-0 -bottom-px h-0.5 overflow-hidden"><span className="block h-full w-1/2 bg-accent animate-indeterminate" /></span>
      </section>
    )
  }

  if (rows.length === 0) return null // e.g. a v0.3-pipeline analysis, which recorded no stages
  const total = analysis.started_at && analysis.finished_at ? (Date.parse(analysis.finished_at) - Date.parse(analysis.started_at)) / 1000 : null
  return (
    <section {...attrs} aria-label="Analysis record" className="border-t border-border pt-3">
      <button type="button" aria-expanded={open} aria-controls="analysis-record" onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 min-h-8 text-small text-text-muted hover:text-text rounded-sm text-left">
        <span>Analysis record</span>
        <span className="num"> · {rows.length} stages · {coverageLine(checks, analysis.coverage)}{total != null ? ` · ${fmtSeconds(total, 1)}` : ''}</span>
        <ChevronRight size={14} strokeWidth={1.5} aria-hidden className={cx('shrink-0 transition-transform duration-base', open && 'rotate-90')} />
      </button>
      <div id="analysis-record" className="disclosure" data-open={open}><div><div className="pt-2">{list}<Checks runs={checks} /></div></div></div>
    </section>
  )
}
