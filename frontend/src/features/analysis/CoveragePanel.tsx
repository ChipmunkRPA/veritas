import type { AnalysisDetail, DetectorRun, Finding } from '../../lib/api/queries'
import { deviceWords } from '../../lib/recordWords'
import { statusWord } from '../signals/ledger'
import { notRunOnServer, ofTheServer, ON_THIS_SERVER } from './answer'
import { checkName } from './ProgressRail'

/** The coverage finding's statement without its first sentence, the count of checks that produced a result (N-19k: "6 of 6
 *  checks produced a result" stood twice in three lines of the block). */
export const withoutCount = (statement: string): string => statement.replace(/^\d+ of \d+ checks? produced a result\.\s*/, '')

const GROUP: Record<string, 'ran' | 'not run' | 'failed'> = { ok: 'ran', skipped: 'not run', not_applicable: 'not run', cancelled: 'not run', failed: 'failed', timeout: 'failed' }

/** Coverage (UX §8.5 partial analysis, P9; ARCHITECTURE §4.5 `coverage` finding): which checks produced a result, which did not and why,
 *  the profile that ran, the devices, and whether the input sits inside a validated slice. */
export default function CoveragePanel({ analysis, runs, findings }: { analysis: AnalysisDetail; runs: readonly DetectorRun[]; findings: readonly Finding[] }) {
  const total = analysis.coverage?.detectors_total ?? runs.length
  const by = analysis.coverage?.by_status ?? {}
  const count = (g: string) => Object.entries(by).filter(([s]) => (GROUP[s] ?? 'not run') === g).reduce((n, [, k]) => n + k, 0)
  const profile = analysis.effective_profile && analysis.effective_profile !== analysis.profile
    ? `${analysis.effective_profile} (requested ${analysis.profile ?? 'default'})` : (analysis.effective_profile ?? analysis.profile ?? 'default')
  const devices = new Map<string, number>()
  runs.filter((r) => r.status === 'ok').forEach((r) => { const k = deviceWords(r.device, r.dtype) ?? 'device not recorded'; devices.set(k, (devices.get(k) ?? 0) + 1) })
  const off = runs.filter((r) => r.status !== 'ok')
  // N-14a: checks skipped for the server's memory budget or for model files that are not installed are counted as the server's
  const server = notRunOnServer(analysis.coverage).total
  const cp = analysis.condition_profile
  const coverageFinding = findings.find((f) => f.family === 'coverage')
  return (
    <div data-testid="coverage-panel">
      <h3 className="m-0 eyebrow">Coverage</h3>
      <p className="mt-2 m-0 text-body text-text num">{total === 0 ? 'No check was applied to this file' : `${count('ran')} of ${total} ${total === 1 ? 'check' : 'checks'} produced a result`} · profile {profile}</p>
      <p className="mt-1 m-0 flex flex-wrap gap-x-5 text-small text-text-muted num">
        <span><span aria-hidden className="font-mono text-text">●</span> {count('ran')} ran</span>
        <span><span aria-hidden className="font-mono">⊘</span> {Math.max(0, count('not run') - server)} not run</span>
        {server > 0 && <span data-testid="coverage-server"><span aria-hidden className="font-mono">⊘</span> {server} {ON_THIS_SERVER}</span>}
        <span><span aria-hidden className="font-mono text-danger">✕</span> {count('failed')} failed or timed out</span>
        {devices.size > 0 && <span>{[...devices.entries()].map(([k, n]) => `${k} (${n})`).join(', ')}</span>}
      </p>
      {off.length > 0 && (
        <ul className="mt-2 m-0 p-0 list-none text-small">
          {off.map((r) => <li key={r.id} data-status={r.status}><span data-check className="text-text" title={`${r.detector_id} ${r.detector_version}`}>{checkName(r)}</span> <span className="text-text-muted">· {ofTheServer(r.status, r.reason) ? ON_THIS_SERVER : statusWord(r.status)}{r.reason ? ` · ${r.reason}` : ''}</span></li>)}
        </ul>
      )}
      {cp && (
        <p data-testid="envelope" data-within={cp.within_validated_envelope} className="mt-2 m-0 text-small text-text max-w-[80ch]">
          <a href="#section-methods" className="text-accent underline underline-offset-[3px]">See validation coverage in Methods &amp; validation.</a>
        </p>
      )}
      {/* N-19k: the count stands in the first line of this block; the finding's own first sentence said it again */}
      {coverageFinding && withoutCount(coverageFinding.statement) && <p data-testid="coverage-finding" className="mt-2 m-0 text-small text-text-muted max-w-[80ch]">{withoutCount(coverageFinding.statement)}</p>}
    </div>
  )
}
