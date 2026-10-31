import { useEffect, useId, useRef, useState } from 'react'
import { BAND_LABEL, bandStep, type Signal, type Verdict, type Analysis } from '../../lib/api'
import { prefersReducedMotion } from '../../lib/theme'
import { cx } from '../../lib/cx'
import { Button, Eyebrow, ConfidenceMeter } from '../../components/ui'
import DocumentShare from '../document/DocumentShare'

type Stats = NonNullable<NonNullable<Analysis['result']>['stats']>

const STEP_TEXT = ['text-verdict-1', 'text-verdict-2', 'text-verdict-3', 'text-verdict-4', 'text-verdict-5'] as const
const STEP_TINT = ['bg-verdict-1-tint', 'bg-verdict-2-tint', 'bg-verdict-3-tint', 'bg-verdict-4-tint', 'bg-verdict-5-tint'] as const
const STEP_BORDER = ['border-verdict-1', 'border-verdict-2', 'border-verdict-3', 'border-verdict-4', 'border-verdict-5'] as const

export const CLOSING_CAVEAT ='Probabilities are statistical evidence from open-source detectors, not proof. Review findings before acting on them.'

/** Reasoning items that merely re-serialise a ledger row (`… (strong, observed p(ai)=… )`); the ledger's "Why this matters" already carries them. */
const PER_SIGNAL_RESTATEMENT = /\((strong|moderate|weak), observed/
/** Backend template that restates the numeral, band and dl one more time: the v0.3 sentence of stored results, and the
 *  fallback sentence since N-7 (`schemas/legacy.py::fallback_reason`). A cap or a documentary reason is not templated and shows. */
export const TEMPLATED_SUMMARY = /is assessed as .* \((estimated probability|score \d+ \/ 100, not calibrated)/
const VISIBLE_REASONING = 3

/** Count 0 -> target over `ms` with the standard ease; jumps when reduced motion is requested. */
function useCountUp(target: number, ms = 320) {
  const [v, setV] = useState(() => (prefersReducedMotion() ? target : 0))
  const raf = useRef<number | null>(null)
  useEffect(() => {
    if (prefersReducedMotion()) { setV(target); return }
    const start = performance.now()
    const ease = (t: number) => 1 - Math.pow(1 - t, 3)
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms)
      setV(Math.round(target * ease(t)))
      if (t < 1) raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => { if (raf.current) cancelAnimationFrame(raf.current) }
  }, [target, ms])
  return v
}

const confLevel = (c: string): 0 | 1 | 2 | 3 => (/high/i.test(c) ? 3 : /med/i.test(c) ? 2 : /low/i.test(c) ? 1 : 0)

export default function VerdictModule({ v, signals, stats, kind }: { v: Verdict; signals: Signal[]; stats?: Stats | null; kind: Analysis['kind'] }) {
  const step = bandStep(v.band)
  const pctTarget = v.probability_ai == null ? null : Math.round(v.probability_ai * 100)  // null: no estimate published (D-49)
  const shown = useCountUp(pctTarget ?? 0)
  const [mounted, setMounted] = useState(false)
  const [allReasoning, setAllReasoning] = useState(false)
  useEffect(() => { const t = requestAnimationFrame(() => setMounted(true)); return () => cancelAnimationFrame(t) }, [])
  const headingId = useId()
  const label = pctTarget == null ? v.label : BAND_LABEL[v.band] ?? v.label
  const decisive = signals.filter((s) => s.decisive).map((s) => s.name)
  const caveats = v.caveats.some((c) => /not proof/i.test(c)) ? v.caveats : [...v.caveats, CLOSING_CAVEAT]
  const reasoning = v.reasoning.filter((r) => !PER_SIGNAL_RESTATEMENT.test(r))
  const visibleReasoning = allReasoning ? reasoning : reasoning.slice(0, VISIBLE_REASONING)
  const summary = v.summary && !TEMPLATED_SUMMARY.test(v.summary) ? v.summary : null

  return (
    <section aria-labelledby={headingId} data-tour="verdict" className="border-t border-border-strong border-b border-b-border py-8">
      <Eyebrow as="h2" id={headingId}>Verdict</Eyebrow>
      <p className="sr-only">{pctTarget == null ? 'No estimate' : `Probability of AI generation ${pctTarget} percent`}, {label.toLowerCase()}, {v.confidence} confidence.</p>

      <div className="mt-4 grid gap-10 min-[960px]:grid-cols-[7fr_5fr]">
        <div className="min-w-0">
          <div className="flex items-end gap-6 flex-wrap">
            {pctTarget == null ? (
              <p className="m-0 text-body text-text-muted" aria-hidden>No estimate</p>
            ) : (
              <div className="flex items-start gap-2" aria-hidden>
                <span className="display text-numeral font-light num text-text">{shown}</span>
                <span className="flex flex-col pt-2">
                  <span className="display text-h1 font-normal text-text leading-none">%</span>
                  <Eyebrow as="span" className="mt-2 normal-case">p(AI)</Eyebrow>
                </span>
              </div>
            )}
            <div className="min-w-0 pb-1">
              <p className={cx('display text-h1 font-normal', STEP_TEXT[step - 1])}>{label}</p>
              {pctTarget != null && <p className="text-small text-text-muted mt-0.5">of 100, where 100 = certainly AI-generated or manipulated</p>}
            </div>
          </div>

          <div className="mt-6 relative" role="img" aria-label={pctTarget == null ? 'No estimate' : `Scale position ${pctTarget} of 100`} data-tour="scale">
            <div className="flex gap-0.5 h-1.5">
              {[1, 2, 3, 4, 5].map((n) => <span key={n} className={cx('flex-1 border', STEP_TINT[n - 1], STEP_BORDER[n - 1])} style={{ borderRadius: 1 }} />)}
            </div>
            {pctTarget != null && <span aria-hidden className="absolute -top-1 w-0.5 h-3.5 bg-text transition-[left] duration-slow ease-standard" style={{ left: `calc(${mounted || prefersReducedMotion() ? pctTarget : 0}% - 1px)` }} />}
            <div className="mt-2 flex justify-between text-caption text-text-muted">
              <span>no indications</span><span>inconclusive</span><span>AI / manipulated</span>
            </div>
          </div>

          {/* Summary and document stats share the scale's right edge (left grid column), not an independent 68ch. */}
          {summary && <p className="display text-reader text-text mt-6">{summary}</p>}

          {kind === 'document' && stats && <DocumentShare stats={stats} estimate={v.probability_ai} />}
        </div>

        <dl className="grid grid-cols-[112px_1fr] gap-x-4 gap-y-3 m-0 self-start">
          <dt className="eyebrow pt-1">Confidence</dt>
          <dd className="m-0 text-body text-text flex items-center gap-2.5"><span className="capitalize">{v.confidence}</span><ConfidenceMeter level={confLevel(v.confidence)} /></dd>
          <dt className="eyebrow pt-1">Signals</dt>
          <dd className="m-0 text-body text-text num">{v.signals_for_ai} → AI · {v.signals_for_authentic} → authentic<span className="text-text-muted"> · of {v.signals_total} informative</span></dd>
          {decisive.length > 0 && (<>
            <dt className="eyebrow pt-1">Decisive</dt>
            <dd className="m-0 text-body text-text">{decisive.join('; ')}</dd>
          </>)}
          {kind === 'document' && stats && (<>
            <dt className="eyebrow pt-1">Corpus</dt>
            <dd className="m-0 text-body text-text num">{stats.words} words · {stats.sentences} sentences</dd>
            {stats.binoculars_document != null && (<>
              <dt className="eyebrow pt-1">Binoculars</dt>
              <dd className="m-0 text-body text-text"><span className="font-mono text-mono num">{stats.binoculars_document.toFixed(3)}</span><span className="text-text-muted"> · whole document</span></dd>
            </>)}
          </>)}
          <dt className="eyebrow pt-1">Basis</dt>
          <dd className="m-0 text-body text-text">Weighted fusion of detector probabilities; see Evidence</dd>
        </dl>
      </div>

      <div className="mt-8 grid gap-10 min-[960px]:grid-cols-[7fr_5fr]">
        {reasoning.length > 0 && (
          <div className="min-w-0" data-tour="reasoning">
            <h3 className="text-h3 font-semibold text-text">Reasoning</h3>
            <ol className="mt-3 m-0 p-0 list-none space-y-2">
              {visibleReasoning.map((r, i) => (
                <li key={i} className="grid grid-cols-[28px_1fr]">
                  <span className="font-mono text-mono text-text-muted num">{i + 1}</span>
                  <span className="text-body text-text">{r}</span>
                </li>
              ))}
            </ol>
            {reasoning.length > VISIBLE_REASONING && (
              <Button variant="text" className="mt-2 ml-7" aria-expanded={allReasoning} onClick={() => setAllReasoning((s) => !s)}>
                {allReasoning ? 'Show fewer' : `Show all ${reasoning.length}`}
              </Button>
            )}
          </div>
        )}
        <div className="min-w-0">
          <h3 className="text-h3 font-semibold text-text">Caveats</h3>
          <ul className="mt-3 m-0 p-0 list-none space-y-2">
            {caveats.map((c, i) => (
              <li key={i} className="grid grid-cols-[16px_1fr] text-small text-text-muted">
                <span aria-hidden className="text-text-faint">·</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}
