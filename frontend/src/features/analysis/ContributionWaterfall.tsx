import { Fragment, useLayoutEffect, useState } from 'react'
import type { AnalysisDetail, Finding, SignalV2 } from '../../lib/api/queries'
import { cx } from '../../lib/cx'
import { Button, HatchPattern, useSvgId } from '../../components/ui'
import { estimateOf, noEstimateFormed, numeral, type Decision } from './answer'
import { aloneWhy, recordOf, recordSetNumber, statisticalWithheld } from './record'
import { littleOf } from './vocabulary'
import { RANGE_WORD, resultNumber, type ResultNumber } from './resultNumber'
import {
  agreementDots, axisPosition, cappedCombined, fallbackText, integrity, logit, LOG_ODDS_TICKS, narrative, NOT_COUNTED, PROB_TICKS, reachedTitle, RECORD_ROW_NO_AMOUNT, sigmoid,
  signed,
  waterfall, type Bar, type Integrity,
} from './contributions'

const GRID = 'grid grid-cols-[280px_minmax(0,1fr)_64px] gap-x-4 max-[959px]:grid-cols-[180px_minmax(0,1fr)_56px] max-[639px]:grid-cols-[112px_minmax(0,1fr)_52px] max-[639px]:gap-x-3'

/** The width of the element the returned ref is put on. N-19b (browser QA "Layout"): the element is held in state, so the
 *  measure starts when it is drawn. The strip draws nothing until the checks are loaded, and a ref read once on mount left
 *  it at the fallback, an 800 px drawing at every window width. */
function useWidth<T extends HTMLElement>(fallback: number) {
  const [el, ref] = useState<T | null>(null)
  const [width, setWidth] = useState(fallback)
  useLayoutEffect(() => {
    if (!el) return
    setWidth(el.clientWidth || fallback)
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth || fallback))
    ro.observe(el)
    return () => ro.disconnect()
  }, [el, fallback])
  return [ref, width] as const
}

/** N-19h (E-20, 1): the last row where a record set the number (`forensic.RECORD_ROW`): its value, not an estimate. */
export const RECORD_ROW = 'Value set by the record'

/** Agreement strip (UX §8.4): each scored check at its standalone p; hidden with two checks or fewer. N-17 (the number rule of
 *  D-68, left open by N-7): the strip shows the number the summary shows: the estimate and its marker for a decisive calibrated
 *  result, the range as text without a marker for an inconclusive or one-check result, the fallback's score as text, and no
 *  number for a capped result. */
function AgreementStrip({ signals, number, little }: { signals: readonly SignalV2[]; number: ResultNumber; little: number }) {
  const estimate = number.kind === 'calibrated' ? number.estimate : null
  const said = number.kind === 'text' || number.kind === 'uncalibrated' ? number.text
    : number.kind === 'record' ? `${number.text.charAt(0).toUpperCase()}${number.text.slice(1)}` : null
  const [ref, width] = useWidth<HTMLDivElement>(800)
  const dots = agreementDots(signals, width, little)
  if (dots.length <= 2) return null
  const x = (p: number) => Math.max(1, Math.min(width - 1, p * width))
  return (
    <div className="mt-6" data-testid="agreement-strip">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h3 className="m-0 eyebrow">Detector estimates</h3>
        <p className="m-0 flex flex-wrap items-center gap-x-4 gap-y-1 text-caption text-text-muted">
          <span className="inline-flex items-center gap-1.5"><span aria-hidden className="block size-(--viz-dot) rounded-full bg-dir-ai" />toward</span>
          <span className="inline-flex items-center gap-1.5"><span aria-hidden className="block size-(--viz-dot) rounded-full border-[1.5px] border-dir-authentic" />against</span>
          <span className="inline-flex items-center gap-1.5"><span aria-hidden className="block size-(--viz-dot-small) rounded-full bg-dir-neutral" />little effect</span>
          {estimate != null && <span className="inline-flex items-center gap-1.5"><span aria-hidden className="block w-0.5 h-3.5 bg-text" />estimate {numeral(estimate)}</span>}
          {said != null && <span data-testid="strip-number">{said}</span>}
        </p>
      </div>
      {/* the drawing is out of the flow: it takes the section's width and never gives it one */}
      <div ref={ref} className="relative mt-2 h-16 w-full">
        <svg width={width} height={64} role="img" aria-label={`Standalone estimates of ${dots.length} checks, from 0 to 100${estimate != null ? `; fused estimate ${numeral(estimate)}` : said != null ? `; ${said}` : ''}.`} className="absolute left-0 top-0 block overflow-visible">
          <line x1={0} x2={width} y1={46} y2={46} style={{ stroke: 'var(--color-graphic)' }} strokeWidth={1} />
          {[0, 50, 100].map((t) => (
            <text key={t} x={x(t / 100)} y={60} textAnchor={t === 0 ? 'start' : t === 100 ? 'end' : 'middle'} className="font-mono" style={{ fontSize: 11, fill: 'var(--color-text-muted)' }}>{t}</text>
          ))}
          {estimate != null && <rect x={x(estimate) - 1} y={32} width={2} height={14} style={{ fill: 'var(--color-text)' }} />}
          {dots.map((d) => (
            <g key={d.id} data-dot={d.direction}>
              {d.direction === 'ai' && <circle cx={x(d.p)} cy={39} r={3.5} style={{ fill: 'var(--color-dir-ai)' }} />}
              {d.direction === 'authentic' && <circle cx={x(d.p)} cy={39} r={2.75} strokeWidth={1.5} style={{ fill: 'none', stroke: 'var(--color-dir-authentic)' }} />}
              {d.direction === 'neutral' && <circle cx={x(d.p)} cy={39} r={1.5} style={{ fill: 'var(--color-dir-neutral)' }} />}
              <text x={Math.max(14, Math.min(width - 14, x(d.p)))} y={27 - d.row * 10} textAnchor="middle" className="font-mono" style={{ fontSize: 11, fill: 'var(--color-text-muted)' }}>{d.displayId}</text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  )
}

const pct = (at: number) => `${(at * 100).toFixed(3)}%`

/** The last row by the number the result shows: the estimate with its marker, the range without one, or no number (capped). */
function lastRow(b: Bar, number: ResultNumber | undefined, flat?: FlatLast, declared?: number | null): { label: string; value: string; marker: boolean; tip: string } | null {
  if (b.kind !== 'estimate' || !number) return null
  // N-19l (R6-02): outside what was validated under a record that set the number, the record's value. N-19n (D3-09): a
  // flat breakdown has no other last row (`flatRows`; it read "Not shown: outside what was validated")
  if (flat?.record != null) return { label: `${RECORD_ROW}: ${numeral(flat.record)} out of 100`, value: '—', marker: false, tip: RECORD_ROW }
  // N-19m (D3-05): the record's value has the unit of the line, "85 out of 100" (it read "85 %" under "Set by the record: 85 out of 100")
  if (number.kind === 'record') return { label: RECORD_ROW, value: `${numeral(number.value)} out of 100`, marker: false, tip: number.text }
  // N-19l (R6-04, 2): under a declaration that set the number the last row is the record's value too (it read "Estimate
  // 97"); where the estimate stands above the declaration's value it is the statistical estimate, as before
  if (number.kind === 'none' && declared != null) return { label: RECORD_ROW, value: `${numeral(declared)} out of 100`, marker: false, tip: RECORD_ROW }
  if (number.kind === 'text') {
    // N-19q (finding 11): a row that holds the plausible range is named so (it read "Estimate 24–85"; `forensic.waterfall`)
    const value = number.range ? `${number.range[0]}–${number.range[1]}` : numeral(number.estimate)
    return { label: number.range ? RANGE_WORD : b.label, value, marker: false, tip: number.text }
  }
  return null
}

/** N-19k (D2-03): the row of a result outside what was validated. Every bar starts at the one origin, zero log-odds, and
 *  holds its own change: the starting row too, which states no percent. Nothing stands on a running total. */
export function flatBar(b: Bar): Bar {
  const own = b.kind === 'start' ? b.to : b.kind === 'estimate' ? 0 : b.delta
  return { ...b, from: 0, to: own, delta: own }
}

/** N-19l (R6-02): the last row of a flat breakdown under a record: the record's value where the record set the number
 *  (`record`); none where it did not (`record: null`). */
type FlatLast = { record: number | null }

/** N-19n (D3-09; `forensic.waterfall`, `flat`): the rows of the breakdown of a result outside what was validated. They are
 *  what the checks returned and the record's row, and nothing else: the rows are not to be added, so there is no row of the
 *  starting point (the constant that completes the sum), none of the fitted term of a check that could not score, no row of
 *  several checks together, and no last row but the value that a record set. */
export function flatRows(bars: readonly Bar[], record?: number | null): Bar[] {
  return bars.flatMap((b) => (b.kind === 'group' ? b.members : [b]))
    .filter((b) => b.kind === 'detector' || b.kind === 'adjustment' || (b.kind === 'estimate' && record != null))
}

/** N-19k (D2-03), N-19l (R6-02; `forensic.breakdown_flat`): the breakdown of a result outside what was validated lets no
 *  estimate be read off it, with a record or without one: the result is capped, or the statistical estimate that stands
 *  beside its record is withheld. */
export const breakdownFlat = (d: Decision, profile: AnalysisDetail['condition_profile'], number: ResultNumber): boolean =>
  number.kind === 'capped' || statisticalWithheld(d, profile)

function BarRow({ b: drawn, next, hatchId, member = false, open, onToggle, number, flat = false, little = 0, flatLast, declared }: { b: Bar; next: Bar | undefined; hatchId: string; member?: boolean; open?: boolean; onToggle?: () => void; number?: ResultNumber; flat?: boolean; little?: number; flatLast?: FlatLast; declared?: number | null }) {
  const b = flat ? flatBar(drawn) : drawn
  const from = axisPosition(b.from)
  const to = axisPosition(b.to)
  const left = Math.min(from.at, to.at)
  const width = Math.abs(to.at - from.at)
  const last = lastRow(b, number, flat ? flatLast : undefined, declared)
  const point = !flat && (b.kind === 'start' || b.kind === 'estimate')
  // N-19l (R6-02): the record's row outside what was validated holds no amount and no bar: its amount is no check's. The
  // bound on a provenance check's influence in a result that no record decides is what that check returned, and keeps it
  const noAmount = flat && flatLast != null && b.kind === 'adjustment'
  const marker = point && (last?.marker ?? true)
  const toward = b.delta >= 0
  const bold = point || b.kind === 'estimate'
  const label = last?.label ?? b.label
  // L-7 (design P2): the column holds changes in log-odds; the two marker rows state a probability and say so with their unit
  // N-19p (D6-04): the starting point is out of 100, as "The bars start at 8 out of 100" above it says (it read "8 %")
  // N-19p (D6-05): a row that the decision does not count reads "not counted", with no bar (it printed "−0.00" with a tick)
  const uncounted = !!b.notCounted
  const value = last?.value ?? (uncounted ? NOT_COUNTED : noAmount ? '—' : point ? `${numeral(sigmoid(b.to))}${b.kind === 'start' ? ' out of 100' : ' %'}` : signed(b.delta))
  const tip = last?.tip ?? (uncounted ? `${b.label}: not counted in this decision` : noAmount ? RECORD_ROW_NO_AMOUNT : point
    ? `${b.label}: ${signed(b.to)} log-odds (${numeral(sigmoid(b.to))})`
    : flat && b.kind === 'start' ? `${b.label}: ${signed(b.delta, 3)} log-odds`
      : `Δ log-odds ${signed(b.delta, 3)}${b.weight != null ? ` · weight ${b.weight.toFixed(2)}` : ''}${b.p != null ? ` · p ${b.p.toFixed(2)}` : ''}${b.approximate ? ' · approximate' : ''}`)
  const whole = flat ? b.kind === 'estimate' || noAmount || uncounted : point || uncounted
  // N-19k (design re-review D2-12): a check that the text calls "little effect" (a change below the threshold of D-63) is a
  // thin bar in the neutral ink, as its dot in the strip is small and neutral; it was a full bar in a direction's colour
  const slight = b.kind === 'detector' && Math.abs(b.delta) < little
  return (
    <div data-bar={b.kind} data-from={pct(from.at)} data-signal={b.signalId ?? undefined} tabIndex={0} title={tip} aria-label={`${b.displayId ? `${b.displayId} ` : ''}${label}: ${whole ? value : `${value} log-odds`}`}
      className={cx(GRID, 'items-center min-h-5 rounded-xs hover:bg-surface-3 focus-visible:bg-surface-3', member && 'text-text-muted')}>
      <span className={cx('min-w-0 flex items-baseline gap-2 text-small', member && 'pl-4')}>
        {b.displayId && <span className="font-mono text-micro text-text-muted shrink-0">{b.displayId}</span>}
        {b.kind === 'group' && onToggle
          ? <button type="button" aria-expanded={open} onClick={onToggle} className="min-w-0 break-words text-left text-text underline decoration-border-strong underline-offset-[3px] hover:decoration-text">{b.label}</button>
          : <span className={cx('min-w-0 break-words', bold ? 'font-medium text-text' : member ? '' : 'text-text')}>{label}</span>}
      </span>
      <span className="relative h-5">
        <span aria-hidden className="absolute -top-3 h-8 w-px bg-graphic" style={{ left: '50%' }} />
        {marker && <span aria-hidden className="absolute top-[3px] w-0.5 h-3.5 -translate-x-1/2 bg-text" style={{ left: pct(to.at) }} />}
        {!point && !uncounted && b.kind !== 'adjustment' && !(flat && b.kind === 'estimate') && (
          // N-19k (D2-03): one ink outside what was validated; the side of the origin says the direction
          <span aria-hidden data-fill data-slight={slight || undefined} className={cx('absolute', slight ? 'top-[7px] h-1.5' : 'top-0 h-5', flat || slight ? 'bg-dir-neutral' : toward ? 'bg-dir-ai' : 'bg-dir-authentic', member && 'opacity-60')}
            style={{ left: pct(left), width: `max(1px, ${pct(width)})`, borderRadius: 1 }} />
        )}
        {b.kind === 'adjustment' && !noAmount && (
          <svg aria-hidden className="absolute top-0 h-5 overflow-visible" style={{ left: pct(left), width: `max(1px, ${pct(width)})` }}>
            <rect width="100%" height="100%" fill={`url(#${hatchId})`} />
            <line x1="0" x2="0" y1="0" y2="100%" strokeDasharray="3 2" style={{ stroke: 'var(--color-graphic)' }} />
            <line x1="100%" x2="100%" y1="0" y2="100%" strokeDasharray="3 2" style={{ stroke: 'var(--color-graphic)' }} />
          </svg>
        )}
        {(to.overflow !== 0) && <span aria-hidden className="absolute top-0 font-mono text-small text-text" style={to.overflow > 0 ? { right: -10 } : { left: -10 }}>{to.overflow > 0 ? '›' : '‹'}</span>}
        {next && !member && !flat && <span aria-hidden className="absolute top-full h-3 w-px bg-graphic" style={{ left: pct(to.at) }} />}
      </span>
      {/* N-19m (D3-05): a record's value, "85 out of 100", is wider than the column: it is set on one line from the column's
          right edge and runs into the track, which holds no bar on that row (it broke over two lines) */}
      <span className={cx('justify-self-end whitespace-nowrap font-mono text-mono num text-right', bold ? 'text-text font-medium' : 'text-text')}>{value}</span>
    </div>
  )
}

function Waterfall({ bars, check, number, little, flat, flatLast, declared }: { bars: Bar[]; check: Extract<Integrity, { ok: true }>; number: ResultNumber; little: number; flat: boolean; flatLast?: FlatLast; declared?: number | null }) {
  const [open, setOpen] = useState(false)
  const [table, setTable] = useState(false)
  const hatchId = useSvgId('waterfall-hatch')
  // `flat` (N-19k, D2-03; N-19l, R6-02): one origin, log-odds, no tick of a probability, no running total
  return (
    <div className="mt-6">
      <h3 className="m-0 eyebrow">Contributions</h3>
      <div role="group" aria-label={flat ? 'What each check returned' : 'Contribution waterfall'} data-testid="waterfall" data-flat={flat || undefined} className="mt-3 flex flex-col gap-3">
        <svg width={0} height={0} aria-hidden className="absolute"><defs><HatchPattern id={hatchId} tone="graphic" /></defs></svg>
        <div aria-hidden className={cx(GRID, 'items-end')}>
          <span />
          {/* N-19b: below 640 px each end's caption has its half of the axis and takes two lines, and the one tick is 50 */}
          <span data-testid="waterfall-axis" className="relative h-9 max-[639px]:h-[52px] block">
            <span className="absolute top-0 left-0 max-w-[calc(50%-4px)] text-caption text-text-muted">toward no indications</span>
            <span className="absolute top-0 right-0 max-w-[calc(50%-4px)] text-right text-caption text-text-muted">toward indications</span>
            {flat
              ? LOG_ODDS_TICKS.map((t) => (
                <span key={t} data-tick className={cx('absolute bottom-0 -translate-x-1/2 font-mono text-micro text-text-muted num', t !== 0 && 'max-[639px]:hidden')} style={{ left: pct(axisPosition(t).at) }}>{t === 0 ? '0' : signed(t, 0)}</span>
              ))
              : PROB_TICKS.map((t) => (
                <span key={t} data-tick className={cx('absolute bottom-0 -translate-x-1/2 font-mono text-micro text-text-muted num', t !== 50 && 'max-[639px]:hidden')} style={{ left: pct(axisPosition(logit(t / 100)).at) }}>{t}</span>
              ))}
          </span>
          <span className="text-caption text-text-muted text-right">Δ <span className="whitespace-nowrap">log-odds</span></span>
        </div>
        {bars.map((b, i) => (
          <Fragment key={b.key}>
            <BarRow b={b} next={bars[i + 1]} hatchId={hatchId} open={open} onToggle={b.kind === 'group' ? () => setOpen(!open) : undefined} number={number} flat={flat} little={little} flatLast={flatLast} declared={declared} />
            {b.kind === 'group' && open && b.members.map((m) => <BarRow key={m.key} b={m} next={undefined} hatchId={hatchId} member flat={flat} little={little} />)}
          </Fragment>
        ))}
      </div>
      {check.clamped && !flat && <p className="mt-3 m-0 text-caption text-text-muted">The estimate is displayed clamped to the 2–98 range; the contributions add up to the unclamped value.</p>}
      <div className="mt-3">
        <Button variant="text" aria-expanded={table} onClick={() => setTable(!table)}>{table ? 'Hide table' : 'View as table'}</Button>
      </div>
      {table && (
        <table data-testid="waterfall-table" className="mt-2 w-full border-collapse text-small">
          <thead>
            <tr className="border-b border-border-strong">
              <th scope="col" className="eyebrow text-left font-medium py-2 pr-4">Row</th>
              <th scope="col" className="eyebrow text-right font-medium py-2 pr-4">Δ log-odds</th>
              {!flat && <th scope="col" className="eyebrow text-right font-medium py-2 pr-4">Running total</th>}
              {!flat && <th scope="col" className="eyebrow text-right font-medium py-2">As estimate</th>}
            </tr>
          </thead>
          <tbody>
            {bars.flatMap((b) => (b.kind === 'group' ? b.members : [b])).map((b) => {
              const last = lastRow(b, number, flat ? flatLast : undefined, declared)
              if (flat) {  // the row's own change, the starting row's too; no total and no percent
                const own = flatBar(b)
                return (
                  <tr key={b.key} className="border-b border-border">
                    <th scope="row" className="text-left font-normal py-1.5 pr-4 text-text">{b.displayId ? `${b.displayId} ` : ''}{last?.label ?? b.label}</th>
                    <td className="text-right py-1.5 font-mono text-mono num">{b.notCounted ? NOT_COUNTED : b.kind === 'estimate' || (flatLast != null && b.kind === 'adjustment') ? '—' : signed(own.delta, 4)}</td>
                  </tr>
                )
              }
              return (
                <tr key={b.key} className="border-b border-border">
                  <th scope="row" className="text-left font-normal py-1.5 pr-4 text-text">{b.displayId ? `${b.displayId} ` : ''}{last?.label ?? b.label}</th>
                  <td className="text-right py-1.5 pr-4 font-mono text-mono num">{b.notCounted ? NOT_COUNTED : b.kind === 'start' || b.kind === 'estimate' ? '—' : signed(b.delta, 4)}</td>
                  <td className="text-right py-1.5 pr-4 font-mono text-mono num">{signed(b.to, 4)}</td>
                  <td className="text-right py-1.5 font-mono text-mono num">{last?.value ?? `${numeral(sigmoid(b.to))}${b.kind === 'start' ? ' out of 100' : ''}`}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

/** L3 "How the estimate was reached" (UX §5.6): narrative from counts, agreement strip, and the contribution waterfall, drawn only
 *  when the recorded rows add up to the estimate; otherwise the explained fallback. */
export default function HowReached({ analysis, signals, findings = [] }: { analysis: AnalysisDetail; signals: readonly SignalV2[]; findings?: readonly Finding[] }) {
  const d = analysis.decision as Decision | null | undefined
  if (!d) return null
  const check = integrity(d)
  const number = resultNumber(d, findings, analysis.condition_profile)
  const flat = breakdownFlat(d, analysis.condition_profile, number)
  const flatLast = flat && number.kind !== 'capped' ? { record: recordSetNumber(d) ? estimateOf(d) : null } : undefined
  const bars = !check.ok ? [] : flat ? flatRows(waterfall(d, signals), flatLast?.record) : waterfall(d, signals)
  if (d.code === 'not_assessable') {
    // N-17 (D-68; design review DS-10): no narrative, strip or breakdown of an estimate that was not formed, and no heading that names one
    return (
      <section id="section-reached" data-testid="how-reached" data-integrity="not_assessable" aria-labelledby="reached-heading" className="scroll-mt-14">
        <h2 id="reached-heading" className="m-0 eyebrow">How the result was reached</h2>
        <p data-testid="no-estimate-formed" className="mt-2 m-0 text-body text-text max-w-[72ch]">{noEstimateFormed(d)}</p>
      </section>
    )
  }
  return (
    <section id="section-reached" data-testid="how-reached" data-integrity={check.ok ? 'additive' : check.reason} aria-labelledby="reached-heading" className="scroll-mt-14">
      <h2 id="reached-heading" className="m-0 eyebrow">{reachedTitle(flat ? { kind: 'capped' } : number, d)}</h2>
      {flat
        // N-19k (D2-03): one sentence, in place of what "pushed the estimate" and where "the bars start"; N-19l (R6-02):
        // under a record too, with why the record's row carries no amount
        ? <p data-testid="capped-combined" className="mt-2 m-0 text-body text-text max-w-[72ch]">{cappedCombined(analysis.kind, bars.length > 0, recordOf(d) != null, recordOf(d) != null && (d.contributions ?? []).some((r) => r.kind === 'override'))}</p>
        : check.ok && narrative(d).map((line) => <p key={line} className="mt-2 m-0 text-body text-text max-w-[72ch]">{line}</p>)}
      <AgreementStrip signals={signals} number={number} little={littleOf(d)} />
      {check.ok
        ? (bars.length > 0 || !flat) && <Waterfall bars={bars} check={check} number={number} little={littleOf(d)} flat={flat} flatLast={flatLast}
            declared={recordSetNumber(d) ? estimateOf(d) : null} />
        : (
          <div data-testid="waterfall-fallback" className="mt-3">
            {check.reason === 'record'
              // N-19h (E-19, 3): where a record decides alone the breakdown is a sentence, not a chart that is unavailable
              ? <p className="m-0 text-body text-text max-w-[72ch]">{fallbackText(check, d, aloneWhy(d, analysis.assessments, analysis.kind), analysis.condition_profile)}</p>
              : (
                <>
                  <p className="m-0 text-body text-text">Contribution breakdown unavailable for this analysis.</p>
                  <p className="mt-1 m-0 text-small text-text-muted max-w-[72ch]">{fallbackText(check, d, null, analysis.condition_profile)}</p>
                </>
              )}
          </div>
        )}
    </section>
  )
}
