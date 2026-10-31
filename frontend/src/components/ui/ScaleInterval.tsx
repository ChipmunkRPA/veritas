import { cx } from '../../lib/cx'
import { Skeleton } from './Skeleton'

const TINT = ['bg-verdict-1-tint', 'bg-verdict-2-tint', 'bg-verdict-3-tint', 'bg-verdict-4-tint', 'bg-verdict-5-tint'] as const
const SOLID = ['bg-verdict-1', 'bg-verdict-2', 'bg-verdict-3', 'bg-verdict-4', 'bg-verdict-5'] as const
const BORDER = ['border-verdict-1', 'border-verdict-2', 'border-verdict-3', 'border-verdict-4', 'border-verdict-5'] as const

const BOUNDARIES = [0, 15, 35, 65, 85, 100] as const

const clamp = (v: number) => Math.min(100, Math.max(0, v))

/**
 * Scale with interval (UX §8.4): 6 px track split at the 15/35/65/85 verdict thresholds, with 2 px gaps; the plausible range drawn inside the
 * track at 2 px with 10 px `graphic` end ticks; a 2×14 ink marker at the estimate. Step 0 (insufficient) draws an
 * empty track and no marker.
 *
 * `band` (D-68) is for a result whose number is not a point on this scale: an uncalibrated score, a result outside what was
 * validated, an inconclusive result. No marker is drawn and `label` says what the number is. With a plausible range the range
 * is the mark; without one the verdict's segment is filled.
 *
 * `note` (D-63, "Headline rule in words") replaces the three labels under the track for a result of a calibrated fusion model,
 * whose headline follows the calibrated levels and not the segment its marker is in: the labels would name the segments as
 * headlines.
 */
export function ScaleInterval({ step, estimate, low, high, band = false, label: said, note, loading = false, className }: {
  step: 0 | 1 | 2 | 3 | 4 | 5
  estimate?: number | null
  low?: number | null
  high?: number | null
  band?: boolean
  label?: string
  note?: string | null
  loading?: boolean
  className?: string
}) {
  if (loading) {
    return (
      <div data-state="loading" aria-busy="true" className={cx('w-full', className)}>
        <Skeleton w="100%" h={6} className="block" />
        <div className="mt-2 flex justify-between"><Skeleton w={88} /><Skeleton w={72} /><Skeleton w={104} /></div>
      </div>
    )
  }
  const banded = band && step > 0
  const insufficient = !banded && (step === 0 || estimate == null || !Number.isFinite(estimate))
  const e = insufficient || banded || estimate == null ? null : clamp(estimate)
  const range = !insufficient && low != null && high != null && Number.isFinite(low) && Number.isFinite(high)
    ? [clamp(Math.min(low, high)), clamp(Math.max(low, high))] as const
    : null
  const filled = banded && !range ? step : null
  const label = said ?? (e == null
    ? (step === 0 ? 'Insufficient evidence: not assessed' : 'No estimate')
    : `Estimate ${Math.round(e)} of 100` + (range ? `, plausible range ${Math.round(range[0])}–${Math.round(range[1])}` : ''))

  return (
    <div data-state={insufficient ? 'insufficient' : banded ? 'band' : 'ready'} className={cx('w-full', className)}>
      <div className="relative" role="img" aria-label={label}>
        <div className="relative h-(--viz-scale-track)">
          {[1, 2, 3, 4, 5].map((n) => (
            <span key={n} data-scale-band={n} data-filled={filled === n ? '' : undefined}
              className={cx('absolute inset-y-0 border', insufficient ? 'bg-surface-2 border-border-strong' : cx(filled === n ? SOLID[n - 1] : TINT[n - 1], BORDER[n - 1]))}
              style={{ left: `${BOUNDARIES[n - 1]}%`, width: n === 5 ? '15%' : `calc(${BOUNDARIES[n] - BOUNDARIES[n - 1]}% - 2px)`, borderRadius: 1 }} />
          ))}
        </div>
        {range && (
          <>
            <span aria-hidden data-mark="interval" className="absolute top-0.5 h-(--viz-interval-h) bg-interval" style={{ left: `${range[0]}%`, width: `${range[1] - range[0]}%` }} />
            <span aria-hidden className="absolute -top-0.5 h-(--viz-interval-tick) w-px bg-graphic" style={{ left: `calc(${range[0]}% - 0.5px)` }} />
            <span aria-hidden className="absolute -top-0.5 h-(--viz-interval-tick) w-px bg-graphic" style={{ left: `calc(${range[1]}% - 0.5px)` }} />
          </>
        )}
        {e != null && <span aria-hidden data-mark="estimate" className="absolute -top-1 w-(--viz-marker-w) h-(--viz-marker-h) bg-text" style={{ left: `calc(${e}% - 1px)` }} />}
      </div>
      {note
        ? <p data-testid="scale-note" className="mt-2 m-0 text-caption text-text-muted">{note}</p>
        : (
          <div aria-hidden data-testid="scale-labels" className="mt-2 flex justify-between text-caption text-text-muted">
            <span>no indications</span><span>inconclusive</span><span>strong indications</span>
          </div>
        )}
    </div>
  )
}
