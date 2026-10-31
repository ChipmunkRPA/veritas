import { buildLut, lutCss } from '../../lib/colormap'
import { cx } from '../../lib/cx'
import { HatchSwatch } from './Hatch'

const LUT = buildLut()
const RAMP_W = 160
const RAMP_H = 8

/**
 * Heat legend (UX §8.3): 160×8 inferno ramp drawn as vector slices, ticks 0 · 0.5 · 1 in 11 px mono, the title,
 * a hatch swatch "unreliable" and the integrity score. `onStage` switches to the media-stage ink and lens hatch.
 */
export function HeatLegend({ title, integrity, attention = false, showUnreliable = true, onStage = false, className, ticks = ['0', '0.5', '1'], testId }: {
  title: string
  /** The labels at the dark end, the middle and the light end of the ramp; a map's score by default (N-19d: a spectrogram's dB) */
  ticks?: readonly [string, string, string]
  testId?: string
  integrity?: string | null
  attention?: boolean
  showUnreliable?: boolean
  onStage?: boolean
  className?: string
}) {
  const ink = onStage ? 'text-stage-text' : 'text-text'
  const muted = onStage ? 'text-stage-muted' : 'text-text-muted'
  return (
    <figure data-legend="heat" data-testid={testId} className={cx('m-0 inline-flex flex-col gap-1.5', className)}>
      <figcaption className={cx('flex items-center gap-2 text-caption', ink)}>
        {attention && <span className="px-1 rounded-xs border border-current text-micro uppercase tracking-[0.08em]">Attention</span>}
        <span>{title}</span>
      </figcaption>
      <svg width={RAMP_W} height={22} role="img" aria-label={`${title}: dark at ${ticks[0]}, light at ${ticks[2]}`} className={cx('block overflow-visible', muted)}>
        {Array.from({ length: RAMP_W }, (_, i) => (
          <rect key={i} x={i} y={0} width={1} height={RAMP_H} fill={lutCss(LUT, (i * 255) / (RAMP_W - 1))} shapeRendering="crispEdges" />
        ))}
        {[0, RAMP_W / 2, RAMP_W - 1].map((x) => <rect key={x} x={x} y={RAMP_H} width={1} height={3} fill="currentColor" />)}
        <text x={0} y={21} textAnchor="start" className="font-mono text-micro" fill="currentColor">{ticks[0]}</text>
        <text x={RAMP_W / 2} y={21} textAnchor="middle" className="font-mono text-micro" fill="currentColor">{ticks[1]}</text>
        <text x={RAMP_W} y={21} textAnchor="end" className="font-mono text-micro" fill="currentColor">{ticks[2]}</text>
      </svg>
      {(showUnreliable || integrity) && (
        <span className={cx('flex items-center gap-4 text-caption', muted)}>
          {showUnreliable && <HatchSwatch tone={onStage ? 'lens' : 'graphic'} label="unreliable" />}
          {integrity && <span>integrity <span className={cx('font-mono num', ink)}>{integrity}</span></span>}
        </span>
      )}
    </figure>
  )
}
