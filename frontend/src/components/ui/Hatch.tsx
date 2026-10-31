import { useId, type ReactNode } from 'react'
import { cx } from '../../lib/cx'

export type HatchTone = 'lane' | 'lens' | 'graphic'

const TONE: Record<HatchTone, string> = {
  lane: 'var(--color-lane-hatch)',     // "not analysed" / "not run" on page backgrounds
  lens: 'var(--color-lens-hatch)',     // "unreliable" over media
  graphic: 'var(--color-graphic)',     // adjustments in charts
}

/** SVG id without the colons React puts in `useId()` values (they break `url(#…)` references). */
export function useSvgId(prefix: string): string {
  return prefix + useId().replace(/[^a-zA-Z0-9_-]/g, '')
}

/** 45° hatch texture, 6 px pitch, 1 px lines (UX §8.1, §8.3). Always paired with a text label. Place inside `<defs>`. */
export function HatchPattern({ id, tone = 'lane' }: { id: string; tone?: HatchTone }) {
  return (
    <pattern id={id} patternUnits="userSpaceOnUse" width={6} height={6} patternTransform="rotate(45)">
      <rect width={1} height={6} style={{ fill: TONE[tone] }} />
    </pattern>
  )
}

/** A 24×12 hatch swatch with its label, for legends and "not run" rows. */
export function HatchSwatch({ label, tone = 'lane', className }: { label?: ReactNode; tone?: HatchTone; className?: string }) {
  const id = useSvgId('hatch')
  return (
    <span className={cx('inline-flex items-center gap-2', className)}>
      <svg width={24} height={12} aria-hidden className="block shrink-0">
        <defs><HatchPattern id={id} tone={tone} /></defs>
        <rect width={24} height={12} fill={`url(#${id})`} />
      </svg>
      {label}
    </span>
  )
}
