import { cx } from '../../lib/cx'

/** Icon array (UX §8.4): 10×10 grid of 6 px dots, 3 px gap, filled ink vs `border-strong` ring, filled in reading order. */
export function IconArray({ filled, label, className }: { filled: number; label: string; className?: string }) {
  const n = Number.isFinite(filled) ? Math.max(0, Math.min(100, Math.round(filled))) : 0
  return (
    <span role="img" aria-label={label} data-filled={n}
      className={cx('inline-grid grid-cols-[repeat(10,var(--viz-icon-dot))] gap-(--viz-icon-gap)', className)}>
      {Array.from({ length: 100 }, (_, i) => (
        <span key={i} data-dot={i < n ? 'on' : 'off'} className={cx('block size-(--viz-icon-dot) rounded-full', i < n ? 'bg-text' : 'border border-border-strong')} />
      ))}
    </span>
  )
}
