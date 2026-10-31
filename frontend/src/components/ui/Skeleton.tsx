import { cx } from '../../lib/cx'

/** A surface-3 bar with a 1.6 s opacity pulse. Never wider than what holds it (N-19b): a bar of 360 or 420 px widened a
 *  375 px page while its rows loaded. */
export function Skeleton({ w, h = 12, className }: { w: number | string; h?: number; className?: string }) {
  return <span aria-hidden className={cx('inline-block max-w-full rounded-xs bg-surface-3 animate-pulse-skeleton', className)} style={{ width: w, height: h }} />
}
