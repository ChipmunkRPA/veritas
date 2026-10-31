import type { ReactNode } from 'react'
import { cx } from '../../lib/cx'
import { Button } from './Button'
import { HatchSwatch } from './Hatch'
import { Skeleton } from './Skeleton'

/* Global state patterns (UX §8.5). Each renders in place of one section; other sections keep rendering. */

/** Empty: one sentence and at most one action, left-aligned, no illustration. */
export function EmptyState({ children, action, onAction, className }: { children: ReactNode; action?: string; onAction?: () => void; className?: string }) {
  return (
    <p data-state="empty" className={cx('m-0 flex flex-wrap items-center gap-x-3 text-small text-text-muted', className)}>
      <span>{children}</span>
      {action && onAction && <Button variant="text" onClick={onAction}>{action}</Button>}
    </p>
  )
}

/** Fetch error: an inline danger sentence and a Retry text button. */
export function ErrorState({ children, onRetry, className }: { children: ReactNode; onRetry?: () => void; className?: string }) {
  return (
    <p role="alert" data-state="error" className={cx('m-0 flex flex-wrap items-center gap-x-3 text-small text-danger', className)}>
      <span>{children}</span>
      {onRetry && <Button variant="text" onClick={onRetry}>Retry</Button>}
    </p>
  )
}

/** A check that did not run: hatch texture, the check's name and the reason (P9: drawn, not omitted). */
export function NotRun({ what, reason, className }: { what: string; reason: string; className?: string }) {
  return (
    <span data-state="partial" className={cx('inline-flex flex-wrap items-center gap-x-2 text-small', className)}>
      <HatchSwatch label={<span className="text-text">{what}</span>} />
      <span className="text-text-muted">not run · {reason}</span>
    </span>
  )
}

/** Loading rows shaped like the final content (no spinners in the middle of the page). */
export function SkeletonRows({ widths = ['72%', '64%', '48%'], className }: { widths?: readonly (string | number)[]; className?: string }) {
  return (
    <div data-state="loading" aria-busy="true" className={cx('flex flex-col gap-3', className)}>
      {widths.map((w, i) => <Skeleton key={i} w={w} h={12} className="block" />)}
    </div>
  )
}
