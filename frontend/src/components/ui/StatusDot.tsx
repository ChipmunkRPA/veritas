import { cx } from '../../lib/cx'

export type Status = 'done' | 'queued' | 'running' | 'failed' | 'offline' | 'ok' | 'loading'

const dotColor: Record<Status, string> = {
  done: 'bg-ok', ok: 'bg-ok', queued: 'bg-text-faint', loading: 'bg-text-faint',
  running: 'bg-accent animate-pulse-dot', failed: 'bg-danger', offline: 'bg-danger',
}
const wordColor: Record<Status, string> = {
  done: 'text-text', ok: 'text-text', queued: 'text-text-muted', loading: 'text-text-muted',
  running: 'text-text', failed: 'text-danger', offline: 'text-danger',
}

/** 6 px dot + 13/20 word. Dot is decorative; the word is the accessible text. */
export function StatusDot({ status, word, title, className }: { status: Status; word?: string; title?: string; className?: string }) {
  return (
    <span className={cx('inline-flex items-center gap-2 text-small', wordColor[status], className)} title={title}>
      <span aria-hidden className={cx('inline-block h-1.5 w-1.5 rounded-full shrink-0', dotColor[status])} />
      {word ?? status}
    </span>
  )
}

/** Bare dot for tight spots (sidebar engine block). */
export function Dot({ status, className }: { status: Status; className?: string }) {
  return <span aria-hidden className={cx('inline-block h-1.5 w-1.5 rounded-full shrink-0', dotColor[status], className)} />
}
