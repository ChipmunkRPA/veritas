import { ErrorState } from './ui'
import { cx } from '../lib/cx'

/** Media access is independent of retained evidence: a failed signed-link request must remain retryable. */
export default function MediaAccessNotice({ ready, failed, pending, onRetry, className }: {
  ready: boolean
  failed: boolean
  pending: boolean
  onRetry: () => void
  className?: string
}) {
  if (ready) return null
  return (
    <section data-testid="media-access" aria-label="Media and downloads" className={cx('shrink-0 border-y border-border py-4', className)}>
      {failed && !pending
        ? <ErrorState onRetry={onRetry}>Could not prepare secure media links. Loaded evidence remains available.</ErrorState>
        : <p role="status" className="m-0 text-small text-text-muted">Preparing secure media links… Loaded evidence remains available.</p>}
    </section>
  )
}
