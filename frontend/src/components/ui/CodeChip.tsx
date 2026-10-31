import { useEffect, useRef, useState } from 'react'
import { cx } from '../../lib/cx'

/** Mono chip on surface-2; click copies `copy` (or children text) and shows "Copied" for 1.2 s. */
export function CodeChip({ children, copy, className, title }: { children: string; copy?: string; className?: string; title?: string }) {
  const [copied, setCopied] = useState(false)
  const timer = useRef<number | null>(null)
  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current) }, [])
  const onCopy = async () => {
    const text = copy ?? children
    try { await navigator.clipboard.writeText(text) } catch { return }
    setCopied(true)
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setCopied(false), 1200)
  }
  return (
    <span className={cx('inline-flex items-center gap-2', className)}>
      <button type="button" onClick={onCopy} title={title ?? `Copy ${copy ?? children}`} aria-label={`Copy ${copy ?? children}`}
        className="font-mono text-mono bg-surface-2 px-1.5 py-0.5 rounded-sm text-text hover:bg-surface-3 transition-colors duration-fast">
        <code>{children}</code>
      </button>
      <span aria-live="polite" className={cx('text-caption text-text-muted transition-opacity duration-fast', copied ? 'opacity-100' : 'opacity-0')}>{copied ? 'Copied' : ''}</span>
    </span>
  )
}
