import { useRef, type KeyboardEvent, type ReactNode } from 'react'
import { cx } from '../../lib/cx'

export interface TabItem<T extends string> { id: T; label: string; icon?: ReactNode }

/** Text tabs with a 2 px underline in `text` sitting on the hairline rule. Arrow keys move between tabs. */
export function Tabs<T extends string>({ items, value, onChange, label, className }: { items: TabItem<T>[]; value: T; onChange: (v: T) => void; label: string; className?: string }) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const onKey = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    const n = items.length
    let j = -1
    if (e.key === 'ArrowRight') j = (i + 1) % n
    else if (e.key === 'ArrowLeft') j = (i - 1 + n) % n
    else if (e.key === 'Home') j = 0
    else if (e.key === 'End') j = n - 1
    if (j >= 0) { e.preventDefault(); onChange(items[j].id); refs.current[j]?.focus() }
  }
  return (
    <div role="tablist" aria-label={label} className={cx('flex gap-6 border-b border-border', className)}>
      {items.map((t, i) => {
        const active = t.id === value
        return (
          <button key={t.id} ref={(el) => { refs.current[i] = el }} role="tab" id={`tab-${t.id}`} aria-selected={active} aria-controls={`panel-${t.id}`} tabIndex={active ? 0 : -1}
            onClick={() => onChange(t.id)} onKeyDown={(e) => onKey(e, i)}
            className={cx('relative -mb-px pb-3 text-body font-medium transition-colors duration-fast', active ? 'text-text' : 'text-text-muted hover:text-text')}>
            <span className="inline-flex items-center gap-2">{t.icon}{t.label}</span>
            <span aria-hidden className={cx('absolute left-0 right-0 bottom-0 h-0.5', active ? 'bg-text' : 'bg-transparent')} />
          </button>
        )
      })}
    </div>
  )
}
