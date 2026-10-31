import { useEffect, useId, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { cx } from '../../lib/cx'

export interface MenuItem<T extends string> { id: T; label: string; disabled?: boolean; danger?: boolean }

/**
 * Text + chevron trigger; open menu is a surface popover with hairline, shadow-popover, radius 6.
 * Closes on Esc, outside click, or selection. Arrow keys move focus. `side="top"` opens above the trigger (a trigger at the foot
 * of the sidebar); `className` replaces the root's `inline-block` (a trigger that truncates needs a bounded width).
 */
export function Menu<T extends string>({ label, items, onSelect, align = 'left', side = 'bottom', trigger, triggerClassName, className, ariaLabel }: {
  label?: ReactNode; items: MenuItem<T>[]; onSelect: (id: T) => void; align?: 'left' | 'right'; side?: 'bottom' | 'top'
  trigger?: ReactNode; triggerClassName?: string; className?: string; ariaLabel?: string
}) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  const openAtEnd = useRef(false)
  const id = useId()
  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => { if (root.current && !root.current.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    const choices = list.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')
    choices?.[openAtEnd.current ? choices.length - 1 : 0]?.focus()
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])
  const closeToTrigger = () => {
    setOpen(false)
    button.current?.focus()
  }
  const onRootKey = (e: ReactKeyboardEvent) => {
    if (!open) return
    if (e.key === 'Escape') {
      e.preventDefault(); e.stopPropagation(); closeToTrigger()
    } else if (e.key === 'Tab') {
      // Restore the tab-order origin before the browser moves to the next/previous page control.
      // Do not prevent the native Tab action or trap focus in a nonmodal menu.
      e.stopPropagation(); closeToTrigger()
    }
  }
  const onListKey = (e: ReactKeyboardEvent) => {
    const btns = Array.from(list.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])
    const i = btns.indexOf(document.activeElement as HTMLButtonElement)
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) return
    e.preventDefault(); e.stopPropagation()
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? btns.length - 1
      : e.key === 'ArrowDown' ? (i + 1) % btns.length : (i - 1 + btns.length) % btns.length
    btns[next]?.focus()
  }
  return (
    <div ref={root} className={cx('relative', className ?? 'inline-block')} onKeyDown={onRootKey}
      onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOpen(false) }}>
      <button ref={button} id={`${id}-trigger`} type="button" aria-haspopup="menu" aria-expanded={open} aria-controls={open ? id : undefined} aria-label={ariaLabel}
        onClick={() => { openAtEnd.current = false; setOpen((o) => !o) }}
        onKeyDown={(e) => {
          if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
          e.preventDefault(); e.stopPropagation()
          openAtEnd.current = e.key === 'ArrowUp'; setOpen(true)
        }}
        className={cx('inline-flex items-center gap-1 text-small text-text-muted hover:text-text min-h-8 px-1 rounded-sm transition-colors duration-fast', triggerClassName)}>
        {trigger ?? <>{label}<ChevronDown size={14} strokeWidth={1.5} aria-hidden /></>}
      </button>
      {open && (
        <div ref={list} id={id} role="menu" aria-labelledby={`${id}-trigger`} onKeyDown={onListKey}
          className={cx('absolute z-20 min-w-40 bg-surface border border-border rounded-md shadow-popover py-1 animate-fade-in',
            side === 'top' ? 'bottom-full mb-1' : 'mt-1', align === 'right' ? 'right-0' : 'left-0')}>
          {items.map((it) => (
            <button key={it.id} role="menuitem" type="button" tabIndex={-1} disabled={it.disabled} onClick={() => { closeToTrigger(); onSelect(it.id) }}
              className={cx('block w-full text-left h-8 px-3 text-small whitespace-nowrap hover:bg-surface-3 disabled:opacity-40 disabled:hover:bg-transparent', it.danger ? 'text-danger' : 'text-text')}>
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
