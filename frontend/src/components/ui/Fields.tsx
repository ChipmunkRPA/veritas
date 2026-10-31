import { forwardRef, useEffect, useId, useState, type SelectHTMLAttributes } from 'react'
import { ChevronDown } from 'lucide-react'
import { cx } from '../../lib/cx'
import { fmtDate, parseLocalDateTime } from '../../lib/format'

const field = 'h-9 min-w-0 rounded-sm border border-graphic bg-surface text-small text-text focus-visible:border-accent'

/** A select in the field style of the workspace (L-7, design P2): the native control and its keyboard behaviour, without the
 *  operating system's chrome: a hairline border, the field height and one chevron in the muted ink. `className` sizes the control. */
export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement> & { compact?: boolean }>(
  function Select({ className, compact = false, children, ...rest }, ref) {
    return (
      <span className={cx('relative inline-flex min-w-0 max-w-full align-middle', className)}>
        <select ref={ref} {...rest} className={cx(field, 'appearance-none w-full pl-2.5 pr-8 disabled:opacity-40', compact && 'h-8')}>{children}</select>
        <ChevronDown size={14} strokeWidth={1.5} aria-hidden className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
      </span>
    )
  })

const pad = (n: number) => String(n).padStart(2, '0')
/** `2026-09-22T14:30` (the draft's local value) as the reader types it: `2026-09-22 14:30`. */
const shown = (value: string) => value.replace('T', ' ')

/** A local date and time typed as `2026-09-22 14:30` (L-7, design P2): one written format in place of the operating system's
 *  date control, with the instant read back in the workspace's own format (UTC, named) or the reason it cannot be read.
 *  `value` and `onChange` carry `YYYY-MM-DDTHH:MM` in local time, or the text as typed while it is not a date, so the form's
 *  own check still refuses it. */
export function DateTimeField({ id, value, onChange, className }: { id?: string; value: string; onChange: (value: string) => void; className?: string }) {
  const own = useId()
  const [text, setText] = useState(shown(value))
  useEffect(() => { setText((t) => (parseLocalDateTime(t) && toValue(parseLocalDateTime(t)!) === value ? t : shown(value))) }, [value])
  const parsed = text.trim() ? parseLocalDateTime(text) : null
  const invalid = text.trim() !== '' && !parsed
  const note = `${own}-note`
  return (
    <span className={cx('flex flex-wrap items-center gap-x-3 gap-y-1 min-w-0', className)}>
      <input id={id} type="text" inputMode="numeric" autoComplete="off" spellCheck={false} placeholder="YYYY-MM-DD HH:MM" value={text}
        aria-invalid={invalid || undefined} aria-describedby={note}
        onChange={(e) => {
          setText(e.target.value)
          const d = parseLocalDateTime(e.target.value)
          onChange(d ? toValue(d) : e.target.value.trim())
        }}
        className={cx(field, 'w-44 px-2.5 font-mono text-mono num placeholder:text-text-muted', invalid && 'border-danger')} />
      <span id={note} data-testid="datetime-note" className={cx('text-caption', invalid ? 'text-danger' : 'text-text-muted')}>
        {invalid ? 'Write the date and time as 2026-09-22 14:30.' : parsed ? `your local time · recorded as ${fmtDate(parsed.toISOString())}` : 'your local time, 24-hour clock; recorded in UTC'}
      </span>
    </span>
  )
}

function toValue(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}
