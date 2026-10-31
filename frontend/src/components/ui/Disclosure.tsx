import { useId, useState, type ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'
import { cx } from '../../lib/cx'

/** Disclosure (UX §9: extracted from the v1 `Footnote` and `RawDetails`): a muted text trigger with a chevron and an animated
 *  grid-rows panel. `summary` sits beside the label in muted numerals. */
export function Disclosure({ label, summary, children, defaultOpen = false, className, testId }: {
  label: ReactNode
  summary?: ReactNode
  children: ReactNode
  defaultOpen?: boolean
  className?: string
  testId?: string
}) {
  const [open, setOpen] = useState(defaultOpen)
  const id = useId()
  return (
    <div className={className} data-testid={testId} data-open={open}>
      <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 min-h-8 text-small text-text-muted hover:text-text rounded-sm text-left">
        <ChevronRight size={14} strokeWidth={1.5} aria-hidden className={cx('shrink-0 transition-transform duration-base', open && 'rotate-90')} />
        <span className="text-text">{label}</span>
        {summary && <span className="num">· {summary}</span>}
      </button>
      <div id={id} className="disclosure" data-open={open}>
        <div>{open && <div className="pt-2 pb-4 pl-[18px]">{children}</div>}</div>
      </div>
    </div>
  )
}
