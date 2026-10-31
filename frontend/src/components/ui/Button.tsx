import { forwardRef, type ButtonHTMLAttributes, type AnchorHTMLAttributes, type ReactNode } from 'react'
import { cx } from '../../lib/cx'

export type ButtonVariant = 'primary' | 'ghost' | 'text' | 'icon' | 'danger-text' | 'danger'

const base = 'inline-flex items-center justify-center gap-2 select-none whitespace-nowrap transition-[background-color,color,opacity] duration-fast ease-standard'
const variants: Record<ButtonVariant, string> = {
  primary: 'h-9 px-3.5 rounded-sm bg-text text-surface text-small font-medium hover:opacity-[.92] active:translate-y-px disabled:bg-border-strong disabled:text-surface dark:disabled:text-text-muted disabled:hover:opacity-100 disabled:active:translate-y-0 aria-disabled:pointer-events-none',
  ghost: 'h-9 px-3.5 rounded-sm border border-graphic bg-transparent text-text text-small font-medium hover:bg-surface-3 disabled:opacity-40',
  text: 'text-accent underline underline-offset-[3px] decoration-1 hover:text-accent-hover hover:decoration-2 text-small font-medium min-h-8 px-0.5',
  icon: 'h-8 w-8 rounded-sm text-text-muted hover:bg-surface-3 hover:text-text',
  // the design contract's destructive button: the secondary button in the danger colour, never a filled red one (N-19a)
  danger: 'h-9 px-3.5 rounded-sm border border-danger/40 bg-transparent text-danger text-small font-medium hover:bg-danger/10 disabled:opacity-40',
  'danger-text': 'text-danger underline underline-offset-[3px] decoration-1 hover:decoration-2 text-small font-medium min-h-8 px-0.5',
}

type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; children?: ReactNode }

export const Button = forwardRef<HTMLButtonElement, BtnProps>(function Button({ variant = 'ghost', className, type = 'button', ...rest }, ref) {
  if (variant === 'icon' && !rest['aria-label']) {
    // Icon buttons must carry an accessible name (§6).
    console.warn('Button[variant=icon] rendered without aria-label')
  }
  return <button ref={ref} type={type} className={cx(base, variants[variant], className)} {...rest} />
})

type LinkBtnProps = AnchorHTMLAttributes<HTMLAnchorElement> & { variant?: ButtonVariant; children?: ReactNode }

/** Anchor styled as a button — used for download links so the browser handles them natively. */
export function ButtonLink({ variant = 'ghost', className, ...rest }: LinkBtnProps) {
  return <a className={cx(base, variants[variant], className)} {...rest} />
}
