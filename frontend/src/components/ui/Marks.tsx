import type { ReactNode } from 'react'
import { cx } from '../../lib/cx'

/*
 * Typographic marks of the encoding matrix (UX §8.2). Each carries a glyph or shape and a word, so none depends on
 * hue alone and all survive grayscale print.
 */

export type Severity = 'info' | 'notable' | 'anomaly' | 'documentary'

const SEVERITY: Record<Severity, { glyph: string; colour: string; word: string }> = {
  info: { glyph: '○', colour: 'text-sev-info', word: 'Info' },
  notable: { glyph: '◐', colour: 'text-sev-notable', word: 'Notable' },
  anomaly: { glyph: '●', colour: 'text-sev-anomaly', word: 'Anomaly' },
  documentary: { glyph: '◆', colour: 'text-sev-documentary', word: 'Documentary' },
}

/** Metadata severity: glyph + word always; optional detail after a middle dot. */
export function SeverityMark({ severity, children, className }: { severity: Severity; children?: ReactNode; className?: string }) {
  const s = SEVERITY[severity]
  return (
    <span data-severity={severity} className={cx('inline-flex items-baseline gap-1.5 text-small', className)}>
      <span aria-hidden className={cx('inline-block w-3 text-center', s.colour)}>{s.glyph}</span>
      <span className={severity === 'info' ? 'text-text-muted' : 'text-text'}>{s.word}</span>
      {children && <span className="text-text-muted">· {children}</span>}
    </span>
  )
}

export type Direction = 'ai' | 'authentic' | 'neutral'

const DIRECTION: Record<Direction, { word: string; title: string }> = {
  ai: { word: 'for', title: 'Supports AI generation or manipulation' },
  authentic: { word: 'against', title: 'Against AI generation or manipulation' },
  neutral: { word: 'little effect', title: 'Little effect either way' },
}

/** Direction of a signal: filled dot (for), ring (against), small muted dot (little effect). */
export function DirectionMark({ direction, showWord = true, className, word }: { direction: Direction; showWord?: boolean; className?: string; word?: string }) {
  // `word`: another word for the same mark (D-63: "little effect" is said of a check that entered the estimate; a check
  // outside it that leans neither way is "neutral")
  const d = word ? { word, title: word.charAt(0).toUpperCase() + word.slice(1) } : DIRECTION[direction]
  return (
    <span data-direction={direction} title={d.title} className={cx('inline-flex items-center gap-1.5 text-small text-text', className)}>
      <span aria-hidden className="inline-flex size-(--viz-dot) items-center justify-center shrink-0">
        {direction === 'ai' && <span className="block size-(--viz-dot) rounded-full bg-dir-ai" />}
        {direction === 'authentic' && <span className="block size-(--viz-dot) rounded-full border-[1.5px] border-dir-authentic" />}
        {direction === 'neutral' && <span className="block size-(--viz-dot-small) rounded-full bg-dir-neutral" />}
      </span>
      {showWord ? <span>{d.word}</span> : <span className="sr-only">{d.title}</span>}
    </span>
  )
}

export type ReviewState = 'machine' | 'in_review' | 'reviewed' | 'approved' | 'disputed' | 'superseded' | 'withdrawn'

export const REVIEW_STATE: Record<ReviewState, { glyph: string; word: string }> = {
  machine: { glyph: '○', word: 'Machine assessment — not reviewed by an examiner' },
  in_review: { glyph: '◐', word: 'In review' },
  reviewed: { glyph: '●', word: 'Reviewed' },
  approved: { glyph: '●', word: 'Reviewed · second review' },
  disputed: { glyph: '≠', word: 'Disputed · peer reviewer does not concur' },
  superseded: { glyph: '⊘', word: 'Superseded' },
  withdrawn: { glyph: '✕', word: 'Withdrawn' },
}

/** Review state line (UX §5.14): glyph + the state sentence; `children` replaces the default words with the full sentence.
 *  `outstanding` marks a signed revision whose case still needs its second review: the half glyph, since nothing is issued yet. */
export function ReviewStateMark({ state, children, className, outstanding = false }: { state: ReviewState; children?: ReactNode; className?: string; outstanding?: boolean }) {
  const r = REVIEW_STATE[state]
  return (
    <span data-review-state={state} data-outstanding={outstanding || undefined} className={cx('inline-flex items-baseline gap-2 text-small', state === 'withdrawn' ? 'text-danger' : 'text-text', className)}>
      <span aria-hidden className="inline-block w-3 text-center">{outstanding ? REVIEW_STATE.in_review.glyph : r.glyph}</span>
      <span>{children ?? r.word}</span>
    </span>
  )
}

/**
 * Identity chip: F2, S04, Fig. 3, R1, Seg. b, Face A (evidence ids, hairline) or N1 (analyst marks, accent fill).
 * The analyst's voice uses the accent; evidence never does (§8.1).
 */
export function IdChip({ children, analyst = false, title, className }: { children: ReactNode; analyst?: boolean; title?: string; className?: string }) {
  return (
    <span title={title} data-chip={analyst ? 'analyst' : 'evidence'}
      className={cx('inline-flex items-center h-5 px-1.5 rounded-xs font-mono text-micro num whitespace-nowrap',
        analyst ? 'bg-annotation text-on-accent' : 'border border-border-strong text-text', className)}>
      {children}
    </span>
  )
}
