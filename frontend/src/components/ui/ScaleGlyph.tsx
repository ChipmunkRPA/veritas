import { cx } from '../../lib/cx'

/**
 * Five 12x6 cells, 2 px gap, radius 1. All filled cells use the current verdict hue.
 * Step 3 stays achromatic; step 0 (insufficient) leaves every cell empty.
 */
export function ScaleGlyph({ step, className, label }: { step: 0 | 1 | 2 | 3 | 4 | 5; className?: string; label?: string }) {
  const fill = (n: number) => {
    if (n > step) return 'bg-border'
    return (['bg-verdict-1', 'bg-verdict-2', 'bg-verdict-3', 'bg-verdict-4', 'bg-verdict-5'] as const)[step - 1]
  }
  return (
    <span className={cx('inline-flex items-center gap-0.5', className)} role="img" aria-label={label ?? (step === 0 ? 'insufficient evidence' : `scale ${step} of 5`)}>
      {[1, 2, 3, 4, 5].map((n) => <span key={n} className={cx('inline-block h-1.5 w-3', fill(n))} style={{ borderRadius: 1 }} />)}
    </span>
  )
}

/** Three-cell confidence / agreement meter (low / medium / high), vertical bars in `text`. Stays at three (UX §8.1). */
export function ConfidenceMeter({ level, className }: { level: 0 | 1 | 2 | 3; className?: string }) {
  return (
    <span aria-hidden className={cx('inline-flex items-end gap-0.5', className)}>
      {[1, 2, 3].map((n) => <span key={n} className={cx('inline-block w-1.5', n <= level ? 'bg-text' : 'bg-border')} style={{ height: 12, borderRadius: 1 }} />)}
    </span>
  )
}

/** `self_declared` (N-19n; D3-07): a record in the file that anyone can write, drawn as the documentary mark in outline. */
export type StrengthLevel = 'none' | 'weak' | 'moderate' | 'strong' | 'very_strong' | 'documentary' | 'self_declared'
type RecordLevel = 'documentary' | 'self_declared'

const BARS: Record<Exclude<StrengthLevel, RecordLevel>, number> = { none: 0, weak: 1, moderate: 2, strong: 3, very_strong: 4 }
const WORDS: Record<StrengthLevel, string> = {
  none: 'little effect', weak: 'weak support', moderate: 'moderate support', strong: 'strong support',
  very_strong: 'very strong support', documentary: 'documentary', self_declared: 'self-declared',
}

/** Likelihood ratio in either direction → strength band (UX §8.1; band edges "to be confirmed (ENFSI)", D-16). */
export function strengthFromLikelihoodRatio(lr: number): Exclude<StrengthLevel, RecordLevel> {
  if (!Number.isFinite(lr) || lr <= 0) throw new RangeError(`likelihood ratio must be a positive number, got ${lr}`)
  const r = lr >= 1 ? lr : 1 / lr
  return r < 1.5 ? 'none' : r < 3 ? 'weak' : r < 10 ? 'moderate' : r < 100 ? 'strong' : 'very_strong'
}

export function strengthWords(level: StrengthLevel, direction?: 'for' | 'against'): string {
  if (level === 'none' || level === 'documentary' || level === 'self_declared' || !direction) return WORDS[level]
  return `${WORDS[level]} ${direction}`
}

/**
 * Evidence strength (UX §8.1): four achromatic 6×12 bars with a 2 px gap, count-encoded; documentary is a ◆ seal.
 * The words are the accessible name, or visible beside the meter with `showWord`.
 *
 * N-19n (design review D3-07): a record's mark stands in the meter's place. The two diamonds are one drawing at one size:
 * filled for a record that was validated (`documentary`), in outline for a record that anyone can write (`self_declared`),
 * and in the muted ink (`muted`) where the decision did not apply the record. `label`: the accessible name where it is not
 * the level's word ("not applied").
 */
export function StrengthMeter({ level, direction, showWord = false, className, muted = false, label }: {
  level: StrengthLevel
  direction?: 'for' | 'against'
  showWord?: boolean
  className?: string
  muted?: boolean
  label?: string
}) {
  const words = label ?? strengthWords(level, direction)
  const record = level === 'documentary' || level === 'self_declared'
  const bars = record ? 0 : BARS[level]
  const glyph = record
    // The seal occupies the four-bar width (4 × 6 px + 3 × 2 px) so the words line up with the bar meters.
    ? (
      <span aria-hidden data-mark={level === 'documentary' ? 'filled' : 'outline'} data-muted={muted ? 'true' : undefined}
        className={cx('inline-flex w-[30px] items-center justify-center', muted ? 'text-text-muted' : 'text-text')}>
        <svg viewBox="0 0 10 10" width="12" height="12" focusable="false">
          {level === 'documentary'
            ? <path d="M5 0.6 L9.4 5 L5 9.4 L0.6 5 Z" fill="currentColor" />
            : <path d="M5 1.3 L8.7 5 L5 8.7 L1.3 5 Z" fill="none" stroke="currentColor" strokeWidth="1" />}
        </svg>
      </span>
    )
    : (
      <span aria-hidden className="inline-flex items-end gap-0.5">
        {[1, 2, 3, 4].map((n) => (
          <span key={n} data-bar={n <= bars ? 'on' : 'off'} style={{ borderRadius: 1 }}
            className={cx('inline-block w-(--viz-strength-bar-w) h-(--viz-strength-bar-h)', n <= bars ? 'bg-strength-on' : 'bg-strength-off')} />
        ))}
      </span>
    )
  if (showWord) {
    return <span data-strength={level} className={cx('inline-flex items-center gap-2 text-small text-text', className)}>{glyph}<span>{words}</span></span>
  }
  return <span data-strength={level} role="img" aria-label={words} title={words} className={cx('inline-flex items-center', className)}>{glyph}</span>
}
