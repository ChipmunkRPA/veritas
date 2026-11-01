/** Public check evaluation facts. Free-form API records are narrowed field by
 * field; absent or malformed metrics never become apparent measured results. */
import type { ModelCard } from '../../lib/api/queries'

export const NOT_ESTABLISHED = 'not established'
export const ADMISSION_WORD: Record<string, string> = { pending: 'pending', admitted: 'admitted', rejected: 'rejected', retired: 'retired' }
export interface PublicAdmission { decision: string | null; date: string | null }
const dateOnly = (value: unknown): string | null => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : null

export function admissionRecord(card: ModelCard | undefined): PublicAdmission | null {
  const value = card?.admission
  if (!value) return null
  const decision = typeof value.decision === 'string' && Object.prototype.hasOwnProperty.call(ADMISSION_WORD, value.decision) ? value.decision : null
  return { decision, date: dateOnly(value.date) }
}
export function admissionWord(card: ModelCard): string {
  const record = admissionRecord(card)
  if (!record?.decision) return 'not recorded'
  return [ADMISSION_WORD[record.decision], record.date].filter(Boolean).join(' · ')
}

export interface EvalRow {
  slice: string | null; auc: number | null; tpr_at_fpr01: number | null; ece: number | null; n: number | null
  date: string | null; benchmark_recorded: boolean
}
const metric = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null
export function measuredRows(card: ModelCard): EvalRow[] {
  return (card.eval ?? []).map((row) => ({
    slice: typeof row.slice === 'string' ? row.slice : null,
    auc: metric(row.auc), tpr_at_fpr01: metric(row.tpr_at_fpr01), ece: metric(row.ece),
    n: typeof row.n === 'number' && Number.isSafeInteger(row.n) && row.n >= 0 ? row.n : null,
    date: dateOnly(row.date), benchmark_recorded: typeof row.bench === 'string' && row.bench.trim().length > 0,
  })).filter((row) => row.auc != null || row.tpr_at_fpr01 != null || row.ece != null)
}
export const fmtMetric = (value: number | null | undefined, digits = 3): string =>
  value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)
export function withCi(value: number | null | undefined, interval: readonly (number | null | undefined)[] | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return NOT_ESTABLISHED
  return interval && interval.length === 2 && interval.every((bound) => bound != null && Number.isFinite(bound))
    ? `${fmtMetric(value, digits)} [${fmtMetric(interval[0], digits)}, ${fmtMetric(interval[1], digits)}]` : fmtMetric(value, digits)
}
