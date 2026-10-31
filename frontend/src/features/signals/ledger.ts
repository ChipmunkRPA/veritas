/** Evidence ledger v2 (UX §5.7; ARCHITECTURE §4.2). Pure derivations from the machine layer: direction, strength, probability,
 *  lens family, the not-run group and the "In testing" line. Nothing here invents a value the analysis does not carry. */
import type { AnalysisDetail, DetectorRun, ExhibitMeta, Finding, SignalV2 } from '../../lib/api/queries'
import type { StrengthLevel } from '../../components/ui'
import { frequency, isDocumentary, markOf, media, type Decision } from '../analysis/answer'
import { standaloneP } from '../analysis/contributions'
import { LN10, LN100, LN2, entered, signalDirection, strength as effectStrength } from '../analysis/vocabulary'

type Calibration = NonNullable<AnalysisDetail['calibration']>[number]
export type LedgerDirection = 'ai' | 'authentic' | 'neutral'

/** Contribution column full scale (UX §5.7): ±1.5 log-odds, clamped with an overflow glyph. */
export const CONTRIBUTION_SCALE = 1.5

export { standaloneP }

export { signalDirection }

const LEGACY_STRENGTH: Record<string, StrengthLevel> = { strong: 'strong', moderate: 'moderate', weak: 'weak', none: 'none' }

const FINDING_STRENGTH: Record<Finding['strength'], StrengthLevel> = { none: 'none', weak: 'weak', moderate: 'moderate', strong: 'strong', decisive: 'documentary' }
const NOT_KEY = new Set(['coverage', 'F9_distribution_conditions'])

/** The finding synthesis built from this signal: one per signal, citing the signal's own id first (EA-07; the report's `view.own_finding`). */
export function ownFinding(s: SignalV2, findings: readonly Finding[]): Finding | undefined {
  return findings.find((f) => !NOT_KEY.has(f.family) && (f.evidence_ids ?? [])[0] === s.id)
}

/** Strength, one per signal on every surface (L-4, design P1-7): the strength of the signal's own finding (family caps and
 *  corroboration applied, ◆ when documentary); without one, ◆ for documentary provenance overrides, the ln-LR bands of IF §6.2
 *  (weak ≥ ln 2, moderate ≥ ln 10, strong ≥ ln 100) when a calibrated LR exists, else the v1 mirror, labelled uncalibrated. */
export function signalStrength(s: SignalV2, findings: readonly Finding[] = [], measured = false, decision?: Decision | null):
  { level: StrengthLevel; calibrated: boolean; outside?: boolean; muted?: boolean; word?: string } {
  // N-19n (D3-07): the signal of a self-declared marker has a record's mark, as its finding has: no marker is "strong"
  const mark = s.provenance_override === 'self_declared_ai' ? ownMark(s, findings, decision) : null
  if (mark) return { level: mark.level, calibrated: true, muted: mark.muted, word: mark.word }
  // D-63: in a result measured from neutral points, a check that entered the estimate takes the strength of its contribution;
  // a scored check outside the estimate keeps its stand-alone strength and is said to be outside
  if (measured && s.provenance_override == null && entered(s.contribution_logit)) return { level: effectStrength(s.contribution_logit), calibrated: true }
  const outside = measured && s.provenance_override == null && s.hypothesis != null && standaloneP(s) != null
  const standalone = standaloneStrength(s, findings)
  return outside ? { ...standalone, outside: true } : standalone
}

function ownMark(s: SignalV2, findings: readonly Finding[], decision?: Decision | null) {
  const own = ownFinding(s, findings)
  return own ? markOf(own, decision) : null
}

function standaloneStrength(s: SignalV2, findings: readonly Finding[]): { level: StrengthLevel; calibrated: boolean } {
  const own = ownFinding(s, findings)
  if (own) return { level: isDocumentary(own) ? 'documentary' : FINDING_STRENGTH[own.strength], calibrated: s.log_lr != null || !!s.provenance_override }
  if (s.provenance_override === 'decisive_ai' || s.provenance_override === 'trusted_capture') return { level: 'documentary', calibrated: true }
  if (s.log_lr != null) {
    const r = Math.abs(s.log_lr)
    return { level: r >= LN100 ? 'strong' : r >= LN10 ? 'moderate' : r >= LN2 ? 'weak' : 'none', calibrated: true }
  }
  return { level: LEGACY_STRENGTH[s.legacy?.strength ?? 'none'] ?? 'none', calibrated: false }
}

/** UX §5.7 lens family tag. */
export function lensFamily(s: SignalV2, exhibits: readonly ExhibitMeta[]): string {
  const mine = exhibits.filter((x) => x.created_by === s.detector_id || (x.source_signal_ids ?? []).includes(s.id) || (s.exhibit_ids ?? []).includes(x.id))
  const family = `${s.family} ${s.category}`.toLowerCase()
  if (s.hypothesis === 'H3_provenance_integrity' || /provenance|metadata/.test(family)) return 'Provenance'
  if (/codec|temporal|lipsync|lip-sync|voice/.test(family) || (s.segments ?? []).length > 0) return 'Temporal'
  if (mine.some((x) => x.semantics === 'p_manipulated') || /locali/.test(family)) return 'Localisation'
  if (mine.some((x) => x.semantics === 'attribution')) return 'Attention'
  if (mine.some((x) => x.semantics === 'residual' || x.semantics === 'error_level') || /forensic|filter/.test(family)) return 'Filter'
  return 'Classifier'
}

export interface NotRunRow { key: string; displayId: string | null; signalId: string | null; name: string; status: string; reason: string }

const STATUS_WORD: Record<string, string> = { not_applicable: 'not applicable', skipped: 'skipped', failed: 'failed', timeout: 'timed out', cancelled: 'cancelled' }
export const statusWord = (status: string) => STATUS_WORD[status] ?? status.replace(/_/g, ' ')

/** Checks that did not run (UX §5.7): signals with a non-ok status, and detector runs that produced no signal at all. */
export function notRunRows(signals: readonly SignalV2[], runs: readonly DetectorRun[]): NotRunRow[] {
  const out: NotRunRow[] = signals.filter((s) => s.status !== 'ok').map((s) => ({
    key: s.id, displayId: s.display_id ?? null, signalId: s.id, name: s.name, status: s.status ?? 'skipped', reason: s.status_reason || statusWord(s.status ?? 'skipped'),
  }))
  const covered = new Set(signals.map((s) => s.detector_id))
  for (const r of runs) {
    if (r.status === 'ok' || covered.has(r.detector_id)) continue
    covered.add(r.detector_id)
    out.push({ key: `run-${r.id}`, displayId: null, signalId: null, name: r.detector_id, status: r.status, reason: r.reason || statusWord(r.status) })
  }
  return out
}

/** Centred-zero contribution bar: half-width fraction 0–1 on the signal's side, with overflow beyond ±1.5 log-odds. */
export function contributionBar(logit: number | null | undefined): { half: number; side: 'left' | 'right' | null; overflow: boolean } {
  if (logit == null || !Number.isFinite(logit) || logit === 0) return { half: 0, side: null, overflow: false }
  return { half: Math.min(Math.abs(logit), CONTRIBUTION_SCALE) / CONTRIBUTION_SCALE, side: logit > 0 ? 'right' : 'left', overflow: Math.abs(logit) > CONTRIBUTION_SCALE }
}

/** "In testing" (UX §5.7, §7.4): the detector's own calibration record at its 5 % conformal (else fixed) threshold, per context.
 *  With fewer than 100 test items it says "k of n". Without a record it says so; it never borrows another detector's numbers. */
export function testingLines(s: SignalV2, calibration: readonly Calibration[], kind: string, benchmarkRows = 0): string[] {
  const subject = `${s.detector_id}@${s.detector_version}`
  const record = calibration.find((c) => c.subject === subject)
  const rows = (record?.error_rates ?? []).filter((r) => r.threshold_kind === 'conformal_0.05' || r.threshold_kind === 'fixed')
  // N-19d (browser QA "Copy"): the line said "No evaluation on record" for a check whose model card lists benchmark results
  // and whose Methods row counts them. `benchmarkRows` is that count (the measured rows of the card).
  if (rows.length === 0) {
    return [benchmarkRows > 0
      ? `No error rate at a flagging threshold is on record for this check. Its model card lists ${benchmarkRows} benchmark ${benchmarkRows === 1 ? 'result' : 'results'}.`
      : 'No evaluation on record for this detector.']
  }
  const [, many] = media(kind)
  return rows.map((r) =>
    `${r.context}: flagged ${frequency(r.fpr, r.n_neg)} genuine ${many} and found ${frequency(r.tpr, r.n_pos)} AI-generated or manipulated ${many} (${r.bench_version}, ${r.threshold_kind === 'fixed' ? 'fixed threshold' : '5 % conformal threshold'}).`)
}

/** Effect used to order the ledger: the recorded contribution, else the distance of p from 50 times the hand-set weight. */
export function effect(s: SignalV2): number {
  if (s.contribution_logit != null) return Math.abs(s.contribution_logit)
  const p = standaloneP(s)
  return p == null ? -1 : Math.abs(p - 0.5) * (s.effective_weight ?? 0)
}

/** Anchor id of a ledger row. Key findings link to `#signal-<v1 id or v2 id>` (E-2), so both analyses and v1 upgrades resolve. */
export const anchorOf = (s: Pick<SignalV2, 'id' | 'legacy'>) => `signal-${s.legacy?.v1_id || s.id}`
