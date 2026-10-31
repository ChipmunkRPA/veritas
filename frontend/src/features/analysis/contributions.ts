/** How the estimate was reached (UX §5.6, §8.4; PRD EA-06; ARCHITECTURE §4.4 invariant 1). Pure functions: the waterfall is drawn
 *  only from contribution rows that add up to the published estimate, so the integrity gate runs before any bar exists. */
import type { SignalV2 } from '../../lib/api/queries'
import { halfUp } from '../../lib/format'
import { countedRows, estimateOf, type Decision } from './answer'
import { recordAlone, recordAloneText, recordOf, recordSetNumber, type MarkerKind } from './record'
import { LN2, littleOf, signalDirection } from './vocabulary'

export type ContributionRow = NonNullable<Decision['contributions']>[number]

/** `app/engine/fusion/calibration.py` PROB_FLOOR / PROB_CEILING: the published estimate is clamped to 2–98 while the rows add up
 *  to the unclamped log-odds. */
export const PROB_FLOOR = 0.02
export const PROB_CEILING = 0.98
/** BUILD_SPEC E-3 and PRD EA-06: contributions plus the prior equal the fused log-odds within 1e-6. */
export const TOLERANCE = 1e-6
/** `app/core/canonical.py` rounds every float of the published document to 6 dp. */
const HALF_DP = 5e-7
/** Waterfall x domain: logit(0.01) to logit(0.99). */
export const DOMAIN = Math.log(99)
export const PROB_TICKS = [1, 5, 20, 50, 80, 95, 99] as const
/** N-19k (D2-03; `forensic.LOG_ODDS_TICKS`): the axis of the breakdown of a result outside what was validated is in
 *  log-odds. It has no tick of a probability, so no estimate is read off where a bar ends. */
export const LOG_ODDS_TICKS = [-4, -2, 0, 2, 4] as const
const MEDIA_ONE: Record<string, string> = { image: 'image', video: 'video', document: 'document', audio: 'recording' }
export const CAPPED_BARS = 'Each bar below starts at zero and shows what one check returned, in log-odds.'
/** N-19k (D2-03; `forensic.capped_combined`): the one sentence over the breakdown of a result outside what was validated,
 *  in place of the sentences on what pushed the estimate and where the bars start. `bars`: rows are drawn. */
export const cappedCombined = (kind: string, bars = true, record = false, recordRow = false): string => {
  const one = MEDIA_ONE[kind] ?? 'file'
  // N-19l (R6-02): under a provenance record the result states the record's value; what is not stated is the checks' estimate
  const what = record ? 'the estimate of the statistical checks is not stated' : 'no estimate is stated'
  return `The checks were combined as for any ${one}. That combination is not validated for this kind of ${one}, so ${what}.${bars ? ` ${CAPPED_BARS}` : ''}${bars && recordRow ? ` ${RECORD_ROW_NO_AMOUNT}` : ''}`
}
/** N-19l (R6-02; `forensic.RECORD_ROW_NO_AMOUNT`): the row of the record in the breakdown of a result outside what was
 *  validated. Its amount would be the distance from the combination of the checks, which is not shown, to the record's value. */
export const RECORD_ROW_NO_AMOUNT = 'The provenance record is listed without an amount: it is not what a check returned.'
/** N-19k (D2-03; `forensic.reached_title`): the section is headed by what it shows: outside what was validated that is
 *  what the checks returned, and under a score of hand-set weights it is not an estimate. */
export const reachedTitle = (number: { kind: string }, d?: Decision | null): string => {
  // N-19n (D3-04; D-68): where a record decides alone there is no estimate and no score to reach, on both fusion paths; every
  // result of the fallback that has a score is headed by the score, with a record or without one
  if (recordAlone(d)) return 'How the result was reached'
  if (number.kind === 'capped') return 'What the checks returned'
  return number.kind === 'uncalibrated' || d?.fusion_mode === 'fallback' ? 'How the score was reached' : 'How the estimate was reached'
}
export const logit = (p: number) => Math.log(p / (1 - p))
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

function fsum(xs: readonly number[]): number {
  // Neumaier summation: rows can mix large opposite terms (context prior, overrides).
  let sum = 0
  let c = 0
  for (const x of xs) {
    const t = sum + x
    c += Math.abs(sum) >= Math.abs(x) ? (sum - t) + x : (x - t) + sum
    sum = t
  }
  return sum + c
}

export type IntegrityFailure = 'record' | 'fallback' | 'capped' | 'no_estimate' | 'no_rows' | 'mismatch'
export type Integrity =
  | { ok: true; sum: number; target: number; tolerance: number; clamped: boolean }
  | { ok: false; reason: IntegrityFailure; sum: number | null; target: number | null; tolerance: number | null }

/** The integrity gate. The tolerance is 1e-6 plus the rounding bound of the published document: 5e-7 per row, and 5e-7 on the
 *  estimate magnified by d logit / dp. An estimate at the 2 / 98 clamp passes when the rows add up to at least that far out. */
export function integrity(d: Decision): Integrity {
  const rows = d.contributions ?? []
  const p = estimateOf(d)
  // N-19h (E-19, 3; `forensic.integrity`): where a record decides alone, one override row is no breakdown of an estimate
  if (recordAlone(d)) return { ok: false, reason: 'record', sum: null, target: null, tolerance: null }
  if (d.fusion_mode === 'fallback') return { ok: false, reason: 'fallback', sum: null, target: null, tolerance: null }
  // D-49: an artefact decision other than not assessable without an estimate was capped because a check it depends on did not run
  if (p == null && d.code !== 'not_assessable') return { ok: false, reason: 'capped', sum: null, target: null, tolerance: null }
  if (p == null || p <= 0 || p >= 1) return { ok: false, reason: 'no_estimate', sum: null, target: null, tolerance: null }
  const target = logit(p)
  if (rows.length === 0) return { ok: false, reason: 'no_rows', sum: null, target, tolerance: null }
  const sum = fsum(rows.map((r) => r.logit))
  const tolerance = TOLERANCE + HALF_DP * rows.length + HALF_DP / (p * (1 - p))
  if (Math.abs(sum - target) <= tolerance) return { ok: true, sum, target, tolerance, clamped: false }
  const clamped = (p >= PROB_CEILING - HALF_DP && sum >= logit(PROB_CEILING) - tolerance)
    || (p <= PROB_FLOOR + HALF_DP && sum <= logit(PROB_FLOOR) + tolerance)
  return clamped ? { ok: true, sum, target, tolerance, clamped } : { ok: false, reason: 'mismatch', sum, target, tolerance }
}

/** The starting row (`reports/view.py` START_KINDS). D-63: since N-17 it is `neutral_start`, the model's estimate with every
 *  scoring check at its neutral point (the score at which that check's stand-alone likelihood ratio is 1), in the context of this
 *  input. A result published earlier holds `context_prior` (L-2: every check at p = 0.5) and keeps the wording it was published
 *  with. Either is a property of the fitted model, never a finding about the file. */
const START_KINDS = ['neutral_start', 'context_prior'] as const
export type StartKind = (typeof START_KINDS)[number]
export const STARTING_POINT_LABEL = 'Model starting point (before any check)'
export const NEUTRAL_START_LABEL = 'Starting point (every check at its neutral point)'
/** D-63: a check without a neutral point of its own keeps the 50 % baseline, and its row says so. */
export const FROM_FIFTY = ' (measured from 50 %)'

export function startKind(d: Decision): StartKind | null {
  const kinds = new Set((d.contributions ?? []).map((r) => r.kind as string))
  return START_KINDS.find((k) => kinds.has(k)) ?? null
}

const isStart = (r: ContributionRow) => (START_KINDS as readonly string[]).includes(r.kind)

/** The starting row out of 100, as the reports print it (`view.start_where`; N-19o, design review D5-02: the page read "8 %"
 *  where the reports and the sentence of D4-04 read "8 out of 100"). */
function startWhere(d: Decision): string {
  const p = sigmoid(fsum((d.contributions ?? []).filter(isStart).map((r) => r.logit)))
  return `${p >= 0.995 ? 'above 99' : p < 0.005 ? 'below 1' : halfUp(p * 100)} out of 100`
}

export function startingPointLine(d: Decision): string | null {
  const kind = startKind(d)
  if (kind == null) return null
  if (kind === 'context_prior') return `The bars start at ${startWhere(d)}, where the fused model stands before it reads any check (every check at 50 %). That starting point comes from fitting the model; it is not evidence about this file.`
  return `The bars start at ${startWhere(d)}: the model’s estimate for an input like this one with every check that scored at its neutral point, the score at which that check alone favours neither answer. That starting point comes from fitting the model; it is not evidence about this file.`
}

/** D-63, beside the key findings (also in Plain view; `view.starting_point_note` on the one-page reports): where the estimate
 *  starts and that the start is not evidence about the file. Null for a result without a `neutral_start` row or without an
 *  estimate that the rows explain. */
export function startingPointNote(d: Decision): string | null {
  if (d.fusion_mode !== 'artefact' || startKind(d) !== 'neutral_start' || estimateOf(d) == null) return null
  return `The estimate starts at ${startWhere(d)} with every check at its neutral point, the score at which a check alone favours neither answer. That start comes from fitting the model; it is not evidence about this file.`
}

/** D-63: the checks whose rows keep the 50 % baseline, named. */
export function fromFiftyLine(d: Decision): string | null {
  const names = (d.contributions ?? []).filter((r) => r.baseline === 'p50').map((r) => r.label)
  if (names.length === 0) return null
  const one = names.length === 1
  const listed = one ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
  return `${listed} ${one ? 'has' : 'have'} no neutral point of ${one ? 'its' : 'their'} own, so ${one ? 'its bar is' : 'their bars are'} measured from a score of 50 %.`
}

/** N-19l (code review R6-04, 1; `decisive.MARKER_LABELS`): the row of a self-declared record names the kind of that
 *  record. A result published before holds one label for all of them (`STORED_MARKER_LABEL`); it is printed by the kind of
 *  the record that decides (`view.row_label`). */
export const STORED_MARKER_LABEL = 'Generator named in unsigned metadata or a public watermark (self-declared)'
export const MARKER_LABELS: Record<MarkerKind, string> = {
  metadata: 'Generator named in unsigned metadata (self-declared)',
  watermark: 'Public AI-generator watermark message (self-declared)',
  credentials_listed_signer: 'Content Credentials name an AI generator without declaring AI generation (self-declared)',
  credentials_unlisted_signer_declared: 'Content Credentials declare AI generation; their signer is not on the trust list (self-declared)',
  credentials_unlisted_signer_named: 'Content Credentials name an AI generator; their signer is not on the trust list (self-declared)',
  metadata_declared: 'AI generation declared in unsigned metadata (self-declared)',
  metadata_declared_composite: 'A composite with AI-generated parts declared in unsigned metadata (self-declared)',
  metadata_parameters: 'AI generation parameters in unsigned metadata (self-declared)',
}
export function rowLabel(label: string, d: Decision | null | undefined): string {
  const marker = recordOf(d)?.marker
  return label === STORED_MARKER_LABEL && marker ? MARKER_LABELS[marker] : label
}

export type BarKind = 'start' | 'detector' | 'missing' | 'group' | 'adjustment' | 'estimate'

export interface Bar {
  key: string
  kind: BarKind
  label: string
  /** D-26 display id of the signal (S02), when the row belongs to one. */
  displayId: string | null
  signalId: string | null
  delta: number
  from: number
  to: number
  weight: number | null
  p: number | null
  approximate: boolean
  members: Bar[]
  /** N-19p (D6-05): a check's row that the decision publishes and does not count (`counted` false): "not counted", no bar. */
  notCounted?: boolean
}

const bar = (over: Partial<Bar> & Pick<Bar, 'key' | 'kind' | 'label' | 'delta' | 'from' | 'to'>): Bar =>
  ({ displayId: null, signalId: null, weight: null, p: null, approximate: false, members: [], ...over })

/** N-19p (design review D6-05, code review R10-05; `view.row_not_counted`): the signal of a check whose row the decision
 *  publishes and does not count (`counted` false: a row of zero, the row of a check that declares H3). Its row reads "not
 *  counted", with no bar and no "−0.00", in the breakdown, in the ledger and in the full reports' ledger. */
export const NOT_COUNTED = 'not counted'
export function rowNotCounted(signalId: string, d: Pick<Decision, 'contributions'> | null | undefined): boolean {
  return (d?.contributions ?? []).some((r) => r.kind === 'detector' && r.counted === false && r.signal_id === signalId)
}

/** N-19o (browser QA Q5-01; `view.is_mention`): the metadata check's hit in free text (a caption, a description, a comment) is
 *  a mention, no marker: it sets no number, so its ledger row points neither way, has no P(AI) and no dot in the strip. */
export const isMention = (s: Pick<SignalV2, 'detector_id' | 'provenance_override' | 'metrics'>): boolean =>
  s.detector_id === 'img.meta.generator' && s.provenance_override == null && (s.metrics?.free_text_hits ?? 0) > 0
export const MENTION_DIRECTION = 'neither way'
/** The check's labels for what it matched, in the finding's words (`synthesis._mention`): not a field the file holds. */
const MENTION_LABELS: Record<string, string> = { 'Generative-AI marker': 'a phrase such as “AI-generated”',
  'IPTC DigitalSourceType = trainedAlgorithmicMedia': 'the term trainedAlgorithmicMedia' }
export const mentionWords = (observed: string): string =>
  observed.split(';').map((l) => l.trim()).filter(Boolean).map((l) => MENTION_LABELS[l] ?? l).join('; ') || 'a term of its list of AI tools'
/** N-19p (browser QA Q6-01; `synthesis._mention_where`): the fields that hold a mention, from the check's result
 *  (`details.mentions`), joined; null for a signal stored before the check named them. */
export function mentionFields(details: unknown): string | null {
  const rows = (details as { mentions?: unknown } | null | undefined)?.mentions
  const fields = Array.isArray(rows) ? [...new Set(rows.map((r) => (r as { field?: unknown })?.field).filter((f): f is string => typeof f === 'string' && f !== ''))] : []
  return fields.length === 0 ? null : fields.length === 1 ? fields[0] : `${fields.slice(0, -1).join(', ')} and ${fields[fields.length - 1]}`
}
/** N-19p (Q6-01): what a mention means, in the place of the check's own "weak evidence only" */
export const MENTION_WHY = 'A text field of the file’s metadata mentions AI. Anyone can write such a field, and Veritas looks for words, not their meaning, so it sets no number in this result.'

/** Standalone probability of a signal: calibrated when available, else a raw score that is itself a probability. */
export function standaloneP(s: SignalV2): number | null {
  if (isMention(s)) return null  // N-19o (Q5-01): a mention sets no number
  if (s.calibrated_probability != null) return s.calibrated_probability
  return s.raw_kind === 'probability' && s.raw_score != null ? s.raw_score : null
}

/** Waterfall bars (UX §8.4): starting point, the top `top` detector rows by |Δ|, the rest as "n smaller checks (net)", then the
 *  adjustments (bounds and provenance overrides) and the estimate. Running totals are exact sums of the recorded rows. */
export function waterfall(d: Decision, signals: readonly SignalV2[], top = 6): Bar[] {
  const rows = d.contributions ?? []
  const byId = new Map(signals.map((s) => [s.id, s]))
  const start = fsum(rows.filter(isStart).map((r) => r.logit))
  const checks = rows.filter((r) => r.kind === 'detector' || r.kind === 'missing')
    .map((r, i) => ({ r, i }))
    .sort((a, b) => Math.abs(b.r.logit) - Math.abs(a.r.logit) || a.i - b.i)
    .map(({ r }) => r)
  const adjustments = rows.filter((r) => r.kind === 'override')
  const shown = checks.length > top + 1 ? checks.slice(0, top) : checks
  const rest = checks.slice(shown.length)

  const out: Bar[] = [bar({ key: 'start', kind: 'start', label: startKind(d) === 'neutral_start' ? NEUTRAL_START_LABEL : STARTING_POINT_LABEL, delta: start, from: start, to: start })]
  let running = start
  const detectorBar = (r: ContributionRow, key: string): Bar => {
    const s = r.signal_id ? byId.get(r.signal_id) : undefined
    const b = bar({
      key, kind: r.kind === 'missing' ? 'missing' : 'detector', label: (s?.name ?? r.label) + (r.baseline === 'p50' ? FROM_FIFTY : ''), displayId: s?.display_id ?? null,
      signalId: r.signal_id ?? null, delta: r.logit, from: running, to: running + r.logit, weight: s?.effective_weight ?? null,
      p: s ? standaloneP(s) : null, approximate: r.approximate ?? false, notCounted: r.kind === 'detector' && r.counted === false,
    })
    running = b.to
    return b
  }
  shown.forEach((r, i) => out.push(detectorBar(r, `check-${i}`)))
  if (rest.length > 0) {
    const from = running
    const members = rest.map((r, i) => detectorBar(r, `rest-${i}`))
    out.push(bar({ key: 'rest', kind: 'group', label: `${rest.length} smaller checks (net)`, delta: running - from, from, to: running, members }))
  }
  adjustments.forEach((r, i) => {
    const s = r.signal_id ? byId.get(r.signal_id) : undefined
    const b = bar({ key: `adjust-${i}`, kind: 'adjustment', label: rowLabel(r.label, d), displayId: s?.display_id ?? null, signalId: r.signal_id ?? null,
      delta: r.logit, from: running, to: running + r.logit, approximate: r.approximate ?? false })
    running = b.to
    out.push(b)
  })
  out.push(bar({ key: 'estimate', kind: 'estimate', label: 'Estimate', delta: running, from: running, to: running }))
  return out
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`
const WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve']
const spelled = (n: number) => (n < WORDS.length ? WORDS[n] : String(n))
/** A count as the subject of a clause, as the reports say it (`forensic._counted`): none, one, two … */
const counted = (n: number, capital = false) => {
  const word = n === 0 ? 'none' : spelled(n).toLowerCase()
  return capital ? word.charAt(0).toUpperCase() + word.slice(1) : word
}

/** N-19i (`view.not_counted_lines`, word for word): what the narrative says of a row that is published and not counted: a row
 *  of zero, and the row of a check that declares H3. */
export function notCountedLines(d: Decision): string[] {
  const rows = (d.contributions ?? []).filter((r) => r.kind === 'detector')
  const zero = rows.filter((r) => Math.abs(r.logit) <= 1e-9).length
  const record = rows.filter((r) => r.counted === false && Math.abs(r.logit) > 1e-9).length
  const out: string[] = []
  if (zero) out.push(zero === 1 ? 'One more row is shown and not counted: it is zero, so its check did not move the estimate.'
    : `${zero} more rows are shown and not counted: they are zero, so their checks did not move the estimate.`)
  if (record) out.push(record === 1 ? 'One more row is shown and not counted: its check tests the file’s provenance and integrity, and is not one of the statistical checks.'
    : `${record} more rows are shown and not counted: their checks test the file’s provenance and integrity, and are not statistical checks.`)
  return out
}

/** UX §5.6 narrative sentence, built from counts only (§7.3 rule 11). N-19i (R5-01, 4): it counts the checks the decision counts
 *  (`countedRows`: what the number line counts), and its three counts are of those checks; it counted every row of a check. */
export function narrative(d: Decision): string[] {
  const rows = d.contributions ?? []
  const detectors = countedRows(d)
  const floor = littleOf(d)  // D-63: ln 2; 0.05 for a result published before N-17
  const up = detectors.filter((r) => r.logit >= floor).length
  const down = detectors.filter((r) => r.logit <= -floor).length
  const little = detectors.length - up - down
  const missing = rows.filter((r) => r.kind === 'missing').length
  const adjustments = rows.filter((r) => r.kind === 'override')
  const out = [
    `${spelled(detectors.length)} ${detectors.length === 1 ? 'check' : 'checks'} entered the fusion model. Veritas combines them by adding each check’s evidence on a log-odds scale, with weights fitted on the benchmark's training data, not on its test data.`,
    `${counted(up, true)} pushed the estimate up, ${counted(down)} pushed it down, ${counted(little)} had little effect.`,
    ...notCountedLines(d),
  ]
  const start = startingPointLine(d)
  if (start) out.push(start)
  const fifty = fromFiftyLine(d)
  if (fifty) out.push(fifty)
  if (missing) out.push(`${plural(missing, 'check')} produced no score and ${missing === 1 ? 'adds its' : 'add their'} fitted “missing” term instead.`)
  if (adjustments.length) out.push(`The result was then adjusted: ${adjustments.map((r) => rowLabel(r.label, d)).join('; ')}.`)
  return out
}

export interface Dot { id: string; displayId: string; p: number; direction: 'ai' | 'authentic' | 'neutral'; row: number }

/** Agreement strip (UX §8.4): one dot per scored check at its standalone p; its direction is the ledger's for the check
 *  (`vocabulary.signalDirection`; design review DS-17: never the side of 50 it falls on). Labels climb up to 3 rows to avoid
 *  collisions (greedy, in order of p). */
export function agreementDots(signals: readonly SignalV2[], widthPx: number, little: number = LN2): Dot[] {
  const scored = signals
    .filter((s) => s.status === 'ok' && s.hypothesis != null && s.provenance_override == null)
    .map((s) => ({ s, p: standaloneP(s) }))
    .filter((x): x is { s: SignalV2; p: number } => x.p != null)
    .sort((a, b) => a.p - b.p)
  const lastX: number[] = [-Infinity, -Infinity, -Infinity]
  const labelPx = 30
  return scored.map(({ s, p }) => {
    const direction = signalDirection(s, little)
    const x = p * widthPx
    let row = lastX.findIndex((last) => x - last >= labelPx)
    if (row < 0) row = lastX.indexOf(Math.min(...lastX))
    lastX[row] = x
    return { id: s.id, displayId: s.display_id ?? s.id, p, direction, row }
  })
}

/** Position of a log-odds value on the waterfall axis, as a fraction 0–1, clamped to the domain with an overflow flag. */
export function axisPosition(x: number): { at: number; overflow: -1 | 0 | 1 } {
  const overflow = x > DOMAIN ? 1 : x < -DOMAIN ? -1 : 0
  const clamped = Math.max(-DOMAIN, Math.min(DOMAIN, x))
  return { at: (clamped + DOMAIN) / (2 * DOMAIN), overflow }
}

export const signed = (x: number, dp = 2) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(dp)}`

/** The explained fallback (UX §5.6 integrity gate): why no chart is drawn. */
/** N-19k (Q2-14; `view.no_model`): a fusion model is installed and this analysis did not use it; the result's limiting
 *  condition says so (`runtime.FALLBACK_ENVELOPE_UNUSED`). "No fusion model is loaded" is said only where none is. */
export const MODEL_UNUSED = 'no calibrated fusion model was used for this analysis'
export const modelUnused = (profile?: { limiting_conditions?: readonly string[] | null } | null): boolean =>
  (profile?.limiting_conditions ?? []).some((c) => c.startsWith(MODEL_UNUSED))

export function fallbackText(check: Extract<Integrity, { ok: false }>, d?: Decision | null, why: string | null = null,
  profile?: { limiting_conditions?: readonly string[] | null } | null): string {
  switch (check.reason) {
    case 'record':  // N-19h (E-19, 3): there is no statistical estimate to break down; `why`: `aloneWhy`
      return recordAloneText(d ?? {}, why)
    case 'fallback': {
      // N-19h (E-20, 1; `forensic.fallback_text`): a floor or a cap does not come from hand-set weights
      const set = recordSetNumber(d)
      return `No calibrated fusion model ${modelUnused(profile) ? 'was used for this analysis' : 'is loaded'}, so the ${set ? 'statistical score' : 'score'} came from the uncalibrated fallback with hand-set weights. That path records no additive contributions, and Veritas does not draw a breakdown that does not add up. Each check’s weight is listed in the evidence ledger.`
    }
    case 'capped':
      return 'There is no estimate to break down: a check that this analysis depends on did not run.'
    case 'no_estimate':
      return 'There is no estimate to break down: the file could not be assessed.'
    case 'no_rows':
      return 'The fusion model recorded no contribution rows for this analysis.'
    default:
      return `The recorded contributions add up to ${signed(check.sum ?? 0, 4)} log-odds, but the estimate corresponds to ${signed(check.target ?? 0, 4)}. The mismatch is recorded in the technical appendix.`
  }
}
