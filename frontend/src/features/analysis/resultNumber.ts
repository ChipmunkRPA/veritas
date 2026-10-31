/** The number a result shows (D-68; N-7). One decision, read by the answer module, the likelihood scale, the sign-off dialog and
 *  the analyses list, so a score that is not a calibrated likelihood never reads as one on one surface and not on another. */
import type { AnalysisDetail, Finding } from '../../lib/api/queries'
import { halfUp } from '../../lib/format'
import { ANSWERS_PART, basisOf, countedRows, estimateOf, fewChecks, interval, numeral, offScale, offScalePart, ruleTaken, scoredChecks, unsupported, type Decision, type DecisionCode } from './answer'
import { OTHER_WAY, OTHER_WAY_ONE, RECORD_SET, STATISTICAL_WITHHELD, aloneInconclusive, aloneOtherWay, aloneNumber, aloneUndecided, outsideValidated, recordOf, recordSetNumber, statisticalWithheld } from './record'

type ConditionProfile = NonNullable<AnalysisDetail['condition_profile']>
type Range = readonly [number, number]

export type ResultNumber =
  /** A decisive calibrated result in which at least two checks scored: the large numeral and the scale marker. */
  | { kind: 'calibrated'; estimate: number; range: Range | null }
  /** A calibrated result that is inconclusive, or in which fewer than two checks scored: the estimate as body text.
   *  N-19l (R6-05; D-68 as amended): also the statistical number under a record that decides and did not set it, where the
   *  statistical checks alone are inconclusive: "Estimate of the statistical checks alone 83–96: they alone do not decide". */
  | { kind: 'text'; estimate: number; range: Range | null; text: string }
  /** The uncalibrated fallback: "score n / 100, not calibrated", no scale marker. */
  | { kind: 'uncalibrated'; score: number; text: string }
  /** Outside what was validated: no number and no scale marker. The reason is in the decision (`capReason`) and the methods. */
  | { kind: 'capped' }
  /** N-19h (E-20): a number that the provenance record set (a marker's floor, a capture's cap): said in words with what set
   *  it, never as an estimated likelihood, a score of the fallback or a marker on the scale. The statistical checks' own
   *  number stands beside it (`statisticalAlone`). */
  | { kind: 'record'; value: number; text: string }
  /** No estimate, a result that is not assessable, or a result that documentary evidence decides. */
  | { kind: 'none' }

const NONE: ResultNumber = { kind: 'none' }
const H3 = 'H3_provenance_integrity'

/** How an uncalibrated score is printed wherever it appears (`schemas/legacy.py::uncalibrated_score`). */
export const uncalibratedScore = (p: number) => `score ${halfUp(p * 100)} / 100, not calibrated`

/** "Plausible range 53–78: inconclusive": the range when there is one, named as the range (N-19q, finding 11: "Estimate 24–85"
 *  stood beside "49 out of 100"; `view.RANGE_WORD`), else the estimate. */
export const RANGE_WORD = 'Plausible range'
const estimateText = (p: number, range: Range | null, why: string) =>
  range ? `${RANGE_WORD} ${range[0]}–${range[1]}: ${why}` : `Estimate ${numeral(p)}: ${why}`

export function resultNumber(d: Decision | null | undefined, findings: readonly Finding[] = [], profile?: ConditionProfile | null): ResultNumber {
  if (!d || d.code === 'not_assessable') return NONE
  if (basisOf(d, findings) === 'documentary_declared') return NONE
  const p = estimateOf(d)
  if (p == null) return NONE
  // N-19h (E-20, 1): the record's value, on the fallback as on the calibrated path (`view.result_number`)
  if (recordSetNumber(d)) return { kind: 'record', value: p, text: recordPhrase(d, p) }
  // The fallback has no validated envelope at all (`runtime.fallback_profile`), so only what its reason says counts (`outsideValidated`)
  if (outsideValidated(d, profile)) return { kind: 'capped' }
  if (d.fusion_mode === 'fallback') return { kind: 'uncalibrated', score: p, text: uncalibratedScore(p) }
  const range = interval(d)
  // N-19l (R6-05): as the result of the statistical checks alone would show it, in words of its own beside a record
  if (aloneUndecided(d)) return { kind: 'text', estimate: p, range, text: `${ALONE_ESTIMATE} ${range ? `${range[0]}–${range[1]}` : numeral(p)}: ${ALONE_UNDECIDED}` }
  if (d.code === 'inconclusive') return { kind: 'text', estimate: p, range, text: estimateText(p, range, 'inconclusive') }
  const scored = scoredChecks(d, findings)
  if (scored < 2) return { kind: 'text', estimate: p, range, text: estimateText(p, range, fewChecks(d, scored)) }
  return { kind: 'calibrated', estimate: p, range }
}

/** N-19l (R6-05; `view.ALONE_ESTIMATE`, `view.ALONE_UNDECIDED`): the statistical number under a record that decides and did
 *  not set it, where the statistical checks alone do not decide. The result is the record's and is not inconclusive. */
export const ALONE_ESTIMATE = 'Estimate of the statistical checks alone'
export const ALONE_UNDECIDED = 'they alone do not decide'

/** A number that a record set, beside a headline (`view.record_phrase`): "set by the record: 85 out of 100, the floor of
 *  self-declared evidence". Not an estimated likelihood and not a score of the fallback. */
export function recordPhrase(d: Decision, p: number): string {
  const kind = recordOf(d)?.kind
  const what = (kind && RECORD_SET[kind]) ?? 'the value of the provenance record'
  return `set by the record: ${numeral(p)} out of 100, ${what}`
}

/** N-19n (D4-03; `view.OWN_LIKELIHOOD`): under a record that did not set the number, the calibrated numeral shown is the
 *  statistical checks' own, and its caption and lines say so ("estimated likelihood" stood under a capture's headline). */
export const OWN_LIKELIHOOD = 'estimated likelihood, statistical checks alone'

/** The number in a sentence beside a headline (the sign-off dialog's machine assessment), or null when the result shows none.
 *  `own` (N-19n, D4-03): a record decides and did not set the number, so the calibrated numeral is the checks' own. */
export function numberPhrase(n: ResultNumber, own = false): string | null {
  switch (n.kind) {
    case 'calibrated': return own ? `${OWN_LIKELIHOOD}: ${numeral(n.estimate)} %` : `estimated likelihood ${numeral(n.estimate)} %`
    case 'text': return n.text
    case 'uncalibrated': return n.text
    case 'record': return n.text
    case 'capped': return 'no estimate shown: outside what was validated'
    default: return null
  }
}

/** The statistical checks beside a documentary result or beside a number that a record set: a calibrated estimate, the
 *  fallback's score said as one, or that there is none (`view.statistical_alone`). */
export function statisticalAlone(d: Decision, profile?: ConditionProfile | null): string {
  const p = d.statistical_estimate_before_provenance
  if (p == null) return 'no estimate'
  if (statisticalWithheld(d, profile)) return STATISTICAL_WITHHELD  // N-19h (E-20, 8)
  const otherWay = aloneOtherWay(d)  // N-19n (D4-02): "…, they point the other way"
  if (d.fusion_mode === 'fallback') return otherWay ? `${uncalibratedScore(p)}; ${OTHER_WAY_ONE}` : uncalibratedScore(p)
  // N-19m (R7-01; D3-05): the number with its unit, as the record's value beside it has one ("85 out of 100" stood over
  // "13"), and with what the checks alone decided where they do not decide ("Statistical checks alone: 90" stood bare)
  if (aloneInconclusive(d)) return `${aloneNumber(d)} out of 100, ${ALONE_UNDECIDED}`
  // N-19n: nor bare where one check alone counts and decides ("98 out of 100, only one check scored"; "…; it points the other way")
  if (aloneOneCheck(d)) return `${numeral(p)} out of 100, ${ONE_CHECK}${otherWay ? `; ${OTHER_WAY_ONE}` : ''}`
  return `${numeral(p)} out of 100${otherWay ? `, ${OTHER_WAY}` : ''}`
}

export const ONE_CHECK = 'only one check counts in the estimate'  // N-19q (finding 5): `answer.ts` `fewChecks`
/** N-19n (the code reviewer's read of N-19m's draft; `view.alone_one_check`): beside a record, the statistical checks alone
 *  decide and one check counts. The same picture without the record reads "Estimate 98–98: only one check scored"; beside a
 *  capture it read the bare number, and N-19m took the fact "Agreement — only one check scored" away there. False where they
 *  alone do not decide, on the fallback, and for a result stored before N-19l, which does not say what they decide. */
export function aloneOneCheck(d: Decision | null | undefined): boolean {
  const alone = d?.statistical_alone?.code
  return alone != null && alone !== 'inconclusive' && d?.fusion_mode === 'artefact' && countedRows(d).length === 1
}

/** N-19n (`TechnicalAppendix.tsx`): the line of the technical appendix on the statistical number that a result holds beside
 *  its record. It gives the unit ("98 out of 100."; it read "98."), and on the fallback the number is a score ("Statistical
 *  score before provenance: 30 / 100, not calibrated."; it read "Statistical estimate before provenance: 30."). Where the
 *  checks alone do not decide it says so (N-19m), and outside what was validated that the number is not shown (N-19h). */
export function appendixStatistical(d: Decision, profile?: ConditionProfile | null): string | null {
  const p = d.statistical_estimate_before_provenance
  if (p == null) return null
  if (statisticalWithheld(d, profile)) return `Statistical estimate before provenance: ${STATISTICAL_WITHHELD}.`
  if (d.fusion_mode === 'fallback') return `Statistical score before provenance: ${halfUp(p * 100)} / 100, not calibrated.`
  const undecided = recordOf(d) && aloneInconclusive(d) ? `, ${ALONE_UNDECIDED}` : ''
  return `Statistical estimate before provenance: ${Math.round(p * 100)} out of 100${undecided}.`
}

export interface ListRow {
  status: string
  /** The decision code; for a record from before the decision codes, the code its stored band maps to. */
  decision_code?: DecisionCode | null
  driving_hypothesis?: string | null
  fusion_mode?: string | null
  probability_ai?: number | null
}

export type ListNumber = { kind: 'calibrated'; estimate: number } | { kind: 'uncalibrated'; score: number; text: string } | { kind: 'none'; why: string | null }

/** What the list says in place of a number for an inconclusive row. */
export const INCONCLUSIVE_ROW = 'Inconclusive: the result page shows the number, where the result has one'

/** The same rules for a list row, which holds the decision code, the fusion mode and the number but neither the validated
 *  envelope nor the findings. What a row cannot tell apart it does not print: an inconclusive row may be a capped one, and a
 *  row decided by a provenance record may show the documentary seal on its page. The sentence of an inconclusive row promises
 *  nothing the page may not hold (N-10b; code re-verification R2-08): a capped result shows no estimate there, and a fallback
 *  result a score without a range. */
export function listNumber(row: ListRow): ListNumber {
  const p = row.probability_ai
  if (row.status !== 'done' || !row.decision_code || row.decision_code === 'not_assessable') return { kind: 'none', why: null }
  if (p == null || !Number.isFinite(p) || p < 0 || p > 1) return { kind: 'none', why: null }
  if (row.driving_hypothesis === H3) return { kind: 'none', why: 'Decided by a provenance record, not by a statistical estimate' }
  if (row.decision_code === 'inconclusive') return { kind: 'none', why: INCONCLUSIVE_ROW }
  if (row.fusion_mode !== 'artefact') return { kind: 'uncalibrated', score: p, text: uncalibratedScore(p) }
  return { kind: 'calibrated', estimate: p }
}

/** D-63 "Headline rule in words" (browser QA P1-2; `view.scale_note`): under a result of a calibrated fusion model the scale
 *  says what its marker is and what the headline follows, in place of the labels that name its segments. Null on the fallback,
 *  whose headline does follow the segments, and where no estimate was formed. */
/** N-19n (the design reviewer's read of the sentences of N-19m's third part): the note read "…, not the segment, and a headline
 *  of one answer also needs the marker in its part of the scale.", which reads for a moment as a list of what the headline does
 *  not follow, and "its part" has three possible owners. */
export const SCALE_NOTE_MARKER = 'The marker is the estimate. The headline follows the calibrated levels for inputs like this one, not the segment; a headline of one answer also needs the marker in that answer’s part of the scale.'
export const SCALE_NOTE = 'The headline follows the calibrated levels for inputs like this one, not the segments of this scale; a headline of one answer also needs the estimate in that answer’s part of the scale.'
/** N-19n (code review R8-02; `view.SCALE_NOTE_MARKER_BEFORE`): the notes as they were before D-77, without the condition. They
 *  stand under a stored result whose headline of one answer has its marker outside that answer's part of the scale: a clean
 *  headline at 42 read "… also needs the marker in its part of the scale." under a marker that is not there. */
export const SCALE_NOTE_MARKER_BEFORE = 'The marker is the estimate; the headline follows the calibrated levels for inputs like this one, not the segment.'
export const SCALE_NOTE_BEFORE = 'The headline follows the calibrated levels for inputs like this one, not the segments of this scale.'
const AI_CODES: readonly string[] = ['ai_supported', 'ai_supported_strong']
const CLEAN_CODES: readonly string[] = ['no_ai_indicators', 'no_ai_indicators_strong']
/** D-77 (`abstain.in_part_of_scale`): whether the estimate as published stands in the part of the scale of the code's answer,
 *  34 or less for a code of no indications and 65 or more for an AI code, in the whole points that the result prints. True
 *  for a code that is not decisive. */
export function inPartOfScale(code: string | null | undefined, estimate: number): boolean {
  const points = halfUp(Math.round(estimate * 1e6) / 1e4)
  if (code != null && AI_CODES.includes(code)) return points >= 65
  if (code != null && CLEAN_CODES.includes(code)) return points <= 34
  return true
}
/** N-19i (D-76; `view.SCALE_NOTE_UNSUPPORTED`): under a result that is inconclusive because no check supports the answer of
 *  its estimate, the headline does not follow the levels: the estimate reached one. */
export const SCALE_NOTE_UNSUPPORTED = "The estimate reaches a calibrated level for inputs like this one, but no check's part in it gives that answer weak support, so the headline is inconclusive."  // N-19q (finding 1)
/** N-19m (D-76 as amended; `view.SCALE_NOTE_TOGETHER`): a check supports the answer, and the checks together do not. */
export const SCALE_NOTE_TOGETHER = 'The estimate reaches a calibrated level for inputs like this one, but the checks together do not support that answer, so the headline is inconclusive.'
/** N-19m (D-77; `view.SCALE_NOTE_OFF_SCALE`): the estimate stands in the middle part of the scale. The note names the answer of
 *  the levels and does not quote the headline that the result does not have. */
export const scaleNoteOffScale = (side: 'for' | 'against') =>
  `The estimate stands in the middle part of the scale, so the headline is inconclusive, although the calibrated levels for inputs like this one would have given ${side === 'for' ? 'the AI answer' : 'the clean answer'}.`
/** N-19n (supervisor check 29; `view.SCALE_NOTE_OTHER_PART`): under a configured prior the estimate of such a result can stand
 *  in the part of the other answer; the note names the part in which it prints. */
export const scaleNoteOtherPart = (part: 'for' | 'against', side: 'for' | 'against') =>
  `The estimate stands in ${ANSWERS_PART[part]}, although the calibrated levels for inputs like this one would have given ${side === 'for' ? 'the AI answer' : 'the clean answer'}, so the headline is inconclusive.`
export function scaleNote(d: Decision, kind: ResultNumber['kind']): string | null {
  // N-19h (E-20): not in a result that a provenance record decides, whose headline is the record's (`view.scale_note`)
  // N-19n (D3-10): nor under a result that shows no estimate because its input is outside what was validated: there are
  // no calibrated levels for such an input, and "the headline follows the calibrated levels for inputs like this one" stood there
  if (d.fusion_mode !== 'artefact' || kind === 'none' || kind === 'capped' || recordOf(d)) return null
  const side = offScale(d)
  const part = offScalePart(d)
  if (side !== null) return part !== null ? scaleNoteOtherPart(part, side) : scaleNoteOffScale(side)
  if (unsupported(d) !== null) return ruleTaken(d)?.rule === 'no_check' ? SCALE_NOTE_UNSUPPORTED : SCALE_NOTE_TOGETHER
  const estimate = estimateOf(d)
  if (estimate != null && !inPartOfScale(d.code, estimate)) return kind === 'calibrated' ? SCALE_NOTE_MARKER_BEFORE : SCALE_NOTE_BEFORE  // made before D-77 (R8-02)
  return kind === 'calibrated' ? SCALE_NOTE_MARKER : SCALE_NOTE
}
