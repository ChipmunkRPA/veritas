/** Examiner layer rules for the review panel (PRD RW-01…RW-04; UX §5.13, §5.14; ARCHITECTURE §5.4). Pure functions that mirror
 *  `app/cases/review.py`, `app/cases/examiner.py` and `app/api/v2_annotations.py`, so the UI withholds what the API refuses. */
import type { AnalysisDetail, Annotation, Conclusion, Signoff } from '../../lib/api/queries'
import { HEADLINE } from '../analysis/answer'
import { STEP_OF_CAPTURE, STEP_OF_DECLARATION, recordOf } from '../analysis/record'
export { POSITION_SHORT, examinerPosition } from '../analysis/answer'

export type ExaminerStatus = NonNullable<Annotation['examiner_status']>
export type ScalePosition = Conclusion['scale_position']
export type SignoffDecision = Signoff['decision']
export type SignoffKind = Signoff['kind']
export type Visibility = Annotation['visibility']

/** N-19a (browser QA P1-12; design review DS-09): a revision the examiner has signed on a case that requires a second review
 *  reads this, with the half glyph, wherever its state is shown: it is not reviewed until the peer has reviewed it (D-56). */
export const AWAITING_SECOND_REVIEW = 'awaiting second review'

export const EXAMINER_STATUSES: readonly ExaminerStatus[] = ['accepted', 'disputed', 'needs_check', 'not_relevant']
export const STATUS_WORD: Record<ExaminerStatus, string> = {
  accepted: 'Accepted', disputed: 'Disputed', needs_check: 'Needs check', not_relevant: 'Not relevant',
}
/** RW-01: a disputed or not-relevant status needs a note. */
export const NOTE_REQUIRED: ReadonlySet<ExaminerStatus> = new Set(['disputed', 'not_relevant'])

/** The verbal scale in UX §7.1 headline wording, strongest AI position first; `no_opinion` has no position. */
export const SCALE_POSITIONS: readonly ScalePosition[] = ['ai_supported_strong', 'ai_supported', 'inconclusive', 'no_ai_indicators', 'no_ai_indicators_strong', 'no_opinion']
export const SCALE_WORD: Record<ScalePosition, string> = {
  ai_supported_strong: HEADLINE.very_likely_ai,
  ai_supported: HEADLINE.likely_ai,
  inconclusive: HEADLINE.uncertain,
  no_ai_indicators: HEADLINE.likely_authentic,
  no_ai_indicators_strong: HEADLINE.very_likely_authentic,
  no_opinion: 'Unable to form an opinion',
}
const STEP: Record<string, number> = { ai_supported_strong: 2, ai_supported: 1, inconclusive: 0, no_ai_indicators: -1, no_ai_indicators_strong: -2 }
/** RW-03: a position this many steps from the machine decision needs a justification. */
export const JUSTIFY_FROM_STEPS = 2

/** N-19j (browser QA Q2-03): a sign-off decision is said with its object, in one wording for the dialog and the record (the
 *  dialog offered "Different opinion" and the record said "Do not concur"; "Examiner sign-off · Do not concur" beside "Peer
 *  review · Concur" read as a dispute where the examiner departed from the machine and the reviewer agreed with the examiner).
 *  The examiner concurs or does not concur with the machine assessment; the reviewer with the examiner's opinion. */
export const DECISION_WORD: Record<SignoffKind, Record<SignoffDecision, string>> = {
  examiner: { concur: 'Concur with the machine assessment', concur_with_changes: 'Concur with the machine assessment, with changes', non_concur: 'Do not concur with the machine assessment' },
  peer_review: { concur: 'Concur with the examiner’s opinion', concur_with_changes: 'Concur with the examiner’s opinion, with changes', non_concur: 'Do not concur with the examiner’s opinion' },
}
/** Where the machine's result is not assessable (or there is no machine decision) the examiner's sign-off has no object. */
export const NO_COMPARISON = 'Opinion recorded, no comparison with the machine assessment'
/** N-19j (Q2-02, Q2-08): these decisions need a comment, in the dialog as in the API: the reason, or the changes asked for. */
export const needsComment = (decision: SignoffDecision | null): boolean => decision === 'non_concur' || decision === 'concur_with_changes'

/** The decision of a recorded sign-off in words, with its object. `examinerDeparts`: whether the signed conclusion departs from
 *  the machine assessment; null where there is nothing to compare with (the sign-off records the opinion alone); undefined
 *  while that is not known. N-19 (browser QA P1-7): a record made before the rule can pair an examiner's concurring decision
 *  with a conclusion that departs; it prints the departure, not "Concur". N-19j: and the mirror, a record that does not concur
 *  under a conclusion equal to the machine's. */
export function recordDecision(s: { kind: SignoffKind; decision: SignoffDecision }, examinerDeparts: boolean | null | undefined, reference: string | null = null): string {
  if (s.kind === 'examiner' && examinerDeparts === null) return NO_COMPARISON
  // N-19n (D4-01): under a capture such a record is said in the words of the record's step (`view.signoff_decision`)
  if (s.kind === 'examiner' && s.decision !== 'non_concur' && examinerDeparts) return reference ? `Not on ${reference}` : 'Departs from the machine assessment'
  if (s.kind === 'examiner' && s.decision === 'non_concur' && examinerDeparts === false)
    return `Signed as not concurring; the conclusion ${reference ? `is on ${reference}` : 'agrees with the machine assessment'}`
  return DECISION_WORD[s.kind][s.decision]
}
export const KIND_WORD: Record<SignoffKind, string> = { examiner: 'Examiner sign-off', peer_review: 'Peer review' }

const byId = (x: { id: string }, y: { id: string }) => (x.id < y.id ? -1 : x.id > y.id ? 1 : 0)

/** N-19n (design review D4-01, P1; `review.STEPS_FROM`, `RECORD_STEP`): under a trusted capture record and under a trusted
 *  declaration the steps of a conclusion are counted from the step of that record on the scale, the first and the fifth, on
 *  both fusion paths. On the fallback the decision code stands one step inside, so that an examiner on the record's own step
 *  was "1 step from the machine decision" and could not sign as concurring. */
export type StepsFrom = 'capture_record' | 'declaration'
export const RECORD_STEP: Record<StepsFrom, ScalePosition> = { capture_record: 'no_ai_indicators_strong', declaration: 'ai_supported_strong' }

/** What the steps of a new conclusion on this result are counted from (`review.steps_from`); null: the decision code. */
export function stepsFromOf(d: AnalysisDetail['decision'] | null | undefined): StepsFrom | null {
  if (!d || STEP[d.code] === undefined) return null
  const kind = recordOf(d)?.kind
  return kind === 'trusted_capture' ? 'capture_record' : kind === 'decisive_ai' ? 'declaration' : null
}

/** The position that a new conclusion's steps are counted from: the record's step, else the machine decision. */
export const comparedWith = (machineCode: string | null | undefined, stepsFrom: StepsFrom | null = null): string | null =>
  machineCode != null && STEP[machineCode] !== undefined && stepsFrom ? RECORD_STEP[stepsFrom] : machineCode ?? null

/** Steps to the examiner's position (positive: towards AI), or null without two positions: from the step of the record
 *  that `stepsFrom` names, else from the machine decision (`review.deviation`). */
export function deviationSteps(machineCode: string | null | undefined, position: ScalePosition, stepsFrom: StepsFrom | null = null): number | null {
  const from = comparedWith(machineCode, stepsFrom)
  const m = from == null ? undefined : STEP[from]
  const p = STEP[position]
  return m === undefined || p === undefined ? null : p - m
}

type Compared = Pick<Conclusion, 'machine_code'> & { steps_from?: Conclusion['steps_from'] | null }

/** N-19n (D4-01; `view.review_reference`): what the examiner's layer names where it compares, in a result that a trusted
 *  capture decides: "the step of the capture record". Null: the words of before, which stand under a declaration, under a
 *  marker and without a record. For a recorded conclusion the words apply where its steps are the record's: it says so, or
 *  the code's step is the record's own (a fusion model); one recorded before on the fallback keeps the words of before with
 *  its steps. Without the conclusion's record the analysis's review summary says it (`examiner_reference`); with neither
 *  (the form of a new conclusion) the step is the record's. */
export function reviewReference(analysis: Pick<AnalysisDetail, 'decision' | 'review'>, conclusion?: Compared | null): string | null {
  const from = stepsFromOf(analysis.decision)
  // N-19o (D5-01): on the fallback a declaration's code is the fourth step and the steps are the fifth's: the layer names it.
  // With a fusion model the code is that step, and a conclusion recorded before N-19n counted from the code: words of before
  if (from === 'declaration' && analysis.decision?.code !== RECORD_STEP.declaration) {
    if (conclusion) return conclusion.steps_from === 'declaration' ? STEP_OF_DECLARATION : null
    if (analysis.review?.examiner_position) return analysis.review.examiner_reference === 'declaration' ? STEP_OF_DECLARATION : null
    return STEP_OF_DECLARATION
  }
  if (from !== 'capture_record') return null
  if (conclusion) return conclusion.steps_from === 'capture_record' || conclusion.machine_code === RECORD_STEP.capture_record ? STEP_OF_CAPTURE : null
  if (analysis.review?.examiner_position) return analysis.review.examiner_reference === 'capture_record' ? STEP_OF_CAPTURE : null
  return STEP_OF_CAPTURE
}

export const stepsWord = (n: number): string => `${n} ${n === 1 ? 'step' : 'steps'}`

/** A position in the words of the scale, as the reports print it in the opinion line: the headline without its explanation
 *  ("Inconclusive", not "Inconclusive: the evidence does not support a firm assessment"). */
export const opinionPosition = (position: ScalePosition): string =>
  position === 'no_opinion' ? 'No opinion on the scale' : SCALE_WORD[position].split(':')[0]

export interface ExaminerOpinion { line: string; departs: boolean | null; label: string | null; justification: string | null; also?: { label: string; text: string } | null }

/** The examiner's opinion of a signed record, as it stands directly under the machine headline (N-19, D-69; browser QA P1-6:
 *  a departing conclusion was one small line, the first sentence of its rationale). Mirrors `view.examiner_opinion`: the
 *  position, whether it departs from or agrees with the machine assessment and by how many steps, and the justification in
 *  full (the justification of the departure when one is recorded, else the rationale). Without the conclusion's record (it
 *  is still loading, or cannot be read) the line comes from the analysis's review summary, without the steps. */
export function examinerOpinion(conclusion: Pick<Conclusion, 'scale_position' | 'departs_from_machine' | 'deviation_steps' | 'deviation_justification' | 'rationale' | 'machine_code'> | null | undefined,
  summary?: { examiner_position?: ScalePosition | null; examiner_departs?: boolean | null } | null, reference: string | null = null): ExaminerOpinion | null {
  const position = conclusion?.scale_position ?? summary?.examiner_position
  if (!position) return null
  const departs = conclusion ? (conclusion.machine_code == null ? null : conclusion.departs_from_machine) : summary?.examiner_departs ?? null
  const steps = Math.abs(conclusion?.deviation_steps ?? 0)
  // N-19n (D4-01): under a capture the line names the step of the capture record (`reference`: `reviewReference`). "No
  // opinion" has its line with the negation; from the summary alone, which holds no steps, a position is "away from" it
  const relation = departs == null ? 'no comparison with the machine assessment is recorded'
    : reference ? (!departs ? `on ${reference}` : steps ? `${stepsWord(steps)} from ${reference}` : `${position === 'no_opinion' ? 'not on' : 'away from'} ${reference}`)
    : departs ? `departs from the machine assessment${steps ? ` by ${stepsWord(steps)}` : ''}` : 'agrees with the machine assessment'
  // N-19p (design review D6-02; `view.examiner_opinion`): "Justification of the departure" only where the conclusion departs;
  // a justification typed on the step (an API client can send one) follows the rationale as "Justification"
  const justified = !!conclusion?.deviation_justification && departs === true
  const typed = conclusion?.deviation_justification
  return {
    line: `Examiner’s opinion: ${opinionPosition(position)}, ${relation}`, departs,
    label: conclusion ? (justified ? 'Justification of the departure' : 'Rationale') : null,
    justification: conclusion ? (justified ? typed ?? null : conclusion.rationale) : null,
    also: typed && !justified ? { label: 'Justification', text: typed } : null,
  }
}

export function needsJustification(machineCode: string | null | undefined, position: ScalePosition, stepsFrom: StepsFrom | null = null): boolean {
  const steps = deviationSteps(machineCode, position, stepsFrom)
  return steps !== null && Math.abs(steps) >= JUSTIFY_FROM_STEPS
}

export interface Disposition { status: ExaminerStatus; note: string | null; annotation: Annotation }

/** The examiner status of each finding (or signal): the status most recently set among its live annotations, with that
 *  annotation's note (`app/cases/examiner.py`). */
export function dispositions(annotations: readonly Annotation[], targetType: 'finding' | 'signal' = 'finding'): Map<string, Disposition> {
  const at = (a: Annotation) => Date.parse(a.edited_at ?? a.created_at)
  const rows = annotations
    .filter((a) => a.target_type === targetType && a.deleted_at == null && a.examiner_status != null)
    .sort((x, y) => at(x) - at(y) || byId(x, y))
  const out = new Map<string, Disposition>()
  for (const a of rows) out.set(a.target_id ?? '', { status: a.examiner_status as ExaminerStatus, note: a.body, annotation: a })
  return out
}

/** A note is a live annotation with a body or a drawn shape. A status set without a note is not one. */
export const isNote = (a: Annotation): boolean => a.deleted_at == null && (a.body != null || a.shape !== 'none')

/** N-numbers in creation order, shared by the notes thread, the viewer pins and the report (QA B6). */
export function noteNumbers(annotations: readonly Annotation[]): Map<string, string> {
  const notes = annotations.filter(isNote).sort((x, y) => Date.parse(x.created_at) - Date.parse(y.created_at) || byId(x, y))
  return new Map(notes.map((a, i) => [a.id, `N${i + 1}`]))
}

const seconds = (pts: unknown, timeBase: unknown): number | null => {
  const m = typeof timeBase === 'string' ? /^(\d+)\/(\d+)$/.exec(timeBase) : null
  return m && typeof pts === 'number' ? (pts * Number(m[1])) / Number(m[2]) : null
}

/** The index of the sentence a note is attached to (`geometry.sentence_idx`), or null: another target, or character offsets. */
export function sentenceIndex(a: Pick<Annotation, 'target_type' | 'geometry'>): number | null {
  const idx = a.target_type === 'sentence' ? (a.geometry as Record<string, unknown> | null | undefined)?.sentence_idx : null
  return typeof idx === 'number' && Number.isInteger(idx) && idx >= 0 ? idx : null
}

/** The anchor chip of a note (UX §5.13): the finding or signal id, or the kind of place it is attached to. */
export function anchorOf(a: Annotation): string {
  switch (a.target_type) {
    case 'finding':
    case 'signal':
      return a.target_id ?? a.target_type
    case 'interval': {
      const g = (a.geometry ?? {}) as Record<string, unknown>
      const start = seconds(g.start_pts, g.time_base)
      const end = seconds(g.end_pts, g.time_base)
      return start !== null && end !== null ? `${start.toFixed(2)}–${end.toFixed(2)} s` : 'Interval'
    }
    case 'region': return 'Region'
    case 'exhibit': return 'Exhibit'
    case 'sentence': {
      const idx = sentenceIndex(a)
      return idx === null ? 'Sentence' : `Sentence ${idx + 1}`  // as the manuscript numbers it
    }
    case 'case': return 'Case'
    default: return 'Analysis'
  }
}

export interface Limitation { id: string; text: string }

/** The analysis caveats as acknowledgeable limitations. A result is immutable per revision, so `L-01…` are stable ids. */
export const limitations = (analysis: Pick<AnalysisDetail, 'caveats'>): Limitation[] =>
  (analysis.caveats ?? []).map((text, i) => ({ id: `L-${String(i + 1).padStart(2, '0')}`, text }))

/** Whether the examiner layer of this revision can still change, and why not (C-4, C-5 refusals). */
export function reviewGate(a: Pick<AnalysisDetail, 'status' | 'review'>): { writable: boolean; reason: string | null } {
  if (a.status !== 'done') return { writable: false, reason: 'The analysis has not finished.' }
  if (a.review?.state === 'withdrawn') return { writable: false, reason: 'This revision was withdrawn and should not be relied on.' }
  if (a.review?.state === 'superseded') return { writable: false, reason: 'A newer revision of this analysis exists; review the latest revision.' }
  if (a.review?.locked_at) return { writable: false, reason: 'This revision is signed off and locked. Notes, statuses and the conclusion are read-only; a later change needs a new revision.' }
  return { writable: true, reason: null }
}

/** The caller's standing on the analysis's case: instance roles and the case role (null when not a member). */
export interface Standing { roles: readonly string[]; caseRole: string | null | undefined }

/** Whether the standing may sign this kind (`app/api/v2_review.py` post_signoff; ADR 0011): an administrator always; the examiner
 *  sign-off needs an examiner with the owner or examiner case role; a peer review needs an examiner or reviewer with the owner or
 *  reviewer case role (RW-04: no instance reviewer role). Without a standing the check is left to the server. */
export function mayCosign(kind: SignoffKind, standing: Standing | undefined): boolean {
  if (!standing || standing.roles.includes('administrator')) return true
  const role = (r: string) => standing.roles.includes(r)
  return kind === 'examiner'
    ? role('examiner') && (standing.caseRole === 'owner' || standing.caseRole === 'examiner')
    : (role('examiner') || role('reviewer')) && (standing.caseRole === 'owner' || standing.caseRole === 'reviewer')
}

/** The sign-off the caller can make now, if any (`app/api/v2_review.py` post_signoff). */
export function signoffAction(a: Pick<AnalysisDetail, 'status' | 'review'>, conclusion: Conclusion | null | undefined,
  records: readonly Signoff[], meId: string | null | undefined, standing?: Standing): SignoffKind | null {
  if (!meId || a.status !== 'done' || a.review?.state === 'withdrawn' || a.review?.state === 'superseded') return null
  const examiner = records.find((s) => s.kind === 'examiner')
  if (!examiner) return conclusion && conclusion.author_id === meId && !a.review?.locked_at && mayCosign('examiner', standing) ? 'examiner' : null
  if (!a.review?.requires_second_review || records.some((s) => s.kind === 'peer_review')) return null
  return meId !== examiner.principal_id && meId !== conclusion?.author_id && mayCosign('peer_review', standing) ? 'peer_review' : null
}

/** UX §5.14: a different opinion should name the findings it disagrees with (warned, never blocking). */
export const citesEvidence = (text: string): boolean => /\b[FS]\d{1,3}\b/.test(text)
