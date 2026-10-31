/** What a check did in a decision, in one vocabulary (D-63; BUILD_SPEC N-17). The mirror of `backend/app/core/effect.py`.
 *
 *  For a check that entered the estimate, direction and strength come from its contribution, the change in log-odds when the
 *  check moves from its neutral point to its score: `for` or `against` by sign; weak from ln 2, moderate from ln 10, strong from
 *  ln 100 (the D-16 bands); below ln 2 it had little effect (strength `none`, direction `neutral`). A check outside the estimate
 *  keeps its stand-alone words and is said to be not in the estimate. The key findings and their counts, the agreement label,
 *  the ledger and the strip read these functions, so one check has one word on every surface.
 *
 *  A result published before N-17 (a `context_prior` starting row) was labelled with a threshold of 0.05 log-odds and with
 *  stand-alone strength words: it keeps both (`LEGACY_LITTLE`). */
import { fmtDay } from '../../lib/format'
import { storedWords } from '../../lib/sliceWords'

export const LN2 = Math.log(2)
export const LN10 = Math.log(10)
export const LN100 = Math.log(100)
/** A contribution of at most this size is a check with no weight in the estimate. */
export const ENTERED = 1e-9
export const LEGACY_LITTLE = 0.05
export const NOT_IN_ESTIMATE = 'not in this estimate'

export type EffectStrength = 'none' | 'weak' | 'moderate' | 'strong'
export type EffectDirection = 'for' | 'against' | 'neutral'
interface Rows { fusion_mode?: string | null; contributions?: readonly { kind: string }[] | null }

export const entered = (c: number | null | undefined): c is number => c != null && Math.abs(c) > ENTERED

export function strength(c: number): EffectStrength {
  const size = Math.abs(c)
  return size >= LN100 ? 'strong' : size >= LN10 ? 'moderate' : size >= LN2 ? 'weak' : 'none'
}

/** `for`, `against`, or `neutral` when the check had little effect (below `little`). */
export const direction = (c: number, little: number = LN2): EffectDirection => (c >= little ? 'for' : c <= -little ? 'against' : 'neutral')

/** "little effect", or "weak support for" … "strong support against". */
export function words(c: number): string {
  const level = strength(c)
  return level === 'none' ? 'little effect' : `${level} support ${c > 0 ? 'for' : 'against'}`
}

/** The size below which a contribution had little effect in this decision: ln 2, and 0.05 for a result published before N-17. */
export const littleOf = (d: Rows | null | undefined): number =>
  (d?.contributions ?? []).some((r) => r.kind === 'context_prior') ? LEGACY_LITTLE : LN2

/** An artefact decision whose rows are measured from each check's neutral point (a `neutral_start` row). */
export const measuredFromNeutral = (d: Rows | null | undefined): boolean =>
  d?.fusion_mode === 'artefact' && (d.contributions ?? []).some((r) => r.kind === 'neutral_start')

/** A check's direction on every surface (the ledger, the strip): its recorded contribution when there is one (neutral below
 *  `little`: D-63), else its log-likelihood ratio (≥ ln 2 either way), else the v1 mirror. Design review DS-17: never "p above
 *  50". The reports' rule is `forensic._direction_word`. */
export function signalDirection(
  s: { contribution_logit?: number | null; log_lr?: number | null; legacy?: { direction?: string | null } | null;
    detector_id?: string; provenance_override?: string | null; metrics?: Record<string, number> },
  little: number = LN2,
): 'ai' | 'authentic' | 'neutral' {
  // N-19o (browser QA Q5-01): a mention of AI in a text field is no marker and points neither way (`view.is_mention`)
  if (s.detector_id === 'img.meta.generator' && s.provenance_override == null && (s.metrics?.free_text_hits ?? 0) > 0) return 'neutral'
  if (s.contribution_logit != null) return s.contribution_logit >= little ? 'ai' : s.contribution_logit <= -little ? 'authentic' : 'neutral'
  if (s.log_lr != null) return s.log_lr >= LN2 ? 'ai' : s.log_lr <= -LN2 ? 'authentic' : 'neutral'
  const legacy = s.legacy?.direction
  return legacy === 'supports_ai' ? 'ai' : legacy === 'supports_authentic' ? 'authentic' : 'neutral'
}


/** N-19d (the N-5 record, "Not done"): the review of a fitted fusion model as its effective state, never the raw field of the
 *  artefact. While no review record names the artefact that field holds the flag of the automated fit, which names the build
 *  tooling: it reads "pending human review". A review record gives "reviewed by <name> on <date>" (`formatting.review_phrase`).
 *  A result published before N-19d holds no date. */
export function fuserReview(reviewedBy: string | null | undefined, reviewDate?: string | null): string {
  if (!reviewedBy) return 'not reviewed'
  if (/pending human review/i.test(reviewedBy)) return 'pending human review'
  return `reviewed by ${reviewedBy}${reviewDate ? ` on ${fmtDay(reviewDate)}` : ''}`
}

/** N-18b (operations re-check O4-01): the caveats of a result that say a step of its run failed, and what the result lacks
 *  for it ("The step that marks regions failed: this result shows no regions from it."). The sentences are written by
 *  `app/core/effect.py` (`step_failed`); the page recognises them by their form, as `effect.failed_steps` does. */
const STEP_FAILED = /^The step that .+ failed: .+\.$/
export const failedSteps = (caveats: readonly string[] | null | undefined): string[] => (caveats ?? []).filter((c) => STEP_FAILED.test(c))

// --- a result whose step for the findings failed (N-19n; browser QA Q4-01; `app.core.effect`) --------------------------------
// Such a result is done, with its decision and no finding. Where the findings would stand the page says that they could not
// be written, and nothing there describes the evidence ("No individual check produced more than weak support either way."
// is true of a result with findings only). The caveat is `effect.step_failed("synthesise")`; a result stored before N-19n
// holds it with the stage's id in backticks, which `storedWords` prints in the step's words.
export const FINDINGS_STEP_FAILED = 'The step that writes the findings failed: this result holds no findings from it.'
export const findingsFailed = (caveats: readonly string[] | null | undefined): boolean => (caveats ?? []).some((c) => storedWords(c) === FINDINGS_STEP_FAILED)
const FINDINGS_NOT_WRITTEN = 'The findings of this result could not be written: a step of the analysis failed.'
const SCORES_IN = { ledger: 'the evidence ledger', expert: 'the expert view' } as const
/** The sentence in the findings' place (`effect.findings_not_written`): the Plain view, which holds no ledger, names the expert view. */
export const findingsNotWritten = (where: keyof typeof SCORES_IN): string => `${FINDINGS_NOT_WRITTEN} The checks and their scores are in ${SCORES_IN[where]}.`
export const AGREEMENT_NOT_WRITTEN = 'the findings could not be written'
/** The next step of such a result on the page (`effect.run_again`; a report names "the Veritas server that made this report"). */
export const RUN_AGAIN = 'Run the analysis again before relying on this result: its findings could not be written. If they fail again, tell the administrator of this server.'

// --- a check that scored and carries no weight (N-19k; browser QA Q2-12; `app.core.effect`) -----------------------------------
// One sentence carries both halves: the score, and why it is not counted, from the record of that check for this input
// (`SignalV2.outside_estimate`, written by the fusion) and from what the check reached in testing (its benchmark note).

/** The fusion's record of why a check that scored is not counted in the estimate (`schemas.evidence.OutsideEstimate`). */
export interface OutsideEstimate { reason: 'not_in_model' | 'gate' | 'zero_weight'; gate?: readonly string[] | null }

const AT_CHANCE = 0.05
export const NO_RECORD_WHY = 'it carries no weight in the fused model for this input'  // a result stored before N-19k names no reason
const NOT_IN_MODEL_WHY = 'it is not in the fitted model for inputs like this one'
const ZERO_WEIGHT_WHY = 'the model gives it no weight for inputs like this one'
export const VOICE_STEP = 'To assess the voice, examine the sound track as audio.'
const GATE_TERMS: Record<string, readonly [string, string]> = {
  has_face: ['inputs with a face', 'inputs without a face'],
  'face_px:120-256': ['inputs with a face of 120 to 256 px', 'inputs without a face of 120 to 256 px'],
  'face_px>=256': ['inputs with a face of 256 px or more', 'inputs without a face of 256 px or more'],
  'jpeg_q>=90': ['JPEG quality of 90 or more', 'JPEG quality under 90'],
  'jpeg_q:70-89': ['JPEG quality of 70 to 89', 'JPEG quality outside 70 to 89'],
  'jpeg_q<70': ['JPEG quality under 70', 'JPEG quality of 70 or more'],
  'min_side<512': ['a shorter side under 512 px', 'a shorter side of 512 px or more'],
  'min_side>1024': ['a shorter side over 1,024 px', 'a shorter side of 1,024 px or less'],
  upscaled: ['upscaled inputs', 'inputs that were not upscaled'],
  has_camera_exif: ['inputs with a camera\u2019s EXIF block', 'inputs without a camera\u2019s EXIF block'],
  video_frame: ['frames of a video', 'still images'],
  screenshot_like: ['inputs that look like a screenshot', 'inputs that do not look like a screenshot'],
}

/** The inputs a gate lets in, in words: `['!has_face']` reads "inputs without a face" (`effect.gate_words`). */
export function gateWords(literals: readonly string[]): string {
  const said = literals.map((literal) => {
    const wanted = !literal.startsWith('!')
    const term = wanted ? literal : literal.slice(1)
    return term in GATE_TERMS ? GATE_TERMS[term][wanted ? 0 : 1] : 'inputs under another condition'
  })
  return said.join(' and ') || 'other inputs'
}

/** (AUC, n) of a check's benchmark note ("… this check reached AUC 0.52 (n = 29, 2026-09-17)."), or null (`effect.measured_auc`). */
export function measuredAuc(note: string | null | undefined): { auc: number; n: string | null } | null {
  const m = /reached AUC (\d\.\d\d)(?: \(n = (\d+))?/.exec(note ?? '')
  return m ? { auc: Number(m[1]), n: m[2] ?? null } : null
}

/** Why a check that scored is not counted in this estimate (`effect.outside_why`): not in the fitted model, its gate closed
 *  and by which condition, or no weight in the model, with what the check reached in testing: "at chance" only where its AUC is. */
export function outsideWhy(outside: OutsideEstimate | null | undefined, note?: string | null, figures = true): string {
  if (outside?.reason === 'not_in_model') return NOT_IN_MODEL_WHY
  if (outside?.reason === 'gate') return `its gate is closed: the model uses this check only for ${gateWords(outside.gate ?? [])}`
  if (outside?.reason !== 'zero_weight') return NO_RECORD_WHY
  const measured = measuredAuc(note)
  if (!measured) return ZERO_WEIGHT_WHY
  const reached = `AUC ${measured.auc.toFixed(2)}${measured.n ? `, n = ${measured.n}` : ''}`
  if (Math.abs(measured.auc - 0.5) <= AT_CHANCE + 1e-9) return `in testing it was at chance on inputs like this one${figures ? ` (${reached})` : ''}`
  return `${ZERO_WEIGHT_WHY}${figures ? ` (in testing it reached ${reached})` : ''}`
}

/** The sentence of a check that scored and is not counted (`effect.outside_sentence`). */
export const outsideSentence = (score: string, why: string, standalone = false): string =>
  standalone ? `Its score of ${score} is not counted in this estimate: ${why}.` : `This check scored ${score} and is not counted in this estimate: ${why}.`
