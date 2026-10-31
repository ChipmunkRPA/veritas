/** BUILD_SPEC N-19h (E-21): the provenance record that decides a result (`reports/view.py record_of`, `stored_record`).
 *
 *  A record decides a result when the provenance hypothesis (H3) drives it; which record is the override the decision
 *  applied by its precedence (declaration, capture, marker). The decision says it (`provenance_record`), and the page takes
 *  the basis of the result from there, not from its findings: a file with a trusted capture record and an AI marker is the
 *  capture's result, whatever the marker's finding says.
 *
 *  A result stored before the decision said so is read by the server (N-19l, code review R6-01): the reports' rule
 *  (`view.stored_record`) works the record out from the stored signals, which the analysis detail does not hold, and the
 *  detail gives it as `stored_record`. The page keeps no rule of its own for such a result: `withRecord` puts that record
 *  on the decision in the query layer. */

import { halfUp } from '../../lib/format'

export type RecordKind = 'decisive_ai' | 'trusted_capture' | 'self_declared_ai'
export type MarkerKind = 'metadata' | 'watermark' | 'credentials_listed_signer' | 'credentials_unlisted_signer_declared' | 'credentials_unlisted_signer_named'
  // N-19n (Q4-01; D-15 as amended): unsigned metadata whose marker is the IPTC digital source type names no generator
  | 'metadata_declared' | 'metadata_declared_composite'
  // N-19n (D3-07, 9): generation parameters that match no fingerprint and name no tool
  | 'metadata_parameters'
export type DecidingRecord = { kind: RecordKind; marker?: MarkerKind | null; signal_ids?: string[] }

type DecisionLike = { driving_hypothesis?: string | null; reason?: string | null; verbal?: string | null; provenance_record?: DecidingRecord | null }
type NumbersLike = DecisionLike & { probability_ai_calibrated?: number | null; probability_ai_uncalibrated?: number | null;
  statistical_estimate_before_provenance?: number | null }
/** N-19l (R6-05; `Decision.statistical_alone`): what the statistical checks alone decided, where a record decides beside a
 *  statistical estimate of a fusion model. Absent in a result stored before. */
export type StatisticalAlone = { code: string; unsupported?: 'for' | 'against' | null; rule?: 'scale' | 'no_check' | 'together_less' | 'together_against' | null }
type AloneLike = NumbersLike & { statistical_alone?: StatisticalAlone | null; ci90?: readonly number[] | null }
type AssessmentLike = { hypothesis: string; note?: string | null }

/** The record that decides: what the decision says. For a result stored before N-19h the query layer has put the server's
 *  record there (`withRecord`); a decision that reaches a component without one has no record that decides. */
export function recordOf(d: DecisionLike | null | undefined): DecidingRecord | null {
  return d?.provenance_record ?? null
}

/** N-19h (E-20, 1; `view.stored_statistical`): a result of the image fallback stored before the stage published the
 *  statistical score beside a record holds that score only in its caveat, in whole points ("Statistical score before the
 *  provenance override: 30 / 100, not calibrated."; before N-7 "Statistical estimate before the provenance override: 30%.").
 *  It is read from there, so that the result is not taken for one that a record decides alone. A result in which no check
 *  of the content scored holds its empty "50 / 100" there, which is no score. The reports tell the two apart by the signals;
 *  the analysis detail holds none, so here a caveat of exactly 50 is read as no score. */
const STORED_STATISTICAL = /^Statistical (?:score before the provenance override: (\d+) \/ 100, not calibrated|estimate before the provenance override: (\d+)\s?%)\./
export function storedStatistical(d: NumbersLike & { fusion_mode?: string | null }, caveats: readonly string[] = []): number | null {
  if (d.fusion_mode !== 'fallback' || d.statistical_estimate_before_provenance != null || !recordOf(d)) return null
  for (const c of caveats) {
    const m = STORED_STATISTICAL.exec(c)
    if (m) {
      const points = Number(m[1] ?? m[2])
      return points === 50 ? null : points / 100
    }
  }
  return null
}

/** N-19h (E-20, 8; `abstain.outside_validated`): whether the input of a result that publishes a number is outside what was
 *  validated. It is the key of what stands beside a record: the statistical number is printed beside no record there. A
 *  calibrated result says it in its condition profile. The fallback has no validated envelope at all (the flag is false
 *  for every fallback result), so its results say it in their reason: a statistical one in the reason of its cap, one that
 *  a provenance record decides in the sentence the stage adds to the record's reason. A result without a number (not
 *  assessable; a check did not run and no record decides) is not asked. */
type OutsideLike = NumbersLike & { code?: string | null; fusion_mode?: string | null }
type ProfileLike = { within_validated_envelope?: boolean | null } | null | undefined
const CAP_REASON = /^Capped at inconclusive: (.+?)(?:; conditions: .*?)?\. Before the cap/
const OUTSIDE_RECORD = / This input is outside what was validated: (.+)\.$/
export function outsideValidated(d: OutsideLike | null | undefined, profile: ProfileLike): boolean {
  if (!d || d.code === 'not_assessable') return false
  if (d.probability_ai_calibrated == null && d.probability_ai_uncalibrated == null) return false
  const reason = d.reason ?? ''
  if (CAP_REASON.test(reason) || (d.fusion_mode === 'fallback' && OUTSIDE_RECORD.test(reason))) return true
  return d.fusion_mode === 'artefact' && profile?.within_validated_envelope === false
}

/** N-19h (E-20, 8; `view.statistical_withheld`): the statistical number that stands beside a record is not shown, because
 *  the input is outside what was validated. The decision's field keeps it. False where there is none to withhold. */
export const statisticalWithheld = (d: OutsideLike | null | undefined, profile: ProfileLike): boolean =>
  d?.statistical_estimate_before_provenance != null && outsideValidated(d, profile)

/** What stands in place of the number (`view.STATISTICAL_WITHHELD`), and the caveats that say so (`decisive.py`). */
export const STATISTICAL_WITHHELD = 'not shown (outside what was validated)'
export const STATISTICAL_WITHHELD_CAVEAT = 'The statistical estimate before the provenance override is not shown: this input is outside what was validated.'
export const FALLBACK_STATISTICAL_WITHHELD_CAVEAT = 'The statistical score before the provenance override is not shown: this input is outside what was validated.'
const ESTIMATE_CAVEAT = 'Statistical estimate before the provenance override:'
const SCORE_CAVEAT = 'Statistical score before the provenance override:'

/** N-19m (code review R7-01; D-76 beside a record; `view.alone_inconclusive`): the result says that the statistical checks
 *  alone are inconclusive, by the band or by a decision rule. Their number beside the record that decides is then printed
 *  with the words that they alone do not decide, whether the record set the result's number or not. False for a result
 *  stored before N-19l, which does not say: its number stands as it did. */
export const aloneInconclusive = (d: AloneLike | null | undefined): boolean => d?.statistical_alone?.code === 'inconclusive'

/** N-19n (design review D4-02; D-68 as amended on 2026-10-05; `view.alone_other_way`): the statistical checks alone decide
 *  the other answer than the record that decides: an AI code under a trusted capture, a clean code under a marker or a
 *  declaration. The fallback publishes no such code: its score's own ranges are read, 65 or more under a capture and under
 *  35 under a marker or a declaration. False without a record, on the fallback without a statistical score, and for a
 *  result stored before N-19l on the fusion model, which does not say what the checks alone decided. */
export function aloneOtherWay(d: (AloneLike & { fusion_mode?: string | null }) | null | undefined): boolean {
  const kind = recordOf(d)?.kind
  if (!d || kind == null) return false
  const capture = kind === 'trusted_capture'
  if (d.fusion_mode === 'fallback') {
    const p = d.statistical_estimate_before_provenance
    // N-19o (code review R9-04; `view.alone_other_way`): on the printed whole points, as D-77 reads them
    return p != null && (capture ? halfUp(p * 100) >= 65 : halfUp(p * 100) <= 34)
  }
  const alone = d.statistical_alone?.code
  return capture ? alone === 'ai_supported' || alone === 'ai_supported_strong' : alone === 'no_ai_indicators' || alone === 'no_ai_indicators_strong'
}
/** `view.ALONE_OTHER_WAY`, `OTHER_WAY`, `OTHER_WAY_ONE`: the sentence, and the words beside the number. */
export const ALONE_OTHER_WAY = 'The statistical checks alone point the other way'
export const OTHER_WAY = 'they point the other way'
export const OTHER_WAY_ONE = 'it points the other way'

/** `view.alone_number`: the statistical number where the checks alone do not decide: the range where the result holds one
 *  (the record did not set the number, so the range is the statistical one: CAPUNS), else the estimate, as the page
 *  prints an estimate (`numeral`). */
export function aloneNumber(d: AloneLike): string {
  const ci = recordSetNumber(d) ? null : d.ci90
  if (ci && ci.length === 2) return `${halfUp(ci[0] * 100)}–${halfUp(ci[1] * 100)}`
  const v = (d.statistical_estimate_before_provenance ?? 0) * 100
  return v < 1 ? '<1' : v > 99 ? '>99' : String(halfUp(v))
}

/** N-19m (R7-02; `view.alone_caveat`): the caveat with the statistical number as it is shown where the checks alone do not
 *  decide ("Statistical estimate before the provenance override: 90%." stood bare). The stored caveat stays as it is. */
export const aloneCaveat = (d: AloneLike): string =>
  `Statistical estimate without the provenance record: ${aloneNumber(d)} out of 100; the statistical checks alone do not decide.`

/** N-19n (D4-04; `view.without_record_caveat`): the caveat with the statistical number as it is shown where the checks alone
 *  decide ("Statistical estimate before the provenance override: 98%." stood beside "Statistical checks alone: 98 out of 100").
 *  The stored caveat stays as it is. */
export function withoutRecordCaveat(d: AloneLike & { fusion_mode?: string | null }): string {
  const p = d.statistical_estimate_before_provenance ?? 0
  if (d.fusion_mode === 'fallback') return `Statistical score without the provenance record: ${halfUp(p * 100)} / 100, not calibrated.`
  return `Statistical estimate without the provenance record: ${aloneNumberOf(p)} out of 100.`
}
const aloneNumberOf = (p: number): string => (p * 100 < 1 ? '<1' : p * 100 > 99 ? '>99' : String(halfUp(p * 100)))

/** `decisive.withhold_statistical`: the caveat that prints the statistical number before the override, replaced in its
 *  place by the one that says that it is not shown (a caveat keeps its line: limitations are acknowledged by number). */
export function withholdStatistical(caveats: readonly string[], fallback = false): string[] {
  return caveats.map((c) => (c.startsWith(SCORE_CAVEAT) || (fallback && c.startsWith(ESTIMATE_CAVEAT)) ? FALLBACK_STATISTICAL_WITHHELD_CAVEAT
    : c.startsWith(ESTIMATE_CAVEAT) ? STATISTICAL_WITHHELD_CAVEAT : c))
}

/** The analysis detail with its deciding record on the decision, also for a stored result (the query layer applies it, so
 *  every reader of the decision finds the record in one place), with the statistical score that a stored result of the
 *  image fallback holds only in its caveat, and with the caveats as they are shown: a result stored before N-19h holds,
 *  outside what was validated, the caveat with the statistical number beside its record (`view.shown_caveats`). */
export function withRecord<T extends { decision?: (OutsideLike & AloneLike) | null; stored_record?: DecidingRecord | null;
  caveats?: readonly string[] | null; condition_profile?: ProfileLike }>(detail: T): T {
  const d = detail.decision
  if (!d) return detail
  const record = d.provenance_record ?? detail.stored_record ?? null  // N-19l: the server's record of a stored result
  if (!record) return detail
  const named = d.provenance_record ? d : { ...d, provenance_record: record }
  const statistical = storedStatistical(named, detail.caveats ?? [])
  const decision = statistical === null ? named : { ...named, statistical_estimate_before_provenance: statistical }
  const stored = detail.caveats ?? []
  // N-19m (R7-02): where the statistical checks alone do not decide, the caveat does not give their estimate bare; N-19n
  // (D4-04): where they decide it is shown in one form too, with the unit of the number beside it (on the fallback by its
  // score; a result of the fusion model stored before N-19l does not say what they decided, and keeps its caveat)
  const shown = outsideValidated(decision, detail.condition_profile) ? withholdStatistical(stored, decision.fusion_mode === 'fallback')
    : aloneInconclusive(decision) && decision.statistical_estimate_before_provenance != null
      ? stored.map((c) => (c.startsWith(ESTIMATE_CAVEAT) ? aloneCaveat(decision) : c))
      : decision.statistical_estimate_before_provenance != null && (decision.fusion_mode === 'fallback' || decision.statistical_alone != null)
        ? stored.map((c) => (c.startsWith(ESTIMATE_CAVEAT) || c.startsWith(SCORE_CAVEAT) ? withoutRecordCaveat(decision) : c)) : stored
  const caveats = shown.some((c, i) => c !== stored[i]) ? shown : null
  if (decision === d && caveats === null) return detail
  return { ...detail, decision, ...(caveats ? { caveats } : {}) }
}

const WATERMARK_DETECTORS: readonly string[] = ['img.watermark.dwtdct']
const CREDENTIAL_DETECTORS: readonly string[] = ['prov.c2pa', 'prov.c2pa.video']
/** N-19l (`decisive.marker_kind`): the kind of the self-declared marker among signals that hold such an override; null
 *  without one. A watermark message alone is a watermark; with another marker beside it the words are the other marker's. */
export function markerKindOf(markers: readonly { detector_id?: string | null; metrics?: Record<string, unknown> | null }[]): MarkerKind | null {
  if (markers.length === 0) return null
  const first = markers.find((s) => !WATERMARK_DETECTORS.includes(s.detector_id ?? ''))
  if (!first) return 'watermark'
  if (!CREDENTIAL_DETECTORS.includes(first.detector_id ?? '')) {
    // N-19n: the metadata check hands the kind on in its metrics; a result stored before holds none and names a generator
    if (first.metrics?.names_generator === 0) {
      // (9): a declaration by the source type or by the phrase in a field that software writes; parameters without a tool
      if (first.metrics?.source_type_trained || first.metrics?.phrase_declared) return 'metadata_declared'
      if (first.metrics?.source_type_composite) return 'metadata_declared_composite'
      if (first.metrics?.sd_params) return 'metadata_parameters'
    }
    return 'metadata'
  }
  if (first.metrics?.trusted) return 'credentials_listed_signer'
  return first.metrics?.ai_declared ? 'credentials_unlisted_signer_declared' : 'credentials_unlisted_signer_named'
}

/** The kind of the marker that decides a result; unsigned metadata where the result does not say (a result stored before
 *  N-19h, whose words were those of metadata). `view.marker_of`. */
export const markerOf = (d: DecisionLike | null | undefined): MarkerKind => recordOf(d)?.marker ?? 'metadata'

/** N-19h (E-21): the words of a marker, by its kind (`decisive.py`, `view.marker_headline`, `view.marker_guidance`; the
 *  vectors `backend/tests/fixtures/marker_words_vectors.json` hold both sides to the same sentences). Each says what the
 *  file carries and who can make such a marker: nothing says of Content Credentials that metadata can be edited, and nothing
 *  says of a signer on the trust list that anyone can sign. */
const HEADLINE: Record<MarkerKind, string> = {
  metadata: 'The file’s metadata names an AI generator',
  watermark: 'The image carries a public AI-generator watermark',
  credentials_listed_signer: 'The file’s Content Credentials name an AI generator',
  credentials_unlisted_signer_declared: 'The file’s Content Credentials declare AI generation; their signer is not on the trust list',
  credentials_unlisted_signer_named: 'The file’s Content Credentials name an AI generator; their signer is not on the trust list',
  metadata_declared: 'The file’s metadata declares AI generation',
  metadata_declared_composite: 'The file’s metadata declares a composite with AI-generated parts',
  metadata_parameters: 'The file’s metadata holds AI generation parameters',
}
const MEANS: Record<MarkerKind, (one: string) => string> = {
  metadata: () => 'The file’s metadata names an AI generator. Metadata can be added or edited by anyone.',
  watermark: (one) => `The ${one} carries a public AI-generator watermark message. The message is public, so anyone can embed it.`,
  credentials_listed_signer: () => 'The file’s Content Credentials name an AI generator and declare no AI generation. Their signer is on the trust list.',
  credentials_unlisted_signer_declared: () => 'The file’s Content Credentials declare AI generation, and their signer is not on the trust list. Anyone can sign such credentials.',
  credentials_unlisted_signer_named: () => 'The file’s Content Credentials name an AI generator, and their signer is not on the trust list. Anyone can sign such credentials.',
  metadata_declared: () => 'The file’s metadata declares AI generation. Metadata can be added or edited by anyone.',
  metadata_declared_composite: () => 'The file’s metadata declares a composite with AI-generated parts. Metadata can be added or edited by anyone.',
  metadata_parameters: () => 'The file’s metadata holds AI generation parameters. Metadata can be added or edited by anyone.',
}
const NOT: Record<MarkerKind, (one: string) => string> = {
  metadata: () => 'the same fields can be added to any file',
  watermark: (one) => `the same message can be embedded in any ${one}`,
  credentials_listed_signer: (one) => `the credentials name a generator and do not declare that it made the ${one}`,
  credentials_unlisted_signer_declared: () => 'credentials from a signer that is not on the trust list can be attached to any file',
  credentials_unlisted_signer_named: () => 'credentials from a signer that is not on the trust list can be attached to any file',
  metadata_declared: () => 'the same fields can be added to any file',
  metadata_declared_composite: () => 'the same fields can be added to any file',
  metadata_parameters: () => 'the same fields can be added to any file',
}
/** What stands before the count of checks in the fact "Basis". */
const BASIS_WORD: Record<MarkerKind, string> = {
  metadata: 'Self-declared metadata', watermark: 'Self-declared watermark message', credentials_listed_signer: 'Self-declared Content Credentials',
  credentials_unlisted_signer_declared: 'Self-declared Content Credentials', credentials_unlisted_signer_named: 'Self-declared Content Credentials',
  metadata_declared: 'Self-declared metadata', metadata_declared_composite: 'Self-declared metadata', metadata_parameters: 'Self-declared metadata',
}

export const markerHeadline = (marker: MarkerKind): string => HEADLINE[marker]
export const markerBasisWord = (marker: MarkerKind): string => BASIS_WORD[marker]
/** "What this means" and "What this does not mean" of a result that a marker decides; `one`: the kind of file in a word. */
export function markerGuidance(marker: MarkerKind, one: string): { means: string; doesNotMean: string } {
  return { means: MEANS[marker](one), doesNotMean: `It is not a finding that the ${one} is AI-generated: ${NOT[marker](one)}. The statistical checks are still reported below.` }
}

/** N-19h (E-21): where a record decides, every finding of that record entered the decision, with or without an override
 *  row (`view.of_record`): the findings of the record's signals. A stored result names no signals; there the finding of a
 *  marker that decides is the one of the family of self-declared metadata that points toward AI (a declaration's and a
 *  capture's findings are documentary, which entered already). */
export function ofRecord(f: { evidence_ids?: readonly string[] | null; family: string; direction?: string | null },
  d: DecisionLike | null | undefined): boolean {
  const record = recordOf(d)
  if (!record) return false
  const ids = record.signal_ids ?? []
  if (ids.length > 0) return (f.evidence_ids ?? []).some((id) => ids.includes(id))
  return record.kind === 'self_declared_ai' && f.family === 'F2_self_declared_metadata' && f.direction === 'for'
}

/** N-19h (E-20; `view.record_set_number`): the record that decides set the number the result publishes. Read from the
 *  result's numbers, not from the words of its reason: a record decides, a number is published, and it differs from the
 *  statistical one beside it or none stands beside it. A marker or a capture that did not move the estimate did not. */
export function recordSetNumber(d: NumbersLike | null | undefined): boolean {
  if (!d || !recordOf(d)) return false
  const published = d.probability_ai_calibrated ?? d.probability_ai_uncalibrated
  const statistical = d.statistical_estimate_before_provenance
  return published != null && (statistical == null || Math.abs(statistical - published) > 1e-9)
}

/** N-19l (R6-05; D-68 as amended; `view.alone_undecided`): a record decides, the number the result publishes is the
 *  statistical one, and the statistical checks alone are inconclusive, by the band or by D-76. The number is then no large
 *  numeral and no marker on the scale. A result stored before the decision said so keeps the form it had. */
export function aloneUndecided(d: AloneLike | null | undefined): boolean {
  if (!d || !recordOf(d) || recordSetNumber(d)) return false
  return (d.probability_ai_calibrated ?? d.probability_ai_uncalibrated) != null && d.statistical_alone?.code === 'inconclusive'
}

/** N-19h (E-20, 1): what set a number that a record set (`view.RECORD_SET`). A declaration shows no number of its own. */
export const RECORD_SET: Partial<Record<RecordKind, string>> = {
  self_declared_ai: 'the floor of self-declared evidence', trusted_capture: 'the cap of a trusted capture record',
}

/** N-19h (E-19; `view.record_alone`): a record decides alone: it decides the result, and the result holds no statistical
 *  number beside it (a check did not run; no check that scored has weight for the input; on the fallback no check of the
 *  content scored). Nothing then says that a statistical estimate is reported, that checks entered a model or that a
 *  model's estimate was adjusted. Read from what the result holds, so a stored result follows the same rule. */
export function recordAlone(d: NumbersLike | null | undefined): boolean {
  if (!d || !recordOf(d)) return false
  return (d.probability_ai_calibrated ?? d.probability_ai_uncalibrated) != null && d.statistical_estimate_before_provenance == null
}

/** Why no statistical number stands beside a record that decides alone, as a clause (`view.alone_why`): on the fallback no
 *  check of the content scored; a calibrated result says it in the note of its H1 assessment. Null where it does not say. */
const CONTENT_NAMES: Record<string, string> = { image: 'pixels', video: 'frames or the sound track', audio: 'sound', document: 'text' }
const ALONE_WHY: readonly (readonly [string, string])[] = [
  ['no estimate: a check that this analysis depends on did not run', 'a check that this analysis depends on did not run'],
  ['for an input like this one the fusion model gives the checks that scored no weight', 'for an input like this one the fusion model gives the checks that scored no weight'],
  ['no artefact detector produced a score', 'no check produced a score that can be used'],
]
export function aloneWhy(d: { fusion_mode?: string | null }, assessments: readonly AssessmentLike[] | null | undefined, kind: string): string | null {
  if (d.fusion_mode === 'fallback') return `no check of the ${CONTENT_NAMES[kind] ?? 'content'} produced a score`
  const note = (assessments ?? []).find((a) => a.hypothesis === 'H1_whole_content_synthesis')?.note ?? ''
  return ALONE_WHY.find(([prefix]) => note.startsWith(prefix))?.[1] ?? null
}

/** What stands in place of the breakdown where a record decides alone (`forensic.record_alone_text`). */
export function recordAloneText(d: DecisionLike & { fusion_mode?: string | null }, why: string | null): string {
  const none = `There is no statistical ${d.fusion_mode === 'fallback' ? 'score' : 'estimate'} to break down${why ? `: ${why}.` : '.'}`
  return recordOf(d)?.kind === 'decisive_ai' ? `${none} The provenance record decides the result.` : `${none} The number is the value Veritas gives the provenance record.`
}

/** N-19h (E-20, 2): a capture's result is the signed record's, not what the combined evidence indicates (`view.CAPTURE_MEANS`). */
export const CAPTURE_MEANS = 'A signed capture record that covers the file’s current bytes decides this result.'
/** N-19m (design review D3-01; D-15): a trusted capture that decides has a headline, a basis, a "does not mean" and a next
 *  step of its own, as a declaration and a marker have (`view.HEADLINE`, `CAPTURE_BASIS`, `CAPTURE_NOT`, `CAPTURE_STEP`,
 *  `CAPTURE_STEP_AGAINST`). It was headed by its decision code and read as a clean statistical result. */
export const CAPTURE_HEADLINE = 'The file’s Content Credentials record a capture of these bytes'
export const CAPTURE_BASIS = 'Documentary · Content Credentials (capture record)'
export const CAPTURE_NOT = 'The record states what the capture device declared. It does not show what was in front of the device: a screen or a print can be captured. A signing key can be misused.'
export const CAPTURE_STEP = 'Check who signed the capture record, then confirm where the file came from.'
export const CAPTURE_STEP_AGAINST = 'The statistical checks alone point the other way: weigh the record’s signer against them.'
/** N-19n (design review D4-01, P1; `review.STEP_NAMES`): what the examiner's layer names where it compares, in a result that a
 *  trusted capture decides. The page heads such a result by the record's sentence, and the layer said "the machine decision"
 *  and "the machine assessment" of a decision code that no surface prints there. In a list, where the row says "Capture
 *  record" beside it and the cell is narrow, the short name. */
export const STEP_OF_CAPTURE = 'the step of the capture record'
export const STEP_OF_CAPTURE_SHORT = 'the record’s step'
/** N-19o (design review D5-01; `review.STEP_NAMES`): on the fallback under a trusted declaration the code is the fourth step and
 *  the steps are counted from the fifth, the declaration's; the layer names that step, in a list by the short name above. */
export const STEP_OF_DECLARATION = 'the step of the declaration'
/** What a list says of a row that a capture decides, where it words a row from its code (the Analyses list, a batch page). */
export const CAPTURE_ROW = 'Capture record'

/** N-19p (design review D6-01, the rest of D5-01): what the lists say of a row that a trusted declaration decides on the
 *  fallback, where the code is the fourth step and the scale fills the declaration's, the fifth; and the case row prints the
 *  sentence that heads its page (`HEADLINE.documentary_declared`). The rows read the code's word ("Indications") beside
 *  "Examiner: …, on the record's step", and the record's step was printed nowhere on them. */
export const DECLARATION_ROW = 'AI declaration'
export const DECLARATION_HEADLINE = 'The file’s Content Credentials declare it was created with AI'

/** A row that a trusted declaration decides with a code short of the declaration's step (the fallback). A marker gives the
 *  same code and hypothesis, so the row says what decides it (`deciding_record`, from the stored result); with a fusion
 *  model the code is the fifth step and the row keeps its word. */
export function declarationRow(row: { decision_code?: string | null; deciding_record?: string | null }): boolean {
  return row.deciding_record === 'declaration' && row.decision_code != null && row.decision_code !== 'ai_supported_strong' && row.decision_code !== 'not_assessable'
}

/** N-19m: a list row holds the decision code and the driving hypothesis, not the record. A row that a record decides with
 *  a clean code is a capture's: a declaration and a marker give an AI code, and no other record gives a clean one. */
export function captureRow(row: { decision_code?: string | null; driving_hypothesis?: string | null }): boolean {
  return row.driving_hypothesis === 'H3_provenance_integrity' && (row.decision_code === 'no_ai_indicators' || row.decision_code === 'no_ai_indicators_strong')
}
