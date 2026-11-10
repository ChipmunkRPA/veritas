/** L1 answer and L2 key findings (UX §4.3, §5.3, §5.4, §7.1–7.4; D-16). Pure functions, so every sentence the answer module
 *  prints comes from a fixed template filled with values from the analysis (UX §7.3 rule 11). */
import type { AnalysisDetail, Finding, Signoff } from '../../lib/api/queries'
import { halfUp } from '../../lib/format'
import { knownSpeechReference } from './speechReference'
import { contextWords } from '../../lib/recordWords'
import { ALONE_OTHER_WAY, CAPTURE_HEADLINE, CAPTURE_MEANS, CAPTURE_NOT, CAPTURE_STEP, CAPTURE_STEP_AGAINST, DECLARATION_HEADLINE, STEP_OF_CAPTURE, STEP_OF_CAPTURE_SHORT, STEP_OF_DECLARATION, aloneInconclusive, aloneOtherWay, aloneNumber, markerGuidance, markerHeadline, markerKindOf, markerOf, ofRecord, outsideValidated, recordAlone, recordOf, recordSetNumber, statisticalWithheld, type MarkerKind } from './record'
import { AGREEMENT_NOT_WRITTEN, LEGACY_LITTLE, LN2, RUN_AGAIN, VOICE_STEP, direction, entered, littleOf, measuredFromNeutral, outsideWhy, type OutsideEstimate } from './vocabulary'

export type Decision = NonNullable<AnalysisDetail['decision']>
export type DecisionCode = Decision['code']
type Calibration = NonNullable<AnalysisDetail['calibration']>[number]
type ConditionProfile = NonNullable<AnalysisDetail['condition_profile']>
type Review = AnalysisDetail['review']

export type Basis = 'inferential' | 'documentary_declared' | 'documentary_metadata' | 'documentary_capture' | 'insufficient'
export type Direction = 'for' | 'against' | 'neutral'
type BandKey = 'very_likely_ai' | 'likely_ai' | 'uncertain' | 'likely_authentic' | 'very_likely_authentic'

const H3 = 'H3_provenance_integrity'

/** D-16: the headline follows the decision code, so a decision capped by the uncalibrated fallback or by abstention reads as
 *  its code, not as the band of the raw estimate. */
const CODE_BAND: Record<Exclude<DecisionCode, 'not_assessable'>, BandKey> = {
  ai_supported_strong: 'very_likely_ai', ai_supported: 'likely_ai', inconclusive: 'uncertain',
  no_ai_indicators: 'likely_authentic', no_ai_indicators_strong: 'very_likely_authentic',
}
const BAND_STEP: Record<BandKey, 1 | 2 | 3 | 4 | 5> = {
  very_likely_authentic: 1, likely_authentic: 2, uncertain: 3, likely_ai: 4, very_likely_ai: 5,
}

/** UX §7.1 L1 headlines, verbatim. */
export const HEADLINE: Record<BandKey | Exclude<Basis, 'inferential'>, string> = {
  very_likely_ai: 'Strong indications of AI generation or manipulation',
  likely_ai: 'Indications of AI generation or manipulation',
  uncertain: 'Inconclusive: the evidence does not support a firm assessment',
  likely_authentic: 'No significant indications of AI generation or manipulation',
  very_likely_authentic: 'No indications of AI generation or manipulation found',
  documentary_declared: DECLARATION_HEADLINE,  // N-19p (D6-01): the case row prints it too
  documentary_metadata: 'The file’s metadata names an AI generator',
  documentary_capture: CAPTURE_HEADLINE,  // N-19m (D3-01; D-15)
  insufficient: 'Chipmunk Verify could not assess this file reliably',
}

const MEDIA: Record<string, readonly [string, string]> = {
  image: ['image', 'images'], video: ['video', 'videos'], document: ['document', 'documents'], audio: ['recording', 'recordings'],
}
/** UX §7.1 `{confounders}`, per media type. */
const CONFOUNDERS: Record<string, string> = {
  image: 'heavy compression or re-saving, or beauty filters and HDR processing',
  video: 'heavy compression or re-saving, or dubbing and poor audio sync',
  audio: 'heavy compression or re-saving',
  document: 'templated or grammar-tool-edited writing',
}

export const media = (kind: string): readonly [string, string] => MEDIA[kind] ?? ['file', 'files']
const capitalise = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

/** Direction of a finding: the synthesis stage records it; older results only carry it in the title pattern. */
export function findingDirection(f: Finding): Direction {
  if (f.direction) return f.direction
  const m = /support (for|against)\b/.exec(f.title)
  return m ? (m[1] as Direction) : 'neutral'
}

/** Documentary findings carry the ◆ seal instead of the meter (UX §5.4): decisive credentials and the synthesis stage's fixed
 *  "Documentary:" titles (`app/engine/synthesis/engine.py::_title`). */
export const isDocumentary = (f: Finding) => f.strength === 'decisive' || f.title.startsWith('Documentary:')

// --- what a finding of the provenance families is (N-19n; design review D3-07; `view.is_marker`, `finding_mark`) ------------
// The finding of a self-declared marker is a record's, not a likelihood's: an open diamond stands where its meter stood,
// in the muted ink where the decision did not apply it, and its strength word is "self-declared", or "not applied". What
// a finding is, is read from what a result stores (it holds no variant), so a stored marker's finding loses its bars too.
const WATERMARK_CHECKS: readonly string[] = ['img.watermark.dwtdct']
const CREDENTIAL_CHECKS: readonly string[] = ['prov.c2pa', 'prov.c2pa.video', 'prov.c2pa.audio']
/** The check of the finding's first evidence id (a signal's id begins with its detector's: "prov.c2pa:global"). */
export const firstCheck = (f: Pick<Finding, 'evidence_ids'>): string => ((f.evidence_ids ?? [])[0] ?? '').replace(/^detector:/, '').split(':')[0]
/** A marker's finding: of self-declared metadata with strength strong (a mention's has none); or of provenance, strong and
 *  "for" and no documentary record, where the first check is a watermark decoder (the other is credentials that do not validate). */
export function isMarker(f: Finding): boolean {
  if (isDocumentary(f) || f.strength !== 'strong') return false
  if (f.family === 'F2_self_declared_metadata') return true
  return f.family === 'F1_provenance' && findingDirection(f) === 'for' && WATERMARK_CHECKS.includes(firstCheck(f))
}
/** N-19n (D3-07, 8; `view.is_invalid_credentials`): Content Credentials that do not validate: of provenance, strong and "for"
 *  and no documentary record, where the first check reads Content Credentials. No marker and no check. */
export const isInvalidCredentials = (f: Finding): boolean =>
  !isDocumentary(f) && f.family === 'F1_provenance' && f.strength === 'strong' && findingDirection(f) === 'for' && CREDENTIAL_CHECKS.includes(firstCheck(f))
/** What the result means says of them, under every headline and on both fusion paths (`view.invalid_sentence`). */
export function invalidSentence(findings: readonly Finding[]): string | null {
  const invalid = [...findings].sort((a, b) => a.rank - b.rank).find(isInvalidCredentials)
  return invalid ? `The file’s Content Credentials do not validate (${invalid.id}). A certificate, signer-trust, signature or content-binding check failed. This alone does not establish tampering, AI generation or originality.` : null
}
export type RecordMark = { level: 'documentary' | 'self_declared'; muted: boolean; word: string }
/** The mark in the meter's place, with its word; null for a finding with a meter. A documentary record that the decision
 *  did not apply (a capture under a declaration) keeps its word and takes the muted ink. */
export function markOf(f: Finding, d: Decision | null | undefined): RecordMark | null {
  if (isDocumentary(f)) {
    const ids = recordOf(d)?.signal_ids ?? []
    return { level: 'documentary', muted: ids.length > 0 && !(f.evidence_ids ?? []).some((id) => ids.includes(id)), word: 'documentary' }
  }
  if (isInvalidCredentials(f)) return { level: 'self_declared', muted: true, word: 'invalid' }  // (8): the muted outline
  if (!isMarker(f)) return null
  const applied = ofRecord(f, d)
  return { level: 'self_declared', muted: !applied, word: applied ? 'self-declared' : 'not applied' }
}
/** The word that the page sets beside a record's finding where no picture stands: the kind of the record (a watermark
 *  message is not "Credentials", and Content Credentials are not "Metadata"); null for any other finding. */
export function recordPlate(f: Finding): string | null {
  if (!isDocumentary(f) && !isMarker(f) && !isInvalidCredentials(f)) return null
  const check = firstCheck(f)
  return WATERMARK_CHECKS.includes(check) ? 'Watermark' : CREDENTIAL_CHECKS.includes(check) ? 'Credentials' : f.family === 'F2_self_declared_metadata' ? 'Metadata' : null
}

/** N-19h (E-21): the basis comes from the record the decision applied (`record.ts`), not from the findings. A trusted
 *  declaration has the declaration's basis and a marker the marker's. N-19m (design review D3-01; D-15): a trusted capture
 *  that decides has a basis of its own too, also when the file carries a marker that was not applied (it had none, so its
 *  result was headed by its decision code and read as a clean statistical one). A stored result that names its record
 *  nowhere keeps the reading by its findings. */
export function basisOf(d: Decision, findings: readonly Finding[]): Basis {
  if (d.code === 'not_assessable') return 'insufficient'
  const record = recordOf(d)
  if (record) {
    return record.kind === 'decisive_ai' ? 'documentary_declared' : record.kind === 'self_declared_ai' ? 'documentary_metadata'
      : record.kind === 'trusted_capture' ? 'documentary_capture' : 'inferential'
  }
  if (d.driving_hypothesis === H3) {
    if (findings.some((f) => f.strength === 'decisive' && findingDirection(f) === 'for')) return 'documentary_declared'
    if (findings.some((f) => f.family === 'F2_self_declared_metadata' && findingDirection(f) === 'for')) return 'documentary_metadata'
  }
  return 'inferential'
}

/** Verdict step of the headline colour and glyph; 0 = insufficient. */
export function stepOf(d: Decision, basis: Basis): 0 | 1 | 2 | 3 | 4 | 5 {
  if (basis === 'insufficient' || d.code === 'not_assessable') return 0
  if (basis === 'documentary_declared') return 5
  if (basis === 'documentary_metadata') return 4
  if (basis === 'documentary_capture') return 1  // N-19m (D-15): the first segment, where the record's 10 stands, on both fusion paths
  return BAND_STEP[CODE_BAND[d.code]]
}

export function headline(d: Decision, basis: Basis): string {
  if (basis === 'documentary_metadata') return markerHeadline(markerOf(d))  // N-19h: the words of the marker's kind
  if (basis !== 'inferential') return HEADLINE[basis]
  return d.code === 'not_assessable' ? HEADLINE.insufficient : HEADLINE[CODE_BAND[d.code]]
}

/** The calibrated estimate, else the fallback estimate (flagged uncalibrated by the facts). */
export const estimateOf = (d: Decision): number | null => d.probability_ai_calibrated ?? d.probability_ai_uncalibrated ?? null

/** Displayed estimate: whole numbers clamped to 1–99, with `<1` and `>99` as text (UX §5.3). */
export function numeral(p: number): string {
  const v = p * 100
  if (v < 1) return '<1'
  if (v > 99) return '>99'
  return String(halfUp(v))
}

export function interval(d: Decision): readonly [number, number] | null {
  return d.ci90 && d.ci90.length === 2 ? [halfUp(d.ci90[0] * 100), halfUp(d.ci90[1] * 100)] as const : null
}

/** ML §5.1 error-rate contexts that apply to the input, most specific first. Mirrors `synthesis/engine.py::_contexts`. */
export function errorRateContexts(kind: string, profile: ConditionProfile | null | undefined): string[] {
  const slice = profile?.slice_id ?? ''
  const face = slice.includes('/face') && !slice.includes('/noface')
  if (kind === 'image') {
    const out: string[] = []
    if (profile?.screenshot_likelihood) out.push('screenshot')
    if (slice.includes('/face<120')) out.push('small face (< 120 px)')
    else if (face) out.push('portrait face')
    if (slice.includes('/side<512')) out.push('low-res (< 512)')
    if (slice.includes('/q<70')) out.push('social-recompressed')
    return out
  }
  if (kind === 'video') return [face ? 'video talking head' : 'video T2V']
  // L-3: "audio lossy" only for a recording the condition stage found lossy (a FLAC or WAV file is not)
  return kind === 'audio' && (profile?.limiting_conditions ?? []).some((c) => c.startsWith('lossy compression')) ? ['audio lossy'] : []
}

/** Natural frequency (UX §7.4): whole numbers per 100, "fewer than 1 in 100", or "k of n" when fewer than 100 items were tested. */
export function frequency(rate: number, n: number): string {
  if (n < 100) return `${halfUp(rate * n)} of ${n}`
  const v = rate * 100
  return v > 0 && v < 0.5 ? 'fewer than 1 in 100' : `${halfUp(v)} in 100`
}

/** N-19i (R5-01; `view.counted_rows`): the rows of the checks that the decision counts, which is one count wherever a count is
 *  taken: the number line and the agreement, the count "in the estimate" and the sentence on how many checks entered the model.
 *  The runtime says it with the row (`counted`): not a row of zero, and not the row of a check that declares H3 (the container
 *  check of a video). A result published before N-19i has no such word: its rows count when they are not zero. */
export function countedRows(d: Pick<Decision, 'contributions'> | null | undefined): NonNullable<Decision['contributions']> {
  return (d?.contributions ?? []).filter((r) => r.kind === 'detector' && r.counted !== false && Math.abs(r.logit) > 1e-9)
}

/** N-19d: the count beside a result says what it counts ("Inferential · 14 checks" stood beside "Three checks entered the
 *  fusion model"). Under a fusion model with published rows: "14 checks ran, 3 in the estimate", the second number being
 *  the one of "… entered the fusion model" (N-19i: the rows the decision counts). The reports' `view.checks_counted`. */
export function checksCounted(coverage: Coverage, d: Pick<Decision, 'fusion_mode' | 'contributions'> | null | undefined, outside = false): string {
  const ok = coverage?.by_status?.ok ?? 0
  const ran = `${ok} ${ok === 1 ? 'check' : 'checks'} ran`
  const rows = d?.contributions ?? []
  if (d?.fusion_mode !== 'artefact' || rows.length === 0) return ran
  // N-19n (D3-09): outside what was validated no estimate of the checks is shown: "2 combined" (it read "2 in the estimate"
  // beside "No estimate is shown")
  return `${ran}, ${countedRows(d).length || 'none'} ${outside ? 'combined' : 'in the estimate'}`
}

/** UX §7.4 T1 + T2 (headline toward AI or inconclusive) or T3 (headline against), at the fuser's conformal 5 % threshold for the
 *  first matching context. Without a row the line says so; it never borrows another context's numbers. */
export function errorRateLine(kind: string, d: Decision, profile: ConditionProfile | null | undefined, calibration: readonly Calibration[]): string {
  const [one, many] = media(kind)
  const rows = calibration.filter((c) => c.subject.startsWith('fuser-')).flatMap((c) => c.error_rates ?? [])
  const contexts = errorRateContexts(kind, profile)
  for (const context of contexts) {
    const row = rows.find((r) => r.context === context && r.threshold_kind === 'conformal_0.05')
    if (!row) continue
    const scope = `${row.bench_version}, ${contextWords(context)} inputs`
    if (d.code === 'no_ai_indicators' || d.code === 'no_ai_indicators_strong') {
      return `Detectors miss some: in testing (${scope}), ${frequency(1 - row.tpr, row.n_pos)} AI-generated or manipulated ${many} scored below the flagging threshold.`
    }
    return `In testing (${scope}), ${frequency(row.fpr, row.n_neg)} genuine ${many} scored above the flagging threshold, and ${frequency(row.tpr, row.n_pos)} AI-generated or manipulated ${many} did.`
  }
  return contexts.length
    ? `Known error rates are not established: no evaluation is on record for ${contextWords(contexts[0])} inputs.`
    : `Known error rates are not established: no evaluation is on record for this kind of ${one}.`
}

export interface Guidance {
  means: string[]
  doesNotMean: string[]
}

/** L-3: `docs/limitations.md`, "Synthetic voice" (reports/view.py `AUDIO_MISS`). */
export const AUDIO_MISS = 'Cloned voices are often missed: in testing, only 1 of 89 cloned voices recorded in the wild was flagged.'

/** N-14a (reports/view.py, the same sentences): checks that were skipped for a reason of the server and not of the file, its
 *  model memory budget or model files that are not installed. `coverage.not_run_on_server` counts them by reason. */
type Coverage = AnalysisDetail['coverage'] | null | undefined
const SERVER_REASONS = ['memory budget', 'model not installed'] as const
export const ON_THIS_SERVER = 'could not run on this server'
export const SERVER_FEWER = 'This result rests on fewer checks than the profile holds.'
const OBTAIN_ANOTHER = ' Obtain the original file or another source.'

/** Whether a detector run was skipped for a reason of the server (`detector.of_the_server`). */
export const ofTheServer = (status: string | null | undefined, reason: string | null | undefined): boolean =>
  status === 'skipped' && (SERVER_REASONS as readonly string[]).includes(reason ?? '')

export function notRunOnServer(coverage: Coverage): { memory: number; files: number; total: number } {
  const found = coverage?.not_run_on_server ?? {}
  const count = (reason: string) => { const n = Number(found[reason] ?? 0); return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0 }
  const memory = count(SERVER_REASONS[0])
  const files = count(SERVER_REASONS[1])
  return { memory, files, total: memory + files }
}

/** N-19n (the browser QA's lead 3; `view.not_run_of_server`): a result without an estimate because a check did not run (D-49),
 *  where the checks that did not run are those that could not run on this server. The reason of such a result names each check
 *  with why: "… did not run: A (could not run on this server: memory budget); B (failed: …). No estimate …". The top block of
 *  an image's page then says so as the document's does: in the line under the headline and in one sentence under "What this
 *  means", not in three. */
const NOT_RUN_REASON = /^Capped at inconclusive: a check that this analysis depends on did not run: ([\s\S]+)\. No estimate is published while a check did not run\.$/
export const NOT_RUN_MEANS = 'A check that this analysis depends on did not run, so no estimate is published.'
export const SERVER_NO_ESTIMATE = 'No estimate is published without every check that this analysis depends on.'
export function notRunOfServer(d: Decision | null | undefined, coverage: Coverage): boolean {
  if (!d || d.fusion_mode !== 'artefact' || d.code === 'not_assessable' || estimateOf(d) != null || serverCount(coverage) === null) return false
  const named = NOT_RUN_REASON.exec(d.reason ?? '')
  return named !== null && named[1].split('; ').every((part) => part.includes(ON_THIS_SERVER))
}

/** "3 of the 4 checks could not run on this server"; null when no check was skipped for a reason of the server. */
export function serverCount(coverage: Coverage): string | null {
  const n = notRunOnServer(coverage).total
  const total = coverage?.detectors_total ?? 0
  if (n === 0) return null
  if (n >= total) return n === 1 ? 'The one check of this analysis could not run on this server' : `None of the ${n} checks could run on this server`
  return `${n} of the ${total} checks could not run on this server`
}

/** The count with the server's reason: its memory budget, model files that are not installed, or both. */
export function serverSentence(coverage: Coverage): string | null {
  const head = serverCount(coverage)
  if (head === null) return null
  const { memory, files } = notRunOnServer(coverage)
  if (memory && files) return `${head}: its memory budget is too small for ${memory}, and the model files of ${files} are not installed.`
  if (memory) return `${head}: its memory budget is too small for ${memory === 1 ? 'it' : 'them'}.`
  return `${head}: ${files === 1 ? 'its' : 'their'} model files are not installed.`
}

export function serverStep(coverage: Coverage): string | null {
  const n = notRunOnServer(coverage).total
  return n === 0 ? null : `Ask the administrator of this server to make ${n === 1 ? 'this check' : 'these checks'} able to run, then analyse the file again.`
}

const lowerFirst = (s: string): string => s.charAt(0).toLowerCase() + s.slice(1)

/** UX §7.1 "What this means / What this does not mean" templates; the natural-frequency line ends "What this means" (§5.3), or
 *  the does-not-mean column when it is the T3 miss rate. The insufficient variant's reasons render as a list beside it.
 *  N-14a: a result in which checks could not run on this server says so (the insufficient variant among its reasons), and is
 *  not told to obtain another file where the server is what was short. */
export function guidance(kind: string, d: Decision, basis: Basis, rateLine: string, coverage?: Coverage, findings: readonly Finding[] = [], signals: readonly FlagSignal[] = [],
  profile?: ConditionProfile | null, caveats?: readonly string[] | null): Guidance {
  const base = baseGuidance(kind, d, basis, rateLine, profile)
  // D-63: an AI headline no check supports; N-18 (QA P1-1): a clean headline over a flagged check outside the estimate
  // N-19l (R6-03, 1): a marker that a capture's decision did not apply, on both fusion paths
  // N-19l (R6-05): D-76's sentence in its form beside a record, after the record's own, whatever the basis
  // N-19m: a capture's result has a basis of its own; a check outside the estimate that flagged the file is still named
  // under it, and so is a marker that its decision did not apply (`view.READ_AS_STATISTICAL`)
  // N-19m (D-77): a result gives one reason, D-77's or D-76's
  // N-19n (supervisor check 29): under the caveat of a configured prior the sentence of a condition that took the code is
  // followed by the sentence on the prior, with the caveat's share (`view.PRIOR_ADJUSTED`)
  const share = priorShare(caveats)
  const withPrior = (sentence: string | null, start: boolean) => (sentence === null || share === null ? sentence : `${sentence} ${priorAdjusted(share, start)}`)
  const own = basis !== 'inferential' ? null
    : unsupportedAiSentence(d) ?? withPrior(unsupportedSentence(d), true) ?? withPrior(offScaleSentence(d), false)
  const beside = withPrior(unsupportedBesideRecord(d, profile), true)
  // N-19q (finding 1): after D-76's sentence in its first form, the sentence on a check's other readings (`view._readings_own`)
  const ownReadings = basis === 'inferential' && unsupportedAiSentence(d) === null && unsupportedSentence(d) !== null ? readingsOwn(d, findings) : null
  const besideReadings = beside !== null ? readingsBeside(d, findings) : null
  // N-19o (browser QA Q5-02; `view.PRIOR_SHARE`): wherever the number of a result under a configured prior is printed, one
  // sentence gives the share, once: the sentences of D-76 and D-77 above state it already
  const whose = share !== null && ![own, beside].some((s) => s?.includes(PRIOR_SAID)) ? priorNumberPrinted(d, basis, profile) : null
  const priorSaid = share !== null && whose !== null ? priorShareSentence(share, whose === 'beside') : null
  const added = [beside, besideReadings, own, ownReadings, priorSaid,
    ...(basis === 'inferential' || basis === 'documentary_capture'
      ? [flaggedOutsideSentence(kind, d, findings, signals), unappliedSentence(kind, d, findings, signals)] : []),
    // N-19n (D3-07, 8): credentials that do not validate, under every headline and on both fusion paths
    d.code !== 'not_assessable' ? invalidSentence(findings) : null].filter((s): s is string => s !== null)
  const g = added.length ? { ...base, means: [base.means[0], ...added, ...base.means.slice(1)] } : base
  const server = serverSentence(coverage)
  if (server === null || basis === 'insufficient') return g
  if (notRunOfServer(d, coverage)) {  // N-19n: one sentence, with the server's reason
    return {
      means: [`${server} ${SERVER_NO_ESTIMATE}`, ...g.means.filter((text) => text !== NOT_RUN_MEANS)],
      doesNotMean: g.doesNotMean.map((text) => (text.endsWith(OBTAIN_ANOTHER) ? text.slice(0, -OBTAIN_ANOTHER.length) : text)),
    }
  }
  return {
    means: [...g.means, `${server} ${SERVER_FEWER}`],
    doesNotMean: g.doesNotMean.map((text) => (text.endsWith(OBTAIN_ANOTHER) ? text.slice(0, -OBTAIN_ANOTHER.length) : text)),
  }
}

/** D-63 (N-17; browser QA P0-1): an AI headline under which no check gives at least weak support says where the estimate comes
 *  from: the starting point, the fitted term of a check that could not score, and checks that each had little effect
 *  (`view.unsupported_ai_sentence`). Null for any other result, and for a result published before N-17. */
export function unsupportedAiSentence(d: Decision): string | null {
  if (!measuredFromNeutral(d) || !(d.code === 'ai_supported' || d.code === 'ai_supported_strong') || estimateOf(d) == null) return null
  if ((d.contributions ?? []).some((r) => r.kind === 'detector' && r.logit >= LN2)) return null
  return `No check gave at least weak support for it on its own. The estimate comes from ${comesFrom(d, 1)}.`
}

const joined = (parts: string[]) => (parts.length === 1 ? parts[0] : `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`)

/** The starting point of the model for the input, and the fitted terms of checks that could not score on the side `sign`
 *  (+1 toward AI, −1 against), counted: what an estimate stands on before any check that scored (`view._start_and_terms`). */
function startAndTerms(d: Decision, sign: 1 | -1): string[] {
  const rows = d.contributions ?? []
  const start = rows.filter((r) => r.kind === 'neutral_start' || r.kind === 'context_prior').reduce((sum, r) => sum + r.logit, 0)
  const p = 1 / (1 + Math.exp(-start))
  const where = p >= 0.995 ? 'above 99' : p < 0.005 ? 'below 1' : `${halfUp(p * 100)}`
  const parts = [`the starting point of ${where} out of 100 for inputs like this one`]
  const missing = rows.filter((r) => r.kind === 'missing' && sign * r.logit > 0).length
  if (missing) parts.push(missing === 1 ? 'the fitted term of a check that could not score' : `${missing} fitted terms of checks that could not score`)
  return parts
}

/** Where an estimate that no check supports comes from, on the side `sign` (+1 toward AI, −1 against): the starting point, the
 *  fitted terms of checks that could not score, and the checks that each had little effect (`view._comes_from`). */
function comesFrom(d: Decision, sign: 1 | -1): string {
  const parts = startAndTerms(d, sign)
  const rows = countedRows(d)  // N-19i: the checks that count
  // N-19n (D4-02 … D4-05, point 3; `view._comes_from`): the checks of little effect in either direction, as the breakdown
  // counts them, and a check of ln 2 or more against the answer named as one that pushed the other way
  const against = rows.filter((r) => sign * r.logit <= -LN2).length
  const small = rows.filter((r) => Math.abs(r.logit) < LN2).length
  if (against) parts.push(against === 1 ? 'one check that pushed the other way' : `${against} checks that pushed the other way`)
  if (small) parts.push(small === 1 ? 'one check that had little effect' : `${small} checks that each had little effect`)
  return joined(parts)
}

/** N-19m (D-76, the second condition; `view._rests_on`): the sentence on what an estimate comes from counts only checks of
 *  little effect; where checks moved it, it is said to rest on the start. N-19n (supervisor check 29): and on the fitted
 *  terms of the checks that could not score, on the side of the answer: on the video model a result holds such terms beside
 *  a start of 69, and the sentence named the start alone. */
function restsOn(d: Decision, sign: 1 | -1): string {
  return joined(startAndTerms(d, sign))
}

/** N-19i (D-76; `abstain.unsupported_side`, word for word: a backend test reads these lines): a result that is inconclusive
 *  because no check gives at least weak support for the answer of its estimate. The reason of such a result ends with the
 *  answer and these words; the side is "for" or "against" AI generation or manipulation. Null for any other result.
 *  N-19m (D-77; D-76 as amended; `abstain.RULE_REASONS`): the three endings of the conditions of the third part stand under
 *  the same pin. Without them the page shows such a result as a plain inconclusive one. `unsupported` is D-76 in either
 *  form (no single check, the checks together); a result that D-77 took is `offScale`. */
export const NO_SUPPORT_REASON = "is the answer of the estimate, and no check gives at least weak support for it: the estimate rests on the model's starting point for inputs like this one, so the result is inconclusive."
export const OFF_SCALE_REASON = 'is the answer of the calibrated levels for inputs like this one, but the estimate stands in the middle part of the scale, so the result is inconclusive.'
/** N-19n (supervisor check 29; `abstain.OFF_SCALE_OTHER_REASON`): under a configured prior the published estimate of a result
 *  that D-77 took can stand in the part of the scale of the other answer; the reason of such a result has this ending. */
export const OFF_SCALE_OTHER_REASON = "is the answer of the calibrated levels for inputs like this one, but the estimate stands in the other answer's part of the scale, so the result is inconclusive."
export const TOGETHER_LESS_REASON = 'is the answer of the estimate, and a check gives at least weak support for it, but the checks together give it less than weak support, so the result is inconclusive.'
export const TOGETHER_AGAINST_REASON = 'is the answer of the estimate, and a check gives at least weak support for it, but the checks together point the other way, so the result is inconclusive.'
const NO_SUPPORT_ANSWER = { for: 'AI generation or manipulation', against: 'No AI generation or manipulation' } as const
export type Rule = 'scale' | 'no_check' | 'together_less' | 'together_against'
const RULE_REASONS: readonly (readonly [Rule, string])[] = [['scale', OFF_SCALE_REASON], ['scale', OFF_SCALE_OTHER_REASON], ['no_check', NO_SUPPORT_REASON],
  ['together_less', TOGETHER_LESS_REASON], ['together_against', TOGETHER_AGAINST_REASON]]
type Reasoned = Pick<Decision, 'code' | 'reason' | 'fusion_mode'> | null | undefined

/** N-19m (`abstain.rule_taken`): the condition that took the decisive code of an inconclusive result's estimate and the side of
 *  the answer that the result does not give, read from the end of its reason. Null for any other result. */
export function ruleTaken(d: Reasoned): { rule: Rule; side: 'for' | 'against' } | null {
  if (!d || d.code !== 'inconclusive' || d.fusion_mode !== 'artefact') return null
  const reason = d.reason ?? ''
  // matched from the sentence's first word: "No AI generation …" ends as "AI generation …" does
  for (const [rule, ending] of RULE_REASONS) {
    const side = (['for', 'against'] as const).find((s) => reason.endsWith(`. ${NO_SUPPORT_ANSWER[s]} ${ending}`))
    if (side) return { rule, side }
  }
  return null
}

export function unsupported(d: Reasoned): 'for' | 'against' | null {
  const taken = ruleTaken(d)
  return taken !== null && taken.rule !== 'scale' ? taken.side : null
}

/** N-19m (D-77; `view.off_scale`): the side of the answer that the calibrated levels would have given, in a result that is
 *  inconclusive because its estimate stands in the middle part of the scale. Null for any other result. */
export function offScale(d: Reasoned): 'for' | 'against' | null {
  const taken = ruleTaken(d)
  return taken !== null && taken.rule === 'scale' ? taken.side : null
}

/** N-19n (`abstain.part_of_scale`): the part of the scale in which an estimate as published prints, in whole points: "against"
 *  for the clean answer's part (34 or less), "for" for the AI answer's (65 or more), null for the middle part (35 to 64). */
export function partOfScale(estimate: number): 'for' | 'against' | null {
  const points = halfUp(Math.round(estimate * 1e6) / 1e4)
  return points <= 34 ? 'against' : points >= 65 ? 'for' : null
}

/** N-19n (supervisor check 29; `view.off_scale_part`): in a result that D-77 took, the part of the scale of the other answer in
 *  which its published estimate prints; null where it prints in the middle part, and for any other result. Read from where
 *  the estimate prints and not from the end of the reason: a result stored with the ending of the middle part reads right too. */
export function offScalePart(d: Decision): 'for' | 'against' | null {
  const side = offScale(d)
  const p = estimateOf(d)
  if (side === null || p == null) return null
  const part = partOfScale(p)
  return part !== side ? part : null
}

/** The owner of a part of the scale and its bound, by the side of that part (`view.ANSWERS_PART`, `PART_BOUND`). */
export const ANSWERS_PART = { for: 'the AI answer’s part of the scale', against: 'the clean answer’s part of the scale' } as const
const PART_BOUND = { for: '65 or more', against: '34 or less' } as const

/** N-19m (D-77; `view.off_scale_sentence`): what such a result says where "What this means" stands. It does not quote the
 *  headline that the result does not have. N-19n: under a configured prior the estimate can stand in the other answer's
 *  part of the scale (it said "the middle part of the scale (35 to 64)" of a printed 21). */
export function offScaleSentence(d: Decision): string | null {
  const side = offScale(d)
  const p = estimateOf(d)
  if (side === null || p == null) return null
  const [read, bound] = side === 'for' ? ['indications', '65 or more'] : ['no significant indications', '34 or less']
  const part = offScalePart(d)
  if (part !== null) return `The estimate itself, ${numeral(p)} out of 100, stands in ${ANSWERS_PART[part]} (${PART_BOUND[part]}). The calibrated levels for inputs like this one would have read ${read}; a headline of that answer also needs an estimate of ${bound}.`
  return `The estimate itself, ${numeral(p)} out of 100, stands in the middle part of the scale (35 to 64), where Chipmunk Verify gives no answer. The calibrated levels for inputs like this one would have read ${read}; a headline of that answer also needs an estimate of ${bound}.`
}

/** N-19n (supervisor check 29; `abstain.PRIOR_CAVEAT`, word for word: a backend test reads these lines): the caveat of a
 *  configured prior, which the result holds with the share between these words. It is known by its fixed words, not by its
 *  number: a prior of 0.004 prints "0 %", and one of 0.5 writes the caveat and shifts nothing. */
export const PRIOR_CAVEAT_HEAD = 'Probabilities assume that '
export const PRIOR_CAVEAT_TAIL = ' of inputs like this are AI-generated or manipulated (configured prior); without it they assume 50:50.'

/** The share of a configured prior as the result's caveat prints it ("10 %"; `view.prior_share`); null without that caveat. */
export function priorShare(caveats: readonly string[] | null | undefined): string | null {
  for (const text of caveats ?? []) {
    if (text.startsWith(PRIOR_CAVEAT_HEAD) && text.endsWith(PRIOR_CAVEAT_TAIL) && text.length > PRIOR_CAVEAT_HEAD.length + PRIOR_CAVEAT_TAIL.length)
      return `${text.slice(PRIOR_CAVEAT_HEAD.length, -PRIOR_CAVEAT_TAIL.length).replace(/%$/, '').trim()} %`
  }
  return null
}

/** N-19o (browser QA Q5-02; `view.PRIOR_SHARE`, `PRIOR_SAID`, `prior_number_printed`): the share beside a printed number, and
 *  whether a number that the prior shifts is printed: the result's own (not a record's value, nothing outside what was
 *  validated), or the statistical checks' beside a record where it is shown. */
export const PRIOR_SAID = 'this server is set to assume'
// N-19p (browser QA Q6-02; `view.PRIOR_SHARE_CHECKS`): beside a record the number that the prior shifts is the statistical checks'
export const priorShareSentence = (share: string, checks = false) =>
  `The ${checks ? 'statistical checks\' estimate' : 'estimate'} is adjusted for the share of AI-generated or manipulated inputs that this server is set to assume, ${share}.`
function priorNumberPrinted(d: Decision, basis: Basis, profile?: ConditionProfile | null): 'own' | 'beside' | null {
  if (d.code !== 'not_assessable' && basis !== 'documentary_declared' && estimateOf(d) != null && !recordSetNumber(d) && !outsideValidated(d, profile))
    return 'own'
  return d.statistical_estimate_before_provenance != null && !statisticalWithheld(d, profile) ? 'beside' : null
}

/** N-19n (`view.PRIOR_ADJUSTED`, `PRIOR_ADJUSTED_START`): why the levels and the estimate part ways under a configured prior.
 *  `start`: after a sentence of D-76, which names the starting point: it is shifted too. */
export const priorAdjusted = (share: string, start: boolean) =>
  `The estimate ${start ? 'and its starting point are' : 'is'} adjusted for the share of AI-generated or manipulated inputs that this server is set to assume, ${share}; the calibrated levels are not.`

/** N-19i (D-76; `view.unsupported_sentence`): what such a result says where "What this means" stands: which way its estimate
 *  points, that no check supports that answer, and where the estimate comes from. Null for any other result. */
export function unsupportedSentence(d: Decision): string | null {
  const side = unsupported(d)
  if (side === null || estimateOf(d) == null || !hasContributionRows(d)) return null
  const answer = side === 'for' ? 'toward AI generation or manipulation' : 'away from AI generation or manipulation'
  const rule = ruleTaken(d)?.rule  // N-19m: the second condition of D-76 has two forms of its own
  const sign = side === 'for' ? 1 : -1
  if (rule === 'together_less') return `The estimate alone points ${answer}, but taken together the checks gave less than weak support for that, so the result is inconclusive: what one check adds, another takes away. The estimate rests on ${restsOn(d, sign)}.`
  if (rule === 'together_against') return `The estimate alone points ${answer}, but taken together the checks point the other way, so the result is inconclusive. The estimate rests on ${restsOn(d, sign)}.`
  return `The estimate alone points ${answer}, but ${NO_PART} toward that answer, so the result is inconclusive. The estimate comes from ${comesFrom(d, sign)}.`
}

/** N-19q (N-20b's read of the reports, finding 1; `view.NO_PART`): D-76 reads each check's row in the estimate (D-63), and the
 *  sentence said "no check gave at least weak support for that on its own" beside the finding "Taken alone, the score is weak
 *  support for manipulation (likelihood ratio about 2.0)" and a check outside the estimate at moderate support. */
export const NO_PART = "no check's part in this estimate reached weak support (a factor of 2)"

/** N-19q (finding 1; `view.other_readings`): the checks of a result that D-76 took whose reading other than their part in the
 *  estimate gives the answer `side` at least weak support: taken alone (a check that entered, by the stand-alone words its
 *  finding stores) and outside the estimate (a check that did not enter, by its finding's own words). A check that declares H3
 *  is not counted by the decision and is left out; a result stored before N-19q stores no stand-alone words. */
export function otherReadings(d: Decision, findings: readonly Finding[], side: 'for' | 'against'): { alone: number; outside: number } {
  const alone = findings.filter((f) => !NOT_KEY.has(f.family) && !PROVENANCE_FINDINGS.has(f.family) && f.hypothesis !== H3 && enteredDecision(f, d)
    && f.standalone_direction === side && f.standalone_strength != null && f.standalone_strength !== 'none').length
  const outside = contextFindings(findings, d).filter((f) => f.hypothesis !== H3 && findingDirection(f) === side).length
  return { alone, outside }
}

/** N-19q (finding 1; `view.other_readings_sentence`): beside the sentence of D-76 in its first form, one sentence where a check
 *  taken alone or a check outside the estimate gives the answer weak support or more: such a reading does not decide the
 *  result, and why. Null where no reading does. */
export function otherReadingsSentence(d: Decision, findings: readonly Finding[], side: 'for' | 'against', within = 'this estimate'): string | null {
  const { alone, outside } = otherReadings(d, findings, side)
  const scores = alone === 1 ? "one check's score gives" : `the scores of ${alone} checks give`
  if (alone && outside) {
    const checks = outside === 1 ? 'does one check' : `do ${outside} checks`
    return `Taken alone, ${scores} that answer weak support or more, and so ${checks} outside ${within}, but neither reading decides the result here: a check's part in ${within} is small, and a check outside it carries no weight.`
  }
  if (alone === 1) return `Taken alone, ${scores} that answer weak support or more, but its part in ${within} is small, so that reading does not decide the result here.`
  if (alone) return `Taken alone, ${scores} that answer weak support or more, but their parts in ${within} are small, so those readings do not decide the result here.`
  if (outside === 1) return `One check outside ${within} gives that answer weak support or more, but it carries no weight, so that reading does not decide the result here.`
  if (outside) return `${outside} checks outside ${within} give that answer weak support or more, but they carry no weight, so those readings do not decide the result here.`
  return null
}

function readingsOwn(d: Decision, findings: readonly Finding[]): string | null {
  const side = unsupported(d)
  const rule = ruleTaken(d)?.rule
  return side === null || rule === 'together_less' || rule === 'together_against' ? null : otherReadingsSentence(d, findings, side)
}

function readingsBeside(d: Decision, findings: readonly Finding[]): string | null {
  const side = d.statistical_alone?.unsupported
  const rule = d.statistical_alone?.rule
  if ((side !== 'for' && side !== 'against') || rule === 'together_less' || rule === 'together_against') return null
  return otherReadingsSentence(d, findings, side, 'that estimate')
}

/** N-19n (the design reviewer's read of the sentences of N-19m's third part; `view.BESIDE_OPENS`): beside a record the three
 *  sentences of D-76 open alike. The first form opened "The statistical checks alone point away from AI generation or
 *  manipulation, but no check gave at least weak support for that on its own", which says of the checks that they point and
 *  that none of them does. Under a capture the sentence follows the sentence of the number ("The statistical checks alone do
 *  not decide: their estimate is …"), so it begins with the pronoun, and the sentence after it with "It". */
export const BESIDE_OPENS = ['The estimate of the statistical checks alone points', 'Their estimate'] as const
export const BESIDE_OPENS_AFTER_NUMBER = ['Their estimate points', 'It'] as const
/** What follows "…, but " in each form (`view.BESIDE_HOW`, `BESIDE_NO_CHECK`); the form of less than weak support keeps its
 *  last words beside a record too. */
export const BESIDE_HOW = {
  together_less: 'taken together the checks gave less than weak support for that: what one check adds, another takes away',
  together_against: 'taken together the checks point the other way',
} as const
export const BESIDE_NO_CHECK = "no check's part in it reached weak support (a factor of 2) toward that answer"  // N-19q (finding 1)

/** N-19l (R6-05; D-76 in its form beside a record; `view.unsupported_beside_record`): where a record decides and D-76 took
 *  the decisive code of the statistical estimate away: which way the statistical checks alone point, that no check
 *  supports that on its own, and where their estimate comes from. It does not say that the result is inconclusive: the
 *  result is the record's. N-19m (R7-01): said wherever their number is shown, whether the record set the result's number
 *  or not (it was said only where the record did not set it). Null for any other result, where no statistical number
 *  stands beside the record, and outside what was validated. */
export function unsupportedBesideRecord(d: Decision, profile?: ConditionProfile | null): string | null {
  const side = d.statistical_alone?.unsupported
  if ((side !== 'for' && side !== 'against') || !recordOf(d) || d.statistical_estimate_before_provenance == null) return null
  if (estimateOf(d) == null || !hasContributionRows(d) || outsideValidated(d, profile)) return null
  const answer = side === 'for' ? 'toward AI generation or manipulation' : 'away from AI generation or manipulation'
  // N-19m (code review, the read of the third part): the sentence follows the condition; under the second one a check does
  // support the answer. A result stored before the third part names no condition and reads as the first form
  const rule = d.statistical_alone?.rule
  const sign = side === 'for' ? 1 : -1
  const [opens, then] = recordOf(d)?.kind === 'trusted_capture' ? BESIDE_OPENS_AFTER_NUMBER : BESIDE_OPENS  // after `statisticalSentence`
  if (rule === 'together_less' || rule === 'together_against') return `${opens} ${answer}, but ${BESIDE_HOW[rule]}. ${then} rests on ${restsOn(d, sign)}.`
  return `${opens} ${answer}, but ${BESIDE_NO_CHECK}. ${then} comes from ${comesFrom(d, sign)}.`
}

/** N-18 (browser QA P1-1): what a flagged check outside the estimate looked at, by the family of its finding (`view.FLAGGED_SUBJECT`). */
const FLAGGED_SUBJECT: Record<string, string> = {
  A1_voice: 'the sound track', V1_face_track: 'a face track', V2_lip_sync: 'the lip movement', V3_synthetic_video: 'the frames',
  V4_codec: 'the encoding', F3_whole_image_generation: 'the image', F4_localised_manipulation: 'a region of the image',
  F7_duplication: 'a region of the image', F10_face: 'a face', T1_ai_text: 'the text',
}

/** Odds of 10 to 1, the lower edge of "moderate" for a score that is not calibrated (`view.FLAGGED_RAW_SCORE`). */
const FLAGGED_RAW_SCORE = 10 / 11

/** The score of a check whose stand-alone fit in the model says nothing at any score (a likelihood ratio of exactly 1) while
 *  the score itself is at least 10 to 1 for manipulation (`view._flat_raw_score`). Null for any other check. */
function flatRawScore(f: Finding, signals: ReadonlyMap<string, FlagSignal>): number | null {
  for (const id of f.evidence_ids ?? []) {
    const s = signals.get(id)
    if (!s || s.raw_kind !== 'probability' || s.raw_score == null || s.log_lr == null) continue
    if (Math.abs(s.log_lr) <= 1e-9 && s.raw_score >= FLAGGED_RAW_SCORE) return s.raw_score
  }
  return null
}

/** What the rule reads of a signal (a `SignalV2` has these fields). */
export interface FlagSignal { id: string; raw_score?: number | null; raw_kind?: string | null; log_lr?: number | null; outside_estimate?: OutsideEstimate | null;
  provenance_override?: string | null; detector_id?: string | null; metrics?: Record<string, unknown> | null }

/** The checks outside the estimate that flagged the file, under a clean headline of a calibrated result (`view.flagged_outside`):
 *  a finding with at least moderate stand-alone support for manipulation, or, where the model's stand-alone fit of the check
 *  says nothing, a score of the check itself of at least 10 to 1. Empty for any other headline. */
export function flaggedOutside(d: Decision, findings: readonly Finding[], signals: readonly FlagSignal[] = []): { finding: Finding; raw: number | null }[] {
  if (SIDE[d.code] !== 'against' || !hasContributionRows(d)) return []
  const byId = new Map(signals.map((s) => [s.id, s]))
  const out: { finding: Finding; raw: number | null }[] = []
  for (const f of [...findings].sort((a, b) => a.rank - b.rank)) {
    if (NOT_KEY.has(f.family) || enteredDecision(f, d)) continue
    // N-19l (R6-03, 1): a provenance record is no check, and it flags no part of the file (`unappliedSentence`). N-19o (R9-02):
    // known by its family, as `provenanceAside` knows it: where the request for the signals failed, the marker under a capture
    // read "A check of part of the file (F4) gives strong support for manipulation on its own …"
    if (PROVENANCE_FINDINGS.has(f.family) || (f.evidence_ids ?? []).some((id) => byId.get(id)?.provenance_override)) continue
    if (isInvalidCredentials(f)) continue  // N-19n (D3-07, 8): nor are credentials that do not validate (`invalidSentence`)
    if (findingDirection(f) === 'for' && (f.strength === 'moderate' || f.strength === 'strong')) out.push({ finding: f, raw: null })
    else if (f.strength === 'none') {
      const raw = flatRawScore(f, byId)
      if (raw !== null) out.push({ finding: f, raw })
    }
  }
  return out
}

/** N-19l (R6-03, 1; `view.UNAPPLIED_MARKER`): what a self-declared marker that the decision did not apply is, by its kind. */
const UNAPPLIED_MARKER: Record<MarkerKind, (one: string) => string> = {
  metadata: () => 'the file’s metadata names an AI generator',
  watermark: (one) => `the ${one} carries a public AI-generator watermark message`,
  credentials_listed_signer: () => 'the file’s Content Credentials name an AI generator',
  credentials_unlisted_signer_declared: () => 'the file’s Content Credentials declare AI generation, from a signer that is not on the trust list',
  credentials_unlisted_signer_named: () => 'the file’s Content Credentials name an AI generator, from a signer that is not on the trust list',
  metadata_declared: () => 'the file’s metadata declares AI generation',
  metadata_declared_composite: () => 'the file’s metadata declares a composite with AI-generated parts',
  metadata_parameters: () => 'the file’s metadata holds AI generation parameters',
}

/** N-19l (R6-03, 1; `view.unapplied_markers`): under a trusted capture that decides, the self-declared markers that the
 *  file also carries: the kind of each and the id of its finding, in the order of the findings. Empty under any other
 *  result: under a declaration a marker points the way of the headline, and nothing more is said of it. */
export function unappliedMarkers(d: Decision, findings: readonly Finding[], signals: readonly FlagSignal[] = []): { marker: MarkerKind; id: string }[] {
  if (recordOf(d)?.kind !== 'trusted_capture') return []
  const byId = new Map(signals.map((s) => [s.id, s]))
  const out: { marker: MarkerKind; id: string }[] = []
  for (const f of [...findings].sort((a, b) => a.rank - b.rank)) {
    const marker = markerKindOf((f.evidence_ids ?? []).map((id) => byId.get(id)).filter((s): s is FlagSignal => s?.provenance_override === 'self_declared_ai'))
    if (marker) out.push({ marker, id: f.id })
  }
  return out
}

/** N-19l (R6-03, 1; `view.unapplied_sentence`): what the result means says of every record that the decision did not apply
 *  and that points the other way than the headline what it is, in the words of its kind, and that the signed capture
 *  record outranks it. It read N-18's sentence of a check without weight; on the fallback nothing was said. */
export function unappliedSentence(kind: string, d: Decision, findings: readonly Finding[], signals: readonly FlagSignal[] = []): string | null {
  const markers = unappliedMarkers(d, findings, signals)
  if (markers.length === 0) return null
  const one = MEDIA[kind]?.[0] ?? 'file'
  const said = markers.map(({ marker, id }) => `${UNAPPLIED_MARKER[marker](one)} (${id})`)
  const listed = said.length === 1 ? said[0] : `${said.slice(0, -1).join(', ')} and ${said[said.length - 1]}`
  const [it, them] = said.length === 1 ? ['that marker', 'it'] : ['those markers', 'them']
  return `Beside the capture record, ${listed}. The decision did not apply ${it}: the signed capture record outranks ${them}, and a self-declared marker can be added to any file.`
}

/** N-18 (browser QA P1-1): real footage with a synthetic voice read "No significant indications … 2 %" while the voice check,
 *  which carries no weight in the video model, scored 0.91. A clean headline names such a check (`view.flagged_outside_sentence`).
 *  N-19k (browser QA Q2-12): the sentence carries both halves, the score and why it is not counted, from the record of that
 *  check for this input (`outsideWhy`); it does not say "flagged" beside a count of flags that leaves the check out. One
 *  sentence for each such check; the figures stand in its finding. */
export function flaggedOutsideSentence(kind: string, d: Decision, findings: readonly Finding[], signals: readonly FlagSignal[] = []): string | null {
  const flagged = flaggedOutside(d, findings, signals)
  if (flagged.length === 0) return null
  const byId = new Map(signals.map((s) => [s.id, s]))
  const subject = (f: Finding) => (kind === 'audio' && f.family === 'A1_voice' ? 'the recording' : FLAGGED_SUBJECT[f.family] ?? 'part of the file')
  const said = flagged.map(({ finding: f, raw }) => {
    const record = (f.evidence_ids ?? []).map((id) => byId.get(id)?.outside_estimate).find((r) => r != null) ?? null
    const basis = raw === null ? `gives ${f.strength} support for manipulation on its own` : `scored ${raw.toFixed(2)}`
    return `A check of ${subject(f)} (${f.id}) ${basis} and is not counted in this estimate: ${outsideWhy(record, f.benchmark_note, false)}.`
  })
  said.push(flagged.length === 1 ? 'The headline does not take it into account.' : 'The headline does not take them into account.')
  if (kind === 'video' && flagged.some((x) => x.finding.family === 'A1_voice')) said.push(VOICE_STEP)
  return said.join(' ')
}

function baseGuidance(kind: string, d: Decision, basis: Basis, rateLine: string, profile?: ConditionProfile | null): Guidance {
  const [one] = media(kind)
  const confounders = CONFOUNDERS[kind] ?? CONFOUNDERS.image
  if (basis === 'insufficient') return { means: ['Chipmunk Verify could not assess this file, for these reasons:'], doesNotMean: ['It is not a finding either way.'] }  // N-17 (D-68)
  if (basis === 'documentary_declared') {
    return {
      means: ['A signed record inside the file, validated as untampered, states that the signing software generated or edited it.'],
      doesNotMean: ['Credentials record what the signing software declared. They don’t describe the scene or who directed it.'],
    }
  }
  if (basis === 'documentary_metadata') {  // N-19h: in the words of the marker's kind
    const words = markerGuidance(markerOf(d), one)
    return { means: [words.means, rateLine], doesNotMean: [words.doesNotMean] }
  }
  if (d.fusion_mode === 'artefact' && estimateOf(d) == null) {  // D-49: capped because a check that it depends on did not run
    return {
      means: [NOT_RUN_MEANS],
      doesNotMean: [`It is not a finding of manipulation, and not a finding that the ${one} is unaltered. Obtain the original file or another source.`],
    }
  }
  if (recordOf(d)?.kind === 'trusted_capture') {  // N-19h (E-20, 2): the record decides, and what stands beside it
    // N-19m (D3-01): what it does not mean is said of the record, not of detectors; the line on error rates stays
    return {
      means: [`${CAPTURE_MEANS} ${statisticalSentence(d, profile)}`],
      doesNotMean: [CAPTURE_NOT, ...(kind === 'audio' && !knownSpeechReference(d.fuser_sha256) ? [AUDIO_MISS] : []), rateLine],
    }
  }
  switch (CODE_BAND[d.code as Exclude<DecisionCode, 'not_assessable'>]) {
    case 'very_likely_ai':
      return {
        means: ['The combined evidence strongly points toward AI generation or manipulation.', rateLine],
        doesNotMean: [`It does not prove the ${one} was made or edited with AI, say who made it, or show exactly what changed. ${capitalise(confounders)} can produce similar patterns.`],
      }
    case 'likely_ai':
      return {
        means: ['The combined evidence points toward AI generation or manipulation.', rateLine],
        doesNotMean: [`It is not a finding that the ${one} is AI-generated. Similar patterns can come from ${confounders}. Treat it as a reason to examine further.`],
      }
    case 'uncertain': {
      // N-19n (D3-09, Q3-02): why the input is outside what was validated, also where the cap changed no code
      const outside = outsideInputSentence(d, profile)
      const levels = levelsSentence(d)  // N-19q (finding 12)
      return {
        means: ['The combined evidence does not support a conclusive assessment.', ...(outside ? [outside] : []), ...(levels ? [levels] : []), rateLine],
        doesNotMean: [`It is not a finding of manipulation, and not a finding that the ${one} is unaltered. Obtain the original file or another source.`],
      }
    }
    case 'likely_authentic':
      return {
        means: ['The combined evidence does not show significant indications of AI generation or manipulation.'],
        doesNotMean: [`It does not confirm the ${one} is authentic or unedited.`, ...(kind === 'audio' && !knownSpeechReference(d.fuser_sha256) ? [AUDIO_MISS] : []), rateLine],
      }
    default:
      return {
        means: ['The combined evidence does not indicate AI generation or manipulation.'],
        doesNotMean: [`It does not confirm the ${one} is authentic or unedited. Detectors miss some AI content.`, ...(kind === 'audio' && !knownSpeechReference(d.fuser_sha256) ? [AUDIO_MISS] : []), rateLine],
      }
  }
}

/** L-1: the validation reason of a decision the envelope rule capped ("Capped at inconclusive: <reason>[; conditions: …]. Before
 *  the cap …", `runtime.fuse_with_artefact`), or null. */
/** N-19q (N-20b's read of the reports, finding 12; `view.levels_sentence`): why the calibrated levels give no answer, read from
 *  the conformal set that the decision's reason records; null for any other result (a condition of D-76 or D-77, a cap). */
const LEVELS_SETS: readonly (readonly [string, string])[] = [['{AI, non-AI}', 'both answers'], ['∅', 'neither answer']]
export function levelsSentence(d: Decision): string | null {
  if (d.code !== 'inconclusive' || d.fusion_mode !== 'artefact' || ruleTaken(d) !== null || capReason(d) !== null) return null
  const which = LEVELS_SETS.find(([text]) => (d.reason ?? '').includes(` ${text}.`))?.[1]
  return which ? `At the calibrated levels for inputs like this one, the estimate fits ${which}, so the result is inconclusive.` : null
}

export function capReason(d: Decision): string | null {
  const m = /^Capped at inconclusive: (.+?)(?:; conditions: .*?)?\. Before the cap/.exec(d.reason ?? '')
  return m ? m[1] : null
}

/** N-19n (D3-09, Q3-02; `view.outside_input_sentence`): wherever a result says that its input is outside what was validated,
 *  the summary says why, from the condition of the input. A result that is inconclusive by its own estimate names no cap in
 *  its reason: the page gave no reason for "outside what was validated", and Plain view held it nowhere. Where the cap
 *  changed the code the sentence stands in the place of "The result is capped at inconclusive because …". With a fusion
 *  model the reason is the cap's, else the envelope's own, which the runtime puts last among the input's conditions
 *  (`abstain.outside_why`); on the fallback it is the reason of the cap. Null under a record that decides. */
export const CAPPED_THEREFORE = 'The result is therefore capped at inconclusive.'
/** the reason of a share of flat blocks says what the image looks like, then the measurement behind it: the summary gives the first */
const FLAT_MEASURE = /(it looks like a screenshot of text or an interface, or a flat graphic): .*? stay below/
export function outsideInputSentence(d: Decision | null | undefined, profile?: ConditionProfile | null): string | null {
  if (!d || recordOf(d) !== null) return null
  const capped = capReason(d)
  const conditions = profile?.limiting_conditions ?? []
  const why = capped ?? (d.fusion_mode === 'artefact' && outsideValidated(d, profile) ? conditions[conditions.length - 1] ?? null : null)
  if (!why) return null
  const said = `This input is outside what was validated: ${why.replace(FLAT_MEASURE, '$1')}.`
  return capped !== null ? `${said} ${CAPPED_THEREFORE}` : said
}

/** N-17 (D-68; design review DS-10): a result that is not assessable does not speak of "the estimate" or "the probability".
 *  Its one sentence on the matter starts with these words, on the page and in the reports (`view.NO_ESTIMATE_FORMED`). */
export const NO_ESTIMATE_FORMED = 'No estimate was formed'

/** N-19f (code review R4-05): the reason of a result that is not assessable because the fusion model gives the checks that
 *  scored no weight for the input (`abstain.NO_WEIGHT_REASON`, word for word: a backend test reads this line). The checks
 *  scored and the fusion was reached, so the page never says of it that nothing ran, scored or was reached. */
export const NO_WEIGHT_REASON = 'For an input like this one the fusion model gives the checks that scored no weight.'
export const noWeight = (d: Pick<Decision, 'code' | 'reason'>): boolean => d.code === 'not_assessable' && d.reason === NO_WEIGHT_REASON

/** L-3: the not-assessable line states what happened, never "too few checks" beside "4 of 4 checks ran". */
export function noEstimateLine(coverage: AnalysisDetail['coverage'], d?: Pick<Decision, 'code' | 'reason'>): string {
  const ok = coverage.by_status.ok ?? 0
  const total = coverage.detectors_total
  // N-19f: checks scored; the model gives them no weight here (`view.no_estimate_line`)
  if (d && noWeight(d)) return `${NO_ESTIMATE_FORMED}: ${ok} of ${total} ${total === 1 ? 'check' : 'checks'} ran, and for an input like this one the fusion model gives those that scored no weight.`
  const server = serverCount(coverage)
  if (server !== null) return `${NO_ESTIMATE_FORMED}: ${lowerFirst(server)}.`  // N-14a: the server, not the file
  if (total === 0) return `${NO_ESTIMATE_FORMED}: no check could be applied to this file.`  // browser QA P1-4: never "none of the 0 checks"
  if (ok === 0) return `${NO_ESTIMATE_FORMED}: none of the ${total} ${total === 1 ? 'check' : 'checks'} produced a result.`
  return `${NO_ESTIMATE_FORMED}: ${ok} of ${total} ${total === 1 ? 'check' : 'checks'} ran, but none produced a score that can be used.`
}

/** What "How the result was reached" says of a not-assessable result (`forensic.no_estimate_formed`). */
export function noEstimateFormed(d: Decision): string {
  const reason = (d.reason ?? '').trim() || 'the file could not be assessed'
  const text = /[.!?]$/.test(reason) ? reason : `${reason}.`
  return `${NO_ESTIMATE_FORMED}: ${/^[A-Z][a-z]/.test(text) ? lowerFirst(text) : text}`
}

/** L-3: beside an uncalibrated score, why it is not a likelihood, and the floor when a provenance marker set it. */
export function uncalibratedNote(d: Decision): string {
  // "floors the estimate at 85%" in results stored before N-7, "floors the uncalibrated score at 85 / 100" since (D-68)
  const floor = /floors the (?:estimate at (\d+)\s?%|uncalibrated score at (\d+) \/ 100)/.exec(d.reason ?? '')
  const base = 'Not a calibrated likelihood: no calibrated fusion model covers this kind of file, so the score is not a frequency estimate.'
  // N-19h (E-20, 3): the note of a floor only where the score is the floor: a marker over a score above its floor did not set it
  const value = floor ? Number(floor[1] ?? floor[2]) : null
  const score = d.probability_ai_uncalibrated ?? d.probability_ai_calibrated
  const isFloor = value !== null && (score == null || halfUp(score * 100) === value)
  return isFloor ? `${base} The score is the ${value} floor set by self-declared metadata, not a measurement of the pixels.` : base
}

/** N-19h (E-20; `view.statistical_sentence`): what the statistical checks alone give beside a record's value, as a sentence;
 *  outside what was validated (8), that their number is not shown. */
export function statisticalSentence(d: Decision, profile?: ConditionProfile | null): string {
  const p = d.statistical_estimate_before_provenance
  if (p == null) return 'The statistical checks alone gave no estimate.'
  if (statisticalWithheld(d, profile)) return `The ${d.fusion_mode === 'fallback' ? 'score' : 'estimate'} of the statistical checks alone is not shown: this input is outside what was validated.`
  const otherWay = aloneOtherWay(d)  // N-19n (D4-02): where the checks alone decide the other answer than the record's
  if (d.fusion_mode === 'fallback') {
    return otherWay ? `${ALONE_OTHER_WAY}: their score is ${halfUp(p * 100)} / 100, not calibrated.` : `The statistical checks alone give a score ${halfUp(p * 100)} / 100, not calibrated.`
  }
  // N-19m (R7-01, R7-02): never the bare estimate where the checks alone do not decide
  if (aloneInconclusive(d)) return `The statistical checks alone do not decide: their estimate is ${aloneNumber(d)} out of 100.`
  // N-19n (`view.alone_one_check`): nor where one check alone counts and decides; with the way it points (D4-02)
  if (d.statistical_alone?.code != null && d.fusion_mode === 'artefact' && countedRows(d).length === 1) {
    return `Only one statistical check scored${otherWay ? ', and it points the other way' : ''}: its estimate is ${numeral(p)} out of 100.`
  }
  if (otherWay) return `${ALONE_OTHER_WAY}: their estimate is ${numeral(p)} out of 100.`
  return `The statistical checks alone estimate ${numeral(p)} out of 100.`
}

/** Reasons of the insufficient variant: the server's first (N-14a), then the decision reason and the input's limiting conditions.
 *  N-19k (browser QA lead 3; `view.insufficient_reasons`): the line under the headline already counts the checks that could
 *  not run on this server, so where the decision's reason names them the count is not given a third time. */
export const insufficientReasons = (d: Decision, profile: ConditionProfile | null | undefined, coverage?: Coverage): string[] =>
  [...new Set([(d.reason ?? '').includes(ON_THIS_SERVER) ? null : serverSentence(coverage), d.reason, ...(profile?.limiting_conditions ?? [])]
    .filter((r): r is string => Boolean(r)))]

/** Recommended next step (PRD RP-01 (1), J5), by basis and decision code. N-14a: where the step was to obtain another file and
 *  checks could not run on this server, the step is the administrator's; a result with an estimate keeps its step and adds it. */
export function nextStep(d: Decision, basis: Basis, coverage?: Coverage, failed = false): string {
  // N-19n (Q4-01): the step that writes the findings failed: there are no key findings to examine
  if (failed) return RUN_AGAIN
  const server = serverStep(coverage)
  if (server === null || basis === 'documentary_declared' || basis === 'documentary_metadata') return baseNextStep(d, basis)
  const noEstimate = d.fusion_mode === 'artefact' && d.code !== 'not_assessable' && estimateOf(d) == null  // D-49
  if (basis === 'insufficient' || noEstimate || d.code === 'inconclusive') return server
  return `${baseNextStep(d, basis)} For a result with every check: ${lowerFirst(server)}`
}

function baseNextStep(d: Decision, basis: Basis): string {
  if (basis === 'insufficient') return 'Obtain the original file or a higher-quality copy, then analyse it again.'
  if (basis === 'documentary_declared') return 'Check who signed the credentials, then ask the source for the original file.'
  if (basis === 'documentary_metadata') return 'Ask the source for the original file and how it was made.'
  if (basis === 'documentary_capture') {  // N-19m (D3-01): the record's step; where the checks alone give an AI code, it says so
    // N-19n (D4-02): on the fallback too, by its score's own ranges (`aloneOtherWay`)
    return aloneOtherWay(d) ? `${CAPTURE_STEP} ${CAPTURE_STEP_AGAINST}` : CAPTURE_STEP
  }
  if (d.code === 'ai_supported_strong' || d.code === 'ai_supported') {
    return 'Examine the key findings below, and look for the original source before relying on this file.'
  }
  if (d.code === 'inconclusive') return 'Look for the original file or another source, then analyse it again.'
  return 'If this file supports a consequential decision, confirm where it came from: detectors miss some AI content.'
}

/** Checks that scored in this decision: the rows the decision counts (`countedRows`), else the findings with at least weak support.
 *  N-19h (E-19, 4): a finding of the record that decides is not a check that scored (`view.scored_checks`). */
export function scoredChecks(d: Decision, findings: readonly Finding[] = []): number {
  return hasContributionRows(d)
    ? countedRows(d).length
    : findings.filter((f) => !NOT_KEY.has(f.family) && !isDocumentary(f) && f.strength !== 'none' && !ofRecord(f, d)).length
}

/** Agreement (UX §5.3; L-3): the decision's confidence, shown as "—" without an estimate or with fewer than two scored checks, capped
 *  at Medium when a check failed or entered checks point against the headline, and at Low when documentary evidence does.
 *  `notWritten`: the step that writes the findings failed (`findingsFailed`). */
/** N-19q (N-20b's read of the reports, finding 5; `view.few_checks`): under contribution rows the count is the decision's, the
 *  checks whose rows it counts (it read "only one check scored" beside "0 of 5 scored checks flagged this file"); without
 *  rows, the fallback's words of what scored. `resultNumber.ts` `ONE_CHECK` is the first. */
export const fewChecks = (d: Decision, scored: number): string =>
  hasContributionRows(d) ? (scored === 1 ? 'only one check counts in the estimate' : 'no check counts in the estimate')
    : (scored === 1 ? 'only one check scored' : 'no check scored')

export function agreement(d: Decision, coverage: AnalysisDetail['coverage'], findings: readonly Finding[] = [], notWritten = false):
  { word: 'High' | 'Medium' | 'Low' | '—'; level: 0 | 1 | 2 | 3; capped: boolean; why: string | null } {
  const scored = scoredChecks(d, findings)
  if (d.code === 'not_assessable') return { word: '—', level: 0, capped: false, why: 'not assessed' }  // N-17 (D-68)
  if (estimateOf(d) == null) return { word: '—', level: 0, capped: false, why: 'no estimate' }
  // N-19h (E-19, 4): a record decides alone: no statistical estimate to agree on; never "no check scored" of a check that scored
  if (recordAlone(d)) return { word: '—', level: 0, capped: false, why: `no statistical ${d.fusion_mode === 'fallback' ? 'score' : 'estimate'}` }
  // N-19n (Q4-01; `view.agreement`): the agreement is worked out from the findings; where they could not be written it is not reported
  if (notWritten) return { word: '—', level: 0, capped: false, why: AGREEMENT_NOT_WRITTEN }
  if (scored < 2) return { word: '—', level: 0, capped: false, why: fewChecks(d, scored) }  // N-19q (finding 5)
  const side = SIDE[d.code]
  const rowsKnown = hasContributionRows(d)
  const counts = (f: Finding) => (rowsKnown ? enteredDecision(f, d) : !NOT_KEY.has(f.family) && f.strength !== 'none')
  const against = side ? findings.filter((f) => counts(f) && decisionDirection(f, d) === (side === 'for' ? 'against' : 'for')) : []
  const failed = (coverage.by_status.failed ?? 0) + (coverage.by_status.timeout ?? 0) > 0
  const level = d.confidence === 'high' ? 3 : d.confidence === 'medium' ? 2 : 1
  const cap = against.some(isDocumentary) ? 1 : (failed || against.length > 0) ? 2 : 3
  const shown = Math.min(level, cap) as 1 | 2 | 3
  const why = shown < level ? (against.some(isDocumentary) ? 'capped: documentary evidence points the other way'
    : against.length > 0 ? 'capped: counter-evidence' : 'capped: a check failed') : null
  return { word: shown === 3 ? 'High' : shown === 2 ? 'Medium' : 'Low', level: shown, capped: shown < level, why }
}

/** Checks that did not produce a result, from the coverage counts. */
export const notRunCount = (coverage: AnalysisDetail['coverage']) =>
  Object.entries(coverage.by_status).filter(([s]) => s !== 'ok').reduce((n, [, k]) => n + k, 0)

/** UX §8.5 low agreement: low confidence while at least moderately strong findings point both ways.
 *  N-19n (D3-11): the headline says "— the checks disagree" only where the result gives no other reason for what it
 *  says. Not under a record that decides (a marker's headline carried it where its confidence is low and a check outside
 *  the estimate points away), not where the input is outside what was validated (the cap of the envelope), not where
 *  D-76 or D-77 took the code, and not where a check did not run and no estimate was formed. It reads the findings that
 *  count in the estimate: the two that differed on the QA's page were a capture record that does not validate and a
 *  check whose gate is closed. */
export function lowAgreement(d: Decision, findings: readonly Finding[], profile?: ConditionProfile | null): boolean {
  if (d.confidence !== 'low' || d.code === 'not_assessable') return false
  if (recordOf(d) !== null || outsideValidated(d, profile) || ruleTaken(d) !== null || estimateOf(d) == null) return false
  const counted = hasContributionRows(d) ? findings.filter((f) => entered(f.contribution_logit)) : findings
  const strong = counted.filter((f) => f.strength === 'moderate' || f.strength === 'strong' || f.strength === 'decisive')
  return strong.some((f) => findingDirection(f) === 'for') && strong.some((f) => findingDirection(f) === 'against')
}

/** N-19n (D3-09; the reports' `view.CAPPED_CHECKS_BELOW`): what stands at the head of the key findings of a result outside
 *  what was validated, in the place of the line that counts the findings toward and against. In N-19f's form. */
export const OUTSIDE_FINDINGS = 'No test shows how well the checks work on inputs like this one, so none of their findings is given as a reason here. They are listed below.'

export interface KeyFinding {
  finding: Finding
  direction: Direction
  counter: boolean
}

/** Families that describe the run rather than the file: they appear in coverage and the full list, never as key findings. */
const NOT_KEY = new Set(['coverage', 'F9_distribution_conditions'])

/** N-19k (Q2-14; D-64; `view.set_aside`): under a result that is not assessable, the finding of a record of the file (its
 *  metadata, its container, its compression history) that points away from AI generation. It is listed with what it
 *  read; it is not headed "support against AI generation or manipulation", and it is no key finding. */
const RECORD_FAMILIES = new Set(['F1_provenance', 'F2_self_declared_metadata', 'F5_compression_history', 'F6_capture_pipeline',
  'F8_metadata_consistency', 'V4_codec'])
export const SET_ASIDE_TITLE = 'A record of the file, not evidence of authenticity'
export const SET_ASIDE_NOTE = 'No check of the content gave a score that can be used, so this finding is shown for inspection only: metadata is never evidence of authenticity.'
export const setAside = (f: Finding, d?: Pick<Decision, 'code'> | null): boolean =>
  d?.code === 'not_assessable' && !isDocumentary(f) && RECORD_FAMILIES.has(f.family) && f.strength !== 'none' && findingDirection(f) === 'against'
const SIDE: Partial<Record<DecisionCode, Direction>> = {
  ai_supported_strong: 'for', ai_supported: 'for', no_ai_indicators: 'against', no_ai_indicators_strong: 'against',
}

/** L-3: an artefact decision with contribution rows, where "entered this decision" is measurable. */
export const hasContributionRows = (d: Decision) => d.fusion_mode === 'artefact' && (d.contributions?.length ?? 0) > 0

/** L-3: a finding entered the decision when its check has a non-zero contribution after gates, or it is documentary (a Stage 0
 *  override row). Checks with weight 0 (not in the fitted model, gated out or not run) did not. */
export const enteredDecision = (f: Finding, d?: Decision) =>
  isDocumentary(f) || (f.contribution_logit != null && Math.abs(f.contribution_logit) > 1e-9) || ofRecord(f, d)  // N-19h: the record that decides

/** Direction of a finding within this decision (D-63, `vocabulary.ts`): by its contribution, neutral when it had little
 *  effect (below ln 2; 0.05 in a result published before N-17), else its own direction. */
export function decisionDirection(f: Finding, d?: Decision): Direction {
  if (isDocumentary(f) || f.contribution_logit == null) return findingDirection(f)
  return direction(f.contribution_logit, littleOf(d))
}

/** UX §5.4: up to six findings, the strongest counter-evidence last (§7.3 rule 8).
 *  - Artefact decision with contribution rows (L-3): only findings that entered the decision, in synthesis rank order, with the
 *    direction of their contribution; the counter-evidence is the largest contribution against the headline.
 *  - Otherwise (uncalibrated fallback, no estimate): the synthesis rank order with at least weak support either way. */
/** N-19n (D3-07, 3; `view.provenance_aside`): a finding of the provenance families (Content Credentials, self-declared
 *  metadata, a watermark message) that did not enter the decision is no check outside the estimate. It is listed after
 *  them under words of its own, each row with its diamond and no meter. The page knows it by its family, as the reports do,
 *  and not by the signals: when the request for the signals fails, nothing says of it a sentence of a check without weight.
 *  With contribution rows: at least weak strength. Without (the fallback): the finding of a record that was not applied
 *  (strong); a provenance check that the fallback weighs stays among the key findings. */
const PROVENANCE_FINDINGS = new Set(['F1_provenance', 'F2_self_declared_metadata'])
export const ASIDE_TITLE = 'Provenance findings that did not decide this result'
export const ASIDE_NOTE = 'These records of the file did not decide this result; each says why. They are shown for the examiner.'
export function provenanceAside(all: readonly Finding[], d: Decision): Finding[] {
  const rows = hasContributionRows(d) || noWeight(d)
  return [...all].filter((f) => PROVENANCE_FINDINGS.has(f.family) && !enteredDecision(f, d) && (rows ? f.strength !== 'none' : f.strength === 'strong'))
    .sort((a, b) => a.rank - b.rank)
}

export function keyFindings(all: readonly Finding[], d: Decision, max = 6): KeyFinding[] {
  const side = SIDE[d.code]
  const opposite = side === 'for' ? 'against' : 'for'
  if (noWeight(d)) return []  // N-19f: every finding of a check is outside the estimate (`contextFindings`); none is a reason
  // N-19k (Q2-14; D-64; `view.key_findings`): under a result that is not assessable no check of the content scored, so a
  // finding that points away from AI generation is a record's (a metadata block), never evidence of authenticity: not key
  if (d.code === 'not_assessable') all = all.filter((f) => findingDirection(f) !== 'against')
  // N-19l (R6-01, 4; `view.key_findings`): the findings of the record that decides stand first, with the ids they were
  // stored with. A result stored before N-19h ranked a marker's and a capture's finding behind the checks that entered
  const order = (a: Finding, b: Finding) => Number(ofRecord(b, d)) - Number(ofRecord(a, d)) || a.rank - b.rank
  if (hasContributionRows(d)) {
    const ranked = [...all].filter((f) => !NOT_KEY.has(f.family) && enteredDecision(f, d)).sort(order)
    const against = side ? ranked.filter((f) => decisionDirection(f, d) === opposite && !ofRecord(f, d)) : []
    const counter = against.sort((a, b) => Math.abs(b.contribution_logit ?? 0) - Math.abs(a.contribution_logit ?? 0))[0]
    const rest = ranked.filter((f) => f !== counter).slice(0, counter ? max - 1 : max)
    return [...rest, ...(counter ? [counter] : [])].map((f) => ({ finding: f, direction: decisionDirection(f, d), counter: f === counter }))
  }
  // N-19n (D3-07, 3): the finding of a record that was not applied is no key finding (it stood last, as counter-evidence)
  const aside = new Set(provenanceAside(all, d))
  const ranked = [...all].filter((f) => !NOT_KEY.has(f.family) && !aside.has(f) && (f.strength !== 'none' || ofRecord(f, d))).sort(order)
  const counter = side ? ranked.find((f) => findingDirection(f) === opposite && !ofRecord(f, d)) : undefined
  const rest = ranked.filter((f) => f !== counter).slice(0, counter ? max - 1 : max)
  return [...rest, ...(counter ? [counter] : [])].map((f) => ({ finding: f, direction: findingDirection(f), counter: f === counter }))
}

/** L-3: findings with at least weak support that did not enter an artefact decision (weight 0): context, never key.
 *  N-19f: under a result that is not assessable because the model gives the checks that scored no weight, all of them. */
export function contextFindings(all: readonly Finding[], d: Decision): Finding[] {
  if (!hasContributionRows(d) && !noWeight(d)) return []
  // N-19n (D3-07, 3): a record of the file is no check outside the estimate (`provenanceAside`)
  return [...all].filter((f) => !NOT_KEY.has(f.family) && !PROVENANCE_FINDINGS.has(f.family) && f.strength !== 'none' && !enteredDecision(f, d))
    .sort((a, b) => a.rank - b.rank)
}

/** L-3: what the check did in this decision. D-63: measured from the check's neutral point; a result published before N-17
 *  (a `context_prior` row) was measured from the model's starting point and keeps that wording. */
/** N-19o (D5-04; `view.not_counted`): the finding of a check whose row the decision publishes and does not count (`counted`
 *  false: a check that declares H3, the container check of a video). */
export function notCounted(f: Finding, d: Decision | null | undefined): boolean {
  const ids = new Set(f.evidence_ids ?? [])
  return (d?.contributions ?? []).some((r) => r.kind === 'detector' && r.counted === false && Math.abs(r.logit) > 1e-9 && r.signal_id != null && ids.has(r.signal_id))
}

/** The size of a contribution as the line prints it: never "0.00" (N-19o, D5-04; `view.log_odds_size`). */
const logOddsSize = (c: number) => (Math.abs(c) < 0.005 ? 'less than 0.01' : Math.abs(c).toFixed(2))

export function contributionLine(f: Finding, d: Decision, plain = false): string | null {
  // N-19o (D5-04): a row that is not counted has no sentence on the estimate (it read "lowered the estimate by 0.00 log-odds")
  if (!hasContributionRows(d) || isDocumentary(f) || f.contribution_logit == null || notCounted(f, d)) return null
  const c = f.contribution_logit
  if (plain) {
    // N-19d (browser QA "Copy": Plain view said "raised the estimate by 1.34 log-odds"): the size in a word, at the bands of
    // the strength words (ln 2, ln 10, ln 100)
    const size = Math.abs(c)
    if (size < littleOf(d)) return entered(c) || littleOf(d) === LEGACY_LITTLE ? 'In this decision it had little effect on the estimate.' : null
    const how = size >= Math.log(100) ? 'strongly' : size >= Math.log(10) ? 'clearly' : 'slightly'
    return `In this decision it ${c > 0 ? 'raised' : 'lowered'} the estimate ${how}.`
  }
  const moved = `it ${c > 0 ? 'raised' : 'lowered'} the estimate by ${logOddsSize(c)} log-odds`
  // D-63: the word is the finding's own (its title and meter); this line gives the number behind it
  if (littleOf(d) !== LEGACY_LITTLE) return entered(c) ? `In this decision ${moved} from its neutral point.` : null
  if (Math.abs(c) < LEGACY_LITTLE) return 'In this decision it had little effect on the estimate.'
  return `In this decision ${moved} from the model’s starting point.`
}

/** The reason line: the finding's first template sentence, plus its dissent note when it is counter-evidence (UX §5.4 F5). */
export function reasonOf(k: KeyFinding): string[] {
  const f = k.finding
  const first = f.sentences?.[0]?.text ?? f.statement
  return k.counter && f.dissent_note ? [first, f.dissent_note] : [first]
}

/** N-19d (browser QA "Copy": Plain view printed "DDA (DINOv2 ViT-L/14 + LoRA, native 336 px crops) returned …"): a sentence
 *  with the bracket after a check's name left out. `names` are the names of the checks of the analysis; only a name that
 *  ends in a bracket is shortened, and only where the sentence holds it whole. The expert view prints the full name. */
export function plainNames(text: string, names: readonly string[]): string {
  return names.reduce((said, name) => {
    const bracket = /^(.*\S)\s*\([^()]*\)$/.exec(name)
    return bracket ? said.split(name).join(bracket[1]) : said
  }, text)
}

export interface ReviewLineEvidence {
  signoff?: Pick<Signoff, 'kind' | 'principal_id' | 'signed_at' | 'decision' | 'signature_valid'> | null
  reviewerName?: string
  /** The examiner's sign-off of a revision that has its peer review, so the line names both reviewers (N-19a). */
  examiner?: { name?: string; principal_id: string; signed_at: string } | null
  successorId?: string | null
  loading?: boolean
}

type Position = NonNullable<NonNullable<Review>['examiner_position']>
/** Short position words for one-line rows (the review line, case table, analyses list). */
export const POSITION_SHORT: Record<Position, string> = {
  ai_supported_strong: 'Strong indications', ai_supported: 'Indications', inconclusive: 'Inconclusive',
  no_ai_indicators: 'No significant indications', no_ai_indicators_strong: 'No indications', no_opinion: 'No opinion',
}

/** "Examiner: Inconclusive, differs from the machine assessment" (design review P0-3): the examiner's position beside the machine's,
 *  for the review line; `compact` ("…, differs from machine") for the case exhibit row and the analyses list. Null without a
 *  conclusion. */
export function examinerPosition(position: Position | null | undefined, departs: boolean | null | undefined, compact = false,
  reference: boolean | 'capture_record' | 'declaration' | null | undefined = false): string | null {
  if (!position) return null
  const machine = compact ? 'machine' : 'the machine assessment'
  // N-19n (D4-01): under a capture the position is on the step of the capture record, away from it, or (no opinion, which
  // has no position to be away from) not on it; in a list "the record's step". N-19o (D5-01): `reference` is the review
  // summary's (`examiner_reference`); `declaration` names the step of the declaration (a declaration on the fallback)
  if (reference && departs != null) {
    const step = compact ? STEP_OF_CAPTURE_SHORT : reference === 'declaration' ? STEP_OF_DECLARATION : STEP_OF_CAPTURE
    return `Examiner: ${POSITION_SHORT[position]}, ${!departs ? 'on' : position === 'no_opinion' ? 'not on' : 'away from'} ${step}`
  }
  // N-19j (design re-review D2-12): beside a result that is not assessable there is nothing to differ from; the page said
  // "differs from the machine assessment" where the report says that no comparison is recorded
  return `Examiner: ${POSITION_SHORT[position]}${departs ? `, differs from ${machine}` : departs === false ? `, agrees with ${machine}` : departs === null ? `, no comparison with ${machine}` : ''}`
}

/** N-19j (browser QA Q2-03): a decision with its object, as the reports print it (`view.decision_words`). */
const OPINION_WORD: Record<Signoff['kind'], Record<Signoff['decision'], string>> = {
  examiner: { concur: 'concurs with the machine assessment', concur_with_changes: 'concurs with the machine assessment, with changes', non_concur: 'does not concur with the machine assessment' },
  peer_review: { concur: 'concurs with the examiner’s opinion', concur_with_changes: 'concurs with the examiner’s opinion, with changes', non_concur: 'does not concur with the examiner’s opinion' },
}

/** The name a sign-off carries (N-10b; code re-verification R2-08): the signer need not be a member of the case, so the
 *  member list cannot name every signer. The API fills `principal_name` with the id when the account is gone; then `fallback`. */
export const signerName = (s: Pick<Signoff, 'principal_id' | 'principal_name'>, fallback?: string): string | undefined =>
  s.principal_name && s.principal_name !== s.principal_id ? s.principal_name : fallback

/** The sign-off that names a state's line: the peer review once one is recorded, else the examiner's. */
export const reviewLineKind = (state: string | null | undefined): Signoff['kind'] =>
  state === 'approved' || state === 'disputed' ? 'peer_review' : 'examiner'

/** Review state is authoritative; names, opinions and dates come from that revision's recorded sign-off, never the lock date.
 *  The line reads "Reviewed" only for an issuable revision: one signed while the case's second review is outstanding reads
 *  "Signed by …; awaiting second review" (N-19a), and a peer review that does not concur reads "Disputed", with the peer's decision. */
export function reviewLine(review: Review, when: (iso: string | null | undefined) => string, evidence: ReviewLineEvidence = {}):
  { state: NonNullable<Review>['state']; text: string } {
  const state = review?.state ?? 'machine'
  const record = evidence.signoff
  const peer = state === 'approved' || state === 'disputed'
  const outstanding = state === 'reviewed' && !!review?.requires_second_review
  const position = examinerPosition(review?.examiner_position, review?.examiner_departs, false, review?.examiner_reference)
  if ((state === 'reviewed' || peer) && record?.kind === reviewLineKind(state)) {
    const name = evidence.reviewerName || `principal ${record.principal_id}`
    // N-19a (design review, P2 "screens" 3): a twice-reviewed record names both reviewers, the examiner first
    const first = peer && evidence.examiner
      ? `${evidence.examiner.name || `principal ${evidence.examiner.principal_id}`}, ${when(evidence.examiner.signed_at)} · second review by` : null
    const prefix = state === 'approved' ? (first ? `Reviewed by ${first}` : 'Reviewed · second review by')
      : state === 'disputed' ? (first ? `Disputed · signed by ${first}` : 'Disputed · second review by')
      : outstanding ? 'Signed by' : 'Reviewed by'
    // The examiner's position, not the bare opinion word, so the line never reads "concurs" beside a departure (design P0-3).
    const opinion = peer ? `peer review: ${OPINION_WORD.peer_review[record.decision]}` : null
    return { state, text: [`${prefix} ${name}, ${when(record.signed_at)}${outstanding ? '; awaiting second review' : ''}`,
      [opinion, position ?? (opinion ? null : `opinion: ${OPINION_WORD.examiner[record.decision]}`)].filter(Boolean).join(' · ')].join(' — ')
      + (record.signature_valid ? '' : ' · signature does not verify') }
  }
  const details = evidence.loading ? 'loading sign-off details' : 'sign-off details unavailable'
  switch (state) {
    case 'machine': return { state, text: 'Machine assessment — not reviewed by an examiner' }
    case 'in_review': return { state, text: `In review — not yet signed off${position ? ` · ${position}` : ''}` }
    case 'reviewed': return { state, text: `${outstanding ? 'Signed; awaiting second review' : 'Reviewed'} — ${details}` }
    case 'approved': return { state, text: `Reviewed · second review — ${details}` }
    case 'disputed': return { state, text: `Disputed · peer reviewer does not concur — ${details}` }
    case 'superseded': return { state, text: evidence.successorId ? `Superseded by analysis ${evidence.successorId.slice(0, 8)}`
      : 'Superseded by a newer analysis of this file' }
    default: return { state, text: `Withdrawn — ${review?.withdrawn_reason ?? 'no reason recorded'}` }
  }
}
