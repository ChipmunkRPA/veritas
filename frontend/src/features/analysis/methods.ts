/** Methods & validation rules (PRD RP-04, UX-01; ARCHITECTURE §6.4 section 4; BUILD_SPEC E-7): the validated-envelope statement,
 *  the dossier slice metrics with their intervals, the fused error rates in testing, per-detector validation and the
 *  reliability-factor table. Where a record is absent the cell says "not established"; it never borrows another slice's numbers. */
import type { AnalysisDetail, ModelCard } from '../../lib/api/queries'
import { shortHash } from '../../lib/format'
import { calibrationMethod, policyWords } from '../../lib/recordWords'
import { sliceWords } from '../../lib/sliceWords'
import { fuserReview } from './vocabulary'
import { ADMISSION_WORD, NOT_ESTABLISHED, admissionWord, fmtMetric, measuredRows, withCi, type EvalRow } from '../models/catalogue'

type Calibration = NonNullable<AnalysisDetail['calibration']>[number]
export type ErrorRate = NonNullable<Calibration['error_rates']>[number]
type ConditionProfile = NonNullable<AnalysisDetail['condition_profile']>

export interface Envelope { within: boolean | null; text: string }

/** The "outside validated envelope" statement (RP-04, EA-02). A slice is said in words, as the reports say it (N-19d). */
export function envelopeStatement(cp: ConditionProfile | null | undefined): Envelope {
  if (!cp) return { within: null, text: 'No condition profile is recorded for this analysis, so no validation slice applies to it.' }
  if (cp.within_validated_envelope) {
    if (cp.validation_support === 'pooled') {
      return { within: true, text: `Validation uses the broader group ${cp.validation_measured_at ? sliceWords(cp.validation_measured_at) : 'recorded in the model'}. Performance for this exact input slice (${sliceWords(cp.slice_id)}) has not been established separately.` }
    }
    if (cp.validation_support == null || cp.validation_support === 'legacy') {
      return { within: true, text: `This input meets the recorded validation AUC threshold (${sliceWords(cp.slice_id)}). The legacy record does not identify whether validation was direct or pooled.` }
    }
    return { within: true, text: `This input meets the recorded AUC support criterion for its slice (${sliceWords(cp.slice_id)}). This does not establish a false-positive or detection rate for this exact input slice.` }
  }
  const why = (cp.limiting_conditions ?? []).filter(Boolean)
  return {
    within: false,
    // N-19q (finding 9; `forensic.methods`): the slice in brackets is the input's, not the envelope's
    text: `This input (${sliceWords(cp.slice_id)}) is outside the validated envelope${why.length ? `: ${why.join('; ')}` : ''}. No measured error rate describes inputs like it.`,
  }
}

export const fuserRecord = (calibration: readonly Calibration[] | null | undefined): Calibration | undefined =>
  (calibration ?? []).find((c) => c.subject.startsWith('fuser-'))

/** The fused decision's measured error rates, per context and threshold. */
export const fusedErrorRates = (calibration: readonly Calibration[] | null | undefined): ErrorRate[] => fuserRecord(calibration)?.error_rates ?? []

export interface Row { key: string; value: string; established: boolean }

const row = (key: string, value: string | null | undefined, missing = NOT_ESTABLISHED): Row =>
  value == null || value === '' ? { key, value: missing, established: false } : { key, value, established: true }

/** Dossier slice metrics (VG-02): TPR, FPR and AUC with bootstrap 95 % intervals, sample sizes, the dossier version and ECE. */
export function sliceRows(a: AnalysisDetail): Row[] {
  const ref = a.validation_ref
  const m = ref?.slice_metrics
  const fuser = fuserRecord(a.calibration)
  const n = m && m.n_pos + m.n_neg > 0 ? `${m.n_pos} AI-generated or manipulated · ${m.n_neg} genuine` : null
  return [
    row('Slice', a.condition_profile?.slice_id ? sliceWords(a.condition_profile.slice_id) : null, 'not recorded'),
    row('Validation record', ref?.dossier_version, `${NOT_ESTABLISHED}: no validation record is in force`),
    row('TPR', m?.tpr != null ? withCi(m.tpr, m.tpr_ci95) : null),
    row('FPR', m?.fpr != null ? withCi(m.fpr, m.fpr_ci95) : null),
    row('AUC', m?.auc != null ? fmtMetric(m.auc) : null),
    row('Samples', n),
    row('Calibration ECE', fuser?.ece != null ? `${fmtMetric(fuser.ece)} (${[fuser.bench_version, testViewWords(fuser.test_view)].filter(Boolean).join('; ')})` : null),
  ]
}

const TEST_VIEWS: Record<string, string> = { size_matched: 'size-matched test set' }

/** J-3d: the test view the calibration numbers come from, in words; null when the artefact did not record one. */
export function testViewWords(view: string | null | undefined): string {
  return view ? (TEST_VIEWS[view] ?? view.replace(/_/g, ' ')) : 'test view not recorded'
}

/** "3 of the 4 checks in this analysis cite", "1 of the 4 … cites", "The only check in this analysis cites" (L-7, as the report's
 *  `of_checks`): the count names its population and has no "1 of 1 checks". */
export function ofChecks(k: number, n: number, singular: string, plural: string): string {
  if (n === 1 && k === 1) return `The only check in this analysis ${singular}`
  return `${k} of the ${n} checks in this analysis ${k === 1 ? singular : plural}`
}

type DetectorRef = NonNullable<NonNullable<AnalysisDetail['manifest']>['detectors']>[number]

/** RP-04: the admission the manifest recorded when the analysis ran, with a dated addendum once the card has changed; a result
 *  without one prints the current card, marked. `decision` is what the Acceptance factor counts. */
export function recordedAdmission(ref: DetectorRef, card: ModelCard | undefined): { text: string; decision: string | null } {
  const recorded = ref.admission
  const now = card?.admission
  if (recorded) {
    const changed = card != null && (now?.decision !== recorded.decision || (now?.date ?? null) !== (recorded.date ?? null))
    const addendum = changed ? ` (card now: ${[now ? admissionWord(card).split(' · ')[0] : 'not recorded', now?.date].filter(Boolean).join(', ')})` : ''
    return { text: `${ADMISSION_WORD[recorded.decision] ?? recorded.decision}${addendum}`, decision: recorded.decision }
  }
  if (card && now) return { text: `${admissionWord(card)} (not recorded at analysis time)`, decision: now.decision }
  return { text: 'not recorded', decision: null }
}

export interface DetectorValidation { id: string; version: string; card: ModelCard | undefined; measured: string; established: boolean; admission: string }

/** Each detector of the reproducibility manifest with its benchmark row for this input's slice, else what is on its card. */
export function detectorValidation(a: AnalysisDetail, cards: readonly ModelCard[] | undefined): DetectorValidation[] {
  const slice = a.condition_profile?.slice_id
  return (a.manifest?.detectors ?? []).map((d) => {
    const card = cards?.find((c) => c.id === d.id)
    const rows = card ? measuredRows(card) : []
    const hit: EvalRow | undefined = slice ? rows.find((r) => r.slice === slice) : undefined
    const measured = hit
      ? [`AUC ${fmtMetric(hit.auc)}`, `TPR at 1 % FPR ${fmtMetric(hit.tpr_at_fpr01)}`, `ECE ${fmtMetric(hit.ece)}`, hit.n != null ? `n ${hit.n}` : null, [hit.bench, hit.date].filter(Boolean).join(', ')]
        .filter(Boolean).join(' · ')
      : rows.length ? `${NOT_ESTABLISHED} for this slice (${rows.length} benchmark ${rows.length === 1 ? 'row' : 'rows'} on the card)`
        : card ? `${NOT_ESTABLISHED}: no benchmark results on the card` : `${NOT_ESTABLISHED}: no model card on record`
    return { id: d.id, version: d.version, card, measured, established: hit != null, admission: recordedAdmission(d, card).text }
  })
}

/** RP-04 reliability-factor table: tested; peer review and publication; known error rate; standards and controls; acceptance. */
export function reliabilityFactors(a: AnalysisDetail, cards: readonly ModelCard[] | undefined): Row[] {
  const d = a.decision
  const fuser = fuserRecord(a.calibration)
  const ref = a.validation_ref
  const detectors = a.manifest?.detectors ?? []
  const ran = detectors.map((x) => cards?.find((c) => c.id === x.id))
  const cited = ran.filter((c) => c?.citation).length
  const admitted = detectors.filter((x, i) => recordedAdmission(x, ran[i]).decision === 'admitted').length  // recorded at analysis time
  const rates = fusedErrorRates(a.calibration)
  const m = ref?.slice_metrics

  const tested = fuser?.bench_version
    ? `Fusion calibrated and tested on ${fuser.bench_version}${fuser.bench_manifest_sha256 ? ` (manifest ${shortHash(fuser.bench_manifest_sha256)})` : ''}${ref ? `; validation record ${ref.dossier_version}` : ''}`
    : ref ? `Validation record ${ref.dossier_version}` : null
  const publication = detectors.length === 0 ? null : cited > 0 ? `${ofChecks(cited, detectors.length, 'cites', 'cite')} a publication (listed on the model cards)` : null
  const errorRate = m && (m.tpr != null || m.fpr != null)
    ? `This slice (${sliceWords(a.condition_profile?.slice_id)}): TPR ${withCi(m.tpr, m.tpr_ci95)}, FPR ${withCi(m.fpr, m.fpr_ci95)}`
    : rates.length ? `${rates.length} measured error-rate ${rates.length === 1 ? 'row' : 'rows'} for the fusion (${[...new Set(rates.map((r) => r.bench_version))].join(', ')})` : null
  const controls = [
    a.manifest ? `reproducibility manifest recorded${a.manifest.config_sha256 ? ` (config ${shortHash(a.manifest.config_sha256)})` : ''}` : 'no reproducibility manifest',
    d?.fusion_mode === 'artefact' && d.fuser_sha256 ? `fusion artefact ${shortHash(d.fuser_sha256)}, ${fuserReview(fuser?.reviewed_by, fuser?.review_date)}` : 'uncalibrated fallback fusion with hand-set weights',
    `release check ${ref ? `against validation record ${ref.dossier_version}` : NOT_ESTABLISHED}`,
  ].join('; ')
  const acceptance = detectors.length > 0 && admitted > 0
    ? `${ofChecks(admitted, detectors.length, 'is', 'are')} admitted on the internal benchmark; independent benchmark results ${NOT_ESTABLISHED}`
    : null

  return [
    row('Tested (harness and dataset)', tested),
    row('Peer review and publication', publication),
    row('Known error rate', errorRate),
    // The manifest and the fusion record always exist for a v2 result: this row is established even when the gate is not.
    { key: 'Standards and controls', value: controls, established: a.manifest != null },
    row('Acceptance (published methods, independent benchmarks)', acceptance,
      `${NOT_ESTABLISHED}: no detector that ran is admitted on the internal benchmark, and no independent benchmark result is recorded`),
  ]
}

/** How the estimate was produced, in plain terms (UX §6.3 R9). */
export function methodRows(a: AnalysisDetail): Row[] {
  const d = a.decision
  const fuser = fuserRecord(a.calibration)
  const fusion = !d ? null : d.fusion_mode === 'fallback'
    ? 'Uncalibrated fallback: hand-set weights combine the detector scores. The result is a score, not a calibrated probability.'
    : `Calibrated fusion (${fuser?.subject ?? 'fuser artefact'}): a monotone logistic stacker over the detector scores, Platt calibration per input context, and conformal bands for the headline.`
  const m = a.manifest
  return [
    row('Pipeline', m ? `${m.pipeline_version} · profile ${m.effective_profile ?? m.profile}` : null, 'not recorded'),
    row('Fusion', fusion, 'not recorded'),
    row('Calibration', fuser ? `${calibrationMethod(fuser.method)}${fuser.bench_version ? ` on ${fuser.bench_version}` : ''}` : d?.fusion_mode === 'fallback' ? 'none (uncalibrated)' : null),
    row('Licence posture', policyWords(m?.licence_posture), 'not recorded'),
  ]
}
