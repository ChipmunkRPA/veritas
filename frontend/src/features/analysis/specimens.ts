/** Synthetic specimens for the /design gallery and the unit tests of the E-3 panels (UX §5.6–5.8). Every value is invented and
 *  labelled as a specimen; nothing here is an analysis result or a measured number. */
import type { AnalysisDetail, DetectorRun, Finding, SignalV2, TableExhibit } from '../../lib/api/queries'
import type { Decision } from './answer'
import { sigmoid } from './contributions'
import { direction, strength } from './vocabulary'

type Row = NonNullable<Decision['contributions']>[number]

export function specimenSignal(n: number, over: Partial<SignalV2> = {}): SignalV2 {
  const id = over.id ?? `spec.detector_${n}:global`
  return {
    id, display_id: `S${String(n).padStart(2, '0')}`, detector_id: id.split(':')[0], detector_version: '1.0.0', model_revisions: {},
    name: `Specimen check ${n}`, family: 'learned-whole-image', modality: 'image', category: 'model', status: 'ok', status_reason: null,
    scope: 'global', scope_ref: null, hypothesis: 'H1_whole_content_synthesis', lane: null, threat: [], track_id: null, segments: [], fp_modes: [],
    contribution_logit: null, contribution_pp: null, effective_weight: 1, raw_score: 0.5, raw_kind: 'probability', calibrated_probability: 0.5,
    log_lr: 0, reliability: { factor: 1, reasons: [], inputs: {} }, observed: 'specimen value', reference: 'specimen reference', reasoning: '',
    metrics: {}, details: {}, exhibit_ids: [], provenance_override: null, calibration_ref: null, legacy: null, ...over,
  } as SignalV2
}

const CHECKS: [number, string, number, number, string][] = [
  // display n, name, Δ log-odds, calibrated p, family
  // D-63: moderate from ln 10 (2.30), weak from ln 2 (0.69), little effect below; the first four are the key findings
  [2, 'Localisation specimen (camera-noise map)', 2.4, 0.91, 'localiser'],
  [5, 'Whole-image classifier specimen A', 1.1, 0.84, 'learned-whole-image'],
  [6, 'Whole-image classifier specimen B', 0.95, 0.78, 'learned-whole-image'],
  [7, 'Face-swap specimen', -0.9, 0.31, 'learned-face'],
  [3, 'Noise residual specimen', 0.2, 0.6, 'forensic'],
  [4, 'Frequency specimen', 0.12, 0.57, 'forensic'],
  [8, 'Compression specimen', 0.06, 0.53, 'forensic'],
  [9, 'Colour statistics specimen', 0.03, 0.52, 'forensic'],
  [10, 'Texture specimen', -0.02, 0.49, 'forensic'],
]

export const SPECIMEN_SIGNALS: SignalV2[] = [
  ...CHECKS.map(([n, name, delta, p, family]) => specimenSignal(n, {
    name, family, category: family === 'forensic' ? 'forensic' : 'model', contribution_logit: delta, calibrated_probability: p,
    log_lr: Math.log(p / (1 - p)), effective_weight: 1 + n / 10, calibration_ref: `spec.detector_${n}@1.0.0/specimen-cal`,
    observed: `score ${p.toFixed(2)}`, reference: 'specimen: no reference distribution',
    reasoning: n === 2 ? 'Specimen text: the check compares local noise fingerprints across the image; regions that differ are candidates for editing.' : '',
    fp_modes: n === 2 ? ['specimen false alarm: heavy HDR tone-mapping', 'specimen false alarm: text overlays'] : [],
    reliability: n === 7 ? { factor: 0.4, reasons: ['largest face 96 px (rule face_px < 120)'], inputs: { face_px: 96 } } : { factor: 1, reasons: [], inputs: {} },
  })),
  specimenSignal(1, { id: 'spec.c2pa:global', name: 'Content Credentials specimen', family: 'provenance', category: 'provenance', hypothesis: 'H3_provenance_integrity',
    contribution_logit: null, calibrated_probability: null, raw_score: null, log_lr: null, provenance_override: null, observed: 'no manifest (specimen)' }),
  specimenSignal(11, { id: 'spec.lipsync:global', name: 'Lip-sync specimen', family: 'lipsync', status: 'not_applicable', status_reason: 'no speech detected',
    contribution_logit: null, calibrated_probability: null }),
]

export const SPECIMEN_RUNS: DetectorRun[] = [
  ...SPECIMEN_SIGNALS.filter((s) => s.status === 'ok').map((s, i) => ({
    id: `run-${i}`, detector_id: s.detector_id, detector_version: '1.0.0', model_revisions: {}, device: 'mps', dtype: 'float16', status: 'ok', reason: null,
    started_at: null, duration_ms: 120 + i * 10, peak_memory_mb: null, raw_output_artefact_id: null,
  })),
  { id: 'run-lipsync', detector_id: 'spec.lipsync', detector_version: '1.0.0', model_revisions: {}, device: null, dtype: null, status: 'not_applicable',
    reason: 'no speech detected', started_at: null, duration_ms: null, peak_memory_mb: null, raw_output_artefact_id: null },
  { id: 'run-voice', detector_id: 'spec.voice', detector_version: '1.0.0', model_revisions: {}, device: 'mps', dtype: null, status: 'failed',
    reason: 'specimen: the op is not supported on this device', started_at: null, duration_ms: null, peak_memory_mb: null, raw_output_artefact_id: null },
] as DetectorRun[]

const ROWS: Row[] = [
  { signal_id: null, label: 'every check at its neutral point', kind: 'neutral_start', logit: -1.6, pp: 0, approximate: false },
  ...CHECKS.map(([n, name, delta]) => ({ signal_id: `spec.detector_${n}:global`, label: name, kind: 'detector' as const, logit: delta, pp: 0, approximate: false, baseline: 'neutral_point' as const })),
  { signal_id: null, label: 'missing: Voice specimen', kind: 'missing', logit: 0.05, pp: 0, approximate: false },
  { signal_id: null, label: 'bound on provenance influence (|Δs| ≤ 0.05)', kind: 'override', logit: -0.04, pp: 0, approximate: false },
]
const SUM = ROWS.reduce((s, r) => s + r.logit, 0)

export const SPECIMEN_DECISION: Decision = {
  code: 'ai_supported', verbal: 'specimen', band: 'likely_ai', confidence: 'medium', reason: 'Specimen decision.', fusion_mode: 'artefact',
  fuser_sha256: 'f'.repeat(64), probability_ai_calibrated: Number(sigmoid(SUM).toFixed(6)), probability_ai_uncalibrated: null, ci90: [0.84, 0.95],
  driving_hypothesis: 'H1_whole_content_synthesis', statistical_estimate_before_provenance: Number(sigmoid(SUM + 0.04).toFixed(6)), disagreement: 0.2,
  contributions: ROWS,
} as Decision

/** Key-finding specimens (UX §5.4): one finding for each of the three largest rows of the specimen decision and one for its
 *  counter-evidence, each carrying the contribution of the check it cites, as the synthesis stage records it. Their families
 *  have typographic plates: the gallery holds no stored source image for a thumbnail. */
const FINDING_FAMILIES = ['F7_duplication', 'F5_compression_history', 'F8_metadata_consistency', 'F5_compression_history'] as const
export const SPECIMEN_FINDINGS: Finding[] = FINDING_FAMILIES.map((family, i) => {
  const [n, , delta] = CHECKS[i]
  const level = strength(delta)  // D-63: the words of a finding that entered the estimate come from its contribution
  const side = direction(delta)
  return {
    id: `F${i + 1}`, rank: i + 1, family, hypothesis: 'H1_whole_content_synthesis', lane: null, strength: level, direction: side, review_priority: 'medium',
    title: `Specimen finding ${i + 1}: ${level === 'none' ? 'little effect on the estimate' : `${level} support ${side} AI generation`}`,
    statement: `Specimen sentence about the check S${String(n).padStart(2, '0')}.`, sentences: [], evidence_ids: [`spec.detector_${n}:global`], scope: null,
    probability: null, contribution_logit: delta, corroborated_by: [], reference_range: '', benchmark_note: null, fp_modes: [], conditions: [],
    dissent_note: side === 'against' ? 'Specimen note: this check addresses a different question.' : null, examiner: null,
  } as unknown as Finding
})

/** The same decision with one extra row, so the rows no longer add up to the estimate (integrity-gate specimen). */
export const SPECIMEN_BROKEN_DECISION: Decision = {
  ...SPECIMEN_DECISION,
  contributions: [...ROWS, { signal_id: null, label: 'unrecorded specimen row', kind: 'detector', logit: 0.5, pp: 0, approximate: false }],
} as Decision

export function specimenAnalysis(decision: Decision = SPECIMEN_DECISION, over: Record<string, unknown> = {}): AnalysisDetail {
  return {
    id: 'specimen', kind: 'image', status: 'done', filename: 'specimen.jpg', sha256: 'a'.repeat(64), size_bytes: 1024, decision, review: null,
    coverage: { detectors_total: 12, by_status: { ok: 10, not_applicable: 1, failed: 1 } }, calibration: [
      { subject: 'spec.detector_2@1.0.0', method: 'platt', artefact_sha256: null, bench_version: 'specimen-bench', bench_manifest_sha256: null, ece: 0.04, brier: 0.12, reviewed_by: null,
        error_rates: [{ context: 'specimen photos', threshold_kind: 'conformal_0.05', threshold: 0.8, tpr: 0.71, fpr: 0.04, tpr_ci95: [0.6, 0.8], fpr_ci95: [0.02, 0.07], n_pos: 600, n_neg: 600, bench_version: 'specimen-bench' },
          { context: 'specimen small faces', threshold_kind: 'conformal_0.05', threshold: 0.8, tpr: 0.5, fpr: 0.065, tpr_ci95: [0.3, 0.7], fpr_ci95: [0.01, 0.2], n_pos: 40, n_neg: 46, bench_version: 'specimen-bench' }] },
    ], condition_profile: { slice_id: 'image/jpeg/side>=512/noface', within_validated_envelope: true, validation_support: 'direct', limiting_conditions: [] }, caveats: ['Specimen caveat.'],
    assessments: [], manifest: { veritas_version: '0.0.0-specimen', pipeline_version: 'img-specimen', profile: 'full', detectors: [
      { id: 'spec.detector_2', version: '1.0.0', weights_sha256: 'b'.repeat(64), revision: null, licence: 'MIT (specimen)', coverage_date: null, device: 'mps', precision: 'float16' },
    ], library_versions: {}, sampling: {}, disclosures: [] }, profile: 'full', effective_profile: 'full', revision: 1, ...over,
  } as unknown as AnalysisDetail
}

/** Methods & validation specimen (E-7): a dossier slice and a fuser record with error rates; synthetic numbers, labelled as such. */
export const SPECIMEN_VALIDATION: Record<string, unknown> = {
  validation_ref: { dossier_version: 'specimen-2026.10', slice_metrics: { tpr: 0.83, fpr: 0.04, auc: 0.93, n_pos: 612, n_neg: 904, tpr_ci95: [0.8, 0.86], fpr_ci95: [0.029, 0.055] } },
  calibration: [
    { subject: 'fuser-image-specimen', method: 'platt', artefact_sha256: 'f'.repeat(64), bench_version: 'specimen-bench', bench_manifest_sha256: 'e'.repeat(64), ece: 0.031, brier: 0.1, reviewed_by: 'specimen reviewer',
      error_rates: [{ context: 'specimen photos', threshold_kind: 'conformal_0.05', threshold: 0.8, tpr: 0.71, fpr: 0.04, tpr_ci95: [0.6, 0.8], fpr_ci95: [0.02, 0.07], n_pos: 600, n_neg: 600, bench_version: 'specimen-bench' }] },
  ],
}

export const SPECIMEN_MODEL_CARDS = [
  { id: 'spec.detector_2', name: 'Specimen detector 2', citation: 'Specimen et al., specimen venue 2025', admission: { decision: 'admitted', date: '2026-09-01', reviewed_by: 'specimen reviewer' },
    eval: [{ bench: 'specimen-bench', slice: 'image/jpeg/side>=512/noface', auc: 0.912, tpr_at_fpr01: 0.44, ece: 0.03, n: 1516, date: '2026-09-01' }] },
] as unknown as import('../../lib/api/queries').ModelCard[]

const table = (title: string, columns: string[], rows: (string | number | null)[][], caption = ''): TableExhibit =>
  ({ id: `spec-${title}`, kind: 'table', title, caption, columns, rows, source_signal_ids: [], created_by: 'specimen', media_ref: 'original', sha256: 'c'.repeat(64) }) as unknown as TableExhibit

export const SPECIMEN_TREE = table('Metadata tree', ['group', 'tag', 'value', 'severity', 'rule', 'note'], [
  ['JPEG structure', 'Quantisation tables', 'standard libjpeg, quality ≈ 75', 'notable', 'M5', 'Camera originals from this device use device tables'],
  ['JPEG structure', 'Embedded thumbnail', '160×120 · differs from main image', 'anomaly', 'M3', 'Main image changed after the thumbnail was written'],
  ['EXIF', 'Make · Model', 'Specimen Camera · Model 1', 'info', '', ''],
  ['EXIF', 'Software', 'Specimen Editor 25.2', 'notable', 'M1', 'Edited after capture'],
  ['EXIF', 'DateTimeOriginal', '2026:08:21 13:58:02', 'info', '', ''],
  ['EXIF', 'ModifyDate', '2026:08:22 09:10:44', 'notable', 'M2', 'Modified 19 h after capture'],
  ['EXIF', 'GPS', 'absent', 'info', '', 'Uninformative'],
  ['XMP', 'CreatorTool', 'Specimen Editor 25.2', 'info', '', ''],
  ['ICC profile', 'Description', 'Display P3', 'info', '', ''],
], 'Specimen tree: synthetic fields and severities.')

export const SPECIMEN_CREDENTIALS = table('C2PA Content Credentials', ['field', 'value'], [
  ['state', 'valid'], ['signer', 'Specimen Signer Inc.'], ['issuer', 'Specimen CA'], ['signed at', '2026-08-22T09:10:00Z'],
  ['claim generator', 'Specimen Generator 1.0'], ['actions', 'c2pa.created, c2pa.cropped'], ['digital source types', 'trainedAlgorithmicMedia (current)'],
  ['hash binding', 'matches the current bytes'],
])

export const SPECIMEN_CHAIN = table('C2PA ingredient chain', ['depth', 'title', 'relationship', 'format', 'signer', 'digital source types', 'recorded validation failures', 'manifest'], [
  ['0', 'specimen.jpg', 'active', 'image/jpeg', 'Specimen Signer Inc.', 'trainedAlgorithmicMedia', '0', 'urn:specimen:0'],
  ['1', 'beach.jpg', 'parentOf', 'image/jpeg', 'Specimen Camera AG', 'digitalCapture', '0', 'urn:specimen:1'],
  ['1', 'sky_gen.png', 'componentOf', 'image/png', 'Specimen Signer Inc.', 'trainedAlgorithmicMedia', '0', 'urn:specimen:2'],
])

export const SPECIMEN_CODEC = table('Codec facts', ['fact', 'value', 'reference'], [
  ['container', 'mov,mp4,m4a,3gp,3g2,mj2', null],
  ['video codec', 'h264 High yuv420p', null],
  ['frame rate, nominal / average', '30 / 29.97 fps', 'a difference suggests variable frame rate or frame-rate conversion'],
  ['GOP', '30 frames, regular', 'camera and phone encoders usually keep a fixed GOP (not measured here)'],
  ['frame timing', 'variable frame rate', 'phones and screen recorders often write variable frame rate'],
  ['edit lists', 'none', 'an empty edit delays a track; a media time well past the first frames hides them'],
])
