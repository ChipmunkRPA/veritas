/** Mirrored by engine/finding_assessment.py; fixtures verify interpretation and calibration gates. */
export interface FindingAssessment {
  version: string; finding_id: string | null
  included_in_overall: boolean | null
  direction: 'ai' | 'original' | 'inconclusive' | 'context'; label: string
  confidence_pct: number | null; confidence_label: string | null
  scope_label: string; confidence_note: string; interpretation: string
}
export interface FindingInput {
  id?: string; family: string; evidence_ids?: readonly string[] | null
  sentences?: readonly { text?: string; evidence_ids?: readonly string[] | null }[] | null
  scope?: unknown; standalone_direction?: string | null; direction?: string | null
  standalone_strength?: string | null; strength?: string | null; dissent_note?: string | null
}
interface SignalInput {
  id: string; detector_id?: string | null; detector_version?: string; category?: string; family?: string
  legacy?: { v1_id?: string | null; direction?: string } | null
  status?: string; scope?: string; observed?: string; calibrated_probability?: number | null
  calibration_ref?: string | null; reliability?: { factor?: number } | null; details?: Record<string, unknown>
}
export interface FindingContext {
  signals?: readonly SignalInput[]; status?: string
  condition_profile?: { within_validated_envelope?: boolean | null } | null
  calibration?: readonly { subject: string; method: string; bench_version?: string | null
    artefact_sha256?: string | null; bench_manifest_sha256?: string | null }[]
  kind?: string
  decision?: { fuser_sha256?: string | null; contributions?: readonly { kind: string; signal_id?: string | null; counted?: boolean | null; logit?: number | null }[] | null } | null
}
const CONTEXT = new Set(['coverage', 'F1_provenance', 'F2_self_declared_metadata', 'F5_compression_history',
  'F6_capture_pipeline', 'F7_duplication', 'F8_metadata_consistency', 'F9_distribution_conditions',
  'F11_physical_scene', 'V2_lip_sync', 'V4_codec'])
const SCOPES: Record<string, string> = { F3_whole_image_generation: 'Whole-image generation indicators',
  F4_localised_manipulation: 'Local editing indicators', F10_face: 'Assessed face content', V1_face_track: 'Sampled face content',
  V3_synthetic_video: 'Sampled visual content', A1_voice: 'Speech in the assessed recording', T1_ai_text: 'Assessed writing' }
const LABELS = { ai: 'Leans AI', original: 'Leans original', inconclusive: 'Inconclusive', context: 'Context only' }
const probability = (p: unknown): p is number => typeof p === 'number' && Number.isFinite(p) && p >= 0 && p <= 1
const pct = (p: number) => Math.min(99.9, Math.floor(Number((p * 1000).toFixed(8))) / 10)
const hex = (v: unknown) => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v)

function linked(finding: FindingInput, result: FindingContext) {
  const refs = new Set([...(finding.evidence_ids ?? []), ...(finding.sentences ?? []).flatMap(s => s.evidence_ids ?? [])])
  return (result.signals ?? []).filter(s => [s.id, s.legacy?.v1_id, `detector:${s.detector_id}`].some(r => r && refs.has(r)))
}

export function findingObservations(finding: FindingInput, result: FindingContext): string[] {
  if (['F1_provenance', 'F2_self_declared_metadata'].includes(finding.family)
    && (finding.sentences ?? []).some(s => s.text?.includes("sets the result's number at no less than"))) {
    return [...linked(finding, result).filter(s => (s.status ?? 'ok') === 'ok' && s.observed).map(s => s.observed!),
      'Historical unverified marker. It does not authenticate AI generation or support a confidence percentage. The saved calculation used a retired marker rule.']
  }
  if (['coverage', 'F9_distribution_conditions', 'F1_provenance', 'F2_self_declared_metadata', 'F8_metadata_consistency'].includes(finding.family)) {
    const facts = (finding.sentences ?? []).map(s => s.text).filter((text): text is string => typeof text === 'string' && text.length > 0)
    if (facts.length) return facts
  }
  const observations = linked(finding, result).filter(s => (s.status ?? 'ok') === 'ok' && s.observed).map(s => {
    const text = s.observed!
    const frames = text.match(/(\d+) of (\d+) frame\(s\) at logit > 0/)
    if (frames) return `${frames[1]} of ${frames[2]} sampled frames were flagged for inspection. This does not assess every frame or the soundtrack.`
    const faces = text.match(/(\d+)\/(\d+) face frames > [\d.]+/)
    if (faces) return `${faces[1]} of ${faces[2]} sampled face frames were flagged for inspection. This does not locate edited pixels or establish a face swap.`
    if (text.includes('mean p(synthetic)=')) return `${text.split(/; per (?:model|check):/)[0].replace(/mean p\(synthetic\)=/g, 'Raw speech inspection response ')}. This response is not a calibrated confidence percentage.`
    return text.split(/; per (?:model|check):/)[0].replace(/p\(fake\)=/g, 'inspection response ').replace(/fused p\(synthetic\)=/g, 'Combined voice-check response ')
  })
  return observations.length ? [...new Set(observations)] : ['No separate scored observation is available here. Review the linked source locations and check records.']
}

export function findingAssessment(finding: FindingInput, result: FindingContext): FindingAssessment {
  const signals = linked(finding, result)
  let scopeLabel = SCOPES[finding.family] ?? 'Recorded observation'
  if (finding.scope && typeof finding.scope === 'object') {
    const scope = finding.scope as Record<string, unknown>
    scopeLabel = ['bbox', 'polygon', 'mask'].includes(String(scope.kind)) ? 'Selected image region'
      : 'start_s' in scope ? 'Selected time interval' : 'Selected face track'
  } else if (signals.length === 1) {
    const labels: Record<string, string> = { window: 'Selected recording window', segment: 'Selected time interval',
      sentence: 'Selected writing context', region: 'Selected image region', face: 'Selected face input', track: 'Selected face track' }
    scopeLabel = labels[signals[0].scope ?? ''] ?? scopeLabel
  }
  const refs = new Set(signals.flatMap(s => [s.id, s.legacy?.v1_id, `detector:${s.detector_id}`]).filter(Boolean))
  const rows = result.decision?.contributions
  const included = rows == null ? null : rows.some(r => r.kind === 'detector' && refs.has(r.signal_id ?? '') && r.counted !== false
    && (r.counted === true || (typeof r.logit === 'number' && Math.abs(r.logit) > 1e-9)))
  const base: FindingAssessment = { version: 'finding-interpretation/v1', finding_id: finding.id ?? null,
    included_in_overall: included,
    direction: 'inconclusive', label: LABELS.inconclusive, confidence_pct: null, confidence_label: null,
    scope_label: scopeLabel,
    confidence_note: 'A calibrated confidence percentage has not been established for this finding. Raw responses and contribution strengths are not AI probabilities.',
    interpretation: 'The recorded evidence does not establish an AI or original tendency for this finding.' }
  const learned = signals.some(s => s.category === 'model' || s.family?.startsWith('learned-')
    || (finding.family === 'A1_voice' && s.details?.audio_screening))
  if (CONTEXT.has(finding.family) || (signals.length > 0 && !learned)) return {
    ...base, direction: 'context', label: LABELS.context, included_in_overall: false,
    interpretation: 'This records provenance, timing, file history or conditions. It does not by itself establish AI generation or original authorship.' }
  if ((result.status != null && result.status !== 'done') || !signals.length || signals.some(s => (s.status ?? 'ok') !== 'ok')) return {
    ...base, interpretation: 'The checks needed for this finding are missing, unavailable or incomplete. Missing evidence does not favour original content.' }
  const individual = new Set(signals.map(s => s.legacy?.direction))
  if ((individual.has('supports_ai') && individual.has('supports_authentic')) || finding.dissent_note) return {
    ...base, interpretation: 'The linked checks disagree. Their responses are not averaged into a finding confidence.' }
  const standalone = finding.standalone_direction || finding.direction
  const strength = finding.standalone_strength || finding.strength
  let direction: 'ai' | 'original' | null = strength && strength !== 'none'
    ? standalone === 'for' ? 'ai' : standalone === 'against' ? 'original' : null : null
  if (direction) Object.assign(base, { direction, label: LABELS[direction], interpretation: direction === 'ai'
    ? 'The recorded check provides an AI-leaning inspection clue; it does not prove AI origin.'
    : 'The recorded check leans original in its assessed scope. Absence of an AI flag does not prove the file is original.' })
  if (signals.length !== 1 || finding.scope != null || (signals[0].scope ?? 'global') !== 'global') return base
  const signal = signals[0]
  let p: unknown = signal.calibrated_probability
  const ref = signal.calibration_ref
  let supported = result.condition_profile?.within_validated_envelope === true
  let calibrated = false
  const audio = signal.details?.audio_screening as Record<string, unknown> | undefined
  if (finding.family === 'A1_voice' && audio && typeof audio === 'object') {
    const sha = '912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf'
    supported = audio.scope_supported === true && ref === sha && audio.calibration_sha256 === sha
    direction = audio.decision === 'ai' || audio.decision === 'original' ? audio.decision : null
    if (direction === 'original' && audio.original_stability_veto !== false) direction = null
    p = audio.probability_ai
    if (!direction || !supported) return { ...base, direction: 'inconclusive', label: LABELS.inconclusive,
      interpretation: 'The saved speech calibration does not support a reliable tendency for this recording.' }
    Object.assign(base, { direction, label: LABELS[direction], interpretation: direction === 'ai'
      ? 'The calibrated speech check leans AI within the assessed recording.' : 'The calibrated speech check leans original within the assessed recording.' })
    calibrated = true
    if (result.kind === 'audio' && result.decision?.fuser_sha256 === sha) base.included_in_overall = true
  } else {
    const subject = `${signal.detector_id}@${signal.detector_version}`
    calibrated = (result.calibration ?? []).some(c => c.subject === subject && ['platt', 'isotonic'].includes(c.method)
      && !!c.bench_version && ref === `${subject}/${c.bench_version}` && hex(c.artefact_sha256) && hex(c.bench_manifest_sha256))
  }
  if (!calibrated || !supported || !direction || !probability(p) || signal.reliability?.factor === 0) return base
  if ((direction === 'ai' && p <= .5) || (direction === 'original' && p >= .5)) return { ...base,
    direction: 'inconclusive', label: LABELS.inconclusive,
    interpretation: 'The recorded tendency and calibrated estimate disagree. No finding confidence is assigned.' }
  return { ...base, confidence_pct: pct(direction === 'ai' ? p : 1 - p), confidence_label: 'Reference-calibrated estimate for this check',
    confidence_note: 'This estimate applies to the stated check and assessed input. It is not measured accuracy for this file or a probability for every highlighted pixel, frame, word or sound.' }
}

export function isFindingAssessment(value: unknown): value is FindingAssessment {
  if (!value || typeof value !== 'object') return false
  const row = value as Record<string, unknown>
  return row.version === 'finding-interpretation/v1' && typeof row.finding_id === 'string'
    && (row.included_in_overall === null || typeof row.included_in_overall === 'boolean')
    && ['ai', 'original', 'inconclusive', 'context'].includes(String(row.direction))
    && (row.confidence_pct === null || (typeof row.confidence_pct === 'number' && row.confidence_pct >= 50 && row.confidence_pct < 100))
    && ['label', 'scope_label', 'confidence_note', 'interpretation'].every(k => typeof row[k] === 'string')
    && (row.confidence_label === null || typeof row.confidence_label === 'string')
}
