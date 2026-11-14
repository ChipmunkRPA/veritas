import { findingAssessment, isFindingAssessment, type FindingAssessment, type FindingInput, type FindingContext } from './findingAssessment'
import { STANDARD_SERVICE_POLICY, type StandardServicePolicy } from './standardServicePolicy'
import { ruleTaken, type Decision } from './answer'
import { outsideValidated } from './record'
import { imageAssessmentScope } from './imageAssessmentScope'

/** Mirrored by engine/customer_assessment.py; the parity fixtures test both implementations. */
export interface CustomerAssessment {
  rule: string; finding_assessments: FindingAssessment[]; applies: boolean; service_policy: StandardServicePolicy
  direction: 'ai' | 'original' | 'inconclusive' | 'unavailable' | 'record'
  headline: string | null; confidence_pct: number | null; confidence_label: string | null
  probability_ai_pct: number | null; interval_pct: [number, number] | null
  interval_label: string | null; interval_note: string; meaning: string; limitation: string
  scope: 'whole_file' | 'visual_content' | 'assessed_components' | 'image_components'; scope_label: string; scope_note: string | null
  excluded_channels: ('voice' | 'audio_picture_timing')[]
  soundtrack?: { direction: 'ai' | 'original' | 'inconclusive'; headline: string; confidence_pct: number | null; limitation: string }
  component_estimate?: { direction: 'ai' | 'original'; headline: string; confidence_pct: number; confidence_label: string;
    probability_ai_pct: number | null; interval_pct: [number, number] | null; interval_label: string | null; scope_label: string }
}
type Signal = { id: string; detector_id?: string | null; family?: string | null; modality?: string | null; lane?: string | null
  hypothesis?: string | null; metrics?: Record<string, unknown> | null; details?: Record<string, unknown> | null
  legacy?: { v1_id?: string | null } | null; status?: string | null; outside_estimate?: unknown; effective_weight?: number | null
  segments?: readonly unknown[] | null }
type Input = FindingContext & {
  decision?: Decision | null; status?: string; stored_record?: unknown
  kind?: string; signals?: readonly Signal[]; findings?: readonly FindingInput[]
  condition_profile?: { within_validated_envelope?: boolean | null; slice_id?: string | null } | null
  manifest?: { sampling?: { image_frame_count?: number | null } | null } | null
  caveats?: readonly string[]
}
export const ASSESSMENT_RULE = 'evidence-led-assessment/v7'
const minLandmarkTimingChangeWindows = 4
const INTERVAL_NOTE = 'This range describes uncertainty in the combined estimate. It does not include every detection error or unfamiliar type of content; a 90% interval does not mean the conclusion is 90% accurate.'
const ESTIMATE_NOTE = "This is the system's estimated likelihood, not an independently established accuracy rate or proof of origin. An original-file tendency does not establish that every part of the file is unedited."
const probability = (p: unknown): p is number => typeof p === 'number' && Number.isFinite(p) && p >= 0 && p <= 1
const pct = (p: number) => Math.floor(Number((p * 1000).toFixed(8))) / 10


function signalRefs(signal: Signal): Set<string> {
  return new Set([signal.id, signal.legacy?.v1_id, signal.detector_id ? `detector:${signal.detector_id}` : null]
    .filter((ref): ref is string => !!ref))
}

function counted(row: { counted?: boolean | null; logit?: number | null }): boolean {
  return row.counted === true || (row.counted !== false && typeof row.logit === 'number'
    && Number.isFinite(row.logit) && Math.abs(row.logit) > 1e-9)
}

type AssessmentScope = Pick<CustomerAssessment, 'scope' | 'scope_label' | 'scope_note' | 'excluded_channels'>

function videoScope(result: Input): [AssessmentScope, boolean] {
  const scope = { scope: 'whole_file' as const, scope_label: 'Assessed content', scope_note: 'The overall estimate covers only the checks included in it. Findings assessed separately do not enter this percentage; unassessed regions and types of editing remain unresolved.', excluded_channels: [] }
  if (result.kind !== 'video') return [scope, true]
  const rows = result.decision?.contributions?.filter(r => r.kind === 'detector') ?? []
  const included = new Set<string>()
  const statuses = { visual: new Set<string>(), voice: new Set<string>(), audio_picture_timing: new Set<string>() }
  for (const signal of result.signals ?? []) {
    const refs = signalRefs(signal)
    const families = new Set((result.findings ?? []).filter(f => f.evidence_ids?.some(id => refs.has(id))).map(f => f.family))
    const detector = signal.detector_id ?? ''
    const channel = signal.family === 'av-sync' || signal.lane === 'lip_sync' || families.has('V2_lip_sync') || detector.startsWith('vid.lipsync.') ? 'audio_picture_timing'
      : signal.family === 'voice' || signal.modality === 'audio' || families.has('A1_voice') || detector.startsWith('aud.') ? 'voice'
        : ['image', 'video'].includes(signal.modality ?? '') || /^(img|vid|face)\./.test(detector)
          || ['F3_whole_image_generation', 'F4_localised_manipulation', 'F10_face', 'V1_face_track', 'V3_synthetic_video'].some(f => families.has(f)) ? 'visual' : null
    if (channel === null) continue
    let status = signal.status || 'unknown'
    const windows = signal.metrics?.windows
    if (channel === 'audio_picture_timing' && status === 'ok' && typeof windows === 'number'
      && (windows === 0 || (signal.metrics?.longest_measured_run !== undefined && windows > 0 && windows < minLandmarkTimingChangeWindows))
      && !signal.segments?.length && !signal.metrics?.steps && !signal.metrics?.collapses) status = 'insufficient_timing_windows'
    statuses[channel].add(status)
    const matched = rows.filter(r => r.signal_id != null && refs.has(r.signal_id))
    const omitted = !!signal.outside_estimate || signal.effective_weight === 0 || matched.some(r => r.counted === false)
    if (status === 'ok' && !omitted && matched.some(counted)) included.add(channel)
  }
  const labels = { visual: 'Visual content', voice: 'Voice/audio', audio_picture_timing: 'Audio/picture timing' }
  const subjects = ([['visual', 'sampled visual content'], ['voice', 'speech'], ['audio_picture_timing', 'audio/picture timing']] as const)
    .filter(([channel]) => included.has(channel)).map(([, label]) => label)
  const notes = [subjects.length ? `The main assessment uses ${subjects.join(' and ')}.`
    : 'The saved result does not establish which content checks contributed to the main assessment.']
  for (const channel of ['visual', 'voice', 'audio_picture_timing'] as const) {
    const states = statuses[channel], label = labels[channel]
    if (states.has('ok') && !included.has(channel)) notes.push(`${label} was checked separately from the main assessment.`)
    if (['failed', 'timeout', 'timed_out'].some(status => states.has(status))) notes.push(`Some ${label.toLowerCase()} checks could not complete.`)
    if (states.has('skipped')) notes.push(`Some ${label.toLowerCase()} checks were skipped.`)
    if (states.has('not_applicable')) notes.push(`Some ${label.toLowerCase()} checks were unavailable or not applicable.`)
    if (states.has('insufficient_timing_windows')) notes.push('Audio/picture timing had too few measured speech windows to assess a change on at least one track.')
    if (!states.size || (states.size === 1 && states.has('unknown'))) notes.push(`${label} check coverage is unavailable in the saved result.`)
  }
  const visualOnly = included.size === 1 && included.has('visual')
  return [{ scope: visualOnly ? 'visual_content' : 'assessed_components',
    scope_label: visualOnly ? 'Visual content only' : included.size ? 'Assessed components only' : 'Check coverage not yet available',
    scope_note: notes.join(' '), excluded_channels: (['voice', 'audio_picture_timing'] as const).filter(key => !included.has(key)) }, included.size > 0]
}

function unavailableContributor(result: Input): boolean {
  const rows = result.decision?.contributions?.filter(r => r.kind === 'detector' && counted(r)) ?? []
  const available = new Set((result.signals ?? []).filter(signal => signal.status === 'ok').flatMap(signal => [...signalRefs(signal)]))
  return rows.some(row => row.signal_id == null || !available.has(row.signal_id))
}

/** Prefer the API's shared presentation; old or mismatched servers use the tested local mirror. */
export function presentedCustomerAssessment(result: Input & { customer_assessment?: unknown }): CustomerAssessment {
  const value = result.customer_assessment
  if (value && typeof value === 'object') {
    const row = value as Record<string, unknown>
    const percent = (p: unknown) => p === null || (typeof p === 'number' && Number.isFinite(p) && p >= 0 && p <= 100)
    const text = (x: unknown) => x === null || typeof x === 'string'
    const range = row.interval_pct
    const soundtrack = row.soundtrack as Record<string, unknown> | null | undefined
    const component = row.component_estimate as Record<string, unknown> | null | undefined
    const componentValid = component === undefined || (result.kind === 'image' && component !== null && typeof component === 'object'
      && ['ai', 'original'].includes(String(component.direction)) && typeof component.confidence_pct === 'number'
      && percent(component.confidence_pct) && percent(component.probability_ai_pct)
      && ['headline', 'confidence_label', 'scope_label'].every(k => typeof component[k] === 'string') && text(component.interval_label)
      && (component.interval_pct === null || (Array.isArray(component.interval_pct) && component.interval_pct.length === 2
        && component.interval_pct.every(x => typeof x === 'number' && percent(x)) && component.interval_pct[0] <= component.interval_pct[1])))
    if ((row.service_policy as StandardServicePolicy | undefined)?.version === STANDARD_SERVICE_POLICY.version
      && row.rule === ASSESSMENT_RULE && componentValid && Array.isArray(row.finding_assessments) && row.finding_assessments.every(isFindingAssessment) && typeof row.applies === 'boolean'
      && ['ai', 'original', 'inconclusive', 'unavailable', 'record'].includes(String(row.direction))
      && text(row.headline) && text(row.confidence_label) && text(row.interval_label)
      && percent(row.confidence_pct) && percent(row.probability_ai_pct)
      && (range === null || (Array.isArray(range) && range.length === 2 && range.every(x => typeof x === 'number' && percent(x)) && range[0] <= range[1]))
      && ['whole_file', 'visual_content', 'assessed_components', 'image_components'].includes(String(row.scope)) && typeof row.scope_label === 'string' && text(row.scope_note)
      && Array.isArray(row.excluded_channels) && row.excluded_channels.every(c => c === 'voice' || c === 'audio_picture_timing')
      && (soundtrack === undefined || (result.kind === 'video' && soundtrack !== null && typeof soundtrack === 'object'
        && ['ai', 'original', 'inconclusive'].includes(String(soundtrack.direction))
        && typeof soundtrack.headline === 'string' && typeof soundtrack.limitation === 'string'
        && (soundtrack.direction === 'inconclusive' ? soundtrack.confidence_pct === null
          : typeof soundtrack.confidence_pct === 'number' && soundtrack.confidence_pct >= 60 && soundtrack.confidence_pct < 100)))
      && ['meaning', 'limitation', 'interval_note'].every(key => typeof row[key] === 'string')) return row as unknown as CustomerAssessment
  }
  return customerAssessment(result)
}

function baseCustomerAssessment(result: Input): CustomerAssessment {
  const d = result.decision
  const profile = result.condition_profile
  const [scope, contentAssessed] = videoScope(result)
  const answer: CustomerAssessment = {
    rule: ASSESSMENT_RULE, finding_assessments: (result.findings ?? []).filter(f => /^F[1-9]\d*$/.test(f.id ?? '')).map(f => findingAssessment(f, result)), applies: true, service_policy: STANDARD_SERVICE_POLICY, direction: 'unavailable',
    headline: 'No reliable automated conclusion', confidence_pct: null, confidence_label: null,
    probability_ai_pct: null, interval_pct: null, interval_label: null, interval_note: INTERVAL_NOTE,
    meaning: 'The analysis did not establish a usable whole-file estimate.',
    limitation: 'Review the recorded observations and any checks that could not run. Missing evidence does not favour either answer.',
    ...scope,
  }
  const contentFindings = answer.finding_assessments.filter(f => f.direction !== 'context')
  const included = [...new Set(contentFindings.filter(f => f.included_in_overall === true).map(f => f.scope_label))]
  const separate = [...new Set(contentFindings.filter(f => f.included_in_overall === false).map(f => f.scope_label))]
  if (included.length || separate.length) answer.scope_note = [answer.scope_note,
    included.length ? `Included finding scopes: ${included.join('; ')}.` : null,
    separate.length ? `Finding scopes assessed separately from this percentage: ${separate.join('; ')}.` : null,
  ].filter(Boolean).join(' ')
  const originRecord = d?.provenance_record ?? result.stored_record
  if (originRecord && typeof originRecord === 'object' && 'kind' in originRecord && originRecord.kind === 'self_declared_ai') return {
    ...answer, direction: 'inconclusive', headline: 'Inconclusive: unverified origin marker',
    meaning: 'This saved result used a retired rule that let an unverified marker decide the answer. The marker does not authenticate AI generation.',
    limitation: 'The original calculation is retained in the analysis record. Run a new analysis for a content-based assessment; no confidence percentage is assigned to the old marker-led verdict.',
  }
  if (originRecord || d?.driving_hypothesis === 'H3_provenance_integrity') return {
    ...answer, applies: false, direction: 'record', headline: null,
    meaning: "The conclusion follows the file's provenance record, not a confidence percentage.",
  }
  const frameCount = result.manifest?.sampling?.image_frame_count
  if (result.kind === 'image' && typeof frameCount === 'number' && Number.isInteger(frameCount) && frameCount > 1) return {
    ...answer, direction: 'inconclusive', headline: 'Inconclusive: multiple images in this file',
    meaning: 'The still-image checks cover only the first stored image.',
    scope: 'image_components', scope_label: 'First stored image',
    scope_note: `The header reports ${frameCount} stored images. No combined AI confidence is available for the sequence.`,
  }
  if ((result.status != null && result.status !== 'done') || !d || d.code === 'not_assessable') return answer
  if (unavailableContributor(result)) return { ...answer, direction: 'inconclusive',
    headline: 'Inconclusive: contributing checks could not complete',
    meaning: 'A check required by this estimate failed or was unavailable. Review the recorded check coverage.' }
  if (result.kind === 'audio' && d.fuser_sha256 === '912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf') {
    const p = d.probability_ai_calibrated
    const direction = d.code === 'ai_supported' && probability(p) && p >= .6 ? 'ai'
      : d.code === 'no_ai_indicators' && probability(p) && p <= .4 + 1e-12 ? 'original' : 'inconclusive'
    return { ...answer, direction, headline: direction === 'ai' ? 'Leaning towards AI-generated speech' : direction === 'original' ? 'Leaning towards original speech' : 'Inconclusive: no reliable speech tendency',
      confidence_pct: direction !== 'inconclusive' && probability(p) ? Math.min(99.9, pct(direction === 'ai' ? p : 1 - p)) : null,
      confidence_label: direction !== 'inconclusive' ? 'Reference-calibrated estimate' : null,
      probability_ai_pct: probability(p) ? pct(p) : null, meaning: direction !== 'inconclusive' ? 'The recorded speech checks support this tendency only within the assessed speech and its calibrated scope.' : 'The recorded speech checks do not support a reliable directional conclusion in their calibrated scope.',
      limitation: 'Local screening estimate calibrated on a balanced reference set of authentic and AI speech. It is not a measured accuracy rate for this recording. Validation used short English speech; heavy background sound can cause errors or an inconclusive result. No per-file confidence interval is established.',
      scope_label: 'Speech assessment', scope_note: 'The check assesses the speech interval as a whole. When voices overlap, the check cannot identify which speaker drove the result.',
      interval_note: 'No per-file confidence interval has been established for this audio calibration.' }
  }
  if (d.fusion_mode !== 'artefact') return { ...answer, headline: 'Inconclusive: no calibrated estimate',
    meaning: 'The available checks provide inspection scores, but those scores have not been calibrated into a reliable whole-file likelihood.',
    limitation: 'An inspection response is not a calibrated confidence percentage. Use the linked evidence to examine the file further.',
  }
  if (outsideValidated(d, profile) || profile?.within_validated_envelope !== true) return { ...answer,
    headline: 'Inconclusive: this input is outside the verified scope',
    meaning: 'The saved validation does not cover this kind of input well enough to support a reliable combined assessment.',
    limitation: 'A score can still be recorded for inspection, but it cannot support a confidence percentage for this file.',
  }
  const p = d.probability_ai_calibrated
  if (result.kind === 'document' && profile?.slice_id === 'document/local-email-v1') {
    const ai = d.code === 'ai_supported' && probability(p) && p >= .6
    const ci = d.ci90
    const validInterval = ai && ci?.length === 2 && probability(ci[0]) && probability(ci[1]) && ci[0] <= ci[1]
    return { ...answer, direction: ai ? 'ai' : 'inconclusive',
      headline: ai ? 'Leaning towards AI-written or AI-rewritten email text' : 'Inconclusive: no reliable AI-writing tendency',
      confidence_pct: ai ? Math.min(99.9, pct(p!)) : null, confidence_label: ai ? 'Reference-calibrated estimate' : null,
      probability_ai_pct: ai ? pct(p!) : null, interval_pct: validInterval ? [pct(ci[0]), pct(ci[1])] : null,
      interval_label: validInterval ? '90% reference-fit interval' : null,
      interval_note: 'The range reflects uncertainty in the fitted reference. It does not measure all possible detector errors.',
      meaning: ai ? 'The reference-calibrated email-body check supports an AI-writing tendency for the selected message body.' : 'The recorded email-body check does not establish a reliable authorship tendency.', limitation: 'Some AI rewrites are missed, so this check can leave email authorship inconclusive.',
      scope_label: 'Selected email body',
      scope_note: 'This assessment covers writing in the message body. Sender identity and attachments require separate verification.' }
  }
  if (!probability(p)) return answer
  const textReference = result.kind === 'document' && profile?.slice_id === 'document/local-article-v1'
  Object.assign(answer, { direction: 'inconclusive', headline: 'Inconclusive: no reliable overall tendency',
    probability_ai_pct: pct(p), meaning: 'The recorded estimate does not establish a reliable tendency. Review the observations, source links and check coverage.', limitation: ESTIMATE_NOTE })
  if (textReference) Object.assign(answer, { scope_label: 'Extracted writing',
    scope_note: 'Reference based on English news and articles. The estimate covers the extracted writing.',
    interval_note: 'The range reflects uncertainty in the fitted reference. It does not measure all possible detector errors.' })
  const ci = d.ci90
  if (ci?.length === 2 && probability(ci[0]) && probability(ci[1]) && ci[0] <= ci[1]) {
    answer.interval_pct = [pct(ci[0]), pct(ci[1])]
    answer.interval_label = '90% interval for AI likelihood'
  }
  if (d.code === 'inconclusive' || ruleTaken(d) || result.caveats?.some(c => c.toLowerCase().includes('strongly disagree')) || !contentAssessed) return { ...answer,
    headline: 'Inconclusive: the checks do not support a reliable tendency',
    meaning: 'The recorded result remains inconclusive. Review the linked findings and the coverage of checks that could not complete.',
  }
  const direction = ['ai_supported', 'ai_supported_strong'].includes(d.code) && p >= .6 ? 'ai'
    : ['no_ai_indicators', 'no_ai_indicators_strong'].includes(d.code) && p <= .4 + 1e-12 ? 'original' : null
  if (!direction) {
    if (p >= .6 || p <= .4 + 1e-12) return { ...answer, headline: 'Inconclusive: the checks do not support a reliable tendency',
      meaning: 'The recorded decision does not support the direction suggested by the estimate.' }
    return answer
  }
  const ai = direction === 'ai'
  Object.assign(answer, { direction,
    headline: ai ? 'Leaning towards AI-generated or AI-altered content' : 'Leaning towards original content in the assessed scope',
    confidence_pct: Math.min(99.9, pct(ai ? p : 1 - p)), confidence_label: 'Reference-calibrated overall estimate',
    meaning: ai ? 'The applicable, reference-calibrated checks lean towards AI generation or alteration. Review the supporting findings and their scope.'
      : 'The applicable, reference-calibrated checks lean towards original content in the assessed scope. Local flags and checks outside the estimate require separate consideration.',
  })
  if (!ai && answer.interval_pct && ci) {
    answer.interval_pct = [pct(1 - ci[1]), pct(1 - ci[0])]
    answer.interval_label = '90% interval for original-file likelihood'
  }
  if (textReference) Object.assign(answer, {
    headline: ai ? 'Leaning towards AI-generated or AI-rewritten text' : 'Leaning towards original writing',
    confidence_label: 'Reference-calibrated estimate',
    meaning: 'The reference-calibrated writing check supports this tendency for the assessed writing, not the authorship of every highlighted passage.',
    interval_label: answer.interval_pct ? '90% reference-fit interval' : null,
  })
  if (answer.scope !== 'whole_file') {
    const subject = answer.scope === 'visual_content' ? 'visual content' : 'content in the assessed components'
    answer.headline = ai ? `Leaning towards AI-generated or AI-altered ${subject}` : `Leaning towards original ${subject}`
    answer.meaning = `The reference-calibrated checks included in this estimate lean towards ${ai ? 'AI generation or alteration of' : 'original'} ${subject}. Local flags and excluded checks require separate consideration.`
    if (answer.interval_pct) answer.interval_label = '90% interval for this scoped estimate'
  }
  return answer
}

export function customerAssessment(result: Input): CustomerAssessment {
  return imageAssessmentScope(result, baseCustomerAssessment(result))
}
