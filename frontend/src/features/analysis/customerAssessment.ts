import { ruleTaken, type Decision } from './answer'
import { outsideValidated } from './record'

/** Mirrored by engine/customer_assessment.py; the parity fixtures test both implementations. */
export interface CustomerAssessment {
  rule: string; threshold_pct: number; applies: boolean
  direction: 'ai' | 'original' | 'inconclusive' | 'unavailable' | 'record'
  headline: string | null; confidence_pct: number | null; confidence_label: string | null
  probability_ai_pct: number | null; interval_pct: [number, number] | null
  interval_label: string | null; interval_note: string; meaning: string; limitation: string
  scope: 'whole_file' | 'visual_content' | 'assessed_components'; scope_label: string; scope_note: string | null
  excluded_channels: ('voice' | 'audio_picture_timing')[]
}
type Signal = { id: string; detector_id?: string | null; family?: string | null; modality?: string | null; lane?: string | null
  legacy?: { v1_id?: string | null } | null; status?: string | null; outside_estimate?: unknown; effective_weight?: number | null }
type Input = {
  decision?: Decision | null; status?: string; stored_record?: unknown
  kind?: string; signals?: readonly Signal[]; findings?: readonly { family: string; evidence_ids?: readonly string[] | null }[]
  condition_profile?: { within_validated_envelope?: boolean | null } | null
  caveats?: readonly string[]
}
export const ASSESSMENT_RULE = '80-percent-leaning/v1'
const INTERVAL_NOTE = 'This range describes uncertainty in the combined estimate. It does not include every detection error or unfamiliar type of content; a 90% interval does not mean the conclusion is 90% accurate.'
const ESTIMATE_NOTE = "This is the system's estimated likelihood, not an independently established accuracy rate or proof of origin. An original-file tendency does not establish that every part of the file is unedited."
const probability = (p: unknown): p is number => typeof p === 'number' && Number.isFinite(p) && p >= 0 && p <= 1
const pct = (p: number) => Math.floor(Number((p * 1000).toFixed(8))) / 10

function videoScope(result: Input): Pick<CustomerAssessment, 'scope' | 'scope_label' | 'scope_note' | 'excluded_channels'> {
  const scope = { scope: 'whole_file' as const, scope_label: 'Combined file assessment', scope_note: null, excluded_channels: [] }
  if (result.kind !== 'video') return scope
  if (!result.signals?.length) return { scope: 'assessed_components', scope_label: 'Check coverage not yet available',
    scope_note: 'The contribution of picture, voice and timing checks has not been established from the available record. This estimate must not be read as clearing the whole video.', excluded_channels: [] }
  const rows = result.decision?.contributions?.filter(r => r.kind === 'detector') ?? []
  const channels = new Map<string, string>()
  const excluded = new Set<'voice' | 'audio_picture_timing'>()
  for (const signal of result.signals ?? []) {
    const refs = new Set([signal.id, signal.legacy?.v1_id, `detector:${signal.detector_id}`].filter((s): s is string => s != null))
    const families = new Set((result.findings ?? []).filter(f => f.evidence_ids?.some(id => refs.has(id))).map(f => f.family))
    const detector = signal.detector_id ?? ''
    const channel = signal.family === 'av-sync' || signal.lane === 'lip_sync' || families.has('V2_lip_sync') || detector.startsWith('vid.lipsync.') ? 'audio_picture_timing'
      : signal.family === 'voice' || signal.modality === 'audio' || families.has('A1_voice') || detector.startsWith('aud.') ? 'voice'
        : ['image', 'video'].includes(signal.modality ?? '') || /^(img|vid|face)\./.test(detector)
          || ['F3_whole_image_generation', 'F4_localised_manipulation', 'F10_face', 'V1_face_track', 'V3_synthetic_video'].some(f => families.has(f)) ? 'visual' : 'other'
    for (const ref of refs) channels.set(ref, channel)
    const omitted = !!signal.outside_estimate || signal.effective_weight === 0 || rows.some(r => r.signal_id != null && refs.has(r.signal_id) && r.counted === false)
    if ((signal.status ?? 'ok') === 'ok' && omitted && (channel === 'voice' || channel === 'audio_picture_timing')) excluded.add(channel)
  }
  if (!excluded.size) return scope
  const counted = rows.filter(r => r.counted === true || (r.counted !== false && typeof r.logit === 'number' && Math.abs(r.logit) > 1e-9))
  const visualOnly = counted.length > 0 && counted.every(r => channels.get(String(r.signal_id)) === 'visual')
  const omittedLabels = ([['voice', 'voice/audio evidence'], ['audio_picture_timing', 'audio/picture timing evidence']] as const)
    .filter(([channel]) => excluded.has(channel)).map(([, label]) => label)
  const verb = omittedLabels.length > 1 ? 'were' : 'was'
  const note = (visualOnly ? 'This percentage applies to the visual checks that contributed to the estimate. '
    : 'This percentage applies only to the checks that contributed to the estimate. ')
    + `The ${omittedLabels.join(' and ')} ${verb} checked separately but ${verb} not included in this percentage.`
    + (excluded.has('voice') ? ' The audio has not been cleared as original.' : '')
  return { scope: visualOnly ? 'visual_content' : 'assessed_components', scope_label: visualOnly ? 'Visual content only' : 'Assessed components only',
    scope_note: note, excluded_channels: (['voice', 'audio_picture_timing'] as const).filter(key => excluded.has(key)) }
}

/** Prefer the API's shared presentation; old or mismatched servers use the tested local mirror. */
export function presentedCustomerAssessment(result: Input & { customer_assessment?: unknown }): CustomerAssessment {
  const value = result.customer_assessment
  if (value && typeof value === 'object') {
    const row = value as Record<string, unknown>
    const percent = (p: unknown) => p === null || (typeof p === 'number' && Number.isFinite(p) && p >= 0 && p <= 100)
    const text = (x: unknown) => x === null || typeof x === 'string'
    const range = row.interval_pct
    if (row.rule === ASSESSMENT_RULE && row.threshold_pct === 80 && typeof row.applies === 'boolean'
      && ['ai', 'original', 'inconclusive', 'unavailable', 'record'].includes(String(row.direction))
      && text(row.headline) && text(row.confidence_label) && text(row.interval_label)
      && percent(row.confidence_pct) && percent(row.probability_ai_pct)
      && (range === null || (Array.isArray(range) && range.length === 2 && range.every(x => typeof x === 'number' && percent(x)) && range[0] <= range[1]))
      && ['whole_file', 'visual_content', 'assessed_components'].includes(String(row.scope)) && typeof row.scope_label === 'string' && text(row.scope_note)
      && Array.isArray(row.excluded_channels) && row.excluded_channels.every(c => c === 'voice' || c === 'audio_picture_timing')
      && ['meaning', 'limitation', 'interval_note'].every(key => typeof row[key] === 'string')) return row as unknown as CustomerAssessment
  }
  return customerAssessment(result)
}

export function customerAssessment(result: Input): CustomerAssessment {
  const d = result.decision
  const profile = result.condition_profile
  const answer: CustomerAssessment = {
    rule: ASSESSMENT_RULE, threshold_pct: 80, applies: true, direction: 'unavailable',
    headline: 'No reliable automated conclusion', confidence_pct: null, confidence_label: null,
    probability_ai_pct: null, interval_pct: null, interval_label: null, interval_note: INTERVAL_NOTE,
    meaning: 'The analysis did not establish a usable whole-file estimate.',
    limitation: 'Review the recorded observations and any checks that could not run. Missing evidence does not favour either answer.',
    ...videoScope(result),
  }
  if (d?.provenance_record || result.stored_record || d?.driving_hypothesis === 'H3_provenance_integrity') return {
    ...answer, applies: false, direction: 'record', headline: null,
    meaning: "The conclusion follows the file's provenance record, not a confidence percentage.",
  }
  if ((result.status != null && result.status !== 'done') || !d || d.code === 'not_assessable') return answer
  if (result.kind === 'audio' && d.fuser_sha256 === '912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf') {
    const p = d.probability_ai_calibrated
    const direction = d.code === 'ai_supported' && probability(p) && p >= .8 ? 'ai'
      : d.code === 'no_ai_indicators' && probability(p) && p <= .2 + 1e-12 ? 'original' : 'inconclusive'
    return { ...answer, direction, headline: d.verbal,
      confidence_pct: direction !== 'inconclusive' && probability(p) ? Math.min(99.9, pct(direction === 'ai' ? p : 1 - p)) : null,
      confidence_label: direction !== 'inconclusive' ? 'Reference-calibrated estimate' : null,
      probability_ai_pct: probability(p) ? pct(p) : null, meaning: d.reason,
      limitation: 'Local screening estimate calibrated on a balanced reference set of authentic and AI speech. It is not a measured accuracy rate for this recording. Validation used short English speech; heavy background sound can cause errors or an inconclusive result. No per-file confidence interval is established.',
      scope_label: 'Speech assessment', scope_note: 'Local experimental audio screening; the percentage concerns speech, not every sound or edit in the recording.',
      interval_note: 'No per-file confidence interval has been established for this audio calibration.' }
  }
  if (d.fusion_mode !== 'artefact') return { ...answer, headline: 'Inconclusive: no calibrated estimate',
    meaning: 'The available checks provide inspection scores, but those scores have not been calibrated into a reliable whole-file likelihood.',
    limitation: 'An inspection score of 80 out of 100 does not mean 80% confidence. Use the linked evidence to examine the file further.',
  }
  if (outsideValidated(d, profile) || profile?.within_validated_envelope !== true) return { ...answer,
    headline: 'Inconclusive: this input is outside the verified scope',
    meaning: 'The saved validation does not cover this kind of input well enough to apply the 80% leaning rule.',
    limitation: 'A score can still be recorded for inspection, but it cannot support a confidence percentage for this file.',
  }
  const p = d.probability_ai_calibrated
  if (!probability(p)) return answer
  Object.assign(answer, { direction: 'inconclusive', headline: 'Inconclusive: neither tendency reaches 80%',
    probability_ai_pct: pct(p), meaning: 'Neither the AI-generated nor original-file estimate reaches the 80% threshold.', limitation: ESTIMATE_NOTE })
  const ci = d.ci90
  if (ci?.length === 2 && probability(ci[0]) && probability(ci[1]) && ci[0] <= ci[1]) {
    answer.interval_pct = [pct(ci[0]), pct(ci[1])]
    answer.interval_label = '90% interval for AI likelihood'
  }
  if (ruleTaken(d) || result.caveats?.some(c => c.toLowerCase().includes('strongly disagree'))) return { ...answer,
    headline: 'Inconclusive: the evidence does not support a reliable tendency',
    meaning: 'The recorded checks are conflicting or do not provide enough support for the direction of the combined estimate.',
  }
  const direction = p >= .8 ? 'ai' : p <= .2 ? 'original' : null
  if (!direction) return answer
  const ai = direction === 'ai'
  Object.assign(answer, { direction,
    headline: ai ? 'Leaning towards an AI-generated or AI-altered file' : 'Leaning towards an original file',
    confidence_pct: Math.min(99.9, pct(ai ? p : 1 - p)), confidence_label: 'Automated estimate',
    meaning: ai ? 'The combined automated estimate reaches the 80% threshold for AI generation or AI alteration.'
      : 'The combined automated estimate reaches the 80% threshold in favour of original content among the checks that contributed to this estimate. Local flags and checks outside the estimate require separate consideration.',
  })
  if (!ai && answer.interval_pct && ci) {
    answer.interval_pct = [pct(1 - ci[1]), pct(1 - ci[0])]
    answer.interval_label = '90% interval for original-file likelihood'
  }
  if (answer.scope !== 'whole_file') {
    const subject = answer.scope === 'visual_content' ? 'visual content' : 'content in the assessed components'
    answer.headline = ai ? `Leaning towards AI-generated or AI-altered ${subject}` : `Leaning towards original ${subject}`
    answer.meaning = `Among the checks included in this estimate, the combined estimate reaches the 80% threshold in favour of ${ai ? 'AI generation or alteration of' : 'original'} ${subject}. Local flags and excluded checks require separate consideration.`
    if (answer.interval_pct) answer.interval_label = '90% interval for this scoped estimate'
  }
  return answer
}
