import type { CustomerAssessment } from './customerAssessment'

type ScopeInput = {
  kind?: string
  signals?: readonly { id: string; detector_id?: string | null; family?: string | null; hypothesis?: string | null
    status?: string | null; outside_estimate?: unknown; effective_weight?: number | null
    legacy?: { v1_id?: string | null } | null; metrics?: Record<string, unknown> | null; details?: Record<string, unknown> | null }[]
  decision?: { contributions?: readonly { kind: string; signal_id?: string | null; counted?: boolean | null; logit?: number | null }[] } | null
}

const counted = (row: { counted?: boolean | null; logit?: number | null }) => row.counted === true
  || (row.counted !== false && typeof row.logit === 'number' && Number.isFinite(row.logit) && Math.abs(row.logit) > 1e-9)

/** Read-only mirror of image_assessment_scope.py; never converts a region response to AI confidence. */
export function imageAssessmentScope(result: ScopeInput, answer: CustomerAssessment): CustomerAssessment {
  if (result.kind !== 'image' || !answer.applies || !['ai', 'original'].includes(answer.direction)) return answer
  const rows = result.decision?.contributions?.filter(r => r.kind === 'detector') ?? []
  const included = new Set<string>()
  let localCues = false
  for (const signal of result.signals ?? []) {
    if ((signal.status ?? 'ok') !== 'ok') continue
    const refs = new Set([signal.id, signal.legacy?.v1_id, signal.detector_id ? `detector:${signal.detector_id}` : null].filter(Boolean))
    const family = signal.family ?? ''
    const rule = signal.details?.localisation as { candidate_eligibility?: { eligible?: boolean } } | undefined
    const regions = signal.metrics?.regions
    if (family === 'localiser' && typeof regions === 'number' && Number.isFinite(regions) && regions > 0
      && (!rule?.candidate_eligibility || rule.candidate_eligibility.eligible === true)) localCues = true
    const matched = rows.filter(r => r.signal_id != null && refs.has(r.signal_id))
    const omitted = !!signal.outside_estimate || signal.effective_weight === 0 || matched.some(r => r.counted === false)
    if (omitted || !matched.some(counted)) continue
    if (signal.hypothesis === 'H1_whole_content_synthesis' || family === 'learned-whole-image') included.add('generation')
    else if (family.includes('face')) included.add('face')
    else if (signal.hypothesis === 'H2_localised_manipulation' && family === 'localiser') included.add('editing')
  }
  if (!included.size) return { ...answer, direction: 'inconclusive', headline: 'Inconclusive: image-check coverage is not established',
    confidence_pct: null, confidence_label: null, probability_ai_pct: null, interval_pct: null, interval_label: null,
    meaning: 'The saved result does not identify usable contributing image checks. It cannot establish original or AI authorship.' }
  const subject = ([['generation', 'whole-image generation'], ['face', 'face-pattern'], ['editing', 'local-editing']] as const)
    .filter(([key]) => included.has(key)).map(([, label]) => label).join(' and ')
  const label = subject[0].toUpperCase() + subject.slice(1) + ' checks'
  const scoped: CustomerAssessment = { ...answer, scope: 'image_components', scope_label: label,
    scope_note: `The percentage covers ${subject} checks. It does not verify every pixel or rule out localized AI edits. Regional editing cues are assessed separately and do not enter this percentage.`,
    confidence_label: 'Scoped image-check estimate', interval_label: answer.interval_pct ? '90% interval for these image checks' : null,
    headline: answer.direction === 'ai' ? `${label} lean towards AI generation or alteration` : `${label} lean original; local edits remain unresolved`,
    meaning: 'The calibrated estimate applies to the listed image checks, not a finding that every part of the file is original or AI-generated.' }
  if (answer.direction === 'original' && localCues) {
    const component = { direction: answer.direction, headline: scoped.headline!, confidence_pct: scoped.confidence_pct!,
      confidence_label: scoped.confidence_label!, probability_ai_pct: scoped.probability_ai_pct,
      interval_pct: scoped.interval_pct, interval_label: scoped.interval_label, scope_label: scoped.scope_label }
    const uncertainty = answer.interval_pct ? ` (90% reference-fit interval ${answer.interval_pct[0]}-${answer.interval_pct[1]}%)` : ''
    return { ...scoped, direction: 'inconclusive', headline: 'Local editing cues found; AI origin is inconclusive',
      confidence_pct: null, confidence_label: null, probability_ai_pct: null, interval_pct: null, interval_label: null,
      scope_note: `The separate ${subject} estimate leans original at ${answer.confidence_pct}%${uncertainty}. That percentage does not cover the highlighted editing cues. Ordinary edits and AI edits can produce such cues; their AI origin remains inconclusive. No whole-file confidence percentage is assigned.`,
      component_estimate: component,
      meaning: 'A local check highlighted editing cues that the whole-image estimate cannot resolve. Inspect the source crops, masks and limitations; neither AI editing nor an entirely original file is established.' }
  }
  return scoped
}
