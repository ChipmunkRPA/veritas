import type { CandidateEligibility, ImageInspectionCheck } from './imagePackets'

const response = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
const decimal = (exact: string | undefined, rounded: number): number | null => {
  if (exact === undefined) return response(rounded) ? rounded : null
  if (!/^(?:0|1)(?:\.\d+)?(?:e[+-]?\d+)?$|^\d(?:\.\d+)?e[+-]?\d+$/i.test(exact)) return null
  const value = Number(exact)
  return response(value) ? value : null
}

export function eligibilityView(gate?: CandidateEligibility) {
  if (!gate) return { status: 'not_recorded' as const, title: 'No whole-image gate recorded', score: null, threshold: null,
    explanation: 'This record does not document a whole-image eligibility gate. Missing eligibility information does not establish authenticity.' }
  const score = decimal(gate.score_exact_decimal, gate.score)
  const threshold = decimal(gate.threshold_exact_decimal, gate.threshold)
  if (!['image-candidate-eligibility/1', 'image-candidate-eligibility/2'].includes(gate.schema)
    || gate.status !== 'evaluated' || gate.scope !== 'whole_image'
    || gate.comparison !== 'greater_or_equal' || typeof gate.eligible !== 'boolean' || score === null || threshold === null
    || gate.eligible !== (score >= threshold)) {
    return { status: 'unavailable' as const, title: 'Image eligibility unavailable', score: null, threshold: null,
      explanation: 'An exact eligibility response could not be displayed. This does not establish that the image is unaltered.' }
  }
  if (gate.schema === 'image-candidate-eligibility/2') {
    const { active_cells: active, minimum_active_cells: minimum, input_cells: total } = gate
    if (active === undefined || minimum === undefined || total === undefined
      || ![active, minimum, total].every(Number.isSafeInteger) || active < 0 || active > total
      || minimum < 1 || minimum > total || active / total !== score || minimum / total !== threshold) {
      return { status: 'unavailable' as const, title: 'Image eligibility unavailable', score: null, threshold: null,
        explanation: 'The recorded response cell counts could not be verified.' }
    }
    return { status: gate.eligible ? 'eligible' as const : 'withheld' as const, totalCells: String(total),
      title: gate.eligible ? 'Eligible for region inspection' : 'Regions withheld by the image gate',
      score: String(active), threshold: String(minimum), explanation: gate.eligible
        ? 'The map met the minimum area rule for showing candidates. Normal processing can also create a highlight.'
        : 'The map was below the minimum area rule. Small or weak edits may be missed.' }
  }
  return { status: gate.eligible ? 'eligible' as const : 'withheld' as const,
    title: gate.eligible ? 'Eligible for region inspection' : 'Regions withheld by the image gate',
    score: gate.score_exact_decimal ?? String(score), threshold: gate.threshold_exact_decimal ?? String(threshold),
    explanation: gate.eligible
      ? 'The whole-image response meets the display threshold. Region masks still require inspection and can contain false alarms or miss edits.'
      : 'The whole-image response is below the display threshold, so this check contributes no candidate regions. This does not establish that the image is unaltered.' }
}

export function inspectionCounts(check: ImageInspectionCheck) {
  const { raw_components: raw, eligible_components: eligible, selected_components: selected, omitted_components: omitted } = check
  if (![raw, eligible, selected, omitted].every((v) => Number.isSafeInteger(v) && v >= 0)
    || eligible > raw || selected > eligible || omitted !== eligible - selected) return null
  return { raw, eligible, selected, omitted, suppressed: raw - eligible }
}

export function inspectionCheckLabel(check: ImageInspectionCheck): string {
  const labels = check.signal_refs.map((ref) => ref.display_id).filter((label): label is string => !!label && /^S\d+$/.test(label))
  return labels.length ? `Check ${[...new Set(labels)].join(', ')}` : 'Recorded image check'
}
