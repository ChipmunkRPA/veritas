import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { COMBINED_SPEECH_REFERENCE, combinedSpeechTendency } from './speechReference'
import { customerAssessment, presentedCustomerAssessment } from './customerAssessment'
import { findingAssessment } from './findingAssessment'

export const cases = JSON.parse(readFileSync(new URL('../../../../backend/tests/fixtures/combined_speech_reference_cases.json', import.meta.url), 'utf8')) as {
  cases: { name: string; screen: Record<string, unknown>; expected: { probability_ai: number | null; direction: string; confidence_pct: number | null } }[]
}

describe('saved speech-reference contract shared with the backend', () => {
  for (const row of cases.cases) it(row.name, () => {
    const before = JSON.stringify(row.screen)
    expect(combinedSpeechTendency(row.screen)).toEqual([row.expected.probability_ai, row.expected.direction])
    expect(JSON.stringify(row.screen)).toBe(before)
  })
})

function example(screen: Record<string, unknown>) {
  const finding = { id: 'F1', family: 'A1_voice', direction: screen.decision === 'original' ? 'against' : 'for',
    strength: 'weak', scope: null, evidence_ids: ['speech:global'] } as Parameters<typeof findingAssessment>[0]
  const result = { kind: 'audio', status: 'done', findings: [finding], signals: [{ id: 'speech:global',
    family: 'voice', category: 'model', modality: 'audio', scope: 'global', status: 'ok', calibration_ref: COMBINED_SPEECH_REFERENCE,
    calibrated_probability: screen.probability_ai, effective_weight: 1, details: { audio_screening: screen } }],
    decision: { fuser_sha256: COMBINED_SPEECH_REFERENCE, probability_ai_calibrated: screen.probability_ai, fusion_mode: 'artefact',
      code: screen.decision === 'ai' ? 'ai_supported' : screen.decision === 'original' ? 'no_ai_indicators' : 'inconclusive' },
  } as unknown as Parameters<typeof customerAssessment>[0]
  return { result, finding }
}

describe('whole and finding presentation use the same saved speech scope', () => {
  for (const row of cases.cases) it(row.name, () => {
    const { result, finding } = example(row.screen)
    const answer = customerAssessment(result)
    expect(answer.direction).toBe(row.expected.direction)
    expect(answer.confidence_pct).toBe(row.expected.confidence_pct)
    const fact = findingAssessment(finding, result)
    expect(fact.direction).toBe(row.expected.direction)
    expect(fact.confidence_pct).toBe(row.expected.confidence_pct)
    const video = { ...result, kind: 'video', decision: undefined }
    const shown = customerAssessment(video)
    expect(shown.confidence_pct).toBeNull()
    expect(shown.soundtrack?.direction).toBe(row.expected.direction)
    expect(shown.soundtrack?.confidence_pct).toBe(row.expected.confidence_pct)
    expect(presentedCustomerAssessment({ ...video, customer_assessment: shown })).toBe(shown)
    expect(presentedCustomerAssessment({ ...result, customer_assessment: answer })).toBe(answer)
  })

  it('does not accept server confidence when the combined source support is missing', () => {
    const { result } = example({ ...cases.cases[0].screen, scope_supported: false })
    const counterfeit = { ...customerAssessment(result), direction: 'original', confidence_pct: 95, probability_ai_pct: 5 }
    const shown = presentedCustomerAssessment({ ...result, customer_assessment: counterfeit })
    expect(shown.direction).toBe('inconclusive')
    expect(shown.confidence_pct).toBeNull()
  })

  it('does not turn a checked window into whole-file confidence', () => {
    const { result } = example(cases.cases[0].screen)
    result.signals![0].scope = 'window'
    const shown = customerAssessment(result)
    expect(shown.direction).toBe('inconclusive')
    expect(shown.confidence_pct).toBeNull()
  })
})
