import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { findingAssessment, findingObservations, type FindingContext, type FindingInput } from './findingAssessment'

const cases = JSON.parse(readFileSync(new URL('../../../../backend/tests/fixtures/finding_assessment_cases.json', import.meta.url), 'utf8')) as {
  name: string; finding: FindingInput; result: FindingContext; expected: Record<string, unknown>
}[]

it('keeps coverage facts separate from a linked voice response', () => {
  const finding = { id: 'F3', family: 'coverage', evidence_ids: ['voice'],
    sentences: [{ text: '2 of 2 checks produced a result.', evidence_ids: ['voice'] }] }
  expect(findingObservations(finding, { signals: [{ id: 'voice', status: 'ok', observed: 'mean p(synthetic)=0.999' }] }))
    .toEqual(['2 of 2 checks produced a result.'])
})

describe('finding confidence shared with reports and JSON', () => {
  for (const row of cases) it(row.name, () => {
    const before = JSON.stringify(row)
    expect(findingAssessment(row.finding, row.result)).toEqual(row.expected)
    expect(JSON.stringify(row)).toBe(before)
  })
  it('keeps scope and calibration gates independent of high raw scores', () => {
    const original = cases[0]
    expect(findingAssessment(original.finding, original.result).confidence_pct).toBe(86)
    expect(findingAssessment({ ...original.finding, scope: { start_s: 2, end_s: 4 } }, original.result).confidence_pct).toBeNull()
    expect(findingAssessment(original.finding, { ...original.result, calibration: [] }).confidence_pct).toBeNull()
    expect(findingAssessment({ ...original.finding, family: 'V2_lip_sync' }, original.result).direction).toBe('context')
  })
})
