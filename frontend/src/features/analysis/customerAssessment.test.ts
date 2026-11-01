import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { customerAssessment, presentedCustomerAssessment } from './customerAssessment'

const cases = JSON.parse(readFileSync(new URL('../../../../backend/tests/fixtures/customer_assessment_cases.json', import.meta.url), 'utf8')) as {
  name: string; input: Parameters<typeof customerAssessment>[0]; expected: Record<string, unknown>
}[]

describe('customer 80% tendency, shared report/API cases', () => {
  for (const c of cases) it(c.name, () => {
    const before = JSON.stringify(c.input)
    const got = customerAssessment(c.input)
    expect(got).toMatchObject(c.expected)
    expect(JSON.stringify(c.input)).toBe(before)
    if (got.confidence_pct !== null) {
      expect(got.limitation).toContain('not an independently established accuracy')
      expect(got.confidence_label).toBe('Automated estimate')
    }
    expect(got.interval_note).toContain('90% accurate')
    if (got.scope === 'visual_content') {
      expect(got.scope_note).toContain('checked separately')
      expect(got.scope_note).toContain('audio has not been cleared as original')
    }
  })
  it('accepts only a complete, matching API presentation', () => {
    const input = cases[0].input
    const expected = customerAssessment(input)
    expect(presentedCustomerAssessment({ ...input, customer_assessment: expected })).toBe(expected)
    expect(presentedCustomerAssessment({ ...input, customer_assessment: { ...expected, rule: 'future-rule' } })).toEqual(expected)
    expect(presentedCustomerAssessment({ ...input, customer_assessment: { ...expected, confidence_pct: Infinity } })).toEqual(expected)
    expect(presentedCustomerAssessment({ ...input, customer_assessment: { ...expected, interval_pct: [90, 10] } })).toEqual(expected)
  })
  it('an API-scoped result is used even before the browser has fetched the check records', () => {
    const input = cases.find(c => c.name === 'real video excludes checked voice and timing from visual percentage')!.input
    const fromApi = customerAssessment(input)
    expect(presentedCustomerAssessment({ ...input, signals: undefined, customer_assessment: fromApi })).toBe(fromApi)
    expect(presentedCustomerAssessment({ ...input, signals: undefined, customer_assessment: {} }).scope).toBe('assessed_components')
  })
})

describe('local speech calibration', () => {
  for (const [code, p, direction, confidence] of [
    ['ai_supported', .92, 'ai', 92], ['no_ai_indicators', .08, 'original', 92],
    ['inconclusive', .92, 'inconclusive', null], ['inconclusive', null, 'inconclusive', null],
  ] as const) it(`${code} at ${p} respects the saved uncertainty decision`, () => {
    const input = { kind: 'audio', status: 'done',
      decision: { code, fuser_sha256: '912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf',
        probability_ai_calibrated: p, verbal: 'Speech assessment', reason: 'Uncertainty check' },
      condition_profile: { within_validated_envelope: false },
    } as Parameters<typeof customerAssessment>[0]
    const got = customerAssessment(input)
    expect(got.direction).toBe(direction)
    expect(got.confidence_pct).toBe(confidence)
    expect(got.interval_pct).toBeNull()
    expect(got.limitation).toContain('short English speech')
    expect(presentedCustomerAssessment({ ...input, customer_assessment: got })).toBe(got)
  })
})
