import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { ASSESSMENT_RULE, customerAssessment, presentedCustomerAssessment } from './customerAssessment'

const cases = JSON.parse(readFileSync(new URL('../../../../backend/tests/fixtures/customer_assessment_cases.json', import.meta.url), 'utf8')) as {
  name: string; input: Parameters<typeof customerAssessment>[0]; expected: Record<string, unknown>
}[]

describe('evidence-led customer interpretation, shared report/API cases', () => {
  for (const c of cases) it(c.name, () => {
    const before = JSON.stringify(c.input)
    const got = customerAssessment(c.input)
    expect(got).toMatchObject({ ...c.expected, rule: ASSESSMENT_RULE })
    expect(JSON.stringify(c.input)).toBe(before)
    if (got.confidence_pct !== null) {
      expect(got.limitation).toContain('not an independently established accuracy')
      expect(got.confidence_label).toBe('Reference-calibrated overall estimate')
    }
    expect(got.interval_note).toContain('90% accurate')
    if (got.scope === 'visual_content') {
      expect(got.scope_note).toContain('uses sampled visual content')
      expect(got.excluded_channels).toEqual(['voice', 'audio_picture_timing'])
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
  it('withholds combined confidence when one of several stored images was assessed', () => {
    const got = customerAssessment({ ...cases[0].input, kind: 'image', manifest: { sampling: { image_frame_count: 3 } } })
    expect(got).toMatchObject({ direction: 'inconclusive', scope: 'image_components', scope_label: 'First stored image',
      confidence_pct: null, probability_ai_pct: null, interval_pct: null })
  })
  it('an API-scoped result is used even before the browser has fetched the check records', () => {
    const input = cases.find(c => c.name === 'real video excludes checked voice and timing from visual percentage')!.input
    const fromApi = customerAssessment(input)
    expect(presentedCustomerAssessment({ ...input, signals: undefined, customer_assessment: fromApi })).toBe(fromApi)
    expect(presentedCustomerAssessment({ ...input, signals: undefined, customer_assessment: {} }).scope).toBe('assessed_components')
  })
  it('states when one measured timing window cannot assess a change', () => {
    const input = cases.find(c => c.name === 'real video excludes checked voice and timing from visual percentage')!.input
    const got = customerAssessment({ ...input, signals: input.signals?.map(s => s.family === 'av-sync'
      ? { ...s, metrics: { windows: 1, longest_measured_run: 1, steps: 0, collapses: 0 } } : s) })
    expect(got.scope_note).toContain('too few measured speech windows')
    expect(got.scope_note).not.toContain('Audio/picture timing was checked')
    expect(got.confidence_pct).toBe(customerAssessment(input).confidence_pct)
  })
  it('recomputes a cached v1 original headline when the recorded decision is inconclusive', () => {
    const input = { ...cases[2].input, decision: { ...cases[2].input.decision!, code: 'inconclusive' as const } }
    const previous = { ...customerAssessment(cases[2].input), rule: '80-percent-leaning/v1' }
    const got = presentedCustomerAssessment({ ...input, customer_assessment: previous })
    expect(got.direction).toBe('inconclusive')
    expect(got.confidence_pct).toBeNull()
    expect(got.headline).toContain('Inconclusive')
  })
})

describe('local speech calibration', () => {
  for (const [code, p, direction, confidence] of [
    ['ai_supported', .92, 'ai', 92], ['no_ai_indicators', .08, 'original', 92],
    ['inconclusive', .92, 'inconclusive', null], ['inconclusive', .08, 'inconclusive', null], ['inconclusive', null, 'inconclusive', null],
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
    expect(got.scope_note).toContain('cannot identify which speaker drove the result')
    expect(presentedCustomerAssessment({ ...input, customer_assessment: got })).toBe(got)
  })
})

describe('email headline fallback', () => {
  for (const [code, p, direction] of [
    ['ai_supported', .92, 'ai'], ['inconclusive', .08, 'inconclusive'], ['no_ai_indicators', .08, 'inconclusive'],
  ] as const) it(`keeps the saved email policy for ${code}`, () => {
    const input = { kind: 'document', status: 'done', condition_profile: {
      slice_id: 'document/local-email-v1', within_validated_envelope: true },
    decision: { code, probability_ai_calibrated: p, fusion_mode: 'artefact', ci90: [.05, .95],
      verbal: 'Saved email headline', reason: 'Saved email result' },
    } as Parameters<typeof customerAssessment>[0]
    const result = customerAssessment(input)
    expect(result.direction).toBe(direction)
    expect(result.scope_label).toBe('Selected email body')
    if (direction === 'inconclusive') {
      expect(result.headline).toBe('Inconclusive: no reliable AI-writing tendency')
      expect(result.confidence_pct).toBeNull()
    }
  })
})

describe('article presentation scope', () => {
  for (const [code, p, direction] of [
    ['ai_supported', .92, 'ai'], ['no_ai_indicators', .08, 'original'], ['inconclusive', .92, 'inconclusive'],
  ] as const) it(`preserves ${direction} without reusing old threshold copy`, () => {
    const input = { kind: 'document', status: 'done', condition_profile: {
      slice_id: 'document/local-article-v1', within_validated_envelope: true },
    decision: { code, probability_ai_calibrated: p, fusion_mode: 'artefact', ci90: [.05, .95],
      verbal: '60% screening threshold', reason: 'Saved threshold copy' },
    } as Parameters<typeof customerAssessment>[0]
    const result = customerAssessment(input)
    expect(result.direction).toBe(direction)
    expect(result.scope_label).toBe('Extracted writing')
    expect(JSON.stringify(result)).not.toContain('60% screening threshold')
    if (direction === 'inconclusive') expect(result.confidence_pct).toBeNull()
    else expect(result.meaning).toContain('not the authorship of every highlighted passage')
  })
})
