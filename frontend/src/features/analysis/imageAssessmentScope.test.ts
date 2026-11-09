import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { customerAssessment } from './customerAssessment'

const fixture = JSON.parse(readFileSync(new URL('../../../../backend/tests/fixtures/image_scope_cases_v50.json', import.meta.url), 'utf8'))

describe('image scope shared with backend v50', () => {
  for (const c of fixture.cases) it(c.name, () => {
    const value = structuredClone(fixture.base)
    const local = value.signals[1]
    local.metrics.regions = c.regions
    local.status = c.local_status
    local.details.localisation.candidate_eligibility.eligible = c.eligible
    if (c.ai) Object.assign(value.decision, { code: 'ai_supported', probability_ai_calibrated: .97, ci90: [.92, .99] })
    if (c.unknown_scope) Object.assign(value.signals[0], { family: 'unknown', hypothesis: null })
    if (c.whole_status) value.signals[0].status = c.whole_status
    if (c.code) value.decision.code = c.code
    const before = JSON.stringify(value)
    const got = customerAssessment(value)
    expect(got).toMatchObject(c.expected)
    expect(JSON.stringify(value)).toBe(before)
    if (c.component_confidence) {
      expect(got.component_estimate?.confidence_pct).toBe(c.component_confidence)
      expect(got.component_estimate?.interval_pct).toEqual([92, 99])
      expect(got.scope_note).toContain('No whole-file confidence')
    }
  })
})
