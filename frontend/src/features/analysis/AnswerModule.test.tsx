// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { afterEach, describe, expect, it } from 'vitest'
import type { AnalysisDetail } from '../../lib/api/queries'
import AnswerModule from './AnswerModule'
import { customerAssessment } from './customerAssessment'

afterEach(cleanup)
const analysis = (probability: number, extra: Record<string, unknown> = {}) => ({
  id: 'fixture', kind: 'image', status: 'done', coverage: { detectors_total: 2, by_status: { ok: 2 } },
  condition_profile: { within_validated_envelope: true }, caveats: [], calibration: [], not_validated: [],
  decision: { code: 'inconclusive', confidence: 'low', fusion_mode: 'artefact', probability_ai_calibrated: probability,
    reason: 'Saved calibrated estimate.', ci90: [0.7, 0.9], ...extra },
}) as unknown as AnalysisDetail

describe('automated customer conclusion', () => {
  it('shows the AI tendency at 80% without the contradictory old inconclusive prose', () => {
    render(<AnswerModule analysis={analysis(.8)} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toBe('Leaning towards an AI-generated or AI-altered file')
    expect(screen.getByTestId('customer-assessment').textContent).toContain('Automated estimate: 80%')
    expect(screen.getByTestId('means').textContent).toContain('reaches the 80% threshold')
    expect(screen.getByTestId('means').textContent).not.toContain('does not support a conclusive assessment')
    expect(screen.getByTestId('does-not-mean').textContent).toContain('not an independently established accuracy')
    expect(screen.getByTestId('fact-calibrated').textContent).toContain('falsely flagged')
  })
  it('shows confidence and interval for the original direction, not the complementary AI score', () => {
    render(<AnswerModule analysis={analysis(.126264, { code: 'no_ai_indicators', ci90: [.059251, .289501] })} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toBe('Leaning towards an original file')
    expect(screen.getByTestId('customer-assessment').textContent).toContain('87.3%')
    expect(screen.getByTestId('fact-plausible').textContent).toContain('71–94%')
    expect(screen.getByTestId('fact-plausible').textContent).toContain('original-file likelihood')
  })
  it('does not promote an interval endpoint or raw fallback score to confidence', () => {
    const { rerender } = render(<AnswerModule analysis={analysis(.755266)} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toContain('neither tendency reaches 80%')
    expect(screen.getByTestId('customer-assessment').textContent).toContain('75.5%')
    expect(screen.getByTestId('customer-assessment').textContent).not.toContain('Automated estimate:')
    rerender(<AnswerModule analysis={analysis(.99, { fusion_mode: 'fallback', probability_ai_calibrated: null, probability_ai_uncalibrated: .99 })} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toBe('Inconclusive: no calibrated estimate')
    expect(screen.getByTestId('customer-assessment').textContent).not.toContain('99%')
  })
  it('puts the excluded audio scope beside a visual tendency before any check data is fetched', () => {
    const cases = JSON.parse(readFileSync('../backend/tests/fixtures/customer_assessment_cases.json', 'utf8'))
    const input = cases.find((c: { name: string }) => c.name === 'real video excludes checked voice and timing from visual percentage').input
    const dto = { ...analysis(.126264), ...input, customer_assessment: customerAssessment(input) } as AnalysisDetail
    const { rerender } = render(<AnswerModule analysis={dto} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toBe('Leaning towards original visual content')
    expect(screen.getByTestId('assessment-scope').textContent).toContain('voice/audio evidence and audio/picture timing evidence were checked separately')
    expect(screen.getByTestId('assessment-scope').textContent).toContain('The audio has not been cleared as original.')
    expect(screen.getByTestId('customer-assessment').textContent).toContain('In favour of original visual content.')
    rerender(<AnswerModule analysis={{ ...dto, customer_assessment: undefined }} findings={[]} signals={input.signals} />)
    expect(screen.getByTestId('headline').textContent).toBe('Leaning towards original visual content')
  })
})
