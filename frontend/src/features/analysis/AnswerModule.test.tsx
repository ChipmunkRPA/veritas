// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { afterEach, describe, expect, it } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import type { AnalysisDetail, Finding } from '../../lib/api/queries'
import AnswerModule from './AnswerModule'
import { invalidSentence } from './answer'
import { customerAssessment } from './customerAssessment'
import { c2paState, c2paWords } from './provenance'

afterEach(cleanup)
const analysis = (probability: number, extra: Record<string, unknown> = {}) => ({
  id: 'fixture', kind: 'image', status: 'done', coverage: { detectors_total: 2, by_status: { ok: 2 } },
  condition_profile: { within_validated_envelope: true }, caveats: [], calibration: [], not_validated: [],
  decision: { code: 'inconclusive', confidence: 'low', fusion_mode: 'artefact', probability_ai_calibrated: probability,
    reason: 'Saved calibrated estimate.', ci90: [0.7, 0.9], ...extra },
}) as unknown as AnalysisDetail

describe('automated customer conclusion', () => {
  it('explains invalid audio credentials without inferring changed content', () => {
    const finding = { id: 'F1', title: 'Content Credentials that do not validate', family: 'F1_provenance', strength: 'strong', direction: 'for', rank: 1,
      evidence_ids: ['prov.c2pa.audio:global'] } as unknown as Finding
    expect(invalidSentence([finding])).toContain('certificate')
    expect(invalidSentence([finding])).not.toContain('changed after signing')
    expect(c2paWords('invalid', true).line).toContain('alone do not establish tampering')
    expect(c2paState({ assessments: [] }, new Map([['state', 'Invalid']]))).toBe('invalid')
  })
  it('shows a supported AI tendency with a supported reference estimate', () => {
    render(<AnswerModule analysis={analysis(.6, { code: 'ai_supported' })} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toBe('Leaning towards AI-generated or AI-altered content')
    expect(screen.getByTestId('customer-assessment').textContent).toContain('Reference-calibrated overall estimate: 60%')
    expect(screen.getByTestId('means').textContent).toContain('reference-calibrated checks lean')
    expect(screen.getByTestId('means').textContent).not.toContain('does not support a conclusive assessment')
    expect(screen.getByTestId('does-not-mean').textContent).toContain('not an independently established accuracy')
    expect(screen.getByTestId('fact-calibrated').textContent).toContain('Comparable error rates')
  })
  it('shows confidence and interval for the original direction, not the complementary AI score', () => {
    render(<AnswerModule analysis={analysis(.126264, { code: 'no_ai_indicators', ci90: [.059251, .289501] })} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toBe('Leaning towards original content in the assessed scope')
    expect(screen.getByTestId('customer-assessment').textContent).toContain('87.3%')
    expect(screen.getByTestId('fact-plausible').textContent).toContain('71–94%')
    expect(screen.getByTestId('fact-plausible').textContent).toContain('original-file likelihood')
  })
  it('does not promote an interval endpoint or raw fallback score to confidence', () => {
    const { rerender } = render(<AnswerModule analysis={analysis(.755266)} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toContain('Inconclusive')
    expect(screen.getByTestId('customer-assessment').textContent).toContain('75.5%')
    expect(screen.getByTestId('customer-assessment').textContent).not.toContain('Reference-calibrated overall estimate:')
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
    expect(screen.getByTestId('assessment-scope').textContent).toContain('Voice/audio was checked separately from the main assessment.')
    expect(screen.getByTestId('assessment-scope').textContent).toContain('Audio/picture timing was checked separately from the main assessment.')
    expect(screen.getByTestId('customer-assessment').textContent).toContain('In favour of original visual content.')
    rerender(<AnswerModule analysis={{ ...dto, customer_assessment: undefined }} findings={[]} signals={input.signals} />)
    expect(screen.getByTestId('headline').textContent).toBe('Leaning towards original visual content')
  })
  it('keeps an inconclusive low AI estimate from becoming an original verdict on screen', () => {
    render(<AnswerModule analysis={analysis(.01)} findings={[]} />)
    expect(screen.getByTestId('headline').textContent).toContain('Inconclusive')
    expect(screen.getByTestId('headline').textContent).not.toContain('original')
    expect(screen.getByTestId('customer-assessment').textContent).not.toContain('Reference-calibrated overall estimate:')
  })
  it('describes skipped audio beside a supported visual assessment', () => {
    const cases = JSON.parse(readFileSync('../backend/tests/fixtures/customer_assessment_cases.json', 'utf8'))
    const input = cases.find((c: { name: string }) => c.name === 'skipped audio checks retain visual-only scope').input
    render(<AnswerModule analysis={{ ...analysis(.126264), ...input }} findings={[]} signals={input.signals} />)
    expect(screen.getByTestId('headline').textContent).toBe('Leaning towards original visual content')
    expect(screen.getByTestId('assessment-scope').textContent).toContain('voice/audio checks were skipped')
    expect(screen.getByTestId('assessment-scope').textContent).not.toContain('checked separately')
  })
  it('presents a separate speech estimate and links to the matching video evidence lane', () => {
    const input = { ...analysis(.126264), kind: 'video' as const }
    const soundtrack = { direction: 'ai' as const, headline: 'Leaning towards AI-generated speech', confidence_pct: 95.6,
      limitation: 'Separate from the visual and audio-picture timing results.' }
    const dto = { ...input, customer_assessment: { ...customerAssessment(input), soundtrack } } as AnalysisDetail
    const { rerender } = render(<MemoryRouter><AnswerModule analysis={dto} findings={[]} /></MemoryRouter>)
    expect(screen.getByRole('region', { name: 'Speech in this soundtrack' }).textContent).toContain('95.6%')
    expect(screen.getByRole('link', { name: 'Inspect voice evidence' }).getAttribute('href'))
      .toBe('/analyses/fixture/media-evidence?view=plain#media-evidence-voice')
    dto.customer_assessment = { ...dto.customer_assessment, soundtrack: { ...soundtrack, direction: 'inconclusive', confidence_pct: null } }
    rerender(<MemoryRouter><AnswerModule analysis={dto} findings={[]} /></MemoryRouter>)
    expect(screen.getByRole('region', { name: 'Speech in this soundtrack' }).textContent).not.toContain('%')
  })
})
