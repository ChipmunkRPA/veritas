// @vitest-environment jsdom
import { readFileSync } from 'node:fs'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import type { Finding, SignalV2 } from '../../lib/api/queries'
import FindingSupportPanel from './FindingSupportPanel'

const cases = JSON.parse(readFileSync('../backend/tests/fixtures/finding_assessment_cases.json', 'utf8'))
afterEach(cleanup)

function show(name: string) {
  const row = cases.find((r: { name: string }) => r.name === name)
  const finding = { title: 'Recorded check', statement: 'Saved technical statement', rank: 1, review_priority: 'low',
    reference_range: '', ...row.finding } as Finding
  const signals = row.result.signals.map((s: SignalV2) => ({ ...s, display_id: s.display_id ?? 'S01',
    name: s.name ?? 'Recorded check', observed: s.observed ?? 'A measured pattern was recorded in the source.' }))
  render(<MemoryRouter><FindingSupportPanel finding={finding} data={{ analysisId: 'analysis-one', ...row.result, signals }} /></MemoryRouter>)
}

describe('readable evidence-led findings', () => {
  it('shows the observation, calibrated finding estimate, scope and supporting check', () => {
    show('matched whole-input calibration')
    expect(screen.getByText('A measured pattern was recorded in the source.')).toBeTruthy()
    expect(screen.getByTestId('finding-interpretation').textContent).toContain('Leans AI')
    expect(screen.getByTestId('finding-interpretation').textContent).toContain('86%')
    expect(screen.getByTestId('finding-interpretation').textContent).toContain('Whole-image generation indicators')
    expect(screen.getByTestId('finding-check-support').getAttribute('href')).toContain('/analyses/analysis-one/ledger')
    expect(screen.queryByText(/screening threshold/)).toBeNull()
  })
  it('does not give a selected crop the global confidence percentage', () => {
    show('a crop cannot inherit a whole-image confidence')
    const text = screen.getByTestId('finding-interpretation').textContent
    expect(text).toContain('Not established for this finding')
    expect(text).toContain('Selected image region')
    expect(text).not.toContain('86%')
  })
  it('makes a separately assessed finding visible without adding it to the overall estimate', () => {
    show('checked independently from overall')
    expect(screen.getByTestId('finding-interpretation').textContent).toContain('Assessed separately')
    expect(screen.getByTestId('finding-interpretation').textContent).toContain('86%')
  })
})
