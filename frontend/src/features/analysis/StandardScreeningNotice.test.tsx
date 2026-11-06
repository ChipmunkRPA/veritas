// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import StandardScreeningNotice from './StandardScreeningNotice'
import ExpertAnalysisInvitation from './ExpertAnalysisInvitation'

afterEach(cleanup)

describe('standard SaaS screening scope', () => {
  it('states the cutoff without claiming measured accuracy and requires a private engagement', () => {
    render(<StandardScreeningNotice />)
    const notice = screen.getByTestId('standard-screening-notice')
    expect(notice.textContent).not.toContain('screening threshold')
    expect(notice.textContent).toContain('Findings link observations')
    expect(notice.textContent).toContain('not verified accuracy or proof')
    expect(notice.textContent).toContain('limited scientific reliability')
    expect(notice.textContent).toContain('Do not rely on it as court evidence')
    expect(notice.textContent).toContain('separate signed engagement')
    expect(notice.textContent).toContain('does not guarantee court admissibility')
    expect(screen.getByRole('link').getAttribute('href')).toBe('mailto:sales@chipmunkrpa.com')
  })

  it('offers the private report through the sales contact with an analysis reference', () => {
    render(<ExpertAnalysisInvitation analysisId="case-reference-1234" />)
    const invitation = screen.getByTestId('expert-analysis-invitation')
    expect(invitation.textContent).toContain('privately researched expert report')
    expect(invitation.textContent).toContain('separate signed engagement')
    expect(invitation.textContent).toContain('not prepared for court filing')
    const href = screen.getByRole('link', { name: /Discuss expert analysis/ }).getAttribute('href')!
    expect(href).toContain('mailto:sales@chipmunkrpa.com')
    expect(decodeURIComponent(href)).toContain('case-reference-1234')
  })
})
