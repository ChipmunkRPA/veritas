// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import SourceLabels from './SourceLabels'

afterEach(cleanup)
const source = 'a'.repeat(64)
const data = { version: 'source-labels/1', source_sha256: source, status: 'checked', notice: 'Unverified metadata; not used in confidence.',
  checked_metadata: ['PNG AIGC text'], issues: [], records: [{ id: 'SL01', declaration: 'Declares AI generation or synthesis',
    location: 'PNG AIGC text', byte_offset: 33, payload_sha256: 'b'.repeat(64), claimed_provider: 'Google / Gemini' }] }

describe('source labels', () => {
  it('links the source and label while keeping provider identity and confidence unverified', () => {
    render(<SourceLabels value={data} source={source} />)
    expect(screen.getByText(/Google \/ Gemini/).textContent).toContain('not authenticated')
    expect(screen.getByText(/No confidence percentage/)).toBeTruthy()
    fireEvent.click(screen.getByText('Source reference'))
    expect(screen.getByText(/Bound source SHA-256/).textContent).toContain(source)
    expect(screen.queryByText('100%')).toBeNull()
  })
  it('does not turn absent or source-mismatched evidence into originality', () => {
    const { rerender } = render(<SourceLabels source={source} />)
    expect(screen.getByText(/not recorded/).textContent).toContain('Missing labels do not establish originality')
    rerender(<SourceLabels source={'c'.repeat(64)} value={data} />)
    expect(screen.queryByText(/Google \/ Gemini/)).toBeNull()
    expect(screen.getByText(/not recorded/)).toBeTruthy()
  })
  it('preserves incomplete coverage and a neutral no-label outcome', () => {
    const { rerender } = render(<SourceLabels source={source} value={{ ...data, records: [] }} />)
    expect(screen.getByText(/No recognized label/).textContent).toContain('does not establish originality')
    rerender(<SourceLabels source={source} value={{ ...data, records: [], status: 'partial' }} />)
    expect(screen.getByText(/No conclusion follows/)).toBeTruthy()
    expect(screen.getByText(/Coverage is incomplete/)).toBeTruthy()
  })
})
