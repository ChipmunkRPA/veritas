// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Analysis, SentenceOut, Verdict } from '../../lib/api'
import VerdictModule from '../legacy/VerdictModule'
import ManuscriptView from './ManuscriptView'
import { sentenceScoreRows } from './sentenceScores'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

const sentence = (text: string, overrides: Partial<SentenceOut> = {}): SentenceOut => ({ idx: 0, start: 0,
  end: Array.from(text).length, text, paragraph: 0, p_ai: 0.7, band: 'ai', reasons: [], scores: {}, ...overrides })

describe('public document measurements and source anchors', () => {
  it('retains scores, reason thresholds and source quotes without printing implementation keys', () => {
    const text = 'The source mentions Binoculars and DeBERTa.'
    const s = sentence(text, { scores: { classification_response: 0.7, classification_response_secondary: 0,
      pattern_ratio: 1.2345, pattern_response: 0.45, style: 0.2, 'check.0123456789abcdef': 0.8 },
      reasons: ['Pattern ratio 1.235 vs threshold 0.901 — review the context.', 'Source says “Binoculars”.'] })
    render(<ManuscriptView text={text} sentences={[s]} paragraphs={[[0, s.end]]} />)
    fireEvent.click(screen.getByRole('button', { name: /Sentence 1, AI-like band/ }))
    const rows = within(screen.getByRole('table'))
    for (const label of ['Text response', 'Second text response', 'Pattern ratio', 'Pattern response', 'Style response', 'Additional check 1']) {
      expect(rows.getByText(label)).toBeTruthy()
    }
    expect(rows.getByText('0.000')).toBeTruthy()
    expect(rows.getByText('1.234')).toBeTruthy()
    expect(screen.getByRole('table').textContent).not.toMatch(/classifier|binoculars|check\./i)
    expect(screen.getByText(s.reasons[0])).toBeTruthy()
    expect(screen.getByText(s.reasons[1])).toBeTruthy()
    expect(screen.getByText(/ratios and responses use different scales/)).toBeTruthy()
  })

  it('uses exact Unicode codepoint offsets in manuscript and rail after emoji and non-BMP letters', () => {
    const first = '😀 First 𐐷 sentence.'
    const second = 'Second e\u0301 sentence.' // Keep decomposed source characters; do not normalize.
    const text = `${first} ${second}`
    const start = Array.from(first).length + 1
    const sentences = [sentence(first), sentence(second, { idx: 1, start, end: Array.from(text).length })]
    const { container } = render(<ManuscriptView text={text} sentences={sentences} paragraphs={[[0, Array.from(text).length]]} />)
    const firstButton = screen.getByRole('button', { name: /Sentence 1, AI-like band/ })
    const secondButton = screen.getByRole('button', { name: /Sentence 2, AI-like band/ })
    expect(firstButton.textContent).toBe(first)
    expect(secondButton.textContent).toBe(second)
    fireEvent.click(firstButton)
    expect(container.querySelector('blockquote')?.textContent).toBe(first)
    fireEvent.click(secondButton)
    expect(container.querySelector('blockquote')?.textContent).toBe(second)
    expect(container.querySelector('blockquote')?.textContent).not.toBe(second.normalize('NFC'))
  })

  it('uses finite raw numbers, preserves zero and does not expose unknown score keys', () => {
    const rows = sentenceScoreRows({ pattern_ratio: 0, future_private_architecture: 1.5,
      failed: Number.NaN, overflow: Infinity, missing: undefined })
    expect(rows.map(({ label, value }) => ({ label, value }))).toEqual([
      { label: 'Pattern ratio', value: 0 }, { label: 'Additional check 1', value: 1.5 },
    ])
    expect(sentenceScoreRows({ constructor: 0.2 })[0].label).toBe('Additional check 1')
  })

  it('shows the public whole-document ratio as a raw value and supports its absence', () => {
    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    const verdict: Verdict = { probability_ai: null, band: 'uncertain', confidence: 'low', label: 'Inconclusive',
      summary: '', reasoning: [], caveats: [], signals_for_ai: 0, signals_for_authentic: 0, signals_total: 0 }
    const stats: NonNullable<NonNullable<Analysis['result']>['stats']> = { words: 100, sentences: 3,
      ai_sentences: 1, mixed_sentences: 1, human_sentences: 1, ai_share: 0.3, mixed_share: 0.3,
      weighted_ai_share: 0.3, document_pattern_ratio: 1.125 }
    const { rerender } = render(<VerdictModule v={verdict} signals={[]} stats={stats} kind="document" />)
    expect(screen.getByText('Pattern ratio')).toBeTruthy()
    expect(screen.getByText('1.125')).toBeTruthy()
    expect(screen.getByText(/whole document, not a probability/)).toBeTruthy()
    expect(screen.queryByText(/Binoculars/i)).toBeNull()
    rerender(<VerdictModule v={verdict} signals={[]} stats={{ ...stats, document_pattern_ratio: null }} kind="document" />)
    expect(screen.queryByText('Pattern ratio')).toBeNull()
    rerender(<VerdictModule v={verdict} signals={[]} stats={{ ...stats, document_pattern_ratio: 0 }} kind="document" />)
    expect(screen.getByText('0.000')).toBeTruthy()
  })
})
