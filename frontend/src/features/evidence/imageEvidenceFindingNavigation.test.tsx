// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { Link, MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider, type UseQueryResult } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { AnalysisDetail, Finding } from '../../lib/api/queries'
import KeyFindings from '../analysis/KeyFindings'

vi.mock('../../lib/api/queries', () => ({ useModelCards: () => ({ data: [] }) }))
afterEach(cleanup)

describe('evidence finding navigation', () => {
  it('reveals a referenced finding once in plain view without requiring the evidence ledger', async () => {
    const scroll = vi.fn()
    Element.prototype.scrollIntoView = scroll
    const finding = { id: 'F5', title: 'Compression difference', statement: 'A recompression comparison found an inspection region.',
      family: 'F5_compression_history', rank: 1, strength: 'weak', review_priority: 'medium', evidence_ids: [] } as unknown as Finding
    const query = { data: [finding], isSuccess: true, isPending: false, error: null } as unknown as UseQueryResult<Finding[], Error>
    const analysis = { id: 'analysis-1', kind: 'image', decision: null, caveats: [] } as unknown as AnalysisDetail
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    client.setQueryData(['analysis', analysis.id, 'evidence-packets'], {
      status: 'not_generated', schema: 'chipmunk.image-evidence/1', analysis_id: analysis.id, packets: [],
    })
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/analyses/analysis-1?view=plain']}>
      <KeyFindings analysis={analysis} query={query} signals={[]} exhibits={[]} token="" plain />
      <Link to="#finding-F5">Evidence reference F5</Link>
    </MemoryRouter></QueryClientProvider>)
    expect(document.getElementById('finding-F5')).toBeNull()
    fireEvent.click(screen.getByRole('link', { name: 'Evidence reference F5' }))
    expect(await screen.findByText(finding.statement)).toBeTruthy()
    expect(document.querySelectorAll('#finding-F5')).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'Hide the full list' }).getAttribute('aria-expanded')).toBe('true')
    expect(scroll).toHaveBeenCalledWith({ block: 'center' })
  })
})
