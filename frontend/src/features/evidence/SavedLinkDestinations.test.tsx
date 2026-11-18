// @vitest-environment jsdom

import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import { MultimediaEvidenceContent } from './MultimediaEvidenceCards'
import type { DocumentEvidenceSet, DocumentSavedLinkHosts, DocumentSavedLinkLocation } from './multimediaPackets'

const set: DocumentEvidenceSet = {
  schema: 'chipmunk.document-evidence/1', status: 'ready', id: null, version: 'test',
  sha256: null, analysis_id: 'analysis', revision: 1, original_result_hash: null, limitations: [],
  packets: [], omitted_windows: 0, source: { sha256: 'a'.repeat(64), offset_unit: 'unicode_codepoint' },
}

const anchors: { method: DocumentSavedLinkHosts['method']; location: DocumentSavedLinkLocation; shown: string }[] = [
  {
    method: 'html',
    location: { kind: 'decoded_html', ranges: [[31, 35], [42, 52]] },
    shown: 'Original decoded HTML source characters [31, 35), [42, 52)',
  },
  {
    method: 'email',
    location: { kind: 'decoded_email_html', mime_part: '1.2.1', ranges: [[113, 129]] },
    shown: 'Email MIME part 1.2.1 · Decoded email HTML source characters [113, 129)',
  },
  {
    method: 'python-docx',
    location: { kind: 'word_paragraph', body_path: [4], ranges: [[12, 28]] },
    shown: 'Word body path 4 · Original Word paragraph characters [12, 28)',
  },
]

function display(evidence: DocumentEvidenceSet) {
  return render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
    <MultimediaEvidenceContent set={evidence} token="" mediaReady={false} />
  </MemoryRouter>)
}

afterEach(cleanup)

describe('saved link destination evidence', () => {
  it.each(anchors)('shows exact $method locations and inert hostnames in the evidence view', ({ method, location, shown }) => {
    const review: DocumentSavedLinkHosts = {
      schema: 'document-saved-link-hosts/1', status: 'checked', method, offset_unit: 'unicode_codepoint',
      counts: { different_host: 2, same_host: 1 }, omitted: 1,
      findings: [{ label_host: 'public.example', saved_host: 'redirect.example', location }],
      authorship_confidence_contribution: 0, destinations_contacted: 0,
    }
    display({ ...set, saved_link_hosts: review })
    const region = within(screen.getByRole('region', { name: 'Saved link destinations' }))
    expect(region.getByText('Displayed hostname')).toBeTruthy()
    expect(region.getByText('public.example').tagName).toBe('BDI')
    expect(region.getByText('Saved destination hostname')).toBeTruthy()
    expect(region.getByText('redirect.example').tagName).toBe('BDI')
    expect(region.queryByRole('link')).toBeNull()
    expect(region.getByText(shown.slice(shown.indexOf('['))).parentElement?.textContent).toBe(shown)
    expect(region.getByText(/Ordinary redirects can explain a difference/)).toBeTruthy()
    expect(region.getByText(/final destination remains unverified/).textContent).toContain('adds no weight to the AI authorship estimate')
    expect(region.getByText('1 additional difference exceeds the display limit.')).toBeTruthy()
    expect(screen.queryByTestId('multimedia-evidence-empty')).toBeNull()
  })

  it('keeps the existing empty view for historical results without a saved-link record', () => {
    display(set)
    expect(screen.queryByRole('region', { name: 'Saved link destinations' })).toBeNull()
    expect(screen.getByTestId('multimedia-evidence-empty')).toBeTruthy()
  })
})
