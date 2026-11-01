// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import { ImageEvidenceContent } from './ImageEvidenceCards'
import { evidenceImageUrl, evidenceSourceUrl, type CandidateEligibility, type ImageEvidenceSet, type ImageInspectionCheck } from './imagePackets'
import { parseViewerState, serialiseViewerState } from '../viewer/viewerState'

const sourceHash = 'a'.repeat(64)
const artifact = (hash: string) => ({ sha256: hash.repeat(64), media_type: 'image/png' as const, width: 300, height: 200, url: 'https://unrelated.invalid/image.png' })
const evidence: ImageEvidenceSet = {
  schema: 'chipmunk.image-evidence/1', status: 'ready', id: 'evidence-set-1', version: 'image-evidence-1', sha256: 'f'.repeat(64),
  analysis_id: 'analysis-1', revision: 1, original_result_hash: 'e'.repeat(64),
  source: { sha256: sourceHash, width: 1000, height: 800, coordinate_space: 'exif-oriented-original' }, limitations: [], omitted_regions: 0,
  packets: [{
    id: 'E01', finding_ids: ['F5'], title: 'Recompression difference', evidence_type: 'anomaly', semantics: 'error_level',
    region: { x: 120, y: 80, w: 60, h: 40 }, crop_region: { x: 105, y: 70, w: 90, h: 60 },
    observation: 'This area has stronger recompression differences than nearby map values.',
    interpretation: 'A diagnostic comparison identifies this area for inspection.',
    overlay_legend: 'Orange intensity shows relative recompression difference within this image; it is not an AI probability.',
    limitations: ['Normal compression and detailed texture can also produce this difference.'],
    measurements: [{ label: 'Mean diagnostic value', value: 0.7134, unit: 'map units' }],
    signal_refs: [{ id: 'check.opaque:global', display_id: 'S04' }], exhibit_refs: ['regions-1'],
    artifacts: { context: artifact('b'), crop: artifact('c'), overlay: artifact('d') },
  }],
}

const show = (value = evidence, token = 'signed token') => render(<MemoryRouter><ImageEvidenceContent set={value} token={token} mediaReady /></MemoryRouter>)
afterEach(cleanup)

const gate = (score: number, eligible = score >= .5): CandidateEligibility => ({
  schema: 'image-candidate-eligibility/1', status: 'evaluated', scope: 'whole_image', comparison: 'greater_or_equal',
  score, score_hex: '0x1.0000000000000p-1', score_exact_decimal: String(score),
  threshold: .5, threshold_hex: '0x1.0000000000000p-1', threshold_exact_decimal: '0.5', eligible,
  reason: 'Internal method identifiers must not be used as the display label.', source_sha256: sourceHash,
})
const check = (overrides: Partial<ImageInspectionCheck> = {}): ImageInspectionCheck => ({
  signal_refs: [{ id: 'loc.private-implementation:global', display_id: 'S06' }], exhibit_ref: 'heat-map-1',
  candidate_eligibility: gate(.2), map_grid: [512, 320], excluded_area_fraction: .125,
  raw_components: 12, eligible_components: 0, selected_components: 0, omitted_components: 0, ...overrides,
})

describe('image evidence cards', () => {
  it('shows original crops and cautious regional evidence without requiring Expert view', () => {
    show()
    expect(screen.getByText('Inspection region')).toBeTruthy()
    expect(screen.getByText('Mean diagnostic value')).toBeTruthy()
    expect(screen.getByText('0.713 map units')).toBeTruthy()
    expect(screen.getByText(evidence.packets[0].limitations[0])).toBeTruthy()
    expect(screen.getByText(/not confirmed edit locations/)).toBeTruthy()
    const crop = screen.getByAltText(/unaltered source crop/)
    expect(crop.getAttribute('src')).toContain(`${'c'.repeat(64)}.png?token=signed%20token`)
    expect(crop.getAttribute('src')).not.toContain('unrelated.invalid')
    expect(screen.getByRole('link', { name: 'F5' }).getAttribute('href')).toBe('/#finding-F5')
  })

  it('switches only the crop display while preserving the same observation and region', () => {
    show()
    const toggle = screen.getByRole('button', { name: 'Show diagnostic overlay for E01' })
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByAltText(/diagnostic overlay on the same source crop/).getAttribute('src')).toContain(`${'d'.repeat(64)}.png`)
    expect(screen.getByText(evidence.packets[0].overlay_legend)).toBeTruthy()
    expect(screen.getByText(evidence.packets[0].observation)).toBeTruthy()
    fireEvent.click(toggle)
    expect(screen.getByAltText(/unaltered source crop/)).toBeTruthy()
  })

  it('frames the exact source region without inheriting a heatmap', () => {
    const href = evidenceSourceUrl(evidence.analysis_id, evidence.packets[0])
    const parsed = parseViewerState(new URL(href, 'https://local.invalid').searchParams)
    expect(parsed.focus).toEqual(evidence.packets[0].region)
    expect(parsed.layers).toEqual({ map: null, marks: [] })
    expect(parseViewerState(serialiseViewerState(parsed)).focus).toEqual(parsed.focus)
    for (const invalid of ['-1,2,3,4', '0,0,0,40', '0,0,20,Infinity', '1,2,3']) {
      expect(parseViewerState(new URLSearchParams({ focus: invalid })).focus).toBeNull()
    }
  })

  it('shows an actual localisation mask separately from the source crop and overlay', () => {
    const localised: ImageEvidenceSet = { ...evidence, version: 'image-evidence-2', packets: [{
      ...evidence.packets[0], evidence_type: 'localisation', semantics: 'p_manipulated',
      artifacts: { ...evidence.packets[0].artifacts, mask: artifact('e') },
      localisation: { threshold: 0.5, grid: [512, 320], region_cells: 1000, region_area_fraction: 0.03,
        reliable_area_fraction: 0.9, source_pixels_per_cell: [2, 2], validation_status: 'local_evaluation_pending',
        candidate_eligibility: gate(.75) },
    }] }
    show(localised)
    expect(screen.getByText(/Measured grid 512 × 320/)).toBeTruthy()
    expect(screen.getByTestId('image-candidate-eligibility').textContent).toContain('Whole-image response 0.75')
    const toggle = screen.getByRole('button', { name: 'Show measured mask for E01' })
    fireEvent.click(toggle)
    expect(screen.getByAltText(/selected connected region on the full measured grid/).getAttribute('src')).toContain(`${'e'.repeat(64)}.png`)
    expect(screen.getByText(/black means outside this component, not verified authentic/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Show diagnostic overlay for E01' }))
    expect(toggle.getAttribute('aria-pressed')).toBe('false')
    expect(screen.getByAltText(/diagnostic overlay on the same source crop/)).toBeTruthy()
  })

  it('separates a prepared empty result from an older result without evidence', () => {
    const { unmount } = show({ ...evidence, packets: [] })
    expect(screen.getByTestId('image-evidence-empty').textContent).toContain('does not establish that the image is unaltered')
    expect(screen.queryAllByRole('img')).toHaveLength(0)
    unmount()
    show({ ...evidence, status: 'not_generated', id: null, sha256: null, packets: [] })
    expect(screen.getByRole('status').textContent).toContain('has not been prepared')
    expect(screen.queryByTestId('image-evidence-empty')).toBeNull()
  })

  it('keeps measurements readable when an evidence image fails and recovers after token renewal', () => {
    const { rerender } = show()
    fireEvent.error(screen.getByAltText(/unaltered source crop/))
    expect(screen.getByRole('alert').textContent).toContain('could not be loaded')
    expect(screen.getByText(evidence.packets[0].observation)).toBeTruthy()
    rerender(<MemoryRouter><ImageEvidenceContent set={evidence} token="renewed" mediaReady /></MemoryRouter>)
    expect(screen.getByAltText(/unaltered source crop/).getAttribute('src')).toContain('?token=renewed')
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('waits for a secure media token and refuses invalid artifact hashes', () => {
    render(<MemoryRouter><ImageEvidenceContent set={evidence} token="" mediaReady={false} /></MemoryRouter>)
    expect(screen.queryAllByRole('img')).toHaveLength(0)
    expect(screen.getAllByText('Preparing secure media links…')).toHaveLength(2)
    expect(evidenceImageUrl(evidence, { ...artifact('b'), sha256: '../other' }, 'secret')).toBeNull()
  })

  it('shows suppressed regions and measurement limits even with no localisation packets', () => {
    show({ ...evidence, version: 'image-evidence-3', packets: [], coverage: { inspection_checks: [check()] } })
    const coverage = screen.getByTestId('image-inspection-checks')
    expect(screen.getByText('Regions withheld by the image gate')).toBeTruthy()
    expect(coverage.textContent).toContain('Whole-image response 0.2')
    expect(coverage.textContent).toContain('12 measured regions were withheld')
    expect(coverage.textContent).toContain('12.5% of the measured grid was excluded')
    expect(coverage.textContent).toContain('not a region or pixel probability')
    expect(coverage.textContent).toContain('Skipped, failed or unavailable checks')
    expect(coverage.textContent).not.toContain('loc.private')
    expect(coverage.textContent).not.toContain('Internal method')
    expect(screen.queryAllByRole('img')).toHaveLength(0)
    expect(screen.getByTestId('image-evidence-empty').textContent).toContain('does not establish that the image is unaltered')
  })

  it('separates image eligibility from bounded display and keeps exact near-threshold responses', () => {
    const rounded = { ...gate(.49999999999999994, false), score: .5, score_hex: '0x1.fffffffffffffp-2' }
    const { unmount } = show({ ...evidence, packets: [], coverage: { inspection_checks: [check({ candidate_eligibility: rounded })] } })
    expect(screen.getByText('0.49999999999999994')).toBeTruthy()
    expect(screen.getByText('Regions withheld by the image gate')).toBeTruthy()
    unmount()
    show({ ...evidence, coverage: { inspection_checks: [check({ candidate_eligibility: gate(.5), eligible_components: 12,
      selected_components: 2, omitted_components: 10 })] } })
    expect(screen.getByText('Eligible for region inspection')).toBeTruthy()
    expect(screen.getByText(/10 eligible regions are not shown/)).toBeTruthy()
    expect(screen.queryByText(/measured regions were withheld/)).toBeNull()
  })

  it('does not infer passed eligibility from absent or inconsistent metadata', () => {
    const { unmount } = show({ ...evidence, coverage: { inspection_checks: [check({ candidate_eligibility: undefined,
      eligible_components: 12, selected_components: 2, omitted_components: 10 })] } })
    expect(screen.getByText(/does not document a whole-image eligibility gate/)).toBeTruthy()
    expect(screen.getByText('Available for region selection')).toBeTruthy()
    expect(screen.queryByText('Eligible after image gate')).toBeNull()
    expect(screen.queryByText('Eligible for region inspection')).toBeNull()
    unmount()
    const roundedWithoutExact = { ...gate(.5, false), score_exact_decimal: undefined }
    show({ ...evidence, packets: [], coverage: { inspection_checks: [check({ candidate_eligibility: roundedWithoutExact,
      selected_components: 4, omitted_components: 0 })] } })
    expect(screen.getByText('Image eligibility unavailable')).toBeTruthy()
    expect(screen.getByText(/Region coverage counts are unavailable/)).toBeTruthy()
  })
})
