// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ImageEvidenceCards, { ImageEvidenceContent } from './ImageEvidenceCards'
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
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const reflected: NonNullable<ImageEvidenceSet['copy_similarity']> = {
  schema: 'same-image-reflection-similarity/1', status: 'supported_pair',
  source_sha256: sourceHash, coordinate_space: 'exif-oriented-original', authorship_confidence_contribution: 0,
  interpretation: 'Copying, intentional symmetry, reflections and repeated patterns can all produce this result. The comparison cannot identify which area came first or establish that the image was edited or made with AI.',
  pairs: [{ id: 'R01', transform: 'horizontal_reflection',
    regions: { area_a: { x: 108, y: 168, w: 192, h: 128 }, area_b: { x: 594, y: 324, w: 192, h: 128 } },
    artifacts: { area_a: { ...artifact('1'), width: 192, height: 128 }, area_b: { ...artifact('2'), width: 192, height: 128 } },
  }],
}
const copyOnly: ImageEvidenceSet = { ...evidence, version: 'image-evidence-7', packets: [], copy_similarity: reflected }
const translated: NonNullable<ImageEvidenceSet['copy_translation']> = {
  schema: 'same-image-copy-similarity/1', status: 'supported_pair',
  source_sha256: sourceHash, coordinate_space: 'exif-oriented-original', authorship_confidence_contribution: 0,
  interpretation: 'Copies and naturally repeated detail can both produce a match.',
  pairs: [{ id: 'T01', transform: 'same_orientation',
    regions: { area_a: { x: 89, y: 154, w: 193, h: 124 }, area_b: { x: 514, y: 319, w: 189, h: 120 } },
    artifacts: { area_a: { ...artifact('5'), width: 193, height: 124 }, area_b: { ...artifact('6'), width: 189, height: 120 } },
  }],
}

const geometric: NonNullable<ImageEvidenceSet['copy_affine']> = {
  schema: 'same-image-affine-support/1', status: 'supported_pair', source_sha256: sourceHash,
  authorship_confidence_contribution: 0, interpretation: 'The masks show where image detail matched after alignment.',
  pairs: [{ id: 'G01', regions: translated.pairs[0].regions, selected_pixels: { area_a: 4251, area_b: 4102 },
    artifacts: { ...translated.pairs[0].artifacts, mask_a: artifact('7'), mask_b: artifact('8') } }],
}

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
  const exactPair: NonNullable<ImageEvidenceSet['embedded_pair_difference']> = {
    schema: 'same-image-exact-difference/1', status: 'local_difference', source_sha256: sourceHash,
    authorship_confidence_contribution: 0, interpretation: 'The comparison carries zero weight in AI authorship confidence.',
    pair: { id: 'D01', regions: { area_a: { x: 19, y: 38, w: 206, h: 206 }, area_b: { x: 459, y: 46, w: 206, h: 206 } },
      difference_regions: { area_a: { x: 113, y: 112, w: 23, h: 19 }, area_b: { x: 553, y: 120, w: 23, h: 19 } },
      matching_pixels: 41999, different_pixels: 437, witnesses: [],
      artifacts: { area_a: artifact('1'), area_b: artifact('2'), overlay_a: artifact('3'), overlay_b: artifact('4'), difference_mask: artifact('5') },
    },
  }
  it('shows exact paired differences, toggles the source crops and links to the measured bounds', () => {
    show({ ...evidence, version: 'image-evidence-15', embedded_pair_difference: exactPair, packets: [] })
    const card = within(screen.getByRole('region', { name: 'Pixel differences between matching areas' }))
    expect(card.getByText(/437 differing pixels/)).toBeTruthy()
    expect(card.getByText(/other matching areas, including resized copies/)).toBeTruthy()
    expect(card.getByText(/zero weight in AI authorship confidence/)).toBeTruthy()
    expect(card.getByAltText('Area A with differing pixels highlighted in orange').getAttribute('src'))
      .toContain(`${'3'.repeat(64)}.png?token=signed%20token`)
    const source = card.getByRole('link', { name: 'Inspect differences in Area B' }).getAttribute('href') ?? ''
    expect(new URL(source, 'http://localhost').searchParams.get('focus')).toBe('553,120,23,19')
    fireEvent.click(card.getByRole('button', { name: 'Show original crops' }))
    expect(card.getByAltText('Area A original crop').getAttribute('src')).toContain(`${'1'.repeat(64)}.png`)
    expect(card.getByAltText('Area B original crop').getAttribute('src')).toContain(`${'2'.repeat(64)}.png`)
  })
  it('shows ambiguity without implying that the submitted image is original', () => {
    show({ ...evidence, version: 'image-evidence-15', packets: [],
      embedded_pair_difference: { ...exactPair, status: 'unassessed_ambiguous_match', pair: null } })
    const card = within(screen.getByRole('region', { name: 'Pixel differences between matching areas' }))
    expect(card.getByText(/made the two-area comparison ambiguous/)).toBeTruthy()
    expect(card.queryByRole('img')).toBeNull()
    expect(card.queryByRole('button')).toBeNull()
  })
  const comparison: NonNullable<ImageEvidenceSet['comparison']> = {
    schema: 'retained-regional-comparison/1', status: 'ready', source_sha256: sourceHash,
    map_grid: [16, 12], selection: 'retained_component_masks_no_refilter', authorship_confidence_contribution: 0,
    members: [{ id: 'sensitive', label: 'Sensitive regional check', status: 'measured', packet_ids: ['E01'] },
      { id: 'complementary', label: 'Complementary regional check', status: 'measured', packet_ids: [] }],
    counts: { sensitive: 10, complementary: 18, wider: 27, shared: 1 },
    artifacts: { wider_overlay: artifact('1'), shared_overlay: artifact('2'), wider_mask: artifact('3'), shared_mask: artifact('4') },
    interpretation: 'Agreement does not establish AI origin or increase confidence.', limitations: ['Only retained components are compared.'],
  }
  it('shows actual wider and shared assets, toggles masks and links to retained cards', () => {
    show({ ...evidence, version: 'image-evidence-4', comparison })
    expect(screen.getByTestId('image-regional-comparison').textContent).toContain('Shared view: 1 measured cells')
    expect(screen.getByAltText(/Shared view: both checks; flagged cells/).getAttribute('src')).toContain(`${'2'.repeat(64)}.png`)
    expect(screen.getByRole('link', { name: 'E01' }).getAttribute('href')).toBe('/#image-evidence-E01')
    fireEvent.click(screen.getByRole('button', { name: 'Show comparison masks' }))
    expect(screen.getByAltText(/Shared view: both checks; white selected/).getAttribute('src')).toContain(`${'4'.repeat(64)}.png`)
    expect(screen.getByText(/A single shared cell is preserved/)).toBeTruthy()
  })
  it('does not substitute a zero shared count when a check failed or withheld its candidates', () => {
    show({ ...evidence, comparison: { ...comparison, status: 'unavailable', counts: null, artifacts: {},
      members: [{ ...comparison.members[0] }, { ...comparison.members[1], status: 'withheld' }] } })
    const text = screen.getByTestId('image-regional-comparison').textContent
    expect(text).toContain('Candidate regions withheld')
    expect(text).toContain('does not establish that the image is unaltered')
    expect(text).not.toContain('Shared view: 0')
    expect(screen.queryByRole('button', { name: 'Show comparison masks' })).toBeNull()
  })
  it('declares bounded coverage and does not clear a zero intersection', () => {
    show({ ...evidence, comparison: { ...comparison, status: 'bounded', counts: { ...comparison.counts!, shared: 0, wider: 28 } } })
    expect(screen.getByText(/Some eligible regions are not shown/)).toBeTruthy()
    expect(screen.getByText(/disagreement does not clear the file/)).toBeTruthy()
  })

  it('loads the reflected pair in the ordinary image section without any regional maps', async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => copyOnly })
    vi.stubGlobal('fetch', fetch)
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/analyses/analysis-1/image-evidence?view=plain']}>
      <ImageEvidenceCards analysisId={copyOnly.analysis_id} token="signed token" mediaReady />
    </MemoryRouter></QueryClientProvider>)

    const pairSection = await screen.findByRole('region', { name: 'Same-image copy similarity' })
    expect(fetch).toHaveBeenCalledWith('/api/v2/analyses/analysis-1/evidence-packets', expect.anything())
    const pair = within(pairSection)
    expect(pair.getByRole('heading', { name: 'R01 · Reflected pair' })).toBeTruthy()
    expect(pairSection.textContent).toContain('Similar areas were found after horizontal reflection.')
    expect(pairSection.textContent).toContain('Coordinates use native pixels in the oriented original.')
    expect(pairSection.textContent).toContain('cannot identify which area came first or establish that the image was edited or made with AI')
    const figures = pair.getAllByRole('figure')
    expect(figures).toHaveLength(2)
    for (const [index, role] of (['area_a', 'area_b'] as const).entries()) {
      const label = index === 0 ? 'Area A' : 'Area B'
      const { x, y, w, h } = reflected.pairs[0].regions[role]
      const crop = within(figures[index]).getByRole('img', { name: `R01: ${label} cropped from the submitted image` })
      const link = within(figures[index]).getByRole('link', {
        name: `${label} in the original image`, description: `x ${x}, y ${y}, ${w} × ${h} px`,
      })
      expect(crop.getAttribute('src')).toBe(`/api/v2/analyses/analysis-1/evidence-packets/evidence-set-1/images/${reflected.pairs[0].artifacts[role].sha256}.png?token=signed%20token`)
      expect(crop.getAttribute('width')).toBe(String(w))
      expect(crop.getAttribute('height')).toBe(String(h))
      const source = new URL(link.getAttribute('href')!, 'https://local.invalid')
      expect(source.pathname).toBe('/analyses/analysis-1/evidence')
      expect(parseViewerState(source.searchParams)).toMatchObject({ focus: { x, y, w, h }, layers: { map: null, marks: [] } })
    }
    expect(screen.queryByTestId('image-regional-comparison')).toBeNull()
    expect(screen.queryByTestId('image-evidence-card')).toBeNull()
    expect(screen.getByTestId('image-evidence-empty').textContent).toContain('No other local editing regions were recorded.')
  })

  it.each([
    ['no_pair', 'This check found no qualifying reflected pair.'],
    ['unsupported_geometry', 'This check requires native dimensions'],
    ['unassessed_match_budget', 'The check found too many candidate matches'],
  ] as const)('describes %s without fabricating a pair or clearing the image', (status, message) => {
    show({ ...copyOnly, copy_similarity: { ...reflected, status, pairs: [] } })
    const summary = screen.getByRole('region', { name: 'Same-image copy similarity' })
    expect(summary.textContent).toContain(message)
    expect(within(summary).queryAllByRole('img')).toHaveLength(0)
    expect(within(summary).queryAllByRole('link')).toHaveLength(0)
    expect(summary.textContent).not.toContain('Similar areas were found after horizontal reflection.')
    expect(screen.getByTestId('image-evidence-empty').textContent).toContain('does not establish that the image is unaltered')
  })

  it('uses the supported sizes recorded by each reflection evidence version', () => {
    const unsupported = { ...copyOnly, copy_similarity: { ...reflected, status: 'unsupported_geometry' as const, pairs: [] } }
    const { rerender } = show({ ...unsupported, version: 'image-evidence-8' })
    const summary = screen.getByRole('region', { name: 'Same-image copy similarity' })
    expect(summary.textContent).toContain('long side of 513–1,024 px.')
    for (const version of ['image-evidence-9', 'image-evidence-10', 'image-evidence-11', 'image-evidence-12'] as const) {
      rerender(<MemoryRouter><ImageEvidenceContent set={{ ...unsupported, version }} token="signed token" mediaReady /></MemoryRouter>)
      expect(summary.textContent).toContain('a long side of 1,025–2,048 px and a short side of at least 768 px')
    }
  })

  it('describes both signed-credential skips once and shows no copy crops', () => {
    const status = 'skipped_trusted_ai_credentials'
    show({ ...copyOnly, version: 'image-evidence-12',
      copy_similarity: { ...reflected, status, pairs: [] }, copy_translation: { ...translated, status, basis: null, pairs: [] } })
    expect(screen.getAllByText(/Trusted Content Credentials cover this image/)).toHaveLength(1)
    expect(screen.queryByRole('region', { name: /copy similarity/i })).toBeNull()
    expect(screen.queryByTestId('image-evidence-empty')).toBeNull()
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('shows both original non-mirrored crops with their own locations alongside the reflected check', () => {
    show({ ...copyOnly, version: 'image-evidence-10', copy_translation: translated })
    const pair = within(screen.getByRole('region', { name: 'Copy similarity without mirroring' }))
    expect(pair.getByRole('heading', { name: 'T01 · Non-mirrored pair' })).toBeTruthy()
    expect(pair.getByText(translated.interpretation)).toBeTruthy()
    expect(pair.getByText('This check carries zero weight in AI authorship confidence.')).toBeTruthy()
    expect(pair.queryByText(/^The match/)).toBeNull()
    expect(screen.getByRole('heading', { name: 'R01 · Reflected pair' })).toBeTruthy()
    for (const [role, label] of [['area_a', 'Area A'], ['area_b', 'Area B']] as const) {
      const region = translated.pairs[0].regions[role]
      const { x, y, w, h } = region
      const crop = pair.getByRole('img', { name: `T01: ${label} cropped from the submitted image` })
      expect(crop.getAttribute('src')).toContain(`/${translated.pairs[0].artifacts[role].sha256}.png?token=signed%20token`)
      const link = pair.getByRole('link', { name: `${label} in the original image`, description: `x ${x}, y ${y}, ${w} × ${h} px` })
      const source = new URL(link.getAttribute('href')!, 'https://local.invalid')
      expect(parseViewerState(source.searchParams)).toMatchObject({ focus: region, layers: { map: null, marks: [] } })
    }
  })

  it.each([
    ['native_patch_alignment', 'The match aligns patches on the original pixel grid.'],
    ['geometric_features', 'The match uses corresponding features in both areas.'],
  ] as const)('shows the stored %s basis next to the new pair', (basis, explanation) => {
    show({ ...copyOnly, version: 'image-evidence-11', copy_translation: { ...translated, basis } })
    const translation = within(screen.getByRole('region', { name: 'Copy similarity without mirroring' }))
    const reflection = within(screen.getByRole('region', { name: 'Same-image copy similarity' }))
    expect(translation.getByText(explanation)).toBeTruthy()
    expect(reflection.queryByText(explanation)).toBeNull()
  })

  it('explains exact JPEG block matches and displays completed negative coverage', () => {
    const blockCheck = { ...translated, basis: 'jpeg_brightness_blocks', jpeg_block_review: 'supported_pair' } as const
    const first = show({ ...copyOnly, version: 'image-evidence-16', copy_translation: blockCheck })
    const explanation = 'The complete encoded brightness blocks matched exactly in both areas.'
    expect(screen.getByText(explanation)).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'T01 · Non-mirrored pair' })).toBeTruthy()
    first.unmount()
    show({ ...copyOnly, version: 'image-evidence-16',
      copy_translation: { ...translated, basis: null, jpeg_block_review: 'no_pair', status: 'no_pair', pairs: [] } })
    expect(screen.getByText('The JPEG brightness-block check found no qualifying repeated area.')).toBeTruthy()
    expect(screen.queryByText(explanation)).toBeNull()
  })

  it('pairs each geometric comparison mask with the native source crop and exact area link', () => {
    show({ ...copyOnly, version: 'image-evidence-13', copy_affine: geometric })
    const check = within(screen.getByRole('region', { name: 'Geometric copy comparison' }))
    expect(check.getByText('Matching detail was found after geometric alignment.')).toBeTruthy()
    expect(check.getByText('This check carries zero weight in AI authorship confidence.')).toBeTruthy()
    expect(check.getByText(/4251 pixels selected from neighborhoods/)).toBeTruthy()
    expect(check.getByText(/same selection mapped to 4102 pixels/)).toBeTruthy()
    for (const [role, label] of [['area_a', 'Area A'], ['area_b', 'Area B']] as const) {
      expect(check.getByRole('img', { name: `G01: ${label} cropped from the submitted image` })).toBeTruthy()
      const mask = check.getByRole('img', { name: `G01: ${label} ${role === 'area_a' ? 'selected' : 'mapped'} mask` })
      expect(mask.getAttribute('src')).toContain(`/${geometric.pairs[0].artifacts[role === 'area_a' ? 'mask_a' : 'mask_b'].sha256}.png`)
      const link = check.getByRole('link', { name: `${label} in the original image` })
      expect(parseViewerState(new URL(link.getAttribute('href')!, 'http://localhost').searchParams)).toMatchObject({ focus: geometric.pairs[0].regions[role] })
    }
  })

  it('keeps the non-mirrored search coverage bounded when the wider reflected search applies', () => {
    show({ ...copyOnly, version: 'image-evidence-11',
      copy_translation: { ...translated, status: 'unsupported_geometry', basis: null, pairs: [] } })
    const section = screen.getByRole('region', { name: 'Copy similarity without mirroring' })
    expect(section.textContent).toContain('a long side of 513–1,024 px')
    expect(section.textContent).not.toContain('2,048')
    expect(within(section).queryByRole('img')).toBeNull()
    expect(within(section).queryByText(/^The match/)).toBeNull()
  })

  it('shows the working size separately from native evidence on a large image', () => {
    show({ ...copyOnly, version: 'image-evidence-14', source: { ...copyOnly.source, width: 1920, height: 1280 },
      copy_affine: { ...geometric, schema: 'same-image-affine-support/2',
        proposal_search: { basis: 'reduced', width: 1024, height: 683 } } })
    const section = within(screen.getByRole('region', { name: 'Geometric copy comparison' }))
    expect(section.getByText(/1024 × 683 pixel working image/)).toBeTruthy()
    expect(section.getByText(/Coordinates use native pixels/)).toBeTruthy()
    expect(section.getByRole('link', { name: 'Area A in the original image' })).toBeTruthy()
  })

  it('keeps the other reflected crop and native coordinates available when a crop fails', () => {
    const { rerender } = show(copyOnly)
    fireEvent.error(screen.getByRole('img', { name: /Area A cropped/ }))
    expect(screen.getByRole('alert').textContent).toContain('could not be loaded')
    expect(screen.getByRole('link', { name: 'Area A in the original image', description: 'x 108, y 168, 192 × 128 px' })).toBeTruthy()
    expect(screen.getByRole('img', { name: /Area B cropped/ })).toBeTruthy()
    rerender(<MemoryRouter><ImageEvidenceContent set={copyOnly} token="renewed" mediaReady /></MemoryRouter>)
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.getByRole('img', { name: /Area A cropped/ }).getAttribute('src')).toContain('?token=renewed')
  })
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

  it('shows the bound matching patch with the existing image highlight in a v6 record', () => {
    const packet = evidence.packets[0]
    show({ ...evidence, version: 'image-evidence-6', packets: [{ ...packet, evidence_type: 'localisation',
      interpretation: 'This pixel similarity is compatible with copying or repeated detail.',
      artifacts: { ...packet.artifacts, mask: artifact('e'), copy_donor: artifact('1') },
    }] })
    const match = screen.getByAltText('E01: matching patch elsewhere in the submitted image')
    expect(match.getAttribute('src')).toContain(`${'1'.repeat(64)}.png?token=signed%20token`)
    expect(screen.getByText(/Cropped from the submitted image for comparison/)).toBeTruthy()
    expect(screen.getByText(/compatible with copying or repeated detail/)).toBeTruthy()
    expect(screen.getByAltText(/full source image with the inspection region outlined/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Show measured mask for E01' }))
    expect(screen.getByAltText(/selected connected region on the full measured grid/)).toBeTruthy()
    expect(screen.getByAltText('E01: matching patch elsewhere in the submitted image')).toBe(match)
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
    show({ ...evidence, version: 'image-evidence-3', packets: [], coverage: { inspection_checks: [check({
      source_quality: { schema: 'image-source-quality/1',
        native_pixel_shares: { clipped: .083, flat: .061, compressed: 0, upscaled: 0, unreliable: .127 } },
    })] } })
    const coverage = screen.getByTestId('image-inspection-checks')
    expect(screen.getByText('Regions withheld by the image gate')).toBeTruthy()
    expect(coverage.textContent).toContain('Whole-image response 0.2')
    expect(coverage.textContent).toContain('12 measured regions were withheld')
    expect(coverage.textContent).toContain('12.5% of the measured grid was excluded')
    expect(coverage.textContent).toContain('Clipped highlights or very dark pixels')
    expect(coverage.textContent).toContain('Low local contrast')
    expect(coverage.textContent).not.toContain('Low-detail JPEG blocks')
    expect(coverage.textContent).toContain('not a region or pixel probability')
    expect(coverage.textContent).toContain('Skipped, failed or unavailable checks')
    expect(coverage.textContent).not.toContain('loc.private')
    expect(coverage.textContent).not.toContain('Internal method')
    expect(screen.queryAllByRole('img')).toHaveLength(0)
    expect(screen.getByTestId('image-evidence-empty').textContent).toContain('does not establish that the image is unaltered')
  })

  it('shows only the frozen reliability mask matching the measured grid', () => {
    const mask = { ...artifact('d'), width: 512, height: 320 }
    const value = { ...evidence, packets: [], overview: { inspection_01_reliability: mask },
      coverage: { inspection_checks: [check({ reliability_mask_sha256: mask.sha256 })] } }
    const { rerender } = show(value)
    fireEvent.click(screen.getByText('Show which areas were checked'))
    const image = screen.getByAltText('Reliability mask on the measured image grid')
    expect(image.getAttribute('src')).toBe(evidenceImageUrl(value, mask, 'signed token'))
    expect(image.getAttribute('width')).toBe('512')
    expect(screen.getByText(/Black marks cells excluded from region selection/)).toBeTruthy()

    const wrongGrid = { ...value, overview: { inspection_01_reliability: { ...mask, width: 511 } } }
    rerender(<MemoryRouter><ImageEvidenceContent set={wrongGrid} token="signed token" mediaReady /></MemoryRouter>)
    expect(screen.queryByText('Show which areas were checked')).toBeNull()
    expect(screen.getByText(/12.5% of the measured grid was excluded/)).toBeTruthy()
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

  it('distinguishes flip-check rejection, unreviewed regions, and display limits', () => {
    show({ ...evidence, packets: [], coverage: { inspection_checks: [check({ candidate_eligibility: gate(.5),
      eligible_components: 1, selected_components: 0, omitted_components: 1,
      flip_consistency: { schema: 'image-flip-consistency/1', inspected_components: 2, supported_components: 1,
        withheld_components: 1, candidates: [
          { region_id: 'R1', source_pixels: 11, shared_pixels: 5, retained: false },
          { region_id: 'R2', source_pixels: 10, shared_pixels: 5, retained: true },
        ] },
    })] } })
    expect(screen.getByText('Supported by the flip check')).toBeTruthy()
    expect(screen.getByText(/same detector on a horizontal flip.*50% of its source pixels.*no AI authorship weight/)).toBeTruthy()
    expect(screen.getByText('The flip check withheld 1 of 2 candidate regions.')).toBeTruthy()
    expect(screen.getByText('10 measured regions were outside the two-candidate review limit.')).toBeTruthy()
    expect(screen.getByText(/1 supported region is not shown/)).toBeTruthy()
    expect(screen.queryByText(/measured regions were withheld by the image gate/)).toBeNull()
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

  it('shows the recorded cell counts for an area-based eligibility rule', () => {
    const area = { ...gate(.125, false), schema: 'image-candidate-eligibility/2' as const,
      threshold: .25, threshold_exact_decimal: '0.25', active_cells: 128, minimum_active_cells: 256, input_cells: 1024 }
    show({ ...evidence, packets: [], coverage: { inspection_checks: [check({ candidate_eligibility: area })] } })
    const note = screen.getByTestId('image-candidate-eligibility')
    expect(note.textContent).toContain('Active screening cells 128 · minimum 256')
    expect(note.textContent).toContain('Small or weak edits may be missed.')
    expect(note.textContent).not.toContain('0.125')
  })
})
