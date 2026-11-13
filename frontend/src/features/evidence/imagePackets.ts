import { useQuery } from '@tanstack/react-query'
import { authHeaders, unwrap } from '../../lib/api/client'
import type { Box } from '../../lib/coords'

export interface EvidenceImage {
  sha256: string
  media_type: 'image/png'
  width: number
  height: number
  url: string
}

export interface CandidateEligibility {
  schema: 'image-candidate-eligibility/1' | 'image-candidate-eligibility/2'
  status: 'evaluated'
  eligible: boolean
  scope: 'whole_image'
  comparison: 'greater_or_equal'
  score: number
  score_hex: string
  score_exact_decimal?: string
  threshold: number
  threshold_hex: string
  threshold_exact_decimal?: string
  reason: string
  source_sha256?: string
  scalar_map_sha256?: string
  active_cells?: number
  minimum_active_cells?: number
  input_cells?: number
}

export interface ImageInspectionCheck {
  signal_refs: { id: string; display_id: string | null }[]
  exhibit_ref: string
  candidate_eligibility?: CandidateEligibility
  flip_consistency?: {
    schema: 'image-flip-consistency/1'
    candidates: { region_id: 'R1' | 'R2'; source_pixels: number; shared_pixels: number; retained: boolean }[]
    inspected_components: number
    supported_components: number
    withheld_components: number
  }
  map_grid?: [number, number]
  source_quality?: {
    schema: 'image-source-quality/1'
    native_pixel_shares: Record<'clipped' | 'flat' | 'compressed' | 'upscaled' | 'unreliable', number>
  }
  excluded_area_fraction?: number
  reliability_mask_sha256?: string | null
  raw_components: number
  eligible_components: number
  selected_components: number
  omitted_components: number
}

export interface ImageEvidencePacket {
  id: string
  finding_ids: string[]
  title: string
  evidence_type: 'anomaly' | 'localisation'
  semantics: 'error_level' | 'residual' | 'p_manipulated'
  region: Box
  crop_region: Box
  observation: string
  interpretation: string
  overlay_legend: string
  limitations: string[]
  measurements: { label: string; value: number; unit: string }[]
  signal_refs: { id: string; display_id: string | null }[]
  exhibit_refs: string[]
  artifacts: { context: EvidenceImage; crop: EvidenceImage; overlay: EvidenceImage; mask?: EvidenceImage; copy_donor?: EvidenceImage }
  localisation?: { threshold: number; grid: [number, number]; region_cells: number; region_area_fraction: number;
    reliable_area_fraction: number; source_pixels_per_cell: [number, number]; validation_status: string;
    candidate_eligibility?: CandidateEligibility }
}

interface ImageCopyEvidence<Schema, Transform> {
  schema: Schema
  status: 'supported_pair' | 'no_pair' | 'unsupported_geometry' | 'unassessed_match_budget' | 'skipped_trusted_ai_credentials'
  source_sha256: string
  coordinate_space: 'exif-oriented-original'
  authorship_confidence_contribution: 0
  interpretation: string
  pairs: { id: string; transform: Transform
    regions: Record<'area_a' | 'area_b', Box>
    artifacts: Record<'area_a' | 'area_b', EvidenceImage> }[]
}

export interface ImageEmbeddedPairDifference {
  schema: 'same-image-exact-difference/1'
  status: 'local_difference' | 'matching_without_difference' | 'no_supported_pair' | 'unassessed_ambiguous_match'
    | 'outside_scope' | 'unassessed_runtime' | 'skipped_trusted_ai_credentials'
  source_sha256: string
  authorship_confidence_contribution: 0
  interpretation: string
  pair: {
    id: 'D01'
    regions: Record<'area_a' | 'area_b', Box>
    difference_regions: Record<'area_a' | 'area_b', Box>
    matching_pixels: number
    different_pixels: number
    witnesses: [number, number][]
    artifacts: Record<'area_a' | 'area_b' | 'overlay_a' | 'overlay_b' | 'difference_mask', EvidenceImage>
  } | null
}

export interface ImageEvidenceSet {
  schema: 'chipmunk.image-evidence/1'
  status: 'ready' | 'not_generated' | 'not_applicable'
  id: string | null
  version: 'image-evidence-1' | 'image-evidence-2' | 'image-evidence-3' | 'image-evidence-4' | 'image-evidence-5' | 'image-evidence-6' | 'image-evidence-7' | 'image-evidence-8' | 'image-evidence-9' | 'image-evidence-10' | 'image-evidence-11' | 'image-evidence-12' | 'image-evidence-13' | 'image-evidence-14' | 'image-evidence-15' | 'image-evidence-16' | 'image-evidence-17' | 'image-evidence-18' | 'image-evidence-19' | 'image-evidence-20' | 'image-evidence-21'
  sha256: string | null
  analysis_id: string
  revision: number
  original_result_hash: string | null
  source: { sha256: string; width: number | null; height: number | null; coordinate_space: 'exif-oriented-original' }
  packets: ImageEvidencePacket[]
  limitations: string[]
  omitted_regions: number
  overview?: Record<string, EvidenceImage>
  coverage?: { inspection_checks?: ImageInspectionCheck[] }
  comparison?: ImageRegionalComparison
  copy_similarity?: ImageCopyEvidence<'same-image-reflection-similarity/1', 'horizontal_reflection'>
  copy_translation?: ImageCopyEvidence<'same-image-copy-similarity/1', 'same_orientation'> & {
    basis?: 'native_patch_alignment' | 'geometric_features' | 'jpeg_brightness_blocks' | 'jpeg_offgrid_texture' | 'png_exact_small_copy' | null
    jpeg_block_review?: 'supported_pair' | 'no_pair' | 'unsupported_profile' | 'unavailable' | 'unassessed_match_budget' | null
    small_png_copy_review?: {
      status: 'supported_pair' | 'no_pair' | 'unsupported_profile' | 'unavailable' | 'ambiguous_pairs' | 'unassessed_match_budget'
      measurement: { matching_pixels: number; different_color_values: number; perimeter_mean_absolute_difference: number[] } | null
    } | null
    jpeg_offgrid_review?: {
      status: 'supported_pair' | 'no_pair' | 'unsupported_profile' | 'unavailable' | 'unassessed_match_budget'
      measurement: { interior_mean: number; interior_p90: number; perimeter_means: (number | null)[] } | null
    } | null
  }
  copy_affine?: Omit<ImageCopyEvidence<'same-image-affine-support/1' | 'same-image-affine-support/2', never>, 'pairs' | 'coordinate_space'> & {
    proposal_search?: { basis: 'source' | 'reduced'; width: number; height: number } | null
    pairs: { id: string; regions: Record<'area_a' | 'area_b', Box>; selected_pixels: Record<'area_a' | 'area_b', number>
      artifacts: Record<'area_a' | 'area_b' | 'mask_a' | 'mask_b', EvidenceImage> }[]
  }
  prior_embedded_pair_evidence?: { id: string; version: 'image-evidence-12' | 'image-evidence-13' | 'image-evidence-14' | 'image-evidence-16' | 'image-evidence-17' | 'image-evidence-18' | 'image-evidence-19' | 'image-evidence-20' | 'image-evidence-21'; sha256: string }
  embedded_pair_difference?: ImageEmbeddedPairDifference
  native_interpolation_review?: {
    record: { status: string; observation: string; scope: string; limitation: string; basis?: string | null;
      supported_cells: number; region: [number, number, number, number] | null; affects_overall_confidence: false }
    artifacts: Partial<Record<'context' | 'crop', EvidenceImage>>
  }
}

export interface ImageRegionalComparison {
  schema: 'retained-regional-comparison/1'
  status: 'ready' | 'bounded' | 'partial' | 'unavailable'
  source_sha256: string
  map_grid: [number, number] | null
  selection: 'retained_component_masks_no_refilter'
  authorship_confidence_contribution: 0
  members: { id: 'sensitive' | 'complementary'; label: string
    status: 'measured' | 'withheld' | 'missing' | 'failed' | 'skipped' | 'unavailable'
    packet_ids: string[]; signal_refs?: { id: string; display_id: string | null }[]
    selected_components?: number; eligible_components?: number; omitted_components?: number }[]
  counts: { sensitive: number; complementary: number; wider: number; shared: number } | null
  artifacts: Partial<Record<'sensitive_mask' | 'complementary_mask' | 'wider_mask' | 'shared_mask' | 'wider_overlay' | 'shared_overlay', EvidenceImage>>
  interpretation: string
  limitations: string[]
}

export async function getImageEvidence(analysisId: string): Promise<ImageEvidenceSet> {
  const response = await fetch(`/api/v2/analyses/${encodeURIComponent(analysisId)}/evidence-packets`, { headers: authHeaders() })
  const body: unknown = await response.json().catch(() => null)
  return unwrap(Promise.resolve({
    response,
    data: response.ok ? body as ImageEvidenceSet : undefined,
    error: response.ok ? undefined : body,
  }))
}

export function useImageEvidence(analysisId: string, enabled = true) {
  return useQuery({
    queryKey: ['analysis', analysisId, 'evidence-packets'],
    queryFn: () => getImageEvidence(analysisId),
    enabled,
    // A display-only backfill can publish a new immutable set without changing the analysis revision.
    staleTime: 30_000,
    retry: false,
  })
}

/** Build our own same-origin URL, so an artifact cannot send its media token to an unrelated host. */
export function evidenceImageUrl(set: ImageEvidenceSet, artifact: EvidenceImage, token = ''): string | null {
  if (!set.id || !/^[a-f0-9]{64}$/.test(artifact.sha256)) return null
  const path = `/api/v2/analyses/${encodeURIComponent(set.analysis_id)}/evidence-packets/${encodeURIComponent(set.id)}/images/${artifact.sha256}.png`
  return token ? `${path}?token=${encodeURIComponent(token)}` : path
}

/** Frame the exact source coordinates, with lenses switched off so the original stays visible. */
export function evidenceSourceUrl(analysisId: string, packet: Pick<ImageEvidencePacket, 'region'>): string {
  const { x, y, w, h } = packet.region
  const query = new URLSearchParams({ layers: '', focus: [x, y, w, h].join(',') })
  return `/analyses/${encodeURIComponent(analysisId)}/evidence?${query.toString()}`
}

export function measurementValue(value: number): string {
  return new Intl.NumberFormat('en', { maximumFractionDigits: 3 }).format(value)
}
