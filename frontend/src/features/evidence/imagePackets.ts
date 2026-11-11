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
  map_grid?: [number, number]
  excluded_area_fraction?: number
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

export interface ImageEvidenceSet {
  schema: 'chipmunk.image-evidence/1'
  status: 'ready' | 'not_generated' | 'not_applicable'
  id: string | null
  version: 'image-evidence-1' | 'image-evidence-2' | 'image-evidence-3' | 'image-evidence-4' | 'image-evidence-5' | 'image-evidence-6' | 'image-evidence-7' | 'image-evidence-8'
  sha256: string | null
  analysis_id: string
  revision: number
  original_result_hash: string | null
  source: { sha256: string; width: number | null; height: number | null; coordinate_space: 'exif-oriented-original' }
  packets: ImageEvidencePacket[]
  limitations: string[]
  omitted_regions: number
  coverage?: { inspection_checks?: ImageInspectionCheck[] }
  comparison?: ImageRegionalComparison
  copy_similarity?: {
    schema: 'same-image-reflection-similarity/1'
    status: 'supported_pair' | 'no_pair' | 'unsupported_geometry' | 'unassessed_match_budget'
    interpretation: string
    pairs: { id: string; transform: 'horizontal_reflection'
      regions: Record<'area_a' | 'area_b', Box>
      artifacts: Record<'area_a' | 'area_b', EvidenceImage> }[]
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
