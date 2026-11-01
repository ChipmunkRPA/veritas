import { useQuery } from '@tanstack/react-query'
import { authHeaders, unwrap } from '../../lib/api/client'

export type EvidenceKind = 'video' | 'audio' | 'document'
export interface EvidenceAsset {
  sha256: string; media_type: 'image/png' | 'audio/wav' | 'video/mp4'; url: string
  width?: number; height?: number; samples?: number; sample_rate?: number; channels?: number
  frame_count?: number; time_base?: string; pts?: number[]; durations?: number[]; duration_ticks?: number; duration_s?: number
  codec?: string; has_audio?: boolean
  role?: string; page?: number; source_bbox?: number[]
}
export interface EvidenceRef { id: string; display_id: string | null }
export interface EvidenceMeasure { label: string; value: number; unit: string }
export interface PacketBase {
  id: string; title: string; observation: string; interpretation: string; limitations: string[]
  finding_ids: string[]; signal_refs: EvidenceRef[]; artifacts: Record<string, EvidenceAsset>
}
interface SetBase {
  status: 'ready' | 'not_generated' | 'not_applicable' | 'unavailable'; id: string | null; version: string
  sha256: string | null; analysis_id: string; revision: number; original_result_hash: string | null; limitations: string[]
}
export interface VideoEvidenceFrame {
  id: string; score: number | null; is_peak: boolean; region_meaning: string
  anchor: { frame_index: number; pts: number; time_base: string; stream_time_s: number; playback_time_s: number | null }
  artifact_refs: { original: string; context: string; crop?: string }
  context_region: { x: number; y: number; w: number; h: number } | null
}
export interface VideoEvidencePacket extends PacketBase {
  claim_scope: 'sampled_frames' | 'sampled_window'; measurements: EvidenceMeasure[]; frames: VideoEvidenceFrame[]
  clip?: { status: 'unavailable'; reason: string } | { status: 'ready'; artifact_ref: string;
    start_frame_index: number; end_frame_index_exclusive: number; source_pts: number[]; clip_pts: number[]; clip_time_base: string;
    start_stream_s: number; last_stream_s: number; start_playback_s: number | null;
    /** Packet-selected frames/window inputs inside the clip, not all checks' input frames. */
    sampled_frame_indices: number[];
    last_frame_duration: { basis: string; ticks: number; time_base: string }; limitations: string[] }
  window?: { frame_indices: number[]; start_pts: number; end_pts: number; time_base: string;
    start_stream_s: number; end_stream_s: number; start_playback_s: number | null; end_playback_s: number | null }
}
export interface VideoEvidenceSet extends SetBase {
  schema: 'chipmunk.video-evidence/1'; packets: VideoEvidencePacket[]; omitted_candidates: number
  source: { sha256: string; time_base?: string; width?: number; height?: number; timeline_origin_s?: number | null }
  coverage?: { frame_count: number; scored_frame_count: number; window_frame_indices: number[]; window_check_count: number;
    decoded_frame_indices: number[]; table_truncated: boolean; unmapped_observations: number;
    channels?: Partial<Record<'audio' | 'av_sync' | 'av', { status: string; reason: string | null; successful_check_count?: number }>> } | null
  /** Present in video-evidence-3 and later. Both channels belong to this parent set. */
  audio?: VideoAudioChannel
  av?: VideoAVChannel
}
export interface SampleInterval { start_sample: number; end_sample: number }
export interface AudioEvidencePacket extends PacketBase {
  interval: SampleInterval; context_interval: SampleInterval; start_s: number; end_s: number
  observations: (EvidenceMeasure & { signal_refs: EvidenceRef[]; unit_ref: string })[]
  anchor: { sample_rate: number; source_timestamp_ranges: (SampleInterval & {
    source_start_pts: number; source_end_pts: number; time_base: string })[] }
}
export interface AcousticContextEvent {
  id: string; category: 'energy_change' | 'near_silence' | 'clipping'
  interval: SampleInterval; context_interval: SampleInterval; start_s: number; end_s: number
  source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  observation: string; interpretation: string; limitations: string[]; observations: EvidenceMeasure[]
  artifacts: Record<'selected_clip' | 'context_clip' | 'waveform' | 'spectrogram', EvidenceAsset>
  related_voice_packet_ids: string[]
}
export interface AcousticContext {
  schema: 'chipmunk.acoustic-context/1'; status: 'ready'
  source: { sha256: string; pcm_sha256: string; sample_rate: number; samples: number; coordinate_space: 'analysed-mono-pcm' }
  method: { version: string; frame_samples: number; description: string; thresholds: Record<string, number> }
  summary?: { measured_samples: number; frame_count: number; low_energy_reference_dbfs: number; peak_dbfs: number;
    clipped_samples: number; near_silence_samples: number }
  overview: Record<'energy' | 'spectrogram', EvidenceAsset>
  bins: (SampleInterval & { start_s: number; end_s: number; rms_dbfs: number; peak_dbfs: number;
    clipped_fraction: number; low_band_fraction: number; high_band_fraction: number; spectral_flatness: number })[]
  events: AcousticContextEvent[]; limitations: string[]; omitted_events?: number
}
export interface SoundEventContext {
  schema: 'chipmunk.sound-events/1'; status: 'ready'
  source: { sha256: string; pcm_sha256: string; sample_rate: number; samples: number }
  method: { window_samples: number; hop_samples: number; description: string }
  windows: (SampleInterval & { id: string; start_s: number; end_s: number; labels: { label: string; score: number }[];
    source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'];
    artifacts: Record<'selected_clip' | 'waveform', EvidenceAsset> })[]
  limitations: string[]
}
export interface AudioEvidenceSet extends SetBase {
  schema: 'chipmunk.audio-evidence/1'; packets: AudioEvidencePacket[]; omitted_windows: number
  source: { sha256: string; pcm_sha256?: string; samples?: number; sample_rate?: number; source_channels?: number; source_sample_rate?: number;
    source_duration_s?: number | null; transform?: string; timeline_time_base?: string;
    timeline?: (SampleInterval & { source_pts: number })[] }
  coverage?: { decoded_samples: number; scored_samples: number; unscored_samples: number; decode_limit_reached: boolean;
    selection_threshold: number; windows: (SampleInterval & { score: number; signal_refs: EvidenceRef[] })[] }
  overview?: Record<string, EvidenceAsset>
  acoustic_context?: AcousticContext
  sound_events?: SoundEventContext
}
export type VideoChannelStatus = 'ready' | 'absent' | 'unavailable' | 'not_assessed' | 'not_applicable'
export interface VideoAudioChannel {
  status: VideoChannelStatus; reason: string | null; source: AudioEvidenceSet['source']
  packets: AudioEvidencePacket[]; omitted_windows?: number; limitations: string[]
  coverage?: AudioEvidenceSet['coverage'] | null; overview?: Record<string, EvidenceAsset>
  acoustic_context?: AcousticContext
  sound_events?: SoundEventContext
}
export interface VideoAVFrameInput {
  tick: number; frame_index: number; frame_pts: number; source_audio_pts: number; source_audio_time_base: string
  decoded_rgb_sha256: string; mouth_patch_sha256: string; mouth_region: { x: number; y: number; w: number; h: number }
  mouth_value: number; artifact_ref: string
}
export interface VideoAVPacket extends PacketBase {
  claim_scope: 'sampled_av_window'; interval: SampleInterval; context_interval: SampleInterval
  anchor: AudioEvidencePacket['anchor']; measurements: EvidenceMeasure[]
  frames: (VideoEvidenceFrame & { av_tick: number; artifact_refs: VideoEvidenceFrame['artifact_refs'] & { measured_patch: string } })[]
  clip?: VideoEvidencePacket['clip']
  window: { start_tick: number; end_tick: number; frame_indices: number[]; frame_inputs: VideoAVFrameInput[] }
  measurement: { offset_ticks: number; offset_ms: number; peak_correlation: number; low_correlation: boolean
    lag_profile: { offset_ticks: number; correlation: number }[]
    audio_curve: { tick: number; sample_start: number; sample_end: number; rms: number; speech: boolean }[]
    overlaps_offset_step: boolean; overlaps_correlation_collapse: boolean
    baseline_offset_ticks: number | null; collapse_threshold: number | null; minimum_trusted_peak: number; reading: string }
}
export interface VideoAVChannel {
  status: VideoChannelStatus; reason: string | null; limitations: string[]; packets: VideoAVPacket[]
  source: AudioEvidenceSet['source'] & { tick_rate?: number; samples_per_tick?: number; video_time_base?: string }
  coverage?: { recorded_windows: number; selected_windows: number; omitted_windows: number; selection: string
    unsupported_checks: { status: string; reason: string }[] } | null
}
export interface TextSpan { start: number; end: number; text: string }
export interface DocumentAnchor {
  kind: string; page?: number; paragraph?: number; mime_part?: string; quote_role?: string; extraction?: string
  paragraph_index?: number; body_path?: (string | number)[]
  selected_start?: number; selected_end?: number; highlight_boxes?: number[][]
  ocr_word_confidence?: number | number[]
}
export interface DocumentEvidencePacket extends PacketBase {
  focus: TextSpan; context: TextSpan; anchors: DocumentAnchor[]; measurements: EvidenceMeasure[]
  supporting_windows: DocumentEvidenceWindow[]
}
export interface DocumentEvidenceWindow {
  signal_ref: EvidenceRef; spans: { start: number; end: number }[]; text: string; input_sha256: string; score: number
  score_kind?: 'classifier_probability' | 'perplexity_ratio'; is_calibrated?: false; threshold?: number; ai_when?: 'lt' | 'gte'
  direction?: 'supports_ai' | 'opposes_ai'; direction_semantics?: string; input_semantics?: string
  token_coverage: { tokens?: number; token_windows?: number; [key: string]: unknown }
  scored_spans?: { start: number; end: number }[]; tokens_scored?: number; neighbor_expansion?: boolean
  inference_contexts?: { id: string; source_token_range: [number, number]; scored_token_range: [number, number];
    spans: { start: number; end: number }[]; text: string; input_sha256: string; token_ids_sha256: string;
    anchor_token_count: number; context_limit: number; stride: number }[]
}
export interface DocumentEvidenceSet extends SetBase {
  schema: 'chipmunk.document-evidence/1'; packets: DocumentEvidencePacket[]; omitted_windows: number
  source: { sha256: string; chars?: number; method?: string; offset_unit: 'unicode_codepoint' }
  coverage?: { scored_windows?: number; candidate_spans?: number; unmapped_pages?: number[];
    checks?: { signal_ref: EvidenceRef; status: 'recorded' | 'partial' | 'context_not_recorded'; scored_windows: number; limitation?: string }[];
    ocr?: { status?: string; pages?: { page: number; status: string; reason?: string | null; mean_recognition_confidence?: number;
      language?: { status?: string; language?: string; reason?: string }; [key: string]: unknown }[]; unchecked_pages?: number[] } | null
    email?: Record<string, unknown> | boolean | null }
}
export type MultimediaEvidenceSet = VideoEvidenceSet | AudioEvidenceSet | DocumentEvidenceSet

export async function getMultimediaEvidence(analysisId: string, kind: EvidenceKind): Promise<MultimediaEvidenceSet> {
  const response = await fetch(`/api/v2/analyses/${encodeURIComponent(analysisId)}/evidence-packets`, { headers: authHeaders() })
  const body: unknown = await response.json().catch(() => null)
  const value = await unwrap(Promise.resolve({ response, data: response.ok ? body as MultimediaEvidenceSet : undefined,
    error: response.ok ? undefined : body }))
  if (value.schema !== `chipmunk.${kind}-evidence/1` || value.analysis_id !== analysisId || !Array.isArray(value.packets)) {
    throw new Error('The evidence does not match this analysis.')
  }
  return value
}
export function useMultimediaEvidence(analysisId: string, kind: EvidenceKind, enabled = true) {
  return useQuery({ queryKey: ['analysis', analysisId, 'evidence-packets'], queryFn: () => getMultimediaEvidence(analysisId, kind),
    enabled, staleTime: 30_000, retry: false })
}
/** Ignore backend URL text: a media token may only go to our own case-scoped asset endpoint. */
export function evidenceAssetUrl(set: Pick<SetBase, 'id' | 'analysis_id'>, asset: EvidenceAsset, token = ''): string | null {
  const extension = asset.media_type === 'image/png' ? 'png' : asset.media_type === 'audio/wav' ? 'wav' : asset.media_type === 'video/mp4' ? 'mp4' : null
  if (!set.id || !extension || !/^[a-f0-9]{64}$/.test(asset.sha256)) return null
  const path = `/api/v2/analyses/${encodeURIComponent(set.analysis_id)}/evidence-packets/${encodeURIComponent(set.id)}/assets/${asset.sha256}.${extension}`
  return token ? `${path}?token=${encodeURIComponent(token)}` : path
}
export function evidenceSourceMediaUrl(analysisId: string, token: string): string {
  const path = `/api/analyses/${encodeURIComponent(analysisId)}/source`
  return token ? `${path}?token=${encodeURIComponent(token)}` : path
}
export function seconds(value: number): string { return `${value.toFixed(3)} s` }
export function rationalSeconds(pts: number, timeBase: string): number | null {
  const match = /^(\d+)\/(\d+)$/.exec(timeBase)
  if (!match || Number(match[2]) <= 0) return null
  const value = pts * Number(match[1]) / Number(match[2])
  return Number.isFinite(value) ? value : null
}
