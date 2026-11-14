import { useQuery } from '@tanstack/react-query'
import { authHeaders, unwrap } from '../../lib/api/client'

export type EvidenceKind = 'video' | 'audio' | 'document'
export interface EvidenceAsset {
  sha256: string; media_type: 'image/png' | 'audio/wav' | 'video/mp4'; url: string
  width?: number; height?: number; samples?: number; sample_rate?: number; channels?: number
  frame_count?: number; time_base?: string; pts?: number[]; durations?: number[]; duration_ticks?: number; duration_s?: number
  codec?: string; has_audio?: boolean
  role?: string; page?: number; source_bbox?: number[]; time_axis?: 'source_pts'
}
export interface EvidenceRef { id: string; display_id: string | null }
export interface EvidenceMeasure { label: string; value: number | string; unit: string }
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
  interval_role?: 'before' | 'during' | 'after'
  anchor: { frame_index: number; pts: number; time_base: string; stream_time_s: number; playback_time_s: number | null }
  artifact_refs: { original: string; context: string; crop?: string }
  context_region: { x: number; y: number; w: number; h: number } | null
}
export interface VideoEvidencePacket extends PacketBase {
  claim_scope: 'sampled_frames' | 'sampled_window' | 'consecutive_frames' | 'paired_frame_sequences'; measurements: EvidenceMeasure[]; frames: VideoEvidenceFrame[]
  picture_hold?: { start_frame_index: number; end_frame_index: number; start_pts: number; end_pts: number; time_base: string;
    start_stream_s: number; end_stream_s: number; start_playback_s: number | null; end_playback_s: number | null;
    span_s: number; frame_count: number; decoded_rgb_sha256: string; pts: number[] }
  mouth_local_hold?: { frames: [number, number]; box: [number, number, number, number] }
  scene_return?: { schema: 'picture-change-return/1' | 'picture-change-return/2'; start_stream_s: number; end_stream_s: number; tiles: number[] }
  picture_sequence?: { first: PictureSequenceSpan; second: PictureSequenceSpan; frame_count: number; span_s: number }
  face_interval?: { start_stream_s: number; end_stream_s: number; lift: number;
    samples: Record<'before' | 'during' | 'after', { frame_index: number; pts: number; score: number }[]> }
  face_sequence?: { start_stream_s: number; end_stream_s: number; response: number;
    centers: Record<'before' | 'during' | 'after', number[]>;
    frame_inputs: { frame_index: number; pts: number; decoded_rgb_sha256: string; aligned_rgb_sha256: string }[] }
  face_boundary?: { method?: 'face-appearance-paired-edge/1' | 'face-appearance-boundary/1'; start_stream_s: number; end_stream_s: number; frame_inputs: { frame_index: number }[] }
  face_boundary_frames?: VideoEvidenceFrame[]
  face_spatial_cut?: { start_stream_s: number; end_stream_s: number; shared_tiles: number; sectors: number; frame_inputs: { frame_index: number }[] }
  face_spatial_cut_frames?: VideoEvidenceFrame[]
  face_return?: { method?: string; start_stream_s: number; end_stream_s: number;
    window_start_frame_index?: number;
    frame_inputs: { frame_index: number }[]; scene_frame_inputs?: { frame_index: number }[];
    edges?: Record<'entry' | 'exit', { frame_indices: number[] }> }
  whole_face_review?: { flagged: boolean; time_base: string;
    samples: { frame_index: number; pts: number; responses_hex: [string | null, string | null] }[] }
  clip?: { status: 'unavailable'; reason: string } | { status: 'ready'; artifact_ref: string;
    start_frame_index: number; end_frame_index_exclusive: number; source_pts: number[]; clip_pts: number[]; clip_time_base: string;
    start_stream_s: number; last_stream_s: number; start_playback_s: number | null;
    /** Packet-selected frames/window inputs inside the clip, not all checks' input frames. */
    sampled_frame_indices: number[];
    last_frame_duration: { basis: string; ticks: number; time_base: string }; limitations: string[] }
  window?: { frame_indices: number[]; start_pts: number; end_pts: number; time_base: string;
    start_stream_s: number; end_stream_s: number; start_playback_s: number | null; end_playback_s: number | null }
}
type PictureSequenceSpan = { start_frame_index: number; end_frame_index: number; start_stream_s: number; end_stream_s: number }
export interface VideoEvidenceSet extends SetBase {
  schema: 'chipmunk.video-evidence/1'; packets: VideoEvidencePacket[]; omitted_candidates: number
  source: { sha256: string; time_base?: string; width?: number; height?: number; timeline_origin_s?: number | null }
  picture_hold?: { status: 'matched' | 'no_match' | 'unavailable'; frames_scanned: number; frame_limit: number;
    last_frame_index: number | null; last_stream_time_s: number | null; last_pts: number | null; time_base: string;
    reason: string | null; packet_id: string | null; scan_scope?: 'extended-native-prefix/1' | 'native-sequence-prefix/1'; continuity_breaks?: number;
    sequence_review?: { status: 'matched' | 'no_match' | 'unavailable' }
    mouth_local_hold?: { status: 'matched' | 'no_match' | 'no_stable_face' | 'unavailable'; frames_scanned: number;
      packet_id: 'VPH02' | null; affects_overall_confidence: false; assessed_window_starts?: number[] }
    motion_sequence_review?: { status: 'matched' | 'no_match' | 'unavailable'; frames_scanned: number; parameters: { frame_rate: 20 | 25 }
      reason: 'source_clock' | 'exact_sequence' | null } }
  coverage?: { frame_count: number; scored_frame_count: number; window_frame_indices: number[]; window_check_count: number;
    decoded_frame_indices: number[]; table_truncated: boolean; unmapped_observations: number;
    face_detail_repeat?: { status: 'withheld' | 'retained' | 'unassessed'; observation: string;
      start_pts: number; end_pts: number; time_base: string; start_stream_s: number; end_stream_s: number;
      affects_overall_confidence: false };
    brief_face_review?: { observations: string[]; affects_overall_confidence: false };
    brief_face_windows?: { observations: string[]; affects_overall_confidence: false };
    scene_return_review?: { status: 'matched' | 'no_match' | 'unavailable'; frames_scanned: number;
      reason: 'unsupported_video' | 'profile' | 'provenance' | 'scan_unavailable' | 'incomplete_decode' | 'missing_anchors' | null;
      packet_id: 'VSC01' | null; observation: string; limitation: string; affects_overall_confidence: false };
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
  id: string; category: 'energy_change' | 'near_silence' | 'clipping' | 'spectral_transition' | 'upper_frequency_drop' | 'digital_silence'
  interval: SampleInterval; context_interval: SampleInterval; start_s: number; end_s: number
  source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  observation: string; interpretation: string; limitations: string[]; observations: EvidenceMeasure[]
  artifacts: Record<'selected_clip' | 'context_clip' | 'waveform' | 'spectrogram', EvidenceAsset>
  related_voice_packet_ids: string[]
}
export interface AcousticContext {
  schema: 'chipmunk.acoustic-context/1' | 'chipmunk.acoustic-context/2' | 'chipmunk.acoustic-context/3' | 'chipmunk.acoustic-context/4' | 'chipmunk.acoustic-context/5'; status: 'ready'
  source: { sha256: string; pcm_sha256: string; sample_rate: number; samples: number; coordinate_space: 'analysed-mono-pcm' }
  method: { version: string; frame_samples: number; description: string; thresholds: Record<string, number> }
  summary?: { measured_samples: number; frame_count: number; low_energy_reference_dbfs: number; peak_dbfs: number;
    clipped_samples: number; near_silence_samples: number }
  overview: Record<'energy' | 'spectrogram', EvidenceAsset>
  bins: (SampleInterval & { start_s: number; end_s: number; rms_dbfs: number; peak_dbfs: number;
    clipped_fraction: number; low_band_fraction: number; high_band_fraction: number; spectral_flatness: number })[]
  events: AcousticContextEvent[]; limitations: string[]; omitted_events?: number; omitted_continuity_windows?: number
  upper_frequency_drop?: { assessed_ranges: number; status: 'unassessed' | 'no_supported_drop' | 'measured_drop' }
  digital_silence?: { checked_samples: number; qualifying_intervals: number; selected: SampleInterval[]; affects_overall_confidence: false }
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
export interface StereoChannelContext {
  schema: 'chipmunk.stereo-channel-context/1' | 'chipmunk.stereo-channel-context/2'
  evidence_type: 'context'; affects_overall_confidence: false
  status: 'measured' | 'quiet' | 'mono' | 'unsupported' | 'too_short' | 'unavailable'
  source_sha256: string; source_stream_index: number; source_channel_count: number
  sample_rate: number; max_seconds: number; quiet_floor_dbfs: number; strong_level_gap_db: number; silence_floor_dbfs: number
  mono_pcm_sha256?: string; strong_downmix_attenuation_db?: number
  channel_silence?: {
    schema: 'chipmunk.stereo-channel-silence/1'; affects_overall_confidence: false
    checked_samples: number; qualifying_intervals: number
    selected: (SampleInterval & {
      silent_channel: 'left' | 'right'; flank_rms_dbfs: [number, number]; minimum_other_rms_dbfs: number
      source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
    })[]
  }
  windows: (SampleInterval & {
    status: 'strong_imbalance' | 'measured' | 'quiet'
    left_rms_dbfs: number; right_rms_dbfs: number; level_gap_db: number | null; correlation: number | null
    louder_channel: 'left' | 'right' | null
    downmix?: { mono_rms_dbfs: number; attenuation_db: number | null; strong_attenuation: boolean }
    source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  })[]
}
interface StereoListeningContext {
  schema: 'chipmunk.stereo-listening/1'; affects_overall_confidence: false
  windows: { stereo_window_index: number; artifacts: Record<'left' | 'right' | 'mono' | 'waveform', EvidenceAsset> }[]
  channel_silence?: { artifacts: Record<'left' | 'right' | 'mono' | 'waveform', EvidenceAsset> }[]
}
type AudioRepeatSpan = SampleInterval & { source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'] }
interface AudioRepeatPair {
  id: string; first: AudioRepeatSpan; second: AudioRepeatSpan; sample_count: number; lag_samples: number
  artifacts: Record<'first_clip' | 'second_clip', EvidenceAsset> & { waveform?: EvidenceAsset }
}
interface AudioRepeatRecord {
  status: 'measured' | 'unavailable' | 'ambiguous'; reason: string | null
  source_sha256: string; pcm_sha256: string; sample_rate: number; input_samples: number; searched_interval: SampleInterval | null
}
export interface AudioRepetitionContext extends AudioRepeatRecord {
  schema: 'chipmunk.audio-exact-repetition/1'; pairs: (AudioRepeatPair & { pcm_s16le_sha256: string })[]
}
interface StereoRepetitionContext {
  schema: 'chipmunk.stereo-exact-repetition/1'; source_sha256: string; affects_overall_confidence: false
  channels: { channel: 'left' | 'right'; measurement: AudioRepetitionContext; gain_measurement?: AudioGainRepetitionContext }[]
}
export function repeatedAudioCount(audio?: AudioEvidenceSet | VideoAudioChannel | null) {
  return (audio?.repetition_context?.pairs.length ?? 0) + (audio?.similarity_context?.pairs.length ?? 0)
    + (audio?.stereo_repetition?.channels.reduce((count, row) => count + row.measurement.pairs.length
      + (row.gain_measurement?.pairs.length ?? 0), 0) ?? 0)
}
export interface AudioGainRepetitionContext extends Omit<AudioRepeatRecord, 'searched_interval'> {
  schema: 'chipmunk.audio-gain-repetition/1'
  pairs: (AudioRepeatPair & { gain_numerator: number; gain_denominator: number; scaled_occurrence: 'first' | 'second';
    artifacts: Record<'first_clip' | 'second_clip' | 'first_waveform' | 'second_waveform', EvidenceAsset> })[]
}
export interface AudioSimilarityContext extends AudioRepeatRecord {
  schema: 'chipmunk.audio-repeat-similarity/1' | 'chipmunk.audio-repeat-similarity/2'
  frequency_band_search?: Pick<AudioRepeatRecord, 'status' | 'reason'>
  pairs: (AudioRepeatPair & {
    first: AudioRepeatSpan & { pcm_s16le_sha256: string }; second: AudioRepeatSpan & { pcm_s16le_sha256: string };
    support_windows: number })[]
}
type AudioClockSpan = SampleInterval & { source_start_pts: number; source_end_pts: number }
export interface AudioSourceClockContext {
  recording_voice_estimate_withheld?: true;
  affects_overall_confidence: false; time_base: string; minimum_jump_samples: number; checked_samples: number
  total_discontinuities: number; scored_windows_crossing: number
  breaks: { boundary_sample: number; jump_samples: number; before: AudioClockSpan; after: AudioClockSpan;
    scored_windows_crossing: number }[]
}
export interface ShortVoiceReview {
  schema: 'chipmunk.short-voice-review/1' | 'chipmunk.short-voice-review/2'; status: 'flagged' | 'no_flag' | 'skipped'
  title: string; reason: string; limitation: string; affects_overall_confidence: false
  checked_windows: SampleInterval[]; interval: SampleInterval | null; start_s: number | null; end_s: number | null
  window_coverage?: (SampleInterval & { speech_samples: number; status: 'checked' | 'below_speech_minimum';
    source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'] })[]
  source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  artifacts: Partial<Record<'selected_clip' | 'waveform', EvidenceAsset>>
}
export interface LongVoiceReview {
  schema: 'chipmunk.long-voice-review/1' | 'chipmunk.long-voice-review/2'; title: 'Long speech-window review'
  status: 'corroborated' | 'no_corroboration' | 'skipped'
  reason: string; limitation: string; affects_overall_confidence: false
  windows: (SampleInterval & {
    pcm_sha256: string; direction: 'ai' | 'original' | 'inconclusive' | 'unassessed'
    source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
    artifacts: Partial<Record<'selected_clip' | 'waveform', EvidenceAsset>>
  })[]
  regions?: (SampleInterval & {
    basis: 'overlap' | 'sparse'; supporting_windows: number[]; pcm_sha256: string
    source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
    artifacts: Record<'selected_clip' | 'waveform', EvidenceAsset>
  })[]
}
export interface BackgroundVoiceReview {
  schema: 'chipmunk.background-reduced-voice-review/1'
  title: string; status: 'flagged' | 'no_flag' | 'unassessed' | 'skipped'
  reason: string; limitation: string; affects_overall_confidence: false
  interval: SampleInterval
  source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  processed_pcm_sha256?: string
  artifacts: Partial<Record<'selected_clip' | 'processed_clip' | 'waveform', EvidenceAsset>>
}
export interface AudioEvidenceSet extends SetBase {
  schema: 'chipmunk.audio-evidence/1'; packets: AudioEvidencePacket[]; omitted_windows: number
  source: { sha256: string; pcm_sha256?: string; samples?: number; sample_rate?: number; source_channels?: number; source_sample_rate?: number;
    source_duration_s?: number | null; source_stream_index?: number; audio_stream_indices?: number[]; transform?: string; timeline_time_base?: string;
    timeline?: (SampleInterval & { source_pts: number })[] }
  coverage?: { decoded_samples: number; scored_samples: number; unscored_samples: number; decode_limit_reached: boolean;
    selection_threshold: number; windows: (SampleInterval & { score: number; signal_refs: EvidenceRef[] })[];
    source_clock_context?: AudioSourceClockContext;
    speech_window_selection?: { schema: 'audio-speech-window-selection/1'; minimum_speech_fraction: 0.5;
      affects_overall_confidence: false; raw_output_sha256: string;
      counts: Record<'scored' | 'below_speech_minimum' | 'outside_window_limit', number>;
      windows: (SampleInterval & { speech_samples: number; status: 'scored' | 'below_speech_minimum' | 'outside_window_limit';
        source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'] })[] };
    unscored_intervals?: (SampleInterval & { source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'] })[];
    stereo_attenuation_overlaps?: (SampleInterval & { source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'];
      unscored_interval_index: number; stereo_window_index: number })[] }
  overview?: Record<string, EvidenceAsset>
  acoustic_context?: AcousticContext
  sound_events?: SoundEventContext
  stereo_context?: StereoChannelContext
  stereo_listening?: StereoListeningContext
  repetition_context?: AudioRepetitionContext
  stereo_repetition?: StereoRepetitionContext
  similarity_context?: AudioSimilarityContext
  short_voice_review?: ShortVoiceReview
  long_voice_review?: LongVoiceReview
  background_reduced_voice_review?: BackgroundVoiceReview
}
export type VideoChannelStatus = 'ready' | 'absent' | 'unavailable' | 'not_assessed' | 'not_applicable'
export interface VideoAudioChannel {
  status: VideoChannelStatus; reason: string | null; source: AudioEvidenceSet['source']
  packets: AudioEvidencePacket[]; omitted_windows?: number; limitations: string[]
  coverage?: AudioEvidenceSet['coverage'] | null; overview?: Record<string, EvidenceAsset>
  acoustic_context?: AcousticContext
  sound_events?: SoundEventContext
  stereo_context?: StereoChannelContext
  stereo_listening?: StereoListeningContext
  repetition_context?: AudioRepetitionContext
  stereo_repetition?: StereoRepetitionContext
  similarity_context?: AudioSimilarityContext
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
    unsupported_checks: { status: string; reason: string }[]
    timing_windows?: { windows: { status: VideoTimingStatus; speech_ticks: number
      interval: VideoAVPacket['anchor']['source_timestamp_ranges'][number] }[] }
  } | null
}
export type VideoTimingStatus = 'measured' | 'insufficient_speech' | 'missing_picture_timestamps'
  | 'insufficient_audio_context' | 'no_face_track' | 'missing_landmarks' | 'track_limit'
  | 'missing_mouth_input' | 'unchanged_mouth_pixels' | 'flat_mouth_proxy' | 'speaker_unresolved'
export interface TextSpan { start: number; end: number; text: string }
export interface DocumentAnchor {
  kind: string; page?: number; paragraph?: number; mime_part?: string; quote_role?: string; extraction?: string
  paragraph_index?: number; body_path?: (string | number)[]
  selected_start?: number; selected_end?: number; highlight_boxes?: number[][]
  ocr_word_confidence?: number | number[]
}
export interface DocumentOriginalTextLocation {
  anchor_index: number
  coordinate: 'decoded_text' | 'word_paragraph' | 'decoded_email' | 'decoded_email_html'
  ranges: [number, number][]
}
export interface DocumentEvidencePacket extends PacketBase {
  focus: TextSpan; context: TextSpan; anchors: DocumentAnchor[]; measurements: EvidenceMeasure[]
  supporting_windows: DocumentEvidenceWindow[]
  inspection_scope?: { basis: 'scored_context' | 'ratio_reduction'; spans: Pick<TextSpan, 'start' | 'end'>[] }
  context_comparison?: DocumentContextComparison
  matched_context_review?: DocumentMatchedContextReview
}
interface DocumentMatchedContextReview {
  schema: 'document-matched-context-review/1'; signal_ref: EvidenceRef; authorship_confidence_contribution: 0
}
export interface DocumentComparisonInput {
  spans: { start: number; end: number }[]; text: string; input_sha256: string; score: number
  token_coverage: Record<string, number>
}
export interface DocumentContextComparison {
  schema: 'chipmunk.document-context-comparison/1'; status: 'recorded' | 'not_assessed'; evidence_type: 'context'
  authorship_confidence_pct: null; affects_overall_confidence: false; focus: { start: number; end: number }
  note: string; reason: string | null; omitted_checks: number
  isolated_paragraph?: { signal_ref: EvidenceRef; context: DocumentComparisonInput; paragraph: DocumentComparisonInput
    supports_inspection: boolean; note: string }
  checks: { signal_ref: EvidenceRef; score_kind: 'classifier_probability'; distinct_inputs: number
    lowest_response: number; highest_response: number; response_spread: number; omitted_inputs: number
    inputs: DocumentComparisonInput[] }[]
}
export interface DocumentEvidenceWindow {
  signal_ref: EvidenceRef; spans: { start: number; end: number }[]; text: string; input_sha256: string; score: number
  score_kind?: 'classifier_probability' | 'perplexity_ratio'; is_calibrated?: false; threshold?: number; ai_when?: 'lt' | 'gte'
  direction?: 'supports_ai' | 'opposes_ai'; direction_semantics?: string; input_semantics?: string
  matched_context_selected?: boolean; score_exact_hex?: string
  token_coverage: { tokens?: number; token_windows?: number; [key: string]: unknown }
  scored_spans?: { start: number; end: number }[]; tokens_scored?: number; neighbor_expansion?: boolean
  inference_contexts?: { id: string; source_token_range: [number, number]; scored_token_range: [number, number];
    spans: { start: number; end: number }[]; text: string; input_sha256: string; token_ids_sha256: string;
    anchor_token_count: number; context_limit: number; stride: number }[]
}
export type DocumentCoverageState = 'response_range' | 'context_only' | 'no_local_record'
export interface DocumentPassageCoverage {
  schema: 'document-local-passage-coverage/1' | 'document-local-passage-coverage/2'
  count_unit: 'non_whitespace_unicode_codepoint'
  mapped_characters: number
  counts: Record<DocumentCoverageState, number>
  ranges: { start: number; end: number; state: DocumentCoverageState; characters: number;
    source_block: number; page?: number; paragraph?: number; mime_part?: string; quote_role?: string }[]
  omitted_ranges: number
  pdf_pages?: {
    schema: 'document-pdf-page-coverage/1'
    page_count: number
    mapped_pages: { page: number; extraction: 'text_layer' | 'ocr' | 'mixed'; unchecked_image_areas: boolean;
      counts: Record<DocumentCoverageState, number> }[]
    unmapped_page_ranges: [number, number][]
  }
}
export interface DocumentUnselectedResponses {
  schema: 'document-unselected-passage-responses/1'; affects_overall_confidence: false
  eligible_windows: number; omitted_windows: number
  rows: { id: string; focus: { start: number; end: number }; spans: { start: number; end: number }[]
    sources: ({ kind: 'pdf'; page: number; extraction: 'text_layer' | 'ocr' }
      | { kind: 'email'; mime_part: string; quote_role: string }
      | { kind: 'text' | 'docx'; paragraph: number })[]
    check: string; input_sha256: string; direction: 'supports_ai' | 'opposes_ai'; response: number; text: string }[]
}
interface DocumentPositionSpan { start: number; end: number; source_start: number; source_end: number }
interface DocumentPositionReading { input_sha256: string; response_hex: string; focus_tokens: number; neighbor_tokens: number }
export type DocumentPassagePosition = {
  authorship_confidence_contribution: 0; signal_ref: EvidenceRef } & ({
  schema: 'document-passage-position-review/1' | 'document-passage-position-review/3';
  status: 'flagged' | 'no_flag' | 'unassessed';
  focus_text: string; sentences: DocumentPositionSpan[];
  source_locator?: { kind: 'docx'; paragraph_index: number; body_path: [number] } | { kind: 'email'; mime_part: string };
  readings?: DocumentPositionReading[]; contrast_hex?: string
} | {
  schema: 'document-passage-position-review/2' | 'document-passage-position-review/4';
  status: 'flagged' | 'no_flag' | 'unassessed' | 'partial'; paragraphs: number; sentence_count: number;
  contexts: { paragraph: number; sentences: DocumentPositionSpan[]; readings?: DocumentPositionReading[] }[];
  selected: number | null; focus_text?: string
})
export interface EmailClassifierInputScope {
  schema: 'document-email-classifier-input-scope/1'; mime_part: string; mapping: 'mapped' | 'unmapped'
  checks: { signal_ref: EvidenceRef; counts: Record<'body' | 'quoted' | 'signature' | 'mixed' | 'unmapped', number>;
    windows: { spans: [number, number][]; input_sha256: string; role: 'body' | 'quoted' | 'signature' | 'mixed' | 'unmapped' }[];
    unlisted: number }[]
}
type PercentageSource = { kind: 'pdf'; page: number; block: number; bbox: number[]; coordinate_space: 'unrotated-page-points' }
  | { kind: 'docx'; paragraph_index: number; body_path: [number]; body_paragraph: number }
  | { kind: 'email'; mime_part: string; source_start: number; source_end: number }
export interface DocumentEvidenceSet extends SetBase {
  schema: 'chipmunk.document-evidence/1'; packets: DocumentEvidencePacket[]; omitted_windows: number
  source: { sha256: string; chars?: number; method?: string; offset_unit: 'unicode_codepoint' }
  percentage_consistency?: { schema: 'document-percentage-consistency/1' | 'document-percentage-consistency/2' | 'document-percentage-consistency/3'; status: 'different_values' | 'checked' | 'not_assessed';
    scope: string; limitation: string; authorship_confidence_contribution: 0; checked_sentences: number;
    excluded_blocks: number; different_claims: number; comparable_claims: number;
    groups: { start: number; end: number; text: string; quantity_start: number; quantity_end: number; anchor?: PercentageSource;
      value_location?: { boxes: number[][] } | { ranges: [number, number][] } }[][] }
  repeated_passages?: { schema: 'document-repeated-passages/1'; status: 'matched' | 'no_match' | 'not_assessed';
    coverage: { compared_blocks: number; excluded_ocr_blocks: number; excluded_quoted_or_signature_blocks: number; omitted_groups: number };
    groups: { id: string; text: string; text_sha256: string; word_count: number; total_occurrences: number;
      occurrences: { start: number; end: number; anchor: { kind: 'text' | 'email' | 'docx' | 'pdf'; source_start?: number;
        source_end?: number; mime_part?: string; paragraph_index?: number; page?: number; block?: number; bbox?: number[] } }[] }[] }
  coverage?: { scored_windows?: number; candidate_spans?: number; unmapped_pages?: number[];
    word_tracked_changes?: { status: 'found' | 'partial' | 'no_match'; scope: string; limitation: string;
      part: string; paragraphs: number; inserted: number; deleted: number; unsupported_marks: number;
      omitted_excerpts: number; rows: { kind: 'inserted' | 'deleted'; paragraph: number; body_index: number;
        inline_index: number; text: string; characters: number }[]; authorship_confidence_contribution: 0 };
    email_body_alternatives?: EmailBodyAlternatives;
    email_classifier_input_scope?: EmailClassifierInputScope;
    local_passage_coverage?: DocumentPassageCoverage;
    unselected_passage_responses?: DocumentUnselectedResponses;
    passage_position_review?: DocumentPassagePosition;
    original_text_locations?: { schema: 'document-original-text-locations/1'; packets: Record<string, DocumentOriginalTextLocation[]> };
    writing_second_check?: { schema: 'document-writing-second-check/1' | 'document-writing-second-check/2' | 'document-writing-second-check/3' | 'document-writing-second-check/4' | 'document-writing-second-check/5' | 'document-writing-second-check/6'; authorship_confidence_contribution: 0;
      body_spans?: { start: number; end: number }[];
      pdf_spans?: { start: number; end: number; page: number }[];
      ranges: { start: number; end: number; spans: { start: number; end: number }[];
        state: 'corroborated' | 'different' | 'unavailable' }[] };
    matched_context_review?: DocumentMatchedContextReview & { eligible_sentences: number; selected_sentences: number };
    isolated_paragraph_priority?: { schema: 'document-isolated-paragraph-priority/1'; packet_id: string;
      signal_ref: EvidenceRef; authorship_confidence_contribution: 0 };
    partially_checked_pages?: { page: number; image_boxes: [number, number, number, number][];
      coordinate_space: 'unrotated-page-points'; reason: string }[];
    pdf_text_visibility?: { schema: 'document-pdf-text-visibility/1' | 'document-pdf-text-visibility/2' | 'document-pdf-text-visibility/3';
      total_pages: number; examined_pages: number[];
      locations: { page: number; invisible_spans: number; boxes: [number, number, number, number][] }[];
      replacement_examined_pages?: number[];
      replacement_differences?: { page: number; boxes: [number, number, number, number][]; locations_incomplete: boolean }[];
      coordinate_space: 'unrotated-page-points'; authorship_confidence_contribution: 0 };
    word_text_visibility?: { schema: 'document-word-text-visibility/1'; part: string; checked_paragraphs: number; hidden_runs: number;
      locations: { body_path: (number | 'row' | 'cell')[]; body_paragraph: number | null; run: number; source_start: number; source_end: number }[];
      authorship_confidence_contribution: 0 };
    invisible_characters?: { schema: 'document-invisible-characters/1' | 'document-invisible-characters/2';
      coordinate_space: 'decoded_text' | 'word_paragraph' | 'decoded_email' | 'decoded_email_html';
      checked_blocks: number; checked_chars: number; matches: number; authorship_confidence_contribution: 0;
      locations: { text_offset: number; source_start: number; source_end: number; codepoint: string; looks_like?: string;
        body_path?: (number | 'row' | 'cell')[]; mime_part?: string }[] };
    checks?: { signal_ref: EvidenceRef; status: 'recorded' | 'partial' | 'context_not_recorded'; scored_windows: number; limitation?: string }[];
    ocr?: { status?: string; pages?: { page: number; status: string; coverage_scope?: 'image_region'; reason?: string | null; mean_recognition_confidence?: number;
      language?: { status?: string; language?: string; reason?: string }; [key: string]: unknown }[]; unchecked_pages?: number[] } | null
    email?: { body_part?: string; body_content_type?: 'text/plain' | 'text/html'; attachments?: unknown[]; [key: string]: unknown } | boolean | null }
}
export interface EmailBodyAlternatives {
  status: 'difference' | 'equivalent' | 'unsupported'
  parts?: Record<'plain' | 'html', { id: string }>
  differences?: ({ id: string } & Partial<Record<'plain' | 'html', {
    text: string; source_ranges: [number, number][]; excerpt_truncated: boolean }>>)[]
  omitted_differences?: number
}
export type MultimediaEvidenceSet = VideoEvidenceSet | AudioEvidenceSet | DocumentEvidenceSet

const nestedVideoVersions = new Set(['video-evidence-3', 'video-evidence-4', 'video-evidence-5', 'video-evidence-6', 'video-evidence-7', 'video-evidence-8', 'video-evidence-9', 'video-evidence-10', 'video-evidence-11', 'video-evidence-12', 'video-evidence-13', 'video-evidence-14', 'video-evidence-15', 'video-evidence-16', 'video-evidence-17', 'video-evidence-18', 'video-evidence-19', 'video-evidence-20', 'video-evidence-21'])
nestedVideoVersions.add('video-evidence-22')
nestedVideoVersions.add('video-evidence-23')
nestedVideoVersions.add('video-evidence-24')
nestedVideoVersions.add('video-evidence-25')
nestedVideoVersions.add('video-evidence-26')
nestedVideoVersions.add('video-evidence-27')
nestedVideoVersions.add('video-evidence-28')
nestedVideoVersions.add('video-evidence-29')
nestedVideoVersions.add('video-evidence-30')
nestedVideoVersions.add('video-evidence-31')
export function hasNestedVideoEvidence(version: string): boolean { return nestedVideoVersions.has(version) }

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
