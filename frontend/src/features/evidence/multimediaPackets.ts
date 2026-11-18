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
  interval_role?: 'before' | 'during' | 'transition' | 'after'
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
  lip_timing_review?: {
    audio_source_pts: [number, number]
    picture_source_samples: [number, number]
    change_samples?: number
    silent_mouth?: { near_silent_peak_units: number; visible_mouth_frames: number }
    speech_activity_gap?: {
      activity_millionths: [[number, number], [number, number], [number, number]]
      visible_mouth_frames: number
      margin_millionths: number
      stereo_levels?: { source_samples: [number, number]; rms_dbfs: [number | null, number | null, number | null] }[]
    }
    articulation?: { mismatched_events: number; audio_sample: number; picture_frame: number; picture_pts: number }
    motion_sound_agreement?: { agreement_units: [number, number, number] }
  }
  scene_return?: { schema: 'picture-change-return/1' | 'picture-change-return/2'; start_stream_s: number; end_stream_s: number; tiles: number[] }
  face_single_boundary?: { method: 'single-face-boundary/1' | 'single-face-boundary/2'; boundary: number; pts: number; time_base: string;
    time_s: number; affects_overall_confidence: false }
  picture_sequence?: { first: PictureSequenceSpan; second: PictureSequenceSpan; frame_count: number; span_s: number; order?: 'reverse';
    frame_steps?: Record<'first' | 'second', 1 | 2 | 3 | 4>; spatial_transform?: 'horizontal_reflection';
    framing?: { wider_range: 'first' | 'second'; crop_box: [number, number, number, number]; alignment?: 'centered' | 'landmarks' } }
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
  whole_face_review?: { schema?: 'paired-whole-face-samples/1' | 'same-frame-face-samples/1' | 'relative-face-detail-samples/1';
    flagged: boolean; time_base: string;
    samples: { frame_index: number; pts: number; responses_hex?: [string | null, string | null] }[] }
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
  lip_timing_review?: {
    schema?: string; status: string; observation: string; affects_overall_confidence: false
    shot_coverage?: { frames: [number, number]; audio_samples: [number, number]; observation: string; exhibit: string | null }[]
  }
  source: { sha256: string; time_base?: string; width?: number; height?: number; timeline_origin_s?: number | null }
  picture_hold?: { status: 'matched' | 'no_match' | 'unavailable'; frames_scanned: number; frame_limit: number;
    last_frame_index: number | null; last_stream_time_s: number | null; last_pts: number | null; time_base: string;
    reason: string | null; packet_id: string | null; scan_scope?: 'extended-native-prefix/1' | 'native-sequence-prefix/1'; continuity_breaks?: number;
    sequence_review?: { status: 'matched' | 'no_match' | 'unavailable' }
    reverse_sequence_review?: { status: 'matched' | 'no_match' | 'unavailable' }
    reverse_motion_sequence_review?: { status: 'matched' | 'no_match' | 'unavailable'; frames_scanned: number
      reason: 'exact_sequence' | 'source_clock' | null }
    retimed_motion_sequence_review?: { status: 'matched' | 'no_match' | 'unavailable'; frames_scanned: number;
      parameters?: { playback_step_pairs?: [number, number][] } }
    reflected_motion_sequence_review?: { status: 'matched' | 'no_match' | 'unavailable'; frames_scanned: number }
    cropped_motion_sequence_review?: { status: 'matched' | 'no_match' | 'unavailable'; frames_scanned: number }
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
    face_single_boundary_coverage?: { status: 'localized' | 'no_match' | 'unavailable'; frames_scanned: number;
      method?: 'single-face-boundary/2';
      measured_face_frames: number; reason: 'provenance' | 'profile' | 'unsupported_video' | 'models_unavailable'
        | 'time_budget' | 'native_measurement_unavailable' | 'shot_change' | 'insufficient_face_context' | 'insufficient_native_boundary'
        | 'conflicting_boundary_context' | null };
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
  id: string; category: 'energy_change' | 'near_silence' | 'clipping' | 'spectral_transition' | 'upper_frequency_drop' | 'digital_silence' | 'tonal_phase'
  interval: SampleInterval; context_interval: SampleInterval; start_s: number; end_s: number
  source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  observation: string; interpretation: string; limitations: string[]; observations: EvidenceMeasure[]
  artifacts: Record<'selected_clip' | 'context_clip' | 'waveform' | 'spectrogram', EvidenceAsset>
  related_voice_packet_ids: string[]
}
export interface AcousticContext {
  schema: 'chipmunk.acoustic-context/1' | 'chipmunk.acoustic-context/2' | 'chipmunk.acoustic-context/3' | 'chipmunk.acoustic-context/4'
    | 'chipmunk.acoustic-context/5' | 'chipmunk.acoustic-context/6'; status: 'ready'
  source: { sha256: string; pcm_sha256: string; sample_rate: number; samples: number; coordinate_space: 'analysed-mono-pcm' }
  method: { version: string; frame_samples: number; description: string; thresholds: Record<string, number> }
  summary?: { measured_samples: number; frame_count: number; low_energy_reference_dbfs: number; peak_dbfs: number;
    clipped_samples: number; near_silence_samples: number }
  overview: Record<'energy' | 'spectrogram', EvidenceAsset>
  bins: (SampleInterval & { start_s: number; end_s: number; rms_dbfs: number; peak_dbfs: number;
    clipped_fraction: number; low_band_fraction: number; high_band_fraction: number; spectral_flatness: number })[]
  events: AcousticContextEvent[]; limitations: string[]; omitted_events?: number; omitted_continuity_windows?: number
  upper_frequency_drop?: { assessed_ranges: number; status: 'unassessed' | 'no_supported_drop' | 'measured_drop' }
  tonal_phase?: { comparisons: number; status: 'unassessed' | 'no_supported_change' | 'measured_change' }
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
  identical_channel_interval?: {
    schema: 'chipmunk.stereo-identical-interval/1'; affects_overall_confidence: false
    checked_samples: number; qualifying_intervals: number
    selected: (SampleInterval & {
      flank_difference_dbfs: [number, number]; minimum_signal_dbfs: number
      source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
    })[]
  }
  stereo_difference_reduction?: {
    schema: 'chipmunk.stereo-difference-reduction/1'; affects_overall_confidence: false
    status: 'unassessed' | 'no_qualifying_change' | 'measured_change'; checked_samples: number
    selected: (SampleInterval & {
      side_dbfs: [number, number, number]; common_dbfs: [number, number, number]
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
  schema: 'chipmunk.stereo-listening/1' | 'chipmunk.stereo-listening/2'; affects_overall_confidence: false
  windows: { stereo_window_index: number; artifacts: Record<'left' | 'right' | 'mono' | 'waveform', EvidenceAsset> & { difference?: EvidenceAsset } }[]
  channel_silence?: { artifacts: Record<'left' | 'right' | 'mono' | 'waveform', EvidenceAsset> }[]
  identical_channel_interval?: { artifacts: Record<'left' | 'right' | 'mono' | 'waveform', EvidenceAsset> }[]
  stereo_difference_reduction?: { artifacts: Record<'left' | 'right' | 'mono' | 'difference' | 'waveform', EvidenceAsset> }[]
}
type AudioRepeatSpan = SampleInterval & { source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'] }
interface AudioRepeatPair {
  id: string; first: AudioRepeatSpan; second: AudioRepeatSpan; sample_count: number; lag_samples: number
  artifacts: Record<'first_clip' | 'second_clip', EvidenceAsset>
    & Partial<Record<'waveform' | 'first_recorded_clip' | 'second_recorded_clip' | 'first_waveform' | 'second_waveform', EvidenceAsset>>
}
interface AudioRepeatRecord {
  status: 'measured' | 'unavailable' | 'ambiguous'; reason: string | null
  source_sha256: string; pcm_sha256: string; sample_rate: number; input_samples: number; searched_interval: SampleInterval | null
}
export interface AudioRepetitionContext extends AudioRepeatRecord {
  schema: 'chipmunk.audio-exact-repetition/1'; pairs: (AudioRepeatPair & { pcm_s16le_sha256: string })[]
}
interface StereoRepetitionContext {
  schema: 'chipmunk.stereo-exact-repetition/1' | 'chipmunk.stereo-exact-repetition/2' | 'chipmunk.stereo-exact-repetition/3'
    | 'chipmunk.stereo-exact-repetition/4' | 'chipmunk.stereo-exact-repetition/5' | 'chipmunk.stereo-exact-repetition/6'
    | 'chipmunk.stereo-exact-repetition/7'
  source_sha256: string; affects_overall_confidence: false
  channels: { channel: 'left' | 'right'; measurement: AudioRepetitionContext; gain_measurement?: AudioGainRepetitionContext }[]
  difference_measurement?: AudioRepetitionContext
  aligned_measurement?: AudioAlignedRepetitionContext
  rate_measurement?: AudioAlignedRepetitionContext
}
export function repeatedAudioCount(audio?: AudioEvidenceSet | VideoAudioChannel | null) {
  return (audio?.repetition_context?.pairs.length ?? 0) + (audio?.similarity_context?.pairs.length ?? 0)
    + (audio?.stereo_repetition?.channels.reduce((count, row) => count + row.measurement.pairs.length
      + (row.gain_measurement?.pairs.length ?? 0), 0) ?? 0)
    + (audio?.stereo_repetition?.difference_measurement?.pairs.length ?? 0)
    + (audio?.stereo_repetition?.aligned_measurement?.pairs.length ?? 0)
    + (audio?.stereo_repetition?.rate_measurement?.pairs.length ?? 0)
}
type AlignedRepeatSpan = AudioRepeatSpan & {
  pcm_s16le_sha256: string; recorded_excerpt: AudioRepeatSpan & { pcm_s16le_sha256: string }
}
type RateRepeatPair = Pick<AudioRepeatPair, 'id' | 'artifacts'> & {
  first: AlignedRepeatSpan; second: AlignedRepeatSpan
  faster_occurrence: 'first' | 'second'; duration_numerator: number; duration_denominator: number
}
export interface AudioAlignedRepetitionContext extends Omit<AudioRepeatRecord, 'status' | 'pcm_sha256'> {
  schema: 'chipmunk.aligned-stereo-repetition/1' | 'chipmunk.aligned-stereo-repetition/2' | 'chipmunk.aligned-stereo-repetition/3'
    | 'chipmunk.aligned-stereo-repetition/4' | 'chipmunk.aligned-stereo-repetition/5'
  status: AudioRepeatRecord['status'] | 'skipped'; pcm_sha256: string | null
  alignment: { right_offset_samples: number; right_gain_numerator: number; agreeing_windows: number; usable_windows: number } | null
  pairs: ((AudioRepeatPair & { first: AlignedRepeatSpan; second: AlignedRepeatSpan }) | RateRepeatPair)[]
}
export interface StereoTimingChange {
  schema: 'chipmunk.stereo-timing-change/1'; source_sha256: string; sample_rate: 48000
  input_samples: number; affects_overall_confidence: false; status: 'measured' | 'unavailable'
  measurement: (SampleInterval & {
    before_offset_samples: number; inside_offset_samples: number; after_offset_samples: number
  }) | null
  clips: (AudioRepeatSpan & { role: 'before' | 'during' | 'after' })[]
  artifacts: Partial<Record<'before' | 'during' | 'after' | 'timing_plot' | 'waveform', EvidenceAsset>>
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
  schema: 'chipmunk.short-voice-review/1' | 'chipmunk.short-voice-review/2' | 'chipmunk.short-voice-review/3'
  status: 'flagged' | 'no_flag' | 'skipped'
  title: string; reason: string; limitation: string; affects_overall_confidence: false
  checked_windows: SampleInterval[]; interval: SampleInterval | null; start_s: number | null; end_s: number | null
  window_coverage?: (SampleInterval & { speech_samples: number; status: 'checked' | 'below_speech_minimum';
    source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'] })[]
  nearby_window_review?: { status: 'repeated' | 'varied' | 'incomplete';
    windows: (SampleInterval & { speech_samples: number; status: 'elevated' | 'below_review_point' | 'below_speech_minimum';
      pcm_sha256: string; source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'] })[] }
  source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  artifacts: Partial<Record<'selected_clip' | 'waveform', EvidenceAsset>>
}
export interface FocusedSpeechEditReview {
  schema: 'chipmunk.focused-speech-edit-review/1' | 'chipmunk.focused-speech-edit-review/2' | 'chipmunk.focused-speech-edit-review/3'
  status: 'flagged' | 'no_flag' | 'skipped'
  title: string; reason: string; limitation: string; affects_overall_confidence: false
  checked_windows: number; interval: SampleInterval | null; start_s: number | null; end_s: number | null
  checked_scope?: SampleInterval
  source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  original_timestamp_range?: { start_pts: number; end_pts: number; time_base: '1/48000' } | null
  artifacts: Partial<Record<'selected_clip' | 'waveform', EvidenceAsset>>
}
interface TwoSecondSpeechWindow extends SampleInterval {
  supporting_excerpts: number[]; pcm_sha256: string
  source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
  native_source_range?: { start_pts: number; end_pts: number; time_base: '1/48000' }
  artifacts: Record<'selected_clip' | 'waveform', EvidenceAsset>
}
interface TwoSecondSpeechReview {
  schema: 'chipmunk.extended-short-voice-review/1' | 'chipmunk.extended-short-voice-review/2' | 'chipmunk.extended-short-voice-review/3'
  title: string; status: 'flagged' | 'no_flag' | 'unassessed' | 'skipped'
  reason: string; limitation: string; affects_overall_confidence: false
  excerpts: { start_sample: number; pcm_sha256: string; speech_samples: number[]; selected_window: number | null; elevated_windows?: number[] }[]
  regions: (TwoSecondSpeechWindow & { confirming_window?: TwoSecondSpeechWindow; same_pair_as?: number })[]
}
export interface LongVoiceReview {
  schema: 'chipmunk.long-voice-review/1' | 'chipmunk.long-voice-review/2' | 'chipmunk.long-voice-review/3'; title: 'Long speech-window review'
  status: 'corroborated' | 'no_corroboration' | 'skipped'
  reason: string; limitation: string; affects_overall_confidence: false
  sample_coverage?: {
    samples_by_status: Record<LongVoiceCoverageStatus, number>
    intervals: (SampleInterval & { status: LongVoiceCoverageStatus; source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'] })[]
  }
  windows: (SampleInterval & {
    pcm_sha256: string; direction: 'ai' | 'original' | 'inconclusive' | 'unassessed'
    speech_samples?: number
    source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
    artifacts: Partial<Record<'selected_clip' | 'waveform', EvidenceAsset>>
  })[]
  regions?: (SampleInterval & {
    basis: 'overlap' | 'sparse'; supporting_windows: number[]; pcm_sha256: string
    source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']
    artifacts: Record<'selected_clip' | 'waveform', EvidenceAsset>
  })[]
  two_second_review?: TwoSecondSpeechReview
}
export type LongVoiceCoverageStatus = 'directional' | 'inconclusive' | 'below_speech_minimum' | 'outside_schedule'
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
    identical_stereo_channel_pcm_sha256?: string; source_duration_pts?: number;
    timeline?: (SampleInterval & { source_pts: number })[] }
  coverage?: { decoded_samples: number; scored_samples: number; unscored_samples: number; decode_limit_reached: boolean;
    selection_threshold: number; windows: (SampleInterval & { score: number; signal_refs: EvidenceRef[] })[];
    source_clock_context?: AudioSourceClockContext;
    mono_voice_coverage_limit?: { policy: 'destructive-stereo-downmix/1'; reason: string; stereo_window_indices: number[] };
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
  stereo_timing_change?: StereoTimingChange
  repetition_context?: AudioRepetitionContext
  stereo_repetition?: StereoRepetitionContext
  stereo_opus_repeat_unavailable?: true
  similarity_context?: AudioSimilarityContext
  short_voice_review?: ShortVoiceReview
  focused_speech_edit_review?: FocusedSpeechEditReview
  long_voice_review?: LongVoiceReview
  two_second_review?: TwoSecondSpeechReview
  background_reduced_voice_review?: BackgroundVoiceReview
}
export type VideoChannelStatus = 'ready' | 'absent' | 'unavailable' | 'not_assessed' | 'not_applicable'
export interface SoundtrackPair {
  status: 'duration_outside_scope' | 'second_stream_unavailable' | 'unsupported_clock' | 'insufficient_same_time_support' | 'no_local_shift' | 'local_shift' | 'local_difference'
  stream_indices: [number, number]; limitation: string
  measurement_basis?: 'two-band-corroboration/1' | 'short-shared-sound-difference/1'
  short_compared_samples?: number
  coverage: Partial<Record<'candidate_windows' | 'matched_windows' | 'same_time_windows' | 'sections_with_same_time_matches', number>>
  location: ({
    excerpts: (SampleInterval & { stream_index: number; source_timestamp_ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges'];
      clip: EvidenceAsset; waveform: EvidenceAsset })[]
  } & ({ shift_samples: number; supporting_windows: number; minimum_correlation: number } | {
    bands: { band_hz: number[]; matching_gain: number[]; correlation: number[]; active_windows: number[]; local_relative_gain: number }[]
  })) | null
}
export interface VideoAudioChannel {
  status: VideoChannelStatus; reason: string | null; source: AudioEvidenceSet['source']
  packets: AudioEvidencePacket[]; omitted_windows?: number; limitations: string[]
  coverage?: AudioEvidenceSet['coverage'] | null; overview?: Record<string, EvidenceAsset>
  acoustic_context?: AcousticContext
  sound_events?: SoundEventContext
  stereo_context?: StereoChannelContext
  stereo_listening?: StereoListeningContext
  stereo_timing_change?: StereoTimingChange
  repetition_context?: AudioRepetitionContext
  stereo_repetition?: StereoRepetitionContext
  similarity_context?: AudioSimilarityContext
  soundtrack_pair?: SoundtrackPair
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
  comparison_context?: { role: 'before' | 'after'; reference_packet_id: string }
  measurement: { offset_ticks: number; offset_ms: number; peak_correlation: number; low_correlation: boolean
    lag_profile: { offset_ticks: number; correlation: number }[]
    audio_curve: { tick: number; sample_start: number; sample_end: number; rms: number; speech: boolean }[]
    overlaps_offset_step: boolean; overlaps_correlation_collapse: boolean
    baseline_offset_ticks: number | null; collapse_threshold: number | null; minimum_trusted_peak: number; reading: string
    competing_lag?: { method: 'strongest-at-least-three-ticks-away/1'; offset_ms: number; correlation: number; correlation_difference: number } }
}
export interface VideoAVChannel {
  status: VideoChannelStatus; reason: string | null; limitations: string[]; packets: VideoAVPacket[]
  audio_clock_breaks?: { time_base: '1/16000'; breaks: { boundary_sample: number; before_end_pts: number; after_start_pts: number }[] }
  source: AudioEvidenceSet['source'] & { tick_rate?: number; samples_per_tick?: number; video_time_base?: string }
  coverage?: { recorded_windows: number; selected_windows: number; omitted_windows: number; selection: string
    unsupported_checks: { status: string; reason: string }[]
    repeated_offsets?: { offset_ms: number; packet_ids: string[] }[]
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
  part_kind?: 'slide' | 'epub_entry'; part_index?: number
  selected_start?: number; selected_end?: number; highlight_boxes?: number[][]
  ocr_word_confidence?: number | number[]
}
export interface DocumentOriginalTextLocation {
  anchor_index: number
  coordinate: 'decoded_text' | 'word_paragraph' | 'odt_paragraph' | 'rtf_paragraph' | 'pptx_block' | 'epub_block' |
    'decoded_email' | 'decoded_email_html' | 'decoded_html'
  ranges: [number, number][]
}
export interface DocumentEvidencePacket extends PacketBase {
  review_basis?: 'writing_pattern_only' | 'different_writing_ranges'
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
  mean_negative_log_likelihood?: number; mean_cross_entropy?: number
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
type PublishedWordingRange = { start: number; end: number; archive_range: [number, number] }
type PublishedWordingLocation = { start: number; end: number; page: number; boxes: [number, number, number, number][] }
  | { start: number; end: number; paragraph?: number; ranges: [number, number][];
      coordinate_space?: 'extracted-paragraph-codepoints' }
export interface PublishedWordingSection {
  reference: { id: string; published_date: string; source_record_sha256: string; transcript_sha256: string; text: string }
    & ({ version_url: string } | { publisher: 'bea' | 'census'; title: string; source_url: string });
  submitted_range: [number, number]; archive_range: [number, number];
  changes: (PublishedWordingRange & { submitted: string; locations: PublishedWordingLocation[] })[];
  changed_sentences?: { sentence: number; start: number; end: number; text: string; locations: PublishedWordingLocation[]; differences: number[] }[];
}
export type PublishedWordingColumn = PublishedWordingSection & {
  page: number; column: 'left' | 'right' | 'both'; source_blocks: [number, number][];
}
export type PublishedWordingReview = {
  archive_sha256: string; authorship_confidence_contribution: 0; published_style_matches: PublishedWordingRange[];
} & (({ schema: 'document-published-wording-review/1' | 'document-published-wording-review/3' | 'document-published-wording-review/4' } & PublishedWordingSection)
  | { schema: 'document-published-wording-review/2'; sections: PublishedWordingColumn[] })
export type MappedTextFinding = {
  start: number; end: number; text: string
  anchor: { kind: 'text' | 'docx' | 'pdf' | 'email'; page?: number; block?: number; paragraph_index?: number;
    body_path?: (number | 'row' | 'cell' | 'control')[]; mime_part?: string }
  location: { boxes: number[][] } | { ranges: [number, number][] }
}
export interface DocumentEvidenceSet extends SetBase {
  schema: 'chipmunk.document-evidence/1'; packets: DocumentEvidencePacket[]; omitted_windows: number
  source: { sha256: string; chars?: number; method?: string; offset_unit: 'unicode_codepoint' }
  calendar_consistency?: {
    schema: 'document-calendar-consistency/1'; affects_overall_confidence: false; scope: string; limitation: string
    mapped_blocks: number; checked_dates: number; disagreements: number
    locations: (MappedTextFinding & { stated_weekday: string; expected_weekday: string | null; iso_date: string | null;
      reason: 'weekday_mismatch' | 'invalid_date' })[]
  }
  arithmetic_consistency?: {
    schema: 'document-arithmetic-consistency/1' | 'document-arithmetic-consistency/2'; affects_overall_confidence: false; scope: string; limitation: string
    checked_equations: number; disagreements: number
    length_conversions_checked?: number; length_conversion_disagreements?: number
    locations: (MappedTextFinding & { stated: string; calculated: string })[]
  }
  publication_identifier_consistency?: {
    schema: 'document-publication-identifiers/1'; affects_overall_confidence: false; scope: string; limitation: string
    checked: number; disagreements: number
    locations: (Omit<MappedTextFinding, 'anchor'> & {
      kind: 'ISBN-10' | 'ISBN-13' | 'ISSN'; written: string; expected: string
      anchor: Omit<MappedTextFinding['anchor'], 'kind'> & { kind: MappedTextFinding['anchor']['kind']
        | 'decoded_html' | 'odt_paragraph' | 'rtf_paragraph' }
    })[]
  }
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
    writing_bibliography?: { schema: 'chipmunk.writing-bibliography/1'; heading: string; entry_count: number;
      section_start: number; section_end: number; first_location: BibliographySource; last_location: BibliographySource };
    balanced_passage_selection?: { schema: 'document-balanced-passage-selection/1'; candidate_blocks: number;
      displayed_blocks: number; displayed_passages: number };
    word_tracked_changes?: { status: 'found' | 'partial' | 'no_match'; scope: string; limitation: string;
      writing_input?: 'no_tracked_text_changes' | 'accepted_inline_changes' | 'accepted_paragraph_changes' | 'unsupported_tracked_changes';
      part: string; paragraphs: number; inserted: number; deleted: number; unsupported_marks: number;
      omitted_excerpts: number; rows: { kind: 'inserted' | 'deleted'; paragraph: number; body_index: number;
        inline_index: number; text: string; characters: number; writing_input_range?: [number, number] }[];
      paragraph_marks?: { kind: 'inserted' | 'deleted'; paragraph: number; body_index: number; child_path: [number, number, number] }[];
      omitted_paragraph_marks?: number;
      authorship_confidence_contribution: 0 };
    email_body_alternatives?: EmailBodyAlternatives;
    email_classifier_input_scope?: EmailClassifierInputScope;
    local_passage_coverage?: DocumentPassageCoverage;
    unselected_passage_responses?: DocumentUnselectedResponses;
    passage_position_review?: DocumentPassagePosition;
    published_wording_review?: PublishedWordingReview;
    pdf_saved_wording?: PdfSavedWording;
    writing_passage_intervals?: {
      schema: 'document-writing-passage-intervals/1' | 'document-writing-passage-intervals/2' | 'document-writing-passage-intervals/3' | 'document-writing-passage-intervals/4';
      status: 'flagged' | 'no_ranges' | 'unassessed';
      ranges: ({ start: number; end: number; preview: string; basis?: 'pooled' | 'document_baseline' } & (
        { source_start: number; source_end: number } | { locations: (
          { page: number; boxes: [number, number, number, number][] } |
          { paragraph: number; ranges: [number, number][] } |
          { mime_part: string; ranges: [number, number][] }
        )[] }
      ))[];
      authorship_confidence_contribution: 0;
    };
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
    pdf_text_visibility?: { schema: 'document-pdf-text-visibility/1' | 'document-pdf-text-visibility/2' |
      'document-pdf-text-visibility/3' | 'document-pdf-text-visibility/4';
      total_pages: number; examined_pages: number[];
      locations: { page: number; invisible_spans: number; boxes: [number, number, number, number][] }[];
      replacement_examined_pages?: number[];
      replacement_differences?: { page: number; boxes: [number, number, number, number][]; locations_incomplete: boolean }[];
      covered_examined_pages?: number[];
      covered_locations?: { page: number; covered_spans: number; boxes: [number, number, number, number][] }[];
      coordinate_space: 'unrotated-page-points'; authorship_confidence_contribution: 0 };
    word_saved_fields?: { schema: 'document-word-saved-fields/1'; part: string; found: number; included: number;
      locations: { body_index: number; field_index: number; source_start: number; source_end: number;
        start: number; end: number; text: string }[] };
    word_text_visibility?: { schema: 'document-word-text-visibility/1'; part: string; checked_paragraphs: number; hidden_runs: number;
      locations: { body_path: (number | 'row' | 'cell' | 'control')[]; body_paragraph: number | null; run: number;
        run_basis?: 'extracted'; source_start: number; source_end: number }[];
      authorship_confidence_contribution: 0 };
    invisible_characters?: { schema: 'document-invisible-characters/1' | 'document-invisible-characters/2';
      coordinate_space: 'decoded_text' | 'word_paragraph' | 'decoded_email' | 'decoded_email_html';
      checked_blocks: number; checked_chars: number; matches: number; authorship_confidence_contribution: 0;
      locations: { text_offset: number; source_start: number; source_end: number; codepoint: string; looks_like?: string;
        body_path?: (number | 'row' | 'cell' | 'control')[]; mime_part?: string }[] };
    writing_character_review?: { schema: 'document-writing-character-review/1'; authorship_confidence_contribution: 0;
      input_view: 'english-writing-character-view/1';
      ranges: { spans: { start: number; end: number }[]; excerpt: string;
        location: { kind: 'text'; ranges: [number, number][] } | { kind: 'docx'; body_path: number[]; ranges: [number, number][] }
          | { kind: 'email'; mime_part: string; html: boolean; ranges: [number, number][] } | { kind: 'pdf'; page: number };
        changes: { offset: number; codepoint: string; replacement: string }[];
        state: 'changed' | 'different' | 'consistent' }[] };
    chat_format_markers?: { schema: 'document-chat-format-markers/1'; scope: string; limitation: string;
      checked_chars: number; turns: number; authorship_confidence_contribution: 0;
      locations: { opener: DocumentChatMarker; closer: DocumentChatMarker }[] };
    checks?: { signal_ref: EvidenceRef; status: 'recorded' | 'partial' | 'context_not_recorded'; scored_windows: number; limitation?: string }[];
    writing_check_coverage?: WritingCheckCoverage;
    ocr?: { status?: string; pages?: { page: number; status: string; coverage_scope?: 'image_region'; reason?: string | null; mean_recognition_confidence?: number;
      language?: { status?: string; language?: string; reason?: string }; [key: string]: unknown }[]; unchecked_pages?: number[] } | null
    email?: { body_part?: string; body_content_type?: 'text/plain' | 'text/html'; attachments?: unknown[]; [key: string]: unknown } | boolean | null }
}

type PdfSavedWords = { page: number; page_xref: number; word_range: [number, number]; excerpt_word_range: [number, number];
  boxes: [number, number, number, number][] }
type PdfSave = { version: number; sha256: string; prefix_bytes: number; xref_offset: number; pages: number }
export interface PdfSavedPictures {
  schema: 'chipmunk.pdf-saved-pictures/1'
  status: 'changed' | 'no_change' | 'no_history' | 'unavailable'
  affects_overall_confidence: false
  reason: null | 'size_limit' | 'page_limit' | 'save_limit' | 'unverified_chain' | 'time_limit' | 'reader_unavailable'
    | 'no_comparable_pictures' | 'partial_picture_coverage'
  saves: PdfSave[]
  comparisons: { earlier_version: number; pages: { earlier: number; current: number; compared: number; skipped: number }[];
    unpaired_pages: [number[], number[]]; changed_regions: number;
    changes: { current_page: number; earlier_page: number; page_rotation: number; render_scale: number;
      picture_box: [number, number, number, number]; region_pixels: [number, number, number, number]; marked_pixels: number;
      matching_tiles: number; tested_tiles: number }[] }[]
}
export interface PdfSavedWording {
  inspection: {
    schema: 'chipmunk.pdf-saved-wording/1'; status: 'changed' | 'no_change' | 'no_history' | 'unavailable'
    reason: null | 'size_limit' | 'page_limit' | 'save_limit' | 'page_text_incomplete' | 'unverified_chain' | 'time_limit' | 'reader_unavailable'
    affects_overall_confidence: false
    saves: PdfSave[]
    comparisons: { earlier_version: number; checked: [number, number][]; unassessed: [number, number][];
      order_only: [number, number][]; earlier_only_pages: number[]; current_only_pages: number[]; changed_ranges: number;
      changes: { kind: 'replace' | 'insert' | 'delete'; earlier: PdfSavedWords | null;
        current: (PdfSavedWords & { excerpt: string }) | null }[] }[]
    pictures?: PdfSavedPictures
  }
  artifacts: Partial<Record<string, EvidenceAsset>>
}

type DocumentChatMarker = { literal: string; start: number; end: number; ranges?: [number, number][];
  anchor: { kind: 'text' } | { kind: 'docx'; body_path: (number | 'row' | 'cell' | 'control')[] }
    | { kind: 'pdf'; page: number; block: number } | { kind: 'email'; mime_part: string } };

type BibliographySource = { kind: 'pdf'; page: number } | { kind: 'docx'; body_index: number } | { kind: 'text'; start: number; end: number }
type WritingCheck = 'passage_classifier' | 'writing_pattern_ratio'
type WritingCheckState = 'completed' | 'not_applicable' | 'skipped' | 'failed' | 'timed_out'
export type WritingCheckReason = 'memory_budget' | 'unavailable' | 'verified_credentials'
  | 'unsupported_language' | 'mixed_language' | 'language_unclear' | 'no_text' | 'too_little_after_cover'
  | 'unsupported_word_revisions' | 'too_little_after_word_revisions'
type WritingCheckCoverage =
  | { schema: 'document-writing-check-coverage/1'; checks: Partial<Record<WritingCheck, WritingCheckState>> }
  | { schema: 'document-writing-check-coverage/2'; checks: Partial<Record<WritingCheck, WritingCheckState>>;
      reasons: Partial<Record<WritingCheck, WritingCheckReason>> }
  | { schema: 'document-writing-check-coverage/3'; checks: Partial<Record<WritingCheck, WritingCheckState>>;
      reasons: Partial<Record<WritingCheck, WritingCheckReason>>;
      language_witness: { sentence: number; range: { start: number; end: number; sha256: string };
        context: { start: number; end: number; sha256: string } } }
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
nestedVideoVersions.add('video-evidence-32')
nestedVideoVersions.add('video-evidence-33')
nestedVideoVersions.add('video-evidence-34')
nestedVideoVersions.add('video-evidence-35')
nestedVideoVersions.add('video-evidence-36')
nestedVideoVersions.add('video-evidence-37')
nestedVideoVersions.add('video-evidence-38')
nestedVideoVersions.add('video-evidence-39')
nestedVideoVersions.add('video-evidence-40')
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
