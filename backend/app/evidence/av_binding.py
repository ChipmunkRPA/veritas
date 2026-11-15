"""Standard-library validation of public, source-bound correlation evidence."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise

from app.evidence import av_lag_support
from app.evidence.mp4 import validate_clip_binding

LEGACY_AUDIO_CLOCK_REASON = 'Audio timestamps contain gaps; audio/video correlation does not support this timeline.'
AUDIO_CLOCK_BREAKS = 'audio_clock_breaks'
AV_CLOCK_DISPLAY_LIMIT = 8
AV_LEGACY_SELECTION = 'flagged windows first, then chronological informational measured examples'
AV_CONTEXT_SELECTION = 'first flagged window with nearby measured context before and after, then other measured windows'
AV_CONTEXT_GAP_TICKS = 25
TIMING_WINDOWS = 'timing_windows'
REPEATED_OFFSETS = 'repeated_offsets'
TIMING_COMMITMENT = 'av_timing_windows_sha256'
TIMING_REASONS = {
    'measured': 'Mouth and audio timing was measured.',
    'insufficient_speech': 'Too little speech was detected for this window.',
    'missing_picture_timestamps': 'No source picture was available close enough to every audio tick.',
    'insufficient_audio_context': 'The full timing search would extend beyond the audio.',
    'no_face_track': 'No tracked face was available.',
    'missing_landmarks': 'No single face track had usable mouth landmarks across the entire window.',
    'track_limit': 'The window was outside the selected face tracks.',
    'missing_mouth_input': 'Some source mouth crops could not be verified.',
    'unchanged_mouth_pixels': 'The mouth crops were identical across the window.',
    'flat_mouth_proxy': 'The mouth-opening measurement stayed flat.',
    'speaker_unresolved': 'Speech could not be linked to a visible mouth for this window.',
}
TIMING_DISPLAY_LIMIT = 12


class InvalidAV(ValueError):
    pass


def repeated_offsets(packets):
    """Summarize only displayed, time-disjoint support for the same recorded offset."""
    groups = defaultdict(list)
    for packet in packets:
        m, s = packet['measurement'], packet['source_record']
        lag = m['offset_ticks']
        if (m['reading'] == 'constant_offset' and lag == m['baseline_offset_ticks'] and abs(lag) >= 3
                and not m['overlaps_offset_step'] and not m['overlaps_correlation_collapse'] and not m['low_correlation']
                and m['peak_correlation'] >= m['minimum_trusted_peak']
                and av_lag_support.compare(m['lag_profile'], lag)['correlation_difference'] >= .05):
            groups[s['raw_output_sha256'], s['track_id'], lag].append(packet)
    rows = []
    for (_, _, lag), group in groups.items():
        chosen, end = [], -1
        for packet in sorted(group, key=lambda value: value['window']['start_tick']):
            window = packet['window']
            if window['start_tick'] >= end:
                chosen.append(packet['id'])
                end = window['end_tick']
        if len(chosen) >= 2:
            rows.append({'offset_ms': lag * 40, 'packet_ids': chosen})
    return rows


def timing_ranges(samples: int) -> range:
    """One-second audio windows at the alignment check's half-second cadence."""
    ticks = samples // 640
    return range(max(0, (2 * (ticks - 25) + 1) // 25 + 1))


def timing_interval(index: int, origin: int) -> dict:
    start = (index * 25 // 2) * 640
    end = start + 16000
    return {'start_sample': start, 'end_sample': end, 'source_start_pts': origin + start,
            'source_end_pts': origin + end, 'time_base': '1/16000'}


def _validate_timing_coverage(coverage: dict, samples: int, origin: int) -> None:
    rows = coverage['windows']
    if (set(coverage) != {'raw_output_sha256', 'windows'} or not isinstance(rows, list)
            or not re.fullmatch(r'[a-f0-9]{64}', coverage['raw_output_sha256'])
            or len(rows) != len(timing_ranges(samples))):
        raise InvalidAV('Timing coverage differs from the audio window schedule')
    for index, row in enumerate(rows):
        start = index * 25 // 2
        if (set(row) != {'status', 'speech_ticks', 'interval'} or row['status'] not in TIMING_REASONS
                or type(row['speech_ticks']) is not int or not 0 <= row['speech_ticks'] <= 25
                or row['interval'] != timing_interval(index, origin)
                or any(type(row['interval'][key]) is not int for key in
                       ('start_sample', 'end_sample', 'source_start_pts', 'source_end_pts'))):
            raise InvalidAV('Timing coverage row has an invalid status, speech count or interval')
        if (row['status'] == 'insufficient_speech') != (row['speech_ticks'] < 13):
            raise InvalidAV('Timing coverage differs from the recorded speech decision')
        full_search = start >= 15 and (start + 40) * 640 <= samples
        if ((row['status'] == 'insufficient_audio_context' and full_search)
                or (row['status'] not in {'insufficient_speech', 'missing_picture_timestamps', 'insufficient_audio_context'}
                    and not full_search)):
            raise InvalidAV('Timing coverage disagrees with the available audio search margin')


def iter_audio_clock_breaks(timeline: Sequence[Mapping[str, int]]):
    """Yield each discontinuity in a validated 16 kHz source timeline."""
    for before, after in pairwise(timeline):
        end = before['source_pts'] + before['end_sample'] - before['start_sample']
        start = after['source_pts']
        if end != start:
            yield {'boundary_sample': after['start_sample'], 'before_end_pts': end, 'after_start_pts': start}


def audio_clock_break_evidence(timeline: Sequence[Mapping[str, int]]) -> dict:
    return {'time_base': '1/16000', 'breaks': list(iter_audio_clock_breaks(timeline))}


def audio_clock_discontinuity(timeline: Sequence[Mapping[str, int]]) -> str:
    """Describe the first discontinuity in a validated 16 kHz source timeline."""
    if first := next(iter_audio_clock_breaks(timeline), None):
        end, start = first['before_end_pts'], first['after_start_pts']
        direction = 'forward' if start > end else 'backward'
        return (f'Audio/video timing was unassessed. The decoded audio clock jumps {direction} '
                f'from {Decimal(end) / 16000:.7f} s to {Decimal(start) / 16000:.7f} s on the source clock.')
    return 'Audio/video timing was unassessed because the decoded audio timeline has separate spans.'


def check_audio_clock_breaks(channel, timeline) -> None:
    if AUDIO_CLOCK_BREAKS not in channel:
        return  # Historical exports recorded only the first discontinuity.
    recorded = channel[AUDIO_CLOCK_BREAKS]
    expected = audio_clock_break_evidence(timeline)
    if (not expected['breaks'] or not isinstance(recorded, dict) or recorded != expected
            or any(type(value) is not int for row in recorded['breaks'] for value in row.values())):
        raise InvalidAV('AV clock breaks differ from the decoded audio timeline')


def validate_audio_clock_coverage(document: dict) -> None:
    audio = document['audio']
    channel = document['av']
    timeline = audio['source'].get('timeline', [])
    if AUDIO_CLOCK_BREAKS in channel and (len(timeline) <= 1 or audio['coverage'] is None):
        raise InvalidAV('AV clock breaks have no verified audio timeline')
    check_audio_clock_breaks(channel, timeline)
    if audio['coverage'] is None or len(timeline) <= 1:
        return
    reason = audio_clock_discontinuity(timeline)
    if (channel['status'] != 'unavailable' or channel['coverage'] is not None
            or channel['reason'] not in (LEGACY_AUDIO_CLOCK_REASON, reason)):
        raise InvalidAV('Frozen AV coverage differs from the decoded audio clock')
    summary = document['coverage']['channels'].get('av')
    if summary != {'status': channel['status'], 'reason': channel['reason']}:
        raise InvalidAV('Reported AV coverage differs from the decoded audio clock')


def validate_av_channel(channel: dict, video_source: dict) -> None:
    try:
        _validate(channel, video_source)
    except (KeyError, IndexError, TypeError, AttributeError, ZeroDivisionError, OverflowError) as error:
        raise InvalidAV('Audio/video evidence contains missing fields or invalid types') from error


def _validate(channel: dict, video: dict) -> None:
    def require(condition, message):
        if not condition:
            raise InvalidAV(message)

    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def numeric(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

    def sha(value):
        return bool(re.fullmatch(r'[a-f0-9]{64}', str(value)))

    def interval(span):
        return (integer(span['start_sample']) and integer(span['end_sample'])
                and 0 <= span['start_sample'] < span['end_sample'] <= source['samples'])

    require(not {'schema', 'version', 'id', 'sha256', 'analysis_id', 'revision', 'original_result_hash'}.intersection(channel), 'AV channel cannot claim independent authority')
    source = channel['source']
    require(source['sha256'] == video['sha256'], 'AV source differs from the original video')
    require(channel['status'] in {'ready', 'not_assessed', 'absent', 'unavailable'}, 'AV availability is invalid')
    if AUDIO_CLOCK_BREAKS in channel:
        require(channel['status'] == 'unavailable' and channel['coverage'] is None,
                'AV clock breaks require an unassessed timing channel')
    if channel['coverage'] is None:
        require(channel['status'] != 'ready' and not channel['packets'] and bool(channel['reason']), 'Unavailable AV channel contains evidence or lacks a reason')
        return
    require(source['sample_rate'] == 16000 and source['tick_rate'] == 25 and source['samples_per_tick'] == 640, 'AV analysis clock differs from the recorded tick mapping')
    require(integer(source['samples']) and 0 < source['samples'] <= 120 * 16000, 'AV PCM sample count is invalid')
    require(all(sha(source[key]) for key in ('pcm_sha256', 'wav_sha256', 'mapping_sha256')), 'AV audio commitments are missing')
    require(all(source[key] == video[key] for key in ('frame_table_sha256', 'frame_anchor_sha256')), 'AV video commitments differ from their parent')
    require(source['video_time_base'] == video['time_base'] and source['timeline_time_base'] == '1/16000', 'AV source clocks differ')
    require(len(source['timeline']) == 1, 'Discontinuous audio is not supported for AV correlation')
    timeline = source['timeline'][0]
    require(timeline['start_sample'] == 0 and timeline['end_sample'] == source['samples'] and integer(timeline['source_pts']), 'AV source timeline is not a continuous PCM mapping')
    origin = timeline['source_pts']
    if TIMING_WINDOWS in channel['coverage']:
        timing = channel['coverage'][TIMING_WINDOWS]
        _validate_timing_coverage(timing, source['samples'], origin)
        measured_starts = {row['interval']['start_sample'] for row in timing['windows'] if row['status'] == 'measured'}
        require(all(packet['interval']['start_sample'] in measured_starts for packet in channel['packets']
                    if packet['source_record']['raw_output_sha256'] == timing['raw_output_sha256']),
                'A published AV packet differs from the timing coverage')
    selection = channel['coverage'].get('selection', AV_LEGACY_SELECTION)
    require(selection in {AV_CONTEXT_SELECTION, AV_LEGACY_SELECTION}, 'Unknown AV packet selection policy')
    roles = set()
    primary = channel['packets'][0] if channel['packets'] else None
    ids = set()
    for packet in channel['packets']:
        require(packet['id'] not in ids and packet['id'].startswith('AV'), 'AV packet identities are invalid')
        ids.add(packet['id'])
        require((packet['evidence_type'], packet['claim_scope'], packet['semantics']) == ('anomaly', 'sampled_av_window', 'mouth_audio_correlation'), 'AV measurements are misrepresented as AI probabilities')
        window, measure, anchor = packet['window'], packet['measurement'], packet['anchor']
        start, end = window['start_tick'], window['end_tick']
        require(integer(start) and integer(end) and start >= 15 and end - start == 25 and (end + 15) * 640 <= source['samples'], 'AV measured window cannot support the full lag search')
        require(interval(packet['interval']) and packet['interval'] == {'start_sample': start * 640, 'end_sample': end * 640}, 'AV selected WAV differs from its measurement window')
        require(interval(packet['context_interval']) and packet['context_interval'] == {'start_sample': (start - 15) * 640, 'end_sample': (end + 15) * 640}, 'AV context WAV differs from the complete lag-search input')
        require(anchor['original_sha256'] == video['sha256'] and anchor['pcm_sha256'] == source['pcm_sha256'] and anchor['sample_rate'] == 16000, 'AV selected samples refer to another input')
        expected_range = {'start_sample': start * 640, 'end_sample': end * 640,
                          'source_start_pts': origin + start * 640, 'source_end_pts': origin + end * 640, 'time_base': '1/16000'}
        require(anchor['source_timestamp_ranges'] == [expected_range], 'AV audio interval lost its exact source timestamps')
        inputs = window['frame_inputs']
        require(len(inputs) == 25 and [row['tick'] for row in inputs] == list(range(start, end)), 'AV frame inputs omit or invent measured ticks')
        require(window['frame_indices'] == [row['frame_index'] for row in inputs], 'AV frame list loses repeated VFR samples')
        artifacts = packet['artifacts']
        for row in inputs:
            require(integer(row['frame_index']) and 0 <= row['frame_index'] < video['frame_count'] and integer(row['frame_pts']), 'AV frame index or source PTS is invalid')
            require(row['source_audio_pts'] == origin + row['tick'] * 640 and row['source_audio_time_base'] == '1/16000', 'AV tick is not on the recorded source audio clock')
            require(sha(row['decoded_rgb_sha256']) and sha(row['mouth_patch_sha256']) and numeric(row['mouth_value']), 'AV source pixels or measured patch are uncommitted')
            region = row['mouth_region']
            require(all(integer(region[k]) for k in ('x', 'y', 'w', 'h')) and 0 <= region['x'] < region['x'] + region['w'] <= video['width']
                    and 0 <= region['y'] < region['y'] + region['h'] <= video['height'], 'AV mouth crop lies outside its source frame')
            patch = artifacts[row['artifact_ref']]
            require(patch['media_type'] == 'image/png' and (patch['width'], patch['height']) == (32, 24), 'AV measured patch has different dimensions')
        profile = measure['lag_profile']
        require(len(profile) == 31 and [p['offset_ticks'] for p in profile] == list(range(-15, 16)), 'AV evidence omits lag candidates')
        require(all(numeric(p['correlation']) and -1 <= p['correlation'] <= 1 for p in profile), 'AV lag correlation is invalid')
        lag, peak = measure['offset_ticks'], measure['peak_correlation']
        require(integer(lag) and -15 <= lag <= 15 and numeric(peak) and measure['offset_ms'] == lag * 40, 'AV selected lag or units are invalid')
        require(abs(profile[lag + 15]['correlation'] - peak) <= 0.000002 and max(p['correlation'] for p in profile) - peak <= 0.000002, 'AV selected lag is not supported by its measured profile')
        if av_lag_support.KEY in measure:
            expected = av_lag_support.compare(profile, lag)
            actual = measure[av_lag_support.KEY]
            require(isinstance(actual, dict) and set(actual) == set(expected)
                    and actual['method'] == expected['method'] and integer(actual['offset_ms'])
                    and actual['offset_ms'] == expected['offset_ms'] and numeric(actual['correlation'])
                    and actual['correlation'] == expected['correlation'] and numeric(actual['correlation_difference'])
                    and actual['correlation_difference'] == expected['correlation_difference'],
                    'AV alternative lag differs from the measured profile')
        curve = measure['audio_curve']
        require(len(curve) == 55 and [r['tick'] for r in curve] == list(range(start - 15, end + 15)), 'AV audio curve omits lag-search input ticks')
        require(all(row['sample_start'] == row['tick'] * 640 and row['sample_end'] == (row['tick'] + 1) * 640
                    and numeric(row['rms']) and 0 <= row['rms'] <= 1 and isinstance(row['speech'], bool) for row in curve), 'AV audio curve has invalid sample coordinates or measurements')
        for role, span in (('selected_clip', packet['interval']), ('context_clip', packet['context_interval'])):
            asset = artifacts[role]
            require(asset['media_type'] == 'audio/wav' and asset['sample_rate'] == 16000 and asset['channels'] == 1
                    and asset['samples'] == span['end_sample'] - span['start_sample'], 'AV WAV asset differs from its recorded samples')
        for asset in artifacts.values():
            require(sha(asset['sha256']) and asset['media_type'] in {'image/png', 'audio/wav', 'video/mp4'}, 'AV asset type or hash is invalid')
        by_tick = {row['tick']: row for row in inputs}
        for frame in packet['frames']:
            tick = frame['av_tick']
            require(tick in by_tick, 'Displayed AV frame was not part of its measured window')
            recorded, fa = by_tick[tick], frame['anchor']
            require(fa['original_sha256'] == video['sha256'] and fa['time_base'] == video['time_base']
                    and all(fa[key] == recorded[other] for key, other in [('frame_index', 'frame_index'), ('pts', 'frame_pts'), ('decoded_rgb_sha256', 'decoded_rgb_sha256')]), 'Displayed AV frame differs from its measured source')
            require(abs(fa['stream_time_s'] - float(fa['pts'] * Fraction(video['time_base']))) <= 0.000002, 'AV frame seconds differ from its exact source PTS')
            origin_s, playback_s = video.get('timeline_origin_s'), fa.get('playback_time_s')
            require((origin_s is None) == (playback_s is None) and (origin_s is None or
                    numeric(playback_s) and abs(fa['stream_time_s']-origin_s-playback_s) <= 0.000002),
                    'AV displayed frame playback origin differs from the original video')
            require(frame['context_region'] == recorded['mouth_region'], 'Displayed AV crop differs from its actual input region')
        require(sha(packet['source_record']['raw_output_sha256']), 'AV packet has no immutable run reference')
        if 'comparison_context' in packet:
            context = packet['comparison_context']
            require(selection == AV_CONTEXT_SELECTION and primary and packet is not primary
                    and isinstance(context, dict) and set(context) == {'role', 'reference_packet_id'}
                    and context['reference_packet_id'] == primary['id'] == 'AV01'
                    and context['role'] in {'before', 'after'} and context['role'] not in roles,
                    'AV comparison window has an invalid role or reference')
            first = primary['measurement']
            require((first['overlaps_offset_step'] or first['overlaps_correlation_collapse'])
                    and not (measure['overlaps_offset_step'] or measure['overlaps_correlation_collapse'])
                    and all(packet['source_record'][key] == primary['source_record'][key] for key in ('raw_output_sha256', 'track_id')),
                    'AV comparison window must use the same face track and run as a flagged window')
            gap = (primary['window']['start_tick'] - end if context['role'] == 'before'
                   else start - primary['window']['end_tick'])
            require(0 <= gap <= AV_CONTEXT_GAP_TICKS, 'AV comparison window overlaps its reference or is too far away')
            roles.add(context['role'])
        validate_clip_binding(packet, video)
    if REPEATED_OFFSETS in channel['coverage']:
        require(bool(records := repeated_offsets(channel['packets'])) and channel['coverage'][REPEATED_OFFSETS] == records,
                'Repeated AV offset differs from its measured support')
    require(channel['coverage']['selected_windows'] == len(channel['packets']) and channel['coverage']['recorded_windows'] >= len(channel['packets']), 'AV coverage differs from its selected packets')
    require((channel['status'] == 'ready') == bool(channel['coverage']['recorded_windows']), 'AV availability differs from recorded measurements')
