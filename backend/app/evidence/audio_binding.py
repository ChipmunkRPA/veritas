"""Standard-library validation of audio content nested in a public video set."""
from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right

from app.core.canonical import canonical_hash

_CLOCK_BREAK_MIN_SAMPLES = 320
_CLOCK_BREAK_MAX_ROWS = 8
VOICE_CLOCK_LIMIT = (
    "The source audio timestamps contain a gap or overlap. "
    "A recording-wide voice estimate is unavailable."
)
VOICE_CLOCK_POLICY = "source-clock-split/1"


def current_voice_detection_output(output) -> bool:
    return isinstance(output, dict) and output.get("voice_clock_policy") == VOICE_CLOCK_POLICY


def exclude_recording_voice(entries: list[dict]) -> list[dict]:
    """Keep interval scores available for evidence and omit them from whole-file fusion."""
    return [{**entry, "status": "not_applicable", "reason": VOICE_CLOCK_LIMIT, "aggregate": None, "unit_scores": []}
            if entry.get("family") == "voice" and entry.get("status") == "ok" else entry for entry in entries]


def limit_voice_screening(screening: dict) -> dict:
    return {**screening, "probability_ai": None, "reference_estimate_ai": None, "decision": "inconclusive",
            "scope_supported": False, "reason": VOICE_CLOCK_LIMIT}


def unscored_voice_intervals(coverage: dict, source: dict) -> list[dict]:
    """Take the exact complement of recorded voice windows on the decoded and original clocks."""
    samples = source["samples"]
    if (type(samples) is not int or not 0 < samples <= 600 * 16000
            or source["sample_rate"] != 16000 or source["timeline_time_base"] != "1/16000"):
        raise ValueError("Invalid unscored voice coverage clock")

    spans = []
    for window in coverage["windows"]:
        start, end = window["start_sample"], window["end_sample"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= samples:
            raise ValueError("A recorded voice window is outside the analysed input")
        spans.append((start, end))
    union = []
    for start, end in sorted(spans):
        if union and start <= union[-1][1]:
            union[-1] = (union[-1][0], max(end, union[-1][1]))
        else:
            union.append((start, end))

    scored = sum(end - start for start, end in union)
    if (coverage["scored_intervals"] != [{"start_sample": a, "end_sample": b} for a, b in union]
            or any(type(coverage[key]) is not int or coverage[key] != value for key, value in
                   (("decoded_samples", samples), ("scored_samples", scored), ("unscored_samples", samples - scored)))):
        raise ValueError("Voice coverage totals differ from the recorded windows")

    blocks = source["timeline"]
    ends = []
    cursor = 0
    for block in blocks:
        start, end, pts = (block[key] for key in ("start_sample", "end_sample", "source_pts"))
        if any(type(value) is not int for value in (start, end, pts)) or not start == cursor < end <= samples:
            raise ValueError("Unscored voice coverage has an invalid source timeline")
        ends.append(end)
        cursor = end
    if cursor != samples:
        raise ValueError("Unscored voice coverage source timeline is incomplete")

    gaps = []
    cursor = 0
    for start, end in [*union, (samples, samples)]:
        if cursor < start:
            ranges = []
            for index in range(bisect_right(ends, cursor), bisect_right(ends, start - 1) + 1):
                block = blocks[index]
                a, b = max(cursor, block["start_sample"]), min(start, block["end_sample"])
                offset = block["source_pts"] - block["start_sample"]
                ranges.append({"start_sample": a, "end_sample": b, "source_start_pts": offset + a,
                               "source_end_pts": offset + b, "time_base": "1/16000"})
            gaps.append({"start_sample": cursor, "end_sample": start, "source_timestamp_ranges": ranges})
        cursor = end
    return gaps


def stereo_attenuation_overlaps(gaps: list[dict], stereo: dict | None) -> list[dict]:
    """Locate where a measured mono level drop and a voice-score gap coincide."""
    from app.evidence.stereo_binding import SCHEMA

    if not stereo or stereo.get("schema") != SCHEMA:
        return []
    overlaps = []
    for window_index, window in enumerate(stereo["windows"]):
        if not window["downmix"]["strong_attenuation"]:
            continue
        for gap_index, gap in enumerate(gaps):
            start = max(gap["start_sample"], window["start_sample"])
            end = min(gap["end_sample"], window["end_sample"])
            if start >= end:
                continue
            ranges = []
            for source_range in gap["source_timestamp_ranges"]:
                left = max(start, source_range["start_sample"])
                right = min(end, source_range["end_sample"])
                if left < right:
                    offset = source_range["source_start_pts"] - source_range["start_sample"]
                    ranges.append({"start_sample": left, "end_sample": right,
                                   "source_start_pts": left + offset, "source_end_pts": right + offset,
                                   "time_base": source_range["time_base"]})
            overlaps.append({"start_sample": start, "end_sample": end, "unscored_interval_index": gap_index,
                             "stereo_window_index": window_index, "source_timestamp_ranges": ranges})
    return overlaps


def attach_stereo_attenuation_overlaps(channel: dict) -> None:
    coverage = channel["coverage"]
    overlaps = stereo_attenuation_overlaps(coverage["unscored_intervals"], channel.get("stereo_context"))
    if overlaps:
        coverage["stereo_attenuation_overlaps"] = overlaps


def source_clock_breaks(source: dict) -> list[tuple[int, int, dict, dict]]:
    """Return every material source timestamp jump in the validated PCM mapping."""
    blocks = source["timeline"]
    breaks = []
    for before, after in zip(blocks, blocks[1:]):
        previous_end = before["source_pts"] + before["end_sample"] - before["start_sample"]
        delta = after["source_pts"] - previous_end
        if abs(delta) >= _CLOCK_BREAK_MIN_SAMPLES:
            breaks.append((after["start_sample"], delta, before, after))
    return breaks


def source_clock_context(coverage: dict, source: dict, *, guarded: bool = False) -> dict:
    """Measure material source timestamp jumps in the already validated PCM mapping."""
    breaks = source_clock_breaks(source)
    positions = [position for position, _, _, _ in breaks]
    windows = coverage["windows"]
    cross_count = sum(bisect_right(positions, row["start_sample"]) < bisect_left(positions, row["end_sample"])
                      for row in windows)
    if guarded and cross_count:
        raise ValueError("A clock-guarded voice window crosses a source timestamp break")
    strongest = sorted(breaks, key=lambda item: (-abs(item[1]), item[0]))[:_CLOCK_BREAK_MAX_ROWS]

    def span(block):
        start, end, pts = block["start_sample"], block["end_sample"], block["source_pts"]
        return {"start_sample": start, "end_sample": end, "source_start_pts": pts, "source_end_pts": pts + end - start}

    return {**({"recording_voice_estimate_withheld": True} if guarded and breaks else {}),
            "affects_overall_confidence": False, "time_base": source["timeline_time_base"],
            "minimum_jump_samples": _CLOCK_BREAK_MIN_SAMPLES, "checked_samples": source["samples"],
            "total_discontinuities": len(breaks),
            "scored_windows_crossing": cross_count,
            "breaks": [{"boundary_sample": position, "jump_samples": delta, "before": span(before), "after": span(after),
                        "scored_windows_crossing": sum(row["start_sample"] < position < row["end_sample"] for row in windows)}
                       for position, delta, before, after in sorted(strongest, key=lambda item: item[0])]}


def _timeline_ranges(source: dict, start: int, end: int) -> list[dict]:
    output = []
    for block in source["timeline"]:
        a, b = max(start, block["start_sample"]), min(end, block["end_sample"])
        if a < b:
            offset = block["source_pts"] - block["start_sample"]
            output.append({"start_sample": a, "end_sample": b, "source_start_pts": a + offset,
                           "source_end_pts": b + offset, "time_base": source["timeline_time_base"]})
    return output


def validate_unscored_voice_coverage(coverage: dict, source: dict, stereo: dict | None = None) -> None:
    if not {"unscored_intervals", "stereo_attenuation_overlaps", "source_clock_context"}.intersection(coverage):
        return
    gaps = unscored_voice_intervals(coverage, source)
    if canonical_hash(coverage.get("unscored_intervals")) != canonical_hash(gaps):
        raise ValueError("Unscored voice intervals differ from their source clocks or recorded windows")
    if "source_clock_context" in coverage:
        context = coverage["source_clock_context"]
        guarded = isinstance(context, dict) and context.get("recording_voice_estimate_withheld") is True
        if canonical_hash(context) != canonical_hash(source_clock_context(coverage, source, guarded=guarded)):
            raise ValueError("Audio source clock observations differ from the exact source timeline or recorded windows")
    if "stereo_attenuation_overlaps" not in coverage:
        return
    from app.evidence.stereo_binding import SCHEMA, validate

    if not stereo or stereo.get("schema") != SCHEMA:
        raise ValueError("Unscored stereo overlaps lack their measured mono level drops")

    validate(stereo, source, ranges=lambda a, b: _timeline_ranges(source, a, b))
    expected = stereo_attenuation_overlaps(gaps, stereo)
    if not expected or canonical_hash(coverage["stereo_attenuation_overlaps"]) != canonical_hash(expected):
        raise ValueError("Unscored stereo overlaps differ from their measured windows or source clocks")


def validate_audio_channel(channel: dict, source_sha256: str) -> None:
    try:
        _validate(channel, source_sha256)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('Video voice evidence contains missing fields or invalid types') from error


def _validate(channel, source_sha256):
    def require(value, message):
        if not value:
            raise ValueError(message)

    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def sha(value):
        return bool(re.fullmatch(r'[a-f0-9]{64}', str(value)))

    def seconds(value, sample):
        # Integer sample coordinates remain authoritative. Presentation seconds
        # have the canonical JSON's six decimals after freezing; accept that
        # exact representation as well as the pre-freeze arithmetic value.
        exact = sample / 16000
        return not isinstance(value, bool) and value in (exact, round(exact, 6))

    require(not {'schema', 'version', 'id', 'sha256', 'analysis_id', 'revision', 'original_result_hash'}.intersection(channel),
            'A video voice channel cannot claim independent authority')
    source = channel['source']
    require(source['sha256'] == source_sha256, 'Video voice source differs from its parent')
    require(channel['status'] in {'ready', 'not_assessed', 'absent', 'unavailable'}, 'Invalid video voice availability')
    if channel['coverage'] is None:
        require(channel['status'] in {'absent', 'unavailable'} and not channel['packets'] and not channel.get('overview')
                and not channel.get('acoustic_context')
                and not {'stereo_context', 'stereo_context_sha256', 'repetition_context', 'similarity_context'}.intersection(channel)
                and bool(channel['reason']), 'Unavailable video voice contains unexplained assets')
        return
    samples = source['samples']
    require(integer(samples) and 0 < samples <= 120*16000 and source['sample_rate'] == 16000 and source['channels'] == 1,
            'Invalid video voice PCM dimensions')
    require(all(sha(source[k]) for k in ('pcm_sha256', 'wav_sha256', 'mapping_sha256')), 'Uncommitted video voice input')
    require(source['timeline_time_base'] == '1/16000', 'Video voice source clock differs from its PCM mapping')
    cursor = 0
    for block in source['timeline']:
        require(all(integer(block[k]) for k in ('start_sample', 'end_sample', 'source_pts'))
                and block['start_sample'] == cursor < block['end_sample'] <= samples, 'Invalid video voice source mapping')
        cursor = block['end_sample']
    require(cursor == samples, 'Video voice source mapping omits decoded samples')

    def interval(span):
        a,b = span['start_sample'], span['end_sample']
        require(integer(a) and integer(b) and 0 <= a < b <= samples, 'Voice samples are outside the analysed input')
        return a,b

    def ranges(a,b):
        return _timeline_ranges(source, a, b)

    if {'stereo_context', 'stereo_context_sha256'}.intersection(channel):
        from app.evidence.stereo_binding import validate

        stereo = channel['stereo_context']
        require(sha(channel['stereo_context_sha256']) and canonical_hash(stereo) == channel['stereo_context_sha256'],
                'Video stereo context differs from its committed measurements')
        validate(stereo, source, ranges=ranges)

    if channel.get('repetition_context') is not None:
        from app.evidence.audio_repetition_binding import validate

        validate(channel['repetition_context'], source)
    if 'similarity_context' in channel:
        from app.evidence.audio_similarity_binding import validate

        validate(channel['similarity_context'], source)

    coverage=channel['coverage']
    for window in coverage['windows']:
        interval(window)
        require(isinstance(window['score'], (float,int)) and not isinstance(window['score'],bool)
                and math.isfinite(window['score']) and 0 <= window['score'] <= 1
                and sha(window['raw_output_sha256']) and bool(window['signal_refs']), 'Invalid recorded voice response')
    require((channel['status']=='ready') == bool(coverage['windows']), 'Voice status differs from its actual windows')
    require(coverage['decoded_samples'] == samples and coverage['scored_samples']+coverage['unscored_samples']==samples,
            'Voice sample coverage does not balance')
    validate_unscored_voice_coverage(coverage, source, channel.get('stereo_context'))
    ids=set()
    for packet in channel['packets']:
        require(packet['id'].startswith('A') and packet['id'] not in ids and packet['evidence_type']=='statistical',
                'Invalid voice packet identity or semantics')
        ids.add(packet['id'])
        a,b=interval(packet['interval'])
        ca,cb=interval(packet['context_interval'])
        require(ca<=a<b<=cb and seconds(packet['start_s'],a) and seconds(packet['end_s'],b), 'Voice clock or context differs')
        anchor=packet['anchor']
        require(anchor['original_sha256']==source_sha256 and anchor['pcm_sha256']==source['pcm_sha256']
                and anchor['sample_rate']==16000 and anchor['source_timestamp_ranges']==ranges(a,b), 'Voice source anchor differs')
        require(set(packet['artifacts'])=={'selected_clip','context_clip','waveform','scores'}, 'Voice evidence lacks required assets')
        for role,start,end in (('selected_clip',a,b),('context_clip',ca,cb)):
            asset=packet['artifacts'][role]
            require(asset['media_type']=='audio/wav' and asset['sample_rate']==16000 and asset['channels']==1
                    and asset['samples']==end-start and sha(asset['sha256']), 'Voice WAV differs from its source interval')
        for observation in packet['observations']:
            require(any(w['start_sample']==a and w['end_sample']==b and w['score']==observation['value']
                        and w['raw_output_sha256']==observation['raw_output_sha256'] and w['unit_ref']==observation['unit_ref']
                        and w['signal_refs']==observation['signal_refs'] for w in coverage['windows']),
                    'Voice observation is not one of the recorded windows')
    if channel.get('acoustic_context') is not None:
        from app.evidence.audio_background_binding import validate_acoustic_context

        validate_acoustic_context(channel['acoustic_context'], source=source, voice_packets=channel['packets'])
    if channel.get('sound_events') is not None:
        from app.evidence.sound_event_binding import validate
        validate(channel['sound_events'], source)
