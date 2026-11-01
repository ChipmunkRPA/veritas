"""Bounded, standard-library inspection of the silent context-clip MP4 format.

This is a narrow integrity reader, not a general MP4 decoder. It accepts a single
unrotated AVC video track with explicit sample sizes and positive decode deltas,
and rejects edits, composition offsets, fragmented files and additional tracks.
No compressed media payload is read or allocated. Usable offline without models,
settings, databases, FFmpeg or application services.
"""
from __future__ import annotations

import io
import math
import re
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import BinaryIO

MAX_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 480
MAX_BOXES = 512
MAX_METADATA = 256 * 1024


class InvalidMP4(ValueError):
    pass


def validate_clip_binding(packet: dict, source: dict) -> None:
    """Validate a frozen clip's public source/asset clocks without decoding media.

    The caller independently checks asset bytes with ``inspect_mp4``. Full
    source-table equality is an additional server-side check; an offline public
    package proves the signed correspondence, not detector accuracy.
    """
    try:
        _validate_clip_binding(packet, source)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise InvalidMP4('Context clip binding has invalid field types or missing fields') from error


def _validate_clip_binding(packet: dict, source: dict) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise InvalidMP4(message)

    def integer(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and abs(value) < 2 ** 63

    def near(value, expected) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and abs(value - float(expected)) <= 0.000002

    clip = packet.get('clip')
    videos = {key: value for key, value in packet.get('artifacts', {}).items() if value.get('media_type') == 'video/mp4'}
    if clip is None:
        require(not videos, 'A context clip asset has no source binding')
        return  # Immutable v1 packets do not have clips.
    require(isinstance(clip, dict), 'Context clip binding is invalid')
    if clip.get('status') == 'unavailable':
        require(not videos and isinstance(clip.get('reason'), str) and bool(clip['reason']), 'Unavailable context clip has media or no explanation')
        return
    require(clip.get('status') == 'ready', 'Context clip status is invalid')
    role = 'context_clip_video' if packet.get('claim_scope') == 'sampled_av_window' else 'context_clip'
    require(clip.get('artifact_ref') == role and set(videos) == {role}, 'Context clip has an ambiguous asset reference')
    asset = videos[role]
    require(bool(re.fullmatch(r'[a-f0-9]{64}', str(asset.get('sha256', '')))), 'Context clip has no content hash')
    require(clip.get('source_sha256') == source.get('sha256') and clip.get('source_time_base') == source.get('time_base'), 'Context clip refers to another source')
    try:
        source_tb, clip_tb = Fraction(source['time_base']), Fraction(clip['clip_time_base'])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise InvalidMP4('Context clip clock is invalid') from error
    require(source_tb > 0 and clip_tb == Fraction(1, source_tb.denominator) and clip.get('clip_time_base') == asset.get('time_base'), 'Context clip uses a different clock scale')
    start, end = clip.get('start_frame_index'), clip.get('end_frame_index_exclusive')
    require(integer(start) and integer(end) and integer(source.get('frame_count')) and 0 <= start < end <= source['frame_count'] and end - start <= MAX_FRAMES, 'Context clip source frame range is invalid')
    source_pts, clip_pts = clip.get('source_pts'), clip.get('clip_pts')
    require(isinstance(source_pts, list) and len(source_pts) == end - start and all(integer(v) for v in source_pts), 'Context clip source frame clock is invalid')
    require(all(b > a for a, b in zip(source_pts, source_pts[1:])), 'Context clip source frame clock is not increasing')
    require(isinstance(clip_pts, list) and all(integer(v) for v in clip_pts) and clip_pts == [(v - source_pts[0]) * source_tb.numerator for v in source_pts] and clip_pts == asset.get('pts'), 'Context clip timestamps do not preserve source instants')
    durations = asset.get('durations')
    require(isinstance(durations, list) and len(durations) == end - start and all(integer(v) and v > 0 for v in durations), 'Context clip sample durations are invalid')
    require(durations[:-1] == [b - a for a, b in zip(clip_pts, clip_pts[1:])] and integer(asset.get('duration_ticks')) and sum(durations) == asset['duration_ticks'], 'Context clip durations differ from its frame clock')
    require(asset.get('frame_count') == end - start and asset.get('codec') == 'h264' and asset.get('has_audio') is False and near(asset.get('duration_s'), sum(durations) * clip_tb), 'Context clip media dimensions differ from its binding')
    require(clip_pts[-1] * clip_tb < 8 and sum(durations) * clip_tb <= 20, 'Context clip exceeds the bounded interval')
    require(near(clip.get('start_stream_s'), source_pts[0] * source_tb) and near(clip.get('last_stream_s'), source_pts[-1] * source_tb), 'Context clip source times differ from its exact clock')
    origin = source.get('timeline_origin_s')
    require(clip.get('start_playback_s') is None if origin is None else near(clip.get('start_playback_s'), source_pts[0] * source_tb - origin), 'Context clip playback origin differs from its source')
    transform = clip.get('transform', {})
    expected = {'version': 'video-context-clip/1', 'reencoded': True, 'codec': 'h264', 'pixel_format': 'yuv420p', 'audio': 'omitted',
                'timestamp_rule': 'subtract_first_source_pts', 'source_metadata': 'not_copied', 'scaling': 'lanczos_to_even_dimensions', 'sample_aspect_ratio': '1:1'}
    require(all(transform.get(k) == v for k, v in expected.items()), 'Context clip transformation is not declared accurately')
    require(all(integer(source.get(k)) and source[k] > 0 for k in ('width', 'height')), 'Context clip source dimensions are invalid')
    scale = min(1.0, 960 / max(source['width'], source['height']))
    for key in ('width', 'height'):
        require(transform.get('source_' + key) == source[key] and transform.get(key) == asset.get(key) == max(2, int(source[key] * scale) // 2 * 2), 'Context clip scaling differs from its source geometry')
    last = clip.get('last_frame_duration', {})
    require(last.get('ticks') == durations[-1] and last.get('time_base') == asset['time_base'], 'Context clip final duration differs from its media')
    if end < source['frame_count']:
        require(last.get('basis') == 'next_source_frame_pts' and last.get('next_source_frame_delta_ticks') == durations[-1] and last.get('matches_next_source_frame') is True, 'Context clip final duration is not bound to the next frame')
    else:
        expected_last = durations[-2] if len(durations) > 1 else max(1, round(source_tb.denominator / 25))
        basis = 'inferred_previous_source_interval' if len(durations) > 1 else 'inferred_25fps_single_frame'
        require(last.get('basis') == basis and last.get('next_source_frame_delta_ticks') is None and last.get('matches_next_source_frame') is None and durations[-1] == expected_last, 'Context clip inferred final duration is not disclosed accurately')
    measured = packet.get('window', {}).get('frame_indices') or [frame['anchor']['frame_index'] for frame in packet.get('frames', [])]
    expected_samples = [i for i in measured if start <= i < end]
    require(clip.get('sampled_frame_indices') == expected_samples, 'Context clip confuses playback frames with measured samples')
    for frame in packet.get('frames', []):
        anchor = frame['anchor']
        index = anchor['frame_index']
        if start <= index < end:
            require(anchor.get('pts') == source_pts[index - start], 'Context clip disagrees with an analysed source frame')


@dataclass(frozen=True)
class Box:
    kind: bytes
    start: int
    end: int


def inspect_mp4(source: bytes | BinaryIO) -> dict:
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    stream.seek(0, 2)
    size = stream.tell()
    if not 32 <= size <= MAX_BYTES:
        raise InvalidMP4('Context clip has an unsupported size')
    count = 0

    def read(offset: int, length: int) -> bytes:
        if length < 0 or length > MAX_METADATA or offset < 0 or offset + length > size:
            raise InvalidMP4('Context clip metadata exceeds its bounds')
        stream.seek(offset)
        data = stream.read(length)
        if len(data) != length:
            raise InvalidMP4('Context clip metadata is truncated')
        return data

    def boxes(start: int, end: int) -> list[Box]:
        nonlocal count
        found = []
        while start < end:
            count += 1
            if count > MAX_BOXES or end - start < 8:
                raise InvalidMP4('Context clip has invalid box boundaries')
            length, kind = struct.unpack('>I4s', read(start, 8))
            header = 8
            if length == 1:
                if end - start < 16:
                    raise InvalidMP4('Context clip large box is truncated')
                length, header = struct.unpack('>Q', read(start + 8, 8))[0], 16
            elif length == 0:
                length = end - start
            if length < header or start + length > end:
                raise InvalidMP4('Context clip box exceeds its parent')
            found.append(Box(kind, start + header, start + length))
            start += length
        return found

    def one(children: list[Box], kind: bytes) -> Box:
        selected = [box for box in children if box.kind == kind]
        if len(selected) != 1:
            raise InvalidMP4('Context clip has a missing or duplicate required box')
        return selected[0]

    def payload(box: Box) -> bytes:
        return read(box.start, box.end - box.start)

    top = boxes(0, size)
    if any(box.kind not in {b'ftyp', b'moov', b'mdat', b'free'} for box in top):
        raise InvalidMP4('Context clip has unsupported top-level content')
    ftyp = payload(one(top, b'ftyp'))
    if len(ftyp) < 8 or len(ftyp) % 4 or not {ftyp[:4], *(ftyp[i:i + 4] for i in range(8, len(ftyp), 4))}.intersection({b'isom', b'iso2', b'mp41', b'mp42', b'avc1'}):
        raise InvalidMP4('Context clip does not declare an MP4 video format')
    media = one(top, b'mdat')
    moov = one(top, b'moov')
    movie = boxes(moov.start, moov.end)
    if any(box.kind not in {b'mvhd', b'trak', b'udta'} for box in movie):
        raise InvalidMP4('Context clip has unsupported movie metadata')
    track = one(movie, b'trak')
    track_boxes = boxes(track.start, track.end)
    if any(box.kind not in {b'tkhd', b'mdia'} for box in track_boxes):
        raise InvalidMP4('Context clip track uses unsupported edits or offsets')
    tkhd = payload(one(track_boxes, b'tkhd'))
    if len(tkhd) not in (84, 96) or tkhd[0] not in (0, 1):
        raise InvalidMP4('Context clip track header is invalid')
    if struct.unpack('>9i', tkhd[-44:-8]) != (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824):
        raise InvalidMP4('Context clip still has an unhandled display transform')
    width_fixed, height_fixed = struct.unpack('>II', tkhd[-8:])
    if width_fixed % 65536 or height_fixed % 65536:
        raise InvalidMP4('Context clip has non-integral display dimensions')
    width, height = width_fixed // 65536, height_fixed // 65536
    if min(width, height) < 2 or max(width, height) > 960:
        raise InvalidMP4('Context clip dimensions exceed the export limit')
    mdia = one(track_boxes, b'mdia')
    media_boxes = boxes(mdia.start, mdia.end)
    if any(box.kind not in {b'mdhd', b'hdlr', b'minf'} for box in media_boxes):
        raise InvalidMP4('Context clip has unsupported media metadata')
    hdlr = payload(one(media_boxes, b'hdlr'))
    if len(hdlr) < 12 or hdlr[8:12] != b'vide':
        raise InvalidMP4('Context clip must contain only one video track and no audio')
    mdhd = payload(one(media_boxes, b'mdhd'))
    if len(mdhd) < 24 or mdhd[0] not in (0, 1):
        raise InvalidMP4('Context clip media clock is invalid')
    if mdhd[0] == 0:
        timescale, duration = struct.unpack('>II', mdhd[12:20])
    else:
        if len(mdhd) < 36:
            raise InvalidMP4('Context clip media clock is truncated')
        timescale, duration = struct.unpack('>IQ', mdhd[20:32])
    if not 1 <= timescale <= 2_147_483_647 or not 0 < duration <= 20 * timescale:
        raise InvalidMP4('Context clip has an unsupported duration or time base')
    minf = one(media_boxes, b'minf')
    media_information = boxes(minf.start, minf.end)
    if any(box.kind not in {b'vmhd', b'dinf', b'stbl'} for box in media_information):
        raise InvalidMP4('Context clip has unsupported media references')
    dinf = one(media_information, b'dinf')
    dref = one(boxes(dinf.start, dinf.end), b'dref')
    if read(dref.start, 8) != b'\0\0\0\0\0\0\0\1':
        raise InvalidMP4('Context clip has ambiguous media references')
    references = boxes(dref.start + 8, dref.end)
    if len(references) != 1 or references[0].kind != b'url ' or payload(references[0]) != b'\0\0\0\1':
        raise InvalidMP4('Context clip references external media')
    stbl = one(media_information, b'stbl')
    tables = boxes(stbl.start, stbl.end)
    if any(box.kind not in {b'stsd', b'stts', b'stss', b'stsc', b'stsz', b'stco', b'co64'} for box in tables):
        raise InvalidMP4('Context clip has unsupported composition offsets or sample tables')
    stsd = one(tables, b'stsd')
    if read(stsd.start, 8) != b'\0\0\0\0\0\0\0\1':
        raise InvalidMP4('Context clip needs one AVC sample description')
    descriptions = boxes(stsd.start + 8, stsd.end)
    if len(descriptions) != 1:
        raise InvalidMP4('Context clip has extra sample descriptions')
    entry = one(descriptions, b'avc1')
    if entry.end - entry.start < 78 or read(entry.start + 6, 2) != b'\0\1' or struct.unpack('>HH', read(entry.start + 24, 4)) != (width, height):
        raise InvalidMP4('Context clip coded and display dimensions differ')
    avcc = payload(one(boxes(entry.start + 78, entry.end), b'avcC'))
    if len(avcc) < 7 or avcc[0] != 1 or avcc[4] & 3 != 3:
        raise InvalidMP4('Context clip AVC configuration is unsupported')
    sizes = payload(one(tables, b'stsz'))
    if len(sizes) < 12 or sizes[:4] != b'\0' * 4:
        raise InvalidMP4('Context clip sample-size table is invalid')
    fixed, frames = struct.unpack('>II', sizes[4:12])
    if not 1 <= frames <= MAX_FRAMES or len(sizes) != 12 + (0 if fixed else frames * 4):
        raise InvalidMP4('Context clip sample count exceeds the export limit')
    sample_sizes = [fixed] * frames if fixed else list(struct.unpack(f'>{frames}I', sizes[12:]))
    if any(value <= 0 for value in sample_sizes) or sum(sample_sizes) > media.end - media.start:
        raise InvalidMP4('Context clip samples exceed its compressed-media payload')
    stts = payload(one(tables, b'stts'))
    if len(stts) < 8 or stts[:4] != b'\0' * 4:
        raise InvalidMP4('Context clip timestamp table is invalid')
    entries = struct.unpack('>I', stts[4:8])[0]
    if not 1 <= entries <= frames or len(stts) != 8 + 8 * entries:
        raise InvalidMP4('Context clip timestamp entries exceed their bounds')
    durations = []
    for index in range(entries):
        samples, delta = struct.unpack('>II', stts[8 + 8 * index:16 + 8 * index])
        if samples < 1 or delta < 1 or len(durations) + samples > frames:
            raise InvalidMP4('Context clip timestamps are not strictly increasing')
        durations.extend([delta] * samples)
    if len(durations) != frames or sum(durations) != duration:
        raise InvalidMP4('Context clip durations do not match its media clock')
    offsets_boxes = [box for box in tables if box.kind in {b'stco', b'co64'}]
    if len(offsets_boxes) != 1:
        raise InvalidMP4('Context clip has ambiguous media offsets')
    offset_box = offsets_boxes[0]
    offsets_data = payload(offset_box)
    if len(offsets_data) < 8 or offsets_data[:4] != b'\0' * 4:
        raise InvalidMP4('Context clip offset table is invalid')
    chunks = struct.unpack('>I', offsets_data[4:8])[0]
    stride, fmt = (4, 'I') if offset_box.kind == b'stco' else (8, 'Q')
    if not 1 <= chunks <= frames or len(offsets_data) != 8 + chunks * stride:
        raise InvalidMP4('Context clip chunk offsets exceed their bounds')
    offsets = list(struct.unpack(f'>{chunks}{fmt}', offsets_data[8:]))
    if offsets != sorted(set(offsets)) or any(not media.start <= offset < media.end for offset in offsets):
        raise InvalidMP4('Context clip chunk points outside its compressed-media payload')
    sample_chunks = payload(one(tables, b'stsc'))
    if len(sample_chunks) < 8 or sample_chunks[:4] != b'\0' * 4:
        raise InvalidMP4('Context clip sample/chunk table is invalid')
    entry_count = struct.unpack('>I', sample_chunks[4:8])[0]
    if not 1 <= entry_count <= chunks or len(sample_chunks) != 8 + 12 * entry_count:
        raise InvalidMP4('Context clip sample/chunk entries exceed their bounds')
    runs = [struct.unpack('>III', sample_chunks[8 + i * 12:20 + i * 12]) for i in range(entry_count)]
    if runs[0][0] != 1 or [r[0] for r in runs] != sorted({r[0] for r in runs}) or any(not 1 <= r[0] <= chunks or r[1] < 1 or r[2] != 1 for r in runs):
        raise InvalidMP4('Context clip sample/chunk mapping is invalid')
    cursor = 0
    for chunk_index, offset in enumerate(offsets, 1):
        run = next(run for run in reversed(runs) if run[0] <= chunk_index)
        if cursor + run[1] > frames:
            raise InvalidMP4('Context clip maps too many compressed samples')
        end = offset + sum(sample_sizes[cursor:cursor + run[1]])
        if end > (offsets[chunk_index] if chunk_index < chunks else media.end):
            raise InvalidMP4('Context clip compressed chunks overlap or exceed their payload')
        cursor += run[1]
    if cursor != frames:
        raise InvalidMP4('Context clip leaves compressed samples unmapped')
    pts, clock = [], 0
    for delta in durations:
        pts.append(clock)
        clock += delta
    return {'width': width, 'height': height, 'frame_count': frames, 'time_base': f'1/{timescale}',
            'pts': pts, 'durations': durations, 'duration_ticks': duration,
            'duration_s': duration / timescale, 'codec': 'h264', 'has_audio': False}
