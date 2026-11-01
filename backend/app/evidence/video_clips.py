"""Source-derived, silent playback context with an exact recorded frame clock."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from app.detectors.video.decode import FrameTable
from app.evidence.mp4 import MAX_BYTES, MAX_FRAMES, inspect_mp4

VERSION = 'video-context-clip/1'
MAX_SECONDS = 8
MAX_SIDE = 960
TIMEOUT_SECONDS = 90
_FRAME = re.compile(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)')
_BASE = re.compile(r'config in time_base:\s*(\d+)/(\d+)')


class ClipUnavailable(ValueError):
    """A bounded source clip cannot be exported with a trustworthy timeline."""


def select_range(table: FrameTable, focus_index: int) -> tuple[int, int]:
    if not 0 <= focus_index < len(table):
        raise ClipUnavailable('The selected frame is outside the recorded source')
    if any(b <= a for a, b in zip(table.pts, table.pts[1:])):
        raise ClipUnavailable('The source timestamps are not strictly increasing')
    middle = table.pts[focus_index] * table.tb
    half = Fraction(MAX_SECONDS, 2)
    indices = [i for i, pts in enumerate(table.pts) if middle - half <= pts * table.tb < middle + half]
    start, end = indices[0], indices[-1] + 1
    if end - start > MAX_FRAMES:
        start = max(start, focus_index - MAX_FRAMES // 2)
        end = min(end, start + MAX_FRAMES)
    if set(table.interpolated).intersection(range(start, end)):
        raise ClipUnavailable('A source timestamp in this context interval was interpolated')
    return start, end


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_clip(path: Path, table: FrameTable, *, source_sha256: str, focus_index: int,
               sampled_frame_indices: list[int], timeline_origin_s: float | None) -> tuple[bytes, dict, dict]:
    """Return MP4 bytes, intrinsic asset metadata and its public source derivation.

    FFmpeg's pre-transform showinfo PTS must match every frozen source-table frame.
    The MP4 sample table is then checked independently with a standard-library
    reader. Compressed output is an explicitly lossy derivative; exact analysed
    RGB pixels remain the separate source PNGs.

    sampled_frame_indices lists this packet's selected frame bindings (or all
    inputs of its selected window) inside the clip, not every check's samples.
    """
    binary = shutil.which('ffmpeg')
    if binary is None:
        raise ClipUnavailable('A video transcoder is not available on this server')
    if _sha(path) != source_sha256:
        raise ClipUnavailable('The original video no longer matches its recorded hash')
    start, end = select_range(table, focus_index)
    tb = table.tb
    if tb.denominator > 2_147_483_647:
        raise ClipUnavailable('The source clock cannot be represented by this clip format')
    ratio = min(1.0, MAX_SIDE / max(table.width, table.height))
    width, height = max(2, int(table.width * ratio) // 2 * 2), max(2, int(table.height * ratio) // 2 * 2)
    # MP4 needs a final sample duration although a frame table records instants,
    # not end times. Set that duration explicitly and disclose its source.
    if end < len(table):
        last_delta = (table.pts[end] - table.pts[end - 1]) * tb.numerator
        duration_basis = 'next_source_frame_pts'
    elif end - start > 1:
        last_delta = (table.pts[end - 1] - table.pts[end - 2]) * tb.numerator
        duration_basis = 'inferred_previous_source_interval'
    else:
        last_delta = max(1, round(tb.denominator / 25))
        duration_basis = 'inferred_25fps_single_frame'
    with tempfile.TemporaryDirectory(prefix='chipmunk-video-clip-') as temporary:
        output, log = Path(temporary) / 'context.mp4', Path(temporary) / 'decode.log'
        filters = f'trim=start_frame={start}:end_frame={end},showinfo,setpts=PTS-STARTPTS,scale={width}:{height}:flags=lanczos,setsar=1'
        argv = [binary, '-nostdin', '-hide_banner', '-v', 'info', '-protocol_whitelist', 'file', '-copyts', '-i', str(path.resolve()),
                '-map', '0:v:0', '-an', '-sn', '-dn', '-map_metadata', '-1', '-map_chapters', '-1',
                '-vf', filters, '-fps_mode', 'vfr', '-enc_time_base:v', f'1:{tb.denominator}',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-bf', '0',
                '-bsf:v', f'setts=duration={last_delta}',
                '-threads', '2', '-video_track_timescale', str(tb.denominator), '-use_editlist', '0',
                '-movflags', '+faststart', '-metadata:s:v:0', 'rotate=0', '-metadata', 'title=',
                '-f', 'mp4', str(output)]
        try:
            with log.open('wb') as diagnostic:
                result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=diagnostic, timeout=TIMEOUT_SECONDS, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ClipUnavailable('The bounded context clip could not be encoded in time') from error
        if result.returncode or not output.is_file() or output.stat().st_size > MAX_BYTES or log.stat().st_size > 4 * 1024 * 1024:
            raise ClipUnavailable('The bounded context clip could not be encoded')
        diagnostic = log.read_text(errors='replace')
        observed = [(int(a), int(b)) for a, b in _FRAME.findall(diagnostic)]
        bases = [Fraction(int(a), int(b)) for a, b in _BASE.findall(diagnostic)]
        expected = [(i - start, table.pts[i]) for i in range(start, end)]
        if observed != expected or not bases or any(base != tb for base in bases):
            raise ClipUnavailable('Decoded context timestamps differ from the frozen source table')
        data = output.read_bytes()
    try:
        encoded = inspect_mp4(data)
    except ValueError as error:
        raise ClipUnavailable('The encoded clip has an unsupported media structure') from error
    pts = [(pts - table.pts[start]) * tb.numerator for pts in table.pts[start:end]]
    if (encoded['pts'] != pts or encoded['frame_count'] != end - start or encoded['time_base'] != f'1/{tb.denominator}'
            or (encoded['width'], encoded['height']) != (width, height)):
        raise ClipUnavailable('The encoded context did not preserve the source frame clock')
    if _sha(path) != source_sha256:
        raise ClipUnavailable('The original video changed during context export')
    next_delta = (table.pts[end] - table.pts[end - 1]) * tb.numerator if end < len(table) else None
    metadata = {key: encoded[key] for key in ('width', 'height', 'frame_count', 'time_base', 'duration_s', 'duration_ticks', 'codec', 'has_audio')}
    metadata['pts'] = encoded['pts']
    metadata['durations'] = encoded['durations']
    description = {
        'status': 'ready', 'artifact_ref': 'context_clip', 'source_sha256': source_sha256,
        'source_time_base': table.time_base, 'start_frame_index': start, 'end_frame_index_exclusive': end,
        'source_pts': list(table.pts[start:end]), 'clip_pts': encoded['pts'], 'clip_time_base': encoded['time_base'],
        'start_stream_s': table.seconds(start), 'last_stream_s': table.seconds(end - 1),
        'start_playback_s': None if timeline_origin_s is None else table.seconds(start) - timeline_origin_s,
        'sampled_frame_indices': [i for i in sampled_frame_indices if start <= i < end],
        'last_frame_duration': {'ticks': encoded['durations'][-1], 'time_base': encoded['time_base'],
                                'basis': duration_basis, 'next_source_frame_delta_ticks': next_delta,
                                'matches_next_source_frame': None if next_delta is None else next_delta == encoded['durations'][-1]},
        'transform': {'version': VERSION, 'reencoded': True, 'codec': 'h264', 'pixel_format': 'yuv420p',
                      'audio': 'omitted', 'timestamp_rule': 'subtract_first_source_pts', 'source_metadata': 'not_copied',
                      'source_width': table.width, 'source_height': table.height, 'width': width, 'height': height,
                      'scaling': 'lanczos_to_even_dimensions', 'sample_aspect_ratio': '1:1'},
        'limitations': ['This is a re-encoded, silent context clip, not the original compressed bytes or exact analysed pixels.',
                        'All clip frames are playback context. Only separately recorded sampled frames/windows carry analysis measurements.',
                        'The last displayed frame duration uses the next source-frame interval when available, otherwise an explicitly inferred duration.'],
    }
    return data, metadata, description
