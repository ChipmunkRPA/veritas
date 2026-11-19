# SPDX-License-Identifier: Apache-2.0
"""Verify the native MP4 video timing used for short face-transition evidence."""

from __future__ import annotations

from fractions import Fraction
from itertools import accumulate
from pathlib import Path
from struct import unpack_from

from app.evidence.mp4 import InvalidMP4, bounded_boxes

IDENTITY_TRANSFORM = (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidMP4(message)


def _clock(header: bytes) -> tuple[int, int]:
    _require(header[1:4] == bytes(3) and header[0] in (0, 1), "Invalid MP4 source clock header")
    if header[0] == 0:
        _require(len(header) >= 20, "Truncated MP4 source clock header")
        return unpack_from(">II", header, 12)
    _require(len(header) >= 32, "Truncated MP4 source clock header")
    return unpack_from(">IQ", header, 20)


def _track_duration(header: bytes, width: int, height: int) -> int:
    _require(len(header) in (84, 96) and header[0] == (len(header) == 96), "Invalid MP4 video track header")
    _require(
        unpack_from(">9i", header, len(header) - 44) == IDENTITY_TRANSFORM
        and unpack_from(">II", header, len(header) - 8) == (width << 16, height << 16),
        "MP4 video display differs from the decoded geometry",
    )
    return unpack_from(">Q", header, 28)[0] if header[0] else unpack_from(">I", header, 20)[0]


def _runs(data: bytes, frames: int, *, composition: bool = False) -> list[int]:
    _require(
        len(data) >= 8 and data[1:4] == bytes(3) and data[0] in ((0, 1) if composition else (0,)),
        "Invalid MP4 video timing table",
    )
    count = unpack_from(">I", data, 4)[0]
    _require(1 <= count <= frames and len(data) == 8 + 8 * count, "Invalid MP4 video timing runs")
    fmt = ">Ii" if data[0] else ">II"
    values = []
    for i in range(count):
        run, ticks = unpack_from(fmt, data, 8 + 8 * i)
        _require(0 < run <= frames - len(values), "MP4 video timing exceeds the decoded frame count")
        values.extend([ticks] * run)
    _require(len(values) == frames, "MP4 video timing differs from the decoded frame count")
    return values


def _edit(data: bytes) -> tuple[int, int]:
    _require(
        len(data) in (20, 28)
        and data[:4] == bytes((len(data) == 28, 0, 0, 0))
        and unpack_from(">I", data, 4)[0] == 1,
        "MP4 video requires a single complete edit",
    )
    duration, start, rate, fraction = unpack_from(">QqhH" if data[0] else ">IihH", data, 8)
    _require(
        start >= 0 and (rate, fraction) == (1, 0), "MP4 video edit has an empty or changed-rate interval"
    )
    return duration, start


def _check(stream, *, time_base: Fraction, ticks: int, frames: int, width: int, height: int,
           video_only: bool = False) -> None:
    size, read, boxes, one, payload = bounded_boxes(stream, label="MP4 source video")

    def children(box):
        return boxes(box.start, box.end)

    top = boxes(0, size)
    brands = payload(one(top, b"ftyp"))
    _require(
        len(brands) >= 12
        and len(brands) % 4 == 0
        and brands[:4] in {b"isom", b"mp41", b"mp42", b"iso4"}
        and any(brands[i : i + 4] in {b"mp41", b"mp42"} for i in range(8, len(brands), 4)),
        "Unsupported MP4 source video format",
    )
    _require(all(box.kind != b"moof" for box in top), "Fragmented MP4 video is outside this clock check")
    one(top, b"mdat")
    movie = children(one(top, b"moov"))
    _require(one(movie, b"mvex", optional=True) is None, "Fragmented MP4 video is outside this clock check")
    movie_scale, movie_ticks = _clock(payload(one(movie, b"mvhd")))
    span = time_base * ticks * frames
    _require(movie_scale > 0 and movie_ticks >= span * movie_scale, "MP4 movie ends before the video")

    tracks = [box for box in movie if box.kind == b"trak"]
    _require(not video_only or len(tracks) == 1, "The source must have only one video track")
    videos = []
    for trak in tracks:
        track = children(trak)
        media = children(one(track, b"mdia"))
        handler = payload(one(media, b"hdlr"))
        _require(len(handler) >= 12, "Truncated MP4 media handler")
        if handler[8:12] == b"vide":
            videos.append((track, media))
    _require(len(videos) == 1, "MP4 must have one unambiguous video track")
    track, media = videos[0]
    scale, duration = _clock(payload(one(media, b"mdhd")))
    _require(
        time_base * scale == 1
        and duration == ticks * frames
        and _track_duration(payload(one(track, b"tkhd")), width, height) == span * movie_scale,
        "Declared MP4 video duration or clock differs from the decoded profile",
    )

    information = children(one(media, b"minf"))
    reference = one(children(one(information, b"dinf")), b"dref")
    _require(
        reference.end - reference.start >= 8 and read(reference.start, 8) == bytes(7) + b"\1",
        "MP4 video must use one local media reference",
    )
    locations = boxes(reference.start + 8, reference.end)
    _require(
        len(locations) == 1 and locations[0].kind == b"url " and payload(locations[0]) == bytes(3) + b"\1",
        "MP4 video cannot reference external media",
    )
    tables = children(one(information, b"stbl"))
    _require(one(tables, b"cslg", optional=True) is None, "Unsupported MP4 composition shift")
    durations = _runs(payload(one(tables, b"stts")), frames)
    _require(all(duration == ticks for duration in durations), "MP4 video sample timing is variable")
    composition = one(tables, b"ctts", optional=True)
    offsets = _runs(payload(composition), frames, composition=True) if composition else [0] * frames
    native = sorted(
        dts + offset for dts, offset in zip(accumulate((0, *durations[:-1])), offsets, strict=True)
    )

    edit = one(track, b"edts", optional=True)
    if brands[:4] == b"iso4":
        _require(
            composition is not None and read(composition.start, 1) == b"\1" and edit is None,
            "This MP4 brand requires signed composition timing and no edit list",
        )
    origin = 0
    if edit is not None:
        edits = children(edit)
        _require(len(edits) == 1, "Ambiguous MP4 video edits")
        length, origin = _edit(payload(one(edits, b"elst")))
        _require(length == span * movie_scale, "MP4 video edit does not cover its exact duration")
    _require(
        all(pts == origin + index * ticks for index, pts in enumerate(native)),
        "MP4 video presentation clock differs from its complete decoded interval",
    )


def verified(path: Path, table, ticks: int, *, video_only: bool = False) -> bool:
    """Check an independently decoded profile against bounded native timing boxes."""
    try:
        with path.open("rb") as stream:
            _check(
                stream,
                time_base=table.tb,
                ticks=ticks,
                frames=len(table),
                width=table.width,
                height=table.height,
                video_only=video_only,
            )
    except (InvalidMP4, OSError):
        return False
    return True
