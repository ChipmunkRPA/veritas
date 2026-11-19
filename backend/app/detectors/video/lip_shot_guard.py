"""Additional short-shot admission for the exact 25 fps speech comparison."""

from __future__ import annotations

from contextlib import closing
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import shots
from app.detectors.video.decode import DecodeError, decode_stream
from app.detectors.video.face_return import source_carrier
from app.detectors.video.frames import probe
from app.evidence import video_lip_agreement as agreement


def qualified_delivery(path: Path, table) -> bool:
    if (table.truncated or table.interpolated or table.rotation
            or (table.width, table.height) != (table.coded_width, table.coded_height)
            or not agreement.native25_picture(table.pts, table.time_base, table.width, table.height)):
        return False
    delivery = probe(path, include_codec_tag=True)
    return (source_carrier(path) == 'mp4' and delivery.get('timeline_origin_s') == 0
            and (delivery.get('codec'), delivery.get('pix_fmt'), delivery.get('codec_tag'))
            == ('h264', 'yuv420p', 'avc1'))


def _gray_view(rgb):
    tiny = cv2.resize(rgb, (64, 51), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(tiny, cv2.COLOR_RGB2GRAY)[5:-5, 7:-7]


def _difference(first, second):
    a, ha = first
    b, hb = second
    return float(np.abs(a - b).mean()), float(1. - np.minimum(ha, hb).sum())


def _structure_alignment(first, second):
    margin = 4
    template = first[margin:-margin, margin:-margin]
    if float(template.std()) < 2.:
        return 1.
    return float(cv2.matchTemplate(second, template, cv2.TM_CCOEFF_NORMED).max())


def _supplemental(hsv, gray, index, content, histogram):
    if content < 13. or histogram < .17:
        return False
    flanks = [_difference(hsv[a], hsv[b]) for a, b in (
        (index - 3, index - 2), (index - 2, index - 1), (index, index + 1), (index + 1, index + 2),
    )]
    if content < 2.4 * max(c for c, _ in flanks) or histogram < 2. * max(h for _, h in flanks):
        return False
    support = 0
    for a, b in product(range(index - 3, index), range(index, index + 3)):
        c, h = _difference(hsv[a], hsv[b])
        support += c >= .72 * content and h >= .72 * histogram
    return support >= 5 and _structure_alignment(gray[index - 1], gray[index]) < .94


def scan(path: Path, table, *, timeout_s: float = 8.) -> dict:
    """Verify every native frame and add guarded boundaries to the incumbent scan."""
    hsv, gray = [], []
    legacy = shots.ShotDetector(table)
    with closing(decode_stream(path, table, max_side=None, timeout_s=timeout_s)) as decoded:
        for frame in decoded:
            index = len(hsv)
            if index >= len(table) or frame.index != index or not frame.pts_verified or frame.pts != table.pts[index]:
                raise DecodeError('Shot review requires every exact native source timestamp')
            legacy.update(index, frame.rgb)
            hsv.append(shots._features(frame.rgb))
            gray.append(_gray_view(frame.rgb))
    if len(hsv) != agreement.NATIVE25_FRAMES:
        raise DecodeError('Shot review could not read every original frame')
    additions = [i for i in range(3, len(hsv) - 2) if _supplemental(hsv, gray, i, *_difference(hsv[i - 1], hsv[i]))]
    return agreement.shot_guard_record(sorted({b['frame_index'] for b in legacy.boundaries} | set(additions)))
