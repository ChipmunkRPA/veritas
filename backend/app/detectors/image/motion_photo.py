# SPDX-License-Identifier: Apache-2.0
"""Check the still against the declared frame and its neighbors in a JPEG Motion Photo."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import ParseError

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.detectors.image import jpeg_bitstream as jpeg
from app.detectors.image.reference_crop_resize import matching_points
from app.detectors.video.decode import decode_frames, probe_frames
from app.provenance.source_labels import InvalidMetadata, _xml

XMP = b"http://ns.adobe.com/xap/1.0/\0"
CAMERA = "{http://ns.google.com/photos/1.0/camera/}"
CONTAINER = "{http://ns.google.com/photos/1.0/container/}"
ITEM = "{http://ns.google.com/photos/1.0/container/item/}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
MAX_INPUT = 32 * 1024 * 1024
MAX_SIDE = 768
MAX_PIXELS = 24_000_000
MAX_FRAME_PIXELS = 2560 * 1920
MAX_VIDEO_BYTES = 16 * 1024 * 1024
POINT = .45


@dataclass(frozen=True)
class Carrier:
    image_end: int
    video_start: int
    timestamp_us: int
    xmp: bytes


@dataclass(frozen=True)
class Measurement:
    clock: dict
    still: np.ndarray
    frame: np.ndarray
    original_size: tuple[int, int]
    frame_transform: np.ndarray
    compared_pixels: int
    region: tuple[int, int, int, int] | None
    mask: np.ndarray | None


def _segments(data):
    if data[:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG")
    pos, scanning = 2, False
    for _ in range(256):
        if scanning:
            marker_at = jpeg._ENTROPY_MARKER.search(data, pos)
            if marker_at is None:
                break
            pos = marker_at.start()
        if pos >= len(data) or data[pos] != 255:
            break
        while pos < len(data) and data[pos] == 255:
            pos += 1
        if pos == len(data):
            break
        marker = data[pos]
        pos += 1
        if marker == jpeg.EOI:
            yield marker, pos, pos
            return
        if marker in jpeg.STANDALONE or pos + 2 > len(data):
            break
        size = int.from_bytes(data[pos:pos + 2], "big")
        if size < 2 or pos + size > len(data):
            break
        yield marker, pos + 2, pos + size
        pos += size
        scanning = marker == jpeg.SOS
    raise ValueError("Incomplete JPEG")


def _end(data):
    for marker, _, end in _segments(data):
        if marker == jpeg.EOI:
            return end
    raise ValueError("Missing JPEG end")


def _valid_video(data):
    """Require complete, contiguous MP4 boxes before using the video decoder."""
    if not 100 <= len(data) <= MAX_VIDEO_BYTES or data[4:8] != b"ftyp":
        return False
    pos, boxes = 0, set()
    for _ in range(64):
        if pos + 8 > len(data):
            return pos == len(data) and {b"ftyp", b"moov", b"mdat"} <= boxes
        size = int.from_bytes(data[pos:pos + 4], "big")
        kind = bytes(data[pos + 4:pos + 8])
        if size == 1:
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
            if size < 16:
                return False
        elif size == 0:
            size = len(data) - pos
        if not 8 <= size <= len(data) - pos or (kind in boxes and kind != b"free"):
            return False
        boxes.add(kind)
        pos += size
    return False


def _clock_metadata(data):
    packets = []
    for marker, start, end in _segments(data):
        # The Motion Photo directory is in the still image's XMP header.
        if marker in (jpeg.SOS, jpeg.EOI) or end > 1024 * 1024:
            break
        if marker != 0xE1 or data[start:start + len(XMP)] != XMP:
            continue
        packet = bytes(data[start + len(XMP):end])
        if b"MotionPhoto" not in packet:
            continue
        try:
            root = _xml(packet.rstrip(b"\0"))
        except (InvalidMetadata, ParseError, UnicodeError) as error:
            raise ValueError("Invalid Motion Photo metadata") from error
        packets.extend((root, node, packet) for node in root.iter() if CAMERA + "MotionPhoto" in node.attrib)
    if not packets:
        return None
    if len(packets) != 1:
        raise ValueError("Ambiguous Motion Photo metadata")
    return packets[0]


def carrier(data) -> Carrier | None:
    """Find a complete, supported JPEG plus MP4 attachment and an explicit clock."""
    if data[:2] != b"\xff\xd8" or b"MotionPhoto" not in bytes(data[:1024 * 1024]):
        return None
    if len(data) > MAX_INPUT:
        raise ValueError("Motion Photo exceeds the input limit")
    metadata = _clock_metadata(data)
    if metadata is None:
        return None
    root, camera, packet = metadata
    if camera.get(CAMERA + "MotionPhoto") != "1":
        return None
    if camera.get(CAMERA + "MotionPhotoVersion") != "1":
        raise ValueError("Unsupported Motion Photo version")
    timestamp = int(camera.get(CAMERA + "MotionPhotoPresentationTimestampUs", "-1"))
    if not 0 <= timestamp <= 10_000_000:
        raise ValueError("No explicit Motion Photo timestamp")
    directories = list(root.iter(CONTAINER + "Directory"))
    if len(directories) != 1:
        raise ValueError("No unique Motion Photo directory")
    items = directories[0].findall(RDF + "Seq/" + RDF + "li/" + CONTAINER + "Item")
    primary, movie, gainmap = ("Primary", "image/jpeg"), ("MotionPhoto", "video/mp4"), ("GainMap", "image/jpeg")
    layout = [(item.get(ITEM + "Semantic"), item.get(ITEM + "Mime")) for item in items]
    if layout not in ([primary, movie], [primary, gainmap, movie]):
        raise ValueError("Unsupported Motion Photo layout")
    end = _end(data)
    start = len(data) - int(items[-1].get(ITEM + "Length", "0"))
    padding = int(items[0].get(ITEM + "Padding", "0"))
    if not 0 <= padding <= 4096 or not end + padding <= start < len(data):
        raise ValueError("Missing Motion Photo attachment")
    gap = data[end + padding:start]
    if len(items) == 3:
        length = int(items[1].get(ITEM + "Length", "0"))
        if not 100 <= length <= 2_097_152 or len(gap) != length or _end(gap) != length:
            raise ValueError("Incomplete Motion Photo gain map")
    elif gap and (len(gap) >= 65_536 or gap[:8] != b"aecDebug"):
        raise ValueError("Unrecognized data before Motion Photo video")
    if not _valid_video(data[start:]):
        raise ValueError("Incomplete Motion Photo video")
    return Carrier(end, start, timestamp, packet)


def _read_still(data, attachment):
    with Image.open(BytesIO(data[:attachment.image_end]), formats=["JPEG"]) as image:
        if min(image.size) < 320 or image.width * image.height > MAX_PIXELS or image.mode != "RGB":
            raise ValueError("Unsupported Motion Photo still geometry")
        orientation = image.getexif().get(274, 1)
        size = image.size[::-1] if orientation in (5, 6, 7, 8) else image.size
        image.draft("RGB", (MAX_SIDE, MAX_SIDE))
        with ImageOps.exif_transpose(image) as oriented:
            oriented.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            return np.array(oriented), size


def _read_frames(path, timestamp_us, deadline):
    check = (lambda: deadline.check("Motion Photo comparison")) if deadline is not None else None
    if check:
        check()
    budget = deadline.remaining() if deadline is not None else None
    probe_limit = min(5., budget / 2) if budget is not None else 5.
    table = probe_frames(path, cap=180, timeout_s=probe_limit, max_pixels=MAX_FRAME_PIXELS,
                         stream_timeout_s=min(2., probe_limit))
    pts, tb = table.pts, table.tb
    if table.truncated or table.interpolated or len(pts) < 5 or any(a >= b for a, b in zip(pts, pts[1:])):
        raise ValueError("Unverified Motion Photo frame clock")
    goal = Fraction(timestamp_us, 1_000_000)
    center = min(range(len(pts)), key=lambda index: abs(pts[index] * tb - goal))
    if abs(pts[center] * tb - goal) > Fraction(1, 150):
        raise ValueError("No video frame near the declared Motion Photo timestamp")
    start = min(max(0, center - 1), len(pts) - 3)
    indices = list(range(start, start + 3))
    if pts[start] < 0 or (pts[start + 2] - pts[start]) * tb > Fraction(1, 5):
        raise ValueError("Unsupported Motion Photo frame spacing")
    if check:
        check()
    budget = deadline.remaining() if deadline is not None else None
    decode_limit = min(6., budget) if budget is not None else 6.
    frames = list(decode_frames(path, table, indices, max_side=MAX_SIDE, check=check, timeout_s=decode_limit,
                                max_pixels=MAX_FRAME_PIXELS))
    if [frame.index for frame in frames] != indices or not all(frame.pts_verified for frame in frames):
        raise ValueError("Unverified Motion Photo frame pixels")
    clock = {"declared_timestamp_us": timestamp_us, "time_base": table.time_base,
             "frame_index": center, "frame_pts": pts[center], "supporting_pts": [frame.pts for frame in frames]}
    return clock, center - start, [frame.rgb for frame in frames]


def _align(still, frame):
    match = matching_points(frame, still)
    if match is None:
        raise ValueError("Insufficient corresponding details")
    source, target = match
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=1.25,
                                                 maxIters=1500, confidence=.997, refineIters=20)
    if matrix is None or inliers is None:
        raise ValueError("The still and video could not be aligned")
    inside = inliers.ravel() > 0
    height, width = still.shape
    if (inside.sum() < max(32, .6 * len(source))
            or np.any(np.ptp(source[inside], axis=0) < .37 * np.array((width, height)))):
        raise ValueError("Not enough of the still could be aligned")
    residual = np.linalg.norm(source[inside] @ matrix[:, :2].T + matrix[:, 2] - target[inside], axis=1)
    if np.median(residual) > .69 or np.quantile(residual, .90) > 1.2 or not .50 <= np.hypot(*matrix[:, 0]) <= 1.8:
        raise ValueError("The still and video alignment could not be verified")
    frame_mask = np.zeros(frame.shape, np.uint8)
    frame_mask[2:-2, 2:-2] = 1
    mask = cv2.warpAffine(frame_mask, matrix, (width, height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((19, 19), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if mask.mean() < .43:
        raise ValueError("Too little of the still overlaps the video")
    warp = cv2.warpAffine(frame.astype(np.float32), matrix, (width, height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return warp, mask, matrix


def _difference(first, second, scale):
    def average(array):
        return cv2.GaussianBlur(array, (0, 0), scale)

    first = cv2.GaussianBlur(first, (0, 0), 1.05)
    second = cv2.GaussianBlur(second, (0, 0), 1.05)
    a, b = average(first), average(second)
    sx = np.sqrt(np.maximum(0, average(first * first) - a * a))
    sy = np.sqrt(np.maximum(0, average(second * second) - b * b))
    similarity = (average(first * second) - a * b + 8.) / (sx * sy + 8.)
    return np.clip(1. - similarity, 0., 2.) * np.minimum(1., np.maximum(sx, sy) / 9.)


def _strongest_region(residual, coverage):
    selected = (coverage & (residual >= POINT)).astype(np.uint8)
    selected = cv2.morphologyEx(selected, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    selected &= coverage
    _, labels, stats, _ = cv2.connectedComponentsWithStats(selected, connectivity=8)
    maximum = .09 * coverage.sum()
    candidates = [index for index, (_, _, w, h, area) in enumerate(stats[1:], 1)
                  if 165 <= area <= maximum and min(w, h) >= 16]
    if not candidates:
        return None, None
    index = max(candidates, key=lambda idx: stats[idx, 4])
    x, y, w, h = (int(n) for n in stats[index, :4])
    return (x, y, w, h), labels[y:y + h, x:x + w] == index


def measure(data, attachment, *, deadline=None) -> Measurement:
    still, size = _read_still(data, attachment)
    with TemporaryDirectory(prefix="chipmunk-motion-photo-") as folder:
        path = Path(folder) / "embedded.mp4"
        path.write_bytes(data[attachment.video_start:])
        clock, center, frames = _read_frames(path, attachment.timestamp_us, deadline)
    gray = cv2.cvtColor(still, cv2.COLOR_RGB2GRAY)
    aligned = []
    for frame in frames:
        if deadline is not None:
            deadline.check("Motion Photo alignment")
        aligned.append(_align(gray, cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)))
    coverage = np.logical_and.reduce([entry[1] for entry in aligned])
    if coverage.mean() < .42:
        raise ValueError("Too little of the still overlaps all three video frames")
    gray = gray.astype(np.float32)
    before, current, after = (entry[0] for entry in aligned)
    scores = np.array([np.minimum(_difference(gray, frame, 3.8), _difference(gray, frame, 6.))
                       for frame in (before, current, after)])
    if np.mean((np.median(scores, axis=0) < .25)[coverage]) < .57:
        raise ValueError("The still and nearby video frames differ too broadly for local comparison")
    movement = np.maximum(_difference(before, current, 6.), _difference(current, after, 6.))
    region, mask = _strongest_region(scores.min(axis=0) - .85 * movement, coverage)
    return Measurement(clock, still, frames[center], size, aligned[center][2], int(coverage.sum()), region, mask)
