"""Compare facial appearance and detail on each side of a local video change."""

from __future__ import annotations

import hashlib
from contextlib import closing
from dataclasses import dataclass

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces, iou
from app.detectors.video import face_appearance, face_detail_inputs, face_frequency, face_return
from app.detectors.video.decode import DecodeError, decode_frames
from app.detectors.video.shots import ShotDetector
from app.evidence import face_local_return_binding as local

DIMENSIONS, COUNT, STEP, SCHEDULE = local.DIMENSIONS, local.FRAME_COUNT, local.STEP, local.SCHEDULE
FLANK, CORE_MIN, CORE_MAX, SCENE_POINT = local.FLANK, local.CORE_MIN, local.CORE_MAX, local.SCENE_POINT
MODES = ("same_appearance", "relative_detail", "combined")


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    features: tuple[float, ...]
    support: tuple[tuple[int, ...], ...]

    def anchors(self):
        before, core, after = self.support
        return {"before": before[-1], "during": core[len(core) // 2], "after": after[0]}


def supported(table, encoding):
    return (tuple(encoding[:2]) == ("h264", "yuv420p")
            and (table.width, table.height) == (table.coded_width, table.coded_height) == DIMENSIONS
            and not (table.rotation or table.truncated or table.interpolated) and len(table) == COUNT
            and all((pts - table.pts[0]) * table.tb == index * STEP for index, pts in enumerate(table.pts)))


def shot_changes(path, table, start, end, *, timeout_s=10., check=None):
    """Read every picture in the candidate and its flanks to find abrupt shot changes."""
    before, _, after = local.support(start, end)
    indices = range(before[0], after[-1] + 1)
    # Short intervening shots matter even when too brief to enter the sampling plan.
    detector = ShotDetector(table, min_shot_s=0.)
    first = indices.start
    with closing(decode_frames(path, table, indices, timeout_s=timeout_s, check=check)) as frames:
        for offset, frame in enumerate(frames):
            if check is not None:
                check()
            if frame.index != first + offset or not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise DecodeError("The source clock could not be verified throughout the face comparison")
            detector.update(frame.index, frame.rgb)
    if detector.frames_seen != len(indices):
        raise DecodeError("The face comparison is missing pictures around its proposed range")
    return tuple(row["frame_index"] for row in detector.boundaries)


def observe(pixels, locator, recognizer, *, with_texture):
    bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    faces = detect_faces(bgr, detector=locator, min_size=48)
    face = faces[0] if len(faces) == 1 else None
    scene = face_return.scene(pixels, face)
    vector, detail, pins = None, None, None
    if face is not None:
        detection = face_appearance.face_row(face.landmarks)
        detection[0, :4] = face.box()
        detection[0, 14] = face.score
        normalized = recognizer.alignCrop(bgr, detection)
        vector = face_appearance.unit(recognizer.feature(normalized).reshape(-1))
        if with_texture:
            crop = face_detail_inputs.align(pixels, face.landmarks[:2])
            if crop is not None:
                detail = np.asarray(face_frequency.frequency_detail(crop[None])["relative"])
                pins = {"box": dict(zip("xywh", face.box(), strict=True)), "eyes": face.landmarks[:2],
                        "decoded_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                        "aligned_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest()}
    return face_return.Sample(face, vector, scene), detail, pins, len(faces) > 1


def _summarize(distances, left, middle, right):
    a = distances[np.ix_(middle, left)]
    b = distances[np.ix_(middle, right)]
    outside = np.quantile(distances[np.ix_(left, right)], .75)
    internal = max(np.quantile(distances[np.ix_(part, part)][np.triu_indices(len(part), k=1)], .75)
                   for part in (left, middle, right))
    shared_change = np.quantile(np.minimum(np.quantile(a, .25, axis=1), np.quantile(b, .25, axis=1)), .25)
    paired_edges = min(distances[left[-1], middle[0]], distances[middle[-1], right[0]])
    return float(shared_change - max(outside, internal)), float(paired_edges - outside)


def proposals(samples, details, *, review=lambda: None):
    """Use fixed native slots and require the surrounding scene on both sides."""
    support = np.array([i in details for i in SCHEDULE])
    if np.count_nonzero(support) < 2 * FLANK + CORE_MIN:
        return []
    available = [i for i in SCHEDULE if i in details]
    raw = np.stack([details[i] for i in available]).astype(np.float64)
    vectors = np.stack([samples[i].vector for i in available])
    identities = 1 - np.clip(vectors @ vectors.T, -1, 1)
    # Distances in log spectral detail stay on the same scale for every clip.
    contrast = np.abs(raw[:, None] - raw[None, :]).mean(axis=-1)
    positions = {index: offset for offset, index in enumerate(available)}
    crossings = {}
    for a, b in zip(SCHEDULE, SCHEDULE[1:]):
        if a in positions and b in positions and iou(samples[a].face, samples[b].face) >= .30:
            crossings[b] = face_return.scene_lower_quartile(samples[a], samples[b])
    results = []
    for start in range(FLANK, len(SCHEDULE) - CORE_MIN - FLANK + 1):
        review()
        onset = SCHEDULE[start]
        entry = crossings.get(onset)
        if entry is None or entry < SCENE_POINT:
            continue
        for stop in range(start + CORE_MIN, min(start + CORE_MAX, len(SCHEDULE) - FLANK) + 1):
            end = SCHEDULE[stop]
            exit_score = crossings.get(end)
            if exit_score is None or exit_score < SCENE_POINT or not support[start - FLANK:stop + FLANK].all():
                continue
            if any(crossings.get(k) is None for k in SCHEDULE[start - FLANK + 1:stop + FLANK]):
                continue
            groups = SCHEDULE[start - FLANK:start], SCHEDULE[start:stop], SCHEDULE[stop:stop + FLANK]
            indexed = tuple([positions[i] for i in group] for group in groups)
            identity = _summarize(identities, *indexed)
            texture = _summarize(contrast, *indexed)
            results.append(Interval(onset, end, identity + texture, groups))
    return results
