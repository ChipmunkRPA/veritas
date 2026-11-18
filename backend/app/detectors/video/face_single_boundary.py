# SPDX-License-Identifier: Apache-2.0
"""Source-bound face appearance review for a sustained change within one shot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import iou
from app.detectors.video import face_local_detail, face_return
from app.detectors.video.decode import DecodeError, FrameTable, decode_frames
from app.detectors.video.shots import ShotDetector
from app.engine.boundary import DetectorTimeout
from app.evidence import face_single_boundary_binding as binding

KEY = binding.KEY
COVERAGE = binding.COVERAGE
COMMITMENT = binding.COMMITMENT
COVERAGE_COMMITMENT = binding.COVERAGE_COMMITMENT
METHOD = binding.METHOD
MODEL_IDS = ("yunet.2023mar", "opencv.sface.2021dec")
POINT = binding.POINT
SAMPLE = tuple(range(0, binding.FRAME_COUNT, 2))
FLANK = 6
MINIMUM_PRESENCE = .76
MINIMUM_IOU = .35
MINIMUM_SCENE = binding.MIN_SCENE


def supported(path: Path, table: FrameTable, encoding: tuple[str | None, str | None]) -> bool:
    return face_local_detail.supported(table, encoding) and face_return.source_carrier(path) == "mp4"


def coverage(status: str, reason: str | None = None, *, frames_scanned: int = 0, measured_face_frames: int = 0) -> dict:
    return {"status": status, "reason": reason, "frames_scanned": frames_scanned,
            "measured_face_frames": measured_face_frames}


def _scene(samples, onset):
    a, b = samples[onset - 2], samples[onset]
    if a.face is None or b.face is None or iou(a.face, b.face) < MINIMUM_IOU:
        return None
    result = face_return.scene_lower_quartile(a, b)
    return result if result is not None and result >= MINIMUM_SCENE else None


def proposals(samples: Mapping[int, face_return.Sample], check: Callable[[], None]) -> list[dict]:
    """Reproduce the frozen persistent-state proposals and the adjacent-jump tie breaker."""
    present = [index for index in SAMPLE if samples[index].vector is not None]
    if len(present) < 2 * FLANK:
        return []
    vectors = np.stack([samples[index].vector for index in present])
    distances = 1 - np.clip(vectors @ vectors.T, -1, 1)
    positions = {index: pos for pos, index in enumerate(present)}

    def within(indices):
        values = distances[np.ix_(indices, indices)]
        return np.quantile(values[np.triu_indices(len(indices), k=1)], .75)

    def separation(before, after):
        return float(np.quantile(distances[np.ix_(before, after)], .25) - max(within(before), within(after)))

    result = []
    for edge in range(22, 79, 2):
        check()
        near_before = list(range(edge - 2 * FLANK, edge, 2))
        near_after = list(range(edge, edge + 2 * FLANK, 2))
        if not set(near_before + near_after) <= positions.keys() or (scene := _scene(samples, edge)) is None:
            continue
        before = [positions[index] for index in present if index < edge]
        after = [positions[index] for index in present if index >= edge]
        if min(len(before) / (edge // 2), len(after) / ((binding.FRAME_COUNT - edge) // 2)) < MINIMUM_PRESENCE:
            continue
        left, right = ([positions[index] for index in group] for group in (near_before, near_after))
        persistent = min(separation(before, after), separation(left, right))
        neighbors = np.r_[distances[left[:-1], left[1:]], distances[right[:-1], right[1:]]]
        bump = float(distances[left[-1], right[0]] - np.quantile(neighbors, .75))
        result.append({"coarse": edge, "scene": round(scene, 9), "persistent": round(persistent, 9),
                       "edge_supported": round(min(persistent, bump), 9), "jump_excess": round(bump, 9)})
    return result


def _refine(path, table, samples, reading, locator, encoder, *, check, timeout_s):
    coarse = reading["coarse"]
    centers = []
    for region in (range(coarse - 12, coarse - 3, 2), range(coarse + 2, coarse + 12, 2)):
        center = np.mean([samples[index].vector for index in region], axis=0)
        centers.append(center / max(np.linalg.norm(center), 1e-6))
    wanted = sorted({coarse - 3, coarse - 2, coarse - 1, coarse, coarse + 1, coarse + 2, coarse + 11})
    fine, anchors = {}, {}
    with closing(decode_frames(path, table, wanted, check=check, timeout_s=timeout_s())) as frames:
        for frame in frames:
            check()
            if not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise DecodeError("The precise face boundary lacks native frame timing")
            sample, _, pin, _ = face_local_detail.observe(frame.rgb, locator, encoder, with_texture=True)
            fine[frame.index] = sample
            if pin:
                anchors[frame.index] = pin | {"index": frame.index, "pts": frame.pts,
                                             "eyes": [list(eye) for eye in pin["eyes"]]}
    possibilities = []
    for edge in range(coarse - 2, coarse + 3):
        left, right = fine.get(edge - 1), fine.get(edge)
        if any(sample is None or sample.vector is None for sample in (left, right)):
            continue
        if iou(left.face, right.face) < MINIMUM_IOU:
            continue
        scene = face_return.scene_lower_quartile(left, right)
        if scene is None or scene < MINIMUM_SCENE:
            continue
        old, new = centers
        prior = float(left.vector @ old - left.vector @ new)
        following = float(right.vector @ new - right.vector @ old)
        score = min(prior, following)
        if score > 0:
            possibilities.append((score, -abs(edge - coarse), -edge, scene))
    if not possibilities:
        return None
    margin, _, neg_edge, scene = max(possibilities)
    edge = -neg_edge
    roles = {"before": edge - 1, "transition": edge, "after": coarse + 11}
    if not set(roles.values()) <= anchors.keys():
        return None
    return reading | {"boundary": edge, "pts": table.pts[edge], "time_base": table.time_base,
                      "fine_margin": round(margin, 9), "fine_scene": round(scene, 9),
                      "source_frames": {role: anchors[index] for role, index in roles.items()}}


def measure(path, table, locator, encoder, *, check, timeout_s) -> tuple[dict | None, dict]:
    """Return the refined measurement before applying the point, plus its actual coverage."""
    observer = ShotDetector(table, min_shot_s=0.)
    samples = {}
    scanned = measured = 0

    def result(status, reason=None):
        return coverage(status, reason, frames_scanned=scanned, measured_face_frames=measured)

    try:
        with closing(decode_frames(path, table, None, check=check, timeout_s=timeout_s())) as frames:
            for index, frame in enumerate(frames):
                check()
                if frame.index != index or not frame.pts_verified or frame.pts != table.pts[index]:
                    raise DecodeError("A measured picture lacks its native timestamp")
                observer.update(index, frame.rgb)
                scanned += 1
                if index in SAMPLE:
                    # Only the unused frequency output is omitted on the coarse scan.
                    sample, _, _, _ = face_local_detail.observe(frame.rgb, locator, encoder, with_texture=False)
                    samples[index] = sample
                    measured += sample.vector is not None
        if observer.frames_seen != len(table) or set(samples) != set(SAMPLE):
            raise DecodeError("Native decode omitted a planned comparison picture")
        if observer.boundaries:
            return None, result("unavailable", "shot_change")
        candidates = proposals(samples, check)
        if not candidates:
            return None, result("unavailable", "insufficient_face_context")
        top = max(candidates, key=lambda item: (item["persistent"], item["jump_excess"], -item["coarse"]))
        reading = _refine(path, table, samples, top, locator, encoder, check=check, timeout_s=timeout_s)
        if reading is None:
            return None, result("unavailable", "insufficient_native_boundary")
        return reading, result("localized" if reading["persistent"] > POINT else "no_match")
    except DetectorTimeout:
        return None, result("unavailable", "time_budget")
    except (DecodeError, cv2.error, ValueError):
        return None, result("unavailable", "native_measurement_unavailable")


def inspect(path, table, locator, encoder, *, check, timeout_s) -> tuple[dict | None, dict]:
    reading, disposition = measure(path, table, locator, encoder, check=check, timeout_s=timeout_s)
    if disposition["status"] != "localized" or reading is None:
        return None, disposition
    # Canonical JSON rounds numeric displays. Preserve precise source geometry and measurements for replay.
    frames = {
        role: row | {"eyes_hex": [[float(value).hex() for value in point] for point in row["eyes"]]}
        for role, row in reading["source_frames"].items()
    }
    return reading | {
        "method": METHOD,
        "time_s": float(reading["pts"] * table.tb),
        "affects_overall_confidence": False,
        "source_frames": frames,
        "readings_hex": {
            name: float(reading[name]).hex()
            for name in ("scene", "persistent", "edge_supported", "jump_excess", "fine_margin", "fine_scene")
        },
    }, disposition
