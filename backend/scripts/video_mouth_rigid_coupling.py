# SPDX-License-Identifier: Apache-2.0
"""Test local lip and jaw coordination after rigid upper-face registration."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import grid_reserve_neural_mouth as geometry
from scripts import grid_reserve_word_clocks as grid
from scripts import qualify_lombard_mouth_events as native
from scripts import review_lombard_neural_mouth as construction
from scripts.video_common_sources_v89 import read, require, sha, write

ARMS = ("jaw", "head")
ROLES = {"original", "resample_ordinary", "delayed_pixels_ordinary", "source_time_neural"}
UPPER = (33, 133, 263, 362, 168, 6, 197, 10, 54, 284, 127, 356)
LOWER_LIP = (14, 17, 87, 317)
UPPER_LIP = (13, 0, 82, 312)
JAW = (152, 148, 377, 176, 400)
ALL_POINTS = (*UPPER, *LOWER_LIP, *UPPER_LIP, *JAW)
DATA = Path(__file__).resolve().parents[1] / "data/independent-evaluation"
GRID = DATA / "grid-reserve-neural-mouth-2026-11-12-admitted"
SPEC = {
    "scope": "Use only the submitted native visual landmarks. No other copy, audio, generator signals, source-aligned marker or authored mask enters measurement.",
    "registration": "Per adjacent picture, proper Kabsch rigid transform between twelve upper-face 3D observations in eye units. Eliminate rigid motion before assessing signed lip and lower-jaw excursions.",
    "jaw": "Correlate signed lower-lip opening velocity with mean chin and bilateral jaw velocity. Measure the loss in positive agreement against both adjoining flanks.",
    "head": "Correlate absolute opening velocity with local angular head speed. Measure loss in positive agreement relative to both adjacent flanks, and require angular movement in all three windows.",
    "scan": {
        "core_pictures": [12, 16, 20],
        "flank_edges": 9,
        "support_minimum": 7,
        "baseline_correlation_minimum": 0.25,
        "lip_velocity_std_minimum": 0.005,
        "jaw_velocity_std_minimum": 0.0025,
        "angular_velocity_std_minimum": 0.002,
        "source_clock_scope": "native 23.5–25 fps, 48–85 pictures, skip edges with head-fit error above .045 eye units",
    },
    "development": "Two source groups from retained GRID reserve and one Lombard group. Use all ordinary variants. Set one point per arm as the larger of .18 and largest eligible ordinary score plus .025.",
    "held": "Eleven disjoint Lombard redraw groups' utterances with exact matched resample and delayed mouth controls, plus all 48 newly admitted genuine utterances from twelve distinct speaker groups. Retain unassessed rows.",
    "gate": "At least four of ten verified source-aligned redraws localized at >=0.5 IoU from three source groups, >=50% measured on each source family, <=2 total ordinary flags, zero on untouched recordings.",
}


def plan(lombard, fresh, frozen):
    all_rows = []
    grid_plan, grid_plan_sha = grid.plan()
    completed = read(GRID / "construction.json")
    proof = read(GRID / "independent-replay.json")
    require(
        proof["status"] == "verified" and proof["construction_sha256"] == sha(GRID / "construction.json"),
        "GRID reserve proof differs",
    )
    for entry in read(GRID / "freeze.json")["sources"]:
        k = entry["ordinal"]
        r = grid_plan[k]["input"]
        group = f"grid-{entry['group']}"
        all_rows.append(
            {
                "split": "development",
                "key": ["grid", k, "original"],
                "group": group,
                "path": r["source_file"],
                "file_sha256": r["source_sha256"],
                "truth": None,
            }
        )
        for item in completed["cases"]:
            if item["ordinal"] == k and item["mode"] == "earlier_pixels_ordinary":
                all_rows.append(
                    {
                        "split": "development",
                        "key": ["grid", k, "retime"],
                        "group": group,
                        "path": str(GRID / "deliveries" / item["file"]),
                        "file_sha256": item["proof"]["control_sha256"],
                        "truth": None,
                    }
                )
    for split in ("development", "held"):
        rows = construction.partitions(lombard, split)
        audit = read(lombard / f"review-{split}.json")
        require(audit["freeze_sha256"] == sha(lombard / "freeze.json"), "Retained native review differs")
        labels = {(item["ordinal"], item["role"]): item for item in audit["cases"]}
        for row in rows:
            k, role = row["ordinal"], row["role"]
            if split == "development" and role == "source_time_neural":
                continue
            label = labels[k, role]
            require(
                role in ROLES and row["status"] == "authored" and row["file_sha256"] == label["file_sha256"],
                "Original rendered fixture differs",
            )
            all_rows.append(
                {
                    "split": split,
                    "key": ["lombard", k, role],
                    "group": f"lombard-{label['group']}",
                    "path": str(lombard / "deliveries" / f"{row['token']}.mov"),
                    "file_sha256": row["file_sha256"],
                    "truth": label["authored_frames"] if role == "source_time_neural" else None,
                    "event_aligned": role == "source_time_neural"
                    and label["observation"]["status"] == "source_event_retained",
                }
            )
    fresh_rows = read(fresh / "source-receipt.json")
    require(
        fresh_rows["freeze_sha256"] == sha(fresh / "freeze.json") and len(fresh_rows["rows"]) == 48,
        "Fresh source roster differs",
    )
    old_speakers = {r["source"]["speaker"] for r in read(lombard / "freeze.json")["sources"]}
    require(
        not old_speakers & {r["speaker"] for r in fresh_rows["rows"]},
        "Fresh source speakers overlap development or held redraws",
    )
    for row in fresh_rows["rows"]:
        all_rows.append(
            {
                "split": "held",
                "key": ["fresh", row["ordinal"], "original"],
                "group": f"fresh-{row['group']}",
                "path": str(fresh / "sources" / row["inputs"]["front"]["file"]),
                "file_sha256": row["inputs"]["front"]["sha256"],
                "truth": None,
            }
        )
    require(
        Counter(r["split"] for r in all_rows) == {"development": 10, "held": 92},
        "Unexpected fixed test roles",
    )
    return {
        "protocol": SPEC,
        "cases": all_rows,
        "worker_sha256": sha(Path(__file__)),
        "grid_plan_sha256": grid_plan_sha,
        "lombard_sha256": sha(lombard / "freeze.json"),
        "fresh_sha256": sha(fresh / "source-receipt.json"),
        "lombard_observer_sha256": sha(frozen / "observations-held.json"),
        "model_sha256": sha(geometry.ASSETS / "permissions/face_landmarker.task"),
    }


def rigid(first, last):
    a, b = first[: len(UPPER)], last[: len(UPPER)]
    left, right = a.mean(axis=0), b.mean(axis=0)
    aa, bb = a - left, b - right
    u, _, v = np.linalg.svd(aa.T @ bb)
    if np.linalg.det(u @ v) < 0:
        u[:, -1] *= -1
    rotation = u @ v
    fit = float(np.sqrt(np.mean(np.sum((aa @ rotation - bb) ** 2, axis=1))))
    pose = math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return (last - right) @ rotation.T + left - first, fit, pose


def features(first, last):
    if first is None or last is None:
        return None
    motion, error, pose = rigid(first, last)
    if error > 0.045:
        return None
    h = len(UPPER)
    lower = float(np.mean(motion[h : h + len(LOWER_LIP), 1]))
    upper = float(np.mean(motion[h + len(LOWER_LIP) : h + len(LOWER_LIP) + len(UPPER_LIP), 1]))
    jaw = float(np.mean(motion[-len(JAW) :, 1]))
    return [lower - upper, jaw, pose, error]


def descriptor(module, task, rgb):
    response = task.detect(module.Image(image_format=module.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if len(response.face_landmarks) != 1:
        return None, None
    mesh = response.face_landmarks[0]
    if len(mesh) != 478:
        return None, None
    height, width = rgb.shape[:2]
    xy = np.asarray(
        [(mesh[i].x * width, mesh[i].y * height, mesh[i].z * width) for i in ALL_POINTS], dtype=float
    )
    if not np.isfinite(xy).all() or not ((xy[:, :2] >= 0) & (xy[:, :2] < [width, height])).all():
        return None, None
    eye = np.linalg.norm((xy[0, :2] + xy[1, :2] - xy[2, :2] - xy[3, :2]) / 2)
    if eye < 20:
        return None, None
    x0, y0 = np.maximum(0, np.floor(xy[len(UPPER) :, :2].min(axis=0) - 6)).astype(int)
    x1, y1 = np.minimum([width, height], np.ceil(xy[len(UPPER) :, :2].max(axis=0) + 6)).astype(int)
    return xy, [int(x0), int(y0), int(x1), int(y1)]


def observe(case, module, task, cached):
    path = Path(case["path"])
    require(sha(path) == case["file_sha256"], "Frozen source file differs")
    table = probe_frames(path, timeout_s=15)
    times = [p * table.tb for p in table.pts]
    if not (
        48 <= len(times) <= 85
        and all(Fraction(1, 25) <= b - a <= Fraction(2, 47) for a, b in zip(times, times[1:]))
        and not (table.rotation or table.interpolated or table.truncated)
    ):
        return {"status": "out_of_scope", "picture_pts": list(table.pts), "time_base": table.time_base}
    points, boxes = [], []
    for frame in decode_frames(path, table, None, timeout_s=30):
        require(
            frame.pts_verified and frame.pts == table.pts[frame.index], "Native picture timestamp differs"
        )
        digest = native.digest(frame.rgb.tobytes())
        if digest not in cached:
            value, box = descriptor(module, task, frame.rgb)
            crop = None
            if box:
                x0, y0, x1, y1 = box
                crop = native.digest(frame.rgb[y0:y1, x0:x1].tobytes())
            cached[digest] = value, box, crop
        value, box, crop = cached[digest]
        points.append(value)
        boxes.append(
            None
            if box is None
            else {
                "index": frame.index,
                "pts": frame.pts,
                "rgb_sha256": digest,
                "box": box,
                "crop_sha256": crop,
            }
        )
    require(len(points) == len(times), "Incomplete native picture sequence")
    eyes = [
        np.linalg.norm((row[0, :2] + row[1, :2] - row[2, :2] - row[3, :2]) / 2)
        for row in points
        if row is not None
    ]
    scale = float(np.median(eyes)) if eyes else 1.0
    normalized = [None if row is None else row / scale for row in points]
    return {
        "status": "observed",
        "picture_pts": list(table.pts),
        "time_base": table.time_base,
        "edges": [features(a, b) for a, b in zip(normalized, normalized[1:])],
        "anchors": boxes,
        "geometry_frames": len(eyes),
        "eye_scale": scale,
    }


def correlation(rows, arm):
    if len(rows) < 7:
        return None
    array = np.asarray(rows, dtype=float)
    if arm == "jaw":
        a, b, minimum = array[:, 0], array[:, 1], 0.0025
    else:
        a, b, minimum = abs(array[:, 0]), array[:, 2], 0.002
    if a.std() < 0.005 or b.std() < minimum:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def detect(measured, arm):
    if measured["status"] != "observed":
        return {"measured": False, "proposal": None}
    edges, boxes = measured["edges"], measured["anchors"]
    cache = {}

    def window(a, b):
        if (a, b) not in cache:
            selected = [r for r in edges[a:b] if r is not None]
            cache[a, b] = correlation(selected, arm) if len(selected) * 5 >= (b - a) * 4 else None
        return cache[a, b]

    best = None
    supported = False
    for width in SPEC["scan"]["core_pictures"]:
        for a in range(9, len(edges) - width - 8):
            b = a + width
            before, middle, after = window(a - 9, a), window(a, b), window(b, b + 9)
            if None in (before, middle, after):
                continue
            supported = True
            if min(before, after) < 0.25:
                continue
            score = min(before, after) - middle
            if score <= 0 or best and score <= best["score"]:
                continue
            evidence = [boxes[i] for i in (a - 1, (a + b) // 2, b)]
            if any(v is None for v in evidence):
                continue
            best = {
                "score": score,
                "frames": [a, b],
                "flank_core_flank_correlation": [before, middle, after],
                "anchors": evidence,
            }
    return {"measured": supported, "proposal": best}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "development", "held"))
    p.add_argument("run", type=Path)
    p.add_argument("lombard", type=Path)
    p.add_argument("fresh", type=Path)
    p.add_argument("observations", type=Path)
    a = p.parse_args()
    lock = plan(a.lombard, a.fresh, a.observations)
    if a.phase == "freeze":
        a.run.mkdir(mode=0o700)
        write(a.run / "freeze.json", lock)
        return
    require(lock == read(a.run / "freeze.json"), "Frozen roster or method differs")
    if a.phase == "held":
        calibration = read(a.run / "calibration.json")
        require(
            calibration["development_sha256"] == sha(a.run / "development.json"), "Operating points differ"
        )
    output = []
    cv2.setNumThreads(1)
    module, task = geometry.mp_task()
    recent = None
    with closing(task):
        for case in lock["cases"]:
            if case["split"] != a.phase:
                continue
            if recent != case["key"][:2]:
                recent = case["key"][:2]
                cached = {}
            measured = observe(case, module, task, cached)
            result = {arm: detect(measured, arm) for arm in ARMS}
            output.append(
                {
                    "key": case["key"],
                    "group": case["group"],
                    "truth": case["truth"],
                    "event_aligned": case.get("event_aligned", False),
                    "native": measured,
                    "results": result,
                }
            )
            print(
                case["key"],
                {
                    k: (v["measured"], v["proposal"]["score"] if v["proposal"] else None)
                    for k, v in result.items()
                },
                flush=True,
            )
    path = a.run / f"{a.phase}.json"
    write(path, {"freeze_sha256": sha(a.run / "freeze.json"), "rows": output})
    if a.phase == "development":
        points = {
            arm: max(
                0.18,
                max(
                    (r["results"][arm]["proposal"]["score"] for r in output if r["results"][arm]["proposal"]),
                    default=0,
                )
                + 0.025,
            )
            for arm in ARMS
        }
        write(a.run / "calibration.json", {"development_sha256": sha(path), "points": points})
        print("Frozen development points", points, flush=True)


if __name__ == "__main__":
    main()
