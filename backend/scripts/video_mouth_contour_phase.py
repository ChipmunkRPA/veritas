# SPDX-License-Identifier: Apache-2.0
"""Test native high-frequency agreement across short mouth-contour sectors."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import review_lombard_neural_mouth as redraw
from scripts import video_mouth_jaw_transfer as panel
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = panel.ROOT
GRID = panel.DATA / "video-mouth-jaw-transfer-2026-11-11"
WEIGHT = panel.prior.WEIGHT
RADII = ((13, 8), (17, 10), (22, 13))
SECTORS = 8
STEPS = 12
ARMS = ("fine_phase", "coarse_adjusted")
SPEC = {
    "measurement": "On a rigid eye and lip-corner coordinate system, sample tangential profiles from inner and outer rails of a mouth-centered contour. Compare signed fine-band phase agreement against two off-phase pairings. Subtract a matched outer rail and compare independently to the coarse-band change.",
    "native": "Band-pass filter decoded native luminance before sampling, require all rail points on observed pixels, no synthesis, no alternate file, no authored mask, no audio. Require a verified picture clock and one shot.",
    "candidate": "Scan cores of 8, 12, 16, or 20 pictures with five-picture flanks. Require supported phase losses on at least three common sectors and positive before/after agreement in each. Keep one strongest range per arm.",
    "point": "Third-largest strictly positive ordinary maximum, or .02 if higher, from the earlier GRID development controls only.",
    "gate": "At least four correctly localized independently aligned redraws across three groups, at least 70% coverage on each main held class, at most two ordinary flags total and none on untouched recordings, H.264 recompression or cuts. Temporal IoU >=0.50.",
}


def plan(lombard, fresh):
    groups = read(GRID / "freeze.json")
    controls = read(GRID / "controls.json")
    require(controls["freeze_sha256"] == sha(GRID / "freeze.json"), "Archived delivery controls differ")
    rows = []
    for pin in groups["grid"]["development"]:
        for path, item, _ in panel.prior.inspect(groups, pin):
            if item["condition"] in panel.prior.CONTROL:
                rows.append(
                    {
                        "split": "development",
                        "family": "grid",
                        "id": pin["ordinal"],
                        "group": pin["component"],
                        "role": item["condition"],
                        "file": str(path),
                        "sha": item["file_sha256"],
                    }
                )
    for item in controls["roster"]:
        rows.append(
            {
                "split": "held",
                "family": "codec",
                "id": item["index"],
                "group": item["index"],
                "role": item["condition"],
                "file": str(GRID / "controls" / item["file"]),
                "sha": item["sha256"],
            }
        )
    review = read(lombard / "review-held.json")
    require(review["freeze_sha256"] == sha(lombard / "freeze.json"), "Source redraw audit differs")
    by_role = {(r["ordinal"], r["role"]): r for r in review["cases"]}
    for item in redraw.partitions(lombard, "held"):
        k, role = item["ordinal"], item["role"]
        key = by_role[k, role]
        require(item["file_sha256"] == key["file_sha256"], "Audited redraw file differs")
        positive = role == "source_time_neural"
        rows.append(
            {
                "split": "held",
                "family": "lombard",
                "id": k,
                "group": key["group"],
                "role": role,
                "file": str(lombard / "deliveries" / f"{item['token']}.mov"),
                "sha": item["file_sha256"],
                "edit": key["authored_frames"] if positive else None,
                "aligned": positive and key["observation"]["status"] == "source_event_retained",
            }
        )
    receipt = read(fresh / "source-receipt.json")
    require(receipt["freeze_sha256"] == sha(fresh / "freeze.json"), "Fresh source receipt differs")
    previously_seen = {r["source"]["speaker"] for r in read(lombard / "freeze.json")["sources"]}
    require(
        not previously_seen & {r["speaker"] for r in receipt["rows"]}, "Fresh speakers overlap positive panel"
    )
    for row in receipt["rows"]:
        item = row["inputs"]["front"]
        rows.append(
            {
                "split": "held",
                "family": "fresh",
                "id": row["ordinal"],
                "group": row["group"],
                "role": "original",
                "file": str(fresh / "sources" / item["file"]),
                "sha": item["sha256"],
            }
        )
    require(
        Counter((r["split"], r["family"]) for r in rows)
        == {
            ("development", "grid"): 56,
            ("held", "lombard"): 44,
            ("held", "fresh"): 48,
            ("held", "codec"): 12,
        },
        "Unexpected complete roster",
    )
    return {
        "protocol": SPEC,
        "rows": rows,
        "tool_sha256": sha(Path(__file__)),
        "weights_sha256": sha(WEIGHT),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "source_receipt_sha256": sha(fresh / "source-receipt.json"),
    }


def profiles(filtered, inverse, mouth, radius):
    theta = (np.arange(SECTORS * STEPS) + 0.5) * (2 * np.pi / (SECTORS * STEPS))
    offsets = np.asarray((-0.20, 0.14, 0.42, 0.76))
    radial = np.stack((radius[0] * np.cos(theta), radius[1] * np.sin(theta)), axis=-1)
    xy = np.asarray(mouth) + (1 + offsets[:, None, None]) * radial[None]
    native = xy @ inverse[:, :2].T + inverse[:, 2]
    height, width = filtered.shape
    allowed = ((native >= 2) & (native < np.asarray([width - 2, height - 2]))).all(axis=(0, 2))
    allowed = allowed.reshape(SECTORS, STEPS).all(axis=1)
    values = cv2.remap(
        filtered, native[..., 0].astype(np.float32), native[..., 1].astype(np.float32), cv2.INTER_LINEAR
    )
    return values.reshape(4, SECTORS, STEPS), allowed


def corrected_phase(first, second):
    first, second = (np.diff(p) for p in (first, second))
    first, second = (p - p.mean() for p in (first, second))
    den = np.linalg.norm(first) * np.linalg.norm(second)
    if den < 0.12:
        return None
    null = (first[3:] @ second[:-3] + first[:-3] @ second[3:]) / (2 * (len(first) - 3))
    return float(first @ second / den - null * len(first) / den)


def frame_measure(rgb, geometry):
    _, matrix, mouth, _ = geometry
    inverse = cv2.invertAffineTransform(matrix)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = [cv2.GaussianBlur(gray, (0, 0), s) for s in (0.55, 1.2, 2.3)]
    bands = (blur[0] - blur[1], blur[1] - blur[2])
    output = []
    for radii in RADII:
        captured = [profiles(band, inverse, mouth, radii) for band in bands]
        row = []
        for k in range(SECTORS):
            signals = []
            for values, valid in captured:
                direct = corrected_phase(values[0, k], values[1, k]) if valid[k] else None
                reference = corrected_phase(values[2, k], values[3, k]) if valid[k] else None
                signals.append(None if direct is None or reference is None else [direct, reference])
            row.append(signals)
        output.append(row)
    return output


def observe(path, detector, cache):
    table = probe_frames(path, timeout_s=15)
    require(
        not (table.truncated or table.interpolated or table.rotation) and 40 <= len(table) <= 100,
        "Expected verified upright short picture table",
    )
    timeline = [p * table.tb for p in table.pts]
    shots, measures, anchors = [], [], []
    track = ShotDetector(table)
    for frame in decode_frames(path, table, None, timeout_s=25):
        require(
            frame.index == len(shots) and frame.pts_verified and frame.pts == table.pts[frame.index],
            "Native picture clock differs",
        )
        track.update(frame.index, frame.rgb)
        shots.append(len(track.boundaries))
        digest = panel.raw_digest(frame.rgb)
        if digest not in cache:
            geometry = panel.prior.aligned_face(frame.rgb, detector)
            if geometry is None:
                cache[digest] = None, None
            else:
                x0, y0, x1, y1 = panel.native_bounds(geometry)
                cache[digest] = (
                    frame_measure(frame.rgb, geometry),
                    ([x0, y0, x1, y1], panel.raw_digest(frame.rgb[y0:y1, x0:x1])),
                )
        values, proof = cache[digest]
        measures.append(values)
        anchors.append(
            None
            if proof is None
            else {
                "frame": frame.index,
                "pts": frame.pts,
                "rgb_sha256": digest,
                "crop_xyxy": proof[0],
                "crop_sha256": proof[1],
            }
        )
    require(len(shots) == len(table), "Decoded frame inventory differs")
    return timeline, table.time_base, shots, measures, anchors


def summary(measured):
    values = np.full((len(measured), len(RADII), SECTORS, 2, 2), np.nan)
    for i, row in enumerate(measured):
        if row is None:
            continue
        for r, sector in enumerate(row):
            for k, pair in enumerate(sector):
                for band, channels in enumerate(pair):
                    if channels is not None:
                        values[i, r, k, band] = channels
    return values


def select(timeline, tb, shots, raw, anchors):
    data = summary(raw)
    selection = dict.fromkeys(ARMS)
    supported = dict.fromkeys(ARMS, False)
    cache = {}

    def aggregate(a, b):
        if (a, b) not in cache:
            part = data[a:b]
            valid = np.sum(np.isfinite(part[..., 0]), axis=0) * 5 >= 4 * (b - a)
            median = np.ma.median(np.ma.masked_invalid(part), axis=0).filled(np.nan)
            cache[a, b] = np.where(valid[..., None], median, np.nan)
        return cache[a, b]

    for width in (8, 12, 16, 20):
        for a in range(5, len(data) - width - 4):
            b = a + width
            if b >= len(data) or shots[a - 5] != shots[b + 4]:
                continue
            before, during, after = aggregate(a - 5, a), aggregate(a, b), aggregate(b, b + 5)
            deficit = np.minimum(before, after) - during
            adjusted = deficit[..., 0] - deficit[..., 1]
            signals = (adjusted[:, :, 0], adjusted[:, :, 0] - np.maximum(0.0, adjusted[:, :, 1]))
            support = np.minimum(before[..., 0, 0], after[..., 0, 0]) >= 0.04
            for arm, response in zip(ARMS, signals, strict=True):
                for radius in range(len(RADII)):
                    eligible = np.flatnonzero(support[radius] & np.isfinite(response[radius]))
                    if len(eligible) < 3:
                        continue
                    supported[arm] = True
                    positive = [int(k) for k in eligible if response[radius, k] > 0]
                    if len(positive) < 3:
                        continue
                    score = float(np.median(response[radius, eligible]))
                    if score <= 0 or selection[arm] is not None and score <= selection[arm]["score"]:
                        continue
                    pictures = [anchors[i] for i in (a - 1, (a + b) // 2, b)]
                    if not all(pictures):
                        continue
                    selection[arm] = {
                        "score": score,
                        "frames": [a, b],
                        "seconds": [str(timeline[a]), str(timeline[b])],
                        "time_base": tb,
                        "radius": radius,
                        "support_sectors": eligible.tolist(),
                        "positive_sectors": positive,
                        "anchors": pictures,
                    }
    return {arm: {"measured": supported[arm], "proposal": selection[arm]} for arm in ARMS}


def score(lock, run, phase):
    cv2.setNumThreads(1)
    detector = load_yunet(WEIGHT)
    result = []
    grouped = {}
    for row in lock["rows"]:
        if row["split"] == phase:
            grouped.setdefault((row["family"], row["id"]), []).append(row)
    for group in grouped.values():
        cache = {}
        for row in group:
            path = Path(row["file"])
            require(sha(path) == row["sha"], "Frozen source bytes differ")
            inputs = observe(path, detector, cache)
            result.append(
                {
                    **row,
                    "picture_ticks": len(inputs[0]),
                    "geometry_frames": sum(a is not None for a in inputs[4]),
                    "arms": select(*inputs),
                }
            )
        print(phase, group[0]["family"], group[0]["id"], len(result), flush=True)
    path = run / f"{phase}.json"
    write(path, {"freeze_sha256": sha(run / "freeze.json"), "rows": result})
    if phase == "development":
        points = {}
        for arm in ARMS:
            values = sorted(
                (r["arms"][arm]["proposal"]["score"] for r in result if r["arms"][arm]["proposal"]),
                reverse=True,
            )
            points[arm] = max(0.02, values[2] if len(values) > 2 else 0.02)
        write(run / "points.json", {"development_sha256": sha(path), "points": points})
        print(points, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "development", "held"))
    p.add_argument("run", type=Path)
    p.add_argument("lombard", type=Path)
    p.add_argument("fresh", type=Path)
    a = p.parse_args()
    lock = plan(a.lombard, a.fresh)
    if a.phase == "freeze":
        a.run.mkdir(mode=0o700)
        write(a.run / "freeze.json", lock)
    else:
        require(lock == read(a.run / "freeze.json"), "Frozen inputs or method changed")
        if a.phase == "held":
            require(
                read(a.run / "points.json")["development_sha256"] == sha(a.run / "development.json"),
                "Held point not committed before scoring",
            )
        score(lock, a.run, a.phase)


if __name__ == "__main__":
    main()
