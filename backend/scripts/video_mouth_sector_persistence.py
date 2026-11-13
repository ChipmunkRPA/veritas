# SPDX-License-Identifier: Apache-2.0
"""Test persistent native perioral sector changes on qualified neural redraws."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_fixed_residual as prior
from scripts import video_mouth_jaw_transfer as panel
from scripts import video_mouth_native_texture_transfer as direct
from scripts.video_common_sources_v89 import read, require, sha, write

ARMS = ("independent_sectors", "persistent_sectors")
RADII = ((17, 10), (23, 14), (29, 19))
WIDTHS, FLANK = (5, 7, 9, 11), 4
SPEC = {
    "schema": "chipmunk.video-mouth-sector-persistence/1",
    "input": "One upright exact native track with a visible face. Eighty-millisecond target grid, native sample times retained.",
    "descriptor": "Fixed eye/mouth geometry selects four contiguous native-coordinate bands around each of three ellipses, in eight angular sectors. In each sector measure signed adjacent-band second differences in median chroma and log fine-detail energy. Subtract the same descriptors measured beside the visible upper face.",
    "geometry": {
        "radii": [list(r) for r in RADII],
        "sectors": 8,
        "band_edges": [-0.32, -0.16, 0, 0.16, 0.32],
        "face_reference_offset": [0, -39],
    },
    "comparison": "Scan 5, 7, 9 or 11 measured pictures inside four-picture flanks. Require same shot, exact native times, and complete descriptors. Use signed changes against the larger absolute flank shift. Independent sectors can change per picture; the persistent arm needs the same three or more sectors, sign and channel in at least three quarters of core pictures, with stable native face position.",
    "coverage": "At least 75% measured in each held class, separate from any confidence estimate.",
    "point": "Strict exceedance of the third largest ordinary development maximum, with floor 0.15. No edited examples fit the point.",
    "gate": {
        "minimum_useful_iou": "1/2",
        "min_grid_added": 4,
        "min_grid_groups": 2,
        "min_compatible_added": 2,
        "max_grid_ordinary": 3,
        "min_crema_added": 2,
        "max_crema_ordinary": 3,
        "max_reencode": 1,
        "max_cut": 0,
        "min_reserve_compatible": 1,
        "max_reserve_ordinary": 0,
    },
}


def freeze():
    manifest = prior.lock(direct.PRIOR, prior.RESERVE)
    return {
        "spec": SPEC,
        "qualified_inputs": manifest,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "worker_sha256": sha(Path(__file__)),
    }


def local_descriptor(values, box, affine, center):
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1]
    x = affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2] - center[0]
    y = affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2] - center[1]
    out = []
    for wx, wy in RADII:
        radius = np.hypot(x / wx, y / wy) - 1
        angle = ((np.arctan2(y / wy, x / wx) + np.pi) * 4 / np.pi).astype(int) % 8
        row = []
        for sector in range(8):
            ranges = [
                (angle == sector) & (a <= radius) & (radius < b)
                for a, b in zip(SPEC["geometry"]["band_edges"], SPEC["geometry"]["band_edges"][1:])
            ]
            if min(np.count_nonzero(mask) for mask in ranges) < 5:
                return None
            levels = np.stack([np.median(values[mask], axis=0) for mask in ranges])
            # A signed curve change discounts a gradual radial color or detail trend.
            row.append(np.diff(levels, 2, axis=0)[0] - np.diff(levels, 2, axis=0)[1])
        out.append(row)
    return np.asarray(out, dtype=np.float32)


def features(rgb, geometry):
    _, affine, center, shape = geometry
    inverse = cv2.invertAffineTransform(affine)
    offsets = (center, center + np.asarray(SPEC["geometry"]["face_reference_offset"]))
    bands = []
    boxes = []
    for pos in offsets:
        corners = (
            np.array(
                [
                    (*p, 1)
                    for p in (
                        (pos[0] - 40, pos[1] - 27),
                        (pos[0] + 40, pos[1] - 27),
                        (pos[0] - 40, pos[1] + 27),
                        (pos[0] + 40, pos[1] + 27),
                    )
                ]
            )
            @ inverse.T
        )
        low = np.floor(corners.min(axis=0)).astype(int)
        high = np.ceil(corners.max(axis=0)).astype(int)
        if np.any(low < 0) or np.any(high > np.asarray(shape[::-1])):
            return None
        box = [int(k) for k in (*low, *high)]
        x0, y0, x1, y1 = box
        pixels = rgb[y0:y1, x0:x1]
        lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).astype(np.float32)
        luminance = lab[..., 0]
        residual = luminance - cv2.GaussianBlur(luminance, (0, 0), 1.0)
        detail = np.log1p(cv2.GaussianBlur(residual * residual, (0, 0), 1.1))
        values = np.dstack((lab[..., 1:] / 3, detail))
        descriptor = local_descriptor(values, box, affine, pos)
        if descriptor is None:
            return None
        bands.append(descriptor)
        boxes.append(box)
    x0, y0, x1, y1 = boxes[0]
    return bands[0] - bands[1], inverse @ np.r_[center, 1], boxes, panel.raw_digest(rgb[y0:y1, x0:x1])


def trial(path, source, detector, cache):
    require(sha(path) == source["file_sha256"], "Qualified source differs")
    table, picks = panel.inspect_frames(path, source["expected"])
    shot = ShotDetector(table)
    observed, shots, anchors = [], [], []
    for frame, digest in panel.exact_rgb(path, table, picks, source["expected"]):
        shot.update(frame.index, frame.rgb)
        shots.append(len(shot.boundaries))
        if digest not in cache:
            geometry = panel.prior.aligned_face(frame.rgb, detector)
            cache[digest] = None if geometry is None else features(frame.rgb, geometry)
        row = cache[digest]
        observed.append(row)
        anchors.append(
            None
            if row is None
            else {
                "frame": frame.index,
                "pts": frame.pts,
                "rgb_sha256": digest,
                "mouth_xyxy": row[2][0],
                "reference_xyxy": row[2][1],
                "mouth_crop_sha256": row[3],
            }
        )
    times = panel.source_clock(table)
    best = dict.fromkeys(ARMS)
    measured = 0
    for width in WIDTHS:
        for a in range(FLANK, len(observed) - width - FLANK + 1):
            b = a + width
            group = observed[a - FLANK : b + FLANK]
            if any(row is None for row in group) or shots[a - FLANK] != shots[b + FLANK - 1]:
                continue
            if any(
                not Fraction(1, 25) <= times[picks[i + 1]] - times[picks[i]] <= Fraction(3, 25)
                for i in range(a - FLANK, b + FLANK - 1)
            ):
                continue
            loc = np.stack([row[1] for row in group])
            if np.max(np.linalg.norm(loc - np.median(loc, axis=0), axis=1)) > 10:
                continue
            values = np.stack([row[0] for row in group])
            left, middle, right = values[:FLANK], values[FLANK:-FLANK], values[-FLANK:]
            base, tail = np.median(left, axis=0), np.median(right, axis=0)
            increase = middle - np.maximum(base, tail)
            decrease = np.minimum(base, tail) - middle
            magnitude = np.maximum(increase, decrease)
            independent = float(
                np.median(np.sort(magnitude, axis=2)[:, :, -3:, :].mean(axis=2), axis=0).max()
            )
            directional = np.stack((np.quantile(increase, 0.25, axis=0), np.quantile(decrease, 0.25, axis=0)))
            persistent = float(np.sort(directional, axis=2)[:, :, -3, :].max())
            measured += 1
            for arm, value in zip(ARMS, (independent, persistent), strict=True):
                if best[arm] is not None and best[arm]["score"] >= value:
                    continue
                best[arm] = {
                    "score": value,
                    "frames": [picks[a], picks[b]],
                    "time_base": table.time_base,
                    "source_clock_s": [str(times[picks[i]]) for i in (a, b)],
                    "anchors": [
                        {**anchors[i], "role": role}
                        for role, i in zip(
                            ("before", "during", "after"), (a - 1, (a + b) // 2, b), strict=True
                        )
                    ],
                }
    return {
        "eligible_windows": measured,
        "arms": best,
        "native_inputs": len(picks),
        "measured_inputs": sum(row is not None for row in observed),
    }


def score(run, phase, detector):
    frozen = read(direct.PRIOR / "freeze.json")
    point = read(run / "points.json") if phase != "development" else None
    require(
        point is None or point["freeze_sha256"] == sha(run / "freeze.json"), "Point must precede held scoring"
    )
    roster = []
    if phase == "development":
        for index, component, rows in panel.grid_cases(frozen, "development"):
            roster.append(
                (
                    "grid",
                    index,
                    component,
                    [
                        (path, source, None, False, None)
                        for path, source, _ in rows
                        if source["condition"] not in panel.prior.NEURAL
                    ],
                )
            )
    elif phase == "held":
        baseline = direct.baselines(direct.PRIOR)
        for (bank, index), (component, rows) in direct.roster(direct.PRIOR, frozen).items():
            roster.append(
                (
                    bank,
                    index,
                    component,
                    [
                        (
                            path,
                            {**source, "condition": condition},
                            old["truth"],
                            old["source_time_compatible"],
                            old["arms"]["direct"],
                        )
                        for condition, (path, source) in rows.items()
                        if (old := baseline[(bank, index, condition)])
                    ],
                )
            )
    else:
        groups = defaultdict(list)
        for row in read(prior.RESERVE)["sources"]:
            neural = row["role"] in ("source_time_neural", "earlier_drive_neural")
            truth = [str(Fraction(i, 25)) for i in row["picture_edit"]] if neural else None
            groups[row["ordinal"]].append(
                (
                    Path(row["path"]),
                    {"condition": row["role"], "expected": None, "file_sha256": row["sha256"]},
                    truth,
                    row["role"] == "source_time_neural",
                    None,
                )
            )
        roster = [("reserve", index, index, rows) for index, rows in groups.items()]
    with (run / f"{phase}.jsonl").open("x") as stream:
        for bank, index, component, cases in roster:
            cache = {}
            for path, source, truth, compatible, old in cases:
                row = trial(path, source, detector, cache)
                row["arms"]["direct"] = old
                stream.write(
                    json.dumps(
                        {
                            "bank": bank,
                            "index": index,
                            "component": component,
                            "truth": truth,
                            "compatible": compatible,
                            "condition": source["condition"],
                            "source_sha256": source["file_sha256"],
                            **row,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            print(phase, bank, index, len(cases), flush=True)


def calibrate(run):
    journal = run / "development.jsonl"
    rows = panel.prior.jsonl(journal)
    require(len(rows) == 56, "Expected all ordinary development views")
    points = {}
    for arm in ARMS:
        values = sorted(row["arms"][arm]["score"] for row in rows if row["arms"][arm] is not None)
        require(len(values) >= 42, "Insufficient measured ordinary views")
        points[arm] = max(0.15, values[-3])
    write(
        run / "points.json",
        {"freeze_sha256": sha(run / "freeze.json"), "journal_sha256": sha(journal), "points": points},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "development", "calibrate", "held", "reserve"))
    p.add_argument("run", type=Path)
    a = p.parse_args()
    lock = freeze()
    if a.phase == "freeze":
        a.run.mkdir(mode=0o700)
        write(a.run / "freeze.json", lock)
    else:
        require(read(a.run / "freeze.json") == lock, "Source or method differs from frozen contract")
        if a.phase == "calibrate":
            calibrate(a.run)
        else:
            cv2.setNumThreads(1)
            score(a.run, a.phase, load_yunet(panel.prior.WEIGHT))


if __name__ == "__main__":
    main()
