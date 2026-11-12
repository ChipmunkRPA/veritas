# SPDX-License-Identifier: Apache-2.0
"""Compare spatial mouth-boundary measurements on source-bound neural redraws."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_mouth_reconstruction_transfer as panel
from scripts.video_common_sources_v89 import read, require, sha, write

ARMS = ("direct", "chroma_rim", "texture_rim", "joint_rim")
RADII = ((17, 10), (23, 14), (29, 19))
N_SECTORS = 8
WIDTHS = (5, 7, 9, 11)
ANGLE = np.arange(64) * 2 * np.pi / 64
POLICY = {
    "source": "Same frozen GRID groups as prior direct and motion-reconstruction comparison.",
    "time": "Every second native frame. Strictly above third-largest ordinary development score.",
    "ranges": "Maximum central median above both neighboring flank medians, on one fixed ellipse.",
    "radii": RADII,
    "sectors": N_SECTORS,
    "widths": WIDTHS,
    "advancement": "Joint arm adds at least two useful ranges across two components, including a source-time-compatible expression redraw, and at most two additional ordinary ranges.",
    "localization": "Half-open native interval IoU at least 1/2.",
}


def pins():
    files = (
        Path(__file__), Path(panel.__file__), panel.WEIGHT,
        panel.ROOT / "backend/app/detectors/image/faces.py", panel.ROOT / "backend/app/detectors/video/decode.py",
    )
    return {str(path.relative_to(panel.ROOT)): sha(path) for path in files}


def freeze(folder: Path, prior: Path) -> None:
    original = read(prior / "freeze.json")
    require(original["schema"] == "chipmunk.video-mouth-reconstruction-freeze/1"
            and original["bindings"] == panel.bindings(), "Exact prior source and method freeze required")
    # Copy only the part of the existing freeze needed for verified native inspection.
    copied = {key: original[key] for key in ("archive", "clips", "development_components", "held_components")}
    frozen = {**copied, "schema": "chipmunk.video-mouth-rim-freeze/1", "policy": POLICY,
              "prior_sha256": sha(prior / "freeze.json"), "bindings": pins(),
              "versions": [cv2.__version__, np.__version__]}
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "freeze.json", frozen)


def frozen(folder: Path) -> dict:
    doc = read(folder / "freeze.json")
    require(doc["schema"] == "chipmunk.video-mouth-rim-freeze/1" and doc["bindings"] == pins()
            and doc["versions"] == [cv2.__version__, np.__version__]
            and doc["policy"] == json.loads(json.dumps(POLICY)), "Frozen comparison differs")
    return doc


def rim_measures(rgb: np.ndarray, geometry: tuple) -> list[list[float]] | None:
    """Measure a mouth-centered chroma edge and fine-residual step in eight sectors."""
    gray, matrix, mouth, _ = geometry
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = cv2.warpAffine(lab[..., 1:], matrix, (panel.SIZE, panel.SIZE),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    gray = gray.astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (5, 5), 1.0)
    texture = cv2.GaussianBlur(residual * residual, (5, 5), 1.0)
    planes = np.dstack((chroma, texture))
    measures = []
    for rx, ry in RADII:
        # Four concentric profiles spaced by three pixels on either side
        # of the proposed rim. Equal spacing distinguishes a local step
        # from a gradual radial change. One sector has eight angular samples.
        offsets = np.array((-4.5, -1.5, 1.5, 4.5), dtype=np.float32)[:, None]
        x = (mouth[0] + (rx + offsets) * np.cos(ANGLE)).astype(np.float32)
        y = (mouth[1] + (ry + offsets) * np.sin(ANGLE)).astype(np.float32)
        if min(x.min(), y.min()) < 0 or max(x.max(), y.max()) > panel.SIZE - 1:
            return None
        values = cv2.remap(planes, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        chroma_steps = np.linalg.norm(np.diff(values[:, :, :2], axis=0), axis=-1)
        chroma_sectors = np.median(chroma_steps.reshape(3, N_SECTORS, -1), axis=2)
        middle = chroma_sectors[1]
        neighboring = np.maximum(chroma_sectors[0], chroma_sectors[2])
        color = float(np.median(np.log((middle + 1.0) / (neighboring + 1.0))))

        # A texture step can be brighter or darker. Take its magnitude
        # only after finding the median signed sector difference. This
        # discounts alternating sector changes caused by local edges.
        energies = np.mean(values[:, :, 2].reshape(4, N_SECTORS, -1), axis=2)
        energy_step = np.diff(np.log(energies + 1.0), axis=0)
        texture_seam = max(0.0, abs(float(np.median(energy_step[1])))
                           - max(abs(float(np.median(energy_step[0]))), abs(float(np.median(energy_step[2])))))
        measures.append([color, texture_seam, (color + texture_seam) / 2])
    return measures


def native_crop(geometry: tuple, radius: int | None) -> list[int]:
    if radius is None:
        return panel.native_bounds(geometry)
    _, matrix, mouth, shape = geometry
    x, y = mouth
    rx, ry = np.asarray(RADII[radius]) + 4.5
    corners = np.array(((x-rx, y-ry, 1), (x+rx, y-ry, 1), (x-rx, y+ry, 1), (x+rx, y+ry, 1)))
    native = corners @ cv2.invertAffineTransform(matrix).T
    low = np.maximum(0, np.floor(native.min(axis=0)))
    high = np.minimum(shape[::-1], np.ceil(native.max(axis=0)))
    return [int(value) for value in (*low, *high)]


def selected(arm: str, arm_series: list, table, source: dict, aligned: list) -> dict | None:
    """Compare only the same radial positions in a core and both flanks."""
    candidates = [(radius, panel.best_window(series)) for radius, series in enumerate(arm_series)]
    best = max(((radius, found) for radius, found in candidates if found is not None),
               key=lambda row: row[1]["score"], default=None)
    if best is None:
        return None
    radius, row = best
    row["radius_index"] = None if arm == "direct" else radius
    row["source_clock_s"] = [str(table.pts[index] * table.tb) for index in row["frames"]]
    row["time_base"] = table.time_base
    row["anchors"] = [
        {"role": label, "frame": panel.POSITIONS[i], "pts": table.pts[panel.POSITIONS[i]],
         "rgb_sha256": source[panel.POSITIONS[i]], "native_crop_xyxy": native_crop(aligned[i], row["radius_index"])}
        for label, i in zip(("before", "during", "after"), row.pop("slots"), strict=True)
    ]
    return row


def measure(path, row, yunet) -> tuple[dict, object]:
    require(sha(path) == row["file_sha256"], "Retained licensed clip differs")
    table = probe_frames(path, timeout_s=15)
    expected = row["expected"]
    require(len(table) == len(expected["rgb_sha256"]) == 75 and not (table.truncated or table.interpolated or table.rotation)
            and [pts * table.tb for pts in table.pts] == [Fraction(t, 1000) for t in expected["picture_pts_ms"]],
            "Exact native frame table differs")
    source, aligned, rims = {}, [], []
    for frame in decode_frames(path, table, panel.POSITIONS, timeout_s=20):
        digest = hashlib.sha256(frame.rgb).hexdigest()
        require(frame.pts_verified and frame.pts == table.pts[frame.index] and digest == expected["rgb_sha256"][frame.index],
                "Native sampled RGB or timestamp differs")
        source[frame.index] = digest
        geometry = panel.aligned_face(frame.rgb, yunet)
        aligned.append(geometry)
        rims.append(None if geometry is None else rim_measures(frame.rgb, geometry))
    require(list(source) == list(panel.POSITIONS), "A prescribed decoded picture was omitted")
    signals = {arm: [[None] * len(aligned) for _ in range(1 if arm == "direct" else len(RADII))] for arm in ARMS}
    for i in range(1, len(aligned) - 1):
        triple = aligned[i-1:i+2]
        if any(value is None for value in triple):
            continue
        before, center, after = (value[0] for value in triple)
        signals["direct"][0][i] = panel.contrast(center, before, after, panel.support_masks(triple[1][2]))
        if rims[i] is not None:
            for radius, values in enumerate(rims[i]):
                for arm, value in zip(ARMS[1:], values, strict=True):
                    signals[arm][radius][i] = value
    return {
        arm: {"best": selected(arm, series, table, source, aligned),
              "measured_centres": sum(value is not None for value in series[0])}
        for arm, series in signals.items()
    }, table


def journal(folder: Path, doc: dict, split: str) -> list[dict]:
    return panel.journal(folder, doc, split)


def run(folder: Path, split: str) -> None:
    doc = frozen(folder)
    if split == "held":
        seal = read(folder / "calibration.json")
        require(seal["freeze_sha256"] == sha(folder / "freeze.json") and seal["development_sha256"] == sha(folder / "development.jsonl"),
                "Set and seal the separate development points before held comparison")
    yunet = load_yunet(panel.WEIGHT)
    with (folder / f"{split}.jsonl").open("x") as out:
        for pin in doc["clips"]:
            if pin["component"] not in doc[f"{split}_components"]:
                continue
            cases = []
            for path, row, motion in panel.inspect(doc, pin):
                arms, table = measure(path, row, yunet)
                truth = None
                if row["condition"] in panel.NEURAL:
                    require(row["neural_provenance_verified"] and row["observed_neural_pixel_change"],
                            "Positive requires a verified native redraw")
                    truth = row["edit_frames_half_open"]
                cases.append({"condition": row["condition"], "sha256": row["file_sha256"], "truth_frames": truth,
                              "truth_clock_s": None if truth is None else [str(table.pts[i] * table.tb) for i in truth],
                              "motion_status": motion, "arms": arms})
            out.write(json.dumps({"ordinal": pin["ordinal"], "component": pin["component"], "cases": cases},
                                 sort_keys=True, allow_nan=False) + "\n")
            out.flush()
            print(split, pin["ordinal"], len(cases), "verified views", flush=True)


def calibrate(folder: Path) -> None:
    doc = frozen(folder)
    rows = journal(folder, doc, "development")
    points = {}
    for arm in ARMS:
        values = sorted(case["arms"][arm]["best"]["score"] for group in rows for case in group["cases"]
                        if case["condition"] in panel.CONTROL and case["arms"][arm]["best"] is not None)
        require(len(values) >= 24, "Too few ordinary measured development cases")
        points[arm] = values[-3]
    write(folder / "calibration.json", {"freeze_sha256": sha(folder / "freeze.json"),
          "development_sha256": sha(folder / "development.jsonl"), "points": points})


def readout(folder: Path) -> None:
    doc = frozen(folder)
    rows = journal(folder, doc, "held")
    points = read(folder / "calibration.json")["points"]
    totals = {arm: Counter() for arm in ARMS}
    details = {condition: {arm: Counter() for arm in ARMS} for condition in panel.ORDER}
    source_time = {arm: Counter() for arm in ARMS}
    groups = {arm: set() for arm in ARMS}
    added_groups = {arm: set() for arm in ARMS}
    transitions = {arm: Counter() for arm in ARMS[1:]}
    for group in rows:
        for case in group["cases"]:
            outcomes = {arm: panel.result(case, arm, points[arm]) for arm in ARMS}
            for arm, result in outcomes.items():
                totals[arm][result] += 1
                details[case["condition"]][arm][result] += 1
                if case["condition"] == "liveportrait_source_time" and case["motion_status"] == panel.SOURCE_CLOCK:
                    source_time[arm][result] += 1
                if result == "correct_range":
                    groups[arm].add(group["component"])
                    if outcomes["direct"] != "correct_range":
                        added_groups[arm].add(group["component"])
                if arm != "direct":
                    transitions[arm][f"{outcomes['direct']} -> {result}"] += 1
    direct, joint = (totals[name] for name in ("direct", "joint_rim"))
    advance = (joint["correct_range"] >= direct["correct_range"] + 2 and len(added_groups["joint_rim"]) >= 2
               and source_time["joint_rim"]["correct_range"] >= 1
               and joint["ordinary_range"] <= direct["ordinary_range"] + 2)
    report = {"schema": "chipmunk.video-mouth-rim-result/1", "freeze_sha256": sha(folder / "freeze.json"),
              "held_sha256": sha(folder / "held.jsonl"), "calibration_sha256": sha(folder / "calibration.json"),
              "held_clips": len(rows), "held_components": len(doc["held_components"]), "totals": totals,
              "by_condition": details, "expression_source_time_compatible_subset": source_time,
              "components_with_correct_ranges": {arm: len(value) for arm, value in groups.items()},
              "components_with_added_ranges": {arm: len(value) for arm, value in added_groups.items()},
              "paired_transitions": transitions, "pilot_advances": advance}
    write(folder / "aggregate.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "calibrate", "held", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "freeze":
        if not args.prior:
            parser.error("freeze requires the exact prior comparison directory")
        freeze(args.folder, args.prior)
    elif args.stage in ("development", "held"):
        run(args.folder, args.stage)
    else:
        {"calibrate": calibrate, "readout": readout}[args.stage](args.folder)


if __name__ == "__main__":
    main()
