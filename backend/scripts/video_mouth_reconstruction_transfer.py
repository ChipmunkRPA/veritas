# SPDX-License-Identifier: Apache-2.0
"""Compare motion-reconstructed native mouth detail on matched GRID edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
WEIGHT = ROOT / "models/weights/face_detection_yunet_2023mar.onnx"
ARMS = ("direct", "motion_reconstruction")
NEURAL = ("musetalk_source_time", "musetalk_earlier_320ms", "liveportrait_source_time", "liveportrait_earlier_320ms")
CONTROL = ("original", "resample_musetalk", "resample_liveportrait", "brightness_musetalk", "brightness_liveportrait",
           "genuine_frame_retime", "audio_delay_320ms")
ORDER = ("original", *NEURAL, *CONTROL[1:])
SALT = "chipmunk-mouth-reconstruction/1"
POINT = "Third largest measured ordinary development score, with strict exceedance."
GATE = "At least 6/32 localized neural views across three held components, two more than direct, at least two source-time-compatible redraws, and no more than two extra ordinary ranges."
SIZE = 96
SOURCE_CLOCK = "source_time_compatible"
POSITIONS = tuple(range(0, 75, 2))
MIN_IOU = Fraction(1, 2)


def files(archive, index):
    return (archive / f"grid-native-clips-v149/clip-{index:02d}/cases.jsonl",
            archive / f"grid-mouth-motion-readout-v158/clip-{index:02d}/motion.json")


def jsonl(path):
    return list(map(json.loads, path.read_text().splitlines()))


def bindings():
    paths = (Path(__file__), WEIGHT, ROOT / "backend/app/detectors/image/faces.py", ROOT / "backend/app/detectors/video/decode.py")
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def freeze(folder, archive):
    pins, components = [], set()
    for index in range(16):
        cases_path, motion_path = files(archive, index)
        rows = jsonl(cases_path)
        component = rows[0]["component_index"]
        require(tuple(row["condition"] for row in rows) == ORDER and all(
            row["status"] == "ready" and row["component_index"] == component and row["asset_index"] == index for row in rows
        ), "Complete matched native case inventory required")
        components.add(component)
        pins.append({"ordinal": index, "component": component, "cases_sha256": sha(cases_path), "motion_sha256": sha(motion_path)})
    require(len(components) == 8, "Expected eight disjoint known source components")
    ordered = sorted(components, key=lambda i: hashlib.sha256(f"{SALT}/{i}".encode()).digest())
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "freeze.json", {"schema": "chipmunk.video-mouth-reconstruction-freeze/1", "bindings": bindings(),
          "archive": str(archive.resolve()), "clips": pins, "development_components": ordered[:4], "held_components": ordered[4:],
          "points": POINT, "gate": GATE, "native_picture_positions": POSITIONS, "iou": str(MIN_IOU),
          "cv_version": cv2.__version__, "numpy_version": np.__version__})


def frozen(folder):
    doc = read(folder / "freeze.json")
    require(doc["bindings"] == bindings() and doc["points"] == POINT and doc["gate"] == GATE
            and (doc["cv_version"], doc["numpy_version"]) == (cv2.__version__, np.__version__), "Frozen comparison differs")
    return doc


def inspect(doc, pin):
    cases_path, motion_path = files(Path(doc["archive"]), pin["ordinal"])
    require(sha(cases_path) == pin["cases_sha256"] and sha(motion_path) == pin["motion_sha256"], "Source archive receipt differs")
    rows, motion = jsonl(cases_path), read(motion_path)["cases"]
    require([(row["condition"], row["case_number"]) for row in rows] == [(row["condition"], row["case_number"]) for row in motion],
            "Motion labels must bind the corresponding render")
    return [(cases_path.parent / Path(row["file"]).name, row, observed["status"]) for row, observed in zip(rows, motion, strict=True)]


def aligned_face(rgb, yunet):
    faces = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=48)
    if len(faces) != 1:
        return None
    eye0, eye1, _, corner0, corner1 = np.asarray(faces[0].landmarks, dtype=np.float64)
    eyes = eye1 - eye0
    magnitude = float(eyes @ eyes)
    if magnitude < 16**2 or eyes[0] < 8:
        return None
    axis = (27 / magnitude) * np.asarray(((eyes[0], eyes[1]), (-eyes[1], eyes[0])))
    matrix = np.column_stack((axis, np.array([34., 25.]) - axis @ eye0))
    mouth = np.rint(matrix @ np.r_[(corner0 + corner1) / 2, 1]).astype(int)
    if not 37 <= mouth[0] <= 61 or not 42 <= mouth[1] <= 73:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    patch = cv2.warpAffine(gray, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return patch, matrix, mouth, rgb.shape[:2]


def support_masks(mouth):
    y, x = np.ogrid[:SIZE, :SIZE]
    horizontal, vertical = x - mouth[0], y - mouth[1]
    core = (horizontal / 17)**2 + (vertical / 9)**2 <= 1
    surround = ((horizontal / 31)**2 + (vertical / 19)**2 <= 1) & (((horizontal / 23)**2 + (vertical / 13)**2) > 1)
    return core, surround


def reconstruct(center, neighbour, pixel_grid):
    flow = cv2.calcOpticalFlowFarneback(center, neighbour, None, .5, 3, 13, 3, 5, 1.2, 0)
    return cv2.remap(neighbour, pixel_grid + flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def contrast(center, before, after, masks):
    # Use float before subtraction so uint8 never wraps. The reference covers
    # neighboring face pixels without reading the original counterpart.
    base = center.astype(np.float32)
    errors = (np.abs(base - before) + np.abs(base - after)) * .5
    mouth, surrounding = (float(np.percentile(errors[mask], 75)) for mask in masks)
    return float(np.log((mouth + 2) / (surrounding + 2)))


def best_window(scores):
    best = None
    for width in (5, 7, 9, 11):
        for start in range(4, len(scores) - width - 3):
            end = start + width
            slots = [list(range(start - 3, start)), list(range(start, end)), list(range(end, end + 3))]
            usable = [[i for i in span if scores[i] is not None] for span in slots]
            if len(usable[0]) < 2 or len(usable[2]) < 2 or len(usable[1]) < (4 * width + 4) // 5:
                continue
            flank_before, core, flank_after = (float(np.median([scores[i] for i in span])) for span in usable)
            score = core - max(flank_before, flank_after)
            if best is None or score > best["score"]:
                best = {"score": score, "frames": [POSITIONS[start], POSITIONS[end]],
                        "slots": [usable[0][-1], usable[1][len(usable[1])//2], usable[2][0]],
                        "medians": [flank_before, core, flank_after]}
    return best


def native_bounds(geometry):
    _, matrix, mouth, shape = geometry
    inverse = cv2.invertAffineTransform(matrix)
    x, y = mouth
    corners = np.array(((x-17, y-9, 1), (x+17, y-9, 1), (x-17, y+9, 1), (x+17, y+9, 1))) @ inverse.T
    low = np.maximum(0, np.floor(corners.min(axis=0))).astype(int)
    high = np.minimum(shape[::-1], np.ceil(corners.max(axis=0))).astype(int)
    return [int(v) for v in [*low, *high]]


def measure(path, row, yunet):
    require(sha(path) == row["file_sha256"], "Licensed native case differs from its archived digest")
    table = probe_frames(path, timeout_s=15)
    expected = row["expected"]
    require(len(table) == len(expected["rgb_sha256"]) == 75 and not (table.truncated or table.interpolated or table.rotation)
            and [pts * table.tb for pts in table.pts] == [Fraction(t, 1000) for t in expected["picture_pts_ms"]],
            "Exact native frame table differs")
    source, aligned = {}, []
    for frame in decode_frames(path, table, POSITIONS, timeout_s=20):
        digest = hashlib.sha256(frame.rgb).hexdigest()
        require(frame.pts_verified and frame.pts == table.pts[frame.index] and digest == expected["rgb_sha256"][frame.index],
                "Native sampled RGB or timestamp differs")
        source[frame.index] = digest
        aligned.append(aligned_face(frame.rgb, yunet))
    require(list(source) == list(POSITIONS), "A prescribed decoded picture was omitted")
    a, b = np.meshgrid(np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32))
    pixels = np.dstack((a, b))
    signals = {name: [None] * len(aligned) for name in ARMS}
    for i in range(1, len(aligned) - 1):
        triple = aligned[i-1:i+2]
        if any(value is None for value in triple):
            continue
        before, center, after = (value[0] for value in triple)
        masks = support_masks(triple[1][2])
        signals["direct"][i] = contrast(center, before, after, masks)
        signals["motion_reconstruction"][i] = contrast(center, reconstruct(center, before, pixels), reconstruct(center, after, pixels), masks)
    selected = {}
    for arm, series in signals.items():
        best = best_window(series)
        if best:
            best["source_clock_s"] = [str(table.pts[index] * table.tb) for index in best["frames"]]
            best["time_base"] = table.time_base
            best["anchors"] = [{"role": label, "frame": POSITIONS[i], "pts": table.pts[POSITIONS[i]],
                                "rgb_sha256": source[POSITIONS[i]], "native_crop_xyxy": native_bounds(aligned[i])}
                               for label, i in zip(("before", "during", "after"), best.pop("slots"), strict=True)]
        selected[arm] = {"best": best, "measured_centres": sum(val is not None for val in series)}
    return selected, table


def run(folder, split):
    doc = frozen(folder)
    if split == "held":
        seal = read(folder / "calibration.json")
        require(seal["freeze_sha256"] == sha(folder / "freeze.json") and seal["development_sha256"] == sha(folder / "development.jsonl"),
                "Freeze separate development points before held comparison")
    yunet = load_yunet(WEIGHT)
    with (folder / f"{split}.jsonl").open("x") as out:
        for pin in doc["clips"]:
            if pin["component"] not in doc[f"{split}_components"]:
                continue
            cases = []
            for path, row, motion in inspect(doc, pin):
                arms, table = measure(path, row, yunet)
                truth = None
                if row["condition"] in NEURAL:
                    require(row["neural_provenance_verified"] and row["observed_neural_pixel_change"], "Positive must have verified native redraw")
                    truth = row["edit_frames_half_open"]
                cases.append({"condition": row["condition"], "sha256": row["file_sha256"], "truth_frames": truth,
                              "truth_clock_s": None if truth is None else [str(table.pts[i] * table.tb) for i in truth],
                              "motion_status": motion, "arms": arms})
            out.write(json.dumps({"ordinal": pin["ordinal"], "component": pin["component"], "cases": cases}, sort_keys=True, allow_nan=False) + "\n")
            out.flush()
            print(split, pin["ordinal"], "verified 11 views", flush=True)


def journal(folder, doc, split):
    rows = jsonl(folder / f"{split}.jsonl")
    require([row["ordinal"] for row in rows] == [pin["ordinal"] for pin in doc["clips"] if pin["component"] in doc[f"{split}_components"]]
            and all(tuple(case["condition"] for case in row["cases"]) == ORDER for row in rows), "Incomplete case journal")
    return rows


def calibrate(folder):
    doc = frozen(folder)
    rows = journal(folder, doc, "development")
    points = {}
    for arm in ARMS:
        controls = sorted(best["score"] for row in rows for case in row["cases"] if case["truth_frames"] is None
                          and (best := case["arms"][arm]["best"]) is not None)
        require(len(controls) >= 24, "Enough ordinary development cases must be measured")
        points[arm] = controls[-3]
    write(folder / "calibration.json", {"freeze_sha256": sha(folder / "freeze.json"),
          "development_sha256": sha(folder / "development.jsonl"), "points": points})


def exact_iou(a, b):
    left, right = map(Fraction, a)
    start, end = map(Fraction, b)
    return max(0, min(right, end) - max(left, start)) / (max(right, end) - min(left, start))


def result(case, arm, point):
    best = case["arms"][arm]["best"]
    if best is None:
        return "unassessed"
    if best["score"] <= point:
        return "measured_no_range"
    if not case["truth_clock_s"]:
        return "ordinary_range"
    return "correct_range" if exact_iou(best["source_clock_s"], case["truth_clock_s"]) >= MIN_IOU else "wrong_range"


def readout(folder):
    doc = frozen(folder)
    rows = journal(folder, doc, "held")
    points = read(folder / "calibration.json")["points"]
    totals = {arm: Counter() for arm in ARMS}
    by_condition = {condition: {arm: Counter() for arm in ARMS} for condition in ORDER}
    synchronized = {arm: Counter() for arm in ARMS}
    per_component = {arm: set() for arm in ARMS}
    for row in rows:
        for case in row["cases"]:
            for arm in ARMS:
                label = result(case, arm, points[arm])
                totals[arm][label] += 1
                by_condition[case["condition"]][arm][label] += 1
                if case["condition"] == "liveportrait_source_time" and case["motion_status"] == SOURCE_CLOCK:
                    synchronized[arm][label] += 1
                if label == "correct_range":
                    per_component[arm].add(row["component"])
    direct, new = (totals[name] for name in ARMS)
    clears = (new["correct_range"] >= max(6, direct["correct_range"] + 2) and len(per_component[ARMS[1]]) >= 3
              and new["ordinary_range"] <= direct["ordinary_range"] + 2 and synchronized[ARMS[1]]["correct_range"] >= 2)
    record = {"schema": "chipmunk.video-mouth-reconstruction-result/1", "freeze_sha256": sha(folder / "freeze.json"),
              "held_sha256": sha(folder / "held.jsonl"), "calibration_sha256": sha(folder / "calibration.json"),
              "held_clips": len(rows), "held_components": len(doc["held_components"]), "totals": totals, "by_condition": by_condition,
              "expression_source_time_compatible_subset": synchronized,
              "components_with_correct_ranges": {arm: len(value) for arm, value in per_component.items()}, "pilot_advances": clears}
    write(folder / "aggregate.json", record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "calibrate", "held", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.stage == "freeze":
        if not args.archive:
            parser.error("freeze requires the retained native source archive")
        freeze(args.folder, args.archive)
    elif args.stage in ("development", "held"):
        run(args.folder, args.stage)
    else:
        {"calibrate": calibrate, "readout": readout}[args.stage](args.folder)


if __name__ == "__main__":
    main()
