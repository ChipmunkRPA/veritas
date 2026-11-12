# SPDX-License-Identifier: Apache-2.0
"""Replay an independent mouth-to-jaw motion measurement on two native source banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_reconstruction_transfer as prior
from scripts.video_common_sources_v89 import read, require, sha, write
from scripts.video_native_inserts_v95 import encode_rgb, verify_rgb

ROOT = prior.ROOT
DATA = ROOT / "backend/data/independent-evaluation"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
PRIOR_GRID = DATA / "video-mouth-reconstruction-transfer-2026-11-11/freeze.json"
NATIVE = DATA / "native-mouth-clips-v120"
NATIVE_ORDER = ("original", "resample", "blur2", "blur4", "ai_aligned", "ai_earlier_320ms")
ARMS = ("direct", "jaw_coupling")
STEP = Fraction(2, 25)
CUT = (27, 47)
RIDGE = .10
GATE = ("GRID: at least six new correct redraw ranges across three components, including two motion-compatible original-time cases; "
        "at most three added ordinary flags and zero added cut flags. Native confirmation: at least four new correct ranges "
        "across three source groups, at most two added ordinary flags. Measure at least 80% of positives and controls separately.")
POINT = "The third largest measured ordinary development response for each arm. Strict exceedance."
SPEC = {"feature_fit": "genuine development frames only", "step_s": str(STEP), "core_samples": [5, 7, 9, 11],
        "flank_samples": 3, "max_flow_gap_s": "3/25", "ridge": RIDGE, "pose_reference_xy": [0, -20],
        "mouth_strain_boxes_xy": [[0, 5], [0, -5], [8, 0], [-8, 0]],
        "jaw_boxes_xy": [[-20, 12], [0, 20], [20, 12], [-22, -2], [22, -2]],
        "minimum_iou": "1/2", "scope": "visual motion only"}


def bindings():
    paths = (Path(__file__), Path(prior.__file__), ROOT / "backend/app/detectors/video/shots.py",
             ROOT / "backend/app/detectors/video/decode.py", ROOT / "backend/app/detectors/image/faces.py", prior.WEIGHT)
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def native_groups(archive):
    output = {}
    for path in sorted(NATIVE.glob("extract-*/output/cases.jsonl")):
        for row in prior.jsonl(path):
            if row["study_role"] != "confirmation":
                continue
            require(row["status"] == "ready", "A planned confirmation clip is unavailable")
            index = row["asset_index"]
            recorded = Path(row["file"])
            destination = archive / "native-mouth-clips-v120" / Path(*recorded.parts[-3:])
            require(destination.is_file(), "Native confirmation media missing")
            entry = {**row, "file": str(destination.relative_to(archive)), "journal": str(path.relative_to(DATA))}
            output.setdefault(index, []).append(entry)
    require(len(output) == 10 and all(tuple(row["condition"] for row in rows) == NATIVE_ORDER for rows in output.values()),
            "Confirmation must contain all 60 prespecified views")
    return output


def freeze(folder, archive):
    old = read(PRIOR_GRID)
    groups = native_groups(archive)
    grid = {split: [pin for pin in old["clips"] if pin["component"] in old[f"{split}_components"]]
            for split in ("development", "held")}
    require(all(len(grid[part]) == 8 for part in grid), "Exactly eight GRID sources per role required")
    for pins in grid.values():
        for pin in pins:
            prior.inspect({"archive": str(archive)}, pin)
    native = [{"index": i, "files": [row["file_sha256"] for row in rows], "journal": rows[0]["journal"],
               "journal_sha256": sha(DATA / rows[0]["journal"])} for i, rows in groups.items()]
    held = [pin["ordinal"] for pin in grid["held"]]
    cut_pairs = [[held[i], held[i + 2]] for i in (0, 1)] + [[held[i], held[i + 2]] for i in (4, 5)]
    require(all(next(pin["component"] for pin in grid["held"] if pin["ordinal"] == a)
                != next(pin["component"] for pin in grid["held"] if pin["ordinal"] == b) for a, b in cut_pairs),
            "Cut pairs must be across known source components")
    folder.mkdir(mode=0o700, parents=True, exist_ok=False)
    write(folder / "freeze.json", {"schema": "chipmunk.video-mouth-jaw-freeze/1", "archive": str(archive.resolve()),
        "bindings": bindings(), "prior_grid_sha256": sha(PRIOR_GRID), "grid": grid, "native": native,
        "cut_pairs": cut_pairs, "cut_frames": CUT, "recompress_crf": 28, "gate": GATE, "points": POINT, "spec": SPEC,
        "opencv": cv2.__version__, "numpy": np.__version__})


def frozen(folder):
    doc = read(folder / "freeze.json")
    require(doc["bindings"] == bindings() and doc["gate"] == GATE and doc["spec"] == SPEC
            and doc["prior_grid_sha256"] == sha(PRIOR_GRID) and doc["opencv"] == cv2.__version__
            and doc["numpy"] == np.__version__, "Frozen method or runtime changed")
    return doc


def raw_digest(rgb):
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def source_clock(table):
    return [p * table.tb for p in table.pts]


def inspect_frames(path, expected, full=False):
    table = probe_frames(path, timeout_s=20)
    times = source_clock(table)
    require(not (table.rotation or table.interpolated or table.truncated) and len(times) >= 40,
            "Need an upright complete native clock with enough context")
    if expected is not None:
        require(times == [Fraction(ms, 1000) for ms in expected["picture_pts_ms"]], "Native media clock differs")
    selected = tuple(range(len(table))) if full else sample_indices(times)
    return table, selected


def sample_indices(times):
    anchor = times[0]
    targets = (anchor + i * STEP for i in range(int((times[-1] - anchor) / STEP) + 1))
    picks = []
    for target in targets:
        index = min(range(len(times)), key=lambda i: (abs(times[i] - target), i))
        if not picks or index > picks[-1]:
            picks.append(index)
    return tuple(picks)


def exact_rgb(path, table, wanted, expected):
    found = []
    for frame in decode_frames(path, table, wanted, timeout_s=45):
        require(frame.pts_verified and frame.pts == table.pts[frame.index], "Sample lacks its native timestamp")
        digest = raw_digest(frame.rgb)
        require(expected is None or digest == expected["rgb_sha256"][frame.index], "Decoded RGB differs from receipt")
        found.append(frame.index)
        yield frame, digest
    require(found == list(wanted), "Native sample roster differs")


def original_grid(doc, index):
    pin = next(p for p in doc["grid"]["held"] if p["ordinal"] == index)
    path, row, _ = prior.inspect(doc, pin)[0]
    require(sha(path) == row["file_sha256"], "GRID original file differs")
    return path, row


def prepare(folder):
    doc = frozen(folder)
    output = folder / "controls"
    output.mkdir(mode=0o700)
    roster, originals = [], {}
    for pin in doc["grid"]["held"]:
        index = pin["ordinal"]
        path, row = original_grid(doc, index)
        table, wanted = inspect_frames(path, row["expected"], full=True)
        originals[index] = (table, [frame.rgb for frame, _ in exact_rgb(path, table, wanted, row["expected"])])
        dest = output / f"reencode-{index:02d}.mkv"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-copyts", "-threads", "1", "-i", str(path),
                        "-map", "0:v:0", "-map", "0:a:0", "-fps_mode", "passthrough", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-threads", "1", "-preset", "veryfast", "-crf", "28", "-c:a", "copy",
                        str(dest)], check=True, timeout=45)
        clock, ids = inspect_frames(dest, {"picture_pts_ms": row["expected"]["picture_pts_ms"]}, full=True)
        roster.append({"condition": "reencode", "index": index, "file": dest.name, "sha256": sha(dest),
                       "expected": {"picture_pts_ms": row["expected"]["picture_pts_ms"],
                                    "rgb_sha256": [digest for _, digest in exact_rgb(dest, clock, ids, None)]}})
    for n, (left, right) in enumerate(doc["cut_pairs"]):
        table, base = originals[left]
        other, donor = originals[right]
        require((table.width, table.height, source_clock(table)) == (other.width, other.height, source_clock(other))
                and source_clock(table) == [Fraction(i, 25) for i in range(75)], "Cut needs matched native geometry/clock")
        a, b = CUT
        dest = output / f"cut-{n:02d}.mp4"
        expected = encode_rgb(dest, [*base[:a], *donor[a:b], *base[b:]], table.width, table.height, Fraction(1, 25))
        verify_rgb(dest, expected, Fraction(1, 25))
        roster.append({"condition": "hard_cut", "index": n, "file": dest.name, "sha256": sha(dest),
                       "expected": {"picture_pts_ms": [40 * i for i in range(75)], "rgb_sha256": expected}})
    write(folder / "controls.json", {"freeze_sha256": sha(folder / "freeze.json"), "roster": roster})


def native_bounds(geometry):
    _, matrix, mouth, shape = geometry
    inverse = cv2.invertAffineTransform(matrix)
    x, y = mouth
    corners = np.asarray(((x-30, y-25, 1), (x+30, y-25, 1), (x-30, y+25, 1), (x+30, y+25, 1))) @ inverse.T
    low = np.maximum(0, np.floor(corners.min(axis=0))).astype(int)
    high = np.minimum(shape[::-1], np.ceil(corners.max(axis=0))).astype(int)
    return [int(v) for v in [*low, *high]]


def patch_flow(flow, center, offset, radius=(5, 3)):
    x, y = center + offset
    w, h = radius
    sample = flow[max(0, y-h):min(prior.SIZE, y+h+1), max(0, x-w):min(prior.SIZE, x+w+1)]
    if len(sample) < 3 or sample.shape[1] < 3:
        return None
    return np.median(sample.reshape((-1, 2)), axis=0)


def flow(a, b, elapsed):
    return cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 13, 3, 5, 1.2, 0) * float(STEP / elapsed)


def motion_features(flow_map, mouth):
    reference = patch_flow(flow_map, mouth, SPEC["pose_reference_xy"])
    strain = [patch_flow(flow_map, mouth, offset, (4, 2)) for offset in SPEC["mouth_strain_boxes_xy"]]
    nearby = [patch_flow(flow_map, mouth, offset) for offset in SPEC["jaw_boxes_xy"]]
    if reference is None or any(value is None for value in strain + nearby):
        return None
    lower, upper, right, left = strain
    predictors = np.concatenate([point - reference for point in nearby])
    targets = np.array([(lower - upper)[1], (right - left)[0], ((lower + upper) / 2 - reference)[1]])
    return predictors.tolist(), targets.tolist()


def measure(path, row, yunet, geometry_cache):
    require(sha(path) == row["file_sha256"], "Media bytes differ from the qualified input")
    table, indices = inspect_frames(path, row["expected"])
    times = source_clock(table)
    detector = ShotDetector(table)
    frames, shots = [], []
    for frame, digest in exact_rgb(path, table, indices, row["expected"]):
        if digest not in geometry_cache:
            aligned = prior.aligned_face(frame.rgb, yunet)
            crop_hash = None
            if aligned is not None:
                x0, y0, x1, y1 = native_bounds(aligned)
                crop_hash = raw_digest(frame.rgb[y0:y1, x0:x1])
            geometry_cache[digest] = (aligned, crop_hash)
        aligned, crop_hash = geometry_cache[digest]
        detector.update(frame.index, frame.rgb)
        shots.append(len(detector.boundaries))
        frames.append((frame.index, frame.pts, digest, aligned, crop_hash))
    flows = []
    for j, (a, b) in enumerate(zip(frames, frames[1:])):
        elapsed = times[b[0]] - times[a[0]]
        flows.append(flow(a[3][0], b[3][0], elapsed) if a[3] is not None and b[3] is not None
                     and shots[j] == shots[j+1] and 0 < elapsed <= Fraction(SPEC["max_flow_gap_s"]) else None)
    direct, design = [None] * len(frames), [None] * len(frames)
    for i in range(1, len(frames) - 1):
        before, center, after = (frame[3] for frame in frames[i-1:i+2])
        if flows[i-1] is None or flows[i] is None:
            continue
        direct[i] = prior.contrast(center[0], before[0], after[0], prior.support_masks(center[2]))
        design[i] = motion_features((flows[i-1] + flows[i]) * .5, center[2])
    anchors = [None if geometry is None else {"frame": index, "pts": pts, "rgb_sha256": digest,
        "native_crop_xyxy": native_bounds(geometry), "crop_rgb_sha256": crop_hash}
        for index, pts, digest, geometry, crop_hash in frames]
    return table, indices, shots, direct, design, anchors


def best_window(series, table, indices, shots, anchors, symmetric=False):
    best = None
    for width in SPEC["core_samples"]:
        for start in range(4, len(series) - width - 3):
            end = start + width
            if shots[start-3] != shots[end+2]:
                continue
            parts = [list(range(start-3, start)), list(range(start, end)), list(range(end, end+3))]
            use = [[i for i in group if series[i] is not None] for group in parts]
            if min(len(use[0]), len(use[2])) < 2 or len(use[1]) < (4*width + 4)//5:
                continue
            left, middle, right = (float(np.median([series[i] for i in group])) for group in use)
            response = middle - max(left, right)
            if symmetric:
                response = max(response, min(left, right) - middle)
            if best is not None and response <= best["score"]:
                continue
            pick = (use[0][-1], use[1][len(use[1])//2], use[2][0])
            start_frame, end_frame = indices[start], indices[end]
            best = {"score": response, "frames": [start_frame, end_frame], "time_base": table.time_base,
                    "source_clock_s": [str(table.pts[i] * table.tb) for i in (start_frame, end_frame)],
                    "medians": [left, middle, right],
                    "anchors": [{**anchors[i], "role": role} for i, role in zip(pick, ("before", "during", "after"), strict=True)]}
    return best


def fit(design):
    x, y = (np.asarray(values, dtype=np.float64) for values in zip(*design, strict=True))
    require(len(x) >= 120 and x.shape[1] == 10 and y.shape[1] == 3, "Insufficient genuine joint motion")
    x_center, y_center = np.median(x, axis=0), np.median(y, axis=0)
    x_scale, y_scale = [np.maximum(.15, np.median(np.abs(a-c), axis=0) * 1.4826)
                        for a, c in ((x, x_center), (y, y_center))]
    a, b = (x-x_center)/x_scale, (y-y_center)/y_scale
    weights = np.linalg.solve(a.T @ a + np.eye(10) * len(a) * RIDGE, a.T @ b)
    return {key: array.tolist() for key, array in (("x_center", x_center), ("x_scale", x_scale),
        ("y_center", y_center), ("y_scale", y_scale), ("weights", weights))}


def residual(series, model):
    x, xs, y, ys, weights = (np.asarray(model[key]) for key in ("x_center", "x_scale", "y_center", "y_scale", "weights"))
    return [None if item is None else float(np.mean(np.abs((item[1]-y)/ys - ((item[0]-x)/xs) @ weights))) for item in series]


def score(measured, model):
    table, ids, shots, direct, design, anchors = measured
    return {name: best_window(series, table, ids, shots, anchors, symmetric=name == "jaw_coupling")
            for name, series in zip(ARMS, (direct, residual(design, model)), strict=True)}


def case_row(path, row, motion, measured, model):
    table = measured[0]
    edit = row["edit_frames_half_open"] if row.get("known_ai_provenance") or row.get("neural_provenance_verified") else None
    return {"condition": row["condition"], "file": str(path), "sha256": row["file_sha256"],
            "truth": [str(table.pts[i] * table.tb) for i in edit] if edit else None,
            "source_time_compatible": row["condition"] == "liveportrait_source_time" and motion == prior.SOURCE_CLOCK,
            "arms": score(measured, model), "sample_count": len(measured[1]), "cuts": len(set(measured[2])) - 1}


def grid_cases(doc, split):
    return [(pin["ordinal"], pin["component"], prior.inspect(doc, pin)) for pin in doc["grid"][split]]


def development(folder):
    doc = frozen(folder)
    detector = load_yunet(prior.WEIGHT)
    fit_rows, measurements = [], []
    for index, _, cases in grid_cases(doc, "development"):
        cache = {}
        for path, row, motion in cases:
            if row["condition"] in prior.NEURAL:
                continue
            observed = measure(path, row, detector, cache)
            if row["condition"] == "original":
                fit_rows.extend(item for item in observed[4] if item is not None)
            measurements.append((path, row, motion, observed))
        print("development", index, flush=True)
    model = fit(fit_rows)
    rows = [case_row(path, row, motion, observed, model) for path, row, motion, observed in measurements]
    points = {}
    for arm in ARMS:
        scores = sorted(row["arms"][arm]["score"] for row in rows if row["arms"][arm] is not None)
        require(len(scores) >= 40, "Insufficient independent control measurements")
        points[arm] = scores[-3]
    write(folder / "calibration.json", {"freeze_sha256": sha(folder / "freeze.json"), "points": points, "model": model,
        "fit_genuine_motion_pairs": len(fit_rows), "ordinary_development": rows})


def run(folder):
    doc, seal = frozen(folder), read(folder / "calibration.json")
    extras = read(folder / "controls.json")
    require(seal["freeze_sha256"] == extras["freeze_sha256"] == sha(folder / "freeze.json"), "Different sealed source run")
    native = native_groups(Path(doc["archive"]))
    detector = load_yunet(prior.WEIGHT)
    start = time.monotonic()
    with (folder / "scored.jsonl").open("x") as out:
        def emit(bank, index, component, cases):
            cache = {}
            rows = [case_row(path, row, motion, measure(path, row, detector, cache), seal["model"])
                    for path, row, motion in cases]
            out.write(json.dumps({"bank": bank, "index": index, "component": component, "cases": rows},
                                 sort_keys=True, allow_nan=False) + "\n")
            out.flush()
            print(bank, index, len(rows), "views", flush=True)

        for index, component, cases in grid_cases(doc, "held"):
            emit("grid", index, component, cases)
        for pin in doc["native"]:
            rows = native[pin["index"]]
            require(sha(DATA / pin["journal"]) == pin["journal_sha256"] and [row["file_sha256"] for row in rows] == pin["files"],
                    "Confirmation source commitment differs")
            emit("native", pin["index"], pin["index"], [(Path(doc["archive"]) / row["file"], row, None) for row in rows])
        for row in extras["roster"]:
            source = folder / "controls" / row["file"]
            emit("delivery", row["index"], None, [(source, {**row, "file_sha256": row["sha256"]}, None)])
    write(folder / "run.json", {"elapsed_s": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "journal_sha256": sha(folder / "scored.jsonl"), "calibration_sha256": sha(folder / "calibration.json"),
        "controls_sha256": sha(folder / "controls.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare", "development", "run"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.stage == "freeze":
        freeze(args.folder, args.archive)
    else:
        {"prepare": prepare, "development": development, "run": run}[args.stage](args.folder)


if __name__ == "__main__":
    main()
