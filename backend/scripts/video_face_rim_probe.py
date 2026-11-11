"""Model-free face-rim measurements on committed native inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.forensics import noise_map
from app.detectors.video.decode import decode_frames, probe_frames

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "backend/data/independent-evaluation"
POLICY = {
    "selection": "first 12 source groups of each prepared roster by independent salted id digest",
    "salt": "chipmunk-face-rim-study/2026-11-11",
    "side": 112,
    "minimum_native_side": 64,
    "candidate_comparisons": ["rim_spatial", "inner_temporal"],
    "legacy_comparison": "abs(log(face/background noise ratio))",
    "comparison_points": "90th-percentile development negative clip maxima, threshold excludes equality",
    "sampling": "up to sixteen evenly spaced centers over available exact consecutive triplets",
    "score_time": "native frame seconds of greatest score, earlier frame breaks ties",
    "summary": "selected frame must land inside known half-open edit interval",
}
FEATURES = ("rim_spatial", "inner_temporal", "legacy_noise")
grid_y, grid_x = np.ogrid[:112, :112]
radius = np.sqrt(((grid_x / 112 - 0.5) / 0.29) ** 2 + ((grid_y / 112 - 0.62) / 0.31) ** 2)
CORE, INNER = radius <= 0.68, (radius > 0.67) & (radius <= 0.86)
RIM, OUTER = (radius > 0.86) & (radius <= 1.02), (radius > 1.02) & (radius <= 1.17)
INTERIOR = (grid_y >= 8) & (grid_y < 104) & (grid_x >= 8) & (grid_x < 104)
REGIONS = [region & INTERIOR for region in (CORE, INNER, RIM, OUTER)]
MIN_REGION_PIXELS = 100
EPSILON = 1.0


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows(path):
    with path.open() as f:
        return {r["id"]: r for line in f if "id" in (r := json.loads(line))}


def plan():
    runs = (
        ("development", "video-v16-sface", "panel.json", "development.jsonl", "private-features-development"),
        (
            "confirmation",
            "video-v17-sface",
            "private-confirmation-panel.json",
            "candidate.jsonl",
            "private-features-confirmation",
        ),
    )
    selected, inputs = [], {}
    for role, directory, panel, saved, features in runs:
        base = PRIVATE / directory
        saved_rows = rows(base / saved)
        records = [r for r in load(base / panel)["items"] if r["role"] == role]
        assert len(records) == 120
        groups = sorted(
            {r["group"] for r in records},
            key=lambda x: hashlib.sha256((POLICY["salt"] + x).encode()).hexdigest(),
        )
        for r in records:
            if r["group"] not in groups[:12]:
                continue
            filename = base / features / f"{r['id']}.json"
            source = Path(r["source_path"])
            candidate = saved_rows[r["id"]]
            assert candidate["source_sha256"] == r["sha256"]
            assert r["status"] == "ready" and (not r["label"] or r["exact_frames_verified"])
            assert filename.is_file() and source.is_file()
            selected.append(
                {
                    "role": role,
                    "id": r["id"],
                    "group": r["group"],
                    "family": r["family"],
                    "label": r["label"],
                    "edit_s": r["edit_s"],
                    "source": str(source),
                    "source_sha256": r["sha256"],
                    "features": str(filename),
                    "features_sha256": candidate["private_feature_packet_sha256"],
                    "frames": r["frame_count"],
                }
            )
        inputs[role] = {"panel": sha(base / panel), "response_receipts": sha(base / saved)}
    assert len(selected) == 96
    assert not (
        {r["group"] for r in selected if r["role"] == "development"}
        & {r["group"] for r in selected if r["role"] == "confirmation"}
    )
    return {"policy": POLICY, "inputs": inputs, "program_sha256": sha(Path(__file__)), "cases": selected}


def prepare_crop(rgb, row):
    side = row.get("current_alignment_side", 0)
    if side < POLICY["minimum_native_side"]:
        return None
    matrix = np.asarray([[float.fromhex(v) for v in line] for line in row["current_alignment_hex"]])
    native = cv2.warpAffine(rgb, matrix, (side, side), flags=cv2.INTER_LINEAR)
    aligned = cv2.resize(native, (112, 112), interpolation=cv2.INTER_AREA)
    assert hashlib.sha256(aligned.tobytes()).hexdigest() == row["current_crop_rgb_sha256"]
    native_valid = cv2.warpAffine(np.full(rgb.shape[:2], 255, dtype=np.uint8), matrix, (side, side))
    valid = cv2.resize(native_valid, (112, 112), interpolation=cv2.INTER_AREA) == 255
    return cv2.cvtColor(aligned, cv2.COLOR_RGB2YCrCb).astype(np.float32), valid


def regional_values(values, valid):
    sites = [valid & region for region in REGIONS]
    if any(int(region.sum()) < MIN_REGION_PIXELS for region in sites):
        return None
    return [float(np.median(values[region])) for region in sites]


def compare_frame(center, before, after):
    rgb, valid = center
    # Detail discontinuities on the suspected blend rim, relative to its two neighbors.
    detail = np.mean(np.abs(rgb - cv2.GaussianBlur(rgb, (5, 5), 0.8)), axis=2)
    regional = regional_values(detail, valid)
    spatial = None
    if regional is not None:
        _, inner, rim, outer = regional
        spatial = abs(math.log((rim + EPSILON) / (math.sqrt((inner + EPSILON) * (outer + EPSILON)))))
    temporal = None
    if before is not None and after is not None:
        sides = []
        for flank, allowed in (before, after):
            delta = rgb - flank
            common = valid & allowed
            if int(common.sum()) < MIN_REGION_PIXELS:
                break
            delta -= np.median(delta[common], axis=0)
            measured = regional_values(np.mean(np.abs(delta), axis=2), common)
            if measured is None:
                break
            core, _, _, outer = measured
            sides.append(math.log((core + EPSILON) / (outer + EPSILON)))
        if len(sides) == 2:
            temporal = min(sides)
    return spatial, temporal


def measure(r):
    source, feature_path = Path(r["source"]), Path(r["features"])
    assert sha(source) == r["source_sha256"] and sha(feature_path) == r["features_sha256"]
    packet = load(feature_path)
    assert packet["source_sha256"] == r["source_sha256"]
    bindings = {row["frame_index"]: row for row in packet["frame_inputs"]}
    table = probe_frames(source, timeout_s=25)
    assert len(table) == r["frames"] and not (table.rotation or table.truncated or table.interpolated)
    centers = sorted(i for i in bindings if i - 1 in bindings and i + 1 in bindings)
    centers = centers[:: max(1, math.ceil(len(centers) / 16))]
    indices = sorted({i + delta for i in centers for delta in (-1, 0, 1)})
    best = {name: None for name in FEATURES}
    coverage = Counter()
    crops, noise = {}, {}
    seen = []
    frames = decode_frames(source, table, indices, timeout_s=35) if indices else ()
    for frame in frames:
        row = bindings[frame.index]
        assert frame.pts == row["pts"] and Fraction(row["time_base"]) == table.tb
        assert hashlib.sha256(frame.rgb.tobytes()).hexdigest() == row["source_rgb_sha256"]
        crops[frame.index] = prepare_crop(frame.rgb, row)
        if frame.index in centers:
            x, y, w, h = map(round, row["bbox"])
            metrics, _ = noise_map(frame.rgb, face_box=(x, y, w, h))
            ratio = metrics.get("noise_face_bg_ratio")
            if ratio and ratio > 0:
                noise[frame.index] = abs(math.log(ratio))
        seen.append(frame.index)
    assert seen == indices
    for i in centers:
        center = crops[i]
        spatial, temporal = compare_frame(center, crops[i - 1], crops[i + 1]) if center else (None, None)
        for name, value in zip(FEATURES, (spatial, temporal, noise.get(i)), strict=True):
            if value is None:
                continue
            assert math.isfinite(value)
            coverage[name] += 1
            if best[name] is None or value > best[name][1]:
                best[name] = [table.seconds(i), value]
    return {"best": best, "coverage": coverage, "native_frames_verified": len(indices)}


def evaluate(frozen, trial: Path):
    results = []
    for index, row in enumerate(frozen["cases"], start=1):
        measured = measure(row)
        results.append({**row, **measured})
        if index % 8 == 0:
            print(f"{index}/{len(frozen['cases'])} clips checked", flush=True)
    (trial / "private-results.json").write_text(json.dumps(results) + "\n")
    aggregate = {}
    for name in FEATURES:
        negatives = sorted(
            r["best"][name][1]
            for r in results
            if r["role"] == "development" and not r["label"] and r["best"][name]
        )
        point = negatives[max(0, math.ceil(0.90 * len(negatives)) - 1)]
        outcomes = {}
        for role in ("development", "confirmation"):
            grouped = {}
            for r in (r for r in results if r["role"] == role):
                best = r["best"][name]
                hit = bool(best and r["edit_s"] and r["edit_s"][0] <= best[0] < r["edit_s"][1])
                flag = bool(best and best[1] > point)
                counts = grouped.setdefault(r["family"], Counter())
                counts.update(
                    cases=1,
                    assessed=int(best is not None),
                    flags=int(flag),
                    max_inside_edit=int(hit),
                    above_point_inside_edit=int(hit and flag),
                    measured_frames=r["coverage"].get(name, 0),
                )
            outcomes[role] = grouped
        aggregate[name] = {"development_point": point, "outcomes": outcomes}
    answer = {
        "status": "complete",
        "freeze_sha256": sha(trial / "freeze.json"),
        "cases": len(results),
        "native_frames_verified": sum(r["native_frames_verified"] for r in results),
        "neural_calls": 0,
        "measurements": aggregate,
    }
    (trial / "aggregate.json").write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n")
    print(json.dumps(answer))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "run"))
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    trial = args.trial.resolve()
    if args.operation == "freeze":
        frozen = plan()
        trial.mkdir(mode=0o700, exist_ok=False)
        with (trial / "freeze.json").open("x") as file:
            json.dump(frozen, file, indent=2)
        print(sha(trial / "freeze.json"))
        return

    if any((trial / name).exists() for name in ("private-results.json", "aggregate.json")):
        raise ValueError("Use a new trial to preserve prior observations")
    frozen = load(trial / "freeze.json")
    if frozen != plan():
        raise ValueError("Exact frozen program and input plan required")
    cv2.setNumThreads(1)
    evaluate(frozen, trial)


if __name__ == "__main__":
    main()
