# SPDX-License-Identifier: Apache-2.0
"""Local research replay for a short, spatially corroborated face-texture change."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time
from types import SimpleNamespace

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces, iou, load_yunet
from app.detectors.video import face_appearance
from app.detectors.video.decode import decode_frames, probe_frames

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation/video-native-inserts-v95"
WEIGHTS = ROOT / "models/weights"
FACE_FILE = WEIGHTS / "face_detection_yunet_2023mar.onnx"
APPEARANCE_FILE = WEIGHTS / "face_recognition_sface_2021dec.onnx"
CONTROL_CONDITIONS = ("original", "ordinary_light", "ordinary_blur", "ordinary_time_step")
SWAP_CONDITIONS = ("swap_a", "swap_b")
DURATIONS = (Fraction(3, 4), Fraction(3, 2))
REGIONS = ((.30, .62), (.70, .62), (.50, .27), (-.15, .62), (1.15, .62), (.50, -.15))
PATCH_SIZE = 24
PATCH_OFFSETS = (np.arange(PATCH_SIZE, dtype=np.float32) + .5) / PATCH_SIZE * 2 - 1
FREQUENCY = np.hypot(*np.meshgrid(np.arange(PATCH_SIZE), np.arange(PATCH_SIZE)))
MID = (FREQUENCY >= 4) & (FREQUENCY < 10)
HIGH = (FREQUENCY >= 10) & (FREQUENCY < 18)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def bindings():
    files = (
        Path(__file__), FACE_FILE, APPEARANCE_FILE, ROOT / "backend/app/detectors/image/faces.py",
        ROOT / "backend/app/detectors/video/decode.py", ROOT / "backend/app/detectors/video/face_appearance.py",
    )
    return {
        **{str(path.relative_to(ROOT)): digest(path) for path in files},
        "ffmpeg": digest(Path(shutil.which("ffmpeg"))),
        "ffprobe": digest(Path(shutil.which("ffprobe"))),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def freeze(output, media):
    roster = []
    for split, role, limit, conditions in (
        ("fit", "fusion_fit", 4, CONTROL_CONDITIONS),
        ("held", "confirmation", 6, CONTROL_CONDITIONS + SWAP_CONDITIONS),
    ):
        options = []
        for path in sorted(DATA.glob("group-*-complete.json")):
            receipt = read(path)
            if receipt["role"] != role:
                continue
            table = receipt["cases"][0]["frame_table"]
            if table["width"] < 1280 or table["height"] < 720 or len(table["pts"]) < 144:
                continue
            by_name = {case["condition"]: case for case in receipt["cases"]}
            review_path = DATA / f"group-{receipt['index']:03d}-command-review-v96/result.json"
            if not review_path.exists():
                continue
            review = read(review_path)
            receipt_sha = digest(path)
            if review["status"] != "complete" or review["private_group_receipt_sha256"] != receipt_sha:
                continue
            cases = {name: by_name[name] for name in conditions}
            case_root = media / f"group-{receipt['index']:03d}"
            if any((case_root / case["file"]).stat().st_size != case["bytes"] for case in cases.values()):
                continue
            options.append((sum(case["bytes"] for case in cases.values()), receipt["index"], receipt_sha, review_path))
        if len(options) < limit:
            raise RuntimeError(f"Too few eligible exact native groups for {split}")
        for _, index, receipt_sha, review_path in sorted(options)[:limit]:
            roster.append({"index": index, "split": split, "receipt_sha256": receipt_sha,
                           "command_review_sha256": digest(review_path), "conditions": list(conditions)})
    write(output / "freeze.json", {
        "schema": "chipmunk.video-face-texture-return-freeze/1",
        "bindings": bindings(), "roster": roster,
        "eligibility": "smallest total required byte length per frozen prior source role; HD six-second native clips; no score or face-based selection",
        "interventions": "fixed 0.75- and 1.50-second spans centered on 3 seconds; native half-open ceil endpoints; exact matched source frames chosen in memory",
        "sampling": "8 Hz nearest native clock, earlier ties; appearance at 4 Hz with adjacent native frame triplets; single observed face only",
        "candidate": "local six-region luminance spectral ratio; compare signed short interior shift against both flanks, variability, and same-direction scene change; corroborated boundaries add up to one quarter of score",
        "point": "maximum assessed score among fit genuine and ordinary-edit views, strictly greater than; no swaps in fit",
        "held": "six source-disjoint prior confirmation groups, evaluated only after fixed point is sealed",
        "readout": "highest range per arm, overlap at least 0.20, wrong ranges and unassessed separately; union counts exact and duplicate errors",
    })


def relative_times(table):
    return [(pts - table.pts[0]) * table.tb for pts in table.pts]


def schedule(times, hz):
    targets = (Fraction(i, hz) for i in range(int(times[-1] * hz) + 1))
    return list(dict.fromkeys(min(range(len(times)), key=lambda i: (abs(times[i] - target), i)) for target in targets))


def intervention(times, duration):
    return tuple(next(i for i, instant in enumerate(times) if instant >= edge)
                 for edge in (3 - duration / 2, 3 + duration / 2))


def spectral_descriptor(gray, face):
    if min(face.w, face.h) < 64:
        return None
    origin = np.array([face.x + face.w / 2, face.y + face.h / 2])
    eye = np.subtract(face.landmarks[1], face.landmarks[0])
    if np.linalg.norm(eye) < 8 or eye[0] <= 0:
        return None
    horizontal = eye / np.linalg.norm(eye)
    vertical = np.array([-horizontal[1], horizontal[0]])
    height, width = gray.shape
    values = []
    for x, y in REGIONS:
        center = origin + horizontal * ((x - .5) * face.w) + vertical * ((y - .5) * face.h)
        offset_x, offset_y = np.meshgrid(PATCH_OFFSETS * .105 * face.w, PATCH_OFFSETS * .105 * face.h)
        map_x = (center[0] + offset_x * horizontal[0] + offset_y * vertical[0]).astype(np.float32)
        map_y = (center[1] + offset_x * horizontal[1] + offset_y * vertical[1]).astype(np.float32)
        if map_x.min() < 1 or map_x.max() >= width - 1 or map_y.min() < 1 or map_y.max() >= height - 1:
            return None
        pixels = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR).astype(np.float32)
        if not 12 <= np.median(pixels) <= 243:
            return None
        spectrum = cv2.dct(pixels - pixels.mean()) ** 2
        mid = float(spectrum[MID].sum() / pixels.size)
        high = float(spectrum[HIGH].sum() / pixels.size)
        if mid < .25:
            return None
        values.append(float(np.log((high + .125) / (mid + .125))))
    return np.array(values)


def best_texture(indices, times, features):
    """Sweep only adjacent complete samples of one observed face."""
    candidates = []
    for start in range(4, len(indices) - 7):
        for end in range(start + 4, min(start + 15, len(indices) - 3)):
            lo, hi = times[indices[start]], times[indices[end]]
            if not Fraction(1, 2) <= hi - lo <= Fraction(7, 4):
                continue
            chosen = indices[start - 4:end + 4]
            if any(i not in features or features[i]["texture"] is None for i in chosen):
                continue
            if any(times[b] - times[a] > Fraction(1, 5) or iou(features[a]["face"], features[b]["face"]) < .35
                   for a, b in zip(chosen, chosen[1:])):
                continue
            groups = [np.array([features[i]["texture"] for i in part])
                      for part in (indices[start - 4:start], indices[start:end], indices[end:end + 4])]
            before, core, after = (np.median(values, axis=0) for values in groups)
            scatter = np.max([np.median(np.abs(g - m), axis=0) for g, m in zip(groups, (before, core, after))], axis=0)
            change = np.abs(after - before) * .5 + scatter
            evidence = []
            for polarity in (1, -1):
                excess = np.maximum(0, np.minimum(polarity * (core - before), polarity * (core - after)) - change)
                core_support = max(0, np.median(excess[:3]) - np.median(excess[3:]))
                entry = polarity * (groups[1][0] - groups[0][-1])
                exit = polarity * (groups[1][-1] - groups[2][0])
                both_edges = np.maximum(0, np.minimum(entry, exit) - change)
                edge_support = max(0, np.median(both_edges[:3]) - np.median(both_edges[3:]))
                evidence.append(float(.75 * core_support + .25 * min(core_support, edge_support)))
            candidates.append((max(evidence), lo, hi, chosen[3], indices[(start + end - 1) // 2], indices[end]))
    if not candidates:
        return None
    score, lo, hi, before, during, after = min(candidates, key=lambda row: (-row[0], row[1], row[2]))
    return {"score": score, "start": str(lo), "end": str(hi), "anchors": [before, during, after]}


def appearance(centers, table, times, features):
    vectors = {i: item["appearance"] for i, item in features.items() if item["appearance"] is not None}
    tracks = {"single-face": [SimpleNamespace(frame_index=i) for i in sorted(vectors)]}
    windows = face_appearance.windows(set(centers), tracks, vectors, table)
    if not windows:
        return None
    best = windows[0]
    centers = best["centers"]
    return {"score": best["response"], "start": str(times[centers[2]]), "end": str(times[centers[-3]]),
            "anchors": [centers[1], centers[len(centers) // 2], centers[-2]]}


def source_features(path, case, wanted, centers_and_adjacent, yunet, sface):
    if digest(path) != case["sha256"]:
        raise RuntimeError("Source file changed after independent receipt")
    table = probe_frames(path, timeout_s=25)
    expected = case["frame_table"]
    rational = Fraction(expected["time_base"])
    if (table.interpolated or table.truncated or table.width != expected["width"] or table.height != expected["height"]
            or [p * table.tb for p in table.pts] != [p * rational for p in expected["pts"]]):
        raise RuntimeError("Source clock or dimensions differ from verified receipt")
    rows, seen = {}, set()
    if wanted:
        for frame in decode_frames(path, table, sorted(wanted), timeout_s=65):
            index = frame.index
            rgb = hashlib.sha256(frame.rgb).hexdigest()
            if not frame.pts_verified or rgb != case["expected_rgb_sha256"][index]:
                raise RuntimeError("Unverified native sample RGB or clock")
            seen.add(index)
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            faces = detect_faces(bgr, detector=yunet, min_size=48)
            if len(faces) != 1:
                continue
            face = faces[0]
            feature = None
            if index in centers_and_adjacent:
                row = face_appearance.face_row(face.landmarks)
                row[0, :4] = face.box()
                row[0, 14] = face.score
                feature = sface.feature(sface.alignCrop(bgr, row)).reshape(-1).copy()
                face_appearance.unit(feature)
            gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
            rows[index] = {"face": face, "appearance": feature, "texture": spectral_descriptor(gray, face)}
    if seen != wanted:
        raise RuntimeError("Missing a fixed native sample")
    return table, rows, len(seen)


def group_run(entry, media, yunet, sface):
    index = entry["index"]
    receipt_path = DATA / f"group-{index:03d}-complete.json"
    review_path = DATA / f"group-{index:03d}-command-review-v96/result.json"
    if digest(receipt_path) != entry["receipt_sha256"] or digest(review_path) != entry["command_review_sha256"]:
        raise RuntimeError("Precommitted group receipts differ")
    receipt = read(receipt_path)
    cases = {row["condition"]: row for row in receipt["cases"]}
    original = cases["original"]
    expected = original["frame_table"]
    step = Fraction(expected["time_base"])
    times = [(pts - expected["pts"][0]) * step for pts in expected["pts"]]
    indices = schedule(times, 8)
    centers = [i for i in schedule(times, 4) if 0 < i < len(times) - 1]
    adjacent = {i + delta for i in centers for delta in (-1, 0, 1)}
    wanted = set(indices) | adjacent
    case_root = media / f"group-{index:03d}"
    table, original_features, num_decoded = source_features(
        case_root / original["file"], original, wanted, adjacent, yunet, sface
    )
    intervals = {str(d): intervention(times, d) for d in DURATIONS}
    rows = [{"condition": "original", "duration": None, "truth": None,
             "appearance": appearance(centers, table, times, original_features),
             "texture": best_texture(indices, times, original_features)}]
    for condition in entry["conditions"]:
        if condition == "original":
            continue
        case = cases[condition]
        lo, hi = intervals[str(max(DURATIONS))]
        if [i for i in range(lo, hi) if i not in case["actual_changed_frames"]]:
            raise RuntimeError("Condition does not change every declared edit picture")
        variant_indices = {i for i in wanted if lo <= i < hi}
        variant_table, variant_features, decoded = source_features(
            case_root / case["file"], case, variant_indices, adjacent, yunet, sface
        )
        if relative_times(variant_table) != times:
            raise RuntimeError("Matched edit no longer has the same clock")
        num_decoded += decoded
        for duration, (a, b) in intervals.items():
            combined = {i: value for i, value in original_features.items() if not a <= i < b}
            combined.update({i: value for i, value in variant_features.items() if a <= i < b})
            rows.append({"condition": condition, "duration": duration, "truth": [str(times[a]), str(times[b])],
                         "appearance": appearance(centers, table, times, combined),
                         "texture": best_texture(indices, times, combined)})
    return {"group": index, "decoded": num_decoded, "rows": rows}


def run(output, media, split):
    frozen = read(output / "freeze.json")
    if bindings() != frozen["bindings"]:
        raise RuntimeError("Worker, media tools, weights or comparison implementation changed")
    if split == "held" and not (output / "point.json").exists():
        raise RuntimeError("Seal the genuine-only inspection point before opening held results")
    cv2.setNumThreads(2)
    yunet = load_yunet(FACE_FILE)
    sface = cv2.FaceRecognizerSF.create(str(APPEARANCE_FILE), "")
    start = time.monotonic()
    results = []
    with (output / f"{split}.jsonl").open("x") as stream:
        for entry in frozen["roster"]:
            if entry["split"] != split:
                continue
            result = group_run(entry, media, yunet, sface)
            results.append(result)
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            print(split, len(results), "groups", sum(group["decoded"] for group in results), "native pictures", flush=True)
    if split == "fit":
        assessed = [row["texture"]["score"] for group in results for row in group["rows"] if row["texture"] is not None]
        if not assessed:
            raise RuntimeError("No measured genuine training controls")
        write(output / "point.json", {"point": max(assessed), "fit_cases": sum(len(g["rows"]) for g in results),
                                      "assessed": len(assessed), "fit_sha256": digest(output / "fit.jsonl")})
    write(output / f"{split}-run.json", {"groups": len(results), "seconds": time.monotonic() - start,
                                        "native_samples": sum(g["decoded"] for g in results),
                                        "max_self_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def overlap(result, truth):
    start, end = Fraction(result["start"]), Fraction(result["end"])
    lo, hi = map(Fraction, truth)
    return max(0, min(hi, end) - max(lo, start)) / (max(hi, end) - min(lo, start))


def readout(output):
    points = {"appearance": face_appearance.REVIEW_POINT, "texture": read(output / "point.json")["point"]}
    held = [json.loads(line) for line in (output / "held.jsonl").read_text().splitlines()]
    counts, joint, groups = {}, Counter(), {}
    for group in held:
        group_counts = Counter()
        for row in group["rows"]:
            positive = row["condition"] in SWAP_CONDITIONS
            key = row["condition"] + "/" + (row["duration"] or "none")
            cell = counts.setdefault(key, {arm: Counter() for arm in points})
            flagged = {}
            for arm, point in points.items():
                response = row[arm]
                counter = cell[arm]
                counter["cases"] += 1
                flagged[arm] = response is not None and response["score"] > point
                if response is None:
                    counter["unassessed"] += 1
                    continue
                if not flagged[arm]:
                    continue
                counter["flagged"] += 1
                if positive:
                    counter["localized" if overlap(response, row["truth"]) >= Fraction(1, 5) else "wrong_range"] += 1
                else:
                    counter["false"] += 1
                    group_counts[f"{arm}_false"] += 1
            if positive:
                hits = {arm: value and overlap(row[arm], row["truth"]) >= Fraction(1, 5)
                        for arm, value in flagged.items()}
                group_counts["positive_views"] += 1
                for arm in points:
                    group_counts[f"{arm}_localized"] += hits[arm]
                joint["union_localized"] += any(hits.values())
                joint["added_localized"] += hits["texture"] and not hits["appearance"]
                joint["additional_wrong_ranges"] += flagged["texture"] and not hits["texture"]
                group_counts["additional_localized"] += hits["texture"] and not hits["appearance"]
            else:
                joint["union_false_views"] += any(flagged.values())
                joint["added_false_views"] += flagged["texture"] and not flagged["appearance"]
                group_counts["additional_false_views"] += flagged["texture"] and not flagged["appearance"]
        for name in group_counts:
            if group_counts[name]:
                groups[name] = groups.get(name, 0) + 1
    write(output / "aggregate.json", {
        "schema": "chipmunk.video-face-texture-return-result/1", "points": points,
        "fit": read(output / "point.json"), "held_groups": len(held), "by_condition_duration": counts,
        "union": joint, "held_source_groups_with_outcome": groups,
        "runs": {split: read(output / f"{split}-run.json") for split in ("fit", "held")},
        "commitments": {f: digest(output / f) for f in ("freeze.json", "fit.jsonl", "point.json", "held.jsonl")},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "held", "readout"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    if args.stage == "readout":
        readout(args.output)
    elif args.stage == "freeze":
        freeze(args.output, args.media)
    else:
        run(args.output, args.media, args.stage)


if __name__ == "__main__":
    main()
