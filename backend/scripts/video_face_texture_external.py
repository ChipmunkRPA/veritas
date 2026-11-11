# SPDX-License-Identifier: Apache-2.0
"""Apply the frozen local face texture point to independent camera and publisher sources."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import resource
import time

import cv2

from scripts import video_face_texture_return as method

CAMERA_DIR = "video-vcd2-delivery-v206"
PUBLISHER_DIR = "video-faceswapai-matched-v186"
PUBLISHER_RECEIPT = method.ROOT / (
    "backend/data/independent-evaluation/video-faceswapai-matched-v186/bind-00/output/private-manifest.json"
)


def freeze(output, archive):
    cases = []
    camera = archive / CAMERA_DIR
    sizes = []
    for path in sorted(camera.glob("slot-*-attempt-01/qualification.json")):
        receipt = method.read(path)
        size = sum(c["bytes"] for c in receipt["cases"].values())
        sizes.append((size, path))
    if len(sizes) != 12:
        raise RuntimeError("Incomplete camera archive index")
    for _, path in sorted(sizes)[:2]:
        cases.append({"type": "camera", "relative": str(path.relative_to(archive)), "sha256": method.digest(path)})
    cases.append({"type": "publisher", "relative": str(PUBLISHER_RECEIPT.relative_to(method.ROOT)),
                  "sha256": method.digest(PUBLISHER_RECEIPT)})
    method.write(output / "external-freeze.json", {
        "schema": "chipmunk.video-face-texture-external-freeze/1",
        "primary_freeze_sha256": method.digest(output / "freeze.json"),
        "point_sha256": method.digest(output / "point.json"),
        "external_worker_sha256": method.digest(Path(__file__)), "sources": cases,
        "plan": "two smallest complete camera source groups; one earlier publisher source; unchanged and two short intervals per variant",
        "centers": {"camera": "9/2", "publisher": "37/8"},
    })


def unpack(entry, archive):
    camera = entry["type"] == "camera"
    path = (archive if camera else method.ROOT) / entry["relative"]
    if method.digest(path) != entry["sha256"]:
        raise RuntimeError("External receipt differs from freeze")
    receipt = method.read(path)
    if not camera:
        return archive / PUBLISHER_DIR, receipt["cases"]
    table = {k: receipt[k] for k in ("width", "height", "time_base")}
    table["pts"] = receipt["relative_pts"]
    return path.parent, [{"condition": k, "frame_table": table, **v} for k, v in receipt["cases"].items()]


def measure(folder, rows, center, yunet, sface):
    cases = {case["condition"]: {**case, "expected_rgb_sha256": case["rgb_sha256"]} for case in rows}
    base = cases["unchanged"]
    table = base["frame_table"]
    step = Fraction(table["time_base"])
    times = [(p - table["pts"][0]) * step for p in table["pts"]]
    indices = method.schedule(times, 8)
    centers = [i for i in method.schedule(times, 4) if 0 < i < len(times) - 1]
    adjacent = {i + delta for i in centers for delta in (-1, 0, 1)}
    wanted = set(indices) | adjacent
    observed, plain, decoded = method.source_features(folder / base["file"], base, wanted, adjacent, yunet, sface)
    result = [{"condition": "unchanged", "duration": None, "truth": None,
               "appearance": method.appearance(centers, observed, times, plain),
               "texture": method.best_texture(indices, times, plain)}]
    intervals = {str(duration): tuple(next(i for i, t in enumerate(times) if t >= endpoint)
                                      for endpoint in (center - duration / 2, center + duration / 2))
                 for duration in method.DURATIONS}
    a, b = intervals[str(max(method.DURATIONS))]
    variant_indices = {i for i in wanted if a <= i < b}
    for name, case in cases.items():
        if name == "unchanged":
            continue
        if any(case["expected_rgb_sha256"][i] == base["expected_rgb_sha256"][i] for i in range(a, b)):
            raise RuntimeError("Published camera or swap edit fails complete native boundary coverage")
        clock, candidate, n = method.source_features(
            folder / case["file"], case, variant_indices, adjacent, yunet, sface
        )
        decoded += n
        if method.relative_times(clock) != times:
            raise RuntimeError("Variant clock changed")
        for duration, (a, b) in intervals.items():
            inputs = {i: value for i, value in plain.items() if not a <= i < b}
            inputs.update({i: value for i, value in candidate.items() if a <= i < b})
            result.append({"condition": name, "duration": duration,
                           "truth": [str(times[a]), str(times[b])] if name.startswith("swap_") else None,
                           "appearance": method.appearance(centers, observed, times, inputs),
                           "texture": method.best_texture(indices, times, inputs)})
    return result, decoded


def run(output, archive):
    frozen = method.read(output / "external-freeze.json")
    if (frozen["external_worker_sha256"] != method.digest(Path(__file__))
            or frozen["primary_freeze_sha256"] != method.digest(output / "freeze.json")
            or frozen["point_sha256"] != method.digest(output / "point.json")
            or method.bindings() != method.read(output / "freeze.json")["bindings"]):
        raise RuntimeError("Frozen point, model or method changed")
    points = {"appearance": method.face_appearance.REVIEW_POINT, "texture": method.read(output / "point.json")["point"]}
    cv2.setNumThreads(2)
    yunet = method.load_yunet(method.FACE_FILE)
    sface = cv2.FaceRecognizerSF.create(str(method.APPEARANCE_FILE), "")
    start = time.monotonic()
    decoded = 0
    counts = {}
    with (output / "external.jsonl").open("x") as stream:
        for entry in frozen["sources"]:
            cohort = entry["type"]
            folder, cases = unpack(entry, archive)
            rows, n = measure(folder, cases, Fraction(frozen["centers"][cohort]), yunet, sface)
            decoded += n
            bucket = counts.setdefault(cohort, {arm: Counter() for arm in points})
            for row in rows:
                truth = row["truth"]
                for arm, point in points.items():
                    result = row[arm]
                    count = bucket[arm]
                    kind = "swap" if truth else "control"
                    count[f"{kind}_cases"] += 1
                    if result is None:
                        count[f"{kind}_unassessed"] += 1
                    elif result["score"] > point:
                        if truth:
                            count["localized" if method.overlap(result, truth) >= Fraction(1, 5) else "wrong_range"] += 1
                        else:
                            count["false_control_flags"] += 1
            stream.write(json.dumps({"cohort": cohort, "rows": rows, "native_samples": n}, sort_keys=True) + "\n")
            stream.flush()
            print(cohort, len(rows), "views", n, "native samples", flush=True)
    method.write(output / "external-aggregate.json", {
        "cohorts": counts, "native_samples": decoded, "seconds": time.monotonic() - start,
        "max_self_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "freeze_sha256": method.digest(output / "external-freeze.json"),
        "raw_results_sha256": method.digest(output / "external.jsonl"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("output", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output, args.archive)
    else:
        run(args.output, args.archive)
