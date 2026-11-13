# SPDX-License-Identifier: Apache-2.0
"""Freeze and compare native and shortlisted six-picture face proposals on HEVC."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from scripts import video_face_720p30_transfer as clocks
from scripts import video_face_paired_edge_hevc_confirmation as delivery
from scripts import video_face_paired_edge_transfer as frozen

ROOT = frozen.ROOT
DATA = frozen.DATA
EDIT_BANK = DATA / "video-face-two-returns-hd-2026-11-12"
CUT_BANK = DATA / "video-face-hd-retry-scene-challenge-2026-11-12"
SCHEMA = "chipmunk.video-face-unrouted-short-transfer/1"
EDIT_CASES = {"original", "same_source_cuts", "cross_recording_cuts", "two_swaps", "tandem_two_swaps"}
CUT_CASES = {"person_switch", "tandem_person_switch"}
GATE = {
    "minimum_added": 4,
    "minimum_new_groups": 3,
    "minimum_exact": 2,
    "minimum_shortlist_share_of_dense": 0.75,
    "maximum_ordinary_flags": 1,
    "maximum_speaker_cut_flags": 0,
    "maximum_wrong_edit_ranges": 1,
}
PLAN = {
    "durations": [5, 6, 7],
    "dense_edges": [6, 114],
    "candidate_point_floor": 0.10,
    "shape": "720p30 hvc1 yuv420p MP4, 120 native upright frames",
    "anchor_samples": list(range(4, 120, 10)),
    "shortlist_edges": 16,
    "shortlist_max_model_frames": 84,
    "near_head_guard": "fixed paired-edge motion on each proposed boundary",
    "study_design": "Fit selection point against development ordinary cases. Score the fixed development and source-disjoint held bank. "
    "After scoring compare selected ranges to Full on the same held bytes. At least .5 IoU counts.",
}
read, save, sha, require = frozen.read, frozen.save, frozen.sha, frozen.require


def bindings(development):
    from app.detectors.video import face_paired_edge

    dev = read(development / "receipt.json")
    require(
        sha(development / "media.json") == dev["media_sha256"],
        "Development media differs from its frozen receipt",
    )
    files = {
        "development": development / "receipt.json",
        "development_media": development / "media.json",
        "edited_media": EDIT_BANK / "media.json",
        "edited_freeze": EDIT_BANK / "freeze.json",
        "control_media": CUT_BANK / "media.json",
        "control_freeze": CUT_BANK / "freeze.json",
        "method": Path(__file__),
        "guard": Path(face_paired_edge.__file__),
    }
    for folder in (EDIT_BANK, CUT_BANK):
        require(
            read(folder / "media.json")["freeze_sha256"] == sha(folder / "freeze.json"),
            "Archived freeze changed",
        )
    return {
        "schema": SCHEMA,
        "plan": PLAN,
        "gate": GATE,
        "weights": frozen.WEIGHTS,
        "pins": {name: sha(path) for name, path in files.items()},
    }


def prepare(run, development):
    run.mkdir(mode=0o700)
    save(run / "freeze.json", bindings(development))
    rows = []
    for folder, cases, bank in (
        (EDIT_BANK, EDIT_CASES, "source_edits"),
        (CUT_BANK, CUT_CASES, "speaker_cuts"),
    ):
        for item in read(folder / "media.json")["rows"]:
            if item["condition"] not in cases:
                continue
            source = folder / "media" / item["file"]
            require(sha(source) == item["sha256"], "Source delivery changed")
            path = run / f"{bank}-{source.name}"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    str(source),
                    *delivery.ENCODE,
                    str(path),
                ],
                check=True,
                timeout=35,
            )
            require(delivery.ffprobe(path) == [delivery.ENCODING], "Unexpected delivery profile")
            row = clocks.describe(path, item["group"], item["condition"], item["truth"])
            require(row["clock"]["pts"] == item["clock"]["pts"], "Transfer changed native picture times")
            rows.append({**row, "bank": bank, "h264_sha256": item["sha256"]})
    require(len(rows) == len({row["sha256"] for row in rows}) == 56, "Held roster differs")
    save(run / "media.json", {"freeze_sha256": sha(run / "freeze.json"), "rows": rows})
    print("Prepared held sources", len(rows), flush=True)


def edge_indices(edge):
    return list(range(edge - 3, edge + 3))


def proposals(values, reduced, edges, pts):
    import numpy as np

    from app.detectors.image.faces import iou
    from app.detectors.video.face_appearance import distance
    from app.detectors.video.face_paired_edge import motion

    supported = {}
    for edge in edges:
        group = edge_indices(edge)
        if not all(i in values for i in group) or any(
            iou(values[a][1], values[b][1]) < 0.25 for a, b in zip(group, group[1:])
        ):
            continue
        sides = [np.median([values[i][0] for i in indices], axis=0) for indices in (group[:3], group[3:])]
        left, right = sides
        separation = distance(left, right)
        if separation < 0.08:
            continue
        guard = motion(edge, reduced, values[edge - 1][2]["bbox"])
        if guard is None:
            continue
        dispersion = max(
            distance(values[i][0], sides[side]) for side in (0, 1) for i in group[side * 3 : side * 3 + 3]
        )
        supported[edge] = left, right, separation - dispersion, guard
    candidates = []
    for start, (before, first, left, entry) in supported.items():
        for duration in PLAN["durations"]:
            stop = start + duration
            if stop not in supported:
                continue
            last, after, right, exit_edge = supported[stop]
            response = min(
                left,
                right,
                min(distance(before, first), distance(after, last))
                - max(distance(before, after), distance(first, last)),
            )
            during = min(
                (i for i in range(start, stop) if i in values), key=lambda i: (abs(2 * i - start - stop), i)
            )
            anchors = [start - 1, during, stop]
            candidates.append(
                {
                    "frames": [start, stop],
                    "pts": [pts[start], pts[stop]],
                    "response": response,
                    "anchors": [
                        {
                            k: values[i][2][k]
                            for k in (
                                "frame_index",
                                "pts",
                                "bbox",
                                "decoded_rgb_sha256",
                                "aligned_rgb_sha256",
                            )
                        }
                        for i in anchors
                    ],
                    "edges": [entry, exit_edge],
                }
            )
    return sorted(candidates, key=lambda row: (-row["response"], *row["frames"]))


def shortlist(reduced, values, edges):
    import numpy as np

    from app.detectors.video.face_paired_edge import motion

    sampled = [values[i][2]["bbox"] for i in PLAN["anchor_samples"] if i in values]
    if len(sampled) < 4:
        return None
    box = {key: float(np.median([item[key] for item in sampled])) for key in ("x", "y", "w", "h")}
    ranked = []
    for edge in edges:
        if guard := motion(edge, reduced, box):
            ranked.append((float.fromhex(guard["central_hex"]), edge))
    kept = []
    measured = set(PLAN["anchor_samples"])
    for _, edge in sorted(ranked, key=lambda item: (-item[0], item[1])):
        indices = set(edge_indices(edge))
        if len(measured | indices) > PLAN["shortlist_max_model_frames"]:
            continue
        kept.append(edge)
        measured.update(indices)
        if len(kept) == PLAN["shortlist_edges"]:
            break
    return kept, len(measured)


def measure(path, row, models):
    table, values, reduced = frozen.frames(path, row, range(120), *models)
    edges = range(PLAN["dense_edges"][0], PLAN["dense_edges"][1] + 1)
    plan = shortlist(reduced, values, edges)
    measurements = {"dense": proposals(values, reduced, edges, table.pts)}
    measurements["shortlist"] = proposals(values, reduced, plan[0], table.pts) if plan else []
    return {
        "candidates": measurements,
        "faces": len(values),
        "shortlist_model_reads": plan[1] if plan else 12,
    }


def measure_panel(run, development, role):
    import cv2

    from app.detectors.image.faces import load_yunet

    require(read(run / "freeze.json") == bindings(development), "Frozen method or inputs changed")
    cv2.setNumThreads(2)
    weights = ROOT / "models/weights"
    require(
        {name: sha(weights / name) for name in frozen.WEIGHTS} == frozen.WEIGHTS, "Detector weights changed"
    )
    models = (
        load_yunet(weights / "face_detection_yunet_2023mar.onnx"),
        cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), ""),
    )
    folder = development if role == "development" else run
    media = read(folder / "media.json")
    if role == "held":
        require(media["freeze_sha256"] == sha(run / "freeze.json"), "Held media differs from freeze")
        require((run / "point.json").is_file(), "Fix the development comparison point before held scoring")
    all_rows = []
    for ordinal, row in enumerate(media["rows"], 1):
        path = folder / row["file"]
        outcome = measure(path, row, models)
        result = {**{key: row[key] for key in ("sha256", "group", "condition", "truth", "bank")}, **outcome}
        all_rows.append(result)
        print(
            role,
            ordinal,
            row["condition"],
            row["group"],
            len(outcome["candidates"]["dense"]),
            len(outcome["candidates"]["shortlist"]),
            flush=True,
        )
    save(run / f"{role}.json", {"media_sha256": sha(folder / "media.json"), "rows": all_rows})
    if role == "development":
        points = {
            arm: max(
                [
                    PLAN["candidate_point_floor"],
                    *(p["response"] for row in all_rows if not row["truth"] for p in row["candidates"][arm]),
                ]
            )
            for arm in ("dense", "shortlist")
        }
        save(run / "point.json", {"development_sha256": sha(run / "development.json"), "points": points})


def matches(a, b):
    return frozen.overlaps_half(a, b)


def chosen(values, point):
    result = []
    for row in values:
        if row["response"] <= point or any(matches(row["frames"], earlier["frames"]) for earlier in result):
            continue
        result.append(row)
        if len(result) == 2:
            break
    return result


def count(rows, point, incumbent):
    results = {}
    for arm in ("dense", "shortlist"):
        groups = set()
        totals = Counter()
        for row in rows:
            prior = incumbent.get(row["sha256"], [])
            spans = [r["frames"] for r in chosen(row["candidates"][arm], point[arm])]
            target = [r for r in row["truth"] if r[1] - r[0] == 6]
            hits = [
                r
                for r in target
                if any(matches(r, s) for s in spans) and not any(matches(r, s) for s in prior)
            ]
            if hits:
                groups.add(row["group"])
            totals.update(
                files=1,
                target=len(target),
                added=len(hits),
                exact=sum(r in spans for r in hits),
                ordinary_files=not row["truth"],
                ordinary_flags=bool(not row["truth"] and spans),
                cut_flags=bool(row["condition"] in CUT_CASES and spans),
                wrong_edit_ranges=sum(not any(matches(span, t) for t in row["truth"]) for span in spans)
                if row["truth"]
                else 0,
            )
        results[arm] = {**dict(totals), "gained_groups": len(groups)}
    return results


def summary(run, development):
    point = read(run / "point.json")
    require(point["development_sha256"] == sha(run / "development.json"), "Selection measurements changed")
    compare = run / "incumbent.json"
    held = read(run / "held.json")
    require(held["media_sha256"] == sha(run / "media.json"), "Held measurement changed")
    before = read(development / "receipt.json")
    old = {r["sha256"]: r["prior"] for r in before["rows"]}
    comparison = read(compare) if compare.is_file() else None
    if comparison is not None:
        require(comparison["media_sha256"] == held["media_sha256"], "Incumbent ran on different bytes")
    new = comparison["rows"] if comparison else {}
    complete = set(new) == {row["sha256"] for row in held["rows"]}
    counted = count(held["rows"], point["points"], new)
    finalist = counted["shortlist"]
    gate = {
        "added": finalist["added"] >= GATE["minimum_added"],
        "groups": finalist["gained_groups"] >= GATE["minimum_new_groups"],
        "exact": finalist["exact"] >= GATE["minimum_exact"],
        "relative_to_dense": finalist["added"]
        >= counted["dense"]["added"] * GATE["minimum_shortlist_share_of_dense"],
        "ordinary": finalist["ordinary_flags"] <= GATE["maximum_ordinary_flags"],
        "speaker_cuts": finalist["cut_flags"] <= GATE["maximum_speaker_cut_flags"],
        "wrong_ranges": finalist["wrong_edit_ranges"] <= GATE["maximum_wrong_edit_ranges"],
    }
    return {
        "point": point,
        "development": count(read(run / "development.json")["rows"], point["points"], old),
        "held": counted,
        "held_incumbent_complete": complete,
        "gate": gate if complete else None,
        "passed": complete and all(gate.values()),
    }


def incumbent(run):
    import os

    from scripts import screen_video_native_background_challenge as worker

    state = worker.configure(run)
    os.environ["VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT"] = "true"
    compare = worker.scorer(state)
    records = {}
    for index, row in enumerate(read(run / "media.json")["rows"], 1):
        result = compare(run / row["file"], row)
        records[row["sha256"]] = frozen.intervals(result, row["clock"]["pts"])
        print("incumbent", index, row["condition"], row["group"], records[row["sha256"]], flush=True)
    save(run / "incumbent.json", {"media_sha256": sha(run / "media.json"), "rows": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "development", "held", "incumbent", "summary"))
    parser.add_argument("run", type=Path)
    parser.add_argument("development", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run, args.development)
    elif args.phase == "incumbent":
        incumbent(args.run)
    elif args.phase == "summary":
        print(json.dumps(summary(args.run, args.development), indent=2, sort_keys=True))
    else:
        measure_panel(args.run, args.development, args.phase)
