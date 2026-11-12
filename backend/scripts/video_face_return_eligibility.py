# SPDX-License-Identifier: Apache-2.0
"""Compare two independent ways to recover brief native face-change ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from scripts import screen_video_native_background_challenge as worker

PROFILES = "exact upright 640x360 H.264 yuv420p at 25 fps, 120 pictures"
POSITIVES = {"six_even": (50, 56), "six_odd": (63, 69), "ten_odd": (77, 87), "medium": (46, 62)}
CONTROLS = {"original": None, "recompressed": None, "light": (60, 70), "jump": (63, 69), "cut": (77, 87)}
ARMS = ("dense_after_sparse_unassessed", "eligible_before_rank")
SPEC = {
    "schema": "chipmunk.video-face-return-eligibility/1",
    "sources": "Use every complete remaining native recording outside the 77 documented source and donor groups in earlier localization studies, excluding aliases. Previous general archive exposure is possible.",
    "profile": PROFILES, "positives": {name: list(value) for name, value in POSITIVES.items()},
    "controls": {name: list(value) if value else None for name, value in CONTROLS.items()},
    "source_exactness": "Before any scoring, try the named interval with its assigned publisher variant then the other variant. If a native picture is identical, try intervals shifted by +2 then -2 pictures with the same variant order. Retain unmatched planned cases as source-unavailable.",
    "arms": {
        ARMS[0]: "When the 8 Hz pass completed exact measurements but no guarded context existed, attempt the unchanged 12.5 Hz method. Do not run after missing native clocks, multiple faces, partial decode, or a selected result.",
        ARMS[1]: "When production selected no short interval, rank the already measured candidates that first satisfy the same duration, point and coarse non-duplication criteria. Try the measured sparse pass before the measured dense pass. Keep all production ranges.",
    },
    "gate": {"minimum_new_useful": 3, "minimum_source_groups": 2, "maximum_new_ordinary_flags": 1, "maximum_new_cut_flags": 0},
    "localization": "Native half-open IoU at least one half. Count each transformed delivery and each group separately. These review ranges carry zero AI-confidence weight.",
}


def dependencies():
    from scripts import video_native_background_challenge as prior

    return prior


def freeze(archive):
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance

    prior = dependencies()
    base = prior.earlier.previous.prior.DATA / "video-native-background-challenge-2026-11-12"
    parent = prior.common.read(base / "freeze.json")
    candidates, past = prior.pool(archive)
    used = {pin["index"] for pin in parent["roster"]}
    prior.require(len(used) == 12, "The prior exact source roster is required")
    roster = [pin for pin in candidates if pin["index"] not in used]
    roster.sort(key=lambda row: hashlib.sha256(f"face-return-eligibility/1/{row['index']}".encode()).hexdigest())
    prior.require(len(roster) >= 8 and len(roster) % 2 == 0, "At least eight remaining paired native source groups are required")
    files = (Path(__file__), Path(face_return.__file__), Path(face_appearance.__file__), Path(worker.__file__))
    return {"spec": SPEC, "roster": roster, "prior_freezes": [prior.common.digest(p) for p in [*past, base / "freeze.json"]],
            "implementation": {path.name: prior.common.digest(path) for path in files}}


def frozen(folder, archive):
    prior = dependencies()
    lock = prior.common.read(folder / "freeze.json")
    prior.require(lock == freeze(archive), "Source and exact method freeze differs")
    return lock


def _variant(group, name):
    family = int(name in {"six_odd", "medium"}) ^ (group % 2)
    return ("swap_a", "swap_b")[family]


def _resolved_swap(native, assets, group, name):
    import numpy as np

    a, b = POSITIVES[name]
    preferred = _variant(group, name)
    alternate = "swap_b" if preferred == "swap_a" else "swap_a"
    for shift in (0, 2, -2):
        start, end = a + shift, b + shift
        for variant in (preferred, alternate):
            if all(not np.array_equal(native[i], assets[variant][i]) for i in range(start, end)):
                return (start, end), variant, {i: assets[variant][i] for i in range(start, end)}
    return None


def prepare(folder, archive):
    import cv2
    import numpy as np

    prior = dependencies()
    cv2.setNumThreads(1)
    lock = frozen(folder, archive)
    directory = folder / "media"
    directory.mkdir(mode=0o700)
    items, unavailable = [], []
    wanted = sorted({i for a, b in POSITIVES.values() for i in range(a - 2, b + 2)})
    for first in range(0, len(lock["roster"]), 2):
        pair = lock["roster"][first:first + 2]
        native = [prior.earlier.source_pictures(archive, pin, "original", range(120)) for pin in pair]
        for side, pin in enumerate(pair):
            group = first + side
            original = directory / f"g{group:02d}-original.mp4"
            prior.old.encoded.encode(original, frames=native[side].values(), crf=23)
            assets = {name: prior.earlier.source_pictures(archive, pin, name, wanted) for name in ("swap_a", "swap_b")}
            for name, interval in {**CONTROLS, **POSITIVES}.items():
                path = directory / f"g{group:02d}-{name}.mp4"
                truth = interval if name in POSITIVES else None
                variant = None
                if name == "recompressed":
                    prior.old.encoded.encode(path, source=original, crf=36)
                elif interval is not None:
                    if truth is not None:
                        resolved = _resolved_swap(native[side], assets, group, name)
                        if resolved is None:
                            unavailable.append({"group": group, "condition": name, "nominal": list(truth)})
                            continue
                        interval, variant, changed = resolved
                        truth = interval
                    else:
                        a, b = interval
                        if name == "cut":
                            changed = {i: native[1 - side][i] for i in range(a, b)}
                        elif name == "jump":
                            changed = {i: native[side][i + 19] for i in range(a, b)}
                        else:
                            prior.require(name == "light", "Unknown controlled intervention")
                            changed = {i: np.clip(np.rint(native[side][i].astype(np.float32) * 1.20), 0, 255).astype(np.uint8)
                                       for i in range(a, b)}
                    a, b = interval
                    prior.require(all(not np.array_equal(native[side][i], changed[i]) for i in range(a, b)),
                                  "Every authored interval picture must differ")
                    frames = (changed.get(i, rgb) for i, rgb in native[side].items())
                    prior.old.encoded.encode(path, frames=frames, crf=23)
                row = prior.old.describe(path, group, "held", name, truth)
                items.append({**row, "authored": list(interval) if interval else None, "publisher_variant": variant})
            print(f"Authored source {group + 1}/{len(lock['roster'])}", flush=True)
    prior.require(len(items) == len({row["sha256"] for row in items}) and len(items) + len(unavailable) == len(lock["roster"]) * 9,
                  "Incomplete distinct roster")
    prior.common.write(folder / "media.json", {"freeze_sha256": prior.common.digest(folder / "freeze.json"),
                       "media": items, "unavailable": unavailable})


def ranks(times, indices, features, table, coarse):
    import math

    import numpy as np

    from app.detectors.video import face_return as method

    visible = {i: features[i] for i in indices if features[i].vector is not None and features[i].face is not None}
    if len(visible) < 9:
        return None, None
    order = {i: n for n, i in enumerate(visible)}
    vectors = np.asarray([item.vector for item in visible.values()])
    distances = 1. - np.clip(vectors @ vectors.T, -1., 1.)
    adjacent = {(a, b): method.scene_correlation(features[a], features[b]) for a, b in zip(indices, indices[1:])}
    winner = eligible = None
    for width in method.CORE_SIZES:
        needed = max(3, math.ceil(width * .66))
        flank = method.FLANK_SIZE
        for entry in range(flank, len(indices) - width - flank + 1):
            exit_ = entry + width
            support = tuple(tuple(i for i in group if i in visible) for group in (
                indices[entry - flank:entry], indices[entry:exit_], indices[exit_:exit_ + flank]))
            before, middle, after = support
            if len(middle) < needed or min(len(before), len(after)) < method.MINIMUM_FLANK:
                continue
            ordered = before + middle + after
            if any(times[b] - times[a] > method.MAX_GAP or method.iou(visible[a].face, visible[b].face) < method.MINIMUM_FACE_IOU
                   for a, b in zip(ordered, ordered[1:])):
                continue
            edges = indices[entry - 1], indices[entry], indices[exit_ - 1], indices[exit_]
            first, last = adjacent[edges[:2]], adjacent[edges[2:]]
            if first is None or last is None or min(first, last) < method.SCENE_CORRELATION_POINT:
                continue
            response = method.distribution_margin(distances, *([order[i] for i in group] for group in support))
            candidate = method.Range(indices[entry], indices[exit_], response, (first, last), support, edges)
            if winner is None or response > winner.response:
                winner = candidate
            if method.distinct(candidate, table, coarse) and (eligible is None or response > eligible.response):
                eligible = candidate
    return winner, eligible


def _compact(candidate, features, frames, table):
    return {"frames": [candidate.start, candidate.end], "response_hex": float(candidate.response).hex(),
            "scene_correlations": list(candidate.scene_correlations),
            "support": [list(items) for items in candidate.support], "scene_frames": list(candidate.scene_frames),
            "anchors": {role: {"index": index, "pts": table.pts[index], "rgb_sha256": frames[index],
                               "box": list(features[index].face.box())}
                        for role, index in candidate.anchors().items()}}


def evaluate(folder, archive, state):
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance as adapter

    prior = dependencies()
    manifest = prior.common.read(folder / "media.json")
    lock = frozen(folder, archive)
    prior.require(manifest["freeze_sha256"] == prior.common.digest(folder / "freeze.json"), "Native media does not match the freeze")
    run = worker.scorer(state)
    measured = []
    original = adapter._inspect_return
    highest = face_return.highest
    record = {}

    def capture(times, indices, features):
        result = highest(times, indices, features)
        measured.append((times, indices, features, result))
        return result

    def inspect(details, inputs, yunet, recognizer, deadline, incumbent):
        stop = time.monotonic() + adapter.RETURN_BUDGET_S
        with patch.object(face_return, "highest", capture):
            original(details, inputs, yunet, recognizer, deadline, incumbent)
        coverage = details.get("face_return_check") or {}
        table = inputs.table
        if not details.get("face_return"):
            for times, indices, features, observed in measured:
                winner, accepted = ranks(times, indices, features, table, incumbent)
                prior.require(winner == observed, "Reference enumeration diverged from production ranking")
                if accepted is not None and ARMS[1] not in record:
                    record[ARMS[1]] = (accepted, features, table, len(indices))
            if coverage.get("reason") == "insufficient_same_shot_context" and time.monotonic() < stop:
                dense = face_return.dense_schedule(table, inputs.encoding)
                if dense is not None:
                    new = {}
                    status = {"measured_face_frames": 0}
                    with patch.object(face_return, "highest", capture):
                        adapter._return_pass(new, inputs, yunet, recognizer, deadline, incumbent, dense, status,
                                             adapter.DENSE_FRAME_PROFILES[len(table)], stop)
                    record["expanded_coverage"] = status
                    if new.get("face_return"):
                        _, indices, features, winner = measured[-1]
                        record[ARMS[0]] = (winner, features, table, len(indices))

    count = len(manifest["media"])
    prior.require(count + len(manifest["unavailable"]) == len(lock["roster"]) * 9, "Incomplete native roster")
    started = time.monotonic()
    journal = folder / "results.jsonl"
    with journal.open("x") as stream, patch.object(adapter, "_inspect_return", inspect):
        for number, row in enumerate(manifest["media"], 1):
            measured.clear()
            record.clear()
            path = folder / "media" / row["file"]
            actual = run(path, row)
            selected = {}
            for arm in ARMS:
                if arm in record:
                    item, features, table, sample_count = record[arm]
                    selected[arm] = {**_compact(item, features, row["rgb_sha256"], table), "sample_count": sample_count}
            keys = ("file", "group", "condition", "sha256", "truth", "authored")
            result = {**{key: row[key] for key in keys}, "incumbent": actual, "candidates": selected,
                      "expanded_coverage": record.get("expanded_coverage")}
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            if number % 6 == 0:
                print(f"Measured {number}/{count} Full stage runs", flush=True)
    prior.common.write(folder / "score-seal.json", {"cases": count, "journal_sha256": prior.common.digest(journal),
                       "freeze_sha256": prior.common.digest(folder / "freeze.json"), "elapsed_s": time.monotonic() - started})


def interval(record, pts):
    return [pts.index(record["start_pts"]), pts.index(record["end_pts"])] if record else None


def overlap(first, second):
    left, right = first
    a, b = second
    return Fraction(max(0, min(right, b) - max(left, a)), max(right, b) - min(left, a))


def aggregate(folder):
    prior = dependencies()
    seal = prior.common.read(folder / "score-seal.json")
    journal = folder / "results.jsonl"
    prior.require(seal["journal_sha256"] == prior.common.digest(journal), "Exact held decisions required")
    prepared = prior.common.read(folder / "media.json")
    manifest = prepared["media"]
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    prior.require(len(rows) == seal["cases"] == len(manifest), "Incomplete held outcomes")
    counts = defaultdict(Counter)
    groups = defaultdict(set)
    baseline = Counter()
    for row, metadata in zip(rows, manifest, strict=True):
        prior.require(row["sha256"] == metadata["sha256"], "Delivered output changed")
        truth = row["truth"]
        category = "swap" if truth else ("cut" if row["condition"] == "cut" else "ordinary")
        pts = metadata["clock"]["pts"]
        before = [interval(row["incumbent"][key], pts) for key in ("coarse", "short_range", "boundary_refinement")]
        useful = truth is not None and any(item and overlap(item, truth) >= Fraction(1, 2) for item in before)
        baseline.update({f"{category}_cases": 1, f"{category}_useful": useful, f"{category}_flags": any(before)})
        for arm in ARMS:
            item = row["candidates"].get(arm)
            good = item is not None and truth is not None and overlap(item["frames"], truth) >= Fraction(1, 2)
            counts[arm].update({f"{category}_flags": item is not None, f"{category}_useful": good, "added_useful": good and not useful})
            if good and not useful:
                groups[arm].add(row["group"])
    out = {}
    for arm in ARMS:
        measured = counts[arm]
        qualifies = measured["added_useful"] >= 3 and len(groups[arm]) >= 2 and measured["ordinary_flags"] <= 1 and measured["cut_flags"] == 0
        out[arm] = {**measured, "added_source_groups": len(groups[arm]), "advances": qualifies}
    result = {"protocol": SPEC, "baseline": dict(baseline), "arms": out, "score_seal": seal, "serving_change": False,
              "source_unavailable_cases": prepared["unavailable"]}
    prior.common.write(folder / "aggregate.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    state = worker.configure(args.folder) if args.phase == "score" else None
    prior = dependencies()
    archive = args.archive or prior.ARCHIVE
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        prior.common.write(args.folder / "freeze.json", freeze(archive))
    elif args.phase == "prepare":
        prepare(args.folder, archive)
    elif args.phase == "score":
        evaluate(args.folder, archive, state)
    else:
        aggregate(args.folder)


if __name__ == "__main__":
    main()
