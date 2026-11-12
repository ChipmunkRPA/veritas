# SPDX-License-Identifier: Apache-2.0
"""Rebuild native timing fixtures and independently check recovery decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_face_return_eligibility as study

SPAN = {"six_even": [50, 56], "six_odd": [63, 69], "ten_odd": [77, 87], "medium": [46, 62],
        "original": None, "recompressed": None, "light": [60, 70], "jump": [63, 69], "cut": [77, 87]}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def reconstruct(folder, archive):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    prior = study.dependencies()
    lock = study.frozen(folder, archive)
    delivery = read(folder / "media.json")
    require(delivery["freeze_sha256"] == digest(folder / "freeze.json"), "Media is detached from its source freeze")
    rows = {(row["group"], row["condition"]): row for row in delivery["media"]}
    unavailable = {(row["group"], row["condition"]): row["nominal"] for row in delivery["unavailable"]}
    wanted = sorted({i for name, span in SPAN.items() if name in study.POSITIVES for i in range(span[0]-2, span[1]+2)})
    counts = Counter()
    with TemporaryDirectory(prefix="verify-face-eligibility-", dir=folder) as scratch:
        work = Path(scratch)
        for offset in range(0, len(lock["roster"]), 2):
            pins = lock["roster"][offset:offset + 2]
            sources = [list(prior.earlier.source_pictures(archive, pin, "original", range(120)).values()) for pin in pins]
            counts["native_original_pictures"] += 240
            for direction, pin in enumerate(pins):
                group = offset + direction
                plain, alternate = sources[direction], sources[1 - direction]
                native_swaps = {name: prior.earlier.source_pictures(archive, pin, name, wanted) for name in ("swap_a", "swap_b")}
                first = work / "original.mp4"
                for condition, nominal in SPAN.items():
                    span = nominal
                    family = None
                    if condition in study.POSITIVES:
                        variants = ("swap_a", "swap_b")
                        if (condition in ("six_odd", "medium")) != bool(group % 2):
                            variants = variants[::-1]
                        choices = ((a, b, variant) for shift in (0, 2, -2) for variant in variants
                                   for a, b in [(nominal[0]+shift, nominal[1]+shift)]
                                   if all(not np.array_equal(native_swaps[variant][i], plain[i]) for i in range(a, b)))
                        selected = next(choices, None)
                        if selected is None:
                            require(unavailable.pop((group, condition)) == nominal, "Missing source truth is not declared")
                            counts["source_unavailable"] += 1
                            continue
                        a, b, family = selected
                        span = [a, b]
                    row = rows.pop((group, condition))
                    require(row["publisher_variant"] == family and row["authored"] == span
                            and row["truth"] == (span if condition in study.POSITIVES else None),
                            "A declared interval differs from the authored recipe")
                    rebuilt = first if condition == "original" else work / "delivery.mp4"
                    actual = folder / "media" / row["file"]
                    require(digest(actual) == row["sha256"], "The delivered bytes differ from the manifest")
                    if condition == "recompressed":
                        prior.old.encoded.encode(rebuilt, source=first, crf=36)
                    else:
                        pictures = list(plain)
                        if span:
                            a, b = span
                            if condition in study.POSITIVES:
                                replacement = [native_swaps[family][i] for i in range(a,b)]
                                counts["native_publisher_swap_pictures"] += len(replacement)
                            elif condition == "cut":
                                replacement = alternate[a:b]
                            elif condition == "jump":
                                replacement = plain[a + 19:b + 19]
                            else:
                                require(condition == "light", "Unexpected ordinary edit")
                                replacement = [np.rint(np.minimum(p.astype(np.float32) * 1.2, 255)).astype(np.uint8) for p in plain[a:b]]
                            require(len(replacement) == b - a and
                                    all(not np.array_equal(p, q) for p, q in zip(plain[a:b], replacement, strict=True)),
                                    "Changed pictures are not the exact native interval")
                            pictures[a:b] = replacement
                            counts["authored_intervals"] += 1
                        prior.old.encoded.encode(rebuilt, frames=iter(pictures), crf=23)
                    require(digest(rebuilt) == row["sha256"], "Native reconstruction differs from the scored bytes")
                    counts["byte_identical_mp4s"] += 1
                    if rebuilt != first:
                        rebuilt.unlink()
                first.unlink()
            print(f"Rebuilt source pair {offset // 2 + 1}", flush=True)
    require(not rows and not unavailable and counts["byte_identical_mp4s"] == len(delivery["media"]),
            "Independent authoring replay is incomplete")
    receipt = {"status": "verified", "media_sha256": digest(folder / "media.json"), "counts": dict(counts)}
    prior.common.write(folder / "independent-authoring.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def _overlap(left, right):
    return 2 * max(0, min(left[1], right[1]) - max(left[0], right[0])) >= max(left[1], right[1]) - min(left[0], right[0])


def _span(record, clock):
    if record is None:
        return None
    pts = clock["pts"]
    require(record["time_base"] == clock["time_base"], "Incumbent record changed clocks")
    a, b = pts.index(record["start_pts"]), pts.index(record["end_pts"])
    require(a < b, "An incumbent range has invalid edges")
    return [a, b]


def _grid(samples):
    if samples == 60:
        return list(range(0, 120, 2))
    values = [min(range(120), key=lambda i: (abs(8*i - 25*j), i)) for j in range(39)]
    require(samples == len(values), "Unexpected rank sample schedule")
    return values


def _candidate(folder, delivery, result, coarse):
    from app.detectors.video.decode import FrameTable, decode_frames

    clock = delivery["clock"]
    base = Fraction(clock["time_base"])
    start, end = result["frames"]
    require(0 < end-start <= 12 and float.fromhex(result["response_hex"]) > .1
            and all(.55 <= val <= 1.00001 for val in result["scene_correlations"])
            and (coarse is None or not _overlap(coarse, [start, end])), "Recovery range failed an existing eligibility guard")
    grid = _grid(result["sample_count"])
    a, b = grid.index(start), grid.index(end)
    require(a >= 4 and b+4 <= len(grid) and b-a in {3, 4, 6}, "Candidate uses an unqualified support window")
    before, during, after = result["support"]
    for value, available, minimum in zip((before, during, after), (grid[a-4:a], grid[a:b], grid[b:b+4]),
                                         (3, 4 if b-a == 6 else 3, 3), strict=True):
        require(len(value) >= minimum and value == sorted(set(value)) and set(value) <= set(available),
                "Face support differs from the scheduled role window")
    require(result["scene_frames"] == [grid[a-1], start, grid[b-1], end], "Scene support differs from the scheduled edges")
    roles = {"before": before[-1], "during": during[len(during)//2], "after": after[0]}
    require({role: row["index"] for role, row in result["anchors"].items()} == roles, "Selected frame does not match its role")
    path = folder / "media" / delivery["file"]
    table = FrameTable.from_json(clock)
    for frame, (role, index) in zip(decode_frames(path, table, sorted(roles.values()), timeout_s=20), roles.items(), strict=True):
        source = result["anchors"][role]
        require(frame.index == index and frame.pts_verified and frame.pts == source["pts"] == table.pts[index]
                and hashlib.sha256(frame.rgb.tobytes()).hexdigest() == source["rgb_sha256"] == delivery["rgb_sha256"][index]
                and (table.pts[end]-table.pts[start])*base < Fraction(1, 2), "Recovery frame differs from the exact original")
    return len(roles)


def outcomes(folder, publication):
    from app.detectors.video.decode import FrameTable, decode_frames

    manifest = read(folder / "media.json")
    seal = read(folder / "score-seal.json")
    authoring = read(folder / "independent-authoring.json")
    recorded = read(folder / "aggregate.json")
    journal = folder / "results.jsonl"
    require(authoring["status"] == "verified" and authoring["media_sha256"] == digest(folder / "media.json")
            and seal["journal_sha256"] == digest(journal) and seal["freeze_sha256"] == digest(folder / "freeze.json"),
            "Independent authoring or score journal is unverified")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    counts, groups = defaultdict(Counter), defaultdict(set)
    baseline = Counter()
    coverage = Counter()
    pictures = 0
    coarse_inputs = 0
    selected = []
    for row, item in zip(rows, manifest["media"], strict=True):
        require(all(row[key] == item[key] for key in ("file", "group", "condition", "sha256", "truth", "authored")),
                "Outcome is detached from the committed source")
        category = "swap" if item["truth"] else ("cut" if item["condition"] == "cut" else "ordinary")
        prior = row["incumbent"]
        sparse = prior["short_coverage"] or {}
        dense = sparse.get("dense_fallback", {})
        expanded = row["expanded_coverage"] or {}
        coverage.update(
            f"{pass_name}/{value.get('reason') or value.get('status') or 'not_run'}"
            for pass_name, value in (("sparse", sparse), ("dense", dense), ("expanded", expanded))
        )
        coarse = _span(prior["coarse"], item["clock"])
        if coarse is not None:
            saved = {frame["frame_index"]: frame for frame in prior["coarse"]["frame_inputs"]}
            table = FrameTable.from_json(item["clock"])
            path = folder / "media" / item["file"]
            for frame, index in zip(decode_frames(path, table, sorted(saved), timeout_s=20), sorted(saved), strict=True):
                require(frame.index == index and frame.pts_verified and frame.pts == table.pts[index] == saved[index]["pts"]
                        and hashlib.sha256(frame.rgb.tobytes()).hexdigest() == item["rgb_sha256"][index]
                        == saved[index]["decoded_rgb_sha256"], "Incumbent coarse input differs from its source")
            coarse_inputs += len(saved)
        already = [coarse, _span(prior["short_range"], item["clock"]), _span(prior["boundary_refinement"], item["clock"])]
        useful_before = bool(item["truth"] and any(value and _overlap(value, item["truth"]) for value in already))
        baseline.update({category+"_cases": 1, category+"_flags": any(already), category+"_useful": useful_before})
        for arm in study.ARMS:
            chosen = row["candidates"].get(arm)
            useful = bool(chosen and item["truth"] and _overlap(chosen["frames"], item["truth"]))
            counts[arm].update({category+"_flags": chosen is not None, category+"_useful": useful,
                                "added_useful": useful and not useful_before})
            if useful and not useful_before:
                groups[arm].add(row["group"])
            if chosen:
                require(prior["short_range"] is None, "Recovery displaced an existing short range")
                if arm == study.ARMS[0]:
                    require(sparse["reason"] == "insufficient_same_shot_context"
                            and expanded["status"] == "localized", "Unexpected fallback scope")
                else:
                    measured = sparse if chosen["sample_count"] == 39 else dense
                    require(measured.get("status") == "no_distinct_range",
                            "Reranking used a pass that did not finish its timing and eligibility checks")
                pictures += _candidate(folder, item, chosen, coarse)
                selected.append({"source_sha256": row["sha256"], "group": row["group"], "condition": row["condition"],
                                 "method": arm, "frames": chosen["frames"], "known_edit": item["truth"],
                                 "useful": useful, "added_useful": useful and not useful_before})
    for arm, result in counts.items():
        result["added_source_groups"] = len(groups[arm])
        advances = result["added_useful"] >= 3 and len(groups[arm]) >= 2 and result["ordinary_flags"] <= 1 and not result["cut_flags"]
        replayed = {**result, "advances": advances}
        require(replayed == recorded["arms"][arm], "Recovered range counts or decision differ")
    require(dict(baseline) == recorded["baseline"] and len(rows) == seal["cases"] == authoring["counts"]["byte_identical_mp4s"],
            "The independent decision replay is incomplete")
    result = {"aggregate": recorded, "independent_authoring": authoring, "decisions_replayed": len(rows),
              "recovered_range_source_pictures_verified": pictures, "coarse_input_pictures_verified": coarse_inputs,
              "selected_ranges": selected, "coverage": dict(coverage)}
    with publication.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"decisions_replayed": len(rows), "recovered_range_source_pictures_verified": pictures,
                      "coarse_input_pictures_verified": coarse_inputs}, sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("authoring", "outcomes"))
    cli.add_argument("folder", type=Path)
    cli.add_argument("--archive", type=Path)
    cli.add_argument("--publication", type=Path)
    args = cli.parse_args()
    if args.phase == "authoring":
        reconstruct(args.folder, args.archive or study.dependencies().ARCHIVE)
    else:
        if not args.publication:
            cli.error("outcomes requires --publication")
        outcomes(args.folder, args.publication)


if __name__ == "__main__":
    main()
