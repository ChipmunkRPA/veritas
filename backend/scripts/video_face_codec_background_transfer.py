# SPDX-License-Identifier: Apache-2.0
"""Test the frozen background quartile guard on new VP9 and HEVC face-review deliveries."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from scripts import diagnose_video_face_six_codec_cuts as diagnostic
from scripts import video_face_scene_continuity_transfer as scene_study
from scripts import video_face_six_codec_transfer as codec_study
from scripts import video_native_background_challenge as matching

common, require = codec_study.common, codec_study.require
ARCHIVE = codec_study.ARCHIVE
DEVELOPMENT = scene_study.DISCOVERY
PRIOR = codec_study.DATA / "video-native-background-challenge-2026-11-12" / "freeze.json"
FRAME_COUNT = codec_study.original.N
SPANS = codec_study.original.SPANS
POINT = scene_study.POINTS["lower_quartile"]
CASES = ("original", "recompressed", "light", "swap_a", "swap_b", "cut_a", "cut_b")
PROTOCOL = {
    "schema": "chipmunk.video-face-codec-background-transfer/1",
    "codecs": codec_study.CODECS,
    "frame_count": FRAME_COUNT,
    "spans": list(map(list, SPANS)),
    "source_exclusions": "Earlier 77 source groups, known source links and donor keys from all short review and scene challenges.",
    "matching": "Use only the genuine source previews and fixed palette/outer gray distance from the prior scene challenge. Keep the three closest mutually disjoint pairs from the minimum-weight maximum-cardinality matching. No detector scores enter selection.",
    "controls": "Unchanged, tandem encoding, local light and two brief genuine whole-picture cuts on both directions of each pair.",
    "guard": "Retain all incumbent ranges. If the unchanged rank fallback selected a distinct range, require the weaker of its two face-masked background correlation quartiles to be at least 0.60. Suppressing it cannot select a different candidate.",
    "guard_point": POINT,
    "gate": {"minimum_additions": 4, "minimum_groups": 3, "maximum_new_cut_ranges": 0,
             "maximum_new_ordinary_ranges": 1, "minimum_fallback_coverage": .9,
             "maximum_lost_correct_dense": 1, "minimum_exposed_cut_removals": 2},
}


def cases(archive):
    roster, histories = matching.pool(archive)
    old = common.read(PRIOR)
    excluded = {row["index"] for row in old["roster"]}
    require(len(excluded) == 12, "Expected all source keys from the previous challenge")
    return [row for row in roster if row["index"] not in excluded], [*histories, PRIOR]


def bindings(archive):
    pool, histories = cases(archive)
    prior_audit = common.read(DEVELOPMENT / "independent-audit.json")
    require(prior_audit["status"] == "verified"
            and prior_audit["aggregate_sha256"] == common.digest(DEVELOPMENT / "aggregate.json"),
            "The exposed codec development panel must have its exact independent audit")
    return pool, {"previous_freezes": [common.digest(path) for path in histories],
                  "development_audit": common.digest(DEVELOPMENT / "independent-audit.json"),
                  "tool_bindings": common.bindings(),
                  "method": common.digest(Path(__file__)),
                  "incumbent": codec_study.original.bindings(archive)[0],
                  "boundary_replay": common.digest(Path(diagnostic.__file__)),
                  "frozen_point_source": common.digest(Path(scene_study.__file__))}


def freeze(run, archive):
    pool, pins = bindings(archive)
    require(len(pool) >= 6, "Need at least six fresh eligible source groups")
    previews = {row["index"]: matching.preview_stats(matching.earlier.source_pictures(
        archive, row, "original", matching.PROBES, verify_file=False)) for row in pool}
    graph = nx.Graph()
    edges = sorted((matching.compare(previews[a], previews[b])[0], a, b)
                   for a in previews for b in previews if a < b)
    for order, (distance, a, b) in enumerate(edges):
        graph.add_edge(a, b, weight=round(distance * 10**9) * 10_000 + order, distance=distance)
    pairs = sorted((graph[a][b]["distance"], *sorted((a, b)))
                   for a, b in nx.min_weight_matching(graph))[:3]
    chosen = {index for _, a, b in pairs for index in (a, b)}
    require(len(chosen) == 6, "Six distinct sources required")
    run.mkdir(mode=0o700)
    common.write(run / "freeze.json", {"protocol": PROTOCOL, "bindings": pins, "available": len(pool),
                 "sources": [row for row in pool if row["index"] in chosen], "pairs": pairs})
    print("Frozen six of", len(pool), "unused sources", flush=True)


def frozen(run, archive):
    lock = common.read(run / "freeze.json")
    pool, expected = bindings(archive)
    source_pins = {row["index"]: row for row in pool}
    require(lock["protocol"] == PROTOCOL and lock["bindings"] == expected
            and all(source_pins.get(row["index"]) == row for row in lock["sources"]),
            "Source or implementation freeze differs")
    return lock


def prepare(run, archive):
    lock = frozen(run, archive)
    source_pins = {row["index"]: row for row in lock["sources"]}
    out = run / "media"
    out.mkdir(mode=0o700)
    rows = []
    for pair, (_, a, b) in enumerate(lock["pairs"]):
        originals = {idx: matching.earlier.source_pictures(
            archive, source_pins[idx], "original", range(FRAME_COUNT)) for idx in (a, b)}
        for side, (host, guest) in enumerate(((a, b), (b, a))):
            group = 2 * pair + side
            base = originals[host]
            edits = {condition: matching.earlier.source_pictures(
                archive, source_pins[host], condition, range(*span))
                for condition, span in zip(("swap_a", "swap_b"), SPANS, strict=True)}
            for cut, span in zip(("cut_a", "cut_b"), SPANS, strict=True):
                edits[cut] = {i: originals[guest][i] for i in range(*span)}
            edits["light"] = {i: np.clip(np.rint(base[i].astype(np.float32) * 1.20), 0, 255).astype(np.uint8)
                              for i in range(*SPANS[1])}
            require(all(not np.array_equal(base[i], altered) for edit in edits.values() for i, altered in edit.items()),
                    "An authored interval contains unchanged pictures")
            for codec, settings in codec_study.CODECS.items():
                original_path = codec_study.filename(out, group, codec, "original")
                for condition in CASES:
                    path = codec_study.filename(out, group, codec, condition)
                    truth = None
                    if condition == "recompressed":
                        codec_study.encode(path, codec, settings["tandem"], source=original_path)
                    else:
                        altered = edits.get(condition, {})
                        codec_study.encode(path, codec, settings["primary"],
                                           pictures=(altered.get(i, rgb) for i, rgb in base.items()))
                        if condition.startswith("swap_"):
                            truth = list(SPANS[condition[-1] == "b"])
                    row = codec_study.describe(path, group, "held", codec, condition, truth)
                    row["authored_range"] = list(SPANS[condition[-1] == "b"]) if condition.startswith(("swap_", "cut_")) else None
                    rows.append(row)
                print("prepared source", group, codec, flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == 84, "Require all 84 distinct output files")
    common.write(run / "media.json", {"freeze_sha256": common.digest(run / "freeze.json"), "rows": rows})


def measured(path, row, models, prior=None):
    require(common.digest(path) == row["sha256"], "Encoded source bytes differ")
    if prior is None:
        arms, _, _ = codec_study.original.compare(path, row, models)
        first = arms["eight_hz"]
        fallback = arms["dense_rank"] if first and not first["selected"] else None
        arms = {key: arms[key] for key in ("coarse", "eight_hz")} | {"fallback": fallback}
    else:
        arms = {key: prior[key] for key in ("coarse", "eight_hz", "fallback")}
    evidence = None
    fallback = arms["fallback"]
    if fallback and fallback["selected"]:
        _, quartiles, frames = diagnostic.boundary_profile(path, row, fallback, models[0])
        evidence = {"quartiles": quartiles, "frames": frames,
                    "frame_sha256": [row["rgb_sha256"][i] for i in frames], "retained": min(quartiles) >= POINT}
    return {k: row[k] for k in ("file", "group", "codec", "condition", "sha256", "truth")} | arms | {"guard": evidence}


def score(run, archive):
    frozen(run, archive)
    manifest = common.read(run / "media.json")
    require(manifest["freeze_sha256"] == common.digest(run / "freeze.json"), "Authored delivery manifest differs")
    manifest_rows = common.read(DEVELOPMENT / "media.json")["media"]
    development = {row["file"]: row for row in manifest_rows}
    models = common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    for phase, source, records in (
        ("exposed", DEVELOPMENT / "media", ((development[row["file"]], row) for row in codec_study.journal(DEVELOPMENT, "held"))),
        ("held", run / "media", ((row, None) for row in manifest["rows"])),
    ):
        started = time.monotonic()
        with (run / (phase + ".jsonl")).open("x") as log:
            for row, prior in records:
                outcome = measured(source / row["file"], row, models, prior)
                log.write(json.dumps(outcome, allow_nan=False) + "\n")
                log.flush()
                print(phase, row["group"], row["codec"], row["condition"], flush=True)
        common.write(run / (phase + "-seal.json"), {"seconds": time.monotonic() - started,
                      "journal_sha256": common.digest(run / (phase + ".jsonl")),
                      "manifest_sha256": common.digest(run / "media.json")})


def aggregate(rows):
    output = {}
    for codec in codec_study.CODECS:
        counters = {kind: Counter() for kind in ("swap", "cut", "ordinary")}
        groups = set()
        for row in rows:
            if row["codec"] != codec:
                continue
            truth = row["truth"]
            condition = row["condition"]
            kind = "swap" if truth else "cut" if condition.startswith("cut_") or condition == "hard_cut" else "ordinary"
            selected = [row[key]["frames"] for key in ("coarse", "eight_hz") if row[key] and row[key]["selected"]]
            found = row["fallback"]
            supported = bool(row["guard"])
            retained = supported and row["guard"]["retained"]
            located = truth and any(codec_study.original.iou(span, truth) >= Fraction(1, 2) for span in selected)
            dense_correct = truth and supported and codec_study.original.iou(found["frames"], truth) >= Fraction(1, 2)
            if dense_correct and retained and not located:
                groups.add(row["group"])
            counters[kind].update({"cases": 1, "baseline_flags": bool(selected), "fallback_measured": found is not None,
                                   "dense_flags": supported, "guarded_dense_flags": bool(retained),
                                   "baseline_located": bool(located), "unfiltered_located": bool(located or dense_correct),
                                   "guarded_located": bool(located or (retained and dense_correct))})
        output[codec] = {"counts": {k: dict(v) for k, v in counters.items()}, "added_held_groups": len(groups)}
    return output


def report(run, archive):
    frozen(run, archive)
    recorded = {}
    for phase, count in (("exposed", 64), ("held", 84)):
        log = run / (phase + ".jsonl")
        seal = common.read(run / (phase + "-seal.json"))
        require(seal["journal_sha256"] == common.digest(log)
                and seal["manifest_sha256"] == common.digest(run / "media.json"), "Scoring seal differs")
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        require(len(rows) == count and len({row["sha256"] for row in rows}) == count, "Incomplete study results")
        recorded[phase] = rows
    hold = aggregate(recorded["held"])
    prior = aggregate(recorded["exposed"])
    for codec, stats in hold.items():
        swap, cut, ordinary = (stats["counts"][group] for group in ("swap", "cut", "ordinary"))
        development_cut = prior[codec]["counts"]["cut"]
        targets = PROTOCOL["gate"]
        checks = {
            "swap_gain": swap["guarded_located"] - swap["baseline_located"] >= targets["minimum_additions"],
            "source_groups": stats["added_held_groups"] >= targets["minimum_groups"],
            "dense_swap_loss": swap["unfiltered_located"] - swap["guarded_located"] <= targets["maximum_lost_correct_dense"],
            "ordinary": ordinary["guarded_dense_flags"] <= targets["maximum_new_ordinary_ranges"],
            "cut": cut["guarded_dense_flags"] <= targets["maximum_new_cut_ranges"],
            "coverage": all(d["fallback_measured"] >= targets["minimum_fallback_coverage"] * d["cases"] for d in (swap, ordinary)),
            "exposed_cut_correction": development_cut["dense_flags"] - development_cut["guarded_dense_flags"]
                                      >= targets["minimum_exposed_cut_removals"],
        }
        stats.update(criteria=checks, advances=all(checks.values()))
    common.write(run / "aggregate.json", {"protocol": PROTOCOL, "held": hold, "exposed": prior,
                 "held_seconds": common.read(run / "held-seal.json")["seconds"]})
    print(json.dumps(hold, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    {"freeze": freeze, "prepare": prepare, "score": score, "report": report}[args.phase](args.run, args.archive)


if __name__ == "__main__":
    main()
