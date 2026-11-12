# SPDX-License-Identifier: Apache-2.0
"""Freeze and render a source-separated panel for native background cut review."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from scripts import video_face_scene_continuity_transfer as previous

common, require, old = previous.common, previous.require, previous.old
ARCHIVE = previous.ARCHIVE
EARLIER = previous.prior.DATA / "video-face-scene-continuity-2026-11-12"
N = 120
PAIR_LIMIT = .15
PAIR_COUNT = 8
SPEC = {
    "schema": "chipmunk.video-native-background-panel/1",
    "source_role": "One prospectively locked held roster. Exclude all 49 original/source/donor groups from preceding localization and scene-continuity studies, including both tracked cut discoveries. This excludes documented links and does not imply unknown portrait identity or general project independence.",
    "output": "Exact H.264 MP4, 640x360, yuv420p, 25 fps, 120 pictures, CRF 23, silent. Select the closest authenticated native source picture by presentation time, earlier on tie, independently for each output tick. No interpolation or fractional timing correspondence between unmatched originals and swap streams.",
    "matching": "Read only genuine source pictures for the prior scene study's fixed outer 8x6 grayscale descriptor at output ticks 0,60,118. Maximum-cardinality matching over pairs with a normalized distance at most .15; minimum total distance breaks ties. Keep eight closest resulting pairs, ordered by source keys on distance ties. Skip source overlap and preserve complete assessment of the locked media.",
    "conditions": {
        "cross_cut": "Both directions per pair. Switch all source pictures and return at exact authored edges. Across pair/direction indices alternate the four fixed short intervals. Keep known non-AI construction.",
        "long_swap": "One fixed 18- or 22-picture local publisher-swap insert per held source. Alternate publisher families by direction.",
        "short_swap": "One fixed six- or eleven-picture publisher-swap insert on both sources of the first four locked pairs, with the other publisher family.",
        "ordinary_control": "Each source has either a localized 8 px-per-edge camera punch-in or 1.20 RGB gain, alternating by direction. Include one original per pair. The optical punch-in is a deterministic edit, not physical new acquisition.",
    },
    "cut_intervals": [[54, 60], [67, 75], [54, 64], [65, 77]],
    "long_swap_intervals": [[50, 68], [56, 78]],
    "short_swap_intervals": [[54, 60], [62, 73]],
    "ordinary_interval": [67, 75],
    "app_comparator": "Run the actual Full shot, track and face-appearance lanes with a simulated signed Stripe checkout in an isolated database. Keep normal detector selection and measured face/signed-clock evidence within those lanes; omit unrelated large checks. The paired guard can only suppress an existing selected coarse range, and does not pick another candidate. Keep existing short-pass behavior for the counterfactual.",
    "advancement": {"min_useful_genuine_cut_removals": 2, "min_distinct_source_pairs": 2,
                    "maximum_lost_useful_actual_swap_ranges": 0, "maximum_false_matches_on_swaps_or_controls": 0,
                    "useful_swap_iou": "1/2", "accepted_cut_max_native_endpoint_error_pictures": 1,
                    "min_coarse_useful_swap_assessment": 1},
}


def eligible(archive):
    _, paths = old.bindings(archive)
    p = previous.prior
    paths = [*paths, *(folder / "freeze.json" for folder in p.PRIOR), previous.DISCOVERY / "freeze.json", EARLIER / "freeze.json"]
    exposure = [common.read(path) for path in paths]
    excluded = set.union(*(old.trial.previous_rtf_sources(doc) for doc in exposure),
                         *({row["index"] for row in doc.get("roster", []) if "index" in row} for doc in exposure))
    require(len(excluded) == 49, "Exact prior exposure roster required")
    metadata = old.trial.prior.comparison.metadata()
    used = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in excluded]
    urls, donors = {row["url"] for row in used}, set.union(*(row["donors"] for row in used))
    roster = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        receipt = common.read(path)
        index, meta = receipt["index"], metadata[receipt["group"]]
        if index in excluded or meta["url"] in urls or meta["donors"] & donors or "child" in meta["tags"]:
            continue
        cases = {row["condition"]: row for row in receipt["cases"]}
        table = p.FrameTable.from_json(cases["original"]["frame_table"])
        if table.rotation or table.truncated or table.interpolated or (table.width, table.height) not in old.encoded.PROFILES:
            continue
        times = common.relative_times(table)
        if times[-1] < Fraction(N-1, 25) or len(times) < N:
            continue
        if any(common.relative_times(p.FrameTable.from_json(cases[condition]["frame_table"])) != times for condition in ("swap_a", "swap_b")):
            continue
        selected = [min(range(len(times)), key=lambda i: (abs(times[i]-Fraction(k,25)), i)) for k in range(N)]
        if len(set(selected)) != N or max(abs(times[i]-Fraction(k,25)) for k,i in enumerate(selected)) > Fraction(1, 40):
            continue
        roster.append({**old.encoded.prior.archive_io.source_pin(index, archive), "shape": [table.width, table.height],
                       "native_step": str(times[1]-times[0]), "source_frame_indices": selected,
                       "archive_role": receipt["role"], "motion_tagged": "head rotation" in meta["tags"]})
        urls.add(meta["url"])
        donors.update(meta["donors"])
    return roster, paths


def source_pictures(archive, pin, condition, output_indices, *, verify_file=True):
    shape, cases, root = previous.prior.durations.source_set(pin, archive)
    case = cases[condition]
    path = root / case["file"]
    if verify_file:
        require(common.digest(path) == case["sha256"], "Source bytes differ from the archive receipt")
    table = common.probe_frames(path, timeout_s=25)
    require(table.to_json() == case["frame_table"] and shape == (table.width, table.height), "Source clock differs from archive receipt")
    wanted = {pin["source_frame_indices"][k]: k for k in output_indices}
    values = {}
    for frame in common.decode_frames(path, table, sorted(wanted), timeout_s=90):
        require(frame.pts_verified and frame.pts == table.pts[frame.index]
                and hashlib.sha256(frame.rgb.tobytes()).hexdigest() == case["expected_rgb_sha256"][frame.index],
                "Unverified source picture, RGB, or native clock")
        values[wanted[frame.index]] = cv2.resize(frame.rgb, old.DIMENSIONS, interpolation=cv2.INTER_AREA)
    require(list(values) == list(output_indices), "Incomplete native source-picture roster")
    return values


def intake(folder, archive):
    roster, paths = eligible(archive)
    require(len(roster) >= PAIR_COUNT*2, "Insufficient mutually source-disjoint media")
    previews = {}
    for pin in roster:
        idx = pin["index"]
        previews[idx] = previous.background(source_pictures(archive, pin, "original", (0, 60, 118), verify_file=False))
        print("source-only match read", len(previews), "of", len(roster), flush=True)
    graph = nx.Graph()
    graph.add_nodes_from(previews)
    eligible_edges = []
    for i, a in enumerate(previews):
        for b in list(previews)[i+1:]:
            distance = float(np.mean(np.abs(previews[a]-previews[b])))
            if distance <= PAIR_LIMIT:
                eligible_edges.append((distance, a, b))
    for k, (distance, a, b) in enumerate(sorted(eligible_edges)):
        graph.add_edge(a, b, weight=round(distance*10**9)*10000+k, distance=distance)
    pairs = sorted((graph[a][b]["distance"], *sorted((a,b))) for a,b in nx.min_weight_matching(graph))
    require(len(pairs) >= PAIR_COUNT, f"Only {len(pairs)} source-disjoint close pairs qualify")
    chosen = {idx for _,a,b in pairs[:PAIR_COUNT] for idx in (a,b)}
    folder.mkdir(mode=0o700)
    lock = {"spec": SPEC, "available": len(roster), "eligible_close_pairs": len(pairs),
            "source_roster": [row for row in roster if row["index"] in chosen], "pairs": pairs[:PAIR_COUNT],
            "eligible_pool_sha256": hashlib.sha256(repr(roster).encode()).hexdigest(),
            "previous_freeze_sha256": [common.digest(path) for path in paths],
            "implementation_sha256": common.digest(Path(__file__)),
            "decoder_and_encoder_bindings": old.bindings(archive)[0]}
    common.write(folder / "freeze.json", lock)
    print("frozen", len(chosen), "sources", len(pairs), "close pairs available", flush=True)
    print("distances", [round(d,4) for d,_,_ in pairs[:PAIR_COUNT]], flush=True)


def frozen(folder, archive):
    lock = common.read(folder / "freeze.json")
    pool, paths = eligible(archive)
    require(lock["spec"] == SPEC and lock["implementation_sha256"] == common.digest(Path(__file__))
            and lock["decoder_and_encoder_bindings"] == old.bindings(archive)[0]
            and lock["eligible_pool_sha256"] == hashlib.sha256(repr(pool).encode()).hexdigest()
            and lock["previous_freeze_sha256"] == [common.digest(path) for path in paths], "Frozen source pool or implementation differs")
    return lock


def recipe(pair_number, direction):
    cases = [("cross_cut", SPEC["cut_intervals"][(pair_number+direction*2)%4], None),
             ("long_swap", SPEC["long_swap_intervals"][direction], "swap_a" if direction == 0 else "swap_b"),
             ("camera_punch" if direction == 0 else "light", SPEC["ordinary_interval"], None)]
    if pair_number < PAIR_COUNT//2:
        cases.append(("short_swap", SPEC["short_swap_intervals"][direction], "swap_b" if direction == 0 else "swap_a"))
    if direction == 0:
        cases.append(("original", None, None))
    return cases


def encode_case(path, plain, condition, span, altered, donor):
    if span is None:
        old.encoded.encode(path, frames=plain.values(), crf=23)
        return
    a,b = span
    if condition == "cross_cut":
        replacement = {i: donor[i] for i in range(a,b)}
    elif condition.endswith("swap"):
        replacement = altered
    elif condition == "light":
        replacement = {i: np.clip(np.rint(plain[i].astype(np.float32)*1.20), 0, 255).astype(np.uint8) for i in range(a,b)}
    else:
        require(condition == "camera_punch", "Unexpected test condition")
        replacement = {i: cv2.resize(plain[i][8:-8, 8:-8], old.DIMENSIONS, interpolation=cv2.INTER_LINEAR) for i in range(a,b)}
    require(list(replacement) == list(range(a,b)) and all(not np.array_equal(plain[i], rgb) for i,rgb in replacement.items()),
            "The authored intervention must change every interval picture")
    old.encoded.encode(path, frames=(replacement.get(i,rgb) for i,rgb in plain.items()), crf=23)


def prepare(folder, archive):
    lock = frozen(folder, archive)
    directory = folder / "media"
    directory.mkdir(mode=0o700)
    roster = {pin["index"]: pin for pin in lock["source_roster"]}
    cases = []
    for pair, (_, a, b) in enumerate(lock["pairs"]):
        originals = {i: source_pictures(archive, roster[i], "original", range(N)) for i in (a,b)}
        for direction, (idx, other) in enumerate(((a,b), (b,a))):
            for condition, span, swap in recipe(pair, direction):
                altered = source_pictures(archive, roster[idx], swap, range(*span)) if swap else None
                name = f"pair-{pair:02d}-{direction}-{condition}.mp4"
                path = directory / name
                encode_case(path, originals[idx], condition, span, altered, originals[other])
                truth = span if swap else None
                row = old.describe(path, pair*2+direction, "held", condition, truth)
                cases.append({**row, "pair":pair,"direction":direction,"interval":span})
            print("prepared held source", pair, direction, flush=True)
    require(len(cases) == len({row["sha256"] for row in cases}) == 64, "The exact frozen 64-case panel is required")
    common.write(folder / "media.json", {"freeze_sha256": common.digest(folder / "freeze.json"), "media": cases,
                 "conditions":dict(Counter(row["condition"] for row in cases))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        intake(args.folder, args.archive)
    else:
        prepare(args.folder, args.archive)


if __name__ == "__main__":
    main()
