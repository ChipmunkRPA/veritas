# SPDX-License-Identifier: Apache-2.0
"""Prospective background-only suppression for the dense short face range."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import face_return
from scripts import diagnose_video_face_six_codec_cuts as diagnostic
from scripts import video_face_six_codec_transfer as prior

common, require, old = prior.common, prior.require, prior.original
ARCHIVE = prior.ARCHIVE
DISCOVERY = prior.DATA / "video-face-six-codec-2026-11-12"
POINTS = {"median": .83, "lower_quartile": .60}
N = old.N
CROSS_SPANS = (old.SPANS[0], (67, 75))
CONTROL_SPAN = old.SPANS[1]
CASES = ("original", "recompressed", "swap_a", "swap_b", "light", "camera_punch", "jump_cut", "cross_cut_a", "cross_cut_b")
SPEC = {
    "schema": "chipmunk.video-face-scene-continuity/1", "method": "After incumbent and fallback range selection, suppress only an existing dense range when its weaker entry or exit falls below the candidate point. Keep all incumbent ranges and never select a different winner.",
    "points": POINTS, "incumbent_scene_point": face_return.SCENE_CORRELATION_POINT, "rank_point": face_return.POINT,
    "scope": "120 exact 640x360 H.264/yuv420p frames at 25 fps, sourced from even decoded pictures on a verified 50 fps clock",
    "spans": {"publisher_swaps": list(map(list, old.SPANS)), "cross_cuts": list(map(list, CROSS_SPANS)), "same_recording_controls": list(CONTROL_SPAN)},
    "controls": {"crop_each_edge_px": 8, "light_gain": 1.20, "time_jump_output_pictures": 22, "crfs": [23, 36]},
    "role_rule": "Two prior selection-role sources for development, and six sources outside development; prioritize prior confirmation roles, then include prior fusion or confidence roles. Fixed SHA order within each priority. Exclude all known prior localization source URLs and donor keys.",
    "pairing": "Use only genuine source pictures 0, 60, and 118 in 640x360 output numbering. Resize to 8x6 gray and omit center 4x4 tiles. Pair minimum total mean absolute outer intensity difference within each role. A mean of at most .15 qualifies as similar by this proxy. Both directions and both spans are included. Same-source jumps provide independent same-background controls.",
    "advancement": {"zero_useful_swap_losses": True, "minimum_useful_swaps": 8, "minimum_swap_sources": 4,
                    "minimum_false_cut_suppression": 2, "minimum_false_cut_pair_sources": 2, "minimum_false_cut_fraction": .5},
}


def selection(archive):
    bindings, paths = old.bindings(archive)
    paths = [*paths, *(folder / "freeze.json" for folder in prior.PRIOR), DISCOVERY / "freeze.json"]
    discovery = common.read(DISCOVERY / "independent-audit.json")
    require(discovery["status"] == "verified" and discovery["aggregate_sha256"] == common.digest(DISCOVERY / "aggregate.json"),
            "Independent discovery audit required")
    archives = [common.read(path) for path in paths]
    excluded = set.union(*(old.trial.previous_rtf_sources(doc) for doc in archives),
                         *({row["index"] for row in doc.get("roster", []) if "index" in row} for doc in archives))
    require(len(excluded) == 41, "All 41 prior short-localization source groups required")
    metadata = old.trial.prior.comparison.metadata()
    used = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in excluded]
    urls, donors = {row["url"] for row in used}, set.union(*(row["donors"] for row in used))
    eligible = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        item = common.read(path)
        i, meta = item["index"], metadata[item["group"]]
        if i in excluded or meta["url"] in urls or donors & meta["donors"] or "child" in meta["tags"]:
            continue
        base = next(row for row in item["cases"] if row["condition"] == "original")
        table = prior.FrameTable.from_json(base["frame_table"])
        shape = table.width, table.height
        if (shape not in old.encoded.PROFILES or table.rotation or table.truncated or table.interpolated
                or common.relative_times(table) != [Fraction(j, 50) for j in range(300)]):
            continue
        order = hashlib.sha256(f"scene-continuity/1/{item['group']}".encode()).hexdigest()
        eligible.append((item["role"] != "confirmation", order, i, item["role"], shape, meta))
    roster = []
    for role, count in (("development", 2), ("held", 6)):
        for _, _, i, source_role, shape, meta in sorted(eligible):
            if ((source_role == "selection") != (role == "development") or meta["url"] in urls or donors & meta["donors"]):
                continue
            roster.append({**old.encoded.prior.archive_io.source_pin(i, archive), "role": role,
                           "shape": list(shape), "prior_archive_role": source_role, "motion_tagged": "head rotation" in meta["tags"]})
            urls.add(meta["url"])
            donors.update(meta["donors"])
            if sum(row["role"] == role for row in roster) == count:
                break
        require(sum(row["role"] == role for row in roster) == count, f"Need {count} eligible {role} source groups")
    return {"spec": SPEC, "roster": roster, "eligible_groups": len(eligible), "excluded_prior_groups": len(excluded),
            "archive_bindings": bindings, "discovery_audit_sha256": common.digest(DISCOVERY / "independent-audit.json"),
            "prior_roster_sha256": [common.digest(path) for path in paths], "worker_sha256": common.digest(Path(__file__)),
            "prior_diagnostic_sha256": common.digest(Path(diagnostic.__file__))}


def frozen(folder, archive):
    locked = common.read(folder / "freeze.json")
    require(locked == selection(archive), "Frozen rules or inputs changed")
    return locked


def source_frames(root, receipt, shape, indices):
    path = root / receipt["file"]
    require(common.digest(path) == receipt["sha256"], "Exact source bytes required")
    table = common.probe_frames(path, timeout_s=25)
    require(table.to_json() == receipt["frame_table"] and shape == (table.width, table.height)
            and common.relative_times(table) == [Fraction(i, 50) for i in range(300)], "Archived 50 fps source clock differs")
    native = {}
    for frame in common.decode_frames(path, table, [2*i for i in indices], timeout_s=65):
        require(frame.pts_verified and frame.pts == table.pts[frame.index] and frame.index % 2 == 0
                and hashlib.sha256(frame.rgb.tobytes()).hexdigest() == receipt["expected_rgb_sha256"][frame.index],
                "Archived native picture or PTS differs")
        native[frame.index // 2] = cv2.resize(frame.rgb, old.DIMENSIONS, interpolation=cv2.INTER_AREA)
    require(list(native) == list(indices), "Exact even-picture source set required")
    return native


def source_group(archive, pin, conditions):
    shape, cases, root = prior.durations.source_set(pin, archive)
    return {condition: source_frames(root, cases[condition], shape, indices) for condition, indices in conditions.items()}


def background(frames):
    mask = np.ones((6, 8), bool)
    mask[1:5, 2:6] = False
    tiles = [cv2.cvtColor(cv2.resize(rgb, (8, 6), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)[mask]
             for rgb in frames.values()]
    return np.stack(tiles).astype(np.float64) / 255


def all_pairings(groups):
    if not groups:
        yield ()
    else:
        first, *rest = groups
        for other in rest:
            for suffix in all_pairings([i for i in rest if i != other]):
                yield ((first, other), *suffix)


def pairs(lock, archive):
    previews = {i: background(source_group(archive, row, {"original": (0, 60, 118)})["original"])
                for i, row in enumerate(lock["roster"])}

    def distance(a, b):
        return float(np.mean(np.abs(previews[a] - previews[b])))

    out = []
    for role in ("development", "held"):
        groups = [i for i, row in enumerate(lock["roster"]) if row["role"] == role]
        chosen = min(all_pairings(groups), key=lambda match: (sum(distance(a, b) for a, b in match), match))
        out.extend({"groups": [a, b], "role": role, "distance": distance(a, b), "similar": distance(a, b) <= .15}
                   for a, b in chosen)
    return out


def swap_span(condition, group):
    return old.SPANS[(group + (condition == "swap_b")) % 2]


def conditions(group):
    return {"original": range(N), **{case: range(*swap_span(case, group)) for case in ("swap_a", "swap_b")}}


def case_span(condition, group):
    if condition.startswith("swap"):
        return swap_span(condition, group)
    if condition.startswith("cross_cut"):
        return CROSS_SPANS[int(condition.endswith("_b"))]
    if condition in ("light", "camera_punch", "jump_cut"):
        return CONTROL_SPAN
    return None


def pictures(plain, changed, donor, condition, interval):
    if interval is None:
        yield from plain.values()
        return
    a, b = interval
    for i, rgb in plain.items():
        if a <= i < b:
            if condition.startswith("swap"):
                replacement = changed[i]
            elif condition.startswith("cross_cut"):
                replacement = donor[i]
            elif condition == "jump_cut":
                replacement = plain[i+22]
            elif condition == "light":
                replacement = np.clip(np.rint(rgb.astype(np.float32)*1.20), 0, 255).astype(np.uint8)
            else:
                require(condition == "camera_punch", "Unexpected edit condition")
                replacement = cv2.resize(rgb[8:-8, 8:-8], old.DIMENSIONS, interpolation=cv2.INTER_LINEAR)
            require(not np.array_equal(rgb, replacement), "Authored edit did not change its picture")
            rgb = replacement
        yield rgb


def prepare(folder, archive):
    lock = frozen(folder, archive)
    pair_rows = pairs(lock, archive)
    common.write(folder / "source-pairs.json", {"pairs": pair_rows, "freeze_sha256": common.digest(folder / "freeze.json")})
    output = folder / "media"
    output.mkdir(mode=0o700)
    rows = []
    for pair in pair_rows:
        a, b = pair["groups"]
        sources = {i: source_group(archive, lock["roster"][i], conditions(i)) for i in (a,b)}
        for group, other in ((a,b), (b,a)):
            plain, donor = sources[group]["original"], sources[other]["original"]
            original = output / f"g{group}-original.mp4"
            for case in CASES:
                path = output / f"g{group}-{case}.mp4"
                interval = case_span(case, group)
                if case == "recompressed":
                    old.encoded.encode(path, source=original, crf=36)
                else:
                    frames = pictures(plain, sources[group].get(case), donor, case, interval)
                    old.encoded.encode(path, frames=frames, crf=23)
                row = old.describe(path, group, pair["role"], case, interval if case.startswith("swap") else None)
                rows.append({**row, "codec": "h264", "authored_interval": interval,
                             "partner": other if case.startswith("cross_cut") else None,
                             "similar_background": pair["similar"] if case.startswith("cross_cut") else case == "jump_cut"})
            print("prepared", pair["role"], group, flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == len(lock["roster"])*len(CASES), "Complete distinct media required")
    common.write(folder / "media.json", {"media": rows, "pairs_sha256": common.digest(folder / "source-pairs.json"),
                 "freeze_sha256": common.digest(folder / "freeze.json")})


def journal(folder, role):
    path = folder / f"{role}.jsonl"
    seal = common.read(folder / f"{role}-seal.json")
    require(seal["journal_sha256"] == common.digest(path) and seal["media_sha256"] == common.digest(folder / "media.json"),
            "Sealed measurement journal differs")
    return [json.loads(line) for line in path.read_text().splitlines()]


def score(folder, archive, role):
    lock = frozen(folder, archive)
    prepared = common.read(folder / "media.json")
    require(prepared["freeze_sha256"] == common.digest(folder / "freeze.json")
            and prepared["pairs_sha256"] == common.digest(folder / "source-pairs.json"), "Source preparation differs")
    if role == "held":
        journal(folder, "development")
    models = common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    started = time.monotonic()
    count = 0
    path = folder / f"{role}.jsonl"
    with path.open("x") as stream:
        for row in prepared["media"]:
            if row["split"] != role:
                continue
            media = folder / "media" / row["file"]
            arms, decoded, supported = old.compare(media, row, models)
            fallback = arms["dense_rank"] if arms["eight_hz"] and not arms["eight_hz"]["selected"] else None
            edges = None
            if fallback and fallback["selected"]:
                median, quartile, indices = diagnostic.boundary_profile(media, row, fallback, models[0])
                edges = {"median": median, "lower_quartile": quartile, "indices": indices}
            stream.write(json.dumps({**{k: row[k] for k in ("file", "group", "condition", "truth", "partner", "similar_background", "sha256")},
                         "coarse": arms["coarse"], "eight_hz": arms["eight_hz"], "fallback": fallback,
                         "edges": edges, "decoded": decoded, "supported": supported}, allow_nan=False) + "\n")
            stream.flush()
            count += 1
            print("scored", role, row["group"], row["condition"], flush=True)
    require(count == sum(row["role"] == role for row in lock["roster"])*len(CASES), "Complete role measurement required")
    common.write(folder / f"{role}-seal.json", {"journal_sha256": common.digest(path),
                 "media_sha256": common.digest(folder / "media.json"), "elapsed_s": time.monotonic()-started,
                 "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def category(row):
    if row["condition"].startswith("cross_cut"):
        return "similar_cross_cut" if row["similar_background"] else "distinct_cross_cut"
    return "swap" if row["truth"] else row["condition"]


def quality(ranges, truth):
    return max((old.iou(window, truth) for window in ranges), default=Fraction(0))


def summarize(rows):
    table = {arm: defaultdict(Counter) for arm in ("current", *POINTS)}
    suppressed_groups = {arm: set() for arm in POINTS}
    suppressed_pairs = {arm: set() for arm in POINTS}
    useful_sources = {arm: set() for arm in ("current", *POINTS)}
    for row in rows:
        kind = category(row)
        incumbent = [arm["frames"] for arm in (row["coarse"], row["eight_hz"]) if arm and arm["selected"]]
        fallback = [row["fallback"]["frames"]] if row["edges"] else []
        current = incumbent + fallback
        for arm, result in table.items():
            suppress = bool(arm in POINTS and fallback and min(row["edges"][arm]) < POINTS[arm])
            ranges = incumbent if suppress else current
            result[kind]["cases"] += 1
            result[kind]["measured_dense"] += row["fallback"] is not None
            result[kind]["dense_ranges"] += bool(fallback) and not suppress
            result[kind]["flagged"] += bool(ranges)
            if row["truth"]:
                baseline = quality(current, row["truth"])
                observed = quality(ranges, row["truth"])
                result[kind]["correct"] += observed >= Fraction(1, 2)
                result[kind]["useful_losses"] += baseline >= Fraction(1, 2) and observed < Fraction(1, 2)
                result[kind]["exact_range_losses"] += any(window == row["truth"] for window in current if window not in ranges)
                if observed >= Fraction(1, 2):
                    useful_sources[arm].add(row["group"])
            elif suppress and bool(current) and not ranges:
                result[kind]["false_case_removals"] += 1
                suppressed_groups[arm].add(row["group"])
                if row["partner"] is not None:
                    suppressed_pairs[arm].add(tuple(sorted((row["group"], row["partner"]))))
    outputs = {}
    cut = ("similar_cross_cut", "distinct_cross_cut", "jump_cut")
    for arm, result in table.items():
        current_cut_flags = sum(table["current"][kind]["flagged"] for kind in cut)
        removed = current_cut_flags - sum(result[kind]["flagged"] for kind in cut)
        gates = {"zero_useful_swap_losses": result["swap"]["useful_losses"] == 0,
                 "minimum_useful_swaps": result["swap"]["correct"] >= 8, "minimum_swap_sources": len(useful_sources[arm]) >= 4,
                 "minimum_false_cut_suppression": removed >= 2,
                 "minimum_false_cut_pair_sources": len(suppressed_pairs.get(arm, ())) >= 2,
                 "minimum_false_cut_fraction": removed*2 >= max(1, current_cut_flags)}
        outputs[arm] = {"counts": {kind: dict(values) for kind, values in result.items() if values},
                        "correct_swap_sources": len(useful_sources[arm]), "cut_case_removals": removed,
                        "removed_false_source_groups": len(suppressed_groups.get(arm, ())),
                        "removed_false_cross_cut_pairs": len(suppressed_pairs.get(arm, ())),
                        "gate": gates, "advances": arm in POINTS and all(gates.values())}
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "development", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        lock = selection(args.archive)
        args.folder.mkdir(mode=0o700)
        common.write(args.folder / "freeze.json", lock)
    elif args.phase == "prepare":
        prepare(args.folder, args.archive)
    elif args.phase in ("development", "held"):
        score(args.folder, args.archive, args.phase)
    else:
        lock = frozen(args.folder, args.archive)
        summary = {role: summarize(journal(args.folder, role)) for role in ("development", "held")}
        pair_rows = common.read(args.folder / "source-pairs.json")["pairs"]
        summary.update(protocol=lock["spec"], freeze_sha256=common.digest(args.folder / "freeze.json"),
                       held_similar_pairs=sum(row["role"] == "held" and row["similar"] for row in pair_rows),
                       held_scene_pair_distances=sorted(row["distance"] for row in pair_rows if row["role"] == "held"),
                       eligible_source_groups=lock["eligible_groups"], excluded_prior_source_groups=lock["excluded_prior_groups"],
                       runtime=common.read(args.folder / "held-seal.json"), serving_change=False)
        common.write(args.folder / "aggregate.json", summary)


if __name__ == "__main__":
    main()
