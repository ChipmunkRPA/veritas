# SPDX-License-Identifier: Apache-2.0
"""Recheck the H.264 earlier-copy trial against native clocks and source crops."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_reference_h264_earlier_transfer as study
from scripts.video_common_sources_v89 import read, require, sha, write


def clocks(table):
    return [(tick - table.pts[0]) * table.tb for tick in table.pts]


def area(left, right):
    a, b = left
    c, d = right
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def read_pictures(path, table, wanted):
    found = {}
    for picture in decode_frames(path, table, sorted(wanted), timeout_s=15, max_pixels=study.previous.MAX_PIXELS):
        require(picture.index not in found and picture.pts_verified and picture.pts == table.pts[picture.index],
                "Evidence has a missing or unverified timestamp")
        found[picture.index] = picture.rgb
    require(found.keys() == set(wanted), "Native picture set differs")
    return found


def measure_spatial(earlier, earlier_table, source, source_table, tile, peak):
    a = read_pictures(earlier, earlier_table, (peak,))[peak]
    b = read_pictures(source, source_table, (peak,))[peak]
    x, y, w, h = tile
    changed = np.any(a != b, axis=-1)
    return [int(np.count_nonzero(changed[y:y+h, x:x+w])), int(np.count_nonzero(changed)), w*h]


def approved(row, native):
    known, best, spatial = row["known_edit"], row["best"], row["spatial"]
    flagged = best is not None and best["score"] > study.local.POINT
    assessed = row["status"] in {"local_difference", "no_local_range", "measured"}
    exact = any(area(r, known) >= Fraction(1, 2) for r in row["exact"]) if known else False
    time = bool(known and flagged and area(best["frames"], known) >= Fraction(1, 2))
    spatial_correct = bool(flagged and spatial and spatial[0] * 8 >= spatial[2] and spatial[0] * 12 >= spatial[1])
    if native is not None and known and flagged:
        observed_time = area([native[i] for i in best["frames"]], [native[i] for i in known])
        require(time == (observed_time >= Fraction(1, 2)), "Native time and frame overlap classify the range differently")
    return {"flag": flagged, "assessed": assessed, "exact_correct": exact, "time_correct": time,
            "spatial_correct": spatial_correct, "joint_correct": time and spatial_correct}


def review(run, prior):
    lock = study.validate(run, prior)
    aggregate, runtime = (read(run / name) for name in ("aggregate.json", "runtime.json"))
    require(runtime == aggregate["runtime"] and runtime["journal_sha256"] == sha(run / "held.jsonl")
            and runtime["freeze_sha256"] == sha(run / "freeze.json"), "Held commitment differs")
    data = {(row["bank"], row["slot"], row["condition"]): row for row in study.previous.rows(run, "held")}
    require(len(data) == 229, "Incomplete independent replay roster")
    counts, status, groups = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    originals = crops = frames = exact_pairs = spatial_checks = 0
    folder = run / "deliveries"
    for name, index, group, source, original, cases, donor in study.roster(prior):
        ref = folder / f"{name}-{index:03d}-earlier.mkv"
        expected = [Fraction(t, 1000) for t in study.relative_native(original)]
        authored_source_table = probe_frames(source, timeout_s=12)
        ref_table = study.table(ref)
        ref_hash = sha(ref)
        require(clocks(ref_table) == clocks(authored_source_table) == expected
                and sha(source) == lock[name, index]["native_earlier_sha256"], "Earlier picture clock or bytes differ")
        originals += 1
        extra = [(kind, f"tandem_crf{crf}", None, None, folder / f"{name}-{index:03d}-tandem_crf{crf}.mkv", None, False, None)
                 for kind in ("tandem",) for crf in (28, 33)]
        if donor:
            extra.append(("different_source", "different_source_native_prefix", donor[2], donor[3],
                          folder / f"{name}-{index:03d}-unrelated.mkv", None, False, None))
        for kind, condition, authored, authored_row, delivery, known, compatible, previous in [*cases, *extra]:
            recorded = data.pop((name, index, condition))
            observed_table = study.table(delivery)
            require(clocks(observed_table) == expected and (ref_table.width, ref_table.height)
                    == (observed_table.width, observed_table.height) and not (observed_table.rotation
                       or observed_table.interpolated or observed_table.truncated)
                    and (recorded["earlier_sha256"], recorded["submitted_sha256"]) == (ref_hash, sha(delivery))
                    and recorded["known_edit"] == known and recorded["source_group"] == group
                    and recorded["visible_source_time"] == compatible
                    and (not authored or sha(authored) == authored_row["file_sha256"] == recorded["authored_sha256"]),
                    "Delivered carrier, geometry, source clocks, or truth differs")
            exact_pairs += 1
            current = approved(recorded, expected)
            old = approved(previous, expected) if previous else {"joint_correct": False}
            joint, prior_joint = current["joint_correct"], old["joint_correct"]
            counts[name, kind].update({"views": 1, **current, "prior_joint_correct": prior_joint,
                "prior_correct_retained": prior_joint and joint, "prior_correct_lost": prior_joint and not joint,
                "new_correct": joint and not prior_joint, "visible_source_time": compatible,
                "visible_source_time_joint": compatible and joint})
            status[name][f"{kind}/{recorded['status']}"] += 1
            if joint and kind == "neural":
                groups[name].add(group)
            if not current["flag"]:
                continue
            best = recorded["best"]
            first, end = best["frames"]
            before, during, after = best["samples"]
            require(before + 1 == first <= during < end == after < len(expected), "Selected picture indices differ")
            x, y, w, h = best["tile_xywh"]
            require(w > 0 and h > 0 and 0 <= x <= ref_table.width - w and 0 <= y <= ref_table.height - h,
                    "Native crop coordinates differ")
            for path, table in ((ref, ref_table), (delivery, observed_table)):
                saved = read_pictures(path, table, best["samples"])
                for picture in saved.values():
                    require(picture[y:y+h, x:x+w].shape == (h, w, 3), "A selected native crop exceeds its source")
                    crops += 1
                frames += len(saved)
            if known:
                prior_table = probe_frames(authored, timeout_s=12)
                actual = measure_spatial(source, authored_source_table, authored, prior_table, best["tile_xywh"], during)
                require(actual == recorded["spatial"], "Authored native spatial pixel counts differ")
                spatial_checks += 1
    require(not data and exact_pairs == 229, "Independent held roster differs")
    for name, bank in aggregate["panels"].items():
        require({kind: dict(counts[name, kind]) for kind in bank["counts"]} == bank["counts"]
                and dict(status[name]) == bank["statuses"] and len(groups[name]) == bank["correct_source_groups"],
                "Published held decision counts differ")
        rules, counts_here = study.SPEC["gate"], bank["counts"]
        neural, neutral, tandem, unrelated = (counts_here[key] for key in ("neural", "neutral", "tandem", "different_source"))
        criteria = {
            "prior_recall": neural["prior_correct_retained"] * 10 >= neural["prior_joint_correct"] * 9,
            "group_coverage": len(groups[name]) >= rules["minimum_new_groups"],
            "assessed": all(row["assessed"] * 5 >= row["views"] * 4 for row in (neural, neutral, tandem)),
            "neutral_flags": neutral["flag"] <= rules["neutral_flags_per_bank"],
            "tandem_flags": tandem["flag"] == rules["tandem_flags"],
            "unrelated_flags": unrelated["flag"] == rules["unrelated_flags"],
        }
        require(criteria == bank["criteria"] and all(criteria.values()) == bank["advances"], "Frozen gate result differs")
    require(all(row["advances"] for row in aggregate["panels"].values()) == aggregate["advances_both"], "Joint gate differs")
    return {"status": "verified", "earlier_h264_clocks": originals, "exact_held_pairs": exact_pairs,
            "native_frames_reopened": frames, "selected_native_crops_checked": crops,
            "authored_spatial_counts_replayed": spatial_checks, "aggregate_sha256": sha(run / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--prior", type=Path, default=study.PRIOR)
    args = parser.parse_args()
    result = review(args.run, args.prior)
    write(args.run / "independent-audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
