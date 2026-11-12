# SPDX-License-Identifier: Apache-2.0
"""Verify the paired background experiment's held geometry and decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-native-background-results/1"
CANDIDATE = "ab7908fbaafe35a5e879190ada426401d7b248e8d37069db733b330a5df2eca6"
SPAN_COUNTS = {"cross_cut": 16, "long_swap": 16, "short_swap": 8, "light": 8, "camera_punch": 8, "original": 8}
DECISION_KEYS = (
    "cut_both_edges_within_one_picture", "cut_both_edges_exact", "useful_cut_removal",
    "false_matched_background_cut", "useful_swap_coarse", "lost_useful_swap_range",
    "current_useful_swap", "counterfactual_useful_swap",
)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sha(value: str) -> bool:
    return re.fullmatch(r"[a-f0-9]{64}", value) is not None


def expected(pair: int, direction: int) -> dict:
    return {
        "cross_cut": ([54, 60], [67, 75], [54, 64], [65, 77])[(pair + 2 * direction) % 4],
        "long_swap": ([50, 68], [56, 78])[direction],
        **({"short_swap": ([54, 60], [62, 73])[direction]} if pair < 4 else {}),
        ("light" if direction else "camera_punch"): [67, 75],
        **({} if direction else {"original": None}),
    }


def span(record):
    return type(record) is list and len(record) == 2 and all(type(i) is int for i in record) and 0 <= record[0] < record[1] < 120


def useful(ranges, truth):
    if truth is None:
        return False
    for first, last in ranges:
        if 2 * (min(last, truth[1]) - max(first, truth[0])) >= max(last, truth[1]) - min(first, truth[0]):
            return True
    return False


def verify_candidate(candidate, incumbent, manifest):
    edges = candidate["edge_frames"]
    require(incumbent and span(edges) and 5 <= edges[1] - edges[0] <= 12 and candidate["coarse_frames"] == incumbent
            and candidate["method"] == "paired-background-edge/1"
            and all(abs(a - b) <= 9 for a, b in zip(edges, incumbent, strict=True)), "Candidate range is outside its frozen scope")
    a, b = edges
    require(2 * (min(b, incumbent[1]) - max(a, incumbent[0])) >= min(b - a, incumbent[1] - incumbent[0]),
            "Candidate pair did not overlap the selected coarse range")
    eligible, support = candidate["eligible_tiles"], candidate["support_tiles"]
    require(len(eligible) >= 16 and eligible == sorted(set(eligible)) and support == sorted(set(support))
            and all(type(t) is int and 0 <= t < 48 for t in eligible) and set(support) <= set(eligible)
            and 100 * len(support) >= 35 * len(eligible), "Invalid non-face tile support")
    quadrants = Counter((i // 8 >= 3, i % 8 >= 4) for i in support)
    require(sum(value >= 2 for value in quadrants.values()) >= 3, "Candidate support lacks spatial coverage")
    indices = [a - 1, a, b - 1, b]
    require([p["index"] for p in candidate["pictures"]] == indices
            and all(valid_sha(p["rgb_sha256"]) for p in candidate["pictures"]), "Evidence pictures must straddle both native edges")
    if manifest is not None:
        clock = manifest["clock"]
        require(candidate["time_base"] == clock["time_base"] and candidate["edge_pts"] == [clock["pts"][a], clock["pts"][b]]
                and all(p["pts"] == clock["pts"][i] and p["rgb_sha256"] == manifest["rgb_sha256"][i]
                        for p, i in zip(candidate["pictures"], indices, strict=True)), "Native picture pixels or clock differ")
    else:
        points = [p["pts"] for p in candidate["pictures"]]
        time_base = Fraction(candidate["time_base"])
        require(candidate["edge_pts"] == [points[1], points[3]] and
                all((q - p) * time_base == Fraction(y - x, 25) for p, q, x, y in zip(points, points[1:], indices, indices[1:])),
                "Candidate timestamp geometry differs")


def decisions(rows, media=None):
    seen, hashes, removed_pairs = set(), set(), set()
    counts = defaultdict(Counter)
    for row in rows:
        pair, direction, kind = (row[key] for key in ("source_pair", "direction", "condition"))
        key = pair, direction, kind
        require(type(pair) is int and 0 <= pair < 8 and type(direction) is int and direction in (0, 1)
                and kind in expected(pair, direction) and key not in seen, "Repeated or undeclared held role")
        seen.add(key)
        sha = row["source_sha256"]
        require(valid_sha(sha) and sha not in hashes, "Repeated or invalid submitted file identity")
        hashes.add(sha)
        edited = expected(pair, direction)[kind]
        swap = kind in ("long_swap", "short_swap")
        require(row["authored"] == edited and row["true_swap"] is swap, "An authored boundary differs from the locked recipe")
        if media is not None:
            item = media[key]
            require(item["sha256"] == sha and item["interval"] == edited, "Report and media manifest bind different inputs")
        else:
            item = None
        before, displayed, after = (row[key] for key in ("incumbent", "displayed", "counterfactual_displayed"))
        correspondence = {"face_sequence": "coarse", "face_boundary": "boundary_refinement", "face_return": "short_range"}
        require(all(k in correspondence.values() and span(v) for k, v in before.items())
                and all(k in correspondence and span(v) and v == before.get(correspondence[k]) for k, v in displayed.items()),
                "A displayed finding differs from its measured native range")
        candidate = row["candidate"]
        if candidate:
            verify_candidate(candidate, before.get("coarse"), item)
        kept = {k: v for k, v in displayed.items() if candidate is None or k == "face_return"}
        require(kept == after, "The candidate must only remove the affected displayed coarse and refined range")
        close = bool(candidate and kind == "cross_cut" and all(abs(a - b) <= 1 for a, b in zip(candidate["edge_frames"], edited, strict=True)))
        useful_coarse = useful((v for k, v in displayed.items() if k != "face_return"), edited if swap else None)
        removed = close and "face_sequence" in displayed
        if removed:
            removed_pairs.add(pair)
        checks = (
            close, close and candidate["edge_frames"] == edited, removed,
            bool(candidate and kind != "cross_cut"), useful_coarse, bool(candidate and useful_coarse),
            useful(displayed.values(), edited if swap else None), useful(kept.values(), edited if swap else None),
        )
        require(all(row[name] is value for name, value in zip(DECISION_KEYS, checks, strict=True)), "A held decision failed independent replay")
        summary = {
            "cases": 1, "shot_transitions": bool(row["shot_boundaries"]), "coarse_review_measured": row["coarse_status"] == "ok",
            "coarse_ranges": "coarse" in before, "displayed_coarse_ranges": "face_sequence" in displayed,
            "displayed_short_ranges": "face_return" in displayed, "counterfactual_coarse_ranges": "face_sequence" in kept,
            "counterfactual_short_ranges": "face_return" in kept, "candidate_matched_background_pairs": candidate is not None,
            **dict(zip(DECISION_KEYS, checks, strict=True)),
        }
        counts[kind].update(summary)
    require({name: c["cases"] for name, c in counts.items()} == SPAN_COUNTS and len(seen) == 64, "The held panel is incomplete")
    gates = {
        "useful_cut_removals": counts["cross_cut"]["useful_cut_removal"] >= 2,
        "distinct_source_pairs": len(removed_pairs) >= 2,
        "useful_swap_coarse_coverage": sum(c["useful_swap_coarse"] for c in counts.values()) >= 1,
        "useful_swap_preservation": all(c["lost_useful_swap_range"] == 0 for c in counts.values()),
        "no_false_matched_background_cuts": all(c["false_matched_background_cut"] == 0 for c in counts.values()),
    }
    return {"held_counts": {k: dict(c) for k, c in counts.items()}, "useful_cut_source_pairs": len(removed_pairs),
            "gate": gates, "advances": all(gates.values())}


def read(path: Path):
    return json.loads(path.read_bytes())


def review(document, media=None):
    require(document["schema"] == SCHEMA and document["aggregate"]["candidate_sha256"] == CANDIDATE,
            "The report names a different candidate")
    calculated = decisions(document["cases"], media)
    require(all(document["aggregate"][k] == v for k, v in calculated.items()), "The public aggregate differs from its verified decisions")
    return {"cases": len(document["cases"]), "gate": calculated["gate"], "advances": calculated["advances"]}


def export(run: Path, output: Path):
    aggregate = read(run / "aggregate.json")
    authoring = read(run / "independent-authoring.json")
    media_path = run / "media.json"
    candidate_journal = run / "private-candidate.jsonl"
    require(authoring["status"] == "verified" and authoring["counts"]["identical_mp4s"] == 64
            and authoring["freeze_sha256"] == aggregate["panel_freeze_sha256"] == digest(run / "freeze.json")
            and authoring["media_sha256"] == digest(media_path)
            and aggregate["candidate_journal_sha256"] == digest(candidate_journal)
            and aggregate["candidate_freeze_sha256"] == digest(run / "candidate-freeze.json")
            and aggregate["incumbent_seal_sha256"] == digest(run / "incumbent-seal.json"),
            "The held run lacks matching source, execution and independent authoring receipts")
    media = {(m["pair"], m["direction"], m["condition"]): m for m in read(media_path)["media"]}
    cases = [json.loads(line) for line in candidate_journal.read_text().splitlines()]
    document = {"schema": SCHEMA, "aggregate": aggregate, "cases": cases,
                "independent_authoring": authoring, "independent_authoring_sha256": digest(run / "independent-authoring.json")}
    result = review(document, media)
    with output.open("x") as stream:
        stream.write(json.dumps(document, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    result = export(args.source, args.export) if args.export else review(read(args.source))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
