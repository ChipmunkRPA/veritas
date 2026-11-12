"""Check the original-audio replication clocks, controls and frozen comparisons."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from review_audio_enrolled_target_transfer import pcm
from review_audio_original_pcm_boundary_transfer import RATE, counts, digest, read, require, sha
from review_audio_remote_neighbor_transfer import best_range
from review_audio_variable_boundary_transfer import actual_overlap

ARMS = ("current", "neighbor", "same_time")


def exact_sources(run, rows):
    groups = defaultdict(list)
    for row in rows:
        data = [(run / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes() for suffix in ("", "-foreground", "-background")]
        require(all(len(raw) == 192000 and digest(raw) == row[key] for raw, key in zip(data, ("pcm_sha256", "foreground_sha256", "background_sha256"), strict=True)),
                "Exact original or source component changed")
        require(all(x == y+z for x, y, z in zip(*(pcm(raw) for raw in data), strict=True)), "Original PCM differs from component sum")
        groups[row["group"]].append((row, data))
    captures, voices, donors = defaultdict(list), set(), set()
    for siblings in groups.values():
        first, original = siblings[0]
        start, room = first["background"]["capture_start_sample"], first["room"]
        require(not any(a < start+96000 and start < b for a,b in captures[room]), "An exact room recording was reused")
        captures[room].append((start, start+96000))
        if first["edit"] is None:
            require(len(siblings) == 1 and not any(original[1]), "Genuine room-only control contains foreground")
            continue
        a, b = first["edit"]
        require(len(siblings) == 4 and {r["arm"] for r, _ in siblings} == {"untouched", "ordinary_same_host", "ordinary_other_voice", "synthetic"},
                "Expected two ordinary edits, one untouched and one synthetic")
        require((first["coverage"] == "early" and b <= 16000) or (first["coverage"] == "late" and a >= 80000)
                or (first["coverage"] == "interior" and any(x <= a < b <= x+32000 for x in (16000,48000))), "Frozen truth was moved across coverage strata")
        group_voices = {r["donor"]["canonical_speaker"] for r, _ in siblings if r["donor"] is not None}
        require(len(group_voices) == 3 and not group_voices & voices, "Foreground speaker role repeats")
        voices.update(group_voices)
        for row, data in siblings:
            require(row["background"] == first["background"] and row["edit"] == [a,b]
                    and row["truth"] == ([a,b] if row["arm"] == "synthetic" else None), "Matched edit or background clock differs")
            require(original[2] == data[2] and all(x[:2*a] == y[:2*a] and x[2*b:] == y[2*b:] for x,y in zip(original[:2], data[:2], strict=True)),
                    "Matched controls differ outside the declared edit")
            if donor := row["donor"]:
                cut = donor["cut"]
                require(donor["pcm_sha256"] not in donors and cut["end_sample"]-cut["start_sample"] == cut["source_end_pts"]-cut["source_start_pts"] == b-a,
                        "A donor fragment or its source clock repeats")
                donors.add(donor["pcm_sha256"])
    require(len(voices) == len(donors) == 36, "New source identities or exact fragments incomplete")


def same_time(observation, current):
    available = [(w["start"], w["end"]) for w in current if w["score"] is not None]
    winner = [None, None]
    for a in range(0, 83201, 3200):
        b = a+12800
        coverage = sum(max(0, min(b,y)-max(a,x)) for x,y in available)
        if coverage != 12800:
            continue
        samples = [r["score"] for r in observation["support"] if a <= r["center"] < b]
        if len(samples) >= 24 and (winner[0] is None or median(samples) > winner[0]):
            winner = [median(samples), [a,b]]
    return winner


def replay(run, roster, discovery, prior):
    lock, result, neighbor, current, decisions = (read(run / f"{key}.json") for key in ("freeze", "result", "neighbor", "current", "decisions"))
    require(all(sha(run / f"{key}.json") == pin for key,pin in result["commitments"].items())
            and neighbor["freeze"] == current["freeze"] == sha(run / "freeze.json"), "Run receipt differs")
    points = {"neighbor": read(discovery / "points.json")["neighbor"], "current": read(prior / "points.json")["values"]["points"]["current"]}
    require(points == lock["points"] and lock["discovery_points"] == sha(discovery / "points.json") and lock["current_points"] == sha(prior / "points.json"),
            "A prior fixed point was changed")
    for row, n, c, out in zip(roster, neighbor["records"], current["records"], decisions, strict=True):
        raw = (run / "pcm" / f"{row['id']}.pcm").read_bytes()
        require((row["id"], digest(raw)) == (n["id"], n["pcm_sha256"]) == (c["id"], c["pcm_sha256"]), "Original model source differs")
        windows = c["windows"]
        require([w["start"] for w in windows] == [RATE, 3*RATE], "Incumbent native clock changed")
        for w in windows:
            a, b = w["start"], w["end"]
            require(b-a == 2*RATE and digest(raw[2*a:2*b]) == w["pcm_sha256"], "Incumbent original input window differs")
        scores = max(([w["score"], [w["start"], w["end"]]] for w in windows if w["score"] is not None), key=lambda x: x[0], default=[None,None])
        selected = {"current": scores, "neighbor": best_range(n), "same_time": same_time(n, windows)}
        require(out == row | {"responses": selected}, "Native or restricted-time decision differs")
    for panel, table in result["results"].items():
        chosen = [r for r in decisions if panel == "all" or r["family"] == panel or (panel == "interior" and r["coverage"] in ("room", "interior"))
                  or (panel == "edge" and r["coverage"] in ("room", "early", "late"))]
        require(Counter(r["arm"] for r in chosen) == table["counts"], "Reported panel scope differs")
        for arm in ARMS:
            require(counts(chosen, arm, points["neighbor" if arm == "same_time" else arm]) == table["fixed"][arm]
                    and counts(chosen, arm, float("-inf"))["useful_ranges"] == table["locations_before_cutoff"][arm], "Fixed or unthresholded results differ")
            scores = defaultdict(list)
            for row in chosen:
                value = row["responses"][arm][0]
                if row["truth"] is None and value is not None:
                    scores[row["group"]].append(value)
            ranking = sorted((max(v) for v in scores.values()), reverse=True)
            for budget, expected in table["equal_false_groups"].items():
                i = int(budget)
                require(expected[arm] == (counts(chosen, arm, ranking[i]) if i < len(ranking) else None), "Equal false group result differs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "discovery", "prior", "annotations"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    rows = read(args.run / "roster.json")
    require(len(rows) == 52 and Counter(r["coverage"] for r in rows if r["truth"]) == {"interior": 8, "early": 2, "late": 2}, "Original roster scopes differ")
    exact_sources(args.run, rows)
    local = actual_overlap(rows, args.annotations)
    if not args.source_only:
        replay(args.run, rows, args.discovery, args.prior)
    print(f"Verified 52 exact original files and source support. The 12 edits each have {min(local)}–{max(local)} ms of overlapping background words.")
