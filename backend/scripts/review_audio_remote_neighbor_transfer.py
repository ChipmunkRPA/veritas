"""Independently check original audio, remote frame clocks, and result arithmetic."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from review_audio_enrolled_target_transfer import pcm
from review_audio_original_pcm_boundary_transfer import RATE, counts, digest, read, require, sha, useful
from review_audio_variable_boundary_transfer import actual_overlap

ARMS = ("current", "neighbor")


def best_range(record):
    support = {}
    for item in record["support"]:
        t, targets = item["center"], item["neighbors"]
        require(t not in support and len(targets) == len({x//3200 for x in targets}) == 4 and math.isfinite(item["score"])
                and all(200 <= x <= 95560 and (x-200) % 320 == 0 for x in [t, *targets])
                and all(abs(t-x) >= 12800 for x in targets), "A remote neighbor is outside its native support")
        support[t] = item["score"]
    best = [None, None]
    for a in range(0, 83201, 3200):
        score = [value for t, value in support.items() if a <= t < a+12800]
        if len(score) >= 24:
            value = median(score)
            if best[0] is None or value > best[0]:
                best = [value, [a, a+12800]]
    return best


def exact_fresh(run, rows):
    groups = defaultdict(list)
    for row in rows:
        raw = [(run / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes() for suffix in ("", "-foreground", "-background")]
        require(all(len(data) == 192000 and digest(data) == row[k] for data, k in zip(raw, ("pcm_sha256", "foreground_sha256", "background_sha256"), strict=True)),
                "Exact new original PCM or components differ")
        require(all(x == y+z for x, y, z in zip(*(pcm(data) for data in raw), strict=True)), "Original PCM no longer equals its exact components")
        groups[row["group"]].append((row, raw))
    voices, captures = set(), defaultdict(list)
    for group in groups.values():
        row, ref = group[0]
        room, offset = row["room"], row["background"]["capture_start_sample"]
        require(not any(offset < b and a < offset+96000 for a, b in captures[room]), "Duplicate room recording support")
        captures[room].append((offset, offset+96000))
        if row["edit"] is None:
            require(len(group) == 1 and not any(ref[1]), "Isolated room contains foreground")
            continue
        a, b = row["edit"]
        require(len(group) == 3 and {r["arm"] for r, _ in group} == {"ordinary_same_host", "ordinary_other_voice", "synthetic"}, "Matched sources differ")
        speakers = {r["donor"]["canonical_speaker"] for r, _ in group}
        require(len(speakers) == 3 and not voices & speakers, "Fresh held source identities overlap")
        voices.update(speakers)
        for chosen, raw in group:
            require(chosen["edit"] == [a, b] and chosen["truth"] == ([a,b] if chosen["arm"] == "synthetic" else None)
                    and ref[2] == raw[2] and all(x[:2*a] == y[:2*a] and x[2*b:] == y[2*b:] for x, y in zip(ref[:2], raw[:2], strict=True)),
                    "Original matched input differs outside exact edit support")
    require(len(voices) == 12, "Fresh held voice denominator differs")


def fixed_union(rows, cutoffs):
    available, flags, located, room, false = 0, 0, 0, 0, set()
    for row in rows:
        scores = [(arm, score, span) for arm, (score, span) in row["responses"].items() if score is not None]
        available += bool(scores)
        kept = [span for arm, score, span in scores if score > cutoffs[arm]]
        if kept and row["truth"] is None:
            false.add(row["group"])
            room += row["arm"] == "background_only"
        if kept and row["truth"] is not None:
            flags += 1
            located += any(useful(span, row["truth"]) for span in kept)
    return {"ai_flags": flags, "useful_ranges": located, "unassessed": len(rows)-available,
            "background_false_flags": room, "false_groups": len(false)}


def verify(run, point_root, same, earlier, short):
    receipt, roster, point, held, decisions, seal = (read(run / f"{n}.json") for n in ("result", "roster", "point", "held", "decisions", "points"))
    require(all(sha(run / f"{n}.json") == pin for n, pin in receipt["commitments"].items()) and held["points_sha256"] == sha(run / "points.json")
            and seal["point_sha256"] == sha(run / "point.json"), "Run receipts differ")
    require(Counter(row["cohort"] for row in roster) == {"point": 12, "earlier_expressive": 26, "earlier_two_family": 20, "fresh": 14}, "Frozen cohort sizes differ")
    for row, scores in zip((x for x in roster if x["cohort"] == "point"), point, strict=True):
        raw = (same / "pcm" / f"{row['id']}-original.pcm").read_bytes()
        require(row["truth"] is None and len(raw) == 192000 and digest(raw) == row["pcm_sha256"] == scores["pcm_sha256"], "Genuine point input differs")
    ranking = [best_range(r)[0] for r in point]
    require(seal["neighbor"] == max(x for x in ranking if x is not None), "Genuine-only neighbor point differs")
    sources = {"earlier_expressive": earlier, "earlier_two_family": short, "fresh": run}
    artifacts = ((earlier, "current.json"), (short, "held.json"), (run, "fresh-current.json"))
    ledger = {(name, row["id"]): row for name, (folder, file) in zip(sources, artifacts, strict=True) for row in read(folder / file)["records"]}
    for row, observation, output in zip((r for r in roster if r["cohort"] != "point"), held["rows"], decisions, strict=True):
        key = row["asset"], row["id"]
        current = ledger[key]
        raw = (sources[row["asset"]] / "pcm" / f"{row['id']}.pcm").read_bytes()
        require(key == (observation["asset"], observation["id"]) and len(raw) == 192000
                and digest(raw) == row["pcm_sha256"] == observation["pcm_sha256"] == current["pcm_sha256"], "Held score source differs")
        windows = current.get("current", current.get("windows"))
        require([w["start"] for w in windows] == [RATE, 3*RATE], "Baseline native clock differs")
        for w in windows:
            a, b = w["start"], w["end"]
            require(b-a == 2*RATE and digest(raw[2*a:2*b]) == w["pcm_sha256"], "Baseline native samples differ")
        best = max(([w["score"], [w["start"], w["end"]]] for w in windows if w["score"] is not None), key=lambda x: x[0], default=[None, None])
        require(output == row | {"responses": {"current": best, "neighbor": best_range(observation)}}, "Held localization did not replay")
    cutoffs = {"current": read(point_root / "points.json")["values"]["points"]["current"], "neighbor": seal["neighbor"]}
    for cohort, result in receipt["result"].items():
        selected = [row for row in decisions if row["cohort"] == cohort]
        require(fixed_union(selected, cutoffs) == result["fixed_union"], "Fixed union differs")
        for arm in ARMS:
            require(counts(selected, arm, cutoffs[arm]) == result["fixed"][arm]
                    and counts(selected, arm, float("-inf"))["useful_ranges"] == result["locations_before_cutoff"][arm], "Reported fixed results differ")
            flags = defaultdict(list)
            for row in selected:
                score = row["responses"][arm][0]
                if row["truth"] is None and score is not None:
                    flags[row["group"]].append(score)
            ranked = sorted((max(v) for v in flags.values()), reverse=True)
            for budget, row in result["equal_false_groups"].items():
                i = int(budget)
                require(row[arm] == (counts(selected, arm, ranked[i]) if i < len(ranked) else None), "Equal false group arithmetic differs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "point", "same", "earlier", "short", "annotations"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    fresh = [row for row in read(args.run / "roster.json") if row["cohort"] == "fresh"]
    require(len(fresh) == 14, "Fresh source scope differs")
    exact_fresh(args.run, fresh)
    exact = actual_overlap(fresh, args.annotations)
    if not args.source_only:
        verify(args.run, args.point, args.same, args.earlier, args.short)
    print(f"Verified fresh exact PCM and 320–760 ms local overlap: {exact}")
