"""Independently verify source windows, localized responses and false-group counts."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from review_audio_spectra_context_transfer import RATE, digest, read, require, useful

ARMS = ("current", "boundary", "combined")
STEP, LENGTH = 1280, 12800


def sha(path):
    return digest(path.read_bytes())


def exact_windows(data, observation):
    samples = len(data)//2
    require([r["start"] for r in observation["current"]] == list(range(RATE, samples-2*RATE+1, 2*RATE)), "Current clock changed")
    for item in observation["current"]:
        a, b = item["start"], item["end"]
        require(b-a == 2*RATE and digest(data[2*a:2*b]) == item["pcm_sha256"], "Current model window input changed")
    boundary = observation["boundary"]
    require([r["start"] for r in boundary] == list(range(3*STEP, samples-2*STEP-LENGTH, STEP)), "Boundary clock changed")
    for item in boundary:
        a, b = item["start"], item["end"]
        require(a >= 2880 and b+2880 <= samples and b-a == LENGTH and len(item["endpoints"]) == 2
                and all(-1e-12 <= value <= math.log(2)+1e-12 for value in item["endpoints"])
                and item["score"] == min(item["endpoints"]), "Boundary score lacks paired native support")


def paired(observation):
    for edge in observation["boundary"]:
        a, b = edge["start"], edge["end"]
        context = [w["score"] for w in observation["current"] if w["score"] is not None and w["start"] <= a < b <= w["end"]]
        if context:
            yield edge, max(context)


def replay(observation, scaling):
    def peak(rows):
        return max(rows, key=lambda x: x[0], default=[None, None])

    current = peak([w["score"], [w["start"], w["end"]]] for w in observation["current"] if w["score"] is not None)
    edge = peak([w["score"], [w["start"], w["end"]]] for w in observation["boundary"])
    mixed = []
    for window, score in paired(observation):
        x = (score-scaling[0][0])/scaling[0][1]
        y = (window["score"]-scaling[1][0])/scaling[1][1]
        mixed.append([(x+y)/2, [window["start"], window["end"]]])
    return dict(zip(ARMS, (current, edge, peak(mixed)), strict=True))


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered)-1)*quantile
    lower = int(index)
    upper = math.ceil(index)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower)


def counts(rows, arm, point):
    available = [r for r in rows if r["responses"][arm][0] is not None]
    flags = [r for r in available if r["responses"][arm][0] > point]
    ai = [r for r in flags if r["truth"] is not None]
    return {"ai_flags": len(ai), "useful_ranges": sum(useful(r["responses"][arm][1], r["truth"]) for r in ai),
            "false_groups": len({r["group"] for r in flags if r["truth"] is None}), "unassessed": len(rows)-len(available),
            "background_false_flags": sum(r["arm"] == "background_only" for r in flags)}


def verify(run, same, alternative):
    lock, result, roster, point, decisions = (read(run / f"{key}.json") for key in ("freeze", "result", "roster", "points", "decisions"))
    require(all(sha(run / f"{name}.json") == commitment for name, commitment in result["commitments"].items())
            and lock["roster_sha256"] == sha(run / "roster.json") and point["freeze_sha256"] == sha(run / "freeze.json")
            and point["point_sha256"] == sha(run / "point.json"), "Study receipt differs")
    require(Counter(row["panel"] for row in roster) == {"point": 12, "point_positive": 6, "clone_hold": 24, "exposed_transfer": 20}, "Scope changed")
    folders = {"same": same, "alt": alternative}
    rows = {(r["asset"], r["id"]): r for r in roster}
    require(len({r["pcm_sha256"] for r in roster}) == len(rows) == 62, "Exact PCM has duplicated source identity")
    responses = {(r["asset"], r["id"]): r for r in decisions}
    ledger = {role: read(run / f"{role}.json") for role in ("point", "held", "alternate")}
    frozen = point["values"]
    paired_scores = [(current, edge["score"]) for observation in ledger["point"] for edge, current in paired(observation)]
    for values, (center, iqr) in zip(zip(*paired_scores, strict=True), frozen["scalers"], strict=True):
        require(math.isclose(center, percentile(values, .5), abs_tol=1e-12)
                and math.isclose(iqr, max(1e-6, percentile(values, .75)-percentile(values, .25)), abs_tol=1e-12),
                "Paired local score scaling differs")
    windows, candidates = 0, 0
    for role, observations in ledger.items():
        asset = "alt" if role == "alternate" else "same"
        for obs in observations:
            key = asset, obs["id"]
            row = rows.pop(key)
            native = f"{obs['id']}-original.pcm" if asset == "same" else f"{obs['id']}.pcm"
            data = (folders[asset] / "pcm" / native).read_bytes()
            require(len(data) == 2*row["samples"] and digest(data) == row["pcm_sha256"] == obs["pcm_sha256"], "Original input changed")
            require((role == "point") == (row["panel"] == "point" and row["truth"] is None), "A positive entered genuine-only point selection")
            exact_windows(data, obs)
            expected = replay(obs, frozen["scalers"])
            require(responses[key] == row | {"responses": expected}, "Exact localized response differs")
            windows += len(obs["current"])
            candidates += len(obs["boundary"])
    require(not rows and len(responses) == 62, "Missing frozen responses")
    for arm in ARMS:
        values = [r["responses"][arm][0] for r in decisions if r["panel"] == "point" and r["responses"][arm][0] is not None]
        require(frozen["points"][arm] == max(values), "Genuine-only fixed point differs")
    for panel, record in result["panels"].items():
        chosen = [r for r in decisions if r["panel"] == panel or (panel == "all_held" and r["panel"] in {"clone_hold", "exposed_transfer"})
                  or (panel in {"stock", "expressive"} and r["panel"] == "exposed_transfer" and r["family"] == panel)]
        require(record["edits"] == sum(r["truth"] is not None for r in chosen) and record["genuine_views"] == sum(r["truth"] is None for r in chosen),
                "Reported panel counts differ")
        for arm in ARMS:
            require(record["fixed"][arm] == counts(chosen, arm, frozen["points"][arm]), "Fixed operating results differ")
            maxima = defaultdict(list)
            for row in chosen:
                value = row["responses"][arm][0]
                if row["truth"] is None and value is not None:
                    maxima[row["group"]].append(value)
            ordered = sorted((max(v) for v in maxima.values()), reverse=True)
            for budget, table in record["equal_false_groups"].items():
                i = int(budget)
                require(table[arm] == (counts(chosen, arm, ordered[i]) if i < len(ordered) else None), "Descriptive false-group counts differ")
    print(f"Verified 62 exact original PCM inputs, {windows} native model windows, {candidates} paired ranges and all false-group counts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "same", "alternative"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(args.run, args.same, args.alternative)
