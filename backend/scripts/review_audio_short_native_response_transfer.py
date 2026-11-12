"""Independently replay native short-audio input ranges and comparison counts."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from review_audio_enrolled_target_transfer import pcm
from review_audio_original_pcm_boundary_transfer import RATE, counts, digest, percentile, read, require, sha
from review_audio_variable_boundary_transfer import actual_overlap

LONG, SHORT, STEP = 32000, 19200, 9600
ARMS = ("current", "short", "cross_scale")


def exact_sources(run, rows):
    groups = defaultdict(list)
    for row in rows:
        audio = [(run / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes() for suffix in ("", "-foreground", "-background")]
        for raw, key in zip(audio, ("pcm_sha256", "foreground_sha256", "background_sha256"), strict=True):
            require(len(raw) == 192000 and digest(raw) == row[key], "Original input or component differs")
        require(all(x == y+z for x, y, z in zip(*(pcm(raw) for raw in audio), strict=True)), "Original PCM differs from source component sum")
        groups[row["group"]].append((row, audio))
    clips, voices = defaultdict(list), set()
    for cases in groups.values():
        first, original = cases[0]
        t = first["background"]["capture_start_sample"]
        room = first["room"]
        require(not any(t < b and a < t+96000 for a, b in clips[room]), "Room capture was reused")
        clips[room].append((t, t+96000))
        if first["truth"] is None and first["edit"] is None:
            require(len(cases) == 1 and not any(original[1]), "Room-only control contains foreground")
            continue
        require({row["arm"] for row, _ in cases} == {"ordinary_same_host", "ordinary_other_voice", "synthetic"}, "Matched controls differ")
        a, b = first["edit"]
        require(b-a in {10240, 12800, 15360}, "Edited duration differs")
        matched_voices = {r["donor"]["canonical_speaker"] for r, _ in cases}
        require(len(matched_voices) == 3 and not voices & matched_voices, "Matched donor groups repeat")
        voices.update(matched_voices)
        for row, inputs in cases:
            require(row["background"] == first["background"] and row["edit"] == [a, b] and row["truth"] == ([a, b] if row["arm"] == "synthetic" else None),
                    "Matched edit or background clock differs")
            require(original[2] == inputs[2] and all(x[:2*a] == y[:2*a] and x[2*b:] == y[2*b:] for x, y in zip(original[:2], inputs[:2], strict=True)),
                    "Matched files differ outside the exact edit")
    require(len(voices) == 18, "Held foreground speaker denominator differs")


def exact_windows(raw, observation):
    for arm, width, starts in (("current", LONG, (RATE, 3*RATE)), ("short", SHORT, range(0, len(raw)//2-SHORT+1, STEP))):
        require([w["start"] for w in observation[arm]] == list(starts), "Source window starts differ")
        for win in observation[arm]:
            a, b = win["start"], win["end"]
            require(b-a == width and digest(raw[2*a:2*b]) == win["pcm_sha256"], "Native model input window hash differs")


def links(observation):
    for short in observation["short"]:
        if short["score"] is None:
            continue
        left, right = short["start"], short["end"]
        context = [w["score"] for w in observation["current"] if w["score"] is not None
                   and min(right, w["end"])-max(left, w["start"]) >= STEP]
        if context:
            yield short, max(context)


def responses(observation, scaling):
    def peak(items):
        return max(items, key=lambda row: row[0], default=[None, None])

    result = {arm: peak([w["score"], [w["start"], w["end"]]] for w in observation[arm] if w["score"] is not None) for arm in ARMS[:2]}
    short_scale = scaling[1]
    long_scale = scaling[0]
    result["cross_scale"] = peak([((s["score"]-short_scale[0])/short_scale[1]+(value-long_scale[0])/long_scale[1])/2,
                                  [s["start"], s["end"]]] for s, value in links(observation))
    return result


def compare(run, roster, point_root, same):
    outcomes, points, point, held, decisions = (read(run / f"{name}.json") for name in ("result", "points", "point", "held", "decisions"))
    require(all(sha(run / f"{name}.json") == pin for name, pin in outcomes["commitments"].items())
            and points["point_sha256"] == sha(run / "point.json") and points["freeze_sha256"] == sha(run / "freeze.json")
            and held["points_sha256"] == sha(run / "points.json"), "Fixed point and score receipt commitments differ")
    prior_rows = [r for r in read(point_root / "roster.json") if r["panel"] == "point"]
    prior_observations = read(point_root / "point.json")
    for src, obs, before in zip(prior_rows, point["records"], prior_observations, strict=True):
        raw = (same / "pcm" / f"{src['id']}-original.pcm").read_bytes()
        require(src["truth"] is None and (src["id"], src["pcm_sha256"]) == (obs["id"], digest(raw))
                and obs["current"] == before["current"], "Point score does not use the earlier genuine receipt")
        exact_windows(raw, obs)
    saved = points["values"]
    features = [(long, short["score"]) for observation in point["records"] for short, long in links(observation)]
    for values, (center, iqr) in zip(zip(*features, strict=True), saved["scalers"], strict=True):
        require(math.isclose(center, percentile(values, .5), abs_tol=1e-12)
                and math.isclose(iqr, max(1e-6, percentile(values, .75)-percentile(values, .25)), abs_tol=1e-12), "Cross-scale reference scaling differs")
    selected = [responses(obs, saved["scalers"]) for obs in point["records"]]
    require(saved["points"] == {arm: max(r[arm][0] for r in selected if r[arm][0] is not None) for arm in ARMS}, "Genuine-only score points differ")
    for row, obs, decision in zip(roster, held["records"], decisions, strict=True):
        raw = (run / "pcm" / f"{row['id']}.pcm").read_bytes()
        require((row["id"], row["pcm_sha256"]) == (obs["id"], obs["pcm_sha256"]) and decision == row | {"responses": responses(obs, saved["scalers"])},
                "Held source or selected interval differs")
        exact_windows(raw, obs)
    for name, result in outcomes["results"].items():
        family_groups = {r["group"] for r in roster if r["truth"] and r["donor"]["family"] == name}
        chosen = decisions if name == "held" else [r for r in decisions if r["group"] in family_groups]
        for arm in ARMS:
            require(counts(chosen, arm, saved["points"][arm]) == result["fixed"][arm], "Fixed outcomes differ")
            genuine = defaultdict(list)
            for row in chosen:
                score = row["responses"][arm][0]
                if row["truth"] is None and score is not None:
                    genuine[row["group"]].append(score)
            ranking = sorted((max(v) for v in genuine.values()), reverse=True)
            for budget, table in result["equal_false_groups"].items():
                rank = int(budget)
                require(table[arm] == (counts(chosen, arm, ranking[rank]) if rank < len(ranking) else None), "Matched false-group results differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "point_root", "same", "annotations"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    roster = read(args.run / "roster.json")
    require(Counter(r["arm"] for r in roster) == {"ordinary_same_host": 6, "ordinary_other_voice": 6, "synthetic": 6, "background_only": 2}, "Held denominator differs")
    exact_sources(args.run, roster)
    overlap = actual_overlap(roster, args.annotations)
    if not args.source_only:
        compare(args.run, roster, args.point_root, args.same)
    print(f"Verified 20 exact held inputs and original sample clocks, with {min(overlap)}–{max(overlap)} ms of actual competing speech during edits")
