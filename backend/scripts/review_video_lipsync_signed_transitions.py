# SPDX-License-Identifier: Apache-2.0
"""Recount source-bound signed timing results without an ML runtime."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.review_video_lipsync_window_schedule import CASES, _useful

SCHEMA = "chipmunk.video-lipsync-signed-transitions/1"
ARMS = ("aperture_level", "signed_transitions", "agreement")
SIGNALS = ARMS[:2]
ROLES = ("before", "during", "after")
WIDTHS = (12, 16, 22)
FLANK = 12
GAP = 1
MAX_LAG = 10
MIN_LAG_CHANGE, MAX_LAG_CHANGE = 3, 8
MIN_CORRELATION = .30
MIN_FLANK_GAIN = .035
MIN_SCORE, MARGIN = .10, .03
HOP, RATE = 640, 16000
SHA = re.compile(r"[0-9a-f]{64}")
EXTRA = ("matched_picture_early", "matched_picture_late")
DESIGN = {
    "articulation": "At successive distinct observed pictures, correlate signed change in mouth aperture with "
                    "signed change in compressed speech level. A static arm uses aperture and RMS level.",
    "bracket": "Search 480, 640 and 880 ms middle intervals. Leave one tick between the middle and each "
               "480 ms flank. Both flanks must favor a common lag over the changed lag.",
    "agreement": "Both measurements must pass on the same bracket, with a shared pair of lags.",
    "point": "For each arm use max(0.10, highest observed development control score + 0.03), then freeze.",
    "useful": "A reported interval overlaps the known shift by at least 0.40 IoU and signs and estimates "
              "the lag within 80 ms.",
    "controls": "Unchanged, gain, filtering, local noise, constant shifts and two matched picture and sound shifts. "
                "Matched picture controls reassign verified mouth observations on the same grid without re-encoding.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scores(comparisons, reference, changed):
    """Admit signed hypotheses only when each flank and the middle support them."""
    delta = changed - reference
    if not (abs(reference) <= MAX_LAG and abs(changed) <= MAX_LAG and MIN_LAG_CHANGE <= abs(delta) <= MAX_LAG_CHANGE):
        return {}
    scores_by_signal = {}
    for name, ((left, left_moved), (center, center_moved), (right, right_moved)) in comparisons.items():
        confidence = min(left, center_moved, right)
        flank_gain = min(left - left_moved, right - right_moved)
        change_gain = center_moved - center
        if confidence >= MIN_CORRELATION and flank_gain >= MIN_FLANK_GAIN and change_gain > 0:
            scores_by_signal[name] = round(min(change_gain, 2 * flank_gain), 6)
    if all(name in scores_by_signal for name in SIGNALS):
        scores_by_signal["agreement"] = min(scores_by_signal.values())
    return scores_by_signal


def brackets(start, width):
    stop = start + width
    return ((start - GAP - FLANK, start - GAP), (start, stop), (stop + GAP, stop + GAP + FLANK))


def thresholds(rows):
    maxima = {name: 0.0 for name in ARMS}
    for group in rows:
        for case in group.get("cases", []):
            if case["role"].startswith("local_"):
                continue
            for name, arm in case["candidate"].items():
                if arm["best"]:
                    maxima[name] = max(maxima[name], arm["best"]["score"])
    return {name: round(max(MIN_SCORE, value + MARGIN), 6) for name, value in maxima.items()}


def _validate_candidate(arm, name, frame_ticks, pcm_samples):
    count, best = arm["brackets"], arm["best"]
    require(type(count) is int and 0 <= count <= len(WIDTHS) * frame_ticks, "Invalid bracket coverage")
    if best is None:
        return
    start, end = best["ticks"]
    lag, reference = best["changed_lag"], best["reference_lag"]
    spans = brackets(start, end - start)
    require(count and type(start) is int and type(end) is int and end - start in WIDTHS
            and type(lag) is int and type(reference) is int
            and spans[0][0] - MAX_LAG > 0 and spans[-1][1] + MAX_LAG <= pcm_samples // HOP
            and spans[-1][1] <= frame_ticks,
            "Selected bracket exceeds declared picture or sound coverage")
    pairs = best["correlations"]
    needed = set(SIGNALS if name == "agreement" else (name,))
    require(set(pairs) == needed and all(len(rows) == 3 and all(len(row) == 2
            and all(math.isfinite(x) and -1 <= x <= 1 for x in row) for row in rows) for rows in pairs.values()),
            "Invalid flank or middle correlations")
    value = scores(pairs, reference, lag).get(name)
    require(value is not None and value == best["score"], "A selected score does not replay")


def _validate_witness(witness, interval, reference, changed, frames, time_base, origin, pcm_samples, role, group, ticks):
    edit_start = (ticks - 16) // 2 + group % 5 - 2
    edit_end = edit_start + (16, 22)[group % 2]
    shift = (5, 7)[group % 2] * (-1 if role.endswith("early") else 1)
    require(set(witness) == set(ROLES), "Incomplete source evidence for selected interval")
    for position, (start, stop) in zip(ROLES, brackets(interval[0], interval[1] - interval[0]), strict=True):
        part = witness[position]
        midpoint = (start + stop - 1) // 2
        require(set(part) == {"video", "audio"} and {frame["tick"] for frame in part["video"]} == {start, midpoint, stop - 1}
                and len(part["video"]) == 3, "Missing verified source pictures")
        for frame in part["video"]:
            index = frame["frame_index"]
            picture_tick = frame["tick"]
            source_tick = picture_tick - shift if role.startswith("matched_") and edit_start <= picture_tick < edit_end else picture_tick
            require(type(index) is int and 0 <= index < frames and type(frame["pts"]) is int
                    and frame["source_tick"] == source_tick and 0 <= source_tick < ticks
                    and SHA.fullmatch(frame["decoded_rgb_sha256"]) and SHA.fullmatch(frame["mouth_patch_sha256"])
                    and abs(frame["pts"] * time_base - origin - Fraction(source_tick, 25)) <= Fraction(1, 25),
                    "A sequence witness has invalid native video support")
        require(set(part["audio"]) == {"reference", "changed"}, "Missing decoded sound support")
        for key, lag in (("reference", reference), ("changed", changed)):
            a, b = (start + lag - 1) * HOP, (stop + lag) * HOP
            recorded = part["audio"][key]
            require(0 <= a < b <= pcm_samples and recorded["samples"] == [a, b]
                    and [Fraction(t) for t in recorded["time"]] == [origin + Fraction(n, RATE) for n in (a, b)]
                    and SHA.fullmatch(recorded["sha256"]), "Selected sound support differs from the source clock")


def review(document):
    freeze, rows = document["freeze"], document["rows"]
    roster = freeze["sources"]
    require(freeze["schema"] == SCHEMA and freeze["design"] == DESIGN and len(rows) == len(roster) == 37
            and Counter(s["bank"] for s in roster) == {"acted": 15, "conversation": 11, "later_excerpt": 11}
            and len({(s["bank"], s["group"]) for s in roster}) == len(roster)
            and sum(s["split"] == "development" for s in roster) == 4, "Incomplete frozen roster")
    development = [row for row, plan in zip(rows, roster, strict=True) if plan["split"] == "development"]
    points = thresholds(development)
    require(points == document["points"], "Development control points differ")
    outputs = {}
    arms = ("current", *ARMS, "combined")
    totals = {bank: {name: Counter() for name in arms} for bank in ("acted", "conversation", "later_excerpt")}
    coverage = Counter()
    for plan, row in zip(roster, rows, strict=True):
        bank = plan["bank"]
        require((row["bank"], row["group"], row["source_sha256"]) == (bank, plan["group"], plan["sha256"])
                and SHA.fullmatch(row["source_sha256"]), "A source group differs from the freeze")
        held = plan["split"] == "held"
        if held:
            coverage[bank, "groups"] += 1
            coverage[bank, "edits"] += 2
            coverage[bank, "controls"] += len(CASES) + len(EXTRA) - 2
        if row.get("unavailable"):
            if held:
                coverage[bank, "unavailable"] += 1
            continue
        samples, ticks, frame_count = row["samples"], row["ticks"], row["frame_count"]
        require(55 <= ticks <= samples // HOP and frame_count > 0 and Fraction(row["time_base"]) > 0
                and len(row["cases"]) == len(CASES) + len(EXTRA)
                and {case["role"] for case in row["cases"]} == {*CASES, *EXTRA}, "Incomplete native deliveries")
        for case in row["cases"]:
            role = case["role"]
            truth, lag = case["truth"], case["lag"]
            require(SHA.fullmatch(case["pcm_sha256"]) and set(case["candidate"]) == set(ARMS), "Invalid delivery binding")
            if role.startswith("local_"):
                a, b = truth
                require(type(lag) is int and abs(lag) == (5, 7)[row["group"] % 2] and (lag < 0) == role.endswith("early")
                        and all(type(n) is int and n % HOP == 0 for n in truth)
                        and 0 <= a < b <= samples and (b - a) // HOP == (16, 22)[row["group"] % 2],
                        "Authored timing references differ")
            else:
                require(truth is None and lag is None, "A control has edited-interval truth")
            current = case["current"]
            require(all(type(t) is int and 15 <= t <= ticks - 40 for t in current["starts"])
                    and all(kind in ("offset_step", "confidence_collapse") and 0 <= a < b <= 2 * ticks
                            and (type(delta) is int if kind == "offset_step" else delta is None)
                            for kind, a, b, delta in current["ranges"]), "Invalid current timing evidence")
            hits = {"current": _useful(current["ranges"], truth, lag)}
            alerts = {"current": bool(current["ranges"])}
            measured = {"current": bool(current["starts"])}
            for name, arm in case["candidate"].items():
                _validate_candidate(arm, name, ticks, samples)
                best = arm["best"]
                reported = bool(held and best and best["score"] >= points[name])
                require(("witness" in best if best else False) == reported, "Candidate release differs from its fixed point")
                if reported:
                    _validate_witness(best["witness"], best["ticks"], best["reference_lag"], best["changed_lag"],
                                      frame_count, Fraction(row["time_base"]), Fraction(row["origin"]), samples,
                                      role, row["group"], ticks)
                ranges = [["offset_step", 2 * best["ticks"][0], 2 * best["ticks"][1],
                           best["changed_lag"] - best["reference_lag"]]] if reported else []
                hits[name] = _useful(ranges, truth, lag)
                alerts[name] = reported
                measured[name] = arm["brackets"] > 0
            hits["combined"] = hits["current"] or hits["agreement"]
            alerts["combined"] = alerts["current"] or alerts["agreement"]
            measured["combined"] = measured["current"] or measured["agreement"]
            if held:
                for name in arms:
                    total = totals[bank][name]
                    total["measured_edits" if truth else "measured_controls"] += measured[name]
                    if truth:
                        total["useful"] += hits[name]
                        total["wrong_location_or_lag"] += alerts[name] and not hits[name]
                    else:
                        total["control_flags"] += alerts[name]
                        total["global_or_synchronized_flags"] += alerts[name] and role.startswith(("global_", "matched_"))
    for bank, methods in totals.items():
        outputs[bank] = {**{key: coverage[bank, key] for key in ("groups", "edits", "controls", "unavailable")},
                         "arms": {name: dict(value) for name, value in methods.items()}}
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    doc = json.loads(parser.parse_args().receipt.read_text())
    outcome = review(doc)
    require(outcome == doc["outcome"], "Published comparison counts do not recompute")
    print(json.dumps(outcome, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
