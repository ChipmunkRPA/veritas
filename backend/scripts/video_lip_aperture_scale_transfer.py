# SPDX-License-Identifier: MIT
"""Locate the causes of native lip-timing misses and test clipwise aperture scaling."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from app.evidence import video_viseme_ranked as ranked
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_dense_native as native
from scripts import video_lip_dense_posteriors as dense

prior = dense.prior
SCHEMA = "chipmunk.video-lip-aperture-scale/1"
SCALED = ("per_clip_scale", "per_clip_ranks")
WIDE_LAGS = tuple(range(-6400, 6401, timing.STEP))
WIDE = ("normalized_wide_grid", "ranked_wide_grid")
TRIALS = (*SCALED, *WIDE)
CANDIDATES = ("sparse", "normalized_aperture", *TRIALS)
PANELS = prior.prior.PANEL
PRIMARY = PANELS / "video-lip-dense-posteriors-2026-11-15.json.gz"
CARRIER = PANELS / "video-lip-dense-carrier-2026-11-15.json.gz"
NATIVE = PANELS / "video-lip-dense-native-2026-11-15.json"
METHOD = {
    "diagnostic": "Give the same dense and sparse events their exact authored edit span and signed timing change. "
                  "Try the existing common-flank lag and count the first unmet condition before a flag. "
                  "This uses known labels only for diagnosis.",
    "scale": "Rescale each clip's dense aperture prediction to its observed 10th to 90th percentile independently. "
             "Keep the existing mouth-gap scale. The rank arm instead uses midranks on both series. Use the same "
             "20 ms prediction cadence, 40 ms measurement grid, native pictures, timing search and range overlap.",
    "points": "Use each candidate's largest and second-largest speaker maxima among the frozen ordinary "
              "calibration deliveries, including uniform delays. Keep the prior sparse and dense cutoffs.",
    "wide_grid": "After the first native diagnostic, lock a separate trial widening the absolute lag grid from "
                 "±240 ms to ±400 ms. Keep the signed local change requirement, score scales and interval search. "
                 "Evaluate original aperture and clipwise ranks separately on this grid, including the held actors.",
    "transfer": "Use the original GRID and matched PCM and AAC sources, the first 24 qualified native actors, "
                "and the remaining qualified native actors. The same frozen fitting and points apply.",
}


def scaled_scores(row, prediction):
    points = timing.observations(row)
    valid = sorted(p for p in prediction if p is not None)
    if points is None or len(valid) < 8 or valid[-1] == valid[0]:
        return []
    count = len(valid) // 10
    low, high = valid[count], valid[-1 - count]
    if high - low < 100:
        return []
    result = []
    # Retain the original observed frame and presentation time for every score.
    for i in range(2, len(prediction) - 2, 2):
        value = prediction[i]
        if value is None:
            continue
        sample = dense.CENTER + i * dense.STRIDE
        reads = [timing.nearest(points, sample - lag) for lag in timing.LAGS]
        if any(item is None for item in reads):
            continue
        frames = [frame for _, frame in reads]
        target = min(1.5, max(-.5, (value - low) / (high - low)))
        strengths = [round(-timing.SCORE_UNITS * (min(1.5, abs(gap - target)) / timing.SCALE) ** 2) for gap, _ in reads]
        result.append((sample, "aperture", strengths, frames))
    return result


def tagged_case(row, role, values, phones, references, *, expected):
    if values["pcm_sha256"] != expected:
        raise ValueError("The acoustic prediction has a different source")
    shift = prior.parent.EDITS.get(role)
    kind = "local" if shift else "constant" if role.startswith("global_") else "ordinary"
    prediction = values["prediction"]
    scores = {
        "sparse": timing.compatible(row, phones, references),
        "normalized_aperture": dense.match_scores(row, prediction)["normalized_aperture"],
        "per_clip_scale": scaled_scores(row, prediction),
        "per_clip_ranks": ranked.scores(row, prediction),
        "normalized_wide_grid": dense.match_scores(row, prediction, lags=WIDE_LAGS)["normalized_aperture"],
        "ranked_wide_grid": ranked.scores(row, prediction, lags=WIDE_LAGS),
    }
    selections = {}
    for arm, events in scores.items():
        selected = timing.hypotheses(events, row["samples"], lags=WIDE_LAGS if arm in WIDE else timing.LAGS)
        if selected["status"] == "measured":
            plates = timing.picture_evidence(row["frames"], selected, row["samples"])
            if not all(plates["picture_brackets"].values()):
                selected = {"status": "incomplete_context", "eligible_events": len(events)}
        selections[arm] = selected
    return row, role, kind, shift, selections, scores


def primary_cases(primary):
    sparse = prior.load(PANELS / "video-lip-viseme-transfer-2026-11-15.json.gz")
    references = sparse["result"]["references"]
    old = {row["source"]: row for row in sparse["acoustic"]["rows"]}
    development, held = [], []
    for row in prior.timings()["rows"]:
        if prior.is_fit(row):
            continue
        for role, case in row["cases"].items():
            value = primary["predictions"][row["source"]][role]
            if prior.is_calibration(row) and role in prior.parent.EDITS:
                continue
            phones = old[row["source"]]["cases"][role]["phones"]
            unit = tagged_case(row, role, value, phones, references, expected=case["pcm_sha256"])
            (development if prior.is_calibration(row) else held).append(unit)
    points = {arm: primary["result"]["points"][arm] for arm in ("sparse", "normalized_aperture")}
    for arm in TRIALS:
        maxima = defaultdict(float)
        for row, _, _, _, selections, _ in development:
            maxima[row["speaker"]] = max(maxima[row["speaker"]], selections[arm].get("score", 0.))
        ordered = sorted(maxima.values(), reverse=True)
        if len(ordered) != 4:
            raise ValueError("Expected four unchanged calibration speakers")
        points[arm] = {"tight": ordered[0], "wider": ordered[1]}
    if points["per_clip_ranks"]["tight"] != ranked.POINT or tuple(primary["fit"]["weights"]) != ranked.WEIGHTS:
        raise ValueError("The local ranking rule differs from the measured fit and point")
    return points, held


def carrier_cases(carrier, primary):
    reference = prior.load(dense.CARRIER_SOURCE)
    if (carrier["schema"], carrier["model"], carrier["primary_sha256"], carrier["source_sha256"]) != (
        dense.CARRIER_SCHEMA, dense.MODEL, prior.phone.sha(PRIMARY), prior.phone.sha(dense.CARRIER_SOURCE)
    ) or carrier["result"] != dense.carrier_results(reference["rows"], carrier["predictions"],
                                                    primary["result"]["points"], reference["references"]):
        raise ValueError("Carrier inputs differ from the frozen measurements")
    cases = defaultdict(list)
    for row in reference["rows"]:
        for codec, deliveries in row["cases"].items():
            for role, entry in deliveries.items():
                prediction = carrier["predictions"][row["source"]][codec][role]
                cases[codec].append(tagged_case(row, role, prediction, entry["phones"], reference["references"],
                                                expected=entry["sha256"]))
    return cases


def actor_cases(document, qualification, primary):
    if (document["schema"], document["model"], document["primary_sha256"]) != (
        native.SCHEMA, dense.MODEL, prior.phone.sha(PRIMARY)
    ) or document["result"] != native.evaluate(qualification, document["rows"], primary):
        raise ValueError("Native speech inputs differ from the frozen clocks or predictions")
    for source in document["rows"]:
        scale = Fraction(source["time_base"]) * timing.RATE
        frames = [f | {"sample": round(f["pts"] * scale) - source["audio_origin"],
                       "gap": None if f["gap_units"] is None else f["gap_units"] / timing.SCORE_UNITS}
                  for f in source["frames"]]
        row = source | {"frames": frames}
        for role, values in source["cases"].items():
            yield tagged_case(row, role, values, values["phones"], native.binding.REFERENCE, expected=values["pcm_sha256"])


def oracle(events, truth, shift, point, *, lags=timing.LAGS):
    """Keep both the scored support and known edit clock fixed for diagnosis."""
    start, stop = truth
    before = [e for e in events if e[0] < start]
    inside = [e for e in events if start <= e[0] < stop]
    after = [e for e in events if e[0] >= stop]
    if not before or len(inside) < 2 or not after:
        return "missing_before_during_or_after"
    left, local, right = (timing.totals(group, lags=lags) for group in (before, inside, after))
    outer = min(range(len(lags)), key=lambda i: (-(left[i] + right[i]), abs(lags[i]), lags[i]))
    changed = lags[outer] + shift
    if changed not in lags:
        return "signed_lag_outside_grid"
    inner = lags.index(changed)
    if min(left[outer] - left[inner], right[outer] - right[inner]) <= 0:
        return "flanks_do_not_prefer_common_lag"
    if local[inner] <= local[outer]:
        return "inside_does_not_prefer_authored_shift"
    numerator = left[outer] + local[inner] + right[outer] - max(timing.totals(events, lags=lags))
    gain = round(numerator / (timing.SCORE_UNITS * len(events)), 7)
    if gain <= 0:
        return "constant_lag_fits_better"
    if gain <= point:
        return "below_frozen_point"
    return "authored_shift_passes"


def summarize(cases, points):
    condensed, diagnostics = [], defaultdict(Counter)
    fallback = defaultdict(Counter)
    for row, role, kind, shift, scored, events in cases:
        condensed.append((row, role, kind, shift, scored))
        existing, additional = scored["sparse"], scored["per_clip_ranks"]
        for level in ("tight", "wider"):
            first = existing.get("score", 0.) > points["sparse"][level]
            second = not first and additional.get("score", 0.) > points["per_clip_ranks"][level]
            picked = existing if first else additional if second else None
            useful = bool(shift and picked and prior.parent.useful(picked["span"], picked["change"], row["edit"], shift))
            fallback[f"{level}/{kind}"].update(files=1, baseline_flags=first, added_flags=second,
                                                useful=useful, added_useful=second and useful)
        if kind == "local":
            for arm in CANDIDATES:
                cutoff = points[arm]["tight"]
                where = oracle(events[arm], row["edit"], shift, cutoff, lags=WIDE_LAGS if arm in WIDE else timing.LAGS)
                measurement = scored[arm]
                useful = measurement.get("score", 0.) > cutoff and prior.parent.useful(
                    measurement["span"], measurement["change"], row["edit"], shift)
                diagnostics[arm][where] += 1
                diagnostics[arm]["measured_and_useful"] += useful
                diagnostics[arm]["oracle_passes_but_search_misses"] += where == "authored_shift_passes" and not useful
    result = dense.summarize(condensed, points)
    return {"outcomes": result["outcomes"], "oracle": dict(diagnostics), "fallback": dict(fallback),
            "located_ranges": sum(name.split("/")[0] in TRIALS for name in result["evidence"])}


def evaluate(*, audit=None, previous=None, remaining=None):
    checkpoint = dense.review(PRIMARY)
    points, main = primary_cases(checkpoint)
    reproducible = {"primary": summarize(main, points)}
    for codec, cases in carrier_cases(prior.load(CARRIER), checkpoint).items():
        reproducible[codec] = summarize(cases, points)
    result = {"schema": SCHEMA, "method": METHOD, "source": {p.name: prior.phone.sha(p) for p in (PRIMARY, CARRIER, NATIVE)},
              "points": points, "portable": reproducible}
    if audit is not None:
        public = prior.load(NATIVE)
        if prior.phone.sha(previous) != public["source_sha256"]:
            raise ValueError("Original native measurements differ from their earlier receipt")
        if prior.load(remaining).get("remaining_actors") is not True:
            raise ValueError("Expected the separate native actors")
        result["native"] = {}
        for label, path in (("previous", previous), ("remaining", remaining)):
            qualified = native.native.roster(audit, remaining=label == "remaining")
            entries = actor_cases(prior.load(path), qualified, checkpoint)
            result["native"][label] = {"source_sha256": prior.phone.sha(path), "actors": len(qualified)} | summarize(entries, points)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("receipt", type=Path)
    for name in ("audit", "previous", "remaining"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if any((args.audit, args.previous, args.remaining)) and not all((args.audit, args.previous, args.remaining)):
        parser.error("Supply --audit and both native actor receipts to verify native outcomes")
    new = evaluate(audit=args.audit, previous=args.previous, remaining=args.remaining)
    if args.phase == "run":
        with args.receipt.open("x") as stream:
            json.dump(new, stream, sort_keys=True, indent=2)
            stream.write("\n")
    else:
        stored = prior.load(args.receipt)
        used = set(new)
        if any(stored.get(key) != new[key] for key in used):
            raise ValueError("Published lip timing outcomes differ from replay")
    print(json.dumps(new, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
