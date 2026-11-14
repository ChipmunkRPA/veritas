"""Replay a calibrated agreement rule across three views of a writing response."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .review_writing_f1000_position_transfer import POINTS, check, digest
from .review_writing_long_position import select_case
from .writing_omission_attribution import log_odds, responses, verify_panel

PANELS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
FILES = {
    "omission": ("writing-omission-attribution-2026-11-13.json", "0c8721669f689ad8cba9344600c2757a3a5ce4835155836911dc380533dfd51d"),
    "point": ("writing-long-position-transfer-2026-11-12.json", "1b4713fb1fc38053a1f28d2c4e564fa587cb16950d261070e8dd7ab6d6871801"),
    "f1000": ("writing-f1000-position-transfer-2026-11-13.json", "2de36e115c061e9cf4e353a567a6987ccf116adb00d49a18478c2cf12e5a6b3d"),
    "elife": ("writing-elife-long-position-transfer-2026-11-13.json", "227d5287d17e3f8cf92cbccfedc30c6e134b4e04767be56805d024538bd15e99"),
}
SCHEMA = "chipmunk.writing-context-agreement-replay/1"
FEATURES = ("position_contrast", "isolated_log_odds", "two_sided_omission")
METHOD = {
    "features": FEATURES,
    "normalization": "Each feature uses the median and median absolute deviation of 14 ordinary work maxima. Scale floor is 0.25 raw units.",
    "score": "The median of the three normalized features at the same exact sentence. The largest unique complete response can select one sentence.",
    "point": "Strict excess over the second-highest ordinary work maximum of the normalized median. Both versions per work participate.",
    "geometry": "Reuse only exact source and input hash matches. All three measurements must be complete at each eligible center.",
    "comparison": "Preserve both existing context and longer-position locations and count new exact locations separately from flags elsewhere.",
    "advance": "Across the two panels, at least three added exact locations from distinct works, both generators and both publishers. At most one added ordinary work per publisher and at most two wrong locations per added exact location.",
}


def load(name):
    filename, expected = FILES[name]
    path = PANELS / filename
    check(digest(path) == expected, f"The pinned {name} receipt differs")
    return json.loads(path.read_bytes())


def bound_cases(name, omission):
    data = omission[name]
    verify_panel(name, data)
    cases = load(name)["cases"]
    prior = {(row["group"], row["variant"]): row for row in cases
             if name != "point" or row["role"] == "calibration"}
    check(len(prior) == len(data["cases"]), "The two receipts cover different source views")
    for case in data["cases"]:
        old = prior.pop((case["group"], case["variant"]))
        check(all(old[key] == case[key] for key in ("text_sha256", "truth", "positions", "nonspace")),
              "The two receipts identify different source positions")
        selected = select_case(old, POINTS["long_document"], allow_punctuation_unit=name != "point",
                               allow_touching=name == "elife")
        incumbent = case["incumbent"]
        check((
            incumbent["context_candidates"], incumbent["context_useful"], incumbent["position_candidate"],
            incumbent["position_useful"]
        ) == (selected["current"], selected["current_useful"], selected["position"], selected["position_useful"]),
              "The two receipts identify different incumbent locations")
        position = {item["idx"]: item for item in old["review"]}
        inputs = data["inputs"]
        omission_scores = responses(case, inputs)
        scores = {}
        for index, left_pair, right_pair, _left, isolated, _right in case["centers"]:
            item = position[index]
            left, center, right = item["responses"]
            check(left is not None and center is not None and right is not None
                  and [left["input_sha256"], center["input_sha256"], right["input_sha256"]]
                  == [left_pair, old["native_current"][index]["input_sha256"], right_pair],
                  "The two receipts do not identify the same measured context")
            raw = inputs[isolated][1]
            contrast = item["contrast"]
            if index in omission_scores and raw is not None and contrast is not None:
                scores[index] = (contrast, log_odds(raw), omission_scores[index][0])
        yield case, scores


def normal_score(features, centers, scales):
    return median((value - center) / scale for value, center, scale in zip(features, centers, scales, strict=True))


def calibration(omission):
    groups = defaultdict(list)
    for case, scores in bound_cases("point", omission):
        check(case["truth"] is None and scores, "Calibration requires complete ordinary source views")
        groups[case["group"]].extend(scores.values())
    peaks = {key: [max(component) for component in zip(*rows, strict=True)] for key, rows in groups.items()}
    components = list(zip(*peaks.values(), strict=True))
    centers = [median(component) for component in components]
    scales = [max(.25, median(abs(value - center) for value in component))
              for component, center in zip(components, centers, strict=True)]
    maxima = {group: max(normal_score(row, centers, scales) for row in rows) for group, rows in groups.items()}
    cutoff = sorted(maxima.values())[-2]
    return {"schema": SCHEMA, "method": json.loads(json.dumps(METHOD)), "point_sha256": FILES["point"][1],
            "measurements_sha256": FILES["omission"][1], "work_maxima": maxima, "feature_maxima": peaks,
            "centers": centers, "scales": scales, "cutoff": cutoff}


def evaluate(name, omission, point):
    counts = Counter()
    groups = defaultdict(set)
    flags = []
    for case, features in bound_cases(name, omission):
        group, variant, truth, incumbent = (case[key] for key in ("group", "variant", "truth", "incumbent"))
        scores = {index: normal_score(row, point["centers"], point["scales"]) for index, row in features.items()}
        ranked = sorted(scores, key=scores.get, reverse=True)
        winner = ranked[0] if ranked and (len(ranked) == 1 or scores[ranked[0]] > scores[ranked[1]]) else None
        chosen = case["positions"][winner][:2] if winner is not None and scores[winner] > point["cutoff"] else None
        counts["files"] += 1
        counts["assessed_files"] += bool(scores)
        counts["measured_sentences"] += len(scores)
        if truth is None:
            counts[f"{variant}_ordinary"] += 1
            counts[f"{variant}_ordinary_flags"] += chosen is not None
            if incumbent["context_candidates"] or incumbent["position_candidate"] is not None:
                groups["incumbent_ordinary"].add(group)
            if chosen is not None:
                groups["candidate_ordinary"].add(group)
        else:
            counts[f"{variant}_edits"] += 1
            counts["target_assessed"] += any(case["positions"][index][:2] == truth for index in scores)
            counts["target_unique_top"] += winner is not None and case["positions"][winner][:2] == truth
            current = incumbent["context_useful"] or incumbent["position_useful"]
            exact = chosen == truth
            counts["incumbent_located"] += current
            counts["combined_located"] += current or exact
            counts["wrong_locations"] += chosen is not None and not exact
            if exact and not current:
                groups["additional_works"].add(group)
                groups["additional_generators"].add(variant)
        if chosen is not None:
            flags.append({"group": group, "variant": variant, "span": chosen, "known_edit": truth})
    groups["additional_ordinary"] = groups["candidate_ordinary"] - groups["incumbent_ordinary"]
    return {"counts": dict(counts), "groups": {key: sorted(value) for key, value in sorted(groups.items())}, "flags": flags}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "review"))
    parser.add_argument("freeze", type=Path)
    args = parser.parse_args()
    omission = load("omission")
    point = calibration(omission)
    if args.mode == "calibrate":
        with args.freeze.open("x") as stream:
            print(json.dumps(point, indent=2, sort_keys=True), file=stream)
        return
    check(json.loads(args.freeze.read_bytes()) == point, "The ordinary calibration freeze differs")
    outcomes = {name: evaluate(name, omission, point) for name in ("f1000", "elife")}
    panels = [(result["counts"], result["groups"]) for result in outcomes.values()]
    added = sum(counts["combined_located"] - counts["incumbent_located"] for counts, _ in panels)
    advance = (
        all(groups.get("additional_works") and len(groups["additional_ordinary"]) <= 1 for _, groups in panels)
        and sum(len(groups.get("additional_works", ())) for _, groups in panels) >= 3
        and len({family for _, groups in panels for family in groups.get("additional_generators", ())}) == 2
        and sum(counts["wrong_locations"] for counts, _ in panels) <= 2 * added
    )
    print(json.dumps({"advance": advance, "outcomes": outcomes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
