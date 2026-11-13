"""Replay a frozen expression-feature transfer to source-separated article passages."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

SCHEMA = "chipmunk.article-expression-transfer/1"
ARMS = {"style": 12, "style_plus_position": 13}
POSITION_POINTS = {"short_passage": 2.7579683180027486, "article": 3.6821095168341915}


def read(path):
    return json.loads(path.read_bytes())


def response(weights, vector):
    return (
        sum(
            (value - mean) * beta / scale
            for value, mean, beta, scale in zip(
                vector, weights["center"], weights["beta"][:-1], weights["scale"], strict=True
            )
        )
        + weights["beta"][-1]
    )


def stable_sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def select(weights, rows, name):
    ranked = sorted(((response(weights, row[name]), row["idx"]) for row in rows), reverse=True)
    if not ranked or ranked[0][0] <= weights["point"] or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        return None
    return ranked[0][1]


def verify_fit(weights, name, development, ridge):
    width = ARMS[name]
    train = []
    calibration = defaultdict(list)
    for source in development:
        for row in source["rows"]:
            values = row[name]
            assert len(values) == width and all(math.isfinite(v) for v in values)
            if source["role"] == "calibration":
                assert source["truth"] is None
                calibration[source["group"]].append(values)
            elif source["truth"] is None or row["span"] == source["truth"]:
                train.append((values, source["truth"] is not None))
    labels = [label for _, label in train]
    positive = sum(labels)
    negative = len(labels) - positive
    assert positive == weights["generated_targets"] and negative == weights["genuine_centers"]
    assert positive == 24 and negative == 164 and len(calibration) == 14
    for axis, values in enumerate(zip(*(v for v, _ in train), strict=True)):
        assert math.isclose(fmean(values), weights["center"][axis], abs_tol=1e-10)
        assert math.isclose(max(0.15, pstdev(values)), weights["scale"][axis], abs_tol=1e-10)
    gradient = [ridge * b / len(train) for b in weights["beta"][:-1]] + [0.0]
    for values, label in train:
        error = (stable_sigmoid(response(weights, values)) - label) * (
            0.5 / (positive if label else negative)
        )
        for i, value in enumerate(values):
            gradient[i] += error * (value - weights["center"][i]) / weights["scale"][i]
        gradient[-1] += error
    assert max(map(abs, gradient)) < 2e-7
    maxima = {group: max(response(weights, row) for row in rows) for group, rows in calibration.items()}
    for group, value in maxima.items():
        assert math.isclose(value, weights["per_genuine_work_maximum"][group], abs_tol=1e-10)
    assert math.isclose(sorted(maxima.values())[-2], weights["point"], abs_tol=1e-10)


def baseline(case):
    return [i for i, item in enumerate(case["native_current"]) if item["score"] >= 0.5][:6]


def baseline_useful(case, indices, truth):
    nonspace = case["nonspace"]
    return any(
        truth in (span := range(max(0, i - 1), min(i + 2, len(nonspace))))
        and 10 * nonspace[truth] >= 3 * sum(nonspace[k] for k in span)
        for i in indices
    )


def position_only(cases):
    result = {}
    for name, point in POSITION_POINTS.items():
        ordinary = set()
        useful, wrong = [], []
        for case in cases:
            complete = sorted(case["position"], key=lambda row: row["contrast"], reverse=True)
            if not complete or complete[0]["contrast"] <= point:
                continue
            if len(complete) > 1 and complete[0]["contrast"] == complete[1]["contrast"]:
                continue
            selected = case["positions"][complete[0]["idx"]][:2]
            if case["truth"] is None:
                ordinary.add(case["group"])
            elif selected == case["truth"]:
                useful.append([case["group"], case["variant"], selected])
            else:
                wrong.append([case["group"], case["variant"], selected])
        result[name] = {"point": point, "genuine_groups": sorted(ordinary), "useful": useful, "wrong": wrong}
    return result


def tally(cases, arms):
    results = {}
    for name, weights in arms.items():
        controls, positive = [], []
        for row in cases:
            assert len(row["positions"]) == len(row["nonspace"]) == len(row["native_current"])
            current = baseline(row)
            selected = select(weights, row["rows"], name)
            for item in row["rows"]:
                i = item["idx"]
                assert 0 < i < len(row["positions"]) - 1 and item["span"] == row["positions"][i][:2]
                assert len({p[2] for p in row["positions"][i - 1 : i + 2]}) == 1 and item["background"] >= 6
                assert all(math.isfinite(v) for v in item[name]) and len(item[name]) == ARMS[name]
            if row["truth"] is None:
                controls.append((row, bool(current), selected is not None))
                continue
            truth = next(i for i, source in enumerate(row["positions"]) if source[:2] == row["truth"])
            good_current = baseline_useful(row, current, truth)
            eligible = truth in {item["idx"] for item in row["rows"]}
            positive.append(
                (row, good_current, selected == truth, selected is not None and selected != truth, eligible)
            )
        current_groups = {row["group"] for row, current, _ in controls if current}
        arm_groups = {row["group"] for row, _, arm in controls if arm}
        additional = {row["group"] for row, current, arm, _, _ in positive if arm and not current}
        by_variant = {}
        for variant in ("before", "after"):
            views = [(c, a) for row, c, a in controls if row["variant"] == variant]
            by_variant[variant] = {
                "views": len(views),
                "current_flagged": sum(c for c, _ in views),
                "style_flagged": sum(a for _, a in views),
                "union_flagged": sum(c or a for c, a in views),
            }
        for variant in ("qwen", "granite"):
            views = [(c, a, w, e) for row, c, a, w, e in positive if row["variant"] == variant]
            by_variant[variant] = {
                "views": len(views),
                "current_useful": sum(c for c, _, _, _ in views),
                "style_useful": sum(a for _, a, _, _ in views),
                "union_useful": sum(c or a for c, a, _, _ in views),
                "added_useful": sum(a and not c for c, a, _, _ in views),
                "eligible": sum(e for _, _, _, e in views),
                "style_wrong_locations": sum(w for _, _, w, _ in views),
            }
        results[name] = {
            "by_variant": by_variant,
            "current_genuine_groups": len(current_groups),
            "style_genuine_groups": len(arm_groups),
            "union_genuine_groups": len(current_groups | arm_groups),
            "added_genuine_groups": sorted(arm_groups - current_groups),
            "added_useful_groups": sorted(additional),
            "advance": len(additional) >= 2
            and len(arm_groups - current_groups) <= 1
            and all(by_variant[g]["added_useful"] for g in ("granite", "qwen")),
        }
    return results


def verify(receipt):
    assert receipt["schema"] == SCHEMA
    assert receipt["panel_status"] in ("diagnostic_exposed", "fresh_prospective")
    assert receipt["tokenization_error_excluded"] is True
    admission = receipt["admission"]
    assert admission["detector_runs"] == 0 and admission["generator_counts"] == {"granite": 12, "qwen": 12}
    sources = receipt["sources"]
    assert len({s["group"] for s in sources}) == 12
    assert len({s["publisher"] for s in sources}) == 1
    publisher = sources[0]["publisher"]
    assert (publisher, receipt["panel_status"]) in (
        ("plos-one", "diagnostic_exposed"),
        ("ore", "fresh_prospective"),
    )
    cases = {(r["group"], r["variant"]): r for r in receipt["held"]}
    assert len(cases) == len(receipt["held"]) == 48
    for case in receipt["held"]:
        readings = {row["idx"]: row["contrast"] for row in case["position"]}
        assert len(readings) == len(case["position"])
        for row in case["rows"]:
            assert row["style"] == row["style_plus_position"][:-1]
            assert row["style_plus_position"][-1] == readings[row["idx"]]
    for row in sources:
        assert row["license"] == "CC-BY-4.0"
        if publisher == "ore":
            assert len(row["doi"]) == len(row["dates"]) == 2
            assert row["dates"][0] < row["dates"][1]
            assert "donor" not in row["views"]["after"]
        else:
            assert row["views"]["after"]["kind"] == "constructed_same_work_human_substitution"
        for variant in ("before", "after", "qwen", "granite"):
            case = cases[row["group"], variant]
            if variant in ("before", "after"):
                assert case["truth"] is None and case["text_sha256"] == row["views"][variant]["text_sha256"]
            else:
                assert case["truth"][0] == row["views"]["before"]["focus"][0]
    assert not {row["group"] for row in receipt["development"]} & {row["group"] for row in sources}
    for name, weights in receipt["fit"]["arms"].items():
        verify_fit(weights, name, receipt["development"], receipt["method"]["ridge"])
    outcome = tally(receipt["held"], receipt["fit"]["arms"])
    assert outcome == receipt["outcomes"]
    assert position_only(receipt["held"]) == receipt["position_only"]
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(read(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
