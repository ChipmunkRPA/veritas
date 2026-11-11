"""Evaluate one-sentence context shifts from exact retained text responses."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from app.detectors.text.segment import segment
from scripts import review_text_gohumanize_transfer_v198 as review
from scripts import text_goce_span_v249 as goce
from scripts import text_gohumanize_transfer_v196 as prior
from scripts import text_opai_bench_score_v260 as opai
from scripts.text_short_insert_intake_v203 import fingerprints

ROOT, PANEL = prior.ROOT, prior.PANEL
TRIAL = opai.BASE / "text-context-shift-v266"
PUBLIC = PANEL / "text-context-shift-freeze-v266.json"
POINT = prior.historical.PRIOR
RECEIPTS = {
    "writing-spans-v7-result.json": "eaf7ed4ef8cef0a8bc44a4d6d52e68a96ed1885223df9b03eee6e6e253cce5e1",
    "text-opai-bench-score-complete-v260.json": "52ecd358f9832fc65727ac009079a9812ad7b635563114cad98ee1e7543b183e",
    "text-gohumanize-transfer-complete-v196.json": "eff16a6244be6dcb3ee8b200b4fc55e1b157794372bf63300b5729003a45ea53",
    "text-goce-span-complete-v249.json": "40ec9bfa79968064b27a39f5db5599a2d36283bb33cbd4b7788b8c08097e9790",
}
METHODS = ("current", "context_shift", "current_plus_shift")
RECIPES = {"human", "human-small-matched", "human-large-matched", "human-many-matched"}
RULE = "Minimum available signed logit change when a focus replaces the outer sentence in an adjacent full triple"
require, read, write, sha = prior.require, prior.read, prior.write, prior.sha


def logit(value):
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1, "Invalid response")
    value = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(value / (1 - value))


def shift_scores(raw):
    logits = [logit(value) for value in raw]
    for i in range(len(logits)):
        flanks = []
        if i >= 3:
            flanks.append(logits[i - 1] - logits[i - 2])
        if i + 3 < len(logits):
            flanks.append(logits[i + 1] - logits[i + 2])
        yield min(flanks) if flanks else None


def foci(raw, proposals, proof):
    doc = prior.from_string(raw)
    segments = segment(doc)
    require(
        isinstance(proposals, list)
        and len(segments) == len(proposals) == proof["foci"]
        and proof["actual_input_windows"] >= len(segments)
        and [(row["start"], row["end"]) for row in proposals] == [(s.start, s.end) for s in segments],
        "Exact recorded three-sentence current foci required",
    )
    mapping = prior.correspondence(raw, doc.text)
    support = [frozenset(i for i in mapping[s.start : s.end] if i is not None) for s in segments]
    require(
        support
        and all(support)
        and sum(map(len, support))
        == len(set().union(*support))
        == len({i for i in mapping if i is not None}),
        "Current foci must cover each non-whitespace source position once",
    )
    for row in proposals:
        require(
            type(row["score"]) in (float, int) and math.isfinite(row["score"]) and 0 <= row["score"] <= 1,
            "Exact calibrated current response required",
        )
    return list(zip(support, proposals, shift_scores([row["raw"] for row in proposals]), strict=True))


def select_point(groups):
    require(
        len(groups) == 32
        and all(len(rows) == 4 and any(v is not None for v in rows) for rows in groups.values()),
        "Exact point split with measured support in every group required",
    )
    maxima = [max(value for value in rows if value is not None) for rows in groups.values()]
    cutoff = math.nextafter(max(0.0, sorted(maxima)[-2]), math.inf)
    return cutoff, sum(value >= cutoff for value in maxima)


def point_inputs(public):
    freeze, seal, receipt = (
        read(POINT / file) for file in ("freeze.json", "control-seal.json", "point.json")
    )
    require(
        sha(POINT / "freeze.json") == public["freeze_sha256"]
        and sha(POINT / "control-seal.json") == public["control_seal_sha256"]
        and sha(POINT / "point.json") == public["roles"]["point"]["receipt_sha256"]
        and sha(POINT / "inputs.json") == freeze["input_sha256"] == public["source"]["input_sha256"]
        and sha(POINT / "point.jsonl") == seal["files"]["point.jsonl"] == receipt["journal_sha256"]
        and sha(POINT / "points.json") == seal["files"]["points.json"] == receipt["points_sha256"]
        and receipt["status"] == "complete"
        and receipt["role"] == "point"
        and receipt["documents"] == 128,
        "Exact sealed human point inputs required",
    )
    cases = [row for row in read(POINT / "inputs.json")["cases"] if row["role"] == "point"]
    responses = prior.records(POINT / "point.jsonl")
    require(len(cases) == len(responses) == 128, "All 128 point inputs required")
    groups, current, recipes = defaultdict(list), defaultdict(list), defaultdict(set)
    shift_covered = point_positions = 0
    for case, response in zip(cases, responses, strict=True):
        text, group = case["text"], case["group"]
        require(
            all(case[k] == response[k] for k in ("id", "group", "recipe", "role", "label"))
            and case["label"] == 0
            and response["status"] == "ok"
            and case["text_sha256"] == prior.text_sha(text)
            and case["truth_spans"] == []
            and case["recipe"] not in recipes[group],
            "Ordered genuine-only point responses required",
        )
        recipes[group].add(case["recipe"])
        rows = foci(text, response["arms"]["current"], response["current_native"])
        usable = [(support, value) for support, _, value in rows if value is not None]
        groups[group].append(max((value for _, value in usable), default=None))
        current[group].append(max(row["score"] for _, row, _ in rows))
        shift_covered += sum(len(support) for support, _ in usable)
        point_positions += sum(len(support) for support, _, _ in rows)
    require(
        all(values == RECIPES for values in recipes.values()), "Exact four point controls per group required"
    )
    cutoff, marked = select_point(groups)
    baseline = read(POINT / "points.json")["current"]["cut"]
    require(
        baseline == max(0.001, math.nextafter(sorted(max(rows) for rows in current.values())[-2], math.inf))
        and sum(max(rows) >= baseline for rows in current.values()) == 1
        and marked <= 1,
        "Reproduce the original current operating point and one-group allowance",
    )
    return cases, {
        "source_groups": len(groups),
        "inputs": len(cases),
        "source_positions": point_positions,
        "shift_covered_positions": shift_covered,
        "maximum_flagged_groups": 1,
        "flagged_groups": marked,
        "current_cut": baseline,
        "shift_cut": cutoff,
        "point_journal_sha256": receipt["journal_sha256"],
    }


def identity():
    for name, expected in RECEIPTS.items():
        require(sha(PANEL / name) == expected, "Published predecessor changed")
    cases, point = point_inputs(read(PANEL / "writing-spans-v7-result.json"))
    paths = (
        Path(__file__),
        ROOT / "backend/tests/test_text_context_shift_v266.py",
        ROOT / "docs/text-context-shift-v266-method.md",
    )
    return cases, {
        "schema": "chipmunk.text-context-shift-freeze/266",
        "source_receipts": RECEIPTS,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "rule": RULE,
        "point": point,
        "scope": "Retrospective comparison on saved responses. No evaluation input used to choose the point.",
        "new_neural_calls": 0,
        "serving_changed": False,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier attempts")
    _, lock = identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC)}


def evaluation_inputs():
    opai_rows = opai.verify()
    opai_saved, profiles, calls = opai.observations(opai_rows)
    opai_result = read(PANEL / "text-opai-bench-score-complete-v260.json")
    require(
        profiles == opai_result["resource_profile_sha256"] and calls == opai_result["actual_forwards"],
        "Exact v260 response profiles required",
    )
    articles, _, locked, frozen_sha = review.load_plan()
    article_saved, article_profiles = [], []
    width = review.PAIRS_PER_BATCH * len(review.ARMS)
    for batch in range(review.BATCHES):
        saved, _, profile = review.verify_batch(
            batch,
            articles[batch * width : (batch + 1) * width],
            frozen_sha,
            locked["code_sha256"]["backend/scripts/text_gohumanize_transfer_v196.py"],
        )
        article_saved.extend(saved)
        article_profiles.append(profile)
    article_result = read(PANEL / "text-gohumanize-transfer-complete-v196.json")
    require(article_profiles == article_result["profile_sha256"], "Exact v196 response profiles required")
    controls = goce.verify()
    control_result = read(PANEL / "text-goce-span-complete-v249.json")
    require(goce.readout() == control_result, "Exact v249 coordinate replay required")
    control_saved = prior.records(goce.TRIAL / "extract-00/output/private-responses.jsonl")
    return {
        "opai_v260": (opai_rows, opai_saved),
        "gohumanize_v196": (articles, article_saved),
        "goce_v249": (controls, control_saved),
    }


def source_overlap(cohorts):
    seen, comparisons = {}, 0
    for name, rows in cohorts.items():
        source = set().union(*(fingerprints(row["text"]) for row in rows if row["text"] is not None))
        for previous in seen.values():
            require(
                not (previous & source), "A 12-token fragment crossed the point/evaluation cohort boundary"
            )
            comparisons += 1
        seen[name] = source
    return comparisons


def evidence(rows, current_cut, shift_cut):
    covered, shift_covered, current, shift = set(), set(), set(), set()
    for support, proposal, value in rows:
        covered.update(support)
        if proposal["score"] >= current_cut:
            current.update(support)
        if value is not None:
            shift_covered.update(support)
            if value >= shift_cut:
                shift.update(support)
    return (
        prior.layout.Evidence(frozenset(current), frozenset(covered)),
        prior.layout.Evidence(frozenset(shift), frozenset(shift_covered)) if shift_covered else None,
        prior.layout.Evidence(frozenset(current | shift), frozenset(covered)),
    )


def ranks(rows, labels):
    tallies = [defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])]
    for support, proposal, value in rows:
        if value is not None:
            ai = sum(labels[i] for i in support)
            for tally, score in zip(tallies, (proposal["raw"], value), strict=True):
                tally[score][0] += ai
                tally[score][1] += len(support) - ai
    ai = sum(a for a, _ in tallies[0].values())
    human = sum(h for _, h in tallies[0].values())
    if ai and human:
        result = []
        for tally in tallies:
            lower = wins = 0
            for a, h in (tally[score] for score in sorted(tally)):
                wins += a * (2 * lower + h)
                lower += h
            result.append(wins / (2 * ai * human))
        return result
    return None


def summarize(cases, responses, point):
    require([row["case"] for row in cases] == [row["case"] for row in responses], "Exact case order required")
    fields = (
        "planned",
        "preparation_unavailable",
        "fully_assessed",
        "partially_assessed",
        "unassessed",
        "genuine_inputs",
        "genuine_flagged_inputs",
        "positive_inputs",
        "positive_inputs_with_overlap",
        "useful_locations",
        "ai_characters",
        "human_characters",
        "covered_ai_characters",
        "covered_human_characters",
        "located_ai_characters",
        "highlighted_human_characters",
        "known_ai_intervals",
        "recovered_ai_intervals",
    )
    counts = defaultdict(lambda: Counter(dict.fromkeys(fields, 0)))
    flagged, ranking = defaultdict(set), defaultdict(list)
    for case, response in zip(cases, responses, strict=True):
        masks, labels = (None,) * len(METHODS), {}
        if case["status"] == "constructed":
            labels = prior.labels_for(case)
            require(
                response["status"] in {"measured", "unassessed"}, "Unexpected recorded response disposition"
            )
            if response["status"] == "measured" and response["primary_current"] is not None:
                rows = foci(case["text"], response["primary_current"], response["native_proofs"][0])
                masks = evidence(rows, point["current_cut"], point["shift_cut"])
                pair = ranks(rows, labels)
                if pair is not None:
                    ranking[case["article_group"]].append(pair)
        else:
            require(response["status"] == "preparation_unavailable", "Unavailable input has recorded scores")
        ai = sum(labels.values())
        human = len(labels) - ai
        values = list(labels.values())
        intervals = sum(curr > prev for prev, curr in zip([0, *values], values))
        for method, mask in zip(METHODS, masks, strict=True):
            result = prior.layout.measure(labels, mask) if labels else {}
            for arm in ("all", case["arm"]):
                key = arm, method
                counter = counts[key]
                counter["planned"] += 1
                counter["preparation_unavailable"] += not labels
                counter["unassessed"] += bool(labels) and mask is None
                counter["fully_assessed"] += mask is not None and mask.covered == labels.keys()
                counter["partially_assessed"] += mask is not None and mask.covered != labels.keys()
                counter["genuine_inputs"] += bool(labels) and not ai
                counter["positive_inputs"] += bool(ai)
                counter["ai_characters"] += ai
                counter["human_characters"] += human
                if mask is not None:
                    for field in (
                        "located_ai_characters",
                        "highlighted_human_characters",
                        "recovered_ai_intervals",
                    ):
                        counter[field] += result[field]
                    counter["covered_ai_characters"] += ai - result["unscored_ai_characters"]
                    counter["covered_human_characters"] += human - result["unscored_human_characters"]
                    counter["useful_locations"] += result["useful_ai_location"]
                    counter["positive_inputs_with_overlap"] += bool(result["located_ai_characters"])
                    counter["genuine_flagged_inputs"] += result["genuine_document_flagged"]
                    if result["genuine_document_flagged"]:
                        flagged[key].add(case["article_group"])
                counter["known_ai_intervals"] += intervals
    tables = {}
    for (arm, method), counter in counts.items():
        tables.setdefault(arm, {})[method] = {**counter, "genuine_flagged_groups": len(flagged[arm, method])}
    return {
        "inputs": len(cases),
        "source_groups": len({case["article_group"] for case in cases}),
        "by_arm": tables,
        "common_position_rank": {
            "source_groups": len(ranking),
            "inputs": sum(map(len, ranking.values())),
            "source_balanced_current": mean(mean(r[0] for r in group) for group in ranking.values())
            if ranking
            else None,
            "source_balanced_shift": mean(mean(r[1] for r in group) for group in ranking.values())
            if ranking
            else None,
        },
    }


def readout():
    point_cases, lock = identity()
    require(lock == read(PUBLIC) == read(TRIAL / "freeze.json"), "Exact frozen comparison required")
    cohorts = evaluation_inputs()
    comparisons = source_overlap({"point": point_cases, **{name: pair[0] for name, pair in cohorts.items()}})
    return {
        "schema": "chipmunk.text-context-shift-result/266",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "cohort_pairs_without_shared_12_token_fragment": comparisons,
        "cohorts": {name: summarize(*pair, lock["point"]) for name, pair in cohorts.items()},
        "new_neural_calls": 0,
        "serving_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "readout"))
    result = {"freeze": freeze, "readout": readout}[parser.parse_args().command]()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
