"""Separate standard-library arithmetic review; never imports the v101 scorer."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "text-layout-methods-v101"
REVIEW = BASE / "text-layout-review-v102"
PUBLIC = ROOT / "docs/evaluation-panels/text-layout-review-freeze-v102.json"
NAMES = ("baseline", "canonical_current", "canonical_union", "baseline_plus_canonical_union")
CODE = (
    "backend/scripts/review_text_layout_v102.py",
    "backend/tests/test_text_layout_review_v102.py",
    "backend/scripts/text_layout_review_runtime_v102.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "docs/text-layout-review-v102-protocol.md",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def geometry(first, view):
    source = [(i, c, first["labels"][i]) for i, c in enumerate(first["text"]) if not c.isspace()]
    pairs = [(i, c) for i, c in enumerate(view["text"]) if not c.isspace()]
    require(len(view["text"]) == len(view["source_positions"]) == len(view["labels"]), "Missing positions")
    require(len(source) == len(pairs) and all(x[1] == y[1] for x, y in zip(source, pairs)), "Content changed")
    require(
        [x[1] - x[0] > 1 for x in zip([p[0] for p in source], [p[0] for p in source][1:])]
        == [x[1] - x[0] > 1 for x in zip([p[0] for p in pairs], [p[0] for p in pairs][1:])],
        "Token boundary changed",
    )
    expected = [None] * len(view["text"])
    for (position, _, label), (index, _) in zip(source, pairs, strict=True):
        require(
            first["source_positions"][position] == position and type(label) is int and label in (0, 1),
            "Unknown source truth",
        )
        expected[index] = position
        require(view["labels"][index] == label, "Truth shifted")
    require(expected == view["source_positions"], "Source mapping changed")
    require(
        all(v == -1 for c, v in zip(view["text"], view["labels"], strict=True) if c.isspace()),
        "Whitespace has authorship",
    )
    return {i: label for i, _, label in source}


def observed(view, response, arm, cut):
    if response["status"] != "ok":
        return None
    positive, covered = set(), set()
    for item in response["arms"][arm]:
        a, b, score = item["start"], item["end"], item["score"]
        require(
            type(a) is type(b) is int
            and 0 <= a <= b <= len(view["text"])
            and type(score) in (int, float)
            and math.isfinite(score)
            and 0 <= score <= 1,
            "Invalid proposal",
        )
        for i in range(a, b):
            if not view["text"][i].isspace():
                covered.add(view["source_positions"][i])
                if score >= cut:
                    positive.add(view["source_positions"][i])
    return positive, covered


def joined(*values):
    if None in values:
        return None
    return set().union(*(v[0] for v in values)), set().union(*(v[1] for v in values))


def metrics(labels, value):
    ai = [i for i in sorted(labels) if labels[i] == 1]
    human = [i for i in sorted(labels) if labels[i] == 0]
    require(len(ai) + len(human) == len(labels), "Unknown provenance")
    if value is None:
        return {"status": "unassessed", "ai_characters": len(ai), "human_characters": len(human)}
    flag, cover = value
    require(flag <= cover <= labels.keys(), "Out-of-source mask")
    tp = sum(i in flag for i in ai)
    fp = sum(i in flag for i in human)
    miss = len(ai) - tp
    iou = tp / (len(ai) + fp) if len(ai) + fp else 1.0
    spans = [
        list(values) for truth, values in itertools.groupby(sorted(labels), key=labels.__getitem__) if truth
    ]
    return {
        "status": "ok",
        "ai_characters": len(ai),
        "human_characters": len(human),
        "located_ai_characters": tp,
        "highlighted_human_characters": fp,
        "missed_ai_characters": miss,
        "unscored_ai_characters": sum(i not in cover for i in ai),
        "unscored_human_characters": sum(i not in cover for i in human),
        "character_iou": iou,
        "useful_ai_location": bool(ai) and iou >= 0.3,
        "known_ai_intervals": len(spans),
        "recovered_ai_intervals": sum(sum(i in flag for i in s) * 2 >= len(s) for s in spans),
        "genuine_document_flagged": not ai and bool(flag),
        "ai_document_with_no_ai_located": bool(ai) and tp == 0,
    }


def identity():
    return {
        "schema": "chipmunk.text-layout-review-freeze/102",
        "new_neural_calls": 0,
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "comparison_result_sha256": sha(TRIAL / "evaluation/result.json"),
        "comparison_freeze_sha256": sha(TRIAL / "freeze.json"),
        "scope": "Separate post-result arithmetic/mapping implementation, not new inference, a second model, calibration refit, independent authorship truth or outside human review.",
    }


def freeze():
    REVIEW.mkdir(exist_ok=False)
    body = identity()
    write(REVIEW / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def review():
    require(read(REVIEW / "freeze.json") == identity() == read(PUBLIC), "Review freeze changed")
    # Resource observations are outside this pure-standard-library reviewer.
    lock = read(TRIAL / "freeze.json")
    case_path = BASE / "text-format-audit-v93/inputs.jsonl"
    require(sha(case_path) == lock["input_packet_sha256"], "Input packet changed")
    cases = [json.loads(line) for line in case_path.read_text().splitlines()]
    rows = []
    for i, digest in enumerate(lock["response_journal_sha256"]):
        path = BASE / f"text-format-comparison-v94/batch-{i:02d}.jsonl"
        require(sha(path) == digest, "Response changed")
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    point_path = BASE / "writing-spans-v7/points.json"
    require(sha(point_path) == lock["research_points_sha256"], "Points changed")
    points = read(point_path)
    result = read(TRIAL / "evaluation/result.json")
    path = TRIAL / "evaluation/measurements.jsonl"
    require(sha(path) == result["journal_sha256"], "Measurement journal changed")
    actual = iter(json.loads(line) for line in path.read_text().splitlines())
    totals, groups, references, changed = defaultdict(Counter), defaultdict(set), {}, Counter()
    count = 0
    require(len(cases) == len(rows) == 288 and len({c["group"] for c in cases}) == 32, "Incomplete panel")
    for case, response in zip(cases, rows, strict=True):
        require(case["id"] == response["id"], "Case identity changed")
        first = case["effective"][0]
        normalized = " ".join(first["text"].split())
        canonical = [i for i, v in enumerate(case["effective"]) if v["text"] == normalized]
        require(len(canonical) == 1, "Ambiguous exact canonical input")
        index = canonical[0]
        fixed = case["effective"][index]
        geometry(first, fixed)
        current = observed(fixed, response["effective_responses"][index], "current", points["current"]["cut"])
        interval = observed(
            fixed, response["effective_responses"][index], "interval", points["interval"]["cut"]
        )
        for request in case["requested_views"]:
            i = request["effective_index"]
            view = case["effective"][i]
            labels = geometry(first, view)
            require(" ".join(view["text"].split()) == normalized, "Canonical input changed")
            ordinary = observed(view, response["effective_responses"][i], "current", points["current"]["cut"])
            variants = (ordinary, current, joined(current, interval), joined(ordinary, current, interval))
            reference = metrics(labels, ordinary)
            for name, value in zip(NAMES, variants, strict=True):
                values = metrics(labels, value)
                record = next(actual)
                require(
                    record
                    == {
                        "case_id": case["id"],
                        "view": request["view"],
                        "method": name,
                        "canonical_effective_index": index,
                        "values": values,
                    },
                    "Recomputed observation differs",
                )
                key = (request["view"], name, case["recipe"])
                total = totals[key]
                total["documents"] += 1
                total[values["status"]] += 1
                if values["status"] == "ok":
                    total.update({k: v for k, v in values.items() if k != "status"})
                    if values["genuine_document_flagged"]:
                        groups[key].add(case["group"])
                    if reference["status"] == "ok":
                        total["useful_ai_gained_vs_same_view_baseline"] += (
                            values["useful_ai_location"] and not reference["useful_ai_location"]
                        )
                        total["useful_ai_lost_vs_same_view_baseline"] += (
                            reference["useful_ai_location"] and not values["useful_ai_location"]
                        )
                pair = (case["id"], name)
                if request["view"] == "original":
                    references[pair] = value
                elif value is not None and references[pair] is not None and value[0] != references[pair][0]:
                    changed[(request["view"], name)] += 1
                count += 1
    require(
        next(actual, None) is None and count == result["method_view_document_pairs"] == 5760,
        "Extra or missing results",
    )
    expected = [
        {
            "view": v,
            "method": m,
            "recipe": recipe,
            **values,
            "flagged_genuine_groups": len(groups[(v, m, recipe)]),
        }
        for (v, m, recipe), values in sorted(totals.items())
    ]
    require(expected == result["results"], "Stratified arithmetic differs")
    for row in result["aggregate"]:
        counts, flagged = Counter(), set()
        for (v, m, recipe), values in totals.items():
            if (v, m) == (row["view"], row["method"]):
                counts.update(values)
                flagged.update(groups[(v, m, recipe)])
        require(
            row
            == {
                "view": row["view"],
                "method": row["method"],
                **counts,
                "flagged_genuine_groups": len(flagged),
            },
            "Aggregate differs",
        )
    require(
        all(r["documents"] == changed[(r["view"], r["method"])] for r in result["changed_highlights"]),
        "Stability arithmetic differs",
    )
    require(
        not {"scripts.text_layout_methods_v101", "torch", "numpy", "transformers", "scipy"}.intersection(
            sys.modules
        ),
        "Reviewer imported scorer/runtime",
    )
    body = {
        "schema": "chipmunk.text-layout-review-result/102",
        "status": "passed",
        "reviewed_pairs": count,
        "freeze_sha256": sha(REVIEW / "freeze.json"),
        "comparison_result_sha256": sha(TRIAL / "evaluation/result.json"),
        "new_neural_calls": 0,
        "outside_human_reviewers": 0,
        "scope": identity()["scope"],
    }
    write(REVIEW / "result.json", body)
    write(ROOT / "docs/evaluation-panels/text-layout-review-result-v102.json", body)
    print(json.dumps(body), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "review"))
    freeze() if parser.parse_args().command == "freeze" else review()
