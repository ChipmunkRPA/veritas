"""Independent standard-library replay of recorded formatting comparisons.

No scorer, extraction, detector, calibration or scientific-library imports.
This does not independently rerun inference or adjudicate source authorship.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
INPUT = BASE / "text-format-audit-v93"
TRIAL = BASE / "text-format-comparison-v94"
PRIOR = BASE / "writing-spans-v7"
REVIEW = BASE / "text-format-independent-review-v99"
PUBLIC = ROOT / "docs/evaluation-panels/text-format-independent-review-freeze-v99.json"
CODE = (
    "backend/scripts/review_text_format_v99.py",
    "backend/scripts/text_review_runtime_v99.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/tests/test_text_format_review_v99.py",
    "docs/text-format-independent-review-v99-protocol.md",
)
ARMS = ("current", "interval")
METHODS = (*ARMS, "union", "intersection")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def identity():
    return {
        "schema": "chipmunk.text-format-independent-review-freeze/99",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "response_freeze_sha256": sha(TRIAL / "freeze.json"),
        "input_packet_sha256": sha(INPUT / "inputs.jsonl"),
        "original_response_journal_sha256": sha(PRIOR / "diagnostic.jsonl"),
        "fixed_research_points_sha256": sha(PRIOR / "points.json"),
        "arithmetic": "Integer counts/positions and reference responses exact; floating summaries abs1e-9/rel1e-12. Missing is never zero. Every requested view retained.",
        "scope": "Separate local recorded-response/mapping/arithmetic review, not new inference, calibration reproduction, independent source truth or outside human review.",
        "new_neural_calls": 0,
    }


def freeze():
    REVIEW.mkdir(exist_ok=False)
    body = identity()
    write(REVIEW / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def mapping(first, view):
    """Reconstruct character correspondence from the two actual text strings."""

    def nonwhite(row):
        require(
            len(row["text"]) == len(row["source_positions"]) == len(row["labels"]),
            "Complete character arrays required",
        )
        pairs = []
        for index, char in enumerate(row["text"]):
            position, label = row["source_positions"][index], row["labels"][index]
            if char.isspace():
                require(
                    position is None and label == -1, "Whitespace is not an authorship-labelled character"
                )
            else:
                require(
                    type(position) is int and position >= 0 and type(label) is int and label in (0, 1),
                    "Source character and binary provenance required",
                )
                pairs.append((index, char, position, label))
        require(all(a[2] < b[2] for a, b in zip(pairs, pairs[1:])), "Source positions must increase")
        return pairs

    original, altered = nonwhite(first), nonwhite(view)
    require(
        [(c, p, y) for _, c, p, y in original] == [(c, p, y) for _, c, p, y in altered],
        "Original non-whitespace content, positions or labels changed",
    )
    require(
        [b[0] > a[0] + 1 for a, b in zip(original, original[1:])]
        == [b[0] > a[0] + 1 for a, b in zip(altered, altered[1:])],
        "Existing token boundaries changed",
    )
    require(
        hashlib.sha256(view["text"].encode("utf-8", "surrogatepass")).hexdigest() == view["text_sha256"],
        "Effective text bytes changed",
    )
    return {p: label for _, _, p, label in original}


def project(view, proposals, cut):
    require(type(cut) in (float, int) and math.isfinite(cut) and cut >= 0, "Finite fixed point required")
    scores = defaultdict(list)
    for row in proposals:
        a, b, score = row["start"], row["end"], row["score"]
        require(
            type(a) is int and type(b) is int and 0 <= a <= b <= len(view["text"]),
            "Source-bounded proposal required",
        )
        require(
            type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
            "Finite response required",
        )
        for index in range(a, b):
            position = view["source_positions"][index]
            if position is not None:
                scores[position].append(score)
    strongest = {p: max(values) for p, values in scores.items()}
    return {p for p, score in strongest.items() if score >= cut}, strongest


def methods(view, response, points):
    if response.get("status") == "unassessed":
        require(
            response.get("arms") is None
            and response.get("reason") == "unsupported_native_character_coverage",
            "Unexplained missing response",
        )
        return {method: None for method in METHODS}
    require(response.get("status") == "ok", "Unknown response state")
    arms = {arm: project(view, response["arms"][arm], points[arm]["cut"]) for arm in ARMS}
    a, b = (arms[arm][0] for arm in ARMS)
    return {**arms, "union": (a | b, {}), "intersection": (a & b, {})}


def compare(first, second, labels):
    if first is None or second is None:
        return None
    a, x = first
    b, y = second
    require(set(labels).issuperset(a | b | set(x) | set(y)), "A response covers unknown original characters")
    shared = set(x).intersection(y)
    errors = [abs(x[p] - y[p]) for p in sorted(shared)]
    counts = {
        "highlight_changed": int(a != b),
        "original_highlighted": len(a),
        "view_highlighted": len(b),
        "added_ai_characters": sum(labels[p] for p in b - a),
        "lost_ai_characters": sum(labels[p] for p in a - b),
        "added_human_characters": sum(1 - labels[p] for p in b - a),
        "removed_human_characters": sum(1 - labels[p] for p in a - b),
        "shared_scored_characters": len(shared),
        "only_original_scored_characters": len(set(x) - set(y)),
        "only_view_scored_characters": len(set(y) - set(x)),
        "absolute_score_change_sum": sum(errors),
    }
    return counts, (len(a & b) / len(a | b) if a | b else 1.0), (max(errors) if errors else None)


def equal(expected, actual):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Aggregate fields differ")
        for key in expected:
            equal(expected[key], actual[key])
    elif type(expected) is float:
        require(
            type(actual) in (int, float) and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-9),
            "Floating aggregate differs",
        )
    else:
        require(type(actual) is type(expected) and actual == expected, "Exact aggregate differs")


def review():
    require(
        read(PUBLIC) == read(REVIEW / "freeze.json") == identity(),
        "Published independent review freeze changed",
    )
    freeze = read(TRIAL / "freeze.json")
    require(
        freeze == read(ROOT / "docs/evaluation-panels/text-format-response-freeze-v94.json"),
        "Response freeze differs",
    )
    require(
        all(sha(ROOT / p) == h for p, h in freeze["code_sha256"].items()),
        "Frozen response implementation differs",
    )
    cases = [json.loads(line) for line in (INPUT / "inputs.jsonl").read_text().splitlines()]
    prior = {
        row["id"]: row
        for row in (json.loads(line) for line in (PRIOR / "diagnostic.jsonl").read_text().splitlines())
    }
    points = read(PRIOR / "points.json")
    expected = read(TRIAL / "result.json")
    counters, groups = defaultdict(Counter), defaultdict(set)
    overlaps, maxima = defaultdict(list), defaultdict(list)
    effective, pairs, forwards = Counter(), 0, Counter()
    profiles, journals = [], []
    for batch in range(8):
        path = TRIAL / f"batch-{batch:02d}.jsonl"
        profile_path = TRIAL / f"batch-{batch:02d}-profile/profile.json"
        profile = read(profile_path)
        complete = read(TRIAL / f"batch-{batch:02d}-complete.json")
        require(
            profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
            "Unsuccessful scientific batch",
        )
        require(
            sha(path) == complete["journal_sha256"]
            and complete["freeze_sha256"] == sha(TRIAL / "freeze.json"),
            "Response journal commitment differs",
        )
        require(
            sha(profile_path.parent / "resources.jsonl") == profile["resource_journal_sha256"],
            "Resource journal differs",
        )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        selected = cases[batch * 36 : (batch + 1) * 36]
        require(
            len(rows) == complete["documents"] == 36
            and [r["id"] for r in rows] == [r["id"] for r in selected],
            "Complete fixed case order required",
        )
        batch_counts = Counter()
        for case, row in zip(selected, rows, strict=True):
            require(
                case["group"] == row["group"] and len(row["effective_responses"]) == len(case["effective"]),
                "Source group or effective views differ",
            )
            first = case["effective"][0]
            labels = mapping(first, first)
            original = row["effective_responses"][0]
            old = prior[case["id"]]
            require(
                original["status"] == "ok" and original["arms"] == {arm: old["arms"][arm] for arm in ARMS},
                "Original proposal parity differs",
            )
            require(
                all(
                    original["native"][k] == old["native"][k]
                    for k in ("complete_features_sha256", "native_offsets_sha256")
                )
                and original["current_native"]["actual_input_ids_sha256"]
                == old["current_native"]["actual_input_ids_sha256"],
                "Original feature, offset or input parity differs",
            )
            original_methods = methods(first, original, points)
            for view, response in zip(case["effective"], row["effective_responses"], strict=True):
                mapping(first, view)
                batch_counts[response["status"]] += 1
            for request in case["requested_views"]:
                index = request["effective_index"]
                require(
                    type(index) is int and 0 <= index < len(case["effective"]),
                    "Invalid effective input index",
                )
                values = methods(case["effective"][index], row["effective_responses"][index], points)
                for method in METHODS:
                    key = request["view"], method
                    counters[key]["documents"] += 1
                    pairs += 1
                    delta = compare(original_methods[method], values[method], labels)
                    if delta is None:
                        counters[key]["unassessed_pairs"] += 1
                        continue
                    counts, overlap, maximum = delta
                    counters[key]["assessed_pairs"] += 1
                    counters[key].update(counts)
                    if counts["highlight_changed"]:
                        groups[key].add(case["group"])
                    overlaps[key].append(overlap)
                    if maximum is not None:
                        maxima[key].append(maximum)
        require(dict(batch_counts) == complete["counts"], "Batch coverage counts differ")
        effective.update(batch_counts)
        forwards.update(complete["actual_pretrained_forwards"])
        profiles.append(sha(profile_path))
        journals.append(sha(path))
    result = {}
    for (view, method), counts in counters.items():
        key = view, method
        n = counts["shared_scored_characters"]
        result.setdefault(view, {})[method] = {
            **counts,
            "source_groups_with_changed_highlight": len(groups[key]),
            "mean_highlight_jaccard": sum(overlaps[key]) / len(overlaps[key]) if overlaps[key] else None,
            "mean_absolute_score_change_on_shared_coverage": counts["absolute_score_change_sum"] / n
            if n
            else None,
            "maximum_score_change": max(maxima[key]) if maxima[key] else None,
        }
    equal(expected["results"], result)
    equal(expected["actual_pretrained_forwards"], dict(forwards))
    require(
        len(cases) == 288
        and len({c["group"] for c in cases}) == 32
        and pairs == 5760
        and sum(effective.values()) == 800
        and effective["ok"] == expected["effective_texts_scored"]
        and effective["unassessed"] == expected["effective_texts_unassessed"],
        "Complete study denominators required",
    )
    require(
        expected["profile_sha256"] == profiles and expected["journal_sha256"] == journals,
        "Aggregate source hashes differ",
    )
    body = {
        "schema": "chipmunk.text-format-independent-review/99",
        "status": "passed",
        "documents": 288,
        "known_source_groups": 32,
        "requested_views": 1440,
        "method_view_pairs": pairs,
        "effective_statuses": dict(effective),
        "new_neural_calls": 0,
        "response_result_sha256": sha(TRIAL / "result.json"),
        "review_freeze_sha256": sha(REVIEW / "freeze.json"),
        "profile_sha256": profiles,
        "journal_sha256": journals,
        "scope": read(PUBLIC)["scope"],
    }
    write(REVIEW / "result.json", body)
    print(json.dumps(body), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "review"))
    freeze() if parser.parse_args().command == "freeze" else review()
