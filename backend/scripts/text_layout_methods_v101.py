"""Post-result development comparison of layout-consistent text evidence.

Only exact retained model inputs/responses are reused. No model is executed,
map fitted, operating point selected or authorship probability combined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
INPUT = BASE / "text-format-audit-v93"
RESPONSE = BASE / "text-format-comparison-v94"
PRIOR = BASE / "writing-spans-v7"
TRIAL = BASE / "text-layout-methods-v101"
PUBLIC = ROOT / "docs/evaluation-panels/text-layout-methods-freeze-v101.json"
CODE = (
    "backend/scripts/text_layout_methods_v101.py",
    "backend/tests/test_text_layout_methods_v101.py",
    "docs/text-layout-methods-v101-protocol.md",
)
METHODS = ("baseline", "canonical_current", "canonical_union", "baseline_plus_canonical_union")


def require(value, detail):
    if not value:
        raise ValueError(detail)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def canonical(view):
    """Collapse whitespace without changing tokens or inventing source positions."""
    text, positions = view["text"], view["source_positions"]
    require(len(text) == len(positions), "Complete source positions required")
    require(
        all((p is None) == c.isspace() for c, p in zip(text, positions, strict=True)),
        "Only whitespace may lack a source position",
    )
    actual = [p for p in positions if p is not None]
    require(
        all(type(p) is int and p >= 0 for p in actual) and all(a < b for a, b in zip(actual, actual[1:])),
        "Ordered original positions required",
    )
    chunks, mapped = [], []
    for match in re.finditer(r"\S+", text):
        if chunks:
            chunks.append(" ")
            mapped.append(None)
        chunks.append(match.group())
        mapped.extend(positions[match.start() : match.end()])
    result = {"text": "".join(chunks), "source_positions": mapped}
    require(bool(actual), "Whitespace alone cannot supply authorship evidence")
    return result


@dataclass(frozen=True)
class Evidence:
    mask: frozenset[int]
    covered: frozenset[int]


def project(view, proposals, cut):
    require(
        type(cut) in (int, float) and math.isfinite(cut) and cut >= 0, "Finite fixed research point required"
    )
    # Validate source correspondence even when no proposal is selected.
    canonical(view)
    mask, coverage = set(), set()
    for proposal in proposals:
        a, b, q = proposal["start"], proposal["end"], proposal["score"]
        require(
            type(a) is type(b) is int
            and 0 <= a <= b <= len(view["text"])
            and type(q) in (int, float)
            and math.isfinite(q)
            and 0 <= q <= 1,
            "Finite source-bounded proposal required",
        )
        support = {p for p in view["source_positions"][a:b] if p is not None}
        coverage.update(support)
        if q >= cut:
            mask.update(support)
    return Evidence(frozenset(mask), frozenset(coverage))


def union(*parts):
    if any(part is None for part in parts):
        return None  # An unavailable component is not an observed negative.
    return Evidence(
        frozenset().union(*(p.mask for p in parts)), frozenset().union(*(p.covered for p in parts))
    )


def compare(ordinary, fixed_current, fixed_interval):
    combined = union(fixed_current, fixed_interval)
    return dict(zip(METHODS, (ordinary, fixed_current, combined, union(ordinary, combined)), strict=True))


def measure(labels, evidence):
    require(
        labels
        and all(type(p) is int and p >= 0 and type(v) is int and v in (0, 1) for p, v in labels.items()),
        "Known source character truth required",
    )
    ai = {p for p, value in labels.items() if value}
    human = set(labels) - ai
    if evidence is None:
        return {"status": "unassessed", "ai_characters": len(ai), "human_characters": len(human)}
    require(evidence.mask <= evidence.covered <= labels.keys(), "Evidence exceeds known source support")
    true = len(evidence.mask & ai)
    false = len(evidence.mask & human)
    missed = len(ai) - true
    iou = true / (true + false + missed) if true + false + missed else 1.0
    intervals, active = [], set()
    for p, value in sorted(labels.items()):
        if value:
            active.add(p)
        elif active:
            intervals.append(active)
            active = set()
    if active:
        intervals.append(active)
    return {
        "status": "ok",
        "ai_characters": len(ai),
        "human_characters": len(human),
        "located_ai_characters": true,
        "highlighted_human_characters": false,
        "missed_ai_characters": missed,
        "unscored_ai_characters": len(ai - evidence.covered),
        "unscored_human_characters": len(human - evidence.covered),
        "character_iou": iou,
        "useful_ai_location": bool(ai) and iou >= 0.3,
        "known_ai_intervals": len(intervals),
        "recovered_ai_intervals": sum(len(evidence.mask & s) / len(s) >= 0.5 for s in intervals),
        "genuine_document_flagged": not ai and bool(evidence.mask),
        "ai_document_with_no_ai_located": bool(ai) and true == 0,
    }


def identity():
    prior = read(RESPONSE / "result.json")
    return {
        "schema": "chipmunk.text-layout-method-freeze/101",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "input_packet_sha256": sha(INPUT / "inputs.jsonl"),
        "response_result_sha256": sha(RESPONSE / "result.json"),
        "response_journal_sha256": prior["journal_sha256"],
        "research_points_sha256": sha(PRIOR / "points.json"),
        "methods": list(METHODS),
        "scope": "Post-result development on 288 consumed controlled documents/32 groups. Previous formatting results are known. No blind confirmation, new inference, calibration, point tuning, combined probability or serving adoption.",
        "new_neural_calls": 0,
    }


def freeze():
    TRIAL.mkdir(exist_ok=False)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def run():
    lock = read(TRIAL / "freeze.json")
    require(lock == identity() == read(PUBLIC), "Published unchanged comparison freeze required")
    out = TRIAL / "evaluation"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    observations = []

    def observe(initial=False):
        row = {
            "elapsed_s": time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "available_ram_bytes": psutil.virtual_memory().available,
            "disk_free_bytes": shutil.disk_usage(out).free,
        }
        observations.append(row)
        minimum = 3 * 1024**3 if initial else 1536 * 1024**2
        require(
            row["elapsed_s"] <= 900
            and row["rss_bytes"] <= 1536 * 1024**2
            and min(row["available_ram_bytes"], row["disk_free_bytes"]) >= minimum,
            "Data-only budget exceeded",
        )

    observe(initial=True)
    cases = [json.loads(line) for line in (INPUT / "inputs.jsonl").read_text().splitlines()]
    require(len(cases) == 288 and len({c["group"] for c in cases}) == 32, "All consumed cases required")
    rows = []
    for i, digest in enumerate(lock["response_journal_sha256"]):
        path = RESPONSE / f"batch-{i:02d}.jsonl"
        require(sha(path) == digest, "Original response journal changed")
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    require([r["id"] for r in rows] == [c["id"] for c in cases], "Ordered complete response panel required")
    points = read(PRIOR / "points.json")
    totals, groups, changed = defaultdict(Counter), defaultdict(set), Counter()
    references = {}
    journal = out / "measurements.jsonl"
    with journal.open("x") as stream:
        for case, row in zip(cases, rows, strict=True):
            original = case["effective"][0]
            labels = {
                p: v
                for p, v in zip(original["source_positions"], original["labels"], strict=True)
                if p is not None
            }
            target = canonical(original)
            matches = [
                i
                for i, v in enumerate(case["effective"])
                if v["text"] == target["text"] and v["source_positions"] == target["source_positions"]
            ]
            require(len(matches) == 1, "Canonical input must match one exact retained native input")
            canonical_index = matches[0]
            fixed = row["effective_responses"][canonical_index]

            def arm(view, response, name):
                return (
                    project(view, response["arms"][name], points[name]["cut"])
                    if response["status"] == "ok"
                    else None
                )

            current = arm(target, fixed, "current")
            interval = arm(target, fixed, "interval")
            for request in case["requested_views"]:
                i = request["effective_index"]
                view, response = case["effective"][i], row["effective_responses"][i]
                require(
                    canonical(view) == target,
                    "Layout-consistent actual input or source correspondence differs",
                )
                methods = compare(arm(view, response, "current"), current, interval)
                reference_measure = measure(labels, methods["baseline"])
                for name, evidence in methods.items():
                    values = measure(labels, evidence)
                    key = (request["view"], name, case["recipe"])
                    total = totals[key]
                    total["documents"] += 1
                    total[values["status"]] += 1
                    if values["status"] == "ok":
                        for k, v in values.items():
                            if k != "status":
                                total[k] += v
                        if values["genuine_document_flagged"]:
                            groups[key].add(case["group"])
                        if reference_measure["status"] == "ok":
                            total["useful_ai_gained_vs_same_view_baseline"] += (
                                values["useful_ai_location"] and not reference_measure["useful_ai_location"]
                            )
                            total["useful_ai_lost_vs_same_view_baseline"] += (
                                reference_measure["useful_ai_location"] and not values["useful_ai_location"]
                            )
                    pair = (case["id"], name)
                    if request["view"] == "original":
                        references[pair] = evidence
                    else:
                        baseline = references[pair]
                        if evidence is not None and baseline is not None and evidence.mask != baseline.mask:
                            changed[(request["view"], name)] += 1
                    stream.write(
                        json.dumps(
                            {
                                "case_id": case["id"],
                                "view": request["view"],
                                "method": name,
                                "canonical_effective_index": canonical_index,
                                "values": values,
                            }
                        )
                        + "\n"
                    )
            observe()
    result_rows = []
    for (view, method, recipe), counts in sorted(totals.items()):
        result_rows.append(
            {
                "view": view,
                "method": method,
                "recipe": recipe,
                **counts,
                "flagged_genuine_groups": len(groups[(view, method, recipe)]),
            }
        )
    aggregate = []
    for view in [r["view"] for r in cases[0]["requested_views"]]:
        for method in METHODS:
            counts, flagged = Counter(), set()
            for (v, m, recipe), values in totals.items():
                if (v, m) == (view, method):
                    counts.update(values)
                    flagged.update(groups[(v, m, recipe)])
            aggregate.append(
                {"view": view, "method": method, **counts, "flagged_genuine_groups": len(flagged)}
            )
    require(
        not {"torch", "transformers", "numpy", "scipy"}.intersection(sys.modules),
        "Non-neural replay required",
    )
    result = {
        "schema": "chipmunk.text-layout-method-result/101",
        "status": "passed",
        "documents": len(cases),
        "known_source_groups": 32,
        "requested_views": 1440,
        "method_view_document_pairs": sum(c["documents"] for c in totals.values()),
        "results": result_rows,
        "aggregate": aggregate,
        "changed_highlights": [
            {"view": v, "method": m, "documents": changed[(v, m)]}
            for v in [r["view"] for r in cases[0]["requested_views"]]
            for m in METHODS
        ],
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "journal_sha256": sha(journal),
        "duration_s": time.monotonic() - started,
        "maximum_observed_self_rss_bytes": max(r["rss_bytes"] for r in observations),
        "minimum_available_ram_bytes": min(r["available_ram_bytes"] for r in observations),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": lock["scope"],
    }
    write(out / "result.json", result)
    write(ROOT / "docs/evaluation-panels/text-layout-methods-result-v101.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"results", "aggregate", "changed_highlights"}}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.command == "freeze" else run()
