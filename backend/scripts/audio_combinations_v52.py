"""Post-result audio combination comparison on all consumed v13 development outcomes.

This is exact decision-set reanalysis, not fresh neural inference or a fitted
combined confidence. The combination functions receive no truth or condition.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.evaluation_tradeoffs_v44 import observed_frontier

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "backend/data/independent-evaluation/audio-combinations-v52"
PACKET = ROOT / "docs/evaluation-panels/audio-v13-development-review.json.gz"
PACKET_SHA256 = "84848472fef312c5fb5fd2ceddacd5fbe217caebadd09aacd85f28f5ee21cac3"
PUBLIC = ROOT / "docs/evaluation-panels/audio-combinations-v52-freeze.json"
DECISIONS = frozenset({"ai", "original", "inconclusive", "unassessed"})
METHODS = {
    "current": (("current",), "single"),
    "new_features_only": (("standalone",), "single"),
    "features_plus_current": (("complement",), "single"),
    "pair_conflict_abstain": (("current", "complement"), "available_no_conflict"),
    "current_then_complement": (("current", "complement"), "priority"),
    "complement_then_current": (("complement", "current"), "priority"),
    "pair_any_ai": (("current", "complement"), "any_ai"),
    "pair_agreement_only": (("current", "complement"), "agreement_only"),
    "three_conflict_abstain": (("current", "complement", "standalone"), "available_no_conflict"),
    "three_any_ai": (("current", "complement", "standalone"), "any_ai"),
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def need(value, message):
    if not value:
        raise ValueError(message)


def combine(choices: tuple[str, ...], policy: str) -> str:
    need(bool(choices) and all(choice in DECISIONS for choice in choices), "Known complete decision states required")
    need(policy in {"single", "available_no_conflict", "priority", "any_ai", "agreement_only"}, "Explicit fixed combination required")
    need((len(choices) == 1) == (policy == "single"), "One component needs single policy; combinations need multiple components")
    if "unassessed" in choices:
        return "unassessed"
    if policy == "single":
        return choices[0]
    stated = set(choices) - {"inconclusive"}
    if policy == "available_no_conflict":
        return next(iter(stated)) if len(stated) == 1 else "inconclusive"
    if policy == "priority":
        return next((choice for choice in choices if choice != "inconclusive"), "inconclusive")
    if policy == "any_ai":
        return "ai" if "ai" in stated else "original" if "original" in stated else "inconclusive"
    return choices[0] if len(set(choices)) == 1 else "inconclusive"


def summarize(rows: list[dict], predictions: list[str]) -> dict:
    need(len(rows) == len(predictions) and bool(rows), "All planned inputs must remain")
    need(all(row["label"] in {0, 1} and decision in DECISIONS for row, decision in zip(rows, predictions, strict=True)),
         "Known truth and explicit outcomes required")
    pairs = list(zip(rows, predictions, strict=True))
    counts = {
        "inputs": len(rows), "source_files": len({row["base_id"] for row in rows}),
        "connected_groups": len({row["group"] for row in rows}),
        "ai_inputs": sum(row["label"] == 1 for row in rows), "genuine_inputs": sum(row["label"] == 0 for row in rows),
        "ai_found": sum(row["label"] == 1 and decision == "ai" for row, decision in pairs),
        "genuine_correct": sum(row["label"] == 0 and decision == "original" for row, decision in pairs),
        "false_ai": sum(row["label"] == 0 and decision == "ai" for row, decision in pairs),
        "false_original": sum(row["label"] == 1 and decision == "original" for row, decision in pairs),
        "inconclusive_assessed": predictions.count("inconclusive"), "unassessed": predictions.count("unassessed"),
        "genuine_groups_flagged": len({row["group"] for row, decision in pairs if row["label"] == 0 and decision == "ai"}),
    }
    counts["without_direction"] = counts["inconclusive_assessed"] + counts["unassessed"]
    counts["ai_without_ai_conclusion"] = counts["ai_inputs"] - counts["ai_found"]
    need(sum(counts[k] for k in ("ai_found", "genuine_correct", "false_ai", "false_original", "without_direction")) == len(rows),
         "Every outcome must have one operational category")
    return counts


def changes(rows, current, candidate):
    def truth(row):
        return "ai" if row["label"] else "original"

    triples = list(zip(rows, current, candidate, strict=True))
    return {
        "new_correct": sum(old != truth(row) and new == truth(row) for row, old, new in triples),
        "lost_correct": sum(old == truth(row) and new != truth(row) for row, old, new in triples),
        "new_ai_hits": sum(row["label"] == 1 and old != "ai" and new == "ai" for row, old, new in triples),
        "lost_ai_hits": sum(row["label"] == 1 and old == "ai" and new != "ai" for row, old, new in triples),
        "new_false_ai": sum(row["label"] == 0 and old != "ai" and new == "ai" for row, old, new in triples),
        "removed_false_ai": sum(row["label"] == 0 and old == "ai" and new != "ai" for row, old, new in triples),
        "new_false_original": sum(row["label"] == 1 and old != "original" and new == "original" for row, old, new in triples),
        "removed_false_original": sum(row["label"] == 1 and old == "original" and new != "original" for row, old, new in triples),
    }


def prepare():
    need(sha(PACKET) == PACKET_SHA256, "Original complete public development packet changed")
    paths = [Path(__file__), Path(__file__).with_name("audio_combinations_runtime_v52.py"),
             ROOT / "backend/scripts/evaluation_tradeoffs_v44.py", ROOT / "backend/scripts/local_probe_limits_v24.py",
             ROOT / "backend/scripts/data_resource_guard_v21.py", ROOT / "backend/scripts/image_regional_continuation_runtime_v41.py",
             ROOT / "backend/tests/test_audio_combinations_v52.py", ROOT / "docs/audio-combinations-v52-protocol.md"]
    frozen = {"schema": "chipmunk.audio-combination-reanalysis/52", "input_sha256": PACKET_SHA256,
              "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}, "methods": METHODS,
              "inputs": 1449, "source_files": 161, "connected_groups": 106, "fresh_validation": False,
              "new_model_calls_permitted": 0, "refit_permitted": False, "combined_confidence_admitted": False,
              "protected_audio_sha256": sha(ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json")}
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"methods": len(METHODS), "inputs": 1449, "new_model_calls": 0}), flush=True)


def admission():
    frozen, start = read(TRIAL / "freeze.json"), read(TRIAL / "execution-start.json")
    need(sha(PUBLIC) == sha(TRIAL / "freeze.json") == start["freeze_sha256"]
         and start["remote_verified_before_reanalysis"] is True, "Published exact reanalysis freeze required")
    need(all(sha(ROOT / path) == value for path, value in frozen["code_sha256"].items())
         and sha(PACKET) == PACKET_SHA256, "Frozen comparison or complete source packet changed")
    need(json.loads(json.dumps(METHODS)) == frozen["methods"], "Method pool changed")
    path = TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json")
    profile = read(path)
    need(sha(path) == start["preparation_profile_sha256"] and profile["child_exit_code"] == 0
         and profile["guard"] is None and profile["sample_errors"] == 0
         and not profile["resource_limit_observations"], "Bounded preparation must pass")
    return frozen


def evaluate():
    admission()
    body = json.loads(gzip.decompress(PACKET.read_bytes()))
    rows = body["outcomes"]
    need(len(rows) == len({r["id"] for r in rows}) == 1449 and {r["role"] for r in rows} == {"development"}, "Complete original development role required")
    need(len({r["base_id"] for r in rows}) == 161 and len({r["group"] for r in rows}) == 106, "Original connected source units changed")
    need(Counter(r["label"] for r in rows) == {0: 693, 1: 756}, "Every original truth case must remain")
    predictions = {name: [] for name in METHODS}
    conflicts = Counter()
    for row in rows:
        components = {"current": row["original_decision"], **{name: value["decision"] for name, value in row["candidates"].items()}}
        need(set(components) == {"current", "standalone", "complement"}, "Original three assessed components required")
        need(row["status"] in {"ok", "not_applicable"} and all(value in DECISIONS - {"unassessed"} for value in components.values()),
             "Missing or failed extraction cannot be dropped or coerced")
        if row["status"] != "ok":
            need(set(components.values()) == {"inconclusive"}, "An unassessed input cannot supply a directional call")
            components = {name: "unassessed" for name in components}
        for name, (members, policy) in METHODS.items():
            selected = tuple(components[member] for member in members)
            conflicts[name] += {"ai", "original"}.issubset(selected)
            predictions[name].append(combine(selected, policy))
    totals = {name: summarize(rows, values) for name, values in predictions.items()}
    expected = {"current": (648, 356, 58, 0, 387), "new_features_only": (30, 28, 0, 4, 1387),
                "features_plus_current": (522, 437, 4, 0, 486)}
    for name, values in expected.items():
        need(tuple(totals[name][key] for key in ("ai_found", "genuine_correct", "false_ai", "false_original", "without_direction")) == values,
             "Historical standalone operational counts did not reproduce exactly")
    strata = {}
    for field in ("condition", "generator"):
        strata[field] = {}
        for value in sorted({r[field] for r in rows}):
            indices = [n for n, r in enumerate(rows) if r[field] == value]
            part = [rows[n] for n in indices]
            strata[field][value] = {name: summarize(part, [values[n] for n in indices]) for name, values in predictions.items()}
    frontier = observed_frontier({name: {key: row[key] for key in ("ai_found", "genuine_correct", "false_ai", "false_original", "without_direction")}
                                  for name, row in totals.items()}, maximize=("ai_found", "genuine_correct"),
                                 minimize=("false_ai", "false_original", "without_direction"))
    result = {"schema": "chipmunk.audio-combination-result/52", "status": "complete", "input_sha256": PACKET_SHA256,
              "scope": "Post-result decision-set reanalysis of consumed four-second development windows; not fresh accuracy, whole-recording validation or combined confidence calibration.",
              "methods": METHODS, "totals": totals, "changes_vs_current": {name: changes(rows, predictions["current"], values) for name, values in predictions.items()},
              "directional_conflicts": dict(conflicts), "strata": strata, "observed_frontier": frontier,
              "historical_counts_reproduced": True, "unassessed_preserved": 29, "new_model_calls": 0,
              "new_parameters_fitted": 0, "new_combined_probability": None, "defaults_changed": False,
              "selection_rule": "Compare gains, losses, false calls and coverage together; no automatic rejection for one added false alarm or lost hit.",
              "commercial_terms": "Reuses the already reviewed frozen component responses only; no new weights, source audio or code with new license terms.",
              "fresh_confirmation_required": True, "workflow_runs": 0, "outreach_sent": False, "deployment_done": False}
    write(TRIAL / "result.json", result)
    print(json.dumps({"methods": len(METHODS), "inputs": 1449, "historical_counts_reproduced": True, "new_model_calls": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate}[args.command]()
