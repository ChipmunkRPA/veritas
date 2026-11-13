# SPDX-License-Identifier: Apache-2.0
"""Compare independently set positive-only operating points for existing voice scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import audio_common_prior_context_v87 as common
from scripts import audio_score_fusion_v83 as fusion
from scripts.audio_combinations_v52 import changes, need, sha, summarize, write

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-combined-confidence-v84"
PUBLIC = ROOT / "docs/evaluation-panels"
COMMON = BASE / "audio-common-reference-study-v88"
SCHEMA = "chipmunk.audio-selective-fusion-transfer/1"
ARMS = ("raw", "global", "context", "common_reference", "common_agreement")
POOLED = {"raw", "global"}
CONTEXTS = ("low", "high")
METHOD = {
    "candidate_scope": "One four-second 16 kHz mono assessment whose incumbent is assessed and inconclusive.",
    "candidates": list(ARMS),
    "routing": "Use the original fitting-only measured spectral flatness split. True background, family and labels are evaluation only.",
    "calibration_point": "Strictly exceed the third-highest genuine connected-group maximum on the original confidence-calibration role.",
    "minimum_genuine_groups": 8,
    "merge": "Preserve all current directions and unassessed results. Add only AI flags when the named arm clears its own point.",
    "agreement": "Common-reference and raw must each clear their own point. This is corroboration between correlated responses.",
    "advancement": {"new_ai_sources": 4, "new_ai_families": 2, "new_genuine_groups_max": 2, "separate_fresh_confirmation_required": True},
}


def read(path):
    return json.loads(path.read_bytes())


def setup(trial, common_trial):
    freeze = PUBLIC / "audio-combined-confidence-v84-freeze.json"
    training = PUBLIC / "audio-combined-confidence-v84-training.json"
    selection = PUBLIC / "audio-combined-confidence-v84-selection.json"
    final = PUBLIC / "audio-combined-confidence-v84-result.json"
    common_freeze = PUBLIC / "audio-common-reference-v88-freeze.json"
    common_training = common_trial / "training-summary.json"
    parent_fit, second_fit = trial / "trained.json", common_trial / "trained.json"
    need(sha(trial / "freeze.json") == sha(freeze) == read(selection)["freeze_sha256"] == read(final)["freeze_sha256"]
         and sha(selection) == read(final)["selection_sha256"], "Historical freeze or selection differs")
    training_doc = read(training)
    need(sha(parent_fit) == training_doc["trained_sha256"] == read(selection)["trained_sha256"], "Original calibration bytes differ")
    need(sha(common_trial / "freeze.json") == sha(common_freeze)
         and read(common_freeze)["primary_trained_sha256"] == sha(parent_fit)
         and read(common_training)["trained_sha256"] == sha(second_fit), "Common-reference calibration bytes differ")
    commitments = {path.name: sha(path) for path in (freeze, training, selection, final, common_freeze)}
    commitments |= {"historical_global_and_context": sha(parent_fit), "historical_common_reference": sha(second_fit),
                    "historical_common_training_summary": sha(common_training), "runner": sha(Path(__file__)),
                    "fixed_global_predictor": sha(Path(fusion.__file__)), "fixed_common_predictor": sha(Path(common.__file__))}
    stages = {"confidence_cal": training_doc["calibration_stages"], "selection": read(selection)["source_stages"],
              "confirmation": read(final)["source_stages"]}
    return read(parent_fit), read(second_fit), stages, commitments


def source_role(trial, stages, role):
    rows = []
    need(len(stages) == 12, "All original chunks are required")
    for index, receipt in sorted(stages.items(), key=lambda item: int(item[0])):
        path = trial / f"chunk-{int(index):02d}.jsonl"
        need(receipt["completed"] == 87 and receipt["compact_sha256"] == sha(path), "Original compact chunk differs")
        chunk = [json.loads(line) for line in path.read_bytes().splitlines()]
        need(len(chunk) == 87 and all(row["role"] == role for row in chunk), "Original role or chunk length differs")
        rows.extend(chunk)
    need(len(rows) == len({row["id"] for row in rows}) == 1044, "Original unique role roster required")
    return rows


def predictors(row, fits, second):
    eligible = row["status"] == "ok" and row["decisions"]["current"] == "inconclusive"
    context = fusion.branch(row, fits["flatness_context"])
    if not eligible:
        return context, {}
    return context, {"raw": row["raw"], "global": fusion.predict(row, fits["global"])["probability_ai"],
                     "context": fusion.predict(row, fits["flatness_context"])["probability_ai"],
                     "common_reference": common.predict(row, second)["probability_ai"]}


def calibrate(rows, fits, second):
    points = {arm: {} for arm in ARMS[:-1]}
    groups = {(arm, ctx): {} for arm in points for ctx in (("all",) if arm in POOLED else CONTEXTS)}
    for row in rows:
        if row["label"] != 0:
            continue
        context, scores = predictors(row, fits, second)
        for arm, score in scores.items():
            if score is None:
                continue
            bucket = groups[arm, "all" if arm in POOLED else context]
            bucket[row["group"]] = max(score, bucket.get(row["group"], -1.0))
    for (arm, context), scores in groups.items():
        maximums = sorted(scores.values(), reverse=True)
        point = maximums[2] if len(maximums) >= METHOD["minimum_genuine_groups"] else None
        points[arm][context] = {"point": point, "genuine_group_maxima": maximums}
    return points


def score(rows, fits, second, points):
    source_alias = {key: i for i, key in enumerate(sorted({row["base_id"] for row in rows}), 1)}
    group_alias = {key: i for i, key in enumerate(sorted({row["group"] for row in rows}), 1)}
    ledger = []
    for row in rows:
        incumbent = row["decisions"]["current"]
        context, scores = predictors(row, fits, second)
        positives = {}
        for arm in ARMS[:-1]:
            point = points[arm]["all" if arm in POOLED else context]["point"]
            value = scores.get(arm)
            positives[arm] = point is not None and value is not None and value > point
        positives["common_agreement"] = positives["raw"] and positives["common_reference"]
        ledger.append({"base_id": source_alias[row["base_id"]], "group": group_alias[row["group"]], "label": row["label"],
                       "kind": row["kind"], "condition": row["condition"], "context": context, "current": incumbent,
                       "scores": scores, "directions": {arm: "ai" if found else incumbent for arm, found in positives.items()}})
    return ledger


def result(ledger):
    current = [row["current"] for row in ledger]
    groups = sorted({row["group"] for row in ledger})
    outputs = {}
    for arm in ARMS:
        prediction = [row["directions"][arm] for row in ledger]
        newly_flagged = [row for row, finding in zip(ledger, prediction, strict=True) if finding == "ai" and row["current"] != "ai"]
        new_ai = [row for row in newly_flagged if row["label"] == 1]
        new_genuine = [row for row in newly_flagged if row["label"] == 0]
        prior_false = {row["group"] for row in ledger if row["label"] == 0 and row["current"] == "ai"}
        families, sources = sorted({row["kind"] for row in new_ai}), {row["base_id"] for row in new_ai}
        new_genuine_groups = len({row["group"] for row in new_genuine} - prior_false)
        summary = {"total": summarize(ledger, prediction), "changes": changes(ledger, current, prediction),
                   "new_ai_sources": len(sources), "new_ai_families": families, "new_genuine_groups": new_genuine_groups,
                   "new_genuine_views": len(new_genuine), "new_ai_views": len(new_ai), "by_condition": {}, "by_context": {}}
        for target, field in (("by_condition", "condition"), ("by_context", "context")):
            for name in sorted({row[field] for row in ledger}):
                subset = [row for row in ledger if row[field] == name]
                summary[target][name] = {"current": summarize(subset, [row["current"] for row in subset]),
                                         "candidate": summarize(subset, [row["directions"][arm] for row in subset])}
        per_group = [sum(1 if row["label"] else -1 for row in newly_flagged if row["group"] == group) for group in groups]
        summary["paired_net_gain_without_each_group"] = [sum(per_group) - max(per_group), sum(per_group) - min(per_group)]
        gate = METHOD["advancement"]
        summary["qualifies_for_fresh_test"] = (len(sources) >= gate["new_ai_sources"] and len(families) >= gate["new_ai_families"]
                                                and new_genuine_groups <= gate["new_genuine_groups_max"])
        outputs[arm] = summary
    return {"current": summarize(ledger, current), "arms": outputs}


def run(trial, common_trial, output, mode):
    fits, second, stages, commitments = setup(trial, common_trial)
    lock = {"schema": SCHEMA, "method": METHOD, "commitments": commitments, "source_stages": stages,
            "context_cutoff": fits["flatness_context"]["routing_cutoff"]}
    if mode == "freeze":
        output.mkdir(mode=0o700)
        write(output / "freeze.json", lock)
        return
    need(read(output / "freeze.json") == lock, "Study parameters or code changed after freeze")
    points = calibrate(source_role(trial, stages["confidence_cal"], "confidence_cal"), fits, second)
    if mode == "fit":
        write(output / "points.json", points)
        return
    need(read(output / "points.json") == points, "Points differ from independent genuine calibration")
    roles = {}
    for role in ("selection", "confirmation"):
        records = score(source_role(trial, stages[role], role), fits, second, points)
        roles[role] = {"result": result(records), "cases": records}
    receipt = {"schema": SCHEMA, "freeze_sha256": sha(output / "freeze.json"), "method": METHOD,
               "model_free_replay": True, "scope": "Post hoc policy on two previously scored, source-separated roles. No new inference or serving change.",
               "points": points, "roles": roles}
    if mode == "report":
        write(output / "receipt.json", receipt)
    else:
        need(read(output / "receipt.json") == receipt, "Scoring or held results differ")
    for role, review in roles.items():
        print(role, {arm: {key: value[key] for key in ("new_ai_sources", "new_genuine_groups", "new_ai_views", "new_genuine_views",
                                                       "qualifies_for_fresh_test")} for arm, value in review["result"]["arms"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "fit", "report", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--trial", type=Path, default=TRIAL)
    parser.add_argument("--common", type=Path, default=COMMON)
    args = parser.parse_args()
    run(args.trial, args.common, args.output, args.mode)
