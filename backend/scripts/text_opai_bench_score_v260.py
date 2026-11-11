"""Compare fixed passage methods on source-verified OpAI-Bench report edits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

from scripts import text_gohumanize_transfer_v196 as prior

ROOT, PANEL = prior.ROOT, prior.PANEL
BASE = prior.TRIAL.parent
TRIAL = BASE / "text-opai-bench-score-v260"
CASES = BASE / "text-opai-bench-intake-v257/private-cases.jsonl"
INTAKE = PANEL / "text-opai-bench-intake-complete-v257.json"
PUBLIC = PANEL / "text-opai-bench-score-freeze-v260.json"
MODULE = "scripts.text_opai_bench_score_v260"
METHOD_DOC = ROOT / "docs/text-opai-bench-score-v260-method.md"
GROUPS_PER_BATCH = 3
ROLES = (
    ("shared_human", "human"),
    ("qwen3-8b", "polish_ai"),
    ("qwen3-8b", "compress_ai"),
    ("gpt-5.4-nano", "polish_ai"),
    ("gpt-5.4-nano", "compress_ai"),
)
WIDTH = len(ROLES) * GROUPS_PER_BATCH
read, write, sha, require, scoped = prior.read, prior.write, prior.sha, prior.require, prior.scoped


def inputs():
    packet = read(INTAKE)
    require(
        packet["schema"] == "chipmunk.text-opai-bench-intake/257"
        and packet["status"] == "complete_intake"
        and packet["private_cases_sha256"] == sha(CASES),
        "Exact verified intake required",
    )
    rows = prior.records(CASES)
    require(
        0 < len(rows) <= 60
        and len(rows) % len(ROLES) == 0
        and packet["selected_cases"] == len(rows)
        and packet["selected_groups"]
        == packet["selected_seeds_with_exact_normalized_govreport_match"]
        == len(rows) // len(ROLES)
        and len({row["case"] for row in rows}) == len(rows),
        "One human seed and four related edits per report required",
    )
    groups = set()
    for start in range(0, len(rows), len(ROLES)):
        batch = rows[start : start + len(ROLES)]
        group = batch[0]["article_group"]
        require(
            group not in groups
            and all(row["article_group"] == group and row["status"] == "constructed" for row in batch)
            and tuple((row["generator"], row["arm"]) for row in batch) == ROLES,
            "Intake must preserve exact source-group order and edit roles",
        )
        groups.add(group)
        for row in batch:
            prior.labels_for(row)
            require(len(row["text"].split()) >= 15, "Retained input is below the current text length floor")
    return rows


def identity(rows):
    require(prior.identity() == read(prior.PUBLIC), "Exact prior native method and calibration required")
    return {
        "schema": "chipmunk.text-opai-bench-score-freeze/260",
        "intake_sha256": sha(INTAKE),
        "private_cases_sha256": sha(CASES),
        "native_reference_sha256": sha(prior.PUBLIC),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (Path(__file__), METHOD_DOC)},
        "source_groups": len(rows) // len(ROLES),
        "planned_inputs": len(rows),
        "roles": [list(role) for role in ROLES],
        "batches": ceil(len(rows) / WIDTH),
        "batch_max_inputs": WIDTH,
        "methods": list(prior.layout.METHODS),
        "serving_change_permitted": False,
    }


def freeze():
    rows = inputs()
    frozen = identity(rows)
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve all earlier attempts")
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return {"status": "frozen", "planned_inputs": len(rows), "new_neural_calls": 0}


def verify():
    rows = inputs()
    require(
        identity(rows) == read(TRIAL / "freeze.json") == read(PUBLIC), "Exact prospective freeze required"
    )
    return rows


worker = scoped(
    prior.worker,
    {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "verify": verify, "ARMS": ROLES, "PAIRS_PER_BATCH": GROUPS_PER_BATCH},
)


def run(batch):
    count = ceil(len(verify()) / WIDTH)
    scoped(
        prior.run,
        {"TRIAL": TRIAL, "verify": verify, "BATCHES": count, "MODULE": MODULE, "__file__": __file__},
    )(batch)


def observations(rows):
    responses, profiles, calls = [], [], Counter()
    for batch, start in enumerate(range(0, len(rows), WIDTH)):
        stage = TRIAL / f"extract-{batch:02d}"
        result_path = stage / "output/result.json"
        result = read(result_path)
        journal = stage / "output/private-responses.jsonl"
        saved = prior.records(journal)
        prior.qualification_profile(
            stage, result["new_neural_calls"], {}, Path(__file__).name, "extract", batch
        )
        require(
            result["status"] == "complete"
            and result["batch"] == batch
            and result["freeze_sha256"] == sha(PUBLIC)
            and result["journal_sha256"] == sha(journal)
            and result["planned_cases"] == len(saved)
            and result["new_neural_calls"] == sum(result["actual_forwards"].values())
            and [row["case"] for row in saved] == [row["case"] for row in rows[start : start + WIDTH]],
            "Expected the exact completed response for each planned input",
        )
        responses.extend(saved)
        calls.update(result["actual_forwards"])
        profiles.append(sha(stage / "profile.json"))
    return responses, profiles, dict(calls)


def summarize(rows, responses, points):
    tables = defaultdict(Counter)
    for row, response in zip(rows, responses, strict=True):
        for method, metrics in prior.measurements(row, response, points).items():
            counter = tables[row["generator"], row["arm"], method]
            counter["planned"] += 1
            counter[metrics["status"]] += 1
            counter.update({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
    result = {}
    for (generator, arm, method), counter in tables.items():
        iou = counter.pop("character_iou", 0)
        result.setdefault(generator, {}).setdefault(arm, {})[method] = {
            **counter,
            "mean_character_iou_assessed": iou / counter["ok"] if counter["ok"] else None,
        }
    return result


def readout():
    rows = verify()
    responses, profiles, calls = observations(rows)
    points = read(prior.historical.PRIOR / "points.json")
    return {
        "schema": "chipmunk.text-opai-bench-score-result/260",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "planned_inputs": len(rows),
        "source_groups": len(rows) // len(ROLES),
        "actual_forwards": calls,
        "resource_profile_sha256": profiles,
        "by_generator_arm_and_method": summarize(rows, responses, points),
        "serving_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("preflight", "freeze", "readout"):
        commands.add_parser(action)
    for action in ("run", "_worker"):
        command = commands.add_parser(action)
        command.add_argument("batch", type=int)
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.batch)
    elif args.command == "_worker":
        worker(args.batch, args.output)
    else:
        action = {"preflight": lambda: identity(inputs()), "freeze": freeze, "readout": readout}[args.command]
        print(json.dumps(action(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
