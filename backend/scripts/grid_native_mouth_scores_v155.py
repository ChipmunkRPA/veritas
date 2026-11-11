"""Continue the frozen v151 pilot scores at exclusive paths after its monitor stop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts import grid_native_mouth_scores_v151 as original
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL, ARCHIVE = original.ROOT, original.BASE, original.PANEL, original.ARCHIVE
TRIAL = BASE / "grid-native-mouth-scores-v155"
CACHE = ARCHIVE / "grid-native-mouth-scores-v155"
PUBLIC = PANEL / "grid-native-mouth-scores-freeze-v155.json"
TAIL = tuple(range(1, 16))
CALLER = Path(__file__).name
ORIGINAL_CALLER = Path(original.__file__).name
CODE = (
    "backend/scripts/grid_native_mouth_scores_v155.py",
    "backend/tests/test_grid_native_mouth_scores_v155.py",
    "docs/grid-native-mouth-scores-v155-method.md",
)
read, write, sha, require = original.read, original.write, original.sha, original.require


def tail_ordinal(ordinal):
    require(type(ordinal) is int and ordinal in TAIL, "One original pilot tail ordinal 1 through 15 required")


def stage_path(trial, ordinal):
    return trial / f"clip-{ordinal:02d}/bind-00"


def commitments(trial, cache, ordinal):
    stage = stage_path(trial, ordinal)
    folder = cache / f"clip-{ordinal:02d}"
    return {
        "result_sha256": sha(stage / "output/result.json"),
        "profile_sha256": sha(stage / "profile.json"),
        "resource_journal_sha256": sha(stage / "resources.jsonl"),
        "worker_log_sha256": sha(stage / "worker.log"),
        "private_score_journal_sha256": sha(folder / "scored-cases.jsonl"),
        "private_input_binding_sha256": sha(folder / "input-binding.json"),
        "private_visual_invariance_sha256": sha(folder / "visual-invariance.json"),
    }


def failure_history():
    stage = stage_path(original.TRIAL, 1)
    profile_path = stage / "profile.json"
    journal, log = stage / "resources.jsonl", stage / "worker.log"
    profile = read(profile_path)
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["runtime_sha256"] == sha(ROOT / "backend/scripts" / ORIGINAL_CALLER)
        and profile["child_started"] is True
        and profile["child_exit_code"] == -15
        and profile["guard"] == "monitor_unavailable"
        and profile["sample_errors"] == 1
        and profile["error_types"] == ["PermissionError"]
        and profile["resource_limit_observations"] == []
        and profile["worker_receipt_sha256"] is None
        and profile["resource_journal_sha256"] == sha(journal)
        and log.stat().st_size == 0
        and not (stage / "output").exists()
        and not (original.CACHE / "clip-01").exists(),
        "Actual v151 clip 1 monitor failure and absence of worker output required",
    )
    return {
        "clip_ordinal": 1,
        "guard": profile["guard"],
        "error_types": profile["error_types"],
        "child_exit_code": profile["child_exit_code"],
        "worker_receipt_present": False,
        "profile_sha256": sha(profile_path),
        "resource_journal_sha256": sha(journal),
        "worker_log_sha256": sha(log),
    }


def identity():
    require(len(original.load_completed(0)) == 11, "Actual complete v151 clip 0 required")
    plan = original.planning.load_completed()
    require(
        [r["ordinal"] for r in original.pilots(plan)] == [0, *TAIL], "Exact original pilot order required"
    )
    return {
        "schema": "chipmunk.grid-native-mouth-score-continuation-freeze/155",
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "v151_freeze_sha256": sha(original.PUBLIC),
        "retained_v151_clip0": commitments(original.TRIAL, original.CACHE, 0),
        "retained_v151_clip1_monitor_failure": failure_history(),
        "tail_clip_ordinals": list(TAIL),
        "planned_cases": 176,
        "new_tail_cases": 165,
        "binaries_sha256": read(original.PUBLIC)["binaries_sha256"],
        "scope": "Unchanged v151 scoring, inputs, learned assets, settings and data-stage monitor. Clip 0 uses its completed v151 receipt. Clips 1 through 15 use new result and score journal paths.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "retained_clips": 1, "tail_clips": len(TAIL)}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(), "Unchanged published continuation required"
    )


def score_function():
    return scoped(original.score, {"CACHE": CACHE, "PUBLIC": PUBLIC})


def tail_stage(folder, freeze_sha, caller, batch, bindings):
    require(
        caller == ORIGINAL_CALLER
        and batch == 0
        and freeze_sha == sha(PUBLIC)
        and folder in {stage_path(TRIAL, ordinal) for ordinal in TAIL},
        "Only the continuation receipt caller may be rerouted",
    )
    return original.planning.reviewed.stage(folder, freeze_sha, CALLER, batch, bindings)


def tail_reader(check):
    planning = SimpleNamespace(
        load_completed=original.planning.load_completed,
        reviewed=SimpleNamespace(stage=tail_stage, lines=original.planning.reviewed.lines),
    )
    return scoped(
        original.load_completed,
        {
            "TRIAL": TRIAL,
            "CACHE": CACHE,
            "PUBLIC": PUBLIC,
            "verify": check,
            "planning": planning,
        },
    )


def load_completed(ordinal):
    if type(ordinal) is int and ordinal == 0:
        return original.load_completed(0)
    tail_ordinal(ordinal)
    return tail_reader(verify)(ordinal)


def aggregate_completed(*, include_private=False):
    verify()
    plan = original.planning.load_completed()
    first = original.load_completed(0)
    # The continuation was verified above. Each tail read still verifies its
    # actual receipt, resource journal, input binding and score inventory.
    reader = tail_reader(lambda: None)
    rows = [*first, *(row for ordinal in TAIL for row in reader(ordinal))]
    totals = original.aggregate_records(plan, rows)
    collection = [commitments(original.TRIAL, original.CACHE, 0)]
    collection.extend(commitments(TRIAL, CACHE, ordinal) for ordinal in TAIL)
    collection_bytes = json.dumps(collection, sort_keys=True, separators=(",", ":")).encode()
    if include_private:
        totals["private_stage_commitments"] = collection
    else:
        totals.pop("source_slot_paired_aggregates")
        totals.pop("component_paired_aggregates")
    return {
        "schema": "chipmunk.grid-native-mouth-score-continuation-result/155",
        "status": "complete",
        **totals,
        "retained_v151_clips": 1,
        "completed_v155_clips": len(TAIL),
        "new_neural_calls": 0,
        "new_head_or_calibration_fits": 0,
        "v151_freeze_sha256": sha(original.PUBLIC),
        "v155_freeze_sha256": sha(PUBLIC),
        "receipt_collection_sha256": hashlib.sha256(collection_bytes).hexdigest(),
    }


def routed_popen(ordinal, argv, **kwargs):
    tail_ordinal(ordinal)
    expected = stage_path(TRIAL, ordinal) / "output"
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"]
        and argv[5:] == [str(expected), "0"],
        "Exact exclusive data-only continuation worker required",
    )
    return subprocess.Popen(
        [*argv[:3], f"scripts.{Path(__file__).stem}", "score", str(expected), str(ordinal)], **kwargs
    )


def run(ordinal):
    tail_ordinal(ordinal)
    scoped(
        original.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "routed_popen": routed_popen,
            "__file__": str(Path(__file__)),
        },
    )(ordinal)


def worker(ordinal, output):
    tail_ordinal(ordinal)
    stage = stage_path(TRIAL, ordinal)
    expected = stage / "output"
    require(output == expected, "Exact exclusive continuation output required")
    resources, log = stage / "resources.jsonl", stage / "worker.log"
    require(
        resources.is_file()
        and resources.stat().st_size > 0
        and log.is_file()
        and not (stage / "profile.json").exists()
        and all(os.path.samestat(os.fstat(fd), log.stat()) for fd in (1, 2)),
        "Active outer monitor files and an unfinished profile required",
    )
    verify()
    output.mkdir(exist_ok=False)
    score_function()(ordinal, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "score", "aggregate"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "No freeze arguments")
        freeze()
    elif args.command == "aggregate":
        require(not args.arguments, "No aggregate arguments")
        print(json.dumps(aggregate_completed(), allow_nan=False), flush=True)
    elif args.command == "run":
        require(len(args.arguments) == 1, "One explicit tail ordinal required")
        run(int(args.arguments[0]))
    else:
        require(len(args.arguments) == 2, "Exclusive output and tail ordinal required")
        worker(int(args.arguments[1]), Path(args.arguments[0]))


if __name__ == "__main__":
    main()
