"""Run the frozen JPEG comparison on the twelve v262 BG-REAL backup pairs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts import image_bg_real_backup_intake_v262 as intake
from scripts import image_bg_real_jpeg_v229 as previous
from scripts import image_jpeg_diagnostic_runtime_v189 as monitor
from scripts.video_appearance_corrected_v112 import scoped

original = previous.original
ROOT, PANELS = previous.ROOT, previous.PANELS
NAME = "image-bg-real-backup-jpeg-v264"
MODULE = "scripts.image_bg_real_backup_jpeg_v264"
TRIAL = intake.DATA / NAME
PUBLIC = PANELS / f"{NAME}-freeze.json"
QUALIFICATION = PANELS / f"{intake.NAME}-complete.json"
RECEIPTS = {
    QUALIFICATION: "74a07ec74a63e0e252904a37f915d1cb1d70d1b0cba01aae4be6a94f7412e51e",
    intake.PUBLIC: "8dc84fd639ecd599ee46c411af8345aaca8bcae5f2952c040c9fe70f8b38ab06",
    intake.SCORE: intake.RECEIPTS[intake.SCORE],
}
CODE = (
    "backend/scripts/image_bg_real_backup_jpeg_v264.py",
    "backend/tests/test_image_bg_real_backup_jpeg_v264.py",
    f"docs/{NAME}-method.md",
)
PHOTOS = intake.PAIR_COUNT
KINDS, QUALITIES, ARMS = previous.KINDS, previous.QUALITIES, previous.ARMS
PHASES = (PHOTOS * len(KINDS) + original.CHUNK - 1) // original.CHUNK
read, write, sha, require = previous.read, previous.write, previous.sha, previous.require


def public_contract():
    require(
        all(sha(path) == digest for path, digest in RECEIPTS.items()),
        "Exact published v262/v229 receipts required",
    )
    _, policy, inherited = previous.public_contract()
    summary, frozen = read(QUALIFICATION), read(intake.PUBLIC)
    require(
        summary["freeze_sha256"] == RECEIPTS[intake.PUBLIC]
        and summary["schema"] == "chipmunk.image-bg-real-backup-intake-result/262"
        and summary["status"] == "complete"
        and summary["qualified_pairs"]
        == summary["qualified_publisher_bound_masks"]
        == summary["attempted_pairs"]
        == PHOTOS
        and summary["target_shortfall"] == summary["neural_calls"] == summary["detector_calls"] == 0,
        "Exact twelve independently qualified v262 pairs required",
    )
    inherited = (
        inherited | frozen["intake_code_commitments"] | {name: sha(ROOT / name) for name in previous.CODE}
    )
    require(
        all(sha(ROOT / name) == digest for name, digest in inherited.items()),
        "A comparison dependency changed",
    )
    return summary, policy, inherited


def source_cases(summary):
    frozen = read(intake.PUBLIC)
    cache, planned, _, _ = intake.lease(intake.TRIAL, frozen["private_plan_sha256"])
    sealed_file = intake.TRIAL / "qualified-private.json"
    require(
        sha(sealed_file) == summary["qualified_private_sha256"]
        and sha(intake.TRIAL / "attempts.jsonl") == summary["acquisition_journal_sha256"],
        "Exact sealed v262 pair index and journal required",
    )
    sealed = read(sealed_file)
    require(
        sealed["schema"] == "chipmunk.image-bg-real-pixel-qualified-private/221"
        and sealed["plan_sha256"] == frozen["private_plan_sha256"]
        and sealed["acquisition_journal_sha256"] == summary["acquisition_journal_sha256"]
        and Path(sealed["cache"]) == cache
        and sealed["attempt_statuses"] == [{"ordinal": n, "status": "qualified"} for n in range(PHOTOS)]
        and [entry["source"] for entry in sealed["qualified"]] == planned,
        "Qualified rows must preserve the twelve unopened source commitments",
    )
    cases, assets = [], {}
    for n, row in enumerate(sealed["qualified"]):
        source, observation = row["source"], row["measurement"]
        source_id = source["recipient"]
        require(
            source_id not in assets
            and source["ordinal"] == observation["ordinal"] == n
            and source["v218_ordinal"] == n + 24
            and source["publisher_split"] == observation["publisher_split"]
            and observation["pair_failure"] is None
            and not observation["prior_matches"]
            and observation["measured"]["semantic_mask_binding"] == "publisher_asset_commitment"
            and row["truth_file"] == f"{n:02d}/exact_truth.png",
            "Qualified source identity, mask, or exposure differs",
        )
        cases.append(
            {
                "source_id": source_id,
                "component": f"bg-real-backup-{n:02d}",
                "publisher_split": source["publisher_split"],
            }
        )
        assets[source_id] = row
    require(
        len(cases) == PHOTOS
        and all(
            len({row["source"][key] for row in assets.values()}) == PHOTOS
            for key in ("flickr_owner", "flickr_photo_id")
        )
        and dict(Counter(row["publisher_split"] for row in cases)) == summary["qualified_by_publisher_split"],
        "Expected twelve separate source groups",
    )
    return cases, {"cache": cache, "assets": assets}


base_identity = scoped(
    previous.frozen_identity,
    {
        "PHOTOS": PHOTOS,
        "PHASES": PHASES,
        "CODE": CODE,
        "QUALIFICATION": QUALIFICATION,
        "TRANSPORT_FREEZE": intake.PUBLIC,
        "PUBLISHED_SHA": previous.PUBLISHED_SHA | RECEIPTS,
    },
)


def frozen_identity(summary, policy, inherited):
    return {
        **base_identity(summary, policy, inherited),
        "schema": "chipmunk.image-bg-real-backup-jpeg-freeze/264",
        "previous_v229_result_sha256": RECEIPTS[intake.SCORE],
        "previous_v229_panel_rescored": False,
    }


prepare = scoped(
    previous.prepare,
    {
        "SOURCE": intake.TRIAL,
        "public_contract": public_contract,
        "source_cases": source_cases,
        "frozen_identity": frozen_identity,
    },
)
tasks = scoped(original.tasks, {"PHOTOS": PHOTOS, "PHASES": PHASES})
batch_views = scoped(original.batch_views, {"tasks": tasks, "material": previous.material})
MONITOR_IDENTITY = {"MODULE": MODULE, "__file__": __file__}


def launch(argv, **kwargs):
    require(argv[1:3] == ["-m", MODULE], "Exact isolated JPEG worker required")
    return subprocess.Popen([*argv[:3], "_worker", *argv[3:]], **kwargs)


assert_profile = scoped(monitor.assert_profile, MONITOR_IDENTITY)
run = scoped(
    monitor.run,
    {
        **MONITOR_IDENTITY,
        "assert_profile": assert_profile,
        "subprocess": SimpleNamespace(
            Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
        ),
    },
)


def private_lease(trial, expected, profile=None):
    require(trial == TRIAL and sha(trial / "freeze.json") == expected, "Exact prepared v264 trial required")
    summary, policy, inherited = public_contract()
    frozen = frozen_identity(summary, policy, inherited)
    frozen["private_input_bindings_sha256"] = sha(trial / "inputs.json")
    require(read(trial / "freeze.json") == frozen, "Source, JPEG bindings or frozen policy changed")
    inputs = read(trial / "inputs.json")
    cases, index = source_cases(summary)
    require(
        cases
        == [
            {key: row[key] for key in ("source_id", "component", "publisher_split")}
            for row in inputs["sources"]
        ]
        and Path(inputs["raw_root"]) == index["cache"].parent / f"{NAME}-maps",
        "Prepared source groups, order or private map destination changed",
    )
    if profile is not None:
        assert_profile(profile, "prepare", ["prepare", str(trial), inputs["raw_root"]])
    return inputs, cases, index


def lease(trial, expected, profile=None):
    require(sha(PUBLIC) == expected, "Prospective v264 freeze required before scoring")
    return private_lease(trial, expected, profile)


def phase_rows(trial, name, command, args, expected):
    assert_profile(trial / f"{name}-profile", command, args)
    part = trial / name
    finished, journal = read(part / "completed.json"), part / "rows.jsonl"
    require(
        finished["freeze_sha256"] == expected and finished["rows_sha256"] == sha(journal),
        "Phase receipt or journal changed",
    )
    return finished, [json.loads(line) for line in journal.read_text().splitlines()]


producer_rows = scoped(original.producer_rows, {"phase_rows": phase_rows, "tasks": tasks})
score = scoped(original.score, {"lease": lease, "batch_views": batch_views, "tasks": tasks})
readout = scoped(
    original.readout,
    {"lease": lease, "producer_rows": producer_rows, "batch_views": batch_views, "tasks": tasks},
)
coverage = scoped(
    original.coverage, {"lease": lease, "producer_rows": producer_rows, "PHASES": PHASES, "PHOTOS": PHOTOS}
)


def write_result(path, result):
    write(
        path,
        {
            **result,
            "schema": "chipmunk.image-bg-real-backup-jpeg-result/264",
            "previous_v229_result_sha256": RECEIPTS[intake.SCORE],
            "scope": "Previously unopened BG-REAL LaMa groups. Both JPEG qualities reuse the same twelve source groups. "
            "Paired intervals are conditional and pointwise. Cue localization is separate from whole-file authorship confidence.",
        },
    )


result_write = scoped(previous.result_write, {"write": write_result})
seal = scoped(
    original.seal,
    {
        "lease": lease,
        "producer_rows": producer_rows,
        "phase_rows": phase_rows,
        "PHASES": PHASES,
        "tasks": tasks,
        "write": result_write,
    },
)


def preflight():
    summary, policy, inherited = public_contract()
    require(
        intake.readout(read(intake.PUBLIC)["private_plan_sha256"]) == summary,
        "Independent v262 readout differs",
    )
    cases, index = source_cases(summary)
    frozen = frozen_identity(summary, policy, inherited)
    for n in range(PHOTOS):
        previous.material(cases, index, n)
    return {
        "status": "ready_for_prepare",
        "pairs": len(cases),
        "jpeg_inputs": PHOTOS * len(KINDS) * len(QUALITIES),
        "views_per_phase": [len(tasks(n)) for n in range(PHASES)],
        "unique_forwards_per_arm": frozen["unique_model_calls"] // len(ARMS),
        "first_input_repeats_total": frozen["exact_first_input_repeats"],
        "model_calls": 0,
    }


def freeze():
    prepared = TRIAL / "freeze.json"
    expected = sha(prepared)
    private_lease(TRIAL, expected, original.prep_profile(TRIAL))
    with PUBLIC.open("xb") as stream:
        stream.write(prepared.read_bytes())
    return {"status": "frozen", "freeze_sha256": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    worker = sys.argv[1:2] == ["_worker"]
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        ("prepare", "score", "readout", "seal")
        if worker
        else ("preflight", "prepare", "freeze", "score", "readout", "seal", "coverage")
    ):
        command = commands.add_parser(name)
        if worker:
            command.add_argument("trial", type=Path)
            if name == "prepare":
                command.add_argument("raw_root", type=Path)
            else:
                command.add_argument("freeze_sha256")
        if name == "score":
            command.add_argument("arm", choices=ARMS)
        if name in {"score", "readout"}:
            command.add_argument("quality", type=int, choices=QUALITIES)
            command.add_argument("phase", type=int, choices=range(PHASES))
    args = parser.parse_args(sys.argv[2:] if worker else None)
    if worker:
        require(args.trial == TRIAL, "Exact isolated trial required")
        return {"prepare": prepare, "score": score, "readout": readout, "seal": seal}[args.command](
            *list(vars(args).values())[1:]
        )
    if args.command in {"preflight", "freeze"}:
        result = preflight() if args.command == "preflight" else freeze()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "prepare":
        _, index = source_cases(public_contract()[0])
        run("prepare", [str(TRIAL), str(index["cache"].parent / f"{NAME}-maps")])
    elif args.command == "coverage":
        coverage(TRIAL, sha(PUBLIC))
    else:
        run(args.command, [str(TRIAL), sha(PUBLIC), *(str(value) for value in list(vars(args).values())[1:])])


if __name__ == "__main__":
    main()
