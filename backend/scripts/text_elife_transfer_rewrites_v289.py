"""Prepare fixed development and confirmation rewrites from the fresh eLife intake."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import FunctionType

from scripts import text_elife_rewrites_v273 as previous

ROOT, PANEL, BASE = previous.ROOT, previous.PANEL, previous.BASE
NAME = "text-elife-transfer-rewrites-v289"
MODULE = "scripts.text_elife_transfer_rewrites_v289"
TRIAL = BASE / NAME
FREEZE = PANEL / "text-elife-transfer-rewrites-freeze-v289.json"
INTAKE = PANEL / "text-elife-transfer-intake-complete-v286.json"
SOURCE = BASE / "text-elife-transfer-intake-v286"
INTAKE_SHA = "5dd8010b4b81fc6598528a19a0b22f5a904da960e6edba2e04cf053ed7794440"
ROSTER_SHA = "6d359711bcad0d96b71d6a90053af43a530e8c47a30ef564285b96b33eab6a8f"
PREVIOUS_SHA = "3566e9a413d9671b0cd56517ceae711a9aaa8f90e0c2017f9309bf4f293d492d"
ROLES = ("development", "confirmation")
PER_ROLE = 13
SALT = "chipmunk-elife-v289-split-v1:"
STAGES = tuple(f"{role}:{family}" for role in ROLES for family in previous.GENERATORS)
STAGE_GENERATORS = {stage: previous.GENERATORS[stage.split(":")[1]] for stage in STAGES}
DISPOSITIONS = {"admitted": 26, "no_eligible_excerpt": 8, "license_not_cc_by_4": 1}
read, write, sha, require = previous.read, previous.write, previous.sha, previous.require


def bind(function, **overrides):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **overrides},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )


def validate(receipt, rows):
    admitted = [record for record in receipt["works"] if record["status"] == "admitted"]
    groups = [row["group"] for row in rows]
    require(
        receipt["schema"] == "chipmunk.text-elife-transfer-intake/286"
        and receipt["status"] == "complete_intake"
        and receipt["roster_sha256"] == ROSTER_SHA
        and receipt["planned_works"] == len(receipt["works"]) == sum(DISPOSITIONS.values())
        and receipt["dispositions"]
        == dict(Counter(row["status"] for row in receipt["works"]))
        == DISPOSITIONS
        and receipt["admitted_source_groups"]
        == len(admitted)
        == len(set(groups))
        == len(rows)
        == 2 * PER_ROLE
        and groups == [record["group"] for record in admitted],
        "Exact v286 groups and the nine source exclusions required",
    )
    for work, row in zip(admitted, rows, strict=True):
        fields = ("group", "text_sha256", "body_slots", "paragraph_codepoints", "edit_target_codepoints")
        require(all(work[key] == row[key] for key in fields), "The source target changed after v286 intake")
        text, edges, target = (row[key] for key in ("text", "paragraph_codepoints", "edit_target_codepoints"))
        require(
            previous.text_sha(text) == row["text_sha256"]
            and target in edges[1:-1]
            and edges[0][0] == 0
            and edges[-1][1] == len(text)
            and all(type(a) is type(b) is int and 0 <= a < b <= len(text) for a, b in edges)
            and all(text[b:c] == "\n\n" for (_, b), (c, _) in zip(edges, edges[1:]))
            and [text[a:b] for a, b in edges] == text.split("\n\n"),
            "Exact Unicode paragraphs and a fixed interior target required",
        )
    return rows


def inputs():
    originals = SOURCE / "private-originals.jsonl"
    receipt = read(INTAKE)
    require(
        sha(INTAKE) == INTAKE_SHA
        and receipt == read(SOURCE / "aggregate.json")
        and receipt["private_originals_sha256"] == sha(originals),
        "Exact public and private v286 intake required",
    )
    return validate(receipt, previous.records(originals))


def partition(rows):
    require(
        len({row["group"] for row in rows}) == len(rows) == 2 * PER_ROLE, "Distinct source groups required"
    )
    ordered = sorted(rows, key=lambda row: (previous.text_sha(SALT + row["group"]), row["group"]))
    return dict(zip(ROLES, (ordered[:PER_ROLE], ordered[PER_ROLE:]), strict=True))


def identity(partitions):
    predecessor = read(previous.FREEZE)
    require(
        sha(previous.FREEZE) == PREVIOUS_SHA
        and all(sha(ROOT / name) == digest for name, digest in predecessor["code_sha256"].items()),
        "Keep exact v273 generator, prompt, loader and resource guard code",
    )
    for family, (repo, revision, weights, path) in previous.GENERATORS.items():
        recorded = predecessor["generators"][family]
        require(
            recorded["repo"] == repo
            and recorded["revision"] == revision
            and recorded["weights_sha256"] == weights
            and recorded["license"] == "Apache-2.0"
            and (path / "model.safetensors").is_file()
            and {name: sha(path / name) for name in previous.ASSETS if (path / name).is_file()}
            == recorded["local_config_sha256"],
            "Exact v273 local generator assets required",
        )
    files = (
        Path(__file__),
        ROOT / f"docs/{NAME}-method.md",
        ROOT / "backend/tests/test_text_elife_transfer_rewrites_v289.py",
    )
    return {
        "schema": "chipmunk.text-elife-transfer-rewrites-freeze/289",
        "v286_intake_sha256": INTAKE_SHA,
        "v286_roster_sha256": ROSTER_SHA,
        "private_originals_sha256": sha(SOURCE / "private-originals.jsonl"),
        "v273_freeze_sha256": PREVIOUS_SHA,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "split_salt": SALT,
        "source_groups_by_role": {role: [row["group"] for row in rows] for role, rows in partitions.items()},
        "source_admission_dispositions": DISPOSITIONS,
        "stages": list(STAGES),
        "planned_rewrites_per_stage": PER_ROLE,
        "planned_rewrites": len(STAGES) * PER_ROLE,
        "generators": predecessor["generators"],
        "runtime": predecessor["runtime"],
        "new_detector_calls_permitted": False,
    }


def freeze():
    require(not TRIAL.exists() and not FREEZE.exists(), "Preserve every earlier attempt")
    locked = identity(partition(inputs()))
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(FREEZE, locked)
    return {"status": "frozen", "planned_rewrites": locked["planned_rewrites"], "source_groups": 2 * PER_ROLE}


def verify():
    partitions = partition(inputs())
    require(
        read(FREEZE) == read(TRIAL / "freeze.json") == identity(partitions),
        "Exact prospective v289 freeze required",
    )
    return partitions


def rewrite(row, stage, raw):
    role, family = stage.split(":")
    return {**previous.rewrite(row, family, raw), "role": role}


def worker(batch, output):
    role, _ = STAGES[batch].split(":")
    bind(
        previous.worker,
        TRIAL=TRIAL,
        FREEZE=FREEZE,
        GENERATORS=STAGE_GENERATORS,
        verify=lambda: verify()[role],
        rewrite=rewrite,
    )(batch, output)


def run(batch):
    require(
        all(
            (TRIAL / f"extract-{previous_batch:02d}/profile.json").is_file()
            for previous_batch in range(batch)
        ),
        "Run the four fixed stages in order",
    )
    bind(
        previous.run,
        TRIAL=TRIAL,
        MODULE=MODULE,
        GENERATORS=STAGE_GENERATORS,
        verify=verify,
        __file__=__file__,
    )(batch)


def control(row, role):
    return {
        "case": f"{row['group']}-human",
        "article_group": row["group"],
        "generator": None,
        "arm": "human",
        "role": role,
        "status": "constructed",
        "text": row["text"],
        "text_sha256": row["text_sha256"],
        "ai_spans": [],
    }


def replay_stage(partitions, batch):
    from scripts.grid_source_speech_v147 import qualification_profile

    stage = STAGES[batch]
    role, _ = stage.split(":")
    folder = TRIAL / f"extract-{batch:02d}"
    receipt = qualification_profile(folder, PER_ROLE, {}, Path(__file__).name, "extract", batch)
    journal = folder / "output/private-rewrites.jsonl"
    require(
        receipt["family"] == stage
        and receipt["freeze_sha256"] == sha(FREEZE)
        and receipt["journal_sha256"] == sha(journal)
        and receipt["weights_sha256"] == STAGE_GENERATORS[stage][2],
        "Exact guarded generation receipt required",
    )
    cases = []
    for source, attempt in zip(partitions[role], previous.records(journal), strict=True):
        case = rewrite(source, stage, attempt["raw_output"])
        require(case == attempt["case"], "Recorded generation disposition or Unicode coordinates changed")
        cases.append(case)
    return cases, sha(folder / "profile.json")


def readout():
    from scripts.text_wikinews_archive_v205 import retain

    partitions = verify()
    status = {}
    for batch, stage in enumerate(STAGES):
        path = TRIAL / f"extract-{batch:02d}/profile.json"
        profile = read(path) if path.is_file() else None
        status[stage] = (
            "pending"
            if profile is None
            else (
                "unavailable"
                if profile["guard"] or profile["child_exit_code"] != 0 or profile["sample_errors"]
                else "complete"
            )
        )
    if any(value != "complete" for value in status.values()):
        return {
            "status": "pending_generation" if "pending" in status.values() else "generation_unavailable",
            "planned_rewrites": len(STAGES) * PER_ROLE,
            "stage_dispositions": status,
            "rewrites_without_completed_stage": PER_ROLE
            * sum(value != "complete" for value in status.values()),
        }
    by_group = {row["group"]: [control(row, role)] for role, rows in partitions.items() for row in rows}
    profiles, counts = {}, {}
    for batch, stage in enumerate(STAGES):
        cases, profiles[stage] = replay_stage(partitions, batch)
        counts[stage] = dict(Counter(case["reason"] or "constructed" for case in cases))
        for case in cases:
            by_group[case["article_group"]].append(case)
    cases = TRIAL / "private-cases.jsonl"
    retain(
        cases,
        "".join(
            json.dumps(case, ensure_ascii=False) + "\n" for group in by_group.values() for case in group
        ).encode(),
    )
    result = {
        "schema": "chipmunk.text-elife-transfer-rewrites-result/289",
        "status": "complete",
        "freeze_sha256": sha(FREEZE),
        "source_groups_by_role": {role: len(rows) for role, rows in partitions.items()},
        "source_admission_dispositions": DISPOSITIONS,
        "human_controls": len(by_group),
        "planned_rewrites": len(STAGES) * PER_ROLE,
        "generation_dispositions_by_stage": counts,
        "private_cases_sha256": sha(cases),
        "resource_profile_sha256": profiles,
        "new_detector_calls": 0,
    }
    path = TRIAL / "result.json"
    if path.exists():
        require(result == read(path), "Retained result differs from exact replay")
    else:
        write(path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for action in ("preflight", "freeze", "readout"):
        commands.add_parser(action)
    for action in ("run", "_worker"):
        command = commands.add_parser(action)
        command.add_argument("batch", type=int, choices=range(len(STAGES)))
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.batch)
    elif args.mode == "_worker":
        worker(args.batch, args.output)
    else:
        action = {"preflight": lambda: identity(partition(inputs())), "freeze": freeze, "readout": readout}[
            args.mode
        ]
        print(json.dumps(action(), indent=2, allow_nan=False))
