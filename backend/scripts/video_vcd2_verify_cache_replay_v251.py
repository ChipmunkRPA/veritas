"""Replay the sealed verification-cache failure with exactly one app source change."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts import video_vcd2_case_archive_v202 as archives
from scripts import video_vcd2_full_app_v202 as raw
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL, ARM = raw.ROOT, raw.BASE, raw.PANEL, raw.ARM
baseline = raw.prior.original.baseline
read, write, sha, require = raw.read, raw.write, raw.sha, raw.require
NAME = "video-vcd2-verify-cache-replay-v251"
MODULE = "scripts.video_vcd2_verify_cache_replay_v251"
TRIAL, WORK = BASE / NAME, BASE / "video-vcd2-verify-cache-replay-work-v251"
SEALED = raw.ARCHIVE / NAME
PUBLIC = PANEL / f"{NAME}-freeze.json"
CONDITION = "slot-10-chroma_conversion_only"
SLOTS = (1, 3, 4, 5, 7, 8, 9, 10)
OLD_RESULT = PANEL / "video-vcd2-full-app-eight-triplets-v202.json"
OLD_RESULT_SHA = "66a91b0e09ec8c3a317acd17ad576b6b580deb1c5efee6e9079631d262cedd5a"
OLD_FREEZE_SHA = "0e63e032a3cae35590862ca43cf2eceeb29b7558d884fa9ce76124b3e06c5489"
OLD_SEAL_SHA = "360648cf7b7c0bba72106c798fd8bdd2f6757ef2d2239c83034dc8e8f44640f5"
BASELINE_SHA = "ec9032d3b79daad0997cb9c64a2a4d39522f7a4f34ae01f9abfe10a786467194"
FETCHERS = "backend/app/engine/fetchers.py"
OLD_FETCHERS = "e83fa48fef53026f30930b71feb57b114b9c5fcdf307bb47a0b75ae8e3fba366"
PATCHED_FETCHERS = "02b01adbab53261c33c98cd83fdde607dd4d9f6941536fa8123b79a334560c09"
AUDIT_FIELDS = (
    "status",
    "customer_direction",
    "legacy_frame_status",
    "appearance_route_status",
    "appearance_check_status",
    "appearance_measured_windows",
    "verified_sampled_frames",
    "verified_native_anchors",
    "verified_evidence_frames",
    "verified_evidence_assets",
)
CODE = (
    f"backend/scripts/{MODULE.split('.')[-1]}.py",
    "backend/tests/test_video_vcd2_verify_cache_replay_v251.py",
    f"docs/{NAME}-method.md",
)


def check_app_sources(historical, actual, *, patched):
    expected = PATCHED_FETCHERS if patched else OLD_FETCHERS
    require(
        historical.get(FETCHERS) == OLD_FETCHERS and actual == {**historical, FETCHERS: expected},
        "Expected exact patched application source"
        if patched
        else "Expected exact historical application source",
    )


def verify_named_code(records):
    for name, digest in records.items():
        paths = [ROOT / folder / name for folder in ("backend/scripts", "backend/tests", "docs")]
        matches = [path for path in paths if path.is_file()]
        require(len(matches) == 1 and sha(matches[0]) == digest, "Historical runner source differs")


def runtime_bindings(old, *, patched):
    raw.require_fresh()
    frozen = read(raw.direct_file(baseline.PUBLIC))
    frame_path = raw.direct_file(raw.prior.runtime.TRIAL / "freeze.json")
    frame = read(frame_path)
    app_path = raw.direct_file(raw.prior.PUBLIC)
    app = read(app_path)
    inventory = raw.direct_file(raw.prior.original.INVENTORY)
    edge_path = raw.direct_file(raw.edge.PUBLIC)
    edge = read(edge_path)
    require(
        sha(baseline.PUBLIC) == BASELINE_SHA == frame["baseline_freeze_sha256"]
        and sha(app_path) == old["app_freeze_sha256"]
        and sha(frame_path) == app["runtime_freeze_sha256"]
        and old["app_manifest_sha256"] == app["arm_manifest_sha256"] == frame["arm_manifest_sha256"][ARM]
        and old["device"] == app["device"] == frame["device"]
        and old["model_manager_budget_bytes"]
        == app["model_manager_budget_bytes"]
        == frame["model_manager_budget_bytes"]
        and sha(inventory) == frozen["asset_inventory_sha256"]
        and sha(baseline.MANIFEST) == frozen["manifest_sha256"]
        and baseline.package_versions() == frozen["runtime_packages"]
        and frozen["platform"]
        == {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()}
        and raw.prior.runtime.dependency_identity() == frame["dependency_sources"]
        and sha(edge_path) == old["edge_freeze_sha256"]
        and sha(raw.edge.WEIGHT) == edge["weight_sha256"] == raw.edge.WEIGHT_SHA
        and edge["separation_point"] == raw.edge.SEPARATION
        and edge["appearance_replay_tolerance"] == raw.edge.REPLAY_TOLERANCE
        and edge["code_sha256"][Path(raw.edge.__file__).name] == sha(Path(raw.edge.__file__)),
        "Historical assets, model inventory, runtime or refinement differ",
    )
    check_app_sources(frozen["code_sha256"], baseline.code_identity(), patched=patched)
    require(
        all(sha(ROOT / name) == digest for name, digest in old["code_sha256"].items()),
        "Historical replay source changed",
    )
    verify_named_code(app["code_sha256"] | frame["code_sha256"] | frame["correction_code_sha256"])
    baseline.checked_programs(frozen)
    return {
        "baseline_freeze_sha256": BASELINE_SHA,
        "historical_app_freeze_sha256": sha(app_path),
        "historical_frame_runtime_sha256": sha(frame_path),
        "model_inventory_sha256": sha(inventory),
        "model_manifest_sha256": old["app_manifest_sha256"],
        "edge_freeze_sha256": sha(edge_path),
        "device": old["device"],
        "model_manager_budget_bytes": old["model_manager_budget_bytes"],
        "resource_limits": frame["resource_limits"],
    }


def archive_set_commitment(seals):
    return hashlib.sha256(json.dumps(seals, sort_keys=True).encode()).hexdigest()


def collision_log(text):
    return "FileNotFoundError" in text and ".veritas-verified.json.tmp" in text


def statuses(audit):
    return {key: audit[key] for key in AUDIT_FIELDS if key in audit}


def has_uploaded_bytes(entries, size, digest):
    return any(
        entry["kind"] == "file"
        and entry["path"].startswith("work/data/")
        and entry["size"] == size
        and entry["sha256"] == digest
        for entry in entries
    )


def historical(*, full_archive=False):
    require(
        sha(raw.PUBLIC) == OLD_FREEZE_SHA and sha(OLD_RESULT) == OLD_RESULT_SHA,
        "Exact published v202 freeze and eight-triplet result required",
    )
    old, result = read(raw.PUBLIC), read(OLD_RESULT)
    seals = {}
    for slot in SLOTS:
        for role in raw.ROLES:
            condition = raw.case_name(slot, role)
            archive, receipt = raw.archive_paths(condition)
            sealed = read(raw.direct_file(receipt))
            require(
                sealed["schema"] == archives.SCHEMA
                and sealed["archive_path"] == str(archive)
                and sealed["receipt_path"] == str(receipt),
                "Historical receipt identity differs",
            )
            seals[condition] = {"receipt_sha256": sha(receipt), "archive_sha256": sealed["archive_sha256"]}
    require(
        result["freeze_sha256"] == OLD_FREEZE_SHA
        and result["sealed_cases"] == len(seals) == 24
        and result["archive_set_commitment_sha256"] == archive_set_commitment(seals)
        and seals[CONDITION]["receipt_sha256"] == OLD_SEAL_SHA,
        "Historical seal is not the exact member of the published 24-case result",
    )
    archive, receipt = raw.archive_paths(CONDITION)
    sealed = read(raw.direct_file(receipt))
    if full_archive:
        archives.verify_archive(archive, sealed)
    stage, previous = raw.authored(CONDITION), raw.TRIAL / CONDITION
    entries = {entry["path"]: entry for entry in sealed["entries"] if entry["kind"] == "file"}
    for name in ("inputs.json", "freeze.json", f"{ARM}.yaml", "audit.json", "profile.json", "worker.log"):
        require(
            sha(raw.direct_file(stage / name)) == entries[f"stage/{name}"]["sha256"],
            "Historical stage metadata differs from the sealed archive",
        )
    require(
        sha(stage / "freeze.json") == OLD_FREEZE_SHA
        and sha(stage / f"{ARM}.yaml") == old["app_manifest_sha256"]
        and all(
            sha(raw.direct_file(previous / name)) == sha(stage / name)
            for name in ("inputs.json", "freeze.json", f"{ARM}.yaml")
        ),
        "Historical worker inputs or manifest differ",
    )
    plan = read(raw.direct_file(raw.TRIAL / "private-plan.json"))
    row, audit = read(stage / "inputs.json"), read(stage / "audit.json")
    profile = read(stage / "profile.json")
    require(
        hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest() == old["plan_sha256"]
        and plan[CONDITION] == row
        and row["slot"] == 10
        and row["role"] == raw.ROLES[1]
        and row["status"] == "prepared"
        and audit["status"] == "scored"
        and audit["condition"] == CONDITION
        and audit["role"] == row["role"]
        and audit["legacy_frame_status"] == audit["appearance_route_status"] == "failed"
        and audit["customer_direction"] == "unavailable"
        and audit["appearance_measured_windows"] == 0
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and collision_log((stage / "worker.log").read_text()),
        "The exact historical failed routes are required",
    )
    require(
        has_uploaded_bytes(entries.values(), row["bytes"], row["sha256"]),
        "Exact prior uploaded bytes must be present in the seal",
    )
    return (
        old,
        row,
        audit,
        {
            "historical_result_sha256": OLD_RESULT_SHA,
            "historical_freeze_sha256": OLD_FREEZE_SHA,
            "historical_sealed_cases": len(seals),
            "historical_archive_set_sha256": result["archive_set_commitment_sha256"],
            "historical_case_receipt_sha256": OLD_SEAL_SHA,
            "historical_case_archive_sha256": sealed["archive_sha256"],
        },
    )


def same_input(row, old):
    preparation = raw.direct_file(raw.PREPARATION / "freeze.json")
    require(
        sha(preparation) == old["preparation_freeze_sha256"]
        and sha(raw.SOURCE_FREEZE) == old["roster_freeze_sha256"],
        "Exact historical preparation and roster required",
    )
    tools = read(preparation)["identity"]["tool_sha256"]
    require(
        all(
            shutil.which(name) and sha(Path(shutil.which(name)).resolve()) == digest
            for name, digest in tools.items()
        ),
        "Exact source-preparation decoder binaries required",
    )
    roster = raw.prepared_roster()
    folder = raw.PREPARATION / f"slot-{row['slot']:02d}-attempt-01"
    qualification = raw.direct_file(folder / "qualification.json")
    document = read(qualification)
    raw.qualify_slot(roster[row["slot"] - 1], document)
    file = raw.direct_file(folder / f"{row['role']}.mp4")
    require(
        sha(qualification) == row["qualification_sha256"]
        and document["cases"][row["role"]]["sha256"] == row["sha256"]
        and file.stat().st_size == row["bytes"]
        and sha(file) == row["sha256"],
        "Prepared input no longer matches the sealed original",
    )
    return file


def context(*, patched, full_archive=False):
    old, row, audit, previous = historical(full_archive=full_archive)
    runtime = runtime_bindings(old, patched=patched)
    file = same_input(row, old)
    return (
        {
            "schema": "chipmunk.video-vcd2-verify-cache-replay-freeze/251",
            **previous,
            **runtime,
            "planned_cases": 1,
            "private_input_record_sha256": sha(raw.authored(CONDITION) / "inputs.json"),
            "only_application_source_change": {
                "path": FETCHERS,
                "before_sha256": OLD_FETCHERS,
                "after_sha256": PATCHED_FETCHERS,
            },
            "code_sha256": {name: sha(ROOT / name) for name in CODE},
            "customer_settings_changed": False,
        },
        row,
        file,
        audit,
    )


def preflight():
    require(
        not any(path.exists() for path in (TRIAL, WORK, SEALED, PUBLIC)),
        "Preserve any earlier replay attempt",
    )
    frozen, row, _, audit = context(patched=False, full_archive=True)
    free, needed = shutil.disk_usage(BASE).free, raw.disk_required(row["bytes"])
    return {
        "status": "ready_for_prospective_freeze" if free >= needed else "pending_disk_headroom",
        "historical_case": statuses(audit),
        "historical_sealed_cases": frozen["historical_sealed_cases"],
        "historical_archive_verified": True,
        "available_disk_bytes": free,
        "required_disk_bytes": needed,
        "model_calls": 0,
    }


def freeze():
    require(
        not any(path.exists() for path in (TRIAL, WORK, SEALED, PUBLIC)),
        "Preserve any earlier replay attempt",
    )
    frozen, row, _, _ = context(patched=False, full_archive=True)
    TRIAL.mkdir(mode=0o700)
    WORK.mkdir(mode=0o700)
    stage = TRIAL / CONDITION
    stage.mkdir(mode=0o700)
    write(TRIAL / "private-plan.json", {CONDITION: row})
    for name in ("inputs.json", f"{ARM}.yaml"):
        with (stage / name).open("xb") as stream:
            stream.write((raw.authored(CONDITION) / name).read_bytes())
    write(stage / "freeze.json", frozen)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return {"status": "frozen_before_patch", "freeze_sha256": sha(PUBLIC), "planned_cases": 1}


def verify(condition=CONDITION):
    require(condition == CONDITION, "One fixed historical case required")
    frozen, row, file, _ = context(patched=True)
    stage = TRIAL / condition
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == read(stage / "freeze.json") == frozen
        and read(TRIAL / "private-plan.json") == {condition: row}
        and read(stage / "inputs.json") == row
        and sha(stage / "inputs.json") == frozen["private_input_record_sha256"]
        and sha(stage / f"{ARM}.yaml") == frozen["model_manifest_sha256"],
        "Exact patched replay freeze required",
    )
    return frozen, file, {"file_sha256": row["sha256"], "bytes": row["bytes"]}


def authored(condition=CONDITION):
    return TRIAL / condition / ARM / "authored"


def work_dir(condition=CONDITION):
    return WORK / condition / ARM


def audit_path(condition=CONDITION):
    return authored(condition) / "audit.json"


def archive_paths(condition=CONDITION):
    archive = SEALED / f"{condition}.tar"
    return archive, archive.with_suffix(".tar.receipt.json")


def active_case():
    return scoped(raw.active_case, {"TRIAL": TRIAL})()


def worker(output):
    return scoped(raw.worker, {"TRIAL": TRIAL, "WORK": WORK, "verify": verify})(CONDITION, output)


def qualification(condition):
    return scoped(raw.qualification, {"TRIAL": TRIAL, "WORK": WORK, "__file__": __file__})(condition)


def run():
    _, _, bound = verify()
    with active_case():
        require(not work_dir().exists() and not authored().exists(), "Preserve any existing replay attempt")
        free, needed = shutil.disk_usage(BASE).free, raw.disk_required(bound["bytes"])
        if free < needed:
            return {
                "status": "pending_disk_headroom",
                "available_disk_bytes": free,
                "required_disk_bytes": needed,
            }
        stage = authored()
        stage.parent.mkdir(exist_ok=True)

        def launch(argv, **kwargs):
            output = stage / "output"
            require(
                argv[3:] == ["scripts.video_appearance_native_v108", "authored", str(output)],
                "Exact replay worker invocation required",
            )
            return subprocess.Popen([*argv[:3], MODULE, "_worker", str(output)], **kwargs)

        scoped(
            raw.prior.original.runtime.run,
            {
                "TRIAL": stage.parent,
                "verify": verify,
                "prerequisites": lambda *_: True,
                "__file__": __file__,
                "subprocess": SimpleNamespace(
                    Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
                ),
            },
        )("authored")
    return None


def audit_case():
    raw.edge.enforce_read_only()
    return scoped(raw.audit_case, {"TRIAL": TRIAL, "verify": verify, "qualification": qualification})(
        CONDITION
    )


def review():
    return scoped(
        raw.review,
        {"TRIAL": TRIAL, "verify": verify, "authored": authored, "audit_path": audit_path, "MODULE": MODULE},
    )(CONDITION)


def seal():
    return scoped(
        raw.seal,
        {
            "TRIAL": TRIAL,
            "SEALED": SEALED,
            "verify": verify,
            "authored": authored,
            "audit_path": audit_path,
            "archive_paths": archive_paths,
            "active_case": active_case,
            "work_dir": work_dir,
        },
    )(CONDITION)


def release():
    return scoped(
        raw.release,
        {
            "verify": verify,
            "authored": authored,
            "archive_paths": archive_paths,
            "active_case": active_case,
            "work_dir": work_dir,
        },
    )(CONDITION)


def readout():
    frozen, _, bound = verify()
    _, _, before, _ = historical(full_archive=True)
    archive, receipt = archive_paths()
    sealed = read(raw.direct_file(receipt))
    archives.verify_archive(archive, sealed)
    stage = authored()
    entries = {entry["path"]: entry for entry in sealed["entries"] if entry["kind"] == "file"}
    require(
        sealed["stage_path"] == str(stage)
        and all(
            sha(raw.direct_file(stage / name)) == entries[f"stage/{name}"]["sha256"]
            for name in (
                "audit.json",
                "profile.json",
                "worker.log",
                "inputs.json",
                "freeze.json",
                f"{ARM}.yaml",
            )
        )
        and read(stage / "freeze.json") == frozen
        and sha(stage / "inputs.json") == frozen["private_input_record_sha256"]
        and sha(stage / f"{ARM}.yaml") == frozen["model_manifest_sha256"],
        "Exact replay seal and bindings required",
    )
    after = read(stage / "audit.json")
    require(
        after["condition"] == CONDITION and after["role"] == raw.ROLES[1],
        "Replay audit names a different case",
    )
    same_bytes = has_uploaded_bytes(entries.values(), bound["bytes"], bound["file_sha256"])
    require(after["status"] != "scored" or same_bytes, "Completed replay seal lacks the exact uploaded input")
    collision = collision_log((stage / "worker.log").read_text())
    return {
        "schema": "chipmunk.video-vcd2-verify-cache-replay-result/251",
        "status": "verified_sealed_replay",
        "freeze_sha256": sha(PUBLIC),
        "historical_sealed_cases": frozen["historical_sealed_cases"],
        "historical_case_receipt_sha256": OLD_SEAL_SHA,
        "replay_receipt_sha256": sha(receipt),
        "same_sealed_input": same_bytes,
        "historical": statuses(before),
        "replay": statuses(after),
        "historical_cache_collision": True,
        "replay_cache_collision": collision,
        "affected_routes_recovered": after["status"] == "scored"
        and not collision
        and after.get("legacy_frame_status") == after.get("appearance_route_status") == "ok",
        "customer_settings_changed": False,
        "historical_aggregate_replaced": False,
    }


def main():
    commands = {
        "preflight": preflight,
        "freeze": freeze,
        "run": run,
        "review": review,
        "seal": seal,
        "release": release,
        "readout": readout,
        "_audit": audit_case,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*commands, "_worker"])
    parser.add_argument("argument", nargs="?")
    args = parser.parse_args()
    if args.command == "_worker":
        # The inherited review wrapper passes a fixed condition; the worker receives only its output path.
        require(args.argument is not None, "Exact worker output required")
        return worker(Path(args.argument))
    require(
        args.argument is None or (args.command == "_audit" and args.argument == CONDITION),
        "One fixed replay case required",
    )
    value = commands[args.command]()
    if value is not None:
        print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
