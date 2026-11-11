"""Keep the frozen full-profile serving run on a hard-link-capable work volume."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

from scripts import grid_serving_baseline_v153 as bootstrap
from scripts.video_appearance_corrected_v112 import scoped

baseline = bootstrap.baseline
ROOT, BASE, PANEL = bootstrap.ROOT, bootstrap.BASE, bootstrap.PANEL
TRIAL = BASE / "grid-serving-baseline-v154"
WORK = BASE / "grid-serving-work-v154"
PUBLIC = PANEL / "grid-serving-baseline-freeze-v154.json"
SYSTEM_DATA = Path("/System/Volumes/Data")
CODE = (
    "backend/scripts/grid_serving_baseline_v154.py",
    "backend/tests/test_grid_serving_baseline_v154.py",
    "docs/grid-serving-baseline-v154-method.md",
)
FREEZE_SCHEMA = "chipmunk.grid-serving-baseline-freeze/154"
RESULT_SCHEMA = "chipmunk.grid-serving-baseline-result/154"
SAMPLE = b"Chipmunk Verify v154 synthetic object-store hard-link smoke\n"
read, write, sha, require = bootstrap.read, bootstrap.write, bootstrap.sha, bootstrap.require


def work_volume():
    """Require the same system APFS data volume the existing monitor samples."""
    target = WORK if WORK.exists() else WORK.parent
    device = target.stat().st_dev
    require(
        WORK.resolve() == WORK
        and target.is_dir()
        and SYSTEM_DATA.stat().st_dev == TRIAL.parent.stat().st_dev == device,
        "Working cache must be on the monitored internal system data volume",
    )
    mounts = subprocess.run(["/sbin/mount"], capture_output=True, text=True, check=True).stdout
    require(
        any(f" on {SYSTEM_DATA} (apfs, local," in line for line in mounts.splitlines()),
        "Internal system data volume must be local APFS",
    )
    return {"filesystem": "apfs", "system_data_volume": True, "device": device}


def capture_link(event, args, objects, linked):
    if event != "os.link":
        return
    source, target = (Path(os.fsdecode(path)) for path in args[:2])
    if not target.is_relative_to(objects):
        return
    require(source.is_relative_to(objects / "tmp"), "Unexpected object-store link source")
    info = source.stat()
    linked.append((info.st_dev, info.st_ino, target))


def storage_smoke(device):
    volume = work_volume()
    require(shutil.disk_usage(WORK.parent).free >= 3 * 1024**3, "Unchanged initial disk reserve required")
    WORK.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".object-store-smoke-", dir=WORK) as directory:
        folder = Path(directory)
        context = bootstrap.initialize(folder, device)
        from app.storage.objects import ObjectStore

        objects = context.settings.objects_dir
        store = ObjectStore(objects)
        linked = []
        sys.addaudithook(lambda event, args: capture_link(event, args, objects, linked))
        created = store.put_bytes(SAMPLE)
        final = created.path.stat()
        digest = hashlib.sha256(SAMPLE).hexdigest()
        require(
            created.created is True
            and created.sha256 == digest
            and created.size_bytes == len(SAMPLE)
            and store.verify(digest) == len(SAMPLE)
            and store.read_bytes(digest) == SAMPLE
            and linked == [(volume["device"], final.st_ino, created.path)]
            and final.st_dev == volume["device"]
            and final.st_nlink == 1,
            "Production ObjectStore must publish the exact bytes by a verified same-inode hard link",
        )
        return {
            **volume,
            "schema": "chipmunk.grid-serving-storage-smoke/154",
            "bytes": len(SAMPLE),
            "sha256": digest,
            "same_inode_hard_link": True,
            "production_object_store_round_trip": True,
        }


def failure_history(context):
    prior, saved = bootstrap.verify_initialized(context)
    cases, upstream = bootstrap.bind_case(context, 0)
    stage = bootstrap.stage_folder(0)
    receipt = baseline.speech.qualification_profile(stage, None, {}, "grid_serving_baseline_v153.py")
    folder = bootstrap.CACHE / "clip-00"
    case_journal = folder / "cases.jsonl"
    artifacts = {
        "cases.jsonl": "private_cases_sha256",
        "planned-cases.json": "private_planned_cases_sha256",
        "runtime.json": "private_runtime_sha256",
        "upstream.json": "private_upstream_sha256",
    }
    plan = baseline.planned_cases(cases)
    require(
        receipt["schema"] == bootstrap.RESULT_SCHEMA
        and receipt["freeze_sha256"] == sha(bootstrap.PUBLIC)
        and receipt["baseline_freeze_sha256"] == prior["baseline_freeze_sha256"]
        and receipt["clip_ordinal"] == 0
        and receipt["profile"] == prior["profile"]
        and receipt["settings_initialized_before_metadata"] is True
        and all(receipt[key] == sha(folder / filename) for filename, key in artifacts.items())
        and read(folder / "planned-cases.json") == plan
        and read(folder / "upstream.json") == upstream,
        "Exact v153 failed work and actual completed monitor receipt required",
    )
    rows = baseline.planning.reviewed.lines(case_journal)
    log = stage / "worker.log"
    lines = log.read_text()
    require(
        len(rows) == 11
        and all(
            row["status"] == "failed"
            and not row["detector_runs"]
            and row["condition"] is None
            and any(item["stage"] == "ingest" and item["status"] == "failed" for item in row["stages"])
            for row in rows
        )
        and "os.link(tmp, final)" in lines
        and "[Errno 45] Operation not supported" in lines,
        "Preserve the actual v153 ingest failures and zero detector runs",
    )
    # The old app condition payload overwrote the label. Check order and IDs against
    # the sealed plan while retaining and hashing the exact original rows.
    matched = [{**row, "condition": spec["condition"]} for row, spec in zip(rows, plan, strict=True)]
    registry = read(baseline.TRIAL / "private-inventory.json")["registry"]
    baseline.validate_case_rows(matched, cases, registry)
    require(
        all(receipt[key] == value for key, value in baseline.count_results(rows).items()),
        "Historical v153 receipt must reconcile with the actual failed outcomes",
    )
    return saved, {
        "v153_freeze_sha256": sha(bootstrap.PUBLIC),
        "v153_result_sha256": sha(stage / "output/result.json"),
        "v153_profile_sha256": sha(stage / "profile.json"),
        "v153_resource_journal_sha256": sha(stage / "resources.jsonl"),
        "v153_worker_log_sha256": sha(log),
        "v153_case_journal_sha256": sha(case_journal),
        "failed_ingest_cases": len(rows),
        "detector_runs": 0,
        "cause": "Seagate exFAT does not support the ObjectStore hard link",
        "null_case_labels_after_field_collision": len(rows),
    }


def identity(saved, history, smoke):
    return {
        "schema": FREEZE_SCHEMA,
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "baseline_freeze_sha256": sha(baseline.PUBLIC),
        "baseline_asset_inventory_sha256": saved["asset_inventory_sha256"],
        "historical_v153_failure": history,
        "storage_smoke_sha256": sha(TRIAL / "storage-smoke.json"),
        "work_filesystem": smoke["filesystem"],
        "configured_device": saved["configured_device"],
        "profile": saved["profile"],
        "pilot_clip_ordinals": saved["pilot_clip_ordinals"],
        "cases_per_clip": saved["cases_per_clip"],
        "scope": "Same frozen detectors, assets, early offline settings and resource monitor. Isolated app working storage is on system APFS.",
    }


def verify_initialized(context):
    saved, history = failure_history(context)
    smoke = read(TRIAL / "storage-smoke.json")
    volume = work_volume()
    require(
        all(smoke[key] == value for key, value in volume.items())
        and smoke["schema"] == "chipmunk.grid-serving-storage-smoke/154"
        and smoke["same_inode_hard_link"] is smoke["production_object_store_round_trip"] is True
        and smoke["bytes"] == len(SAMPLE)
        and smoke["sha256"] == hashlib.sha256(SAMPLE).hexdigest(),
        "Successful ObjectStore smoke on the same monitored APFS volume required",
    )
    frozen = identity(saved, history, smoke)
    require(frozen == read(PUBLIC) == read(TRIAL / "freeze.json"), "Unchanged v154 context required")
    return frozen, saved


def stage_folder(ordinal):
    return TRIAL / f"clip-{ordinal:02d}/authored"


def read_cases(context, ordinal, frozen):
    require(type(ordinal) is int and ordinal in frozen["pilot_clip_ordinals"], "Fixed pilot ordinal required")
    cases, upstream = bootstrap.bind_case(context, ordinal)
    stage = stage_folder(ordinal)
    result = baseline.speech.qualification_profile(stage, None, {}, "grid_serving_baseline_v154.py")
    folder = WORK / f"clip-{ordinal:02d}"
    require(folder.resolve() == folder, "Exact owned working cache path required")
    artifacts = {
        "cases.jsonl": "private_cases_sha256",
        "planned-cases.json": "private_planned_cases_sha256",
        "runtime.json": "private_runtime_sha256",
        "upstream.json": "private_upstream_sha256",
    }
    require(
        result["schema"] == RESULT_SCHEMA
        and result["clip_ordinal"] == ordinal
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["baseline_freeze_sha256"] == frozen["baseline_freeze_sha256"]
        and result["storage_smoke_sha256"] == frozen["storage_smoke_sha256"]
        and result["settings_initialized_before_metadata"] is True
        and result["profile"] == frozen["profile"]
        and result["configured_device"] == frozen["configured_device"]
        and result["defaults_changed"]
        is result["candidate_added"]
        is result["billing_or_service_started"]
        is False
        and all(result[key] == sha(folder / name) for name, key in artifacts.items())
        and read(folder / "planned-cases.json") == baseline.planned_cases(cases)
        and read(folder / "upstream.json") == upstream
        and read(folder / "runtime.json")["resolved_device"] == result["resolved_device"],
        "Actual bounded v154 receipt and private app work records required",
    )
    rows = baseline.planning.reviewed.lines(folder / "cases.jsonl")
    baseline.validate_case_rows(rows, cases, read(baseline.TRIAL / "private-inventory.json")["registry"])
    require(
        all(result[key] == value for key, value in baseline.count_results(rows).items()),
        "Serving outcomes must reconcile with the matched case roster",
    )
    return rows


def metadata_entry(action, ordinal):
    with contextlib.redirect_stdout(sys.stderr):
        device = bootstrap.header()["configured_device"]
        if action == "smoke":
            return storage_smoke(device)
        context = bootstrap.initialize(WORK / "metadata-only", device)
        if action == "history":
            return failure_history(context)
        frozen, _ = verify_initialized(context)
        if action == "verify":
            return frozen
        ordinals = frozen["pilot_clip_ordinals"]
        require(type(ordinal) is int and ordinal in ordinals, "Fixed pilot ordinal required")
        if action == "load":
            return read_cases(context, ordinal, frozen)
        require(action == "prerequisite", "Exact metadata command required")
        previous = ordinals.index(ordinal) - 1
        if previous >= 0:
            read_cases(context, ordinals[previous], frozen)
        bootstrap.bind_case(context, ordinal)
        return True


def metadata(action, ordinal=None):
    args = [sys.executable, "-B", "-m", "scripts.grid_serving_baseline_v154", "_metadata", action]
    if ordinal is not None:
        args.append(str(ordinal))
    env = {
        key: os.environ[key] for key in ("HOME", "PATH", "LANG", "LC_ALL", "LC_CTYPE") if key in os.environ
    }
    env["PYTHONPATH"] = str(ROOT / "backend")
    response = subprocess.run(
        args, cwd=ROOT / "backend", env=env, capture_output=True, text=True, check=False
    )
    if response.returncode:
        raise RuntimeError(response.stderr.strip() or "Isolated v154 metadata failed")
    return json.loads(response.stdout)


def load_completed(ordinal):
    require(type(ordinal) is int, "Fixed pilot ordinal required")
    return metadata("load", ordinal)


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every v154 attempt and freeze")
    saved, history = metadata("history")
    smoke = metadata("smoke")
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "storage-smoke.json", smoke)
    frozen = identity(saved, history, smoke)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"status": "frozen", "freeze_sha256": sha(PUBLIC)}), flush=True)


def analyze_case(*args, **kwargs):
    result = baseline.analyze_case(*args, **kwargs)
    result["condition_profile"] = result.pop("condition")
    return result


def worker_entry(ordinal, output):
    saved = bootstrap.header()
    stage = stage_folder(ordinal)
    require(
        ordinal in saved["pilot_clip_ordinals"]
        and output.resolve() == (stage / "output").resolve()
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists(),
        "Exclusive guarded v154 output required",
    )
    folder = WORK / f"clip-{ordinal:02d}"
    require(folder.resolve() == folder and not folder.exists(), "Exact unused APFS working path required")
    context = bootstrap.initialize(folder, saved["configured_device"])
    frozen, prior = verify_initialized(context)
    output.mkdir(exist_ok=False)

    def write_bound(path, data):
        if path == output / "result.json":
            data["storage_smoke_sha256"] = frozen["storage_smoke_sha256"]
        write(path, data)

    worker = scoped(
        bootstrap.corrected_worker,
        {
            "TRIAL": TRIAL,
            "CACHE": WORK,
            "PUBLIC": PUBLIC,
            "RESULT_SCHEMA": RESULT_SCHEMA,
            "write": write_bound,
        },
    )(context, output, frozen)
    scoped(worker, {"analyze_case": analyze_case})(ordinal, output, prior)


def prerequisites(ordinal, ordinals):
    index = ordinals.index(ordinal)
    stages = [baseline.clips.TRIAL / f"clip-{ordinal:02d}/bind-00"]
    if index:
        stages.append(stage_folder(ordinals[index - 1]))
    return all(
        (stage / name).is_file() for stage in stages for name in ("profile.json", "output/result.json")
    ) and metadata("prerequisite", ordinal)


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime

    ordinals = metadata("verify")["pilot_clip_ordinals"]
    require(type(ordinal) is int and ordinal in ordinals, "Fixed pilot ordinal required")
    stage = stage_folder(ordinal)
    stage.parent.mkdir(exist_ok=True)

    def popen(argv, **kwargs):
        output = stage / "output"
        require(
            argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"]
            and argv[5:] == [str(output)],
            "Only the exclusive v154 full-profile worker is permitted",
        )
        return subprocess.Popen(
            [*argv[:3], "scripts.grid_serving_baseline_v154", "_worker", str(ordinal), str(output)], **kwargs
        )

    scoped(
        runtime.run,
        {
            "TRIAL": stage.parent,
            "verify": lambda: metadata("verify"),
            "prerequisites": lambda _c, _b: prerequisites(ordinal, ordinals),
            "subprocess": types.SimpleNamespace(
                Popen=popen,
                STDOUT=subprocess.STDOUT,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
            "__file__": str(Path(__file__)),
        },
    )("authored")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("run").add_argument("ordinal", type=int)
    worker = commands.add_parser("_worker")
    worker.add_argument("ordinal", type=int)
    worker.add_argument("output", type=Path)
    reader = commands.add_parser("_metadata")
    reader.add_argument("action", choices=("history", "smoke", "verify", "prerequisite", "load"))
    reader.add_argument("ordinal", type=int, nargs="?")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "run":
        run(args.ordinal)
    elif args.command == "_worker":
        worker_entry(args.ordinal, args.output)
    else:
        print(json.dumps(metadata_entry(args.action, args.ordinal), allow_nan=False))


if __name__ == "__main__":
    main()
