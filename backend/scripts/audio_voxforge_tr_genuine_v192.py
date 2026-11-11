"""Prepare the v188 Turkish roster with the audited per-session license notices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import types
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_voxforge_tr_genuine_v188 as source
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE = source.ROOT, source.BASE
TRIAL = BASE / "audio-voxforge-tr-genuine-v192"
STAGE = TRIAL / "bind-00"
OUTPUT = STAGE / "output"
PUBLIC = ROOT / "docs/evaluation-panels/audio-voxforge-tr-genuine-freeze-v192.json"
MODULE = "scripts.audio_voxforge_tr_genuine_v192"
CODE = (
    "backend/scripts/audio_voxforge_tr_genuine_v192.py",
    "backend/tests/test_audio_voxforge_tr_genuine_v192.py",
    "docs/audio-voxforge-tr-genuine-v192-method.md",
)
STOCK_LICENSES = frozenset(
    {
        "38cb1cce91733ded27d417c5d58862ed7e3785847878de213afa43e34e35eaf1",
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
    }
)
STOCK_NORMALIZED = "13e61ba5071e671e299fb8dec45a6786c3c6aa4efef1dc3c7c772f16dfb69cf3"
PRIOR_REASON = "Conflicting per-session noncommercial or research-only terms"
read, write, sha, require = source.read, source.write, source.sha, source.require


def session_metadata(archive, files, contributor):
    stock = [path for path in files if path.name.casefold() == "gpl_license.txt"]
    notice = [path for path in files if path.name.casefold() == "license"]
    require(len(stock) == len(notice) == 1, "Separate exact GPL reference and session LICENSE required")
    reference = source.member_bytes(archive, files[stock[0]], source.MAX_TEXT)
    reference_sha = source.digest(reference)
    require(
        reference_sha in STOCK_LICENSES
        and source.digest(source.norm(reference.decode("utf-8-sig")).encode()) == STOCK_NORMALIZED,
        "Exact audited stock GPL reference required",
    )
    # The stock GPL text is an attached reference. The per-session grant
    # and all other metadata still pass the original strict checks.
    prompts, result = source.metadata(
        archive, {path: member for path, member in files.items() if path != stock[0]}, contributor
    )
    require(
        result["license"] == "GPL-3.0-or-later"
        and result["license_receipt"]["member"] == notice[0].as_posix(),
        "Exact explicit per-session GPL-3.0-or-later notice required",
    )
    return prompts, {
        **result,
        "stock_gpl_reference": {"member": stock[0].as_posix(), "sha256": reference_sha},
    }


def prior():
    plan = source.verify()
    archives, acquisition = source.acquired(plan)
    preparation = source.readout()
    journal = source.OUTPUT / "private-attempts.json"
    attempts = read(journal)
    require(
        preparation["case_outcomes"] == {"unavailable": source.GROUPS}
        and preparation["attempt_outcomes"] == {"session_unavailable": 15}
        and preparation["new_neural_calls"] == 0
        and preparation["holdout_media_opened"] is preparation["holdout_scored"] is False
        and len(archives) == len(attempts) == 15
        and all(row["status"] == "downloaded" for row in archives)
        and [(row["slot"], row["archive"]) for row in archives]
        == [(row["slot"], row["archive"]) for row in attempts]
        and all(
            row["status"] == "session_unavailable" and row["private_reason"] == PRIOR_REASON
            for row in attempts
        ),
        "Preserved exact pre-decode v188 license-parser outcome required",
    )
    return plan, archives, acquisition


def audit(plan, archives):
    groups = {row["slot"]: row["contributor"] for row in plan}
    licenses = Counter()
    references = set()
    for row in archives:
        with source.verify_archive(row) as archive:
            _, evidence = session_metadata(archive, source.regular_members(archive), groups[row["slot"]])
        licenses[evidence["license"]] += 1
        references.add(evidence["stock_gpl_reference"]["sha256"])
    return {
        "archives_with_declared_turkish_and_named_contributor": len(archives),
        "archives_with_explicit_session_grants": dict(licenses),
        "stock_gpl_reference_variants": len(references),
        "audio_members_opened": 0,
        "new_neural_calls": 0,
    }


def context():
    plan, archives, acquisition = prior()
    return (
        {
            "schema": "chipmunk.audio-voxforge-tr-genuine-freeze/192",
            "code_sha256": {path: sha(ROOT / path) for path in CODE},
            "prior_preparation_freeze_sha256": sha(source.PUBLIC),
            "prior_preparation_result_sha256": sha(source.OUTPUT / "result.json"),
            "prior_preparation_profile_sha256": sha(source.STAGE / "profile.json"),
            "prior_private_attempts_sha256": sha(source.OUTPUT / "private-attempts.json"),
            "private_acquisition_sha256": acquisition["private_acquisition_sha256"],
            "stock_gpl_reference_sha256": sorted(STOCK_LICENSES),
            "stock_gpl_normalized_sha256": STOCK_NORMALIZED,
            "planned_cases": len(plan),
            "planned_session_archives": len(archives),
            "language": "tr",
            "origin": "genuine",
            "model_calls_permitted": False,
            "holdout_access_permitted": False,
        },
        plan,
        archives,
    )


def preflight():
    _, plan, archives = context()
    return audit(plan, archives)


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier attempts")
    lock, plan, archives = context()
    audit(plan, archives)
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(json.dumps({"status": "frozen", "planned_cases": lock["planned_cases"], "model_calls": 0}))


def verify():
    lock, plan, archives = context()
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact corrective preparation freeze required"
    )
    return plan, archives


def aggregation(rows, attempts, plan, archives):
    require(
        [row["slot"] for row in rows] == [row["slot"] for row in plan]
        and all(
            row["status"] in {"qualified", "unavailable"} and row["language"] == "tr" and row["label"] == 0
            for row in rows
        ),
        "Exact unchanged roster and ordered case outcomes required",
    )
    acquired = {(row["slot"], row["archive"]): row["sha256"] for row in archives}
    qualified = [row for row in rows if row["status"] == "qualified"]
    require(
        all(
            acquired.get((row["slot"], row["archive"])) == row["archive_sha256"]
            and row["license"] == "GPL-3.0-or-later"
            and row["stock_gpl_reference"]["sha256"] in STOCK_LICENSES
            for row in qualified
        )
        and all(
            len({row[key] for row in qualified}) == len(qualified)
            for key in ("prompt_sha256", "source_sha256", "pcm_sha256")
        ),
        "Selected clips must use the acquired session with an explicit grant and unique prompt and PCM",
    )
    return {
        "planned_cases": len(plan),
        "case_outcomes": dict(Counter(row["status"] for row in rows)),
        "attempt_outcomes": dict(Counter(row["status"] for row in attempts)),
        "qualified_licenses": dict(Counter(row["license"] for row in qualified)),
    }


def prepare_worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded data-only preparation required",
    )
    plan, archives = verify()
    OUTPUT.mkdir(mode=0o700, exist_ok=False)
    prepared = OUTPUT / "prepared"
    prepared.mkdir(mode=0o700)
    prepare_slot = scoped(source.prepare_slot, {"metadata": session_metadata})
    rows, attempts, prompts = [], [], set()
    known = set(read(source.HISTORY)["keys"])
    with TemporaryDirectory(prefix="voxforge-genuine-v192-") as tmp:
        for group in plan:
            candidates = [row for row in archives if row["slot"] == group["slot"]]
            rows.append(prepare_slot(group, candidates, prompts, known, Path(tmp), prepared, attempts))
    observations, journal = OUTPUT / "private-cases.json", OUTPUT / "private-attempts.json"
    write(observations, rows)
    write(journal, attempts)
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-voxforge-tr-genuine-result/192",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_cases_sha256": sha(observations),
            "private_attempts_sha256": sha(journal),
            **aggregation(rows, attempts, plan, archives),
            "language": "tr",
            "origin": "genuine",
            "new_neural_calls": 0,
            "holdout_media_opened": False,
            "holdout_scored": False,
        },
    )


def prepare():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
            "Exact monitored v108 data-only invocation required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "prepare-worker"], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
        },
    )("bind", 0)


def read_prepared_pcm(row):
    return scoped(source.read_prepared_pcm, {"OUTPUT": OUTPUT})(row)


def readout():
    plan, archives = verify()
    result_path, profile_path = OUTPUT / "result.json", STAGE / "profile.json"
    resource_path = STAGE / "resources.jsonl"
    cases, journal = OUTPUT / "private-cases.json", OUTPUT / "private-attempts.json"
    result, profile = read(result_path), read(profile_path)
    rows, attempts = read(cases), read(journal)
    samples = [json.loads(line) for line in resource_path.read_text().splitlines()]
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == profile["resource_limit_observations"] == []
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(resource_path)
        and samples
        and all(
            not item["violations"]
            and not runtime.violations(item, "bind", initial=i == 0)
            and item["archive_free_disk_bytes"] >= (3 * 1024**3 if i == 0 else 1536 * 1024**2)
            for i, item in enumerate(samples)
        )
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(item["tree_rss_bytes"] for item in samples)
        and samples[-1]["elapsed_s"] <= profile["duration_s"] <= 900
        and result["schema"] == "chipmunk.audio-voxforge-tr-genuine-result/192"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_cases_sha256"] == sha(cases)
        and result["private_attempts_sha256"] == sha(journal)
        and all(result[key] == value for key, value in aggregation(rows, attempts, plan, archives).items())
        and result["language"] == "tr"
        and result["origin"] == "genuine"
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and result["holdout_media_opened"] is result["holdout_scored"] is False,
        "Exact completed corrective data-only guard required",
    )
    for row in rows:
        if row["status"] == "qualified":
            read_prepared_pcm(row)
    return {
        **result,
        "resource_profile_sha256": sha(profile_path),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "prepare", "prepare-worker", "readout"))
    command = parser.parse_args().command
    if command in {"preflight", "readout"}:
        print(json.dumps({"preflight": preflight, "readout": readout}[command](), indent=2, allow_nan=False))
    else:
        {"freeze": freeze, "prepare": prepare, "prepare-worker": prepare_worker}[command]()
