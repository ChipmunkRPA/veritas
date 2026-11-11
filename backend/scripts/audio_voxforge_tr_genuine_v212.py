"""Freeze and prepare 16 additional genuine Turkish read-speech groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_voxforge_tr_genuine_v199 as prior
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, RATE, LISTING = prior.ROOT, prior.BASE, prior.RATE, prior.LISTING
NAME = "audio-voxforge-tr-genuine-v212"
MODULE = "scripts.audio_voxforge_tr_genuine_v212"
TRIAL = BASE / NAME
STAGE = TRIAL / "bind-00"
OUTPUT = STAGE / "output"
PANELS = ROOT / "docs/evaluation-panels"
PUBLIC = PANELS / "audio-voxforge-tr-genuine-freeze-v212.json"
V199_COMPLETE = PANELS / "audio-voxforge-tr-genuine-complete-v199.json"
RESULT_SCHEMA = "chipmunk.audio-voxforge-tr-genuine-result/212"
GROUPS = 16
MIN_SAMPLES, MAX_SAMPLES = prior.MIN_SAMPLES, prior.MAX_SAMPLES
read, write, sha, require, digest = prior.read, prior.write, prior.sha, prior.require, prior.digest


def rank(value):
    return digest(f"chipmunk.audio-voxforge-tr/212:{prior.source.LISTING_SHA}:{value}".encode())


def prior_controls():
    earlier_plan, earlier_qualified, _ = prior.prior_metadata()
    recent_plan = prior.verify()
    require(
        len(earlier_plan) == 12
        and len(recent_plan) == 24
        and len({row["contributor"] for row in [*earlier_plan, *recent_plan]}) == 36,
        "Exact disjoint v188 and v199 contributor metadata required",
    )
    cases_path, result_path = prior.OUTPUT / "private-cases.json", prior.OUTPUT / "result.json"
    profile_path = prior.STAGE / "profile.json"
    published, result, profile, rows = map(read, (V199_COMPLETE, result_path, profile_path, cases_path))
    require(
        all(published.get(key) == value for key, value in result.items())
        and result["schema"] == "chipmunk.audio-voxforge-tr-genuine-result/199"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(prior.PUBLIC)
        and result["private_cases_sha256"] == sha(cases_path)
        and result["new_neural_calls"] == 0
        and result["holdout_media_opened"] is result["holdout_scored"] is False
        and published["resource_profile_sha256"] == sha(profile_path)
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and [row["slot"] for row in rows] == [row["slot"] for row in recent_plan],
        "Exact v199 guarded cases metadata required without opening earlier audio",
    )
    recent_qualified = [row for row in rows if row["status"] == "qualified"]
    require(
        len(earlier_qualified) == 3
        and len(recent_qualified) == result["case_outcomes"]["qualified"] == 19
        and all("full_pcm_sha256" in row for row in recent_qualified),
        "All earlier qualified source identities must be present",
    )
    references = {
        "v188_public_freeze": prior.source.PUBLIC,
        "v188_private_plan": prior.source.TRIAL / "private-plan.json",
        "v192_public_freeze": prior.rights.PUBLIC,
        "v192_private_cases": prior.rights.OUTPUT / "private-cases.json",
        "v192_private_result": prior.rights.OUTPUT / "result.json",
        "v199_public_freeze": prior.PUBLIC,
        "v199_public_result": V199_COMPLETE,
        "v199_private_plan": prior.TRIAL / "private-plan.json",
        "v199_private_cases": cases_path,
        "v199_private_result": result_path,
        "v199_resource_profile": profile_path,
        "v58_private_history": prior.source.HISTORY,
    }
    return [*earlier_plan, *recent_plan], [*earlier_qualified, *recent_qualified], references


def exclusion_keys(controls, history):
    prompts = {row["prompt_sha256"] for row in controls}
    fingerprints = set(history)
    for row in controls:
        fingerprints.add(f"bytes:{row['source_sha256']}")
        fingerprints.add(f"pcm:{row['pcm_sha256']}")
        if "full_pcm_sha256" in row:
            fingerprints.add(f"pcm:{row['full_pcm_sha256']}")
    return prompts, fingerprints


def source_plan(raw, excluded):
    return scoped(prior.source_plan, {"GROUPS": GROUPS, "rank": rank})(raw, excluded)


def context(raw):
    inherited, _ = prior.context(raw)
    earlier, _, references = prior_controls()
    history = read(prior.source.HISTORY)
    known = {
        key.casefold().removeprefix("voxforge:")
        for key in history["voice_keys"]
        if key.casefold().startswith("voxforge:")
    }
    plan, available = source_plan(raw, known | {row["contributor"] for row in earlier})
    code = (
        Path(__file__),
        ROOT / "backend/tests/test_audio_voxforge_tr_genuine_v212.py",
        ROOT / "docs/audio-voxforge-tr-genuine-v212-method.md",
    )
    return {
        **inherited,
        "schema": "chipmunk.audio-voxforge-tr-genuine-freeze/212",
        "code_sha256": {
            **inherited["code_sha256"],
            **{str(path.relative_to(ROOT)): sha(path) for path in code},
        },
        "prior_metadata_sha256": {key: sha(path) for key, path in references.items()},
        "excluded_prior_v199_named_groups": 24,
        "excluded_prior_qualified_groups": {"v192": 3, "v199": 19},
        "named_groups_after_exclusions": available,
        "planned_cases": GROUPS,
        "planned_session_archives": len(prior.source.session_list(plan)),
    }, plan


def preflight():
    return context(LISTING.read_bytes())[0]


def freeze():
    return scoped(prior.freeze, {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "context": context})()


def verify():
    return scoped(prior.verify, {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "context": context})()


def acquire():
    scoped(prior.acquire, {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "verify": verify})()


def acquired(plan):
    return scoped(prior.acquired, {"TRIAL": TRIAL, "PUBLIC": PUBLIC})(plan)


def read_prepared_pcm(row):
    return scoped(prior.read_prepared_pcm, {"OUTPUT": OUTPUT})(row)


def aggregation(rows, attempts, plan, archives):
    summary = scoped(prior.aggregation, {"GROUPS": GROUPS})(rows, attempts, plan, archives)
    _, earlier, _ = prior_controls()
    prompts, fingerprints = exclusion_keys(earlier, read(prior.source.HISTORY)["keys"])
    for row in rows:
        if row["status"] == "qualified":
            text, identities = exclusion_keys([row], ())
            require(
                not text & prompts and not identities & fingerprints,
                "Qualified source repeats prior identity",
            )
            prompts.update(text)
            fingerprints.update(identities)
    return summary


def prepare_worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded data-only preparation required",
    )
    plan = verify()
    archives, acquisition = acquired(plan)
    OUTPUT.mkdir(mode=0o700)
    prepared = OUTPUT / "prepared"
    prepared.mkdir(mode=0o700)
    _, earlier, _ = prior_controls()
    prompts, fingerprints = exclusion_keys(earlier, read(prior.source.HISTORY)["keys"])
    prepare_slot = scoped(prior.prepare_slot, {"TRIAL": TRIAL, "rank": rank})
    attempts = []
    with TemporaryDirectory(prefix="voxforge-genuine-v212-") as directory:
        rows = [
            prepare_slot(
                group,
                [row for row in archives if row["slot"] == group["slot"]],
                prompts,
                fingerprints,
                Path(directory),
                prepared,
                attempts,
            )
            for group in plan
        ]
    cases, journal = OUTPUT / "private-cases.json", OUTPUT / "private-attempts.json"
    write(cases, rows)
    write(journal, attempts)
    write(
        OUTPUT / "result.json",
        {
            "schema": RESULT_SCHEMA,
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_acquisition_sha256": acquisition["private_acquisition_sha256"],
            "private_cases_sha256": sha(cases),
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
    scoped(
        prior.prepare,
        {
            "TRIAL": TRIAL,
            "OUTPUT": OUTPUT,
            "verify": verify,
            "acquired": acquired,
            "MODULE": MODULE,
            "__file__": __file__,
        },
    )()


def readout():
    plan = verify()
    archives, acquisition = acquired(plan)
    result_path, profile_path = OUTPUT / "result.json", STAGE / "profile.json"
    cases_path, journal_path = OUTPUT / "private-cases.json", OUTPUT / "private-attempts.json"
    result, profile, rows, attempts = map(read, (result_path, profile_path, cases_path, journal_path))
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and result["schema"] == RESULT_SCHEMA
        and result["status"] == "complete"
        and result["language"] == "tr"
        and result["origin"] == "genuine"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_acquisition_sha256"] == acquisition["private_acquisition_sha256"]
        and result["private_cases_sha256"] == sha(cases_path)
        and result["private_attempts_sha256"] == sha(journal_path)
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and result["holdout_media_opened"] is result["holdout_scored"] is False
        and all(result[key] == value for key, value in aggregation(rows, attempts, plan, archives).items()),
        "Exact guarded v212 receipt, provenance and exclusion replay required",
    )
    for row in rows:
        if row["status"] == "qualified":
            read_prepared_pcm(row)
    return {
        **result,
        "archive_outcomes": acquisition["archive_outcomes"],
        "resource_profile_sha256": sha(profile_path),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {
        "preflight": preflight,
        "freeze": freeze,
        "acquire": acquire,
        "prepare": prepare,
        "prepare-worker": prepare_worker,
        "readout": readout,
    }
    parser.add_argument("command", choices=commands)
    outcome = commands[parser.parse_args().command]()
    if outcome is not None:
        print(json.dumps(outcome, indent=2, allow_nan=False))
