"""Score the guarded v199 genuine Turkish panel with the frozen v191 route."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from scripts import audio_voxforge_tr_transfer_v191 as pilot
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = pilot.ROOT, pilot.BASE, pilot.PANEL
read, require, sha, write = pilot.read, pilot.require, pilot.sha, pilot.write
MODULE = "scripts.audio_voxforge_tr_transfer_v201"
TRIAL = BASE / "audio-voxforge-tr-transfer-v201"
OUTPUT = TRIAL / "authored/output"
PUBLIC = PANEL / "audio-voxforge-tr-transfer-freeze-v201.json"
PREPARATION = BASE / "audio-voxforge-tr-genuine-v199"
PREPARATION_STAGE = PREPARATION / "bind-00"
PREPARATION_PUBLIC = PANEL / "audio-voxforge-tr-genuine-freeze-v199.json"
PRIOR_RESULT = PANEL / "audio-voxforge-tr-transfer-complete-v191.json"
PREPARATION_CODE = "backend/scripts/audio_voxforge_tr_genuine_v199.py"
CODE = (
    "backend/scripts/audio_voxforge_tr_transfer_v201.py",
    "backend/tests/test_audio_voxforge_tr_transfer_v201.py",
    "docs/audio-voxforge-tr-transfer-v201-method.md",
)
MIN_SAMPLES, MAX_SAMPLES = 48000, 64000
PLANNED, ELIGIBLE = 24, 19
SELECTIONS = {"pause_bounded_crop": 18, "natural_short_full": 1}
UNAVAILABLE_CLASS = "preparation_unavailable"
ACTUAL_PREPARATION = {
    "freeze_sha256": "a91ce4a16280dbd477ec1dddc1271c7db5a1177c5cec45e132911d45778bbe8e",
    "resource_profile_sha256": "7857dcab1dfea6ebab0a02790f8d6201193f86b9cc5746290f2644a4142d54fa",
    "private_cases_sha256": "369b7ecf0f0061dd6878a99ad75f432d8dca058ef3e5f0fad6c67ccf994671c0",
    "private_attempts_sha256": "4f2ec01cfcb50dd779019bfbeeec431a65ec5dafbecd439021c87dfe1e84af77",
    "private_acquisition_sha256": "29146ad559a45d71cac34d92d5cf2f2c0f7753628994c112f70a4e716a7a93b6",
}


def selection_class(row):
    return row["selection"] if row["status"] == "qualified" else UNAVAILABLE_CLASS


def verify_clock(row, lock):
    full, crop = row["full_source_range"], row["source_range"]
    start, end, size = crop["start_sample"], crop["end_sample"], row["samples"]
    origin = full["source_start_pts"]

    def extent(a, b, pts):
        return {
            "start_sample": a,
            "end_sample": b,
            "source_start_pts": pts + a,
            "source_end_pts": pts + b,
            "time_base": f"1/{pilot.RATE}",
        }

    require(
        type(size) is int
        and MIN_SAMPLES <= size <= MAX_SAMPLES
        and 0 <= start < end <= row["full_samples"]
        and end - start == size
        and full == extent(0, row["full_samples"], origin)
        and crop == extent(start, end, origin)
        and row["wav_range"] == extent(0, size, 0),
        "Exact source and prepared crop clocks required",
    )
    if row["selection"] == "natural_short_full":
        require(start == 0 and size == row["full_samples"], "Complete natural short source required")
    else:
        frame = lock["crop_frame_samples"]
        require(
            row["selection"] == "pause_bounded_crop"
            and row["full_samples"] > MAX_SAMPLES
            and start % frame == size % frame == 0
            and size // frame in lock["crop_frame_counts_in_preference_order"],
            "Exact frozen pause-bounded crop extent required",
        )


def verify_data_guard(profile, stage, result_path, code_sha):
    journal = stage / "resources.jsonl"
    samples = pilot.speech.planning.reviewed.lines(journal)
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == profile["resource_limit_observations"] == []
        and profile["new_neural_calls"] == 0
        and profile["runtime_sha256"] == code_sha
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(journal)
        and samples
        and all(
            not item["violations"]
            and not pilot.runtime.violations(item, "bind", initial=i == 0)
            and item["archive_free_disk_bytes"] >= (3 * 1024**3 if i == 0 else 1536 * 1024**2)
            for i, item in enumerate(samples)
        )
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(item["tree_rss_bytes"] for item in samples)
        and profile["minimum_available_ram_bytes"] == min(item["ram_available_bytes"] for item in samples)
        and samples[-1]["elapsed_s"] <= profile["duration_s"] <= 900,
        "Exact successful v199 data-only guard required",
    )


def verified_preparation():
    stage = PREPARATION_STAGE
    output = stage / "output"
    acquisition_dir = PREPARATION / "acquired"
    result_path, profile_path = output / "result.json", stage / "profile.json"
    cases = output / "private-cases.json"
    acquisition_path = acquisition_dir / "private-acquisition.jsonl"
    actual = {
        "freeze_sha256": sha(PREPARATION_PUBLIC),
        "resource_profile_sha256": sha(profile_path),
        "private_cases_sha256": sha(cases),
        "private_attempts_sha256": sha(output / "private-attempts.json"),
        "private_acquisition_sha256": sha(acquisition_path),
    }
    lock = read(PREPARATION_PUBLIC)
    require(
        actual == ACTUAL_PREPARATION
        and lock == read(PREPARATION / "freeze.json")
        and lock["schema"] == "chipmunk.audio-voxforge-tr-genuine-freeze/199"
        and lock["production_sample_rate"] == pilot.RATE
        and lock["prepared_sample_bounds"] == [MIN_SAMPLES, MAX_SAMPLES]
        and lock["language"] == "tr"
        and lock["origin"] == "genuine"
        and lock["model_calls_permitted"] is lock["holdout_access_permitted"] is False
        and lock["private_plan_sha256"] == sha(PREPARATION / "private-plan.json")
        and lock["publisher_index_sha256"] == sha(PREPARATION / "publisher-index.html")
        and all(sha(ROOT / name) == digest for name, digest in lock["code_sha256"].items()),
        "Exact published v199 freeze, preparation code and private data commitments required",
    )
    verify_data_guard(read(profile_path), stage, result_path, lock["code_sha256"][PREPARATION_CODE])
    result, rows = read(result_path), read(cases)
    acquisition = read(acquisition_dir / "result.json")
    archive_rows = pilot.speech.planning.reviewed.lines(acquisition_path)
    archives = {
        (row["slot"], row["archive"]): row["sha256"] for row in archive_rows if row["status"] == "downloaded"
    }
    require(
        result["schema"] == "chipmunk.audio-voxforge-tr-genuine-result/199"
        and result["status"] == "complete"
        and all(result[key] == value for key, value in actual.items() if key != "resource_profile_sha256")
        and result["new_neural_calls"] == 0
        and result["holdout_media_opened"] is result["holdout_scored"] is False
        and result["language"] == "tr"
        and result["origin"] == "genuine"
        and acquisition["schema"] == "chipmunk.audio-voxforge-tr-acquisition/188"
        and acquisition["status"] == "complete"
        and acquisition["freeze_sha256"] == actual["freeze_sha256"]
        and acquisition["private_acquisition_sha256"] == actual["private_acquisition_sha256"]
        and acquisition["archive_outcomes"] == {"downloaded": 34}
        and len(archives) == len(archive_rows) == lock["planned_session_archives"] == 34
        and result["planned_cases"] == lock["planned_cases"] == len(rows) == PLANNED
        and result["case_outcomes"]
        == {"qualified": ELIGIBLE, "unavailable": PLANNED - ELIGIBLE}
        == Counter(row["status"] for row in rows)
        and result["selection_outcomes"] == SELECTIONS
        and result["qualified_licenses"] == {"GPL-3.0-or-later": ELIGIBLE}
        and [row["slot"] for row in rows] == [f"genuine-{i:02d}" for i in range(PLANNED)]
        and all(row["language"] == "tr" and row["label"] == 0 for row in rows),
        "Exact completed v199 preparation and acquisition outcomes required",
    )
    qualified = [row for row in rows if row["status"] == "qualified"]
    require(
        Counter(row["selection"] for row in qualified) == SELECTIONS
        and all(row["license"] == "GPL-3.0-or-later" for row in qualified)
        and all(archives.get((row["slot"], row["archive"])) == row["archive_sha256"] for row in qualified)
        and all(
            len({row[key] for row in qualified}) == ELIGIBLE
            for key in ("prompt_sha256", "source_sha256", "full_pcm_sha256", "pcm_sha256")
        ),
        "Distinct admitted source and prompt receipts required",
    )
    verify_wave = scoped(pilot.read_prepared_wave, {"PREPARATION_STAGE": stage, "MIN_SAMPLES": MIN_SAMPLES})
    for row in qualified:
        verify_clock(row, lock)
        verify_wave({**row, "source_range": row["wav_range"]})
    return rows, {
        **actual,
        "worker_receipt_sha256": sha(result_path),
        "acquisition_receipt_sha256": sha(acquisition_dir / "result.json"),
    }


def prior_results():
    alania, packages = pilot.verified_alania()
    older_lock = read(pilot.PUBLIC)
    older_stage = pilot.TRIAL / "authored"
    require(
        older_lock == read(pilot.TRIAL / "freeze.json")
        and older_lock["schema"] == "chipmunk.audio-voxforge-tr-transfer-freeze/191"
        and older_lock["synthetic_result_sha256"] == sha(pilot.ALANIA_RESULT)
        and older_lock["voice_route_freeze_sha256"] == sha(pilot.VOICE_FREEZE)
        and older_lock["local_asset_commitments"] == pilot.ASSETS
        and older_lock["runtime_packages"] == packages
        and all(sha(ROOT / name) == digest for name, digest in older_lock["code_sha256"].items()),
        "Exact previously completed v191 serving-route freeze required",
    )
    published = read(PRIOR_RESULT)
    older = pilot.speech.qualification_profile(
        older_stage, published["new_neural_calls"], {}, Path(pilot.__file__).name
    )
    profile_path = older_stage / "profile.json"
    profile = read(profile_path)
    genuine = older["comparison"]["genuine_voxforge"]
    require(
        published
        == {
            **older,
            "resource_profile_sha256": sha(profile_path),
            "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
            "worker_duration_seconds": profile["duration_s"],
        }
        and older["schema"] == "chipmunk.audio-voxforge-tr-transfer-result/191"
        and older["freeze_sha256"] == sha(pilot.PUBLIC)
        and older["private_observations_sha256"] == sha(older_stage / "output/private-observations.jsonl")
        and older["holdout_media_opened"] is older["holdout_scored"] is older["defaults_changed"] is False
        and genuine["planned_slots"] == 12
        and genuine["qualified_slots"] == 3
        and genuine["speech_gate_rejections"] == 2
        and genuine["scored_slots"] == 1,
        "Exact published guarded v191 descriptive comparison required",
    )
    return genuine, alania, packages


def context():
    pilot.require_model_free()
    prepared, receipt = verified_preparation()
    genuine, alania, packages = prior_results()
    pilot.require_model_free()
    lock = {
        "schema": "chipmunk.audio-voxforge-tr-transfer-freeze/201",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "prepared_genuine_receipt_sha256": receipt,
        "v191_freeze_sha256": sha(pilot.PUBLIC),
        "v191_result_sha256": sha(PRIOR_RESULT),
        "v179_result_sha256": sha(pilot.ALANIA_RESULT),
        "voice_route_freeze_sha256": sha(pilot.VOICE_FREEZE),
        "local_asset_commitments": pilot.ASSETS,
        "runtime_packages": packages,
        "planned_genuine_slots": PLANNED,
        "eligible_genuine_slots": ELIGIBLE,
        "eligible_selection_counts": SELECTIONS,
        "language": "tr",
        "route_reference_language": "en",
        "new_fitting_permitted": False,
        "holdout_access_permitted": False,
    }
    return lock, prepared, genuine, alania


def preflight():
    lock, _, _, _ = context()
    return {
        "status": "ready_for_freeze",
        "planned_genuine_slots": lock["planned_genuine_slots"],
        "eligible_genuine_slots": lock["eligible_genuine_slots"],
        "eligible_selection_counts": lock["eligible_selection_counts"],
        "model_calls": 0,
        "settings_imported": False,
    }


def verify():
    lock, prepared, genuine, alania = context()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact frozen v201 scoring study required")
    return prepared, genuine, alania


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every earlier attempt")
    lock, _, _, _ = context()
    require(
        all(sha(pilot.WEIGHTS / name) == digest for name, digest in pilot.ASSETS.items()),
        "Exact frozen local voice assets required",
    )
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return {"status": "frozen", "planned_genuine_slots": PLANNED, "model_calls": 0}


def summary(rows, prepared, genuine, alania):
    result = pilot.summary(rows, prepared, alania)
    by_selection = {}
    for row, case in zip(rows, prepared, strict=True):
        classification = selection_class(case)
        require(
            row["selection_class"] == classification and row["source_range"] == case.get("source_range"),
            "Scored journal selection or original source range differs",
        )
        subset = by_selection.setdefault(classification, ([], []))
        subset[0].append(row)
        subset[1].append(case)
    return {
        "genuine_voxforge": result["genuine_voxforge"],
        "genuine_voxforge_by_selection": {
            name: pilot.summary(observations, cases, alania)["genuine_voxforge"]
            for name, (observations, cases) in by_selection.items()
        },
        "genuine_voxforge_v191_prior_result": genuine,
        "synthetic_alania_v179_prior_result": result["synthetic_alania_prior_result"],
    }


def start_scoring(stack, count):
    # Keep settings and all model imports out of admission and preflight.
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts import audio_alania_transfer_v179 as alania
    from scripts import audio_voxforge_tr_genuine_v199 as source
    from scripts.run_audio_specialists_v15 import load_voice

    model = load_voice({"weights": str(pilot.WEIGHTS), "assets": pilot.ASSETS})
    activity = SpeechActivity(pilot.WEIGHTS / "silero_vad.jit")
    stack.callback(model.register_forward_pre_hook(count).remove)

    def score(case):
        return alania.score_voice(source.read_prepared_pcm(case), model, activity, Deadline(120))

    return score


def worker(output):
    pilot.require_model_free()
    stage = TRIAL / "authored"
    require(
        output == OUTPUT
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Fresh offline owned-worker guard required",
    )
    prepared, genuine, alania = verify()
    pilot.require_model_free()
    output.mkdir(mode=0o700)
    forwards, rows = 0, []

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    journal = output / "private-observations.jsonl"
    with ExitStack() as stack, journal.open("x") as stream:
        score = None
        for case in prepared:
            before = forwards
            if case["status"] == "qualified":
                if score is None:
                    score = start_scoring(stack, count)
                outcome = score(case)
            else:
                outcome = pilot.unavailable()
            row = {
                "slot": case["slot"],
                "selection_class": selection_class(case),
                "source_range": case.get("source_range"),
                **outcome,
                "pella_forwards": forwards - before,
            }
            pilot.check_case(row, case)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    raw = sum(len(row["windows"]) for row in rows)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-voxforge-tr-transfer-result/201",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "comparison": summary(rows, prepared, genuine, alania),
            "new_neural_calls": forwards,
            "serving_raw_forwards": raw,
            "serving_stability_forwards": forwards - raw,
            "holdout_media_opened": False,
            "holdout_scored": False,
            "defaults_changed": False,
        },
    )


def run():
    scoped(
        pilot.run,
        {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "MODULE": MODULE, "verify": verify, "__file__": __file__},
    )()


def readout():
    prepared, genuine, alania = verify()
    stage = TRIAL / "authored"
    journal = OUTPUT / "private-observations.jsonl"
    rows = pilot.speech.planning.reviewed.lines(journal)
    compared = summary(rows, prepared, genuine, alania)
    calls = sum(row["pella_forwards"] for row in rows)
    raw = sum(len(row["windows"]) for row in rows)
    result = pilot.speech.qualification_profile(stage, calls, {}, Path(__file__).name)
    require(
        result["schema"] == "chipmunk.audio-voxforge-tr-transfer-result/201"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["comparison"] == compared
        and result["serving_raw_forwards"] == raw
        and result["serving_stability_forwards"] == calls - raw
        and result["holdout_media_opened"] is result["holdout_scored"] is result["defaults_changed"] is False,
        "Exact guarded v201 outcome and resource receipt required",
    )
    profile_path = stage / "profile.json"
    profile = read(profile_path)
    return {
        **result,
        "resource_profile_sha256": sha(profile_path),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run", "worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require(
        (args.command == "worker") == (args.output is not None),
        "Only the guarded worker takes an output path",
    )
    if args.command == "worker":
        worker(args.output)
    elif args.command == "run":
        run()
    else:
        commands = {"preflight": preflight, "freeze": freeze, "readout": readout}
        print(json.dumps(commands[args.command](), indent=2, allow_nan=False))
