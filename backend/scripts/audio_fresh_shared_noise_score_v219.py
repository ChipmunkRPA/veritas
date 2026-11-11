"""Score the frozen v217 shared-noise panel with the unchanged speech route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from scripts import audio_demand_transfer_v208 as serving
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = serving.ROOT, serving.BASE, serving.PANEL
TRIAL = BASE / "audio-fresh-shared-noise-score-v219"
PUBLIC = PANEL / "audio-fresh-shared-noise-score-freeze-v219.json"
PREP = PANEL / "audio-fresh-shared-noise-complete-v217.json"
PREP_FREEZE = PANEL / "audio-fresh-shared-noise-freeze-v217.json"
INPUT_TRIAL = BASE / "audio-fresh-shared-noise-v217"
INPUT = INPUT_TRIAL / "bind-00"
METHOD = ROOT / "docs/audio-fresh-shared-noise-score-v219-method.md"
MODULE = "scripts.audio_fresh_shared_noise_score_v219"
CONDITIONS = ("matched_clean", "kitchen_0db", "kitchen_5db", "traffic_0db", "traffic_5db")
ARMS = ("genuine", "alania", "drifting")
CODE = (
    "backend/scripts/audio_fresh_shared_noise_score_v219.py",
    "backend/tests/test_audio_fresh_shared_noise_score_v219.py",
    "docs/audio-fresh-shared-noise-score-v219-method.md",
    "backend/scripts/audio_fresh_shared_noise_v217.py",
    "backend/scripts/audio_demand_transfer_v208.py",
    "backend/scripts/video_appearance_runtime_v108.py",
)
read, write, sha, require = serving.read, serving.write, serving.sha, serving.require


def waveform(row):
    return scoped(serving.waveform, {"INPUT": INPUT})(row)


def cases():
    """Verify the guarded, published v217 result without importing its source modules."""
    serving.prior.pilot.require_model_free()
    frozen, result, published = read(PREP_FREEZE), read(INPUT / "output/result.json"), read(PREP)
    profile, journal = INPUT / "profile.json", INPUT / "output/private-cases.jsonl"
    serving.prior.verify_data_guard(
        read(profile),
        INPUT,
        INPUT / "output/result.json",
        frozen["code_sha256"]["backend/scripts/audio_fresh_shared_noise_v217.py"],
    )
    require(
        frozen == read(INPUT_TRIAL / "freeze.json")
        and frozen["schema"] == "chipmunk.audio-fresh-shared-noise-freeze/217"
        and frozen["planned_triplets"] == 16
        and frozen["planned_cases"] == 240
        and tuple(frozen["arms"]) == ARMS
        and tuple(frozen["conditions"]) == CONDITIONS
        and frozen["source_range"] == [0, 48000]
        and frozen["sample_rate"] == 16000
        and frozen["model_calls_permitted"] is frozen["detector_calls_permitted"] is False
        and frozen["holdout_access_permitted"] is False
        and result["schema"] == "chipmunk.audio-fresh-shared-noise-result/217"
        and result["status"] == "complete"
        and result["planned_cases"] == 240
        and result["new_neural_calls"] == result["detector_calls"] == 0
        and result["holdout_scored"] is False
        and all(published.get(key) == value for key, value in result.items())
        and result["freeze_sha256"] == sha(PREP_FREEZE)
        and result["private_cases_sha256"] == sha(journal)
        and published["profile_sha256"] == sha(profile)
        and all(sha(ROOT / name) == digest for name, digest in frozen["code_sha256"].items()),
        "Exact published guarded v217 preparation required",
    )
    rows = serving.records(journal)
    expected = [
        (f"triplet-{i:02d}", arm, condition, f"triplet-{i:02d}-{arm}-{condition}")
        for i in range(16)
        for arm in ARMS
        for condition in CONDITIONS
    ]
    require(
        [(r["triplet"], r["arm"], r["condition"], r["case"]) for r in rows] == expected
        and all(r["status"] in {"prepared", "unavailable"} for r in rows),
        "Exact ordered v217 preparation roster required",
    )
    for i in range(16):
        chunk = rows[i * 15 : (i + 1) * 15]
        for arm_index in range(3):
            group = chunk[arm_index * 5 : arm_index * 5 + 5]
            require(
                len({r["status"] for r in group}) == 1, "One availability per prepared foreground required"
            )
            if group[0]["status"] == "prepared":
                require(all(r["samples"] == 48000 for r in group), "Exact three-second foreground required")
                for row in group:
                    waveform(row)
                require(
                    len({r["source_prefix_sha256"] for r in group}) == 1
                    and group[0]["noise_sample_offset"] is None
                    and all(r["noise_sample_offset"] == (4 + 6 * i) * 16000 for r in group[1:]),
                    "Matched source and fixed noise window required",
                )
        for j in range(1, 5):
            prepared = [chunk[k * 5 + j] for k in range(3) if chunk[k * 5 + j]["status"] == "prepared"]
            require(
                len({r["noise_residual_sha256"] for r in prepared}) <= 1
                and len({r["noise_excerpt_sha256"] for r in prepared}) <= 1,
                "Arms must share exact recorded-noise residuals",
            )
    for arm in ARMS:
        for condition in CONDITIONS:
            counts = dict(
                Counter(r["status"] for r in rows if r["arm"] == arm and r["condition"] == condition)
            )
            require(counts == result["by_arm_and_condition"][arm][condition], "Preparation coverage differs")
    return rows


def identity():
    serving.prior.pilot.require_model_free()
    rows = cases()
    serving.prior.verify()
    assets = serving.prior.pilot.ASSETS
    require(
        all(sha(serving.prior.pilot.WEIGHTS / name) == digest for name, digest in assets.items()),
        "Exact frozen local speech assets required",
    )
    serving.prior.pilot.require_model_free()
    asset_digest = hashlib.sha256(
        json.dumps(assets, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "chipmunk.audio-fresh-shared-noise-score-freeze/219",
        "v217_preparation_sha256": sha(PREP),
        "v217_freeze_sha256": sha(PREP_FREEZE),
        "route_freeze_sha256": sha(serving.prior.PUBLIC),
        "asset_set_commitment_sha256": asset_digest,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "planned_triplets": 16,
        "planned_cases": len(rows),
        "conditions": list(CONDITIONS),
        "scoring_batches": 5,
        "calibration_change_permitted": False,
        "holdout_access_permitted": False,
    }, rows


def freeze():
    require(not PUBLIC.exists() and not TRIAL.exists(), "Preserve any earlier scoring attempt")
    locked, _ = identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return locked


def verify():
    locked, rows = identity()
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == locked, "Exact frozen shared-noise scoring required"
    )
    return rows


def audit_case(row, case):
    require(
        all(row[key] == case[key] for key in ("case", "triplet", "arm", "condition")),
        "Outcome roster differs",
    )
    serving.prior.pilot.check_case(
        {**row, "slot": case["case"]},
        {
            **case,
            "slot": case["case"],
            "status": "qualified" if case["status"] == "prepared" else "unavailable",
        },
    )


def worker(batch, output):
    require(type(batch) is int and 0 <= batch < len(CONDITIONS), "One fixed batch is required")
    rows = verify()
    stage = TRIAL / f"extract-{batch:02d}"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact guarded offline scoring worker required",
    )
    selected = rows[batch :: len(CONDITIONS)]
    serving.prior.pilot.require_model_free()
    output.mkdir(mode=0o700)
    journal, forwards = output / "private-observations.jsonl", 0

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    with ExitStack() as stack, journal.open("x") as stream:
        score = serving.start(stack, count)
        for case in selected:
            before = forwards
            outcome = (
                score(waveform(case)) if case["status"] == "prepared" else serving.prior.pilot.unavailable()
            )
            row = {key: case[key] for key in ("case", "triplet", "condition", "arm")}
            row.update(outcome, pella_forwards=forwards - before)
            audit_case(row, case)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    write(
        output / "result.json",
        {
            "status": "complete",
            "batch": batch,
            "condition": CONDITIONS[batch],
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": forwards,
        },
    )


def completed_batch(batch, cases_):
    require(type(batch) is int and 0 <= batch < len(CONDITIONS), "One fixed batch is required")
    stage = TRIAL / f"extract-{batch:02d}"
    journal, result_path = stage / "output/private-observations.jsonl", stage / "output/result.json"
    rows, result = serving.records(journal), read(result_path)
    count = sum(row["pella_forwards"] for row in rows)
    serving.prior.pilot.speech.qualification_profile(stage, count, {}, Path(__file__).name, "extract", batch)
    require(
        result["status"] == "complete"
        and result["batch"] == batch
        and result["condition"] == CONDITIONS[batch]
        and result["private_observations_sha256"] == sha(journal)
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["new_neural_calls"] == count
        and len(rows) == 48,
        "Exact complete scored batch required",
    )
    for row, case in zip(rows, cases_[batch :: len(CONDITIONS)], strict=True):
        audit_case(row, case)
    return rows, sha(stage / "profile.json"), count


def run(batch):
    require(type(batch) is int and 0 <= batch < len(CONDITIONS), "One fixed batch is required")
    sources = verify()
    for previous in range(batch):
        completed_batch(previous, sources)

    def launch(argv, **kwargs):
        output = TRIAL / f"extract-{batch:02d}/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(output), str(batch)],
            "Exact owned scoring command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", str(batch), str(output)], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
        },
    )("extract", batch)


def summarize(rows):
    require(len(rows) == 240, "Every planned source and condition needs a result")
    expected = {
        (f"triplet-{i:02d}", arm, condition) for i in range(16) for arm in ARMS for condition in CONDITIONS
    }
    require(
        {(row["triplet"], row["arm"], row["condition"]) for row in rows} == expected,
        "Exact unique planned outcomes required",
    )
    require(
        all(
            row["route_status"] in {"ok", "not_applicable", "preparation_unavailable"}
            and row["decision"] in serving.prior.pilot.DIRECTIONS
            and ((row["route_status"] == "ok") == (row["decision"] != "unavailable"))
            for row in rows
        ),
        "Recorded direction and availability must agree",
    )
    by_key = {(row["triplet"], row["arm"], row["condition"]): row for row in rows}
    outcomes, transitions, coverage = {}, {}, {}
    for arm in ARMS:
        outcomes[arm], transitions[arm] = {}, {}
        for condition in CONDITIONS:
            group = [by_key[f"triplet-{i:02d}", arm, condition] for i in range(16)]
            counts = Counter(row["decision"] for row in group)
            outcomes[arm][condition] = {
                "planned": 16,
                "scored": sum(r["route_status"] == "ok" for r in group),
                "speech_gate_rejected": sum(r["route_status"] == "not_applicable" for r in group),
                "preparation_unavailable": sum(r["route_status"] == "preparation_unavailable" for r in group),
                "directions": {key: counts[key] for key in serving.prior.pilot.DIRECTIONS},
                "false_ai": counts["ai"] if arm == "genuine" else None,
                "missed_ai_over_planned": 16 - counts["ai"] if arm != "genuine" else None,
                "wrong_original": counts["original"] if arm != "genuine" else None,
            }
        for condition in CONDITIONS[1:]:
            pairs = [
                (by_key[f"triplet-{i:02d}", arm, "matched_clean"], by_key[f"triplet-{i:02d}", arm, condition])
                for i in range(16)
            ]
            transitions[arm][f"matched_clean_to_{condition}"] = {
                "planned_pairs": 16,
                "both_scored": sum(a["route_status"] == b["route_status"] == "ok" for a, b in pairs),
                "directions": dict(Counter(f"{a['decision']}->{b['decision']}" for a, b in pairs)),
                "route_statuses": dict(
                    Counter(f"{a['route_status']}->{b['route_status']}" for a, b in pairs)
                ),
            }
    for condition in CONDITIONS:
        coverage[condition] = {
            f"{a}_{b}": sum(
                by_key[f"triplet-{i:02d}", a, condition]["route_status"]
                == by_key[f"triplet-{i:02d}", b, condition]["route_status"]
                == "ok"
                for i in range(16)
            )
            for a, b in (("genuine", "alania"), ("genuine", "drifting"), ("alania", "drifting"))
        }
        coverage[condition]["complete_triplets"] = sum(
            all(by_key[f"triplet-{i:02d}", arm, condition]["route_status"] == "ok" for arm in ARMS)
            for i in range(16)
        )
    return {"outcomes": outcomes, "clean_to_noise_transitions": transitions, "scored_pair_coverage": coverage}


def readout():
    sources = verify()
    rows, profiles, calls = [], [], 0
    for batch in range(len(CONDITIONS)):
        current, profile, count = completed_batch(batch, sources)
        rows.extend(current)
        profiles.append(profile)
        calls += count
    return {
        "schema": "chipmunk.audio-fresh-shared-noise-score-result/219",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": calls,
        "profile_sha256": profiles,
        **summarize(rows),
        "scope": "Fresh Turkish source cohorts and two recorded noise tapes. "
        "Differences cannot isolate source class or establish population performance; no calibration or defaults changed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    sub.add_parser("readout")
    for action in ("run", "_worker"):
        command = sub.add_parser(action)
        command.add_argument("batch", type=int, choices=range(len(CONDITIONS)))
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run(args.batch)
    elif args.action == "_worker":
        worker(args.batch, args.output)
    else:
        print(json.dumps({"freeze": freeze, "readout": readout}[args.action](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
