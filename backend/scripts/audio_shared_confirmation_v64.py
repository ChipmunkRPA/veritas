"""Explicit technical repetition after a descendant-monitor failure; frozen v62 math."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from scripts import audio_shared_confirmation_v62 as prior
from scripts.audio_reserve_audit_v60 import digest, write

ROOT, BASE, SOURCE, CHUNK = prior.ROOT, prior.BASE, prior.SOURCE, prior.CHUNK
TRIAL = BASE / "audio-shared-confirmation-v64"
PUBLIC = ROOT / "docs/evaluation-panels/audio-shared-confirmation-v64-freeze.json"
METHODS, summarize, changes = prior.METHODS, prior.summarize, prior.changes
read, require, decisions = prior.read, prior.require, prior.decisions


def implementation():
    names = {"backend/scripts/audio_shared_confirmation_v64.py", "backend/scripts/audio_shared_runtime_v64.py",
             "backend/scripts/audio_tree_sampler_v64.py", "backend/tests/test_audio_tree_sampler_v64.py",
             "docs/audio-shared-confirmation-v64-protocol.md"}
    return {**prior.implementation(), **{name: digest(ROOT / name) for name in sorted(names)}}


def prepare():
    lock = prior.admission(fresh=True)
    stress_path = BASE / "audio-shared-monitor-v64-a02/result.json"
    stress = read(stress_path)
    require(stress["error"] is None and stress["child_exit_code"] == 0
            and stress["spawned_short_lived_children"] == 250 and stress["samples"] > 0
            and stress["sampler_sha256"] == digest(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
            "Qualified current monitor and real subprocess stress receipt required")
    old = prior.TRIAL
    path = old / "chunk-00.jsonl"
    profile = read(old / "chunk-00-profile/profile.json")
    require(profile["guard"] == "live_sampling_unavailable" and profile["child_exit_code"] == -15
            and not (old / "chunk-00-complete.json").exists(), "Preserved original monitor stop required")
    # Bind only prefix identities here; do not inspect predictions to select a
    # method, source, calibration, operating rule or favorable subset.
    sources = read(SOURCE / "confirmation/variants.json")["variants"]
    count = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row["id"] == sources[count]["id"] and row["sha256"] == sources[count]["sha256"],
                    "Interrupted prefix differs from fixed source ordering")
            count += 1
    require(count == 33, "Original interrupted prefix must remain exact")
    require(not TRIAL.exists() and not PUBLIC.exists(), "Never repeated destination required")
    frozen = {**lock, "schema": "chipmunk.audio-shared-technical-repetition-freeze/64",
              "code_sha256": implementation(), "original_method_freeze_sha256": digest(prior.PUBLIC),
              "original_method_freeze_commit": "6fa5ab0", "original_parity_sha256": digest(old / "parity.json"),
              "original_parity_profile_sha256": digest(old / "parity-profile/profile.json"),
              "original_failed_chunk_sha256": digest(path),
              "original_failed_profile_sha256": digest(old / "chunk-00-profile/profile.json"),
              "monitor_stress_sha256": digest(stress_path),
              "monitor_stress": {key: stress[key] for key in ("samples", "spawned_short_lived_children", "duration_s")},
              "prior_processed_cases": count, "possible_in_flight_additional_cases": 1, "fresh_calls_started": True,
              "full_panel_technical_repetition_started": False, "refit_or_response_based_selection": False,
              "repetition": "All 1080 original cases using the exact original score function, heads, calibrations and combinations. 33 cases were processed under a stopped monitor. That attempt remains failed; no outcome-based subset or method change. Full new execution is a technical repetition, not a second fresh cohort."}
    TRIAL.mkdir()
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"inputs": 1080, "prior_processed_cases": count, "new_model_calls": 0,
                      "freeze_sha256": digest(PUBLIC)}), flush=True)


def admission():
    lock = prior.admission(fresh=True)
    frozen, start = read(TRIAL / "freeze.json"), read(TRIAL / "execution-start.json")
    require(digest(PUBLIC) == digest(TRIAL / "freeze.json") == start["freeze_sha256"]
            and start["remote_verified_before_repetition"] is True, "Published exact technical repetition freeze required")
    require(frozen["code_sha256"] == implementation() and frozen["original_method_freeze_sha256"] == digest(prior.PUBLIC)
            and frozen["original_failed_chunk_sha256"] == digest(prior.TRIAL / "chunk-00.jsonl")
            and frozen["original_failed_profile_sha256"] == digest(prior.TRIAL / "chunk-00-profile/profile.json")
            and frozen["original_parity_sha256"] == digest(prior.TRIAL / "parity.json")
            and frozen["original_parity_profile_sha256"] == digest(prior.TRIAL / "parity-profile/profile.json"),
            "Scientific method, preserved stopped attempt or native qualification changed")
    prior.profile_pass(TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json"))
    return lock


def run_chunk(index):
    require(type(index) is int and 0 <= index < 12, "One predetermined chunk index required")
    lock = admission()
    destination = TRIAL / f"chunk-{index:02d}.jsonl"
    require(not destination.exists(), "Existing or failed output cannot be overwritten")
    for previous in range(index):
        prior.profile_pass(TRIAL / f"chunk-{previous:02d}-profile/profile.json")
        require(read(TRIAL / f"chunk-{previous:02d}-complete.json")["completed"] == CHUNK,
                "Every preceding chunk must finish")
    body = read(SOURCE / "confirmation/variants.json")
    rows = body["variants"][index * CHUNK:(index + 1) * CHUNK]
    require(len(rows) == CHUNK, "Complete fixed chunk required")
    current, activity, candidate, heads, captured, hook = prior.load_models(lock)
    completed = 0
    with destination.open("x") as stream:
        try:
            for source in rows:
                started = time.monotonic()
                result = {key: source[key] for key in ("id", "base_id", "group", "role", "label", "condition", "generator", "sha256", "pcm_sha256")}
                try:
                    result.update(prior.score(prior.input_pcm(body, source), current, activity, candidate, heads, captured))
                except Exception as error:
                    result.update(status="failed", error_type=type(error).__name__, detail=str(error)[:250])
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    stream.flush()
                    raise
                result["runtime_s"] = time.monotonic() - started
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                completed += 1
                if completed % 15 == 0:
                    print(json.dumps({"chunk": index, "completed": completed, "planned": CHUNK,
                                      "status": result["status"]}), flush=True)
        finally:
            hook.remove()
    write(TRIAL / f"chunk-{index:02d}-complete.json", {"completed": completed, "journal_sha256": digest(destination)})


def evaluate():
    lock = read(TRIAL / "freeze.json")
    admission()
    sources = read(SOURCE / "confirmation/variants.json")["variants"]
    rows, bindings = [], {}
    for index in range(12):
        prior.profile_pass(TRIAL / f"chunk-{index:02d}-profile/profile.json")
        path = TRIAL / f"chunk-{index:02d}.jsonl"
        receipt = read(TRIAL / f"chunk-{index:02d}-complete.json")
        require(receipt == {"completed": CHUNK, "journal_sha256": digest(path)}, "Complete unaltered chunk required")
        bindings[path.name] = digest(path)
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    require(len(rows) == 1080, "Every predetermined input required")
    for row, source in zip(rows, sources, strict=True):
        require(all(row[key] == source[key] for key in ("id", "base_id", "group", "role", "label", "condition", "generator", "sha256", "pcm_sha256"))
                and row["decisions"] == decisions(row), "Fresh result identity or fixed combination changed")
    names = [*METHODS, "v7"]
    predictions = {name: [row["decisions"][name] for row in rows] for name in names}
    totals = {name: summarize(rows, values) for name, values in predictions.items()}
    strata = {}
    for field in ("condition", "generator"):
        strata[field] = {}
        for value in sorted({row[field] for row in rows}):
            indices = [n for n, row in enumerate(rows) if row[field] == value]
            strata[field][value] = {name: summarize([rows[n] for n in indices], [values[n] for n in indices])
                                    for name, values in predictions.items()}
    gain = {name: changes(rows, predictions["current"], values) for name, values in predictions.items()}
    groups = sorted({row["group"] for row in rows})
    rng = np.random.default_rng(lock["bootstrap"]["seed"])
    draws = rng.integers(0, len(groups), size=(lock["bootstrap"]["replicates"], len(groups)))
    intervals = {}
    for name, values in predictions.items():
        deltas = np.asarray([sum(int(values[n] == ("ai" if row["label"] else "original"))
                                - int(predictions["current"][n] == ("ai" if row["label"] else "original"))
                                for n, row in enumerate(rows) if row["group"] == group) for group in groups])
        intervals[name] = [float(value) for value in np.quantile(deltas[draws].sum(axis=1), [.025, .975])]
    result = {"schema": "chipmunk.audio-shared-confirmation-result/64", "freeze_sha256": digest(PUBLIC),
              "journal_sha256": bindings, "totals": totals, "strata": strata, "changes_from_current": gain,
              "paired_group_bootstrap_95pct_net_correct_count_change": intervals,
              "assessed": sum(row["status"] == "ok" for row in rows),
              "unassessed": sum(row["status"] == "not_applicable" for row in rows),
              "combined_probability": None, "serving_changed": False,
              "limits": "Panel fixed before initial 33 records in interrupted v62; full technical repetition with unchanged methods, some historically seen voices, three AI generators, short English speech, shared backgrounds and correlated variants. Decision combinations are not calibrated probabilities. No independent outside reviewer or general accuracy guarantee."}
    write(TRIAL / "result.json", result)
    # No record IDs, learned coefficients, raw media or voice representations.
    write(ROOT / "docs/evaluation-panels/audio-shared-confirmation-v64-result.json", result)
    print(json.dumps({"totals": totals, "changes_from_current": gain}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "chunk", "evaluate"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.stage == "chunk":
        run_chunk(args.index)
    elif args.index is not None:
        parser.error("Only a chunk takes an index")
    else:
        {"prepare": prepare, "evaluate": evaluate}[args.stage]()
