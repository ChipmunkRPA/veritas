"""Prospective common-input comparison of frozen audio heads and combinations.

Research only: no refitting, combined probability, service change or retry.
Reserve metadata is already fixed; fresh forwards require published freeze and
consumed-source native parity. All failed and unassessed cases stay explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration as serving
from scripts import audio_wavlm_features_v13 as features
from scripts import audio_wavlm_response_v13 as method
from scripts import run_audio_wavlm_v13 as original
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_combinations_v52 import METHODS, changes, combine, summarize
from scripts.audio_guard_v7 import initial
from scripts.audio_history_v58 import row_keys
from scripts.audio_multisource_v8 import final
from scripts.audio_reserve_audit_v60 import digest, write
from scripts.run_audio_guard_v7 import capture
from scripts.run_audio_multisource_v8 import input_pcm
from scripts.score_audio_confirmation_v7 import models, runtime, score_one, weights

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-shared-confirmation-v62"
PUBLIC = ROOT / "docs/evaluation-panels/audio-shared-confirmation-v62-freeze.json"
SOURCE = BASE / "audio-v9"
HEADS = {"v7": BASE / "audio-v7/development/trained/head.json",
         "v13": BASE / "audio-v13-wavlm/trained/head.json"}
INPUT_SHA = "14828469353f3b37a60da86f6588b817d946846303c275afa46e4f752bce4e26"
SOURCE_SHA = "f3d7a7f2e20a9d2aa9092c371f5af4d2a05ed9172e4652677049a8286f61605f"
HEAD_SHA = {"v7": "fa1a0faec12fd6926cefb8f7ffe9c9736f517ea89b44a2ff13ab17f4d028c408",
            "v13": "23350d709c9dc4c968066e2ab36bb16792f5c18d2513408b5f5c3298a8f83ae9"}
CHUNK = 90


def parity_plan(identifiers):
    encoded = json.dumps(identifiers, separators=(",", ":")).encode()
    return {"parity_inputs": len(identifiers), "parity_ids_sha256": hashlib.sha256(encoded).hexdigest()}


def v7_overlap():
    source = read(SOURCE / "source-plan.json")
    reserve = [row for row in source["sources"] if row["role"] == "confirmation"]
    manifest = read(BASE / "audio-v7/development/manifest.json")
    roles = {row["base_id"]: row["role"] for row in manifest["variants"]}
    trained = [row for row in manifest["sources"] if roles.get(row["id"]) in {"fit", "cal"}]
    common = set().union(*(row_keys(row) for row in reserve)) & set().union(*(row_keys(row) for row in trained))
    return {"fit_cal_sources": len(trained), "reserve_sources": len(reserve),
            "known_identity_keys_in_common": dict(Counter(key.split(":", 1)[0] for key in common)),
            "scope": "Older v7 head has two known fitting/calibration voice overlaps. No known recording/PCM/text overlap. Current protected calibration and pretrained exposure are not claimed speaker-independent."}


def read(path):
    return json.loads(path.read_text())


def require(value, reason):
    if not value:
        raise ValueError(reason)


def parity_selection(rows):
    groups = {}
    for row in rows:
        if row["role"] == "development":
            groups.setdefault((row["condition"], row["label"]), []).append(row)
    require(len(groups) == 18, "Both truths in all nine consumed conditions required")
    return [min(members, key=lambda row: row["id"])["id"] for _, members in sorted(groups.items())]


def implementation():
    names = set(original.implementation()) | {
        "backend/scripts/audio_shared_confirmation_v62.py", "backend/scripts/audio_shared_runtime_v62.py",
        "backend/tests/test_audio_shared_confirmation_v62.py", "docs/audio-shared-confirmation-v62-protocol.md",
        "backend/scripts/audio_combinations_v52.py", "backend/scripts/evaluation_tradeoffs_v44.py",
        "backend/scripts/audio_reserve_audit_v60.py", "backend/scripts/review_audio_reserve_v60.py",
        "backend/scripts/audio_history_v58.py", "backend/scripts/audio_source_identity_v9.py",
        "backend/scripts/local_probe_limits_v24.py", "backend/scripts/data_resource_guard_v21.py",
        "backend/scripts/image_regional_continuation_runtime_v41.py",
    }
    return {name: digest(ROOT / name) for name in sorted(names)}


def prepare():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Never scored new destination required")
    audit = read(BASE / "audio-reserve-audit-v60/summary.json")
    review = read(BASE / "audio-reserve-audit-v60/additive-review.json")
    require(not audit["errors"] and not audit["screening_record_paths"]
            and not review["normalized_cross_role_overlaps"] and not review["alias_screening_matches"],
            "Complete no-recorded-exposure and separated-role review required")
    require(digest(SOURCE / "source-plan.json") == SOURCE_SHA
            and digest(SOURCE / "confirmation/variants.json") == INPUT_SHA, "Original reserve must stay fixed")
    body = read(SOURCE / "confirmation/variants.json")
    rows = body["variants"]
    require(len(rows) == len({r["id"] for r in rows}) == 1080 and {r["role"] for r in rows} == {"confirmation"},
            "All 1080 reserve variants required")
    require(Counter(r["label"] for r in rows) == {0: 540, 1: 540}, "Complete balanced reserve required")
    for row in rows:
        require(digest(Path(body["source_root"]) / row["path"]) == row["sha256"], "Prespecified input bytes changed")
    require({name: digest(path) for name, path in HEADS.items()} == HEAD_SHA, "Previously fitted heads changed")
    old_lock = read(BASE / "audio-v13-wavlm/lock.json")
    wavlm_root = Path(old_lock["weights_root"])
    assets = features.assets(wavlm_root)
    require(assets == old_lock["pretrained_assets"], "Original commercially permitted assets required")
    consumed = read(SOURCE / "development/variants.json")
    artifacts = {str(path.relative_to(ROOT)): digest(path) for path in [
        *HEADS.values(), BASE / "audio-reserve-audit-v60/summary.json",
        BASE / "audio-reserve-audit-v60/additive-review.json", SOURCE / "development/variants.json",
        BASE / "audio-v7/development/manifest.json",
        SOURCE / "development/representations.jsonl", BASE / "audio-v13-wavlm/features.jsonl",
        BASE / "audio-v13-wavlm/responses-callable.jsonl",
    ]}
    frozen = {"schema": "chipmunk.audio-shared-confirmation-freeze/62", "inputs_sha256": INPUT_SHA,
              "source_sha256": SOURCE_SHA, "heads_sha256": HEAD_SHA, "artifacts_sha256": artifacts,
              "code_sha256": implementation(), "runtime": runtime(),
              "installed_implementation_sha256": original.installed_implementation(),
              "baseline_weights": weights(), "wavlm_assets": assets, "methods": METHODS,
              "additional_method": "v7 with its own unchanged fitted head, guards and stability rule",
              "v7_source_overlap": v7_overlap(),
              **parity_plan(parity_selection(consumed["variants"])), "inputs": 1080,
              "foregrounds": 120, "connected_foreground_groups": 118, "chunk_size": CHUNK,
              "chunks": 12, "probability_refit": False, "combined_probability": None,
              "bootstrap": {"seed": 6207, "replicates": 2000, "unit": "connected foreground group",
                            "limits": "Exploratory paired count-change interval; shared backgrounds are not independent and no customer confidence interval is established."},
              "decision_policy": "Report gains, lost correct cases, both wrong directions, unassessed/inconclusive, generators and conditions. No zero-error or no-loss automatic rejection. Compare tradeoffs; no automatic default or service confidence change.",
              "fresh_calls_started": False, "deployment_eligible": False}
    TRIAL.mkdir()
    write(TRIAL / "private-paths.json", {"wavlm_root": str(wavlm_root), "parity_ids": parity_selection(consumed["variants"])})
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"planned": 1080, "parity_inputs": 18, "methods": len(METHODS) + 1,
                      "new_model_calls": 0, "freeze_sha256": digest(PUBLIC)}), flush=True)


def profile_pass(path):
    result = read(path)
    require(result["child_exit_code"] == 0 and result["guard"] is None and result["sample_errors"] == 0
            and not result["resource_limit_observations"], "Required preceding bounded stage failed")


def admission(*, fresh):
    lock, start = read(TRIAL / "freeze.json"), read(TRIAL / "execution-start.json")
    require(digest(PUBLIC) == digest(TRIAL / "freeze.json") == start["freeze_sha256"]
            and start["remote_verified_before_project_forwards"] is True, "Published exact prospective freeze required")
    require(lock["code_sha256"] == implementation() and lock["runtime"] == runtime()
            and lock["installed_implementation_sha256"] == original.installed_implementation(),
            "Frozen code or numerical runtime changed")
    require(json.loads(json.dumps(METHODS)) == lock["methods"], "Fixed combinations changed")
    require(digest(SOURCE / "source-plan.json") == lock["source_sha256"] == SOURCE_SHA
            and digest(SOURCE / "confirmation/variants.json") == lock["inputs_sha256"] == INPUT_SHA,
            "Reserve metadata changed")
    require(all(digest(ROOT / path) == expected for path, expected in lock["artifacts_sha256"].items()),
            "Frozen fitting, parity or source-review artifact changed")
    require(lock["baseline_weights"] == weights(), "Original checkpoint changed")
    require(digest(serving.PATH) == serving.SHA256 == "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf",
            "Protected current calibration changed")
    profile_pass(TRIAL.parent / (TRIAL.name + "-preparation-profile-a02/profile.json"))
    if fresh:
        profile_pass(TRIAL / "parity-profile/profile.json")
        parity = read(TRIAL / "parity.json")
        require(parity["passed"] is True and parity["cases"] == 18, "Complete consumed-source native parity required")
    return lock


def decisions(result):
    if result["status"] == "not_applicable":
        return {name: "unassessed" for name in [*METHODS, "v7"]}
    require(result["status"] == "ok", "Failed measured input cannot become a conclusion")
    parts = {"current": result["original_decision"],
             **{arm: result["candidates"][arm]["decision"] for arm in method.ARMS}}
    output = {name: combine(tuple(parts[member] for member in members), policy)
              for name, (members, policy) in METHODS.items()}
    output["v7"] = result["candidate_decision"]
    return output


def score(pcm, current, activity, candidate, heads, captured):
    result = score_one(pcm, current, activity, heads["v7"], captured)
    result.update(features.extract(pcm, candidate) if result["status"] == "ok" else
                  {name: None for name in ("wavlm", "wavlm_sha256", "feature_frames", "valid_feature_frames", "input_pcm_sha256")})
    probabilities = {arm: method.predict(result, heads["v13"]["fitted"][arm]) for arm in method.ARMS}
    probe = {"status": "not_required", "raw": None, "wavlm": None}
    if any(initial(probabilities[arm], heads["v13"]["guards"][arm]) == "original" for arm in method.ARMS):
        band = band_view(pcm)
        if result["probe"]["status"] == "computed":
            require(result["probe"]["filtered_pcm_sha256"] == hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest(),
                    "Shared original stability probe has different input samples")
            raw = result["probe"]["raw"]
        else:
            raw, _ = capture(current, band, captured)
        probe.update(features.extract(band, candidate), raw=raw, status="computed")
    result["candidates"] = {}
    for arm in method.ARMS:
        band_p = method.predict(probe, heads["v13"]["fitted"][arm]) if probe["status"] == "computed" else None
        detail = final(probabilities[arm], band_p, result["raw"], probe["raw"], heads["v13"]["guards"][arm])
        result["candidates"][arm] = {"p": probabilities[arm], "decision": detail["decision"], "detail": detail}
    result["wavlm_probe"] = probe
    result["decisions"] = decisions(result)
    return result


def load_models(lock):
    current, activity = models()  # Sole owner of inter-op initialization.
    root = Path(read(TRIAL / "private-paths.json")["wavlm_root"])
    require(features.assets(root) == lock["wavlm_assets"], "Frozen feature checkpoint changed")
    candidate = features.load(root)
    captured = []
    hook = current.head.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().cpu().numpy().copy()))
    return current, activity, candidate, {name: read(path) for name, path in HEADS.items()}, captured, hook


def selected_journal(path, identifiers):
    result = {}
    with path.open() as stream:
        next(stream)
        for line in stream:
            row = json.loads(line)
            if row["id"] in identifiers:
                result[row["id"]] = row
    require(set(result) == set(identifiers), "All predetermined parity references required")
    return result


def parity():
    lock = admission(fresh=False)
    body = read(SOURCE / "development/variants.json")
    identifiers = read(TRIAL / "private-paths.json")["parity_ids"]
    require(parity_plan(identifiers) == {key: lock[key] for key in ("parity_inputs", "parity_ids_sha256")}
            and identifiers == parity_selection(body["variants"]), "Private deterministic parity plan changed")
    selected = {row["id"]: row for row in body["variants"] if row["id"] in identifiers}
    raw = selected_journal(SOURCE / "development/representations.jsonl", selected)
    wavlm = selected_journal(BASE / "audio-v13-wavlm/features.jsonl", selected)
    old = selected_journal(BASE / "audio-v13-wavlm/responses-callable.jsonl", selected)
    current, activity, candidate, heads, captured, hook = load_models(lock)
    completed = []
    with (TRIAL / "parity-records.jsonl").open("x") as stream:
        try:
            for identifier in identifiers:
                measured = score(input_pcm(body, selected[identifier]), current, activity, candidate, heads, captured)
                exact = {"status": measured["status"] == raw[identifier]["status"],
                         "raw": measured["raw"] == raw[identifier]["raw"],
                         "embedding": measured["embedding"] == raw[identifier]["embedding"],
                         "wavlm": measured["wavlm"] == wavlm[identifier]["wavlm"],
                         "current_decision": measured["original_decision"] == old[identifier]["original_decision"],
                         "candidate_decisions": all(measured["candidates"][arm] == old[identifier]["candidates"][arm]
                                                    for arm in method.ARMS)}
                row = {"id": identifier, "checks": exact, "passed": all(exact.values())}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                completed.append(row)
                print(json.dumps({"parity_completed": len(completed), "planned": 18, "passed": row["passed"]}), flush=True)
                require(row["passed"], "Preserve parity failure; no fresh reserve calls")
        finally:
            hook.remove()
    write(TRIAL / "parity.json", {"passed": all(row["passed"] for row in completed), "cases": len(completed),
                                      "records_sha256": digest(TRIAL / "parity-records.jsonl"),
                                      "scope": "Exact consumed native features, current decisions and complete candidate decision details across 18 fixed truth/condition cells. Not fresh accuracy."})


def run_chunk(index):
    require(type(index) is int and 0 <= index < 12, "One predetermined chunk index required")
    lock = admission(fresh=True)
    destination = TRIAL / f"chunk-{index:02d}.jsonl"
    require(not destination.exists(), "Existing or failed chunk is never retried or overwritten")
    for previous in range(index):
        profile_pass(TRIAL / f"chunk-{previous:02d}-profile/profile.json")
        require(read(TRIAL / f"chunk-{previous:02d}-complete.json")["completed"] == CHUNK,
                "Every preceding chunk must be complete")
    body = read(SOURCE / "confirmation/variants.json")
    rows = body["variants"][index * CHUNK:(index + 1) * CHUNK]
    require(len(rows) == CHUNK, "Complete predetermined chunk required")
    current, activity, candidate, heads, captured, hook = load_models(lock)
    completed = 0
    with destination.open("x") as stream:
        try:
            for source in rows:
                started = time.monotonic()
                result = {key: source[key] for key in ("id", "base_id", "group", "role", "label", "condition", "generator", "sha256", "pcm_sha256")}
                try:
                    result.update(score(input_pcm(body, source), current, activity, candidate, heads, captured))
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
    lock = admission(fresh=True)
    sources = read(SOURCE / "confirmation/variants.json")["variants"]
    rows, bindings = [], {}
    for index in range(12):
        profile_pass(TRIAL / f"chunk-{index:02d}-profile/profile.json")
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
    result = {"schema": "chipmunk.audio-shared-confirmation-result/62", "freeze_sha256": digest(PUBLIC),
              "journal_sha256": bindings, "totals": totals, "strata": strata, "changes_from_current": gain,
              "paired_group_bootstrap_95pct_net_correct_count_change": intervals,
              "assessed": sum(row["status"] == "ok" for row in rows),
              "unassessed": sum(row["status"] == "not_applicable" for row in rows),
              "combined_probability": None, "serving_changed": False,
              "limits": "Untouched fixed recordings, some historically seen voices, three AI generators, short English speech, shared backgrounds and correlated variants. Decision combinations are not calibrated probabilities. No independent outside reviewer or general accuracy guarantee."}
    write(TRIAL / "result.json", result)
    # No record IDs, learned coefficients, raw media or voice representations.
    write(ROOT / "docs/evaluation-panels/audio-shared-confirmation-v62-result.json", result)
    print(json.dumps({"totals": totals, "changes_from_current": gain}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "parity", "chunk", "evaluate"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.stage == "chunk":
        run_chunk(args.index)
    elif args.index is not None:
        parser.error("Only a chunk takes an index")
    else:
        {"prepare": prepare, "parity": parity, "evaluate": evaluate}[args.stage]()
