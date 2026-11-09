"""Prospective common-input comparison with independent fitting/calibration roles."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from scripts import audio_fusion_evaluation_v84 as evaluation
from scripts import audio_score_fusion_v83 as fusion
from scripts import audio_shared_confirmation_v62 as prior
from scripts.audio_calibration_roles_v71 import ROLES
from scripts.audio_reserve_audit_v60 import digest, write

ROOT, BASE = prior.ROOT, prior.BASE
TRIAL = BASE / "audio-combined-confidence-v84"
SOURCE = BASE / "audio-confidence-mixtures-v82/variants.json"
SOURCE_SHA = "8411c76325bc4dcde649db08ed05747c5eb917df13d811d1068b7395ee562d24"
SOURCE_MANIFEST = BASE / "audio-complete-sources-v81/manifest.json"
SOURCE_MANIFEST_SHA = "8b667675a803bd03851c27251e08a63b040b2bd29474462f9f0a4e3af9108ea6"
HISTORY = BASE / "audio-history-v58/catalogue.json"
HISTORY_SHA = "530df71b6f6443a71233bb56ccd5ed9743f4874d34fccc50a7172f2bf050d2c1"
OUTPUT = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify/audio-confidence-extraction-v84")
PUBLIC = ROOT / "docs/evaluation-panels/audio-combined-confidence-v84-freeze.json"
CHUNK, ROLE_CHUNKS = 87, 12
read, require, profile_pass = prior.read, prior.require, prior.profile_pass


def offline():
    attempts = []

    def audit(event, arguments):
        if event == "socket.connect":
            attempts.append(event)
            raise RuntimeError("Offline comparison refused a network connection")

    sys.addaudithook(audit)
    return attempts


def check_compact(record, full_line):
    require(hashlib.sha256(full_line.encode()).hexdigest() == record["full_record_sha256"],
            "Compact record differs from its complete private response")
    full = json.loads(full_line)
    require(all(value == full[key] for key, value in record.items() if key != "full_record_sha256"),
            "Compact scores/identity differ from complete private response")


def implementation():
    names = {
        "backend/scripts/audio_combined_confidence_v84.py", "backend/scripts/audio_combined_runtime_v84.py",
        "backend/scripts/audio_fusion_evaluation_v84.py", "backend/tests/test_audio_fusion_evaluation_v84.py",
        "backend/scripts/audio_score_fusion_v83.py", "backend/tests/test_audio_score_fusion_v83.py",
        "backend/tests/test_audio_combined_confidence_v84.py", "docs/audio-combined-confidence-v84-protocol.md",
        "docs/audio-score-fusion-v83-method.md", "backend/scripts/audio_tree_sampler_v64.py",
    }
    return {**prior.implementation(), **{name: digest(ROOT / name) for name in sorted(names)}}


def validate_panel(body):
    rows = body["variants"]
    require(len(rows) == len({row["id"] for row in rows}) == 4176, "Every unique prepared variant required")
    roles = [row["role"] for row in rows]
    require(roles == [role for role in ROLES for _ in range(1044)], "All four roles must remain in fixed order")
    owners = {}
    for row in rows:
        require(row["samples"] == 64000 and row["sample_rate"] == 16000
                and type(row["label"]) is int and row["label"] in {0, 1}, "Exact four-second inputs and known truths required")
        for value in ("group:" + row["group"], "bytes:" + row["sha256"], "pcm:" + row["pcm_sha256"]):
            require(value not in owners or owners[value] == row["role"], "Known foreground or variant alias crosses roles")
            owners[value] = row["role"]
    return rows


def prepare():
    old = prior.admission(fresh=True)
    require(digest(SOURCE) == SOURCE_SHA and digest(SOURCE_MANIFEST) == SOURCE_MANIFEST_SHA
            and digest(HISTORY) == HISTORY_SHA, "Exact qualified sources, mixtures and exclusion history required")
    profile_pass(BASE / "audio-confidence-mixtures-v82/finish-profile/profile.json")
    body = read(SOURCE)
    rows = validate_panel(body)
    history = set(read(HISTORY)["keys"])
    overlaps = {key for row in rows for key in ("bytes:" + row["sha256"], "pcm:" + row["pcm_sha256"]) if key in history}
    require(not overlaps, "Prepared variants must pass recorded exact historical byte/PCM exclusions")
    for row in rows:
        require(digest(Path(body["source_root"]) / row["path"]) == row["sha256"], "Prepared variant bytes changed")
    require(not TRIAL.exists() and not OUTPUT.exists() and not PUBLIC.exists(), "Never extracted study destinations required")
    TRIAL.mkdir()
    OUTPUT.mkdir(parents=True)
    lock = {"schema": "chipmunk.audio-combined-confidence-freeze/84", "inputs_sha256": SOURCE_SHA,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA, "historical_catalogue_sha256": HISTORY_SHA,
            "recorded_exact_historical_variant_aliases": len(overlaps), "code_sha256": implementation(),
            "old_method_freeze_sha256": digest(prior.PUBLIC), "runtime": old["runtime"],
            "installed_implementation_sha256": old["installed_implementation_sha256"],
            "baseline_weights": old["baseline_weights"], "wavlm_assets": old["wavlm_assets"],
            "heads_sha256": old["heads_sha256"], "component_methods": old["methods"],
            "additional_component_method": "v7 unchanged calibration and guards",
            "fusion_method": fusion.METHOD, "selection_rule": evaluation.SELECTION,
            "research_operating_points": evaluation.OPERATING_POINTS,
            "inputs": 4176, "roles": list(ROLES), "inputs_per_role": 1044,
            "chunk_size": CHUNK, "chunks_per_role": ROLE_CHUNKS, "chunks": 48,
            "native_parity_inputs": 18, "fresh_forwards_started": False,
            "raw_feature_storage": "Private external full-response journals; compact scores retain whole-record hashes",
            "calibration_prior": "Balanced 50% AI reference prior and equal observed family/group/source/variant weights; not customer prevalence",
            "confirmation_guard": "Final-role forwards require a remotely published, exact fixed selection receipt",
            "uncertainty": "Descriptive held-role metrics only; correlated variants and shared backgrounds do not establish independent confidence intervals",
            "protected_audio_sha256": digest(prior.serving.PATH), "defaults_changed": False,
            "scope": "Four-second short English speech; three known generator families, historical expressive profiles/music, unknown pretraining overlap. Local research comparison only, no outside human review or deployment."}
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(json.dumps({"inputs": 4176, "methods": len(prior.METHODS) + 3, "new_model_calls": 0,
                      "freeze_sha256": digest(PUBLIC)}), flush=True)


def admission(*, fresh=False):
    old = prior.admission(fresh=True)
    lock, start = read(TRIAL / "freeze.json"), read(TRIAL / "execution-start.json")
    require(digest(PUBLIC) == digest(TRIAL / "freeze.json") == start["freeze_sha256"]
            and start["remote_verified_before_study_forwards"] is True, "Published exact confidence-study freeze required")
    require(lock["code_sha256"] == implementation() and digest(SOURCE) == lock["inputs_sha256"] == SOURCE_SHA
            and digest(SOURCE_MANIFEST) == lock["source_manifest_sha256"] == SOURCE_MANIFEST_SHA,
            "Frozen experiment code or inputs changed")
    require(lock["runtime"] == old["runtime"] and lock["fusion_method"] == fusion.METHOD
            and lock["selection_rule"] == evaluation.SELECTION and lock["old_method_freeze_sha256"] == digest(prior.PUBLIC),
            "Frozen models, calibration method or selection rule changed")
    profile_pass(TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json"))
    if fresh:
        profile_pass(TRIAL / "parity-profile/profile.json")
        require(read(TRIAL / "parity.json")["passed"] is True, "Complete exact native parity must pass before new sources")
    return old


def parity():
    lock = admission()
    attempts = offline()
    body = read(prior.SOURCE / "development/variants.json")
    identifiers = read(prior.TRIAL / "private-paths.json")["parity_ids"]
    selected = {row["id"]: row for row in body["variants"] if row["id"] in identifiers}
    raw = prior.selected_journal(prior.SOURCE / "development/representations.jsonl", selected)
    wavlm = prior.selected_journal(BASE / "audio-v13-wavlm/features.jsonl", selected)
    old = prior.selected_journal(BASE / "audio-v13-wavlm/responses-callable.jsonl", selected)
    current, activity, candidate, heads, captured, hook = prior.load_models(lock)
    checks = []
    with (TRIAL / "parity-records.jsonl").open("x") as stream:
        try:
            for identifier in identifiers:
                pcm = prior.input_pcm(body, selected[identifier])
                measured = prior.score(pcm, current, activity, candidate, heads, captured)
                passed = (measured["status"] == raw[identifier]["status"] and measured["raw"] == raw[identifier]["raw"]
                          and measured["embedding"] == raw[identifier]["embedding"]
                          and measured["wavlm"] == wavlm[identifier]["wavlm"]
                          and measured["original_decision"] == old[identifier]["original_decision"]
                          and all(measured["candidates"][arm] == old[identifier]["candidates"][arm] for arm in prior.method.ARMS))
                require(fusion.flatness(pcm) == fusion.flatness(pcm), "Measured quality must repeat exactly")
                row = {"id": identifier, "passed": passed}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                checks.append(passed)
                print(json.dumps({"parity_completed": len(checks), "planned": 18, "passed": passed}), flush=True)
                require(passed, "Stopped native parity must remain failed; no fresh inputs")
        finally:
            hook.remove()
    require(len(checks) == 18 and all(checks), "All native parity cells required")
    require(not attempts, "Parity attempted network access")
    write(TRIAL / "parity.json", {"passed": True, "cases": 18,
                                  "records_sha256": digest(TRIAL / "parity-records.jsonl"),
                                  "scope": "Exact consumed native raw/features/decisions and repeatable measured flatness. Not fresh accuracy."})


def final_guard():
    path = TRIAL / "selection.json"
    public = ROOT / "docs/evaluation-panels/audio-combined-confidence-v84-selection.json"
    start = read(TRIAL / "confirmation-start.json")
    require(digest(path) == digest(public) == start["selection_sha256"]
            and start["remote_verified_before_confirmation_forwards"] is True,
            "Exact selection must be remotely published before any final-role forward")


def extract(index):
    require(type(index) is int and 0 <= index < 48, "One fixed extraction chunk required")
    lock = admission(fresh=True)
    for previous in range(index):
        profile_pass(TRIAL / f"chunk-{previous:02d}-profile/profile.json")
        require(read(TRIAL / f"chunk-{previous:02d}-complete.json")["completed"] == CHUNK, "Every earlier chunk must finish")
    if index >= 24:
        profile_pass(TRIAL / "train-profile/profile.json")
        require(digest(TRIAL / "trained.json") == read(TRIAL / "training-summary.json")["trained_sha256"],
                "Fit and own calibration must finish before selection sources")
    if index >= 36:
        profile_pass(TRIAL / "select-profile/profile.json")
        final_guard()
    body = read(SOURCE)
    sources = validate_panel(body)[index * CHUNK:(index + 1) * CHUNK]
    attempts = offline()
    current, activity, candidate, heads, captured, hook = prior.load_models(lock)
    destination, full_path = TRIAL / f"chunk-{index:02d}.jsonl", OUTPUT / f"chunk-{index:02d}-full.jsonl"
    completed = 0
    with destination.open("x") as compact, full_path.open("x") as full:
        try:
            for source_row in sources:
                started = time.monotonic()
                metadata = {key: source_row[key] for key in
                            ("id", "base_id", "group", "role", "label", "kind", "condition", "sha256", "pcm_sha256")}
                try:
                    pcm = prior.input_pcm(body, source_row)
                    result = prior.score(pcm, current, activity, candidate, heads, captured)
                    quality = fusion.flatness(pcm)
                except Exception as error:
                    compact.write(json.dumps({**metadata, "status": "failed", "error_type": type(error).__name__}) + "\n")
                    compact.flush()
                    raise
                record = {**metadata, **result, "quality_flatness": quality, "runtime_s": time.monotonic() - started}
                payload = json.dumps(record, allow_nan=False) + "\n"
                full.write(payload)
                full.flush()
                small = {**metadata, "quality_flatness": quality, "runtime_s": record["runtime_s"],
                         "full_record_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                         **{key: result[key] for key in ("status", "raw", "original_p", "original_decision", "candidate_p",
                                                        "candidate_decision", "candidates", "decisions", "embedding_sha256",
                                                        "wavlm_sha256", "input_pcm_sha256")}}
                compact.write(json.dumps(small, allow_nan=False) + "\n")
                compact.flush()
                completed += 1
                if completed % 15 == 0:
                    print(json.dumps({"chunk": index, "completed": completed, "planned": CHUNK}), flush=True)
        finally:
            hook.remove()
    require(completed == CHUNK, "Complete fixed chunk required")
    require(not attempts, "Extraction attempted network access")
    write(TRIAL / f"chunk-{index:02d}-complete.json", {"completed": completed, "compact_sha256": digest(destination),
                                                      "full_sha256": digest(full_path)})


def load_role(role):
    index = ROLES.index(role)
    sources = read(SOURCE)["variants"][index * 1044:(index + 1) * 1044]
    rows, bindings = [], {}
    for chunk in range(index * ROLE_CHUNKS, (index + 1) * ROLE_CHUNKS):
        profile = TRIAL / f"chunk-{chunk:02d}-profile/profile.json"
        profile_pass(profile)
        path, full = TRIAL / f"chunk-{chunk:02d}.jsonl", OUTPUT / f"chunk-{chunk:02d}-full.jsonl"
        receipt = read(TRIAL / f"chunk-{chunk:02d}-complete.json")
        require(receipt == {"completed": CHUNK, "compact_sha256": digest(path), "full_sha256": digest(full)},
                "Complete unchanged compact/full journals required")
        with path.open() as compact_stream, full.open() as full_stream:
            for compact_line, full_line in zip(compact_stream, full_stream, strict=True):
                record = json.loads(compact_line)
                check_compact(record, full_line)
                rows.append(record)
        bindings[str(chunk)] = {**receipt, "profile_sha256": digest(profile)}
    require(len(rows) == 1044, "Every assigned role input required")
    for row, original in zip(rows, sources, strict=True):
        require(all(row[key] == original[key] for key in
                    ("id", "base_id", "group", "role", "label", "kind", "condition", "sha256", "pcm_sha256"))
                and row["decisions"] == prior.decisions(row), "Fixed source identity and component decisions required")
        fusion.predictors(row)
    return rows, bindings


def train():
    admission(fresh=True)
    fitting, fit_bindings = load_role("fusion_fit")
    calibration, cal_bindings = load_role("confidence_cal")
    artifacts = {arm: fusion.calibrate(fusion.fit(fitting, arm), calibration) for arm in fusion.METHOD["arms"]}
    write(TRIAL / "trained.json", artifacts)
    result = {"schema": "chipmunk.audio-combined-training-summary/84", "trained_sha256": digest(TRIAL / "trained.json"),
              "fit_stages": fit_bindings, "calibration_stages": cal_bindings,
              "fit_inputs": len(fitting), "calibration_inputs": len(calibration),
              "branches": {arm: {name: {key: value for key, value in detail.items() if key not in {"coefficients", "calibration"}}
                                  for name, detail in artifact["branches"].items()} for arm, artifact in artifacts.items()},
              "selection_inputs_used": 0, "confirmation_inputs_used": 0, "defaults_changed": False}
    write(TRIAL / "training-summary.json", result)
    print(json.dumps({"fit_inputs": 1044, "calibration_inputs": 1044,
                      "branch_statuses": {arm: {name: value["status"] for name, value in artifact["branches"].items()}
                                          for arm, artifact in artifacts.items()}}), flush=True)


def select():
    admission(fresh=True)
    profile_pass(TRIAL / "train-profile/profile.json")
    require(digest(TRIAL / "trained.json") == read(TRIAL / "training-summary.json")["trained_sha256"], "Frozen trained artifacts required")
    rows, bindings = load_role("selection")
    artifacts = read(TRIAL / "trained.json")
    measured = evaluation.evaluate(rows, artifacts)
    choice = evaluation.choose(measured, artifacts)
    result = {"schema": "chipmunk.audio-combined-selection/84", "freeze_sha256": digest(PUBLIC),
              "trained_sha256": digest(TRIAL / "trained.json"), "source_stages": bindings,
              "selection_rule": evaluation.SELECTION, "selected_arm": choice,
              "evaluation": measured, "confirmation_inputs_used": 0, "defaults_changed": False,
              "scope": "Fixed independent-role selection, balanced-reference-prior scores, correlated controlled variants. No general reliability or public threshold claim."}
    write(TRIAL / "selection.json", result)
    write(ROOT / "docs/evaluation-panels/audio-combined-confidence-v84-selection.json", result)
    print(json.dumps({"selected_arm": choice, "inputs": len(rows), "totals": measured["totals"]}), flush=True)


def confirm():
    admission(fresh=True)
    final_guard()
    rows, bindings = load_role("confirmation")
    selection = read(TRIAL / "selection.json")
    require(digest(TRIAL / "trained.json") == selection["trained_sha256"], "Selected fitted/calibrated methods changed")
    result = {"schema": "chipmunk.audio-combined-confirmation/84", "freeze_sha256": digest(PUBLIC),
              "selection_sha256": digest(TRIAL / "selection.json"), "selected_arm": selection["selected_arm"],
              "source_stages": bindings, "evaluation": evaluation.evaluate(rows, read(TRIAL / "trained.json")),
              "defaults_changed": False,
              "scope": "Fresh assigned final-role comparison only. Controlled four-second English speech and known generator families with correlated variants/shared backgrounds and some historically seen profiles/music. No independent external human review, population confidence, public threshold or court-reliability claim. No serving change."}
    write(TRIAL / "result.json", result)
    write(ROOT / "docs/evaluation-panels/audio-combined-confidence-v84-result.json", result)
    print(json.dumps({"selected_arm": result["selected_arm"], "inputs": len(rows), "totals": result["evaluation"]["totals"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "parity", "extract", "train", "select", "confirm"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.index)
    else:
        require(args.index is None, "Only one fixed extraction chunk takes an index")
        {"prepare": prepare, "parity": parity, "train": train, "select": select, "confirm": confirm}[args.command]()
