"""Execute the corrected identity profile with an unchanged statistical learner.

The old experiment remains intact. Exact-input cached features are accepted only
with pinned lineage, weights/runtime/head parity and preselected native replay.
Fresh confirmation is never accepted by this development runner.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from app.detectors.audio import calibration as serving
from scripts import audio_multisource_v8 as method
from scripts import audio_residual_head_v6 as learner
from scripts import run_audio_multisource_v8 as prior
from scripts.audio_source_identity_v9 import digest, role_violations
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.score_audio_confirmation_v7 import models, runtime, speech_windows, weights

SOURCE_SHA = "f3d7a7f2e20a9d2aa9092c371f5af4d2a05ed9172e4652677049a8286f61605f"
VARIANTS_SHA = "6cf19bbe44433b4dd85dd7758cbef9a33c9dd184417dd7a2ef72c1505a6d6f7c"
CONFIRMATION_SHA = "14828469353f3b37a60da86f6588b817d946846303c275afa46e4f752bce4e26"
CACHE_SHA = "68673bfb01c3e5b80609299c4455f3e436d723312f328536bacf95fbd2a7e6f4"
OLD_VARIANTS_SHA = "911a865638c1a49f367071a1a0eaebcf9c196b9dc2ae64b3f965c92d81e79515"
OLD_LOCK_SHA = "94528b1f27633d00ab82d1485542874d7a7e9200b08ac364130864adb4fe527e"
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}
CONDITIONS = (
    "clean",
    "noise_0",
    "noise_5",
    "noise_15",
    "music_5",
    "music_15",
    "speech_0",
    "speech_5",
    "speech_15",
)


def implementation():
    project = Path(__file__).resolve().parents[2]
    names = set(prior.implementation()) | {
        "backend/scripts/audio_source_identity_v9.py",
        "backend/scripts/plan_audio_identity_v9.py",
        "backend/scripts/prepare_audio_identity_inputs_v9.py",
        "backend/scripts/export_audio_identity_v9.py",
        "backend/scripts/review_audio_identity_plan_v9.py",
        "backend/scripts/rebuild_audio_identity_inputs_v9.py",
        "backend/scripts/run_audio_identity_v9.py",
        "backend/scripts/review_audio_identity_execution_v9.py",
        "backend/app/engine/fetchers.py",
        "backend/app/engine/registry.py",
        "docs/audio-identity-v9-execution-protocol.md",
    }
    return {name: digest(project / name) for name in sorted(names)}


def validate(body, plan):
    rows = body["variants"]
    if body["panel"] != "development" or any(r["role"] not in COUNTS for r in rows):
        raise ValueError("Development runner refuses fresh confirmation")
    if Counter(r["role"] for r in rows) != COUNTS or len({r["id"] for r in rows}) != 4248:
        raise ValueError("Every corrected consumed input and role required")
    if role_violations(plan["sources"] + plan["backgrounds"]):
        raise ValueError("Canonical speaker connection crosses a role or statistical group")
    sources = {r["id"]: r for r in plan["sources"] if r["role"] != "confirmation"}
    backgrounds = {r["sha256"]: r for r in plan["backgrounds"]}
    by_source = defaultdict(list)
    for row in rows:
        source = sources[row["base_id"]]
        if row["source_sha256"] != source["sha256"] or any(
            row[k] != source[k] for k in ("role", "group", "label", "generator", "dataset")
        ):
            raise ValueError("Corrected source/role/group/truth binding changed")
        if (
            row["samples"] != 64000
            or row["source_interval_samples"] != [0, 64000]
            or row["sample_rate"] != 16000
            or row["seconds"] != 4.0
        ):
            raise ValueError("Original four-second source clock changed")
        if row["condition"] == "clean":
            if any(
                row[k] is not None
                for k in ("background_sha256", "background_role", "background_kind", "snr_db")
            ):
                raise ValueError("Clean input cannot have a hidden background")
        elif row["condition"] in CONDITIONS:
            kind, snr = row["condition"].split("_")
            if (
                row["background_sha256"] is None
                or row["background_kind"] != kind
                or row["snr_db"] != int(snr)
            ):
                raise ValueError("Condition must match its actual background recipe")
            bg = backgrounds[row["background_sha256"]]
            if (
                bg["role"] != row["role"]
                or row["background_role"] != row["role"]
                or bg["kind"] != row["background_kind"]
            ):
                raise ValueError("Backgrounds cannot cross corrected roles")
        by_source[row["base_id"]].append(row)
    if set(by_source) != set(sources) or any(
        Counter(r["condition"] for r in rows) != Counter(CONDITIONS) for rows in by_source.values()
    ):
        raise ValueError("Complete source/condition matrix required")


def validate_feature(record, reference):
    if record["status"] not in ("ok", "not_applicable"):
        raise ValueError("Failed features cannot be reused or fitted")
    if record["status"] == "not_applicable":
        if record["raw"] is not None or record["embedding"] is not None or record["windows"]:
            raise ValueError("Unassessed feature status disagrees with its payload")
        return
    feature = np.asarray(record["embedding"], np.float64)
    if feature.shape != (1024,) or not np.isfinite(feature).all() or not serving.valid(record["raw"]):
        raise ValueError("Finite original representation/response required")
    if learner.digest_embedding(record["embedding"]) != record["embedding_sha256"]:
        raise ValueError("Stored representation changed")
    direction = np.asarray(reference["head_weight"], np.float64)
    if (
        direction.shape != (1024,)
        or not np.isfinite(direction).all()
        or not np.isfinite(reference["head_bias"])
    ):
        raise ValueError("Finite original checkpoint head required")
    value = float(expit(feature @ direction + reference["head_bias"]))
    if abs(value - record["raw"]) > 1e-6 or record["windows"] != [{"start_sample": 0, "end_sample": 64000}]:
        raise ValueError("Original head/complete-window parity failed")


def cached_rows(receipt, old_manifest, expected_weights, expected_runtime):
    if digest(receipt) != CACHE_SHA or digest(old_manifest) != OLD_VARIANTS_SHA:
        raise ValueError("Retained original cache/input receipt changed")
    header, records = load_rows(receipt)
    original = json.loads(old_manifest.read_text())["variants"]
    if (
        header["lock_sha256"] != OLD_LOCK_SHA
        or header["weights"] != expected_weights
        or header["runtime"] != expected_runtime
    ):
        raise ValueError("Cached model/runtime lineage differs from the current verified inputs")
    if len(records) != 1882 or [r["id"] for r in records] != [r["id"] for r in original[:1882]]:
        raise ValueError("Retained cache must be the exact terminal prefix, including unassessed cases")
    result = {}
    for old, record in zip(original, records, strict=False):
        if any(old[k] != record[k] for k in ("id", "role", "sha256", "pcm_sha256")):
            raise ValueError("Cached source/PCM/role binding changed")
        validate_feature(record, header)
        if record["sha256"] in result:
            raise ValueError("Ambiguous repeated cache waveform")
        result[record["sha256"]] = record
    return header, result


def native_selection(rows, cache):
    eligible = [
        r for r in rows if r["sha256"] in cache and cache[r["sha256"]]["pcm_sha256"] == r["pcm_sha256"]
    ]
    cells = defaultdict(list)
    for row in eligible:
        cells[row["role"], row["label"], row["condition"]].append(row["id"])
    selected = {min(ids) for ids in cells.values()}
    unassessed = [r["id"] for r in eligible if cache[r["sha256"]]["status"] == "not_applicable"]
    if unassessed:
        selected.add(min(unassessed))
    return sorted(selected)


def freeze(root, output):
    plan_path, path = root / "source-plan.json", root / "development/variants.json"
    if (
        digest(plan_path) != SOURCE_SHA
        or digest(path) != VARIANTS_SHA
        or digest(root / "confirmation/variants.json") != CONFIRMATION_SHA
    ):
        raise ValueError("Published corrected source/recipe set changed")
    plan, body = json.loads(plan_path.read_text()), json.loads(path.read_text())
    validate(body, plan)
    project = Path(__file__).resolve().parents[2]
    if (
        body["preparation_code_sha256"]
        != digest(project / "backend/scripts/prepare_audio_identity_inputs_v9.py")
        or plan["identity_code_sha256"] != digest(project / "backend/scripts/audio_source_identity_v9.py")
        or plan["planning_code_sha256"] != digest(project / "backend/scripts/plan_audio_identity_v9.py")
    ):
        raise ValueError("Corrected source preparation code changed")
    rebuilt = json.loads((root / "reproduction/public-only-rebuild/summary.json").read_text())
    if (
        not rebuilt["passed"]
        or rebuilt["counts"] != {"ok": 5328}
        or rebuilt["receipts_sha256"] != "eb44576b8d804126e6ea5c9069a9e9bf64b4cddea4c3bd7496db0814a65a7ec8"
    ):
        raise ValueError("Complete public-only corrected input replay required")
    checkpoint, environment = weights(), runtime()
    old_root = Path(plan["original_source_plan_root"])
    receipt, old_manifest = (
        old_root / "development/representations.jsonl",
        old_root / "development/variants.json",
    )
    reference, cache = cached_rows(receipt, old_manifest, checkpoint, environment)
    eligible = [
        r
        for r in body["variants"]
        if r["sha256"] in cache and cache[r["sha256"]]["pcm_sha256"] == r["pcm_sha256"]
    ]
    if len(eligible) != 1846:
        raise ValueError("Retained exact-input eligibility changed")
    for row in body["variants"]:
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError("Corrected waveform bytes changed")
    result = {
        "schema": "chipmunk.audio-identity-execution-lock/9",
        "source_plan_path": str(plan_path.resolve()),
        "source_plan_sha256": SOURCE_SHA,
        "variants_path": str(path.resolve()),
        "variants_sha256": VARIANTS_SHA,
        "reserved_confirmation_sha256": CONFIRMATION_SHA,
        "method": method.METHOD,
        "implementation_sha256": implementation(),
        "runtime": environment,
        "weights": checkpoint,
        "serving_calibration_sha256": serving.SHA256,
        "serving_bundle": serving.load(),
        "cache_path": str(receipt.resolve()),
        "cache_sha256": CACHE_SHA,
        "cache_old_manifest_path": str(old_manifest.resolve()),
        "cache_old_manifest_sha256": OLD_VARIANTS_SHA,
        "cache_reference_header": reference,
        "cache_eligible_inputs": len(eligible),
        "native_cache_ids": native_selection(body["variants"], cache),
        "planned": 4248,
        "development_sources": 161,
        "deployment_eligible": False,
    }
    write(output, result)
    print(
        json.dumps(
            {
                "lock_sha256": digest(output),
                "planned": 4248,
                "eligible_cache_inputs": len(eligible),
                "native_cache_checks": len(result["native_cache_ids"]),
                "fitting_started": False,
                "new_neural_inference_run": False,
            }
        ),
        flush=True,
    )


def check(lock_path):
    lock = json.loads(lock_path.read_text())
    if (
        lock["implementation_sha256"] != implementation()
        or lock["runtime"] != runtime()
        or lock["method"] != method.METHOD
    ):
        raise ValueError("Frozen corrected implementation/runtime/statistical method changed")
    if lock["serving_calibration_sha256"] != serving.SHA256 or lock["serving_bundle"] != serving.load():
        raise ValueError("Actual service calibration changed")
    path, source = Path(lock["variants_path"]), Path(lock["source_plan_path"])
    if digest(path) != VARIANTS_SHA or digest(source) != SOURCE_SHA:
        raise ValueError("Corrected exact sources/recipes changed")
    body = json.loads(path.read_text())
    validate(body, json.loads(source.read_text()))
    return lock, body


def extract_one(pcm, model, activity, captured):
    spans, windows = speech_windows(pcm, activity)
    if windows and (len(windows) != 1 or windows[0][0] != 0 or len(windows[0][1]) != 64000):
        raise ValueError("Unexpected original speech window")
    record = {
        "status": "ok" if windows else "not_applicable",
        "raw": None,
        "embedding": None,
        "speech_spans_s": [list(s) for s in spans],
        "windows": [{"start_sample": 0, "end_sample": 64000} for _ in windows],
    }
    if windows:
        raw, feature = capture(model, windows[0][1], captured)
        record.update(raw=raw, embedding=feature.tolist(), embedding_sha256=learner.digest_embedding(feature))
    return record


def reuse_record(row, cached):
    if cached["status"] not in ("ok", "not_applicable"):
        raise ValueError("Failed features cannot be reused or fitted")
    if row["sha256"] != cached["sha256"] or row["pcm_sha256"] != cached["pcm_sha256"]:
        raise ValueError("Cached representation is not this exact input")
    result = {k: cached[k] for k in ("status", "raw", "embedding", "speech_spans_s", "windows")}
    if cached["status"] == "ok":
        result["embedding_sha256"] = cached["embedding_sha256"]
    result.update({k: row[k] for k in ("id", "role", "sha256", "pcm_sha256")})
    result.update(reused_exact_input=True, cached_row_id=cached["id"], cached_prior_role=cached["role"])
    return result


def native_audit(lock_path, body, cache, model, activity, captured, journal):
    lock = json.loads(lock_path.read_text())
    if not lock["native_cache_ids"] or lock["native_cache_ids"] != native_selection(body["variants"], cache):
        raise ValueError("Complete predetermined native cache selection required")
    by_id = {r["id"]: r for r in body["variants"]}
    records = []
    for identifier in lock["native_cache_ids"]:
        row = by_id[identifier]
        expected = cache[row["sha256"]]
        actual = extract_one(prior.input_pcm(body, row), model, activity, captured)
        same = (
            actual["status"] == expected["status"]
            and actual["raw"] == expected["raw"]
            and actual["windows"] == expected["windows"]
            and actual["speech_spans_s"] == expected["speech_spans_s"]
        )
        if actual["status"] == "ok":
            same = same and actual["embedding_sha256"] == expected["embedding_sha256"]
        records.append(
            {
                "id": identifier,
                "status": actual["status"],
                "passed": bool(same),
                "sha256": row["sha256"],
                "pcm_sha256": row["pcm_sha256"],
            }
        )
    audit = {
        "lock_sha256": digest(lock_path),
        "cache_sha256": CACHE_SHA,
        "passed": all(r["passed"] for r in records),
        "cases": records,
    }
    if journal.exists():
        for line in journal.read_text().splitlines():
            if json.loads(line)["lock_sha256"] != audit["lock_sha256"]:
                raise ValueError("Native cache audit journal belongs to another execution")
    with journal.open("a") as stream:
        stream.write(json.dumps(audit) + "\n")
    if not audit["passed"]:
        raise ValueError(
            "Native cached-feature replay differs; preserve the failure, never silently fall back"
        )
    return audit


def extract(lock_path, output):
    lock, body = check(lock_path)
    if weights() != lock["weights"]:
        raise ValueError("Verified commercial model bytes changed")
    reference, cache = cached_rows(
        Path(lock["cache_path"]), Path(lock["cache_old_manifest_path"]), lock["weights"], lock["runtime"]
    )
    model, activity = models()
    direction, bias = model.head.weight.detach().cpu().numpy()[0], float(model.head.bias.detach().cpu()[0])
    if (
        not np.array_equal(direction, np.asarray(reference["head_weight"], np.float32))
        or bias != reference["head_bias"]
    ):
        raise ValueError("Live frozen representation head differs from cached checkpoint")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    header = {
        "schema": "chipmunk.audio-identity-representations/9",
        "lock_sha256": digest(lock_path),
        "planned": 4248,
        "weights": lock["weights"],
        "runtime": lock["runtime"],
        "cache_sha256": CACHE_SHA,
        "head_weight": direction.tolist(),
        "head_bias": bias,
    }
    try:
        audit = native_audit(
            lock_path, body, cache, model, activity, captured, output.parent / "native-cache-audits.jsonl"
        )
        print(
            json.dumps({"native_cache_replays": len(audit["cases"]), "passed": audit["passed"]}), flush=True
        )
        completed = prior.open_journal(output, header, body["variants"])
        with output.open("a") as stream:
            for index, row in enumerate(body["variants"][len(completed) :], len(completed) + 1):
                started = time.monotonic()
                result = {k: row[k] for k in ("id", "role", "sha256", "pcm_sha256")}
                try:
                    pcm = prior.input_pcm(body, row)
                    old = cache.get(row["sha256"])
                    if old is not None and old["pcm_sha256"] == row["pcm_sha256"]:
                        result.update(reuse_record(row, old))
                    else:
                        result.update(extract_one(pcm, model, activity, captured), reused_exact_input=False)
                    validate_feature(result, header)
                except Exception as error:
                    result.update(status="failed", error_type=type(error).__name__, raw=None, embedding=None)
                result["runtime_s"] = time.monotonic() - started
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if index % 30 == 0 or result["status"] == "failed" or index == 4248:
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "planned": 4248,
                                "role": row["role"],
                                "status": result["status"],
                                "reused": result.get("reused_exact_input", False),
                            }
                        ),
                        flush=True,
                    )
    finally:
        hook.remove()


def feature_rows(lock_path, receipt):
    lock, body = check(lock_path)
    header, records = load_rows(receipt)
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["weights"] != lock["weights"]
        or header["runtime"] != lock["runtime"]
        or header["cache_sha256"] != CACHE_SHA
        or header["head_weight"] != lock["cache_reference_header"]["head_weight"]
        or header["head_bias"] != lock["cache_reference_header"]["head_bias"]
        or [r["id"] for r in records] != [r["id"] for r in body["variants"]]
    ):
        raise ValueError("Every corrected input needs a bound terminal feature record")
    rows = []
    for row, record in zip(body["variants"], records, strict=True):
        if any(row[k] != record[k] for k in ("id", "role", "sha256", "pcm_sha256")):
            raise ValueError("Corrected feature source/role/PCM binding changed")
        validate_feature(record, header)
        rows.append({**row, **record})
    return body, header, rows


def train(lock_path, receipt, output):
    _, header, rows = feature_rows(lock_path, receipt)
    output.mkdir(parents=True, exist_ok=False)
    fitted = method.fit([r for r in rows if r["role"] == "fit"], header["head_weight"])
    write(
        output / "fit.json",
        {
            "lock_sha256": digest(lock_path),
            "receipt_sha256": digest(receipt),
            "fitted": fitted,
            "deployment_eligible": False,
        },
    )
    guards = method.calibrate([r for r in rows if r["role"] == "cal"], fitted)
    write(
        output / "head.json",
        {
            "lock_sha256": digest(lock_path),
            "receipt_sha256": digest(receipt),
            "fit_sha256": digest(output / "fit.json"),
            "fitted": fitted,
            "guards": guards,
            "deployment_eligible": False,
        },
    )
    print(
        json.dumps(
            {
                "head_sha256": digest(output / "head.json"),
                "fit_groups_by_class": fitted["fit_groups_by_class"],
                "cal_groups_by_class": [len(guards[str(c)]) for c in (0, 1)],
            }
        ),
        flush=True,
    )


def compare(lock_path, receipt, head_path, output):
    body, reference, all_rows = feature_rows(lock_path, receipt)
    lock, _ = check(lock_path)
    head = json.loads(head_path.read_text())
    if (
        head["lock_sha256"] != digest(lock_path)
        or head["receipt_sha256"] != digest(receipt)
        or weights() != lock["weights"]
    ):
        raise ValueError("Fitted head or model differs from corrected execution")
    rows = [r for r in all_rows if r["role"] == "development"]
    header = {
        "schema": "chipmunk.audio-identity-development-responses/9",
        "lock_sha256": digest(lock_path),
        "receipt_sha256": digest(receipt),
        "head_sha256": digest(head_path),
        "planned": 1449,
    }
    completed = prior.open_journal(output, header, rows)
    model, _ = models()
    if (
        not np.array_equal(
            model.head.weight.detach().cpu().numpy()[0], np.asarray(reference["head_weight"], np.float32)
        )
        or float(model.head.bias.detach().cpu()[0]) != reference["head_bias"]
    ):
        raise ValueError("Live original head differs")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        with output.open("a") as stream:
            for index, row in enumerate(rows[len(completed) :], len(completed) + 1):
                result = {k: v for k, v in row.items() if k not in ("path", "production_input", "embedding")}
                try:
                    result.update(prior.compare_one(prior.input_pcm(body, row), row, model, head, captured))
                except Exception as error:
                    result.update(
                        status="failed",
                        error_type=type(error).__name__,
                        original_p=None,
                        original_decision="inconclusive",
                        candidate_p=None,
                        candidate_decision="inconclusive",
                        probe={"status": "failed", "raw": None, "p": None, "embedding": None},
                    )
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if index % 30 == 0 or result["status"] == "failed" or index == 1449:
                    print(
                        json.dumps({"completed": index, "planned": 1449, "status": result["status"]}),
                        flush=True,
                    )
    finally:
        hook.remove()


def review(lock_path, receipt, head_path, responses, output):
    _, _, rows = feature_rows(lock_path, receipt)
    expected = [r for r in rows if r["role"] == "development"]
    header, actual = load_rows(responses)
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["head_sha256"] != digest(head_path)
        or header["receipt_sha256"] != digest(receipt)
        or [r["id"] for r in actual] != [r["id"] for r in expected]
    ):
        raise ValueError("Every corrected development outcome must remain bound")
    for row, record in zip(expected, actual, strict=True):
        if any(
            row[k] != record[k]
            for k in ("sha256", "pcm_sha256", "role", "base_id", "group", "condition", "generator", "label")
        ):
            raise ValueError("Corrected outcome source/class/group differs")
    result = summarize(actual, 161)
    result.update(
        schema="chipmunk.audio-identity-development-comparison/9",
        lock_sha256=digest(lock_path),
        receipt_sha256=digest(receipt),
        head_sha256=digest(head_path),
        responses_sha256=digest(responses),
    )
    write(output, result)
    print(
        json.dumps(
            {
                "gates": result["gates"],
                "advance_to_fresh_confirmation": result["advance_to_fresh_confirmation"],
                "deployment_eligible": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "extract", "train", "compare", "review"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--responses", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.input, args.output)
    elif args.stage == "extract":
        extract(args.input, args.output)
    elif args.stage == "train" and args.receipt:
        train(args.input, args.receipt, args.output)
    elif args.stage == "compare" and args.receipt and args.head:
        compare(args.input, args.receipt, args.head, args.output)
    elif args.stage == "review" and args.receipt and args.head and args.responses:
        review(args.input, args.receipt, args.head, args.responses, args.output)
    else:
        parser.error("train needs --receipt; compare needs --receipt/--head; review also needs --responses")
