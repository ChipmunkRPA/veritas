"""Export/reproduce complete development arithmetic without media or private IDs.

Original feature JSONL is reconstructed byte-for-byte; source recipes supply
hashed group/source identities. Outcome exports sanitize provider identifiers,
so original outcome digests are lineage commitments, not byte reproduction.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts import run_audio_identity_v9 as execution
from scripts.audio_source_identity_v9 import digest
from scripts.audit_audio_identity_forwards_v9 import select
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_identity_numerics_v9 import reconstruct
from scripts.run_audio_guard_v7 import load_rows

REFERENCES = (
    "audio-v9-identity-plan.json.gz",
    "audio-v9-execution-freeze.json",
    "audio-v9-result-review-freeze.json",
    "audio-v9-native-development-plan.json",
)
REFERENCE_SHA = dict(
    zip(
        REFERENCES,
        (
            "526a660d812c3afec959fb45946101d643181816e4b7c4248e93559280adbf83",
            "b73e5933deb418129f570500a90672e062d159ed84d39ddf8bc8744d2f13ff8c",
            "d8c23526d6b37ecefafa22d7f5079d7d3da4f5bda7b30eeca76a7c350813a88b",
            "48795283220152d5de07356ff54c018b8da797749957b6480d27582de97eed1e",
        ),
        strict=True,
    )
)
FEATURE_FIELDS = {
    "id",
    "role",
    "sha256",
    "pcm_sha256",
    "status",
    "raw",
    "embedding",
    "embedding_sha256",
    "speech_spans_s",
    "windows",
    "reused_exact_input",
    "cached_row_id",
    "cached_prior_role",
    "runtime_s",
    "error_type",
}
OUTCOME_FIELDS = (
    "status",
    "error_type",
    "raw",
    "original_p",
    "original_decision",
    "candidate_p",
    "candidate_decision",
    "probe_needed_by_service",
    "probe_needed_by_candidate",
)
BASELINE_FIELDS = (
    "version",
    "calibration_sha256",
    "probability_ai",
    "reference_estimate_ai",
    "decision",
    "scope_supported",
    "seconds",
    "probe_status",
    "stability_log_odds_change",
    "original_stability_veto",
    "reason",
    "limitation",
)
PROBE_FIELDS = ("status", "raw", "embedding", "embedding_sha256", "filtered_pcm_sha256", "p", "error_type")
CANDIDATE_FIELDS = (
    "proposed",
    "decision",
    "calibrated_stability_log_odds_change",
    "raw_stability_log_odds_change",
    "original_stability_veto",
)
SUMMARY_FIELDS = (
    "schema",
    "groups",
    "paired_differences",
    "strata",
    "gates",
    "advance_to_fresh_confirmation",
    "complete_inputs",
    "source_files",
    "connected_foreground_groups",
    "deployment_eligible",
    "scope",
)


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def pick(body, names):
    return {k: body[k] for k in names if k in body}


def privacy_check(value):
    forbidden = {
        "path",
        "source_path",
        "source_root",
        "transcript",
        "transcripts",
        "provider_id",
        "speaker",
        "speaker_id",
        "prompt_speaker_id",
        "voice_key",
        "source_key",
    }
    if isinstance(value, dict):
        if set(value) & forbidden:
            raise ValueError("Private source fields cannot enter the public capsule")
        for item in value.values():
            privacy_check(item)
    elif isinstance(value, list):
        for item in value:
            privacy_check(item)
    elif isinstance(value, str) and any(prefix in value for prefix in ("/Users/", "/private/", "/home/")):
        raise ValueError("Local paths cannot enter the public capsule")


def verify_implementation(freeze, supplement):
    project = Path(__file__).resolve().parents[2]
    for name, expected in {
        **freeze["lock"]["implementation_sha256"],
        **supplement["implementation_sha256"],
    }.items():
        path = project / name
        if not path.resolve().is_relative_to(project) or digest(path) != expected:
            raise ValueError("Published method or verification source bytes changed")


def jsonl_digest(header, rows):
    h = hashlib.sha256()
    for row in (header, *rows):
        h.update((json.dumps(row, allow_nan=False) + "\n").encode())
    return h.hexdigest()


def metadata(variant):
    result = dict(variant)
    result["base_id"] = result.pop("base_id_sha256")
    result["background_id"] = result["background_sha256"]
    return result


def comparable(summary):
    result = pick(copy.deepcopy(summary), SUMMARY_FIELDS)
    result.pop("schema", None)
    for panel in result["paired_differences"].values():
        panel.pop("interpretation", None)
    return result


def public_summary(summary, private_rows):
    result = pick(copy.deepcopy(summary), SUMMARY_FIELDS)
    mapping = {str(r["background_id"]): str(r["background_sha256"]) for r in private_rows}
    result["strata"]["background_id"] = {mapping[k]: v for k, v in result["strata"]["background_id"].items()}
    for panel in result["paired_differences"].values():
        panel["interpretation"] = (
            "Consumed-source development conditional on the fixed fitting/calibration and selected backgrounds; excludes fitting/calibration and background-population uncertainty. Not fresh testing, outside validation or per-upload accuracy."
        )
    return result


def review(body, sources, freeze, native_plan, *, refit=False):
    privacy_check(body)
    if (
        body["schema"] != "chipmunk.audio-identity-public-development/9"
        or body["deployment_eligible"] is not False
        or body["references"] != REFERENCE_SHA
    ):
        raise ValueError("Original public development schema/reference commitments required")
    header, features = body["representation_header"], body["features"]
    if set(header) != {
        "schema",
        "lock_sha256",
        "planned",
        "weights",
        "runtime",
        "cache_sha256",
        "head_weight",
        "head_bias",
    }:
        raise ValueError("Exact safe feature header required")
    if any(set(row) - FEATURE_FIELDS for row in features):
        raise ValueError("Unknown feature fields cannot enter the public capsule")
    if jsonl_digest(header, features) != body["original_receipt_sha256"]:
        raise ValueError("Exact original feature receipt cannot be reconstructed")
    if (
        header["lock_sha256"] != freeze["private_lock_sha256"]
        or header["weights"] != freeze["lock"]["weights"]
        or header["runtime"] != freeze["lock"]["runtime"]
        or header["cache_sha256"] != freeze["lock"]["cache_sha256"]
    ):
        raise ValueError("Published execution/model/runtime/cache commitment differs")
    variants = sources["variants"]["development"]
    if [r["id"] for r in features] != [r["id"] for r in variants]:
        raise ValueError("Complete public source/feature order required")
    rows = []
    for feature, variant in zip(features, variants, strict=True):
        if any(feature[k] != variant[k] for k in ("role", "sha256", "pcm_sha256")):
            raise ValueError("Feature differs from corrected public input")
        rows.append({**metadata(variant), **feature})
    for key, original in (("head", "original_head_sha256"), ("fit_record", "original_fit_sha256")):
        encoded = (json.dumps(body[key], indent=2, allow_nan=False) + "\n").encode()
        if hashlib.sha256(encoded).hexdigest() != body[original]:
            raise ValueError("Exact original fitted artifact differs")
    if (
        not same_tree(body["fit_record"]["fitted"], body["head"]["fitted"])
        or body["head"]["receipt_sha256"] != body["original_receipt_sha256"]
        or body["head"]["lock_sha256"] != freeze["private_lock_sha256"]
        or body["head"]["fit_sha256"] != body["original_fit_sha256"]
    ):
        raise ValueError("Fitting-before-calibration artifact lineage differs")
    native = body["native_audit"]
    if (
        native_plan["ids"] != select(variants)
        or len(native_plan["ids"]) != 99
        or [r["id"] for r in native["cases"]] != native_plan["ids"]
        or not native["passed"]
        or not all(r["passed"] for r in native["cases"])
        or native["receipt_sha256"] != body["original_receipt_sha256"]
        or native["responses_sha256"] != body["original_responses_sha256"]
        or native["lock_sha256"] != freeze["private_lock_sha256"]
        or native["plan_sha256"] != REFERENCE_SHA[REFERENCES[3]]
    ):
        raise ValueError("Complete predetermined native result audit required")
    by_id = {r["id"]: r for r in rows}
    if any(
        any(case[k] != by_id[case["id"]][k] for k in ("sha256", "pcm_sha256")) for case in native["cases"]
    ):
        raise ValueError("Native receipt source/PCM differs from the published input")
    numerical = body["numerical_audit"]
    if (
        not numerical["passed"]
        or not numerical["refit_performed"]
        or numerical["receipt_sha256"] != body["original_receipt_sha256"]
        or numerical["responses_sha256"] != body["original_responses_sha256"]
        or numerical["head_sha256"] != body["original_head_sha256"]
        or numerical["summary_sha256"] != body["original_summary_sha256"]
    ):
        raise ValueError("Complete original numerical/refit review lineage required")
    reconstructed, result = reconstruct(
        header, body["head"], rows, body["outcomes"], freeze["lock"]["serving_bundle"], refit=refit
    )
    if not same_tree(comparable(reconstructed), comparable(body["summary"])):
        raise ValueError("Every summary value and frozen gate must agree")
    result.update(
        schema="chipmunk.audio-identity-public-result-review/9",
        native_cases=99,
        exact_original_feature_receipt_reproduced=True,
        original_fit_head_reproduced=True,
        gates=reconstructed["gates"],
        advance_to_fresh_confirmation=reconstructed["advance_to_fresh_confirmation"],
        serving_changed=False,
        outside_reviewer_participation=False,
        native_reexecution_performed_by_this_review=False,
    )
    return result


def export(root, destination, references):
    lock_path, receipt = root / "lock.json", root / "representations.jsonl"
    _, _, private_rows = execution.feature_rows(lock_path, receipt)
    numerical = load(root / "numerical-review.json")
    native = load(root / "native-development-review.json")
    if not numerical["passed"] or not numerical["refit_performed"] or not native["passed"]:
        raise ValueError(
            "Complete successful refit/numerical/native reviews required, even for a rejected candidate"
        )
    sources, freeze = load(references / REFERENCES[0]), load(references / REFERENCES[1])
    header, features = load_rows(receipt)
    response_header, outcomes = load_rows(root / "responses.jsonl")
    expected = [r for r in sources["variants"]["development"] if r["role"] == "development"]
    if [r["id"] for r in outcomes] != [r["id"] for r in expected]:
        raise ValueError("Every development outcome required")
    public_outcomes = []
    for row, variant in zip(outcomes, expected, strict=True):
        if any(
            row[k] != variant[k]
            for k in (
                "id",
                "role",
                "sha256",
                "pcm_sha256",
                "group",
                "condition",
                "label",
                "generator",
                "dataset",
            )
        ):
            raise ValueError("Outcome source/truth differs from public recipe")
        public = {**metadata(variant), **pick(row, OUTCOME_FIELDS)}
        public["probe"] = pick(row["probe"], PROBE_FIELDS)
        for key, fields in (("baseline_detail", BASELINE_FIELDS), ("candidate_detail", CANDIDATE_FIELDS)):
            if key in row:
                public[key] = pick(row[key], fields)
        public_outcomes.append(public)
    body = {
        "schema": "chipmunk.audio-identity-public-development/9",
        "references": {name: digest(references / name) for name in REFERENCES},
        "original_receipt_sha256": digest(receipt),
        "original_responses_sha256": digest(root / "responses.jsonl"),
        "original_head_sha256": digest(root / "trained/head.json"),
        "original_fit_sha256": digest(root / "trained/fit.json"),
        "original_summary_sha256": digest(root / "summary.json"),
        "representation_header": header,
        "features": features,
        "head": load(root / "trained/head.json"),
        "fit_record": load(root / "trained/fit.json"),
        "response_header": response_header,
        "outcomes": public_outcomes,
        "summary": public_summary(load(root / "summary.json"), private_rows),
        "native_audit": native,
        "numerical_audit": numerical,
        "deployment_eligible": False,
        "scope": "Complete consumed-source arithmetic with optional CPU refit; no source audio, transcripts, raw provider identifiers, local paths or full pretrained checkpoints. Sanitized outcome/summary bytes differ from original lineage artifacts. No fresh result or outside reviewer participation.",
        "data_terms": sources["data_terms"],
    }
    verify_implementation(freeze, load(references / REFERENCES[2]))
    result = review(body, sources, freeze, load(references / REFERENCES[3]))
    encoded = gzip.compress(
        (json.dumps(body, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0
    )
    with destination.open("xb") as stream:
        stream.write(encoded)
    print(
        json.dumps(
            {
                "capsule_sha256": digest(destination),
                "bytes": len(encoded),
                "features": len(features),
                "outcomes": len(outcomes),
                "review_passed": result["passed"],
            }
        ),
        flush=True,
    )


def tamper_review(body, sources, freeze, native_plan):
    feature_index = next(i for i, r in enumerate(body["features"]) if r["status"] == "ok")
    assessed_index = next(i for i, r in enumerate(body["outcomes"]) if r["status"] == "ok")
    unassessed_index = next(i for i, r in enumerate(body["outcomes"]) if r["status"] == "not_applicable")
    names = (
        "changed representation",
        "source role moved to confirmation",
        "changed fitted coefficient",
        "changed candidate probability",
        "unassessed source called original",
        "changed complete denominator",
        "dropped native replay case",
        "hidden local path",
    )
    cases = []
    for name in names:
        altered = copy.deepcopy(body)
        if name == names[0]:
            altered["features"][feature_index]["embedding"][0] += 0.01
        elif name == names[1]:
            altered["features"][0]["role"] = "confirmation"
        elif name == names[2]:
            altered["head"]["fitted"]["coefficients"][0] += 0.01
        elif name == names[3]:
            altered["outcomes"][assessed_index]["candidate_p"] = None
        elif name == names[4]:
            altered["outcomes"][unassessed_index]["candidate_decision"] = "original"
            altered["outcomes"][unassessed_index]["candidate_detail"]["decision"] = "original"
        elif name == names[5]:
            altered["summary"]["complete_inputs"] -= 1
        elif name == names[6]:
            altered["native_audit"]["cases"].pop()
        else:
            altered["notes"] = "/Users/placeholder/private-input.wav"
        try:
            review(altered, sources, freeze, native_plan)
        except ValueError as error:
            cases.append({"alteration": name, "rejected": True, "reason": str(error)})
        else:
            raise ValueError("Altered actual capsule incorrectly passed: " + name)
    return {
        "schema": "chipmunk.audio-identity-public-result-tamper-review/9",
        "passed": len(cases) == 8 and all(r["rejected"] for r in cases),
        "cases": cases,
        "scope": "Local deliberate changes to the complete actual public capsule; no neural execution, outside reviewer or performance claim.",
    }


def guard_only_ceiling(outcomes):
    families = defaultdict(list)
    for row in outcomes:
        if row["label"] == 1:
            families[row["generator"]].append(row)
    if not families:
        raise ValueError("Generated-source outcomes required for a retention bound")
    records = []
    for generator, rows in sorted(families.items()):
        service = sum(r["original_decision"] == "ai" for r in rows)
        ceiling = sum(r["candidate_p"] is not None and r["candidate_p"] >= 0.8 for r in rows)
        records.append(
            {
                "generator": generator,
                "service_correct_ai": service,
                "candidate_correct_ai": sum(r["candidate_decision"] == "ai" for r in rows),
                "fixed_probability_ai_ceiling_without_any_guard_or_stability_veto": ceiling,
                "frozen_minimum_to_retain": service - 2,
                "guard_only_retention_possible": ceiling >= service - 2,
            }
        )
    return {
        "schema": "chipmunk.audio-identity-guard-only-bound/9",
        "families": records,
        "guard_only_cannot_pass": any(not r["guard_only_retention_possible"] for r in records),
        "scope": "Post-result arithmetic on consumed development. Holding fitted probabilities/internal operating points fixed, even removing every class/stability veto cannot meet all per-generator retention requirements. An optimistic ceiling, not permission to remove guards. No new method, point selection, fresh scoring or causal claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "review", "tamper", "diagnose"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    if args.stage == "export":
        export(args.input, args.output, args.references or Path("docs/evaluation-panels"))
    else:
        body = load(args.input)
        folder = args.references or args.input.parent
        if body["references"] != REFERENCE_SHA or any(
            digest(folder / name) != expected for name, expected in body["references"].items()
        ):
            raise ValueError("Fixed public source/execution/native references differ")
        verify_implementation(load(folder / REFERENCES[1]), load(folder / REFERENCES[2]))
        if args.stage == "tamper":
            result = tamper_review(
                body, load(folder / REFERENCES[0]), load(folder / REFERENCES[1]), load(folder / REFERENCES[3])
            )
        elif args.stage == "diagnose":
            review(
                body, load(folder / REFERENCES[0]), load(folder / REFERENCES[1]), load(folder / REFERENCES[3])
            )
            result = guard_only_ceiling(body["outcomes"])
            result.update(
                responses_sha256=body["original_responses_sha256"], head_sha256=body["original_head_sha256"]
            )
        else:
            result = review(
                body,
                load(folder / REFERENCES[0]),
                load(folder / REFERENCES[1]),
                load(folder / REFERENCES[3]),
                refit=args.refit,
            )
        result["capsule_sha256"] = digest(args.input)
        write(args.output, result)
        print(json.dumps(result), flush=True)
