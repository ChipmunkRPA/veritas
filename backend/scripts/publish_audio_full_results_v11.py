"""Publish/reproduce the full-vector result using existing public feature packets.

Retains exact original small fit/head objects and all sanitized outcomes.
Cached probe vectors are referenced, not duplicated. No private manifest, source
audio, customer files or pretrained checkpoints are needed for numerical review.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path

from scripts import export_audio_full_representation_v11 as prospective
from scripts import publish_audio_identity_results_v9 as original
from scripts import run_audio_full_representation_v11 as execution
from scripts.audit_audio_identity_forwards_v9 import select
from scripts.prepare_audio_guard_v7 import write
from scripts.publish_audio_conditional_results_v10 import exact_artifact
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_full_representation_v11 import reconstruct
from scripts.run_audio_guard_v7 import load_rows

REFERENCES = {
    "audio-v9-identity-plan.json.gz": prospective.SOURCE_SHA,
    "audio-v9-execution-freeze.json": prospective.REFERENCE_SHA,
    "audio-v9-development-review.json.gz": "0546940a656ce0f718318975491ee041f2a429be15ac57d316e8e5e3ea6345c3",
    "audio-v9-native-development-plan.json": "48795283220152d5de07356ff54c018b8da797749957b6480d27582de97eed1e",
    "audio-v10-development-review.json.gz": "c45553d229814aad9acbde3603408a42652ac2a3f9bf0b8d79b80afaf7d377b0",
    "audio-v11-full-representation-freeze.json": "9f23fe7ab11c4672ca13949956fbf57ed88a74308507015fb20db90f5a356c82",
}
LINEAGE = {
    "lock_sha256": "bce7b7cb7cb142a28cc68c60b54a559b3eb6f9ea52efc67a0259a49c8ee5c384",
    "fit_sha256": "b9f12ee6de6491abbe94266e1f15eac7758bbe149230f710bdf04e65de0578ac",
    "head_sha256": "bf2a363ad6e491ad4c4fac1ec18658c58bc01eb20788a2a198c198426d481607",
    "responses_sha256": "683f56f383244257ddb47edb1e3d5c87a3071db19df7d13c4cc707f48129bf0f",
    "summary_sha256": "cab6bf33a644774d565a5bc3f949143b3d3ab35c84c1e9e5425a84bef79bcdcf",
}
RESPONSE_FIELDS = {
    "status",
    "error_type",
    "raw",
    "original_p",
    "original_decision",
    "candidate_p",
    "candidate_decision",
    "candidate_detail",
    "probe",
}
PROBE_FIELDS = {
    "status",
    "raw",
    "embedding",
    "p",
    "embedding_sha256",
    "filtered_pcm_sha256",
    "reused_exact_probe",
    "error_type",
}
NATIVE_FIELDS = {
    "schema",
    "lock_sha256",
    "head_sha256",
    "responses_sha256",
    "plan_sha256",
    "audit_code_sha256",
    "passed",
    "predetermined_cells",
    "new_probe_cases",
    "unique_cases",
    "cases",
    "scope",
}


def inputs(folder, project=None):
    if any(prospective.digest(folder / name) != value for name, value in REFERENCES.items()):
        raise ValueError("Exact original public source/vector/freeze references required")
    source = original.load(folder / "audio-v9-identity-plan.json.gz")
    reference = original.load(folder / "audio-v9-execution-freeze.json")
    prior = original.load(folder / "audio-v9-development-review.json.gz")
    conditional = original.load(folder / "audio-v10-development-review.json.gz")
    freeze = original.load(folder / "audio-v11-full-representation-freeze.json")
    prospective.review(freeze, source, reference, project)
    rows = []
    for variant, feature in zip(source["variants"]["development"], prior["features"], strict=True):
        if any(variant[k] != feature[k] for k in ("id", "role", "sha256", "pcm_sha256")):
            raise ValueError("Original complete source/feature binding differs")
        execution.original.validate_feature(feature, prior["representation_header"])
        rows.append({**original.metadata(variant), **feature})
    cached = {}
    for row in prior["outcomes"] + conditional["outcomes"]:
        probe = row["probe"]
        if probe["status"] == "computed":
            execution.original.validate_feature(
                {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]},
                prior["representation_header"],
            )
            if row["id"] in cached and any(
                probe[k] != cached[row["id"]][k]
                for k in ("raw", "embedding", "embedding_sha256", "filtered_pcm_sha256")
            ):
                raise ValueError("Original cached probe references disagree")
            cached[row["id"]] = probe
    if len(cached) != 424:
        raise ValueError("Complete original424-probe cache required")
    return source, prior, rows, cached


def expand_probe(probe, cached):
    if set(probe) - PROBE_FIELDS:
        raise ValueError("Unknown probe fields cannot enter the numerical review")
    result = dict(probe)
    if probe["status"] == "computed" and probe["reused_exact_probe"]:
        if (
            "embedding" in probe
            or cached is None
            or any(probe[k] != cached[k] for k in ("raw", "embedding_sha256", "filtered_pcm_sha256"))
        ):
            raise ValueError("Referenced probe must match the exact original cache")
        result["embedding"] = cached["embedding"]
    elif probe["status"] != "computed" and any(probe[k] is not None for k in ("raw", "embedding", "p")):
        raise ValueError("An uncomputed probe cannot carry measurements")
    return result


def review_native(native, rows, outcomes, folder, project):
    plan = original.load(folder / "audio-v9-native-development-plan.json")
    new = {r["id"] for r in outcomes if r["probe"].get("reused_exact_probe") is False}
    ids = sorted(set(plan["ids"]) | new)
    if (
        set(native) != NATIVE_FIELDS
        or plan["ids"] != select(rows)
        or [r["id"] for r in native["cases"]] != ids
        or native["passed"] is not True
        or native["predetermined_cells"] != 99
        or native["new_probe_cases"] != len(new)
        or len(new) != 26
        or native["unique_cases"] != len(ids)
        or native["plan_sha256"] != REFERENCES["audio-v9-native-development-plan.json"]
        or any(native[k] != LINEAGE[k] for k in ("lock_sha256", "head_sha256", "responses_sha256"))
    ):
        raise ValueError("All predetermined native cells and every new probe required")
    by_id = {r["id"]: r for r in rows}
    response = {r["id"]: r for r in outcomes}
    for case in native["cases"]:
        source, probe = by_id[case["id"]], response[case["id"]]["probe"]
        if (
            any(case[k] != source[k] for k in ("sha256", "pcm_sha256"))
            or case["predetermined_cell"] != (case["id"] in plan["ids"])
            or case["new_probe_case"] != (case["id"] in new)
            or case["probe_status"] != probe["status"]
            or any(case[k] is not True for k in ("passed", "original_passed", "mapped_original_passed"))
            or (probe["status"] == "computed" and case["probe_passed"] is not True)
        ):
            raise ValueError("Native source/PCM/raw/vector/mapped/probe receipt differs")
    if (
        project is not None
        and prospective.digest(project / "backend/scripts/audit_audio_full_representation_v11.py")
        != native["audit_code_sha256"]
    ):
        raise ValueError("Native review implementation bytes changed")


def review(body, folder, *, refit=False, project=None):
    original.privacy_check(body)
    if (
        body["schema"] != "chipmunk.audio-full-representation-public-development/11"
        or body["references"] != REFERENCES
        or body["original_lineage"] != LINEAGE
        or body["deployment_eligible"] is not False
    ):
        raise ValueError("Original public schema/reference/lineage commitments required")
    source, prior, rows, cached = inputs(folder, project)
    head, fit_record = body["head"], body["fit_record"]
    exact_artifact(head, LINEAGE["head_sha256"])
    exact_artifact(fit_record, LINEAGE["fit_sha256"])
    if (
        head["lock_sha256"] != LINEAGE["lock_sha256"]
        or fit_record["lock_sha256"] != LINEAGE["lock_sha256"]
        or head["fit_sha256"] != LINEAGE["fit_sha256"]
        or head["fitted"] != fit_record["fitted"]
        or head["deployment_eligible"] is not False
        or fit_record["deployment_eligible"] is not False
    ):
        raise ValueError("Original fitting-before-calibration artifact lineage required")
    if body["response_header"] != {
        "schema": "chipmunk.audio-full-representation-responses/11",
        "lock_sha256": LINEAGE["lock_sha256"],
        "head_sha256": LINEAGE["head_sha256"],
        "original_responses_sha256": prospective.ORIGINAL["original_responses_sha256"],
        "planned": 1449,
    }:
        raise ValueError("Original complete response header required")
    variants = [r for r in source["variants"]["development"] if r["role"] == "development"]
    if [r["id"] for r in body["outcomes"]] != [r["id"] for r in variants]:
        raise ValueError("Every sanitized development outcome in original order required")
    outcomes = []
    for row, variant in zip(body["outcomes"], variants, strict=True):
        metadata = original.metadata(variant)
        if set(row) - set(metadata) - RESPONSE_FIELDS or any(row[k] != v for k, v in metadata.items()):
            raise ValueError("Sanitized source/truth/group/role binding differs")
        expanded = {**row, "probe": expand_probe(row["probe"], cached.get(row["id"]))}
        outcomes.append(expanded)
    review_native(body["native_audit"], rows, outcomes, folder, project)
    numerical = body["numerical_audit"]
    if (
        numerical["passed"] is not True
        or numerical["refit_performed"] is not True
        or any(
            numerical[k] != LINEAGE[k]
            for k in ("lock_sha256", "head_sha256", "responses_sha256", "summary_sha256")
        )
        or numerical["inputs"] != 4248
        or numerical["outcomes"] != 1449
        or numerical["computed_probes"] != 427
        or numerical["new_probes"] != 26
        or any(v != 0.0 for v in numerical["refit_max_absolute_difference"].values())
    ):
        raise ValueError("Complete original numerical/refit receipt required")
    if (
        project is not None
        and prospective.digest(project / "backend/scripts/review_audio_full_representation_v11.py")
        != numerical["review_code_sha256"]
    ):
        raise ValueError("Numerical review implementation bytes changed")
    rebuilt, result = reconstruct(
        head,
        rows,
        {r["id"]: r for r in prior["outcomes"]},
        cached,
        prior["representation_header"],
        outcomes,
        refit=refit,
    )
    if not same_tree(original.comparable(rebuilt), original.comparable(body["summary"])):
        raise ValueError("Every summary, stratum, uncertainty interval and gate must agree")
    result.update(
        schema="chipmunk.audio-full-representation-public-result-review/11",
        original_fit_head_reproduced=True,
        gates=rebuilt["gates"],
        advance_to_fresh_confirmation=rebuilt["advance_to_fresh_confirmation"],
        native_cases=len(body["native_audit"]["cases"]),
        serving_changed=False,
        outside_reviewer_participation=False,
        native_reexecution_performed_by_this_review=False,
    )
    result["scope"] = (
        "Public-input numerical reconstruction with optional fitting-only CPU refit; original native receipts checked without rerunning audio. Exact original fit/head bytes, sanitized outcome/summary lineage. No private manifest, source audio, pretrained download, fresh confirmation or outside human validation."
    )
    return result


def export(root, destination, folder):
    source, _, rows, _ = inputs(folder, Path(__file__).resolve().parents[2])
    files = {
        "lock_sha256": "lock.json",
        "fit_sha256": "trained/fit.json",
        "head_sha256": "trained/head.json",
        "responses_sha256": "responses.jsonl",
        "summary_sha256": "summary.json",
    }
    if any(prospective.digest(root / path) != LINEAGE[name] for name, path in files.items()):
        raise ValueError("Original completed experiment artifacts changed")
    header, responses = load_rows(root / "responses.jsonl")
    expected = [r for r in rows if r["role"] == "development"]
    variants = [r for r in source["variants"]["development"] if r["role"] == "development"]
    outcomes = []
    for row, source_row, variant in zip(responses, expected, variants, strict=True):
        if any(
            row[k] != source_row[k]
            for k in (
                "id",
                "raw",
                "group",
                "role",
                "label",
                "generator",
                "dataset",
                "condition",
                "sha256",
                "pcm_sha256",
            )
        ):
            raise ValueError("Original outcome/source binding differs")
        public = {**original.metadata(variant), **original.pick(row, RESPONSE_FIELDS)}
        public["candidate_detail"] = original.pick(row["candidate_detail"], original.CANDIDATE_FIELDS)
        public["probe"] = original.pick(row["probe"], PROBE_FIELDS)
        if public["probe"].get("reused_exact_probe") is True:
            public["probe"].pop("embedding")
        outcomes.append(public)
    mapping_rows = [
        {**r, "background_sha256": s["background_sha256"]} for r, s in zip(responses, expected, strict=True)
    ]
    body = {
        "schema": "chipmunk.audio-full-representation-public-development/11",
        "references": REFERENCES,
        "original_lineage": LINEAGE,
        "head": original.load(root / "trained/head.json"),
        "fit_record": original.load(root / "trained/fit.json"),
        "response_header": header,
        "outcomes": outcomes,
        "summary": original.public_summary(original.load(root / "summary.json"), mapping_rows),
        "native_audit": original.load(root / "native-review.json"),
        "numerical_audit": original.load(root / "numerical-review.json"),
        "deployment_eligible": False,
        "data_terms": source["data_terms"],
        "scope": "Complete consumed-source result with original small fitted JSON parameters. Existing public vectors and424 cached probes referenced instead of duplicated; every new probe vector retained. Sanitized outcome/summary bytes differ from original lineage artifacts. No source audio, customer files, transcripts, provider IDs, private paths or full pretrained checkpoints. No fresh confirmation or outside human participation.",
    }
    result = review(body, folder, project=Path(__file__).resolve().parents[2])
    encoded = gzip.compress(
        (json.dumps(body, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0
    )
    with destination.open("xb") as stream:
        stream.write(encoded)
    print(
        json.dumps(
            {
                "sha256": prospective.digest(destination),
                "bytes": len(encoded),
                "outcomes": len(outcomes),
                "passed": result["passed"],
            }
        ),
        flush=True,
    )


def tamper(body, folder):
    assessed = next(i for i, r in enumerate(body["outcomes"]) if r["status"] == "ok")
    unassessed = next(i for i, r in enumerate(body["outcomes"]) if r["status"] == "not_applicable")
    cached = next(i for i, r in enumerate(body["outcomes"]) if r["probe"].get("reused_exact_probe") is True)
    new = next(i for i, r in enumerate(body["outcomes"]) if r["probe"].get("reused_exact_probe") is False)
    names = (
        "changed fitted coefficient",
        "changed source group",
        "changed probability",
        "unassessed called original",
        "changed cached probe PCM",
        "changed new probe vector",
        "changed complete denominator",
        "dropped native case",
        "hidden local path",
    )
    cases = []
    for name in names:
        changed = copy.deepcopy(body)
        if name == names[0]:
            changed["head"]["fitted"]["coefficients"][0] += 0.1
        elif name == names[1]:
            changed["outcomes"][0]["group"] = "altered"
        elif name == names[2]:
            changed["outcomes"][assessed]["candidate_p"] = None
        elif name == names[3]:
            changed["outcomes"][unassessed]["candidate_decision"] = "original"
        elif name == names[4]:
            changed["outcomes"][cached]["probe"]["filtered_pcm_sha256"] = "a" * 64
        elif name == names[5]:
            changed["outcomes"][new]["probe"]["embedding"][0] += 0.1
        elif name == names[6]:
            changed["summary"]["complete_inputs"] -= 1
        elif name == names[7]:
            changed["native_audit"]["cases"].pop()
        else:
            changed["notes"] = "/Users/placeholder/private-source.wav"
        try:
            review(changed, folder)
        except ValueError as error:
            cases.append({"alteration": name, "rejected": True, "reason": str(error)})
        else:
            raise ValueError("Altered actual capsule incorrectly passed: " + name)
    return {
        "schema": "chipmunk.audio-full-representation-public-result-tamper/11",
        "passed": True,
        "cases": cases,
        "scope": "Deliberate local actual-capsule alterations, not neural execution or outside human review.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("export", "review", "tamper", "diagnose"))
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--references", type=Path, default=Path("docs/evaluation-panels"))
    p.add_argument("--project", type=Path)
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    if a.stage == "export":
        export(a.input, a.output, a.references)
    else:
        body = original.load(a.input)
        if a.stage == "tamper":
            result = tamper(body, a.references)
        elif a.stage == "diagnose":
            review(body, a.references, project=a.project)
            result = original.guard_only_ceiling(body["outcomes"])
            result.update(schema="chipmunk.audio-full-representation-guard-only-bound/11")
        else:
            result = review(body, a.references, refit=a.refit, project=a.project)
        write(a.output, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "families")}), flush=True)
