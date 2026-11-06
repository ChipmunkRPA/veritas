"""Publish corrected canonical connections and exact recipes, never private text."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.audio_source_identity_v9 import connection_keys, digest, speaker_key
from scripts.export_audio_multisource_v8 import public_source, public_variant


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source(row):
    result = public_source(row)
    for old, new in (
        ("voice_key_sha256", "declared_voice_sha256"),
        ("source_key_sha256", "declared_source_sha256"),
    ):
        if old in result:
            result[new] = result.pop(old)
    canonical = speaker_key(row)
    result["canonical_speaker_sha256"] = fingerprint(canonical) if canonical else None
    result["connection_fingerprints"] = sorted(fingerprint(key) for key in connection_keys(row))
    for key in ("prior_role", "prior_group", "recipe_policy"):
        if key in row:
            result[key] = row[key]
    return result


def export(root, output):
    path = root / "source-plan.json"
    plan = json.loads(path.read_text())
    old_root = Path(plan["original_source_plan_root"])
    audit_path = old_root / "source-planning/cross-corpus-identity-audit.json"
    if plan["identity_audit_sha256"] != digest(audit_path):
        raise ValueError("Original identity audit changed")
    variants, hashes = {}, {}
    for name in ("development", "confirmation"):
        manifest_path = root / name / "variants.json"
        body = json.loads(manifest_path.read_text())
        if body["source_plan_sha256"] != digest(path):
            raise ValueError("Repaired recipes have a different source plan")
        hashes[name] = digest(manifest_path)
        variants[name] = []
        for row in body["variants"]:
            current = public_variant(row)
            current.update(
                retained_exact_v8_input=row["retained_exact_v8_input"],
                prior_input_sha256=row["prior_input_sha256"],
            )
            variants[name].append(current)
    body = {
        "schema": "chipmunk.audio-canonical-identity-public-plan/9",
        "source_plan_sha256": digest(path),
        "identity_audit_sha256": plan["identity_audit_sha256"],
        "planning_code_sha256": plan["planning_code_sha256"],
        "identity_code_sha256": plan["identity_code_sha256"],
        "export_code_sha256": digest(Path(__file__)),
        "original_source_catalogue": "audio-v8-source-plan.json.gz",
        "original_source_catalogue_sha256": "ba58b6646be0d9ce36f8675d80fff3ea146ced3b767882d86bcd02cfa53af0f3",
        "sources": [source(r) for r in plan["sources"]],
        "backgrounds": [source(r) for r in plan["backgrounds"]],
        "retired_consumed_sources": [
            {
                "source_sha256": r["sha256"],
                "prior_role": r["role"],
                "canonical_speaker_sha256": fingerprint(speaker_key(r)),
                "reason": r["retirement_reason"],
            }
            for r in plan["retired_consumed_sources"]
        ],
        "variants": variants,
        "variants_sha256": hashes,
        "role_counts": plan["role_counts"],
        "role_group_counts": plan["role_group_counts"],
        "role_generator_counts": plan["role_generator_counts"],
        "role_changes": plan["role_changes"],
        "freshness": plan["freshness"],
        "recipe_policy": plan["recipe_policy"],
        "learner_policy": plan["learner_policy"],
        "limitations": plan["limitations"],
        "fitting_started": False,
        "new_neural_inference_run": False,
        "deployment_eligible": False,
        "data_terms": "The original v8 catalogue's per-source terms/attributions continue to apply, including CC-BY-SA4 for Dictation Speech-derived metadata. No raw audio, transcripts, provider IDs, local paths or pretrained weights are included.",
    }
    with output.open("xb") as stream:
        stream.write(gzip.compress((json.dumps(body, indent=2, allow_nan=False) + "\n").encode(), mtime=0))
    print(
        json.dumps(
            {
                "capsule_sha256": digest(output),
                "bytes": output.stat().st_size,
                "sources": len(body["sources"]),
                "inputs": sum(len(v) for v in variants.values()),
                "fitting_started": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.root, args.output)
