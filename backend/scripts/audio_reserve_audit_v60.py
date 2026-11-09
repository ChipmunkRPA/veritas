"""Audit recorded research metadata for exposure of the fixed audio reserve.

Plans and reconstructions are not detector exposures. This inventories both and
separately reports identity-matched records carrying measured screening fields.
It never opens customer uploads, decodes media, imports models or fits anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-reserve-audit-v60"
SCALAR_SIGNALS = frozenset({"raw", "score", "probability", "p", "original_p", "candidate_p",
                            "decision", "original_decision", "candidate_decision", "verdict"})
VECTOR_SIGNALS = frozenset({"embedding", "wavlm", "logits", "layer_embeddings"})


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def signals(row):
    numeric = {key for key in SCALAR_SIGNALS if type(row.get(key)) in (int, float)}
    directional = {key for key in SCALAR_SIGNALS if isinstance(row.get(key), str)
                   and row[key] in {"ai", "original", "inconclusive", "unassessed", "leaning_ai",
                                    "leaning_original", "synthetic", "genuine"}}
    vectors = {key for key in VECTOR_SIGNALS if isinstance(row.get(key), list) and row[key]}
    return numeric | directional | vectors


def matches(value, identities, inherited=frozenset(), pointer=""):
    if isinstance(value, dict):
        context = inherited | signals(value)
        fields = sorted(key for key, item in value.items() if isinstance(item, str) and item in identities)
        if fields:
            yield {"pointer": pointer, "identity_fields": fields, "screening_fields": sorted(context)}
        for key, item in value.items():
            if isinstance(item, dict | list):
                yield from matches(item, identities, context, pointer + "/" + key)
    elif isinstance(value, list):
        # Sibling measurements must never be attached to a different source row.
        for index, item in enumerate(value):
            if isinstance(item, dict | list):
                yield from matches(item, identities, inherited, pointer + "/" + str(index))


def inventory(base, excluded):
    paths, seen_directories = [], set()
    for directory, names, files in os.walk(base, followlinks=True):
        folder = Path(directory)
        real = folder.resolve()
        if folder == excluded or real in seen_directories:
            names[:] = []
            continue
        seen_directories.add(real)
        for name in files:
            path = folder / name
            if path.suffix in {".json", ".jsonl"}:
                paths.append(path)
    return sorted(paths)


def audit():
    source_path, variants_path = BASE / "audio-v9/source-plan.json", BASE / "audio-v9/confirmation/variants.json"
    source = json.loads(source_path.read_text())
    variants = json.loads(variants_path.read_text())["variants"]
    foregrounds = [row for row in source["sources"] if row["role"] == "confirmation"]
    if len(variants) != 1080 or len(foregrounds) != 120 or {row["role"] for row in variants} != {"confirmation"}:
        raise ValueError("Complete unchanged reserve required")
    fields = {"id", "sha256", "pcm_sha256", "payload_pcm_sha256", "source_sha256",
              "production_pcm_sha256", "decoded_pcm_sha256", "source_prefix_pcm_sha256"}
    identities = {value for row in [*foregrounds, *variants] for key, value in row.items()
                  if key in fields and isinstance(value, str)}
    paths = inventory(BASE, TRIAL)
    TRIAL.mkdir(exist_ok=False)
    bindings = {str(path.relative_to(BASE)): digest(path) for path in paths}
    write(TRIAL / "input-bindings.json", bindings)
    details, errors, objects, records = [], [], 0, 0
    for path in paths:
        relative = str(path.relative_to(BASE))
        found, parsed = [], 0
        try:
            with path.open() as stream:
                values = enumerate(stream, 1) if path.suffix == ".jsonl" else [(None, stream.read())]
                for line, raw in values:
                    value = json.loads(raw)
                    parsed += 1
                    found.extend({"line": line, **item} for item in matches(value, identities))
            if digest(path) != bindings[relative]:
                raise ValueError("Historical metadata changed during audit")
        except (ValueError, OSError) as error:
            errors.append({"path": relative, "error": type(error).__name__, "detail": str(error)[:250]})
        objects += parsed
        records += len(found)
        if found:
            details.append({"path": relative, "records": len(found),
                            "screening_record_count": sum(bool(row["screening_fields"]) for row in found),
                            "matches": found})
    write(TRIAL / "matched-records.json", details)
    write(TRIAL / "errors.json", errors)
    consumed = [row for row in source["sources"] if row["role"] != "confirmation"]
    overlap = {}
    for field in ("sha256", "canonical_speaker", "transcript_sha256", "source_prefix_pcm_sha256"):
        prior = {row[field] for row in consumed if row.get(field)}
        overlap[field] = len({row[field] for row in foregrounds if row.get(field)} & prior)
    screening_paths = [{"path": row["path"], "records": row["screening_record_count"]}
                       for row in details if row["screening_record_count"]]
    summary = {"schema": "chipmunk.audio-reserve-recorded-exposure-audit/60",
               "source_plan_sha256": digest(source_path), "variants_sha256": digest(variants_path),
               "input_bindings_sha256": digest(TRIAL / "input-bindings.json"),
               "audit_code_sha256": digest(Path(__file__)), "metadata_files": len(paths),
               "metadata_bytes": sum(path.stat().st_size for path in paths), "parsed_objects": objects,
               "identity_matched_records": records, "identity_matched_files": len(details),
               "screening_record_paths": screening_paths, "errors": errors,
               "reserve_foregrounds": len(foregrounds), "reserve_groups": len({r["group"] for r in foregrounds}),
               "reserve_inputs": len(variants), "generators": dict(Counter(r["generator"] for r in foregrounds)),
               "overlap_with_v9_consumed_sources": overlap,
               "matched_records_sha256": digest(TRIAL / "matched-records.json"),
               "new_model_calls": 0, "fresh_test_started": False,
               "scope": "Recorded research JSON/JSONL identity matches, including plans and reconstruction. Screening-field matches require review; no matches cannot prove absence of unrecorded calls, biometric overlap or pretraining exposure. Historical voice reuse outside the fitting pool remains disclosed."}
    write(TRIAL / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise ValueError("Preserve incomplete audit; do not claim reserve eligibility")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    audit()
