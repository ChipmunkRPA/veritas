"""Offline review of published fresh-audio preparation or full arithmetic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.audio_confirmation_v7 import CONDITIONS, HEAD_SHA256, PREPARED_SHA256
from scripts.prepare_audio_confirmation_v7 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.review_audio_confirmation_v7 import reconstruct


def same_tree(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10)
    if isinstance(a, int):
        return isinstance(b, int) and a == b
    return a == b


def review(body, source_catalogue):
    lock = body["lock"]
    if lock["head_sha256"] != HEAD_SHA256 or lock["prepared_sources_sha256"] != PREPARED_SHA256:
        raise ValueError("Original source/candidate binding differs")
    head_bytes = (json.dumps(body["head"], indent=2, allow_nan=False) + "\n").encode()
    if hashlib.sha256(head_bytes).hexdigest() != HEAD_SHA256:
        raise ValueError("Published candidate head differs from frozen original artifact")
    variants = body["variants"]
    if (
        len(variants) != 1080
        or len({r["id"] for r in variants}) != 1080
        or len({r["base_id"] for r in variants}) != 120
    ):
        raise ValueError("Every planned input/source required")
    foreground = {r["id"]: r for r in source_catalogue["sources"] if r["role"] == "foreground"}
    backgrounds = {r["id"]: r for r in source_catalogue["backgrounds"]}
    backgrounds.update({r["id"]: r for r in source_catalogue["sources"] if r["role"] == "background"})
    if {r["base_id"] for r in variants} != set(foreground):
        raise ValueError("Original public foreground source set differs")
    for row in variants:
        source = foreground[row["base_id"]]
        if (
            any(row[k] != source[k] for k in ("label", "generator", "dataset"))
            or row["source_sha256"] != source["sha256"]
            or row["source_prefix_pcm_sha256"] != source["source_prefix_pcm_sha256"]
            or row["group"] != source["connected_group"]
        ):
            raise ValueError("Truth/source/group differs from original public source catalogue")
        if row["background_id"] is not None:
            bg = backgrounds[row["background_id"]]
            if row["background_sha256"] != bg["sha256"] or row["background_kind"] != (
                bg.get("kind") or "speech"
            ):
                raise ValueError("Background source differs from original catalogue")
    for source in {r["base_id"] for r in variants}:
        current = [r for r in variants if r["base_id"] == source]
        if (
            Counter(r["condition"] for r in current) != Counter(CONDITIONS)
            or len({r["group"] for r in current}) != 1
        ):
            raise ValueError("Repeated source conditions or connected group changed")
        if any(
            r["samples"] != 64000 or r["source_interval_samples"] != [0, 64000] or r["sample_rate"] != 16000
            for r in current
        ):
            raise ValueError("Source sample clock differs")
    if len({r["group"] for r in variants}) != 110:
        raise ValueError("Known shared prompt groups lost")
    audit = body["input_audit"]
    if (
        not audit["passed"]
        or audit["variants_sha256"] != lock["variants_sha256"]
        or {r["id"] for r in audit["cases"]} != {r["id"] for r in variants}
    ):
        raise ValueError("Complete source/mixture audit receipt required")
    parity = body["consumed_parity"]
    if (
        not parity["passed"]
        or len(parity["cases"]) != 13
        or parity["implementation_sha256"] != lock["implementation_sha256"]
    ):
        raise ValueError("Current-runner consumed-source parity differs")
    result = {
        "passed": True,
        "inputs": len(variants),
        "source_groups": 110,
        "consumed_parity_cases": 13,
        "head_artifact_sha256_verified": True,
        "detector_outcomes_reviewed": False,
        "scope": "Local portable source-role/count/head/receipt review; no source decoding, neural forwards or outside reviewer.",
    }
    if body["detector_inference_run"]:
        if body["response_header"]["lock_sha256"] != body["original_lock_sha256"]:
            raise ValueError("Responses bound to different original lock")
        native = body["native_audit"]
        selected = {r["id"] for r in body["input_audit"]["cases"] if r["sampled_production_decode"]}
        if (
            native["lock_sha256"] != body["original_lock_sha256"]
            or native["scores_sha256"] != body["summary"]["scores_sha256"]
            or {r["id"] for r in native["cases"]} != selected
            or len(native["cases"]) != 18
        ):
            raise ValueError("Native audit did not preserve the18 preselected source cells")
        rebuilt = reconstruct(lock, body["head"], variants, body["outcomes"])
        if any(
            not same_tree(rebuilt[k], body["summary"][k])
            for k in (
                "groups",
                "differences",
                "gates",
                "passes",
                "strata",
                "execution_status",
                "probe_status",
            )
        ):
            raise ValueError("Published complete summary/gates differ")
        result.update(
            detector_outcomes_reviewed=True,
            outcomes=1080,
            gates_passed=sum(rebuilt["gates"].values()),
            candidate_passes=rebuilt["passes"],
            native_replay_cells=len(native["cases"]),
            native_replay_receipt_passed=native["passed"] and all(r["passed"] for r in native["cases"]),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    body = json.loads(gzip.decompress(args.capsule.read_bytes()))
    if body["source_catalogue"] != "audio-v7-fresh-sources.json":
        raise ValueError("Expected fixed source catalogue beside capsule")
    sources = args.capsule.parent / body["source_catalogue"]
    if digest(sources) != body["source_catalogue_sha256"]:
        raise ValueError("Original public source catalogue hash differs")
    result = review(body, json.loads(sources.read_text()))
    result["capsule_sha256"] = digest(args.capsule)
    write(args.output, result)
    print(json.dumps(result), flush=True)
