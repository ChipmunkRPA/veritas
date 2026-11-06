"""Freeze unchanged source roles and the actual guarded-service comparator."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from app.detectors.audio import calibration as serving
from scripts import audio_calibration_v3_methods as guard
from scripts.audio_guard_v7 import CONDITIONS, METHOD, source_roles
from scripts.audio_quality_v5 import digest, verify


def implementation():
    root = Path(__file__).resolve().parents[2]
    names = [
        "backend/scripts/prepare_audio_guard_v7.py",
        "backend/scripts/audio_guard_v7.py",
        "backend/scripts/run_audio_guard_v7.py",
        "backend/scripts/audio_residual_head_v6.py",
        "backend/scripts/review_audio_noise_v4.py",
        "backend/scripts/audio_quality_v5.py",
        "backend/scripts/audio_calibration_v3_methods.py",
        "backend/app/detectors/audio/calibration.py",
        "backend/app/detectors/audio/upgraded.py",
        "backend/app/detectors/audio/decode.py",
        "backend/app/third_party/kaldi/fbank.py",
        "backend/app/models.yaml",
        "docs/audio-guard-v7-development.md",
    ]
    return {p: digest(root / p) for p in names}


def write(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def current_reference(row, old, bundle):
    p = serving.probability(row["raw"], bundle)
    if p != row["serving_p"]:
        raise ValueError("Saved baseline probability differs from actual service")
    decision = serving.initial_decision(p, bundle)
    if decision == "original":
        band = old["new_band"]
        if band is None or abs(
            serving.logit(round(row["raw"], 6)) - serving.logit(round(band, 6))
        ) > math.log(2):
            decision = "inconclusive"
    if decision != row["serving_decision"]:
        raise ValueError("Saved baseline guard/probe decision differs from actual service")
    if not (bundle["min_seconds"] <= row["seconds"] <= bundle["max_seconds"]):
        raise ValueError("Development input outside frozen service calibration scope")
    return {
        "id": row["id"],
        "probability_ai": p,
        "decision": decision,
        "saved_band_response": old["new_band"],
        "probability_and_decision_parity": True,
    }


def freeze(root, output):
    source = root / "audio-v3"
    prior_path = root / "audio-v6/development/manifest.json"
    prior = json.loads(prior_path.read_text())
    verify(prior)
    m = json.loads((source / "manifest.json").read_text())
    expected = {r["id"]: r for r in json.loads((source / "predictions.json").read_text())}
    old_by_id = {r["id"]: r for r in prior["variants"]}
    eval_ids = {r["base_id"] for r in prior["variants"]}
    bases = [b for b in m["base_files"] if b["split"] in ("fit", "cal") or b["id"] in eval_ids]
    assignment, groups = source_roles(bases)
    if Counter(r["role"] for r in assignment.values()) != {"fit": 40, "cal": 80, "development": 120}:
        raise ValueError("All original40/80 and120 development sources required")
    hashes = {}
    for b in bases:
        role = assignment[b["id"]]["role"]
        if b["sha256"] in hashes and hashes[b["sha256"]] != role:
            raise ValueError("Original source bytes cross roles")
        hashes[b["sha256"]] = role
    scores = {}
    with (source / "scores.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["detector"] == "new" and row["split"] in ("fit", "cal"):
                if row["variant_id"] in scores:
                    raise ValueError("Duplicate fitting/calibration response")
                scores[row["variant_id"]] = row
    bundle = serving.load()
    if guard.METHOD["class_set_alpha"] != 0.1 or guard.METHOD["lean_threshold"] != 0.8:
        raise ValueError("Frozen internal diagnostic/guard policy changed")
    rows, reference = [], []
    for role in ("fit", "cal", "development"):
        for original in m["variants"]:
            if original["base_id"] not in assignment or assignment[original["base_id"]]["role"] != role:
                continue
            row = {**original, **assignment[original["base_id"]]}
            if role == "development":
                old = old_by_id[row["id"]]
                if any(row[k] != old[k] for k in ("sha256", "pcm_sha256", "samples", "condition", "label")):
                    raise ValueError("Existing complete development panel changed")
                row.update(
                    raw=old["raw"],
                    raw_status=old["raw_status"],
                    original_windows=old["original_windows"],
                    serving_p=old["serving_p"],
                    serving_decision=old["serving_decision"],
                )
                reference.append(current_reference(row, expected[row["id"]], bundle))
            else:
                score = scores[row["id"]]
                if (
                    score["input_sha256"] != row["sha256"]
                    or score["production_input"]["pcm_sha256"] != row["pcm_sha256"]
                ):
                    raise ValueError("Fit/cal raw response bound to another source")
                row.update(
                    raw=score["raw_unrounded"], raw_status=score["status"], original_windows=score["windows"]
                )
            if row["raw"] is not None and (
                len(row["original_windows"]) != 1
                or row["original_windows"][0]["start_sample"] != 0
                or row["original_windows"][0]["end_sample"] != row["samples"]
            ):
                raise ValueError("One complete original speech window is required")
            rows.append(row)
    if (
        Counter(r["role"] for r in rows) != {"fit": 240, "cal": 480, "development": 720}
        or len({r["id"] for r in rows}) != 1440
    ):
        raise ValueError("Every planned input must be retained exactly once")
    for base in assignment:
        variants = [r for r in rows if r["base_id"] == base]
        if len(variants) != 6 or {r["condition"] for r in variants} != set(CONDITIONS):
            raise ValueError("Missing ordinary/noise/music condition")
    backgrounds = {
        role: {r["background_sha256"] for r in rows if r["role"] == role and r["background_sha256"]}
        for role in ("fit", "cal", "development")
    }
    if any(
        backgrounds[a] & backgrounds[b]
        for a, b in (("fit", "cal"), ("fit", "development"), ("cal", "development"))
    ):
        raise ValueError("Background bytes cross source roles")
    output.mkdir(parents=True, exist_ok=False)
    body = {
        "schema": "chipmunk.audio-source-guard-lock/7",
        "method": METHOD,
        "source_root": str(source.resolve()),
        "variants": rows,
        "sources": bases,
        "groups": groups,
        "implementation_sha256": implementation(),
        "serving_bundle_sha256": serving.SHA256,
        "sources_bound": {
            str(p.resolve()): digest(p)
            for p in (
                source / "manifest.json",
                source / "scores.jsonl",
                source / "predictions.json",
                prior_path,
                root / "audio-v6/development/representations.jsonl",
            )
        },
        "prior_representation_receipt": str((root / "audio-v6/development/representations.jsonl").resolve()),
        "reference": reference,
        "deployment_eligible": False,
    }
    write(output / "manifest.json", body)
    print(
        json.dumps(
            {
                "manifest_sha256": digest(output / "manifest.json"),
                "inputs": 1440,
                "roles": dict(Counter(r["role"] for r in rows)),
                "baseline_replays": len(reference),
                "connected_groups": dict(Counter(r["role"] for r in groups)),
            }
        ),
        flush=True,
    )


def check(path):
    body = json.loads(path.read_text())
    if body["implementation_sha256"] != implementation() or body["serving_bundle_sha256"] != serving.SHA256:
        raise ValueError("Frozen method or serving calibration changed")
    serving.load()
    if any(digest(Path(p)) != sha for p, sha in body["sources_bound"].items()):
        raise ValueError("Frozen source/response receipt changed")
    return body


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze(args.root, args.output)
