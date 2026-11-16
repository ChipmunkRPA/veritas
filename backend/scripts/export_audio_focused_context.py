# SPDX-License-Identifier: MIT
"""Export source-timed longer-audio truth and responses for independent comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.bench import independent
from app.bench.independent_locations import TRUTH_SCHEMA, count_events, count_locations
from scripts import audio_focused_context_transfer as study

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ASSETS = ROOT / "docs/evaluation-panels/audio-joint-generator-conversion-2026-11-16.json.gz"
INCUMBENT_SOURCE = ROOT / "docs/evaluation-panels/audio-short-edit-detector-training-2026-11-14.json"


def export(receipt, output):
    summary = study.validate(receipt)
    frozen = receipt["freeze"]
    origins = {row["id"]: row for row in frozen["origins"]}
    rooms = {row["room"]: row for row in frozen["backgrounds"]}
    membership = []
    targets = []
    for row in receipt["rows"]:
        case, treatment, kind = row["id"].split("/")
        room = rooms.get(row["room"])
        origin = origins[case]
        assert (treatment, kind, origin["group"]) == (row["treatment"], row["kind"], row["group"])
        reader = f"llama-partial-spoof/{row['group']}"
        background = kind == "background"
        condition = treatment + ("/background" if background else "")
        group = f"ami/{room['room']}" if background else reader
        source_hash = room["pcm_sha256"] if background else origin["source_sha256"]
        lineage = [reader, room["pcm_sha256"]] if room else [reader]
        membership.append({"id": row["id"], "modality": "audio", "split": "confirmation", "group": group,
                           "label": int(kind == "edited"), "family": row["family"] or kind, "condition": condition,
                           "source_sha256": source_hash, "input_sha256": row["pcm_sha256"], "lineage": lineage,
                           "license": {"id": "CC BY 4.0 (LlamaPartialSpoof and AMI)",
                                       "evaluation_permitted": True, "redistribution_permitted": True},
                           "localization_truth": {"unit": "samples", "source_units": row["samples"],
                                                  "target_units": sum(b - a for a, b in row["truth"])}})
        targets.append({"id": row["id"], "input_sha256": row["pcm_sha256"],
                        "extent": [row["samples"]], "regions": row["truth"]})
    manifest = {"schema": independent.SCHEMA, "sample_rate_hz": study.RATE, "cases": membership}
    identity = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    truth = identity | {"schema": TRUTH_SCHEMA, "cases": targets}
    output.mkdir(mode=0o700)
    study.save(output / "manifest.json", manifest)
    study.save(output / "truth.json", truth)
    prior = study.read(SOURCE_ASSETS)
    speech_encoder = prior["freeze"]["encoder"]["assets_sha256"]["pytorch_model.bin"]
    assets = study.read(INCUMBENT_SOURCE)["incumbent"]["assets"]
    baseline = [assets["pellav2_detector.pt"], assets["silero_vad.jit"]]
    method_sha = study.digest(Path(study.__file__).read_bytes())
    focused_only = receipt.get("stage") == "focused-only"
    names = ({"current_plus_focused": "focused_excerpt", "current_plus_repeated": "repeated_focused_excerpt"}
             if focused_only else {arm: arm for arm in study.ARMS})
    for arm, name in names.items():
        predictions = []
        for case, row in zip(membership, receipt["rows"], strict=True):
            proposed = study.proposed(row)[arm]
            checked = {
                "current": False,
                "current_plus_short": any(r["short_assessed"] for r in row["excerpts"]),
                "current_plus_focused": any(r["checked_windows"] for r in row["excerpts"]),
                "current_plus_repeated": sum(r["checked_windows"] > 0 for r in row["excerpts"]) >= 2,
            }[arm]
            result = {"id": case["id"], "input_sha256": case["input_sha256"]}
            if row["current_assessed"] or checked:
                result.update(status="ok", response=int(bool(proposed)),
                              localization={"status": "ok", "regions": [list(span) for span in proposed]})
            else:
                reason = "No usable responses on the required speech excerpts"
                result.update(status="not_applicable", response=None, reason=reason,
                              localization={"status": "not_applicable", "reason": reason})
            predictions.append(result)
        weights = baseline[1:] if focused_only else baseline
        response = identity | {"system": name, "implementation_sha256": independent.digest([method_sha, name]),
                               "method_kind": "weighted", "score_kind": "raw",
                               "weight_sha256": weights + ([speech_encoder, frozen["focused_head_sha256"]]
                                                             if arm in study.ARMS[2:] else []),
                               "response_units": "A source interval was flagged (0 or 1)", "predictions": predictions}
        events = count_events(manifest, truth, response, minimum_iou=.30, minimum_recall=.50)
        locations = count_locations(manifest, truth, response)
        for treatment, results in summary.items():
            scalar = results[arm]
            rows = [item for item in events["strata"] if item["condition"] == treatment and item["label"] == 1]
            assert sum(item["matched_events"] for item in rows) == scalar["located"]
            assert sum(item["target_events"] for item in rows) == scalar["truth_ranges"]
            genuine = 0
            for case, result in zip(membership, locations["predictions"], strict=True):
                if case["condition"] != treatment or result["status"] != "ok":
                    continue
                stats = result["localization"]
                genuine += stats["reported_units"] - stats["overlap_units"]
            assert genuine == scalar.get("flagged_genuine_samples", 0)
        study.save(output / f"{name}.json", response)
        study.save(output / f"{name}-events.json", events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(study.read(args.receipt), args.output)
