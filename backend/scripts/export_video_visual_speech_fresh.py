"""Rebuild fresh word edits and export exact video-frame truth for local review."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts import export_video_lip_correspondence as export
from scripts import export_video_lip_sequence as sequence
from scripts import video_lip_correspondence as data
from scripts import video_visual_speech_evaluation as trial


def prediction(case, span, times, assessed):
    if span or assessed:
        regions = [export.frame_range(span, times)] if span else []
        return case | {"status": "ok", "response": int(bool(regions)), "localization": {"status": "ok", "regions": regions}}
    reason = "The clip did not support a localized speech correspondence measurement."
    return case | {"status": "inconclusive", "response": None, "reason": reason,
                   "localization": {"status": "inconclusive", "reason": reason}}


def prepare(plan, locked, measured, sources, output):
    from app.detectors.audio.phone_events import WEIGHT_PIN
    from app.evidence.video_reference_timing import MODEL_SHA256

    roster, freeze, result = map(data.read, (plan, locked, measured))
    trial.review(locked)
    trial.review(measured)
    if (result["freeze"] != data.sha(locked) or result["selected"] != freeze["selected"]
            or result["wide_offset"] != freeze["wide_offset"]):
        raise ValueError("The result differs from its frozen candidate")
    entries = roster["rows"]
    locations = {case["target"]["front_sha256"]: case["target"]["span"] for case in entries}
    rows = {(row["picture_sha256"], row["carrier"], row["role"]): row | {"source_pcm_sha256": row["audio_sha256"]}
            for row in result["rows"]}
    expected = {(picture, delivery, role) for picture in locations for delivery in export.PROFILES for role in export.ORDER}
    if (rows.keys() != expected or len(result["rows"]) != len(expected) or len(locations) != len(entries)
            or any(row["split"] != "fresh" or row["truth"] != locations[picture] for (picture, _, _), row in rows.items())):
        raise ValueError("The fresh source and delivery rosters differ")

    output.mkdir(mode=0o700)
    manifest, truth, index, baseline, combined = [], [], [], [], []
    choice = result["selected"]
    for submitted, source, times, row in sequence._readbacks(entries, rows, sources, output):
        case = {"id": submitted.stem, "input_sha256": data.sha(submitted)}
        group = f"lombard-grid/{row['speaker']}"
        role = row["role"]
        region = export.frame_range(row["truth"], times) if role == "different_words" else []
        manifest.append(case | {"split": "confirmation", "modality": "video", "group": group,
            "label": int(bool(region)), "family": export.LABELS[role], "condition": f"{row['carrier']}/{role}",
            "source_sha256": row["picture_sha256"], "lineage": [group],
            "license": {"id": "CC BY 4.0 (Lombard Grid)", "evaluation_permitted": True, "redistribution_permitted": True},
            "localization_truth": {"unit": "frames", "source_units": len(times),
                                   "target_units": region[1] - region[0] if region else 0}})
        truth.append(case | {"extent": [len(times)], "regions": [region] if region else []})
        index.append({"id": case["id"], "input": submitted.relative_to(output).as_posix(), "source": source.relative_to(output).as_posix()})
        assessed = row["baseline_status"] in {"matched", "no_match"}
        baseline.append(prediction(case, row["baseline"], times, assessed))
        combined.append(prediction(case, row["baseline"] or trial.propose(row, choice), times,
                                   assessed or bool(row["choices"].get(choice["arm"]))))
    roster = {"schema": independent.SCHEMA, "purpose": "Locate short audible word substitutions on unchanged native pictures.",
              "publisher": "https://zenodo.org/records/3736465", "cases": manifest}
    binding = {"manifest_sha256": independent.audit_manifest(roster)["manifest_sha256"], "split": "confirmation"}
    truth_doc = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    files = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": index}
    checked = verify_inputs(roster, files, output)
    for name, document in (("manifest", roster), ("truth", truth_doc), ("file-index", files), ("input-verification", checked)):
        export.json_file(output / f"{name}.json", document)
    weights = [WEIGHT_PIN, MODEL_SHA256]
    for name, predictions in (("current", baseline), ("combined", combined)):
        response = binding | {"system": name, "method_kind": "weighted", "score_kind": "raw",
            "implementation_sha256": independent.digest([name, result["freeze"], data.sha(measured)]),
            "weight_sha256": weights if name == "current" else weights + [freeze["weight"]],
            "response_units": "Selected local speech mismatch (1 or 0)", "predictions": predictions}
        summary = count_events(roster, truth_doc, response, minimum_iou=.40, minimum_recall=.40)
        export.json_file(output / f"{name}.json", response)
        export.json_file(output / f"{name}-events.json", summary)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "locked", "measured", "sources", "output"):
        cli.add_argument(arg, type=Path)
    prepare(**vars(cli.parse_args()))
