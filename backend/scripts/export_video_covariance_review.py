# SPDX-License-Identifier: Apache-2.0
"""Verify original stereo dropout inputs and export an exact soundtrack comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts import video_shared_sound_panel as prior
from scripts import video_stereo_covariance_panel as covariance

SCHEMA = "chipmunk.video-covariance-independent-review/1"
SOURCE_SCHEMA = "chipmunk.video-stereo-covariance-sources/1"
SYSTEMS = ("selected_covariance", "prior_stereo", "near_silence")
INPUT_FIELDS = ("group", "snr_db", "carrier", "role", "input_sha256", "stereo_pcm_sha256", "samples", "truth",
                "mono_baseline", "audio_eligible")
REASONS = {
    "selected_covariance": "No comparable stereo interval with measured mouth motion.",
    "prior_stereo": "No eligible window in the previous stereo scan.",
    "near_silence": "The retained mono comparator records a proposal when present and omits assessment status.",
}


def eligibility(row, held):
    setting = held["points"][row["carrier"]]["selected"]
    yield covariance.active(row, setting["arm"], setting["mode"], setting["point"])
    earlier = held["baseline"][row["carrier"]]
    choice = row["incumbent"][earlier["arm"]]
    span = choice["span"] if choice and choice["margin_millidb"] > earlier["point"] else None
    yield span, row["audio_eligible"]
    span = row["mono_baseline"]
    yield span, span is not None


def prediction(ident, span, assessed, *, reason):
    if assessed:
        return ident | {"status": "ok", "response": int(span is not None),
                        "localization": {"status": "ok", "regions": [span] if span else []}}
    return ident | {"status": "not_applicable", "response": None, "reason": reason,
                    "localization": {"status": "not_applicable", "reason": reason}}


def check_counts(events, measured, *, coverage):
    for item in events["strata"]:
        row = measured[item["condition"]]
        positive = item["label"]
        population = "edit" if positive else "ordinary"
        if (item["planned"] != row[f"{population}_files"]
                or positive and (item["matched_events"] != row["correct"]
                                 or item["unmatched_proposals"] != row["misplaced"])
                or not positive and item["reported_events"] != row["ordinary_flags"]
                or coverage and item["assessed"] != row[f"{population}_assessed"]):
            raise ValueError("Portable event counts differ from the frozen sample intervals")


def export(receipt, freeze, inputs, deliveries, sources, output):
    held = covariance.read(receipt)
    covariance.verify(held, freeze)
    planned = covariance.read(inputs)
    if (covariance.file_hash(inputs) != held["source_receipt_sha256"]
            or planned.get("schema") != SOURCE_SCHEMA or planned.get("split") != "held"
            or covariance.file_hash(sources / "sources.json.gz") != planned["source_sha256"]
            or planned["sources"] != held["sources"]):
        raise ValueError("The inputs and publisher inventory must match the frozen held receipt")
    root = deliveries.parent
    parents = {row["group"]: row for row in planned["sources"]}
    cases, truth, files = [], [], []
    predictions = {name: [] for name in SYSTEMS}
    for row, source in zip(held["rows"], planned["rows"], strict=True):
        if (row["delivery"] != source["delivery"] or row["incumbent"] != source["with_picture"]
                or any(row[key] != source[key] for key in INPUT_FIELDS)):
            raise ValueError("A scored treatment differs from its verified source row")
        original = parents[row["group"]]
        delivered = covariance.delivery_file(deliveries, row["delivery"])
        parent = covariance.delivery_file(sources, original["name"])
        ident = {"id": delivered.stem, "input_sha256": row["input_sha256"]}
        label = row["truth"]
        cases.append(ident | {
            "split": "confirmation", "modality": "audio", "group": original["speaker"],
            "label": int(label is not None), "source_sha256": original["sha256"], "family": row["role"],
            "condition": f'{row["carrier"]}/{row["snr_db"]}', "lineage": [original["speaker"], original["room_sha256"]],
            "license": {"id": "CREMA-D ODbL 1.0 and DbCL 1.0; DCASE 2013 CC BY 2.0 UK",
                        "evaluation_permitted": True, "redistribution_permitted": False},
            "localization_truth": {"unit": "samples", "source_units": row["samples"],
                                   "target_units": label[1] - label[0] if label else 0},
        })
        truth.append(ident | {"extent": [row["samples"]], "regions": [label] if label else []})
        files.append({"id": ident["id"], "input": delivered.relative_to(root).as_posix(),
                      "source": parent.relative_to(root).as_posix()})
        for system, (span, assessed) in zip(SYSTEMS, eligibility(row, held), strict=True):
            predictions[system].append(prediction(ident, span, assessed, reason=REASONS[system]))
    manifest = {"schema": independent.SCHEMA, "purpose": "Locate staged speech removals in original video soundtracks.",
                "publisher": "https://github.com/CheyneyComputerScience/CREMA-D", "cases": cases}
    manifest_hash = independent.digest(manifest)
    binding = {"manifest_sha256": manifest_hash, "split": "confirmation"}
    truth_doc = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    index = {"schema": INDEX_SCHEMA, "manifest_sha256": manifest_hash, "cases": files}
    verified = verify_inputs(manifest, index, root)
    result = {"schema": SCHEMA, "soundtrack_sample_rate_hz": prior.RATE,
              "measured_sha256": covariance.file_hash(receipt), "input_receipt_sha256": covariance.file_hash(inputs),
              "freeze_sha256": covariance.file_hash(freeze), "manifest": manifest, "file_index": index,
              "truth": truth_doc, "inputs_verified": verified, "readings": {}, "events": {}}
    summary_keys = ("selected", "prior_stereo", "near_silence")
    for (system, rows), summary in zip(predictions.items(), summary_keys, strict=True):
        response = binding | {"system": system, "method_kind": "weighted", "score_kind": "raw",
                              "implementation_sha256": independent.digest([result["measured_sha256"], system]),
                              "weight_sha256": [prior.GEOMETRY_SHA], "response_units": "A localized interval (1 or 0)",
                              "predictions": rows}
        events = count_events(manifest, truth_doc, response, minimum_iou=.4, minimum_recall=.4)
        check_counts(events, held["summary"][summary], coverage=system != "near_silence")
        result["readings"][system], result["events"][system] = response, events
    covariance.save(output, result)
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("receipt", "freeze", "inputs", "deliveries", "sources", "output"):
        parser.add_argument(field, type=Path)
    print(export(**{key: path.resolve() for key, path in vars(parser.parse_args()).items()}))
