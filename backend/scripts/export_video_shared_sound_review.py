# SPDX-License-Identifier: Apache-2.0
"""Bind measured stereo-dropout predictions to the exact videos and parent clips."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts.export_video_lip_correspondence import frame_range
from scripts.video_shared_sound_panel import RATE, digest, read, review, save


def proposals(row, choices):
    choice = choices[row["carrier"]]
    yield "mono", row["mono_baseline"], True
    for mode in ("audio_only", "with_picture"):
        candidate = row[mode][choice["arm"]]
        yield mode, candidate["span"] if candidate and candidate["margin_millidb"] >= choice["point"] else None, row["audio_eligible"]


def prepare(deliveries, sources, freeze, output):
    receipt = deliveries / "result.json.gz"
    review(receipt, freeze)
    data = read(receipt)
    if digest(sources / "sources.json") != data["source_sha256"]:
        raise ValueError("Source inventory differs from the held recipe")
    root = deliveries.parent.resolve()
    parents = {source["group"]: source for source in data["sources"]}
    times = {group: [Fraction(tick * RATE, 30_000) for tick in source["ticks"]] for group, source in parents.items()}
    cases, files, truth = [], [], []
    readings = {"mono": [], "audio_only": [], "with_picture": []}
    for row in data["rows"]:
        parent, clock = parents[row["group"]], times[row["group"]]
        clip = deliveries / row["delivery"]
        case = {"id": clip.stem, "input_sha256": row["input_sha256"]}
        regions = [frame_range(row["truth"], clock)] if row["truth"] else []
        cases.append(case | {"split": "confirmation", "modality": "video", "group": parent["speaker"],
            "label": int(bool(regions)), "source_sha256": parent["sha256"], "family": row["role"],
            "condition": f'{row["carrier"]}/{row["snr_db"]:+d} dB/{row["role"]}',
            "lineage": [parent["speaker"], parent["room_sha256"]],
            "license": {"id": "CREMA-D ODbL 1.0, DbCL 1.0 and DCASE 2013 CC BY 2.0 UK",
                        "evaluation_permitted": True, "redistribution_permitted": False},
            "localization_truth": {"unit": "frames", "source_units": len(clock),
                                   "target_units": regions[0][1] - regions[0][0] if regions else 0}})
        files.append({"id": case["id"], "input": clip.relative_to(root).as_posix(),
                      "source": (sources / parent["name"]).relative_to(root).as_posix()})
        truth.append(case | {"extent": [len(clock)], "regions": regions})
        for mode, span, assessed in proposals(row, data["choices"]):
            reason = "The stereo scan found no eligible windows for a comparison."
            if assessed:
                readings[mode].append(case | {"status": "ok", "response": int(span is not None),
                    "localization": {"status": "ok", "regions": [frame_range(span, clock)] if span else []}})
            else:
                readings[mode].append(case | {"status": "not_applicable", "response": None, "reason": reason,
                    "localization": {"status": "not_applicable", "reason": reason}})
    manifest = {"schema": independent.SCHEMA, "purpose": "Locate staged speech removals under continued ambience.",
                "publisher": "https://github.com/CheyneyComputerScience/CREMA-D", "cases": cases}
    binding = {"manifest_sha256": independent.digest(manifest), "split": "confirmation"}
    truth_doc = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    index = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": files}
    verified = verify_inputs(manifest, index, root)
    result = {"manifest": manifest, "truth": truth_doc, "file_index": index, "verified": verified,
              "sample_receipt_sha256": digest(receipt), "readings": {}, "frame_events": {}}
    for mode, rows in readings.items():
        weights = [] if mode == "audio_only" else [data["programs"]["landmarks"]]
        response = binding | {"system": mode, "method_kind": "weighted" if weights else "model_free", "score_kind": "raw",
            "implementation_sha256": independent.digest([digest(receipt), mode]), "weight_sha256": weights,
            "response_units": "An interval proposed (1 or 0)", "predictions": rows}
        result["readings"][mode] = response
        result["frame_events"][mode] = count_events(manifest, truth_doc, response, minimum_iou=.40, minimum_recall=.40)
    if output.exists():
        raise FileExistsError(output)
    save(output, result)
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("deliveries", "sources", "freeze", "output"):
        parser.add_argument(arg, type=Path)
    options = {key: value.resolve() for key, value in vars(parser.parse_args()).items()}
    print(prepare(**options))
