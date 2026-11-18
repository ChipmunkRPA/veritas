# SPDX-License-Identifier: Apache-2.0
"""Verify the exact held portrait files and export their labeled frame intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts import video_common_sources_v89 as io
from scripts.review_video_face_portrait import METHODS, read, save, selected, verify
from scripts.video_face_portrait_panel import GRID


def export(receipt, freeze, root, output, *, method):
    result = read(receipt)
    verify(result, freeze)
    cases, truth, files = [], [], []
    predictions = {system: [] for system in ("without_shot_check", "with_shot_check")}
    for height, panel in sorted(result["banks"].items()):
        point = result["points"][height][method]
        for row in panel["rows"]:
            ident = {"id": f'{height}-{Path(row["file"]).stem}', "input_sha256": row["input_sha256"]}
            label = row["truth"]
            cases.append(ident | {"split": "confirmation", "modality": "video", "group": row["group"],
                "label": int(label is not None), "source_sha256": row["source_sha256"], "family": row["kind"],
                "condition": f'{height}p/{row["delivery"]}/{row["kind"]}',
                "lineage": [row["group"], row["treated_sha256"]],
                "license": {"id": "RTFS publisher CC BY-SA 4.0", "evaluation_permitted": True,
                            "redistribution_permitted": False},
                "localization_truth": {"unit": "frames", "source_units": GRID.frame_count,
                                       "target_units": label[1] - label[0] if label else 0}})
            truth.append(ident | {"extent": [GRID.frame_count], "regions": [label] if label else []})
            files.append({"id": ident["id"], "input": f'held-{height}/{row["file"]}',
                          "source": f'sources/{row["source_index"]:02d}-ordinary.mp4'})
            for system, rows in predictions.items():
                if row["proposal"] is not None:
                    span = selected(row, point, system == "with_shot_check")
                    rows.append(ident | {"status": "ok", "response": int(span is not None),
                        "localization": {"status": "ok", "regions": [span] if span else []}})
                else:
                    reason = "No comparable face range and surrounding pictures."
                    rows.append(ident | {"status": "not_applicable", "response": None, "reason": reason,
                        "localization": {"status": "not_applicable", "reason": reason}})
    manifest = {"schema": independent.SCHEMA, "purpose": "Locate staged brief face swaps on a portrait clock.",
                "publisher": io.RTFS, "cases": cases}
    binding = {"manifest_sha256": independent.digest(manifest), "split": "confirmation"}
    truth_doc = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    index = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": files}
    verified = verify_inputs(manifest, index, root)
    weights = sorted({pin for bank in result["banks"].values() for pins in bank["header"]["weights"].values() for pin in pins})
    exported = {"manifest": manifest, "file_index": index, "truth": truth_doc, "inputs_verified": verified,
                "measured_sha256": io.sha(receipt), "readings": {}, "events": {}}
    for system, rows in predictions.items():
        response = binding | {"system": system, "method_kind": "weighted", "score_kind": "raw",
            "implementation_sha256": independent.digest([io.sha(receipt), method, system]),
            "weight_sha256": weights, "response_units": "A localized interval (1 or 0)", "predictions": rows}
        exported["readings"][system] = response
        exported["events"][system] = count_events(manifest, truth_doc, response, minimum_iou=.5, minimum_recall=.5)
    save(output, exported)
    print(verified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "freeze", "root", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--method", choices=METHODS, required=True)
    export(**vars(parser.parse_args()))
