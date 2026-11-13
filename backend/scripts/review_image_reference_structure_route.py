# SPDX-License-Identifier: Apache-2.0
"""Reproduce the two complete earlier-copy routes on the frozen JPEG confirmation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from app.detectors.image import reference_structure_jpeg
from app.evidence.image_reference import inspect
from scripts import image_reference_monotone_jpeg_confirmation as previous
from scripts import image_reference_monotone_jpeg_transfer as tone
from scripts import image_reference_structure_jpeg_transfer as trial

SCHEMA = "chipmunk.image-reference-structure-serving/1"
OPTIONS = {"allow_proportional_resize": True, "allow_oriented_crop": True, "allow_oriented_resize": True,
           "allow_embedded_photo": True, "allow_exact_jpeg_identity": True, "allow_shifted_jpeg": True}


def evaluate(args):
    profile = tone.read(args.held / "freeze.json")
    tone.require(profile["schema"] == trial.SCHEMA and profile["stage"] == "held", "Expected the frozen held comparison")
    banks = {}
    for bank, directory in (("v225", args.source_roster), ("v262", args.backup_roster)):
        for row in tone.read(directory / "qualified-private.json")["qualified"]:
            source = tone.sha(str(row["source"]["flickr_photo_id"]).encode())
            banks[source] = bank
    rows = []
    with TemporaryDirectory(prefix="chipmunk-jpeg-structure-route-") as work:
        directory = Path(work)
        earlier, submitted = directory / "earlier.jpg", directory / "submitted.jpg"
        for (row, before, after, truth), witness in zip(trial.cases(args, "held"), profile["rows"], strict=True):
            tone.require(row == witness, "The source, JPEG delivery or exact truth mask differs from the freeze")
            earlier.write_bytes(before)
            submitted.write_bytes(after)
            masks = {}

            def capture(path, _record, _reference, _submitted, mask, *, masks=masks, truth=truth):
                masks[path.name] = np.zeros_like(truth) if mask is None else mask

            with patch("app.evidence.image_reference._write_evidence", side_effect=capture):
                disabled = inspect(earlier, submitted, evidence_dir=directory / "baseline", **OPTIONS)
                enabled = inspect(earlier, submitted, evidence_dir=directory / "candidate", allow_structure_jpeg=True, **OPTIONS)
            bank, role = banks[row["source"]], row["role"]
            outcome = {"bank": bank, "role": role, "group": row["source"], "earlier_sha256": row["before_sha256"],
                       "submitted_sha256": row["after_sha256"], "truth_sha256": row["truth_sha256"]}
            for arm, response in (("candidate", enabled), ("baseline", disabled)):
                selected = masks[arm]
                metrics = tone.mask_metrics(selected, truth)
                useful = metrics["iou"] >= .3 and (metrics["recall"] or 0) >= .5
                outcome[arm] = {"useful": useful, "selected": metrics["predicted_positive_pixels"], "schema": response["schema"],
                                "status": response["status"], "mask_sha256": tone.sha(selected.tobytes())}
            if enabled["schema"] == reference_structure_jpeg.SCHEMA:
                tone.require(disabled["status"] != "comparable", "The new fallback replaced an assessed comparison")
            else:
                tone.require(enabled == disabled and np.array_equal(masks["candidate"], masks["baseline"]),
                             "An incumbent result changed without the new fallback")
            rows.append(outcome)
    return rows


def summarize(rows):
    arms = {arm: Counter() for arm in ("baseline", "candidate")}
    for row in rows:
        bank, role = row["bank"], row["role"]
        for arm, counts in arms.items():
            measured = row[arm]
            if role in previous.CONTROLS:
                counts[f"{bank}/ordinary_files"] += 1
                counts[f"{bank}/ordinary_flagged"] += bool(measured["selected"])
            elif role in previous.WRONG:
                counts[f"{bank}/incompatible_pairs"] += 1
                counts[f"{bank}/incompatible_compared"] += measured["status"] == "comparable"
            elif role in ("sharp", "feathered"):
                counts[f"{bank}/{role}_files"] += 1
                counts[f"{bank}/{role}_useful"] += measured["useful"]
    return {arm: dict(totals) for arm, totals in arms.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-roster", "backup-roster", "later-f5", "external", "held", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rows = evaluate(args)
    encoded = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    result = {"schema": SCHEMA, "comparisons": len(rows), "full_route_sha256": tone.sha(encoded.encode()),
              "held_freeze_sha256": tone.file_sha(args.held / "freeze.json"), "results": summarize(rows),
              "fallback_uses": sum(row["candidate"]["schema"] == reference_structure_jpeg.SCHEMA for row in rows)}
    tone.write(args.receipt, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
