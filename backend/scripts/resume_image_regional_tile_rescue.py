# SPDX-License-Identifier: Apache-2.0
"""Finish and replay the frozen two-bank JPEG regional tile comparison."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts import image_regional_tile_rescue_transfer as trial
from scripts.localization_benchmark import mask_metrics

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "docs/evaluation-panels/image-regional-tile-rescue-main-bank-2026-11-13.json"
FROZEN_SHA256 = "bd73686c7befc6c21a63078ced142cf8a9499533ddc57354341edb55141436d3"
MAIN_JOURNAL_SHA256 = "6df93114c566dc8276a49658f384dc07fd96d10d408c1b3a3d5a52cf79267e83"
MAIN_ROWS = 136
read, write, require, sha, file_sha = trial.read, trial.write, trial.require, trial.sha, trial.file_sha


def evidence_rows(path, frozen):
    raw = (path / "rows.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    require(all(line.endswith(b"\n") for line in lines), "A partial journal line cannot become a trial result")
    require(sha(b"".join(lines[:MAIN_ROWS])) == MAIN_JOURNAL_SHA256, "Historical main-bank bytes differ")
    rows = [json.loads(line) for line in lines]
    require(MAIN_ROWS <= len(rows) <= len(frozen["rows"]), "Journal falls outside the frozen input roster")
    for index, row in enumerate(rows):
        require(all(row[k] == v for k, v in frozen["rows"][index].items()), "Journal ordering or binding differs")
        require(file_sha(path / "masks" / f"{index:03}.bits") == row["mask_sha256"], "Saved native masks differ")
    return rows


def resume(args, frozen, rows):
    start = len(rows)
    if start == len(frozen["rows"]):
        return
    model = trial.regional.load_model(trial.previous.CHECKPOINT)
    with (args.output / "rows.jsonl").open("a") as journal:
        pairs = zip(islice(trial.material(args), start, None), frozen["rows"][start:], strict=True)
        for index, ((key, body, truth, _), expected) in enumerate(pairs, start):
            require(key["panel"] == "external" and trial.witness(key, body, truth) == expected,
                    "Resumed delivery or truth differs from its original freeze")
            masks, trace = trial.readout(model, body)
            packed = b"".join(np.packbits(mask, axis=None).tobytes() for mask in masks)
            mask_path = args.output / "masks" / f"{index:03}.bits"
            with mask_path.open("wb") as mask_file:
                mask_file.write(packed)
            row = expected | trace | {"mask_sha256": sha(packed), "measures": {
                arm: mask_metrics(mask, truth) for arm, mask in zip(trial.ARMS, masks, strict=True)}}
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            if (index + 1) % 8 == 0:
                print(index + 1, "/", len(frozen["rows"]), flush=True)


def public_rows(rows):
    historical = read(HISTORICAL)["anonymous_pixel_counts"]
    prior_groups = {row["group"]: record["group"] for row, record in zip(rows[:MAIN_ROWS], historical, strict=True)}
    new_groups = {group: i for i, group in enumerate(sorted({r["group"] for r in rows[MAIN_ROWS:]}), 16)}
    require(len(prior_groups) == 15 and len(new_groups) == 7 and not prior_groups.keys() & new_groups.keys(),
            "Anonymized receipt groups must remain disjoint")
    groups = prior_groups | new_groups
    result = []
    for i, row in enumerate(rows, 1):
        metrics = row["measures"]
        result.append({"case": i, "panel": row["panel"], "group": groups[row["group"]],
                       "role": row["role"], "family": row["family"], "tile_forwards": len(row["tiles"]),
                       "image_pixels": metrics["native"]["pixels"],
                       "pixels": {arm: [m["reference_positive_pixels"], m["predicted_positive_pixels"], m["intersection"]]
                                  for arm, m in metrics.items()}})
    require([dict((k, v) for k, v in row.items() if k != "panel") for row in result[:MAIN_ROWS]] == historical,
            "Public main-bank rows differ from the historical receipt")
    return result


def review(args, frozen, rows):
    from scripts.review_image_regional_tile_rescue_main import verify_full

    require(len(rows) == len(frozen["rows"]), "A partial second bank cannot produce a completed receipt")
    require(trial.audit(args, frozen) == rows, "Exact two-bank replay differs")
    result = {k: v for k, v in frozen.items() if k != "rows"} | trial.summary(rows) | {
        "schema": "chipmunk.image-regional-tile-rescue-complete/1",
        "view_count": len(rows), "exact_source_mask_replays": len(rows),
        "source_roles": {"retained_b": MAIN_ROWS, "external": len(rows) - MAIN_ROWS},
        "freeze_sha256": FROZEN_SHA256,
        "rows_sha256": file_sha(args.output / "rows.jsonl"),
        "historical_main_journal_sha256": MAIN_JOURNAL_SHA256,
        "historical_main_receipt_sha256": file_sha(HISTORICAL),
        "recovery_code_sha256": file_sha(Path(__file__)),
        "anonymous_pixel_counts": public_rows(rows),
    }
    verify_full(result)
    write(args.publication, result)
    print(result["paired"], "pass", result["passes"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("resume", "review"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint",
                 "output-fresh", "external", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    trial.smoke()
    path = args.output / "freeze.json"
    require(file_sha(path) == FROZEN_SHA256, "Original prospective freeze differs")
    frozen = read(path)
    require(trial.profile(args) == frozen, "Frozen study implementation or inputs differ")
    rows = evidence_rows(args.output, frozen)
    if args.stage == "resume":
        print("verified", len(rows), "retained rows before resume", flush=True)
        resume(args, frozen, rows)
    else:
        require(args.publication is not None, "Completed receipt needs a publication path")
        review(args, frozen, rows)


if __name__ == "__main__":
    main()
