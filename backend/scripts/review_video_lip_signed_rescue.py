"""Bind every public lip-sync label to the plan frozen before measurement."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import video_lip_signed_rescue as measured


def review(plan_path, calibration_path, held_path):
    plan = measured.read(plan_path)
    calibration = measured.read(calibration_path)
    held = measured.read(held_path)
    records = {row["speaker"]: row for row in plan["rows"]}
    if len(records) != len(plan["rows"]):
        raise ValueError("The held speaker and calibration speaker allocations overlap")
    for report in (calibration, held):
        for row in report["rows"]:
            source = records[row["speaker"]]
            kind = row["input"]
            truth = source["interval"] if kind.startswith("local_") else None
            shift = (-source["shift_samples"] if kind == "local_early" else source["shift_samples"]) if truth else None
            if report["phase"] != source["role"] or (row["truth"], row["shift_samples"]) != (truth, shift):
                raise ValueError("A delivered clip was relabeled after the plan froze")
    measured.review(plan_path, calibration_path, held_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plan_path", "calibration_path", "held_path"):
        parser.add_argument(field, type=Path)
    review(**vars(parser.parse_args()))
