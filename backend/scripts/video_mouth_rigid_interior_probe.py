# SPDX-License-Identifier: Apache-2.0
"""Measure a separately frozen interior-landmark scope on the consumed visual panel."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

import cv2

from scripts import video_mouth_rigid_coupling as prior
from scripts.video_common_sources_v89 import read, require, sha, write

EXCLUDED_POINT = 10
SCOPE = "Exploratory remeasurement on the consumed source panel. Omit the single forehead point found outside the picture in tightly framed originals. Keep clock, movement, thresholds and gate unchanged."


def freeze(previous):
    frozen = read(previous / "freeze.json")
    require(frozen["worker_sha256"] == sha(Path(prior.__file__)), "Original frozen implementation differs")
    require(
        read(previous / "held.json")["freeze_sha256"] == sha(previous / "freeze.json"),
        "Original held freeze differs",
    )
    return {
        "scope": SCOPE,
        "omitted_point": EXCLUDED_POINT,
        "original_freeze_sha256": sha(previous / "freeze.json"),
        "original_held_sha256": sha(previous / "held.json"),
        "original_point_sha256": sha(previous / "calibration.json"),
        "tool_sha256": sha(Path(__file__)),
    }


def run(previous, dest):
    require(read(dest / "freeze.json") == freeze(previous), "Interior-only scope differs from its freeze")
    priors = read(previous / "freeze.json")
    prior.UPPER = tuple(i for i in prior.UPPER if i != EXCLUDED_POINT)
    prior.ALL_POINTS = (*prior.UPPER, *prior.LOWER_LIP, *prior.UPPER_LIP, *prior.JAW)
    cv2.setNumThreads(1)
    module, task = prior.geometry.mp_task()
    rows = []
    recent = None
    with closing(task):
        for case in priors["cases"]:
            if case["split"] != "held":
                continue
            if recent != case["key"][:2]:
                recent, cached = case["key"][:2], {}
            measured = prior.observe(case, module, task, cached)
            result = {arm: prior.detect(measured, arm) for arm in prior.ARMS}
            rows.append(
                {
                    "key": case["key"],
                    "group": case["group"],
                    "truth": case["truth"],
                    "event_aligned": case.get("event_aligned", False),
                    "native": measured,
                    "results": result,
                }
            )
            print(
                case["key"],
                measured.get("geometry_frames"),
                {k: v["measured"] for k, v in result.items()},
                flush=True,
            )
    write(dest / "held.json", {"freeze_sha256": sha(dest / "freeze.json"), "rows": rows})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "run"))
    p.add_argument("original", type=Path)
    p.add_argument("probe", type=Path)
    a = p.parse_args()
    if a.phase == "freeze":
        lock = freeze(a.original)
        a.probe.mkdir(mode=0o700)
        write(a.probe / "freeze.json", lock)
    else:
        run(a.original, a.probe)


if __name__ == "__main__":
    main()
