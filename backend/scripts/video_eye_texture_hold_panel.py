# SPDX-License-Identifier: MIT
"""Compare normalized eye texture prospectively on a distinct camera roster."""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from scripts import video_eye_hold_panel as first
from scripts.video_cropped_reverse_panel import load, save, sha

POINT = .115
GUARD = True
SCHEMA = "chipmunk.eye-texture-hold-evaluation/1"
DEV = {"files": 27, "edits": 9, "controls": 18, "points": [.045, .075, .115],
       "useful": [2, 2, 5], "misplaced": [0, 0, 1], "control_flags": [0, 0, 0]}


def _pins():
    folder = Path(__file__).parent
    return first._pins() | {name: sha(folder / name) for name in
                            ("video_eye_texture_hold.py", "video_eye_texture_hold_panel.py")}


def _distinct(new, prior):
    for field in ("group", "capture_key_sha256"):
        assert not ({row[field] for row in new} & {row[field] for row in prior})
    assert not ({row["files"]["ordinary"]["sha256"] for row in new}
                & {row["files"]["ordinary"]["sha256"] for row in prior})


def freeze(source_roster, sources, earlier_receipt, output):
    earlier = load(earlier_receipt)
    _distinct(load(source_roster)["groups"], earlier["roster"]["groups"])
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        base = Path(scratch) / "geometry-plan.json"
        first.freeze(source_roster, sources, base)
        plan = load(base)
    save(output, plan | {"focus": SCHEMA, "point": POINT, "continuity_guard": GUARD,
                         "development": DEV, "method_sha256": _pins(), "earlier_receipt_sha256": sha(earlier_receipt)})
    print("Frozen texture test", sha(output), flush=True)


def _measure(path, detector):
    from app.detectors.video import mouth_local_hold
    from scripts import video_eye_texture_hold as comparator

    earlier = first._measure(path, detector)
    table, rgb = mouth_local_hold.native(path)
    started = time.monotonic()
    status, match = comparator.observe(rgb, detector, POINT, guard=GUARD)
    elapsed = round(time.monotonic() - started, 4)
    return earlier | {"first_eye_candidate": earlier["candidate"], "status": status, "candidate": match,
                      "elapsed_s": elapsed, "verified_anchors": first._anchors(path, table, rgb, match) if match else {}}


def _earlier_counts(rows):
    return {
        "located": sum(first._matches(row["first_eye_candidate"], row["truth"]) for row in rows),
        "misplaced": sum(bool(row["truth"] and row["first_eye_candidate"]
                              and not first._matches(row["first_eye_candidate"], row["truth"])) for row in rows),
        "control_flags": sum(bool(row["first_eye_candidate"] and not row["truth"]) for row in rows),
    }


def score(plan_path, sources, output):
    import cv2

    cv2.setNumThreads(1)
    plan = load(plan_path)
    if (plan["focus"] != SCHEMA or plan["method_sha256"] != _pins() or plan["point"] != POINT
            or plan["continuity_guard"] != GUARD or plan["development"] != DEV):
        raise ValueError("The frozen eye texture rule changed")
    detector, weight = first._locator()
    if weight != plan["geometry_weight_sha256"]:
        raise ValueError("The pinned face locator changed")
    rows = []
    expected = [row for row in plan["cases"] if row["split"] == "held"]
    for group in (row for row in plan["roster"]["groups"] if row["role"] == "held"):
        asset = sources / f"{group['index']:02d}-ordinary.mp4"
        if sha(asset) != group["files"]["ordinary"]["sha256"]:
            raise ValueError("The pinned original changed")
        prepared = first._prepare(asset, group["offset"])
        with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
            for case in (row for row in expected if row["index"] == group["index"]):
                path = Path(scratch) / case["delivery"]
                first._encode(prepared, case["role"], group["eye_boxes"], path)
                rows.append(case | _measure(path, detector))
                path.unlink()
        save(output, plan | {"rows": rows, "split": "held", "complete": len(rows) == len(expected),
                             "plan_sha256": sha(plan_path), "summary": first.summary(rows),
                             "first_eye_summary": _earlier_counts(rows)})
        print("Held texture", len(rows), "/", len(expected), flush=True)


def verify(receipt, earlier):
    value = load(receipt)
    assert value["focus"] == SCHEMA and value["point"] == POINT and value["continuity_guard"] == GUARD
    assert value["gate"] == first.GATE and value["development"] == DEV
    assert value["earlier_receipt_sha256"] == sha(earlier)
    _distinct(value["roster"]["groups"], load(earlier)["roster"]["groups"])
    assert value["first_eye_summary"] == _earlier_counts(value["rows"])
    first.verify(receipt)
    print("First eye candidate on the same deliveries", value["first_eye_summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prescribe = commands.add_parser("freeze")
    prescribe.add_argument("roster", type=Path)
    prescribe.add_argument("sources", type=Path)
    prescribe.add_argument("earlier_receipt", type=Path)
    prescribe.add_argument("plan", type=Path)
    measure = commands.add_parser("score")
    measure.add_argument("plan", type=Path)
    measure.add_argument("sources", type=Path)
    measure.add_argument("receipt", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("receipt", type=Path)
    check.add_argument("earlier_receipt", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.roster, args.sources, args.earlier_receipt, args.plan)
    elif args.action == "score":
        score(args.plan, args.sources, args.receipt)
    else:
        verify(args.receipt, args.earlier_receipt)


if __name__ == "__main__":
    main()
