# SPDX-License-Identifier: Apache-2.0
"""Score another detector's single face-transition calls on the published panel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.single-face-boundary-predictions/1"
PANEL_SCHEMA = "chipmunk.video-face-single-boundary/1"
STATUSES = {"localized", "no_match", "unassessed"}
SHA256 = re.compile("[a-f0-9]{64}")
FRAME_COUNT = 100


def panel(path, split):
    payload = path.read_bytes()
    document = json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)
    if document.get("schema") != PANEL_SCHEMA:
        raise ValueError("This comparison needs the published single-transition face panel")
    rows = document["rows"] if split == "held" else document["development_rows"]
    indexed = {}
    for row in rows:
        digest, expected = row["input_sha256"], row["authored_edge"]
        if (not isinstance(digest, str) or not SHA256.fullmatch(digest) or digest in indexed or row["pictures"] != FRAME_COUNT
                or (row["edited_side"] is not None and expected is None)
                or expected is not None and (type(expected) is not int or not 1 <= expected < FRAME_COUNT)):
            raise ValueError("The panel has ambiguous input identities or invalid authored frame labels")
        indexed[digest] = row
    if not indexed:
        raise ValueError("The requested panel has no cases")
    return hashlib.sha256(payload).hexdigest(), document, indexed


def prepare(source, split):
    digest, _, cases = panel(source, split)
    return {"schema": SCHEMA, "panel_sha256": digest, "split": split,
            "predictions": [{"sha256": name, "status": None, "frame": None} for name in sorted(cases)]}


def validate_predictions(document, expected_hash, expected_cases, split):
    if (set(document) != {"schema", "panel_sha256", "split", "predictions"} or document["schema"] != SCHEMA
            or document["panel_sha256"] != expected_hash or document["split"] != split):
        raise ValueError("The predictions refer to a different panel or split")
    predictions = document["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list of exact-file results")
    rows = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {"sha256", "status", "frame"}:
            raise ValueError("Each prediction needs only SHA-256, status and frame")
        digest, status, frame = row["sha256"], row["status"], row["frame"]
        if (not isinstance(digest, str) or digest in rows or digest not in expected_cases
                or not isinstance(status, str) or status not in STATUSES
                or (not (type(frame) is int and 1 <= frame < FRAME_COUNT) if status == "localized" else frame is not None)):
            raise ValueError("Invalid, unrecognized or repeated prediction")
        rows[digest] = row
    if rows.keys() != expected_cases.keys():
        raise ValueError("Provide an explicit result for each exact input, including unassessed cases")
    return rows


def score(source, predictions, split):
    digest, document, cases = panel(source, split)
    comparator = validate_predictions(predictions, digest, cases, split)
    arm = document["thresholds"]["selected"]
    point = document["thresholds"]["options"][arm]["point"]
    tolerance = document["frozen_context"]["maximum_boundary_error"]
    total, by_delivery, by_control = Counter(), defaultdict(Counter), defaultdict(Counter)
    gain_groups, loss_groups, ordinary_alert_groups = set(), set(), set()
    for name, row in cases.items():
        truth, group = row["authored_edge"] if row["edited_side"] else None, row["group"]
        other = comparator[name]
        frame = other["frame"]
        reference = row["candidates"][arm]
        reference_frame = reference["boundary"] if reference is not None and reference[arm] > point else None
        if truth is None:
            alert, prior = frame is not None, reference_frame is not None
            result = {"ordinary": 1, "ordinary_alerts": alert, "new_ordinary_alerts": alert and not prior,
                      "removed_ordinary_alerts": prior and not alert, "unassessed_ordinary": other["status"] == "unassessed"}
            if alert:
                ordinary_alert_groups.add(group)
            by_control[row["role"]].update(result)
        else:
            located = frame is not None and abs(frame - truth) <= tolerance
            prior = reference_frame is not None and abs(reference_frame - truth) <= tolerance
            result = {"edited": 1, "located": located, "exact": frame == truth, "missed": not located,
                      "misplaced": frame is not None and not located, "unassessed_edits": other["status"] == "unassessed",
                      "added_locations": located and not prior, "lost_locations": prior and not located}
            if located and not prior:
                gain_groups.add(group)
            if prior and not located:
                loss_groups.add(group)
        total.update(result)
        by_delivery[row["encoding"]].update(result)
    return {"split": split, "panel_sha256": digest, "cases": len(cases), "boundary_tolerance_frames": tolerance,
            "total": dict(total), "added_source_groups": len(gain_groups), "lost_source_groups": len(loss_groups),
            "ordinary_alert_source_groups": len(ordinary_alert_groups),
            "by_delivery": {key: dict(values) for key, values in by_delivery.items()},
            "by_control": {key: dict(values) for key, values in by_control.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("prepare", help="Emit a prediction template without source labels")
    compare = commands.add_parser("score", help="Score exact-file predictions against the test labels")
    for command in (template, compare):
        command.add_argument("panel", type=Path)
        command.add_argument("--split", choices=("development", "held"), default="held")
    compare.add_argument("predictions", type=Path)
    args = parser.parse_args()
    output = (prepare(args.panel, args.split) if args.command == "prepare"
              else score(args.panel, json.loads(args.predictions.read_bytes()), args.split))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
