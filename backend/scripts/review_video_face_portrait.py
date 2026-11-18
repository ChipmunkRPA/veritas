# SPDX-License-Identifier: Apache-2.0
"""Freeze portrait face-change points and independently recount held intervals."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from app.evidence import face_local_return_binding as local
from scripts import video_common_sources_v89 as io
from scripts.video_face_local_fit import overlap
from scripts.video_face_portrait_panel import DELIVERIES, DIMENSIONS, GENERATORS, GRID, SCHEMA

REPORT = "chipmunk.video-portrait-face-report/1"
METHODS = ("legacy", "strict", "one_group", "two_groups")


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def save(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload)


def validate(bank):
    manifest, header, rows = (bank[key] for key in ("manifest", "header", "rows"))
    expected = {row["file"]: row for row in manifest["rows"]}
    if (header["schema"] != SCHEMA or header["role"] != manifest["role"] or header["size"] != manifest["size"]
            or tuple(header["size"]) not in DIMENSIONS.values()
            or len(expected) != len(rows) or len(rows) != len({row["file"] for row in rows})):
        raise ValueError("Portrait result is missing its declared inputs")
    for row in rows:
        source = expected[row["file"]]
        if (any(row.get(field) != value for field, value in source.items())
                or row["delivery"] not in DELIVERIES or row["first_pts"] != 0
                or Fraction(row["time_base"]) * row["step_pts"] != GRID.step
                or not 0 <= row["face_frames"] <= len(GRID.schedule)):
            raise ValueError("Portrait measurement differs from its delivered input")
        truth = row["truth"]
        if truth is not None:
            if row["kind"] not in GENERATORS or truth != row["treatment"] or len(row["changed_clock"]) != truth[1] - truth[0]:
                raise ValueError("A planted edit differs from its source timing")
            GRID.support(*truth)
        if row["proposal"] is None:
            continue
        candidate = row["proposal"]
        before, inside, after = GRID.support(*candidate["range"])
        if sorted(set(candidate["cuts"])) != candidate["cuts"] or any(
                not before[0] < frame <= after[-1] for frame in candidate["cuts"]):
            raise ValueError("A shot change is outside the exact compared frames")
        score = local.response([float.fromhex(value) for value in candidate["features"]])
        if not math.isfinite(score) or score.hex() != candidate["score"]:
            raise ValueError("Appearance measurements do not reproduce the raw response")
        anchors = {"before": before[-1], "during": inside[len(inside) // 2], "after": after[0]}
        if row["face_frames"] < len(before) + len(inside) + len(after) or set(candidate["anchors"]) != set(anchors):
            raise ValueError("Portrait source locators are incomplete")
        for role, frame in anchors.items():
            index, tick, checksum = candidate["anchors"][role]
            if index != frame or tick != row["first_pts"] + frame * row["step_pts"] or len(checksum) != 64:
                raise ValueError("Portrait source locator differs from the native clock")


def bank(path, journal):
    lines = journal.read_text().splitlines()
    manifest = path / "manifest.json"
    result = {"manifest": io.read(manifest), "header": json.loads(lines[0]), "rows": [json.loads(row) for row in lines[1:]]}
    if result["header"]["manifest_sha256"] != io.sha(manifest):
        raise ValueError("Measured video manifest differs from its input digest")
    validate(result)
    return result


def selected(row, point, guard=True):
    candidate = row["proposal"]
    return (candidate["range"] if candidate and float.fromhex(candidate["score"]) > point
            and (not guard or not candidate["cuts"]) else None)


def counts(rows, point, guard=True):
    tallies = Counter(dict.fromkeys(("edit_files", "ordinary_files", "edit_assessed", "ordinary_assessed",
                                    "located", "misplaced", "ordinary_flags", "guard_vetoes"), 0))
    groups = set()
    for row in rows:
        truth, span = row["truth"], selected(row, point, guard)
        kind = "edit" if truth else "ordinary"
        tallies[f"{kind}_files"] += 1
        tallies[f"{kind}_assessed"] += row["proposal"] is not None
        if truth:
            tallies["located"] += overlap(truth, span) >= .5
            tallies["misplaced"] += span is not None and overlap(truth, span) < .5
        elif span:
            tallies["ordinary_flags"] += 1
            groups.add(row["group"])
        tallies["guard_vetoes"] += bool(guard and span is None and selected(row, point, False))
    tallies["ordinary_groups"] = len(groups)
    return dict(tallies)


def calibrate(bank_doc):
    rows = bank_doc["rows"]
    maxima = {}
    for row in rows:
        if (candidate := row["proposal"]) and not (row["truth"] or candidate["cuts"]):
            group = row["group"]
            score = float.fromhex(candidate["score"])
            maxima[group] = max(score, maxima.get(group, -math.inf))
    if len(maxima) <= 2:
        raise ValueError("Development has too few ordinary groups with measurable portrait comparisons")
    maxima = sorted(maxima.values(), reverse=True)
    return dict(zip(METHODS, (local.POINT, *maxima[:3]), strict=True))


def primary(document, points):
    """Allow one additional ordinary development source only if it finds more edits there."""
    rows = document["rows"]
    strict, wider = (counts(rows, points[name]) for name in ("strict", "one_group"))
    return "one_group" if wider["located"] > strict["located"] else "strict"


def summarize(banks, points):
    return {height: {delivery: {method: counts([row for row in document["rows"] if row["delivery"] == delivery], point)
            | {"before_shot_guard": counts([row for row in document["rows"] if row["delivery"] == delivery], point, False)}
            for method, point in points[height].items()} for delivery in DELIVERIES} for height, document in banks.items()}


def qualify(banks, role):
    if set(banks) != {str(height) for height in DIMENSIONS}:
        raise ValueError("Both portrait sizes must be accounted for")
    for height, document in banks.items():
        validate(document)
        if document["header"]["role"] != role or document["header"]["size"][1] != int(height):
            raise ValueError("Portrait panel has incorrect experimental role")
    first, second = banks.values()
    if (first["manifest"]["roster_sha256"] != second["manifest"]["roster_sha256"]
            or first["manifest"]["sources"] != second["manifest"]["sources"]
            or first["manifest"]["omissions"] != second["manifest"]["omissions"]):
        raise ValueError("Portrait profiles do not share the exact staged source population")


def evaluate(paths, output, frozen=None):
    role = "held" if frozen else "development"
    banks = {}
    for dossier, journal in paths:
        result = bank(Path(dossier), Path(journal))
        banks[str(result["header"]["size"][1])] = result
    qualify(banks, role)
    development = read(frozen) if frozen else None
    points = development["points"] if development else {height: calibrate(item) for height, item in banks.items()}
    methods = development["primary"] if development else {height: primary(doc, points[height]) for height, doc in banks.items()}
    result = {"schema": REPORT, "role": role, "banks": banks, "points": points, "summary": summarize(banks, points),
              "primary": methods,
              "freeze_sha256": io.sha(frozen) if frozen else None, "scoring_sha256": io.sha(Path(__file__))}
    verify(result, frozen)
    save(output, result)
    print_summary(result)


def verify(result, frozen=None):
    qualify(result["banks"], result["role"])
    if result["schema"] != REPORT or (result["role"] == "held") != (frozen is not None):
        raise ValueError("Portrait results require their separate calibration")
    if frozen:
        development = read(frozen)
        verify(development)
        fields = {"role", "manifest_sha256"}
        for height, held in result["banks"].items():
            fit = development["banks"][height]
            if (held["manifest"]["roster_sha256"] != fit["manifest"]["roster_sha256"]
                    or any(held["header"][key] != fit["header"][key] for key in held["header"].keys() - fields)
                    or {row["source_sha256"] for row in held["rows"]} & {row["source_sha256"] for row in fit["rows"]}
                    or {row["group"] for row in held["rows"]} & {row["group"] for row in fit["rows"]}):
                raise ValueError("Held inputs differ from the freeze or reuse development sources")
        if (result["freeze_sha256"] != io.sha(frozen) or development["points"] != result["points"]
                or development["primary"] != result["primary"]):
            raise ValueError("Held operating points differ from the original development comparison")
    elif result["freeze_sha256"] is not None or result["points"] != {
            height: calibrate(item) for height, item in result["banks"].items()} or result["primary"] != {
            height: primary(doc, result["points"][height]) for height, doc in result["banks"].items()}:
        raise ValueError("The development operating points do not reproduce")
    if result["summary"] != summarize(result["banks"], result["points"]):
        raise ValueError("Held portrait results do not reproduce")


def print_summary(result):
    for height, deliveries in result["summary"].items():
        for delivery, variants in deliveries.items():
            for method, row in variants.items():
                print(height, delivery, method, "controls", row["ordinary_files"], "ordinary flags", row["ordinary_flags"],
                      "locations", f'{row["located"]}/{row["edit_files"]}', "cut vetoes", row["guard_vetoes"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    take = commands.add_parser("evaluate")
    take.add_argument("output", type=Path)
    take.add_argument("--panel", nargs=2, action="append", required=True, metavar=("DIRECTORY", "MEASUREMENTS"))
    inspect = commands.add_parser("verify")
    inspect.add_argument("receipt", type=Path)
    for item in (take, inspect):
        item.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.panel, args.output, args.freeze)
    else:
        result = read(args.receipt)
        verify(result, args.freeze)
        print_summary(result)
