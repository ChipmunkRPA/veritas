# SPDX-License-Identifier: Apache-2.0
"""Compare bounded windows and incumbent findings against exact source edit ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

EXPECTED_ROLES = (
    "genuine",
    "whole_loop",
    "jump",
    "light",
    "early_swap",
    "late_swap",
    "tandem_late",
    "two_swaps",
    "genuine_cut",
)
LATE_ROLES = ("late_swap", "tandem_late", "two_swaps")
WINDOW_STARTS = (0, 60, 120)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def iou(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def check_witness(row, window):
    observed = window["match"]
    if not observed:
        return
    start = window["window_start"]
    a, b = observed["frames"]
    if not (
        start + 6 <= a < b <= start + 114
        and 5 <= b - a <= 7
        and observed["response"] > 0.10
        and a in window["edges"]
        and b in window["edges"]
        and len(observed["anchors"]) == 3
    ):
        raise ValueError("Proposal differs from native window and appearance response")
    indices = [item["frame_index"] for item in observed["anchors"]]
    if not (indices[0] == a - 1 and a <= indices[1] < b and indices[2] == b):
        raise ValueError("Before, during, and after anchors differ from the proposed interval")
    for idx, frame in zip(indices, observed["anchors"], strict=True):
        if (
            (frame["pts"] - row["first_pts"]) * Fraction(row["time_base"]) != Fraction(idx, 30)
            or frame["decoded_rgb_sha256"] != row["anchor_rgb"][str(idx)]
            or any(len(bytes.fromhex(frame[k])) != 32 for k in ("decoded_rgb_sha256", "aligned_rgb_sha256"))
        ):
            raise ValueError("Source proof does not match a delivered frame")


def result(rows):
    output = {arm: Counter() for arm in ("current", "prefix", "overlap")}
    roles = {arm: {role: Counter() for role in EXPECTED_ROLES} for arm in output}
    recovered = set()
    lost_prefix = 0
    maximum_native_face_reads = 0
    for row in rows:
        if (
            row["native_count"] != 240
            or (row["last_pts"] - row["first_pts"]) * Fraction(row["time_base"]) != Fraction(239, 30)
            or any(len(bytes.fromhex(row[key])) != 32 for key in ("native_rgb_sha256", "native_pts_sha256"))
            or len({w["window_start"] for w in row["windows"]}) != 3
        ):
            raise ValueError("Incomplete continuous native timeline")
        if tuple(w["window_start"] for w in row["windows"]) != WINDOW_STARTS:
            raise ValueError("Window plan differs")
        if not 18 <= row["face_inputs"] <= 3 * 84:
            raise ValueError("Recorded face reads exceed the comparison plan")
        maximum_native_face_reads = max(maximum_native_face_reads, row["face_inputs"])
        old = row["current"]
        rolling = []
        for window in row["windows"]:
            check_witness(row, window)
            if (match := window["match"]) is not None and not any(
                iou(match["frames"], earlier) >= Fraction(1, 2) for earlier in old + rolling
            ):
                rolling.append(match["frames"])
        first = row["windows"][0]["match"]
        prefix = (
            [first["frames"]]
            if first and not any(iou(first["frames"], earlier) >= Fraction(1, 2) for earlier in old)
            else []
        )
        late = next((span for span in row["truth"] if span[0] >= 120), None)
        lost_prefix += sum(
            any(iou(truth, found) >= Fraction(1, 2) for found in old + prefix)
            and not any(iou(truth, found) >= Fraction(1, 2) for found in old + rolling)
            for truth in row["truth"]
        )
        if (
            late
            and any(iou(late, range_) >= Fraction(1, 2) for range_ in rolling)
            and not any(iou(late, range_) >= Fraction(1, 2) for range_ in old + prefix)
        ):
            recovered.add(row["group"])
        for arm, ranges in (("current", old), ("prefix", old + prefix), ("overlap", old + rolling)):
            totals = roles[arm][row["role"]]
            totals["files"] += 1
            totals["expected"] += len(row["truth"])
            totals["assessed"] += int(
                arm == "current"
                and row["current_assessed"]
                or arm == "prefix"
                and row["windows"][0]["assessed"]
                or arm == "overlap"
                and all(w["assessed"] for w in row["windows"])
            )
            for expected in row["truth"]:
                totals["found"] += int(any(iou(expected, range_) >= Fraction(1, 2) for range_ in ranges))
                totals["exact"] += int(expected in ranges)
                if expected[0] >= 120:
                    totals["late_found"] += int(
                        any(iou(expected, range_) >= Fraction(1, 2) for range_ in ranges)
                    )
                    totals["late_exact"] += int(expected in ranges)
            if not row["truth"]:
                totals["ordinary_flagged"] += bool(ranges)
                totals["added_ordinary"] += bool(ranges and not old)
                if row["role"] == "genuine_cut":
                    totals["added_cut"] += bool(ranges and not old)
            else:
                totals["wrong_proposed_ranges"] += sum(
                    not any(iou(found, expected) >= Fraction(1, 2) for expected in row["truth"])
                    for found in ranges[len(old) :]
                )
    for arm, mapping in roles.items():
        for counters in mapping.values():
            output[arm].update(counters)
    return {
        "total": {k: dict(v) for k, v in output.items()},
        "by_role": {k: {name: dict(c) for name, c in v.items()} for k, v in roles.items()},
        "recovered_late_groups": len(recovered),
        "maximum_native_face_reads": maximum_native_face_reads,
        "lost_prefix_ranges": lost_prefix,
    }


def advancement(metrics, gate):
    totals = metrics["total"]
    current, prefix, overlap = (totals[k] for k in ("current", "prefix", "overlap"))
    return {
        "late_recovered": overlap.get("late_found", 0) - prefix.get("late_found", 0)
        >= gate["minimum_recovered_late"],
        "groups": metrics["recovered_late_groups"] >= gate["minimum_groups"],
        "exact": overlap.get("late_exact", 0) - current.get("late_exact", 0) >= gate["minimum_exact"],
        "ordinary": overlap.get("added_ordinary", 0) <= gate["max_new_ordinary"],
        "cuts": overlap.get("added_cut", 0) <= gate["max_new_cut"],
        "coverage": overlap["assessed"] / overlap["files"] >= gate["minimum_measured"],
        "preserves_prefix": metrics["lost_prefix_ranges"] <= gate["maximum_lost_first_window"],
    }


def assemble(folder, target):
    frozen = read(folder / "freeze.json")
    publication = {
        "schema": "chipmunk.video-face-short-prefix-public/1",
        "freeze_sha256": sha(folder / "freeze.json"),
        "method": frozen["method"],
        "fixed_point": frozen["existing_point"],
        "gate": frozen["gate"],
        "panels": [],
    }
    for phase in ("development", "held"):
        raw = read(folder / f"{phase}.json")
        incumbent = read(folder / f"{phase}-incumbent.json")
        if raw["freeze_sha256"] != publication["freeze_sha256"] or incumbent["scored_panel_sha256"] != sha(
            folder / f"{phase}.json"
        ):
            raise ValueError("Measured sources differ from their fixed receipts")
        rows = []
        for measured, old in zip(raw["rows"], incumbent["rows"], strict=True):
            if any(measured[k] != old[k] for k in ("group", "role", "sha256")):
                raise ValueError("Incumbent was not applied to the identical media")
            pts, time_base = (measured["clock"][key] for key in ("pts", "time_base"))
            rgb = measured["native_hashes"]
            if (
                len(pts) != 240
                or len(rgb) != 240
                or any((p - pts[0]) * Fraction(time_base) != Fraction(i, 30) for i, p in enumerate(pts))
            ):
                raise ValueError("The native ledger must bind all source pictures")
            anchors = {
                a["frame_index"]
                for window in measured["windows"]
                if window["match"]
                for a in window["match"]["anchors"]
            }
            rows.append(
                {
                    **{
                        k: measured[k]
                        for k in ("group", "role", "sha256", "truth", "windows", "face_inputs", "elapsed")
                    },
                    "first_pts": pts[0],
                    "last_pts": pts[-1],
                    "native_count": len(pts),
                    "time_base": time_base,
                    "native_pts_sha256": hashlib.sha256(
                        json.dumps(pts, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "native_rgb_sha256": hashlib.sha256(b"".join(bytes.fromhex(p) for p in rgb)).hexdigest(),
                    "anchor_rgb": {str(i): rgb[i] for i in sorted(anchors)},
                    "current": old["ranges"],
                    "current_assessed": old["track_status"] == "ok",
                    "current_tracks": old["tracks"],
                }
            )
        expected = {(i, role) for i in range(1, len(frozen[phase]) + 1) for role in EXPECTED_ROLES}
        if {(r["group"], r["role"]) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError("The matched source roster is incomplete")
        metrics = result(rows)
        publication["panels"].append(
            {
                "role": phase,
                "results": metrics,
                "advancement": advancement(metrics, frozen["gate"]),
                "rows": rows,
            }
        )
    target.write_text(json.dumps(publication, indent=2, allow_nan=False, sort_keys=True) + "\n")
    print(json.dumps({p["role"]: p["results"]["total"] for p in publication["panels"]}, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--assemble", type=Path)
    a = p.parse_args()
    if a.assemble:
        assemble(a.assemble, a.path)
        return
    receipt = read(a.path)
    for panel in receipt["panels"]:
        if (
            result(panel["rows"]) != panel["results"]
            or advancement(panel["results"], receipt["gate"]) != panel["advancement"]
        ):
            raise ValueError("Independent count differs from the report")
    print(json.dumps({p["role"]: p["results"]["total"] for p in receipt["panels"]}, sort_keys=True))


if __name__ == "__main__":
    main()
