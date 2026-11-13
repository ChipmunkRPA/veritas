# SPDX-License-Identifier: Apache-2.0
"""Recount spatial cut evidence and verify selected native source images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def review(data, selection):
    lock, rows, point = data["freeze"], data["held"], data["point"]["point"]
    need(
        lock["schema"] == "chipmunk.video-spatial-cut-context-transfer/1"
        and point in [0.10, 0.20, 0.30]
        and len(rows) == 56
        and len({r["sha256"] for r in rows}) == 56,
        "Incorrect frozen roster or cutoff",
    )
    metrics = recount(rows, point)
    need(metrics == data["metrics"], "Held result differs from its case rows")
    gate = {
        "enough_explanations": metrics["cut_exact"] >= 8,
        "enough_groups": metrics["source_groups_with_cut_explanation"] >= 4,
        "swap_preservation": metrics["wrong_swaps"] == 0,
        "ordinary_preservation": metrics["wrong_ordinary"] <= 1,
        "matched_edges": metrics["cut_wrong"] == 0,
        "coverage": 5 * metrics["cut_measured"] >= 4 * metrics["cut_routed"],
    }
    need(gate == data["gate"] and all(gate.values()) == data["passed"], "Held advancement gate differs")
    if selection:
        need(pin(selection) == data["point"]["selection_sha256"], "Threshold-selection digest differs")
        dev = read(selection)["rows"]
        need(
            len(dev) == 61 and not ({r["sha256"] for r in dev} & {r["sha256"] for r in rows}),
            "Selection overlapped held inputs",
        )
        points = [{"point": p, "metrics": recount(dev, p)} for p in (0.1, 0.2, 0.3)]
        acceptable = [
            p for p in points if not p["metrics"]["wrong_ordinary"] and not p["metrics"]["wrong_swaps"]
        ]
        need(
            points == data["point"]["selection"]
            and point == max(acceptable, key=lambda p: (p["metrics"]["cut_exact"], p["point"]))["point"],
            "Selection did not choose the frozen point",
        )
    return {"passed": all(gate.values()), "metrics": metrics, "gate": gate}


def recount(rows, point):
    metrics, groups = Counter(), set()
    for row in rows:
        scope, known, result = row["status"], row.get("authored_cut"), row.get("result")
        cut = row["condition"] in {"cutaway", "tandem_cutaway", "person_switch", "tandem_person_switch"}
        need(
            not result
            or (
                result["shared_tiles"] >= 5
                and result["sectors"] >= 2
                and result["score"] >= 0.20
                and 5 <= result["interval"][1] - result["interval"][0] <= 24
            ),
            "Invalid native tile or timing evidence",
        )
        found = bool(result and result["score"] >= point)
        exact = bool(
            cut
            and known
            and found
            and all(abs(a - b) <= 1 for a, b in zip(known, result["interval"], strict=True))
        )
        if exact:
            groups.add((row["bank"], row["group"]))
        metrics.update(
            files=1,
            routed=scope != "not_routed",
            measured=scope == "measured",
            cut_files=cut,
            cut_routed=cut and scope != "not_routed",
            cut_measured=cut and scope == "measured",
            cut_exact=exact,
            cut_wrong=found and cut and not exact,
            wrong_swaps=bool(found and row["truth"]),
            wrong_ordinary=found and not (row["truth"] or cut),
            swap_routed=bool(row["truth"]) and scope != "not_routed",
        )
    return {**metrics, "source_groups_with_cut_explanation": len(groups)}


def native(data, folder):
    from app.detectors.video.decode import decode_frames, probe_frames

    manifest = folder / "media.json"
    need(pin(manifest) == data["freeze"]["pins"]["held_media"], "Wrong held source ledger")
    index = {row["sha256"]: row for row in read(manifest)["rows"]}
    checked = Counter()
    for row in data["held"]:
        record = row.get("result")
        if not record or record["score"] < data["point"]["point"]:
            continue
        media = index[row["sha256"]]
        path = folder / media["file"]
        need(pin(path) == row["sha256"], "Wrong held original bytes")
        table = probe_frames(path, timeout_s=15)
        need(
            table.to_json() == media["clock"]
            and all((pts - table.pts[0]) * table.tb == Fraction(i, 30) for i, pts in enumerate(table.pts)),
            "Wrong held original clock",
        )
        a, b = record["interval"]
        anchors = {row["index"]: row for row in record["anchors"]}
        need(
            list(anchors) == [a - 1, a, (a + b) // 2, b - 1, b]
            and record["pts"] == [table.pts[a], table.pts[b]],
            "Wrong boundary timing or representative positions",
        )
        seen = set()
        for frame in decode_frames(path, table, sorted(anchors), timeout_s=20):
            original = anchors[frame.index]
            need(
                frame.pts_verified
                and frame.pts == original["pts"] == table.pts[frame.index]
                and hashlib.sha256(frame.rgb).hexdigest()
                == original["sha256"]
                == media["rgb_sha256"][frame.index],
                "Wrong original frame pixels",
            )
            seen.add(frame.index)
            checked["frames"] += 1
        need(seen == anchors.keys(), "One or more native roles are missing")
        checked["files"] += 1
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    document = read(args.receipt)
    result = review(document, args.selection)
    if args.native:
        result["native"] = native(document, args.native)
    print(json.dumps(result, sort_keys=True, indent=2))
