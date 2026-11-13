# SPDX-License-Identifier: Apache-2.0
"""Recompute native lip coupling decisions and replay a sanitized evaluation ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

ARMS = ("jaw", "head")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def correlation(rows, arm):
    a = [row[0] if arm == "jaw" else abs(row[0]) for row in rows]
    b = [row[1] if arm == "jaw" else row[2] for row in rows]
    ma, mb = math.fsum(a) / len(a), math.fsum(b) / len(b)
    da, db = [x - ma for x in a], [y - mb for y in b]
    va, vb = math.fsum(x * x for x in da), math.fsum(y * y for y in db)
    if va < len(a) * 0.005**2 or vb < len(b) * (0.0025 if arm == "jaw" else 0.002) ** 2:
        return None
    return math.fsum(x * y for x, y in zip(da, db, strict=True)) / math.sqrt(va * vb)


def redo(record, arm):
    if record["status"] != "observed":
        return False, None
    edges, anchors = record["edges"], record["anchors"]
    cache = {}

    def selected(a, b):
        if (a, b) not in cache:
            usable = [r for r in edges[a:b] if r is not None]
            cache[a, b] = (
                correlation(usable, arm) if len(usable) >= 7 and len(usable) * 5 >= (b - a) * 4 else None
            )
        return cache[a, b]

    measured, best = False, None
    for width in (12, 16, 20):
        for a in range(9, len(edges) - width - 8):
            b = a + width
            values = selected(a - 9, a), selected(a, b), selected(b, b + 9)
            if None in values:
                continue
            measured = True
            left, middle, right = values
            score = min(left, right) - middle
            if (
                min(left, right) >= 0.25
                and score > 0
                and (best is None or score > best[0] + 1e-12)
                and all(anchors[i] is not None for i in (a - 1, (a + b) // 2, b))
            ):
                best = score, [a, b]
    return measured, best


def iou(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def totals(rows, points):
    result = {}
    for arm in ARMS:
        groups = set()
        counts = {}
        ordinary_flags, original_flags, useful = 0, 0, 0
        for row in rows:
            method = row["arms"][arm]
            counter = counts.setdefault(row["family"] + ":" + row["role"], Counter())
            measured, score, span = (method[k] for k in ("measured", "score", "frames"))
            require(
                (score is None) == (span is None) and (score is None or measured), "Invalid support or score"
            )
            flag = score is not None and score > points[arm]
            localized = flag and bool(row["truth"]) and iou(span, row["truth"]) >= Fraction(1, 2)
            counter.update({"cases": 1, "measured": measured, "flags": flag, "localizations": localized})
            if row["event_aligned"]:
                counter.update(
                    {"aligned_cases": 1, "aligned_measured": measured, "aligned_localizations": localized}
                )
                if localized:
                    groups.add(row["group"])
                    useful += 1
            if flag and row["truth"] is None:
                ordinary_flags += 1
                original_flags += row["role"] == "original"
        support = Counter()
        for row in rows:
            support[(row["family"], "cases")] += 1
            support[(row["family"], "measured")] += row["arms"][arm]["measured"]
        families = {row["family"] for row in rows}
        result[arm] = {
            "roles": counts,
            "source_groups_localized": len(groups),
            "ordinary_flags": ordinary_flags,
            "original_flags": original_flags,
            "aligned_localizations": useful,
            "gate": {
                "aligned_localizations": useful >= 4,
                "groups": len(groups) >= 3,
                "coverage": all(2 * support[f, "measured"] >= support[f, "cases"] for f in families),
                "ordinary": ordinary_flags <= 2 and original_flags == 0,
            },
        }
    return result


def report(run):
    freeze, dev, held, point = (run / (k + ".json") for k in ("freeze", "development", "held", "calibration"))
    protocol, points = read(freeze), read(point)
    require(points["development_sha256"] == digest(dev), "Frozen point changed")
    expected = {tuple(case["key"]): case for case in protocol["cases"]}
    result = {
        "schema": "chipmunk.video-mouth-rigid-coupling/1",
        "protocol": protocol["protocol"],
        "freeze_sha256": digest(freeze),
        "development_sha256": digest(dev),
        "held_sha256": digest(held),
        "points": points["points"],
        "splits": {},
    }
    audit_path = run / "source-audit.json"
    audit = read(audit_path)
    require(audit["held_sha256"] == digest(held), "Source evidence audit differs")
    result["source_audit"] = {"ledger_sha256": digest(audit_path), **audit}
    for split, path in (("development", dev), ("held", held)):
        contents = read(path)
        require(contents["freeze_sha256"] == digest(freeze), "Result uses a different freeze")
        source, groups, rows = {}, {}, []
        n_pts, n_geometry, n_admitted = 0, 0, 0
        for r in contents["rows"]:
            case = expected[tuple(r["key"])]
            require(
                case["split"] == split and r["truth"] == case["truth"] and r["group"] == case["group"],
                "Incorrect frozen case",
            )
            n_pts += len(r["native"]["picture_pts"])
            n_geometry += r["native"].get("geometry_frames", 0)
            n_admitted += r["native"]["status"] == "observed"
            out = {
                "source": source.setdefault(tuple(r["key"][:2]), len(source) + 1),
                "group": groups.setdefault(r["group"], len(groups) + 1),
                "family": r["key"][0],
                "role": r["key"][2],
                "truth": r["truth"],
                "event_aligned": r["event_aligned"],
                "clock_admitted": r["native"]["status"] == "observed",
                "arms": {},
            }
            for arm, candidate in r["results"].items():
                measured, selected = redo(r["native"], arm)
                p = candidate["proposal"]
                require(
                    measured == candidate["measured"] and (selected is None) == (p is None),
                    "Independent decision differs",
                )
                require(
                    p is None or (abs(selected[0] - p["score"]) < 1e-10 and selected[1] == p["frames"]),
                    "Independent localization differs",
                )
                out["arms"][arm] = {
                    "measured": measured,
                    "score": p["score"] if p else None,
                    "frames": p["frames"] if p else None,
                }
            rows.append(out)
        require(len(rows) == (10 if split == "development" else 92), "Incomplete scored split")
        result["splits"][split] = {
            "rows": rows,
            "native_picture_ticks": n_pts,
            "landmark_observations": n_geometry,
            "clock_admitted": n_admitted,
            "summary": totals(rows, result["points"]),
        }
    for arm in ARMS:
        ordinary = [r["arms"][arm]["score"] for r in result["splits"]["development"]["rows"]]
        require(
            points["points"][arm]
            == max(0.18, max((x for x in ordinary if x is not None), default=0) + 0.025),
            "Point does not replay",
        )
    require(
        sum(arm["ordinary_flags"] for arm in result["splits"]["held"]["summary"].values())
        == audit["verified_flag_ranges"]
        and audit["verified_native_flag_crops"] == 3 * audit["verified_flag_ranges"],
        "Exact picture audit differs from held range counts",
    )
    return result


def summarize_probe(result, run, probe):
    original = result["splits"]["held"]["rows"]
    evidence, freeze = (read(probe / f"{k}.json") for k in ("held", "freeze"))
    require(
        evidence["freeze_sha256"] == digest(probe / "freeze.json")
        and freeze["original_freeze_sha256"] == result["freeze_sha256"]
        and freeze["original_held_sha256"] == result["held_sha256"]
        and freeze["original_point_sha256"] == digest(run / "calibration.json"),
        "Interior comparison does not bind the frozen source data and points",
    )
    rows = []
    expected = read(run / "held.json")["rows"]
    for base, old, new in zip(original, expected, evidence["rows"], strict=True):
        require(
            all(new[k] == old[k] for k in ("key", "group", "truth", "event_aligned"))
            and new["native"]["picture_pts"] == old["native"]["picture_pts"],
            "Interior case changed",
        )
        arms = {}
        for arm, candidate in new["results"].items():
            measured, selected = redo(new["native"], arm)
            p = candidate["proposal"]
            require(
                measured == candidate["measured"]
                and (selected is None) == (p is None)
                and (p is None or (abs(selected[0] - p["score"]) < 1e-10 and selected[1] == p["frames"])),
                "Interior outcome does not independently replay",
            )
            arms[arm] = {
                "measured": measured,
                "score": p["score"] if p else None,
                "frames": p["frames"] if p else None,
            }
        rows.append({**base, "arms": arms})
    return {
        "scope": freeze["scope"],
        "omitted_point": freeze["omitted_point"],
        "freeze_sha256": digest(probe / "freeze.json"),
        "held_sha256": digest(probe / "held.json"),
        "rows": rows,
        "landmark_observations": sum(r["native"].get("geometry_frames", 0) for r in evidence["rows"]),
        "summary": totals(rows, result["points"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    p.add_argument("--interior-probe", type=Path)
    a = p.parse_args()
    if a.assemble:
        assembled = report(a.assemble)
        if a.interior_probe:
            assembled["exploratory_interior_probe"] = summarize_probe(assembled, a.assemble, a.interior_probe)
        a.receipt.write_text(json.dumps(assembled, indent=2, sort_keys=True, allow_nan=False) + "\n")
    result = read(a.receipt)
    require(
        result["schema"] == "chipmunk.video-mouth-rigid-coupling/1"
        and set(result["splits"]) == {"development", "held"},
        "Unknown scope",
    )
    for split, block in result["splits"].items():
        replay = totals(block["rows"], result["points"])
        require(replay == block["summary"], "Public ledger totals do not match")
        print(split, json.dumps(replay, sort_keys=True))
    if probe := result.get("exploratory_interior_probe"):
        replay = totals(probe["rows"], result["points"])
        require(replay == probe["summary"], "Interior probe totals do not replay")
        print("exploratory interior", json.dumps(replay, sort_keys=True))


if __name__ == "__main__":
    main()
