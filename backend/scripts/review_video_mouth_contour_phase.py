# SPDX-License-Identifier: Apache-2.0
"""Replay the sanitized phase correspondence study and its native picture anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

ARMS = ("fine_phase", "coarse_adjusted")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def overlap(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def results(rows, points):
    arms = {}
    for arm in ARMS:
        counts, localized_groups = {}, set()
        aligned, flags = 0, 0
        flagged_ordinary, coverage = 0, Counter()
        positive_scores, ordinary_scores = [], []
        for r in rows:
            candidate = r["arms"][arm]
            proposed = candidate["score"] is not None
            require(
                proposed == bool(candidate["seconds"]) and (not proposed or candidate["measured"]),
                "Measured decision differs",
            )
            alert = proposed and candidate["score"] > points[arm]
            localized = bool(
                proposed
                and r["edit"]
                and overlap(list(map(Fraction, candidate["seconds"])), list(map(Fraction, r["edit"])))
                >= Fraction(1, 2)
            )
            counts.setdefault(r["family"] + ":" + r["role"], Counter()).update(
                {
                    "rows": 1,
                    "measured": candidate["measured"],
                    "raw_localized": localized,
                    "flags": alert,
                    "localized": alert and localized,
                    "aligned_rows": r["aligned"],
                    "aligned_measured": r["aligned"] and candidate["measured"],
                    "aligned_raw_localized": r["aligned"] and localized,
                }
            )
            if alert and localized and r["aligned"]:
                aligned += 1
                localized_groups.add(r["group"])
            if alert:
                flags += not r["edit"]
                flagged_ordinary += r["role"] in {"original", "reencode", "hard_cut"}
            if r["family"] in {"fresh", "lombard"}:
                name = "positive" if r["edit"] else "ordinary" if r["family"] == "lombard" else "fresh"
                coverage[name + "_total"] += 1
                coverage[name + "_measured"] += candidate["measured"]
            if proposed:
                if r["aligned"]:
                    positive_scores.append(candidate["score"])
                elif r["edit"] is None:
                    ordinary_scores.append(candidate["score"])
        denominator = len(positive_scores) * len(ordinary_scores)
        auc = sum((2 if p > n else 1 if p == n else 0) for p in positive_scores for n in ordinary_scores)
        arms[arm] = {
            "by_role": counts,
            "aligned_localized": aligned,
            "aligned_groups": len(localized_groups),
            "ordinary_flags": flags,
            "essential_control_flags": flagged_ordinary,
            "coverage": coverage,
            "raw_proposal_auc_aligned_vs_ordinary": str(Fraction(auc, 2 * denominator))
            if denominator
            else None,
            "gate": {
                "localized": aligned >= 4,
                "groups": len(localized_groups) >= 3,
                "ordinary": flags <= 2 and flagged_ordinary == 0,
                "coverage": all(
                    10 * coverage[k + "_measured"] >= 7 * coverage[k + "_total"]
                    for k in ("positive", "ordinary", "fresh")
                ),
            },
        }
    return arms


def audit_file(path, row, wanted):
    from app.detectors.video.decode import decode_frames, probe_frames

    require(sha(path) == row["sha"], "Native evidence source hash differs")
    table = probe_frames(path, timeout_s=15)
    found = set()
    for frame in decode_frames(path, table, sorted(wanted), timeout_s=20):
        anchor = wanted[frame.index]
        x0, y0, x1, y1 = anchor["crop_xyxy"]
        require(
            frame.pts_verified
            and frame.pts == anchor["pts"]
            and hashlib.sha256(frame.rgb.tobytes()).hexdigest() == anchor["rgb_sha256"]
            and hashlib.sha256(frame.rgb[y0:y1, x0:x1].tobytes()).hexdigest() == anchor["crop_sha256"],
            "Frame timestamp, full pixels or native crop differs",
        )
        found.add(frame.index)
    require(found == set(wanted), "Selected native proof incomplete")
    return len(found)


def assemble(run, do_audit):
    freeze, dev, held, points = (run / (k + ".json") for k in ("freeze", "development", "held", "points"))
    lock, point = read(freeze), read(points)
    require(point["development_sha256"] == sha(dev), "Frozen ordinary comparison point differs")
    result = {
        "schema": "chipmunk.video-mouth-contour-phase/1",
        "protocol": lock["protocol"],
        "freeze_sha256": sha(freeze),
        "points": point["points"],
        "development_sha256": sha(dev),
        "held_sha256": sha(held),
        "splits": {},
    }
    groups, subjects = {}, {}
    audited = Counter()
    for split, path in (("development", dev), ("held", held)):
        record = read(path)
        require(record["freeze_sha256"] == sha(freeze), "Observed panel does not bind frozen method")
        committed = {(r["family"], r["id"], r["role"]): r for r in lock["rows"] if r["split"] == split}
        rows = []
        for row in record["rows"]:
            k = (row["family"], row["id"], row["role"])
            require(
                all(row[name] == val for name, val in committed.pop(k).items()), "Observed source differs"
            )
            group = groups.setdefault((row["family"], row["group"]), len(groups) + 1)
            subject = subjects.setdefault(k[:2], len(subjects) + 1)
            arms = {}
            wanted = {}
            truth = None
            if row.get("edit"):
                from app.detectors.video.decode import probe_frames

                table = probe_frames(Path(row["file"]), timeout_s=12)
                truth = [str(table.pts[i] * table.tb) for i in row["edit"]]
            for arm, entry in row["arms"].items():
                p = entry["proposal"]
                arms[arm] = {
                    "measured": entry["measured"],
                    "score": p["score"] if p else None,
                    "seconds": p["seconds"] if p else None,
                }
                if (
                    split == "held"
                    and p
                    and (
                        p["score"] > point["points"][arm]
                        or truth
                        and overlap(list(map(Fraction, p["seconds"])), list(map(Fraction, truth)))
                        >= Fraction(1, 2)
                    )
                ):
                    audited["selected_ranges"] += 1
                    require(
                        len(p["support_sectors"]) >= 3 and len(p["positive_sectors"]) >= 3,
                        "Proposal lacks supported sectors",
                    )
                    wanted.update({a["frame"]: a for a in p["anchors"]})
            if wanted and do_audit:
                audited["files"] += 1
                audited["native_crops"] += audit_file(Path(row["file"]), row, wanted)
            rows.append(
                {
                    "family": row["family"],
                    "group": group,
                    "clip": subject,
                    "role": row["role"],
                    "edit": truth,
                    "aligned": row.get("aligned", False),
                    "arms": arms,
                }
            )
        require(
            not committed and len(rows) == (56 if split == "development" else 104),
            "Expected all frozen source rows",
        )
        result["splits"][split] = {
            "rows": rows,
            "native_picture_ticks": sum(r["picture_ticks"] for r in record["rows"]),
            "geometry_frames": sum(r["geometry_frames"] for r in record["rows"]),
            "summary": results(rows, point["points"]),
        }
    if do_audit:
        result["native_audit"] = dict(audited)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    a = p.parse_args()
    if a.assemble:
        result = assemble(a.assemble, True)
        a.receipt.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    result = read(a.receipt)
    require(
        result["schema"] == "chipmunk.video-mouth-contour-phase/1"
        and set(result["splits"]) == {"development", "held"},
        "Unexpected result scope",
    )
    for arm in ARMS:
        found = sorted(
            (
                v["arms"][arm]["score"]
                for v in result["splits"]["development"]["rows"]
                if v["arms"][arm]["score"] is not None
            ),
            reverse=True,
        )
        require(
            result["points"][arm] == max(0.02, found[2] if len(found) > 2 else 0.02),
            "Point does not independently replay",
        )
    for split, block in result["splits"].items():
        require(
            results(block["rows"], result["points"]) == block["summary"], "Published result totals differ"
        )
        print(split, json.dumps(block["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
