# SPDX-License-Identifier: Apache-2.0
"""Recompute the frozen trajectory decisions and reopen selected native pictures."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from collections import Counter
from fractions import Fraction
from pathlib import Path

ARMS = ("trajectory_loss", "cycle_separation")
FLOORS = (0.08, 0.10)
SCHEMA = "chipmunk.video-mouth-track-cycle/1"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def overlap(a, b):
    a0, a1 = map(Fraction, a)
    b0, b1 = map(Fraction, b)
    return max(0, min(a1, b1) - max(a0, b0)) / (max(a1, b1) - min(a0, b0))


def points_for(rows):
    result = {}
    for arm, floor in zip(ARMS, FLOORS, strict=True):
        values = sorted(
            (r["arms"][arm]["score"] for r in rows if r["arms"][arm]["score"] is not None), reverse=True
        )
        result[arm] = max(floor, values[2] if len(values) >= 3 else 0)
    return result


def totals(rows, points):
    result = {}
    for arm in ARMS:
        stats = Counter()
        by_role = {}
        groups = set()
        positive_scores, ordinary_scores = [], []
        for row in rows:
            data = row["arms"][arm]
            require(
                bool(data["seconds"]) == (data["score"] is not None)
                and (data["measured"] or data["score"] is None),
                "Proposal and measurement availability differ",
            )
            alerted = data["score"] is not None and data["score"] > points[arm]
            localized = bool(
                data["seconds"] and row["truth"] and overlap(data["seconds"], row["truth"]) >= Fraction(1, 2)
            )
            family = row["family"]
            by_role.setdefault(f"{family}:{row['role']}", Counter()).update(
                {
                    "files": 1,
                    "measured": data["measured"],
                    "alerts": alerted,
                    "correct_ranges": alerted and localized,
                    "useful_before_point": localized,
                }
            )
            if row["truth"] is None:
                stats.update(
                    {
                        "ordinary": 1,
                        "ordinary_measured": data["measured"],
                        "ordinary_alerts": alerted,
                        "essential_alerts": alerted and row["role"] in {"original", "reencode", "hard_cut"},
                    }
                )
                if data["measured"]:
                    ordinary_scores.append(data["score"] or 0)
                continue
            stats.update(
                {
                    "neural": 1,
                    "neural_measured": data["measured"],
                    "neural_alerts": alerted,
                    "neural_localized": alerted and localized,
                    "neural_wrong_alerts": alerted and not localized,
                }
            )
            if row["aligned"]:
                stats.update(
                    {
                        f"strict_{family}": 1,
                        f"strict_{family}_measured": data["measured"],
                        f"strict_{family}_localized": alerted and localized,
                        f"strict_{family}_raw_localized": localized,
                    }
                )
                if data["measured"]:
                    positive_scores.append(data["score"] or 0)
                if family == "lombard" and alerted and localized:
                    groups.add(row["group"])
            if family == "musetalk" and row["source_event_retained"]:
                stats.update(
                    {
                        "musetalk_retained": 1,
                        "musetalk_retained_measured": data["measured"],
                        "musetalk_retained_localized": alerted and localized,
                    }
                )
        auc_count = sum(2 if a > b else 1 if a == b else 0 for a in positive_scores for b in ordinary_scores)
        denom = 2 * len(positive_scores) * len(ordinary_scores)
        result[arm] = {
            "counts": stats,
            "by_role": by_role,
            "strict_liveportrait_localized_groups": len(groups),
            "strict_aligned_auc_among_measured": str(Fraction(auc_count, denom)) if denom else None,
            "gate": {
                "liveportrait": stats["strict_lombard_localized"] >= 4 and len(groups) >= 3,
                "musetalk": stats["strict_musetalk_localized"] == stats["strict_musetalk"] == 1,
                "ordinary": stats["ordinary_alerts"] <= 2 and not stats["essential_alerts"],
                "coverage": all(
                    stats[k] and stats[k + "_measured"] * 10 >= 7 * stats[k]
                    for k in ("strict_lombard", "strict_musetalk", "ordinary")
                ),
            },
        }
    return result


def assemble(run):
    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video.decode import decode_frames, probe_frames
    from scripts import video_mouth_track_cycle as worker

    replay = runpy.run_path(str(Path(__file__).with_name("review_video_mouth_contour_phase.py")))
    lock, controls, dev, held, points = (
        run / (n + ".json") for n in ("freeze", "controls", "development", "held", "points")
    )
    source, derived = read(lock), read(controls)
    require(
        derived["freeze_sha256"] == sha(lock)
        and len(derived["rows"]) == 12
        and len(source["control_sources"]) == 6,
        "Derived roster does not match the frozen plan",
    )
    originals = {r["id"]: r for r in source["control_sources"]}
    derivatives = Counter()
    cv2.setNumThreads(1)
    detector = load_yunet(worker.prior.WEIGHT)
    for row in derived["rows"]:
        original = originals[row["id"]]
        a, b = row["authored_control_frames"]
        require(
            sha(Path(original["file"])) == original["sha"] == row["source_sha"]
            and sha(Path(row["file"])) == row["sha"]
            and b - a == 16
            and all(a <= i < b for i in row["changed_pictures"]),
            "Ordinary derived pixels are unbound",
        )
        first, second = (probe_frames(p, timeout_s=15) for p in (Path(original["file"]), Path(row["file"])))
        require(
            len(first) == len(second) == row["verified_pictures"]
            and [p * first.tb for p in first.pts] == [p * second.tb for p in second.pts],
            "Derived control source clock differs",
        )
        rgb = []
        for frame in decode_frames(Path(original["file"]), first, None, timeout_s=25):
            require(
                frame.pts_verified and frame.pts == first.pts[frame.index], "Original control PTS differs"
            )
            rgb.append(frame.rgb)
        require(len(rgb) == len(first), "Original control did not reopen all pictures")
        changed = []
        for frame in decode_frames(Path(row["file"]), second, None, timeout_s=25):
            i = frame.index
            geometry = worker.prior.panel.prior.aligned_face(rgb[i], detector) if a <= i < b else None
            expected = worker.derived_picture(rgb, i, [a, b], geometry, row["role"])
            require(
                frame.pts_verified and frame.pts == second.pts[i] and np.array_equal(expected, frame.rgb),
                "Control geometry, outside pixels or native picture differs",
            )
            if not np.array_equal(rgb[i], frame.rgb):
                changed.append(i)
            derivatives["reconstructed_pictures"] += 1
        require(changed == row["changed_pictures"], "Ordinary changed-picture roster differs")
        derivatives.update(
            {
                "verified_pictures": len(first),
                "committed_changed_pictures": len(row["changed_pictures"]),
                "files": 1,
            }
        )
    point = read(points)
    require(
        point["development_sha256"] == sha(dev), "Held point is not based on the committed development run"
    )
    report = {
        "schema": SCHEMA,
        "protocol": source["protocol"],
        "freeze_sha256": sha(lock),
        "points": point["points"],
        "development_sha256": sha(dev),
        "held_sha256": sha(held),
        "derived_sha256": sha(controls),
        "ordinary_derivation_readback": dict(derivatives),
        "splits": {},
    }
    aliases, audit = {}, Counter()
    for split, file in (("development", dev), ("held", held)):
        observed = read(file)
        require(
            observed["freeze_sha256"] == sha(lock)
            and observed["derived_controls_sha256"] == (sha(controls) if split == "held" else None),
            "Scored source did not bind the frozen clock",
        )
        roster = [*source["rows"], *derived["rows"]] if split == "held" else source["rows"]
        pending = {(r["family"], r["id"], r["role"]): r for r in roster if r["split"] == split}
        result = []
        for row in observed["rows"]:
            frozen = pending.pop((row["family"], row["id"], row["role"]))
            require(all(row[k] == v for k, v in frozen.items()), "Scored roster row differs")
            family = "fresh" if row["family"] == "derived_ordinary" else row["family"]
            key = (family, row["group"])
            alias = aliases.setdefault(key, len(aliases) + 1)
            record = {
                "file": len(result) + 1,
                "family": row["family"],
                "group": alias,
                "role": row["role"],
                "truth": row["edit_truth"],
                "aligned": row.get("aligned", False),
                "source_event_retained": row.get("source_event_retained", False),
                "arms": {},
            }
            wanted = {}
            for arm in ARMS:
                arm_input = row["arms"][arm]
                p = arm_input["proposal"]
                record["arms"][arm] = {
                    "measured": arm_input["measured"],
                    "score": p["score"] if p else None,
                    "seconds": p["seconds"] if p else None,
                }
                if (
                    split == "held"
                    and p
                    and (
                        p["score"] > point["points"][arm]
                        or record["truth"]
                        and overlap(p["seconds"], record["truth"]) >= Fraction(1, 2)
                    )
                ):
                    a, b = p["frames"]
                    table = probe_frames(Path(row["file"]), timeout_s=15)
                    require(
                        table.time_base == p["time_base"]
                        and p["seconds"] == [str(table.pts[j] * table.tb) for j in (a, b)]
                        and [x["frame"] for x in p["anchors"]] == [a - 1, (a + b) // 2, b],
                        "Native proposal clock or exhibits differ",
                    )
                    wanted.update({v["frame"]: v for v in p["anchors"]})
                    audit["proposals"] += 1
            if wanted:
                audit["files"] += 1
                audit["native_frame_crops"] += replay["audit_file"](Path(row["file"]), row, wanted)
            result.append(record)
        require(
            not pending and len(result) == (56 if split == "development" else 146), "Incomplete comparison"
        )
        report["splits"][split] = {
            "rows": result,
            "totals": totals(result, point["points"]),
            "native_frames": sum(r["native_frames"] for r in observed["rows"]),
            "geometry_frames": sum(r["geometry_frames"] for r in observed["rows"]),
        }
    report["native_audit"] = dict(audit)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    a = p.parse_args()
    if a.assemble:
        with a.receipt.open("x") as handle:
            json.dump(assemble(a.assemble), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    data = read(a.receipt)
    require(
        data["schema"] == SCHEMA and set(data["splits"]) == {"development", "held"},
        "Unexpected public replay scope",
    )
    require(
        points_for(data["splits"]["development"]["rows"]) == data["points"], "Frozen operating points differ"
    )
    for split, block in data["splits"].items():
        require(totals(block["rows"], data["points"]) == block["totals"], "Published decision totals differ")
        print(split, json.dumps(block["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
