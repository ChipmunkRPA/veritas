# SPDX-License-Identifier: Apache-2.0
"""Replay published decisions and reopen private native chroma-study evidence."""

from __future__ import annotations

import argparse
import json
import runpy
from collections import Counter
from fractions import Fraction
from pathlib import Path

if __package__:
    from scripts.review_video_mouth_contour_phase import audit_file, overlap, read, require, sha
else:
    helper = runpy.run_path(str(Path(__file__).with_name("review_video_mouth_contour_phase.py")))
    audit_file, overlap, read, require, sha = (
        helper[key] for key in ("audit_file", "overlap", "read", "require", "sha")
    )

ARMS = ("luma_control", "color_coupling", "conditioned_chroma")
SCHEMA = "chipmunk.video-mouth-chroma-coupling/1"


def points_for(rows):
    points = {}
    for arm in ARMS:
        scores = sorted(
            (r["arms"][arm]["score"] for r in rows if r["arms"][arm]["score"] is not None), reverse=True
        )
        points[arm] = max(0.03, scores[2] if len(scores) >= 3 else 0)
    return points


def exact_range(entry, row):
    return bool(
        entry["seconds"]
        and row["edit"]
        and overlap(list(map(Fraction, entry["seconds"])), list(map(Fraction, row["edit"]))) >= Fraction(1, 2)
    )


def totals(rows, points):
    summaries = {}
    for arm in ARMS:
        roles = {}
        groups, auc_positive, auc_control = set(), [], []
        false_flags, correct, missed_range_flags, core_flags = 0, 0, 0, 0
        coverage = Counter()
        for row in rows:
            evidence = row["arms"][arm]
            proposed = evidence["score"] is not None
            require(
                proposed == bool(evidence["seconds"]) and (not proposed or evidence["measured"]),
                "Public proposal availability differs",
            )
            localized = exact_range(evidence, row)
            flag = proposed and evidence["score"] > points[arm]
            measured = evidence["measured"]
            positive = row["edit"] is not None
            valid_positive = row["aligned"]
            if measured and valid_positive:
                auc_positive.append(evidence["score"] or 0)
            if measured and not positive:
                auc_control.append(evidence["score"] or 0)
            if flag and localized and valid_positive:
                correct += 1
                groups.add(row["group"])
            if flag and not positive:
                false_flags += 1
                core_flags += row["role"] in {"original", "reencode", "hard_cut"}
            missed_range_flags += flag and positive and not localized
            summary = roles.setdefault(row["family"] + ":" + row["role"], Counter())
            summary.update(
                {
                    "rows": 1,
                    "measured": measured,
                    "flags": flag,
                    "raw_localized": localized,
                    "aligned": valid_positive,
                    "aligned_localized": flag and localized and valid_positive,
                    "aligned_raw_localized": localized and valid_positive,
                }
            )
            key = "aligned_positive" if valid_positive else "unaligned_neural" if positive else row["family"]
            coverage.update({key + "_total": 1, key + "_measured": measured})
        denominator = len(auc_positive) * len(auc_control)
        auc = sum(2 if p > n else 1 if p == n else 0 for p in auc_positive for n in auc_control)
        summaries[arm] = {
            "roles": roles,
            "coverage": coverage,
            "aligned_localized": correct,
            "localized_groups": len(groups),
            "ordinary_flags": false_flags,
            "essential_ordinary_flags": core_flags,
            "wrong_positive_ranges": missed_range_flags,
            "auc_among_measured": str(Fraction(auc, 2 * denominator)) if denominator else None,
            "criteria": {
                "localized": correct >= 4 and len(groups) >= 3,
                "ordinary": false_flags <= 2 and core_flags == 0,
                "coverage": all(
                    coverage[k + "_total"] and coverage[k + "_measured"] * 10 >= coverage[k + "_total"] * 7
                    for k in ("aligned_positive", "lombard", "fresh", "new_ordinary")
                ),
            },
        }
    base = summaries["luma_control"]
    for arm in ARMS[1:]:
        row = summaries[arm]
        row["criteria"]["over_luma"] = (
            row["aligned_localized"] > base["aligned_localized"]
            and row["ordinary_flags"] <= base["ordinary_flags"]
        )
    return summaries


def verify_scope(lock, controls, freeze_sha):
    require(
        controls["freeze_sha256"] == freeze_sha and len(controls["rows"]) == 24, "Held control roster differs"
    )
    expected = {(r["group"], r["id"], r["sha"]) for r in lock["new_control_sources"]}
    for role in ("reencode", "local_chroma_denoise"):
        rows = [r for r in controls["rows"] if r["role"] == role]
        require(
            len(rows) == 12 and {(r["group"], r["id"], r["source_sha256"]) for r in rows} == expected,
            "Ordinary derivation sources differ",
        )
        for row in rows:
            if role == "local_chroma_denoise":
                start, end = row["ordinary_edit_frames"]
                require(
                    end - start == 16 and all(start <= i < end for i in row["changed_pictures"]),
                    "Ordinary local adjustment escaped its range",
                )


def verify_new_native_controls(lock, controls):
    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video.decode import decode_frames, probe_frames
    from scripts import video_mouth_chroma_coupling as worker

    cv2.setNumThreads(1)
    detector = load_yunet(worker.earlier.WEIGHT)
    by_id = {row["id"]: row for row in lock["new_control_sources"]}
    result = Counter()
    for row in controls["rows"]:
        source = by_id[row["id"]]
        path, original = Path(row["file"]), Path(source["file"])
        require(
            sha(path) == row["sha"] and sha(original) == source["sha"], "Derived ordinary source bytes differ"
        )
        target, base = (probe_frames(p, timeout_s=15) for p in (path, original))
        require(
            [i * target.tb for i in target.pts] == [i * base.tb for i in base.pts]
            and (target.width, target.height) == (base.width, base.height),
            "Derived native picture table differs",
        )
        result["derived_clock_entries"] += len(target)
        if row["role"] != "local_chroma_denoise":
            continue
        start, end = row["ordinary_edit_frames"]
        changed = []
        for a, b in zip(
            decode_frames(original, base, None, timeout_s=25),
            decode_frames(path, target, None, timeout_s=25),
            strict=True,
        ):
            require(
                a.pts_verified and b.pts_verified and a.pts * base.tb == b.pts * target.tb,
                "Derived source presentation time differs",
            )
            if start <= a.index < end:
                geometry = worker.earlier.panel.prior.aligned_face(a.rgb, detector)
                require(geometry is not None, "Expected the fixed ordinary mouth crop")
                opacity = 0.35 * min(2, a.index - start + 1, end - a.index) / 2
                expected = worker.denoise(a.rgb, geometry, opacity)
            else:
                expected = a.rgb
            require(
                np.array_equal(expected, b.rgb),
                "Ordinary denoise differs from reconstruction or outside pixels",
            )
            if not np.array_equal(a.rgb, b.rgb):
                changed.append(a.index)
            result["denoise_pictures_reconstructed"] += 1
        require(changed == row["changed_pictures"], "Ordinary changed picture inventory differs")
        result["denoise_changed_pictures"] += len(changed)
    return result


def assemble(run):
    from app.detectors.video.decode import probe_frames

    freeze, development, held, points, controls = (
        run / (key + ".json") for key in ("freeze", "development", "held", "points", "controls")
    )
    lock, point = read(freeze), read(points)
    freeze_sha = sha(freeze)
    receipt = read(controls)
    verify_scope(lock, receipt, freeze_sha)
    require(
        point["development_sha256"] == sha(development), "Frozen points differ from the development scores"
    )
    result = {
        "schema": SCHEMA,
        "protocol": lock["protocol"],
        "freeze_sha256": freeze_sha,
        "points": point["points"],
        "development_sha256": sha(development),
        "held_sha256": sha(held),
        "ordinary_derivatives_sha256": sha(controls),
        "splits": {},
    }
    group_alias, subject_alias = {}, {}
    native_audit = verify_new_native_controls(lock, receipt)
    for split, file in (("development", development), ("held", held)):
        observed = read(file)
        require(
            observed["freeze_sha256"] == freeze_sha
            and observed["controls_sha256"] == (sha(controls) if split == "held" else None),
            "Results do not bind frozen protocol and derived controls",
        )
        original = [*lock["rows"], *receipt["rows"]] if split == "held" else lock["rows"]
        expected = {(r["family"], r["id"], r["role"]): r for r in original if r["split"] == split}
        rows = []
        for row in observed["rows"]:
            key = row["family"], row["id"], row["role"]
            committed = expected.pop(key)
            require(all(row[k] == v for k, v in committed.items()), "Observed source row differs")
            truth = row["authored_native_range"]
            if truth:
                table = probe_frames(Path(row["file"]), timeout_s=12)
                require(
                    truth == [str(table.pts[i] * table.tb) for i in row["edit"]],
                    "Authored native range differs",
                )
            family = "fresh" if row["family"] == "new_ordinary" else row["family"]
            group = group_alias.setdefault((family, row["group"]), len(group_alias) + 1)
            subject = subject_alias.setdefault((family, row["id"]), len(subject_alias) + 1)
            brief = {
                "family": row["family"],
                "role": row["role"],
                "group": group,
                "clip": subject,
                "edit": truth,
                "aligned": row.get("aligned", False),
                "arms": {},
            }
            if row.get("changed_pictures") is not None:
                brief["ordinary_changed_pictures"] = len(row["changed_pictures"])
            wanted = {}
            for arm in ARMS:
                evidence = row["arms"][arm]
                p = evidence["proposal"]
                compact = {
                    "measured": evidence["measured"],
                    "score": p["score"] if p else None,
                    "seconds": p["seconds"] if p else None,
                }
                brief["arms"][arm] = compact
                if (
                    split == "held"
                    and p
                    and (p["score"] > point["points"][arm] or exact_range(compact, brief))
                ):
                    require(
                        len(p["support_sectors"]) >= 3 and len(p["positive_sectors"]) >= 3,
                        "Selected interval lacks required support",
                    )
                    a, b = p["frames"]
                    require(
                        [anchor["frame"] for anchor in p["anchors"]] == [a - 1, (a + b) // 2, b],
                        "Evidence anchor positions differ",
                    )
                    table = probe_frames(Path(row["file"]), timeout_s=12)
                    require(
                        p["time_base"] == table.time_base
                        and p["seconds"] == [str(table.pts[i] * table.tb) for i in (a, b)],
                        "Proposal native clock differs",
                    )
                    wanted.update({a["frame"]: a for a in p["anchors"]})
                    native_audit["ranges"] += 1
            if wanted:
                native_audit["files"] += 1
                native_audit["picture_crops"] += audit_file(Path(row["file"]), row, wanted)
            rows.append(brief)
        require(
            not expected and len(rows) == (56 if split == "development" else 128),
            "Frozen row inventory differs",
        )
        result["splits"][split] = {
            "rows": rows,
            "summary": totals(rows, point["points"]),
            "native_pictures": sum(r["picture_ticks"] for r in observed["rows"]),
            "observed_geometry": sum(r["geometry_frames"] for r in observed["rows"]),
        }
    result["native_audit"] = dict(native_audit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    args = parser.parse_args()
    if args.assemble:
        with args.receipt.open("x") as handle:
            handle.write(
                json.dumps(assemble(args.assemble), sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
    receipt = read(args.receipt)
    require(
        receipt["schema"] == SCHEMA and set(receipt["splits"]) == {"development", "held"},
        "Unexpected public scope",
    )
    dev = receipt["splits"]["development"]["rows"]
    require(receipt["points"] == points_for(dev), "Frozen points do not independently replay")
    for split, block in receipt["splits"].items():
        require(
            totals(block["rows"], receipt["points"]) == block["summary"], "Public arithmetic does not replay"
        )
        print(split, json.dumps(block["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
