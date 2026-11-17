"""Recount a frozen local face comparison with a native shot continuity check."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import face_local_return_binding as local
from scripts import video_common_sources_v89 as io
from scripts import video_face_local_fit as fit
from scripts.video_face_local_replay import report

SCHEMA = "chipmunk.video-face-shot-guard/1"
FIELDS = ("file", "group", "family", "delivery", "truth", "existing", "existing_assessed",
          "source_sha256", "input_sha256")


def summarize(rows):
    by_delivery = defaultdict(Counter)
    names = set()
    for row in rows:
        if row["file"] in names:
            raise ValueError("Duplicate video in the replay")
        names.add(row["file"])
        truth = row["truth"]
        prior, suggestion, cuts = row["existing"], row["proposal"], row["cuts"]
        if suggestion is not None:
            before, _, after = local.support(*suggestion)
            if cuts is None or cuts != sorted(set(cuts)) or any(not before[0] < c <= after[-1] for c in cuts):
                raise ValueError("Shot changes must match the proposed range and its flanks")
        elif cuts is not None:
            raise ValueError("A shot review needs a proposed face change")
        guarded = None if cuts else suggestion
        counts = by_delivery[row["delivery"]]
        counts.update({
            "edited": int(truth is not None), "control": int(truth is None),
            "old_useful": int(fit.overlap(prior, truth) >= .5),
            "old_ordinary_flags": int(truth is None and prior is not None),
            "ungated_useful": int(fit.overlap(prior or suggestion, truth) >= .5),
            "ungated_ordinary_flags": int(truth is None and bool(prior or suggestion)),
            "guarded_useful": int(fit.overlap(prior or guarded, truth) >= .5),
            "guarded_ordinary_flags": int(truth is None and bool(prior or guarded)),
            "above_point": int(suggestion is not None), "crossed_cut": int(bool(cuts)),
            "reviewed_face_context": int(row["considered"] or row["existing_assessed"]),
        })
    return {delivery: dict(stats) for delivery, stats in sorted(by_delivery.items())}


def scan(delivered, frozen, output, measurements=None):
    import cv2

    from app.detectors.video.face_local_detail import shot_changes
    from scripts.video_face_temporal_readout import table_for

    cv2.setNumThreads(1)
    model = report(frozen)
    lock = model["lock"]
    selected = lock["choices"][lock["selected"]]
    arm, point = selected["arm"], selected["point"]
    head = lock["heads"][arm]
    if (point != local.POINT or head["columns"] != [0, 1] or tuple(head["center"]) != local.CENTER
            or tuple(head["scale"]) != local.SCALE or tuple(head["weights"]) != local.COEFFICIENTS):
        raise ValueError("The app comparison differs from the frozen appearance method")
    if measurements is None:
        rows = model["rows"]
    else:
        header, measured = fit.measured(measurements)
        fit.verify_manifest(header, measured, delivered / "manifest.json")
        if ({row["group"] for row in measured} & {row["group"] for row in model["rows"]} or
                {row["source_sha256"] for row in measured} & {row["source_sha256"] for row in model["rows"]}):
            raise ValueError("The comparison group overlaps the earlier face study")
        rows = fit.readout(measured, {arm: head})
    results = []
    for row in rows:
        path = delivered / row["file"]
        if io.sha(path) != row["input_sha256"]:
            raise ValueError("Video differs from the measured receiver")
        candidate = row["candidate"][arm]
        proposed = candidate["range"] if candidate is not None and candidate["score"] > point else None
        cuts = list(shot_changes(path, table_for(path), *proposed)) if proposed is not None else None
        results.append({name: row[name] for name in FIELDS} |
                       {"proposal": proposed, "considered": candidate is not None, "cuts": cuts})
    value = {"schema": SCHEMA, "frozen_comparison_sha256": io.sha(frozen),
             "manifest_sha256": io.sha(delivered / "manifest.json"), "rows": results, "counts": summarize(results)}
    io.write(output, value)
    return value["counts"]


def verify(receipt, original_worker=None):
    data = io.read(receipt)
    counts = summarize(data["rows"])
    if data["schema"] != SCHEMA or data["counts"] != counts:
        raise ValueError("The replay counts differ from the committed picture ranges")
    if original_worker is not None:
        former = {row["file"]: row for row in io.read(original_worker)["cases"]}
        useful, flagged, refused, lost = [0] * 4
        for row in data["rows"]:
            old = former.get(row["file"])
            if old is None or old["added_range"] is None:
                continue
            if old["input_sha256"] != row["input_sha256"] or old["added_range"] != row["proposal"]:
                raise ValueError("The earlier worker saw a different video or proposed range")
            useful += old["new_useful_range"] and not row["cuts"]
            flagged += old["truth"] is None and not row["cuts"]
            refused += old["truth"] is None and bool(row["cuts"])
            lost += old["new_useful_range"] and bool(row["cuts"])
        counts["previous_worker_recheck"] = {"extra_useful": useful, "remaining_ordinary_flags": flagged,
                                            "removed_ordinary_flags": refused, "lost_useful": lost}
    return counts


def verify_worker(receipt, worker):
    expected = {row["file"]: row for row in io.read(receipt)["rows"] if row["proposal"] is not None}
    payload = io.read(worker)
    if payload["comparison_sha256"] != io.sha(receipt) or payload["unpaid_jobs"] != 0:
        raise ValueError("App replay uses a different panel or failed its payment check")
    seen, counts, useful_groups = set(), Counter(), set()
    for case in payload["cases"]:
        name = case["file"]
        if name in seen or name not in expected:
            raise ValueError("Unexpected or duplicate app replay")
        seen.add(name)
        row = expected[name]
        marked, truth = case["new_range"], case["truth"]
        if (any(case[key] != row[key] for key in ("input_sha256", "delivery", "group", "family", "truth"))
                or (marked is not None and (row["cuts"] or marked != row["proposal"]))):
            raise ValueError("An app range differs from the allowed source proposal")
        extra = fit.overlap(marked, truth) >= .5 and not any(fit.overlap(old, truth) >= .5 for old in case["prior_ranges"])
        if case["extra_useful"] is not extra:
            raise ValueError("An added location differs from its labeled range or incumbent")
        if extra:
            useful_groups.add(row["group"])
        cut = (case["coverage"] or {}).get("reason") == "local_shot_change"
        if cut and not row["cuts"]:
            raise ValueError("App attributed the rejection to a cut absent from the source review")
        counts.update({"extra_useful": int(extra), "ordinary_new_flags": int(truth is None and bool(marked)),
                       "misplaced_edit_ranges": int(truth is not None and bool(marked) and fit.overlap(marked, truth) < .5),
                       "shot_vetoes": int(cut)})
    if (seen != expected.keys() or len(seen) != payload["comparisons"] or dict(counts) != payload["counts"]
            or payload["simulated_paid_jobs"] != len(seen) + int(payload["confidence_toggle_verified"])):
        raise ValueError("App replay is incomplete or its counts do not match")
    return dict(counts) | {"additional_useful_source_groups": len(useful_groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan")
    for name in ("delivered", "frozen", "output"):
        scan_parser.add_argument(name, type=Path)
    scan_parser.add_argument("--measurements", type=Path)
    review = commands.add_parser("verify")
    review.add_argument("receipt", type=Path)
    review.add_argument("--original-worker", type=Path)
    review.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.command == "scan":
        print(scan(args.delivered, args.frozen, args.output, args.measurements))
    else:
        print(verify(args.receipt, args.original_worker))
        if args.worker:
            print(verify_worker(args.receipt, args.worker))


if __name__ == "__main__":
    main()
