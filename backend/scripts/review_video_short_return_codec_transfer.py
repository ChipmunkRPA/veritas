# SPDX-License-Identifier: Apache-2.0
"""Independently score actual 360p30 media and replay its precise evidence crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.review_video_short_return_encoded import overlap, verify_case
from scripts.review_video_short_return_margin import digest, read, require

CODECS = ("hevc", "av1")
RATE = 30
SPANS = {"swap_a": [84, 95], "swap_b": [91, 102]}
MATCHED = ("original", "ordinary_light", "ordinary_blur", *SPANS)
DATA = Path(__file__).resolve().parents[1] / "data/independent-evaluation"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")


def camera_sources(lock):
    verified, drift = 0, []
    parent = DATA / "av-sentence-support-v104/preparation/plan.json"
    require(digest(parent) == lock["camera_plan_sha256"], "Camera source metadata differs from its freeze")
    metadata = {row["asset_index"]: row for row in read(parent)["records"]}
    require(len({metadata[pin["index"]]["actor"] for pin in lock["camera"]}) == len(lock["camera"]) == 6,
            "Camera actor code repeated")
    for pin in lock["camera"]:
        receipt = DATA / pin["journal"]
        require(digest(receipt) == pin["journal_sha256"], "Camera source receipt differs from its freeze")
        [row] = [row for row in map(json.loads, receipt.read_text().splitlines())
                 if row["asset_index"] == pin["index"] and row["condition"] == "original"]
        source = ARCHIVE / "native-mouth-clips-v120" / Path(*Path(row["file"]).parts[-3:])
        require(digest(source) == pin["media_sha256"] == row["file_sha256"], "Natural camera media differs")
        table = probe_frames(source, timeout_s=30)
        native = [p * table.tb for p in table.pts]
        require((table.width, table.height) == (480, 360) and not (table.rotation or table.interpolated or table.truncated)
                and len(native) == pin["pictures"]
                and native == [Fraction(ms, 1000) for ms in row["expected"]["picture_pts_ms"]],
                "Original natural camera clock or geometry differs")
        worst = max(abs(p - native[0] - Fraction(i, RATE)) for i, p in enumerate(native))
        require(worst == Fraction(pin["maximum_relative_retime_s"]) and worst <= Fraction(1, 250),
                "Camera delivery exceeded its frozen timestamp movement")
        drift.append(worst)
        seen = 0
        for frame in decode_frames(source, table, list(range(len(native))), timeout_s=30):
            require(frame.index == seen and frame.pts_verified
                    and hashlib.sha256(frame.rgb).hexdigest() == row["expected"]["rgb_sha256"][seen],
                    "Original camera picture differs from its receipt")
            seen += 1
        require(seen == len(native), "Natural camera picture roster is incomplete")
        verified += seen
    return {"source_groups": len(drift), "original_pictures_verified": verified,
            "maximum_relative_timestamp_movement_ms": float(1000 * max(drift))}


def audit(folder, stage):
    freeze, media, journal = folder / "freeze.json", folder / "media.json", folder / stage / "journal.jsonl"
    lock, manifest = read(freeze), read(media)
    require(lock["delivery_size"] == [640, 360] and lock["delivery_rate_hz"] == RATE and lock["edit_frame_spans"] == SPANS
            and lock["review_point"] == .10 and digest(freeze) == manifest["freeze_sha256"]
            and freeze.stat().st_mtime_ns <= media.stat().st_mtime_ns <= journal.stat().st_mtime_ns, "Frozen run order or method differs")
    rows = manifest["media"]
    expected = {(codec, i, role) for codec in CODECS for i in range(3) for role in MATCHED} | {
        (codec, i, role) for codec in CODECS for role, count in (("camera", 6), ("hard_cut", 2)) for i in range(count)}
    require({(row["codec"], row["group"], row["condition"]) for row in rows} == expected
            and len(rows) == len({row["file"] for row in rows}) == 46
            and all(row["truth_frames"] == SPANS.get(row["condition"]) for row in rows), "Frozen media roster differs")
    by_name = {row["file"]: row for row in rows}
    scores = [json.loads(line) for line in journal.read_text().splitlines()]
    names = [score["file"] for score in scores]
    proof = stage == "experimental-report-proof"
    if stage == "scoring":
        require(names == list(by_name), "Incomplete measured roster")
    else:
        require(len(names) == len(set(names)) == 4 and set(names) <= set(by_name), "Report proof needs four distinct inputs")
    counts = defaultdict(lambda: defaultdict(Counter))
    gains, examples, ious, errors = defaultdict(set), defaultdict(dict), defaultdict(list), defaultdict(list)
    precision = defaultdict(list)
    pictures, images = 0, 0
    for score in scores:
        require(score["paid_before_analysis_and_idempotency"] and score["decision_parity"] == proof,
                "Simulated payment or off/on parity proof differs")
        row = by_name[score["file"]]
        path = folder / "media" / row["file"]
        codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
            "-of", "default=nw=1:nk=1", str(path)], timeout=15, text=True).strip()
        require(codec == row["codec"], "Delivery codec differs from its label")
        n = len(row["rgb_sha256"])
        require(n == (lock["camera"][row["group"]]["pictures"] if row["condition"] == "camera" else 180),
                "Delivered picture count differs")
        clock = [Fraction(i, RATE) for i in range(n)]
        current, candidate, verified = verify_case(path, row, score, folder / stage / path.stem, native=clock)
        if proof:
            exported = read(folder / stage / f"{path.stem}-export.json")["evidence"]
            require([packet for packet in exported["packets"] if packet.get("face_return")] ==
                    ([score["packet"]] if score["packet"] else []), "Retained JSON export differs from its evidence packet")
        pictures += n
        images += verified
        kind = "swap" if row["truth_frames"] else row["condition"] if row["condition"] in {"camera", "hard_cut"} else "matched"
        status = (score["return_check"] or {}).get("status")
        require((candidate is not None) == (status == "localized"), "Selected range status differs")
        count = counts[codec][kind]
        count["views"] += 1
        count["assessed"] += status in {"no_distinct_range", "localized"}
        count["incumbent_ranges"] += current is not None
        count["added_ranges"] += candidate is not None
        if kind == "swap":
            truth = tuple(clock[i] for i in row["truth_frames"])
            old = current is not None and overlap(current, truth) >= Fraction(1, 2)
            iou = overlap(candidate, truth) if candidate is not None else 0
            new = iou >= Fraction(1, 2)
            count["incumbent_correct"] += old
            count["combined_correct"] += old or new
            count["added_misplaced"] += candidate is not None and not new
            if old and new:
                precision[codec].append(iou - overlap(current, truth))
            if new:
                examples[codec].setdefault("swap_range", {"truth_seconds": list(map(float, truth)),
                    "range_seconds": list(map(float, candidate)), "iou": float(iou)})
                if proof:
                    require(score["report_replay"]["signed_original_replay_verified"], "Signed report or source replay missing")
            if new and not old:
                count["additional_correct"] += 1
                gains[codec].add(row["group"])
                ious[codec].append(iou)
                errors[codec].extend(abs(x-y) for x, y in zip(candidate, truth, strict=True))
                examples[codec].setdefault("swap", {"file": row["file"], "truth_seconds": list(map(float, truth)),
                    "range_seconds": list(map(float, candidate)), "iou": float(iou)})
        elif candidate is None:
            examples[codec].setdefault(kind, row["file"])
    result = {}
    for codec, classes in counts.items():
        c = classes["swap"]
        ordinary = [classes[key] for key in ("matched", "camera")]
        if stage == "scoring":
            assessed = {"swaps": str(Fraction(c["assessed"], c["views"])),
                        "ordinary": str(Fraction(sum(row["assessed"] for row in ordinary), sum(row["views"] for row in ordinary)))}
            passes = (c["additional_correct"] >= 3 and len(gains[codec]) >= 2
                      and min(map(Fraction, assessed.values())) >= Fraction(4, 5)
                      and sum(row["added_ranges"] for row in ordinary) <= 1 and classes["hard_cut"]["added_ranges"] == 0)
        else:
            require(c["added_ranges"] == c["views"] == 1 and c["added_misplaced"] == 0 and classes["camera"]["views"] == 1
                    and all(classes[role]["added_ranges"] == 0 for role in ("matched", "camera", "hard_cut")),
                    "Experimental report replay lost a positive or added a control flag")
            assessed, passes = {}, None
        result[codec] = {"counts": classes, "added_correct_source_groups": len(gains[codec]), "coverage": assessed,
                         "median_addition_iou": float(median(ious[codec])) if ious[codec] else None,
                         "maximum_added_edge_error_ms": float(1000 * max(errors[codec])) if errors[codec] else None,
                         "retrospective_precision_on_shared_correct_cases": {
                            "pairs": len(precision[codec]), "improved": sum(delta > 0 for delta in precision[codec]),
                            "worsened": sum(delta < 0 for delta in precision[codec]),
                            "median_iou_change": float(median(precision[codec])) if precision[codec] else None},
                         "pilot_advances": passes}
    require(set(result) == set(CODECS), "Both codec arms must be replayed")
    return {"schema": "chipmunk.video-short-return-codec-audit/1", "views": len(scores), "delivered_pictures_replayed": pictures,
            "frame_and_crop_pngs_replayed": images, "freeze_sha256": digest(freeze), "media_sha256": digest(media),
            "journal_sha256": digest(journal), "gate": lock["gate"], "camera_originals": camera_sources(lock),
            "by_codec": result, "all_codecs_advance": all(panel["pilot_advances"] for panel in result.values()) if stage == "scoring" else None,
            "private_examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--stage", choices=("scoring", "experimental-report-proof"), default="scoring")
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.stage), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
