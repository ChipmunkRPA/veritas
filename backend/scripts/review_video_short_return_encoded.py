# SPDX-License-Identifier: Apache-2.0
"""Recompute app range outcomes and decode the committed 360p evidence independently."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

from PIL import Image

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.review_video_short_return_margin import digest, read, require

POINT = .10
RATE = 25
NATIVE = [Fraction(i, RATE) for i in range(150)]
ROLES = ("before", "during", "after")
CONTROLS = ("original", "recompressed", "ordinary_light", "ordinary_blur", "ordinary_time_step")
SWAPS = {"swap_a": [67, 76], "swap_b": [75, 84]}


def exact_range(record, table):
    if record is None:
        return None
    first, end = record["start_pts"], record["end_pts"]
    require(record["time_base"] == table.time_base and first < end and first in table.pts and end in table.pts,
            "Range does not use the encoded clock")
    return tuple((stamp - table.pts[0]) * table.tb for stamp in (first, end))


def overlap(a, b):
    common = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, common) / (max(a[1], b[1]) - min(a[0], b[0]))


def verify_case(path, qualified, scored, directory):
    require(digest(path) == scored["input_sha256"] == qualified["sha256"], "Encoded input differs")
    table = probe_frames(path, timeout_s=30)
    require((table.width, table.height) == (640, 360) and not (table.truncated or table.rotation or table.interpolated)
            and [(p - table.pts[0]) * table.tb for p in table.pts] == NATIVE
            and table.to_json() == qualified["frame_table"], "Encoded clock or picture grid differs")
    packet = scored["packet"]
    assets = {(item["role"], item["asset"]): item for item in scored["images"]}
    required, anchors = {}, {}
    if packet:
        require(not packet["affects_overall_confidence"] and packet["claim_scope"] == "sampled_window"
                and len(assets) == 6 and set(assets) == {(role, label) for role in ROLES for label in ("original", "crop")},
                "Local review did not remain a separate, complete evidence packet")
        record = packet["face_return"]
        required = {row["frame_index"]: row for row in record["frame_inputs"] + record["scene_frame_inputs"]}
        anchors = {row["anchor"]["frame_index"]: row for row in packet["frames"]}
        candidate = exact_range(record, table)
        require(0 < candidate[1] - candidate[0] < Fraction(1, 2) and record["response"] > POINT
                and exact_range(packet["window"], table) == candidate, "Supplemental range exceeds its frozen limit")
        times = [NATIVE[record["anchors"][role]] for role in ROLES]
        require(times[0] < candidate[0] <= times[1] < candidate[1] <= times[2], "Picture roles disagree with range")
    else:
        require(not assets, "Frame artifacts have no committed range")
        candidate = None

    seen = 0
    matched_images = 0
    for frame in decode_frames(path, table, list(range(150)), timeout_s=60):
        i = frame.index
        raw = hashlib.sha256(frame.rgb).hexdigest()
        require(frame.pts_verified and i == seen and raw == qualified["rgb_sha256"][i],
                "Encoded picture differs from the preflight digest")
        seen += 1
        if i in required:
            require((required[i]["pts"], required[i]["decoded_rgb_sha256"]) == (frame.pts, raw),
                    "Measured frame differs from its encoded picture")
        if i not in anchors:
            continue
        row = anchors[i]
        role = row["interval_role"]
        require(row["anchor"]["pts"] == frame.pts and row["anchor"]["decoded_rgb_sha256"] == raw
                and packet["face_return"]["anchors"][role] == i, "Displayed anchor differs from measurement")
        box = row["context_region"]
        x, y, width, height = (box[k] for k in ("x", "y", "w", "h"))
        require(0 <= x < x + width <= 640 and 0 <= y < y + height <= 360, "Invalid source crop")
        for label, expected in (("original", frame.rgb), ("crop", frame.rgb[y:y + height, x:x + width])):
            file = assets[role, label]
            require(file["file"] == f"{role}-{label}.png", "Artifact path differs from roster")
            png = directory / file["file"]
            require(digest(png) == file["sha256"], "Artifact bytes differ from the API receipt")
            with Image.open(png) as image:
                rgb = image.convert("RGB")
                require(rgb.size == (expected.shape[1], expected.shape[0])
                        and rgb.tobytes() == expected.tobytes(), "Artifact pixels differ from the actual input")
            matched_images += 1
    require(seen == 150 and matched_images == len(assets), "Incomplete independent decode")
    current = exact_range(scored["incumbent"], table)
    require(candidate is None or current is None or overlap(candidate, current) < Fraction(1, 2),
            "Supplemental range duplicates the incumbent")
    return current, candidate, matched_images


def audit(folder, stage):
    freeze, media, journal = folder / "freeze.json", folder / "media.json", folder / stage / "journal.jsonl"
    lock, manifest = read(freeze), read(media)
    require(manifest["freeze_sha256"] == digest(freeze) and freeze.stat().st_mtime_ns <= media.stat().st_mtime_ns <= journal.stat().st_mtime_ns,
            "Prospective freeze, media, and score order differs")
    require(lock["review_point"] == POINT and lock["minimum_correct_iou"] == "1/2"
            and lock["edit_frame_spans"] == list(SWAPS.values()) and lock["delivery_size"] == [640, 360]
            and lock["delivery_rate_hz"] == RATE, "Fixed method differs")
    rows = manifest["media"]
    expected = {(i, condition) for i in range(4) for condition in (*SWAPS, *CONTROLS)}
    ordinary = {(row["group"], row["condition"]) for row in rows if row["condition"] != "hard_cut"}
    require(ordinary == expected and Counter(row["group"] for row in rows if row["condition"] == "hard_cut") == {0: 2, 1: 2}
            and len(rows) == len({row["sha256"] for row in rows}) == len({row["file"] for row in rows}) == 32,
            "Frozen case roster differs")
    require(all(row["truth_frames"] == SWAPS.get(row["condition"]) for row in rows), "Frozen swap label differs")
    expected_by_name = {row["file"]: row for row in rows}
    observations = [json.loads(line) for line in journal.read_text().splitlines()]
    scored_names = [row["file"] for row in observations]
    if stage == "scoring":
        require(scored_names == list(expected_by_name), "Missing or reordered evaluation case")
    else:
        require(len(scored_names) == len(set(scored_names)) == 3 and set(scored_names) <= set(expected_by_name),
                "The serving replay must cover three distinct inputs")
    counts, groups_gained = defaultdict(Counter), set()
    images, errors, overlaps, examples = 0, [], [], {}
    for scored in observations:
        name = scored["file"]
        require(scored["paid_before_analysis_and_idempotency"] and scored["decision_parity"] == (stage == "serving"),
                "Required payment or decision parity evidence missing")
        row = expected_by_name[name]
        current, added, verified_images = verify_case(folder / "media" / name, row, scored, folder / stage / Path(name).stem)
        images += verified_images
        truth = tuple(NATIVE[i] for i in row["truth_frames"]) if row["truth_frames"] else None
        kind = "swap" if truth else ("hard_cut" if row["condition"] == "hard_cut" else "ordinary")
        count = counts[kind]
        count["views"] += 1
        check = (scored["return_check"] or {}).get("status")
        require((added is not None) == (check == "localized"), "Selected review status differs from packet")
        count["assessed"] += check in {"localized", "no_distinct_range"}
        count["not_assessed"] += check not in {"localized", "no_distinct_range"}
        count["incumbent_ranges"] += current is not None
        count["added_ranges"] += added is not None
        if truth:
            old_correct = current is not None and overlap(current, truth) >= Fraction(1, 2)
            added_iou = overlap(added, truth) if added else 0
            added_correct = added_iou >= Fraction(1, 2)
            count["incumbent_correct"] += old_correct
            count["combined_correct"] += old_correct or added_correct
            count["added_wrong_ranges"] += added is not None and not added_correct
            if added_correct and not old_correct:
                count["additional_correct"] += 1
                groups_gained.add(row["group"])
                errors.extend(abs(observed - expected) for observed, expected in zip(added, truth, strict=True))
                overlaps.append(added_iou)
                if stage == "serving":
                    require(scored["report_replay"]["signed_original_replay_verified"], "Serving signed replay did not finish")
                if "positive" not in examples:
                    examples["positive"] = name
                    examples["positive_range_s"] = list(map(float, added))
                    examples["positive_truth_s"] = list(map(float, truth))
                    examples["positive_iou"] = float(added_iou)
        elif not added:
            examples.setdefault(kind, name)
    if stage == "scoring":
        assessed = {kind: str(Fraction(counts[kind]["assessed"], counts[kind]["views"])) for kind in ("swap", "ordinary", "hard_cut")}
        advances = (counts["swap"]["additional_correct"] >= 4 and len(groups_gained) >= 3
                    and all(Fraction(assessed[kind]) >= Fraction(4, 5) for kind in ("swap", "ordinary"))
                    and counts["ordinary"]["added_ranges"] <= 1 and counts["hard_cut"]["added_ranges"] == 0)
    else:
        require(counts["swap"]["additional_correct"] == 1 and counts["ordinary"]["added_ranges"] == counts["hard_cut"]["added_ranges"] == 0
                and counts["ordinary"]["views"] == counts["hard_cut"]["views"] == 1, "Serving did not repeat one distinct swap and two controls")
        assessed, advances = {}, None
    return {"schema": "chipmunk.video-short-return-encoded-audit/1", "freeze_sha256": digest(freeze),
            "media_manifest_sha256": digest(media), "journal_sha256": digest(journal), "gate": lock["gate"],
            "views": len(observations), "decoded_picture_digests_verified": len(observations) * 150,
            "frame_and_crop_pngs_verified": images, "counts": counts, "assessed_coverage": assessed,
            "distinct_gain_source_groups": len(groups_gained), "median_added_iou": float(median(overlaps)) if overlaps else None,
            "maximum_added_endpoint_error_ms": float(max(errors) * 1000) if errors else None, "pilot_advances": advances,
            "private_examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--stage", choices=("scoring", "serving"), default="scoring")
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.stage), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
