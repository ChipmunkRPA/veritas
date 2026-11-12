# SPDX-License-Identifier: Apache-2.0
"""Replay H.264 source clocks, selected crops and face-return decisions."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from PIL import Image

from scripts import video_face_six_picture_transfer as study

common, require = study.common, study.require


def intersection(a, b):
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return Fraction(max(0, hi-lo), max(a[1], b[1])-min(a[0], b[0]))


def selection(records):
    earlier = []
    for arm in study.ARMS:
        finding = records[arm]
        if finding is None:
            continue
        start, end = finding["frames"]
        require(0 <= start < end < study.N, "Selected interval is outside native pictures")
        if arm == "coarse":
            flag = finding["score"] > study.trial.prior.POINT
        else:
            before, middle, after = finding["support"]
            require(max(before) < start <= min(middle) <= max(middle) < end <= min(after)
                    and [finding["anchors"][role]["index"] for role in ("before", "during", "after")]
                    == [before[-1], middle[len(middle)//2], after[0]]
                    and min(finding["scene_correlations"]) >= .55, "Face or scene support differs")
            if arm == "consecutive":
                support = before + middle + after
                require(all(b-a == 2 for a, b in zip(support, support[1:])) and len(before) == len(after) == 4
                        and len(middle) in (3, 4) and len(middle) == len(finding["minimum_each_center"])
                        and min(finding["minimum_each_center"]) == finding["score"], "Contiguous evidence differs")
            flag = finding["score"] > .10 and end-start < Fraction(25, 2) and all(
                intersection(finding["frames"], interval) < Fraction(1, 2) for interval in earlier)
        require(flag == finding["selected"], "Supplemental range routing differs")
        if flag and arm in ("coarse", "eight_hz"):
            earlier.append(finding["frames"])
    return earlier


def source_pixels(path, media, arms, examples):
    table = common.probe_frames(path, timeout_s=20)
    require(common.digest(path) == media["sha256"] and table.to_json() == media["clock"], "Encoded media binding differs")
    codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
                                    "-of", "csv=p=0", str(path)], timeout=10, text=True).strip()
    require(codec == "h264,yuv420p" and (table.width, table.height) == study.DIMENSIONS, "Actual codec or image shape differs")
    targets = defaultdict(list)
    for arm, finding in arms.items():
        if arm != "coarse" and finding is not None and finding["selected"]:
            for label, anchor in finding["anchors"].items():
                targets[anchor["index"]].append((arm, label, anchor))
    count = crops = 0
    for decoded in common.decode_frames(path, table, range(study.N), timeout_s=45):
        pts = decoded.pts
        index = decoded.index
        rgb = decoded.rgb
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        require(decoded.pts_verified and index == count and pts == table.pts[index]
                and (pts-table.pts[0])*table.tb == Fraction(index, 25)
                and digest == media["rgb_sha256"][index], "Native decoded picture or exact clock differs")
        for arm, role, item in targets[index]:
            x, y, w, h = item["xywh"]
            require(0 <= x < x+w <= table.width and 0 <= y < y+h <= table.height, "Crop is outside its source")
            crop = rgb[y:y+h, x:x+w]
            require(item["pts"] == pts and item["rgb_sha256"] == digest
                    and item["crop_sha256"] == hashlib.sha256(crop.tobytes()).hexdigest(), "Selected crop bytes differ")
            crops += 1
            if arm in examples:
                Image.fromarray(rgb).save(examples[arm] / f"{role}-picture.png")
                Image.fromarray(crop).save(examples[arm] / f"{role}-crop.png")
        count += 1
    require(count == study.N, "Every actual delivered picture must be reopened")
    return count, crops


def authoring(folder, archive, plan, media):
    """Rebuild every held swap and hard-cut file from verified source pictures."""
    checked, source_pictures, shifted = 0, 0, 0
    all_media = {(row["group"], row["condition"]): row for row in media}
    pair = []
    with TemporaryDirectory(prefix="private-verify-", dir=folder) as scratch:
        for group, pin in enumerate(plan["roster"]):
            if pin["split"] != "held":
                continue
            index, size = pin["index"], tuple(pin["size"])
            doc = common.read(common.DATA / f"group-{index:03d}-complete.json")
            source = archive / "video-native-inserts-v95" / f"group-{index:03d}"
            cases = {row["condition"]: row for row in doc["cases"]}
            base = cases["original"]
            plain = study.encoded.native_frames(source / base["file"], base, size, range(study.N))
            source_pictures += len(plain)
            for condition in study.SWAPS:
                record = all_media[group, condition]
                case = cases[condition[:-2]]
                start, end = study.SPANS[int(condition[-1])]
                indices = range(start, end)
                require(record["truth"] == [start, end] and all(base["expected_rgb_sha256"][i] != case["expected_rgb_sha256"][i]
                        for i in indices), "Known authored source pictures or bounds differ")
                alternate = study.encoded.native_frames(source / case["file"], case, size, indices)
                intended = dict(plain)
                intended.update(alternate)
                output = Path(scratch) / f"r{group}-{condition}.mp4"
                study.encoded.encode(output, frames=(intended[i] for i in range(study.N)), crf=23)
                require(common.digest(output) == record["sha256"], "Encoded replay of six authored pictures differs")
                checked += 1
                source_pictures += len(alternate)
                shifted += len(alternate)
            pair.append(plain)
            if len(pair) == 2:
                for phase, (original, alternate) in enumerate((pair, pair[::-1])):
                    record = all_media[group-1+phase, "hard_cut"]
                    require(record["truth"] is None, "Genuine cuts are ordinary controls")
                    a, b = study.SPANS[phase]
                    output = Path(scratch) / f"r{group}-{phase}-cut.mp4"
                    study.encoded.encode(output, frames=(alternate[i] if a <= i < b else original[i] for i in range(study.N)), crf=23)
                    require(common.digest(output) == record["sha256"], "Encoded whole-picture cut reconstruction differs")
                    checked += 1
                pair.clear()
    return checked, source_pictures, shifted


def audit(folder, archive):
    frozen = study.frozen(folder, archive)
    require(frozen["roster"] == study.roster(archive, study.bindings(archive)[1]), "Known source-disjoint roster differs")
    aggregate, manifest = (common.read(folder / name) for name in ("aggregate.json", "media.json"))
    rows = study.journal(folder, "held")
    media = [row for row in manifest["media"] if row["split"] == "held"]
    require(len(rows) == len(media) == 40 and len({row["sha256"] for row in media}) == 40
            and manifest["freeze_sha256"] == common.digest(folder / "freeze.json")
            and all(common.read(folder / f"{role}-seal.json")["media_sha256"] == common.digest(folder / "media.json")
                    for role in ("development", "held")), "Held media journal or freeze differs")
    counts = {arm: defaultdict(Counter) for arm in (*study.ARMS, "current_combined")}
    actual, crops = 0, 0
    examples = {}
    for row, clip in zip(rows, media, strict=True):
        require(all(row[key] == clip[key] for key in ("file", "group", "condition", "sha256", "truth")), "Held source row differs")
        truth = clip["truth"]
        outcomes = row["arms"]
        condition = "swap" if truth else ("hard_cut" if clip["condition"] == "hard_cut" else "ordinary")
        old = selection(outcomes)
        base_correct = bool(truth and any(intersection(interval, truth) >= Fraction(1, 2) for interval in old))
        counts["current_combined"][condition].update({"views": 1, "correct": base_correct, "flagged": bool(old),
            "assessed": any(outcomes[arm] is not None for arm in ("coarse", "eight_hz"))})
        selected_examples = {}
        for arm, outcome in outcomes.items():
            marked = bool(outcome and outcome["selected"])
            correct = bool(marked and truth and intersection(outcome["frames"], truth) >= Fraction(1, 2))
            counts[arm][condition].update({"views": 1, "assessed": outcome is not None, "flagged": marked, "correct": correct,
                "adds_correct": correct and not base_correct, "wrong_range": bool(marked and truth and not correct)})
            if correct and not base_correct and arm in ("consecutive", "dense_rank") and arm not in examples:
                examples[arm] = {"known_extent_s": [str(Fraction(i,25)) for i in truth],
                                 "selected_extent_s": [str(Fraction(i,25)) for i in outcome["frames"]],
                                 "iou": str(intersection(outcome["frames"], truth)),
                                 "anchor_s": [str(Fraction(outcome["anchors"][role]["index"],25))
                                              for role in ("before", "during", "after")]}
                target = folder / f"private-example-{arm}"
                target.mkdir(exist_ok=True, mode=0o700)
                selected_examples[arm] = target
        seen, count = source_pixels(folder / "media" / clip["file"], clip, outcomes, selected_examples)
        actual += seen
        crops += count
    require(counts == aggregate["held"]["strata"], "Independent decisions differ")
    authored, source_pictures, inserts = authoring(folder, archive, frozen, media)
    require(authored == 20 and inserts == 96 and actual == 4800, "Known picture reconstructions are incomplete")
    return {"status": "verified", "actual_h264_held_files": len(rows), "actual_decoded_held_pictures": actual,
            "native_source_pictures_reopened_for_authoring": source_pictures, "exact_encoded_source_reconstructions": authored,
            "authored_swapped_native_pictures": inserts, "replayed_selected_crops": crops,
            "recomputed_arm_decisions": len(rows) * len(study.ARMS), "examples": examples,
            "aggregate_sha256": common.digest(folder / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=study.ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    receipt = audit(args.folder, args.archive)
    common.write(args.folder / "independent-audit.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
