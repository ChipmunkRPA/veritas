# SPDX-License-Identifier: Apache-2.0
"""Independently check the whole-image source manifest and paired evaluation arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
SOURCE = PANELS / "image-whole-fresh-sources-2026-11-12.json"
PRIOR = PANELS / "image-v7-smaug-development-panel.json"
LICENSES = {f"https://creativecommons.org/licenses/{kind}/{version}"
            for kind in ("by", "by-sa") for version in ("2.0", "2.5", "3.0", "4.0")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(body):
    return hashlib.sha256(body).hexdigest()


def verify_sources(path, directory=None):
    public = read(path)
    frozen = public["frozen_record"]
    rows = public["sources"]
    frozen_bytes = (json.dumps(frozen, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(public["schema"] == "chipmunk.image-whole-fresh-source-manifest/1"
            and digest(frozen_bytes) == public["frozen_input_record"]["sha256"]
            and not public["media_included"], "Source-freeze commitment changed")
    old = read(PRIOR)["record"]
    excluded = old["inventory"] + old["items"]
    encoded = {row["sha256"] for row in excluded} | {row["input_sha256"] for row in old["items"]}
    rgb = {row["rgb_sha256"] for row in excluded}
    similar = {int(row["dhash64"], 16) for row in excluded}
    labels, families, groups = Counter(), Counter(), set()
    for row in rows:
        label, family = row["label"], row["family"]
        labels[label] += 1
        groups.add(row["group"])
        if label == "ordinary":
            require(row["license"] in LICENSES and row["publisher_delivery"] in ("original", "thumb")
                    and row["author"] and row["page"].startswith("https://commons.wikimedia.org/"), "Missing photo rights")
        else:
            families[family] += 1
            require(row["license"] == public["generated_source"]["license"] == "https://www.apache.org/licenses/LICENSE-2.0"
                    and row["url"].startswith("https://huggingface.co/datasets/Qwen/Qwen-Image-Bench/resolve/"
                                              + public["generated_source"]["revision"] + "/"), "Missing synthetic source rights")
        signature = int(row["difference_hash"], 16)
        require(row["sha256"] not in encoded and row["rgb_sha256"] not in rgb
                and all((signature ^ other).bit_count() > 4 for other in similar), "Duplicate or earlier-panel image")
        encoded.add(row["sha256"])
        rgb.add(row["rgb_sha256"])
        similar.add(signature)
    require(len(groups) == 34 and labels == {"synthetic": 18, "ordinary": 16}
            and families == {"Qwen-Image-2512": 6, "GLM-Image": 6, "FLUX.2_max": 6}, "Incomplete source groups")
    deliveries = set()
    for delivery in frozen["items"]:
        index, format_ = delivery["source_index"], delivery["format"]
        row = rows[index]
        require(format_ in {"native", "jpeg75"} and (index, format_) not in deliveries
                and (delivery["label"], delivery["family"]) == (row["label"], row["family"]), "Delivery scope changed")
        require(format_ != "native" or delivery["sha256"] == row["sha256"], "Native image is not the source file")
        if directory:
            local = (directory / delivery["path"]).resolve()
            require(local.is_relative_to(directory.resolve()) and digest(local.read_bytes()) == delivery["sha256"],
                    "Native or JPEG75 bytes differ from freeze")
        deliveries.add((index, format_))
    require(len(deliveries) == 68, "Missing native/JPEG75 pairs")
    return public


def reduction(source, cases, *, baseline_complete):
    lock = source["frozen_record"]
    point = lock["operating_point"]["raw_mean_logit"]
    expected = {f"{item['source_index']:03d}-{item['format']}": item for item in lock["items"]}
    require(len(expected) == len(cases) and {item["id"] for item in cases} == expected.keys(), "Missing scored deliveries")
    fields = ("candidate_flags", "candidate_misses", "baseline_ai", "baseline_unavailable", "baseline_original",
              "baseline_inconclusive", "baseline_record", "added_flags", "combined_flags", "added_original_conflicts")
    totals = {form: {label: Counter(dict.fromkeys(fields, 0)) for label in ("synthetic", "ordinary")}
              for form in ("native", "jpeg75")}
    added_families = {form: set() for form in totals}
    for case in cases:
        item = expected[case["id"]]
        require(item["sha256"] == case["input_sha256"], "Scored delivery is not frozen")
        logits, mean = case["crop_logits"], case["mean_logit"]
        require(len(logits) == 3 and all(math.isfinite(x) for x in [mean, *logits])
                and abs(math.fsum(logits) / 3 - mean) <= 1e-4, "Invalid candidate reduction")
        flag = mean > point
        require(flag == case["candidate_flag"], "Flag differs from inherited point")
        cell = totals[item["format"]][item["label"]]
        cell["candidate_flags"] += flag
        cell["candidate_misses"] += not flag
        if baseline_complete:
            direction = case["baseline_direction"]
            require(direction in {"ai", "unavailable", "original", "inconclusive", "record"}, "Invalid Full interpretation")
            cell[f"baseline_{direction}"] += 1
            cell["added_flags"] += flag and direction != "ai"
            cell["combined_flags"] += flag or direction == "ai"
            cell["added_original_conflicts"] += flag and direction == "original"
            if flag and direction != "ai" and item["label"] == "synthetic":
                added_families[item["format"]].add(item["family"])
    passed = baseline_complete and all(
        cell["synthetic"]["added_flags"] >= lock["criteria"]["format_minimum_added_synthetic"]
        and cell["ordinary"]["added_flags"] <= lock["criteria"]["format_maximum_additional_ordinary_groups"]
        and len(added_families[form]) >= lock["criteria"]["format_minimum_represented_generator_families"]
        for form, cell in totals.items())
    if not baseline_complete:
        totals = {form: {label: {k: cell[k] for k in fields[:2]} for label, cell in by_label.items()}
                  for form, by_label in totals.items()}
    return {"candidate_complete": True, "baseline_complete": baseline_complete, "counts": totals,
            "added_generator_families": {form: sorted(families) for form, families in added_families.items()},
            "gate_passed": passed}


def verify_candidate_crops(source, directory, journal):
    from PIL import Image, ImageOps

    cases = [json.loads(line) for line in journal.read_text().splitlines()]
    rows = {case["id"]: case for case in cases}
    require(len(cases) == len(rows) == 68, "Candidate journal is incomplete or duplicated")
    audited = 0
    for item in source["frozen_record"]["items"]:
        case = rows[f"{item['source_index']:03d}-{item['format']}"]
        require(case["input_sha256"] == item["sha256"], "Crop response belongs to a different source")
        with Image.open(directory / item["path"]) as image, ImageOps.exif_transpose(image) as upright:
            original = upright.convert("RGB")
            width, height = original.size
            short = min(original.size)
            if short > 576:
                size = tuple(round(n * (576 / short)) for n in original.size)
                resampling = Image.Resampling.LANCZOS
            elif short < 528:
                size = tuple(math.ceil(n * (528 / short)) for n in original.size)
                resampling = Image.Resampling.BICUBIC
            else:
                size, resampling = original.size, None
            rw, rh = size
            resized = original if resampling is None else original.resize(size, resampling)
            cx, cy = (rw - 528) // 2, (rh - 528) // 2
            expected = ((0, cy), (cx, cy), (rw - 528, cy)) if rw >= rh else ((cx, 0), (cx, cy), (cx, rh - 528))
            for (x, y), trace, expected_hash in zip(expected, case["traces"], case["crop_rgb_sha256"], strict=True):
                bounds = (x, y, x + 528, y + 528)
                expected_rational = [[edge * scale, denominator] for edge, scale, denominator in
                                     zip(bounds, (width, height, width, height), (rw, rh, rw, rh), strict=True)]
                require(trace["oriented_original_size"] == [width, height] and trace["resized_size"] == list(size)
                        and trace["resized_crop_xyxy"] == list(bounds) and trace["source_edge_bounds_rational"] == expected_rational
                        and trace["resampling"] == (None if resampling is None else resampling.name)
                        and digest(resized.crop(bounds).tobytes()) == expected_hash, "Crop pixels or exact source coordinates differ")
                audited += 1
    require(audited == 204, "Not every planned crop was audited")
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--candidate-journal", type=Path)
    args = parser.parse_args()
    source = verify_sources(args.source, args.directory)
    if args.results:
        result = read(args.results)
        expected = reduction(source, result["cases"], baseline_complete=result["summary"]["baseline_complete"])
        require(result["source_manifest_sha256"] == digest(args.source.read_bytes())
                and result["summary"] == expected, "Reported evaluation differs from independent replay")
        print(json.dumps(expected, sort_keys=True))
    else:
        print("Verified 34 source groups and 68 frozen native/JPEG75 deliveries.")
    if args.candidate_journal:
        require(args.directory is not None, "Native crop review requires the local frozen inputs")
        print(f"Replayed {verify_candidate_crops(source, args.directory, args.candidate_journal)} exact source crops.")


if __name__ == "__main__":
    main()
