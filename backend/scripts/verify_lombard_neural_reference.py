# SPDX-License-Identifier: Apache-2.0
"""Test the existing earlier-copy picture exhibit against the frozen redraw boundaries."""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
from pathlib import Path

from app.evidence import video_reference
from scripts import build_lombard_neural_mouth as build
from scripts import review_lombard_neural_mouth as review
from scripts.video_common_sources_v89 import read, require, sha, write


def inspect(run, earlier, edited, row):
    paths = tuple(run / "deliveries" / f"{r['token']}.mov" for r in (earlier, edited))
    require(
        tuple(map(sha, paths)) == (earlier["file_sha256"], edited["file_sha256"]), "Reference pair differs"
    )
    original_event = Fraction(row["visual"]["seconds"][0])
    record, assets = video_reference.compare(
        *paths,
        include_local=False,
        timing_event_s=original_event,
        timing_model=build.retained.ASSETS / "permissions/face_landmarker.task",
        timing_short_fallback=True,
    )
    require(
        record["status"] == "measured"
        and record["affects_overall_confidence"] is False
        and record["audio"]["status"] == "measured"
        and record["audio"]["changed_samples"] == 0
        and record["audio"]["earlier"]["samples"] == row["audio"]["samples"],
        "Earlier-copy comparison lost source or matched sound support",
    )
    [finding] = record["picture"]["ranges"]
    start, end = row["event"]["native_edit_frames"]
    pts = row["visual"]["pts"]
    expected_pts = {"start_pts": pts[start], "last_pts": pts[end - 1], "end_exclusive_pts": pts[end]}
    require(
        record["picture"]["changed_frames"] == end - start
        and [finding["start_frame"], finding["end_frame_exclusive"]] == [start, end]
        and all(finding["source_ranges"][k] == expected_pts for k in ("earlier", "submitted"))
        and all(record[k]["time_base"] == row["visual"]["time_base"] for k in ("earlier", "submitted")),
        "Earlier-copy finding does not match exact authored picture ticks",
    )
    before, during, after = [finding["previews"][k] for k in ("before", "during", "after")]
    require(
        before["frame"] == start - 1
        and start <= during["frame"] < end
        and after["frame"] == end
        and not before["changed_pixels"]
        and during["changed_pixels"] > 0
        and not after["changed_pixels"],
        "Earlier-copy exhibits are outside their respective native edit roles",
    )
    referenced = set()
    for preview in (before, during, after):
        for key in ("earlier", "submitted"):
            reference = preview["assets"][key]
            name = reference["file"]
            require(
                hashlib.sha256(assets[name]).hexdigest() == reference["sha256"],
                "Picture exhibit bytes differ",
            )
            referenced.add(name)
    require(len(referenced) == 6 and set(assets) == referenced, "Six linked picture exhibits required")
    timing = record["timing"]
    require(
        timing["status"] == "unassessed"
        and timing["reason"] == "profile"
        and timing["requested_earlier_time_s"] == str(original_event)
        and timing["affects_overall_confidence"] is False,
        "Released marked-event timing reports beyond its native profile",
    )
    return {
        "edited_frames": end - start,
        "source_ranges": finding["source_ranges"],
        "previews": len(referenced),
        "marked_event_timing": timing["reason"],
    }


def run(folder, split):
    verified = folder / f"review-{split}.json"
    reviewed = read(verified)
    lock = folder / "freeze.json"
    require(reviewed["freeze_sha256"] == sha(lock), "Native delivery has no matching qualification")
    native = {r["source"]["ordinal"]: r for r in read(lock)["sources"]}
    records = review.partitions(folder, split)
    by_case = {(r["ordinal"], r["role"]): r for r in records}
    output = []
    for item in reviewed["cases"]:
        k = item["ordinal"]
        if item["role"] != build.NEURAL or item["status"] != "authored":
            continue
        result = inspect(folder, by_case[k, "original"], by_case[k, build.NEURAL], native[k])
        output.append({"ordinal": k, **result})
        print("Earlier-copy exact range", split, k, flush=True)
    write(folder / f"reference-marked-{split}.json", {"native_review_sha256": sha(verified), "cases": output})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--split", choices=("development", "held"), default="development")
    a = p.parse_args()
    run(a.run, a.split)


if __name__ == "__main__":
    main()
