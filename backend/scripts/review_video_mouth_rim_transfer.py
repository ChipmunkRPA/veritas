# SPDX-License-Identifier: Apache-2.0
"""Recompute the held mouth-rim outcomes and replay native evidence without detector inference."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from PIL import Image

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_mouth_reconstruction_transfer as panel
from scripts import video_mouth_rim_transfer as transfer
from scripts.review_video_mouth_reconstruction_transfer import verify_mask
from scripts.video_common_sources_v89 import read, require, sha, write


def outcome(case: dict, arm: str, point: float) -> str:
    reading = case["arms"][arm]
    chosen = reading["best"]
    if chosen is None:
        return "unassessed"
    lower, middle, upper = chosen["medians"]
    require(all(math.isfinite(value) for value in (lower, middle, upper))
            and chosen["score"] == min(middle - lower, middle - upper), "Invalid recorded flank comparison")
    if chosen["score"] <= point:
        return "measured_no_range"
    truth = case["truth_clock_s"]
    if truth is None:
        return "ordinary_range"
    start, stop = map(Fraction, chosen["source_clock_s"])
    authored_start, authored_stop = map(Fraction, truth)
    shared = min(stop, authored_stop) - max(start, authored_start)
    combined = max(stop, authored_stop) - min(start, authored_start)
    return "correct_range" if 2 * shared >= combined else "wrong_range"


def check_source(path, row, case):
    condition = row["condition"]
    truth = row["edit_frames_half_open"] if condition in panel.NEURAL else None
    expected = row["expected"]
    clock = [Fraction(ms, 1000) for ms in expected["picture_pts_ms"]]
    require(case["condition"] == condition and case["sha256"] == row["file_sha256"] == sha(path)
            and case["truth_frames"] == truth and (
                case["truth_clock_s"] is None if truth is None else list(map(Fraction, case["truth_clock_s"])) == [
                    clock[index] for index in truth]
            ), "Recorded case differs from its source binding")
    return expected, clock


def check_evidence(path, expected, clock, chosen):
    table = probe_frames(path, timeout_s=15)
    require([pts * table.tb for pts in table.pts] == clock and not table.interpolated and not table.truncated,
            "Decoded clock differs from the native construction")
    wanted = set()
    for record in chosen:
        start, end = record["frames"]
        before, during, after = record["anchors"]
        require([item["role"] for item in (before, during, after)] == ["before", "during", "after"]
                and before["frame"] < start <= during["frame"] < end <= after["frame"]
                and list(map(Fraction, record["source_clock_s"])) == [clock[start], clock[end]]
                and record["time_base"] == table.time_base, "Evidence ranges do not bind the selected native interval")
        radius = record["radius_index"]
        require(radius is None or type(radius) is int and 0 <= radius < len(transfer.RADII), "Invalid mouth ellipse")
        require((end - start) // 2 in transfer.WIDTHS, "Invalid evaluated range width")
        for anchor in (before, during, after):
            frame = anchor["frame"]
            x0, y0, x1, y1 = anchor["native_crop_xyxy"]
            require(frame in panel.POSITIONS and anchor["pts"] == table.pts[frame]
                    and anchor["rgb_sha256"] == expected["rgb_sha256"][frame]
                    and 0 <= x0 < x1 <= table.width and 0 <= y0 < y1 <= table.height,
                    "Source picture binding or crop differs")
            wanted.add(frame)
    frames = {}
    for frame in decode_frames(path, table, sorted(wanted), timeout_s=20):
        require(frame.pts_verified and frame.pts == table.pts[frame.index]
                and hashlib.sha256(frame.rgb).hexdigest() == expected["rgb_sha256"][frame.index],
                "Re-decoded evidence frame differs")
        frames[frame.index] = frame.rgb
    require(wanted == frames.keys(), "A requested source frame was omitted")
    return frames


def review(folder: Path, prior: Path) -> None:
    frozen = transfer.frozen(folder)
    aggregate, cal = (read(folder / f"{part}.json") for part in ("aggregate", "calibration"))
    require(frozen["prior_sha256"] == sha(prior / "freeze.json")
            and aggregate["freeze_sha256"] == cal["freeze_sha256"] == sha(folder / "freeze.json")
            and aggregate["held_sha256"] == sha(folder / "held.jsonl")
            and aggregate["calibration_sha256"] == sha(folder / "calibration.json")
            and cal["development_sha256"] == sha(folder / "development.jsonl"), "Study commitments differ")
    development = transfer.journal(folder, frozen, "development")
    for arm in transfer.ARMS:
        controls = [case["arms"][arm]["best"]["score"] for group in development for case in group["cases"]
                    if case["condition"] in panel.CONTROL and case["arms"][arm]["best"] is not None]
        require(cal["points"][arm] == sorted(controls, reverse=True)[2], "Ordinary development point differs")

    checked = Counter()
    tallies = {condition: {arm: Counter() for arm in transfer.ARMS} for condition in panel.ORDER}
    transitions = {arm: Counter() for arm in transfer.ARMS[1:]}
    source_time = {arm: Counter() for arm in transfer.ARMS}
    groups = {arm: set() for arm in transfer.ARMS}
    added = {arm: set() for arm in transfer.ARMS}
    example = None
    for split in ("development", "held"):
        rows = transfer.journal(folder, frozen, split)
        prior_rows = panel.jsonl(prior / f"{split}.jsonl")
        pins = [pin for pin in frozen["clips"] if pin["component"] in frozen[f"{split}_components"]]
        for group, old_group, pin in zip(rows, prior_rows, pins, strict=True):
            require(group["component"] == old_group["component"] == pin["component"]
                    and group["ordinal"] == old_group["ordinal"] == pin["ordinal"], "Source group partition differs")
            require(group["cases"][0]["arms"] == group["cases"][-1]["arms"], "Audio-only edit changed a visual score")
            for case, old, (path, row, motion) in zip(group["cases"], old_group["cases"], panel.inspect(frozen, pin), strict=True):
                expected, clock = check_source(path, row, case)
                require(motion == case["motion_status"] and old["sha256"] == case["sha256"], "Motion or prior source differs")
                direct = case["arms"]["direct"]
                old_direct = old["arms"]["direct"]
                best = direct["best"]
                require(direct["measured_centres"] == old_direct["measured_centres"] and
                        (None if best is None else {k: v for k, v in best.items() if k != "radius_index"}) == old_direct["best"],
                        "Replayed direct comparison differs from the earlier native run")
                checked["direct_responses_matched"] += 1
                if split != "held":
                    continue
                checked["held_files_verified"] += 1
                decisions = {arm: outcome(case, arm, cal["points"][arm]) for arm in transfer.ARMS}
                for arm, decision in decisions.items():
                    tallies[case["condition"]][arm][decision] += 1
                    if case["condition"] == "liveportrait_source_time" and motion == panel.SOURCE_CLOCK:
                        source_time[arm][decision] += 1
                    if decision == "correct_range":
                        groups[arm].add(group["component"])
                        if decisions["direct"] != "correct_range":
                            added[arm].add(group["component"])
                    if arm != "direct":
                        transitions[arm][f"{decisions['direct']} -> {decision}"] += 1
                findings = [case["arms"][arm]["best"] for arm, decision in decisions.items()
                            if decision in {"correct_range", "wrong_range", "ordinary_range"}]
                frames = check_evidence(path, expected, clock, findings) if findings else {}
                checked["native_frames_redecoded"] += len(frames)
                if case["condition"] not in panel.NEURAL:
                    continue
                for arm, decision in decisions.items():
                    if decision not in {"correct_range", "wrong_range"}:
                        continue
                    finding = case["arms"][arm]["best"]
                    center = finding["anchors"][1]
                    mask = verify_mask(path, row, center["frame"], center["native_crop_xyxy"])
                    checked["native_mask_proofs_replayed"] += mask is not None
                    rank = (decision == "correct_range", arm == "joint_rim",
                            motion == panel.SOURCE_CLOCK, arm != "direct", mask is not None)
                    if example is None or rank > example[0]:
                        example = rank, arm, case, finding, frames, mask, decision

    totals = {arm: sum((row[arm] for row in tallies.values()), Counter()) for arm in transfer.ARMS}
    direct, joint = totals["direct"], totals["joint_rim"]
    advances = (joint["correct_range"] >= direct["correct_range"] + 2
                and len(added["joint_rim"]) >= 2 and source_time["joint_rim"]["correct_range"] >= 1
                and joint["ordinary_range"] <= direct["ordinary_range"] + 2)
    require(tallies == aggregate["by_condition"] and totals == aggregate["totals"]
            and transitions == aggregate["paired_transitions"]
            and source_time == aggregate["expression_source_time_compatible_subset"]
            and {arm: len(group) for arm, group in groups.items()} == aggregate["components_with_correct_ranges"]
            and {arm: len(group) for arm, group in added.items()} == aggregate["components_with_added_ranges"]
            and advances == aggregate["pilot_advances"], "Independent paired counts differ")
    receipt = {"status": "verified", "aggregate_sha256": sha(folder / "aggregate.json"), "counts": checked}
    if example is not None:
        _, arm, case, finding, frames, mask, decision = example
        exhibit = folder / "private-example"
        exhibit.mkdir()
        for anchor in finding["anchors"]:
            data = frames[anchor["frame"]]
            x0, y0, x1, y1 = anchor["native_crop_xyxy"]
            Image.fromarray(data).save(exhibit / f"{anchor['role']}-source.png")
            Image.fromarray(data[y0:y1, x0:x1]).save(exhibit / f"{anchor['role']}-crop.png")
        receipt["example"] = {"arm": arm, "condition": case["condition"], "authored": case["truth_clock_s"],
                              "selected": finding["source_clock_s"], "outcome": decision,
                              "native_pictures_and_crops": len(finding["anchors"]), "mask_overlap": mask}
        write(exhibit / "receipt.json", {**receipt["example"], "source_sha256": case["sha256"], "finding": finding})
    write(folder / "independent-audit.json", receipt)
    print(receipt, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()
    review(args.folder, args.prior)


if __name__ == "__main__":
    main()
