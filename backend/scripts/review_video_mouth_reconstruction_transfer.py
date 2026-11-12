# SPDX-License-Identifier: Apache-2.0
"""Independently audit outcomes and decode selected native mouth evidence."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_mouth_reconstruction_transfer as transfer
from scripts.video_common_sources_v89 import read, require, sha, write


def verified_outcome(case, arm, threshold):
    finding = case["arms"][arm]["best"]
    if finding is None:
        return "unassessed"
    left, center, right = finding["medians"]
    require(finding["score"] == min(center - left, center - right), "Contrast no longer matches its observed flanks")
    if not finding["score"] > threshold:
        return "measured_no_range"
    if not (truth := case["truth_clock_s"]):
        return "ordinary_range"
    a, b, c, d = map(Fraction, [*finding["source_clock_s"], *truth])
    intersection = max(0, min(b, d) - max(a, c))
    return "correct_range" if 2 * intersection >= max(b, d) - min(a, c) else "wrong_range"


def verify_anchors(path, original_row, findings):
    table = probe_frames(path, timeout_s=15)
    expected = original_row["expected"]
    require([pts * table.tb for pts in table.pts] == [Fraction(ms, 1000) for ms in expected["picture_pts_ms"]], "Native timing differs")
    wanted = {}
    for finding in findings:
        start, end = finding["frames"]
        frames = [entry["frame"] for entry in finding["anchors"]]
        require(len(frames) == 3 and frames[0] < start <= frames[1] < end <= frames[2]
                and [entry["role"] for entry in finding["anchors"]] == ["before", "during", "after"]
                and list(map(Fraction, finding["source_clock_s"])) == [table.pts[i] * table.tb for i in (start, end)],
                "Evidence roles or half-open source times differ")
        for anchor in finding["anchors"]:
            index = anchor["frame"]
            require(anchor["rgb_sha256"] == expected["rgb_sha256"][index]
                    and anchor["pts"] == table.pts[index], "The anchor differs from its native archived frame")
            if index in wanted:
                require(wanted[index]["native_crop_xyxy"] == anchor["native_crop_xyxy"], "Crop differs for one common frame")
            wanted[index] = anchor
    frames = {}
    for frame in decode_frames(path, table, sorted(wanted), timeout_s=20):
        anchor = wanted[frame.index]
        digest = hashlib.sha256(frame.rgb).hexdigest()
        x0, y0, x1, y1 = anchor["native_crop_xyxy"]
        require(frame.pts_verified and anchor["pts"] == frame.pts and anchor["rgb_sha256"] == digest
                and 0 <= x0 < x1 <= table.width and 0 <= y0 < y1 <= table.height,
                "Replayed frame or native crop differs")
        frames[frame.index] = frame.rgb
    require(wanted.keys() == frames.keys(), "Incomplete native anchor replay")
    return frames


def verify_mask(path, row, frame, crop):
    proof = next((p for p in row["edit_proofs"] if p["frame_index"] == frame), None)
    if proof is None:
        return None
    mask_path = path.parent / Path(proof["mask_file"]).name
    require(sha(mask_path) == proof["mask_sha256"] and row["source_video_pts"][frame] == proof["pts"], "Mask file or original PTS differs")
    with Image.open(mask_path) as image:
        mask = np.array(image.convert("L"))
    require(np.all((mask == 0) | (mask == 255)) and np.count_nonzero(mask) == proof["changed_pixels"]
            and hashlib.sha256(mask).hexdigest() == proof["mask_raw_sha256"], "Native changed-pixel proof differs")
    x0, y0, x1, y1 = crop
    return {"true_changed_pixels": int(np.count_nonzero(mask)), "inside_inspection_crop": int(np.count_nonzero(mask[y0:y1, x0:x1]))}


def audit(folder):
    doc = transfer.frozen(folder)
    aggregate, cal = (read(folder / (name + ".json")) for name in ("aggregate", "calibration"))
    require(aggregate["held_sha256"] == sha(folder / "held.jsonl") and aggregate["calibration_sha256"] == sha(folder / "calibration.json")
            and aggregate["freeze_sha256"] == sha(folder / "freeze.json")
            and cal["development_sha256"] == sha(folder / "development.jsonl"), "Score journals or calibration differ")
    development = transfer.journal(folder, doc, "development")
    for arm in transfer.ARMS:
        scores = sorted((case["arms"][arm]["best"]["score"] for group in development for case in group["cases"]
                         if case["condition"] in transfer.CONTROL and case["arms"][arm]["best"] is not None), reverse=True)
        require(cal["points"][arm] == scores[2], "Ordinary development point differs")
    tallies = {condition: {arm: Counter() for arm in transfer.ARMS} for condition in transfer.ORDER}
    pictures = held_cases = verified_masks = 0
    example = None
    held = transfer.journal(folder, doc, "held")
    for observed, pin in zip(held, (pin for pin in doc["clips"] if pin["component"] in doc["held_components"]), strict=True):
        cases = transfer.inspect(doc, pin)
        require(observed["cases"][0]["arms"] == observed["cases"][-1]["arms"], "Audio-only control changed visual-only measurements")
        for recorded, (path, row, motion) in zip(observed["cases"], cases, strict=True):
            condition = recorded["condition"]
            require(recorded["sha256"] == row["file_sha256"] == sha(path) and recorded["motion_status"] == motion
                    and recorded["truth_frames"] == (row["edit_frames_half_open"] if condition in transfer.NEURAL else None),
                    "Held source condition differs from its retained construction")
            if recorded["truth_frames"]:
                require(list(map(Fraction, recorded["truth_clock_s"])) == [Fraction(row["expected"]["picture_pts_ms"][i], 1000)
                        for i in recorded["truth_frames"]], "Known native edit times differ")
            outcomes = {arm: verified_outcome(recorded, arm, cal["points"][arm]) for arm in transfer.ARMS}
            for arm, label in outcomes.items():
                tallies[condition][arm][label] += 1
            findings = [recorded["arms"][arm]["best"] for arm in transfer.ARMS if recorded["arms"][arm]["best"] is not None]
            frames = verify_anchors(path, row, findings) if findings else {}
            pictures += len(frames)
            held_cases += 1
            if condition not in transfer.NEURAL:
                continue
            finding = recorded["arms"]["motion_reconstruction"]["best"]
            if finding is None:
                continue
            center = finding["anchors"][1]
            mask = verify_mask(path, row, center["frame"], center["native_crop_xyxy"])
            verified_masks += mask is not None
            status = outcomes["motion_reconstruction"]
            rank = (status == "correct_range", condition == "liveportrait_source_time" and motion == transfer.SOURCE_CLOCK,
                    status == "wrong_range", mask is not None, finding["score"])
            if example is None or rank > example[0]:
                example = rank, recorded, finding, frames, mask, status
    require(tallies == aggregate["by_condition"], "Independent decision counts differ")
    output = {"status": "verified", "held_files_byte_checked": held_cases, "native_frames_redecoded": pictures,
              "native_mask_proofs_replayed": verified_masks, "audio_only_picture_pairs_identical": len(held),
              "aggregate_sha256": sha(folder / "aggregate.json")}
    if example is not None:
        _, case, finding, frames, mask, status = example
        exhibit = folder / "private-example"
        exhibit.mkdir()
        for anchor in finding["anchors"]:
            data = frames[anchor["frame"]]
            x0, y0, x1, y1 = anchor["native_crop_xyxy"]
            Image.fromarray(data).save(exhibit / (anchor["role"] + "-source.png"))
            Image.fromarray(data[y0:y1, x0:x1]).save(exhibit / (anchor["role"] + "-crop.png"))
        write(exhibit / "receipt.json", {"case": case["condition"], "source_sha256": case["sha256"], "known_edit_s": case["truth_clock_s"],
              "chosen": finding, "selected_outcome": status, "mask": mask})
        output["example"] = {"case": case["condition"], "known_edit_s": case["truth_clock_s"], "selected_range_s": finding["source_clock_s"],
                             "outcome": status, "native_pictures_and_crops": len(finding["anchors"]), "mask_overlap": mask}
    write(folder / "independent-audit.json", output)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)


if __name__ == "__main__":
    main()
