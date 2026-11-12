# SPDX-License-Identifier: Apache-2.0
"""Verify held mouth recurrence results and the source-compatible neural subset."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

from scripts import review_video_lipsync_local_delay_transfer as audio
from scripts import review_video_mouth_reconstruction_transfer as replay
from scripts import video_mouth_pose_recurrence_transfer as study
from scripts.video_common_sources_v89 import read, require, sha, write


def compatible_source(path, row, original, original_row):
    """Reopen both actual picture streams, PCM and all 20 authored edit masks."""
    require(row["motion_compatible"] and row["neural_provenance_verified"], "Compatible source record differs")
    pcm, start, discarded = audio.source_pcm(path, row["file_sha256"])
    reference, original_start, original_discarded = audio.source_pcm(original, original_row["file_sha256"])
    require(np.array_equal(pcm, reference) and len(pcm) == row["expected"]["audio_samples"] == 47648
            and start == original_start == 0 and not (discarded or original_discarded)
            and audio.sha(pcm.tobytes()) == row["expected"]["pcm_s16_sha256"] == original_row["expected"]["pcm_s16_sha256"],
            "Compatible redraw changed original soundtrack or clock")
    first, ai = [study.panel.inspect_frames(media, data["expected"], full=True)
                 for media, data in ((original, original_row), (path, row))]
    require(first[1] == ai[1] == tuple(range(75)), "Compatible clips require all native pictures")
    before = study.panel.exact_rgb(original, *first, original_row["expected"])
    after = study.panel.exact_rgb(path, *ai, row["expected"])
    proof_by_frame = {proof["frame_index"]: proof for proof in row["edit_proofs"]}
    changed_frames, changed_pixels = [], 0
    for (a, _), (b, _) in zip(before, after, strict=True):
        require(a.index == b.index and a.pts * first[0].tb == b.pts * ai[0].tb, "Compatible picture clocks differ")
        mask = np.any(a.rgb != b.rgb, axis=2)
        n = int(mask.sum())
        if n:
            changed_frames.append(a.index)
            changed_pixels += n
            proof = proof_by_frame[a.index]
            png = path.parent / Path(proof["mask_file"]).name
            require(n == proof["changed_pixels"] and sha(png) == proof["mask_sha256"], "Authored pixel mask differs")
            with Image.open(png) as image:
                require(np.array_equal(np.asarray(image.convert("L")), np.uint8(mask) * 255),
                        "Actual changed-pixel region differs from its proof")
    require(changed_frames == list(range(*row["edit_frames_half_open"])) == sorted(proof_by_frame)
            and changed_pixels == row["changed_native_pixels"], "Compatible edits differ from source truth")
    return len(changed_frames), len(first[1]) + len(ai[1])


def result(record, point, truth, times, arm):
    if record is None:
        return "unassessed"
    begin, end = record["frames"]
    a, b, c = [frame["frame"] for frame in record["anchors"]]
    require(a < begin <= b < end <= c and list(map(Fraction, record["source_clock_s"])) == [times[begin], times[end]],
            "Review interval or picture role differs from source")
    if arm != "direct":
        first, last, references = record["edge_distances"]
        samples = record["responses"]
        require(median(samples) == record["score"] and min(first, last)-references in samples
                and len(samples) == record["matched_triplets"], "Mouth recurrence arithmetic differs")
        require(min(times[b]-times[a], times[c]-times[b]) >= Fraction(4, 25), "Reference pictures are too close to the target")
        if arm == "pose_matched":
            require(max(record["pose_distances"]) <= study.SPEC["pose_distance_max"], "Reference pose did not meet the bound")
    if record["score"] <= point:
        return "measured_no_range"
    if not truth:
        return "false_range"
    start, stop = map(Fraction, truth)
    overlap = max(Fraction(), min(stop, times[end]) - max(start, times[begin]))
    union = max(stop, times[end]) - min(start, times[begin])
    return "correct_range" if 2*overlap >= union else "misplaced_range"


def audit(folder, prior):
    frozen, calibration, runtime, aggregate = (read(folder / name) for name in
                                                ("freeze.json", "calibration.json", "runtime.json", "aggregate.json"))
    require(frozen == study.freeze(prior) and calibration["freeze_sha256"] == sha(folder / "freeze.json")
            and runtime == aggregate["runtime"] and runtime["calibration_sha256"] == sha(folder / "calibration.json")
            and runtime["held_sha256"] == sha(folder / "held.jsonl")
            and (folder / "calibration.json").stat().st_mtime_ns <= (folder / "held.jsonl").stat().st_mtime_ns,
            "Sealed score or source files differ")
    require(len(calibration["ordinary"]) == 56, "Expected 56 ordinary development views")
    for arm in study.ARMS:
        measured = sorted(row["arms"][arm] for row in calibration["ordinary"] if row["arms"][arm] is not None)
        require(len(measured) >= 42 and max(0., measured[-3]) == calibration["points"][arm] == aggregate["points"][arm],
                "Development point differs from the frozen rule")
    doc = read(prior / "freeze.json")
    roster, reference = study.previous.roster(prior, doc), study.previous.baselines(prior)
    rows = study.panel.prior.jsonl(folder / "held.jsonl")
    seen, decisions, compatible = set(), Counter(), Counter()
    results_by_case = {}
    crops = native_pictures = mask_count = compatible_count = audio_only = 0
    example = None
    for row in rows:
        bank, index, condition = (row[key] for key in ("bank", "index", "condition"))
        key = bank, index, condition
        component, cases = roster[bank, index]
        path, source = cases[condition]
        old = reference[key]
        require(key not in seen and row["component"] == component and row["source_sha256"] == sha(path) == source["file_sha256"]
                and row["arms"]["direct"] == old["arms"]["direct"], "Held source or matched direct comparator differs")
        seen.add(key)
        times = [Fraction(i, 1000) for i in source["expected"]["picture_pts_ms"]]
        edit = source["edit_frames_half_open"] if source.get("neural_provenance_verified") or source.get("known_ai_provenance") else None
        truth = [str(times[j]) for j in edit] if edit else None
        require(row["truth"] == old["truth"] == truth and row["compatible"] == bool(source.get("motion_compatible")),
                "Native truth or visible-motion qualification differs")
        if row["compatible"]:
            masks, pictures = compatible_source(path, source, *cases["original"])
            compatible_count += 1
            mask_count += masks
            native_pictures += pictures
        results_by_case[key] = row["arms"]
        cohort = condition if bank == "delivery" else bank
        for arm, rec in row["arms"].items():
            status = result(rec, aggregate["points"][arm], truth, times, arm)
            decisions[cohort, arm, status] += 1
            if row["compatible"]:
                compatible[arm, status] += 1
            if arm == "direct" or rec is None or status == "measured_no_range" and not row["compatible"]:
                continue
            pictures = replay.verify_anchors(path, source, [rec])
            for anchor in rec["anchors"]:
                x0, y0, x1, y1 = anchor["native_crop_xyxy"]
                native = pictures[anchor["frame"]][y0:y1, x0:x1]
                require(study.panel.raw_digest(native) == anchor["crop_rgb_sha256"], "Selected native crop differs")
                crops += 1
            if arm == "pose_matched" and truth and status in {"correct_range", "misplaced_range"}:
                rank = (status == "correct_range", row["compatible"], rec["score"])
                if example is None or rank > example[0]:
                    example = rank, rec, truth, pictures
    for bank, index, condition in seen:
        if bank == "grid" and condition == "audio_delay_320ms":
            original = results_by_case[bank, index, "original"]
            require(results_by_case[bank, index, condition] == original, "Audio-only change altered a picture check")
            audio_only += 1
    expected = {(bank, arm, status): n for bank, arms in aggregate["outcomes"].items()
                for arm, statuses in arms.items() for status, n in statuses.items()}
    compatible_expected = {(arm, status): n for arm, statuses in aggregate["compatible"].items() for status, n in statuses.items()}
    require(seen == set(reference) and len(seen) == 160 and decisions == expected and compatible == compatible_expected
            and compatible_count == 7 and audio_only == 8, "Public held outcomes differ")
    receipt = {"status": "verified", "source_files": len(seen), "recomputed_decisions": len(seen) * 3,
               "matched_native_crops": crops, "compatible_pairs_redecoded": compatible_count,
               "compatible_picture_stream_frames": native_pictures, "compatible_pcm_pairs_identical": compatible_count,
               "compatible_masks_rebuilt": mask_count, "visual_scores_on_audio_only_pairs_identical": audio_only,
               "aggregate_sha256": sha(folder / "aggregate.json")}
    if example:
        rank, record, truth, pixels = example
        target = folder / "private-example"
        target.mkdir(mode=0o700, exist_ok=True)
        for anchor in record["anchors"]:
            frame = pixels[anchor["frame"]]
            x0, y0, x1, y1 = anchor["native_crop_xyxy"]
            Image.fromarray(frame).save(target / f"{anchor['role']}-picture.png")
            Image.fromarray(frame[y0:y1, x0:x1]).save(target / f"{anchor['role']}-crop.png")
        receipt["anonymous_example"] = {"correct": rank[0], "source_compatible": rank[1], "truth": truth,
            "selected": record["source_clock_s"], "iou": str(study.review.overlap(record["source_clock_s"], truth)),
            "anchors_s": [str(frame["pts"]*Fraction(record["time_base"])) for frame in record["anchors"]]}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, default=study.previous.PRIOR)
    args = parser.parse_args()
    receipt = audit(args.folder, args.prior)
    write(args.folder / "independent-audit.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
