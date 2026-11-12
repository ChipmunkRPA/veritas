# SPDX-License-Identifier: Apache-2.0
"""Replay frozen mouth-detail decisions, exact pictures and aligned source PCM."""

from __future__ import annotations

import argparse
import math
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median

from PIL import Image

from scripts import review_video_lipsync_local_delay_transfer as audio
from scripts import review_video_mouth_pose_recurrence_transfer as earlier_review
from scripts import review_video_mouth_reconstruction_transfer as pictures
from scripts import video_mouth_temporal_detail as study
from scripts.video_common_sources_v89 import read, require, sha, write


def check_selection(row, times, arm):
    frames, shots, edges, anchors = (row[k] for k in ("frames", "shots", "edges", "anchors"))
    require(len(frames) == len(shots) == len(anchors) == len(edges)+1, "Sample roster differs")
    scores = []
    for edge in edges:
        if edge is None:
            scores.append(None)
            continue
        require(len(edge) == 2, "Both tracking directions must contribute")
        for view in edge:
            require(len(view["pixels"]) == 2 and min(view["pixels"]) >= 48
                    and all(-1 <= value <= 1 for band in ("fine", "broad") for value in view[band]),
                    "Paired correlation lacks the frozen pixel support")
        forward, reverse = edge
        f = ((forward["fine"][1]-forward["fine"][0]) + (reverse["fine"][1]-reverse["fine"][0])) / 2
        c = ((forward["broad"][1]-forward["broad"][0]) + (reverse["broad"][1]-reverse["broad"][0])) / 2
        scores.append(f if arm == "fine" else f - c)
    series = [None] + [None if a is None or b is None else min(a, b) for a, b in zip(scores, scores[1:])] + [None]
    eligible = []
    for width in (5, 7, 9, 11):
        for begin in range(4, len(frames) - width - 3):
            end = begin + width
            if shots[begin-3] != shots[end+2]:
                continue
            left = [i for i in range(begin-3, begin) if series[i] is not None]
            middle = [i for i in range(begin, end) if series[i] is not None]
            right = [i for i in range(end, end+3) if series[i] is not None]
            if min(len(left), len(right)) < 2 or len(middle) < math.ceil(.8 * width):
                continue
            a, b, c = [median(series[i] for i in part) for part in (left, middle, right)]
            center = min(middle, key=lambda i: (abs(series[i]-b), i))
            eligible.append((b - max(a, c), begin, end, [left[-1], center, right[0]], [a, b, c]))
    record = row["arms"][arm]
    if not eligible:
        require(record is None, "An unavailable view reported a selected range")
        return
    require(record is not None, "An eligible view lost its selected range")
    score, a, b, samples, medians = max(eligible, key=lambda item: item[0])
    require(record["score"] == score and record["frames"] == [frames[a], frames[b]]
            and record["medians"] == medians
            and list(map(Fraction, record["source_clock_s"])) == [times[frames[a]], times[frames[b]]]
            and record["anchors"] == [{**anchors[i], "role": role} for role, i in zip(study.ROLES, samples, strict=True)],
            "Selected mouth-detail range differs from the held measurements")


def checked_pcm(path, data, record):
    pcm, origin, discarded = audio.source_pcm(path, data["file_sha256"])
    expected = data["expected"]
    if "pcm_s16_sha256" in expected:
        require(audio.sha(pcm.tobytes()) == expected["pcm_s16_sha256"], "Source PCM differs from its native receipt")
    begin, end = map(Fraction, record["source_clock_s"])
    a, b = (math.ceil((t-origin)*audio.RATE) for t in (begin, end))
    require(not discarded and 0 <= a < b <= len(pcm), "Measured video range does not align with continuous source audio")
    return pcm[a:b], {
        "sample_rate_hz": audio.RATE, "interval_source_s": [str(origin + Fraction(i, audio.RATE)) for i in (a, b)],
        "source_pcm_samples": [a, b], "sample_count": b-a, "pcm_sha256": audio.sha(pcm[a:b].tobytes()),
    }


def audit(folder, prior):
    frozen, calibration, runtime, aggregate = (read(folder/name) for name in
                                                ("freeze.json", "calibration.json", "runtime.json", "aggregate.json"))
    require(frozen == study.lock(prior) and calibration["freeze_sha256"] == sha(folder/"freeze.json")
            and runtime["held_sha256"] == sha(folder/"held.jsonl")
            and runtime["calibration_sha256"] == sha(folder/"calibration.json")
            and (folder/"calibration.json").stat().st_mtime_ns <= (folder/"held.jsonl").stat().st_mtime_ns,
            "Frozen source, calibration or held records differ")
    for arm in study.ARMS:
        values = sorted(row["scores"][arm] for row in calibration["ordinary"] if row["scores"][arm] is not None)
        require(len(values) >= 42 and max(0., values[-3]) == aggregate["points"][arm] == calibration["points"][arm],
                "The development point differs")
    baseline = study.prior.baselines(prior)
    roster = study.prior.roster(prior, read(prior/"freeze.json"))
    rows = study.panel.prior.jsonl(folder/"held.jsonl")
    counts, compatible, scores = Counter(), Counter(), {}
    visited = set()
    crop_count = compatible_pictures = compatible_masks = pcm_checked = 0
    example = None
    for row in rows:
        bank, index, condition = (row[key] for key in ("bank", "index", "condition"))
        key = bank, index, condition
        path, source = roster[bank, index][1][condition]
        old = baseline[key]
        require(key not in visited and row["source_sha256"] == sha(path) == source["file_sha256"]
                and row["component"] == roster[bank, index][0] and row["arms"]["direct"] == old["arms"]["direct"]
                and row["truth"] == old["truth"] and row["compatible"] == old["source_time_compatible"],
                "Held media roster or comparison differs")
        visited.add(key)
        times = [Fraction(ms, 1000) for ms in source["expected"]["picture_pts_ms"]]
        for arm in study.ARMS:
            check_selection(row, times, arm)
        scores[key] = {arm:row["arms"][arm] for arm in study.ARMS}
        if row["compatible"]:
            original, original_record = roster[bank, index][1]["original"]
            masks, reopened = earlier_review.compatible_source(path, source, original, original_record)
            compatible_masks += masks
            compatible_pictures += reopened
        group = condition if bank == "delivery" else bank
        for arm, record in row["arms"].items():
            truth = row["truth"]
            outcome = "unassessed" if record is None else "measured_no_range"
            if record is not None and record["score"] > aggregate["points"][arm]:
                a, b = [times[i] for i in record["frames"]]
                if truth:
                    c, d = map(Fraction, truth)
                    overlap = max(Fraction(), min(b, d) - max(a, c))
                    union = max(b, d) - min(a, c)
                    outcome = "correct_range" if 2*overlap >= union else "misplaced_range"
                else:
                    outcome = "false_range"
            counts[group, arm, outcome] += 1
            if row["compatible"]:
                compatible[arm, outcome] += 1
            if arm != study.PRIMARY or record is None or (outcome == "measured_no_range" and not row["compatible"]):
                continue
            source_pixels = pictures.verify_anchors(path, source, [record])
            for item in record["anchors"]:
                x0, y0, x1, y1 = item["native_crop_xyxy"]
                patch = source_pixels[item["frame"]][y0:y1, x0:x1]
                require(study.panel.raw_digest(patch) == item["crop_rgb_sha256"], "Selected original crop differs")
                crop_count += 1
            if truth:
                pcm, wave_receipt = checked_pcm(path, source, record)
                pcm_checked += 1
                rank = (outcome == "correct_range", row["compatible"], record["score"])
                if example is None or rank > example[0]:
                    example = rank, record, truth, source_pixels, pcm, wave_receipt
    pairs = [(bank,index) for bank,index,condition in scores if bank == "grid" and condition == "audio_delay_320ms"]
    for bank, index in pairs:
        require(scores[bank,index,"original"] == scores[bank,index,"audio_delay_320ms"],
                "Identical pictures with delayed audio changed the picture check")
    expected = {(bank, arm, state): count for bank, arms in aggregate["outcomes"].items()
                for arm, outcomes in arms.items() for state, count in outcomes.items()}
    compatible_expected = {(arm, state): count for arm, outcomes in aggregate["compatible"].items()
                           for state, count in outcomes.items()}
    require(visited == set(baseline) and len(visited) == 160 and counts == expected and compatible == compatible_expected
            and len(pairs) == 8 and compatible_masks == 140, "Aggregate or source-compatible roster differs")
    receipt = {"status": "verified", "media_files": len(visited), "decisions": len(visited)*3, "crop_images": crop_count,
               "source_compatible_pcm_pairs_equal": compatible_masks//20, "compatible_pictures": compatible_pictures,
               "compatible_pixel_masks": compatible_masks, "candidate_pcm_intervals": pcm_checked,
               "identical_picture_scores_under_audio_delay": len(pairs), "aggregate_sha256": sha(folder/"aggregate.json")}
    if example:
        rank, record, truth, rgb, pcm, wave_receipt = example
        output = folder/"private-example"
        output.mkdir(mode=0o700)
        for item in record["anchors"]:
            frame = rgb[item["frame"]]
            x0, y0, x1, y1 = item["native_crop_xyxy"]
            Image.fromarray(frame).save(output/f"{item['role']}-native.png")
            Image.fromarray(frame[y0:y1, x0:x1]).save(output/f"{item['role']}-crop.png")
        with wave.open(str(output/"observed-interval.wav"), "wb") as stream:
            stream.setparams((1, 2, audio.RATE, len(pcm), "NONE", "not compressed"))
            stream.writeframes(pcm.tobytes())
        write(output/"pcm-evidence.json", wave_receipt)
        receipt["anonymous_example"] = {
            "correct": rank[0], "source_compatible": rank[1], "authored_s": truth,
            "observed_s": record["source_clock_s"],
            "iou": str(study.review.overlap(record["source_clock_s"], truth)),
            "anchors_s": [str(item["pts"]*Fraction(record["time_base"])) for item in record["anchors"]],
            "aligned_source_pcm_samples": wave_receipt["sample_count"],
        }
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, default=study.prior.PRIOR)
    args = parser.parse_args()
    receipt = audit(args.folder, args.prior)
    write(args.folder/"independent-audit.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
