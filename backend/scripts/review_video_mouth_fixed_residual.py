# SPDX-License-Identifier: Apache-2.0
"""Independently check mouth-residual decisions and reproduce native evidence."""

from __future__ import annotations

import argparse
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path

from PIL import Image

from app.detectors.video.decode import probe_frames
from scripts import review_video_mouth_temporal_detail as prior_review
from scripts import video_mouth_fixed_residual as study
from scripts.video_common_sources_v89 import read, require, sha, write


def exact_outcome(record, point, truth):
    if record is None:
        return "unassessed"
    if record["score"] <= point:
        return "measured_no_range"
    if truth is None:
        return "false_range"
    a, b, c, d = map(Fraction, [*record["source_clock_s"], *truth])
    return "correct_range" if 2 * (min(b, d) - max(a, c)) >= max(b, d) - min(a, c) else "misplaced_range"


def inspect_score(record, arm):
    if record is None:
        return
    raw = record["measurement"]
    spec = study.SPEC
    require(0 <= raw["mouth_drift_eye_distance"] <= spec["max_mouth_drift_eye_distance"]
            and 0 <= raw["relative_eye_scale_change"] <= spec["max_relative_eye_scale_change"]
            and record["sample_support"][0] == record["sample_support"][2] == spec["flank_samples"]
            and record["sample_support"][1] in spec["core_samples"], "Selected range differs from the frozen geometry or support")
    delta = []
    for part in (raw["mouth"], raw["face_reference"]):
        left_right, left_middle, middle_right = part["agreement"]
        require(part["pixels"] >= spec["quiet_pixels"] and left_right >= spec["min_flank_agreement"]
                and all(-2.000001 <= value <= 2.000001 for value in part["agreement"]),
                "The residual agreement lacks its required support")
        delta.append(min(left_right-left_middle, left_right-middle_right))
    require(record["score"] == (delta[0] if arm == "fixed_pixel" else delta[0]-delta[1]),
            "Selected residual score differs from its measurements")


def replay_images(path, source, record):
    table = probe_frames(path, timeout_s=15)
    expected = source["expected"]
    if expected is not None:
        require(study.panel.source_clock(table) == [Fraction(t, 1000) for t in expected["picture_pts_ms"]],
                "Reopened source clock differs")
    require(record["time_base"] == table.time_base and len(record["frames"]) == 2,
            "Evidence time base differs from original")
    a, b = record["frames"]
    require([table.pts[i]*table.tb for i in (a, b)] == list(map(Fraction, record["source_clock_s"])),
            "Range differs from the exact source indices")
    anchors = record["anchors"]
    frames = [item["frame"] for item in anchors]
    require(len(frames) == 3 and frames[0] < a <= frames[1] < b <= frames[2]
            and [item["role"] for item in anchors] == ["before", "during", "after"],
            "Evidence anchors differ from the source interval")
    checked = {}
    for (frame, digest), item in zip(study.panel.exact_rgb(path, table, frames, expected), anchors, strict=True):
        x0, y0, x1, y1 = item["native_crop_xyxy"]
        require(frame.pts == item["pts"] and digest == item["rgb_sha256"]
                and 0 <= x0 < x1 <= table.width and 0 <= y0 < y1 <= table.height
                and study.panel.raw_digest(frame.rgb[y0:y1, x0:x1]) == item["crop_rgb_sha256"],
                "Native original picture or inspection crop differs")
        checked[frame.index] = frame.rgb
    return checked


def audit(folder, prior, reserve):
    freeze, calibration, runtime, aggregate, reserve_result = (read(folder / name) for name in
        ("freeze.json", "calibration.json", "runtime.json", "aggregate.json", "reserve.json"))
    require(freeze == study.lock(prior, reserve)
            and calibration["freeze_sha256"] == aggregate["freeze_sha256"] == sha(folder / "freeze.json")
            and runtime["held_sha256"] == sha(folder / "held.jsonl")
            and runtime["calibration_sha256"] == reserve_result["calibration_sha256"] == sha(folder / "calibration.json")
            and aggregate["reserve_sha256"] == sha(folder / "reserve.json")
            and (folder / "calibration.json").stat().st_mtime_ns <= (folder / "held.jsonl").stat().st_mtime_ns,
            "Frozen protocol, threshold or evaluated inputs differ")
    for arm in study.ARMS:
        measured = sorted(row["scores"][arm] for row in calibration["ordinary"] if row["scores"][arm] is not None)
        require(len(measured) >= 42 and calibration["points"][arm] == aggregate["points"][arm] == max(0., measured[-3]),
                "Ordinary review point differs")
    baseline = study.baseline.baselines(prior)
    old_roster = study.baseline.roster(prior, read(prior / "freeze.json"))
    roster = {(bank, index, role): (path, source, component) for (bank, index), (component, cases) in
              old_roster.items() for role, (path, source) in cases.items()}
    truth_reserve = {}
    for row in read(reserve)["sources"]:
        key = ("reserve", row["ordinal"], row["role"])
        roster[key] = study.panel.ROOT / row["path"], {"file_sha256": row["sha256"], "expected": None}, row["ordinal"]
        neural = row["role"] in {"source_time_neural", "earlier_drive_neural"}
        truth_reserve[key] = [str(Fraction(i, 25)) for i in row["picture_edit"]] if neural else None
    counts, compared, seen = Counter(), {}, set()
    native_crops = aligned_audio = 0
    example = None
    for row in [*study.panel.prior.jsonl(folder / "held.jsonl"), *reserve_result["rows"]]:
        key = row["bank"], row["index"], row["condition"]
        path, source, component = roster[key]
        require(key not in seen and row["source_sha256"] == sha(path) == source["file_sha256"]
                and row["component"] == component, "Case differs from the complete frozen source roster")
        seen.add(key)
        if key in baseline:
            old = baseline[key]
            require(row["truth"] == old["truth"] and row["compatible"] == old["source_time_compatible"]
                    and row["arms"]["direct"] == old["arms"]["direct"], "Paired visual comparison differs")
        else:
            require(row["truth"] == truth_reserve[key] and row["compatible"] == (row["condition"] == "source_time_neural"),
                    "Reserve truth differs from the admitted neural delivery")
        bank = row["condition"] if row["bank"] == "delivery" else row["bank"]
        compared[key] = {arm: row["arms"][arm] for arm in study.ARMS}
        for arm, record in row["arms"].items():
            if arm != "direct":
                inspect_score(record, arm)
                require((row["eligible_windows"] > 0) == (record is not None), "Window coverage differs")
            state = exact_outcome(record, aggregate["points"][arm], row["truth"])
            counts[bank, arm, state] += 1
            if arm != study.PRIMARY or record is None or (state == "measured_no_range" and not row["compatible"]):
                continue
            pictures = replay_images(path, source, record)
            native_crops += len(pictures)
            if row["truth"]:
                audio, waveform = prior_review.checked_pcm(path, {**source, "expected": source["expected"] or {}}, record)
                aligned_audio += 1
                priority = (state == "correct_range", row["compatible"], record["score"])
                if example is None or priority > example[0]:
                    example = priority, row, record, pictures, audio, waveform
    pairs = [index for bank, index, role in compared if bank == "grid" and role == "audio_delay_320ms"]
    require(len(pairs) == 8 and all(compared["grid", index, "original"] == compared["grid", index, "audio_delay_320ms"]
                                    for index in pairs), "Identical pictures changed with delayed sound")
    expected = {(bank, arm, state): count for bank, arms in aggregate["outcomes"].items()
                for arm, states in arms.items() for state, count in states.items()}
    require(seen == set(roster) and len(seen) == 168 and counts == expected, "Held results or independent counts differ")
    receipt = {"status": "verified", "media_files": len(seen), "decisions": sum(counts.values()), "native_crops": native_crops,
               "identical_picture_pairs": len(pairs), "aligned_source_pcm_intervals": aligned_audio,
               "aggregate_sha256": sha(folder / "aggregate.json"), "audit_implementation_sha256": sha(Path(__file__))}
    if example:
        rank, row, record, pictures, audio, waveform = example
        output = folder / "private-example"
        output.mkdir(mode=0o700)
        for anchor in record["anchors"]:
            picture = pictures[anchor["frame"]]
            role = anchor["role"]
            x0, y0, x1, y1 = anchor["native_crop_xyxy"]
            Image.fromarray(picture).save(output / f"{role}-native.png")
            Image.fromarray(picture[y0:y1, x0:x1]).save(output / f"{role}-crop.png")
        with wave.open(str(output / "observed-interval.wav"), "wb") as stream:
            stream.setparams((1, 2, prior_review.audio.RATE, len(audio), "NONE", "not compressed"))
            stream.writeframes(audio.tobytes())
        write(output / "pcm-evidence.json", waveform)
        receipt["anonymous_example"] = {"bank": row["bank"], "correct": rank[0], "source_time_compatible": rank[1],
              "authored_s": row["truth"], "observed_s": record["source_clock_s"],
              "anchors_s": [str(anchor["pts"]*Fraction(record["time_base"])) for anchor in record["anchors"]],
              "aligned_source_pcm_samples": waveform["sample_count"]}
    write(folder / "independent-audit.json", receipt)
    print(receipt, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path, default=study.baseline.PRIOR)
    parser.add_argument("--reserve", type=Path, default=study.RESERVE)
    args = parser.parse_args()
    audit(args.folder, args.prior, args.reserve)


if __name__ == "__main__":
    main()
