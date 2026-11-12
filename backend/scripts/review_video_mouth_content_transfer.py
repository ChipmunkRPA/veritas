# SPDX-License-Identifier: Apache-2.0
"""Verify held speech-content readout and source evidence from persistent clips."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import lipsync, sampling
from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_mouth_content_transfer as trial
from scripts.review_video_lipsync_local_delay_transfer import source_pcm
from scripts.video_common_sources_v89 import read, require, sha, write


def overlaps(predicted, truth):
    a, b = predicted
    c, d = truth
    return 2 * max(0, min(b, d) - max(a, c)) >= max(b, d) - min(a, c)


def outcome(case, name, points):
    if name == "current":
        measured = case["current"]["measured_windows"] > 0
        spans = [item["extent_ticks"] for item in case["current"]["ranges"]]
    else:
        arm = case["arms"][name]
        measured = arm["compared"] > 0
        candidate = arm["best"]
        require(measured == (candidate is not None), "Compared count and chosen range disagree")
        spans = [candidate["extent_ticks"]] if candidate and candidate["score"] > points[name] else []
    truth = case["truth_ticks"]
    if not measured:
        require(not spans, "Unassessed check contains a flagged range")
        return "unassessed"
    if not spans:
        return "measured_no_range"
    if truth is None:
        return "ordinary_range"
    return "correct_range" if any(overlaps(span, truth) for span in spans) else "wrong_range"


def verify_support(best, origin, table, grid, wave, wanted):
    before, during, after = best["medians"]
    require(best["score"] == min(during - before, during - after), "Selected contrast arithmetic differs")
    require([item["role"] for item in best["support"]] == ["before", "during", "after"], "Missing ordered evidence roles")
    start, end = best["extent_ticks"]
    bounds = ((start - 8, start), (start, end), (end, end + 8))
    for row, (lo, hi) in zip(best["support"], bounds, strict=True):
        tick = row["tick"]
        a, b = row["audio_samples"]
        require(lo <= tick < hi and a == tick * 640 and b == a + 640 and b <= len(wave)
                and list(map(Fraction, row["audio_native_s"])) == [origin + Fraction(a, 16000), origin + Fraction(b, 16000)]
                and trial.prior.sha(wave[a:b]) == row["pcm_sha256"], "PCM evidence is outside the selected source-clock range")
        if (frame := row["frame"]) is None:
            require(grid.frames[tick] < 0, "An available exact native picture was omitted")
            continue
        index = frame["frame_index"]
        require(index == grid.frames[tick] and frame["pts"] == table.pts[index]
                and Fraction(row["frame_time_base"]) == table.tb, "Evidence frame differs from the exact source table")
        if index in wanted:
            require(wanted[index] == frame, "A native picture has inconsistent commitments")
        wanted[index] = frame


def verify_pictures(path, table, wanted):
    seen, crops = set(), 0
    for frame in decode_frames(path, table, sorted(wanted), timeout_s=20):
        pin = wanted[frame.index]
        require(frame.pts_verified and frame.pts == pin["pts"] and trial.prior.sha(frame.rgb) == pin["decoded_rgb_sha256"],
                "Native evidence picture differs from its commitment")
        if "mouth_patch_sha256" in pin:
            x0, y0, x1, y1 = pin["mouth_box_xyxy"]
            require(0 <= x0 < x1 <= table.width and 0 <= y0 < y1 <= table.height, "Mouth crop outside source pixels")
            gray = cv2.cvtColor(frame.rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
            patch = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype("<f4")
            require(trial.prior.sha(patch) == pin["mouth_patch_sha256"], "Native mouth crop differs from its commitment")
            crops += 1
        seen.add(frame.index)
    require(seen == wanted.keys(), "Exact native evidence replay was incomplete")
    return len(seen), crops


def audit(folder):
    frozen = trial.frozen(folder)
    aggregate, calibration = read(folder / "aggregate.json"), read(folder / "calibration.json")
    require(aggregate["freeze_sha256"] == sha(folder / "freeze.json")
            and aggregate["held_journal_sha256"] == sha(folder / "held.jsonl")
            and aggregate["calibration_sha256"] == sha(folder / "calibration.json")
            and calibration["fit_sha256"] == sha(folder / "fit.json")
            and calibration["journal_sha256"] == sha(folder / "calibration.jsonl"), "Readout commitments differ")
    development = trial.journal(folder, frozen, "calibration")
    for name in trial.ARMS[1:]:
        scores = [c["arms"][name]["best"]["score"] for row in development for c in row["cases"]
                  if c["truth_ticks"] is None and c["arms"][name]["best"] is not None]
        require(calibration["points"][name] == max(.025, max(scores) + .025), "Calibration point differs")
    for row in frozen["rows"]:
        require(sha(folder / row["file"]) == row["sha256"], "A pinned publisher recording differs")
    totals = {role: {name: Counter() for name in trial.ARMS} for role in trial.ROLE_ORDER}
    inputs = pictures = crops = audio_slices = 0
    for observed, (_, sources) in zip(trial.journal(folder, frozen, "held"), trial.groups(frozen, "held"), strict=True):
        target = sources["target"]
        path = folder / target["file"]
        table = probe_frames(path, timeout_s=15)
        original, origin, _ = source_pcm(path, target["sha256"])
        require(str(origin) == observed["origin_s"] and observed["source_sha256"] == target["sha256"], "Target clock differs")
        grid = lipsync.tick_grid(table.times(), sampling.duration_s(table), len(original), audio_start_s=float(origin))
        wanted = {}
        reconstructed = trial.cases(folder, sources, original, grid.ticks)
        for (role, wave, truth, extent), saved in zip(reconstructed, observed["cases"], strict=True):
            require((role, truth, extent) == (saved["role"], saved["truth_ticks"], saved["changed_samples"])
                    and trial.prior.sha(wave) == saved["pcm_sha256"], "Authored input differs from held comparison")
            if extent is not None:
                changed = np.flatnonzero(wave != original)
                require(extent[0] <= int(changed.min()) <= int(changed.max()) < extent[1], "Edits leak outside their exact interval")
            inputs += 1
            for name in trial.ARMS:
                totals[role][name][outcome(saved, name, calibration["points"])] += 1
                if name != "current" and (best := saved["arms"][name]["best"]) is not None:
                    verify_support(best, origin, table, grid, wave, wanted)
                    audio_slices += len(best["support"])
        count, verified_crops = verify_pictures(path, table, wanted)
        pictures += count
        crops += verified_crops
    require(totals == aggregate["by_condition"], "Independent outcome arithmetic differs from the aggregate")
    output = {"status": "verified", "publisher_sources_byte_checked": len(frozen["rows"]), "held_pcm_inputs_rebuilt": inputs,
              "source_audio_slices": audio_slices, "native_pictures_replayed": pictures, "native_mouth_crops_replayed": crops,
              "aggregate_sha256": sha(folder / "aggregate.json")}
    write(folder / "independent-audit.json", output)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    audit(args.folder)


if __name__ == "__main__":
    main()
