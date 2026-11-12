# SPDX-License-Identifier: Apache-2.0
"""Re-decode, replay and check the frozen short-slip study from source media."""

from __future__ import annotations

import argparse
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video.decode import probe_frames
from app.detectors.video.lipsync import mouth_patch
from scripts import review_video_lipsync_local_delay_transfer as earlier
from scripts import review_video_reference_h264_earlier_transfer as video
from scripts import video_lipsync_local_contrast_transfer as study
from scripts.video_common_sources_v89 import read, require, sha, write


def original(entry):
    source = Path(entry["source_file"])
    wav = Path(entry["wav_file"])
    require(sha(source) == entry["source_sha256"] and sha(wav) == entry["wav_file_sha256"], "Input files differ")
    actual, origin, discarded = earlier.source_pcm(source, entry["source_sha256"])
    with wave.open(str(wav), "rb") as stream:
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes())
                == (1, 2, 16000, 47648), "Stored WAV format differs")
        require(stream.readframes(47648) == actual.tobytes() and not origin and not discarded,
                "Decoded source audio bytes or clock differ")
    return source, actual


def authored(pcm, case, ordinal):
    if case == "recorded":
        return pcm, None, None
    start = 26 + ordinal % 5
    end = start + 16
    if case == "local_gain":
        substitute = .68 * pcm
    elif case == "local_lowpass":
        substitute = sosfiltfilt(butter(4, 1900, fs=16000, output="sos"), pcm)
    elif case == "local_noise":
        energy = np.linalg.norm(pcm[start*640:end*640].astype(float)) / np.sqrt(16*640)
        substitute = pcm + .12*energy*np.random.default_rng(36120 + ordinal).normal(size=len(pcm))
    else:
        delta = (7 if case.startswith("long_") else 5) * (-1 if case.endswith("early") else 1)
        sample_shift = 640 * delta
        ix = np.arange(len(pcm)) - sample_shift
        substitute = np.zeros_like(pcm)
        valid = (ix >= 0) & (ix < len(pcm))
        substitute[valid] = pcm[ix[valid]]
        if case.startswith("constant_"):
            return substitute, None, None
        if case.startswith("long_"):
            start, end = start - 3, end + 3
    out = pcm.copy()
    first, last = 640*start, 640*end
    for index in range(first, last):
        blend = min(index-first+1, last-index, 160) / 160
        out[index] = max(-32768, min(32767, round((1-blend)*pcm[index] + blend*substitute[index])))
    if case.startswith(("short_", "long_")):
        return out, [start, end], delta
    return out, None, None


def compare_flags(method, case, point):
    truth, delay = case["known_extent_ticks"], case["known_delay_ticks"]
    data = case[method]
    if method == "candidate":
        best = data["best"]
        flags = [best] if best and best["score"] > point else []
        measured = data["measured"]
    else:
        flags, measured = data["ranges"], data["measured_windows"]
    correct = False
    if truth:
        for flag in flags:
            overlap = earlier.overlaps(flag["extent_ticks"], truth)
            lag = flag["relative_lag_ticks"]
            correct |= overlap >= Fraction(2, 5) and lag is not None and abs(lag-delay) <= 2
    return {"views": 1, "measured": bool(measured), "flagged": bool(flags), "correct": correct,
            "misplaced_or_wrong_lag": bool(truth and flags and not correct)}


def review(folder, plan):
    pairs = [(row, rec) for row, rec in study.frozen(folder, plan) if rec["split"] == "held"]
    aggregate, head, point = (read(folder / name) for name in ("aggregate.json", "head.json", "point.json"))
    calibration = study.rows(folder, "point")
    negatives = [case["candidate"]["best"]["score"] for row in calibration for case in row["cases"]
                 if not case["known_extent_ticks"] and case["candidate"]["best"]]
    require(point["head_sha256"] == sha(folder / "head.json") and point["journal_sha256"] == sha(folder / "point.jsonl")
            and point["value"] == max(negatives, default=0)+.025 and aggregate["point"] == point
            and aggregate["runtime"]["held_sha256"] == sha(folder / "held.jsonl"), "Sealed score inputs differ")
    totals = {name: {condition: Counter() for condition in study.CASES} for name in ("candidate", "current")}
    examined = set()
    detector = load_yunet(study.previous.common.FACE_FILE)
    crops = audio_slices = rebuilt = native_count = 0
    for (row, roster), recorded in zip(pairs, study.rows(folder, "held"), strict=True):
        require(all(recorded[key] == value for key, value in roster.items()), "Held file is outside the sealed roster")
        entry = row["input"]
        source, pcm = original(entry)
        clock = probe_frames(source)
        require(tuple(video.clocks(clock)) == study.grid.CLOCK and (clock.width, clock.height) == (360, 288),
                "Native picture clock or shape differs")
        images = video.read_pictures(source, clock, range(75))
        patches = {}
        for index, picture in images.items():
            require(earlier.sha(picture.tobytes()) == entry["pictures"][index]["rgb_sha256"], "Native picture bytes differ")
            if index >= 74:
                continue
            faces = detect_faces(cv2.cvtColor(picture, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
            if len(faces) == 1:
                patches[index] = mouth_patch(picture, {"landmarks": faces[0].landmarks})
        require(len(patches) == recorded["mouth_pictures"], "Native patch coverage differs")
        native_count += len(images)
        v = study.visual_features(patches, 74)
        for case, name in zip(recorded["cases"], study.CASES, strict=True):
            audio, truth, lag = authored(pcm, name, roster["ordinal"])
            require(case["case"] == name and case["known_extent_ticks"] == truth and case["known_delay_ticks"] == lag
                    and case["pcm_sha256"] == earlier.sha(audio.tobytes()), "Rebuilt source PCM or authored timing differs")
            require(study.candidate(v, *study.acoustic_features(audio), head) == case["candidate"], "Source score replay differs")
            rebuilt += 1
            if case["evidence"]:
                for role, record in case["evidence"].items():
                    trial = case["candidate"]["best"]
                    lo, hi = trial["roles"][("before", "during", "after").index(role)]
                    picture = record["picture"]
                    index = picture["frame"]
                    require(lo <= index < hi and picture["pts"] == clock.pts[index] and picture["tb"] == clock.time_base
                            and picture["picture_sha256"] == earlier.sha(images[index].tobytes())
                            and picture["mouth_patch_sha256"] == earlier.sha(patches[index].tobytes()), "Mouth frame binding differs")
                    x0, y0, x1, y1 = picture["xyxy"]
                    crop = images[index][y0:y1, x0:x1]
                    patch = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (32, 24), interpolation=cv2.INTER_AREA)
                    require(earlier.sha(patch.astype("<f4").tobytes()) == picture["mouth_patch_sha256"], "Selected native crop differs")
                    crops += 1
                    for label, key in (("reference", "flank_lag_ticks"), ("alternate", "changed_lag_ticks")):
                        a, b = (lo + trial[key])*640, (hi + trial[key])*640
                        require(0 <= a < b <= len(audio) and record["pcm"][label] == {
                            "samples": [a, b], "sha256": earlier.sha(audio[a:b].tobytes())}, "Selected PCM range differs")
                        audio_slices += 1
            for method, conditions in totals.items():
                decision = compare_flags(method, case, point["value"])
                conditions[name].update(decision)
                if method == "candidate" and decision["correct"]:
                    examined.add(roster["group"])
    require(totals == aggregate["by_condition"] and len(examined) == aggregate["qualified_groups"], "Published decisions differ")
    return {"status": "verified", "source_recordings": len(pairs), "native_pictures": native_count,
            "held_pcm_and_candidate_scores_replayed": rebuilt, "native_mouth_crops": crops,
            "exact_pcm_ranges": audio_slices, "aggregate_sha256": sha(folder / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--plan", type=Path, default=study.grid.PLAN)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = review(args.folder, args.plan)
    write(args.folder / "independent-audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
