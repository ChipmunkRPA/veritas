# SPDX-License-Identifier: Apache-2.0
"""Independently reopen native short timing evidence and verify authored witness bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.video_common_sources_v89 import read, require, sha, write


def digest(data):
    return hashlib.sha256(data).hexdigest()


def command(*argv):
    return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=20).stdout


def audit(source, run):
    observed_path, scored_path = (run / n for n in ("observations-held.json", "held.json"))
    observed, scores = (read(p) for p in (observed_path, scored_path))
    receipt = read(source / "source-receipt.json")
    require(
        receipt["freeze_sha256"] == sha(source / "freeze.json")
        and observed["source_receipt_sha256"] == sha(source / "source-receipt.json")
        and scores["observation_sha256"] == sha(observed_path),
        "Source and observation receipts differ",
    )
    indexed = {row["ordinal"]: row for row in receipt["rows"]}
    n_frames, n_samples, crop_count, exact_earlier, exact_later, raw_proposals = 0, 0, 0, 0, 0, 0
    for obs in observed["rows"]:
        k = obs["ordinal"]
        row = indexed[k]
        front, sound = (source / "sources" / row["inputs"][label]["file"] for label in ("front", "audio"))
        require(
            sha(front) == obs["video_sha256"] == row["inputs"]["front"]["sha256"]
            and sha(sound) == obs["audio_sha256"] == row["inputs"]["audio"]["sha256"],
            "Archived input bytes differ",
        )
        inventory = json.loads(
            command(
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_streams",
                "-show_entries",
                "stream=time_base:frame=best_effort_timestamp",
                "-of",
                "json",
                str(front),
            )
        )
        [stream] = inventory["streams"]
        require(
            stream["time_base"] == obs["time_base"]
            and [int(frame["best_effort_timestamp"]) for frame in inventory["frames"]] == obs["pts"],
            "Standalone FFprobe clock differs",
        )
        n_frames += len(obs["pts"])
        width = obs["source_sample_bytes"]
        codec = {2: "s16le", 4: "f32le"}[width]
        native = command(
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(sound),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_" + codec,
            "-f",
            codec,
            "-",
        )
        require(
            len(native) == obs["audio_samples"] * width and digest(native) == obs["audio_pcm_sha256"],
            "Standalone FFmpeg native waveform differs",
        )
        n_samples += obs["audio_samples"]
        wanted = set()
        if truth := obs["truth"]:
            a, b = truth["acoustic_samples"]
            require(
                digest(native[a * width : b * width]) == truth["acoustic_excerpt_sha256"],
                "Acoustic excerpt differs",
            )
            wanted.update(truth["visual_frames"])
            positives = [p for p in scores["cases"] if p["ordinal"] == k and p["truth"] is not None]
            require(
                {p["role"] for p in positives} == {"local_earlier", "local_later"} and len(positives) == 2,
                "Qualified source lacks matched signed variants",
            )
            for positive in positives:
                expected = positive["truth"]
                moved_a, moved_b = expected["audio_bracket"]
                start, stop = expected["edit_samples"]
                delta = moved_a - a
                require(
                    moved_b - b == delta
                    and expected["shift_s"] == delta / 16000
                    and start + 80 <= moved_a < moved_b <= stop - 80
                    and expected["frame_bracket"] == truth["visual_frames"],
                    "Authored witness differs from native source ranges",
                )
                exact_earlier += delta < 0
                exact_later += delta > 0
                if positive["correct"]:
                    raw_proposals += 1
                    for i, _ in positive["candidate"]["inside"]:
                        e = positive["video_events"][i]
                        require(
                            e["hi"] == e["lo"] + 1
                            and abs(
                                e["t"]
                                - float(
                                    Fraction(obs["pts"][e["hi"]] + obs["pts"][e["lo"]], 2)
                                    * Fraction(obs["time_base"])
                                )
                            )
                            < 1e-12,
                            "Native proposal clock differs",
                        )
                        wanted.update((e["lo"], e["hi"]))
        if wanted:
            table = probe_frames(front, timeout_s=15)
            actual = set()
            for frame in decode_frames(front, table, sorted(wanted), timeout_s=20):
                anchor = obs["anchors"][str(frame.index)]
                x0, y0, x1, y1 = anchor["xyxy"]
                require(
                    frame.pts_verified
                    and frame.pts == anchor["pts"]
                    and digest(frame.rgb.tobytes()) == anchor["full_rgb_sha256"]
                    and digest(frame.rgb[y0:y1, x0:x1].tobytes()) == anchor["crop_rgb_sha256"],
                    "Native frame and crop pixels differ",
                )
                actual.add(frame.index)
            require(wanted == actual, "Missing native mouth evidence")
            crop_count += len(actual)
    return {
        "observations_sha256": sha(observed_path),
        "scored_sha256": sha(scored_path),
        "verified_native_picture_ticks": n_frames,
        "verified_audio_samples": n_samples,
        "exact_native_mouth_crops": crop_count,
        "exact_earlier_source_witnesses": exact_earlier,
        "exact_later_source_witnesses": exact_later,
        "raw_useful_proposals": raw_proposals,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", type=Path)
    p.add_argument("run", type=Path)
    a = p.parse_args()
    result = audit(a.sources, a.run)
    write(a.run / "native-independent-audit.json", result)
    print(result)


if __name__ == "__main__":
    main()
