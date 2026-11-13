# SPDX-License-Identifier: Apache-2.0
"""Verify normal audio-comparison evidence on frozen held standalone audio pairs."""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import wave
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence import audio_reference
from scripts.video_common_sources_v89 import read, require, sha, write


def sources(folder, case):
    role = case["role"]
    earlier = role if role in {"unchanged", "silence_all", "preexisting_repeat"} else "unchanged"
    return folder / f"{earlier}-pcm-earlier.wav", folder / f"{role}-pcm-submitted.wav"


def checked_pair(paths, expected):
    record, assets = audio_reference.compare(*paths, include_local_timing=True)
    item = record["local_timing"]
    statuses = {
        "local_shift": "local_shift",
        "no_local_shift": "no_local_shift",
        "too_short": "unassessed",
        "insufficient_support": "unassessed",
    }
    require(
        record["schema"] == audio_reference.SCHEMA_WITH_LOCAL_TIMING
        and item["affects_overall_confidence"] is False
        and item["status"] == statuses[expected["status"]]
        and item.get("matched_windows", 0) == expected["support"],
        "App result differs from held comparison",
    )
    clips = 0
    if item["status"] == "local_shift":
        actual = item["range"]
        wanted = expected["range"]
        require(
            all(
                actual[k] == wanted[k]
                for k in ("earlier_samples", "submitted_samples", "shift_samples", "same_time_support")
            ),
            "App range or support differs from held comparison",
        )
        for label, path in zip(("earlier", "submitted"), paths, strict=True):
            part = actual[label]
            a, b = part["samples"]
            require(
                [a, b] == actual[f"{label}_samples"]
                and part["source_pts"] == [a, b]
                and (part["start_s"], part["end_s"]) == (a / 16000, b / 16000),
                "Playable excerpt clock differs from its range",
            )
            with wave.open(io.BytesIO(assets[part["audio"]["file"]]), "rb") as data:
                require(
                    (data.getnchannels(), data.getframerate(), data.getsampwidth(), data.getnframes())
                    == (1, 16000, 2, b - a),
                    "Playable excerpt has the wrong clock",
                )
                actual_bytes = data.readframes(b - a)
            if path.suffix == ".wav":
                with wave.open(str(path), "rb") as source:
                    source.setpos(a)
                    expected_bytes = source.readframes(b - a)
            else:
                source = subprocess.check_output(
                    (
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        "-map",
                        "0:a:0",
                        "-c:a",
                        "pcm_s16le",
                        "-f",
                        "s16le",
                        "-",
                    ),
                    timeout=12,
                )
                expected_bytes = source[2 * a : 2 * b]
            require(
                actual_bytes == expected_bytes
                and hashlib.sha256(actual_bytes).hexdigest() == part["pcm_sha256"],
                "App exhibit differs from source samples",
            )
            clips += 1
    else:
        require(
            not any(k.startswith("local-timing-") for k in assets),
            "Unmeasured shift contains a playable finding",
        )
    return item["status"], clips


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    a = p.parse_args()
    expected = read(a.run / "held.json")
    total = Counter()
    unchanged_flac = 0
    example = None
    for group in expected["rows"]:
        folder = a.run / f"case-{group['ordinal']:02}"
        for case in group["cases"]:
            if case["carrier"] == "cross" and case["role"] == "codec_flac":
                paths = folder / "unchanged-pcm-earlier.wav", folder / "flac-unchanged.flac"
                status, clips = checked_pair(paths, case)
                require(
                    status == "no_local_shift" and not clips, "Unchanged FLAC contains a local timing finding"
                )
                unchanged_flac += 1
            if case["carrier"] != "pcm":
                continue
            paths = sources(folder, case)
            status, clips = checked_pair(paths, case)
            total[status] += 1
            total["verified_wav_excerpts"] += clips
            if status == "local_shift" and example is None:
                example = paths, case, group["ordinal"]
        print("App WAV source", group["ordinal"], flush=True)
    require(example is not None, "Held local timing examples unavailable")
    paths, case, ordinal = example
    earlier, submitted = paths
    with TemporaryDirectory(dir=a.run) as temp:
        flac = Path(temp) / "matched-flac-positive.flac"
        subprocess.run(
            ("ffmpeg", "-nostdin", "-v", "error", "-i", str(submitted), "-c:a", "flac", str(flac)),
            check=True,
            timeout=12,
        )
        status, clips = checked_pair((earlier, flac), case)
    require(status == "local_shift" and clips == 2, "Mixed native PCM/FLAC positive did not retain evidence")
    receipt = {
        "held_sha256": sha(a.run / "held.json"),
        "native_wav_cases": sum(total[k] for k in ("local_shift", "no_local_shift", "unassessed")),
        "native_wav_results": total,
        "wav_flac_unchanged_no_local_shift": unchanged_flac,
        "wav_flac_positive": {"ordinal": ordinal, "local_timing_status": status, "exact_excerpts": clips},
    }
    output = a.run / "app-held-replay.json"
    if output.exists():
        require(read(output) == receipt, "App replay differs from prior receipt")
    else:
        write(output, receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
