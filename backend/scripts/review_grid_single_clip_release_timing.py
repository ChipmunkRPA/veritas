# SPDX-License-Identifier: Apache-2.0
"""Independently decode and audit the single-clip release timing panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

RATE, STEP, RGB_SIZE = 16000, 640, 288 * 360 * 3


def read(path):
    return json.loads(path.read_text())


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output(argv):
    return subprocess.run(argv, capture_output=True, check=True, timeout=20).stdout


def native(path):
    table = json.loads(
        output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_streams",
                "-show_entries",
                "stream=time_base,width,height:frame=best_effort_timestamp",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    [stream] = table["streams"]
    assert (stream["width"], stream["height"]) == (360, 288)
    tb = Fraction(stream["time_base"])
    pts = [int(frame["best_effort_timestamp"]) for frame in table["frames"]]
    assert [v * tb for v in pts] == [Fraction(i, 25) for i in range(75)]
    common = ["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-i", str(path)]
    pixels = output(
        common
        + [
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
            "-vf",
            "format=rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    audio = output(
        common
        + [
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000",
            "-f",
            "s16le",
            "pipe:1",
        ]
    )
    assert len(pixels) == 75 * RGB_SIZE and len(audio) == 47648 * 2
    return pts, tb, memoryview(pixels), np.frombuffer(audio, dtype="<i2")


def reconstruct(signal, row):
    role, shift = row["role"], row["shift"] * STEP
    if role in ("untouched", "remux"):
        return signal
    if row["span"] is None:
        rolled = np.roll(signal, shift)
        if shift > 0:
            rolled[:shift] = 0
        else:
            rolled[shift:] = 0
        return rolled
    low, high = (v * STEP for v in row["span"])
    result = signal.copy()
    if role == "local_gain":
        result[low:high] = np.rint(signal[low:high] * 0.7)
        return result
    if role == "local_lowpass":
        replacement = np.rint(sosfiltfilt(butter(3, 2200, fs=RATE, output="sos"), signal.astype(float)))
    else:
        replacement = np.roll(signal, shift)
    for i in range(low, high):
        fade = min(160, i - low + 1, high - i) / 160
        result[i] = int(np.clip(round((1 - fade) * signal[i] + fade * replacement[i]), -32768, 32767))
    return result


def event_audit(events, pts, tb, pixels, pcm):
    for event in events:
        frame, audio = event["picture"], event["acoustic_anchor"]
        assert event["native_pts"] == pts[frame : frame + 2] and Fraction(event["time_base"]) == tb
        assert Fraction(event["picture_anchor"], RATE) == (pts[frame] + pts[frame + 1]) * tb / 2
        crops = []
        x0, y0, x1, y1 = event["crop_xyxy"]
        assert 0 <= x0 < x1 <= 360 and 0 <= y0 < y1 <= 288
        for i, digest_at in zip((frame, frame + 1), event["frame_sha256s"], strict=True):
            frame_bytes = pixels[i * RGB_SIZE : (i + 1) * RGB_SIZE]
            assert sha(frame_bytes) == digest_at
            cropped = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(288, 360, 3)[y0:y1, x0:x1]
            crops.append(sha(cropped.tobytes()))
        assert event["crop_sha256s"] == crops
        a, b = event["pcm_range"]
        assert (a, b) == (audio - STEP, audio + STEP) and 0 <= a < b <= len(pcm)
        assert event["pcm_sha256"] == sha(pcm[a:b].tobytes())


def audit(folder):
    seal, media, result, aggregate = (
        read(folder / (name + ".json")) for name in ("freeze", "media", "readout", "aggregate")
    )
    assert (
        digest(folder / "freeze.json")
        == media["freeze_sha256"]
        == result["freeze_sha256"]
        == aggregate["freeze_sha256"]
    )
    assert (
        digest(folder / "media.json") == result["media_sha256"]
        and digest(folder / "readout.json") == aggregate["readout_sha256"]
    )
    held_groups = {row["group"] for row in seal["held"]}
    assert len(held_groups) == 6 and not held_groups.intersection(seal["discovery_groups"])
    assert len(media["cases"]) == len(result["scores"]) == 120
    originals, counts, totals = {}, Counter(), Counter()
    for row, score in zip(media["cases"], result["scores"], strict=True):
        assert {k: row[k] for k in ("ordinal", "group", "role", "span", "shift")} == {
            k: score[k] for k in ("ordinal", "group", "role", "span", "shift")
        }
        path = Path(row["path"])
        assert digest(path) == row["file_sha256"]
        pts, tb, pixels, pcm = native(path)
        frame_hashes = [sha(pixels[k * RGB_SIZE : (k + 1) * RGB_SIZE]) for k in range(75)]
        if row["role"] == "untouched":
            assert row["file_sha256"] == row["source_sha256"]
            originals[row["ordinal"]] = frame_hashes, pcm
        original_rgb, original_pcm = originals[row["ordinal"]]
        assert frame_hashes == row["native_rgb"] == original_rgb
        assert sha(pcm.tobytes()) == row["whole_pcm_sha256"]
        assert np.array_equal(pcm, reconstruct(original_pcm, row))
        assert row["changed_samples"] == np.count_nonzero(pcm != original_pcm)
        proposal = score["proposal"]
        status = proposal["status"]
        correct = False
        if status == "local_timing_change":
            before, middle, after = proposal["events"]
            event_audit(proposal["events"], pts, tb, pixels, pcm)
            totals["native_event_crops"] += 6
            totals["sample_exact_event_excerpts"] += 3
            a, m, b = (
                event["acoustic_anchor"] - event["picture_anchor"] for event in (before, middle, after)
            )
            signed = m - (a + b) // 2
            assert signed == proposal["audio_minus_picture_change_samples"] and 2560 <= abs(signed) <= 6400
            if row["span"] and row["shift"]:
                low, high = (t * STEP for t in row["span"])
                correct = (
                    low + 160 <= middle["acoustic_anchor"] < high - 160
                    and abs(signed - row["shift"] * STEP) <= 1280
                )
        assert correct == score["correct"]
        counts[(row["role"], "views")] += 1
        counts[(row["role"], "assessed")] += status != "unassessed"
        counts[(row["role"], "flags")] += status == "local_timing_change"
        counts[(row["role"], "correct")] += correct
        counts[(row["role"], "current_assessed")] += bool(score["current"]["measured"])
        counts[(row["role"], "current_flags")] += bool(score["current"]["ranges"])
        counts[(row["role"], "current_correct")] += bool(score["current_correct"])
        totals["native_pictures"] += 75
        totals["exact_pcm_samples"] += len(pcm)
        totals["deliveries"] += 1
    assert all(
        counts[role, key] == count
        for role, row in aggregate["held"].items()
        for key, count in row.items()
        if key != "local_shift_truth"
    )
    record = {
        "verified": True,
        "source_groups": len(held_groups),
        "held_originals": len(originals),
        **totals,
        "aggregate_sha256": digest(folder / "aggregate.json"),
        "readout_sha256": digest(folder / "readout.json"),
    }
    with (folder / "independent-replay.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
