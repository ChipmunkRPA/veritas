# SPDX-License-Identifier: Apache-2.0
"""Independently reopen native timestamps and audio evidence for the phone benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRED = ROOT / "backend/data/independent-evaluation/grid-paired-closure-panel-2026-11-12/media.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def capture(*command):
    return subprocess.run(command, capture_output=True, check=True, timeout=25).stdout


def compare_original(file, native):
    assert digest(file.read_bytes()) == native["file_sha256"]
    clock = json.loads(
        capture(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=time_base:frame=best_effort_timestamp",
            "-of",
            "json",
            str(file),
        )
    )
    tb = Fraction(clock["streams"][0]["time_base"])
    times = [str(int(frame["best_effort_timestamp"]) * tb) for frame in clock["frames"]]
    assert times == native["picture_times"]
    pcm = capture(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-copyts",
        "-i",
        str(file),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000",
        "-f",
        "s16le",
        "pipe:1",
    )
    assert len(pcm) == native["samples"] * 2 and digest(pcm) == native["pcm_sha256"]
    return pcm


def rgb(file):
    return digest(
        capture(
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-copyts",
            "-i",
            str(file),
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
        )
    )


def audit(run):
    freeze, scores, aggregate = (read(run / (n + ".json")) for n in ("freeze", "scores", "aggregate"))
    assert digest((run / "freeze.json").read_bytes()) == scores["freeze_sha256"] == aggregate["freeze_sha256"]
    assert digest((run / "scores.json").read_bytes()) == aggregate["scores_sha256"]
    assert len(freeze["roster"]) == len(scores["rows"]) == 28
    assert Counter(r["bank"] for r in freeze["roster"]) == {
        "crema_lexical": 12,
        "grid_signal": 12,
        "grid_shift": 4,
    }
    assert len({r["group"] for r in freeze["roster"] if r["bank"] == "crema_lexical"}) == 12
    plans = {row["ordinal"]: row for row in read(PAIRED)["cases"] if row["role"] == "audio_late_ordinary"}
    originals, counts, totals = {}, Counter(), Counter()
    for row, observed in zip(freeze["roster"], scores["rows"], strict=True):
        bank, ordinal, native = row["bank"], row["id"], row["native"]
        assert (bank, ordinal) == (observed["bank"], observed["id"])
        file = Path(row["file"])
        pcm = compare_original(file, native)
        totals["exact_native_picture_times"] += len(native["pts"])
        totals["exact_decoded_samples"] += len(pcm) // 2
        if bank == "grid_signal":
            originals[ordinal] = (pcm, file, native)
        if bank == "grid_shift":
            prior_pcm, prior_file, prior_native = originals[ordinal]
            shift = plans[ordinal]
            start, stop = (index * 1280 for index in shift["span"])
            delta = shift["audio_shift"] * 1280
            assert pcm[:start] == prior_pcm[:start] and pcm[stop:] == prior_pcm[stop:]
            assert pcm[start:stop] == prior_pcm[start - delta : stop - delta]
            assert native["picture_times"] == prior_native["picture_times"]
            assert rgb(file) == rgb(prior_file)
            totals["byte_exact_paired_clips"] += 1
        if bank == "crema_lexical":
            expected = row["lexical_phones"]
            assert expected == observed["lexical_phones"]
            for arm, got in observed["observed"].items():
                if isinstance(got, list):
                    counts[arm, "measured"] += 1
                    phones = [item["phone"] for item in got]
                    counts[arm, "positive_exact"] += bool(expected) and phones == expected
                    counts[arm, "negative_nonempty"] += not expected and bool(phones)
        for event, evidence in zip(row["anchors"], observed["anchors"], strict=True):
            counts[bank, "lexical_windows"] += 1
            if event["anchor"] is None:
                assert evidence["anchor"] is None
                continue
            counts[bank, "signal_landmarks"] += 1
            point = event["anchor"]
            assert point == evidence["anchor"]
            a, b = point["impulse_sample"], point["flux_sample"]
            center = point["signal_sample"]
            assert abs(a - b) <= 240 and center == (a + b) // 2 + event.get("shift_samples", 0)
            assert point["uncertainty_samples"] == 400 and 400 <= center < len(pcm) // 2 - 400
            assert digest(pcm[(center - 400) * 2 : (center + 400) * 2]) == point["native_pcm_sha256"]
            for arm in ("raw", "known", "blind"):
                observed_error = evidence[arm + "_nearest_error_samples"]
                if arm != "raw":
                    typed = observed["observed"][arm]
                    match = (
                        [r for r in typed if r["phone"] == event["phone"]] if isinstance(typed, list) else []
                    )
                    assert observed_error == min(
                        (abs(center - r["sample_range"][1]) for r in match), default=None
                    )
                counts[bank, arm, "anchors_with_any_matching_event"] += observed_error is not None
                counts[bank, arm, "anchors_with_nearby_event"] += (
                    observed_error is not None and observed_error <= 800
                )
    for arm, summary in aggregate["lexical"].items():
        assert all(counts[arm, key] == value for key, value in summary.items())
    for bank, summary in aggregate["timing"].items():
        for key, value in summary.items():
            if isinstance(value, dict):
                assert all(counts[bank, key, metric] == number for metric, number in value.items())
            else:
                assert counts[bank, key] == value
    receipt = {
        "verified": True,
        "source_views": len(scores["rows"]),
        **totals,
        "freeze_sha256": aggregate["freeze_sha256"],
        "score_sha256": aggregate["scores_sha256"],
    }
    with (run / "independent-audit.json").open("x") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
