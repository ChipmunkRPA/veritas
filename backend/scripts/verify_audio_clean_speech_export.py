"""Independently decode eligible source recordings and verify local speech evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path
from struct import iter_unpack

import numpy as np

from app.core.canonical import canonical_hash
from app.evidence import audio_clean_speech, audio_clean_speech_long
from scripts.verify_audio_local_timing_export import decode
from scripts.verify_stereo_reference import RATE, SIDES, digest, require, verify_clip, waveform

SCHEMA = "chipmunk.audio-reference-comparison/8"
LONG_SCHEMA = "chipmunk.audio-reference-comparison/9"
STATUSES = {"local_change": "localized_difference", "no_local_change": "no_local_difference",
            "no_corroborated_local_change": "no_local_difference",
            "consistent_frequency_balance_change": "broad_processing_difference"}


def verify_export(earlier, submitted, export):
    package = json.loads(export.read_bytes())
    record, assets = package["comparison"], package["assets"]
    result = record["clean_speech"]
    long = record["schema"] == LONG_SCHEMA
    sample_count = audio_clean_speech_long.SAMPLES if long else audio_clean_speech.SAMPLES
    require(record["schema"] in {SCHEMA, LONG_SCHEMA} and result["sample_rate"] == RATE
            and record["affects_overall_confidence"] is False and result["affects_overall_confidence"] is False,
            "The export needs local speech evidence with zero AI weight")
    native = [decode(path, record[side]) for path, side in zip((earlier, submitted), SIDES, strict=True)]
    require(all(origin == 0 and len(raw) == sample_count * 2 for raw, origin in native),
            "This verification requires both equal supported native source clocks")
    samples = [np.frombuffer(raw, dtype="<i2") for raw, _ in native]
    for side, (_, origin), pcm in zip(SIDES, native, samples, strict=True):
        decoded_float = (pcm.astype(np.float32) / 32768).astype("<f4", copy=False)
        expected = {"pcm_sha256": digest(decoded_float.tobytes()), "source_pts": origin, "relative_start": "0", "samples": sample_count}
        require(record["audio"][side] == expected, "Decoded source metadata differs")
    if long:
        reason, measured, profile = audio_clean_speech_long.locate(*samples)
        require(result["sample_count"] == sample_count and result.get("frequency_balance") == profile,
                "Extended speech duration or frequency measurements differ")
    else:
        reason, measured = audio_clean_speech.locate(*samples)
    status = STATUSES.get(reason, "unassessed")
    require(result["status"] == status and result["reason"] == reason, "Local speech coverage or result differs")
    expected_assets = {f"clean-reference-speech-{side}.wav" for side in SIDES} if measured else set()
    require({name for name in assets if name.startswith("clean-reference-speech-")} == expected_assets,
            "Unexpected or missing local speech evidence assets")
    if measured is None:
        require("range" not in result, "No local speech interval was supported")
        return {"status": status, "reason": reason, "excerpts_verified": 0}
    span = result["range"]
    require({key: span[key] for key in measured} == measured
            and span["support_sha256"] == canonical_hash(measured), "Local speech measurement or range differs")
    start, end = measured["samples"]
    for side, (raw, origin) in zip(SIDES, native, strict=True):
        row = span[side]
        require(row["samples"] == [start, end] and row["source_pts"] == [origin + start, origin + end]
                and row["start_s"] == start / RATE and row["end_s"] == end / RATE, "Exported source clock differs")
        excerpt = raw[2 * start:2 * end]
        values = tuple(value for (value,) in iter_unpack("<h", excerpt))
        require(row["pcm_sha256"] == digest(excerpt) and row["waveform"] == waveform(values),
                "Source PCM or waveform differs")
        verify_clip(assets, row["audio"], values, f"clean-reference-speech-{side}.wav")
    return {"status": status, "reason": reason, "samples": [start, end], "excerpts_verified": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("submitted", type=Path)
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_export(args.earlier, args.submitted, args.export), sort_keys=True))
    except (KeyError, TypeError, ValueError, OSError, wave.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f"Local speech evidence verification failed. {error}\n")


if __name__ == "__main__":
    main()
