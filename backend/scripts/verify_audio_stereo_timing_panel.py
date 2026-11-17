# SPDX-License-Identifier: Apache-2.0
"""Recount and verify the frozen stereo timing panel using a separate audio decode."""

import argparse
import hashlib
import json
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.audio_native_alignment_panel import digest, read, save
from scripts.audio_stereo_timing_change_panel import PCM, SAMPLES, SCHEMA, totals


def read_aac(path):
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0",
         "-show_frames", "-show_entries", "frame=pts,nb_samples:stream=codec_name,profile,sample_rate,channels,"
         "time_base,start_pts,duration_ts", "-of", "json", str(path)], timeout=20))
    stream, = probe["streams"]
    if stream != {"codec_name": "aac", "profile": "LC", "sample_rate": "48000", "channels": 2,
                  "time_base": "1/48000", "start_pts": 0, "duration_ts": SAMPLES}:
        raise ValueError(f"Audio track clock differs for {path.name}")
    cursor = 0
    for frame in probe["frames"]:
        if frame["pts"] != cursor or frame["nb_samples"] <= 0:
            raise ValueError(f"Audio frame clock discontinuity in {path.name}")
        cursor += frame["nb_samples"]
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file", "-copyts", "-i", str(path),
         "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "s16le", "-"], timeout=20)
    if len(raw) != 4 * cursor or not 0 <= cursor - SAMPLES < 1024:
        raise ValueError(f"Audio frame extent differs for {path.name}")
    return raw[:4 * SAMPLES], len(probe["frames"]), cursor - SAMPLES


def audit_delivery(folder, row):
    import numpy as np

    from app.evidence.audio_stereo_timing_change import locate

    path = folder / row["delivery"]
    if row["carrier"] == PCM:
        parent = path
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (2, 2, 48000, SAMPLES):
                raise ValueError(f"Unexpected source sample clock for {path.name}")
            raw, frames, tail = wav.readframes(SAMPLES), 0, 0
    else:
        parent = folder / f"{path.stem.rsplit('-', 1)[0]}.wav"
        raw, frames, tail = read_aac(path)
    if digest(path) != row["input_sha256"] or digest(parent) != row["parent_sha256"]:
        raise ValueError(f"Audio delivery or parent differs for {path.name}")
    if hashlib.sha256(raw).hexdigest() != row["presented_pcm_sha256"]:
        raise ValueError(f"Presented samples differ for {path.name}")
    actual = locate(np.frombuffer(raw, dtype="<i2").reshape(-1, 2))
    if actual != (row["assessed"], row["measurement"]):
        raise ValueError(f"Released measurement differs from the frozen reading for {path.name}")
    return frames, tail


def audit(receipt, folder, output):
    from app.evidence import audio_stereo_timing_change
    from scripts import audio_stereo_timing_change_panel

    record = read(receipt)
    implementation = digest(Path(audio_stereo_timing_change.__file__))
    recipe = digest(Path(audio_stereo_timing_change_panel.__file__))
    if (record["schema"] != SCHEMA or record["totals"] != totals(record["rows"])
            or record["implementation_sha256"] != implementation or record["recipe_sha256"] != recipe):
        raise ValueError("Published stereo timing measurements or frozen sources differ")
    aac_frames = tails = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, (frames, tail) in enumerate(pool.map(lambda row: audit_delivery(folder, row), record["rows"]), 1):
            aac_frames += frames
            tails += tail
            if index % 96 == 0:
                print(index, "exact audio inputs reproduced", flush=True)
    save(output, {"schema": "chipmunk.stereo-timing-replay/1", "result_sha256": digest(receipt),
                  "delivered_files_verified": len(record["rows"]), "parent_file_bindings_verified": len(record["rows"]),
                  "aac_frames_verified": aac_frames, "unpresented_samples": tails, "implementation_sha256": implementation,
                  "recipe_sha256": recipe})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "folder", "output"):
        cli.add_argument(name, type=Path)
    audit(**vars(cli.parse_args()))
