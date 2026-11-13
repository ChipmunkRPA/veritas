# SPDX-License-Identifier: Apache-2.0
"""Check an adaptive local noise profile against the current focused speech review."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts import audio_transient_reduction_transfer as prior

sources, io, clock = prior.sources, prior.io, prior.clock
require = prior.require
FFT = 512
HOP = 128
PROFILE_FRAMES = 151
SCHEMA = "chipmunk.audio-local-noise-profile-transfer/1"
METHOD = {
    "fft_samples": FFT,
    "hop_samples": HOP,
    "profile_frames": PROFILE_FRAMES,
    "profile_percentile": 20,
    "power_factor": 1.25,
    "amplitude_floor": 0.60,
    "smoothing_frames_and_bins": [5, 3],
    "rule": "Original focused intervals plus adjacent positive views, with at least one from the local profile.",
}


def freeze():
    return {"schema": SCHEMA, "method": METHOD, "runner": io.sha(Path(__file__)), "parent": prior.freeze()}


def profile(wave):
    import numpy as np
    from scipy.ndimage import percentile_filter, uniform_filter
    from scipy.signal import istft, stft

    settings = {"fs": clock.RATE, "window": "hann", "nperseg": FFT, "noverlap": FFT - HOP}
    _, _, spectrum = stft(wave.astype(np.float64), **settings)
    power = np.abs(spectrum) ** 2
    noise = percentile_filter(power, METHOD["profile_percentile"], size=(1, PROFILE_FRAMES), mode="nearest")
    ratio = np.clip(
        1 - METHOD["power_factor"] * noise / np.maximum(power, 1e-15), METHOD["amplitude_floor"] ** 2, 1
    )
    gain = uniform_filter(np.sqrt(ratio), size=(3, 5), mode="nearest")
    _, reconstructed = istft(spectrum * gain, **settings)
    require(
        len(reconstructed) >= len(wave) and np.isfinite(reconstructed).all(),
        "Local profile changed the input clock",
    )
    return reconstructed[: len(wave)].astype("<f4")


def filtered(folder, row):
    from scripts.audio_gtcrn_traffic import float_pcm

    return profile(float_pcm(sources.native(folder, row)))


def measure(rows, folder, voice, activity, phase):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech

    readings = []
    for row, native in rows:
        wave = filtered(folder, row)
        windows = []
        for a, b in sources.eligible(row, native):
            audio = wave[a:b]
            response = (
                logit(voice.predict([audio])[0]).hex()
                if window_in_speech(activity.spans(audio), 0.0, 3.0)
                else None
            )
            windows.append({"span": [a, b], "sha256": io.digest(audio.tobytes()), "logit": response})
        readings.append({"id": row.get("case", row.get("id")), "windows": windows})
        if len(readings) % 10 == 0:
            print(phase, len(readings), flush=True)
    return readings


def verify_native(rows, recordings, folder):
    if folder is None:
        return
    for (row, _), item in zip(rows, recordings, strict=True):
        wave = filtered(folder, row)
        for window in item["windows"]:
            a, b = window["span"]
            require(io.digest(wave[a:b].tobytes()) == window["sha256"], "Profile input hashes differ")


def run(folder, selection, held):
    import torch

    from scripts.audio_reduced_background_localization import assets_for_scores

    folder.mkdir(mode=0o700)
    lock = freeze()
    io.save(folder / "freeze.json", lock)
    rows, assets = sources.inputs()
    torch.set_num_threads(2)
    voice, activity = assets_for_scores({"voice_assets": assets})
    selected = measure(rows["selection"], selection, voice, activity, "selection")
    point = prior.point(rows["selection"], selected)
    io.save(folder / "point.json", point)
    print("Frozen point", point, flush=True)
    measured = measure(rows["held"], held, voice, activity, "held")
    io.save(folder / "readings.json", {"selection": selected, "held": measured})
    result = prior.evaluate(rows["held"], measured, float.fromhex(point["threshold"]))
    io.save(
        folder / "receipt.json",
        {"freeze": lock, "point": point, "selection": selected, "held": measured, "result": result},
    )


def review(path, selection, held):
    doc = io.read(path)
    rows, _ = sources.inputs()
    require(doc["freeze"] == freeze(), "Frozen local profile changed")
    for role, folder in (("selection", selection), ("held", held)):
        verify_native(rows[role], doc[role], folder)
    point = prior.point(rows["selection"], doc["selection"])
    require(
        point == doc["point"] and math.isfinite(float.fromhex(point["threshold"])), "Selection point changed"
    )
    result = prior.evaluate(rows["held"], doc["held"], float.fromhex(point["threshold"]))
    require(result == doc["result"], "Held results changed")
    print(io.encoded(result).decode(), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--held", type=Path)
    args = parser.parse_args()
    if args.phase == "run" and not (args.selection and args.held):
        parser.error("run requires --selection and --held")
    (run if args.phase == "run" else review)(args.path, args.selection, args.held)
