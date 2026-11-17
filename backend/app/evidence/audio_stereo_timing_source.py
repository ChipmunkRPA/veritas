# SPDX-License-Identifier: Apache-2.0
"""Render and replay a measured stereo timing change on the original clock."""

import hashlib
import io
from copy import deepcopy

import numpy as np

from app.evidence import audio_stereo_timing_binding as binding
from app.evidence import audio_stereo_timing_change as timing
from app.evidence.stereo_repetition import _pcm_wav
from app.storage.objects import ObjectIntegrityError


def _plots(stereo, start, stop):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def chart(title, unit):
        figure = Figure(figsize=(9.6, 2.8), dpi=100, facecolor="#fafaf7")
        figure.subplots_adjust(left=.105, right=.982, bottom=.20, top=.86)
        axes = figure.add_subplot(111)
        axes.axvspan(start / timing.RATE, stop / timing.RATE, color="#cadfd5", alpha=.75)
        axes.axhline(0, color="#919b94", linewidth=.7)
        axes.set(xlim=(0., len(stereo) / timing.RATE), xlabel="Analyzed audio (seconds)", ylabel=unit, title=title)
        axes.grid(axis="y", alpha=.22)
        return figure, axes

    def png(figure):
        output = io.BytesIO()
        FigureCanvasAgg(figure).print_png(output)
        return output.getvalue()

    times, offsets, measured = timing.windows(stereo)
    delay, axis = chart("Measured delay between the stereo channels", "Right vs. left (ms)")
    axis.scatter(times[measured] / timing.RATE, offsets[measured] * 1000 / timing.RATE, s=9, color="#245d44")

    waveform, axis = chart("Original left-channel waveform", "Amplitude")
    left = stereo[:, 0].astype(np.float32) / 32768.
    edges = np.linspace(0, len(left), 801, dtype=int)
    midpoints = (edges[:-1] + edges[1:]) / (2 * timing.RATE)
    axis.vlines(midpoints, np.minimum.reduceat(left, edges[:-1]), np.maximum.reduceat(left, edges[:-1]),
                color="#245d44", linewidth=.52)
    return {"timing_plot": png(delay), "waveform": png(waveform)}


def build(data, source, asset):
    extent = binding.extent(source)
    if extent is None or len(data) != 4 * extent:
        raise ValueError("Stereo timing data differs from the presented audio clock")
    stereo = np.frombuffer(data, dtype="<i2").reshape(-1, 2)
    assessed, reading = timing.locate(stereo)
    context = {"schema": binding.SCHEMA, "source_sha256": source["sha256"], "sample_rate": timing.RATE,
               "input_samples": extent, "stereo_pcm_sha256": hashlib.sha256(data).hexdigest(),
               "affects_overall_confidence": False, "status": "measured" if assessed else "unavailable",
               "measurement": reading, "clips": [], "artifacts": {}}
    if reading:
        start, stop = reading["start_sample"], reading["end_sample"]
        origin = source["timeline"][0]["source_pts"] * binding.NATIVE_FACTOR
        for role, (low, high) in zip(binding.ROLES, binding.ranges(start, stop, extent), strict=True):
            payload = data[4 * low:4 * high]
            context["clips"].append({"role": role, "start_sample": low, "end_sample": high,
                                     "pcm_s16le_sha256": hashlib.sha256(payload).hexdigest(),
                                     "source_timestamp_ranges": [{"start_sample": low, "end_sample": high,
                                         "source_start_pts": origin + low, "source_end_pts": origin + high,
                                         "time_base": "1/48000"}]})
            context["artifacts"][role] = asset(_pcm_wav(payload, rate=timing.RATE), "audio/wav", channels=2,
                                               samples=high - low, sample_rate=timing.RATE)
        for role, image in _plots(stereo, start, stop).items():
            context["artifacts"][role] = asset(image, "image/png", width=960, height=280)
    binding.validate(context, source)
    return context


def verify(record, data, source, read_asset):
    def checked(payload, media_type, **metadata):
        sha = hashlib.sha256(payload).hexdigest()
        if read_asset(sha) != payload:
            raise ValueError("Stereo timing evidence differs from the submitted samples")
        return {"sha256": sha, "media_type": media_type, **metadata}

    binding.validate(record, source)
    declared = deepcopy(record)
    for asset in declared["artifacts"].values():
        asset.pop("url", None)
    if build(data, source, checked) != declared:
        raise ValueError("Stereo timing change does not reproduce on its source clock")


def verify_frozen(store, document, digest, read_asset):
    record = document.get(binding.KEY)
    if (digest is None) != (record is None):
        raise ObjectIntegrityError("Stereo timing availability differs from the frozen audio input")
    if digest is not None:
        if record["stereo_pcm_sha256"] != digest:
            raise ObjectIntegrityError("Stereo timing commits to different audio samples")
        store.verify(digest)
        try:
            verify(record, store.read_bytes(digest), document["source"], read_asset)
        except (KeyError, ValueError) as error:
            raise ObjectIntegrityError("Stereo timing evidence differs from the frozen audio channels") from error
