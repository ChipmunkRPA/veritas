"""Source-locked comparison of decoded stereo channels with the checked mono input."""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageDraw

from app.detectors.audio.decode import SAMPLE_RATE, wav_bytes
from app.evidence.stereo_binding import destructive_downmix_windows
from app.evidence.stereo_listening_binding import (
    CHANNELS,
    DIFFERENCE_DIMENSIONS,
    DIFFERENCE_SCHEMA,
    DIMENSIONS,
    LOCAL_EXCERPTS,
    SCHEMA,
    all_artifacts,
    intervals,
    select_windows,
    validate,
)
from app.storage.artefacts import register_artefact
from app.storage.objects import ObjectIntegrityError

PLOT_BOX = (108, 32, 933, 262)


def _excerpts(stereo):
    for index in select_windows(stereo):
        window = stereo["windows"][index]
        yield index, window["start_sample"], window["end_sample"]
    for key in LOCAL_EXCERPTS:
        for start, end in intervals(stereo, key):
            yield key, start, end


def selected_pcm(stereo: dict, pcm: np.ndarray) -> bytes:
    return b"".join(pcm[start:end].astype("<i2", copy=False).tobytes() for _, start, end in _excerpts(stereo))


def freeze(db, store, analysis_id: str, decoded):
    stereo = decoded.mapping.get("stereo_context")
    if decoded.stereo_pcm is None or not stereo:
        return None
    data = selected_pcm(stereo, decoded.stereo_pcm)
    if not data:
        return None
    return register_artefact(db, store, data, media_type="application/octet-stream", role="stage_output",
                             created_by=SCHEMA, analysis_id=analysis_id)


def _waveform(tracks: tuple[np.ndarray, ...], *, flank: int = 0) -> bytes:
    dimensions = DIFFERENCE_DIMENSIONS if len(tracks) > len(CHANNELS) else DIMENSIONS
    im = Image.new("RGB", dimensions, "#faf9f5")
    draw = ImageDraw.Draw(im)
    x0, x1 = PLOT_BOX[0], PLOT_BOX[2] - 1
    draw.text((x0, 10), "Same amplitude scale | Time within clip", fill="#203b30")
    labels = ("Left channel", "Right channel", "Analyzed mono", "Channel difference")[:len(tracks)]
    colors = ("#176853", "#315b96", "#9b511e", "#74518a")[:len(tracks)]
    sample_count = len(tracks[0])
    for index, (track, label, color) in enumerate(zip(tracks, labels, colors, strict=True)):
        mid = 67 + 80 * index
        draw.text((10, mid - 7), label, fill="#203b30")
        draw.rectangle((x0, mid - 34, x1, mid + 34), outline="#c6cbc0")
        if flank:
            margin = round((x1 - x0) * flank / sample_count)
            draw.rectangle((x0 + margin, mid - 33, x1 - margin, mid + 33), fill="#f0e4ca")
        draw.line((x0, mid, x1, mid), fill="#c6cbc0")
        for i, chunk in enumerate(np.array_split(track, x1 - x0)):
            draw.line((x0+i, mid-float(chunk.max())*32, x0+i, mid-float(chunk.min())*32), fill=color)
    for i in range(5):
        draw.text((x0 - 12 + i * (x1 - x0) / 4, dimensions[1] - 23), f"{i * sample_count / (4 * SAMPLE_RATE):.2f}s", fill="#203b30")
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def payloads(data: bytes, stereo: dict, pcm: np.ndarray, *, waveform: bool = True, include_difference: bool = False):
    excerpts = list(_excerpts(stereo))
    cancelled = set(destructive_downmix_windows(stereo)) if include_difference else set()
    if not excerpts or len(data) != 4 * sum(end - start for _, start, end in excerpts):
        raise ObjectIntegrityError("Stereo listening data does not cover the selected sample intervals")
    values = np.frombuffer(data, dtype="<i2").reshape(-1, 2)
    offset = 0
    for index, start, end in excerpts:
        count = end - start
        pair = values[offset:offset + count]
        offset += count
        tracks = (pair[:, 0].astype(np.float32) / 32768, pair[:, 1].astype(np.float32) / 32768, pcm[start:end])
        files = {name: wav_bytes(track) for name, track in zip(CHANNELS, tracks, strict=True)}
        if index in cancelled:
            difference = (tracks[0] - tracks[1]) / 2
            files["difference"] = wav_bytes(difference)
            tracks = (*tracks, difference)
        if waveform:
            files["waveform"] = _waveform(tracks, flank=LOCAL_EXCERPTS.get(index, 0))
        yield index, count, files


def _frozen_data(store, digest: str) -> bytes:
    store.verify(digest)
    return store.read_bytes(digest)


def build(store, digest: str | None, stereo: dict | None, pcm: np.ndarray, asset, *, include_difference: bool = False) -> dict | None:
    if digest is None:
        return None
    if not stereo:
        raise ObjectIntegrityError("Stereo listening requires its source-clock measurements")
    difference = include_difference and bool(set(select_windows(stereo)).intersection(destructive_downmix_windows(stereo)))
    result = {"schema": DIFFERENCE_SCHEMA if difference else SCHEMA, "affects_overall_confidence": False, "windows": []}
    for index, count, files in payloads(_frozen_data(store, digest), stereo, pcm, include_difference=difference):
        artifacts = {name: asset(body, "audio/wav", samples=count, sample_rate=SAMPLE_RATE, channels=1)
                     for name, body in files.items() if name != "waveform"}
        width, height = DIFFERENCE_DIMENSIONS if "difference" in files else DIMENSIONS
        artifacts["waveform"] = asset(files["waveform"], "image/png", width=width, height=height)
        if isinstance(index, str):
            result.setdefault(index, []).append({"artifacts": artifacts})
        else:
            result["windows"].append({"stereo_window_index": index, "artifacts": artifacts})
    return result


def verify_payloads(listening: dict, stereo: dict, pcm: np.ndarray, data: bytes, *, read_waveform=None) -> set[str]:
    try:
        validate(listening, stereo)
    except ValueError as error:
        raise ObjectIntegrityError(str(error)) from error
    wav_digests = set()
    expected_payloads = payloads(data, stereo, pcm, waveform=read_waveform is not None,
                                 include_difference=listening["schema"] == DIFFERENCE_SCHEMA)
    for assets, (_, _, files) in zip(all_artifacts(listening), expected_payloads, strict=True):
        for name, body in files.items():
            recorded = assets[name]["sha256"]
            if name == "waveform":
                # Replay plot pixels independent of font rendering and PNG encoding.
                try:
                    x0, y0, x1, y1 = PLOT_BOX
                    plot = (x0, y0, x1, y1 + 80) if "difference" in files else PLOT_BOX
                    with Image.open(io.BytesIO(body)) as expected, Image.open(io.BytesIO(read_waveform(recorded))) as displayed:
                        if displayed.size == expected.size and displayed.convert("RGB").crop(plot).tobytes() == expected.crop(plot).tobytes():
                            continue
                except (OSError, SyntaxError) as error:
                    raise ObjectIntegrityError("Stereo listening waveform could not be decoded") from error
                raise ObjectIntegrityError("Stereo listening waveform differs from its committed channel samples")
            digest = hashlib.sha256(body).hexdigest()
            if digest != recorded:
                raise ObjectIntegrityError("Stereo listening asset differs from its committed channel samples")
            wav_digests.add(digest)
    return wav_digests


def verify(store, document: dict, digest: str | None, pcm: np.ndarray, *, read_waveform=None) -> None:
    listening = document.get("stereo_listening")
    if (digest is None) != (listening is None):
        raise ObjectIntegrityError("Stereo listening availability differs from the frozen audio input")
    if digest is not None:
        verify_payloads(listening, document["stereo_context"], pcm, _frozen_data(store, digest), read_waveform=read_waveform)
