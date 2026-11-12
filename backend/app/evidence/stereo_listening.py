"""Source-locked comparison of decoded stereo channels with the checked mono input."""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageDraw

from app.detectors.audio.decode import SAMPLE_RATE, wav_bytes
from app.evidence.stereo_listening_binding import CHANNELS, DIMENSIONS, SCHEMA, select_windows, validate
from app.storage.artefacts import register_artefact
from app.storage.objects import ObjectIntegrityError

PLOT_BOX = (108, 32, 933, 262)


def selected_pcm(stereo: dict, pcm: np.ndarray) -> bytes:
    return b"".join(
        pcm[stereo["windows"][i]["start_sample"]:stereo["windows"][i]["end_sample"]].astype("<i2", copy=False).tobytes()
        for i in select_windows(stereo)
    )


def freeze(db, store, analysis_id: str, decoded):
    stereo = decoded.mapping.get("stereo_context")
    if decoded.stereo_pcm is None or not stereo:
        return None
    data = selected_pcm(stereo, decoded.stereo_pcm)
    if not data:
        return None
    return register_artefact(db, store, data, media_type="application/octet-stream", role="stage_output",
                             created_by=SCHEMA, analysis_id=analysis_id)


def _waveform(tracks: tuple[np.ndarray, ...]) -> bytes:
    im = Image.new("RGB", DIMENSIONS, "#faf9f5")
    draw = ImageDraw.Draw(im)
    x0, x1 = PLOT_BOX[0], PLOT_BOX[2] - 1
    draw.text((x0, 10), "Same amplitude scale | Time within clip", fill="#203b30")
    labels = ("Left channel", "Right channel", "Analyzed mono")
    colors = ("#176853", "#315b96", "#9b511e")
    for index, (track, label, color) in enumerate(zip(tracks, labels, colors, strict=True)):
        mid = 67 + 80 * index
        draw.text((10, mid - 7), label, fill="#203b30")
        draw.rectangle((x0, mid - 34, x1, mid + 34), outline="#c6cbc0")
        draw.line((x0, mid, x1, mid), fill="#c6cbc0")
        for i, chunk in enumerate(np.array_split(track, x1 - x0)):
            draw.line((x0+i, mid-float(chunk.max())*32, x0+i, mid-float(chunk.min())*32), fill=color)
    for i in range(5):
        draw.text((x0 - 12 + i * (x1 - x0) / 4, 267), f"{i/4:.2f}s", fill="#203b30")
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def payloads(data: bytes, stereo: dict, pcm: np.ndarray, *, waveform: bool = True):
    selected = select_windows(stereo)
    if len(data) != len(selected) * 4 * SAMPLE_RATE or not selected:
        raise ObjectIntegrityError("Stereo listening data does not cover the selected sample intervals")
    pairs = np.frombuffer(data, dtype="<i2").reshape(len(selected), SAMPLE_RATE, 2)
    for index, pair in zip(selected, pairs, strict=True):
        window = stereo["windows"][index]
        start, end = window["start_sample"], window["end_sample"]
        tracks = (pair[:, 0].astype(np.float32) / 32768, pair[:, 1].astype(np.float32) / 32768, pcm[start:end])
        files = {name: wav_bytes(track) for name, track in zip(CHANNELS, tracks, strict=True)}
        if waveform:
            files["waveform"] = _waveform(tracks)
        yield index, files


def _frozen_data(store, digest: str) -> bytes:
    store.verify(digest)
    return store.read_bytes(digest)


def build(store, digest: str | None, stereo: dict | None, pcm: np.ndarray, asset) -> dict | None:
    if digest is None:
        return None
    if not stereo:
        raise ObjectIntegrityError("Stereo listening requires its source-clock measurements")
    rows = []
    for index, files in payloads(_frozen_data(store, digest), stereo, pcm):
        artifacts = {name: asset(files[name], "audio/wav", samples=SAMPLE_RATE, sample_rate=SAMPLE_RATE, channels=1)
                     for name in CHANNELS}
        artifacts["waveform"] = asset(files["waveform"], "image/png", width=DIMENSIONS[0], height=DIMENSIONS[1])
        rows.append({"stereo_window_index": index, "artifacts": artifacts})
    return {"schema": SCHEMA, "affects_overall_confidence": False, "windows": rows}


def verify_payloads(listening: dict, stereo: dict, pcm: np.ndarray, data: bytes, *, read_waveform=None) -> set[str]:
    try:
        validate(listening, stereo)
    except ValueError as error:
        raise ObjectIntegrityError(str(error)) from error
    wav_digests = set()
    for row, (_, files) in zip(listening["windows"], payloads(data, stereo, pcm, waveform=read_waveform is not None), strict=True):
        for name, body in files.items():
            recorded = row["artifacts"][name]["sha256"]
            if name == "waveform":
                # Replay plot pixels independent of font rendering and PNG encoding.
                try:
                    with Image.open(io.BytesIO(body)) as expected, Image.open(io.BytesIO(read_waveform(recorded))) as displayed:
                        if displayed.convert("RGB").crop(PLOT_BOX).tobytes() == expected.crop(PLOT_BOX).tobytes():
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
