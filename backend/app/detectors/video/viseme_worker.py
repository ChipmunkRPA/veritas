# SPDX-License-Identifier: MIT
"""Measure a short clip in an isolated, offline local worker."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import closing
from pathlib import Path
from statistics import median

from app.detectors.audio.phone_events import CENTER, STRIDE, collapse, neural_decoder, phone_posteriors
from app.evidence import video_viseme_ranked as ranked
from app.evidence import video_viseme_timing as viseme_timing


def aperture_reader(checkpoint):
    if (CENTER, STRIDE) != (ranked.CENTER, ranked.STRIDE):
        raise ValueError("The dense speech observations are on an unexpected clock")
    posterior, vocabulary, device = phone_posteriors(checkpoint)
    groups = [[vocabulary[name] for name in names] for names in viseme_timing.PHONES.values()]
    included = {index for ids in groups for index in ids}
    excluded = {vocabulary[token] for token in ("<pad>", "<s>", "</s>", "<unk>", "|") if token in vocabulary}
    other = sorted(set(vocabulary.values()) - included - excluded)
    labels = {vocabulary[name]: label for label, names in viseme_timing.PHONES.items() for name in names}

    def measure(pcm, *, include_events=False):
        probabilities = posterior(pcm)
        masses = [probabilities[:, ids].sum(1).tolist() for ids in (*groups, other)]
        series = [[round(ranked.UNIT * value) for value in row] for row in zip(*masses, strict=True)]
        return (series, collapse(probabilities, labels)) if include_events else series

    return measure, device


def observe(request):
    if request["runtime"]:
        sys.path.insert(0, request["runtime"])

    import numpy as np

    from app.detectors.audio.decode import read_wav
    from app.detectors.video.decode import FrameTable, decode_stream
    from app.evidence.video_reference_timing import _landmarker, _mesh, available

    table = FrameTable.from_json(request["frames"])
    geometry = Path(request["geometry"])
    if not available(geometry):
        raise ValueError("The verified facial geometry assets are unavailable")
    pcm = read_wav(Path(request["wav"]).read_bytes())
    if len(pcm) != request["samples"]:
        raise ValueError("The frozen soundtrack length differs")
    measured = []
    pictures = decode_stream(Path(request["video"]), table, max_side=None, timeout_s=20)
    with _landmarker(geometry) as (mp, task), closing(pictures) as frames:
        for frame in frames:
            if frame.index != len(measured) or not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("A decoded source picture differs from its timestamp")
            record = {"index": frame.index, "pts": frame.pts, "gap_units": None}
            if observed := _mesh(task, mp, frame.rgb):
                features, box = observed
                x, y, right, bottom = box
                record.update(gap_units=round(float(median(features[:3])) * viseme_timing.SCORE_UNITS), mouth_box=box,
                              rgb_sha256=hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
                              mouth_crop_sha256=hashlib.sha256(frame.rgb[y:bottom, x:right].tobytes()).hexdigest())
            measured.append(record)
    if len(measured) != len(table):
        raise ValueError("The native picture scan is incomplete")
    if sum(row["gap_units"] is not None for row in measured) < 20:
        return {"frames": measured, "phones": []}
    values = np.rint(pcm[:request["presented_samples"]] * 32768.).astype("<i2")
    if request.get("ranked"):
        decode, _ = aperture_reader(Path(request["checkpoint"]))
        groups, events = decode(values, include_events=True)
        return {"frames": measured, "phones": events, "aperture_prediction": ranked.predict(groups)}
    phones = {name[0]: label for label, names in viseme_timing.PHONES.items() for name in names}
    decoder, _ = neural_decoder(Path(request["checkpoint"]), names=tuple(
        name for names in viseme_timing.PHONES.values() for name in names))
    return {"frames": measured, "phones": [[phones[phone], at, weight] for phone, at, weight in decoder(values)]}


if __name__ == "__main__":
    print(json.dumps(observe(json.loads(sys.stdin.read())), allow_nan=False))
