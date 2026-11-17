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


def aperture_reader(checkpoint, *, articulation=False):
    if (CENTER, STRIDE) != (ranked.CENTER, ranked.STRIDE):
        raise ValueError("The dense speech observations are on an unexpected clock")
    posterior, vocabulary, device = phone_posteriors(checkpoint)
    groups = [[vocabulary[name] for name in names] for names in viseme_timing.PHONES.values()]
    included = {index for ids in groups for index in ids}
    excluded = {vocabulary[token] for token in ("<pad>", "<s>", "</s>", "<unk>", "|") if token in vocabulary}
    other = sorted(set(vocabulary.values()) - included - excluded)
    labels = {vocabulary[name]: label for label, names in viseme_timing.PHONES.items() for name in names}
    if articulation:
        from app.evidence.video_viseme_articulation import PHONES

        shape_phones = {vocabulary[name]: label for label, names in PHONES.items() for name in names}

    def measure(pcm, *, include_events=False):
        probabilities = posterior(pcm)
        masses = [probabilities[:, ids].sum(1).tolist() for ids in (*groups, other)]
        series = [[round(ranked.UNIT * value) for value in row] for row in zip(*masses, strict=True)]
        if articulation:
            return series, collapse(probabilities, labels), collapse(probabilities, shape_phones)
        return (series, collapse(probabilities, labels)) if include_events else series

    return measure, device


def motion_agreement(directory, patches, valid, table, pcm):
    import numpy as np
    from safetensors.torch import load

    from app.detectors.video import av_local_agreement
    from app.evidence import video_lip_agreement

    def state(name):
        value = (directory / f"{name}.safetensors").read_bytes()
        if hashlib.sha256(value).hexdigest() != video_lip_agreement.WEIGHT_PINS[name]:
            raise ValueError("Local motion and speech weights differ from the tested assets")
        return load(value)

    visual = av_local_agreement.picture_windows(
        av_local_agreement.load_picture(state("vision")), patches, valid,
    )
    model = av_local_agreement.Agreement()
    model.load_state_dict(state("shared"), strict=True)
    times = np.asarray([round(pts * table.tb * viseme_timing.RATE) for pts in table.pts])
    evidence = av_local_agreement.similarities(model.eval(), visual, av_local_agreement.sound(pcm), times)
    _, candidate = av_local_agreement.localize(times, evidence, len(pcm))
    return video_lip_agreement.encode(candidate)


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
    models = request.get("motion_models") if len(table) >= 34 and table.pts[0] == 0 else None
    if models:
        from app.detectors.video import viseme_patch

        patches = np.zeros((len(table), viseme_patch.HEIGHT, viseme_patch.WIDTH), dtype=np.uint8)
        valid = np.zeros(len(table), dtype=bool)
    pictures = decode_stream(Path(request["video"]), table, max_side=None, timeout_s=20)
    with _landmarker(geometry) as (mp, task), closing(pictures) as frames:
        for frame in frames:
            if frame.index != len(measured) or not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("A decoded source picture differs from its timestamp")
            record = {"index": frame.index, "pts": frame.pts, "gap_units": None}
            if observed := _mesh(task, mp, frame.rgb, include_landmarks=bool(models)):
                features, box = observed[:2]
                x, y, right, bottom = box
                record.update(gap_units=round(float(median(features[:3])) * viseme_timing.SCORE_UNITS), mouth_box=box,
                              rgb_sha256=hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
                              mouth_crop_sha256=hashlib.sha256(frame.rgb[y:bottom, x:right].tobytes()).hexdigest())
                if request.get("articulation"):
                    record["shape"] = [round(float(value) * viseme_timing.SCORE_UNITS) for value in features]
                if models and (patch := viseme_patch.crop(frame.rgb, observed[2])) is not None:
                    patches[frame.index] = patch
                    valid[frame.index] = True
            measured.append(record)
    if len(measured) != len(table):
        raise ValueError("The native picture scan is incomplete")
    if sum(row["gap_units"] is not None for row in measured) < 20:
        for row in measured:
            row.pop("shape", None)
        return {"frames": measured, "phones": []}
    values = np.rint(pcm[:request["presented_samples"]] * 32768.).astype("<i2")
    # A missing or incompatible optional local model leaves the ordinary timing check available.
    extra = {}
    if models and valid.sum() >= 24:
        from app.evidence import video_lip_agreement

        try:
            if candidate := motion_agreement(Path(models), patches, valid, table, values):
                extra[video_lip_agreement.KEY] = candidate
        except (ImportError, OSError, RuntimeError, ValueError):
            pass
    if request.get("articulation"):
        decode, _ = aperture_reader(Path(request["checkpoint"]), articulation=True)
        groups, events, shape_phones = decode(values, include_events=True)
        result = {"frames": measured, "phones": events, "articulation_phones": shape_phones, **extra}
        if request.get("ranked"):
            result["aperture_prediction"] = ranked.predict(groups)
        return result
    if request.get("ranked"):
        decode, _ = aperture_reader(Path(request["checkpoint"]))
        groups, events = decode(values, include_events=True)
        return {"frames": measured, "phones": events, "aperture_prediction": ranked.predict(groups), **extra}
    phones = {name[0]: label for label, names in viseme_timing.PHONES.items() for name in names}
    decoder, _ = neural_decoder(Path(request["checkpoint"]), names=tuple(
        name for names in viseme_timing.PHONES.values() for name in names))
    return {"frames": measured, "phones": [[phones[phone], at, weight] for phone, at, weight in decoder(values)], **extra}


if __name__ == "__main__":
    print(json.dumps(observe(json.loads(sys.stdin.read())), allow_nan=False))
