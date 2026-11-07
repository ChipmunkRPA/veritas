"""Bound two complete estimates to unchanged frozen voice features, research only."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from app.detectors.audio.calibration import valid
from scripts import audio_separation_v14 as separator
from scripts.run_audio_guard_v7 import capture
from scripts.run_audio_identity_v9 import validate_feature as validate_original

WINDOW = [{"start_sample": 0, "end_sample": 64000}]


def pcm_sha(pcm):
    return hashlib.sha256(np.asarray(pcm, "<f4").tobytes()).hexdigest()


def assets(root):
    result = {}
    for name, expected in (("README.md", separator.CARD_SHA), ("pytorch_model.bin", separator.WEIGHT_SHA)):
        path = Path(root) / name
        actual = separator.digest(path)
        if actual != expected:
            raise ValueError("Pinned separator source/card bytes changed")
        result[name] = {"sha256": actual, "bytes": path.stat().st_size}
    return result


def validate(record, source, reference):
    if (record["status"] != source["status"] or record["raw"] != source["raw"]
            or record["input_pcm_sha256"] != source["pcm_sha256"]):
        raise ValueError("Original assessment/response/PCM binding changed")
    if record["status"] == "not_applicable":
        if record["raw"] is not None or record["stems"] is not None:
            raise ValueError("Unassessed inputs cannot acquire assessed estimates")
        return
    if (record["status"] != "ok" or not valid(record["raw"])
            or not isinstance(record["stems"], list) or len(record["stems"]) != 2):
        raise ValueError("Complete two-estimate assessed input required")
    for index, stem in enumerate(record["stems"]):
        if (type(stem["index"]) is not int or stem["index"] != index
                or type(stem["samples"]) is not int or stem["samples"] != 64000
                or not isinstance(stem["pcm_sha256"], str)
                or re.fullmatch(r"[a-f0-9]{64}", stem["pcm_sha256"]) is None):
            raise ValueError("Both ordered storage positions and full estimate clocks required")
        vector = np.asarray(stem["embedding"], np.float32)
        if (vector.shape != (1024,) or not np.isfinite(vector).all()
                or vector.astype(np.float64).tolist() != stem["embedding"]):
            raise ValueError("Exact finite float32 source features required")
        validate_original({**stem, "status": "ok", "windows": WINDOW}, reference)


def extract(pcm, model, voice, captured, reference, mixture_raw):
    estimates = separator.separate(pcm, model)
    stems = []
    for index, estimate in enumerate(estimates):
        raw, embedding = capture(voice, estimate, captured)
        stems.append({"index": index, "samples": len(estimate), "pcm_sha256": pcm_sha(estimate),
                      "raw": raw, "embedding": embedding.tolist(), "embedding_sha256": pcm_sha(embedding)})
    # Reference-head arithmetic is checked before any result can be journaled.
    temporary = {"status": "ok", "raw": mixture_raw, "input_pcm_sha256": pcm_sha(pcm), "stems": stems}
    validate(temporary, {"status": "ok", "raw": mixture_raw, "pcm_sha256": pcm_sha(pcm)}, reference)
    return {"stems": stems, "input_pcm_sha256": pcm_sha(pcm)}
