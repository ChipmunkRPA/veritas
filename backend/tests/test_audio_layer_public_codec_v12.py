"""Lossless public-vector transport, corruption and numerical-boundary checks."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import struct

import numpy as np
import pytest

from scripts import audio_layer_public_codec_v12 as codec
from scripts.review_audio_layer_v12 import normalization, probability


def vector():
    return np.linspace(-1, 1, 1024, dtype=np.float32).tolist()


def test_vector_roundtrip_restores_original_json_bytes_and_signed_zero():
    v = vector()
    v[0] = -0.0
    record = {"id": "source", "raw": .512345, "embedding": v,
              "layer_embeddings": [v, list(reversed(v)), [float(np.float32(x * .5)) for x in v]],
              "layer_ids": [8, 16, 24], "runtime_s": .01357}
    old = {"source": {"embedding": v}}
    before = json.dumps(record, allow_nan=False)
    packed = codec.pack(record, old)
    assert packed["embedding"]["format"] == codec.REFERENCE
    assert packed["layer_embeddings"]["format"] == codec.FORMAT
    assert json.dumps(codec.unpack(packed, old), allow_nan=False) == before
    assert json.dumps(record, allow_nan=False) == before


@pytest.mark.parametrize("damage", ["transfer", "shape", "extra", "nonfinite", "length"])
def test_packed_payload_cannot_hide_corruption(damage):
    packed = codec.pack(vector(), key="embedding")
    if damage == "transfer":
        packed["data"] = "AAAA" + packed["data"][4:]
    elif damage == "shape":
        packed["shape"] = [1, 1024]
    elif damage == "extra":
        packed["unknown"] = "not a vector field"
    else:
        raw = base64.b64decode(packed["data"])
        raw = struct.pack("<f", math.nan) + raw[4:] if damage == "nonfinite" else raw[:-4]
        packed["data"] = base64.b64encode(raw).decode()
        packed["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        codec.unpack(packed)


@pytest.mark.parametrize("damage", ["rounded", "integer", "nan", "missing_layer"])
def test_transport_refuses_new_precision_or_incomplete_layers(damage):
    v = vector()
    if damage == "rounded":
        v[0] = .123456789
    elif damage == "integer":
        v[0] = 0
    elif damage == "nan":
        v[0] = math.nan
    else:
        with pytest.raises(ValueError):
            codec.pack([v, v], key="layer_embeddings")
        return
    with pytest.raises(ValueError):
        codec.pack(v, key="embedding")


def test_previous_vector_reference_requires_exact_float32_bytes():
    v = vector()
    packed = codec.pack({"id": "a", "embedding": v}, {"a": {"embedding": v}})
    with pytest.raises(ValueError):
        codec.unpack(packed, {})
    changed = copy.deepcopy(v)
    changed[0] = 0.0
    with pytest.raises(ValueError):
        codec.unpack(packed, {"a": {"embedding": changed}})
    v[0] = -0.0
    changed[0] = 0.0
    with pytest.raises(ValueError):
        codec.pack({"id": "a", "embedding": v}, {"a": {"embedding": changed}})


def fitted():
    coefficients = [0.0] * 1026
    coefficients[1] = 1.0
    coefficients[-1] = .1
    return {"arm": "final_control", "coefficients": coefficients,
            "transform": {"center": [0.0] * 1024, "scale": [1.0] * 1024, "variance_factor": .125}}


def test_independent_probability_matches_hand_calculated_logistic_example():
    row = {"raw": .5, "embedding": [2.0] + [0.0] * 1023}
    # Only first representation coordinate and intercept contribute: 2/8 + 0.1.
    assert probability(row, fitted()) == pytest.approx(1 / (1 + math.exp(-.35)), abs=1e-15)
    assert probability({"raw": None}, fitted()) is None


@pytest.mark.parametrize("damage", ["raw_bool", "raw_nan", "shape", "slope", "scale_nan", "coefficient_nan"])
def test_independent_predictor_rejects_changed_numeric_contract(damage):
    row, head = {"raw": .5, "embedding": [0.0] * 1024}, fitted()
    if damage == "raw_bool":
        row["raw"] = True
    elif damage == "raw_nan":
        row["raw"] = math.nan
    elif damage == "shape":
        row["embedding"].pop()
    elif damage == "slope":
        head["coefficients"][0] = -1
    elif damage == "scale_nan":
        head["transform"]["scale"][0] = math.nan
    else:
        head["coefficients"][0] = math.nan
    with pytest.raises(ValueError):
        probability(row, head)


def test_calibration_sources_cannot_learn_normalization():
    with pytest.raises(ValueError, match="fitting sources only"):
        normalization([{"role": "cal"}], fitted())
