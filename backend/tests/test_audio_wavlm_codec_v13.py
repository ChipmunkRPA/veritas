"""Transport preserves original bits and refuses rounding or malformed payloads."""
import base64
import copy
import hashlib
import json
import math
import struct

import pytest

from scripts import audio_wavlm_codec_v13 as codec


def record():
    return {"id": "public-id", "wavlm": [-0., 1., -2., 2**-149] + [0.] * (codec.WIDTH - 4),
            "coefficients": [math.pi], "runtime_s": math.e}


def test_original_json_and_signed_bits_round_trip():
    original = record()
    packed = codec.pack(original)
    restored = codec.unpack(json.loads(json.dumps(packed)))
    assert json.dumps(restored) == json.dumps(original)
    assert struct.pack("<f", restored["wavlm"][0]) == b"\0\0\0\x80"
    assert restored["coefficients"] == [math.pi] and restored["runtime_s"] == math.e
    assert codec.unpack(codec.pack({"wavlm": None})) == {"wavlm": None}


@pytest.mark.parametrize("bad", [None, [0.] * 4607, [0.] * 4609, [0] * 4608,
                               [.1] * 4608, [float("nan")] * 4608, [float("inf")] * 4608])
def test_rounding_or_invalid_original_is_not_hidden(bad):
    if bad is None:
        assert codec.pack({"wavlm": bad}) == {"wavlm": None}
    else:
        with pytest.raises(ValueError):
            codec.pack({"wavlm": bad})


@pytest.mark.parametrize("fault", ["digest", "shape", "length", "extra", "nan"])
def test_corrupt_payload_is_refused(fault):
    packed = copy.deepcopy(codec.pack(record())["wavlm"])
    if fault == "digest":
        packed["sha256"] = "0" * 64
    elif fault == "shape":
        packed["shape"] = [3, 1536]
    elif fault == "length":
        packed["data"] += "A"
    elif fault == "extra":
        packed["unknown"] = True
    else:
        raw = struct.pack(f"<{codec.WIDTH}f", *([float("nan")] + [0.] * (codec.WIDTH - 1)))
        packed["data"] = base64.b64encode(raw).decode()
        packed["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        codec.unpack(packed)
