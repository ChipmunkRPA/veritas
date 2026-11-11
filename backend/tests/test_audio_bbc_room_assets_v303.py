from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
import soundfile as sf

from scripts import audio_bbc_room_assets_v303 as bbc


def encode(values, rate=bbc.RATE, subtype="FLOAT"):
    handle = io.BytesIO()
    sf.write(handle, values, rate, format="WAV", subtype=subtype)
    return handle.getvalue()


@pytest.mark.parametrize("encoding", ["FLOAT", "PCM_32"])
def test_bbc_preserves_original_stereo_clock_and_units(encoding):
    values = np.zeros((bbc.SAMPLES, 2), dtype="<f8")
    values[[97, 119], [0, 1]] = [0.5, -0.25]
    impulse, actual_encoding = bbc.parse_impulse(encode(values, subtype=encoding), "left")
    assert impulse.delay == (0, 0)
    assert actual_encoding == encoding
    assert impulse.identity()["samples_per_channel"] == 16_384
    assert np.array_equal(impulse.coefficients, values)


def test_bbc_rejects_wrong_native_geometry():
    frames = np.zeros((bbc.SAMPLES, 2))
    with pytest.raises(ValueError, match="native BBC stereo"):
        bbc.parse_impulse(encode(frames, rate=44_100), "front")
    with pytest.raises(ValueError, match="native BBC stereo"):
        bbc.parse_impulse(encode(frames[:-1]), "front")
    with pytest.raises(ValueError, match="native BBC stereo"):
        bbc.parse_impulse(encode(frames[:, 0]), "front")
    with pytest.raises(ValueError, match="native BBC stereo"):
        bbc.parse_impulse(encode(frames, subtype="PCM_16"), "front")
    with pytest.raises(ValueError, match="distinct ear channels"):
        bbc.parse_impulse(encode(frames), "front")
    with pytest.raises(ValueError, match="three documented BBC positions"):
        bbc.parse_impulse(encode(frames), "rear")


def test_bbc_intake_qualifies_only_three_pinned_files(tmp_path, monkeypatch):
    authored = {}
    for index, position in enumerate(bbc.POSITIONS):
        values = np.zeros((bbc.SAMPLES, 2))
        values[[99 + index, 101 - index], [0, 1]] = [0.5, -0.25]
        raw = encode(values)
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        asset = bbc.prior.Asset(f"https://example.test/{position}.wav", len(raw), blob)
        (tmp_path / asset.filename).write_bytes(raw)
        authored[position] = asset
    monkeypatch.setattr(bbc, "ASSETS", authored)
    result = bbc.qualify(tmp_path)
    assert result["status"] == "documented_positions_qualified"
    assert result["positions"]["left"]["peak_sample_left_right"] == [100, 100]
    assert {row["impulse"]["position"] for row in result["positions"].values()} == set(bbc.POSITIONS)
    path = tmp_path / "right.wav"
    path.write_bytes(b"bad" + path.read_bytes()[3:])
    with pytest.raises(ValueError, match="Publisher bytes differ"):
        bbc.qualify(tmp_path)


def test_publisher_plan_binds_geometry_and_corrected_static_license():
    data = bbc.plan()
    assert data["license"] == "CC-BY-SA-4.0"
    assert data["native_range"] == [0, 16_384]
    assert data["ear_order"] == ["left", "right"]
    assert data["source_commit"] in data["publisher_sources"]["licence_correction"]
    assert [row["azimuth_degrees_left_positive"] for row in data["positions"].values()] == [0, 90, -90]
    assert {row["asset"]["bytes"] for row in data["positions"].values()} == {131_160}
