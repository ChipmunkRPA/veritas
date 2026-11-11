from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
import soundfile as sf
from scipy.io import savemat

from scripts import audio_room_assets_v299 as assets


def mat(**values):
    buffer = io.BytesIO()
    savemat(buffer, values, do_compression=True)
    return buffer.getvalue()


def wav(values, rate=44_100, subtype="FLOAT"):
    buffer = io.BytesIO()
    sf.write(buffer, values, rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


def test_surrey_documented_axes_and_delays():
    coefficients = np.zeros((24, 2, 24), dtype=np.float32)
    for channel in range(24):
        coefficients[[5, 7], [0, 1], channel] = [0.125 * (channel + 1), -0.25 * (channel + 1)]
    positions = assets.surrey_impulses(mat(brir=coefficients))
    assert tuple(positions) == ("FC", "SiL", "SiR")
    for speaker, index in (("FC", 2), ("SiL", 10), ("SiR", 11)):
        impulse = positions[speaker]
        assert impulse.identity()["sample_rate_hz"] == 48_000
        assert impulse.delay == (0, 0)
        assert impulse.coefficients.dtype == "<f8"
        assert np.array_equal(impulse.coefficients, coefficients[:, :, index])


def test_surrey_rejects_unreviewed_layout_and_metadata():
    with pytest.raises(ValueError, match="samples × ears × loudspeakers"):
        assets.surrey_impulses(mat(brir=np.ones((24, 2, 12))))
    with pytest.raises(ValueError, match="One documented"):
        assets.surrey_impulses(mat(brir=np.ones((12, 2, 24)), delay=np.array([1, 2])))
    with pytest.raises(ValueError, match="finite real"):
        assets.surrey_impulses(mat(brir=np.ones((12, 2, 24)) * (1 + 1j)))


@pytest.mark.parametrize("subtype", ["FLOAT", "PCM_32"])
def test_huddersfield_preserves_linear_native_coefficients(subtype):
    linear = np.array([0.0, -0.125, 0.5, 0.0], dtype="<f8")
    result = assets.huddersfield_channel(wav(linear, subtype=subtype))
    assert result == {
        "sample_rate_hz": 44_100,
        "encoding": subtype,
        "samples": 4,
        "coefficients_sha256": hashlib.sha256(linear.tobytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="44.1 kHz 32-bit"):
        assets.huddersfield_channel(wav(linear, rate=48_000, subtype=subtype))
    with pytest.raises(ValueError, match="44.1 kHz 32-bit"):
        assets.huddersfield_channel(wav(linear, subtype="PCM_24"))


def test_publisher_pin_checks_exact_blob_and_blocks_symlink(tmp_path):
    body = b"test"
    expected = hashlib.sha1(b"blob 4\0" + body).hexdigest()
    asset = assets.Asset("https://example.test/a.wav", 4, expected)
    target = tmp_path / asset.filename
    target.write_bytes(body)
    raw, record = assets.pinned_bytes(tmp_path, asset)
    assert raw == body and record["sha256"] == hashlib.sha256(body).hexdigest()
    target.write_bytes(b"tEst")
    with pytest.raises(ValueError, match="Publisher bytes differ"):
        assets.pinned_bytes(tmp_path, asset)
    target.unlink()
    target.symlink_to(tmp_path / "b.wav")
    with pytest.raises(OSError):
        assets.pinned_bytes(tmp_path, asset)


def test_huddersfield_candidates_remain_without_geometry(monkeypatch):
    def read_pair_member(_directory, asset):
        value = int(asset.git_blob_sha1[:8], 16) / 2**32
        return wav(np.array([value], dtype=np.float32)), {"filename": asset.filename}

    monkeypatch.setattr(assets, "pinned_bytes", read_pair_member)
    result = assets.check_huddersfield(None)
    assert result["status"] == "bytes_and_format_verified_angles_unassigned"
    assert result["positions"] == {}
    assert set(result["candidate_pairs"]) == {"S01", "S03", "S07"}
    assert all(
        [row["ear"] for row in pair] == ["left", "right"] for pair in result["candidate_pairs"].values()
    )
    assert assets.publisher_plan()["holdout"]["speaker_id_angle_mapping"] is None
