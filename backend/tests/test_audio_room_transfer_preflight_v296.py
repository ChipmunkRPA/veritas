from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts import audio_room_transfer_preflight_v296 as room


def test_shared_source_gain_retains_position_asymmetry_and_exact_residual():
    speech, noise, front, left, right = room.fixtures()
    cases = room.compose(speech, noise, front, left, right)
    clean, clean_record = cases[room.CONDITIONS[0]]
    left_case, left_record = cases["noise_left"]
    right_case, right_record = cases["noise_right"]
    assert clean_record["background"] is None
    assert len({record["common_headroom_gain"] for _, record in cases.values()}) == 1
    assert len({record["foreground_stereo_pcm_sha256"] for _, record in cases.values()}) == 1
    assert left_record["background_source_gain"] == pytest.approx(right_record["background_source_gain"])
    assert left_record["background"]["source_pcm_sha256"] == right_record["background"]["source_pcm_sha256"]
    assert (
        left_record["background"]["impulse"]["coefficients_sha256"]
        != right_record["background"]["impulse"]["coefficients_sha256"]
    )
    for pcm, record in ((left_case, left_record), (right_case, right_record)):
        residual = pcm.astype(np.int32) - clean
        assert record["background_stereo_pcm_sha256"] == room.stereo.digest(residual.astype("<i2").tobytes())
        assert record["snr_db"]["joint"] == pytest.approx(-10, abs=0.1)
    left_snr, right_snr = (row["snr_db"]["channels"] for row in (left_record, right_record))
    assert left_snr[1] - left_snr[0] > 10
    assert right_snr[0] - right_snr[1] > 10


def test_geometry_units_and_input_identity_fail_closed():
    speech, noise, front, left, right = room.fixtures()
    with pytest.raises(ValueError, match="distinct source positions"):
        room.compose(speech, noise, front, left, left)
    with pytest.raises(ValueError, match="same native receiver pair"):
        room.compose(speech, noise, front, left, replace(right, rig="different"))
    with pytest.raises(ValueError, match="source content must differ"):
        room.compose(speech, speech, front, left, right)
    with pytest.raises(ValueError, match="linear impulse unit"):
        room.render(noise, replace(left, unit="decibel"))
    with pytest.raises(ValueError, match="integer samples"):
        room.render(noise, replace(left, delay=(1.5, 2)))
    with pytest.raises(ValueError, match="nonzero finite linear impulse pair"):
        room.render(noise, replace(left, coefficients=np.array([[1, 1]])))


def test_native_resampling_preserves_relative_arrival_and_output_extent():
    source = np.zeros(room.SAMPLES, dtype="<i2")
    source[100] = 1024
    impulse = room.Impulse("authored", "front", 48_000, room.IMPULSE_UNIT, (3, 12), np.array([[0.5, 0.25]]))
    rendered, record = room.render(source.tobytes(), impulse)
    assert rendered.shape == (room.SAMPLES, 2)
    assert list(np.argmax(np.abs(rendered), axis=0)) == [101, 104]
    assert rendered[101, 0] == pytest.approx(2 * rendered[104, 1])
    assert record["impulse"]["delay_samples"] == [3, 12]


def test_authored_stereo_native_wav_and_production_downmix(tmp_path):
    results = room.authored_checks(tmp_path)
    assert tuple(results) == room.CONDITIONS
    assert len({row["stereo_pcm_sha256"] for row in results.values()}) == 3
    assert results["noise_left"]["mixed_pcm_sha256"] == results["noise_right"]["mixed_pcm_sha256"]
    assert results["noise_left"]["mixed_pcm_sha256"] != results["matched_room_clean"]["mixed_pcm_sha256"]
    for row in results.values():
        assert row["source_timeline"] == [{"start_sample": 0, "end_sample": room.SAMPLES, "source_pts": 0}]
        assert len(set(row["channel_pcm_sha256"])) == 2
        assert row["mixed_pcm_sha256"] not in row["channel_pcm_sha256"]
