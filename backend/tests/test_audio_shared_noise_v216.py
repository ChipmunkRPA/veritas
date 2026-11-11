"""Verify common-gain, shared-residual, and source-clock arithmetic."""

import numpy as np
import pytest

from scripts import audio_shared_noise_v216 as panel


def pcm(values):
    return np.asarray(values, dtype="<i2").tobytes()


def alternating(amplitude, size=panel.SAMPLES):
    return pcm(np.tile([amplitude, -amplitude], (size + 1) // 2)[:size])


def tapes():
    return {"kitchen": alternating(1500), "traffic": pcm(np.tile([250, 10, -140, 90], panel.SAMPLES // 4))}


def test_crops_first_native_samples_and_level_matches_three_arms():
    base = alternating(5000)
    foreground = {
        "drifting": alternating(10000) + alternating(29000, 77),
        "genuine": alternating(7500) + alternating(5, 29),
        "alania": base + alternating(32000, 201),
    }
    result = panel.mix_shared_noise(foreground, tapes())
    assert result.sample_range == (0, 48000) and result.sample_rate == 16000
    assert result.reference_arm == "alania"
    assert result.common_gain == 1
    assert tuple(result.variants) == panel.ARMS
    assert [arm.source_rms for arm in result.variants.values()] == [7500, 5000, 10000]
    assert [arm.level_match_gain for arm in result.variants.values()] == [2 / 3, 1, 1 / 2]
    assert all(arm.clean_s16le == base for arm in result.variants.values())


def test_both_tapes_share_one_anti_clipping_gain_and_exact_residuals():
    foreground = {"genuine": alternating(27000), "alania": alternating(31000), "drifting": alternating(30000)}
    backgrounds = tapes()
    result = panel.mix_shared_noise(foreground, backgrounds)
    assert result == panel.mix_shared_noise(dict(reversed(tuple(foreground.items()))), backgrounds)
    assert 0 < result.common_gain < 1
    assert result.reference_arm == "genuine"
    assert len(set(result.noise_s16le.values())) == 4
    assert set(result.noise_s16le) == {(tape, db) for tape in panel.TAPES for db in panel.SNR_LEVELS_DB}
    for arm in result.variants.values():
        clean = panel.samples(arm.clean_s16le).astype(np.int32)
        assert np.max(np.abs(clean)) == round(27000 * result.common_gain)
        for condition, payload in arm.mixed_s16le.items():
            mixed = panel.samples(payload).astype(np.int32)
            observed = mixed - clean
            assert len(payload) == 96000
            assert np.max(np.abs(mixed)) <= 0.98 * 32768 + 1
            assert observed.astype("<i2").tobytes() == result.noise_s16le[condition]
            observed_snr = 20 * np.log10(np.linalg.norm(clean) / np.linalg.norm(observed))
            assert observed_snr == pytest.approx(arm.measured_snr_db[condition])
            assert abs(observed_snr - condition[1]) <= 0.1


def test_missing_arms_stay_absent_and_reference_tie_is_stable():
    source = alternating(5000)
    shared = panel.mix_shared_noise({"drifting": source, "alania": source}, tapes())
    assert tuple(shared.variants) == ("alania", "drifting")
    assert shared.reference_arm == "alania"
    single = panel.mix_shared_noise({"drifting": source}, tapes())
    assert tuple(single.variants) == ("drifting",)
    assert single.variants["drifting"].clean_s16le == source


def test_unsupported_geometry_and_degeneracies_remain_distinct():
    real = alternating(4000)
    with pytest.raises(panel.UnsupportedControl, match="no_available_foregrounds"):
        panel.mix_shared_noise({}, tapes())
    with pytest.raises(panel.UnsupportedControl, match="insufficient_source_samples"):
        panel.mix_shared_noise({"genuine": real[:-2]}, tapes())
    with pytest.raises(panel.UnsupportedControl, match="expected_exact_three_second_tape"):
        panel.mix_shared_noise({"genuine": real}, tapes() | {"kitchen": real + b"\0\0"})
    with pytest.raises(panel.UnsupportedControl, match="silent_foreground"):
        panel.mix_shared_noise({"genuine": real, "alania": bytes(len(real))}, tapes())
    with pytest.raises(panel.UnsupportedControl, match="constant_noise"):
        panel.mix_shared_noise({"genuine": real}, tapes() | {"traffic": bytes(len(real))})
    with pytest.raises(panel.UnsupportedControl, match="quantized_snr_out_of_tolerance"):
        panel.mix_shared_noise(
            {"genuine": alternating(1)}, {"kitchen": alternating(1), "traffic": alternating(1)}
        )
