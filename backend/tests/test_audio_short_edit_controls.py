"""Authored PCM checks for the pure short speech control recipe."""

import numpy as np
import pytest

from scripts import audio_short_edit_controls as controls


def pcm(values: np.ndarray) -> bytes:
    return values.astype("<i2").tobytes()


def samples(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2")


def test_exact_three_second_crop_without_padding_or_alignment():
    values = np.arange(controls.SAMPLES + 33, dtype=np.int32) - 24000
    output = controls.crop_pcm(pcm(values), 33)

    assert len(output) == 96000
    assert output == pcm(values[33:])
    with pytest.raises(controls.UnsupportedControl, match="insufficient_source_samples"):
        controls.crop_pcm(pcm(values), 34)
    with pytest.raises(ValueError, match="whole little-endian signed16"):
        controls.crop_pcm(b"\x00")


def test_hard_same_source_copy_uses_pre_edit_donor_and_keeps_outside_bytes():
    values = np.arange(controls.SAMPLES, dtype=np.int32) - 24000
    host = pcm(values)
    edited = controls.splice_pcm(host)
    start, end = edited.target_samples
    donor_start, donor_end = edited.donor_samples

    assert edited.target_samples == (17280, 30080)
    assert edited.donor_samples == (12160, 24960)
    assert start - donor_start == end - donor_end == 5120
    assert edited.pcm_s16le == host[: 2 * start] + host[2 * donor_start : 2 * donor_end] + host[2 * end :]
    assert edited.changed_samples == end - start == 12800
    assert edited.fade_samples == 0
    assert edited.sample_count == controls.SAMPLE_RATE * edited.duration_seconds == 48000


def test_hard_independent_donor_uses_its_fixed_subrange():
    host = pcm(np.arange(controls.SAMPLES, dtype=np.int32) - 24000)
    donor = pcm(18000 - np.arange(controls.SAMPLES, dtype=np.int32))
    result = controls.splice_pcm(host, donor)
    start, end = result.target_samples
    donor_start, donor_end = result.donor_samples
    assert result.pcm_s16le == host[: 2 * start] + donor[2 * donor_start : 2 * donor_end] + host[2 * end :]
    assert result.changed_samples == np.count_nonzero(samples(result.pcm_s16le) != samples(host))


@pytest.mark.parametrize(("fade_ms", "fade_samples"), [(20, 320), (50, 800)])
def test_raised_cosine_fades_keep_native_duration_and_fit_inside_target(fade_ms, fade_samples):
    host = pcm(np.full(controls.SAMPLES, -2000))
    donor = pcm(np.full(controls.SAMPLES, 6000))
    result = controls.splice_pcm(host, donor, fade_ms=fade_ms)
    start, end = result.target_samples
    values = samples(result.pcm_s16le)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_samples) / (fade_samples - 1))
    expected = np.rint(-2000 + 8000 * ramp).astype("<i2")

    assert result.fade_samples == fade_samples
    assert result.sample_count == controls.SAMPLES and result.duration_seconds == 3
    assert result.pcm_s16le[: 2 * start] == host[: 2 * start]
    assert result.pcm_s16le[2 * end :] == host[2 * end :]
    np.testing.assert_array_equal(values[start : start + fade_samples], expected)
    np.testing.assert_array_equal(values[end - fade_samples : end], expected[::-1])
    assert (values[start + fade_samples : end - fade_samples] == 6000).all()
    assert result.changed_samples == np.count_nonzero(values != samples(host))


def test_unsupported_edit_does_not_substitute_samples():
    host = pcm(np.full(controls.SAMPLES, 700))
    with pytest.raises(controls.UnsupportedControl, match="zero_changed_samples"):
        controls.splice_pcm(host, fade_ms=50)
    with pytest.raises(controls.UnsupportedControl, match="expected_three_seconds"):
        controls.splice_pcm(host, host[:-2])
    with pytest.raises(ValueError, match="hard edit or a 20 ms or 50 ms fade"):
        controls.splice_pcm(host, fade_ms=10)


def test_pair_noise_is_reproducible_with_exact_shared_signed16_residual():
    rng = np.random.default_rng(167)
    host = pcm(rng.integers(-6500, 6500, size=controls.SAMPLES))
    donor = pcm(rng.integers(-12000, 12000, size=controls.SAMPLES))
    noise = pcm(rng.integers(-5000, 5000, size=controls.SAMPLES))
    edited = controls.splice_pcm(host, donor, fade_ms=20)
    variants = {"reference": host, "edited": edited.pcm_s16le}
    result = controls.mix_paired_noise(variants, reference="reference", noise_pcm_s16le=noise)
    assert result == controls.mix_paired_noise(variants, reference="reference", noise_pcm_s16le=noise)

    assert result.common_gain == 1.0
    assert set(result.noise_s16le) == {0, 5}
    for name, variant in result.variants.items():
        assert variant.clean_s16le == variants[name]
        clean = samples(variant.clean_s16le).astype(np.int32)
        for level, mixed in variant.mixed_s16le.items():
            assert len(mixed) == len(variant.clean_s16le) == 96000
            residual = samples(mixed).astype(np.int32) - clean
            np.testing.assert_array_equal(residual, samples(result.noise_s16le[level]))
            actual_snr = 10 * np.log10(np.mean(clean**2) / np.mean(residual**2))
            assert variant.measured_snr_db[level] == pytest.approx(actual_snr)
    for level in controls.SNR_LEVELS_DB:
        original = result.variants["reference"].mixed_s16le[level]
        mixture = result.variants["edited"].mixed_s16le[level]
        start, end = edited.target_samples
        assert mixture[: 2 * start] == original[: 2 * start]
        assert mixture[2 * end :] == original[2 * end :]
        assert result.variants["reference"].measured_snr_db[level] == pytest.approx(level, abs=0.005)
        assert result.variants["edited"].measured_snr_db[level] > level


def test_one_gain_attenuates_all_arms_and_explicit_clipping_is_unsupported():
    loud = np.tile(np.array([32767, -32768], dtype=np.int32), controls.SAMPLES // 2)
    variants = {"reference": pcm(loud), "quiet": pcm(loud // 3)}
    result = controls.mix_paired_noise(variants, reference="reference", noise_pcm_s16le=pcm(loud))

    assert 0 < result.common_gain < 1
    for name, raw in variants.items():
        clean = np.rint(samples(raw).astype(np.float64) * result.common_gain)
        np.testing.assert_array_equal(samples(result.variants[name].clean_s16le), clean)
        for level in controls.SNR_LEVELS_DB:
            expected = clean + samples(result.noise_s16le[level])
            np.testing.assert_array_equal(samples(result.variants[name].mixed_s16le[level]), expected)
    with pytest.raises(controls.UnsupportedControl, match="pcm16_clipping"):
        controls.mix_paired_noise(variants, reference="reference", noise_pcm_s16le=pcm(loud), common_gain=1)


def test_noise_degeneracy_and_sub_lattice_treatment_are_unsupported():
    alternating = pcm(np.tile([1000, -1000], controls.SAMPLES // 2))
    constant = pcm(np.full(controls.SAMPLES, 500))
    silence = bytes(2 * controls.SAMPLES)
    with pytest.raises(controls.UnsupportedControl, match="constant_noise"):
        controls.mix_paired_noise({"ref": alternating}, reference="ref", noise_pcm_s16le=constant)
    with pytest.raises(controls.UnsupportedControl, match="silent_reference"):
        controls.mix_paired_noise({"ref": silence}, reference="ref", noise_pcm_s16le=alternating)
    with pytest.raises(controls.UnsupportedControl, match="quantized_silent_noise"):
        controls.mix_paired_noise(
            {"ref": alternating}, reference="ref", noise_pcm_s16le=alternating, common_gain=1e-6
        )
