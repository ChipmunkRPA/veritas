"""Separate family presence, public payload evidence and authenticated origin."""
import numpy as np
import pytest

from scripts import ltx_marking_audio_v1 as study


def bits():
    return np.arange(16, dtype=np.int64) % 2


@pytest.mark.parametrize('matches,p', [(0, 1.), (15, 17 / 65536), (16, 1 / 65536)])
def test_bit_match_has_exact_finite_evidence_not_authorship_confidence(matches, p):
    expected = bits()
    decoded = 1 - expected
    decoded[:matches] = expected[:matches]
    result = study.interpret(np.ones((2, 10), np.float32), decoded, expected)
    assert result['bit_match_p_value'] == p and result['bit_matches'] == matches
    assert result['provider_authenticated'] is result['used_in_confidence'] is result['early_stop_eligible'] is False


def test_wrong_public_payload_with_strong_family_presence_is_not_ltx_attribution():
    got = study.interpret(np.ones((2, 10), np.float32), 1 - bits(), bits())
    assert got['family_presence_detected'] and got['state'] == 'family_signal_only'
    assert not got['public_payload_exact_match'] and not got['early_stop_eligible']


def test_matching_public_bits_without_presence_are_not_a_detected_watermark():
    got = study.interpret(np.zeros((2, 10), np.float32), bits(), bits())
    assert got['state'] == 'not_detected' and not got['family_presence_detected']


def test_sample_probability_and_clip_fraction_use_different_boundary_rules():
    values = np.full((2, 10), .5, np.float32)
    assert not study.interpret(values, bits(), bits())['family_presence_detected']
    values[0, :] = 1
    got = study.interpret(values, bits(), bits())
    assert got['positive_native_sample_fraction'] == .5 and got['family_presence_detected']
    assert got['state'] == 'recognized_public_payload' and not got['provider_authenticated']


@pytest.mark.parametrize('kind', ['mono', 'nonfinite', 'outside', 'wrong_bits'])
def test_invalid_or_missing_native_observation_cannot_become_a_negative(kind):
    values, decoded = np.ones((2, 10), np.float32), bits()
    if kind == 'mono':
        values = values[:1]
    elif kind == 'nonfinite':
        values[0, 0] = np.nan
    elif kind == 'outside':
        values[0, 0] = 1.1
    else:
        decoded = np.ones(15, np.int64)
    with pytest.raises(ValueError):
        study.interpret(values, decoded, bits())


@pytest.mark.parametrize('rate', [24000, 48000])
def test_authored_carriers_are_native_stereo_and_exactly_reproducible(rate):
    value = study.carrier(rate, 2)
    assert value.shape == (2, 2 * rate) and value.dtype == np.float32
    assert np.array_equal(value, study.carrier(rate, 2))
    assert not np.array_equal(value[0], value[1]) and np.abs(value).max() < 1
