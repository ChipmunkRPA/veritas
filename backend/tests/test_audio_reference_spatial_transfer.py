import numpy as np

from scripts.audio_reference_spatial_transfer import RATE, SAMPLES, projection


def signals():
    t = np.arange(SAMPLES) / RATE
    reference = np.rint(2800 * np.sin(2 * np.pi * 263 * t)).astype("<i2")
    room = 4000 * np.sin(2 * np.pi * 173 * t)
    return reference, room


def test_projection_keeps_reference_gain_and_reduces_shared_room_sound():
    reference, room = signals()
    stereo = np.column_stack((.5 * reference + room, reference + room)).round().astype("<i2")

    status, reduced, proof = projection(reference, stereo)

    assert status == "projected"
    assert reduced.shape == reference.shape
    assert proof["estimated_residual_reduction"] > .5
    weights, balance = np.array(proof["projection_weights"]), np.array(proof["target_balance"])
    assert np.isclose(weights @ balance, balance.mean())
    assert np.std(reduced - .75 * reference) < np.std(stereo.mean(axis=1) - .75 * reference)


def test_projection_declines_inconsistent_channel_balance_across_halves():
    reference, room = signals()
    left = .2 * reference.astype(float)
    right = reference.astype(float)
    left[SAMPLES // 2:], right[SAMPLES // 2:] = right[SAMPLES // 2:].copy(), left[SAMPLES // 2:].copy()
    stereo = np.column_stack((left + room / 10, right + room / 10)).round().astype("<i2")

    assert projection(reference, stereo) == ("unconfirmed_direction", None, None)
