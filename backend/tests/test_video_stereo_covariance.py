# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from app.bench.video_stereo_covariance import _directional_reading


def test_stereo_direction_does_not_turn_uniform_gain_into_selective_loss():
    center = np.diag([.0001, .1])
    flank = center * 4

    for covariances in ((flank, center, flank), tuple(part.astype(complex) for part in (flank, center, flank))):
        margin, stable = _directional_reading(*covariances)
        assert not stable
        assert margin <= 0


def test_complex_direction_recovers_loss_hidden_from_real_cross_power():
    center = np.diag([.01, .01])
    flank = center + np.array([[.009, -.009j], [.009j, .009]])

    real_margin, real_stable = _directional_reading(flank.real, center, flank.real)
    complex_margin, complex_stable = _directional_reading(flank, center, flank)

    assert real_stable and complex_stable
    assert real_margin == pytest.approx(0)
    assert complex_margin == pytest.approx(10 * np.log10(2.8))
