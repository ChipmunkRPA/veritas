"""Separate replay agrees at borders, diagonals and non-divisible dimensions."""

import numpy as np

from scripts import image_residual_localizer_v13 as method
from scripts import review_image_residual_v13 as review


def test_alternate_projection_affine_and_median_replay():
    rng = np.random.default_rng(79)
    basis, _ = np.linalg.qr(rng.normal(size=(32, 2)))
    head = {
        "center": rng.normal(size=32).tolist(),
        "scale": rng.uniform(0.2, 2, 32).tolist(),
        "projection": (np.eye(32) - basis @ basis.T).tolist(),
        "weight": rng.normal(size=32).tolist(),
        "bias": 0.3,
    }
    values = rng.normal(size=(64, 64, 32)).astype(np.float32)
    np.testing.assert_allclose(review.scores(values, head), method.response(values, head), atol=2e-12, rtol=0)


def test_independent_flood_fill_and_cell_boundaries():
    values = np.random.default_rng(21).uniform(size=(64, 64))
    values[:5, :5] = 0.95
    values[22:27, 33:38] = 0.95
    for cutoff in (0.5, 0.8, 0.95):
        prediction = review.source_mask(values, cutoff, 137, 91)
        np.testing.assert_array_equal(prediction, method.mask(values, cutoff, 137, 91))
        truth = np.zeros((91, 137), bool)
        truth[25:59, 33:89] = True
        assert review.arithmetic(prediction, truth) == method.metrics(prediction, truth)
