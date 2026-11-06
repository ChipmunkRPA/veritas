import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.mouth_probe_v6 import (
    margin_windows,
    margins,
    mouth_chip,
    preprocessed,
    source_weights,
    train_head,
)


def fitting_rows():
    rng = np.random.default_rng(8)
    rows = []
    for group, role in (("a", "fit"), ("b", "fit"), ("c", "point")):
        for condition in ("original", "local-light", "local-resample", "ai-redraw", "ai-retime"):
            label = int(condition.startswith("ai-"))
            for _ in range(4):
                rows.append(
                    {
                        "group": group,
                        "role": role,
                        "condition": condition,
                        "label": label,
                        "features": (rng.normal(0, 0.1, 5) + label * 0.2).tolist(),
                    }
                )
    return rows


def test_point_actors_and_positive_point_responses_cannot_change_fitted_head():
    rows = fitting_rows()
    first = train_head(rows)
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["role"] == "point" and r["label"]:
            r["features"] = [10000] * 5
    second = train_head(changed)
    for field in ("mean", "scale", "coef", "intercept", "review_point"):
        assert first[field] == second[field]
    for r in changed:
        if r["role"] == "point" and not r["label"]:
            r["features"] = [10000] * 5
    third = train_head(changed)
    for field in ("mean", "scale", "coef", "intercept"):
        assert first[field] == third[field]
    assert third["review_point"] != first["review_point"]


def test_actor_and_class_total_weights_do_not_depend_on_repeated_variants():
    rows = fitting_rows()[:40]
    rows += [copy.deepcopy(rows[0])] * 13
    weights = source_weights(rows)
    for group in ("a", "b"):
        for label in (0, 1):
            assert sum(
                w for w, r in zip(weights, rows, strict=True) if r["group"] == group and r["label"] == label
            ) == pytest.approx(len(rows) / 4)
    assert weights.sum() == pytest.approx(len(rows))


def test_actor_role_overlap_refuses_the_fit():
    rows = fitting_rows()
    for r in rows:
        if r["role"] == "point":
            r["group"] = "a"
    with pytest.raises(ValueError, match="overlap"):
        train_head(rows)


def test_head_reconstruction_is_exact_linear_margin_without_probability_label():
    rows = fitting_rows()
    head = train_head(rows)
    z = margins([r["features"] for r in rows], head)
    assert np.isfinite(z).all()
    assert "not AI probability" in head["response_kind"]
    controls = [r for r in rows if r["role"] == "point" and not r["label"]]
    assert max(margins([r["features"] for r in controls], head)) < head["review_point"]


def test_native_chip_and_preprocessing_preserve_explicit_measurement_region():
    rgb = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    chip, bounds = mouth_chip(rgb)
    assert bounds == [28, 60, 72, 92]
    assert np.array_equal(chip, rgb[60:92, 28:72])
    tensor = preprocessed([chip])
    assert tensor.shape == (1, 3, 112, 112) and tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_probe_windows_retain_actual_source_clock_and_prior_following_context():
    table = SimpleNamespace(seconds=lambda i: 8 + i / 25)
    tracks = {"t0": [SimpleNamespace(frame_index=i) for i in range(63)]}
    centers = set(range(1, 61, 4))
    values = {i: 4.0 if 18 <= i <= 38 else -2.0 for i in range(63)}
    window = margin_windows(centers, tracks, values, table)[0]
    assert window["response"] > 3.0
    assert window["start_s"] >= 8.6 and window["end_s"] <= 9.6
    assert window["centers"][2] >= 17 and window["centers"][-3] <= 41
    assert margin_windows(centers, tracks, {i: values[i] for i in centers}, table) == []
