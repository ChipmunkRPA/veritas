import numpy as np
import pytest
from scipy.optimize import check_grad

from scripts import audio_paired_consistency_v223 as study


def test_consistency_objective_gradient():
    design = np.array([[1.0, 1.0], [-1.0, 1.0], [0.5, 1.0], [-0.1, 1.0]])
    target = np.array([1.0, 0.0, 1.0, 0.0])
    args = (
        design,
        target,
        np.full(4, 0.25),
        np.array([0.001, 0]),
        design[2:] - design[:2],
        np.full(2, 0.5),
        0.15,
    )
    error = check_grad(
        lambda x: study.objective(x, *args)[0], lambda x: study.objective(x, *args)[1], [0.5, -0.2]
    )
    assert error < 1e-7


def test_group_cutpoints_keep_budget_with_tied_windows():
    scores = {"a": 0.9, "b": 0.7, "c": 0.9, "d": 0.4, "x": 0.1, "y": 0.3}
    rows = [
        {"id": identity, "label": label, "group": group}
        for identity, label, group in (
            ("a", 0, "a"),
            ("b", 0, "a"),
            ("c", 0, "b"),
            ("d", 0, "c"),
            ("x", 1, "x"),
            ("y", 1, "y"),
        )
    ]
    assert study.group_tail_cutpoint(rows, scores, truth=0, budget=1) == 0.9
    assert study.group_tail_cutpoint(rows, scores, truth=1, budget=0) == 0.1
    assert study.score_direction(0.9, {"ai_above": 0.9, "original_below": 0.1}) == "inconclusive"
    assert study.score_direction(None, {"ai_above": 0.9, "original_below": 0.1}) == "unavailable"


def test_paired_head_penalizes_noise_drift(monkeypatch):
    monkeypatch.setattr(study, "WAVLM_WIDTH", 2)
    rows = []
    for label in (0, 1):
        for i in range(6):
            for condition, raw, nuisance in (
                ("clean", 0.1 + 0.8 * label, 0),
                ("noise_0", 0.45 + 0.1 * label, 1),
            ):
                rows.append(
                    {
                        "base_id": f"{label}-{i}",
                        "group": f"{label}-{i}",
                        "label": label,
                        "kind": "stock" if label else "genuine",
                        "role": study.ROLES[0],
                        "condition": condition,
                        "status": "ok",
                        "raw": raw,
                        "wavlm": [label + i / 30, nuisance],
                    }
                )
    plain = study.fit_head(rows, "wavlm_raw")
    paired = study.fit_head(rows, "paired_wavlm_raw")
    plain_delta, _ = study.pair_deltas(rows, study.matrix(rows, "wavlm_raw", plain["transform"]))
    paired_delta, _ = study.pair_deltas(rows, study.matrix(rows, "paired_wavlm_raw", paired["transform"]))
    assert np.mean(np.abs(paired_delta @ paired["coefficients"])) < np.mean(
        np.abs(plain_delta @ plain["coefficients"])
    )
    with pytest.raises(ValueError, match="fitting rows"):
        study.fit_head([{**rows[0], "role": study.ROLES[1]}], "raw_only")
