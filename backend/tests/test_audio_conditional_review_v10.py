import numpy as np
import pytest

from scripts.audio_conditional_v10 import forest_predict
from scripts.review_audio_conditional_v10 import guards


def test_guard_reconstruction_keeps_missing_variants_and_paired_truth_classes():
    rows = [{"role": "cal", "group": "same", "label": c} for c in (0, 0, 1)]
    result = guards(rows, [0.1, None, 0.9])
    assert result["0"] == [{"group": "same", "maximum": None, "rows": 2}]
    assert result["1"][0]["group"] == "same"
    assert result["1"][0]["maximum"] == pytest.approx(0.1)


def test_development_cannot_join_calibration_or_lose_a_prediction():
    with pytest.raises(ValueError, match="Independent calibration"):
        guards([{"role": "development"}], [0.5])
    with pytest.raises(ValueError, match="Independent calibration"):
        guards([{"role": "cal"}], [])


def test_json_tree_uses_exact_less_equal_boundary_and_rejects_cycles():
    tree = {
        "value": [0.0, -1.0, 1.0],
        "feature": [0, 0, 0],
        "threshold": [0.0, 0.0, 0.0],
        "left": [1, 0, 0],
        "right": [2, 0, 0],
        "leaf": [False, True, True],
    }
    forest = {"n_features": 1, "max_depth": 1, "baseline_logit": 0.0, "trees": [tree]}
    p = forest_predict([[-1.0], [0.0], [np.nextafter(0.0, 1.0)]], forest)
    assert p[0] == p[1] and p[1] < p[2]
    tree["left"][0] = 0
    with pytest.raises(ValueError, match="depth/cycle"):
        forest_predict([[-1.0]], forest)
