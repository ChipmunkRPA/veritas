import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.portrait_expression_drive_v138 import expression_points


def test_public_arithmetic_import_does_not_load_model_or_app_runtime():
    code = """
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".")
               for name in ("torch", "cv2", "app", "scripts.liveportrait_mouth_benchmark")):
            raise RuntimeError("Public arithmetic tried to load a model/app dependency")
sys.meta_path.insert(0, BlockModels())
from scripts.portrait_expression_drive_v138 import expression_points
import numpy as np
zero = np.zeros((1, 21, 3))
assert np.array_equal(expression_points(zero, zero, zero, np.ones((1, 1, 1))), zero)
print("public_arithmetic_without_models")
"""
    code = f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n" + code
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20, check=True
    )
    assert result.stdout.strip() == "public_arithmetic_without_models"


def test_expression_transfer_matches_rebuilding_with_source_identity_pose_scale_and_translation():
    random = np.random.default_rng(138)
    canonical, source_expression, driving_expression = [random.normal(size=(1, 21, 3)) for _ in range(3)]
    rotation = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], np.float64)
    scale = np.array([[[0.7]]])
    translation = np.array([[[0.15, -0.2, 0]]])
    source = scale * (canonical @ rotation + source_expression) + translation
    actual = expression_points(source, source_expression, driving_expression, scale)
    expected = scale * (canonical @ rotation + driving_expression) + translation
    assert np.allclose(actual, expected, atol=1e-15, rtol=0)
    assert np.array_equal(
        expression_points(source, source_expression, source_expression.copy(), scale), source
    )


def test_expression_displacement_preserves_different_mouth_shapes_with_the_same_aperture():
    source = np.zeros((1, 21, 3))
    a, b = source.copy(), source.copy()
    a[0, 4, 0] = 0.1
    b[0, 4, 2] = 0.1
    # Both have identical vertical displacement but distinct full expressions.
    scale = np.ones((1, 1, 1))
    assert not np.array_equal(
        expression_points(source, source, a, scale), expression_points(source, source, b, scale)
    )


@pytest.mark.parametrize(
    "problem", ["nan", "wrong_points", "bad_scale_shape", "zero_scale", "negative_scale", "infinite_scale"]
)
def test_invalid_expression_correspondence_cannot_be_silently_used(problem):
    source = np.zeros((1, 21, 3))
    driver = source.copy()
    scale = np.ones((1, 1, 1))
    if problem == "nan":
        driver[0, 0, 0] = np.nan
    elif problem == "wrong_points":
        driver = driver[:, :-1]
    elif problem == "bad_scale_shape":
        scale = np.ones((1,))
    elif problem == "zero_scale":
        scale *= 0
    elif problem == "negative_scale":
        scale *= -1
    else:
        scale *= np.inf
    with pytest.raises(ValueError):
        expression_points(source, source, driver, scale)
