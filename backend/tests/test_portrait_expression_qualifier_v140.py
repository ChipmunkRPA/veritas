import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import portrait_expression_qualifier_v140 as qualifier


def hex_array(values):
    return np.vectorize(lambda x: float(x).hex(), otypes=[object])(np.asarray(values)).tolist()


def geometry_record():
    mesh = np.linspace(0, 1, 478 * 3).reshape(478, 3)
    matrix = np.array([[0.8, 0.1, 2], [-0.1, 0.8, 4]])
    return {
        "mesh_xyz_hex": hex_array(np.stack([mesh, mesh + 0.0001, mesh])),
        "source_to_256_matrix_hex": hex_array(matrix),
        "source_to_512_matrix_hex": hex_array(qualifier.original.upscale_matrix(matrix)),
        "source_ratio_hex": (0.125).hex(),
    }


def test_cached_geometry_preserves_repeat_and_full_pixel_centre_matrix_without_inference():
    record = geometry_record()
    meshes, matrix, ratio = qualifier.cached_geometry(record)
    assert meshes.shape == (3, 478, 3) and np.array_equal(meshes[0], meshes[2])
    assert not np.array_equal(meshes[0], meshes[1]) and ratio == 0.125
    assert np.array_equal(
        qualifier.original.upscale_matrix(matrix),
        np.asarray([[float.fromhex(x) for x in row] for row in record["source_to_512_matrix_hex"]]),
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_point",
        "nonfinite",
        "changed_repeat",
        "wrong_matrix",
        "matrix_shape",
        "noncanonical",
        "bad_ratio",
    ],
)
def test_invalid_cached_geometry_is_rejected_before_model_use(problem):
    record = geometry_record()
    if problem == "missing_point":
        record["mesh_xyz_hex"][1].pop()
    elif problem == "nonfinite":
        record["mesh_xyz_hex"][0][0][0] = float("nan").hex()
    elif problem == "changed_repeat":
        record["mesh_xyz_hex"][2][0][0] = (0.5).hex()
    elif problem == "wrong_matrix":
        record["source_to_512_matrix_hex"][0][2] = float(0).hex()
    elif problem == "matrix_shape":
        record["source_to_256_matrix_hex"].pop()
    elif problem == "noncanonical":
        record["mesh_xyz_hex"][0][0][0] = "0"
    else:
        record["source_ratio_hex"] = float("inf").hex()
    with pytest.raises(ValueError):
        qualifier.cached_geometry(record)


def outputs():
    source = np.zeros((512, 512, 3), np.uint8)
    earlier = source.copy()
    earlier[40:42, 10:13, 0] = 1
    return [source, source.copy(), earlier]


def test_response_metrics_distinguish_renderer_response_from_achieved_motion():
    metrics = qualifier.response_metrics(outputs())
    assert metrics == {"renderer_repeat_bitwise": True, "earlier_expression_changed_crop_pixels": 6}
    qualifier.require_response(metrics)
    assert not any("alignment" in key or "accuracy" in key or "achieved_motion" in key for key in metrics)


@pytest.mark.parametrize(
    "problem", ["changed_repeat", "no_earlier_response", "wrong_shape", "wrong_dtype", "missing"]
)
def test_nonrepeat_nonresponse_or_incomplete_outputs_do_not_qualify(problem):
    images = outputs()
    if problem == "changed_repeat":
        images[1][0, 0, 0] = 1
    elif problem == "no_earlier_response":
        images[2] = images[0].copy()
    elif problem == "wrong_shape":
        images[2] = images[2][:-1]
    elif problem == "wrong_dtype":
        images[2] = images[2].astype(np.float32)
    else:
        images.pop()
    with pytest.raises(ValueError):
        qualifier.require_response(qualifier.response_metrics(images))


def test_native_support_metrics_keep_escape_and_empty_edits_visible():
    source = np.zeros((6, 8, 3), np.uint8)
    edited = source.copy()
    alpha = np.zeros(source.shape[:2], np.float32)
    alpha[1:4, 2:5] = 1
    edited[2, 3, 0] = 1
    assert qualifier.support_metrics(source, edited, alpha) == {
        "changed_native_pixels": 1,
        "outside_support_changed_pixels": 0,
    }
    edited[0, 0, 0] = 1
    assert qualifier.support_metrics(source, edited, alpha)["outside_support_changed_pixels"] == 1
    assert qualifier.support_metrics(source, source.copy(), alpha)["changed_native_pixels"] == 0
    alpha[0, 0] = np.nan
    with pytest.raises(ValueError):
        qualifier.support_metrics(source, edited, alpha)


def completion_fixture():
    return {
        "status": "complete",
        "execution_context_schema": "chipmunk.portrait-tiled-cpu-result/139",
        "freeze_sha256": "freeze",
        "strict_exact_checkpoint_loads": 5,
        "stateless_geometry_repeat_bitwise": True,
        "renderer_repeat_bitwise": True,
        "native_coordinate_containment_qualified": True,
        "face_task_inputs": 3,
        "renderer_inputs": 4,
        "top_level_module_forwards": {name: 4 for name in qualifier.EXPECTED},
        "new_neural_calls": 23,
        "private_render_records_sha256": "records",
        "private_forward_progress_sha256": "progress",
    }, {
        "child_exit_code": 0,
        "guard": None,
        "sample_errors": 0,
        "new_neural_calls": 23,
        "worker_receipt_sha256": "result",
    }


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "pending",
        "wrong_freeze",
        "wrong_records",
        "wrong_roots",
        "guarded",
        "wrong_receipt",
        "wrong_profile_calls",
    ],
)
def test_actual_baseline_completion_binding_cannot_be_replaced_by_readiness_or_partial_files(problem):
    result, profile = completion_fixture()
    if problem == "pending":
        result["status"] = "pending_headroom_no_stage_attempt"
    elif problem == "wrong_freeze":
        result["freeze_sha256"] = "changed"
    elif problem == "wrong_records":
        result["private_render_records_sha256"] = "changed"
    elif problem == "wrong_roots":
        result["top_level_module_forwards"]["lip"] = 0
    elif problem == "guarded":
        profile["guard"] = "resource_limit"
    elif problem == "wrong_receipt":
        profile["worker_receipt_sha256"] = "changed"
    elif problem == "wrong_profile_calls":
        profile["new_neural_calls"] = 0
    if problem is None:
        qualifier.completion_binding(
            result,
            profile,
            result_sha="result",
            freeze_sha="freeze",
            records_sha="records",
            progress_sha="progress",
        )
    else:
        with pytest.raises(ValueError):
            qualifier.completion_binding(
                result,
                profile,
                result_sha="result",
                freeze_sha="freeze",
                records_sha="records",
                progress_sha="progress",
            )


def test_only_the_fixed_neural_qualification_command_is_dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(
        qualifier.subprocess, "Popen", lambda argv, **kwargs: calls.append(copy.deepcopy(argv))
    )
    argv = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", "private-output"]
    qualifier.routed_popen(argv)
    assert calls[0][3] == "scripts.portrait_expression_qualifier_v140"
    assert argv[3] == "scripts.video_appearance_native_v108"
    with pytest.raises(ValueError):
        qualifier.routed_popen([*argv[:4], "bind", *argv[5:]])


def test_authored_freeze_body_round_trips_without_tuple_or_private_runtime_dependencies(
    tmp_path, monkeypatch
):
    selected = {"requested_offset_seconds": "-8/25", "actual_offset_seconds": "-167/500"}
    (tmp_path / "selection.json").write_text(json.dumps(selected))
    monkeypatch.setattr(qualifier, "TRIAL", tmp_path)
    monkeypatch.setattr(qualifier, "baseline_verified", lambda: (selected, {}))
    monkeypatch.setattr(qualifier, "sha", lambda path: "authored-digest")
    monkeypatch.setattr(qualifier.importlib.metadata, "version", lambda name: "authored-version")
    body = qualifier.identity()
    assert json.loads(json.dumps(body, allow_nan=False)) == body
    assert body["expected_top_module_forward_units"] == 15 and body["new_face_task_inputs"] == 0
    assert body["driver_actual_offset_seconds"] == "-167/500"


def test_authored_cache_response_helpers_import_without_model_or_face_runtime():
    code = """
import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in
               ("torch", "cv2", "mediapipe", "scripts.liveportrait_mouth_benchmark")):
            raise RuntimeError("Authored helper import tried to load a model/face runtime")
sys.meta_path.insert(0, NoModels())
from scripts import portrait_expression_qualifier_v140
assert portrait_expression_qualifier_v140.EXPECTED["lip"] == 0
print("cache_response_helpers_without_models")
"""
    code = f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n" + code
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20, check=True
    )
    assert result.stdout.strip() == "cache_response_helpers_without_models"
