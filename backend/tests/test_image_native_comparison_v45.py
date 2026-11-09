"""Scope isolation, exact inherited math, combined masks and honest tradeoffs."""

import ast
import copy
import inspect

import numpy as np
import pytest

from scripts import image_native_comparison_readout_v45 as readout
from scripts import image_native_comparison_runtime_v45 as runtime
from scripts import image_native_comparison_v45 as producer
from scripts import image_regional_inference_v28 as old
from scripts import regional_calibration_runtime_v42 as old_runtime


def plan():
    return {"rows": [{"source_id": str(i), "component": str(min(i, 23)), "role": "point"} for i in range(27)]}


def test_source_role_and_all_three_paths_are_explicit():
    value = plan()
    value["rows"].append({"source_id": "fit", "component": "fit", "role": "fit"})
    tasks = producer.comparison_tasks(value)
    assert len(tasks) == 108 and {t["role"] for t in tasks} == {"point"}
    assert [len(producer.phase_tasks(tasks, n)) for n in range(5)] == [24, 24, 24, 24, 12]
    assert producer.ARMS == ("candidate", "baseline", "fused-reference")
    assert len(readout.METHODS) == 6


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "cross_role"])
def test_incomplete_or_leaking_source_plan_is_not_scored(alteration):
    value = plan()
    if alteration == "missing":
        value["rows"].pop()
    elif alteration == "duplicate":
        value["rows"][-1] = copy.deepcopy(value["rows"][-2])
    else:
        value["rows"].append({"source_id": "other", "component": "0", "role": "fit"})
    with pytest.raises(ValueError):
        producer.comparison_tasks(value)


@pytest.mark.parametrize(
    "arm,expected",
    [
        ("candidate", ("candidate", "native")),
        ("baseline", ("baseline", "native")),
        ("fused-reference", ("baseline", "jpeg75")),
    ],
)
def test_model_loader_uses_frozen_graphs_without_changing_native_input_encoding(monkeypatch, arm, expected):
    seen = []
    monkeypatch.setattr(old, "load_predictor", lambda a, f, c: seen.append((a, f, c)))
    producer.load_predictor(arm, "native", {"checkpoint": "bound"})
    assert seen == [(*expected, {"checkpoint": "bound"})]
    with pytest.raises(ValueError):
        producer.load_predictor(arm, "jpeg75", {})


def test_producer_is_the_frozen_numeric_loop_with_only_explicit_role_arm_and_schema_changes():
    expected = ast.parse(inspect.getsource(old.score)).body[0]

    class Scope(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "calibration_tasks":
                node.id = "comparison_tasks"
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace("calibration", "point").replace("producer/28", "producer/45")
            return node

        def visit_Set(self, node):
            self.generic_visit(node)
            if {n.value for n in node.elts if isinstance(n, ast.Constant)} == {"candidate", "baseline"}:
                node.elts.append(ast.Constant(value="fused-reference"))
            return node

    expected = Scope().visit(expected)
    assert ast.dump(expected) == ast.dump(ast.parse(inspect.getsource(producer.score)).body[0])


def test_resource_observer_body_is_unchanged_after_dispatch():
    def body(function):
        source = inspect.getsource(function)
        source = source[source.index("    if output.exists():") :]
        return ast.dump(
            ast.parse(
                "def observer():\n" + source.replace("fixed-monitor-profile/42", "fixed-monitor-profile/45")
            )
        )

    assert body(runtime.run) == body(old_runtime.run)


def test_source_mask_is_the_actual_frozen_production_reconstruction():
    from scripts.score_splice_localizers import score_map

    expected = ast.parse(inspect.getsource(score_map)).body[0]
    expected.name = "source_mask"
    expected.body[-1] = ast.Return(
        value=ast.BinOp(
            left=ast.Name(id="retained", ctx=ast.Load()),
            op=ast.BitAnd(),
            right=ast.Name(id="image_gate", ctx=ast.Load()),
        )
    )
    assert ast.dump(expected) == ast.dump(ast.parse(inspect.getsource(readout.source_mask)).body[0])
    from PIL import Image

    pixels = np.random.default_rng(45).integers(0, 256, size=(192, 256, 3), dtype=np.uint8)
    data, _ = producer.encoded_view(Image.fromarray(pixels), "native")
    truth = np.zeros((192, 256), bool)
    truth[40:150, 60:190] = True
    raw = np.zeros((384, 384), np.float32)
    raw[70:280, 90:290] = 0.99
    for gate in (True, False):
        mask = readout.source_mask(raw, gate, 0.7, data, pixels, truth)
        assert readout.mask_metrics(mask, truth) == score_map(raw, gate, 0.7, data, pixels, truth)["metrics"]
        if not gate:
            assert not mask.any()


def test_complete_producer_records_real_encoded_bytes_maps_and_exact_repeat(tmp_path, monkeypatch):
    import json

    from PIL import Image

    from scripts import image_regional_training_v26 as training

    trial, maps, pairs = (tmp_path / name for name in ("trial", "maps", "pairs"))
    trial.mkdir()
    maps.mkdir()
    (pairs / "qualification").mkdir(parents=True)
    producer.write(trial / "private-paths.json", {"raw_map_root": str(maps)})
    producer.write(trial / "freeze.json", {"authored": True})
    task = {"source_id": "authored", "component": "fixture", "role": "point", "kind": "ai"}
    (pairs / "qualification/pairs.jsonl").write_text(
        json.dumps({"source_id": "authored", "truth_png_sha256": "authored-mask"}) + "\n"
    )
    monkeypatch.setattr(training, "PAIRS", pairs)
    monkeypatch.setattr(producer, "lease", lambda path: ({}, {}, {"rows": [task]}))
    monkeypatch.setattr(producer, "comparison_tasks", lambda value: [task])
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (48, 64, 3), dtype=np.uint8))
    monkeypatch.setattr(training, "source_views", lambda *args: (image, {}, np.ones((48, 64), np.uint8)))
    calls = []

    def predict(value):
        calls.append(value.size)
        return np.full((48, 64), 0.75, np.float32), {"image_gate": None, "image_score": None}

    monkeypatch.setattr(producer, "load_predictor", lambda *args: predict)
    producer.score(trial, "candidate", "native", 0)
    result = producer.read(trial / "candidate-native-000/completed.json")
    row = json.loads((trial / "candidate-native-000/rows.jsonl").read_text())
    assert calls == [(64, 48), (64, 48)]
    assert result["completed_model_forwards"] == 2 and result["exact_repeat_extra_forwards"] == 1
    assert row["role"] == "point" and row["status"] == "ok"
    assert producer.sha(row["raw_map_file"]) == row["raw_map_sha256"]
    assert np.array_equal(
        np.load(row["raw_map_file"], allow_pickle=False), np.full((48, 64), 0.75, np.float32)
    )
    with pytest.raises(FileExistsError):
        producer.score(trial, "candidate", "native", 0)
    assert len(calls) == 2


def test_three_path_measurement_preserves_source_binding_and_actual_mask_union(tmp_path, monkeypatch):
    import hashlib

    from PIL import Image

    pixels = np.random.default_rng(12).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    data, _image = producer.encoded_view(Image.fromarray(pixels), "native")
    truth = np.zeros((192, 256), bool)
    truth[40:150, 60:190] = True
    raw = {
        "candidate": truth.astype(np.float32) * np.float32(0.99),
        "baseline": np.zeros((384, 384), np.float32),
        "fused-reference": np.zeros((512, 512), np.float32),
    }
    raw["baseline"][70:280, 90:290] = 0.99
    raw["fused-reference"][90:350, 130:390] = 0.99
    rows = []
    for arm in producer.ARMS:
        rows.append(
            {
                "source_id": "authored",
                "component": "fixture",
                "role": "point",
                "kind": "ai",
                "status": "ok",
                "format": "native",
                "arm": arm,
                "input_sha256": hashlib.sha256(data).hexdigest(),
                "source_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "source_size": [256, 192],
                "positive_truth_png_sha256": "authored-truth",
                "negative_truth_is_zero": False,
                "image_gate": True if arm != "candidate" else None,
                "image_score": 0.99 if arm == "fused-reference" else None,
                "pixel_threshold": 0.5 if arm == "fused-reference" else 0.7,
            }
        )

    def load(row, _root, shape):
        assert raw[row["arm"]].shape == shape
        return raw[row["arm"]]

    monkeypatch.setattr(readout, "load_map", load)
    result = readout.measure(rows, tmp_path, data, pixels, truth, "authored-truth")

    def count(name):
        value = result["baseline"] if name == "baseline" else result["candidate"][name]
        return value["metrics"]["predicted_positive_pixels"]

    assert set(result["candidate"]) == set(readout.METHODS)
    assert count(producer.POLICIES[0]) > 0
    assert count("current-or-strict") + count("current-and-strict") == count("baseline") + count(
        producer.POLICIES[0]
    )
    assert count("current-or-strict") >= max(count("baseline"), count(producer.POLICIES[0]))
    rows[0]["source_rgb_sha256"] = "changed"
    with pytest.raises(ValueError, match="input bytes/RGB"):
        readout.measure(rows, tmp_path, data, pixels, truth, "authored-truth")


@pytest.mark.parametrize("arguments", [["unused", "baseline", "5"], ["unused", "unknown", "0"]])
def test_dispatch_rejects_unplanned_phases_before_creating_outputs(arguments):
    with pytest.raises(ValueError):
        runtime.run("score", arguments)


def metric(positive, flagged):
    tp, fp, fn = (20, 0, 0) if positive and flagged else (0, 0, 20) if positive else (0, int(flagged), 0)
    return {
        "status": "ok",
        "metrics": {
            "pixels": 100,
            "intersection": tp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "predicted_positive_pixels": tp + fp,
            "reference_positive_pixels": tp + fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
    }


def panel():
    rows = []
    for task in producer.comparison_tasks(plan()):
        i = int(task["source_id"])
        positive = task["kind"] == "ai"
        rows.append(
            {
                **task,
                "format": "native",
                "baseline": metric(positive, i < 6 if positive else i < 3),
                "candidate": {
                    name: metric(positive, i < 18 if positive else i < 6) for name in readout.METHODS
                },
            }
        )
    return rows


def test_large_gain_is_reported_even_when_a_control_false_alarm_count_increases():
    result = readout.summarize(panel())
    selected = result["methods"][producer.POLICIES[0]]
    assert selected["usefully_localized_photos"] == 18
    assert selected["false_groups_by_kind"]["original"] == 6
    assert selected["paired_useful_fraction_difference"] > 0
    assert result["automatic_zero_false_alarm_increase_rule_applied"] is False
    assert result["serving_admitted"] is result["final_confirmation_used"] is False
    assert producer.POLICIES[0] in result["tradeoff_comparison"]["models"]


@pytest.mark.parametrize("failure", ["missing", "unchecked", "wrong_role"])
def test_missing_or_inconclusive_views_cannot_become_originality(failure):
    values = panel()
    if failure == "missing":
        values.pop()
    elif failure == "unchecked":
        values[0]["candidate"][producer.POLICIES[0]] = {"status": "inconclusive"}
    else:
        values[0]["role"] = "confirmation"
    with pytest.raises(ValueError):
        readout.summarize(values)
