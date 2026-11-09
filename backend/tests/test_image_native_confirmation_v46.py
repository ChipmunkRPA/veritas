"""Fixed selection, role isolation, inherited math and confirmation reporting."""

import ast
import copy
import hashlib
import inspect
import re

import numpy as np
import pytest

from scripts import image_native_comparison_readout_v45 as previous_readout
from scripts import image_native_comparison_runtime_v45 as previous_runtime
from scripts import image_native_comparison_v45 as previous
from scripts import image_native_confirmation_readout_v46 as readout
from scripts import image_native_confirmation_runtime_v46 as runtime
from scripts import image_native_confirmation_v46 as producer


def plan():
    return {
        "rows": [
            {"source_id": str(i), "component": str(min(i, 47)), "role": "confirmation"} for i in range(59)
        ]
    }


def test_complete_confirmation_support_and_exact_fixed_methods():
    tasks = producer.comparison_tasks(plan())
    assert len(tasks) == 236 and {r["role"] for r in tasks} == {"confirmation"}
    assert [len(producer.phase_tasks(tasks, n)) for n in range(10)] == [24] * 9 + [20]
    assert producer.ARMS == ("candidate", "baseline")
    assert producer.POLICIES == ("p0.90-a0.01000",)
    assert readout.METHODS == ("p0.90-a0.01000", "current-and-strict")
    assert readout.source_mask is previous_readout.source_mask
    assert len(producer.ASSETS) == 2


@pytest.mark.parametrize("kind", ["missing", "duplicate", "cross_role"])
def test_role_leakage_or_partial_support_is_rejected(kind):
    value = plan()
    if kind == "missing":
        value["rows"].pop()
    elif kind == "duplicate":
        value["rows"][-1] = copy.deepcopy(value["rows"][-2])
    else:
        value["rows"].append({"source_id": "point", "component": "0", "role": "point"})
    with pytest.raises(ValueError):
        producer.comparison_tasks(value)


@pytest.mark.parametrize("arm", ["candidate", "baseline"])
def test_same_frozen_native_predictor_without_new_model_or_format(monkeypatch, arm):
    calls = []
    monkeypatch.setattr(producer.original, "load_predictor", lambda a, f, c: calls.append((a, f, c)))
    producer.load_predictor(arm, "native", {})
    assert calls == [(arm, "native", {})]
    with pytest.raises(ValueError):
        producer.load_predictor(arm, "jpeg75", {})
    with pytest.raises(ValueError):
        producer.load_predictor("fused-reference", "native", {})


def test_scoring_math_is_unchanged_except_explicit_role_schema_and_removed_third_path():
    expected = ast.parse(inspect.getsource(previous.score)).body[0]

    class Scope(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = re.sub(r"\bpoint\b", "confirmation", node.value).replace(
                    "producer/45", "producer/46"
                )
            return node

        def visit_Set(self, node):
            self.generic_visit(node)
            node.elts = [
                x for x in node.elts if not (isinstance(x, ast.Constant) and x.value == "fused-reference")
            ]
            return node

    assert ast.dump(Scope().visit(expected)) == ast.dump(ast.parse(inspect.getsource(producer.score)).body[0])


def test_corrected_observer_resource_math_is_unchanged():
    def body(fn):
        source = inspect.getsource(fn)
        source = source[source.index("    if output.exists():") :]
        source = source.replace("fixed-monitor-profile/45", "fixed-monitor-profile/46")
        source = source.replace(
            "One explicit calibration phase with unchanged inference/readout math and corrected terminal observation.",
            "One explicit confirmation phase with methods fixed before source inference and corrected terminal observation.",
        )
        return ast.dump(ast.parse("def observer():\n" + source))

    assert body(previous_runtime.run) == body(runtime.run)


def test_actual_confirmation_masks_match_selected_v45_views(tmp_path, monkeypatch):
    from PIL import Image

    pixels = np.random.default_rng(46).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    data, _ = producer.encoded_view(Image.fromarray(pixels), "native")
    truth = np.zeros((192, 256), bool)
    truth[40:150, 60:190] = True
    maps = {
        "candidate": truth.astype(np.float32) * np.float32(0.99),
        "baseline": np.zeros((384, 384), np.float32),
    }
    maps["baseline"][70:280, 90:290] = 0.99
    rows = [
        {
            "source_id": "authored",
            "component": "fixture",
            "role": "confirmation",
            "kind": "ai",
            "status": "ok",
            "format": "native",
            "arm": arm,
            "input_sha256": hashlib.sha256(data).hexdigest(),
            "source_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "source_size": [256, 192],
            "positive_truth_png_sha256": "authored-truth",
            "negative_truth_is_zero": False,
            "image_gate": True if arm == "baseline" else None,
            "pixel_threshold": 0.7,
        }
        for arm in producer.ARMS
    ]
    monkeypatch.setattr(readout, "load_map", lambda row, _root, shape: maps[row["arm"]])
    value = readout.measure(rows, tmp_path, data, pixels, truth, "authored-truth")
    strict = previous_readout.source_mask(maps["candidate"], True, 0.9, data, pixels, truth)
    current = previous_readout.source_mask(maps["baseline"], True, 0.7, data, pixels, truth)
    assert value["candidate"][producer.POLICIES[0]]["metrics"] == readout.mask_metrics(strict, truth)
    assert value["candidate"]["current-and-strict"]["metrics"] == readout.mask_metrics(
        current & strict, truth
    )
    rows[0]["role"] = "point"
    with pytest.raises(ValueError):
        readout.measure(rows, tmp_path, data, pixels, truth, "authored-truth")


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
                "baseline": metric(positive, i < 8 if positive else i < 4),
                "candidate": {
                    producer.POLICIES[0]: metric(positive, i < 40 if positive else i < 12),
                    "current-and-strict": metric(positive, i < 8 if positive else False),
                },
            }
        )
    return rows


def test_confirmation_reports_gain_and_false_alarms_without_relabeling_as_selection():
    value = readout.summarize(panel())
    primary = value["methods"][producer.POLICIES[0]]
    assert primary["usefully_localized_photos"] == 40
    assert primary["control_photos"]["original"] == {
        "photos": 59,
        "flagged": 12,
        "mean_wrongly_highlighted_area_fraction": 12 / 5900,
    }
    assert primary["paired_useful_fraction_difference"] > 0
    assert value["final_confirmation_used"] is True and value["comparison_source_role"] == "confirmation"
    assert value["source_groups"] == 48 and value["photos"] == 59
    assert (
        value["serving_admitted"]
        is value["probability_calibration_performed"]
        is value["automatic_zero_false_alarm_increase_rule_applied"]
        is False
    )


@pytest.mark.parametrize("kind", ["missing", "unchecked", "wrong_role"])
def test_bad_measurements_are_never_dropped_or_interpreted_as_original(kind):
    rows = panel()
    if kind == "missing":
        rows.pop()
    elif kind == "unchecked":
        rows[0]["candidate"][producer.POLICIES[0]] = {"status": "inconclusive"}
    else:
        rows[0]["role"] = "point"
    with pytest.raises(ValueError):
        readout.summarize(rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("primary_next_readout", "p0.60-a0.01000"),
        ("secondary_next_readout", "current-or-strict"),
        ("confirmation_forwards_at_selection", 1),
        ("next_source_role", "point"),
        ("next_source_photos", 58),
    ],
)
def test_confirmation_cannot_change_selection_after_seeing_labels(monkeypatch, tmp_path, field, value):
    selection = {
        "primary_next_readout": "p0.90-a0.01000",
        "secondary_next_readout": "current-and-strict",
        "comparator": "baseline",
        "selection_precedes_confirmation_forwards": True,
        "confirmation_forwards_at_selection": 0,
        "next_source_role": "confirmation",
        "next_source_photos": 59,
        "next_source_components": 48,
        "completed_summary_sha256": "7f2bc695455ba5786c59d0a7c63d3483d7d9b1d039860e51bb8c2e273493be63",
    }
    selection[field] = value
    monkeypatch.setattr(producer, "read", lambda path: selection)
    monkeypatch.setattr(
        producer, "sha", lambda path: "7f2bc695455ba5786c59d0a7c63d3483d7d9b1d039860e51bb8c2e273493be63"
    )
    with pytest.raises(ValueError):
        producer.check_selection(tmp_path)


@pytest.mark.parametrize("arguments", [["unused", "candidate", "10"], ["unused", "fused-reference", "0"]])
def test_unplanned_phase_or_extra_model_refuses_before_outputs(arguments):
    with pytest.raises(ValueError):
        runtime.run("score", arguments)
