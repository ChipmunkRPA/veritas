"""Authored geometry, assignment, calibration and role-isolation checks."""

import json

import numpy as np
import pytest
import torch

from scripts import writing_span_calibration_v7 as calibration
from scripts import writing_span_data_v7 as data
from scripts import writing_span_geometry_v7 as geometry
from scripts import writing_span_heads_v7 as heads

torch.set_num_threads(2)


def test_unicode_spans_use_characters_and_do_not_charge_whitespace():
    text = "a 🙂 e\u0301 z"
    labels = [int(2 <= i < 6) for i in range(len(text))]
    assert geometry.truth_spans(text, labels) == [[2, 6]]
    mask = np.ones(len(text), bool)
    result = geometry.metrics(text, labels, mask)
    assert result["ai_characters"] == 3 and result["human_characters"] == 2
    assert result["character_iou"] == 3 / 5


def test_missing_combining_character_is_not_assessed_as_original():
    with pytest.raises(ValueError, match="coverage omits"):
        geometry.native_coverage("e\u0301", [(0, 1)])


def test_out_of_source_native_offset_is_refused():
    with pytest.raises(ValueError, match="exceeds source"):
        geometry.native_coverage("abc", [(0, 4)])


def test_overlapping_native_offsets_are_a_union_for_evidence():
    text = "a🙂z"
    proposals = [{"start": a, "end": b, "score": 0.9} for a, b in [(0, 1), (1, 2), (1, 2), (2, 3)]]
    geometry.native_coverage(text, [(p["start"], p["end"]) for p in proposals])
    result = geometry.metrics(text, [0, 1, 0], geometry.proposals_mask(text, proposals, 0.5))
    assert result["located_ai_characters"] == 1 and result["highlighted_characters"] == 3


def test_token_that_crosses_a_known_boundary_keeps_soft_source_target():
    target, weight = geometry.token_targets("aB", [0, 1], [(0, 2)])
    assert target.tolist() == [0.5] and weight.tolist() == [2]


def test_source_metric_charges_false_and_missed_characters():
    text = "aaBBcc"
    mask = geometry.proposals_mask(text, [{"start": 0, "end": 3, "score": 0.9}], 0.5)
    result = geometry.metrics(text, [0, 0, 1, 1, 0, 0], mask)
    assert result["located_ai_characters"] == 1 and result["missed_ai_characters"] == 1
    assert result["highlighted_human_characters"] == 2
    assert result["character_iou"] == 0.25 and result["character_f1"] == 0.4


def test_interval_integer_readout_is_explicit_floor_start_ceil_end():
    assert geometry.absolute_interval([0.199, 0.601], 10) == [1, 7]
    assert geometry.absolute_interval([0, 1], 7) == [0, 7]


@pytest.mark.parametrize("span", [[0.5, 0.5], [-0.1, 0.3], [0.2, 1.1], [0, float("nan")]])
def test_invalid_interval_cannot_supply_source_evidence(span):
    with pytest.raises(ValueError):
        geometry.absolute_interval(span, 10)


def point_rows():
    return [
        {
            "id": f"{i}:{j}",
            "group": str(i),
            "role": "point",
            "label": 0,
            "status": "ok",
            "proposals": [{"start": 0, "end": 2, "score": 0.2 + i / 1000}],
        }
        for i in range(32)
        for j in range(4)
    ]


def test_points_group_correlated_compositions_and_keep_ties():
    rows = point_rows()
    assert geometry.genuine_point(rows)["groups_highlighted"] == 1
    for r in rows:
        r["proposals"][0]["score"] = 0.4
    point = geometry.genuine_point(rows)
    assert point["groups_highlighted"] == 0 and point["cut"] > 0.4


@pytest.mark.parametrize("fault", ["AI", "diagnostic", "missing_composition", "failed"])
def test_ai_missing_failed_and_diagnostic_cases_cannot_choose_points(fault):
    rows = point_rows()
    if fault == "AI":
        rows[0]["label"] = 1
    elif fault == "diagnostic":
        rows[0]["role"] = "diagnostic"
    elif fault == "failed":
        rows[0]["status"] = "failed"
    else:
        rows.pop()
    with pytest.raises(ValueError):
        geometry.genuine_point(rows)


def test_nine_recipes_keep_exact_source_slices_and_matched_controls():
    sources = {
        k: {"row": k, "text": " ".join(f"{chr(104 + k)}{i}" for i in range(200)), "text_sha256": str(k) * 64}
        for k in (0, 1)
    }
    cases = [
        data.compose("authored-group", "fit", name, parts, sources)
        for name, parts in data.recipes("authored-group").items()
    ]
    assert len(cases) == 9 and len({r["text_sha256"] for r in cases}) == 9
    assert sorted(len(r["truth_spans"]) for r in cases) == [0, 0, 0, 0, 1, 1, 1, 2, 3]
    assert all(len(r["text"].split()) == 160 for r in cases)
    for case in cases:
        for part in case["source_parts"]:
            original = sources[part["label"]]["text"][part["source_start"] : part["source_end"]]
            assert original in case["raw_text"]


def test_interval_iou_has_known_overlap_and_disjoint_penalty():
    a, b = torch.tensor([[0.0, 0.5]]), torch.tensor([[0.25, 0.75], [0.75, 1.0]])
    assert torch.allclose(heads.pair_iou(a, b), torch.tensor([[1 / 3, 0.0]]))
    assert float(heads.pair_iou(a, b, True)[0, 1]) < 0


def test_assignment_loss_is_finite_on_positive_and_empty_documents():
    logits = torch.tensor([[[5.0, -5.0], [5.0, -5.0]], [[-5.0, 5.0], [-5.0, 5.0]]], requires_grad=True)
    spans = torch.tensor([[[0.6, 0.9], [0.1, 0.3]], [[0.2, 0.4], [0.6, 0.8]]], requires_grad=True)
    target = [torch.tensor([[0.1, 0.3], [0.6, 0.9]]), torch.empty((0, 2))]
    value = heads.interval_loss({"logits": logits, "spans": spans}, target)
    assert torch.isfinite(value) and float(value.detach()) < 0.001
    value.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(spans.grad).all()


def test_invalid_training_intervals_are_refused():
    with pytest.raises(ValueError, match="valid source intervals"):
        heads.interval_loss(
            {"logits": torch.zeros((1, 8, 2)), "spans": torch.ones((1, 8, 2)) / 2},
            [torch.tensor([[0.8, 0.2]])],
        )


@pytest.mark.parametrize("kind", ["token", "interval"])
def test_saved_small_head_roundtrip_and_metadata_invariance(kind):
    model = heads.build(kind).eval()
    case = {
        "id": "a",
        "group": "author-a",
        "role": "diagnostic",
        "text": "abcdefghijkl",
        "features": np.random.default_rng(7).normal(size=(12, 768)).astype(np.float32),
        "positions": np.linspace(0, 1, 12).astype(np.float32),
        "offsets": [(i, i + 1) for i in range(12)],
    }
    original = heads.propose(model, kind, case)
    restored = heads.restore(kind, json.loads(json.dumps(heads.save_state(model))))
    changed = {
        **case,
        "id": "b",
        "group": "another-author",
        "label": 1,
        "truth_spans": [[0, 12]],
        "generator": "unused",
    }
    assert heads.propose(restored, kind, changed) == original
    bad = heads.save_state(model)
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError, match="fields differ"):
        heads.restore(kind, bad)


def test_padding_cannot_supply_interval_evidence():
    model = heads.build("interval").eval()
    f = torch.randn(1, 8, 768)
    pos = torch.linspace(0, 1, 8).reshape(1, 8)
    padding = torch.tensor([[False] * 5 + [True] * 3])
    with torch.no_grad():
        a = model(f, pos, padding)
        f[:, 5:] = 1e6
        b = model(f, pos, padding)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_diagnostic_cases_cannot_fit_a_new_head_or_calibration():
    with pytest.raises(ValueError, match="fitting-role"):
        heads.fit([{"role": "diagnostic"}], "token")
    with pytest.raises(ValueError, match="calibration-role"):
        calibration.fit([{}], [{"role": "diagnostic"}])


def test_calibration_gradient_matches_separate_finite_differences():
    x, y, w = (
        np.asarray([-2.0, -0.4, 0.6, 2.0]),
        np.asarray([0.0, 0.2, 0.7, 1.0]),
        np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    theta = np.asarray([-0.3, 0.8])
    _, gradient = calibration.objective(theta, x, y, w)
    finite = []
    for i in range(2):
        direction = np.zeros(2)
        direction[i] = 1e-6
        finite.append(
            (
                calibration.objective(theta + direction, x, y, w)[0]
                - calibration.objective(theta - direction, x, y, w)[0]
            )
            / 2e-6
        )
    assert np.max(np.abs(gradient - finite)) < 1e-8


def test_source_balanced_soft_fraction_calibration_converges_monotonically():
    cases, rows = [], []
    for i in range(288):
        target = i % 3
        cases.append(
            {
                "id": str(i),
                "group": str(i // 9),
                "role": "cal",
                "text": "ab",
                "labels": [[0, 0], [0, 1], [1, 1]][target],
            }
        )
        rows.append(
            {
                "id": str(i),
                "status": "ok",
                "proposals": [{"start": 0, "end": 2, "raw": [0.15, 0.5, 0.85][target]}],
            }
        )
    fitted = calibration.fit(rows, cases)
    assert fitted["projected_gradient"] <= 1e-6 and fitted["slope"] > 0
    assert calibration.predict(0.8, fitted) > calibration.predict(0.2, fitted)
    assert fitted["customer_confidence"] is False


def test_separate_set_count_matches_masks_on_multiple_intervals():
    from scripts.review_writing_spans_v7 import separate_measure
    from scripts.run_writing_spans_v7 import measure

    case = {
        "text": "aa BB cc DD",
        "labels": [0, 0, -1, 1, 1, -1, 0, 0, -1, 1, 1],
        "label": 1,
        "truth_spans": [[3, 5], [9, 11]],
    }
    proposals = [{"start": 1, "end": 5, "score": 0.8}, {"start": 8, "end": 10, "score": 0.9}]
    assert separate_measure(case, proposals, 0.7) == measure(case, proposals, 0.7)


def summary_rows():
    from scripts.run_writing_spans_v7 import ARMS

    rows = []
    for i in range(32):
        for recipe in data.recipes(str(i)):
            ai = recipe not in data.GENUINE
            rows.append(
                {
                    "id": f"{i}:{recipe}",
                    "group": str(i),
                    "recipe": recipe,
                    "role": "diagnostic",
                    "status": "ok",
                    "label": int(ai),
                    "measures": {
                        k: {
                            "highlighted_characters": 0,
                            "useful_location": False,
                            "character_iou": 0.0 if ai else 1.0,
                            "recovered_intervals": 0,
                            "truth_intervals": int(ai),
                            "highlighted_human_characters": 0,
                            "missed_ai_characters": int(ai),
                        }
                        for k in ARMS
                    },
                }
            )
    return rows


def test_inconclusive_silence_is_not_a_candidate_improvement():
    from scripts.run_writing_spans_v7 import summarize

    summary = summarize(summary_rows())
    assert not summary["passes_numerical_development_gates"]
    assert not summary["serving_admitted"]
    assert summary["arms"]["interval"]["genuine_groups_highlighted"] == 0


def test_better_average_cannot_hide_lost_useful_cases_or_false_alarms():
    from scripts.run_writing_spans_v7 import summarize

    rows = summary_rows()
    for row in rows:
        if row["recipe"].startswith("mixed-"):
            row["measures"]["interval"].update(character_iou=0.8, useful_location=True)
    rows[2]["measures"]["current"].update(character_iou=0.7, useful_location=True)
    rows[2]["measures"]["interval"].update(character_iou=0, useful_location=False)
    for row in rows:
        if row["recipe"] == "human" and row["group"] in ("0", "1"):
            row["measures"]["interval"]["highlighted_characters"] = 1
    result = summarize(rows)["interval_vs"]["current"]
    assert result["checks"]["higher_mixed_character_iou"]
    assert not result["checks"]["no_lost_useful_mixed_document"]
    assert not result["checks"]["similar_genuine_group_flags"]
    assert not result["passes_numerical_development_gates"]
