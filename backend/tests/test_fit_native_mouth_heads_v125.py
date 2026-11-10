import copy

import numpy as np
import pytest

from app.detectors.video.decode import FrameTable
from scripts import fit_native_mouth_heads_v125 as fit


def labelled_rows():
    rows = []
    for group in ("source-a", "source-b"):
        for label in (0, 1):
            for condition in ("original", "blur") if label == 0 else ("ai_aligned", "ai_earlier"):
                rows.extend(
                    {
                        "source_group": group,
                        "study_role": "fusion_fit",
                        "condition": condition,
                        "label": label,
                    }
                    for _ in range(4 if group == "source-a" else 7)
                )
    return rows


def test_source_class_condition_weights_keep_short_original_only_source():
    rows = labelled_rows() + [
        {"source_group": "short", "study_role": "fusion_fit", "condition": "original", "label": 0}
    ]
    weights = fit.balanced_weights(rows)
    assert np.isclose(weights.sum(), len(rows))
    for label in (0, 1):
        selected = [i for i, r in enumerate(rows) if r["label"] == label]
        assert np.isclose(weights[selected].sum(), len(rows) / 2)
        groups = {rows[i]["source_group"] for i in selected}
        masses = [weights[[i for i in selected if rows[i]["source_group"] == g]].sum() for g in groups]
        assert np.allclose(masses, masses[0])
    assert weights[-1] > 0


@pytest.mark.parametrize("role", ["confidence_cal", "selection", "confirmation"])
def test_nonfitting_labels_are_rejected_before_learning(role):
    rows = labelled_rows()
    rows[0]["study_role"] = role
    with pytest.raises(ValueError, match="fitting"):
        fit.fit_linear(rows, np.ones((len(rows), 16)))


def test_separate_small_heads_learn_authored_signal_as_raw_margins():
    rows = labelled_rows()
    rng = np.random.default_rng(125)
    x = rng.normal(size=(len(rows), 16)) * 0.01
    x[:, 0] = [2 * r["label"] - 1 for r in rows]
    head = fit.fit_linear(rows, x)
    predictions = fit.margins(x, head)
    assert np.all(predictions[[i for i, r in enumerate(rows) if r["label"] == 1]] > 0)
    assert np.all(predictions[[i for i, r in enumerate(rows) if r["label"] == 0]] < 0)
    assert head["response_kind"] == "raw_signed_margin_not_probability"
    assert head["deployment_eligible"] is False
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        fit.margins(x, head)


def test_all_supported_points_are_selected_independently_of_truth_or_role(monkeypatch):
    table = FrameTable("1/1000", (0, 40, 80, 120, 160, 200), (), "??????", 100, 100, 100, 100)
    image = np.arange(6 * 768, dtype=np.float32).reshape(6, 768) / 1000
    audio = np.arange(4 * 3840, dtype=np.float64).reshape(4, 3840)
    row = {
        "status": "extracted",
        "table": table.to_json(),
        "speech": {"indices": [1, 2, 3, 4]},
        "frames": [{"status": "ok", "frame_index": i, "feature_row": i} for i in range(6)],
    }
    monkeypatch.setattr(
        fit.reviewed, "array_file", lambda record, dtype: image if dtype == np.float32 else audio
    )
    indices, visual, sound = fit.point_features(row)
    assert indices == [1, 2, 3, 4] and visual.shape == (4, 2304) and np.array_equal(sound, audio)
    changed = copy.deepcopy(row)
    changed.update(condition="ai_aligned", study_role="confirmation", edit_frames_half_open=[2, 3])
    other = fit.point_features(changed)
    assert other[0] == indices and np.array_equal(other[1], visual) and np.array_equal(other[2], sound)
    changed["frames"][2]["status"] = "unassessed"
    assert fit.point_features(changed)[0] == [4]


def test_unsupported_audio_and_native_gaps_are_not_bridged(monkeypatch):
    table = FrameTable("1/1000", (0, 40, 80, 1000, 1040, 1080), (), "??????", 100, 100, 100, 100)
    row = {
        "status": "extracted",
        "table": table.to_json(),
        "speech": {"indices": [2, 3, 4]},
        "frames": [{"status": "ok", "frame_index": i, "feature_row": i} for i in range(6)],
    }
    monkeypatch.setattr(
        fit.reviewed,
        "array_file",
        lambda record, dtype: np.ones((6, 768), np.float32) if dtype == np.float32 else np.ones((3, 3840)),
    )
    assert fit.point_features(row)[0] == [4]
    indices, visual, audio = fit.point_features({"status": "unavailable"})
    assert indices == [] and visual.shape == (0, 2304) and audio.shape == (0, 3840)


def test_neural_or_other_batch_stage_cannot_be_routed():
    with pytest.raises(ValueError):
        fit.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract"])
