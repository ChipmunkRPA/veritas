import copy

import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from scripts.mouth_learner_v8 import (
    MouthLearner,
    choose_point,
    fit_positions,
    frozen_identity,
    optimize,
    pool,
    view,
)
from scripts.prepare_mouth_learning_v8 import conditioning_index, validate_parent
from scripts.run_mouth_learning_v8 import advancement


def tiny():
    torch.manual_seed(5)
    config = Dinov2Config(
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=12,
        num_attention_heads=3,
        image_size=28,
        patch_size=14,
    )
    return MouthLearner(Dinov2Model(config))


def fixture():
    rows, bindings = [], []
    for group, role in (("a", "fit"), ("b", "fit"), ("c", "point")):
        for condition, label in (("resample", 0), ("redraw", 1)):
            rows.append(
                {
                    "id": group + condition,
                    "group": group,
                    "role": role,
                    "condition": condition,
                    "label": label,
                }
            )
            bindings.append(
                {
                    "id": rows[-1]["id"],
                    "views": [
                        {"view": k, "offset": len(rows) * 4 - 4 + k}
                        for k in (range(4) if role == "fit" else [0])
                    ],
                }
            )
    return rows, bindings


def test_picture_conditioning_uses_source_clock_not_frame_index():
    mapping = {"sample_rate": 16000, "timeline": [{"start_sample": 0, "source_pts": 0}]}
    assert conditioning_index(1.04, mapping, 71) == 26
    mapping["timeline"][0]["source_pts"] = 640
    assert conditioning_index(1.04, mapping, 71) == 25
    with pytest.raises(ValueError, match="exact supported"):
        conditioning_index(1.05, mapping, 71)
    mapping["timeline"].append({"start_sample": 900, "source_pts": 2000})
    with pytest.raises(ValueError, match="contiguous"):
        conditioning_index(1.04, mapping, 71)


def test_cached_prefix_and_full_forward_have_same_representation():
    learner = tiny()
    x = torch.rand(2, 3, 28, 28)
    with torch.no_grad():
        assert torch.equal(
            learner.representation(learner.prefix(x)),
            pool(learner.backbone(pixel_values=x).last_hidden_state),
        )
    trainable = {k for k, p in learner.named_parameters() if p.requires_grad}
    assert all(
        k.startswith(
            ("backbone.encoder.layer.10.", "backbone.encoder.layer.11.", "backbone.layernorm.", "head.")
        )
        for k in trainable
    )
    assert any(k.startswith("backbone.encoder.layer.10.") for k in trainable)


def test_learning_updates_both_blocks_preserves_prefix_and_ignores_point_inputs():
    torch.set_num_threads(1)
    rows, bindings = fixture()
    tokens = torch.randn(24, 5, 24, generator=torch.Generator().manual_seed(91))
    first = tiny()
    old = frozen_identity(first)
    result = optimize(first, tokens, rows, bindings, epochs=2, batch=4)
    assert old == frozen_identity(first)
    assert result["fit_rows"] == 4
    assert any(k.startswith("backbone.encoder.layer.10.") for k in result["changed_trainable_parameters"])
    changed = copy.deepcopy(rows)
    changed_tokens = tokens.clone()
    for i, r in enumerate(changed):
        if r["role"] == "point":
            r["label"] = 1 - r["label"]
            changed_tokens[bindings[i]["views"][0]["offset"]] = 1000
    second = tiny()
    optimize(second, changed_tokens, changed, bindings, epochs=2, batch=4)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_point_positive_responses_do_not_select_point():
    rows, _ = fixture()
    response = [5.0] * len(rows)
    response[-2:] = [0.4, 1000.0]
    value = choose_point(rows, response)
    response[-1] = -1000
    assert choose_point(rows, response) == value
    assert value > 0.4 and value < 0.400001


def test_actor_leakage_and_point_augmentations_are_refused():
    rows, bindings = fixture()
    bindings[-1]["views"].append({"view": 1, "offset": 23})
    with pytest.raises(ValueError, match="native-only"):
        fit_positions(rows, bindings)
    rows, bindings = fixture()
    rows[-1]["group"] = "a"
    with pytest.raises(ValueError, match="overlap"):
        fit_positions(rows, bindings)


def test_fixed_views_transform_pixels_without_changing_original():
    chip = np.random.default_rng(3).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    original = chip.copy()
    assert np.array_equal(view(chip, 1), chip[:, ::-1])
    assert not np.array_equal(view(chip, 2), chip)
    assert np.array_equal(view(chip, 3), view(chip, 2)[:, ::-1])
    assert np.array_equal(chip, original)


def test_total_false_flags_gate_cannot_be_evaded_across_conditions():
    a, b, families = [], [], []
    for i in range(88):
        ai = i < 32
        base = {
            "id": str(i),
            "condition": "ai" if ai else f"ordinary{i % 3}",
            "label": int(ai),
            "flagged": False,
            "localized": False,
            "assessed": True,
            "spatial_iou": 0.0,
            "temporal_iou": 0.0,
        }
        candidate = {
            **base,
            "localized": ai,
            "flagged": ai or i in (32, 33, 34),
            "spatial_iou": 0.1 if ai else 0.0,
        }
        a.append(base)
        b.append(candidate)
        families.append("liveportrait" if i % 2 else "musetalk")
    rows = [{"status": "ok", "baseline_replay_parity": True}] * 88
    _, _, gates = advancement(a, b, families, rows)
    assert gates["condition_ordinary_flags"]
    assert not gates["total_ordinary_flags"]
    assert gates["both_families"]
    rows[-1] = {"status": "failed"}
    assert not advancement(a, b, families, rows)[2]["all_replays"]


def test_parent_manifest_source_roles_and_labels_cannot_be_substituted():
    with pytest.raises(ValueError, match="twenty"):
        validate_parent({"sources": []}, [])
