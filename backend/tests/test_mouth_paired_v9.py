import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from scripts.mouth_learner_v8 import MouthLearner, frozen_identity
from scripts.mouth_paired_v9 import make_pairs, optimize, teacher_eligible, teacher_targets
from scripts.run_mouth_paired_v9 import gates, native_parity


def fixture():
    rows, bindings = [], []
    offset = 0
    for group, role in (("a", "fit"), ("b", "fit"), ("c", "point")):
        for condition in ("original", "local-light", "local-resample", "ai-redraw", "ai-retime"):
            rows.append(
                {
                    "id": group + condition,
                    "group": group,
                    "role": role,
                    "family": "liveportrait",
                    "condition": condition,
                    "label": int(condition.startswith("ai-")),
                    "frame_index": 10,
                    "source_sha256": group,
                    "original_rgb_sha256": group,
                }
            )
            views = 4 if role == "fit" else 1
            bindings.append(
                {"id": rows[-1]["id"], "views": [{"view": k, "offset": offset + k} for k in range(views)]}
            )
            offset += views
    tokens = torch.randn(offset, 5, 24, generator=torch.Generator().manual_seed(123))
    return rows, bindings, tokens


def tiny():
    torch.manual_seed(7)
    return MouthLearner(
        Dinov2Model(
            Dinov2Config(
                hidden_size=24,
                intermediate_size=48,
                num_hidden_layers=12,
                num_attention_heads=3,
                image_size=28,
                patch_size=14,
            )
        )
    )


def test_pair_identity_is_same_actor_source_and_frame_and_never_point():
    rows, _, _ = fixture()
    pairs = make_pairs(rows)
    assert len(pairs) == 12
    assert all(rows[p]["role"] == rows[n]["role"] == "fit" for p, n in pairs)
    changed = copy.deepcopy(rows)
    changed[0]["source_sha256"] = "different-source"
    with pytest.raises(ValueError, match="identities differ"):
        make_pairs(changed)
    changed = copy.deepcopy(rows)
    changed[0]["frame_index"] += 1
    with pytest.raises(ValueError, match="missing"):
        make_pairs(changed)


def test_teacher_targets_exclude_every_point_input_and_first_family_ai():
    torch.set_num_threads(1)
    rows, bindings, tokens = fixture()
    teacher = tiny()
    _, eligible = teacher_targets(teacher, tokens, rows, bindings)
    assert int(eligible.sum()) == 24
    for row, binding in zip(rows, bindings, strict=True):
        assert all(bool(eligible[v["offset"]]) == teacher_eligible(row) for v in binding["views"])


def test_paired_learning_keeps_teacher_prefix_frozen_and_ignores_point_labels_pixels():
    torch.set_num_threads(1)
    rows, bindings, tokens = fixture()
    first = tiny()
    teacher = copy.deepcopy(first)
    before = {k: p.clone() for k, p in teacher.state_dict().items()}
    prefix = frozen_identity(first)
    result = optimize(first, teacher, tokens, rows, bindings, epochs=1, batch=4)
    assert result["unique_pairs"] == 12
    assert result["teacher_targets"] == 24
    assert frozen_identity(first) == prefix
    assert all(torch.equal(value, teacher.state_dict()[key]) for key, value in before.items())
    altered = copy.deepcopy(rows)
    pixels = tokens.clone()
    for i, row in enumerate(altered):
        if row["role"] == "point":
            row["label"] = 1 - row["label"]
            pixels[bindings[i]["views"][0]["offset"]] = 10000
    second = tiny()
    optimize(second, copy.deepcopy(second), pixels, altered, bindings, epochs=1, batch=4)
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())


def test_learned_features_can_change_but_source_timeline_and_crop_cannot():
    old = [
        {
            "frame_index": 4,
            "pts": 160,
            "decoded_rgb_sha256": "a",
            "model_input_float32_sha256": "b",
            "features_float32_sha256": "old",
            "margin": 0.1,
        }
    ]
    new = copy.deepcopy(old)
    new[0].update(features_float32_sha256="learned", margin=0.5)
    assert native_parity(new, old)
    new[0]["pts"] += 40
    assert not native_parity(new, old)


def test_more_first_family_locations_cannot_hide_lost_second_family_or_extra_flags():
    source = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/mouth-learning-v8-evaluation.json").read_text()
    )
    a = source["cases"]["candidate"]
    b = copy.deepcopy(a)
    families = [
        "musetalk"
        if r["condition"].startswith("muse-") or r["condition"].startswith("local-blur-")
        else "liveportrait"
        for r in a
    ]
    lp = [i for i, r in enumerate(b) if families[i] == "liveportrait" and r["label"]]
    for i in lp[:10]:
        b[i].update(localized=True, flagged=True, spatial_iou=0.3, temporal_iou=0.7)
    muse = [i for i, r in enumerate(b) if families[i] == "musetalk" and r["localized"]]
    for i in muse[:2]:
        b[i].update(localized=False, flagged=False, spatial_iou=0, temporal_iou=0)
    rows = [{"status": "ok", "baseline_replay_parity": True, "prior_native_input_parity": True}] * 88
    result = gates(a, b, families, rows)
    assert result["six_net_additional_locations"]
    assert not result["retained_second_family"]
    assert not result["at_most_one_lost_location"]
    assert result["first_family_six_locations"]
    for i in muse[:2]:
        b[i] = copy.deepcopy(a[i])
    controls = [i for i, r in enumerate(b) if not r["label"]]
    b[controls[0]]["flagged"] = b[controls[1]]["flagged"] = True
    assert not gates(a, b, families, rows)["total_ordinary_flags"]
