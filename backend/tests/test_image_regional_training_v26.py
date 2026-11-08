"""Leakage, schedule, source-coordinate and continuation contracts."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scripts import image_regional_training_v26 as study
from scripts.image_pair_truth_v21 import qualify


def groups():
    rows, components = [], []
    for index in range(320):
        key = f"new/{index}"
        rows.append(
            {
                "key": key,
                "source_id": index,
                "publisher_role": "testing" if index >= 240 else "training",
                "previously_consumed": False,
            }
        )
        component = [key]
        if index < 12:
            old = f"old/{index}"
            rows.append(
                {"key": old, "source_id": index, "publisher_role": "legacy", "previously_consumed": True}
            )
            component.append(old)
        components.append(component)
    return {"sources": rows, "components": components}


def test_related_sources_cannot_cross_roles_or_enter_reserved_groups():
    value = groups()
    value["sources"].append(
        {"key": "new/1000", "source_id": 1000, "publisher_role": "validation", "previously_consumed": False}
    )
    value["components"][0].append("new/1000")
    roles = study.grouped_roles(value)
    assert {
        role: sum(r["role"] == role for r in roles)
        for role in ("fit", "calibration", "point", "diagnostic", "confirmation")
    } == {"fit": 168, "calibration": 32, "point": 24, "diagnostic": 48, "confirmation": 48}
    assert all(r["role"] == "fit" for r in roles if r["old_linked"])
    assert all(
        r["publisher_nontraining"] and not r["old_linked"] for r in roles if r["role"] == "confirmation"
    )
    assert next(r for r in roles if 1000 in r["source_ids"])["source_ids"] == [0, 1000]
    value["sources"].reverse()
    value["components"] = [list(reversed(c)) for c in reversed(value["components"])]
    assert study.grouped_roles(value) == roles


@pytest.mark.parametrize("damage", ["duplicate", "missing", "insufficient"])
def test_incomplete_or_ambiguous_source_partitions_fail(damage):
    value = groups()
    if damage == "duplicate":
        value["components"][0].append("new/319")
    elif damage == "missing":
        value["components"].pop()
    else:
        for r in value["sources"]:
            r["publisher_role"] = "training"
    with pytest.raises(ValueError):
        study.grouped_roles(value)


def test_every_fit_photo_has_exactly_six_prescribed_paired_updates():
    rows = [{"source_id": i, "role": "fit" if i < 5 else "confirmation"} for i in range(8)]
    schedule = study.schedule(rows)
    assert len(schedule) == 30
    for i in range(5):
        source = [r for r in schedule if r["source_id"] == i]
        assert {(r["format"], r["negative"]) for r in source} == {
            (f, n) for f in ("native", "jpeg75") for n in ("original", "tone", "blur")
        }
        assert sorted(r["epoch"] for r in source) == list(range(6))
    assert max(r["source_id"] for r in schedule) == 4
    assert study.schedule(list(reversed(rows))) == schedule


def test_compression_never_relabels_background_as_ai():
    y, x = np.mgrid[:31, :47]
    pixels = np.stack([(x * 7) % 256, (y * 13) % 256, (x + y) % 256], axis=2).astype(np.uint8)
    truth = np.zeros((31, 47), np.uint8)
    truth[9:21, 15:29] = 255
    image = Image.fromarray(pixels)
    native = study.training_view(image, truth, "native", positive=True)
    jpeg = study.training_view(image, truth, "jpeg75", positive=True)
    assert native[2]["source_size"] == jpeg[2]["source_size"] == [47, 31]
    assert native[1].shape == jpeg[1].shape == (1, 1, 31, 47)
    assert np.array_equal(native[1].numpy(), jpeg[1].numpy())
    assert int(jpeg[1].sum()) == 12 * 14
    assert jpeg[2]["source_rgb_sha256"] != native[2]["source_rgb_sha256"]
    negative = study.training_view(image, truth, "jpeg75", positive=False)
    assert not bool(negative[1].any())


def test_controls_and_every_truth_pixel_are_replayed_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    pair_root = tmp_path / "pair"
    monkeypatch.setattr(study, "PAIRS", pair_root)
    (pair_root / "qualification").mkdir(parents=True)
    rng = np.random.default_rng(123)
    source = rng.integers(0, 256, (27, 41, 3), dtype=np.uint8)
    edited = source.copy()
    edited[7:18, 13:31] = 91
    alpha = np.zeros(source.shape[:2], np.uint8)
    alpha[7:18, 13:31] = 255
    changed, proof = qualify(source, edited, alpha)
    paths = {
        "original": tmp_path / "source.png",
        "edited": tmp_path / "edit.png",
        "mask": tmp_path / "mask.png",
        "truth": pair_root / "qualification/1.truth.png",
    }
    for name, value in (
        ("original", source),
        ("edited", edited),
        ("mask", alpha),
        ("truth", changed.astype(np.uint8) * 255),
    ):
        Image.fromarray(value).save(paths[name])
    case = {
        "original": "source.png",
        "original_sha256": study.sha(paths["original"]),
        "source_rgb_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
    }
    qualified = {
        "edited_image": {"root": ".", "file": "edit.png", "sha256": study.sha(paths["edited"])},
        "compositing_mask": {"root": ".", "file": "mask.png", "sha256": study.sha(paths["mask"])},
        "truth_mask": "1.truth.png",
        "truth_png_sha256": study.sha(paths["truth"]),
        "proof": proof,
    }
    positive, controls, truth = study.source_views(case, qualified)
    assert np.array_equal(np.asarray(positive), edited)
    assert set(controls) == {"original", "tone", "blur"}
    assert np.array_equal(truth > 0, changed)
    edited[0, 0] = 0
    Image.fromarray(edited).save(paths["edited"])
    with pytest.raises(ValueError, match="bytes changed"):
        study.source_views(case, qualified)


def test_failed_predecessor_never_resumes(tmp_path):
    previous = tmp_path / "phase-000"
    previous.mkdir()
    profile = tmp_path / "phase-000-profile"
    profile.mkdir()
    (previous / "completed.json").write_text(json.dumps({"phase": 0}))
    (profile / "profile.json").write_text(
        json.dumps({"child_exit_code": -15, "guard": "recorded_resource_limit", "sample_errors": 0})
    )
    with pytest.raises(ValueError, match="successful resource profile"):
        study.checkpoint_predecessor(tmp_path, {}, 1)


@pytest.mark.parametrize("number", [-1, True, 1.5])
def test_phase_identity_is_an_explicit_integer(tmp_path, number):
    with pytest.raises(ValueError):
        study.checkpoint_predecessor(tmp_path, {}, number)
