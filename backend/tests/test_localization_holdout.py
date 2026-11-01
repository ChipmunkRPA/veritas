from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import fused_research_eval as evaluator
from scripts import prepare_localization_holdout as holdout


def row(value: str, *, label: int = 1, key: str = "partial/example", mask: bytes | None = b"mask"):
    return {"key": key, "label": label, "image": {"bytes": value.encode()}, "mask": {"bytes": mask}}


def test_selection_offset_counts_unique_partial_rows_before_exclusions():
    rows = [row("ignored", key="full/x"), row("a"), row("a"), row("b"), row("baseline"),
            row("wrong-label", label=0), row("previous"), row("c"), row("d")]
    selected = {holdout.sha(b"previous")}
    chosen = holdout.select_rows(rows, label=1, offset=2, count=2,
                                excluded={holdout.sha(b"baseline")}, selected=selected)
    assert [(index, rank, value["image"]["bytes"]) for index, rank, value, _ in chosen] == [
        (7, 4, b"c"), (8, 5, b"d")]
    assert selected == {holdout.sha(v) for v in (b"previous", b"c", b"d")}


def test_selection_fails_closed_for_missing_masks_or_incomplete_groups():
    with pytest.raises(ValueError, match="missing"):
        holdout.select_rows([row("a", mask=None)], label=1, offset=0, count=1, excluded=set(), selected=set())
    with pytest.raises(ValueError, match="Only 1"):
        holdout.select_rows([row("a"), row("a")], label=1, offset=0, count=2, excluded=set(), selected=set())
    assert len(holdout.select_rows([row("a", label=0, mask=None)], label=0, offset=0, count=1,
                                   excluded=set(), selected=set())) == 1


def baseline(path):
    value = {"dataset": holdout.REPO, "revision": holdout.REVISION,
             "items": [{"family": "sd15", "label": 1, "image_sha256": holdout.sha(b"baseline")}]}
    path.write_text(json.dumps(value))
    return path


def test_plan_predeclares_policy_before_any_remote_count_or_payload(tmp_path, monkeypatch):
    excluded = baseline(tmp_path / "baseline.json")
    before = excluded.read_bytes()
    root = tmp_path / "holdout"
    calls = []

    def metadata_only(family, label, shard):
        assert (root / "evaluation-policy.json").exists()
        assert not list(root.glob("*.image"))
        calls.append((family, label))
        return {"family": family, "label": label, "shard": shard, "expected_label_partial_rows": 40,
                "row_group_compressed_column_bytes": 100}

    monkeypatch.setattr(holdout, "preflight_shard", metadata_only)
    planned = holdout.plan(root, offset=20, count=20, exclude_paths=[excluded])
    assert len(calls) == 10 and len(set(calls)) == 10
    assert planned["expected_samples"] == 200 and planned["image_payload_columns_read"] is False
    assert planned["excluded_image_sha256"] == [holdout.sha(b"baseline")]
    assert planned["evaluation_policy_sha256"] == holdout.sha((root / "evaluation-policy.json").read_bytes())
    policy = json.loads((root / "evaluation-policy.json").read_text())
    assert policy["image_gate_threshold"] == policy["mask_threshold"] == 0.5
    assert policy["production_component_policy"] == holdout.policy()
    assert policy["threshold_tuning"] is False and policy["evaluation_has_started"] is False
    assert excluded.read_bytes() == before
    with pytest.raises(FileExistsError):
        holdout.plan(root, offset=20, count=20, exclude_paths=[excluded])


def test_insufficient_counts_stop_before_payloads(tmp_path, monkeypatch):
    excluded = baseline(tmp_path / "baseline.json")
    monkeypatch.setattr(holdout, "preflight_shard", lambda *args: {"expected_label_partial_rows": 39})
    root = tmp_path / "holdout"
    with pytest.raises(ValueError, match="Insufficient"):
        holdout.plan(root, offset=20, count=20, exclude_paths=[excluded])
    assert (root / "evaluation-policy.json").exists()
    assert not (root / "download-plan.json").exists()
    frozen_policy = (root / "evaluation-policy.json").read_bytes()
    monkeypatch.setattr(holdout, "preflight_shard", lambda *args: {
        "expected_label_partial_rows": 100, "row_group_compressed_column_bytes": 100})
    holdout.plan(root, offset=20, count=20, exclude_paths=[excluded])
    assert (root / "evaluation-policy.json").read_bytes() == frozen_policy


def test_metadata_retry_cannot_change_the_predeclared_policy(tmp_path):
    excluded = baseline(tmp_path / "baseline.json")
    root = tmp_path / "holdout"
    root.mkdir()
    modified = {**holdout.declared_policy(), "mask_threshold": .8}
    holdout.write_new(root / "evaluation-policy.json", modified)
    with pytest.raises(ValueError, match="different evaluation policy"):
        holdout.plan(root, offset=20, count=20, exclude_paths=[excluded])


def test_original_bytes_and_completed_manifest_cannot_be_overwritten(tmp_path):
    path = tmp_path / "original.image"
    holdout.original_bytes(path, b"original")
    holdout.original_bytes(path, b"original")
    with pytest.raises(FileExistsError):
        holdout.original_bytes(path, b"modified")
    (tmp_path / "manifest.json").write_text("frozen")
    with pytest.raises(FileExistsError):
        holdout.prepare(tmp_path)
    assert path.read_bytes() == b"original"
    assert (tmp_path / "manifest.json").read_text() == "frozen"


def test_excluded_manifest_must_use_pinned_revision(tmp_path):
    path = baseline(tmp_path / "baseline.json")
    value = json.loads(path.read_text())
    value["revision"] = "different"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="same pinned"):
        holdout.exclusions([path])


@pytest.fixture
def evaluation_manifest(tmp_path):
    image = tmp_path / "source.image"
    Image.new("RGB", (12, 10), "blue").save(image, format="PNG")
    policy = tmp_path / "evaluation-policy.json"
    holdout.write_new(policy, holdout.declared_policy())
    manifest = tmp_path / "manifest.json"
    holdout.write_new(manifest, {"evaluation_policy_sha256": holdout.sha(policy.read_bytes()),
                                "items": [{"id": "test", "family": "fixture", "label": 0,
                                           "image": image.name, "image_sha256": holdout.sha(image.read_bytes())}]})
    return SimpleNamespace(manifest=manifest, output="evaluation", per_group=1, max_items=None,
                           device="cpu", dtype="float32")


def test_evaluator_reads_selected_manifest_and_preserves_existing_results(evaluation_manifest, monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, *args):
            calls.append(args)
            self.ops = SimpleNamespace(tensors={}, used=set())

        def predict(self, image):
            assert image.size == (12, 10) and image.getpixel((0, 0)) == (0, 0, 255)
            return np.full((512, 512), .25, dtype=np.float32), .1

    monkeypatch.setattr(evaluator, "FusedResearch", FakeModel)
    args = evaluation_manifest
    original = args.manifest.read_bytes()
    evaluator.evaluate(args)
    path = args.manifest.parent / args.output / "results.json"
    saved = path.read_bytes()
    result = json.loads(saved)
    assert result["manifest_sha256"] == holdout.sha(original)
    assert result["evaluation_policy_sha256"] == json.loads(original)["evaluation_policy_sha256"]
    assert result["rows"][0]["flagged_fraction"] == 0
    with pytest.raises(FileExistsError):
        evaluator.evaluate(args)
    assert len(calls) == 1 and path.read_bytes() == saved and args.manifest.read_bytes() == original


def test_evaluator_rejects_changed_predeclared_policy_before_loading(evaluation_manifest, monkeypatch):
    def forbidden(*args):
        pytest.fail("No model may load for a changed policy")

    monkeypatch.setattr(evaluator, "FusedResearch", forbidden)
    args = evaluation_manifest
    (args.manifest.parent / "evaluation-policy.json").write_text("changed")
    with pytest.raises(ValueError, match="policy changed"):
        evaluator.evaluate(args)
    assert not (args.manifest.parent / args.output).exists()
