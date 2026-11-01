"""CPU-only checks for the fixed loader and raw-measurement/selection boundary.

No test loads the multi-GiB checkpoint or executes a remote/source-repo model.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from app.detectors.image import fused
from app.detectors.image.adapters import fused_loader
from app.engine.manifest import card_gaps, lint, load_manifest
from app.engine.models import ModelUnavailable
from app.engine.registry import DetectorRegistry, parse_policy


def test_fixed_gate_keeps_exact_side_without_becoming_a_probability():
    below = np.nextafter(.5, 0)
    low, boundary = fused.candidate_eligibility(below), fused.candidate_eligibility(.5)
    assert not low["eligible"] and boundary["eligible"]
    assert float.fromhex(low["score_hex"]) == below
    assert round(low["score"], 10) == .5  # Exact hex preserves this otherwise rounded boundary.
    assert low["score_kind"] == "uncalibrated_response"
    assert low["role"] == "candidate_selection_only" and low["raw_mask_retained"]
    assert low["scope"] == "whole_image" and low["status"] == "evaluated"
    for invalid in (float("nan"), float("inf"), -.01, 1.01, True):
        with pytest.raises(ValueError):
            fused.candidate_eligibility(invalid)


@pytest.mark.parametrize(("device", "dtype"), [("cuda", "float32"), ("cpu:1", "float32"),
                                                ("cpu", "float16"), ("cpu", torch.bfloat16)])
def test_unreviewed_execution_targets_fail_before_file_access(device, dtype, monkeypatch):
    monkeypatch.setattr(fused, "_check_hash", lambda *args: pytest.fail("Weights must not be read"))
    with pytest.raises(ValueError):
        fused.load_model(Path("absent"), Path("absent-backbone"), device, dtype)


def test_unavailable_mps_does_not_silently_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        fused._execution_target("mps", "float32")


def test_corrupt_backbone_is_rejected_before_any_pickle_inspection(tmp_path, monkeypatch):
    checkpoint, backbone = tmp_path / "model.pth", tmp_path / "backbone.safetensors"
    checkpoint.write_bytes(b"verified fixture")
    backbone.write_bytes(b"changed")
    monkeypatch.setattr(fused, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(torch.serialization, "get_unsafe_globals_in_checkpoint",
                        lambda *args: pytest.fail("Neither pickle inspection nor deserialization is allowed"))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Must check both file hashes first"))
    with pytest.raises(ValueError, match="SHA-256"):
        fused.load_model(checkpoint, backbone, "cpu")


def test_pickle_globals_refused_without_deserialization(monkeypatch):
    monkeypatch.setattr(fused, "_check_hash", lambda *args: None)
    monkeypatch.setattr(torch.serialization, "get_unsafe_globals_in_checkpoint", lambda *args: ["bad.Class"])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Unsafe globals must not be loaded"))
    with pytest.raises(ValueError, match="unsupported pickle"):
        fused.load_model(Path("fixture"), Path("fixture-backbone"), "cpu")


@pytest.fixture
def tiny_saved(monkeypatch):
    # Preserve collection cardinality while replacing large tensors with scalars.
    state = {f"fixture.{i}": torch.tensor([float(i)]) for i in range(879)}
    schema = [[key, list(tensor.shape), str(tensor.dtype)] for key, tensor in sorted(state.items())]
    monkeypatch.setattr(fused, "STATE_SCHEMA_SHA256",
                        hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest())
    return {"cfg": {"model": copy.deepcopy(fused.EXPECTED_CONFIG)}, "state_dict": state}


@pytest.mark.parametrize("damage", ["target", "extra", "boolean_dimension", "shape", "dtype", "missing", "nontensor"])
def test_checkpoint_config_and_tensor_schema_are_fixed(tiny_saved, damage):
    assert fused._validated_state(tiny_saved) is tiny_saved["state_dict"]
    if damage == "target":
        tiny_saved["cfg"]["model"]["_target_"] = "untrusted.ArbitraryCode"
    elif damage == "extra":
        tiny_saved["cfg"]["model"]["download_remote_weights"] = True
    elif damage == "boolean_dimension":
        tiny_saved["cfg"]["model"]["seg_bridge"] = 1
    elif damage == "shape":
        tiny_saved["state_dict"]["fixture.0"] = torch.zeros(2)
    elif damage == "dtype":
        tiny_saved["state_dict"]["fixture.0"] = torch.zeros(1, dtype=torch.float16)
    elif damage == "missing":
        tiny_saved["state_dict"].pop("fixture.0")
    else:
        tiny_saved["state_dict"]["fixture.0"] = "not a tensor"
    with pytest.raises(ValueError):
        fused._validated_state(tiny_saved)


def test_local_safe_loader_meta_build_and_strict_assignment(tmp_path, monkeypatch, tiny_saved):
    import timm
    from safetensors import torch as safetensors_torch

    checkpoint, backbone_path = tmp_path / "model.pth", tmp_path / "backbone.safetensors"
    torch.save(tiny_saved, checkpoint)
    backbone_path.write_bytes(b"small fixture")
    for key, path in (("CHECKPOINT_SHA256", checkpoint), ("BACKBONE_SHA256", backbone_path)):
        monkeypatch.setattr(fused, key, hashlib.sha256(path.read_bytes()).hexdigest())
    calls = []
    real_load = torch.load

    def safe_load(path, **kwargs):
        calls.append(("pickle", kwargs))
        return real_load(path, **kwargs)

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1))

        def load_state_dict(self, state, **kwargs):
            calls.append(("assignment", sorted(state), kwargs))
            return super().load_state_dict(state, **kwargs)

    def create(architecture, **kwargs):
        assert architecture == fused.BACKBONE_ARCHITECTURE
        assert kwargs == {"pretrained": False, "num_classes": 0}
        model = TinyBackbone()
        assert model.weight.device.type == "meta"
        return model

    def safe_tensors(path, **kwargs):
        assert path == str(backbone_path) and kwargs == {"device": "cpu"}
        return {"weight": torch.ones(1), "head.fc.weight": torch.zeros(1), "head.fc.bias": torch.zeros(1)}

    monkeypatch.setattr(torch, "load", safe_load)
    monkeypatch.setattr(timm, "create_model", create)
    monkeypatch.setattr(safetensors_torch, "load_file", safe_tensors)
    model = fused.load_model(checkpoint, backbone_path, "cpu")
    assert calls == [("pickle", {"map_location": "cpu", "weights_only": True, "mmap": True}),
                     ("assignment", ["weight"], {"strict": True, "assign": True})]
    assert model.device.type == "cpu" and model.backbone.weight.dtype == torch.float32
    assert not model.backbone.training and not model.backbone.weight.requires_grad


class RecordingOps:
    def __init__(self, mask=None, score=.1):
        self.mask = torch.full((1, 1, 512, 512), .9) if mask is None else mask
        self.score = torch.tensor([score])

    def p(self, key):
        values = {"clip_mean": (.1, .2, .3), "clip_std": (.4, .5, .6)}
        return torch.tensor(values[key])[None, :, None, None]

    def forensic_features(self, x):
        self.forensic_input = x
        return x

    def fusion_map(self, x):
        return x

    def finish(self, forensic, semantic):
        return self.mask, self.score


def test_predict_preserves_raw_map_and_stored_orientation_when_gate_fails():
    class RecordingBackbone:
        def forward_features(self, x):
            self.input = x
            return x

    pixels = np.zeros((3, 7, 3), np.uint8)
    pixels[:, :3] = [240, 130, 20]
    image = Image.fromarray(pixels)
    image.getexif()[274] = 6  # Already-oriented inputs must not be transformed again.
    ops, backbone = RecordingOps(), RecordingBackbone()
    result = fused.FusedModel(ops, backbone, torch.device("cpu")).predict(image)
    assert result.source_size == (7, 3) and not result.candidate_eligibility["eligible"]
    np.testing.assert_array_equal(result.raw_mask, np.full((512, 512), .9, np.float32))
    assert not result.raw_mask.flags.writeable
    resized = np.array(image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    semantic = backbone.input[0].permute(1, 2, 0).numpy()
    forensic = ops.forensic_input[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(semantic, (resized - [.1, .2, .3]) / [.4, .5, .6], atol=3e-7)
    np.testing.assert_allclose(forensic, (resized - [.485, .456, .406]) / [.229, .224, .225], atol=4e-7)


def test_predict_rejects_batches_and_invalid_outputs():
    image = Image.new("RGB", (4, 3))
    backbone = SimpleNamespace(forward_features=lambda x: x)
    model = fused.FusedModel(RecordingOps(), backbone, torch.device("cpu"))
    with pytest.raises(ValueError, match="exactly one"):
        model.predict([image, image])
    for mask in (torch.ones(2, 1, 512, 512), torch.full((1, 1, 512, 512), float("nan")),
                 torch.full((1, 1, 512, 512), 1.01)):
        model.ops = RecordingOps(mask)
        with pytest.raises(RuntimeError):
            model.predict(image)


def test_production_lattice_coordinates_roundtrip():
    image = torch.arange(2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    tokens = fused.lattice_tokens(image, 4)
    assert torch.equal(tokens[1, :, 2], image[0, 2, ::4, 1::4].flatten())
    assert torch.equal(fused.lattice_image(tokens, 4, 16), image)


def test_manifest_disabled_unfitted_and_composite_loader_verifies_both(monkeypatch):
    from app.engine import fetchers

    manifest = load_manifest()
    assert not [issue for issue in lint(manifest) if issue.level == "error"]
    detector = manifest.detector("loc.fused")
    assert detector is not None and not detector.default_enabled
    assert detector.models == [fused.MODEL_ID] and detector.legacy_weight == 0
    assert detector.calibration.method == "none" and detector.admission.decision == "pending"
    assert not card_gaps(detector)
    registry = DetectorRegistry(manifest, parse_policy("commercial"))
    assert not registry.decision("loc.fused").enabled
    assert set(registry.required_models(detector)) == {fused.MODEL_ID, fused.BACKBONE_MODEL_ID}
    primary = manifest.model(fused.MODEL_ID)
    assert primary.fp16_safe is False and primary.footprint.fp32_mb >= 3584
    verified = []
    loaded = []
    monkeypatch.setattr(fused_loader, "_entry", manifest.model)
    monkeypatch.setattr(fused_loader, "_file", lambda entry, name: Path(entry.source.path) / name)
    monkeypatch.setattr(fetchers, "ensure_verified", lambda entry: verified.append(entry.id))
    monkeypatch.setattr(fused, "load_model", lambda *args: loaded.append(args) or "composite")
    result = fused_loader.load_fused(fused.MODEL_ID, "cpu", "float32")
    assert result.obj == "composite" and result.revision == fused.CHECKPOINT_REVISION
    assert verified == [fused.MODEL_ID, fused.BACKBONE_MODEL_ID]
    assert loaded == [(Path("weights/fused/fused_opensdi.pth"), Path("weights/fused/convnext.safetensors"),
                       "cpu", "float32")]
    primary.source.files[0].sha256 = "0" * 64
    with pytest.raises(ModelUnavailable, match="unexpected checkpoint"):
        fused_loader.load_fused(fused.MODEL_ID, "cpu", "float32")
