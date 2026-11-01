"""Real exhibit persistence with a bounded synthetic image response; no weights."""
from __future__ import annotations

import hashlib
import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from app.detectors.image import fused
from app.detectors.image import localiser as loc
from app.detectors.image.adapters.localisers import FusedDetector
from app.engine.detector import Profile
from app.engine.exhibits import ExhibitWriter
from app.engine.manifest import load_manifest
from app.engine.models import ManagedModel
from app.storage.objects import ObjectStore


class FakeModel:
    def __init__(self, score=.8, *, gate=None, mask=None):
        self.calls = []
        raw = np.full((512, 512), .1, np.float32)
        raw[120:400, 100:300] = .9
        raw[180:250, 160:220] = .1
        self.prediction = SimpleNamespace(
            raw_mask=raw if mask is None else mask, image_score=score,
            candidate_eligibility=fused.candidate_eligibility(score) if gate is None else gate,
        )

    def predict(self, image):
        assert isinstance(image, Image.Image)
        self.calls.append(np.array(image))
        return self.prediction


@pytest.fixture
def context_factory(tmp_path):
    pixels = np.random.default_rng(15).integers(20, 230, (600, 900, 3), dtype=np.uint8)
    pixels[:100, :150] = 255  # Real deterministic exclusion separate from selection.
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    data = stream.getvalue()
    source_sha = hashlib.sha256(data).hexdigest()
    store = ObjectStore(tmp_path / "objects")

    def make(model, profile=Profile.FULL):
        ctx = SimpleNamespace(
            media=SimpleNamespace(native_pixels=lambda: pixels, original_bytes=lambda: data, sha256=source_sha),
            lease=SimpleNamespace(get=lambda model_id: SimpleNamespace(obj=model, revision=fused.CHECKPOINT_REVISION)),
            device="cpu", profile=profile,
            exhibits=ExhibitWriter("fused-adapter-fixture", store, created_by="loc.fused"),
        )
        return ctx, pixels, store

    return make


@pytest.mark.parametrize("profile", [Profile.FULL, Profile.DEEP])
def test_single_whole_image_screening_and_one_activation_allowance(context_factory, monkeypatch, profile):
    model = FakeModel()
    ctx, pixels, _ = context_factory(model, profile)
    monkeypatch.setattr(loc, "tile_origins", lambda *args: pytest.fail("This check must never create tiles"))
    detector = FusedDetector()
    output = detector.localise(ctx, pixels)
    assert len(model.calls) == 1
    np.testing.assert_array_equal(model.calls[0], pixels)
    assert output.resolution_policy["mode"] == "screening"
    assert output.resolution_policy["input_px"] == 512
    assert output.resolution_policy["grid"] == [900, 600]
    assert output.resolution_policy["tiles"] == output.resolution_policy["overlap_px"] == 0
    assert output.regions
    np.testing.assert_array_equal(output.anomaly_map,
                                  cv2.resize(model.prediction.raw_mask, (900, 600), interpolation=cv2.INTER_LINEAR))
    plan = detector.activation_plan(ctx)
    assert len(plan) == 1 and plan[0].model_id == fused.MODEL_ID and plan[0].max_items == 1
    entry = ManagedModel.from_manifest(load_manifest().model(fused.MODEL_ID))
    assert plan[0].max_items * entry.activation_bytes_per_item == 2048 * 1024**2
    assert detector.spec.requirements.model_ids == (fused.MODEL_ID,)  # Dependency is not leased twice.


def test_failed_gate_preserves_raw_map_conditions_and_integrity(context_factory):
    positive, negative = FakeModel(.8), FakeModel(.2)
    ctx, pixels, _ = context_factory(positive)
    detector = FusedDetector()
    selected = detector.localise(ctx, pixels)
    ctx, pixels, _ = context_factory(negative)
    withheld = detector.localise(ctx, pixels)
    assert selected.regions and not withheld.regions and not withheld.region_stats
    assert selected.candidate_eligibility["eligible"] and not withheld.candidate_eligibility["eligible"]
    np.testing.assert_array_equal(withheld.anomaly_map, selected.anomaly_map)
    np.testing.assert_array_equal(withheld.reliable, selected.reliable)
    assert withheld.reliable.any() and not withheld.reliable.all()
    assert withheld.conditions == selected.conditions and withheld.integrity_score == selected.integrity_score
    assert withheld.integrity_score > .5
    assert len(positive.calls) == len(negative.calls) == 1


@pytest.mark.parametrize("damage", ["false_eligibility", "nan_score", "nan_map", "wrong_map_shape", "out_of_range"])
def test_invalid_response_refused_before_persisting_exhibits(context_factory, damage):
    model = FakeModel()
    if damage == "false_eligibility":
        model.prediction.candidate_eligibility["eligible"] = False
    elif damage == "nan_score":
        model.prediction.candidate_eligibility.update(score=float("nan"), score_hex="nan")
    elif damage == "nan_map":
        model.prediction.raw_mask[5, 5] = np.nan
    elif damage == "wrong_map_shape":
        model.prediction.raw_mask = np.zeros((128, 128), np.float32)
    else:
        model.prediction.raw_mask[5, 5] = 1.1
    ctx, _, _ = context_factory(model)
    with pytest.raises(ValueError):
        FusedDetector().run(ctx)
    assert ctx.exhibits.exhibits == []


@pytest.mark.parametrize("score", [.2, .8])
def test_persisted_source_and_scalar_hash_bind_gate_with_raw_score(context_factory, score):
    model = FakeModel(score)
    ctx, pixels, store = context_factory(model)
    result = FusedDetector().run(ctx)
    assert result.status == "ok" and len(model.calls) == 1
    assert result.aggregate.raw_kind == "score"
    assert result.aggregate.raw > .5  # Gate is not allowed to zero the raw aggregate.
    assert result.metrics["map_max"] > .5 and result.metrics["flagged_share"] > 0
    gate = result.details["localisation"]["candidate_eligibility"]
    heat = next(ex for ex in ctx.exhibits.exhibits if ex.kind == "heatmap")
    assert heat.calibrated is False and gate["role"] == "candidate_selection_only"
    assert gate["source_sha256"] == ctx.media.sha256
    assert gate["scalar_map_sha256"] == heat.sha256
    persisted = store.read_bytes(heat.sha256)
    assert hashlib.sha256(persisted).hexdigest() == gate["scalar_map_sha256"]
    raw = cv2.resize(model.prediction.raw_mask, (900, 600), interpolation=cv2.INTER_LINEAR)
    # The writer encodes the actual float32 values in float64; .9 float32 is
    # just below the half-integer rounding boundary and must encode as 229.
    expected = np.rint(raw.astype(np.float64) * 255).astype(np.uint8)
    with Image.open(io.BytesIO(persisted)) as image:
        np.testing.assert_array_equal(np.asarray(image), expected)
    mask_sha = next(a.sha256 for a in ctx.exhibits.artefacts if a.id == heat.reliability_mask_artefact_id)
    with Image.open(io.BytesIO(store.read_bytes(mask_sha))) as image:
        reliable = np.asarray(image) > 127
    assert gate["eligible"] == (score >= .5)
    regions = [ex for ex in ctx.exhibits.exhibits if ex.kind == "regions"]
    assert bool(regions) == (score >= .5)
    if score >= .5:
        _, recorded_stats = loc.components(expected.astype(np.float64) / 255, reliable, pixels.shape[1], pixels.shape[0])
        assert [unit.raw for unit in result.unit_scores] == [record["mean"] for record in recorded_stats]
    if score < .5:
        assert result.metrics["regions"] == 0
        assert "withheld" in result.reasoning and "not a finding of authenticity" in result.reasoning
    assert "uncalibrated" in result.reasoning and "does not validate any pixel" in result.reasoning
