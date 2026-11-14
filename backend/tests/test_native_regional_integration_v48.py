"""Optional app integration: scope, failure, evidence/export equality and no confidence weight."""
from __future__ import annotations

import copy
import dataclasses
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.detectors.image import regional
from app.detectors.image.adapters.regional import RegionalDetector
from app.engine.detector import Profile
from app.engine.exhibits import ExhibitWriter
from app.engine.manifest import card_gaps, lint, load_manifest
from app.engine.registry import DetectorRegistry, parse_policy
from app.evidence.image_packets import build_packet_set
from app.storage.db import Analysis, Base
from app.storage.objects import ObjectStore


def test_optional_mode_commercial_policy_denial_and_rollback(monkeypatch):
    manifest = load_manifest()
    assert not [r for r in lint(manifest) if r.level == "error"]
    card = manifest.detector(RegionalDetector.spec.id)
    assert not card_gaps(card) and card.default_enabled is False
    assert card.legacy_weight == 0 and card.calibration.method == "none"
    assert card.admission.decision == "pending"
    model = manifest.model(regional.MODEL_ID)
    assert model.load.pickle is False and model.source.files[0].sha256 == regional.CHECKPOINT_SHA256
    registry = DetectorRegistry(manifest, parse_policy("commercial"))
    monkeypatch.setattr(settings, "image_region_mode", "legacy")
    monkeypatch.setattr(settings, "image_region_jpeg_review", False)
    assert not registry.decision(card.id).enabled
    before = registry.decision("loc.splice")
    monkeypatch.setattr(settings, "image_region_mode", "candidate")
    assert registry.decision(card.id).enabled
    assert registry.decision("loc.splice").enabled
    denied = DetectorRegistry(manifest, parse_policy({"preset": "commercial", "deny_detectors": [card.id]}))
    assert not denied.decision(card.id).enabled
    monkeypatch.setattr(settings, "image_region_mode", "legacy")
    assert not registry.decision(card.id).enabled
    assert registry.decision("loc.splice") == before


@pytest.fixture
def context(tmp_path):
    pixels = np.random.default_rng(51).integers(10, 245, (128, 192, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    output = io.BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    store = ObjectStore(tmp_path / "objects")
    source = store.put_bytes(data)
    writer = ExhibitWriter("regional-authored-fixture", store, created_by=RegionalDetector.spec.id)
    fake = SimpleNamespace(raw=np.zeros(pixels.shape[:2], np.float32))

    def predict(image):
        from app.evidence.image_eligibility import area_gate

        _, geometry = regional.prepare(image)
        raw = fake.raw
        gate = area_gate(int((raw > .90).sum()), int(np.ceil(.01 * raw.size)), raw.size)
        return regional.RegionalPrediction(raw, geometry, {"fixture": True}, gate)

    loaded = SimpleNamespace(obj=SimpleNamespace(predict=predict), revision=regional.CHECKPOINT_SHA256)
    ctx = SimpleNamespace(media=SimpleNamespace(native_pixels=lambda: pixels, original_bytes=lambda: data,
                                               sha256=source.sha256), lease=SimpleNamespace(get=lambda _: loaded),
                          profile=Profile.FULL, exhibits=writer, device="cpu")
    return SimpleNamespace(ctx=ctx, fake=fake, store=store, writer=writer, source=source, pixels=pixels)


def test_png_scope_size_and_empty_result_never_assert_original(context, monkeypatch):
    f, detector = context, RegionalDetector()
    monkeypatch.setattr(settings, "image_region_jpeg_review", False)
    assert detector.applicable(f.ctx).ok
    result = detector.run(f.ctx)
    assert result.status == "ok" and result.metrics["regions"] == 0
    assert "does not establish that the file is original" in result.reasoning
    assert result.details["authorship_confidence_contribution"] == 0
    f.ctx.media.original_bytes = lambda: b"jpeg"
    assert not detector.applicable(f.ctx).ok
    f.ctx.media.original_bytes = lambda: b"\x89PNG\r\n\x1a\n"
    f.ctx.media.native_pixels = lambda: f.pixels[:30]
    assert not detector.applicable(f.ctx).ok


@pytest.mark.parametrize(("quality", "sampling", "size", "orientation", "progressive", "supported"), [
    (75, "4:2:0", (768, 512), 1, False, True),
    (75, "4:2:0", (512, 768), 1, False, True),
    (90, "4:2:0", (768, 512), 1, False, False),
    (75, "4:4:4", (768, 512), 1, False, False),
    (75, "4:2:0", (769, 512), 1, False, False),
    (75, "4:2:0", (768, 511), 1, False, False),
    (75, "4:2:0", (768, 512), 6, False, False),
    (75, "4:2:0", (768, 512), 1, True, True),
    (90, "4:2:0", (768, 512), 1, True, False),
    (75, "4:4:4", (768, 512), 1, True, False),
])
def test_jpeg_scope_requires_opt_in_and_evaluated_encoding(monkeypatch, quality, sampling, size, orientation,
                                                         progressive, supported):
    output = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = orientation
    image = Image.new("RGB", size, (90, 100, 110))
    image.save(output, format="JPEG", quality=quality, subsampling=sampling, exif=exif, progressive=progressive)
    media = SimpleNamespace(original_bytes=output.getvalue, native_pixels=lambda: np.asarray(image),
                            info=lambda: {"mode": "RGB", "exif_orientation": orientation})
    ctx = SimpleNamespace(media=media)
    detector = RegionalDetector()
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_region_mode", "candidate")
    monkeypatch.setattr(settings, "image_region_jpeg_review", False)
    assert not detector.applicable(ctx).ok
    monkeypatch.setattr(settings, "image_region_jpeg_review", True)
    assert detector.applicable(ctx).ok is supported
    monkeypatch.setattr(settings, "image_region_mode", "legacy")
    assert detector.applicable(ctx).ok is supported
    monkeypatch.setattr(settings, "image_region_mode", "candidate")
    monkeypatch.setattr(settings, "environment", "prod")
    assert not detector.applicable(ctx).ok


def test_progressive_jpeg_rejects_truncated_scans_and_table_changes():
    from app.detectors.image import jpeg_bitstream
    from app.detectors.image.adapters.regional import evaluated_jpeg

    stream = io.BytesIO()
    Image.new("RGB", (768, 512), (90, 100, 110)).save(stream, "JPEG", quality=75, subsampling=2, progressive=True)
    original = stream.getvalue()
    first_table = original.index(b"\xff\xdb")
    table_bytes = 2 + int.from_bytes(original[first_table + 2:first_table + 4])
    injected = original[:-2] + original[first_table:first_table + table_bytes] + original[-2:]
    frame_at = original.index(b"\xff\xc2")
    frame_size = 2 + int.from_bytes(original[frame_at + 2:frame_at + 4])
    repeated_frame = original[:frame_at] + original[frame_at:frame_at + frame_size] + original[frame_at:]
    filled_end = original[:-2] + b"\xff" + original[-2:]
    assert jpeg_bitstream.read(injected).ijg_quality == 75
    assert jpeg_bitstream.progressive_frame(original)[1] == 10
    assert evaluated_jpeg(original)
    assert evaluated_jpeg(filled_end)
    assert not evaluated_jpeg(injected)
    assert not evaluated_jpeg(repeated_frame)
    assert not evaluated_jpeg(original[:-2])
    assert not evaluated_jpeg(original + b"second frame")


def test_jpeg_scope_rejects_distinct_third_component_quantization():
    from app.detectors.image import jpeg_bitstream
    from app.detectors.image.adapters.regional import evaluated_jpeg

    image = Image.new("RGB", (768, 512), (90, 100, 110))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=75, subsampling="4:2:0")
    original = output.getvalue()
    structure = jpeg_bitstream.parse(original)
    unusual = list(structure.table(1))
    unusual[0] += 1
    output = io.BytesIO()
    image.save(output, format="JPEG", qtables=[list(structure.table(0)), list(structure.table(1)), unusual],
               subsampling="4:2:0")
    altered = output.getvalue()
    assert jpeg_bitstream.parse(altered).ijg_quality == 75
    assert evaluated_jpeg(original)
    assert not evaluated_jpeg(altered)


def test_jpeg_only_toggle_preserves_png_and_splice_selection(context, monkeypatch):
    from app.engine import registry as engine_registry

    manifest = load_manifest()
    registry = DetectorRegistry(manifest, parse_policy("commercial"))
    monkeypatch.setattr(engine_registry, "get_registry", lambda: registry)
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_region_mode", "legacy")
    monkeypatch.setattr(settings, "image_region_jpeg_review", False)
    before = registry.decision("loc.splice")
    monkeypatch.setattr(settings, "image_region_jpeg_review", True)
    assert registry.decision(RegionalDetector.spec.id).enabled
    assert registry.decision("loc.splice") == before
    assert not RegionalDetector().applicable(context.ctx).ok
    monkeypatch.setattr(settings, "image_region_mode", "candidate")
    assert RegionalDetector().applicable(context.ctx).ok
    monkeypatch.setattr(settings, "image_region_mode", "legacy")
    monkeypatch.setattr(settings, "environment", "prod")
    assert not registry.decision(RegionalDetector.spec.id).enabled


@pytest.mark.parametrize("failure", ["wrong_shape", "nonfinite", "exception"])
def test_failed_check_cannot_be_a_negative_authorship_finding(context, failure):
    f = context
    if failure == "wrong_shape":
        f.fake.raw = np.zeros((64, 64), np.float32)
    elif failure == "nonfinite":
        f.fake.raw[:] = np.nan
    else:
        def fail(_image):
            raise RuntimeError("Authored unavailable model")
        f.ctx.lease.get(regional.MODEL_ID).obj.predict = fail
    with pytest.raises((ValueError, RuntimeError)):
        RegionalDetector().run(f.ctx)
    assert not f.writer.exhibits


def test_real_recorded_packets_freeze_same_source_crops_masks_and_export(context, tmp_path):
    from app.api.public_projection import project
    from app.core.canonical import canonical_hash, canonical_json
    from app.engine.stages.image import legacy_signals, legacy_weight
    from app.reports import image_evidence
    from app.reports.public_view import public_manifest
    from app.reports.snapshot import dump, load
    from app.reports.specimens import specimen

    f = context
    f.fake.raw[30:85, 45:125] = .98
    outcome = RegionalDetector().run(f.ctx)
    assert outcome.aggregate.raw_kind == "score"
    aggregate = dataclasses.asdict(outcome.aggregate)
    entry = {"detector_id": RegionalDetector.spec.id, "aggregate": aggregate, "status": "ok",
             "provenance_override": None, "family": "localiser"}
    manifest = load_manifest()
    assert legacy_weight(entry, manifest) == 0
    legacy = legacy_signals([entry], manifest)[0]
    assert legacy.prob is None and legacy.weight == 0 and not legacy.decisive
    result = {"input": {"sha256": f.source.sha256},
              "exhibits": [e.model_dump(mode="json") for e in f.writer.exhibits],
              "signals": [{"id": RegionalDetector.spec.id + ":global", "detector_id": RegionalDetector.spec.id,
                           "display_id": "S01", "status": "ok", "exhibit_ids": outcome.exhibit_ids,
                           "details": outcome.details}],
              "findings": [{"id": "F1", "evidence_ids": [RegionalDetector.spec.id + ":global"]}]}
    engine = create_engine(f"sqlite:///{tmp_path / 'packets.sqlite'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        analysis = Analysis(id="regional-authored-fixture", kind="image", filename="authored.png", content_type="image/png",
                            size_bytes=f.source.size_bytes, sha256=f.source.sha256, status="done", revision=1,
                            result_hash=canonical_hash(result), result_json=canonical_json(result).decode())
        db.add(analysis)
        db.flush()
        f.writer.persist_artefacts(db)
        document = build_packet_set(db, analysis, f.store, result)
        assert len(document["packets"]) == 1
        packet = document["packets"][0]
        assert set(packet["artifacts"]) == {"context", "crop", "overlay", "mask"}
        assert packet["region"] == {"x": 45, "y": 30, "w": 80, "h": 55}
        assert packet["localisation"]["candidate_eligibility"]["eligible"] is True
        snapshot = specimen("image", "standard")
        snapshot = snapshot.model_copy(update={
            "analysis": snapshot.analysis.model_copy(update={"id": analysis.id, "revision": 1}),
            "media": snapshot.media.model_copy(update={"sha256": f.source.sha256}),
            "result": result, "image_evidence": copy.deepcopy(document)})
        frozen = load(dump(snapshot))
        assert frozen.image_evidence == document
        rendered = image_evidence.view(frozen, store=f.store)
        rendered_assets = {im["role"]: im["sha256"] for im in rendered["packets"][0]["images"]}
        assert rendered_assets == {k: v["sha256"] for k, v in packet["artifacts"].items()}
        exported = public_manifest(frozen, "authored test renderer", "authored test template")["image_evidence"]
        assert {a["role"]: a["sha256"] for a in exported["artifacts"] if a["packet_id"]} == rendered_assets
        public = json.dumps(project(result))
        assert regional.MODEL_ID not in public and "loc.native_regional" not in public
        assert regional.CHECKPOINT_SHA256 not in public
    engine.dispose()


def test_worker_configuration_legacy_is_noop_and_candidate_only_changes_threads(monkeypatch):
    import torch

    from app.core.inference_runtime import configure_worker

    observed = []
    monkeypatch.setattr(torch, "set_num_threads", observed.append)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 6)
    monkeypatch.setattr(settings, "image_region_mode", "legacy")
    monkeypatch.setattr(settings, "image_region_jpeg_review", False)
    configure_worker()
    assert observed == []
    monkeypatch.setattr(settings, "image_region_mode", "candidate")
    configure_worker()
    assert observed == [2]
    monkeypatch.setattr(settings, "image_region_mode", "legacy")
    monkeypatch.setattr(settings, "image_region_jpeg_review", True)
    monkeypatch.setattr(settings, "environment", "prod")
    configure_worker()
    assert observed == [2]
    monkeypatch.setattr(settings, "environment", "dev")
    configure_worker()
    assert observed == [2, 2]
