from __future__ import annotations

import copy
import io
from contextlib import closing
from types import SimpleNamespace

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw
from test_image_packets import _bytes
from test_image_packets import packet_fixture as packet_fixture

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.image import local_jpeg_grid
from app.engine.detector import Modality, Profile, StageOutputs
from app.engine.exhibits import ExhibitWriter
from app.engine.manifest import load_manifest
from app.engine.media import StoredMedia
from app.engine.registry import DetectorRegistry, parse_policy
from app.engine.stages import common, image
from app.evidence import service
from app.evidence.image_packets import packet_hash
from app.reports import engine, image_evidence
from app.reports.public_view import public_manifest
from app.reports.snapshot import dump, load, snapshot_hash
from app.reports.specimens import specimen
from app.storage.artefacts import index_object
from app.storage.objects import ObjectIntegrityError, ObjectNotFound, ObjectStore
from scripts.image_local_grid_transfer import jpeg_copy


def packet_snapshot(tmp_path, profile="standard"):
    store = ObjectStore(tmp_path / "objects")
    source = Image.new("RGB", (180, 120), "#d8ddd5")
    drawing = ImageDraw.Draw(source)
    drawing.rectangle((30, 20, 69, 54), fill="#708565")
    drawing.text((10, 90), "QA specimen", fill="#243026")

    def save(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        sha = store.put_bytes(buf.getvalue()).sha256
        return {"sha256": sha, "width": image.width, "height": image.height, "media_type": "image/png",
                "url": f"/api/v2/evidence-image/{sha}"}

    original = save(source)
    context = source.copy()
    ImageDraw.Draw(context).rectangle((30, 20, 69, 54), outline="#ac6e16", width=2)
    crop = source.crop((20, 10, 80, 65))
    overlay = crop.copy()
    ImageDraw.Draw(overlay).rectangle((10, 10, 49, 44), outline="#ac6e16", width=2)
    snapshot = specimen("image", profile)
    result = copy.deepcopy(snapshot.result)
    result["input"]["sha256"] = original["sha256"]
    result["condition_profile"].update(width=180, height=120)
    packet_set = {
        "schema": "chipmunk.image-evidence/1", "status": "ready", "id": "frozen-example-set",
        "version": "image-evidence-1", "analysis_id": snapshot.analysis.id, "revision": snapshot.analysis.revision,
        "original_result_hash": canonical_hash(result),
        "source": {"sha256": original["sha256"], "width": 180, "height": 120,
                   "coordinate_space": "exif-oriented-original"},
        "packets": [{"id": "E01", "finding_ids": ["F1"], "title": "Compression inspection region",
                     "evidence_type": "anomaly", "semantics": "error_level",
                     "region": {"x": 30, "y": 20, "w": 40, "h": 35},
                     "crop_region": {"x": 20, "y": 10, "w": 60, "h": 55},
                     "observation": "The marked region has higher reconstruction error than its surrounding image.",
                     "interpretation": "This inspection candidate does not establish editing or AI generation.",
                     "limitations": ["Ordinary compression can produce the same pattern."],
                     "measurements": [{"label": "Map maximum", "value": 0.75, "unit": "raw score"}],
                     "signal_refs": [{"id": "check-example:global", "display_id": "S05"}], "exhibit_refs": [],
                     "artifacts": {"context": save(context), "crop": save(crop), "overlay": save(overlay)}}],
        "limitations": ["The measured region does not locate every possible edit."], "omitted_regions": 0,
    }
    packet_set["sha256"] = packet_hash(packet_set)
    snapshot = snapshot.model_copy(update={"result": result, "media": snapshot.media.model_copy(update={"sha256": original["sha256"]}),
                                           "image_evidence": packet_set})
    return snapshot, store


def test_historical_snapshots_keep_canonical_bytes_without_added_key():
    old = specimen("image", "standard")
    historical = canonical_json(old.model_dump(mode="python", exclude={"image_evidence"}))
    assert b'"image_evidence"' not in historical
    assert dump(load(historical)) == historical
    assert dump(old.model_copy(update={"image_evidence": None})) == historical


def test_capture_freezes_set_despite_later_loader_mutation(tmp_path, monkeypatch):
    snapshot, store = packet_snapshot(tmp_path)
    mutable = copy.deepcopy(snapshot.image_evidence)
    monkeypatch.setattr("app.evidence.image_packets_v8.load_packet_set", lambda *a, **k: mutable)
    frozen = image_evidence.capture(None, SimpleNamespace(kind="image"), store=store)
    mutable["packets"][0]["observation"] = "A later version has a different observation."
    assert frozen["packets"][0]["observation"] == snapshot.image_evidence["packets"][0]["observation"]
    loaded = load(dump(snapshot.model_copy(update={"image_evidence": frozen})))
    assert snapshot_hash(loaded) == snapshot_hash(snapshot)
    monkeypatch.setattr("app.evidence.image_packets_v7.load_packet_set", lambda *a, **k: pytest.fail("Render queried mutable state"))
    assert image_evidence.view(loaded, store=store)["sha256"] == frozen["sha256"]


def test_frozen_packet_hash_and_source_binding_are_checked(tmp_path):
    snapshot, store = packet_snapshot(tmp_path)
    changed = copy.deepcopy(snapshot.image_evidence)
    changed["packets"][0]["region"]["x"] += 1
    with pytest.raises(ObjectIntegrityError):
        image_evidence.view(snapshot.model_copy(update={"image_evidence": changed}), store=store)
    wrong_source = snapshot.media.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ObjectIntegrityError):
        image_evidence.view(snapshot.model_copy(update={"media": wrong_source}), store=store)


def test_missing_or_corrupt_frozen_plate_is_not_replaced(tmp_path):
    snapshot, store = packet_snapshot(tmp_path)
    artifact = snapshot.image_evidence["packets"][0]["artifacts"]["crop"]
    path = store.path_for(artifact["sha256"])
    path.unlink()
    with pytest.raises(ObjectNotFound):
        image_evidence.view(snapshot, store=store)
    path.write_bytes(b"a different image cannot stand in for the crop")
    with pytest.raises(ObjectIntegrityError):
        image_evidence.view(snapshot, store=store)


def test_measured_mask_is_frozen_rendered_and_committed(tmp_path, monkeypatch):
    snapshot, store = packet_snapshot(tmp_path)
    evidence = copy.deepcopy(snapshot.image_evidence)
    evidence['version'] = 'image-evidence-2'
    packet = evidence['packets'][0]
    packet['evidence_type'] = 'localisation'
    mask = Image.new('RGB', (18, 12), 'black')
    ImageDraw.Draw(mask).rectangle((3, 2, 6, 5), fill='white')
    data = io.BytesIO()
    mask.save(data, format='PNG')
    sha = store.put_bytes(data.getvalue()).sha256
    packet['artifacts']['mask'] = {'sha256': sha, 'media_type': 'image/png', 'width': 18, 'height': 12,
                                    'url': f'/api/v2/evidence-image/{sha}'}
    packet['localisation'] = {'grid': [18, 12], 'threshold': 0.5, 'region_area_fraction': round(16 / 216, 6)}
    evidence['sha256'] = packet_hash(evidence)
    snapshot = snapshot.model_copy(update={'image_evidence': evidence})
    monkeypatch.setattr(image_evidence, 'default_store', lambda: store)
    html = engine.render_html(snapshot, embed_fonts=False)
    assert 'evidence-plates-four' in html
    assert 'White mask cells mark this selected region' in html
    view = image_evidence.view(snapshot, store=store)
    assert view['packets'][0]['images'][3]['role'] == 'mask'
    assert view['packets'][0]['images'][3]['sha256'] == sha
    commitment = public_manifest(snapshot, 'weasyprint test', engine.TEMPLATE_SET)['image_evidence']
    assert {'packet_id': 'E01', 'role': 'mask', 'sha256': sha} in commitment['artifacts']
    assert load(dump(snapshot)).image_evidence == evidence
    store.path_for(sha).unlink()
    with pytest.raises(ObjectNotFound):
        image_evidence.view(snapshot, store=store)


@pytest.mark.parametrize(("basis", "explanation"), [
    ("native_patch_alignment", "The match aligns patches on the original pixel grid."),
    ("geometric_features", "The match uses corresponding features in both areas."),
])
@pytest.mark.parametrize(("affine_review", "version"), [
    (False, "image-evidence-12"),
    (True, "image-evidence-14"),
])
def test_summary_pdf_and_manifest_present_non_mirrored_pair(packet_fixture, monkeypatch, basis, explanation, affine_review, version):
    db, store, analysis, _, _ = packet_fixture
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_embedded_pair_review", False)
    monkeypatch.setattr(settings, "image_small_png_copy_review", False)
    monkeypatch.setattr(settings, "image_jpeg_block_copy_review", False)
    monkeypatch.setattr(settings, "image_jpeg_offgrid_copy_review", False)
    monkeypatch.setattr(settings, "image_affine_copy_review", affine_review)
    if basis == "geometric_features":
        monkeypatch.setattr("app.detectors.image.copy_similarity.analyse_translation", lambda _: ("no_pair", []))
    pixels = np.random.default_rng(812).integers(0, 256, (512, 768, 3), dtype=np.uint8)
    pixels = cv2.GaussianBlur(pixels, (3, 3), 0.65)
    pixels[61:189, 31:159] = pixels[301:429, 527:655]
    source = store.put_bytes(_bytes(Image.fromarray(pixels), format="PNG"))
    analysis.sha256, analysis.size_bytes = source.sha256, source.size_bytes
    original = specimen("image", "summary")
    result = copy.deepcopy(original.result)
    result.update(exhibits=[], signals=[], findings=[])
    result["input"]["sha256"] = source.sha256
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    document = service.persist_packet_set(db, analysis, store, result=result)
    assert document["version"] == version
    assert document["copy_similarity"]["pairs"] == []
    assert document["packets"] == []
    assert document["copy_translation"]["basis"] == basis
    (pair,) = document["copy_translation"]["pairs"]
    snapshot = original.model_copy(update={
        "result": result,
        "image_evidence": document,
        "analysis": original.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
        "media": original.media.model_copy(update={"sha256": source.sha256}),
    })
    monkeypatch.setattr(image_evidence, "default_store", lambda: store)
    assert load(dump(snapshot)).image_evidence == document
    commitment = public_manifest(snapshot, "weasyprint test", engine.TEMPLATE_SET)["image_evidence"]
    assert commitment["copy_translation"] == document["copy_translation"]
    translation_artifacts = [asset for asset in commitment["artifacts"] if asset["packet_id"] == "T01"]
    assert translation_artifacts == [
        {"packet_id": "T01", "role": role, "sha256": asset["sha256"]}
        for role, asset in pair["artifacts"].items()
    ]
    if not affine_review:
        assert commitment["artifacts"] == translation_artifacts
    rendered = engine.render_pdf(snapshot)
    text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert "T01 · Non-mirrored pair" in text
    assert explanation in text
    assert "This check carries zero weight in AI authorship confidence." in text
    expected_pixels = set()
    for role, label in (("area_a", "Area A"), ("area_b", "Area B")):
        region = pair["regions"][role]
        assert f"{label} · x {region['x']}, y {region['y']}, {region['w']} × {region['h']} px" in text
        with Image.open(io.BytesIO(store.read_bytes(pair["artifacts"][role]["sha256"]))) as crop:
            expected_pixels.add((crop.size, crop.convert("RGB").tobytes()))
    with pymupdf.open(stream=rendered.pdf, filetype="pdf") as pdf:
        actual_pixels = {
            ((width, height), pymupdf.Pixmap(pdf, xref).samples)
            for page in pdf
            for xref, _, width, height, *_ in page.get_images()
            if (width, height) in {size for size, _ in expected_pixels}
        }
    assert expected_pixels <= actual_pixels
    redacted = snapshot.model_copy(update={"report": snapshot.report.model_copy(update={"redacted": True})})
    (redacted_pair,) = image_evidence.view(redacted, store=store)["copy_translation"]["pairs"]
    assert all(image["uri"] is None for image in redacted_pair["images"])
    assert redacted_pair["regions"] == pair["regions"]


def test_png_prior_grid_uses_original_crops_and_is_outside_ai_confidence(packet_fixture, monkeypatch):
    db, store, analysis, _, _ = packet_fixture
    registry = DetectorRegistry(load_manifest(), parse_policy("commercial"))
    detector_id = local_jpeg_grid.DETECTOR_ID
    assert detector_id in registry.enabled(Modality.IMAGE, Profile.QUICK)
    detector = registry.instance(detector_id)
    rng = np.random.default_rng(817)
    a, b = [cv2.GaussianBlur(rng.integers(0, 256, (512, 768, 3), dtype=np.uint8), (5, 5), 1.1)
            for _ in range(2)]
    control = jpeg_copy(a, 67, (3, 5))
    pixels = jpeg_copy(a, 67)
    pixels[155:381, 239:515] = jpeg_copy(b, 59, (3, 5))[155:381, 239:515]
    writer = ExhibitWriter(analysis.id, store, created_by=detector_id)

    def run(native, image_format):
        obj = store.put_bytes(_bytes(Image.fromarray(native), format=image_format))
        with closing(StoredMedia(obj.sha256, "image", store)) as media:
            header = {"format": image_format, "mode": "RGB", "width": 768, "height": 512, "n_frames": 1}
            return obj, detector.run(SimpleNamespace(media=media, outputs=StageOutputs(probe=header), exhibits=writer))

    _, jpeg = run(pixels, "JPEG")
    assert jpeg.status == "not_applicable"
    _, uniform = run(control, "PNG")
    assert uniform.status == "ok" and uniform.aggregate is None and not writer.exhibits
    source, output = run(pixels, "PNG")
    assert output.status == "ok" and output.aggregate.raw_kind == "score"
    row = common.detector_entry(detector.spec, output, None)
    context = SimpleNamespace(state={"registry": registry}, outputs={"detect": {"detectors": [row]}})
    fused = image.legacy_fusion(context)
    assert fused["decision"]["code"] == "not_assessable"
    assert fused["signals"][0]["effective_weight"] == 0
    assert fused["signals"][0]["calibrated_probability"] is None

    analysis.sha256, analysis.size_bytes = source.sha256, source.size_bytes
    for asset in writer.artefacts:
        index_object(db, asset.sha256, asset.size_bytes, media_type=asset.media_type, role=asset.role,
                     created_by=asset.created_by, analysis_id=analysis.id, artefact_id=asset.id)
    original = specimen("image", "summary")
    result = copy.deepcopy(original.result)
    result["input"]["sha256"] = source.sha256
    result.update(decision=fused["decision"], signals=fused["signals"],
                  exhibits=[item.model_dump(mode="json") for item in writer.exhibits])
    context.outputs.update(fuse=fused, decode={"width": 768, "height": 512})
    context.outputs["detect"]["exhibits"] = result["exhibits"]
    result["findings"] = image.synthesise(context)["findings"]
    finding, = [f for f in result["findings"] if f["title"] == "Local compression-grid difference"]
    assert finding["direction"] == "neutral" and finding["probability"] is None
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    evidence = service.persist_packet_set(db, analysis, store, result=result)
    packet, = evidence["packets"]
    assert packet["finding_ids"] == [finding["id"]]
    assert packet["region"] == {"x": 248, "y": 152, "w": 256, "h": 224}
    crop = packet["crop_region"]
    with Image.open(io.BytesIO(store.read_bytes(packet["artifacts"]["crop"]["sha256"]))) as stored:
        np.testing.assert_array_equal(np.asarray(stored), pixels[crop["y"]:crop["y"]+crop["h"], crop["x"]:crop["x"]+crop["w"]])
    snapshot = original.model_copy(update={
        "result": result, "image_evidence": evidence,
        "analysis": original.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
        "media": original.media.model_copy(update={"sha256": source.sha256}),
    })
    monkeypatch.setattr(image_evidence, "default_store", lambda: store)
    manifest = public_manifest(snapshot, "weasyprint test", engine.TEMPLATE_SET)["image_evidence"]
    assert {"packet_id": packet["id"], "role": "crop", "sha256": packet["artifacts"]["crop"]["sha256"]} in manifest["artifacts"]
    pdf = engine.render_pdf(snapshot)
    report = " ".join(engine.pdf_text(pdf.pdf).split())
    assert "Local compression-grid difference" in report
    assert "Ordinary cropping, separate saves or compositing can produce this pattern." in report
    assert "carries zero weight in AI confidence" in report
    assert detector_id not in report


@pytest.mark.parametrize("profile", ["standard", "forensic", "summary", "newsroom"])
def test_all_profiles_use_same_packet_and_redaction_policy(tmp_path, monkeypatch, profile):
    snapshot, store = packet_snapshot(tmp_path, profile)
    monkeypatch.setattr(image_evidence, "default_store", lambda: store)
    before = snapshot_hash(snapshot)
    html = engine.render_html(snapshot, embed_fonts=False)
    assert "E01" in html
    assert "Compression inspection region" in html
    assert "does not establish editing or AI generation" in html
    assert "Map maximum" in html
    if profile == "newsroom":
        assert "Source images and crops are withheld" in html
        assert '<img src="data:image/png;base64,' not in html
    else:
        assert '<img src="data:image/png;base64,' in html
        assert "Original crop" in html and "Diagnostic overlay" in html
    commitment = public_manifest(snapshot, "weasyprint test", engine.TEMPLATE_SET)["image_evidence"]
    assert commitment["sha256"] == snapshot.image_evidence["sha256"]
    assert commitment["source_sha256"] == snapshot.media.sha256
    assert {x["sha256"] for x in commitment["artifacts"]} == {
        x["sha256"] for x in snapshot.image_evidence["packets"][0]["artifacts"].values()}
    assert snapshot_hash(snapshot) == before
