from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from app.core.canonical import canonical_hash, canonical_json
from app.evidence.image_packets import packet_hash
from app.reports import engine, image_evidence
from app.reports.public_view import public_manifest
from app.reports.snapshot import dump, load, snapshot_hash
from app.reports.specimens import specimen
from app.storage.objects import ObjectIntegrityError, ObjectNotFound, ObjectStore


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
    monkeypatch.setattr("app.evidence.image_packets_v7.load_packet_set", lambda *a, **k: mutable)
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
