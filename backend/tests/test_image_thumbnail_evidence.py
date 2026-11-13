from __future__ import annotations

import io
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from app.detectors.image.adapters.provenance import MetadataConsistencyDetector
from app.engine.exhibits import ExhibitWriter
from app.engine.stages import image as image_stage
from app.evidence import image_thumbnail_evidence as evidence
from app.provenance import image_thumbnail_localization as localization
from app.reports import engine, figures
from app.reports.specimens import specimen
from app.storage.objects import ObjectStore
from scripts.image_editor_preview_transfer import adversarial, attach, segment, without_exif
from scripts.image_thumbnail_localization_transfer import color_patch, jpeg, preview_from, thumbnail
from scripts.image_thumbnail_mirror_transfer import OPERATIONS as MIRRORS
from scripts.image_thumbnail_orientation_transfer import encode as rotated_carrier
from scripts.review_image_editor_preview import verify as verify_editor_preview


def _pair(orientation=1, *, display_preview=False):
    rng = np.random.default_rng(1)
    colors = rng.integers(0, 256, (15, 20, 3), dtype=np.uint8)
    smooth = cv2.resize(colors, (512, 384), interpolation=cv2.INTER_CUBIC).astype(np.float32) * .60 + 52
    before = np.clip(np.rint(smooth + rng.normal(0, 3, smooth.shape)), 0, 255).astype(np.uint8)
    earlier = Image.fromarray(before)
    changed = Image.fromarray(color_patch(before, 1))
    if orientation != 1:
        inverse = Image.Transpose.ROTATE_90 if orientation == 6 else MIRRORS[orientation]
        before_raw, after_raw = (image.transpose(inverse) for image in (earlier, changed))
        embedded = thumbnail(earlier if display_preview else before_raw)
        return rotated_carrier(after_raw, orientation, embedded), rotated_carrier(before_raw, orientation, embedded)
    embedded = thumbnail(earlier)
    return jpeg(changed, preview=embedded), jpeg(earlier, preview=embedded)


def _run(store, source, monkeypatch):
    writer = ExhibitWriter("thumbnail-test", store)
    media = SimpleNamespace(original_bytes=lambda: source)
    detector = MetadataConsistencyDetector()
    outcome = detector.run(SimpleNamespace(media=media, exhibits=writer.bind(detector.spec.id)))
    source_sha = store.put_bytes(source).sha256
    original_exhibits = [item.model_dump(mode="json") for item in writer.exhibits]
    events = []
    monkeypatch.setattr(image_stage.settings, "image_whole_review_python", None)
    monkeypatch.setattr(image_stage, "_legacy_artifacts", lambda _: events.append("legacy") or {})
    monkeypatch.setattr(image_stage.common, "index_artefacts", lambda *_args, **_kw: events.append("preview"))
    context = SimpleNamespace(analysis_id="thumbnail-test", exhibits=writer, state={"media": media},
                              sessions=lambda: nullcontext(SimpleNamespace(commit=lambda: None)),
                              outputs={"ingest": {"mime": "image/jpeg", "sha256": source_sha}})
    rendered = image_stage.render(context)
    assert events == (["legacy", "preview"] if rendered["artefacts"] else ["legacy"])
    result = {"exhibits": original_exhibits + rendered["exhibits"], "input": {"sha256": source_sha}}
    return outcome, result


@pytest.mark.parametrize("profile", ["standard", "summary"])
@pytest.mark.parametrize(("orientation", "display_preview"), [(1, False), (6, False), (2, False), (4, True), (5, False), (7, True)])
def test_report_and_export_keep_source_crops_with_numeric_coordinates(tmp_path, monkeypatch, profile, orientation, display_preview):
    altered, clean = _pair(orientation, display_preview=display_preview)
    store = ObjectStore(tmp_path / "objects")
    clean_outcome, clean_result = _run(store, clean, monkeypatch)
    assert clean_outcome.status == "ok"
    assert evidence.from_result(clean_result) is None

    positive, result = _run(store, altered, monkeypatch)
    published = evidence.from_result(result)
    assert published["source_sha256"] == result["input"]["sha256"]
    assert published["main_exif_orientation"] == orientation
    assert published["preview_alignment_orientation"] == (orientation if orientation != 1 and not display_preview else 0)
    assert published["authorship_confidence_contribution"] == 0
    with monkeypatch.context() as before:
        before.setattr(localization, "measure", lambda *_args, **_kwargs: ({"status": "unsupported_carrier"}, None))
        baseline, _ = _run(store, altered, before)
    assert positive.aggregate == baseline.aggregate and positive.unit_scores == baseline.unit_scores
    assert any(item["title"] == "Embedded thumbnail vs main image" for item in result["exhibits"])
    _, measured = localization.measure(altered, minimum_component_fraction=localization.CONFIRMATION_COMPONENT_FRACTION)
    for region in published["regions"]:
        for role, source in (("main_image", altered), ("embedded_preview", preview_from(altered))):
            bounds = region[role]
            x, y, w, h = (bounds[key] for key in evidence.BOX_KEYS)
            assert min(w, h) > 0 and min(x, y) >= 0
            crop = region["crops"][role]
            with Image.open(io.BytesIO(source)) as original, Image.open(io.BytesIO(store.read_bytes(crop["sha256"]))) as image:
                axes = ImageOps.exif_transpose(original) if role == "main_image" else original
                expected = axes.convert("RGB").crop((x, y, x + w, y + h))
                assert (crop["display_width"], crop["display_height"]) == image.size
                assert not crop["display_reduced"]
                assert image.size == expected.size
                np.testing.assert_array_equal(np.asarray(image), np.asarray(expected))
        box, footprint = region["footprint_grid"], region["crops"]["measured_footprint"]
        x, y, w, h = (box[key] for key in evidence.BOX_KEYS)
        assert box["coordinate_space"] == "aligned-preview-grid"
        with Image.open(io.BytesIO(store.read_bytes(footprint["sha256"]))) as image:
            assert image.size == (w, h)
            white = np.asarray(image.convert("L")) == 255
            assert white.sum() == region["marked_preview_pixels"]
            np.testing.assert_array_equal(white, measured[0][y:y + h, x:x + w])

    if orientation == 1:
        import copy

        old = copy.deepcopy(result)
        table = next(item for item in old["exhibits"] if item["title"] == evidence.TABLE)
        table["columns"] = table["columns"][:-2]
        table["rows"] = [row[:-2] for row in table["rows"]]
        compatible = copy.deepcopy(published)
        for region in compatible["regions"]:
            region.pop("footprint_grid")
            region["crops"].pop("measured_footprint")
        assert evidence.from_result(old) == compatible

    # A resumed run starts with a fresh writer and cached provenance exhibits.
    # Render must keep its exhibit IDs without reusing the metadata namespace.
    fresh = ExhibitWriter("thumbnail-test", store).bind(evidence.CREATOR)
    resumed = set(evidence.add_exhibits(altered, fresh))
    metadata_ids = {item["id"] for item in result["exhibits"] if item["created_by"] == MetadataConsistencyDetector.spec.id}
    assert metadata_ids and resumed.isdisjoint(metadata_ids)
    assert resumed == {item["id"] for item in result["exhibits"] if item["created_by"] == evidence.CREATOR}

    example = specimen("image", profile=profile)
    snapshot = example.model_copy(update={"result": {**example.result, **result}, "exhibit_images": {
        crop["exhibit_id"]: crop["sha256"] for region in published["regions"] for crop in region["crops"].values()},
        "media": example.media.model_copy(update={"sha256": result["input"]["sha256"]})})
    monkeypatch.setattr(figures, "default_store", lambda: store)
    pdf = engine.render_pdf(snapshot)
    text = " ".join(engine.pdf_text(pdf.pdf).split())
    assert "Ordinary editing and image processing" in text
    assert "White cells mark measured differences on the aligned preview grid" in text
    first = published["regions"][0]
    assert "Embedded preview differences" in text
    for role in ("main_image", "embedded_preview", "footprint_grid"):
        box = first[role]
        assert f"x {box['x']}, y {box['y']}, {box['w']} × {box['h']} px" in text


@pytest.mark.parametrize("error", [OSError("malformed embedded JPEG"), Image.DecompressionBombError("preview too large")])
def test_optional_decoder_error_preserves_original_metadata_score(tmp_path, monkeypatch, error):
    altered, _ = _pair()
    store = ObjectStore(tmp_path / "objects")
    original, _ = _run(store, altered, monkeypatch)

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(localization, "measure", fail)
    degraded, result = _run(store, altered, monkeypatch)
    assert degraded.status == original.status
    assert degraded.aggregate == original.aggregate
    assert degraded.metrics == original.metrics
    assert degraded.observed == original.observed
    assert evidence.from_result(result) is None
    assert any(item["title"] == "Embedded thumbnail vs main image" for item in result["exhibits"])


def test_editor_preview_preserves_source_assets_and_refuses_ambiguous_resources(tmp_path, monkeypatch):
    altered, clean = _pair()
    retained = preview_from(altered)
    raw = without_exif(altered)
    submitted = attach(raw, segment(retained, name=b"preview"))
    store = ObjectStore(tmp_path / "objects")
    outcome, result = _run(store, submitted, monkeypatch)
    baseline, _ = _run(store, raw, monkeypatch)
    assert outcome.aggregate == baseline.aggregate and outcome.unit_scores == baseline.unit_scores
    assert preview_from(submitted) is None
    published = evidence.from_result(result)
    assert published["preview_carrier"] == "jpeg_editor_resource"
    assert published["authorship_confidence_contribution"] == 0
    assert verify_editor_preview(submitted, published, store.read_bytes) == {
        "regions": 1, "source_crops": 2, "footprints": 1,
    }
    import copy

    moved = copy.deepcopy(published)
    moved["regions"][0]["embedded_preview"]["x"] += 1
    with pytest.raises(ValueError, match="thumbnail region differs"):
        verify_editor_preview(submitted, moved, store.read_bytes)
    with pytest.raises(ValueError, match="Submitted source differs"):
        verify_editor_preview(raw, published, store.read_bytes)
    with pytest.raises(ValueError, match="Saved PNG differs from its commitment"):
        verify_editor_preview(submitted, published, lambda pin: store.read_bytes(pin) + b"altered")
    for malformed in adversarial(raw, retained).values():
        _, negative = _run(store, malformed, monkeypatch)
        assert evidence.from_result(negative) is None
    _, negative = _run(store, attach(without_exif(clean), segment(retained)), monkeypatch)
    assert evidence.from_result(negative) is None
    both = attach(altered, segment(retained))
    _, existing = _run(store, both, monkeypatch)
    assert "preview_carrier" not in evidence.from_result(existing)

    report = specimen("image", profile="standard")
    frozen = report.model_copy(update={"result": {**report.result, **result}, "exhibit_images": {
        crop["exhibit_id"]: crop["sha256"] for region in published["regions"] for crop in region["crops"].values()},
        "media": report.media.model_copy(update={"sha256": result["input"]["sha256"]})})
    monkeypatch.setattr(figures, "default_store", lambda: store)
    text = " ".join(engine.pdf_text(engine.render_pdf(frozen).pdf).split())
    assert "The saved editor thumbnail in this JPEG and the main image differ" in text
    assert "White cells mark measured differences" in text
