"""Authored end-to-end packets preserve regional masks and existing decisions."""

from __future__ import annotations

import copy
import io

import cv2
import numpy as np
import pytest
from PIL import Image
from test_image_packets import _bytes
from test_image_packets import packet_fixture as packet_fixture
from test_image_packets_v4 import paired

from app.core.canonical import canonical_hash, canonical_json
from app.evidence import image_packets_v4 as prior
from app.evidence import image_packets_v5 as packets
from app.evidence.service import asset_refs, load_packet_set, persist_packet_set

TARGET = (31, 33, 64, 64)
DONOR = (151, 138, 64, 64)
NATIVE_TARGET = (33, 35, 96, 96)
NATIVE_DONOR = (741, 428, 96, 96)


def _box(box):
    return dict(zip(("x", "y", "w", "h"), box, strict=True))


def _selection(shape, box=TARGET):
    mask = np.zeros(shape, bool)
    x, y, w, h = box
    mask[y : y + h, x : x + w] = True
    return mask


def _texture(shape, seed):
    y, x = np.indices(shape)
    raw = np.random.default_rng(seed).integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    soft = cv2.GaussianBlur(raw, (5, 5), 0.85).astype(np.float32)
    return np.clip(soft + (20 * np.sin(x / 2.5) + 24 * np.cos(y / 4))[..., None], 0, 255).astype(np.uint8)


def _authored(kind):
    if kind == "unsupported_geometry":
        return _texture((240, 1025), 295)
    if kind == "ambiguous_repetition":
        return np.tile(_texture((48, 48), 78), (5, 6, 1))[:, :256]
    if kind == "insufficient_texture":
        return np.full((240, 256, 3), 119, dtype=np.uint8)
    if kind == "native_periodic_budget":
        return np.tile(_texture((8, 8), 51), (96, 128, 1))
    native = kind.startswith("native_")
    operation = kind.removeprefix("native_")
    rgb = _texture((768, 1024) if native else (240, 256), 295)
    if operation in {"translation", "horizontal_reflection"}:
        x, y, w, h = NATIVE_DONOR if native else DONOR
        donor = rgb[y : y + h, x : x + w]
        if operation == "horizontal_reflection":
            donor = donor[:, ::-1]
        rgb[_selection(rgb.shape[:2], NATIVE_TARGET if native else TARGET)] = donor.reshape(-1, 3)
    return rgb


def _localised(packet_fixture, kind, encoding="PNG"):
    db, store, analysis, result, _ = packet_fixture
    rgb = _authored(kind)
    data = _bytes(Image.fromarray(rgb), format=encoding, **({"quality": 90} if encoding == "JPEG" else {}))
    source = store.put_bytes(data)
    analysis.sha256, analysis.size_bytes = source.sha256, source.size_bytes
    result["input"]["sha256"] = source.sha256
    image = Image.open(io.BytesIO(data)).convert("RGB")
    selection = _selection(rgb.shape[:2], NATIVE_TARGET if kind.startswith("native_") else TARGET)
    return paired((db, store, analysis, result, image), a=selection, b=selection)


@pytest.mark.parametrize(
    ("transform", "encoding"), [("translation", "PNG"), ("horizontal_reflection", "JPEG")]
)
def test_product_packet_uses_exact_source_crop_and_keeps_mask(packet_fixture, transform, encoding):
    db, store, analysis, result, selection, _ = _localised(packet_fixture, transform, encoding)
    saved_result = canonical_json(result)
    previous = prior.persist_packet_set(db, analysis, store, result=result)
    document = persist_packet_set(db, analysis, store, result=result)

    assert document["version"] == "image-evidence-6"
    assert document["derived_from"] == {key: previous[key] for key in ("id", "version", "sha256")}
    assert canonical_json(result) == saved_result
    assert load_packet_set(db, analysis, store) == document
    assert load_packet_set(db, analysis, store, set_id=previous["id"]) == previous
    assert document["comparison"]["counts"] == previous["comparison"]["counts"]
    for packet, old in zip(document["packets"], previous["packets"], strict=True):
        cue = packet["copy_corroboration"]
        assert cue["status"] == "copy_consistent"
        assert cue["search_strategy"] == "native_dense"
        assert cue["recipient_box"] == _box(TARGET)
        assert cue["donor"]["box"] == _box(DONOR)
        assert cue["donor"]["transform"] == transform
        assert cue["authorship_confidence_contribution"] == 0
        assert cue["recipient_pixels"] == int(selection.sum())
        assert packet["region"] == old["region"]
        assert packet["artifacts"]["mask"]["sha256"] == old["artifacts"]["mask"]["sha256"]
        assert packet["artifacts"]["overlay"]["sha256"] == old["artifacts"]["overlay"]["sha256"]
        assert "compatible with copying or repeated detail" in packet["interpretation"]
        asset = packet["artifacts"]["copy_donor"]
        assert asset["sha256"] in asset_refs(document)
        source, _ = packets.legacy._source(store.read_bytes(analysis.sha256))
        x, y, w, h = DONOR
        with Image.open(io.BytesIO(store.read_bytes(asset["sha256"]))) as crop:
            assert crop.tobytes() == source.crop((x, y, x + w, y + h)).tobytes()

    changed = copy.deepcopy(document)
    changed["packets"][0]["copy_corroboration"]["donor"]["box"]["x"] += 1
    changed["sha256"] = packets.packet_hash(changed)
    with pytest.raises(packets.PacketIntegrityError, match="pixels differ"):
        packets.validate_assets(changed, store)


@pytest.mark.parametrize(
    "outcome", ["unsupported_geometry", "insufficient_texture", "ambiguous_repetition", "no_donor"]
)
def test_product_packet_preserves_unassessed_and_inconclusive_outcomes(packet_fixture, outcome):
    db, store, analysis, result, _, _ = _localised(packet_fixture, outcome)
    document = persist_packet_set(db, analysis, store, result=result)
    assert document["packets"]
    for packet in document["packets"]:
        cue = packet["copy_corroboration"]
        assert cue["status"] == outcome
        assert cue["donor"] is None and "copy_donor" not in packet["artifacts"]
        assert cue["authorship_confidence_contribution"] == 0
        assert "copy check" in packet["interpretation"]


@pytest.mark.parametrize(
    ("operation", "encoding"), [("translation", "PNG"), ("horizontal_reflection", "JPEG")]
)
def test_large_native_copy_cue_preserves_historical_size_skip(packet_fixture, operation, encoding):
    db, store, analysis, result, selection, _ = _localised(packet_fixture, f"native_{operation}", encoding)
    historical = packets.build_packet_set(db, analysis, store, result, version=packets.V5)
    history_bytes = canonical_json(historical)
    assert load_packet_set(db, analysis, store) == historical
    assert len(historical["packets"]) == 2
    for packet in historical["packets"]:
        assert packet["copy_corroboration"]["status"] == "unsupported_geometry"
        assert "512 px" in packet["interpretation"]
    unchanged_result = canonical_json(result)
    current = persist_packet_set(db, analysis, store, result=result)
    assert load_packet_set(db, analysis, store) == current
    assert canonical_json(load_packet_set(db, analysis, store, set_id=historical["id"])) == history_bytes
    assert canonical_json(result) == unchanged_result
    assert len(current["packets"]) == 2
    for packet, old in zip(current["packets"], historical["packets"], strict=True):
        cue = packet["copy_corroboration"]
        assert cue["status"] == "copy_consistent"
        assert cue["search_strategy"] == "native_anchor_shortlist"
        assert cue["source_size"] == [1024, 768]
        assert cue["recipient_box"] == _box(NATIVE_TARGET)
        assert cue["recipient_pixels"] == int(selection.sum())
        assert cue["donor"]["box"] == _box(NATIVE_DONOR)
        assert cue["donor"]["transform"] == operation
        assert cue["authorship_confidence_contribution"] == 0
        assert packet["artifacts"]["mask"]["sha256"] == old["artifacts"]["mask"]["sha256"]


@pytest.mark.parametrize(
    ("variant", "outcome"),
    [("native_original", "no_donor"), ("native_periodic_budget", "unassessed_candidate_budget")],
)
def test_large_native_controls_keep_independent_coverage(packet_fixture, variant, outcome):
    db, store, analysis, result, _, _ = _localised(packet_fixture, variant)
    current = persist_packet_set(db, analysis, store, result=result)
    assert len(current["packets"]) == 2
    for packet in current["packets"]:
        cue = packet["copy_corroboration"]
        assert cue["search_strategy"] == "native_anchor_shortlist"
        assert cue["status"] == outcome
        assert cue["donor"] is None and "copy_donor" not in packet["artifacts"]
        assert cue["authorship_confidence_contribution"] == 0
    if outcome == "unassessed_candidate_budget":
        assert "too many similar areas" in current["packets"][0]["interpretation"].lower()


def test_report_embeds_matching_original_patch_and_retains_history(packet_fixture, monkeypatch):
    from app.reports import engine, image_evidence
    from app.reports.snapshot import dump, load
    from app.reports.specimens import specimen

    db, store, analysis, measured, _, _ = _localised(packet_fixture, "native_translation")
    snapshot = specimen("image", "standard")
    result = copy.deepcopy(snapshot.result)
    result["input"].update(measured["input"])
    result.update(
        signals=measured["signals"], exhibits=measured["exhibits"], findings=[], kind="image", status="done"
    )
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    document = persist_packet_set(db, analysis, store, result=result)
    snapshot = snapshot.model_copy(
        update={
            "result": result,
            "image_evidence": document,
            "analysis": snapshot.analysis.model_copy(
                update={"id": analysis.id, "revision": analysis.revision}
            ),
            "media": snapshot.media.model_copy(update={"sha256": analysis.sha256}),
        }
    )
    monkeypatch.setattr(image_evidence, "default_store", lambda: store)

    assert image_evidence.capture(db, analysis, store=store) == document
    assert load(dump(snapshot)).image_evidence == document
    view = image_evidence.view(snapshot, store=store)
    first = view["packets"][0]
    image = next(value for value in first["images"] if value["role"] == "copy_donor")
    assert image["uri"].startswith("data:image/png;base64,")
    assert image["sha256"] == first["artifacts"]["copy_donor"]["sha256"]
    assert any(asset["role"] == "copy_donor" for asset in image_evidence.manifest(snapshot)["artifacts"])
    assert "Matching patch from the original image" in engine.render_html(snapshot, embed_fonts=False)
    assert "compatible with copying or repeated detail" in engine.render_html(snapshot, embed_fonts=False)
    redacted = snapshot.model_copy(update={"report": snapshot.report.model_copy(update={"redacted": True})})
    assert all(
        image["uri"] is None
        for packet in image_evidence.view(redacted, store=store)["packets"]
        for image in packet["images"]
    )
