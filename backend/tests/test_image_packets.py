from __future__ import annotations

import io
import json
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api.public_projection import project
from app.core.canonical import canonical_hash, canonical_json
from app.evidence.image_packets import (
    PacketIntegrityError,
    build_packet_set,
    load_packet_set,
    packet_hash,
    persist_packet_set,
    validate_packet_set,
)
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact, Base
from app.storage.objects import ObjectIntegrityError, ObjectStore


def _bytes(image: Image.Image, **kwargs) -> bytes:
    stream = io.BytesIO()
    image.save(stream, **kwargs)
    return stream.getvalue()


@pytest.fixture
def packet_fixture(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    store = ObjectStore(tmp_path / "objects")
    with Session(engine) as db:
        yy, xx = np.mgrid[:120, :160]
        image = Image.fromarray(np.stack([xx, yy, (xx + yy) % 256], axis=2).astype(np.uint8))
        source = store.put_bytes(_bytes(image, format="PNG"))
        analysis = Analysis(id="image-test", kind="image", filename="private-name.png", content_type="image/png",
                            size_bytes=source.size_bytes, sha256=source.sha256, status="done", revision=1,
                            result_schema_version=2)
        db.add(analysis)
        db.flush()
        grid = np.arange(192, dtype=np.uint8).reshape(12, 16)
        scalar = register_artefact(db, store, _bytes(Image.fromarray(grid), format="PNG"), media_type="image/png",
                                   role="exhibit_payload", created_by="img.forensic.ela", analysis_id=analysis.id)
        space = {"width": 160, "height": 120, "scale_to_original": 1.0}
        heat = {"id": "ex_heat", "kind": "heatmap", "created_by": "img.forensic.ela", "media_ref": "original",
                "semantics": "error_level", "calibrated": False, "grid": [16, 12], "value_range": [0.0, 10.0],
                "coordinate_space": space, "payload_artefact_id": scalar.id, "sha256": scalar.sha256,
                "source_signal_ids": []}
        regions = {"id": "ex_regions", "kind": "regions", "created_by": "localise", "media_ref": "original",
                   "coordinate_space": space, "source_signal_ids": ["img.forensic.ela:global"],
                   "regions": [{"region": {"kind": "bbox", "bbox": {"x": 40, "y": 30, "w": 40, "h": 40}},
                                "score": 9.0}]}
        result = {"input": {"sha256": source.sha256}, "exhibits": [heat, regions],
                  "signals": [{"id": "img.forensic.ela:global", "detector_id": "img.forensic.ela", "display_id": "S01",
                               "status": "ok", "name": "PRIVATE MODEL", "observed": "PRIVATE MODEL observation",
                               "exhibit_ids": ["ex_heat"]}],
                  "findings": [{"id": "F1", "family": "F5_compression_history", "evidence_ids": ["img.forensic.ela:global"]}]}
        analysis.result_hash = canonical_hash(result)
        store.put_bytes(canonical_json(result))
        yield db, store, analysis, result, image


def _rehash(analysis, result):
    analysis.result_hash = canonical_hash(result)


def _localised_fixture(packet_fixture):
    from app.detectors.image.localiser import components

    db, store, analysis, result, source = packet_fixture
    scalar = np.zeros((12, 16), dtype=np.uint8)
    scalar[2:7, 3:5] = 240
    scalar[5:7, 5:9] = 220  # L shape: its bounding rectangle contains unselected pixels
    expected = scalar >= 128
    art = register_artefact(db, store, _bytes(Image.fromarray(scalar), format="PNG"), media_type="image/png",
                           role="exhibit_payload", created_by="loc.mesorch", analysis_id=analysis.id)
    heat, regions = result["exhibits"]
    heat.update(created_by="loc.mesorch", semantics="p_manipulated", value_range=[0, 1],
                payload_artefact_id=art.id, sha256=art.sha256)
    regions.update(created_by="loc.mesorch", source_signal_ids=["loc.mesorch:global"])
    regions["regions"], _ = components(scalar.astype(float) / 255, np.ones(scalar.shape, bool), 160, 120)
    result["signals"][0].update(id="loc.mesorch:global", detector_id="loc.mesorch", details={"localisation": {
        "schema": "recorded-mask/1", "threshold": 0.5, "comparison": "greater_or_equal", "scalar_encoding": "linear_uint8",
        "component_connectivity": 8, "minimum_region_fraction": 0.001, "grid": [16, 12], "capability": "local_editing"}})
    _rehash(analysis, result)
    return db, store, analysis, result, source, expected


def test_localisation_mask_preserves_connected_shape_instead_of_box(packet_fixture):
    db, store, analysis, result, _, expected = _localised_fixture(packet_fixture)
    output = build_packet_set(db, analysis, store, result)
    packet = output["packets"][0]
    assert output["version"] == "image-evidence-3"
    assert packet["evidence_type"] == "localisation"
    assert packet["localisation"]["region_cells"] == int(expected.sum())
    mask = Image.open(io.BytesIO(store.read_bytes(packet["artifacts"]["mask"]["sha256"]))).convert("L")
    np.testing.assert_array_equal(np.asarray(mask) > 127, expected)
    assert not expected[2, 8]  # within bbox, outside the selected component
    assert packet["derivation"]["selection"] == "recorded_connected_component"
    assert packet["localisation"]["source_pixels_per_cell"] == [10, 10]
    assert "highest 5%" not in " ".join(packet["limitations"])
    assert "mesorch" not in json.dumps(output).lower()


def test_localisation_does_not_invent_missing_mask_provenance(packet_fixture):
    db, store, analysis, result, _, _ = _localised_fixture(packet_fixture)
    result["signals"][0].pop("details")
    _rehash(analysis, result)
    output = build_packet_set(db, analysis, store, result)
    assert output["packets"] == []
    assert "reproducible component mask" in " ".join(output["limitations"])


def test_uniform_positive_localisation_preserves_measurement_and_excluded_islands(packet_fixture):
    from app.detectors.image.localiser import components

    db, store, analysis, result, _, _ = _localised_fixture(packet_fixture)
    scalar = np.full((12, 16), 240, dtype=np.uint8)
    reliable = np.ones(scalar.shape, bool)
    reliable[:, 7:9] = False
    heat = result['exhibits'][0]
    for field, pixels in [('payload_artefact_id', scalar), ('reliability_mask_artefact_id', reliable.astype(np.uint8) * 255)]:
        art = register_artefact(db, store, _bytes(Image.fromarray(pixels), format='PNG'), media_type='image/png',
                               role='exhibit_payload', created_by='loc.mesorch', analysis_id=analysis.id)
        heat[field] = art.id
        if field == 'payload_artefact_id':
            heat['sha256'] = art.sha256
    result['exhibits'][1]['regions'], _ = components(scalar.astype(float) / 255, reliable, 160, 120)
    _rehash(analysis, result)
    output = build_packet_set(db, analysis, store, result)
    assert len(output['packets']) == 2
    combined = np.zeros(scalar.shape, bool)
    for packet in output['packets']:
        mask = Image.open(io.BytesIO(store.read_bytes(packet['artifacts']['mask']['sha256']))).convert('L')
        combined |= np.asarray(mask) > 127
        assert 'cannot distinguish an edited area' in ' '.join(packet['limitations'])
    np.testing.assert_array_equal(combined, reliable)


def test_localisation_refuses_region_with_wrong_component_anchor(packet_fixture):
    db, store, analysis, result, _, _ = _localised_fixture(packet_fixture)
    result["exhibits"][1]["regions"][0]["region"]["bbox"]["x"] += 20
    _rehash(analysis, result)
    assert build_packet_set(db, analysis, store, result)["packets"] == []


def _gated_fixture(packet_fixture, score):
    db, store, analysis, result, source, expected = _localised_fixture(packet_fixture)
    heat, regions = result["exhibits"]
    heat["created_by"] = regions["created_by"] = "loc.fused"
    signal = result["signals"][0]
    signal.update(id="loc.fused:global", detector_id="loc.fused")
    regions["source_signal_ids"] = [signal["id"]]
    gate = {"schema": "image-candidate-eligibility/1", "status": "evaluated", "scope": "whole_image",
            "comparison": "greater_or_equal", "score": score, "score_hex": float(score).hex(),
            "score_exact_decimal": repr(float(score)), "threshold": .5, "threshold_hex": .5.hex(),
            "eligible": score >= .5, "source_sha256": analysis.sha256, "scalar_map_sha256": heat["sha256"]}
    signal["details"]["localisation"]["candidate_eligibility"] = gate
    _rehash(analysis, result)
    return db, store, analysis, result, source, expected


@pytest.mark.parametrize("score,selected", [(.2, 0), (.499999999999, 0), (.5, 1), (.9, 1)])
def test_image_gate_is_separate_from_raw_map_quality_and_survives_rounding(packet_fixture, score, selected):
    db, store, analysis, result, _, _ = _gated_fixture(packet_fixture, score)
    before = deepcopy(result)
    output = build_packet_set(db, analysis, store, result)
    assert result == before
    check = output["coverage"]["inspection_checks"][0]
    assert check["raw_components"] == 1
    assert check["eligible_components"] == check["selected_components"] == selected
    assert check["omitted_components"] == 0
    assert len(output["packets"]) == selected
    assert check["excluded_area_fraction"] == 0
    assert output["overview"]["inspection_01_response"]["sha256"] == result["exhibits"][0]["sha256"]
    assert np.asarray(Image.open(io.BytesIO(store.read_bytes(check["scalar_map_sha256"])))).max() == 240
    assert check["candidate_eligibility"]["score_hex"] == float(score).hex()
    assert "fused" not in json.dumps(output).lower()
    if selected:
        assert output["packets"][0]["localisation"]["candidate_eligibility"] == check["candidate_eligibility"]
    else:
        # A valid-looking forged region exhibit cannot bypass a failed image gate.
        assert "does not confirm authenticity" in " ".join(output["limitations"])
        assert "No dedicated manipulation-localisation result is available" not in " ".join(output["limitations"])


@pytest.mark.parametrize("field,value", [("source_sha256", "a" * 64), ("scalar_map_sha256", "b" * 64),
                                         ("eligible", False), ("threshold", .4), ("score_hex", "nan"),
                                         ("threshold_exact_decimal", "0.4")])
def test_image_gate_refuses_mismatched_binding_or_rule(packet_fixture, field, value):
    db, store, analysis, result, _, _ = _gated_fixture(packet_fixture, .9)
    result["signals"][0]["details"]["localisation"]["candidate_eligibility"][field] = value
    _rehash(analysis, result)
    output = build_packet_set(db, analysis, store, result)
    assert not output["packets"] and not output["coverage"]["inspection_checks"]
    assert "eligibility record" in " ".join(output["limitations"])


def test_gated_adapter_cannot_fall_back_to_ungated_and_gate_without_regions_is_recorded(packet_fixture):
    db, store, analysis, result, _, _ = _gated_fixture(packet_fixture, .2)
    result["exhibits"] = result["exhibits"][:1]  # genuine adapter emits no regions when gate fails
    _rehash(analysis, result)
    output = build_packet_set(db, analysis, store, result)
    assert output["coverage"]["inspection_checks"][0]["raw_components"] == 1
    assert not output["packets"]
    result["signals"][0]["details"]["localisation"].pop("candidate_eligibility")
    _rehash(analysis, result)
    refused = build_packet_set(db, analysis, store, result)
    assert not refused["packets"] and not refused["coverage"]["inspection_checks"]


def test_frozen_gate_selection_cannot_be_changed_even_with_rehashed_set(packet_fixture):
    db, store, analysis, result, _, _ = _gated_fixture(packet_fixture, .9)
    output = build_packet_set(db, analysis, store, result)
    output["coverage"]["inspection_checks"][0]["candidate_eligibility"]["eligible"] = False
    output["sha256"] = packet_hash(output)
    with pytest.raises(PacketIntegrityError, match="eligibility contradicts"):
        validate_packet_set(output, analysis_id=analysis.id, revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


def test_empty_dedicated_localisation_is_not_resegmented_as_relative_peaks(monkeypatch):
    from app.engine.stages import image as stage

    monkeypatch.setattr(stage, "_stage_exhibits", lambda ctx, creators: {"exhibits": []})
    ctx = SimpleNamespace(outputs={"detect": {"exhibits": [
        {"kind": "heatmap", "semantics": "p_manipulated", "created_by": "loc.fused"},
        {"kind": "heatmap", "semantics": "p_manipulated", "created_by": "loc.mesorch"},
    ]}}, check=lambda *a: pytest.fail("An empty dedicated result cannot become a relative peak region"))
    assert stage.localise(ctx) == {"exhibits": []}


def test_crops_match_source_and_records_stay_unchanged(packet_fixture):
    db, store, analysis, result, source = packet_fixture
    before = deepcopy(result)
    output = build_packet_set(db, analysis, store, result)
    assert result == before and analysis.result_hash == canonical_hash(before)
    assert len(output["packets"]) == 1
    packet = output["packets"][0]
    assert packet["finding_ids"] == ["F1"]
    assert packet["signal_refs"][0]["id"] == project(result)["signals"][0]["id"]
    assert packet["signal_refs"][0]["display_id"] == "S01"
    crop = packet["crop_region"]
    expected = source.crop((crop["x"], crop["y"], crop["x"] + crop["w"], crop["y"] + crop["h"]))
    actual = Image.open(io.BytesIO(store.read_bytes(packet["artifacts"]["crop"]["sha256"])))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.size == (crop["w"], crop["h"])
    assert packet["evidence_type"] == "anomaly"
    assert "highest 5%" in " ".join(packet["limitations"])
    assert "does not establish" in " ".join(packet["limitations"])
    assert "PRIVATE MODEL" not in json.dumps(output)
    assert output["sha256"] == packet_hash(output)


def test_read_is_pure_and_build_is_idempotent(packet_fixture):
    db, store, analysis, result, _ = packet_fixture
    assert load_packet_set(db, analysis, store) is None
    before_count = len(db.scalars(select(Artefact)).all())
    assert load_packet_set(db, analysis, store) is None
    assert len(db.scalars(select(Artefact)).all()) == before_count
    first = persist_packet_set(db, analysis, store, result=result)
    count = len(db.scalars(select(Artefact)).all())
    second = persist_packet_set(db, analysis, store, result=result)
    assert first == second == load_packet_set(db, analysis, store, set_id=first["id"])
    assert len(db.scalars(select(Artefact)).all()) == count


@pytest.mark.parametrize("kind", ["attribution", "failed", "flat", "no_region"])
def test_no_invented_local_evidence(packet_fixture, kind):
    db, store, analysis, result, _ = packet_fixture
    if kind == "attribution":
        result["exhibits"][0]["semantics"] = "attribution"
    elif kind == "failed":
        result["signals"][0]["status"] = "failed"
    elif kind == "no_region":
        result["exhibits"].pop()
    else:
        row = register_artefact(db, store, _bytes(Image.new("L", (16, 12)), format="PNG"), media_type="image/png",
                                role="exhibit_payload", created_by="test", analysis_id=analysis.id)
        result["exhibits"][0].update(payload_artefact_id=row.id, sha256=row.sha256)
    _rehash(analysis, result)
    output = build_packet_set(db, analysis, store, result)
    assert output["packets"] == []
    assert "cannot identify edited pixels" in " ".join(output["limitations"])


def test_bad_region_does_not_silently_clip_to_a_different_region(packet_fixture):
    db, store, analysis, result, _ = packet_fixture
    result["exhibits"][1]["regions"][0]["region"]["bbox"]["x"] = -5
    _rehash(analysis, result)
    output = build_packet_set(db, analysis, store, result)
    assert output["packets"] == [] and output["omitted_regions"] == 1
    assert "out-of-bounds" in " ".join(output["limitations"])


def test_scalar_hash_mismatch_and_corruption_fail_closed(packet_fixture):
    db, store, analysis, result, _ = packet_fixture
    result["exhibits"][0]["sha256"] = "f" * 64
    _rehash(analysis, result)
    with pytest.raises(PacketIntegrityError, match="frozen exhibit"):
        build_packet_set(db, analysis, store, result)


def test_source_corruption_blocks_read_even_when_plates_survive(packet_fixture):
    db, store, analysis, result, _ = packet_fixture
    build_packet_set(db, analysis, store, result)
    path = store.path_for(analysis.sha256)
    path.chmod(0o644)
    path.write_bytes(b"changed source")
    with pytest.raises(ObjectIntegrityError):
        load_packet_set(db, analysis, store)


def test_hash_and_binding_tamper_rejected(packet_fixture):
    db, store, analysis, result, _ = packet_fixture
    output = build_packet_set(db, analysis, store, result)
    output["packets"][0]["region"]["x"] = 0
    with pytest.raises(PacketIntegrityError):
        validate_packet_set(output, analysis_id=analysis.id, revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    output["sha256"] = packet_hash(output)
    with pytest.raises(PacketIntegrityError):
        validate_packet_set(output, analysis_id="another-analysis", revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


def test_exif_orientation_crop_and_scaled_coordinates(packet_fixture):
    db, store, analysis, result, source = packet_fixture
    exif = Image.Exif()
    exif[274] = 6
    data = _bytes(source, format="JPEG", quality=95, exif=exif)
    original = store.put_bytes(data)
    analysis.sha256 = original.sha256
    result["input"]["sha256"] = original.sha256
    # EXIF rotation gives 120 x 160; saved region coordinates are a half-size oriented copy.
    for exhibit in result["exhibits"]:
        exhibit["coordinate_space"] = {"width": 60, "height": 80, "scale_to_original": 2.0}
    result["exhibits"][1]["regions"][0]["region"]["bbox"] = {"x": 10, "y": 15, "w": 20, "h": 20}
    _rehash(analysis, result)
    output = build_packet_set(db, analysis, store, result)
    assert output["source"]["exif_orientation"] == 6
    assert (output["source"]["width"], output["source"]["height"]) == (120, 160)
    packet = output["packets"][0]
    assert packet["region"] == {"x": 20, "y": 30, "w": 40, "h": 40}
    c = packet["crop_region"]
    expected = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB").crop((c["x"], c["y"], c["x"]+c["w"], c["y"]+c["h"]))
    actual = Image.open(io.BytesIO(store.read_bytes(packet["artifacts"]["crop"]["sha256"])))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_plate_dedup_and_api_export_use_identical_frozen_set(packet_fixture, monkeypatch):
    from app.api import routes, v2_evidence, v2_evidence_packets

    db, store, analysis, result, _ = packet_fixture
    analysis.result_json = json.dumps(result)
    output = build_packet_set(db, analysis, store, result)
    crop = output["packets"][0]["artifacts"]["crop"]
    # The object index deduplicates previews regardless of their original creator.
    row = db.scalars(select(Artefact).where(Artefact.sha256 == crop["sha256"])).first()
    row.created_by = "an-earlier-renderer"
    db.flush()

    @contextmanager
    def sessions():
        yield db

    monkeypatch.setattr(v2_evidence_packets, "get_session", sessions)
    monkeypatch.setattr(v2_evidence_packets, "default_store", lambda: store)
    monkeypatch.setattr(v2_evidence, "default_store", lambda: store)
    response = v2_evidence_packets.image_evidence(analysis.id)
    assert json.loads(response.body) == output
    request = SimpleNamespace(headers={}, state=SimpleNamespace(principal=SimpleNamespace(id="local")))
    image = v2_evidence_packets.image_evidence_plate(analysis.id, output["id"], crop["sha256"], request)
    assert image.body == store.read_bytes(crop["sha256"])
    with pytest.raises(v2_evidence_packets.ProblemError):
        v2_evidence_packets.image_evidence_plate(analysis.id, output["id"], analysis.sha256, request)
    monkeypatch.setattr(routes, "_get", lambda _aid: analysis)
    monkeypatch.setattr(routes, "_current_revision", lambda _: None)
    monkeypatch.setattr(routes, "_verified", lambda *_: None)
    monkeypatch.setattr(routes, "get_session", sessions)
    monkeypatch.setattr("app.storage.objects.default_store", lambda: store)
    monkeypatch.setattr(routes.audit, "record", lambda *_, **__: None)
    register_artefact(db, store, canonical_json(result), media_type="application/json", role="stage_output",
                      created_by="publish-result", analysis_id=analysis.id)
    exported = json.loads(routes.export_json(analysis.id, request).body)
    assert exported["image_evidence"] == output
    assert output["packets"][0]["signal_refs"][0]["id"] == exported["result"]["signals"][0]["id"]


def _large_native_fixture(packet_fixture, *, full_extent: bool):
    db, store, analysis, result, _ = packet_fixture
    width, height = 4101, 51  # legacy factor 2 drops one source column and one row
    source = store.put_bytes(_bytes(Image.new("RGB", (width, height), (60, 90, 120)), format="PNG"))
    analysis.sha256 = source.sha256
    result["input"]["sha256"] = source.sha256
    grid = np.tile(np.arange(2050, dtype=np.uint16) % 256, (25, 1)).astype(np.uint8)
    scalar = register_artefact(db, store, _bytes(Image.fromarray(grid), format="PNG"), media_type="image/png",
                               role="exhibit_payload", created_by="img.forensic.ela", analysis_id=analysis.id)
    space = ({"width": width, "height": height, "scale_to_original": 1.0} if full_extent
             else {"width": 2050, "height": 25, "scale_to_original": width / 2050})
    result["exhibits"][0].update(grid=[2050, 25], coordinate_space=space,
                                 payload_artefact_id=scalar.id, sha256=scalar.sha256)
    result["exhibits"][1]["coordinate_space"] = space
    result["exhibits"][1]["regions"][0]["region"]["bbox"] = {"x": 20, "y": 3, "w": 40, "h": 10}
    if full_extent:
        result["signals"][0]["details"] = {"display_mapping_version": "full-extent-area/1"}
    _rehash(analysis, result)
    store.put_bytes(canonical_json(result))
    return db, store, analysis, result


def test_old_nondivisible_native_map_omitted_instead_of_stretched(packet_fixture):
    db, store, analysis, result = _large_native_fixture(packet_fixture, full_extent=False)
    output = build_packet_set(db, analysis, store, result)
    assert output["packets"] == []
    assert output["omitted_regions"] == 1
    assert "ambiguous edge mapping" in " ".join(output["limitations"])
    assert load_packet_set(db, analysis, store) == output


def test_full_extent_native_map_is_accepted_on_same_nondivisible_source(packet_fixture):
    db, store, analysis, result = _large_native_fixture(packet_fixture, full_extent=True)
    output = build_packet_set(db, analysis, store, result)
    assert len(output["packets"]) == 1
    assert output["packets"][0]["region"] == {"x": 20, "y": 3, "w": 40, "h": 10}
    assert load_packet_set(db, analysis, store) == output


def test_existing_unsafe_set_refused_without_rewriting_bytes(packet_fixture, monkeypatch):
    from app.evidence import image_packets

    db, store, analysis, result = _large_native_fixture(packet_fixture, full_extent=False)
    with monkeypatch.context() as old_build:
        old_build.setattr(image_packets, "_native_mapping_issue", lambda *_args: None)
        output = build_packet_set(db, analysis, store, result)
    row = db.get(Artefact, output["id"])
    frozen = store.read_bytes(row.sha256)
    with pytest.raises(PacketIntegrityError, match="ambiguous older"):
        load_packet_set(db, analysis, store)
    assert store.read_bytes(row.sha256) == frozen


def test_legacy_mapping_guard_keeps_divisible_grids_and_small_existing_sets():
    from app.evidence.image_packets import _native_mapping_issue

    heat = {"created_by": "img.forensic.ela"}
    assert _native_mapping_issue(heat, [], 5001, 2501) is not None
    assert _native_mapping_issue(heat, [], 5000, 2500) is None
    assert _native_mapping_issue(heat, [], 2048, 1365) is None
