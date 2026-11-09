"""Authored masks exercise actual membership, caps, failure states and immutable successors."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from PIL import Image
from test_image_packets import _bytes
from test_image_packets import packet_fixture as packet_fixture

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.image.localiser import components
from app.evidence import image_packets as legacy
from app.evidence import image_packets_v4 as packets
from app.evidence.image_eligibility import area_gate
from app.evidence.service import asset_refs
from app.storage.artefacts import register_artefact


def paired(packet_fixture, *, a=None, b=None, second_status="ok", withheld=False, wrong_box=False):
    db, store, analysis, result, source = packet_fixture
    if a is None:
        a = np.zeros((12, 16), dtype=bool)
        a[2:7, 3:5] = True
    if b is None:
        b = np.zeros((12, 16), dtype=bool)
        b[6:9, 4:10] = True  # Intersection is one cell. The boxes must not be filled.
    result["signals"], result["exhibits"] = [], []
    for n, (creator, mask) in enumerate((("loc.native_regional", a), ("loc.splice", b)), 1):
        art = register_artefact(db, store, _bytes(Image.fromarray(mask.astype(np.uint8) * 240), format="PNG"),
                                media_type="image/png", role="exhibit_payload", created_by=creator, analysis_id=analysis.id)
        signal_id, heat_id = f"{creator}:global", f"heat-{n}"
        grid = list(mask.shape[::-1])
        gate = area_gate(int(mask.sum()), max(1, int(np.ceil(mask.size * .01))), mask.size)
        gate.update(source_sha256=analysis.sha256, scalar_map_sha256=art.sha256)
        if n == 2 and withheld:
            gate = area_gate(0, 1, mask.size)
            gate.update(source_sha256=analysis.sha256, scalar_map_sha256=art.sha256)
        policy = {"mode": "screening", "input_px": 1024, "fit_size": [1024, 768],
                  "canvas_size": [1024, 1024], "padding_position": "top_left", "source_cropped": False}
        result["signals"].append({"id": signal_id, "detector_id": creator, "display_id": f"S{n:02d}",
            "status": second_status if n == 2 else "ok", "exhibit_ids": [heat_id],
            "details": {"localisation": {"schema": "recorded-mask/1", "threshold": .5,
                "comparison": "greater_or_equal", "scalar_encoding": "linear_uint8", "component_connectivity": 8,
                "minimum_region_fraction": .001, "grid": grid, "candidate_eligibility": gate,
                "resolution_policy": policy}}})
        space = {"width": source.width, "height": source.height, "scale_to_original": 1.}
        regions, _ = components(mask.astype(float) * 240 / 255., np.ones(mask.shape, bool), *source.size)
        if n == 2 and wrong_box and regions:
            regions[0]["region"]["bbox"]["x"] += 10
        result["exhibits"].extend([
            {"id": heat_id, "kind": "heatmap", "created_by": creator, "media_ref": "original",
             "semantics": "p_manipulated", "calibrated": False, "grid": grid, "value_range": [0., 1.],
             "coordinate_space": space, "payload_artefact_id": art.id, "sha256": art.sha256,
             "source_signal_ids": [signal_id]},
            {"id": f"regions-{n}", "kind": "regions", "created_by": creator, "media_ref": "original",
             "coordinate_space": space, "source_signal_ids": [signal_id], "regions": [] if n == 2 and withheld else regions}])
    result["findings"] = []
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    return db, store, analysis, result, a, b


def test_actual_union_and_one_cell_intersection_are_preserved(packet_fixture):
    db, store, analysis, result, a, b = paired(packet_fixture)
    original = canonical_json(result)
    document = packets.build_packet_set(db, analysis, store, result)
    comparison = document["comparison"]
    assert document["version"] == "image-evidence-4" and comparison["status"] == "ready"
    assert comparison["counts"] == {"sensitive": 10, "complementary": 18, "wider": 27, "shared": 1}
    for name, expected in (("sensitive", a), ("complementary", b), ("wider", a | b), ("shared", a & b)):
        np.testing.assert_array_equal(packets._mask(store, comparison["artifacts"][f"{name}_mask"]), expected)
    assert canonical_json(result) == original
    assert comparison["authorship_confidence_contribution"] == 0
    assert "loc.native" not in json.dumps(document) and "splice" not in json.dumps(document)
    assert packets.load_packet_set(db, analysis, store) == document
    assert packets.persist_packet_set(db, analysis, store) == document
    assert all(asset["sha256"] in asset_refs(document) for asset in comparison["artifacts"].values())


def test_successor_keeps_old_bytes_and_corrects_only_fitted_canvas_description(packet_fixture):
    db, store, analysis, result, _, _ = paired(packet_fixture)
    old = legacy.build_packet_set(db, analysis, store, result)
    old_bytes = canonical_json(old)
    new = packets.build_packet_set(db, analysis, store, result)
    assert new["derived_from"] == {key: old[key] for key in ("id", "version", "sha256")}
    assert old["id"] != new["id"]
    assert canonical_json(legacy.load_packet_set(db, analysis, store, set_id=old["id"])) == old_bytes
    assert packets.load_packet_set(db, analysis, store, set_id=old["id"]) == old
    assert "resized to 1024 × 1024" in " ".join(old["packets"][0]["limitations"])
    assert "fitted without stretching to 1024 × 768" in " ".join(new["packets"][0]["limitations"])
    assert "resized to 1024 × 1024" not in " ".join(new["packets"][0]["limitations"])
    for p, q in zip(old["packets"], new["packets"], strict=True):
        assert p["region"] == q["region"] and p["measurements"] == q["measurements"]
        assert p["localisation"] == q["localisation"]
        assert {r: a["sha256"] for r, a in p["artifacts"].items()} == {r: a["sha256"] for r, a in q["artifacts"].items()}


@pytest.mark.parametrize("status", ["failed", "timeout", "skipped", "not_applicable"])
def test_failed_or_skipped_check_is_not_an_empty_shared_result(packet_fixture, status):
    db, store, analysis, result, _, _ = paired(packet_fixture, second_status=status)
    comparison = packets.build_packet_set(db, analysis, store, result)["comparison"]
    assert comparison["status"] == "unavailable" and not comparison["artifacts"] and comparison["counts"] is None
    assert comparison["members"][1]["status"] in {"failed", "skipped"}


def test_withheld_is_not_a_negative_finding(packet_fixture):
    db, store, analysis, result, _, _ = paired(packet_fixture, withheld=True)
    comparison = packets.build_packet_set(db, analysis, store, result)["comparison"]
    assert comparison["status"] == "unavailable" and comparison["members"][1]["status"] == "withheld"
    assert comparison["counts"] is None


def test_missing_complement_and_incompatible_grids_are_unavailable(packet_fixture):
    b = np.zeros((6, 8), bool)
    b[2:4, 2:4] = True
    db, store, analysis, result, _, _ = paired(packet_fixture, b=b)
    comparison = packets.build_packet_set(db, analysis, store, result)["comparison"]
    assert comparison["status"] == "unavailable" and comparison["counts"] is None
    assert "grids differ" in " ".join(comparison["limitations"])
    result["signals"] = result["signals"][:1]
    analysis.result_hash = canonical_hash(result)
    comparison = packets.build_packet_set(db, analysis, store, result)["comparison"]
    assert comparison["members"][1]["status"] == "missing" and not comparison["artifacts"]


def test_component_cap_is_explicit_and_partial_plates_are_not_full_coverage(packet_fixture):
    a = np.zeros((12, 16), bool)
    a[1:3, 1:3] = a[5:7, 5:7] = a[9:11, 11:13] = True
    db, store, analysis, result, _, _ = paired(packet_fixture, a=a)
    comparison = packets.build_packet_set(db, analysis, store, result)["comparison"]
    assert comparison["status"] == "bounded" and comparison["members"][0]["omitted_components"] == 1
    assert comparison["counts"]["sensitive"] == 8


def test_bad_region_mapping_is_partial_not_an_all_clear(packet_fixture):
    db, store, analysis, result, _, _ = paired(packet_fixture, wrong_box=True)
    comparison = packets.build_packet_set(db, analysis, store, result)["comparison"]
    assert comparison["status"] == "partial" and comparison["members"][1]["selected_components"] == 0
    assert comparison["members"][1]["eligible_components"] == 1


def test_real_empty_measurements_do_not_clear_the_file(packet_fixture):
    db, store, analysis, result, _, _ = paired(packet_fixture, b=np.zeros((12, 16), bool))
    comparison = packets.build_packet_set(db, analysis, store, result)["comparison"]
    assert comparison["status"] == "unavailable"  # area gate withheld this input
    assert "not cleared as original" in comparison["interpretation"]


def test_rehashed_false_intersection_cannot_pass_asset_verification(packet_fixture):
    db, store, analysis, result, _, _ = paired(packet_fixture)
    document = packets.build_packet_set(db, analysis, store, result)
    changed = copy.deepcopy(document)
    changed["comparison"]["counts"].update(shared=0, wider=28)
    changed["sha256"] = packets.packet_hash(changed)
    packets.validate_packet_set(changed, analysis_id=analysis.id, revision=analysis.revision,
                                original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    with pytest.raises(packets.PacketIntegrityError, match="actual retained mask"):
        packets.validate_comparison_assets(changed, store)
    changed["comparison"]["members"][0]["exhibit_ref"] = "wrong-map"
    changed["sha256"] = packets.packet_hash(changed)
    with pytest.raises(packets.PacketIntegrityError, match="different measured map"):
        packets.validate_packet_set(changed, analysis_id=analysis.id, revision=analysis.revision,
                                    original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


@pytest.mark.parametrize("profile", ["standard", "forensic", "summary", "newsroom"])
def test_shared_masks_match_snapshot_json_report_assets_and_redaction(packet_fixture, monkeypatch, profile):
    from app.reports import engine, image_evidence
    from app.reports.public_view import public_manifest
    from app.reports.snapshot import dump, load
    from app.reports.specimens import specimen

    db, store, analysis, measured, _, _ = paired(packet_fixture)
    snapshot = specimen("image", profile)
    result = copy.deepcopy(snapshot.result)
    result["input"].update(measured["input"])
    result.update(signals=measured["signals"], exhibits=measured["exhibits"], findings=[], kind="image", status="done")
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    document = packets.build_packet_set(db, analysis, store, result)
    snapshot = snapshot.model_copy(update={"result": result, "image_evidence": document,
        "analysis": snapshot.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
        "media": snapshot.media.model_copy(update={"sha256": analysis.sha256})})
    monkeypatch.setattr(image_evidence, "default_store", lambda: store)
    view = image_evidence.view(snapshot, store=store)
    images = view["comparison"]["images"]
    assert {image["sha256"] for image in images} == {
        document["comparison"]["artifacts"][key]["sha256"] for key in ("wider_overlay", "shared_overlay", "wider_mask", "shared_mask")}
    assert all((image["uri"] is None) == (profile == "newsroom") for image in images)
    html = engine.render_html(snapshot, embed_fonts=False)
    assert "Wider view: 27 measured cells. Shared view: 1 measured cells." in html
    assert "Agreement does not establish AI origin" in html
    assert ('<img src="data:image/png;base64,' in html) == (profile != "newsroom")
    assert load(dump(snapshot)).image_evidence == document
    manifest = public_manifest(snapshot, "weasyprint test", engine.TEMPLATE_SET)["image_evidence"]
    assert {a["sha256"] for a in manifest["artifacts"]} == set(asset_refs(document))
