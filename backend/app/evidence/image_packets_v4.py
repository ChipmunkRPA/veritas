"""Successor image presentation: retained component unions and exact shared cells.

The v3 builder and numerical selection remain untouched. New immutable records
correct fitted/padded input wording and compare only the masks actually retained
in the evidence cards. Agreement is not an authorship probability.
"""
from __future__ import annotations

import copy
import io
import json
import uuid
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.evidence import image_packets as legacy
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact
from app.storage.objects import ObjectStore

VERSION = "image-evidence-4"
CREATOR = "image-evidence-packets/2"
SCHEMA = legacy.SCHEMA
PacketIntegrityError = legacy.PacketIntegrityError
packet_hash = legacy.packet_hash
_NS = uuid.UUID("a76a9056-ef61-412e-9621-31e4d34c2cfe")
_MEMBERS = (("sensitive", "Sensitive regional check", "loc.native_regional"),
            ("complementary", "Complementary regional check", "loc.splice"))
COMPARISON_NOTE = (
    "The wider view contains cells selected by either check; the shared view contains only cells selected by both. "
    "These are the retained component masks, not filled boxes or a new threshold. Checks can share errors. "
    "Agreement does not establish AI origin or increase confidence, and unmarked cells are not cleared as original.")


def current_set_id(analysis: Analysis) -> str:
    return str(uuid.uuid5(_NS, f"{analysis.id}:{analysis.result_hash}:{VERSION}"))


def _v3(document: dict) -> dict:
    check = copy.deepcopy(document)
    check["version"] = legacy.VERSION
    check["sha256"] = packet_hash(check)
    return check


def validate_packet_set(document: dict[str, Any], **binding: Any) -> None:
    if document.get("version") != VERSION:
        legacy.validate_packet_set(document, **binding)
        return
    if document.get("sha256") != packet_hash(document):
        raise PacketIntegrityError("Image comparison evidence has a different content hash")
    legacy.validate_packet_set(_v3(document), **binding)
    comparison = document.get("comparison")
    if (not isinstance(comparison, dict) or comparison.get("schema") != "retained-regional-comparison/1"
            or comparison.get("source_sha256") != document["source"]["sha256"]
            or comparison.get("status") not in {"ready", "bounded", "partial", "unavailable"}
            or comparison.get("authorship_confidence_contribution") != 0
            or comparison.get("selection") != "retained_component_masks_no_refilter"):
        raise PacketIntegrityError("Image comparison has an unsupported scope or rule")
    members = comparison.get("members")
    if not isinstance(members, list) or [m.get("id") for m in members] != [m[0] for m in _MEMBERS]:
        raise PacketIntegrityError("Image comparison has ambiguous member checks")
    coverage = {c["exhibit_ref"]: c for c in document["coverage"]["inspection_checks"]}
    for member in members:
        if member.get("status") not in {"measured", "withheld", "missing", "failed", "skipped", "unavailable"}:
            raise PacketIntegrityError("Image comparison has an invalid member outcome")
        if member["status"] not in {"measured", "withheld"}:
            continue
        check = coverage.get(member.get("exhibit_ref"))
        if check is None or member.get("scalar_map_sha256") != check["scalar_map_sha256"]:
            raise PacketIntegrityError("Image comparison is bound to a different measured map")
        if member.get("map_grid") != check.get("map_grid"):
            raise PacketIntegrityError("Image comparison uses a different measured grid")
        for key in ("eligible_components", "selected_components", "omitted_components"):
            if member.get(key) != check[key]:
                raise PacketIntegrityError("Image comparison differs from recorded component coverage")
        expected_packets = [p["id"] for p in document["packets"] if p.get("evidence_type") == "localisation"
                            and member["exhibit_ref"] in p.get("exhibit_refs", [])]
        if member.get("packet_ids") != expected_packets:
            raise PacketIntegrityError("Image comparison differs from the retained evidence cards")
    assets = comparison.get("artifacts")
    if not isinstance(assets, dict):
        raise PacketIntegrityError("Image comparison assets are missing")
    if comparison["status"] == "unavailable":
        if assets or comparison.get("counts") is not None:
            raise PacketIntegrityError("Unavailable image checks cannot supply a shared-mask result")
        return
    grid = comparison.get("map_grid")
    if (not isinstance(grid, list) or len(grid) != 2
            or any(type(v) is not int or v <= 0 for v in grid)
            or grid[0] * grid[1] > legacy.MAX_PIXELS
            or any(m["status"] != "measured" or m.get("map_grid") != grid for m in members)):
        raise PacketIntegrityError("Image comparison has incompatible measured grids")
    roles = {"sensitive_mask", "complementary_mask", "wider_mask", "shared_mask", "wider_overlay", "shared_overlay"}
    if set(assets) != roles:
        raise PacketIntegrityError("Image comparison is missing required presentation assets")
    for role, asset in assets.items():
        if asset != document["overview"].get(f"comparison_{role}"):
            raise PacketIntegrityError("Image comparison asset is not declared in this evidence set")
        if (asset.get("media_type") != "image/png" or not legacy._SHA.fullmatch(str(asset.get("sha256", "")))
                or (role.endswith("_mask") and [asset.get("width"), asset.get("height")] != grid)):
            raise PacketIntegrityError("Image comparison has an invalid mask asset")
    counts = comparison.get("counts")
    if (not isinstance(counts, dict) or set(counts) != {"sensitive", "complementary", "wider", "shared"}
            or any(type(v) is not int or not 0 <= v <= grid[0] * grid[1] for v in counts.values())
            or counts["wider"] != counts["sensitive"] + counts["complementary"] - counts["shared"]
            or counts["shared"] > min(counts["sensitive"], counts["complementary"])):
        raise PacketIntegrityError("Image comparison has contradictory selected-cell counts")
    expected_status = ("partial" if any(m["selected_components"] < min(m["eligible_components"], legacy.MAX_PER_MAP)
                                       for m in members) else
                       "bounded" if any(m["omitted_components"] for m in members) else "ready")
    if comparison["status"] != expected_status:
        raise PacketIntegrityError("Image comparison hides incomplete component coverage")


def validate_packet_source_mapping(document: dict, result: dict) -> None:
    # Bind to the canonical stored result, whose decimal presentation is also
    # used by the frozen packet builder. Exact gate values remain in hex fields.
    result = json.loads(canonical_json(result))
    legacy.validate_packet_source_mapping(document, result)
    if document.get("version") != VERSION:
        return
    maps = {e["id"]: e for e in result.get("exhibits", []) if e.get("kind") == "heatmap"}
    for member, (_, _, creator) in zip(document["comparison"]["members"], _MEMBERS, strict=True):
        if member["status"] in {"measured", "withheld"}:
            heat = maps.get(member["exhibit_ref"])
            if not heat or heat.get("created_by") != creator:
                raise PacketIntegrityError("Image comparison links the wrong source check")
    expected = _comparison(document, result)
    if any(document["comparison"].get(key) != expected[key] for key in ("members", "status", "map_grid")):
        raise PacketIntegrityError("Image comparison differs from the successful recorded source checks")


def _mask(store: ObjectStore, asset: dict) -> np.ndarray:
    with Image.open(io.BytesIO(legacy._verified(store, asset["sha256"]))) as image:
        if image.format != "PNG" or image.size != (asset["width"], asset["height"]):
            raise PacketIntegrityError("Image comparison mask dimensions differ from the frozen asset")
        values = np.asarray(image.convert("L"))
    if not np.isin(values, (0, 255)).all():
        raise PacketIntegrityError("Image comparison requires binary retained masks")
    return values > 0


def _unions(document: dict, store: ObjectStore) -> list[np.ndarray]:
    grid = document["comparison"]["map_grid"]
    packets = {p["id"]: p for p in document["packets"]}
    arrays = []
    for member in document["comparison"]["members"]:
        union = np.zeros(tuple(grid[::-1]), dtype=bool)
        for packet_id in member["packet_ids"]:
            component = _mask(store, packets[packet_id]["artifacts"]["mask"])
            if component.shape != union.shape:
                raise PacketIntegrityError("Retained image components use different grids")
            union |= component
        arrays.append(union)
    return arrays


def validate_comparison_assets(document: dict, store: ObjectStore) -> None:
    if document.get("version") != VERSION or document["comparison"]["status"] == "unavailable":
        return
    a, b = _unions(document, store)
    masks = {"sensitive": a, "complementary": b, "wider": a | b, "shared": a & b}
    for name, expected in masks.items():
        recorded = _mask(store, document["comparison"]["artifacts"][f"{name}_mask"])
        if (not np.array_equal(recorded, expected)
                or int(expected.sum()) != document["comparison"]["counts"][name]):
            raise PacketIntegrityError("Image comparison differs from the actual retained mask pixels")


def _geometry_note(policy: dict) -> str | None:
    fit, canvas = policy.get("fit_size"), policy.get("canvas_size")
    if (policy.get("padding_position") != "top_left" or policy.get("source_cropped") is not False
            or any(not isinstance(pair, list) or len(pair) != 2
                   or any(type(v) is not int or v <= 0 for v in pair) for pair in (fit, canvas))
            or any(f > c for f, c in zip(fit, canvas, strict=True))):
        return None
    return (f"The complete oriented image was fitted without stretching to {fit[0]} × {fit[1]} pixels, "
            f"placed at the top left of a {canvas[0]} × {canvas[1]} canvas, and padded on the right or bottom "
            "where needed. Padding is excluded from the source map. A larger displayed map does not add detection detail.")


def _comparison(document: dict, result: dict) -> dict:
    members = []
    for ident, label, creator in _MEMBERS:
        signals = [s for s in result.get("signals", []) if s.get("detector_id") == creator]
        maps = [e for e in result.get("exhibits", []) if e.get("kind") == "heatmap" and e.get("created_by") == creator]
        checks = [c for c in document["coverage"]["inspection_checks"] if c["exhibit_ref"] in {e["id"] for e in maps}]
        member: dict = {"id": ident, "label": label, "status": "unavailable", "packet_ids": []}
        if not signals:
            member["status"] = "missing"
        elif any(s.get("status", "ok") != "ok" for s in signals):
            statuses = {s.get("status") for s in signals}
            member["status"] = "failed" if statuses.intersection({"failed", "error", "timeout"}) else "skipped"
        elif len(checks) == 1:
            check = checks[0]
            gate = check.get("candidate_eligibility")
            member.update(status="withheld" if gate is not None and not gate["eligible"] else "measured",
                          exhibit_ref=check["exhibit_ref"], scalar_map_sha256=check["scalar_map_sha256"],
                          map_grid=check["map_grid"], eligible_components=check["eligible_components"],
                          selected_components=check["selected_components"], omitted_components=check["omitted_components"],
                          signal_refs=check["signal_refs"],
                          packet_ids=[p["id"] for p in document["packets"] if p.get("evidence_type") == "localisation"
                                      and check["exhibit_ref"] in p.get("exhibit_refs", [])])
        members.append(member)
    comparison = {"schema": "retained-regional-comparison/1", "status": "unavailable",
                  "source_sha256": document["source"]["sha256"], "members": members, "map_grid": None,
                  "selection": "retained_component_masks_no_refilter", "counts": None, "artifacts": {},
                  "authorship_confidence_contribution": 0, "interpretation": COMPARISON_NOTE,
                  "limitations": ["This compares the displayed component masks only. Per-map and total display caps still apply."]}
    if any(m["status"] != "measured" for m in members):
        comparison["limitations"].append("Both regional checks did not supply eligible measured masks. Missing, failed or withheld checks are not negative findings.")
    elif members[0]["map_grid"] != members[1]["map_grid"]:
        comparison["limitations"].append("The recorded grids differ; a shared-pixel comparison was not constructed.")
    else:
        comparison["status"] = ("partial" if any(m["selected_components"] < min(m["eligible_components"], legacy.MAX_PER_MAP)
                                                for m in members) else
                                "bounded" if any(m["omitted_components"] for m in members) else "ready")
        comparison["map_grid"] = members[0]["map_grid"]
        if comparison["status"] != "ready":
            comparison["limitations"].append("Some eligible regions are not included. The wider and shared views are restricted to retained cards and are not exhaustive.")
    return comparison


def _build_comparison_assets(db: Session, analysis: Analysis, store: ObjectStore, document: dict) -> None:
    comparison = document["comparison"]
    if comparison["status"] == "unavailable":
        return
    a, b = _unions(document, store)
    masks = {"sensitive": a, "complementary": b, "wider": a | b, "shared": a & b}
    comparison["counts"] = {key: int(mask.sum()) for key, mask in masks.items()}
    source, _ = legacy._source(legacy._verified(store, analysis.sha256))
    source.thumbnail((legacy.CONTEXT_MAX, legacy.CONTEXT_MAX), Image.Resampling.LANCZOS)
    images = {f"{key}_mask": Image.fromarray(mask.astype(np.uint8) * 255) for key, mask in masks.items()}
    for key, colour in (("wider", (244, 158, 45)), ("shared", (166, 110, 235))):
        alpha = Image.fromarray(masks[key].astype(np.uint8) * 150).resize(source.size, Image.Resampling.NEAREST)
        images[f"{key}_overlay"] = Image.composite(Image.new("RGB", source.size, colour), source, alpha)
    for role, image in images.items():
        art = register_artefact(db, store, legacy._png(image), media_type="image/png", role="preview",
                                created_by=CREATOR, analysis_id=analysis.id)
        asset = {"sha256": art.sha256, "media_type": "image/png", "width": image.width, "height": image.height,
                 "url": f"/api/v2/analyses/{analysis.id}/evidence-packets/{document['id']}/images/{art.sha256}.png"}
        comparison["artifacts"][role] = asset
        document["overview"][f"comparison_{role}"] = asset


def build_packet_set(db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any]) -> dict[str, Any]:
    if (canonical_hash(result) != analysis.result_hash
            or result.get("input", {}).get("sha256") != analysis.sha256):
        raise PacketIntegrityError("Published result or source identity does not match")
    base = legacy.persist_packet_set(db, analysis, store, result=result)
    if base is None:
        raise ValueError("Image evidence needs a published image result")
    document = copy.deepcopy(base)
    document.update(id=current_set_id(analysis), version=VERSION,
                    derived_from={"id": base["id"], "version": base["version"], "sha256": base["sha256"]})
    for packet in document["packets"]:
        packet["derivation"]["version"] = VERSION
        policy = packet.get("localisation", {}).get("resolution_policy", {})
        note = _geometry_note(policy)
        if note:
            old = (f"The whole image was resized to {policy.get('input_px')} × {policy.get('input_px')} "
                   "for this check; a larger displayed map does not add detection detail.")
            packet["limitations"] = [note if line == old else line for line in packet["limitations"]]
    assets = [*(document.get("overview") or {}).values(),
              *(art for packet in document["packets"] for art in packet["artifacts"].values())]
    for asset in assets:
        asset["url"] = f"/api/v2/analyses/{analysis.id}/evidence-packets/{document['id']}/images/{asset['sha256']}.png"
    document["comparison"] = _comparison(document, result)
    _build_comparison_assets(db, analysis, store, document)
    document = json.loads(canonical_json(document))
    document["sha256"] = packet_hash(document)
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    validate_packet_source_mapping(document, result)
    validate_comparison_assets(document, store)
    register_artefact(db, store, canonical_json(document), media_type="application/json", role="exhibit_payload",
                      created_by=CREATOR, analysis_id=analysis.id, artefact_id=document["id"])
    return document


def load_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, set_id: str | None = None) -> dict[str, Any] | None:
    query = select(Artefact).where(Artefact.analysis_id == analysis.id, Artefact.created_by == CREATOR,
                                   Artefact.role == "exhibit_payload", Artefact.media_type == "application/json")
    if set_id is not None:
        query = query.where(Artefact.id == set_id)
    row = db.scalars(query.order_by(Artefact.created_at.desc(), Artefact.id)).first()
    if row is None:
        return legacy.load_packet_set(db, analysis, store, set_id=set_id)
    document = json.loads(legacy._verified(store, row.sha256))
    if document.get("id") != row.id or document.get("version") != VERSION:
        raise PacketIntegrityError("Image comparison evidence differs from its indexed identity")
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    result = json.loads(legacy._verified(store, analysis.result_hash))
    validate_packet_source_mapping(document, result)
    validate_comparison_assets(document, store)
    store.verify(analysis.sha256)
    return document


def persist_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if analysis.kind != "image" or not analysis.result_hash:
        return None
    existing = load_packet_set(db, analysis, store, set_id=current_set_id(analysis))
    if existing is not None:
        return existing
    if result is None:
        result = json.loads(legacy._verified(store, analysis.result_hash))
    return build_packet_set(db, analysis, store, result)


def unavailable_packet_set(analysis: Analysis) -> dict[str, Any]:
    return {**legacy.unavailable_packet_set(analysis), "version": VERSION}
