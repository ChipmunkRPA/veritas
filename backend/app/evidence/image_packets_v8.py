"""Append exact same-image differences to an immutable earlier evidence set."""
from __future__ import annotations

import json

import cv2
import numpy as np

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.image import embedded_pair
from app.evidence import image_embedded_pair_binding as binding
from app.evidence import image_embedded_pair_pixels as pixels
from app.evidence import image_interpolation_binding as interpolation
from app.evidence import image_packets_v7 as prior
from app.evidence.image_copy_similarity_binding import SKIPPED_CREDENTIALS, result_skip, url
from app.storage.artefacts import register_artefact
from app.storage.db import Artefact

CREATOR = "image-evidence-packets/13"
PacketIntegrityError = prior.PacketIntegrityError


def _active() -> bool:
    return settings.environment == "dev" and settings.image_embedded_pair_review


def current_set_id(analysis) -> str:
    return (prior.prior.current_set_id(analysis, binding.VERSION) if _active() else prior.current_set_id(analysis))


def validate_packet_set(document, **source_binding) -> None:
    try:
        binding.validate(document, prior.prior.packet_hash)
    except (KeyError, TypeError, ValueError) as exc:
        raise PacketIntegrityError(str(exc)) from exc
    if document.get("version") == binding.VERSION and document.get("sha256") != prior.prior.packet_hash(document):
        raise PacketIntegrityError("Exact two-area evidence differs from its recorded hash")
    prior.validate_packet_set(binding.base_document(document), **source_binding)


def validate_packet_source_mapping(document, result) -> None:
    try:
        binding.validate_scope(document, result_skip(result) is not None)
        record = interpolation.from_result(result)
        interpolation.validate(document, canonical_hash(record) if record is not None else None)
        if record is not None and (record["status"] == "provenance") != (result_skip(result) is not None):
            raise ValueError("Interpolation evidence differs from the frozen provenance disposition")
    except (ValueError, KeyError, TypeError) as exc:
        raise PacketIntegrityError(str(exc)) from exc
    prior.validate_packet_source_mapping(binding.base_document(document), result)


def validate_assets(document, store) -> None:
    prior.validate_assets(binding.base_document(document), store)
    for asset in interpolation.assets(document).values():
        store.verify(asset["sha256"])
    pair = (document.get(binding.KEY) or {}).get("pair")
    if pair is None:
        return
    source = pixels.source_image(prior.legacy._verified(store, document["source"]["sha256"]))
    if source is None:
        raise PacketIntegrityError("Exact two-area evidence uses a source outside the tested PNG geometry")
    with source:
        try:
            pixels.verify(source, pair, store.read_bytes)
        except (KeyError, TypeError, ValueError) as exc:
            raise PacketIntegrityError(str(exc)) from exc


def _box(bounds) -> dict:
    x, y, right, bottom = bounds
    return {"x": x, "y": y, "w": right - x, "h": bottom - y}


def _pair_record(db, analysis, store, document, source, pair) -> dict:
    areas = dict(zip(binding.ROLES, (pair.area_a, pair.area_b), strict=True))
    x, y, right, bottom = pair.difference_bounds
    changed = int(np.count_nonzero(pair.difference))
    a, b = (source.crop(areas[role]) for role in binding.ROLES)
    assets = {}
    for name, image in pixels.plates(a, b, pair.difference).items():
        saved = register_artefact(db, store, pixels.encode(image), media_type="image/png", role="preview",
                                  created_by=CREATOR, analysis_id=analysis.id)
        assets[name] = {"sha256": saved.sha256, "media_type": "image/png", "width": image.width,
                        "height": image.height, "url": url(document, saved.sha256)}
    return {"id": "D01", "regions": {role: _box(bounds) for role, bounds in areas.items()},
            "difference_regions": {role: _box((bounds[0] + x, bounds[1] + y, bounds[0] + right, bounds[1] + bottom))
                                   for role, bounds in areas.items()},
            "matching_pixels": pair.difference.size - changed, "different_pixels": changed,
            "witnesses": [list(point) for point in pair.witnesses], "artifacts": assets}


def _record(db, analysis, store, document, result) -> dict:
    record = {"schema": binding.SCHEMA, "source_sha256": analysis.sha256, "status": "outside_scope",
              "interpretation": binding.NOTE, "authorship_confidence_contribution": 0, "pair": None}
    if result_skip(result) is not None:
        return {**record, "status": SKIPPED_CREDENTIALS}
    size = document["source"]["width"], document["source"]["height"]
    if size not in binding.SIZES:
        return record
    source = pixels.source_image(prior.legacy._verified(store, analysis.sha256))
    if source is None:
        return record
    with source:
        try:
            status, pair = embedded_pair.inspect(np.asarray(source))
        except cv2.error:
            return {**record, "status": "unassessed_runtime"}
        record["status"] = status
        if pair is not None:
            record["pair"] = _pair_record(db, analysis, store, document, source, pair)
    return record


def build_packet_set(db, analysis, store, result):
    if not _active():
        return prior.build_packet_set(db, analysis, store, result)
    if canonical_hash(result) != analysis.result_hash or result.get("input", {}).get("sha256") != analysis.sha256:
        raise PacketIntegrityError("Exact two-area evidence requires the frozen result and original image")
    base = prior.persist_packet_set(db, analysis, store, result=result)
    document = {**base, "version": binding.VERSION, "id": current_set_id(analysis),
                binding.BASE: {field: base[field] for field in ("id", "version", "sha256")}}
    document[binding.KEY] = _record(db, analysis, store, document, result)
    document = json.loads(canonical_json(document))
    document["sha256"] = prior.prior.packet_hash(document)
    validate_packet_set(document, **prior.prior._binding(analysis))
    validate_packet_source_mapping(document, result)
    validate_assets(document, store)
    register_artefact(db, store, canonical_json(document), media_type="application/json", role="exhibit_payload",
                      created_by=CREATOR, analysis_id=analysis.id, artefact_id=document["id"])
    return document


def load_packet_set(db, analysis, store, *, set_id=None):
    expected = prior.prior.current_set_id(analysis, binding.VERSION)
    if set_id in (None, expected) and (row := db.get(Artefact, expected)) is not None:
        if (row.analysis_id != analysis.id or row.created_by != CREATOR or row.role != "exhibit_payload"
                or row.media_type != "application/json"):
            raise PacketIntegrityError("Exact two-area evidence has a different indexed identity")
        document = json.loads(prior.legacy._verified(store, row.sha256))
        if document.get("id") != expected or document.get("version") != binding.VERSION:
            raise PacketIntegrityError("Exact two-area evidence has a different immutable identity")
        validate_packet_set(document, **prior.prior._binding(analysis))
        result = json.loads(prior.legacy._verified(store, analysis.result_hash))
        validate_packet_source_mapping(document, result)
        validate_assets(document, store)
        return document
    document = prior.load_packet_set(db, analysis, store, set_id=set_id)
    if document is not None and interpolation.KEY in document:
        result = json.loads(prior.legacy._verified(store, analysis.result_hash))
        validate_packet_source_mapping(document, result)
        for asset in interpolation.assets(document).values():
            store.verify(asset["sha256"])
    return document


def persist_packet_set(db, analysis, store, *, result=None):
    if not _active():
        return prior.persist_packet_set(db, analysis, store, result=result)
    if analysis.kind != "image" or not analysis.result_hash:
        return None
    existing = load_packet_set(db, analysis, store, set_id=current_set_id(analysis))
    if existing is not None:
        return existing
    if result is None:
        result = json.loads(prior.legacy._verified(store, analysis.result_hash))
    return build_packet_set(db, analysis, store, result)


def unavailable_packet_set(analysis):
    document = prior.unavailable_packet_set(analysis)
    return {**document, "version": binding.VERSION} if _active() else document
