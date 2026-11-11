"""Append automatic, undirected copy similarity to a frozen v6 image evidence set."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.image import copy_similarity
from app.evidence import image_packets_v5 as prior
from app.evidence.image_copy_similarity_binding import (
    NOTE,
    ROLES,
    SCHEMA,
    VERSION,
    box_bounds,
    prior_evidence,
    url,
    validate_similarity,
)
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact
from app.storage.objects import ObjectStore

CREATOR = "image-evidence-packets/5"
legacy = prior.legacy
PacketIntegrityError = prior.PacketIntegrityError


def current_set_id(analysis: Analysis) -> str:
    return prior.current_set_id(analysis, VERSION)


def validate_packet_set(document: dict[str, Any], **binding: Any) -> None:
    try:
        validate_similarity(document, canonical_hash)
    except (KeyError, TypeError, ValueError) as exc:
        raise PacketIntegrityError(str(exc)) from exc
    if document.get("version") != VERSION:
        prior.validate_packet_set(document, **binding)
        return
    if document.get("sha256") != prior.packet_hash(document):
        raise PacketIntegrityError("Copy similarity evidence has a different content hash")
    prior.validate_packet_set(prior_evidence(document), **binding)


def validate_packet_source_mapping(document: dict, result: dict) -> None:
    prior.validate_packet_source_mapping(
        prior_evidence(document) if document.get("version") == VERSION else document, result
    )


def _source(document: dict, store: ObjectStore) -> Image.Image:
    source, orientation = legacy._source(legacy._verified(store, document["source"]["sha256"]))
    if (
        source.size != (document["source"]["width"], document["source"]["height"])
        or orientation != document["source"]["exif_orientation"]
    ):
        raise PacketIntegrityError("Copy similarity uses different source geometry")
    return source


def validate_assets(document: dict, store: ObjectStore) -> None:
    if document.get("version") != VERSION:
        prior.validate_assets(document, store)
        return
    prior.validate_assets(prior_evidence(document), store)
    pairs = document["copy_similarity"]["pairs"]
    if not pairs:
        return
    source = _source(document, store)
    for pair in pairs:
        for role in ROLES:
            expected = source.crop(box_bounds(pair["regions"][role], *source.size))
            with Image.open(io.BytesIO(legacy._verified(store, pair["artifacts"][role]["sha256"]))) as crop:
                if (
                    crop.format != "PNG"
                    or crop.mode != "RGB"
                    or crop.size != expected.size
                    or crop.tobytes() != expected.tobytes()
                ):
                    raise PacketIntegrityError("Copy similarity crop pixels differ from the submitted image")


def build_packet_set(
    db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any]
) -> dict[str, Any]:
    if (
        canonical_hash(result) != analysis.result_hash
        or result.get("input", {}).get("sha256") != analysis.sha256
    ):
        raise PacketIntegrityError("Published result or source identity does not match")
    base = prior.load_packet_set(db, analysis, store, set_id=prior.current_set_id(analysis))
    if base is None:
        base = prior.build_packet_set(db, analysis, store, result)
    document = {
        **base,
        "id": current_set_id(analysis),
        "version": VERSION,
        "prior_image_evidence": {key: base[key] for key in ("id", "version", "sha256")},
    }
    size = document["source"]["width"], document["source"]["height"]
    status, pairs = "unsupported_geometry", []
    if copy_similarity.supported_size(*size):
        source = _source(document, store)
        status, boxes = copy_similarity.analyse(np.asarray(source))
        for index, areas in enumerate(boxes, 1):
            regions, artifacts = {}, {}
            for role, box in zip(ROLES, areas, strict=True):
                region = prior._coordinates(box)
                crop = source.crop(box_bounds(region, *size))
                art = register_artefact(
                    db,
                    store,
                    legacy._png(crop),
                    media_type="image/png",
                    role="preview",
                    created_by=CREATOR,
                    analysis_id=analysis.id,
                )
                regions[role] = region
                artifacts[role] = {
                    "sha256": art.sha256,
                    "media_type": "image/png",
                    "width": crop.width,
                    "height": crop.height,
                    "url": url(document, art.sha256),
                }
            pairs.append(
                {
                    "id": f"R{index:02d}",
                    "transform": "horizontal_reflection",
                    "regions": regions,
                    "artifacts": artifacts,
                }
            )
    document["copy_similarity"] = {
        "schema": SCHEMA,
        "source_sha256": analysis.sha256,
        "coordinate_space": "exif-oriented-original",
        "status": status,
        "authorship_confidence_contribution": 0,
        "pairs": pairs,
        "interpretation": NOTE,
    }
    document = json.loads(canonical_json(document))
    document["sha256"] = prior.packet_hash(document)
    validate_packet_set(document, **prior._binding(analysis))
    validate_packet_source_mapping(document, result)
    validate_assets(document, store)
    register_artefact(
        db,
        store,
        canonical_json(document),
        media_type="application/json",
        role="exhibit_payload",
        created_by=CREATOR,
        analysis_id=analysis.id,
        artefact_id=document["id"],
    )
    return document


def load_packet_set(
    db: Session, analysis: Analysis, store: ObjectStore, *, set_id: str | None = None
) -> dict[str, Any] | None:
    expected = current_set_id(analysis)
    row = db.get(Artefact, expected) if set_id is None or set_id == expected else None
    if row is None:
        return prior.load_packet_set(db, analysis, store, set_id=set_id)
    if (
        row.analysis_id != analysis.id
        or row.created_by != CREATOR
        or row.role != "exhibit_payload"
        or row.media_type != "application/json"
    ):
        raise PacketIntegrityError("Copy similarity evidence has a different indexed identity")
    document = json.loads(legacy._verified(store, row.sha256))
    if document.get("id") != expected or document.get("version") != VERSION:
        raise PacketIntegrityError("Copy similarity evidence has a different immutable identity")
    validate_packet_set(document, **prior._binding(analysis))
    validate_packet_source_mapping(document, json.loads(legacy._verified(store, analysis.result_hash)))
    validate_assets(document, store)
    store.verify(analysis.sha256)
    return document


def persist_packet_set(
    db: Session, analysis: Analysis, store: ObjectStore, *, result: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if analysis.kind != "image" or not analysis.result_hash:
        return None
    existing = load_packet_set(db, analysis, store, set_id=current_set_id(analysis))
    if existing is not None:
        return existing
    if result is None:
        result = json.loads(legacy._verified(store, analysis.result_hash))
    return build_packet_set(db, analysis, store, result)


def unavailable_packet_set(analysis: Analysis) -> dict[str, Any]:
    return {**prior.unavailable_packet_set(analysis), "version": VERSION}
