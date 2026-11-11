"""Append source-bound copy similarity checks to a frozen v6 image evidence set."""

from __future__ import annotations

import io
import json
import math
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.image import copy_move, copy_similarity
from app.evidence import image_packets_v5 as prior
from app.evidence.image_copy_similarity_binding import (
    NOTE,
    PAIR_LIMIT,
    REFLECTION_VERSIONS,
    ROLES,
    SCHEMA,
    SKIPPED_CREDENTIALS,
    TRANSLATION_NOTE,
    TRANSLATION_SCHEMA,
    V7,
    V8,
    V9,
    V10,
    V11,
    VERSION,
    box_bounds,
    copy_records,
    prior_evidence,
    result_skip,
    supported_size,
    url,
    validate_copy_scope,
    validate_similarity,
)
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact
from app.storage.objects import ObjectStore

CREATOR = "image-evidence-packets/10"
_CREATORS = {VERSION: CREATOR, V11: "image-evidence-packets/9", V10: "image-evidence-packets/8", V9: "image-evidence-packets/7",
             V8: "image-evidence-packets/6", V7: "image-evidence-packets/5"}
MAX_COPY_CANDIDATES = 1024  # Each pair adds two orientations to the neighbor graph.
legacy = prior.legacy
PacketIntegrityError = prior.PacketIntegrityError


def current_set_id(analysis: Analysis) -> str:
    return prior.current_set_id(analysis, VERSION)


def validate_packet_set(document: dict[str, Any], **binding: Any) -> None:
    try:
        validate_similarity(document, canonical_hash)
    except (KeyError, TypeError, ValueError) as exc:
        raise PacketIntegrityError(str(exc)) from exc
    if document.get("version") not in REFLECTION_VERSIONS:
        prior.validate_packet_set(document, **binding)
        return
    if document.get("sha256") != prior.packet_hash(document):
        raise PacketIntegrityError("Copy similarity evidence has a different content hash")
    prior.validate_packet_set(prior_evidence(document), **binding)


def validate_packet_source_mapping(document: dict, result: dict) -> None:
    if document.get("version") == VERSION:
        try:
            validate_copy_scope(document, result_skip(result))
        except ValueError as exc:
            raise PacketIntegrityError(str(exc)) from exc
    prior.validate_packet_source_mapping(
        prior_evidence(document) if document.get("version") in REFLECTION_VERSIONS else document, result
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
    if document.get("version") not in REFLECTION_VERSIONS:
        prior.validate_assets(document, store)
        return
    prior.validate_assets(prior_evidence(document), store)
    pairs = [pair for record in copy_records(document) for pair in record["pairs"]]
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


def _crop_box(box: copy_move.Box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    left, top = math.floor(x), math.floor(y)
    return left, top, min(width, math.ceil(x + w)) - left, min(height, math.ceil(y + h)) - top


def translation_pairs(rgb: np.ndarray) -> tuple[str, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    height, width = rgb.shape[:2]
    try:
        result = copy_move.analyse(rgb, max_candidate_pairs=MAX_COPY_CANDIDATES)
    except copy_move.CandidateBudgetExceeded:
        return "unassessed_match_budget", []
    pairs = []
    for pair in result.distinct:
        first = _crop_box(pair.box_a, width, height)
        second = _crop_box(pair.box_b, width, height)
        if not pair.affine.reflected and min(*first[2:], *second[2:]) > 0 and copy_move.iou(first, second) == 0:
            pairs.append((first, second))
    return ("supported_pair" if pairs else "no_pair"), pairs[:PAIR_LIMIT]


def selected_translation_pairs(rgb: np.ndarray) -> tuple[str, list[tuple[copy_move.Box, copy_move.Box]], str | None]:
    status, pairs = copy_similarity.analyse_translation(rgb)
    if status != "no_pair":
        return status, pairs, "native_patch_alignment" if pairs else None
    status, pairs = translation_pairs(rgb)
    return status, pairs, "geometric_features" if pairs else None


def _copy_record(db: Session, analysis: Analysis, store: ObjectStore, document: dict, source: Image.Image | None,
                 status: str, boxes: list, *, reflected: bool) -> dict:
    pairs = []
    for index, areas in enumerate(boxes, 1):
        if source is None:
            raise PacketIntegrityError("Copy similarity evidence is missing its source image")
        regions, artifacts = {}, {}
        for role, box in zip(ROLES, areas, strict=True):
            region = prior._coordinates(box)
            crop = source.crop(box_bounds(region, *source.size))
            art = register_artefact(db, store, legacy._png(crop), media_type="image/png", role="preview",
                                    created_by=CREATOR, analysis_id=analysis.id)
            regions[role] = region
            artifacts[role] = {"sha256": art.sha256, "media_type": "image/png", "width": crop.width,
                               "height": crop.height, "url": url(document, art.sha256)}
        pairs.append({"id": f"{'R' if reflected else 'T'}{index:02d}",
                      "transform": "horizontal_reflection" if reflected else "same_orientation",
                      "regions": regions, "artifacts": artifacts})
    return {"schema": SCHEMA if reflected else TRANSLATION_SCHEMA,
            "source_sha256": analysis.sha256, "coordinate_space": "exif-oriented-original", "status": status,
            "authorship_confidence_contribution": 0, "pairs": pairs, "interpretation": NOTE if reflected else TRANSLATION_NOTE}


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
    status, boxes = "unsupported_geometry", []
    translation_status, translations, basis, source = "unsupported_geometry", [], None, None
    if result_skip(result):
        status = translation_status = SKIPPED_CREDENTIALS
    elif copy_similarity.supported_size(*size):
        source = _source(document, store)
        rgb = np.asarray(source)
        status, boxes = copy_similarity.analyse(rgb)
        if supported_size(*size, version=V8):
            translation_status, translations, basis = selected_translation_pairs(rgb)
    document["copy_similarity"] = _copy_record(db, analysis, store, document, source, status, boxes, reflected=True)
    document["copy_translation"] = _copy_record(db, analysis, store, document, source, translation_status,
                                                translations, reflected=False)
    document["copy_translation"]["basis"] = basis
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
    for version, creator in _CREATORS.items():
        expected = prior.current_set_id(analysis, version)
        if set_id is not None and set_id != expected:
            continue
        row = db.get(Artefact, expected)
        if row is None:
            continue
        if (
            row.analysis_id != analysis.id
            or row.created_by != creator
            or row.role != "exhibit_payload"
            or row.media_type != "application/json"
        ):
            raise PacketIntegrityError("Copy similarity evidence has a different indexed identity")
        document = json.loads(legacy._verified(store, row.sha256))
        if document.get("id") != expected or document.get("version") != version:
            raise PacketIntegrityError("Copy similarity evidence has a different immutable identity")
        validate_packet_set(document, **prior._binding(analysis))
        validate_packet_source_mapping(document, json.loads(legacy._verified(store, analysis.result_hash)))
        validate_assets(document, store)
        store.verify(analysis.sha256)
        return document
    return prior.load_packet_set(db, analysis, store, set_id=set_id)


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
