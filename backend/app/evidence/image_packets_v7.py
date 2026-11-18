"""Append source-bound copy similarity checks to a frozen v6 image evidence set."""

from __future__ import annotations

import io
import json
import math
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.image import copy_affine, copy_affine_large, copy_move, copy_similarity
from app.evidence import image_copy_affine_binding as affine
from app.evidence import image_packets_v5 as prior
from app.evidence.image_copy_similarity_binding import (
    AFFINE_VERSIONS,
    CREDENTIAL_VERSIONS,
    DIAGONAL_EXACT_VERSIONS,
    DIAGONAL_TRANSFORMS,
    DISJOINT_JPEG_ROTATION_AFFINE_VERSION,
    DISJOINT_JPEG_ROTATION_VERSION,
    DISJOINT_JPEG_ROTATION_VERSIONS,
    EXACT_DIAGONAL_NOTE,
    EXACT_REFLECTION_NOTE,
    EXACT_REFLECTION_VERSIONS,
    EXPANDED_JPEG_ROTATION_AFFINE_VERSION,
    EXPANDED_JPEG_ROTATION_VERSION,
    EXPANDED_JPEG_ROTATION_VERSIONS,
    JPEG_BLOCK_AFFINE_VERSION,
    JPEG_BLOCK_BASIS,
    JPEG_BLOCK_VERSION,
    JPEG_BLOCK_VERSIONS,
    JPEG_OFFGRID_AFFINE_VERSION,
    JPEG_OFFGRID_VERSION,
    JPEG_OFFGRID_VERSIONS,
    MULTI_EXACT_BASIS,
    MULTI_EXACT_KEY,
    MULTI_EXACT_NOTE,
    MULTI_EXACT_VERSIONS,
    NATIVE_JPEG_ROTATION_AFFINE_VERSION,
    NATIVE_JPEG_ROTATION_VERSION,
    NATIVE_JPEG_ROTATION_VERSIONS,
    NOTE,
    OFFGRID_BASIS,
    PAIR_LIMIT,
    PNG_COPY_TRANSFORMS,
    PNG_DIAGONAL_EXACT_VERSION,
    PNG_EXACT_REFLECTION_AFFINE_VERSION,
    PNG_EXACT_REFLECTION_BASIS,
    PNG_EXACT_REFLECTION_KEY,
    PNG_EXACT_REFLECTION_VERSION,
    PNG_MULTI_EXACT_AFFINE_VERSION,
    PNG_MULTI_EXACT_VERSION,
    PNG_THREE_EXACT_AFFINE_VERSION,
    PNG_THREE_EXACT_VERSION,
    REFLECTION_VERSIONS,
    ROLES,
    SCHEMA,
    SKIPPED_CREDENTIALS,
    SMALL_PNG_BASIS,
    SMALL_PNG_COPY_AFFINE_VERSION,
    SMALL_PNG_COPY_VERSION,
    SMALL_PNG_VERSIONS,
    THREE_EXACT_VERSIONS,
    TRANSLATION_NOTE,
    TRANSLATION_SCHEMA,
    TURNED_JPEG_BASIS,
    TURNED_JPEG_COPY_AFFINE_VERSION,
    TURNED_JPEG_COPY_VERSION,
    TURNED_JPEG_VERSIONS,
    TURNED_PNG_BASIS,
    TURNED_PNG_COPY_AFFINE_VERSION,
    TURNED_PNG_COPY_VERSION,
    TURNED_PNG_VERSIONS,
    V7,
    V8,
    V9,
    V10,
    V11,
    VERSION,
    WEBP_DIAGONAL_EXACT_VERSION,
    WEBP_EXACT_COPY_AFFINE_VERSION,
    WEBP_EXACT_COPY_BASIS,
    WEBP_EXACT_COPY_KEY,
    WEBP_EXACT_COPY_VERSION,
    WEBP_EXACT_COPY_VERSIONS,
    WEBP_EXACT_REFLECTION_AFFINE_VERSION,
    WEBP_EXACT_REFLECTION_BASIS,
    WEBP_EXACT_REFLECTION_KEY,
    WEBP_EXACT_REFLECTION_VERSION,
    WEBP_MULTI_EXACT_AFFINE_VERSION,
    WEBP_MULTI_EXACT_VERSION,
    WEBP_THREE_EXACT_AFFINE_VERSION,
    WEBP_THREE_EXACT_VERSION,
    box_bounds,
    copy_records,
    exact_copy_size,
    native_rotation_size,
    prior_evidence,
    result_skip,
    rotation_review_follows,
    supported_size,
    url,
    validate_copy_scope,
    validate_similarity,
)
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact
from app.storage.objects import ObjectStore

CREATOR = "image-evidence-packets/10"
_CREATORS = {WEBP_DIAGONAL_EXACT_VERSION: "image-evidence-packets/45",
             PNG_DIAGONAL_EXACT_VERSION: "image-evidence-packets/44",
             WEBP_THREE_EXACT_AFFINE_VERSION: "image-evidence-packets/43", WEBP_THREE_EXACT_VERSION: "image-evidence-packets/42",
             PNG_THREE_EXACT_AFFINE_VERSION: "image-evidence-packets/41", PNG_THREE_EXACT_VERSION: "image-evidence-packets/40",
             WEBP_MULTI_EXACT_AFFINE_VERSION: "image-evidence-packets/39", WEBP_MULTI_EXACT_VERSION: "image-evidence-packets/38",
             PNG_MULTI_EXACT_AFFINE_VERSION: "image-evidence-packets/37", PNG_MULTI_EXACT_VERSION: "image-evidence-packets/36",
             PNG_EXACT_REFLECTION_AFFINE_VERSION: "image-evidence-packets/35",
             PNG_EXACT_REFLECTION_VERSION: "image-evidence-packets/34",
             WEBP_EXACT_REFLECTION_AFFINE_VERSION: "image-evidence-packets/33",
             WEBP_EXACT_REFLECTION_VERSION: "image-evidence-packets/32",
             WEBP_EXACT_COPY_AFFINE_VERSION: "image-evidence-packets/31",
             WEBP_EXACT_COPY_VERSION: "image-evidence-packets/30",
             DISJOINT_JPEG_ROTATION_AFFINE_VERSION: "image-evidence-packets/29",
             DISJOINT_JPEG_ROTATION_VERSION: "image-evidence-packets/28",
             NATIVE_JPEG_ROTATION_AFFINE_VERSION: "image-evidence-packets/27",
             NATIVE_JPEG_ROTATION_VERSION: "image-evidence-packets/26",
             EXPANDED_JPEG_ROTATION_AFFINE_VERSION: "image-evidence-packets/25",
             EXPANDED_JPEG_ROTATION_VERSION: "image-evidence-packets/24",
             TURNED_JPEG_COPY_AFFINE_VERSION: "image-evidence-packets/23",
             TURNED_JPEG_COPY_VERSION: "image-evidence-packets/22",
             TURNED_PNG_COPY_AFFINE_VERSION: "image-evidence-packets/21", TURNED_PNG_COPY_VERSION: "image-evidence-packets/20",
             SMALL_PNG_COPY_AFFINE_VERSION: "image-evidence-packets/19", SMALL_PNG_COPY_VERSION: "image-evidence-packets/18",
             JPEG_OFFGRID_AFFINE_VERSION: "image-evidence-packets/17", JPEG_OFFGRID_VERSION: "image-evidence-packets/16",
             JPEG_BLOCK_AFFINE_VERSION: "image-evidence-packets/15", JPEG_BLOCK_VERSION: "image-evidence-packets/14",
             affine.LARGE_VERSION: "image-evidence-packets/12", affine.VERSION: "image-evidence-packets/11",
             VERSION: CREATOR, V11: "image-evidence-packets/9",
             V10: "image-evidence-packets/8", V9: "image-evidence-packets/7",
             V8: "image-evidence-packets/6", V7: "image-evidence-packets/5"}
MAX_COPY_CANDIDATES = 1024  # Each pair adds two orientations to the neighbor graph.
legacy = prior.legacy
PacketIntegrityError = prior.PacketIntegrityError


def _active_version() -> str:
    if settings.environment != "dev":
        return VERSION
    if settings.image_jpeg_offgrid_copy_review:
        return JPEG_OFFGRID_AFFINE_VERSION if settings.image_affine_copy_review else JPEG_OFFGRID_VERSION
    if settings.image_jpeg_block_copy_review:
        return JPEG_BLOCK_AFFINE_VERSION if settings.image_affine_copy_review else JPEG_BLOCK_VERSION
    return affine.LARGE_VERSION if settings.image_affine_copy_review else VERSION


def _analysis_version(analysis: Analysis) -> str:
    if settings.environment == "dev" and settings.image_disjoint_exact_copy_review:
        if analysis.content_type == "image/png":
            if settings.image_three_exact_copy_review:
                if settings.image_diagonal_exact_copy_review and settings.image_affine_copy_review:
                    return PNG_DIAGONAL_EXACT_VERSION
                return PNG_THREE_EXACT_AFFINE_VERSION if settings.image_affine_copy_review else PNG_THREE_EXACT_VERSION
            return PNG_MULTI_EXACT_AFFINE_VERSION if settings.image_affine_copy_review else PNG_MULTI_EXACT_VERSION
        if analysis.content_type == "image/webp":
            if settings.image_three_exact_copy_review:
                if settings.image_diagonal_exact_copy_review and settings.image_affine_copy_review:
                    return WEBP_DIAGONAL_EXACT_VERSION
                return WEBP_THREE_EXACT_AFFINE_VERSION if settings.image_affine_copy_review else WEBP_THREE_EXACT_VERSION
            return WEBP_MULTI_EXACT_AFFINE_VERSION if settings.image_affine_copy_review else WEBP_MULTI_EXACT_VERSION
    if settings.environment == "dev" and settings.image_png_reflection_review and analysis.content_type == "image/png":
        return PNG_EXACT_REFLECTION_AFFINE_VERSION if settings.image_affine_copy_review else PNG_EXACT_REFLECTION_VERSION
    if settings.environment == "dev" and settings.image_lossless_webp_reflection_review and analysis.content_type == "image/webp":
        return WEBP_EXACT_REFLECTION_AFFINE_VERSION if settings.image_affine_copy_review else WEBP_EXACT_REFLECTION_VERSION
    if settings.environment == "dev" and settings.image_lossless_webp_copy_review and analysis.content_type == "image/webp":
        return WEBP_EXACT_COPY_AFFINE_VERSION if settings.image_affine_copy_review else WEBP_EXACT_COPY_VERSION
    if settings.environment == "dev" and settings.image_turned_jpeg_copy_review and analysis.content_type == "image/jpeg":
        if settings.image_jpeg_native_rotation_review:
            if settings.image_jpeg_disjoint_rotation_review:
                return DISJOINT_JPEG_ROTATION_AFFINE_VERSION if settings.image_affine_copy_review else DISJOINT_JPEG_ROTATION_VERSION
            return NATIVE_JPEG_ROTATION_AFFINE_VERSION if settings.image_affine_copy_review else NATIVE_JPEG_ROTATION_VERSION
        return EXPANDED_JPEG_ROTATION_AFFINE_VERSION if settings.image_affine_copy_review else EXPANDED_JPEG_ROTATION_VERSION
    if settings.environment == "dev" and settings.image_small_png_copy_review and analysis.content_type == "image/png":
        if settings.image_turned_png_copy_review:
            return TURNED_PNG_COPY_AFFINE_VERSION if settings.image_affine_copy_review else TURNED_PNG_COPY_VERSION
        return SMALL_PNG_COPY_AFFINE_VERSION if settings.image_affine_copy_review else SMALL_PNG_COPY_VERSION
    return _active_version()


def current_set_id(analysis: Analysis) -> str:
    return prior.current_set_id(analysis, _analysis_version(analysis))


def validate_packet_set(document: dict[str, Any], **binding: Any) -> None:
    try:
        affine.validate(document, box_bounds=box_bounds, url=url, credential_status=SKIPPED_CREDENTIALS)
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
    if document.get("version") in CREDENTIAL_VERSIONS:
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
    for pair in (document.get(affine.KEY) or {}).get("pairs", []):
        for role in ROLES:
            asset = pair["artifacts"]["mask_" + role[-1]]
            with Image.open(io.BytesIO(legacy._verified(store, asset["sha256"]))) as mask:
                pixels = np.asarray(mask)
                if (mask.format != "PNG" or mask.mode != "L" or mask.size != (asset["width"], asset["height"])
                        or not np.all((pixels == 0) | (pixels == 255))
                        or not all(edge.any() for edge in (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
                        or int(np.count_nonzero(pixels)) != pair["selected_pixels"][role]):
                    raise PacketIntegrityError("Geometric copy mask differs from its recorded geometry")
    translation = document.get("copy_translation") or {}
    if translation.get("basis") == MULTI_EXACT_BASIS:
        from app.evidence.image_exact_small_copy_source import verify_multiple

        try:
            verify_multiple(legacy._verified(store, document["source"]["sha256"]), translation,
                            webp=document["version"] in WEBP_EXACT_COPY_VERSIONS)
        except (OSError, ValueError) as error:
            raise PacketIntegrityError(str(error)) from error
    if translation.get("basis") == JPEG_BLOCK_BASIS:
        from app.evidence.image_jpeg_copy_blocks_source import verify

        try:
            verify(legacy._verified(store, document["source"]["sha256"]), translation["pairs"])
        except (OSError, ValueError) as error:
            raise PacketIntegrityError(str(error)) from error
    if translation.get("basis") in {SMALL_PNG_BASIS, TURNED_PNG_BASIS, PNG_EXACT_REFLECTION_BASIS,
                                    WEBP_EXACT_COPY_BASIS, WEBP_EXACT_REFLECTION_BASIS}:
        from app.evidence.image_exact_small_copy_source import verify

        basis = translation["basis"]
        reflected = basis in {PNG_EXACT_REFLECTION_BASIS, WEBP_EXACT_REFLECTION_BASIS}
        webp = basis in {WEBP_EXACT_COPY_BASIS, WEBP_EXACT_REFLECTION_BASIS}
        key = {WEBP_EXACT_REFLECTION_BASIS: WEBP_EXACT_REFLECTION_KEY, WEBP_EXACT_COPY_BASIS: WEBP_EXACT_COPY_KEY,
               PNG_EXACT_REFLECTION_BASIS: PNG_EXACT_REFLECTION_KEY}.get(basis, "small_png_copy_review")
        try:
            verify(legacy._verified(store, document["source"]["sha256"]), translation["pairs"][0],
                   translation[key]["measurement"], webp=webp, reflected=reflected,
                   include_diagonal=document["version"] in DIAGONAL_EXACT_VERSIONS)
        except (OSError, ValueError) as error:
            raise PacketIntegrityError(str(error)) from error
    if translation.get("basis") == OFFGRID_BASIS:
        from app.evidence.image_jpeg_offgrid_source import verify

        try:
            verify(legacy._verified(store, document["source"]["sha256"]), np.asarray(source), translation["pairs"][0],
                   translation["jpeg_offgrid_review"]["measurement"])
        except (OSError, ValueError) as error:
            raise PacketIntegrityError(str(error)) from error
    if translation.get("basis") == TURNED_JPEG_BASIS:
        from app.evidence.image_jpeg_turned_copy_source import verify_pairs

        try:
            verify_pairs(legacy._verified(store, document["source"]["sha256"]), np.asarray(source), translation,
                         expanded=document["version"] in EXPANDED_JPEG_ROTATION_VERSIONS,
                         native=document["version"] in NATIVE_JPEG_ROTATION_VERSIONS and native_rotation_size(*source.size))
        except (OSError, ValueError) as error:
            raise PacketIntegrityError(str(error)) from error


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
                 status: str, boxes: list, *, reflected: bool, turns: int | tuple[int, ...] = 0,
                 exact_reflection: str | None = None, exact_transforms: tuple[str, ...] = ()) -> dict:
    pairs = []
    rotations = (turns,) * len(boxes) if isinstance(turns, int) else turns
    transforms = exact_transforms or tuple(exact_reflection or ("horizontal_reflection" if reflected else PNG_COPY_TRANSFORMS[rotation])
                                          for rotation in rotations)
    reflection_note = EXACT_DIAGONAL_NOTE if exact_reflection in DIAGONAL_TRANSFORMS else EXACT_REFLECTION_NOTE
    for index, (areas, transform) in enumerate(zip(boxes, transforms, strict=True), 1):
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
                      "transform": transform,
                      "regions": regions, "artifacts": artifacts})
    return {"schema": SCHEMA if reflected else TRANSLATION_SCHEMA,
            "source_sha256": analysis.sha256, "coordinate_space": "exif-oriented-original", "status": status,
            "authorship_confidence_contribution": 0, "pairs": pairs,
            "interpretation": (MULTI_EXACT_NOTE if exact_transforms else reflection_note if exact_reflection
                               else NOTE if reflected else TRANSLATION_NOTE)}


def _affine_record(db: Session, analysis: Analysis, store: ObjectStore, document: dict,
                   source: Image.Image | None, *, skip: bool) -> dict:
    version = document["version"]
    expanded = version in affine.LARGE_VERSIONS
    record = {"schema": affine.LARGE_SCHEMA if expanded else affine.SCHEMA,
              "source_sha256": analysis.sha256, "status": "unsupported_geometry",
              "pairs": [], "authorship_confidence_contribution": 0, "interpretation": affine.NOTE}
    if expanded:
        record["proposal_search"] = None
    if skip:
        return {**record, "status": SKIPPED_CREDENTIALS}
    if source is None or not affine.supported(*source.size, version=version):
        return record
    rgb = np.asarray(source)
    search = affine.search_geometry(*source.size)
    if expanded:
        record["proposal_search"] = search
    reduced = search["basis"] == "reduced"
    try:
        if reduced:
            proposals = copy_affine_large.candidates(rgb)
        else:
            proposals = [(np.array(item.affine.matrix, np.float64),
                          box_bounds(prior._coordinates(_crop_box(item.box_a, *source.size)), *source.size))
                         for item in copy_affine.candidates(rgb)]
    except copy_move.CandidateBudgetExceeded:
        return {**record, "status": "unassessed_match_budget"}
    gray = copy_affine.luma(rgb) if proposals else None
    matches = [(matrix, bounds, masks) for matrix, bounds in proposals
               if (masks := copy_affine.dense_pair(gray, matrix, bounds)) is not None]
    if not matches and not reduced and settings.environment == "dev" and settings.image_small_affine_copy_review:
        from app.detectors.image import copy_affine_small
        from app.detectors.image.reference_lossy_webp import webp_size
        from app.evidence.image_jpeg_offgrid_source import encoded_profile

        data = legacy._verified(store, analysis.sha256)
        jpeg_supported = data.startswith(b"\xff\xd8") and encoded_profile(data) in {75, 90}
        webp_supported = max(source.size) >= 768 and webp_size(data, minimum_side=384) == source.size
        if jpeg_supported or webp_supported:
            try:
                extra = copy_affine_small.confirmed(rgb)
            except copy_move.CandidateBudgetExceeded:
                return {**record, "status": "unassessed_match_budget"}
            if extra is not None:
                matches.append(extra)
    for matrix, bounds, masks in matches:
        x0, y0, x1, y1 = bounds
        seed = prior._coordinates((x0, y0, x1 - x0, y1 - y0))
        regions, counts, artifacts = {}, {}, {}
        for role, mask in zip(ROLES, masks, strict=True):
            ys, xs = np.nonzero(mask)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            region = prior._coordinates((x0, y0, x1 - x0, y1 - y0))
            regions[role], counts[role] = region, int(len(xs))
            crop = source.crop((x0, y0, x1, y1))
            selection = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255)
            with io.BytesIO() as output:
                selection.save(output, format="PNG")
                mask_png = output.getvalue()
            for name, content in ((role, legacy._png(crop)), ("mask_" + role[-1], mask_png)):
                art = register_artefact(db, store, content, media_type="image/png", role="preview",
                                        created_by=_CREATORS[version], analysis_id=analysis.id)
                artifacts[name] = {"sha256": art.sha256, "media_type": "image/png", "width": crop.width,
                                   "height": crop.height, "url": url(document, art.sha256)}
        record["pairs"].append({"id": f"G{len(record['pairs']) + 1:02d}", "seed": seed, "matrix": matrix.tolist(),
                                "regions": regions, "selected_pixels": counts, "artifacts": artifacts})
    record["status"] = "supported_pair" if record["pairs"] else "no_pair"
    return record


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
    version = _analysis_version(analysis)
    expanded_rotation = version in EXPANDED_JPEG_ROTATION_VERSIONS
    document = {
        **base,
        "id": current_set_id(analysis),
        "version": version,
        "prior_image_evidence": {key: base[key] for key in ("id", "version", "sha256")},
    }
    size = document["source"]["width"], document["source"]["height"]
    status, boxes = "unsupported_geometry", []
    translation_status, translations, basis, source = "unsupported_geometry", [], None, None
    jpeg_status = offgrid_review = turned_jpeg_review = exact_review = reflection_review = exact_reflection = None
    webp = version in WEBP_EXACT_COPY_VERSIONS
    exact_reflection_enabled = version in EXACT_REFLECTION_VERSIONS
    multi_review, multi_transforms = None, ()
    turns = 0
    skipped = result_skip(result) is not None
    if skipped:
        status = translation_status = SKIPPED_CREDENTIALS
    elif copy_similarity.supported_size(*size):
        source = _source(document, store)
        rgb = np.asarray(source)
        status, boxes = copy_similarity.analyse(rgb)
        if supported_size(*size, version=V8):
            translation_status, translations, basis = selected_translation_pairs(rgb)
            if version in JPEG_BLOCK_VERSIONS and translation_status == "no_pair":
                from app.detectors.image import jpeg_copy_blocks

                data = legacy._verified(store, analysis.sha256)
                try:
                    jpeg_status, extra = (jpeg_copy_blocks.analyse(data) if data.startswith(b"\xff\xd8")
                                          else ("unsupported_profile", []))
                except cv2.error:
                    jpeg_status, extra = "unavailable", []
                if extra:
                    translation_status, translations, basis = "supported_pair", extra, JPEG_BLOCK_BASIS
                elif version in JPEG_OFFGRID_VERSIONS and jpeg_status == "no_pair":
                    from app.evidence.image_jpeg_offgrid_source import review

                    try:
                        offgrid_review, extra = review(data, rgb)
                    except (cv2.error, OSError, ValueError, SystemError):
                        offgrid_review, extra = {"status": "unavailable", "measurement": None}, []
                    if extra:
                        translation_status, translations, basis = "supported_pair", extra, OFFGRID_BASIS
                    elif version in TURNED_JPEG_VERSIONS and rotation_review_follows(offgrid_review, expanded=expanded_rotation):
                        from app.evidence import image_jpeg_turned_copy_source

                        try:
                            turned_jpeg_review, extra, turns = image_jpeg_turned_copy_source.review(
                                data, rgb, expanded=expanded_rotation, offgrid_status=offgrid_review["status"])
                        except (cv2.error, OSError, ValueError, SystemError):
                            turned_jpeg_review, extra = {"status": "unavailable", "measurement": None}, []
                        if extra:
                            translation_status, translations, basis = "supported_pair", extra, TURNED_JPEG_BASIS
    if version in NATIVE_JPEG_ROTATION_VERSIONS and not skipped and native_rotation_size(*size):
        from app.evidence import image_jpeg_turned_copy_source

        source = source if source is not None else _source(document, store)
        data = legacy._verified(store, analysis.sha256)
        try:
            turned_jpeg_review, extra, turns = image_jpeg_turned_copy_source.review(
                data, np.asarray(source), native=True, offgrid_status=None, allow_disjoint=version in DISJOINT_JPEG_ROTATION_VERSIONS)
        except (cv2.error, OSError, ValueError, SystemError):
            turned_jpeg_review, extra = {"status": "unavailable", "measurement": None}, []
        translation_status = "supported_pair" if extra else "no_pair"
        if extra:
            translations, basis = extra, TURNED_JPEG_BASIS
    if version in MULTI_EXACT_VERSIONS and not skipped and exact_copy_size(*size, webp=webp) and len(translations) < 2:
        from app.evidence.image_exact_small_copy_source import review_multiple

        try:
            multi_review, matches = review_multiple(legacy._verified(store, analysis.sha256), webp=webp,
                                                   allow_three=version in THREE_EXACT_VERSIONS)
        except (cv2.error, OSError, ValueError, SystemError):
            multi_review, matches = {"status": "unavailable", "measurements": []}, []
        if matches:
            source = source if source is not None else _source(document, store)
            translation_status, basis = "supported_pair", MULTI_EXACT_BASIS
            translations = [match["pair"] for match in matches]
            multi_transforms = tuple(match["transform"] for match in matches)
    if ((webp or version in SMALL_PNG_VERSIONS) and not skipped and exact_copy_size(*size, webp=webp)
            and translation_status in {"no_pair", "unsupported_geometry"}):
        from app.evidence.image_exact_small_copy_source import review

        try:
            exact_review, extra, turns = review(legacy._verified(store, analysis.sha256),
                                               include_rotation=webp or version in TURNED_PNG_VERSIONS, webp=webp)
        except (cv2.error, OSError, ValueError, SystemError):
            exact_review, extra = {"status": "unavailable", "measurement": None}, []
        translation_status = "supported_pair" if extra else "no_pair"
        if extra:
            source = source if source is not None else _source(document, store)
            translations, basis = extra, WEBP_EXACT_COPY_BASIS if webp else (TURNED_PNG_BASIS if turns else SMALL_PNG_BASIS)
    if exact_reflection_enabled and version in AFFINE_VERSIONS:
        document[affine.KEY] = _affine_record(db, analysis, store, document, source, skip=skipped)
    affine_status = (document.get(affine.KEY) or {}).get("status")
    if (exact_reflection_enabled and exact_review and status in {"no_pair", "unsupported_geometry"}
            and affine_status in {None, "no_pair", "unsupported_geometry"}):
        if exact_review["status"] == "unsupported_profile":
            reflection_review = exact_review
        elif exact_review["status"] == "no_pair":
            from app.evidence.image_exact_small_copy_source import review_reflection

            try:
                reflection_review, extra, exact_reflection = review_reflection(
                    legacy._verified(store, analysis.sha256), webp=webp,
                    include_diagonal=version in DIAGONAL_EXACT_VERSIONS)
            except (cv2.error, OSError, ValueError, SystemError):
                reflection_review, extra = {"status": "unavailable", "measurement": None}, []
            if extra:
                source = source if source is not None else _source(document, store)
                translation_status, translations = "supported_pair", extra
                basis = WEBP_EXACT_REFLECTION_BASIS if webp else PNG_EXACT_REFLECTION_BASIS
    document["copy_similarity"] = _copy_record(db, analysis, store, document, source, status, boxes, reflected=True)
    document["copy_translation"] = _copy_record(db, analysis, store, document, source, translation_status,
                                                translations, reflected=False, turns=turns, exact_reflection=exact_reflection,
                                                exact_transforms=multi_transforms)
    document["copy_translation"]["basis"] = basis
    if version in JPEG_BLOCK_VERSIONS:
        document["copy_translation"]["jpeg_block_review"] = jpeg_status
    if version in JPEG_OFFGRID_VERSIONS:
        document["copy_translation"]["jpeg_offgrid_review"] = offgrid_review
    if version in TURNED_JPEG_VERSIONS:
        document["copy_translation"]["jpeg_turned_review"] = turned_jpeg_review
    if version in SMALL_PNG_VERSIONS:
        document["copy_translation"]["small_png_copy_review"] = exact_review
    if webp:
        document["copy_translation"][WEBP_EXACT_COPY_KEY] = exact_review
    if exact_reflection_enabled:
        key = WEBP_EXACT_REFLECTION_KEY if webp else PNG_EXACT_REFLECTION_KEY
        document["copy_translation"][key] = reflection_review
    if version in MULTI_EXACT_VERSIONS:
        document["copy_translation"][MULTI_EXACT_KEY] = multi_review
    if not exact_reflection_enabled and version in AFFINE_VERSIONS:
        document[affine.KEY] = _affine_record(db, analysis, store, document, source, skip=skipped)
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
        created_by=_CREATORS[version],
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
    return {**prior.unavailable_packet_set(analysis), "version": _analysis_version(analysis)}
