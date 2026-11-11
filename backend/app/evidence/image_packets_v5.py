"""Immutable image evidence with a native-pixel same-image copy check."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import uuid
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.image import copy_donor
from app.evidence import image_packets as legacy
from app.evidence import image_packets_v4 as prior
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact
from app.storage.objects import ObjectStore

VERSION = "image-evidence-5"
CREATOR = "image-evidence-packets/3"
SCHEMA = prior.SCHEMA
PacketIntegrityError = prior.PacketIntegrityError
packet_hash = prior.packet_hash
_NS = uuid.UUID("a76a9056-ef61-412e-9621-31e4d34c2cfe")
_ASSET = "copy_donor"
_STATUSES = frozenset(
    {
        "copy_consistent",
        "ambiguous_repetition",
        "no_donor",
        "insufficient_texture",
        "unsupported_geometry",
        "empty_source_projection",
    }
)


def current_set_id(analysis: Analysis) -> str:
    return str(uuid.uuid5(_NS, f"{analysis.id}:{analysis.result_hash}:{VERSION}"))


def _v4(document: dict) -> dict:
    base = copy.deepcopy(document)
    base["version"] = prior.VERSION
    for packet in base["packets"]:
        packet.pop("copy_corroboration", None)
        packet["artifacts"].pop(_ASSET, None)
    base["sha256"] = packet_hash(base)
    return base


def _source_mask(store: ObjectStore, packet: dict, size: tuple[int, int]) -> np.ndarray:
    mask = prior._mask(store, packet["artifacts"]["mask"])
    native = Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
    return np.asarray(native, dtype=bool)


def _coordinates(box: list[int]) -> dict[str, int]:
    return dict(zip(("x", "y", "w", "h"), box, strict=True))


def _describe(cue: dict) -> str:
    status = cue["status"]
    if status == "copy_consistent":
        donor = cue["donor"]
        box = donor["box"]
        transform = (
            "after a horizontal reflection"
            if donor["transform"] == "horizontal_reflection"
            else "after a shift"
        )
        return (
            f"The same-image copy check found another patch at x {box['x']}, y {box['y']} ({box['w']} × {box['h']} px) "
            f"that matches {donor['supporting_anchors']} of {cue['anchor_count']} textured blocks {transform}. "
            "This pixel similarity is compatible with copying or repeated detail. The direction and origin of any edit remain unknown."
        )
    if status == "unsupported_geometry":
        return (
            f"Same-image copy check skipped. This native-pixel check requires an image at least "
            f"{copy_donor.MIN_SIDE} px across each side and no larger than {copy_donor.SIDE_LIMIT} px per side."
        )
    return {
        "insufficient_texture": "Same-image copy check was inconclusive. The selected pixels had too few textured blocks to compare.",
        "empty_source_projection": "Same-image copy check skipped. The selected map cells did not project to any native source pixels.",
        "ambiguous_repetition": "Same-image copy check found similarly strong matches in multiple locations. Repeated detail makes the match ambiguous.",
        "no_donor": "Same-image copy check found no qualifying shifted or horizontally reflected patch. Other edits remain possible.",
    }[status]


def _cue(store: ObjectStore, packet: dict, size: tuple[int, int], source: Image.Image | None) -> dict:
    evidence = {
        "schema": "same-image-copy-corroboration/1",
        "source_size": list(size),
        "coordinate_space": "exif-oriented-original",
        "highlight_asset_sha256": packet["artifacts"]["mask"]["sha256"],
        "status": "unsupported_geometry",
        "anchor_count": 0,
        "recipient_box": None,
        "recipient_pixels": None,
        "recipient_mask_sha256": None,
        "donor": None,
        "authorship_confidence_contribution": 0,
    }
    if source is None:
        return evidence
    mask = _source_mask(store, packet, size)
    if not mask.any():
        evidence["status"] = "empty_source_projection"
        return evidence
    match = copy_donor.corroborate(np.asarray(source), mask)
    donor = match["donor"]
    evidence.update(
        status=match["status"],
        anchor_count=match["anchor_count"],
        recipient_box=_coordinates(match["proposal_box_xywh"]),
        recipient_pixels=match["proposal_pixels"],
        recipient_mask_sha256=match["proposal_mask_sha256"],
        donor=(
            {
                "box": _coordinates(donor["box_xywh"]),
                "transform": donor["transform"],
                "supporting_anchors": donor["supporting_anchors"],
                "median_correlation": donor["correlation"],
            }
            if donor
            else None
        ),
    )
    return evidence


def _check_cue(document: dict, packet: dict) -> None:
    cue = packet.get("copy_corroboration")
    asset = packet["artifacts"].get(_ASSET)
    if packet["evidence_type"] != "localisation":
        if cue is not None or asset is not None:
            raise PacketIntegrityError("Copy evidence requires a stored component mask")
        return
    width, height = (document["source"][axis] for axis in ("width", "height"))
    if (
        not isinstance(cue, dict)
        or cue.get("schema") != "same-image-copy-corroboration/1"
        or cue.get("status") not in _STATUSES
        or cue.get("source_size") != [width, height]
        or cue.get("coordinate_space") != "exif-oriented-original"
        or cue.get("authorship_confidence_contribution") != 0
        or cue.get("highlight_asset_sha256") != packet["artifacts"]["mask"]["sha256"]
    ):
        raise PacketIntegrityError("Copy evidence uses a different scope or highlight")
    status = cue["status"]
    anchors = cue.get("anchor_count")
    if (
        type(anchors) is not int
        or not 0 <= anchors <= 9
        or (status == "unsupported_geometry") == copy_donor.supports_size(width, height)
        or (anchors < copy_donor.MIN_ANCHORS)
        != (status in {"unsupported_geometry", "empty_source_projection", "insufficient_texture"})
    ):
        raise PacketIntegrityError("Copy evidence coverage is inconsistent")
    recipient, pixels, mask_hash = (
        cue.get(key) for key in ("recipient_box", "recipient_pixels", "recipient_mask_sha256")
    )
    donor = cue.get("donor")
    if status in {"unsupported_geometry", "empty_source_projection"}:
        if anchors or any(value is not None for value in (recipient, pixels, mask_hash)):
            raise PacketIntegrityError("Unassessed image geometry cannot supply a measured region")
    elif (
        type(pixels) is not int
        or not 0 < pixels <= width * height
        or not legacy._SHA.fullmatch(str(mask_hash))
    ):
        raise PacketIntegrityError("Copy evidence is missing its source mask")
    else:
        _integer_box(recipient, width, height)
    if status != "copy_consistent":
        if donor is not None or asset is not None:
            raise PacketIntegrityError("An unsupported copy finding cannot supply a donor")
        return
    if (
        not isinstance(donor, dict)
        or donor.get("transform") not in copy_donor.TRANSFORMS
        or not isinstance(asset, dict)
    ):
        raise PacketIntegrityError("Supported copy evidence is missing its matching patch")
    count, correlation = donor.get("supporting_anchors"), donor.get("median_correlation")
    if (
        type(count) is not int
        or not math.ceil(anchors * copy_donor.MIN_SUPPORT_FRACTION) <= count <= anchors
        or type(correlation) not in {int, float}
        or not copy_donor.MIN_CORRELATION <= correlation <= 1
    ):
        raise PacketIntegrityError("Copy similarity measurement is invalid")
    box = donor.get("box")
    x, y, right, bottom = _integer_box(box, width, height)
    rx, ry, rr, rb = _integer_box(recipient, width, height)
    if (right - x, bottom - y) != (rr - rx, rb - ry) or (x < rr and right > rx and y < rb and bottom > ry):
        raise PacketIntegrityError("Matching patch geometry is inconsistent")
    if (
        asset.get("media_type") != "image/png"
        or not legacy._SHA.fullmatch(str(asset.get("sha256")))
        or [asset.get("width"), asset.get("height")] != [box["w"], box["h"]]
    ):
        raise PacketIntegrityError("Matching patch does not have a valid original crop")


def _integer_box(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    if (
        not isinstance(box, dict)
        or set(box) != {"x", "y", "w", "h"}
        or any(type(v) is not int for v in box.values())
    ):
        raise PacketIntegrityError("Copy evidence needs exact source-pixel coordinates")
    return legacy._bounds(box, width, height)


def validate_packet_set(document: dict[str, Any], **binding: Any) -> None:
    if document.get("version") != VERSION:
        prior.validate_packet_set(document, **binding)
        return
    if document.get("sha256") != packet_hash(document):
        raise PacketIntegrityError("Copy evidence has a different content hash")
    prior.validate_packet_set(_v4(document), **binding)
    for packet in document["packets"]:
        _check_cue(document, packet)


def validate_packet_source_mapping(document: dict, result: dict) -> None:
    prior.validate_packet_source_mapping(
        _v4(document) if document.get("version") == VERSION else document, result
    )


def validate_assets(document: dict, store: ObjectStore) -> None:
    if document.get("version") != VERSION:
        prior.validate_comparison_assets(document, store)
        return
    prior.validate_comparison_assets(_v4(document), store)
    size = tuple(document["source"][axis] for axis in ("width", "height"))
    source = None
    for packet in document["packets"]:
        cue = packet.get("copy_corroboration")
        if cue is None or cue["status"] == "unsupported_geometry":
            continue
        mask = _source_mask(store, packet, size)
        if cue["status"] == "empty_source_projection":
            if mask.any():
                raise PacketIntegrityError("Copy evidence incorrectly marks its source projection empty")
            continue
        if (
            int(mask.sum()) != cue["recipient_pixels"]
            or hashlib.sha256(mask.tobytes()).hexdigest() != cue["recipient_mask_sha256"]
        ):
            raise PacketIntegrityError("Copy evidence does not match the retained source mask")
        ys, xs = np.nonzero(mask)
        actual_box = legacy._box((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        if cue["recipient_box"] != actual_box:
            raise PacketIntegrityError("Copy evidence recipient coordinates differ from the retained mask")
        if cue["donor"] is None:
            continue
        if source is None:
            source, orientation = legacy._source(legacy._verified(store, document["source"]["sha256"]))
            if source.size != size or orientation != document["source"]["exif_orientation"]:
                raise PacketIntegrityError("Matching patch uses different source geometry")
        expected = source.crop(_integer_box(cue["donor"]["box"], *size))
        data = legacy._verified(store, packet["artifacts"][_ASSET]["sha256"])
        with Image.open(io.BytesIO(data)) as donor_image:
            if (
                donor_image.format != "PNG"
                or donor_image.mode != "RGB"
                or donor_image.size != expected.size
                or donor_image.tobytes() != expected.tobytes()
            ):
                raise PacketIntegrityError("Matching patch pixels differ from the submitted source")


def build_packet_set(
    db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any]
) -> dict[str, Any]:
    if (
        canonical_hash(result) != analysis.result_hash
        or result.get("input", {}).get("sha256") != analysis.sha256
    ):
        raise PacketIntegrityError("Published result or source identity does not match")
    base = prior.persist_packet_set(db, analysis, store, result=result)
    if base is None:
        raise ValueError("Image evidence needs a published image result")
    document = copy.deepcopy(base)
    document.update(
        id=current_set_id(analysis),
        version=VERSION,
        derived_from={key: base[key] for key in ("id", "version", "sha256")},
    )
    packets = document["packets"]
    localised = [packet for packet in packets if packet["evidence_type"] == "localisation"]
    size = (document["source"]["width"], document["source"]["height"])
    source = None
    if localised and copy_donor.supports_size(*size):
        source, orientation = legacy._source(legacy._verified(store, analysis.sha256))
        if source.size != size or orientation != document["source"]["exif_orientation"]:
            raise PacketIntegrityError("Copy check uses different source geometry")
    for packet in localised:
        cue = _cue(store, packet, size, source)
        packet["copy_corroboration"] = cue
        packet["interpretation"] += " " + _describe(cue)
        donor = cue["donor"]
        if donor is not None and source is not None:
            crop = source.crop(_integer_box(donor["box"], *source.size))
            art = register_artefact(
                db,
                store,
                legacy._png(crop),
                media_type="image/png",
                role="preview",
                created_by=CREATOR,
                analysis_id=analysis.id,
            )
            packet["artifacts"][_ASSET] = {
                "sha256": art.sha256,
                "media_type": "image/png",
                "width": crop.width,
                "height": crop.height,
            }
            packet["measurements"].append(
                {
                    "label": "Median matching-block luminance correlation",
                    "value": donor["median_correlation"],
                    "unit": "pixel similarity",
                }
            )
    for packet in packets:
        packet["derivation"]["version"] = VERSION
    for asset in [
        *document["overview"].values(),
        *document["comparison"]["artifacts"].values(),
        *(art for p in packets for art in p["artifacts"].values()),
    ]:
        asset["url"] = (
            f"/api/v2/analyses/{analysis.id}/evidence-packets/{document['id']}/images/{asset['sha256']}.png"
        )
    document = json.loads(canonical_json(document))
    document["sha256"] = packet_hash(document)
    validate_packet_set(document, **_binding(analysis))
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


def _binding(analysis: Analysis) -> dict:
    return {
        "analysis_id": analysis.id,
        "revision": analysis.revision,
        "original_result_hash": analysis.result_hash,
        "source_sha256": analysis.sha256,
    }


def load_packet_set(
    db: Session, analysis: Analysis, store: ObjectStore, *, set_id: str | None = None
) -> dict[str, Any] | None:
    row = db.scalars(
        select(Artefact).where(
            Artefact.analysis_id == analysis.id,
            Artefact.created_by == CREATOR,
            Artefact.role == "exhibit_payload",
            Artefact.media_type == "application/json",
            Artefact.id == (set_id or current_set_id(analysis)),
        )
    ).first()
    if row is None:
        return prior.load_packet_set(db, analysis, store, set_id=set_id)
    document = json.loads(legacy._verified(store, row.sha256))
    if document.get("id") != row.id or document.get("version") != VERSION:
        raise PacketIntegrityError("Copy evidence differs from its indexed identity")
    validate_packet_set(document, **_binding(analysis))
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
