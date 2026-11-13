"""Freeze and render image evidence without consulting mutable analysis state at render time."""
from __future__ import annotations

import base64
import copy
import io
from typing import TYPE_CHECKING, Any

from PIL import Image

from app.core.canonical import canonical_hash
from app.evidence import image_embedded_pair_binding as embedded_pair
from app.reports.image_eligibility import eligibility_view, inspection_check_view
from app.storage.objects import ObjectIntegrityError, ObjectStore, default_store

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.reports.snapshot import ReportSnapshot
    from app.storage.db import Analysis


ROLES = (("context", "Position in the original"), ("crop", "Original crop"), ("overlay", "Diagnostic overlay"))
COPY_CHECKS = ("copy_similarity", "copy_translation", "copy_affine")
TRANSLATION_STATUS = {
    "supported_pair": "Similar image detail was found without mirroring.",
    "no_pair": "This check found no qualifying non-mirrored pair.",
    "unsupported_geometry": (
        "This check requires native dimensions of 384 px or more on both sides and a long side of 513–1,024 px."
    ),
    "unassessed_match_budget": "The check found too many candidate matches to show a distinct comparison.",
}
TRANSLATION_BASIS_TEXT = {
    "native_patch_alignment": "The match aligns patches on the original pixel grid.",
    "geometric_features": "The match uses corresponding features in both areas.",
}


def roles(packet: dict):
    mask = (("mask", "Selected component on the measured grid"),) if "mask" in packet["artifacts"] else ()
    if "copy_donor" in packet["artifacts"]:
        return (*ROLES[:2], ("copy_donor", "Matching patch from the original image"), ROLES[2], *mask)
    return ROLES + mask


def _asset_bytes(asset: dict[str, Any], store: ObjectStore) -> bytes:
    """Every report embeds the exact frozen PNG, not a refreshed or substituted thumbnail."""
    sha = asset["sha256"]
    store.verify(sha)
    data = store.read_bytes(sha)
    # Defend the interval between verification and reading, as well as malformed image metadata.
    import hashlib

    if hashlib.sha256(data).hexdigest() != sha:
        raise ObjectIntegrityError("image evidence artifact changed while reading")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (asset["width"], asset["height"]):
            raise ObjectIntegrityError("image evidence artifact does not match its frozen dimensions")
        image.verify()
    return data


def capture(db: Session, analysis: Analysis, *, store: ObjectStore | None = None) -> dict[str, Any] | None:
    """Read-only capture at report request; never create evidence or alter a stored result."""
    if analysis.kind != "image":
        return None
    from app.evidence.image_packets_v8 import load_packet_set

    store = store or default_store()
    document = load_packet_set(db, analysis, store)
    if document is None:
        return None
    for asset in (document.get("overview") or {}).values():
        _asset_bytes(asset, store)
    for packet in document["packets"]:
        for role, _ in roles(packet):
            _asset_bytes(packet["artifacts"][role], store)
    for asset in embedded_pair.artifacts(document).values():
        _asset_bytes(asset, store)
    return copy.deepcopy(document)


def verified_set(snapshot: ReportSnapshot) -> dict[str, Any] | None:
    """Validate the frozen public evidence content against this exact original result."""
    document = snapshot.image_evidence
    if document is None:
        return None
    from app.evidence.image_packets_v8 import validate_packet_set, validate_packet_source_mapping

    if snapshot.analysis.kind != "image":
        raise ValueError("image evidence is attached to a non-image report")
    validate_packet_set(document, analysis_id=snapshot.analysis.id, revision=snapshot.analysis.revision,
                        original_result_hash=canonical_hash(snapshot.result), source_sha256=snapshot.media.sha256)
    validate_packet_source_mapping(document, snapshot.result)
    return document


def view(snapshot: ReportSnapshot, *, store: ObjectStore | None = None) -> dict[str, Any] | None:
    if snapshot.analysis.kind != "image":
        return None
    document = verified_set(snapshot)
    if document is None:
        return {"available": False, "packets": [],
                "note": "No image-region evidence was captured in this report version. "
                        "An overall classification does not locate edited pixels."}
    store = store or default_store()
    store.verify(document["source"]["sha256"])
    from app.evidence import image_copy_affine_binding as affine
    from app.evidence.image_copy_similarity_binding import (
        JPEG_BLOCK_STATUS_TEXT,
        OFFGRID_STATUS_TEXT,
        SKIPPED_CREDENTIALS,
        SMALL_PNG_STATUS_TEXT,
        status_text,
    )
    from app.evidence.image_packets_v8 import validate_assets

    validate_assets(document, store)
    withheld = snapshot.report.profile == "newsroom" or snapshot.report.redacted
    packets = []
    for packet in document["packets"]:
        images = []
        for role, label in roles(packet):
            asset = packet["artifacts"][role]
            data = _asset_bytes(asset, store)
            images.append({"role": role, "label": label, "sha256": asset["sha256"],
                           "uri": None if withheld else "data:image/png;base64," + base64.b64encode(data).decode("ascii")})
        packets.append({**copy.deepcopy(packet), "images": images,
                        "eligibility_view": eligibility_view(packet["localisation"]["candidate_eligibility"])
                        if packet.get("localisation", {}).get("candidate_eligibility") else None})
    comparison = copy.deepcopy(document.get("comparison"))
    copy_checks = {field: copy.deepcopy(document.get(field)) for field in COPY_CHECKS}
    base_version = embedded_pair.base_document(document)["version"]
    for field, check in copy_checks.items():
        if check is None:
            continue
        status = check["status"]
        if status == SKIPPED_CREDENTIALS and document.get(affine.KEY):
            check["observation"] = "Trusted Content Credentials cover this image and declare AI involvement. The image copy checks were skipped."
        elif field == affine.KEY:
            check["observation"] = affine.status_text(status, base_version)
        else:
            check["observation"] = (status_text(status, version=base_version)
                                    if field == "copy_similarity" or status == SKIPPED_CREDENTIALS
                                    else TRANSLATION_STATUS[status])
        if field == "copy_translation":
            if small_review := check.get("small_png_copy_review"):
                check["basis_text"] = SMALL_PNG_STATUS_TEXT[small_review["status"]]
                if measured := small_review["measurement"]:
                    perimeter = ", ".join(f"{value:.2f}" for value in measured["perimeter_mean_absolute_difference"])
                    check["basis_text"] += (f" Compared {measured['matching_pixels']:,} RGB pixels in each crop. "
                                            f"Mean surrounding differences (top, bottom, left, right) {perimeter} of 255 levels.")
            elif review := check.get("jpeg_offgrid_review"):
                check["basis_text"] = OFFGRID_STATUS_TEXT[review["status"]]
                if measured := review["measurement"]:
                    perimeter = ", ".join("unavailable" if value is None else f"{value:.2f}"
                                          for value in measured["perimeter_means"])
                    check["basis_text"] += (f" Mean RGB difference inside {measured['interior_mean']:.2f} of 255 levels. "
                                             f"Surrounding differences (left, right, top, bottom) {perimeter}.")
            elif (jpeg_review := check.get("jpeg_block_review")) is not None:
                check["basis_text"] = JPEG_BLOCK_STATUS_TEXT[jpeg_review]
            elif check["pairs"] and (basis := check.get("basis")) in TRANSLATION_BASIS_TEXT:
                check["basis_text"] = TRANSLATION_BASIS_TEXT[basis]
        for pair in check["pairs"]:
            pair["images"] = []
            for role, label in (("area_a", "Area A"), ("area_b", "Area B")):
                asset = pair["artifacts"][role]
                data = _asset_bytes(asset, store)
                pair["images"].append({"role": role, "label": label, "sha256": asset["sha256"],
                    "uri": None if withheld else "data:image/png;base64," + base64.b64encode(data).decode("ascii")})
                if field == affine.KEY:
                    mask = pair["artifacts"]["mask_" + role[-1]]
                    data = _asset_bytes(mask, store)
                    title = "Area A selected mask" if role == "area_a" else "Area B mapped mask"
                    pair["images"].append({"role": role, "label": title, "sha256": mask["sha256"],
                        "uri": None if withheld else "data:image/png;base64," + base64.b64encode(data).decode("ascii")})
    exact = copy.deepcopy(document.get(embedded_pair.KEY))
    if exact is not None:
        exact["observation"] = embedded_pair.STATUS_TEXT[exact["status"]]
        if pair := exact["pair"]:
            pair["images"] = []
            for role, label in embedded_pair.ASSETS.items():
                asset = pair["artifacts"][role]
                data = _asset_bytes(asset, store)
                pair["images"].append({"role": role, "label": label, "sha256": asset["sha256"],
                    "uri": None if withheld else "data:image/png;base64," + base64.b64encode(data).decode("ascii")})
    if comparison is not None:
        comparison["images"] = []
        for role, label in (("wider_overlay", "Wider view: selected by either check"),
                            ("shared_overlay", "Shared view: selected by both checks"),
                            ("wider_mask", "Wider mask on the measured grid"),
                            ("shared_mask", "Shared mask on the measured grid")):
            asset = comparison["artifacts"].get(role)
            if asset is not None:
                data = _asset_bytes(asset, store)
                comparison["images"].append({"role": role, "label": label, "sha256": asset["sha256"],
                    "uri": None if withheld else "data:image/png;base64," + base64.b64encode(data).decode("ascii")})
    inspection_checks = []
    overview = (document.get("overview") or {}).values()
    for check in (document.get("coverage") or {}).get("inspection_checks", []):
        display = inspection_check_view(check)
        sha = check.get("reliability_mask_sha256")
        if sha and display["map_grid"] and not withheld:
            for asset in overview:
                if (asset["sha256"] == sha and asset["media_type"] == "image/png"
                        and [asset["width"], asset["height"]] == list(display["map_grid"])):
                    data = _asset_bytes(asset, store)
                    display["coverage_map"] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
                    break
        inspection_checks.append(display)
    return {"available": True, "id": document["id"], "version": document["version"], "sha256": document["sha256"],
            "source_sha256": document["source"]["sha256"], "source": document["source"], "packets": packets,
            "limitations": document["limitations"], "omitted_regions": document["omitted_regions"], "withheld": withheld,
            "inspection_checks": inspection_checks,
            "comparison": comparison, **copy_checks, embedded_pair.KEY: exact,
            "note": "Each plate is derived from the recorded original and the same evidence set used by the JSON export. "
                    "Inspection regions alone do not establish editing or AI generation."}


def manifest(snapshot: ReportSnapshot) -> dict[str, Any] | None:
    document = verified_set(snapshot)
    if document is None:
        return None
    copy_checks = {field: copy.deepcopy(document[field]) for field in COPY_CHECKS if field in document}
    result = {"schema": document["schema"], "id": document["id"], "version": document["version"],
            "sha256": document["sha256"], "source_sha256": document["source"]["sha256"],
            "original_result_hash": document["original_result_hash"],
            "artifacts": [{"packet_id": None, "role": role, "sha256": asset["sha256"]}
                          for role, asset in (document.get("overview") or {}).items()] +
                         [{"packet_id": packet["id"], "role": role, "sha256": packet["artifacts"][role]["sha256"]}
                          for packet in document["packets"] for role, _ in roles(packet)] +
                         [{"packet_id": pair["id"], "role": role, "sha256": asset["sha256"]}
                          for check in copy_checks.values() for pair in check["pairs"]
                          for role, asset in pair["artifacts"].items()],
              **copy_checks}
    if embedded_pair.KEY in document:
        result[embedded_pair.KEY] = copy.deepcopy(document[embedded_pair.KEY])
        result["artifacts"].extend({"packet_id": "D01", "role": role, "sha256": asset["sha256"]}
                                   for role, asset in embedded_pair.artifacts(document).items())
    return result
