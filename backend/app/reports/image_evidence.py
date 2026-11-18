"""Freeze and render image evidence without consulting mutable analysis state at render time."""
from __future__ import annotations

import base64
import copy
import io
from typing import TYPE_CHECKING, Any

from PIL import Image

from app.core.canonical import canonical_hash
from app.evidence import image_embedded_pair_binding as embedded_pair
from app.evidence import image_interpolation_binding as interpolation
from app.evidence import image_previous_grid_binding as prior_grid
from app.evidence.image_eligibility import repeat_pass
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
    for asset in (*interpolation.assets(document).values(), *prior_grid.assets(document).values()):
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
        EXACT_REFLECTION_VERSIONS,
        JPEG_BLOCK_STATUS_TEXT,
        MULTI_EXACT_KEY,
        MULTI_EXACT_STATUS,
        NATIVE_JPEG_ROTATION_VERSIONS,
        OFFGRID_STATUS_TEXT,
        PNG_EXACT_REFLECTION_KEY,
        PNG_EXACT_REFLECTION_STATUS_TEXT,
        PNG_EXACT_REFLECTION_VERSIONS,
        SKIPPED_CREDENTIALS,
        SMALL_PNG_CROP_MAX,
        SMALL_PNG_CROP_MIN,
        SMALL_PNG_STATUS_TEXT,
        TURNED_JPEG_BASIS,
        TURNED_JPEG_STATUS_TEXT,
        TURNED_PNG_BASIS,
        TURNED_PNG_VERSIONS,
        WEBP_EXACT_COPY_KEY,
        WEBP_EXACT_COPY_STATUS_TEXT,
        WEBP_EXACT_COPY_VERSIONS,
        WEBP_EXACT_REFLECTION_KEY,
        WEBP_EXACT_REFLECTION_STATUS_TEXT,
        exact_copy_size,
        status_text,
    )
    from app.evidence.image_packets_v8 import validate_assets

    validate_assets(document, store)
    withheld = snapshot.report.profile == "newsroom" or snapshot.report.redacted
    local_reviews = {}
    for method, labels in ((interpolation, {"context": "Region in the submitted image", "crop": "Original 16-bit source crop"}),
                           (prior_grid, prior_grid.PLATES)):
        review = copy.deepcopy(document.get(method.KEY))
        if review is not None:
            review["images"] = [
                {"label": label, "uri": None if withheld else "data:image/png;base64," + base64.b64encode(_asset_bytes(asset, store)).decode("ascii")}
                for role, label in labels.items() if (asset := review["artifacts"].get(role)) is not None]
        local_reviews[method.KEY] = review
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
    webp_scope = WEBP_EXACT_COPY_STATUS_TEXT["unsupported_profile"] + " The image must be upright."
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
            multi = check.get(MULTI_EXACT_KEY)
            multi_confirmed = multi is not None and multi["status"] == "supported_pair"
            reflected_webp = check.get(WEBP_EXACT_REFLECTION_KEY)
            reflected_png = check.get(PNG_EXACT_REFLECTION_KEY)
            if exact_review := reflected_png or reflected_webp or check.get(WEBP_EXACT_COPY_KEY):
                review_status = exact_review["status"]
                coverage_text = (PNG_EXACT_REFLECTION_STATUS_TEXT if reflected_png else
                                 WEBP_EXACT_REFLECTION_STATUS_TEXT if reflected_webp else WEBP_EXACT_COPY_STATUS_TEXT)
                check["observation"] = (webp_scope if review_status == "unsupported_profile" and not reflected_png
                                        else coverage_text[review_status])
                if measured := exact_review["measurement"]:
                    transform = check["pairs"][0]["transform"]
                    if reflected_webp or reflected_png:
                        direction = "horizontally" if transform == "horizontal_reflection" else "vertically"
                        carrier = "PNG" if reflected_png else "lossless WebP"
                        check["observation"] = f"Flipping Area B {direction} aligns every RGB pixel with Area A in the {carrier}."
                    elif transform == "same_orientation":
                        check["observation"] = "The two lossless WebP crops match exactly in their original orientation."
                    else:
                        rotation = "90° counterclockwise" if transform == "rotate_90_ccw" else "180°"
                        check["observation"] = f"Rotating Area B {rotation} aligns every RGB pixel with Area A in the lossless WebP."
                    perimeter = ", ".join(f"{value:.2f}" for value in measured["perimeter_mean_absolute_difference"])
                    check["basis_text"] = ("Pixels outside the aligned crops differ on at least three sides. "
                                           f"Compared {measured['matching_pixels']:,} RGB pixels in each crop. "
                                           f"Mean surrounding differences (top, bottom, left, right) {perimeter} of 255 levels.")
                elif review_status == "no_pair":
                    check["basis_text"] = (f"This check covers separated, textured crops with sides from {SMALL_PNG_CROP_MIN}"
                                           f" to {SMALL_PNG_CROP_MAX} pixels.")
            elif small_review := check.get("small_png_copy_review"):
                check["basis_text"] = SMALL_PNG_STATUS_TEXT[small_review["status"]]
                if check.get("basis") == TURNED_PNG_BASIS:
                    turns = "90° counterclockwise" if check["pairs"][0]["transform"] == "rotate_90_ccw" else "180°"
                    check["observation"] = f"Area B matches Area A exactly after rotating it {turns}."
                    check["basis_text"] = "After alignment, pixels outside the crops differ on at least three sides."
                elif base_version in TURNED_PNG_VERSIONS and small_review["status"] == "no_pair":
                    check["basis_text"] = "The small-copy check found no qualifying exact pair with or without right-angle rotation."
                if measured := small_review["measurement"]:
                    perimeter = ", ".join(f"{value:.2f}" for value in measured["perimeter_mean_absolute_difference"])
                    check["basis_text"] += (f" Compared {measured['matching_pixels']:,} RGB pixels in each crop. "
                                            f"Mean surrounding differences (top, bottom, left, right) {perimeter} of 255 levels.")
            elif review := check.get("jpeg_turned_review") or check.get("jpeg_offgrid_review"):
                turned_jpeg = check.get("jpeg_turned_review") is not None
                statuses = TURNED_JPEG_STATUS_TEXT if turned_jpeg else OFFGRID_STATUS_TEXT
                check["basis_text"] = statuses[review["status"]]
                if turned_jpeg and (check.get("jpeg_offgrid_review") or {}).get("status") == "unsupported_profile":
                    check["basis_text"] += " The between-block translation check was outside this file’s compression profile."
                if turned_jpeg and base_version in NATIVE_JPEG_ROTATION_VERSIONS and check.get("jpeg_offgrid_review") is None:
                    if status == "no_pair":
                        check["observation"] = check["basis_text"]
                        check["basis_text"] = ""
                    if review["status"] in {"supported_pair", "no_pair"}:
                        check["basis_text"] += " The rotation check used the original pixel grid."
                if check.get("basis") == TURNED_JPEG_BASIS:
                    if measurements := review.get("measurements"):
                        check["observation"] = "Two separate pairs of rotated areas share closely matching detail."
                        for pair, measured in zip(check["pairs"], measurements, strict=True):
                            turns = "90° counterclockwise" if pair["transform"] == "rotate_90_ccw" else "180°"
                            perimeter = ", ".join("unavailable" if value is None else f"{value:.2f}"
                                                  for value in measured["perimeter_means"])
                            pair["rotation_text"] = (f"Rotate Area B {turns} to align with Area A. "
                                                     f"Mean RGB difference inside {measured['interior_mean']:.2f} of 255 levels. "
                                                     f"Surrounding differences (left, right, top, bottom) {perimeter}.")
                    else:
                        turns = "90° counterclockwise" if check["pairs"][0]["transform"] == "rotate_90_ccw" else "180°"
                        check["observation"] = f"Area B closely matches Area A after rotating it {turns}."
                if measured := review["measurement"]:
                    perimeter = ", ".join("unavailable" if value is None else f"{value:.2f}"
                                          for value in measured["perimeter_means"])
                    check["basis_text"] += (f" Mean RGB difference inside {measured['interior_mean']:.2f} of 255 levels. "
                                             f"Surrounding differences (left, right, top, bottom) {perimeter}.")
            elif (jpeg_review := check.get("jpeg_block_review")) is not None:
                check["basis_text"] = JPEG_BLOCK_STATUS_TEXT[jpeg_review]
            elif check["pairs"] and (basis := check.get("basis")) in TRANSLATION_BASIS_TEXT:
                check["basis_text"] = TRANSLATION_BASIS_TEXT[basis]
            if base_version in WEBP_EXACT_COPY_VERSIONS and not check.get(WEBP_EXACT_COPY_KEY) and not multi and status != SKIPPED_CREDENTIALS:
                outside_size = not exact_copy_size(document["source"]["width"], document["source"]["height"], webp=True)
                if outside_size and status == "unsupported_geometry":
                    check["observation"] = webp_scope
                else:
                    scope = webp_scope if outside_size else "Exact lossless WebP comparison did not run for this result."
                    check["basis_text"] = " ".join(filter(None, (check.get("basis_text"), scope)))
            if base_version in WEBP_EXACT_COPY_VERSIONS and not check["pairs"] and status != SKIPPED_CREDENTIALS:
                check["interpretation"] = "Copy comparisons look for repeated image detail. The origin of the image and AI involvement remain undetermined."
            if base_version in PNG_EXACT_REFLECTION_VERSIONS and not reflected_png and not multi and status != SKIPPED_CREDENTIALS:
                outside_size = not exact_copy_size(document["source"]["width"], document["source"]["height"], webp=False)
                scope = (SMALL_PNG_STATUS_TEXT["unsupported_profile"] if outside_size
                         else "The exact PNG reflection check did not run for this result.")
                if outside_size and status == "unsupported_geometry":
                    check["observation"] = scope
                else:
                    check["basis_text"] = " ".join(filter(None, (check.get("basis_text"), scope)))
            if multi_confirmed:
                check["observation"] = MULTI_EXACT_STATUS["supported_pair"]
                alignments = {"same_orientation": "The two crops match in their original orientation.",
                              "horizontal_reflection": "Flip Area B horizontally to align with Area A.",
                              "vertical_reflection": "Flip Area B vertically to align with Area A.",
                              "rotate_90_ccw": "Rotate Area B 90° counterclockwise to align with Area A.",
                              "rotate_180": "Rotate Area B 180° to align with Area A."}
                for pair, measured in zip(check["pairs"], multi["measurements"], strict=True):
                    perimeter = ", ".join(f"{value:.2f}" for value in measured["perimeter_mean_absolute_difference"])
                    pair["rotation_text"] = (f"{alignments[pair['transform']]} Compared {measured['matching_pixels']:,} RGB pixels in each crop. "
                                             f"Mean surrounding differences (top, bottom, left, right) {perimeter} of 255 levels.")
            elif multi:
                check["basis_text"] = " ".join(filter(None, (check.get("basis_text"), MULTI_EXACT_STATUS[multi["status"]])))
        if field == "copy_translation" and base_version in EXACT_REFLECTION_VERSIONS:
            check["title"] = "Exact and geometric copy comparisons"
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
        _, repeat = repeat_pass(check)
        masks = {"coverage_map": check.get("reliability_mask_sha256"),
                 "repeat_mask": (repeat or {}).get("aligned_mask_png_sha256")}
        if display["map_grid"] and not withheld:
            for name, sha in masks.items():
                if sha:
                    for asset in overview:
                        if (asset["sha256"] == sha and asset["media_type"] == "image/png"
                                and [asset["width"], asset["height"]] == list(display["map_grid"])):
                            data = _asset_bytes(asset, store)
                            display[name] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
                            break
        inspection_checks.append(display)
    return {"available": True, "id": document["id"], "version": document["version"], "sha256": document["sha256"],
            "source_sha256": document["source"]["sha256"], "source": document["source"], "packets": packets,
            "limitations": document["limitations"], "omitted_regions": document["omitted_regions"], "withheld": withheld,
            "inspection_checks": inspection_checks,
            "comparison": comparison, **copy_checks, embedded_pair.KEY: exact, **local_reviews,
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
    for method, packet_id in ((interpolation, "I01"), (prior_grid, "J01")):
        if method.KEY in document:
            result[method.KEY] = copy.deepcopy(document[method.KEY]["record"])
            result["artifacts"].extend({"packet_id": packet_id, "role": role, "sha256": asset["sha256"]}
                                       for role, asset in method.assets(document).items())
    return result
