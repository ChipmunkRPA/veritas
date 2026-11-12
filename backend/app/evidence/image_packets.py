"""Frozen image evidence plates, built from recorded measurements, never from a GET.

The published analysis is not changed. A separately indexed CAS document binds source,
result, regions and rendered PNGs. Attribution and classifier input crops are excluded:
they explain a classifier, not a suspicious edit. Relative forensic peaks are explicitly
inspection candidates, including on authentic images.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import uuid
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.public_projection import alias_signal
from app.core.canonical import FLOAT_DP, canonical_hash, canonical_json
from app.detectors.image import sharp_boundary
from app.engine.media import MAX_PIXELS, MAX_SIDE, refuse_truncated_images
from app.evidence import image_flip_mask
from app.evidence.image_eligibility import (
    validate_flip_consistency,
    validate_gate,
    validate_inspection_coverage,
)
from app.evidence.source_geometry import pixel_bounds
from app.storage.artefacts import get_artefact, register_artefact
from app.storage.db import Analysis, Artefact
from app.storage.objects import ObjectIntegrityError, ObjectNotFound, ObjectStore, default_store

SCHEMA = "chipmunk.image-evidence/1"
VERSION = "image-evidence-3"
CREATOR = "image-evidence-packets/1"
MAX_PACKETS = 6
MAX_PER_MAP = 2
CONTEXT_MAX = 960
_SHA = re.compile(r"[a-f0-9]{64}")
_NS = uuid.UUID("d4b14f21-bbd3-46cc-a733-9381949f594b")


class PacketIntegrityError(ObjectIntegrityError):
    """The evidence cannot be bound safely to the recorded source and result."""


def _verified(store: ObjectStore, sha256: str) -> bytes:
    data = store.read_bytes(sha256)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise PacketIntegrityError("Evidence bytes do not match their recorded hash")
    return data


def packet_hash(document: dict[str, Any]) -> str:
    return canonical_hash({k: v for k, v in document.items() if k != "sha256"})


def current_set_id(analysis: Analysis) -> str:
    return str(uuid.uuid5(_NS, f"{analysis.id}:{analysis.result_hash}:{VERSION}"))


def validate_packet_set(document: dict[str, Any], *, analysis_id: str, revision: int,
                        original_result_hash: str, source_sha256: str) -> None:
    if (document.get("schema") != SCHEMA or document.get("status") != "ready"
            or document.get("analysis_id") != analysis_id or document.get("revision") != revision
            or document.get("original_result_hash") != original_result_hash
            or document.get("source", {}).get("sha256") != source_sha256
            or document.get("sha256") != packet_hash(document)):
        raise PacketIntegrityError("Image evidence does not match its frozen analysis, source or hash")
    source = document["source"]
    width, height = source.get("width"), source.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise PacketIntegrityError("Image evidence has invalid source dimensions")
    try:
        validate_inspection_coverage(document)
    except (KeyError, TypeError, ValueError) as exc:
        raise PacketIntegrityError(str(exc)) from exc
    ids = set()
    for packet in document.get("packets", []):
        if packet.get("id") in ids or packet.get("semantics") == "attribution":
            raise PacketIntegrityError("Ambiguous image evidence or unsupported attribution region")
        ids.add(packet.get("id"))
        for key in ("region", "crop_region"):
            _bounds(packet.get(key), width, height)
        r, c = packet["region"], packet["crop_region"]
        if r["x"] < c["x"] or r["y"] < c["y"] or r["x"] + r["w"] > c["x"] + c["w"] or r["y"] + r["h"] > c["y"] + c["h"]:
            raise PacketIntegrityError("The evidence crop does not contain its region")
        artifacts = packet.get("artifacts", {})
        required = {"context", "crop", "overlay"}
        if document.get("version") in {"image-evidence-2", "image-evidence-3"} and packet.get("evidence_type") == "localisation":
            required.add("mask")
            localisation = packet.get("localisation") or {}
            mask = artifacts.get("mask", {})
            if localisation.get("grid") != [mask.get("width"), mask.get("height")]:
                raise PacketIntegrityError("Localisation mask dimensions differ from the measured grid")
        if set(artifacts) != required:
            raise PacketIntegrityError("Image evidence is missing a required plate")
        for art in artifacts.values():
            if not _SHA.fullmatch(str(art.get("sha256", ""))) or art.get("media_type") != "image/png":
                raise PacketIntegrityError("Image evidence has an invalid plate reference")


def _bounds(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
    try:
        return pixel_bounds(box, width, height)
    except ValueError as error:
        raise PacketIntegrityError(str(error)) from error


def _box(rect: tuple[int, int, int, int]) -> dict[str, int]:
    x0, y0, x1, y1 = rect
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _source(data: bytes) -> tuple[Image.Image, int]:
    refuse_truncated_images()
    with Image.open(io.BytesIO(data)) as src:
        if src.width * src.height > MAX_PIXELS:
            raise PacketIntegrityError("Source exceeds the image pixel limit")
        orientation = int(src.getexif().get(274, 1) or 1)
        return ImageOps.exif_transpose(src).convert("RGB"), orientation


def _transform(space: dict[str, Any] | None, width: int, height: int) -> tuple[float, float]:
    if not space:
        raise PacketIntegrityError("No recorded source coordinate transform")
    sw, sh, scale = space.get("width", 0), space.get("height", 0), space.get("scale_to_original", 0)
    if min(sw, sh, scale) <= 0 or not all(math.isfinite(v) for v in (sw, sh, scale)):
        raise PacketIntegrityError("Invalid source coordinate transform")
    # Existing adapters store one scale, with at most one pixel of thumbnail rounding.
    if abs(sw * scale - width) > max(1.01, scale) or abs(sh * scale - height) > max(1.01, scale):
        raise PacketIntegrityError("Recorded map coordinates do not align with the source")
    return width / sw, height / sh


def _read_map(db: Session, store: ObjectStore, aid: str, heat: dict[str, Any], width: int, height: int):
    _transform(heat.get("coordinate_space"), width, height)
    row = get_artefact(db, heat["payload_artefact_id"], analysis_id=aid)
    if heat.get("sha256") != row.sha256 or row.media_type != "image/png":
        raise PacketIntegrityError("The scalar map no longer matches its frozen exhibit")
    data = _verified(store, row.sha256)
    refuse_truncated_images()
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "PNG" or im.mode != "L" or im.width * im.height > MAX_PIXELS or list(im.size) != list(heat.get("grid", [])):
            raise PacketIntegrityError("The stored scalar map has different dimensions")
        grid = np.asarray(im.convert("L"), dtype=np.float64)
    low, high = heat.get("value_range", (None, None))
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in (low, high)) or high <= low:
        raise PacketIntegrityError("The scalar map has no valid measurement range")
    values = low + grid * ((high - low) / 255.0)
    reliable = np.ones(grid.shape, dtype=bool)
    mask_sha = None
    if heat.get("reliability_mask_artefact_id"):
        mask = get_artefact(db, heat["reliability_mask_artefact_id"], analysis_id=aid)
        mask_sha = mask.sha256
        with Image.open(io.BytesIO(_verified(store, mask.sha256))) as im:
            if im.format != "PNG" or im.mode != "L" or im.size != (grid.shape[1], grid.shape[0]):
                raise PacketIntegrityError("Reliability mask does not match the scalar map")
            reliable = np.asarray(im.convert("L")) > 127
    return values, reliable, row.sha256, mask_sha


def _public_signal(signal: dict[str, Any]) -> dict[str, str | None]:
    original = signal["id"]
    public = alias_signal(original)
    display = signal.get("display_id")
    return {"id": public, "display_id": display if isinstance(display, str) and re.fullmatch(r"S\d+", display) else None}


def _png(im: Image.Image) -> bytes:
    out = io.BytesIO()
    # A fresh image strips source EXIF and arbitrary text metadata from generated plates.
    clean = Image.fromarray(np.asarray(im.convert("RGB")))
    clean.save(out, format="PNG", optimize=False)
    return out.getvalue()


def _plates(source: Image.Image, rect: tuple[int, int, int, int], values: np.ndarray, reliable: np.ndarray,
            packet_id: str, component: np.ndarray | None = None) -> tuple[dict[str, Image.Image], dict[str, int]]:
    width, height = source.size
    x0, y0, x1, y1 = rect
    pad = max(24, round(max(x1 - x0, y1 - y0) * 0.12))
    crop_rect = (max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad))
    crop = source.crop(crop_rect)
    valid = values[reliable]
    low, high = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)
    strength = np.zeros(values.shape) if high <= low else np.clip((values - low) / (high - low), 0, 1)
    strength *= reliable
    if component is not None:
        strength = component.astype(np.float64)
    # Nearest-neighbour keeps the stored map grid visible; orange intensity is relative, never a probability scale.
    alpha = Image.fromarray(np.uint8(strength * 150)).resize(source.size, Image.Resampling.NEAREST).crop(crop_rect)
    overlay = Image.composite(Image.new("RGB", crop.size, (244, 158, 45)), crop, alpha)
    offset = (x0 - crop_rect[0], y0 - crop_rect[1], x1 - crop_rect[0] - 1, y1 - crop_rect[1] - 1)
    ImageDraw.Draw(overlay).rectangle(offset, outline=(250, 190, 65), width=max(2, round(max(crop.size) / 300)))
    context = source.copy()
    context.thumbnail((CONTEXT_MAX, CONTEXT_MAX), Image.Resampling.LANCZOS)
    sx, sy = context.width / width, context.height / height
    outline = (round(x0 * sx), round(y0 * sy), round((x1 - 1) * sx), round((y1 - 1) * sy))
    draw = ImageDraw.Draw(context)
    draw.rectangle(outline, outline=(250, 170, 40), width=3)
    tx, ty = outline[0], max(0, outline[1] - 17)
    draw.rectangle((tx, ty, min(context.width - 1, tx + 35), ty + 16), fill=(30, 35, 30))
    draw.text((tx + 3, ty + 2), packet_id, fill=(255, 230, 165))
    plates = {"context": context, "crop": crop, "overlay": overlay}
    if component is not None:
        plates["mask"] = Image.fromarray(component.astype(np.uint8) * 255)
    return plates, _box(crop_rect)


def _mask_rule(signals: list[dict], values: np.ndarray) -> dict:
    rules = [s.get("details", {}).get("localisation") for s in signals]
    rules = [r for r in rules if isinstance(r, dict) and r.get("schema") == "recorded-mask/1"]
    if not rules or any(r != rules[0] for r in rules[1:]):
        raise PacketIntegrityError("A local editing map has no unambiguous recorded mask rule")
    rule = rules[0]
    threshold = rule.get("threshold")
    fraction = rule.get("minimum_region_fraction")
    if (not isinstance(threshold, int | float) or not 0 <= threshold <= 1
            or not isinstance(fraction, int | float) or not 0 < fraction <= 1
            or rule.get("component_connectivity") != 8 or rule.get("comparison") != "greater_or_equal"
            or rule.get("scalar_encoding") != "linear_uint8" or rule.get("grid") != list(values.shape[::-1])):
        raise PacketIntegrityError("The recorded local editing mask rule is unsupported")
    return rule


def _localisation_component(signals: list[dict], values: np.ndarray, reliable: np.ndarray,
                            rect: tuple[int, int, int, int], width: int, height: int):
    """Reconstruct one recorded component, never the enclosing rectangle."""
    rule = _mask_rule(signals, values)
    threshold, fraction = rule["threshold"], rule["minimum_region_fraction"]
    gate = rule.get("candidate_eligibility")
    if gate is not None:
        try:
            validate_gate(gate)
        except ValueError as exc:
            raise PacketIntegrityError(str(exc)) from exc
        if not gate["eligible"]:
            raise PacketIntegrityError("Image candidate selection withheld this map")
    binary = ((values >= threshold) & reliable).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    gh, gw = values.shape
    matches = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < max(1, math.ceil(fraction * values.size)):
            continue
        left, top = round(x * width / gw, 1), round(y * height / gh, 1)
        right, bottom = round((x + w) * width / gw, 1), round((y + h) * height / gh, 1)
        mapped = _bounds({"x": left, "y": top, "w": round(right - left, 1),
                          "h": round(bottom - top, 1)}, width, height)
        if all(abs(a - b) <= 1 for a, b in zip(mapped, rect, strict=True)):
            matches.append(i)
    if len(matches) != 1:
        raise PacketIntegrityError("The recorded region cannot be matched to one local editing component")
    component = labels == matches[0]
    repeat = rule.get("resolution_policy", {}).get("flip_consistency")
    retained = None
    if repeat is not None:
        bounds = list(_box(rect).values())
        retained = next((row for row in repeat["candidates"] if row["retained"] and row["bounds"] == bounds), None)
        if retained is None or retained["source_pixels"] != int(component.sum()):
            raise PacketIntegrityError("Image region does not match its recorded repeat-pass selection")
    return component, {"threshold": threshold, "comparison": "greater_or_equal", "grid": [gw, gh],
                       "region_cells": int(component.sum()), "region_area_fraction": float(component.mean()),
                       "reliable_area_fraction": float(reliable.mean()),
                       "source_pixels_per_cell": [width / gw, height / gh],
                       "resolution_policy": rule.get("resolution_policy", {}),
                       "excluded_area_fraction": float(1 - reliable.mean()),
                       "selection": "connected_component", "connectivity": 8,
                       "capability": "ai_inpainting" if rule.get("capability") == "ai_inpainting" else "local_editing",
                       "validation_status": "limited_local_evaluation" if gate else "local_evaluation_pending",
                       **({"flip_consistency": retained} if retained is not None else {}),
                       **({"candidate_eligibility": gate} if gate is not None else {})}


def _inspection_record(signals: list[dict], heat: dict, values: np.ndarray, reliable: np.ndarray,
                       scalar_sha: str, source_sha: str) -> dict:
    rule = _mask_rule(signals, values)
    gate = rule.get("candidate_eligibility")
    # New gated adapters must never be silently interpreted as an ungated legacy check.
    if heat.get("created_by") in {"loc.fused", "loc.splice"} and gate is None:
        raise PacketIntegrityError("The image candidate eligibility record is missing")
    if gate is not None:
        try:
            validate_gate(gate, source_sha256=source_sha, scalar_map_sha256=scalar_sha)
        except ValueError as exc:
            raise PacketIntegrityError(str(exc)) from exc
    binary = ((values >= rule["threshold"]) & reliable).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum = max(1, math.ceil(rule["minimum_region_fraction"] * values.size))
    raw = sum(int(s[cv2.CC_STAT_AREA]) >= minimum for s in stats[1:])
    eligible = raw if gate is None or gate["eligible"] else 0
    repeat = rule.get("resolution_policy", {}).get("flip_consistency")
    if repeat is not None:
        try:
            eligible = validate_flip_consistency(repeat, eligible, values.shape[1], values.shape[0])
        except ValueError as exc:
            raise PacketIntegrityError(str(exc)) from exc
    return {"signal_refs": [_public_signal(s) for s in signals], "exhibit_ref": heat["id"],
            "scalar_map_sha256": scalar_sha, "map_grid": list(values.shape[::-1]),
            "selection_rule": {k: rule[k] for k in ("threshold", "comparison", "scalar_encoding",
                                "component_connectivity", "minimum_region_fraction")},
            "resolution_policy": rule.get("resolution_policy", {}),
            "excluded_area_fraction": float(1 - reliable.mean()),
            "raw_components": raw, "eligible_components": eligible, "selected_components": 0,
            "omitted_components": eligible,
            **({"flip_consistency": repeat} if repeat is not None else {}),
            **({"candidate_eligibility": gate} if gate is not None else {})}


def _copy_for(semantics: str, creator: str) -> tuple[str, str, str, str]:
    if creator == sharp_boundary.DETECTOR_ID:
        return ("Sharp boundary to review", "Boundary contrast", "relative response",
                "The four sides of this rectangle have abrupt color steps compared with nearby parallel steps. "
                "Pastes, local adjustments and ordinary rectangular edges can produce this pattern. "
                "This evidence carries zero weight in AI confidence.")
    if creator == "img.forensic.local_jpeg_grid":
        return ("Local compression-grid difference", "Earlier grid difference", "relative response",
                "This area carries a different prior JPEG block alignment from the dominant grid. "
                "Ordinary cropping, separate saves or compositing can produce this pattern. "
                "The finding cannot identify AI generation and carries zero weight in AI confidence.")
    if semantics == "error_level":
        return ("Compression difference", "JPEG re-save difference", "channel levels",
                "A JPEG re-save changes this region differently. Texture, sharp edges, earlier saves and ordinary editing can all cause this pattern; it does not identify AI-generated pixels.")
    if semantics == "residual":
        return ("Noise variation", "Local residual variation", "channel levels",
                "Noise residuals vary in this region. Texture, lighting, sharpening, denoising and compression can produce this pattern without AI editing.")
    if creator == "img.forensic.jpeg_dq":
        return ("Compression-history difference", "Compression-history score", "uncalibrated score",
                "This map compares compression traces across the image. Different save histories or smooth and saturated areas can explain a difference; it is not specific to AI editing.")
    return ("Local editing candidate", "Localisation response", "uncalibrated score",
            "A local editing check marked this region. The response is not a calibrated pixel probability, and ordinary edits or unfamiliar images may also trigger it.")


def _sharp_boundary_measurements(signals: list[dict]) -> list[dict]:
    if len(signals) != 1:
        raise PacketIntegrityError("Sharp boundary evidence requires one supporting check")
    signal = signals[0]
    details = (signal.get("details") or {}).get("sharp_boundary") or {}
    sides = details.get("sides") or {}
    order = ("left", "top", "right", "bottom")
    score = signal.get("raw_score")
    minimum = round(sharp_boundary.REVIEW_POINT, FLOAT_DP)
    maximum = sharp_boundary.MAX_STEP_RESPONSE + .00001  # Native float32 accumulation.
    if (signal.get("detector_id") != sharp_boundary.DETECTOR_ID or signal.get("effective_weight") != 0
            or signal.get("raw_kind") != "score" or details.get("schema") != sharp_boundary.SCHEMA
            or set(sides) != set(order) or any(type(sides[key]) not in (int, float) or not 0 < sides[key] <= maximum for key in order)
            or type(score) not in (int, float) or not minimum <= score <= maximum
            or abs(score - math.prod(sides[key] for key in order) ** .25) > .000002):
        raise PacketIntegrityError("Sharp boundary evidence differs from its recorded four-side response")
    return [{"label": "Combined boundary response", "value": score, "unit": "relative response"},
            *[{"label": f"{key.title()} side", "value": sides[key], "unit": "relative response"} for key in order]]


def _signals_for(result: dict[str, Any], heat: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in result.get("signals", []) if s.get("status", "ok") == "ok"
            and (heat["id"] in s.get("exhibit_ids", []) or s.get("id") in heat.get("source_signal_ids", []))]


def _native_mapping_issue(heat: dict[str, Any], signals: list[dict[str, Any]], width: int, height: int) -> str | None:
    """Legacy native-forensic previews trimmed remainder pixels before block averaging.

    Their recorded scale stretched the remaining cells back over the full source. A
    scalar PNG alone cannot recover the missing edge or its actual transform. Keep
    divisible legacy grids, but require the producer's full-extent marker otherwise.
    """
    if heat.get("created_by") not in {"img.forensic.ela", "img.forensic.noise"}:
        return None
    factor = max(1, math.ceil(max(width, height) / MAX_SIDE))
    if factor == 1 or (width % factor == 0 and height % factor == 0):
        return None
    space = heat.get("coordinate_space") or {}
    full_extent = (space.get("width") == width and space.get("height") == height
                   and space.get("scale_to_original") == 1.0)
    marked = any(s.get("detector_id") == heat.get("created_by") and isinstance(s.get("details"), dict)
                 and s["details"].get("display_mapping_version") == "full-extent-area/1" for s in signals)
    if full_extent and marked:
        return None
    return "An older reduced forensic map has an ambiguous edge mapping and was omitted. Re-run the image to obtain correctly aligned evidence."


def validate_packet_source_mapping(document: dict[str, Any], result: dict[str, Any]) -> None:
    """Refuse an already-frozen unsafe map without rewriting its evidence or signature."""
    source = document["source"]
    width, height = source["width"], source["height"]
    maps = {e["id"]: e for e in result.get("exhibits", []) if e.get("kind") == "heatmap"}
    for check in (document.get("coverage") or {}).get("inspection_checks", []):
        heat = maps.get(check["exhibit_ref"])
        if heat is None or heat.get("sha256") != check.get("scalar_map_sha256"):
            raise PacketIntegrityError("Image inspection coverage uses a different original map")
        signals = _signals_for(result, heat)
        rules = [s.get("details", {}).get("localisation", {}) for s in signals]
        if not rules or any(r.get("candidate_eligibility") != check.get("candidate_eligibility") for r in rules):
            raise PacketIntegrityError("Image inspection eligibility differs from the frozen check")
        if any(r.get("resolution_policy", {}).get("flip_consistency") != check.get("flip_consistency") for r in rules):
            raise PacketIntegrityError("Image repeat-pass evidence differs from the frozen check")
    for packet in document.get("packets", []):
        for ref in packet.get("exhibit_refs", []):
            heat = maps.get(ref)
            if heat is not None and _native_mapping_issue(heat, _signals_for(result, heat), width, height):
                raise PacketIntegrityError("The frozen evidence uses an ambiguous older forensic-map edge mapping")


def build_packet_set(db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any]) -> dict[str, Any]:
    """Build and index an immutable public set; transaction commit belongs to the caller."""
    if analysis.kind != "image" or not analysis.result_hash:
        raise ValueError("Image evidence needs a published image result")
    if canonical_hash(result) != analysis.result_hash or result.get("input", {}).get("sha256") != analysis.sha256:
        raise PacketIntegrityError("Published result or source identity does not match")
    set_id = current_set_id(analysis)
    source, orientation = _source(_verified(store, analysis.sha256))
    width, height = source.size
    document: dict[str, Any] = {
        "schema": SCHEMA, "status": "ready", "id": set_id, "version": VERSION,
        "analysis_id": analysis.id, "revision": analysis.revision, "original_result_hash": analysis.result_hash,
        "source": {"sha256": analysis.sha256, "width": width, "height": height,
                   "coordinate_space": "exif-oriented-original", "exif_orientation": orientation},
        "packets": [], "limitations": [], "omitted_regions": 0,
        "overview": {}, "coverage": {"inspection_checks": []},
    }
    exhibits = result.get("exhibits", [])
    region_exhibits = [e for e in exhibits if e.get("kind") == "regions" and e.get("frame_pts") is None]
    eligible = [e for e in exhibits if e.get("kind") == "heatmap" and e.get("semantics") in {"error_level", "residual", "p_manipulated"}
                and e.get("frame_pts") is None and e.get("media_ref") in {"original", "analysis_copy"}]
    diagnostics = []
    def priority(heat: dict) -> tuple[int, str, str]:
        creator = heat.get("created_by", "")
        rank = 0 if creator.startswith("loc.") else 1 if creator == sharp_boundary.DETECTOR_ID else 2
        return rank, heat.get("semantics", ""), heat["id"]

    for heat in sorted(eligible, key=priority):
        signals = _signals_for(result, heat)
        if not signals:
            diagnostics.append("A map without a successful supporting check was omitted.")
            continue
        signal_ids = {s["id"] for s in signals}
        matching = [e for e in region_exhibits if signal_ids.intersection(e.get("source_signal_ids", []))
                    or (e.get("created_by") == heat.get("created_by") and e.get("created_by", "").startswith("loc."))]
        # An ambiguous shared-source map must not borrow a region from a different map.
        competing = [e for e in eligible if signal_ids.intersection(s["id"] for s in _signals_for(result, e))]
        if len(competing) != 1:
            diagnostics.append("A region with an ambiguous map association was omitted.")
            continue
        if issue := _native_mapping_issue(heat, signals, width, height):
            diagnostics.append(issue)
            document["omitted_regions"] += sum(len(e.get("regions", [])) for e in matching)
            continue
        values, reliable, scalar_sha, mask_sha = _read_map(db, store, analysis.id, heat, width, height)
        localiser = heat["semantics"] == "p_manipulated" and heat.get("created_by", "").startswith("loc.")
        inspection = None
        if localiser:
            proof_sha = None
            try:
                inspection = _inspection_record(signals, heat, values, reliable, scalar_sha, analysis.sha256)
                inspection["reliability_mask_sha256"] = mask_sha
                repeat = inspection.get("flip_consistency") or {}
                if proof_sha := repeat.get("aligned_mask_png_sha256"):
                    proof = db.scalar(select(Artefact.id).where(
                        Artefact.analysis_id == analysis.id, Artefact.sha256 == proof_sha,
                        Artefact.role == "mask", Artefact.media_type == "image/png",
                        Artefact.created_by == heat.get("created_by")).limit(1))
                    if proof is None or mask_sha is None:
                        raise PacketIntegrityError("The repeat-pass proof belongs to a different check")
                    image_flip_mask.verify_overlap(values, reliable, inspection["selection_rule"]["threshold"],
                                                   repeat, _verified(store, proof_sha))
            except (ObjectIntegrityError, ObjectNotFound, ValueError, OSError):
                diagnostics.append("A local editing region without a reproducible component mask or image eligibility record was omitted; reanalysis is required.")
                document["omitted_regions"] += sum(len(e.get("regions", [])) for e in matching)
                continue
            index = len(document["coverage"]["inspection_checks"]) + 1
            document["coverage"]["inspection_checks"].append(inspection)
            for role, sha in (("response", scalar_sha), ("reliability", mask_sha), ("aligned_flip_mask", proof_sha)):
                if sha is None:
                    continue
                # Give exactly these declared measurement bytes a presentation
                # index; the general raw artefact namespace remains private.
                register_artefact(db, store, _verified(store, sha), media_type="image/png", role="preview",
                                 created_by=CREATOR, analysis_id=analysis.id)
                document["overview"][f"inspection_{index:02d}_{role}"] = {
                    "sha256": sha, "media_type": "image/png", "width": values.shape[1], "height": values.shape[0],
                    "url": f"/api/v2/analyses/{analysis.id}/evidence-packets/{set_id}/images/{sha}.png"}
            gate = inspection.get("candidate_eligibility")
            if gate is not None and not gate["eligible"]:
                diagnostics.append("A whole-image selection check withheld its local response regions. The measured map remains recorded; this does not confirm authenticity.")
                continue
        if not matching:
            continue
        if not reliable.any() or (not localiser and np.ptp(values[reliable]) == 0):
            continue
        per_map = 0
        for regions in sorted(matching, key=lambda e: e["id"]):
            try:
                sx, sy = _transform(regions.get("coordinate_space"), width, height)
            except PacketIntegrityError:
                diagnostics.append("A region without a verified source-coordinate mapping was omitted.")
                document["omitted_regions"] += len(regions.get("regions", []))
                continue
            for region in regions.get("regions", []):
                if per_map >= MAX_PER_MAP or len(document["packets"]) >= MAX_PACKETS:
                    document["omitted_regions"] += 1
                    continue
                geometry = region.get("region", {})
                if geometry.get("kind") != "bbox" or geometry.get("frame_pts") is not None:
                    document["omitted_regions"] += 1
                    continue
                b = geometry.get("bbox") or {}
                try:
                    rect = _bounds({"x": b["x"] * sx, "y": b["y"] * sy, "w": b["w"] * sx, "h": b["h"] * sy}, width, height)
                except (KeyError, TypeError, PacketIntegrityError):
                    diagnostics.append("An invalid or out-of-bounds source region was omitted.")
                    document["omitted_regions"] += 1
                    continue
                component, localisation = None, None
                if localiser:
                    try:
                        component, localisation = _localisation_component(signals, values, reliable, rect, width, height)
                    except PacketIntegrityError:
                        diagnostics.append("A local editing region without a reproducible component mask was omitted; reanalysis is required.")
                        document["omitted_regions"] += 1
                        continue
                packet_id = f"E{len(document['packets']) + 1:02d}"
                plates, crop_region = _plates(source, rect, values, reliable, packet_id, component)
                x0, y0, x1, y1 = rect
                gh, gw = values.shape
                gx0, gy0 = math.floor(x0 * gw / width), math.floor(y0 * gh / height)
                gx1, gy1 = min(gw, math.ceil(x1 * gw / width)), min(gh, math.ceil(y1 * gh / height))
                measured = values[gy0:gy1, gx0:gx1][reliable[gy0:gy1, gx0:gx1]]
                if component is not None:
                    measured = values[component]
                if measured.size == 0:
                    document["omitted_regions"] += 1
                    continue
                c = crop_region
                cx0, cy0 = math.floor(c["x"] * gw / width), math.floor(c["y"] * gh / height)
                cx1, cy1 = min(gw, math.ceil((c["x"] + c["w"]) * gw / width)), min(gh, math.ceil((c["y"] + c["h"]) * gh / height))
                surrounding = reliable[cy0:cy1, cx0:cx1].copy()
                surrounding[max(0, gy0-cy0):gy1-cy0, max(0, gx0-cx0):gx1-cx0] = False
                around = values[cy0:cy1, cx0:cx1][surrounding]
                title, measure, unit, interpretation = _copy_for(heat["semantics"], heat.get("created_by", ""))
                relative = regions.get("created_by") == "localise" and not heat.get("calibrated", False)
                limits = ["This inspection candidate does not establish AI generation or manipulation.",
                          "Measurements use the stored scalar map and its 8-bit quantisation; colour intensity is relative."]
                if relative:
                    limits.append("Selected from the highest 5% of map values. Unedited images can also have such regions.")
                if not heat.get("calibrated", False):
                    limits.append("No calibrated probability is available for this region.")
                mean = round(float(measured.mean()), 6)
                measurements = [{"label": f"{measure}: region mean", "value": mean, "unit": unit},
                                {"label": "Region peak", "value": round(float(measured.max()), 6), "unit": unit}]
                observation = f"{measure} averages {mean:.3f} in the outlined region"
                if localisation:
                    observation = (f"A local editing check marked {100 * localisation['region_area_fraction']:.2f}% of the image "
                                   f"in this connected region at score ≥ {localisation['threshold']:.2f}; "
                                   f"its mean response is {mean:.3f}")
                    if flip := localisation.get("flip_consistency"):
                        overlap = round(100 * flip["shared_pixels"] / flip["source_pixels"], 3)
                        measurements.append({"label": "Flipped-pass overlap on the source grid", "value": overlap, "unit": "%"})
                        limits.append("The same check supported this region after a horizontal flip. This measures repeatability and carries zero weight in AI authorship confidence.")
                    limits.append(f"The measured mask grid is {gw} × {gh}; displayed boundaries are resampled to the original, not original-pixel certainty.")
                    limits.append("This check is not validated for general AI editing detection. Local testing found missed edits and false alarms on authentic images.")
                    if np.ptp(values[reliable]) == 0:
                        limits.append("The response is uniform over the measurable grid. It cannot distinguish an edited area from its surroundings; component edges may only follow excluded cells.")
                    policy = localisation["resolution_policy"]
                    if policy.get("mode") == "screening" and isinstance(policy.get("input_px"), int):
                        limits.append(f"The whole image was resized to {policy['input_px']} × {policy['input_px']} for this check; a larger displayed map does not add detection detail.")
                    if localisation["excluded_area_fraction"]:
                        limits.append(f"{100 * localisation['excluded_area_fraction']:.2f}% of the measured grid was excluded as less reliable; those areas were not cleared.")
                if around.size:
                    nearby = round(float(around.mean()), 6)
                    measurements.append({"label": "Nearby context mean", "value": nearby, "unit": unit})
                    observation += f", compared with {nearby:.3f} in the surrounding crop"
                    limits.append("The surrounding crop is context, not a matched control; texture and lighting may differ.")
                observation += "."
                boundary_cue = heat.get("created_by") == sharp_boundary.DETECTOR_ID
                if boundary_cue:
                    if (gw, gh, sx, sy) != (width, height, 1.0, 1.0):
                        raise PacketIntegrityError("Sharp boundary evidence must use native source coordinates")
                    measurements = _sharp_boundary_measurements(signals)
                    observation = (f"The combined boundary response was {measurements[0]['value']:.3f}. "
                                   "Each side showed a stronger color step than the nearby comparison lines.")
                    limits = ["The check measures abrupt, roughly rectangular seams. Soft edges and irregular edits can be missed.",
                              "An ordinary object edge or a conventional edit can produce the same evidence."]
                refs = signal_ids | {heat["id"], regions["id"]}
                findings = [f["id"] for f in result.get("findings", [])
                            if f.get("family") not in {"coverage", "F9_distribution_conditions"}
                            and refs.intersection(f.get("evidence_ids", []))]
                artifacts = {}
                for role, image in plates.items():
                    art = register_artefact(db, store, _png(image), media_type="image/png", role="preview",
                                            created_by=CREATOR, analysis_id=analysis.id)
                    artifacts[role] = {"sha256": art.sha256, "media_type": "image/png", "width": image.width,
                                       "height": image.height,
                                       "url": f"/api/v2/analyses/{analysis.id}/evidence-packets/{set_id}/images/{art.sha256}.png"}
                packet = {"id": packet_id, "finding_ids": findings, "title": title,
                          "evidence_type": "localisation" if localiser else "anomaly", "semantics": heat["semantics"],
                          "region": _box(rect), "crop_region": crop_region,
                          "observation": observation, "interpretation": interpretation, "limitations": limits,
                          "overlay_legend": f"Orange intensity shows relative {measure.lower()} within this image, not an AI probability. The outline marks the recorded inspection region.",
                          "measurements": measurements, "signal_refs": [_public_signal(s) for s in signals],
                          "exhibit_refs": [heat["id"], regions["id"]], "artifacts": artifacts,
                          "anchor": {"original_sha256": analysis.sha256, "coordinate_space": "exif-oriented-original",
                                     "exif_orientation": orientation, "region_to_original": [sx, 0, 0, 0, sy, 0],
                                     "scalar_sha256": scalar_sha, "reliability_mask_sha256": mask_sha,
                                     "map_grid": [gw, gh], "value_range": heat["value_range"]},
                          "derivation": {"version": VERSION, "selection": "relative_top_5_percent" if relative else "recorded_region",
                                         "overlay": "relative_orange_nearest_neighbour", "max_alpha": 150 / 255,
                                         "crop_padding": "12% of region long side, minimum 24 source pixels"}}
                if boundary_cue:
                    packet["overlay_legend"] = "Orange shows the recorded four sides and their relative contrast. The outlined interior gives location context."
                if localisation:
                    packet["localisation"] = localisation
                    packet["derivation"]["selection"] = "recorded_connected_component"
                    packet["derivation"]["overlay"] = "threshold_component_orange_nearest_neighbour"
                    packet["overlay_legend"] = (f"Orange marks this connected mask at score ≥ {localisation['threshold']:.2f} "
                                                "on measurable cells. The box is navigation context; uncoloured pixels are not confirmed authentic.")
                document["packets"].append(packet)
                per_map += 1
                if inspection is not None:
                    inspection["selected_components"] += 1
                    inspection["omitted_components"] = inspection["eligible_components"] - inspection["selected_components"]
    document["limitations"] = list(dict.fromkeys(diagnostics))
    if document["packets"]:
        document["limitations"].append("These plates explain recorded local checks. They do not change the original analysis decision.")
        if not any(p["evidence_type"] == "localisation" for p in document["packets"]):
            document["limitations"].append(
                "No dedicated local editing region was selected. The coverage record retains measured maps and any withheld candidates; the regions shown are forensic inspection candidates."
                if document["coverage"]["inspection_checks"] else
                "No dedicated manipulation-localisation result is available. The regions shown are forensic inspection candidates.")
    else:
        document["limitations"].append("No supported local region was recorded. An overall score or classifier attention map cannot identify edited pixels.")
    if document["omitted_regions"]:
        document["limitations"].append(f"Showing at most {MAX_PER_MAP} recorded regions per map and {MAX_PACKETS} in total; {document['omitted_regions']} additional or unusable regions are omitted.")
    # Return exactly the canonical values that readers and report snapshots will see.
    document = json.loads(canonical_json(document))
    document["sha256"] = packet_hash(document)
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    register_artefact(db, store, canonical_json(document), media_type="application/json", role="exhibit_payload",
                     created_by=CREATOR, analysis_id=analysis.id, artefact_id=set_id)
    return document


def load_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, set_id: str | None = None) -> dict[str, Any] | None:
    query = select(Artefact).where(Artefact.analysis_id == analysis.id, Artefact.created_by == CREATOR,
                                   Artefact.role == "exhibit_payload", Artefact.media_type == "application/json")
    if set_id is not None:
        query = query.where(Artefact.id == set_id)
    row = db.scalars(query.order_by(Artefact.created_at.desc(), Artefact.id)).first()
    if row is None:
        return None
    document = json.loads(_verified(store, row.sha256))
    if document.get("id") != row.id:
        raise PacketIntegrityError("Image evidence identity differs from its stored object")
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    width, height = document["source"]["width"], document["source"]["height"]
    factor = max(1, math.ceil(max(width, height) / MAX_SIDE))
    if document["packets"] and factor > 1 and (width % factor or height % factor):
        validate_packet_source_mapping(document, json.loads(_verified(store, analysis.result_hash)))
    # Do not offer verified evidence when its original has been lost or modified.
    store.verify(analysis.sha256)
    return document


def persist_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if analysis.kind != "image" or not analysis.result_hash:
        return None
    current_id = current_set_id(analysis)
    existing = load_packet_set(db, analysis, store, set_id=current_id)
    if existing is not None:
        return existing
    if result is None:
        result = json.loads(_verified(store, analysis.result_hash))
    return build_packet_set(db, analysis, store, result)


def unavailable_packet_set(analysis: Analysis) -> dict[str, Any]:
    return {"schema": SCHEMA, "status": "not_generated" if analysis.kind == "image" else "not_applicable",
            "id": None, "version": VERSION, "sha256": None, "analysis_id": analysis.id, "revision": analysis.revision,
            "original_result_hash": analysis.result_hash, "source": {"sha256": analysis.sha256, "width": None,
            "height": None, "coordinate_space": "exif-oriented-original"}, "packets": [], "omitted_regions": 0,
            "limitations": ["Image evidence plates have not been generated for this revision." if analysis.kind == "image"
                            else "This evidence view currently supports images."]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable image evidence plates for existing local analyses")
    parser.add_argument("analysis_ids", nargs="+")
    args = parser.parse_args()
    from app.audit import chain as audit
    for aid in args.analysis_ids:
        with audit.writing() as db:
            a = db.get(Analysis, aid)
            if a is None or a.status != "done":
                raise ValueError("The requested analysis is not completed")
            existing = load_packet_set(db, a, default_store(), set_id=current_set_id(a))
            packet_set = persist_packet_set(db, a, default_store())
            audit.commit_with(db, [{"action": "image_evidence_generated", "actor_id": audit.SYSTEM_ACTOR,
                                   "analysis_id": aid, "after_hash": packet_set["sha256"],
                                   "object_ids": [packet_set["id"]], "details": {"version": VERSION}}] if packet_set and existing is None else [])
        if packet_set:
            print(json.dumps({"analysis_id": aid, "set_id": packet_set["id"], "packets": len(packet_set["packets"]),
                              "sha256": packet_set["sha256"]}))


if __name__ == "__main__":
    main()
