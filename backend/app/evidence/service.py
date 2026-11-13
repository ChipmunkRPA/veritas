"""Shared access to versioned modality evidence; reads never build missing sets."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from sqlalchemy.orm import Session

from app.evidence import image_embedded_pair_binding as embedded_pair
from app.evidence.image_copy_similarity_binding import copy_records
from app.evidence.video_packet_versions import NESTED_VERSIONS
from app.storage.db import Analysis
from app.storage.objects import ObjectIntegrityError, ObjectStore

KINDS = frozenset({"image", "video", "audio", "document"})
ASSET_TYPES = {"png": "image/png", "wav": "audio/wav", "mp4": "video/mp4"}


def module_for_kind(kind: str) -> ModuleType | None:
    if kind not in KINDS:
        return None
    name = "app.evidence.image_packets_v8" if kind == "image" else f"app.evidence.{kind}_packets"
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        # Missing optional modality implementation is an explicit unavailable state.
        # Missing dependencies inside an installed module are real failures.
        if exc.name == name:
            return None
        raise


def current_set_id(analysis: Analysis) -> str | None:
    module = module_for_kind(analysis.kind)
    return module.current_set_id(analysis) if module is not None else None


def build_packet_set(db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any]) -> dict[str, Any]:
    module = module_for_kind(analysis.kind)
    if module is None:
        raise ValueError("An evidence implementation is not available for this media type")
    return module.build_packet_set(db, analysis, store, result)


def load_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, set_id: str | None = None) -> dict[str, Any] | None:
    module = module_for_kind(analysis.kind)
    return module.load_packet_set(db, analysis, store, set_id=set_id) if module is not None else None


def persist_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
    module = module_for_kind(analysis.kind)
    return module.persist_packet_set(db, analysis, store, result=result) if module is not None else None


def unavailable_packet_set(analysis: Analysis) -> dict[str, Any]:
    module = module_for_kind(analysis.kind)
    if module is not None:
        return module.unavailable_packet_set(analysis)
    return {"schema": "chipmunk.evidence/1", "status": "not_generated", "id": None, "sha256": None,
            "analysis_id": analysis.id, "revision": analysis.revision, "original_result_hash": analysis.result_hash,
            "source": {"sha256": analysis.sha256}, "packets": [],
            "limitations": ["An evidence implementation is not available for this media type on this server."]}


def packet_hash(document: dict[str, Any]) -> str:
    from app.core.canonical import canonical_hash

    return canonical_hash({k: v for k, v in document.items() if k != "sha256"})


def validate_packet_set(document: dict[str, Any], *, analysis_id: str, revision: int,
                        original_result_hash: str, source_sha256: str) -> None:
    schema = document.get("schema", "")
    kind = next((k for k in KINDS if schema.startswith(f"chipmunk.{k}-evidence/")), None)
    module = module_for_kind(kind) if kind else None
    if module is None:
        raise ObjectIntegrityError("The frozen evidence schema is unsupported")
    module.validate_packet_set(document, analysis_id=analysis_id, revision=revision,
                               original_result_hash=original_result_hash, source_sha256=source_sha256)


def asset_refs(document: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Only declared presentation assets may be fetched; source/raw hashes are not download grants."""
    if document is None:
        return {}
    parts = [document]
    if document.get("version") in NESTED_VERSIONS:
        parts.extend(document[key] for key in ("audio", "av"))
    groups = [group for part in parts for group in
              [part.get("overview") or {}, *(packet.get("artifacts") or {} for packet in part.get("packets", []))]]
    for part in parts:
        groups.extend(pair["artifacts"] for record in copy_records(embedded_pair.base_document(part)) for pair in record["pairs"])
        groups.append(embedded_pair.artifacts(part))
        context = part.get("acoustic_context")
        if context is not None:
            if not isinstance(context, dict):
                raise ObjectIntegrityError("The evidence has invalid acoustic context")
            groups.append(context.get("overview") or {})
            groups.extend(event.get("artifacts") or {} for event in context.get("events", []))
        sound = part.get("sound_events")
        if sound is not None:
            if not isinstance(sound, dict) or not isinstance(sound.get("windows"), list):
                raise ObjectIntegrityError("The evidence has invalid sound categories")
            groups.extend(window.get("artifacts") or {} for window in sound["windows"])
        if "short_voice_review" in part:
            review = part["short_voice_review"]
            if part.get("schema") != "chipmunk.audio-evidence/1" or not isinstance(review, dict):
                raise ObjectIntegrityError("The evidence has an invalid short speech-window review")
            groups.append(review.get("artifacts") or {})
        if "long_voice_review" in part:
            from app.evidence.audio_long_review_binding import media_windows

            review = part["long_voice_review"]
            if (part is not document or part.get("schema") != "chipmunk.audio-evidence/1"
                    or not isinstance(review, dict) or not isinstance(review.get("windows"), list)
                    or not isinstance(media_windows(review), list)):
                raise ObjectIntegrityError("The evidence has an invalid long speech-window review")
            groups.extend(window["artifacts"] for window in media_windows(review))
        if "background_reduced_voice_review" in part:
            review = part["background_reduced_voice_review"]
            if (part is not document or part.get("schema") != "chipmunk.audio-evidence/1" or not isinstance(review, dict)):
                raise ObjectIntegrityError("The evidence has an invalid background-reduced speech review")
            groups.append(review["artifacts"])
        if listening := part.get("stereo_listening"):
            groups.extend(window["artifacts"] for window in listening["windows"])
        for key in ("repetition_context", "similarity_context"):
            repetition = part.get(key)
            if repetition is not None:
                audio_part = part.get("schema") == "chipmunk.audio-evidence/1" or part is document.get("audio")
                if not audio_part or not isinstance(repetition, dict) or not isinstance(repetition.get("pairs"), list):
                    raise ObjectIntegrityError("The evidence has invalid repeated audio")
                groups.extend(pair["artifacts"] for pair in repetition["pairs"])
    result: dict[str, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict):
            raise ObjectIntegrityError("The evidence has invalid presentation assets")
        for asset in group.values():
            if not isinstance(asset, dict) or asset.get("media_type") not in ASSET_TYPES.values() or not asset.get("sha256"):
                raise ObjectIntegrityError("The evidence has an unsupported presentation asset")
            previous = result.get(asset["sha256"])
            if previous and previous.get("media_type") != asset["media_type"]:
                raise ObjectIntegrityError("The evidence gives one object contradictory media types")
            result[asset["sha256"]] = asset
    return result
