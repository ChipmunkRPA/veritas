"""Frozen video, audio and document evidence for reports; never query analysis state while rendering."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import re
import wave
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from app.core.canonical import canonical_hash, canonical_json
from app.evidence import document_scope, service
from app.evidence.video_packet_versions import NESTED_VERSIONS as NESTED_VIDEO_VERSIONS
from app.storage.objects import ObjectIntegrityError, ObjectStore, default_store

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.reports.snapshot import ReportSnapshot
    from app.storage.db import Analysis

KINDS = frozenset({"video", "audio", "document"})
_SHA = re.compile(r"[a-f0-9]{64}")


def _bytes(sha256: str, store: ObjectStore) -> bytes:
    if not _SHA.fullmatch(str(sha256)):
        raise ObjectIntegrityError("Evidence has an invalid source checksum")
    store.verify(sha256)
    data = store.read_bytes(sha256)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ObjectIntegrityError("Frozen evidence bytes changed while reading")
    return data


def asset_bytes(asset: dict[str, Any], store: ObjectStore) -> bytes:
    """Verify frozen presentation media and its declared dimensions/format; never fetch its URL."""
    data = _bytes(asset.get("sha256"), store)
    try:
        if asset.get("media_type") == "image/png":
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "PNG" or image.size != (asset.get("width"), asset.get("height")):
                    raise ObjectIntegrityError("Frozen evidence image differs from its declared dimensions or format")
                image.verify()
        elif asset.get("media_type") == "audio/wav":
            with wave.open(io.BytesIO(data), "rb") as audio:
                if (audio.getnframes(), audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (
                        asset.get("samples"), asset.get("sample_rate"), asset.get("channels"), 2, "NONE"):
                    raise ObjectIntegrityError("Frozen evidence audio differs from its declared sample format")
                if len(audio.readframes(audio.getnframes())) != audio.getnframes() * audio.getnchannels() * 2:
                    raise ObjectIntegrityError("Frozen evidence audio is truncated")
        elif asset.get("media_type") == "video/mp4":
            from app.evidence.mp4 import inspect_mp4

            dimensions = inspect_mp4(data)
            if any(canonical_hash(asset.get(key)) != canonical_hash(value) for key, value in dimensions.items()):
                raise ObjectIntegrityError("Frozen video clip differs from its declared frames, timing or track format")
        else:
            raise ObjectIntegrityError("Unsupported frozen evidence asset type")
    except (OSError, SyntaxError, EOFError, wave.Error, ValueError) as exc:
        raise ObjectIntegrityError("Frozen evidence asset cannot be decoded") from exc
    return data


def asset_items(document: dict[str, Any]):
    """Every JSON presentation asset, including overview plots and WAVs that a static PDF cannot play."""
    service.asset_refs(document)
    parts = [(None, document)]
    if document.get("version") in NESTED_VIDEO_VERSIONS:
        parts.extend((key, document[key]) for key in ("audio", "av"))
    for channel, part in parts:
        for role, asset in sorted((part.get("overview") or {}).items()):
            yield None, f"{channel}_{role}" if channel else role, asset
        for packet in part["packets"]:
            for role, asset in sorted(packet.get("artifacts", {}).items()):
                yield packet["id"], role, asset
        for window in (part.get("sound_events") or {}).get("windows", []):
            for role, asset in sorted(window.get("artifacts", {}).items()):
                yield window["id"], role, asset
        for key in ("repetition_context", "similarity_context"):
            for pair in (part.get(key) or {}).get("pairs", []):
                for role, asset in sorted(pair["artifacts"].items()):
                    yield pair["id"], role, asset
        acoustic = part.get("acoustic_context")
        if acoustic:
            for role, asset in sorted(acoustic.get("overview", {}).items()):
                yield None, f"{channel + '_' if channel else ''}acoustic_{role}", asset
            for event in acoustic.get("events", []):
                for role, asset in sorted(event.get("artifacts", {}).items()):
                    yield event["id"], role, asset


def _validate(document: dict, *, kind: str, analysis_id: str, revision: int, result: dict, source_sha256: str) -> None:
    if kind not in KINDS or document.get("schema") != f"chipmunk.{kind}-evidence/1":
        raise ObjectIntegrityError("Evidence modality differs from its report")
    if result.get("input", {}).get("sha256") != source_sha256:
        raise ObjectIntegrityError("Report result and source identity disagree")
    service.validate_packet_set(document, analysis_id=analysis_id, revision=revision,
                                original_result_hash=canonical_hash(result), source_sha256=source_sha256)
    source = document["source"]
    sampling = result.get("manifest", {}).get("sampling", {})
    if kind == "video":
        if (sampling.get("frame_anchor_version") != "stream-pts-rgb24/1"
                or any(source.get(key) != sampling.get(key) or not source.get(key)
                       for key in ("frame_table_sha256", "frame_anchor_sha256"))
                or source.get("timeline_origin_s") != sampling.get("timeline_origin_s")
                or ("picture_hold" in document and source.get("picture_hold_sha256") != sampling.get("picture_hold_sha256"))):
            raise ObjectIntegrityError("Video evidence does not match the published source-frame commitments")
    elif kind == "audio":
        binding = sampling.get("audio_evidence", {}).get("input", {})
        if binding.get("version") != "audio-input/1" or any(source.get(key) != binding.get(key) or source.get(key) is None
                for key in ("pcm_sha256", "wav_sha256", "mapping_sha256", "samples", "sample_rate")):
            raise ObjectIntegrityError("Audio evidence does not match the published sample commitments")
    else:
        binding = sampling.get("document_source", {})
        if binding.get("source_sha256") != source_sha256 or any(source.get(key) != binding.get(key) or not source.get(key)
                for key in ("extracted_text_sha256", "source_map_sha256", "decode_sha256")):
            raise ObjectIntegrityError("Document evidence does not match the published extraction commitments")


def verified_set(snapshot: ReportSnapshot) -> dict[str, Any] | None:
    document = snapshot.evidence
    if document is None:
        return None
    _validate(document, kind=snapshot.analysis.kind, analysis_id=snapshot.analysis.id, revision=snapshot.analysis.revision,
              result=snapshot.result, source_sha256=snapshot.media.sha256)
    return document


def _verify_video(document: dict, store: ObjectStore, assets: dict[str, bytes], *, result: dict | None = None) -> None:
    from app.detectors.video.decode import FrameTable
    from app.evidence import (
        face_boundary_binding,
        face_lighting_context,
        face_return_binding,
        picture_hold_binding,
        whole_face_binding,
    )
    from app.evidence.face_interval_binding import validate_packet as validate_face_interval
    from app.evidence.image_packets import _bounds
    from app.evidence.mp4 import validate_clip_binding

    records = {r["stage"]: json.loads(_bytes(r["sha256"], store)) for r in document.get("source_records", [])}
    table_doc = records.get("probe", {}).get("frame_table")
    anchors = records.get("render", {}).get("frame_anchors")
    source = document["source"]
    if canonical_hash(table_doc) != source["frame_table_sha256"] or canonical_hash(anchors) != source["frame_anchor_sha256"]:
        raise ObjectIntegrityError("Frozen video records no longer match their frame commitments")
    table = FrameTable.from_json(table_doc)
    if (table.width, table.height, len(table), table.time_base) != (source["width"], source["height"], source["frame_count"], source["time_base"]):
        raise ObjectIntegrityError("Frozen video geometry differs from its source frame table")
    by_index = {anchor["frame_index"]: anchor for anchor in anchors}
    if len(by_index) != len(anchors):
        raise ObjectIntegrityError("Frozen video frames have ambiguous identities")
    try:
        picture_hold_binding.validate_record(document, records["render"].get("picture_hold"), table_pts=table.pts,
                                             interpolated=table.interpolated, canonical_hash=canonical_hash)
    except (ValueError, KeyError, TypeError) as error:
        raise ObjectIntegrityError("Picture hold differs from the recorded native inspection") from error
    for packet in document["packets"]:
        validate_clip_binding(packet, source)
        validate_face_interval(packet, source, table_pts=table.pts, interpolated=table.interpolated)
        face_boundary_binding.validate_packet(packet, source, document["version"], table_pts=table.pts,
                                             interpolated=table.interpolated)
        face_return_binding.validate_packet(packet, source, table_pts=table.pts, interpolated=table.interpolated,
                                            rotation=table.rotation, truncated=table.truncated)
        if whole := packet.get("whole_face_review"):
            from app.api.public_projection import alias_signal

            whole_face_binding.validate_packet(packet, source, pts=table.pts)
            if any(i not in by_index or i in table.interpolated for i in packet["window"]["frame_indices"]):
                raise ObjectIntegrityError("Paired face samples lack native source anchors")
            frozen = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
            if not any((signal.get("details") or {}).get("whole_face_review") == whole
                       and signal.get("effective_weight") == 0.0 and alias_signal(signal["id"]) == packet["signal_refs"][0]["id"]
                       for signal in frozen.get("signals", [])):
                raise ObjectIntegrityError("Paired face evidence differs from the frozen analysis measurement")
        if sequence := packet.get("face_sequence"):
            for item in sequence["frame_inputs"]:
                committed = by_index.get(item["frame_index"])
                if committed is None or any(item[key] != committed[key] for key in ("pts", "decoded_rgb_sha256")):
                    raise ObjectIntegrityError("Face sequence input differs from its analysed frame commitment")
            frozen = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
            if not any((s.get("details") or {}).get("face_sequence") == sequence for s in frozen.get("signals", [])):
                raise ObjectIntegrityError("Face sequence differs from the frozen analysis measurement")
        if boundary := packet.get("face_boundary"):
            from app.api.public_projection import alias_signal

            for item in boundary["frame_inputs"]:
                committed = by_index.get(item["frame_index"])
                if committed is None or any(item[key] != committed[key] for key in ("pts", "decoded_rgb_sha256")):
                    raise ObjectIntegrityError("Refined face timing differs from committed decoded source frames")
            if not any((signal.get("details") or {}).get("face_sequence") == packet["face_sequence"]
                       and (signal.get("details") or {}).get("face_boundary") == boundary
                       and signal.get("effective_weight") == 0.0 and alias_signal(signal["id"]) == packet["signal_refs"][0]["id"]
                       for signal in frozen.get("signals", [])):
                raise ObjectIntegrityError("Refined face timing differs from the frozen review check")
        if face_return := packet.get("face_return"):
            from app.api.public_projection import alias_signal

            for item in (*face_return["frame_inputs"], *face_return["scene_frame_inputs"]):
                committed = by_index.get(item["frame_index"])
                if committed is None or any(item[key] != committed[key] for key in ("pts", "decoded_rgb_sha256")):
                    raise ObjectIntegrityError("Short face-change inputs differ from their analyzed frames")
            frozen = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
            commitment = canonical_hash(face_return)
            signal = next((s for s in frozen.get("signals", []) if s.get("effective_weight") == 0.0
                           and alias_signal(s["id"]) == packet["signal_refs"][0]["id"]
                           and canonical_hash((s.get("details") or {}).get("face_return")) == commitment), None)
            if signal is None:
                raise ObjectIntegrityError("Short face-change evidence differs from the frozen review check")
            face_return_binding.validate_distinct(face_return, signal["details"].get("face_sequence"))
        clip = packet.get("clip") or {}
        if clip.get("status") == "ready":
            start, end = clip["start_frame_index"], clip["end_frame_index_exclusive"]
            if clip["source_pts"] != list(table.pts[start:end]):
                raise ObjectIntegrityError("Video context timestamps differ from the committed source frame table")
            if set(table.interpolated).intersection(range(start, end)):
                raise ObjectIntegrityError("Video context contains an inferred source timestamp")
            if end < len(table) and clip["last_frame_duration"]["ticks"] != (table.pts[end] - table.pts[end - 1]) * table.tb.numerator:
                raise ObjectIntegrityError("Video context final duration differs from the next source frame")
        brightness_inputs = {}
        coarse_frames = len(packet["frames"])
        for position, frame in enumerate((*packet["frames"], *packet.get("face_boundary_frames", ()))):
            refined = position >= coarse_frames
            anchor = frame["anchor"]
            recorded = by_index.get(anchor["frame_index"])
            if (table.pts[anchor["frame_index"]] != anchor["pts"]
                    or ("picture_hold" not in packet and (not recorded or any(anchor.get(key) != value for key, value in recorded.items())))):
                raise ObjectIntegrityError("Displayed video frame differs from its recorded inference input")
            artifact = packet["artifacts"][frame["artifact_refs"]["original"]]
            with Image.open(io.BytesIO(assets[artifact["sha256"]])) as image:
                rgb = image.convert("RGB")
                if hashlib.sha256(rgb.tobytes()).hexdigest() != anchor["decoded_rgb_sha256"]:
                    raise ObjectIntegrityError("Displayed video pixels differ from the analysed frame")
                if not refined and packet.get("face_lighting"):
                    brightness_inputs[anchor["frame_index"]] = np.asarray(rgb)
                if sequence := packet.get("face_boundary" if refined else "face_sequence") or packet.get("face_return"):
                    import cv2

                    item = next(row for row in sequence["frame_inputs"] if row["frame_index"] == anchor["frame_index"])
                    matrix = np.array([[float.fromhex(value) for value in row] for row in item["alignment_matrix_hex"]])
                    side = item["alignment_side"]
                    aligned = cv2.resize(cv2.warpAffine(np.asarray(rgb), matrix, (side, side), flags=cv2.INTER_LINEAR),
                                         (112, 112), interpolation=cv2.INTER_AREA)
                    if hashlib.sha256(aligned.tobytes()).hexdigest() != item["aligned_rgb_sha256"]:
                        raise ObjectIntegrityError("Recorded face features refer to a different aligned source crop")
                if "crop" in frame["artifact_refs"]:
                    crop_asset = packet["artifacts"][frame["artifact_refs"]["crop"]]
                    with Image.open(io.BytesIO(assets[crop_asset["sha256"]])) as crop:
                        expected = rgb.crop(_bounds(frame["context_region"], rgb.width, rgb.height))
                        if crop.size != expected.size or crop.convert("RGB").tobytes() != expected.tobytes():
                            raise ObjectIntegrityError("Video crop is not from its recorded source region")
        if packet.get("face_lighting") and face_lighting_context.record(packet, source, brightness_inputs) != packet["face_lighting"]:
            raise ObjectIntegrityError("Video brightness differs from the displayed source pixels")
    if document.get("version") in NESTED_VIDEO_VERSIONS:
        from app.evidence import video_audio, video_av

        result = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
        if canonical_hash(result) != document["original_result_hash"]:
            raise ObjectIntegrityError("Video child evidence uses another frozen result")
        video_audio.verify_source_content(document["audio"], result=result, store=store, source_sha256=source["sha256"])
        if document["audio"]["coverage"] is not None:
            _verify_audio(document["audio"], store, assets, result=result)
        video_av.verify_source_assets(document["av"], video_source=source, result=result, store=store,
                                      table=table, anchors=by_index, assets=assets)


def _verify_audio(document: dict, store: ObjectStore, assets: dict[str, bytes], *, result: dict | None = None) -> None:
    from app.detectors.audio.decode import read_wav, source_ranges

    source = document["source"]
    pcm = read_wav(_bytes(source["wav_sha256"], store))
    mapping = json.loads(_bytes(source["mapping_sha256"], store))
    if (hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest() != source["pcm_sha256"]
            or mapping.get("source_sha256") != source["sha256"] or len(pcm) != source["samples"]
            or any(mapping.get(key) != source.get(key) for key in ("pcm_sha256", "samples", "sample_rate", "channels", "timeline_time_base", "timeline"))):
        raise ObjectIntegrityError("Frozen audio no longer matches its analysed samples or source clock")
    source_ranges(mapping, 0, len(pcm))
    stereo_context = document.get("stereo_context")
    if stereo_sha := document.get("stereo_context_sha256"):
        if canonical_json(stereo_context) != _bytes(stereo_sha, store):
            raise ObjectIntegrityError("Stereo context differs from the frozen video soundtrack measurements")
    elif stereo_context != mapping.get("stereo_context"):
        raise ObjectIntegrityError("Stereo context differs from the frozen audio input mapping")
    if stereo_context is not None:
        from app.detectors.audio.stereo import verify_mono_levels

        try:
            verify_mono_levels(stereo_context, pcm)
        except ValueError as error:
            raise ObjectIntegrityError(str(error)) from error
    for packet in document["packets"]:
        for role, span in (("selected_clip", packet["interval"]), ("context_clip", packet["context_interval"])):
            clip = read_wav(assets[packet["artifacts"][role]["sha256"]])
            if not np.array_equal(clip, pcm[span["start_sample"]:span["end_sample"]]):
                raise ObjectIntegrityError("Frozen audio clip is not the recorded sample interval")
        anchor = packet["anchor"]
        if (anchor.get("original_sha256") != source["sha256"] or anchor.get("pcm_sha256") != source["pcm_sha256"]
                or anchor.get("source_timestamp_ranges") != source_ranges(mapping, packet["interval"]["start_sample"], packet["interval"]["end_sample"])):
            raise ObjectIntegrityError("Audio evidence source timestamps differ from their recorded mapping")
    if document.get("acoustic_context") is not None:
        from app.evidence import audio_background, audio_packets

        binding = {key: source[key] for key in ("sha256", "pcm_sha256", "samples", "sample_rate", "coordinate_space")}
        try:
            audio_background.verify_source_content(document["acoustic_context"], pcm=pcm, mapping=mapping,
                source=binding, voice_packets=document["packets"], read_asset=lambda sha: assets[sha], waveform=audio_packets._chart)
        except (ValueError, KeyError) as error:
            raise ObjectIntegrityError("Frozen acoustic context differs from its measured source input") from error
    if document.get("sound_events") is not None:
        from app.evidence import audio_packets, sound_events

        result = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
        sampling = result.get("manifest", {}).get("sampling", {})
        record = sampling.get("video_audio_evidence") or sampling.get("audio_evidence") or {}
        binding = record.get("input") or {}
        try:
            sound_events.verify(document["sound_events"], store=store, record=record, binding=binding,
                                source=source, pcm=pcm, mapping=mapping, waveform=audio_packets._chart)
        except (ValueError, KeyError) as error:
            raise ObjectIntegrityError("Frozen sound-category evidence differs from its recorded inputs") from error
    if (repetition := document.get("repetition_context")) is not None:
        from app.evidence.audio_repetition import verify

        try:
            verify(repetition, pcm=pcm, mapping=mapping, source=source, read_asset=lambda sha: assets[sha])
        except (ValueError, KeyError) as error:
            raise ObjectIntegrityError("Frozen repeated audio differs from its recorded samples") from error
    if (similarity := document.get("similarity_context")) is not None:
        from app.evidence.audio_similarity import verify

        try:
            verify(similarity, pcm=pcm, mapping=mapping, source=source, read_asset=lambda sha: assets[sha],
                   repetition=document.get("repetition_context"))
        except (ValueError, KeyError, TypeError) as error:
            raise ObjectIntegrityError("Frozen audio similarity differs from its recorded samples") from error


def _verify_document(document: dict, store: ObjectStore, assets: dict[str, bytes], *, result: dict | None = None) -> None:
    from app.detectors.text.extract import text_hash
    from app.evidence import document_matched_context, document_paragraph_priority, document_passage_coverage
    from app.evidence.document_packets import (
        _inspection_anchors,
        _pdf_plates,
        _source_anchors,
        verify_context_window,
    )
    from app.evidence.document_repeat import verify_source as verify_repeated_passages

    source = document["source"]
    decoded = json.loads(_bytes(source["decode_sha256"], store))
    text, mapping = decoded["text"], decoded["source_map"]
    if (text_hash(text) != source["extracted_text_sha256"] or canonical_hash(mapping) != source["source_map_sha256"]
            or decoded.get("meta", {}).get("original_sha256") != source["sha256"] or len(text) != source["chars"]):
        raise ObjectIntegrityError("Frozen text no longer matches its extraction source")
    coverage = document.get("coverage") or {}
    partial_pages = coverage.get("partially_checked_pages", [])
    if canonical_hash(partial_pages) != canonical_hash(decoded.get("meta", {}).get("partially_checked_pages", [])):
        raise ObjectIntegrityError("Frozen partial-page coverage differs from the recorded extraction")
    email = coverage.get("email")
    if isinstance(email, dict) and canonical_hash(email) != canonical_hash(decoded.get("meta", {}).get("email")):
        raise ObjectIntegrityError("Frozen email coverage differs from the recorded extraction")
    try:
        verify_repeated_passages(document, text, mapping)
        if document["version"] == document_scope.MATCHED_VERSION:
            frozen = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
            document_matched_context.verify_source(document, text, mapping, result=frozen)
        if document["version"] == document_scope.PARAGRAPH_PRIORITY_VERSION:
            frozen = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
            document_paragraph_priority.verify_source(document, text, mapping, result=frozen)
        if document_passage_coverage.KEY in coverage:
            frozen = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
            document_passage_coverage.verify_source(document, text, mapping, result=frozen)
        if document_scope.CHECKPOINT_AGREEMENT_KEY in coverage:
            frozen = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
            document_scope.validate_checkpoint_agreement(document, text, result=frozen)
    except (ValueError, KeyError, TypeError) as error:
        raise ObjectIntegrityError("Document review evidence differs from the recorded source extraction") from error
    for packet in document["packets"]:
        for key in ("focus", "context"):
            part = packet[key]
            if part["text"] != text[part["start"]:part["end"]]:
                raise ObjectIntegrityError("A report excerpt differs from its recorded source range")
        focus = packet["focus"]
        if document.get("version") in document_scope.SCOPE_VERSIONS:
            try:
                document_scope.validate(packet, len(text))
            except (ValueError, KeyError, TypeError) as error:
                raise ObjectIntegrityError("Document inspection scope differs from the recorded inputs") from error
            anchors = _inspection_anchors(mapping, packet["inspection_scope"], len(text))
        else:
            anchors = _source_anchors(mapping, focus["start"], focus["end"], len(text))
        if packet["anchors"] != anchors:
            raise ObjectIntegrityError("A document source anchor differs from its recorded extraction mapping")
        for window in packet["supporting_windows"]:
            verify_context_window(window, text)
        if document.get("version") in document_scope.CONTEXT_VERSIONS or "context_comparison" in packet:
            from app.evidence.document_context import build as compare_contexts
            from app.evidence.document_context import validate as validate_comparison

            frozen = result if result is not None else json.loads(_bytes(document["original_result_hash"], store))
            validate_comparison(packet["context_comparison"], focus=focus, chars=len(text), text=text)
            if canonical_hash(packet["context_comparison"]) != canonical_hash(compare_contexts(frozen, text, focus)):
                raise ObjectIntegrityError("Context comparison differs from the actual published classifier inputs")
        if not any(anchor.get("extraction") == "ocr" for anchor in packet["anchors"]):
            continue
        # Historical frozen sets did not record integer crop bounds. Preserve
        # their existing hash/anchor verification without inventing coordinates
        # or rewriting their frozen record from rounded scale metadata.
        if not any("source_crop_pixels" in asset for asset in packet["artifacts"].values()):
            continue
        # Recreate presentation pixels solely from the frozen OCR input and the
        # committed source mapping; this never reruns OCR or a text classifier.
        expected_plates = _pdf_plates(_bytes(source["sha256"], store), packet["anchors"], store)
        for key, (expected, metadata) in expected_plates.items():
            artifact = packet["artifacts"].get(key)
            if artifact is None:
                raise ObjectIntegrityError("An OCR source presentation is missing from the report")
            for field in ("role", "page", "source_crop_pixels", "source_raster_dimensions", "source_crop_coordinate_space",
                          "source_raster_sha256", "source_bbox", "input_to_page"):
                if field in metadata and canonical_hash(artifact.get(field)) != canonical_hash(metadata[field]):
                    raise ObjectIntegrityError("OCR crop metadata differs from its recorded source mapping")
            content = assets[artifact["sha256"]]
            if metadata.get("_exact_png") is not None and content != metadata["_exact_png"]:
                raise ObjectIntegrityError("The displayed OCR input differs from the frozen recognition raster")
            with Image.open(io.BytesIO(content)) as displayed:
                if displayed.size != expected.size or displayed.convert("RGB").tobytes() != expected.convert("RGB").tobytes():
                    raise ObjectIntegrityError("OCR presentation pixels differ from the frozen source raster")


def verify_assets(document: dict[str, Any], store: ObjectStore, *, result: dict | None = None) -> dict[str, bytes]:
    """CAS-only verification for capture/render/package assembly; no extraction, model inference, or database lookup."""
    _bytes(document["source"]["sha256"], store)
    assets = {}
    for _, _, asset in asset_items(document):
        assets[asset["sha256"]] = asset_bytes(asset, store)
    kind = document["schema"].removeprefix("chipmunk.").split("-evidence/")[0]
    if kind == "video":
        _verify_video(document, store, assets, result=result)
    elif kind == "audio":
        _verify_audio(document, store, assets, result=result)
    elif kind == "document":
        _verify_document(document, store, assets, result=result)
    else:
        raise ObjectIntegrityError("Unsupported frozen evidence modality")
    return assets


def capture(db: Session, analysis: Analysis, *, result: dict, store: ObjectStore | None = None) -> dict[str, Any] | None:
    if analysis.kind not in KINDS:
        return None
    store = store or default_store()
    document = service.load_packet_set(db, analysis, store)
    if document is None:
        return None
    _validate(document, kind=analysis.kind, analysis_id=analysis.id, revision=analysis.revision,
              result=result, source_sha256=analysis.sha256)
    verify_assets(document, store, result=result)
    return copy.deepcopy(document)


def manifest(snapshot: ReportSnapshot) -> dict[str, Any] | None:
    document = verified_set(snapshot)
    if document is None:
        return None
    return {"schema": document["schema"], "id": document["id"], "version": document["version"], "sha256": document["sha256"],
            "source_sha256": document["source"]["sha256"], "original_result_hash": document["original_result_hash"],
            "artifacts": [{"packet_id": packet_id, "role": role, "sha256": asset["sha256"], "media_type": asset["media_type"]}
                          for packet_id, role, asset in asset_items(document)]}


def quoted_values(snapshot: ReportSnapshot) -> list[str]:
    """Verbatim submitted text is exempt from generated-prose vocabulary checks."""
    if snapshot.analysis.kind != "document" or snapshot.evidence is None:
        return []
    repeats = (snapshot.evidence.get("repeated_passages") or {}).get("groups", [])
    isolated = [input["text"] for packet in snapshot.evidence["packets"]
                if (review := (packet.get("context_comparison") or {}).get("isolated_paragraph"))
                for input in (review["context"], review["paragraph"])]
    return [group["text"] for group in repeats] + isolated + [text for packet in snapshot.evidence["packets"] for text in
            [packet["focus"]["text"], packet["context"]["text"], *(w["text"] for w in packet["supporting_windows"]),
             *(w["text"] for check in (packet.get("context_comparison") or {}).get("checks", []) for w in check["inputs"])]]


def _coverage(kind: str, document: dict) -> str:
    coverage, source = document.get("coverage") or {}, document["source"]
    if kind == "video":
        label = "visual frame or window checks" if document.get("version") in NESTED_VIDEO_VERSIONS else "frame or window checks"
        return (f"{coverage.get('scored_frame_count', 0)} of {source['frame_count']} source frames were used by recorded {label}; "
                "unscored frames and gaps have no negative finding.")
    if kind == "audio":
        rate = source["sample_rate"]
        return (f"{coverage.get('scored_samples', 0)} uniquely scored samples of {source['samples']} decoded samples at {rate} Hz; "
                f"{coverage.get('unscored_samples', 0)} unscored samples. "
                + ("The decoding limit was reached; later audio was not checked." if coverage.get("decode_limit_reached") else ""))
    summary = f"{coverage.get('scored_windows', 0)} scored contexts; {coverage.get('candidate_spans', 0)} candidate passages."
    if not coverage.get("email"):
        summary += f" {len(coverage.get('unmapped_pages') or [])} source pages have no mapped text."
    partial_count = len(coverage.get("partially_checked_pages") or [])
    if partial_count:
        page_count = "1 source page has" if partial_count == 1 else f"{partial_count} source pages have"
        summary += f" {page_count} mapped text and unchecked image areas."
    return summary


def _document_coverage_details(coverage: dict) -> list[str]:
    notes = [
        f"{check['signal_ref'].get('display_id') or 'Recorded check'}: {check['scored_windows']} local context(s) retained. "
        + check.get('limitation', 'Some input contexts were omitted by the bounded evidence capture; missing context is not a negative finding.')
        for check in coverage.get("checks", []) if check["status"] != "recorded"
    ]
    if review := coverage.get("matched_context_review"):
        notes.append(f"Short-passage review selected {review['selected_sentences']} of {review['eligible_sentences']} "
                     "eligible sentences. This evidence has zero weight in the overall assessment.")
    for page in coverage.get("partially_checked_pages", []):
        boxes = ", ".join(f"[{', '.join(map(str, box))}]" for box in page["image_boxes"])
        notes.append(f"Page {page['page']}. {page['reason']} Unchecked bounds in unrotated PDF points {boxes}.")
    email = coverage.get("email")
    if isinstance(email, dict):
        body_format = {"text/plain": "plain-text", "text/html": "HTML"}.get(email.get("body_content_type"))
        if body_format and email.get("body_part"):
            note = f"The extracted {body_format} body is MIME part {email['body_part']}."
            count = len(email.get("attachments") or [])
            if count:
                phrase = "attachment was" if count == 1 else "attachments were"
                note += f" {count} {phrase} outside the analysis."
            notes.append(note)
    return notes


def _passage_coverage_view(coverage: dict) -> dict | None:
    from app.evidence.document_passage_coverage import KEY, LABELS, STATES

    record = coverage.get(KEY)
    if record is None:
        return None
    rows = record["ranges"][:12]
    return {"mapped_characters": record["mapped_characters"],
            "counts": [(LABELS[state], record["counts"][state]) for state in reversed(STATES)],
            "ranges": [{**row, "label": LABELS[row["state"]]} for row in rows],
            "additional_ranges": len(record["ranges"]) - len(rows),
            "omitted_ranges": record["omitted_ranges"]}


def _writing_agreement_view(coverage: dict) -> dict | None:
    record = coverage.get(document_scope.CHECKPOINT_AGREEMENT_KEY)
    if record is None:
        return None
    return {"ranges": [{**row, "label": document_scope.CHECKPOINT_AGREEMENT_LABELS[row["state"]]} for row in record["ranges"]]}


def _voice_interval_view(interval: dict, rate: int, video_origin_s: float | None) -> dict:
    ranges = []
    for recorded in interval["source_timestamp_ranges"][:8]:
        start, end = recorded["source_start_pts"] / rate, recorded["source_end_pts"] / rate
        ranges.append({**recorded,
                       "playback": (start - video_origin_s, end - video_origin_s)
                       if video_origin_s is not None and start >= video_origin_s else None})
    return {"start_s": interval["start_sample"] / rate, "end_s": interval["end_sample"] / rate,
            "ranges": ranges, "omitted_ranges": len(interval["source_timestamp_ranges"]) - len(ranges)}


def view(snapshot: ReportSnapshot, *, store: ObjectStore | None = None) -> dict[str, Any] | None:
    if snapshot.analysis.kind not in KINDS:
        if snapshot.evidence is not None:
            verified_set(snapshot)  # do not silently ignore an incorrectly attached set
        return None
    kind = snapshot.analysis.kind
    document = verified_set(snapshot)
    if document is None:
        return {"available": False, "kind": kind, "packets": [], "title": f"{kind.capitalize()} evidence",
                "note": "No source-bound evidence set was captured in this report version. An overall classification does not identify altered parts."}
    store = store or default_store()
    verified = verify_assets(document, store, result=snapshot.result)
    withheld = snapshot.report.profile == "newsroom" or snapshot.report.redacted

    def plate(role: str, asset: dict, label: str | None = None) -> dict:
        return {"role": role, "label": label or role.replace("_", " ").capitalize(), "sha256": asset["sha256"],
                "uri": None if withheld else "data:image/png;base64," + base64.b64encode(verified[asset["sha256"]]).decode("ascii")}

    def overview_label(pk: str, body: dict, role: str) -> str:
        source = body["source"]
        if source.get("sample_rate") and source.get("samples"):
            location = f"PCM 0.000-{source['samples'] / source['sample_rate']:.3f} s"
        else:
            location = "recorded source coverage"
        return f"{pk.capitalize()} overview - {location} - {role.replace('_', ' ')}"

    packets = []
    parts = [(kind, document)]
    if document.get("version") in NESTED_VIDEO_VERSIONS:
        parts.extend((key, document[key]) for key in ("audio", "av"))
    entries = []
    acoustic_contexts = []
    for pk, body in parts:
        prefix = f"/{pk}" if kind == "video" and pk != "video" else ""
        entries.extend((pk, body, packet, f"{prefix}/packets/{index}")
                       for index, packet in enumerate(body["packets"]))
        entries.extend(("sound", body, window, f"{prefix}/sound_events/windows/{index}")
                       for index, window in enumerate((body.get("sound_events") or {}).get("windows", [])))
        acoustic = body.get("acoustic_context")
        if acoustic:
            entries.extend(("acoustic", body, event, f"{prefix}/acoustic_context/events/{index}")
                           for index, event in enumerate(acoustic.get("events", [])))
            acoustic_contexts.append({"json_pointer": f"{prefix}/acoustic_context",
                "status": acoustic.get("status"), "method": copy.deepcopy(acoustic.get("method", {})),
                "source": copy.deepcopy(acoustic.get("source", {})), "bin_count": len(acoustic.get("bins", [])),
                "event_count": len(acoustic.get("events", [])), "limitations": copy.deepcopy(acoustic.get("limitations", [])),
                "images": [plate(role, asset, overview_label("acoustic context", acoustic, role))
                           for role, asset in sorted(acoustic.get("overview", {}).items())]})
    for packet_kind, part, packet, json_pointer in entries:
        row = {key: copy.deepcopy(packet.get(key)) for key in
               ("id", "title", "observation", "interpretation", "limitations", "finding_ids", "signal_refs")}
        row.update(kind=packet_kind, measurements=copy.deepcopy(packet.get("measurements", packet.get("observations", []))),
                   images=[], frames=[], excerpts=[], source_locations=[], windows=[], audio=None,
                   json_pointer=json_pointer,
                   assets=[{"role": role, "sha256": asset["sha256"], "media_type": asset["media_type"]}
                           for role, asset in sorted(packet.get("artifacts", {}).items())])
        if packet_kind in {"video", "av"}:
            hold = packet.get("picture_hold")
            row["picture_hold"] = copy.deepcopy(hold)
            if sheet := packet["artifacts"].get("contact_sheet"):
                row["images"] = [plate("contact_sheet", sheet, "Endpoints of the verified hold" if hold else "Sampled frames (not continuous playback)")]
            row["window"] = copy.deepcopy(packet.get("window"))
            row["face_interval"] = copy.deepcopy(packet.get("face_interval"))
            row["face_sequence"] = copy.deepcopy(packet.get("face_sequence"))
            row["face_boundary"] = copy.deepcopy(packet.get("face_boundary"))
            if packet.get("face_return"):
                row["face_return"] = copy.deepcopy(packet["face_return"])
            row["whole_face_review"] = copy.deepcopy(packet.get("whole_face_review"))
            row["clip"] = copy.deepcopy(packet.get("clip"))
            coarse_count = len(packet["frames"])
            for position, frame in enumerate((*packet["frames"], *packet.get("face_boundary_frames", ()))):
                row["frames"].append({"id": frame["id"], "anchor": copy.deepcopy(frame["anchor"]), "score": frame["score"],
                                      "region_meaning": frame["region_meaning"], "context_region": frame.get("context_region"),
                                      "interval_role": frame.get("interval_role"), "refined": position >= coarse_count,
                                      "first_refined": position == coarse_count,
                                      "images": [plate(role, packet["artifacts"][frame["artifact_refs"][role]], {"original": "Exact decoded source frame" if hold else "Exact analysed source frame", "context": "Frame context", "crop": "Original mouth-input crop" if packet_kind == "av" else "Original face-context crop", "measured_patch": "Exact measured gray mouth patch"}[role])
                                                 for role in ("original", "context", "crop", "measured_patch")
                                                 if role in frame["artifact_refs"] and (not hold or role == "original")]})
        elif packet_kind == "audio":
            row["images"] = [plate(role, packet["artifacts"][role]) for role in ("waveform", "scores")]
            row["audio"] = {"interval": copy.deepcopy(packet["interval"]), "context_interval": copy.deepcopy(packet["context_interval"]),
                             "start_s": packet["start_s"], "end_s": packet["end_s"], "sample_rate": part["source"]["sample_rate"],
                             "source_ranges": copy.deepcopy(packet["anchor"]["source_timestamp_ranges"])}
        elif packet_kind == "sound":
            a, b = packet["start_sample"], packet["end_sample"]
            interval = {"start_sample": a, "end_sample": b}
            row.update(title="Recorded sound categories", finding_ids=[], signal_refs=[],
                       observation="; ".join(f"{item['label']}: {item['score']:.3f}" for item in packet["labels"]),
                       interpretation="These labels describe possible sounds in the mixed track. They do not separate the sounds or determine whether the recording is AI-generated.",
                       limitations=copy.deepcopy(part["sound_events"]["limitations"]), measurements=[],
                       sound_labels=copy.deepcopy(packet["labels"]))
            row["images"] = [plate("waveform", packet["artifacts"]["waveform"])]
            row["audio"] = {"interval": interval, "context_interval": interval,
                            "start_s": packet["start_s"], "end_s": packet["end_s"], "sample_rate": part["source"]["sample_rate"],
                            "source_ranges": copy.deepcopy(packet["source_timestamp_ranges"])}
        elif packet_kind == "acoustic":
            row.update(title=packet.get("title") or packet["category"].replace("_", " ").capitalize(),
                       finding_ids=[], signal_refs=[], related_voice_packet_ids=packet.get("related_voice_packet_ids", []))
            row["images"] = [plate(role, packet["artifacts"][role]) for role in ("waveform", "spectrogram")]
            row["audio"] = {"interval": copy.deepcopy(packet["interval"]), "context_interval": copy.deepcopy(packet["context_interval"]),
                            "start_s": packet["start_s"], "end_s": packet["end_s"], "sample_rate": part["source"]["sample_rate"],
                            "source_ranges": copy.deepcopy(packet["source_timestamp_ranges"])}
        else:
            from app.evidence import document_paragraph_priority

            row["images"] = [plate(role, asset, f"Page {asset.get('page', '?')} - {asset.get('role', role)}")
                             for role, asset in sorted(packet["artifacts"].items())]
            focus, context = packet["focus"], packet["context"]
            matched_review = bool(packet.get("matched_context_review"))
            if matched_review:
                row["matched_context_review"] = True
            priority = part.get("coverage", {}).get(document_paragraph_priority.KEY)
            if priority and priority["packet_id"] == packet["id"]:
                row["paragraph_priority_note"] = document_paragraph_priority.NOTE
            if packet.get("context_comparison") is not None:
                row["context_comparison"] = copy.deepcopy(packet["context_comparison"])
                if withheld:
                    for check in row["context_comparison"]["checks"]:
                        for window in check["inputs"]:
                            window["text"] = None
                    if isolated := row["context_comparison"].get("isolated_paragraph"):
                        isolated["context"]["text"] = isolated["paragraph"]["text"] = None
            row["focus_range"] = {"start": focus["start"], "end": focus["end"]}
            scope = packet.get("inspection_scope") if part.get("version") in document_scope.SCOPE_VERSIONS else None
            if scope:
                row["inspection_scope"] = copy.deepcopy(scope)
            if not withheld:
                row["context"] = {"start": context["start"], "end": context["end"],
                                  "before": context["text"][:focus["start"] - context["start"]], "focus": focus["text"],
                                  "after": context["text"][focus["end"] - context["start"]:]}
                if scope:
                    segments, cursor = [], context["start"]
                    for span in scope["spans"]:
                        segments.append({"text": context["text"][cursor - context["start"]:span["start"] - context["start"]], "highlight": False})
                        segment = {"text": context["text"][span["start"] - context["start"]:span["end"] - context["start"]], "highlight": True}
                        if matched_review and span["start"] == focus["start"] and span["end"] == focus["end"]:
                            segment["review_focus"] = True
                        segments.append(segment)
                        cursor = span["end"]
                    segments.append({"text": context["text"][cursor - context["start"]:], "highlight": False})
                    row["context"]["segments"] = segments
                row["source_locations"] = copy.deepcopy(packet["anchors"])
            for window in packet["supporting_windows"]:
                context_window = {"score": window["score"], "input_sha256": window["input_sha256"],
                                       "spans": copy.deepcopy(window["spans"]), "signal_ref": copy.deepcopy(window["signal_ref"]),
                                       "token_coverage": copy.deepcopy(window.get("token_coverage", {})),
                                       "text": None if withheld else window["text"]}
                context_window.update(score_kind=window.get("score_kind", "classifier_probability"),
                                      threshold=window.get("threshold", 0.5), ai_when=window.get("ai_when", "gte"),
                                      input_semantics=window.get("input_semantics"),
                                      scored_spans=copy.deepcopy(window.get("scored_spans", [])),
                                      tokens_scored=window.get("tokens_scored"), neighbor_expansion=window.get("neighbor_expansion"))
                supports = (window["direction"] == "supports_ai" if window.get("direction") else
                            window["score"] < context_window["threshold"] if context_window["ai_when"] == "lt"
                            else window["score"] >= context_window["threshold"])
                context_window["supports_ai"] = supports
                if "matched_context_selected" in window:
                    context_window["matched_context_selected"] = window["matched_context_selected"]
                    context_window["matched_context_eligible"] = "score_exact_hex" in window
                context_window["contexts"] = [{key: copy.deepcopy(context[key]) for key in
                    ("id", "source_token_range", "scored_token_range", "token_ids_sha256", "input_sha256", "anchor_token_count")}
                    for context in window.get("inference_contexts", [])]
                row["windows"].append(context_window)
        if packet_kind == "av":
            row["images"].extend(plate(role, packet["artifacts"][role]) for role in ("lag_profile", "alignment", "waveform"))
            row["measurement"] = copy.deepcopy(packet["measurement"])
            row["audio"] = {"interval": copy.deepcopy(packet["interval"]), "context_interval": copy.deepcopy(packet["context_interval"]),
                            "sample_rate": part["source"]["sample_rate"],
                            "source_ranges": copy.deepcopy(packet["anchor"]["source_timestamp_ranges"])}
            row["measured_inputs"] = [{"tick": item["tick"], "frame_index": item["frame_index"], "frame_pts": item["frame_pts"],
                                       "image": plate(item["artifact_ref"], packet["artifacts"][item["artifact_ref"]],
                                                      f"{packet['id']} - tick {item['tick']}; frame {item['frame_index']}")}
                                      for item in packet["window"]["frame_inputs"]]
        if packet_kind in {"audio", "acoustic", "av", "sound"}:
            interval = row["audio"]["interval"]
            rate = row["audio"]["sample_rate"]
            row["locator"] = (f"PCM {interval['start_sample'] / rate:.3f}-{interval['end_sample'] / rate:.3f} s; "
                              f"samples [{interval['start_sample']}, {interval['end_sample']})")
        elif packet_kind == "video":
            interval = row["picture_hold"] or row["face_sequence"] or row["face_interval"] or row.get("face_return") or (row["window"] if row["whole_face_review"] else None)
            label = "Identical frames" if row["picture_hold"] else "Sampled face positions" if row["whole_face_review"] else "Face review range"
            row["locator"] = (f"{label} {interval['start_stream_s']:.3f}-{interval['end_stream_s']:.3f} s" if interval else
                              "; ".join(f"frame {f['anchor']['frame_index']} at {f['anchor']['stream_time_s']:.6f} s" for f in packet["frames"]))
        else:
            locations = []
            for anchor in packet["anchors"]:
                label = (f"page {anchor['page']}" if anchor["kind"] == "pdf" else
                         f"paragraph {anchor['paragraph_index'] + 1}" if anchor["kind"] == "docx" else
                         f"MIME {anchor.get('mime_part', 'body')}" if anchor["kind"] in {"email", "markup"} else
                         str(anchor["kind"]))
                if label not in locations:
                    locations.append(label)
            if row.get("matched_context_review"):
                position = f"selected sentence [{packet['focus']['start']}, {packet['focus']['end']})"
            elif scope := row.get("inspection_scope"):
                label = "measured ratio range" if scope["basis"] == "ratio_reduction" else "scored context"
                ranges = ", ".join(f"[{span['start']}, {span['end']})" for span in scope["spans"])
                position = f"{label} characters {ranges}"
            else:
                position = f"characters [{packet['focus']['start']}, {packet['focus']['end']})"
            row["locator"] = position + ("; " + ", ".join(locations) if locations else "")
        # Captions travel with their plate across a page break. Repeating the
        # stable ID and source coordinates makes a detached PDF page traceable
        # to the same JSON packet without relying on the previous page heading.
        for image in row["images"]:
            image["label"] = f"{packet['id']} - {row['locator']} - {image['label']}"
        for frame in row["frames"]:
            anchor = frame["anchor"]
            for image in frame["images"]:
                image["label"] = (f"{frame['id']} - source {anchor['stream_time_s']:.6f} s; "
                                  f"frame {anchor['frame_index']}, PTS {anchor['pts']} at {anchor['time_base']} - {image['label']}")
        packets.append(row)
    stereo_source = (document.get("audio") or {}) if kind == "video" else document
    stereo_context = copy.deepcopy(stereo_source.get("stereo_context"))
    video_origin_s = document["source"].get("timeline_origin_s") if kind == "video" else None
    voice_coverage = (stereo_source.get("coverage") or {}) if kind in {"audio", "video"} else {}
    rate = (stereo_source.get("source") or {}).get("sample_rate")
    gaps = voice_coverage.get("unscored_intervals", [])
    overlaps = voice_coverage.get("stereo_attenuation_overlaps", [])
    voice_gaps = [_voice_interval_view(gap, rate, video_origin_s) for gap in gaps[:8]]
    voice_stereo_overlaps = []
    for overlap in overlaps[:8]:
        window = stereo_context["windows"][overlap["stereo_window_index"]]
        voice_stereo_overlaps.append({**_voice_interval_view(overlap, rate, video_origin_s),
                                      "attenuation_db": window["downmix"]["attenuation_db"]})
    repeated_passages = copy.deepcopy(document.get("repeated_passages"))
    if withheld and repeated_passages:
        for group in repeated_passages["groups"]:
            group.pop("text")
    return {"available": True, "kind": kind, "title": f"{kind.capitalize()} evidence", "id": document["id"],
            "stereo_context": stereo_context,
            "repetition_context": copy.deepcopy(stereo_source.get("repetition_context")),
            "repetition_context_pointer": "/audio/repetition_context" if kind == "video" else "/repetition_context",
            "similarity_context": copy.deepcopy(stereo_source.get("similarity_context")),
            "similarity_context_pointer": "/audio/similarity_context" if kind == "video" else "/similarity_context",
            "repeated_passages": repeated_passages,
            "picture_hold": copy.deepcopy(document.get("picture_hold")),
            "voice_gaps": voice_gaps,
            "voice_gap_count": len(gaps),
            "voice_gap_pointer": "/audio/coverage/unscored_intervals" if kind == "video" else "/coverage/unscored_intervals",
            "audio_source_clock": voice_coverage.get("source_clock_context"),
            "audio_source_clock_rate": rate,
            "audio_source_clock_pointer": ("/audio" if kind == "video" else "") + "/coverage/source_clock_context",
            "voice_stereo_overlaps": voice_stereo_overlaps, "voice_stereo_overlap_count": len(overlaps),
            "voice_stereo_overlap_pointer": ("/audio" if kind == "video" else "") + "/coverage/stereo_attenuation_overlaps",
            "stereo_context_pointer": "/audio/stereo_context" if kind == "video" else "/stereo_context",
            "stereo_context_video_origin_s": video_origin_s,
            "version": document["version"], "sha256": document["sha256"], "source_sha256": document["source"]["sha256"],
            "withheld": withheld, "packets": packets, "acoustic_contexts": acoustic_contexts, "coverage_note": _coverage(kind, document),
            "passage_coverage": _passage_coverage_view(document.get("coverage") or {}) if kind == "document" else None,
            "writing_agreement": _writing_agreement_view(document.get("coverage") or {}) if kind == "document" else None,
            "coverage_details": ([f"{'Voice evidence' if key == 'audio' else 'Audio/video synchronization evidence'}: {channel['reason']}"
                                  for key, channel in (document.get("coverage", {}).get("channels") or {}).items()]
                                 if kind == "video" else
                                 _document_coverage_details(document.get("coverage") or {})
                                 if kind == "document" else []),
            "overview": [{**plate(role, asset, overview_label(pk, part, role)),
                          "anchor": "multimedia-evidence-voice-waveform" if pk == "audio" and role == "waveform" else None}
                         for pk, part in parts for role, asset in sorted((part.get("overview") or {}).items())],
            "limitations": [note for _, part in parts for note in part.get("limitations", [])],
            "omitted": sum(part.get("omitted_candidates", part.get("omitted_windows", (part.get("coverage") or {}).get("omitted_windows", 0))) for _, part in parts),
            "note": "These frozen observations and assets are the same evidence set as the JSON export. Scores are inspection clues and do not establish AI generation or an exact edit boundary."}
