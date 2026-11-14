"""Immutable audio evidence from the exact scored PCM and recorded windows.

No inference happens on a read. A window response is not a word-level or edit-
boundary localisation; clips and descriptive charts share one sample anchor.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import uuid
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.public_projection import alias_signal
from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.audio.coverage import interval_union
from app.detectors.audio.decode import SAMPLE_RATE, read_wav, source_ranges, wav_bytes
from app.evidence.audio_binding import (
    SPEECH_SELECTION,
    VOICE_CLOCK_POLICY,
    attach_stereo_attenuation_overlaps,
    source_clock_context,
    speech_window_selection,
    unscored_voice_intervals,
    validate_audio_stream_scope,
    validate_unscored_voice_coverage,
)
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact
from app.storage.objects import ObjectIntegrityError, ObjectStore

SCHEMA = "chipmunk.audio-evidence/1"
VERSION = "audio-evidence-3"
CREATOR = "audio-evidence-packets/1"
MAX_PACKETS = 6
THRESHOLD = 0.5
_NS = uuid.UUID("8dddbf09-cf9b-4d82-bdb2-aa0243d6289b")
_SHA = re.compile(r"[a-f0-9]{64}")


class PacketIntegrityError(ObjectIntegrityError):
    pass


def _verified(store: ObjectStore, sha: str) -> bytes:
    data = store.read_bytes(sha)
    if hashlib.sha256(data).hexdigest() != sha:
        raise PacketIntegrityError("Audio evidence bytes no longer match their recorded hash")
    return data


def packet_hash(document: dict[str, Any]) -> str:
    return canonical_hash({k: v for k, v in document.items() if k != "sha256"})


def current_set_id(analysis: Analysis) -> str:
    return str(uuid.uuid5(_NS, f"{analysis.id}:{analysis.result_hash}:{VERSION}"))


def _interval(value: Any, samples: int) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise PacketIntegrityError("Missing audio sample interval")
    a, b = value.get("start_sample"), value.get("end_sample")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (a, b)) or not 0 <= a < b <= samples:
        raise PacketIntegrityError("Audio sample interval is outside the analysed signal")
    return a, b


def validate_packet_set(document: dict[str, Any], *, analysis_id: str, revision: int,
                        original_result_hash: str, source_sha256: str) -> None:
    from app.evidence.audio_bandwidth_binding import BANDWIDTH_SCHEMAS
    from app.evidence.audio_silence_binding import CONTEXT_SCHEMA

    if (document.get("schema") != SCHEMA or document.get("version") not in {"audio-evidence-1", "audio-evidence-2", VERSION} or document.get("status") != "ready" or document.get("analysis_id") != analysis_id
            or document.get("revision") != revision or document.get("original_result_hash") != original_result_hash
            or document.get("source", {}).get("sha256") != source_sha256 or document.get("sha256") != packet_hash(document)):
        raise PacketIntegrityError("Audio evidence is not bound to its frozen result and source")
    validate_content(document)
    if document.get("version") in {"audio-evidence-2", VERSION} and not document.get("acoustic_context"):
        raise PacketIntegrityError("Current audio evidence lacks frozen acoustic context")
    if document.get("version") == VERSION and document["acoustic_context"]["schema"] not in {
        "chipmunk.acoustic-context/2", CONTEXT_SCHEMA, *BANDWIDTH_SCHEMAS,
    }:
        raise PacketIntegrityError("Current audio evidence lacks continuity measurements")


def validate_content(document: dict[str, Any]) -> None:
    """Validate audio evidence content also used by a parent video set."""
    source = document["source"]
    samples = source.get("samples")
    if not isinstance(samples, int) or not 0 < samples <= SAMPLE_RATE * 600 or source.get("sample_rate") != SAMPLE_RATE:
        raise PacketIntegrityError("Invalid analysed PCM format")
    ids = set()
    for p in document["packets"]:
        if p["id"] in ids or p.get("evidence_type") != "statistical":
            raise PacketIntegrityError("Duplicate or misrepresented audio evidence")
        ids.add(p["id"])
        a, b = _interval(p["interval"], samples)
        ca, cb = _interval(p["context_interval"], samples)
        if not ca <= a < b <= cb:
            raise PacketIntegrityError("Audio context does not contain its scored interval")
        if set(p["artifacts"]) != {"selected_clip", "context_clip", "waveform", "scores"}:
            raise PacketIntegrityError("Incomplete audio evidence assets")
        for role, asset in p["artifacts"].items():
            expected = "audio/wav" if role.endswith("clip") else "image/png"
            if not _SHA.fullmatch(str(asset.get("sha256"))) or asset.get("media_type") != expected:
                raise PacketIntegrityError("Invalid audio evidence asset")
            if role.endswith("clip") and asset.get("samples") != (b - a if role == "selected_clip" else cb - ca):
                raise PacketIntegrityError("Audio clip length disagrees with its source interval")
        if not p["observations"] or any(not 0 <= x["value"] <= 1 for x in p["observations"]):
            raise PacketIntegrityError("Unusable audio window response")
    for window in document["coverage"]["windows"]:
        _interval(window, samples)
    try:
        validate_audio_stream_scope(source)
        validate_unscored_voice_coverage(document["coverage"], source, document.get("stereo_context"))
    except (KeyError, TypeError, ValueError) as error:
        raise PacketIntegrityError(str(error)) from error
    if document.get("acoustic_context") is not None:
        from app.evidence.audio_background_binding import validate_acoustic_context
        try:
            validate_acoustic_context(document["acoustic_context"], source=source, voice_packets=document["packets"])
        except ValueError as error:
            raise PacketIntegrityError(str(error)) from error
    if document.get("sound_events") is not None:
        from app.evidence.sound_event_binding import validate
        validate(document["sound_events"], source)
    if "short_voice_review" in document:
        from app.evidence.audio_short_review_binding import validate as validate_short_voice_review

        try:
            if document.get("schema") != SCHEMA:
                raise ValueError("Short speech-window review is for standalone audio")
            validate_short_voice_review(document["short_voice_review"], source)
        except ValueError as error:
            raise PacketIntegrityError(str(error)) from error
    if "long_voice_review" in document:
        from app.evidence.audio_long_review_binding import validate as validate_long_voice_review

        try:
            validate_long_voice_review(document["long_voice_review"], source)
        except ValueError as error:
            raise PacketIntegrityError(str(error)) from error
    if "background_reduced_voice_review" in document:
        from app.evidence.audio_background_review_binding import validate as validate_background_review

        try:
            if document.get("schema") != SCHEMA:
                raise ValueError("Background-reduced speech review is for standalone audio")
            validate_background_review(document["background_reduced_voice_review"], source)
        except ValueError as error:
            raise PacketIntegrityError(str(error)) from error
    if document.get("repetition_context") is not None:
        from app.evidence.audio_repetition_binding import validate

        try:
            validate(document["repetition_context"], source)
        except ValueError as error:
            raise PacketIntegrityError(str(error)) from error
    if "stereo_repetition" in document:
        from app.evidence.stereo_repetition_binding import validate

        try:
            validate(document["stereo_repetition"], source)
        except ValueError as error:
            raise PacketIntegrityError(str(error)) from error
    if "similarity_context" in document:
        from app.evidence.audio_similarity_binding import validate

        try:
            validate(document["similarity_context"], source, document.get("repetition_context"))
        except ValueError as error:
            raise PacketIntegrityError(str(error)) from error
    if document.get("stereo_context") is not None:
        from app.detectors.audio.stereo import validate
        try:
            validate(document["stereo_context"], source)
        except (KeyError, TypeError, ValueError) as error:
            raise PacketIntegrityError("Invalid source-bound stereo channel context") from error
    if "stereo_listening" in document:
        from app.evidence.stereo_listening_binding import validate as validate_listening

        try:
            validate_listening(document["stereo_listening"], document.get("stereo_context"))
        except ValueError as error:
            raise PacketIntegrityError(str(error)) from error
    elif (document.get("stereo_context") or {}).get("channel_silence", {}).get("selected"):
        raise PacketIntegrityError("Exact channel silence is missing its listening evidence")


def _input(store: ObjectStore, analysis: Analysis, binding: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    _verified(store, analysis.sha256)
    mapping = json.loads(_verified(store, binding["mapping_sha256"]))
    pcm = read_wav(_verified(store, binding["wav_sha256"]))
    actual = hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
    if (mapping.get("version") != "audio-input/1" or mapping.get("source_sha256") != analysis.sha256
            or binding.get("version") != "audio-input/1" or binding.get("pcm_sha256") != actual
            or mapping.get("pcm_sha256") != actual or mapping.get("samples") != len(pcm)
            or binding.get("samples") != len(pcm) or binding.get("sample_rate") != SAMPLE_RATE
            or mapping.get("sample_rate") != SAMPLE_RATE or mapping.get("channels") != 1):
        raise PacketIntegrityError("The audio input no longer matches the input scored by the checks")
    source_ranges(mapping, 0, len(pcm))
    return pcm, mapping


def _signals(result: dict[str, Any], entry: dict[str, Any]) -> list[dict[str, Any]]:
    # DbRunStore persists only the raw detector identity/version; its _SpecView
    # has no family. The frozen published signal supplies that classification.
    # Older explicit voice records may omit family on their signal, but a
    # contradictory non-voice family is never accepted on either side.
    return [s for s in result.get("signals", []) if s.get("status", "ok") == "ok" and
            (s.get("family") == "voice" or (not s.get("family") and entry.get("family") == "voice")) and
            (s.get("detector_id") == entry["detector_id"] or set(s.get("exhibit_ids", [])).intersection(entry.get("exhibit_ids", [])))]


def _windows(store: ObjectStore, result: dict[str, Any], record: dict[str, Any], binding: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for reference in record.get("checks", []):
        entry = json.loads(_verified(store, reference["raw_output_sha256"]))
        if (entry.get("detector_id") != reference["detector_id"] or entry.get("status") != "ok"
                or entry.get("family") not in {None, "", "voice"}):
            raise PacketIntegrityError("The frozen audio check does not identify a successful voice run")
        input_ = entry.get("details", {}).get("audio_input") or {}
        if (input_.get("pcm_sha256") != binding["pcm_sha256"] or input_.get("sample_rate") != SAMPLE_RATE
                or input_.get("samples") != binding["samples"]):
            raise PacketIntegrityError("An audio window was scored from a different input")
        signals = _signals(result, entry)
        if not signals:
            raise PacketIntegrityError("An audio window has no successful published signal")
        refs = [{"id": alias_signal(s["id"]), "display_id": s.get("display_id")} for s in signals]
        for unit in entry.get("unit_scores", []):
            if not str(unit.get("unit_ref", "")).startswith("window:") or unit.get("raw_kind") != "probability":
                continue
            q = unit.get("quality") or {}
            score, a, b = unit.get("raw"), q.get("start_sample"), q.get("end_sample")
            if not all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in (score, a, b)):
                raise PacketIntegrityError("A scored audio window lacks exact sample coordinates")
            if a != int(a) or b != int(b) or not 0 <= score <= 1:
                raise PacketIntegrityError("Invalid audio window measurements")
            a, b = _interval({"start_sample": int(a), "end_sample": int(b)}, binding["samples"])
            if abs(q.get("start_s", -1) - a / SAMPLE_RATE) > 1e-6 or abs(q.get("end_s", -1) - b / SAMPLE_RATE) > 1e-6:
                raise PacketIntegrityError("Audio seconds and sample coordinates disagree")
            out.append({"start_sample": a, "end_sample": b, "score": score, "signal_refs": refs,
                        "unit_ref": unit["unit_ref"], "raw_output_sha256": reference["raw_output_sha256"]})
    return out


def _speech_selection(store, record, source, windows):
    if not (ref := record.get(SPEECH_SELECTION)):
        return None
    checksum = ref["raw_output_sha256"]
    entry = json.loads(_verified(store, checksum))
    details = entry["details"]
    audio_input = details["audio_input"]
    if (entry["detector_id"] != "aud.pella_v2" or entry["status"] not in {"ok", "not_applicable"}
            or audio_input["version"] != "audio-input/1"
            or any(audio_input[key] != source[key] for key in ("pcm_sha256", "samples", "sample_rate"))):
        raise PacketIntegrityError("Speech activity was recorded from a different audio input")
    return speech_window_selection(details[SPEECH_SELECTION], checksum, windows, source)


def verify_speech_selection(document, store, record):
    coverage = document["coverage"]
    expected = _speech_selection(store, record, document["source"], coverage["windows"])
    if canonical_hash(expected) != canonical_hash(coverage.get(SPEECH_SELECTION)):
        raise PacketIntegrityError("Speech window selection differs from the recorded check")


def _chart(pcm: np.ndarray, windows: list[dict[str, Any]], *, start: int, end: int,
           selected: tuple[int, int] | None, scores: bool, title: str | None = None,
           source_offset: int = 0, time_precision: int = 2) -> bytes:
    im = Image.new("RGB", (960, 280), (250, 249, 245))
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 58, 40, 938, 232
    def px(sample: int) -> float:
        return x0 + (sample - start) / (end - start) * (x1 - x0)
    if selected:
        draw.rectangle((px(selected[0]), y0, px(selected[1]), y1), fill=(255, 233, 195))
    label = title or ("Recorded window responses (uncalibrated)" if scores else "Analysed mono waveform (context only)")
    draw.text((x0, 12), label, fill=(30, 50, 40))
    draw.rectangle((x0, y0, x1, y1), outline=(135, 150, 135))
    if scores:
        threshold_y = y1 - THRESHOLD * (y1 - y0)
        draw.line((x0, threshold_y, x1, threshold_y), fill=(180, 150, 120), width=1)
        draw.text((6, threshold_y - 6), "0.5", fill=(70, 80, 70))
        for window in windows:
            a, b = max(start, window["start_sample"]), min(end, window["end_sample"])
            if a < b:
                y = y1 - window["score"] * (y1 - y0)
                draw.line((px(a), y, px(b), y), fill=(190, 108, 34) if window["score"] >= THRESHOLD else (45, 100, 78), width=3)
        draw.text((16, y0 - 5), "1.0", fill=(70, 80, 70))
        draw.text((16, y1 - 5), "0.0", fill=(70, 80, 70))
    else:
        chunks = np.array_split(pcm[start:end], x1 - x0)
        mid = (y0 + y1) / 2
        draw.line((x0, mid, x1, mid), fill=(190, 200, 190))
        for i, chunk in enumerate(chunks):
            if chunk.size:
                draw.line((x0 + i, mid - float(chunk.max()) * 90, x0 + i, mid - float(chunk.min()) * 90), fill=(45, 100, 78))
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = x0 + fraction * (x1 - x0)
        seconds = (source_offset + start + fraction * (end - start)) / SAMPLE_RATE
        draw.text((x - 12, y1 + 10), f"{seconds:.{time_precision}f}s", fill=(50, 70, 60))
    buffer = io.BytesIO()
    im.save(buffer, "PNG")
    return buffer.getvalue()


def _asset(db: Session, store: ObjectStore, analysis: Analysis, set_id: str, data: bytes, media_type: str,
           **metadata: Any) -> dict[str, Any]:
    art = register_artefact(db, store, data, media_type=media_type,
                            role="working_copy" if media_type == "audio/wav" else "preview",
                            created_by=CREATOR, analysis_id=analysis.id)
    ext = "wav" if media_type == "audio/wav" else "png"
    return {"sha256": art.sha256, "media_type": media_type, **metadata,
            "url": f"/api/v2/analyses/{analysis.id}/evidence-packets/{set_id}/assets/{art.sha256}.{ext}"}


def mouth_hold_waveform(pcm: np.ndarray, sound: dict, *, time_axis: str | None) -> bytes:
    from app.evidence import mouth_hold_sound

    start, stop = sound["samples"]
    if time_axis is None:
        return _chart(pcm, [], start=start, end=stop, selected=(start, stop), scores=False,
                      title=mouth_hold_sound.WAVEFORM_TITLE)
    if time_axis != mouth_hold_sound.SOURCE_CLOCK:
        raise ValueError("Unsupported mouth-hold waveform time axis")
    return _chart(pcm, [], start=start, end=stop, selected=(start, stop), scores=False,
                  title=mouth_hold_sound.SOURCE_WAVEFORM_TITLE, source_offset=sound["source_pts"][0] - start,
                  time_precision=3)


def build_packet_set(db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any]) -> dict[str, Any]:
    if analysis.kind != "audio" or canonical_hash(result) != analysis.result_hash or result.get("input", {}).get("sha256") != analysis.sha256:
        raise PacketIntegrityError("Audio evidence requires the exact published audio result")
    record = result.get("manifest", {}).get("sampling", {}).get("audio_evidence")
    if not record:
        raise PacketIntegrityError("This older audio analysis has no frozen PCM input binding; re-analysis is required")
    set_id = current_set_id(analysis)
    document = {"schema": SCHEMA, "version": VERSION, "status": "ready", "id": set_id,
                "analysis_id": analysis.id, "revision": analysis.revision, "original_result_hash": analysis.result_hash,
                **build_content(db, analysis, store, result, record=record, set_id=set_id)}
    document = json.loads(canonical_json(document))
    document["sha256"] = packet_hash(document)
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    register_artefact(db, store, canonical_json(document), media_type="application/json", role="exhibit_payload",
                     created_by=CREATOR, analysis_id=analysis.id, artefact_id=set_id)
    return document


def build_content(db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any], *,
                  record: dict, set_id: str) -> dict[str, Any]:
    """Build shared audio presentation content, without independent set identity.

    A video parent supplies its own set_id, source and result authority. This
    helper registers only its media assets; a read never calls it.
    """
    binding = record["input"]
    pcm, mapping = _input(store, analysis, binding)
    windows = _windows(store, result, record, binding)
    union = interval_union((w["start_sample"], w["end_sample"]) for w in windows)
    scored = sum(b - a for a, b in union)
    document: dict[str, Any] = {"source": {"sha256": analysis.sha256, "pcm_sha256": binding["pcm_sha256"], "wav_sha256": binding["wav_sha256"],
                   "mapping_sha256": binding["mapping_sha256"], "samples": len(pcm), "sample_rate": SAMPLE_RATE, "channels": 1,
                   "coordinate_space": "analysed-mono-pcm", "source_stream_index": mapping["source_stream_index"],
                   "source_channels": mapping["source_channels"], "source_sample_rate": mapping["source_sample_rate"],
                   "source_duration_s": mapping["source_duration_s"], "transform": mapping["transform"],
                   "timeline_time_base": mapping["timeline_time_base"], "timeline": mapping["timeline"]},
        "coverage": {"decoded_samples": len(pcm), "scored_samples": scored, "unscored_samples": len(pcm) - scored,
                     "decode_limit_reached": len(pcm) >= mapping["limit_samples"],
                     "scored_intervals": [{"start_sample": a, "end_sample": b} for a, b in union], "windows": windows,
                     "selection_threshold": THRESHOLD, "threshold_kind": "uncalibrated_window_response"},
        "packets": [], "omitted_windows": 0,
        "limitations": ["A response applies to the complete scored window, not individual words or exact edit boundaries.",
                        "Window responses are uncalibrated scores, not probabilities that a time range was altered.",
                        "The mono input mixes source channels. Compression, noise, speakers and unfamiliar synthesis methods can affect responses.",
                        "Waveforms and exported audio are context for inspection; they are not independent evidence of synthesis.",
                        "Unscored audio and audio beyond the decoded duration have not received a voice-check result."]}
    if selection := _speech_selection(store, record, document["source"], windows):
        document["coverage"][SPEECH_SELECTION] = selection
    if "audio_stream_indices" in mapping:
        document["source"]["audio_stream_indices"] = mapping["audio_stream_indices"]
    if analysis.kind == "audio" and (review := record.get("short_voice_review")):
        from app.evidence.audio_short_review import build as build_short_voice_review
        from app.evidence.audio_short_review_binding import source_metadata

        document["source"].update(source_metadata(mapping))
        document["short_voice_review"] = build_short_voice_review(
            review, store=store, source=document["source"], pcm=pcm, mapping=mapping,
            asset=lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata),
            waveform=_chart,
        )
    if analysis.kind == "audio" and (review := record.get("long_voice_review")):
        from app.evidence.audio_long_review import build as build_long_voice_review
        from app.evidence.audio_short_review_binding import source_metadata

        document["source"].update(source_metadata(mapping))
        document["long_voice_review"] = build_long_voice_review(
            review, store=store, source=document["source"], pcm=pcm, mapping=mapping,
            asset=lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata),
            waveform=_chart,
        )
    if analysis.kind == "audio" and (review := record.get("background_reduced_voice_review")):
        from app.evidence.audio_background_review import build as build_background_review
        from app.evidence.audio_short_review_binding import source_metadata

        document["source"].update(source_metadata(mapping))
        document["background_reduced_voice_review"] = build_background_review(
            review, store=store, source=document["source"], pcm=pcm, mapping=mapping,
            asset=lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata),
            waveform=_chart,
        )
    document["coverage"]["unscored_intervals"] = unscored_voice_intervals(document["coverage"], document["source"])
    clock = source_clock_context(document["coverage"], document["source"], guarded=record.get("voice_clock_policy") == VOICE_CLOCK_POLICY)
    if analysis.kind == "audio" or clock["total_discontinuities"]:
        document["coverage"]["source_clock_context"] = clock
    doc_assets = {}
    for name, scores in (("waveform", False), ("scores", True)):
        doc_assets[name] = _asset(db, store, analysis, set_id,
            _chart(pcm, windows, start=0, end=len(pcm), selected=None, scores=scores), "image/png", width=960, height=280)
    document["overview"] = doc_assets
    by_range: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for window in windows:
        by_range.setdefault((window["start_sample"], window["end_sample"]), []).append(window)
    candidates = [(key, values) for key, values in by_range.items() if max(v["score"] for v in values) >= THRESHOLD]
    candidates.sort(key=lambda item: (-max(v["score"] for v in item[1]), item[0]))
    document["omitted_windows"] = max(0, len(candidates) - MAX_PACKETS)
    for i, ((a, b), values) in enumerate(candidates[:MAX_PACKETS], 1):
        ca, cb = max(0, a - 2 * SAMPLE_RATE), min(len(pcm), b + 2 * SAMPLE_RATE)
        refs = {s["id"]: s for value in values for s in value["signal_refs"]}
        ids = set(refs)
        findings = [f["id"] for f in result.get("findings", []) if ids.intersection(alias_signal(ref) for ref in f.get("evidence_ids", []))]
        artifacts = {}
        for name, start, end in (("selected_clip", a, b), ("context_clip", ca, cb)):
            artifacts[name] = _asset(db, store, analysis, set_id, wav_bytes(pcm[start:end]), "audio/wav",
                                     samples=end-start, sample_rate=SAMPLE_RATE, channels=1)
        for name, scores in (("waveform", False), ("scores", True)):
            artifacts[name] = _asset(db, store, analysis, set_id,
                _chart(pcm, windows, start=ca, end=cb, selected=(a, b), scores=scores), "image/png", width=960, height=280)
        peak = max(v["score"] for v in values)
        document["packets"].append({"id": f"A{i:02d}", "title": "Speech inspection window", "evidence_type": "statistical",
            "interval": {"start_sample": a, "end_sample": b}, "context_interval": {"start_sample": ca, "end_sample": cb},
            "start_s": a / SAMPLE_RATE, "end_s": b / SAMPLE_RATE, "finding_ids": findings, "signal_refs": list(refs.values()),
            "observation": f"A voice check returned {peak:.3f} for the complete {a / SAMPLE_RATE:.3f}–{b / SAMPLE_RATE:.3f} s window.",
            "interpretation": "At least one recorded window response reaches the check's 0.5 inspection threshold. Listen to the exact scored input and its context; this does not locate a voice-replacement boundary.",
            "observations": [{"label": "Voice-check window response", "value": v["score"], "unit": "uncalibrated score",
                              "signal_refs": v["signal_refs"], "unit_ref": v["unit_ref"], "raw_output_sha256": v["raw_output_sha256"]} for v in values],
            "artifacts": artifacts, "limitations": ["Overlapping windows are correlated and do not count as independent confirmations.",
                "A high window response does not identify a specific word, speaker identity or an exact edit boundary."],
            "anchor": {"original_sha256": analysis.sha256, "pcm_sha256": binding["pcm_sha256"],
                       "coordinate_space": "analysed-mono-pcm", "sample_rate": SAMPLE_RATE,
                       "source_timestamp_ranges": source_ranges(mapping, a, b)},
            "derivation": {"version": VERSION, "clip_encoding": "PCM signed 16-bit WAV, 16000 Hz mono, no further normalisation",
                           "selection": "recorded window response >= 0.5; highest response first", "context_padding_samples": 2 * SAMPLE_RATE}})
    if not candidates:
        document["limitations"].append("No recorded window reached the inspection threshold. This does not establish authenticity."
                                       if windows else "No voice windows were scored; no local synthesis claim can be made.")
    from app.evidence import audio_background, audio_bandwidth_binding, audio_silence_binding
    acoustic_source = {key: document["source"][key] for key in
                       ("sha256", "pcm_sha256", "samples", "sample_rate", "coordinate_space")}
    acoustic_version = audio_background.VERSION
    if settings.audio_bandwidth_return_review and analysis.kind == "audio":
        if audio_bandwidth_binding.native_input(len(pcm), mapping, audio_bandwidth_binding.LEGACY_VERSION):
            acoustic_version = audio_bandwidth_binding.LEGACY_VERSION
        elif audio_bandwidth_binding.native_input(len(pcm), mapping, audio_bandwidth_binding.EXPANDED_VERSION):
            from app.evidence.audio_short_review_binding import source_metadata

            acoustic_version = audio_bandwidth_binding.EXPANDED_VERSION
            document["source"].update(source_metadata(mapping))
    document["acoustic_context"] = audio_background.build_content(pcm, mapping, acoustic_source, document["packets"],
        lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata), _chart,
        version=acoustic_version + audio_silence_binding.VERSION_SUFFIX)
    from app.evidence.sound_events import build as build_sound_events
    sound_context = build_sound_events(store, record, binding, document["source"], pcm, mapping,
        lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata), _chart)
    if sound_context is not None:
        document["sound_events"] = sound_context
    if stereo_sha := record.get("stereo_context_sha256"):
        document["stereo_context"] = json.loads(_verified(store, stereo_sha))
        document["stereo_context_sha256"] = stereo_sha
    elif analysis.kind == "audio" and "stereo_context" in mapping:
        document["stereo_context"] = mapping["stereo_context"]
    if "stereo_context" in document:
        attach_stereo_attenuation_overlaps(document)
    from app.evidence import stereo_listening

    listening_digest = binding.get("stereo_listening_pcm_sha256") or mapping.get("stereo_listening_pcm_sha256")
    if listening := stereo_listening.build(store, listening_digest, document.get("stereo_context"), pcm,
            lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata)):
        document["stereo_listening"] = listening
    from app.evidence.audio_repetition import build as build_repetition
    from app.evidence.audio_similarity import build as build_similarity

    document["repetition_context"] = build_repetition(
        pcm, mapping, document["source"],
        lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata),
    )
    from app.evidence.audio_gain_repetition_binding import GAIN_INPUT
    from app.evidence.stereo_repetition import build_from_store
    from app.evidence.stereo_repetition_binding import INPUT as STEREO_REPEAT_INPUT

    if digest := binding.get(STEREO_REPEAT_INPUT) or mapping.get(STEREO_REPEAT_INPUT):
        document["stereo_repetition"] = build_from_store(
            store, digest, document["source"], mapping,
            lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata),
            gain_version=binding.get(GAIN_INPUT) or mapping.get(GAIN_INPUT),
        )
    document["similarity_context"] = build_similarity(
        pcm, mapping, document["source"],
        lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata),
        repetition=document["repetition_context"] if settings.environment == "dev" and settings.audio_waveform_repeat_review else None,
    )
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
        raise PacketIntegrityError("Audio evidence set identity has changed")
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    store.verify(analysis.sha256)
    store.verify(document["source"]["wav_sha256"])
    mapping = json.loads(_verified(store, document["source"]["mapping_sha256"]))
    if document["source"].get("audio_stream_indices") != mapping.get("audio_stream_indices"):
        raise PacketIntegrityError("Audio track coverage differs from the frozen audio input mapping")
    if document.get("stereo_context") != mapping.get("stereo_context"):
        raise PacketIntegrityError("Stereo context differs from the frozen audio input mapping")
    listening_digest = mapping.get("stereo_listening_pcm_sha256")
    if listening_digest is not None or "stereo_listening" in document:
        from app.evidence import stereo_listening

        stereo_listening.verify(store, document, listening_digest, read_wav(_verified(store, document["source"]["wav_sha256"])))
    from app.evidence import stereo_repetition
    from app.evidence.audio_gain_repetition_binding import GAIN_INPUT
    from app.evidence.stereo_repetition_binding import INPUT as STEREO_REPEAT_INPUT

    if mapping.get(STEREO_REPEAT_INPUT) or "stereo_repetition" in document:
        stereo_repetition.verify_frozen(store, document, mapping, mapping.get(STEREO_REPEAT_INPUT), store.read_bytes,
                                       gain_version=mapping.get(GAIN_INPUT))
    return document


def persist_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if analysis.kind != "audio" or not analysis.result_hash:
        return None
    existing = load_packet_set(db, analysis, store, set_id=current_set_id(analysis))
    if existing is not None:
        return existing
    result = result or json.loads(_verified(store, analysis.result_hash))
    if not result.get("manifest", {}).get("sampling", {}).get("audio_evidence"):
        return None
    return build_packet_set(db, analysis, store, result)


def unavailable_packet_set(analysis: Analysis) -> dict[str, Any]:
    return {"schema": SCHEMA, "version": VERSION, "status": "not_generated", "id": None, "sha256": None,
            "analysis_id": analysis.id, "revision": analysis.revision, "original_result_hash": analysis.result_hash,
            "source": {"sha256": analysis.sha256}, "packets": [], "limitations": [
                "No verified audio evidence set is available. Re-run older analyses to freeze the exact scored PCM and sample mapping."]}
