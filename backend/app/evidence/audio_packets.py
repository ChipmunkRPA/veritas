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
from app.detectors.audio.coverage import interval_union
from app.detectors.audio.decode import SAMPLE_RATE, read_wav, source_ranges, wav_bytes
from app.evidence import audio_response_binding as responses
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact
from app.storage.objects import ObjectIntegrityError, ObjectStore

SCHEMA = "chipmunk.audio-evidence/1"
VERSION = "audio-evidence-4"
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
    if (document.get("schema") != SCHEMA or document.get("version") not in {"audio-evidence-1", "audio-evidence-2", "audio-evidence-3", VERSION} or document.get("status") != "ready" or document.get("analysis_id") != analysis_id
            or document.get("revision") != revision or document.get("original_result_hash") != original_result_hash
            or document.get("source", {}).get("sha256") != source_sha256 or document.get("sha256") != packet_hash(document)):
        raise PacketIntegrityError("Audio evidence is not bound to its frozen result and source")
    validate_content(document)
    if document.get("version") in {"audio-evidence-2", "audio-evidence-3", VERSION} and not document.get("acoustic_context"):
        raise PacketIntegrityError("Current audio evidence lacks frozen acoustic context")
    if document.get("version") in {"audio-evidence-3", VERSION} and document["acoustic_context"]["schema"] != "chipmunk.acoustic-context/2":
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
        responses.validate(document, required=document.get("version") == VERSION)
    except ValueError as error:
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
        # This adapter's unit is already its final calibrated estimate. Require
        # the saved source scope and published value; never apply a whole-file
        # calibration to another detector's raw window responses.
        reference_probability = None
        if entry["detector_id"] == "aud.combined_speech_v1":
            from app.detectors.audio.reference import COMBINED_SHA256, combined_tendency

            screen = entry.get("details", {}).get("audio_screening") or {}
            reference_probability, direction = combined_tendency(screen)
            mapping = json.loads(_verified(store, binding["mapping_sha256"]))
            if (reference_probability is None or binding["samples"] != 64000
                    or screen.get("checked_source_ranges") != source_ranges(mapping, 0, 64000)
                    or direction != responses.reference_claim(reference_probability)["direction"]
                    or len(entry.get("unit_scores", [])) != 1
                    or any(s.get("calibration_ref") != COMBINED_SHA256
                           or type(s.get("calibrated_probability")) not in (int, float)
                           or not math.isfinite(s["calibrated_probability"])
                           or abs(s["calibrated_probability"] - reference_probability) > 1e-12 for s in signals)):
                raise PacketIntegrityError("Calibrated speech evidence lacks its matching source-bound reference")
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
            if reference_probability is not None and (a != 0 or b != 64000 or abs(score - reference_probability) > 1e-12):
                raise PacketIntegrityError("Calibrated speech unit differs from its saved reference estimate")
            out.append({"start_sample": a, "end_sample": b, "score": score, "signal_refs": refs,
                        "unit_ref": unit["unit_ref"], "raw_output_sha256": reference["raw_output_sha256"],
                        **({"response_reference": responses.reference_claim(score)} if reference_probability is not None else {})})
    return out


def verify_responses(document: dict, *, store: ObjectStore, result: dict, record: dict, binding: dict) -> None:
    """Rebind score meanings and selected intervals to saved runs, without inference."""
    windows = _windows(store, result, record, binding)
    coverage = document["coverage"]
    union = interval_union((w["start_sample"], w["end_sample"]) for w in windows)
    intervals = [{"start_sample": a, "end_sample": b} for a, b in union]
    if (canonical_hash(coverage["windows"]) != canonical_hash(windows)
            or coverage["scored_intervals"] != intervals
            or coverage["scored_samples"] != sum(b-a for a, b in union)
            or coverage["unscored_samples"] != binding["samples"] - sum(b-a for a, b in union)):
        raise PacketIntegrityError("Speech coverage differs from the actual saved responses")
    selected = responses.candidates(windows)
    if (len(document["packets"]) != min(len(selected), MAX_PACKETS)
            or document["omitted_windows"] != max(0, len(selected) - MAX_PACKETS)):
        raise PacketIntegrityError("Speech packet selection differs from the actual saved responses")
    for packet, ((a, b), values) in zip(document["packets"], selected[:MAX_PACKETS], strict=True):
        refs = {s["id"]: s for value in values for s in value["signal_refs"]}
        if (packet["interval"] != {"start_sample": a, "end_sample": b}
                or canonical_hash(packet["observations"]) != canonical_hash([responses.observation(w) for w in values])
                or canonical_hash(packet["signal_refs"]) != canonical_hash(list(refs.values()))):
            raise PacketIntegrityError("Speech packet differs from its actual saved response and score meaning")


def _chart(pcm: np.ndarray, windows: list[dict[str, Any]], *, start: int, end: int,
           selected: tuple[int, int] | None, scores: bool) -> bytes:
    im = Image.new("RGB", (960, 280), (250, 249, 245))
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 58, 40, 938, 232
    def px(sample: int) -> float:
        return x0 + (sample - start) / (end - start) * (x1 - x0)
    if selected:
        draw.rectangle((px(selected[0]), y0, px(selected[1]), y1), fill=(255, 233, 195))
    reference_count = sum(bool(w.get("response_reference")) for w in windows)
    title = ("Calibrated speech reference estimate (AI tendency)" if reference_count == len(windows) and windows
             else "Window responses (different score scales)" if reference_count else "Recorded window responses (uncalibrated)")
    draw.text((x0, 12), title if scores else "Analysed mono waveform (context only)", fill=(30, 50, 40))
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
        draw.text((x - 12, y1 + 10), f"{(start + fraction * (end - start)) / SAMPLE_RATE:.2f}s", fill=(50, 70, 60))
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
                     "selection_threshold": THRESHOLD, "threshold_kind": "uncalibrated_window_response",
                     "response_semantics": responses.SEMANTICS},
        "packets": [], "omitted_windows": 0,
        "limitations": ["A response applies to the complete scored window, not individual words or exact edit boundaries.",
                        "Each response states its own score meaning. Raw responses are uncalibrated; calibrated speech estimates use a balanced reference, not a per-file accuracy guarantee.",
                        "The mono input mixes source channels. Compression, noise, speakers and unfamiliar synthesis methods can affect responses.",
                        "Waveforms and exported audio are context for inspection; they are not independent evidence of synthesis.",
                        "Unscored audio and audio beyond the decoded duration have not received a voice-check result."]}
    doc_assets = {}
    for name, scores in (("waveform", False), ("scores", True)):
        doc_assets[name] = _asset(db, store, analysis, set_id,
            _chart(pcm, windows, start=0, end=len(pcm), selected=None, scores=scores), "image/png", width=960, height=280)
    document["overview"] = doc_assets
    calibrated = any(w.get("response_reference") for w in windows)
    if calibrated:
        document["coverage"].update(selection_threshold=None, threshold_kind="reference_directions_and_raw_inspection",
            selection="Retain calibrated speech windows in either direction or without a direction; retain other raw responses at or above 0.5.")
    candidates = responses.candidates(windows)
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
        reference = next((v.get("response_reference") for v in values if v.get("response_reference")), None)
        if reference:
            p = next(v["score"] for v in values if v.get("response_reference"))
            confidence = min(99.9, math.floor(round(max(p, 1-p) * 1000, 8)) / 10)
            label = {"ai": "Leaning towards AI-generated speech", "original": "Leaning towards original speech"}.get(reference["direction"])
            observation = f"{label} with {confidence:g}% reference confidence for the complete {a / SAMPLE_RATE:.3f}–{b / SAMPLE_RATE:.3f} s window." if label else "The calibrated speech evidence has no directional preference for this complete window."
            interpretation = "The AI reference estimate uses a balanced AI/original reference population. It is not a probability that this particular recording is fake, does not assess background synthesis and does not locate an edit boundary."
        else:
            observation = f"A voice check returned {peak:.3f} for the complete {a / SAMPLE_RATE:.3f}–{b / SAMPLE_RATE:.3f} s window."
            interpretation = "At least one recorded window response reaches the check's 0.5 inspection threshold. Listen to the exact scored input and its context; this does not locate a voice-replacement boundary."
        document["packets"].append({"id": f"A{i:02d}", "title": "Speech inspection window", "evidence_type": "statistical",
            "interval": {"start_sample": a, "end_sample": b}, "context_interval": {"start_sample": ca, "end_sample": cb},
            "start_s": a / SAMPLE_RATE, "end_s": b / SAMPLE_RATE, "finding_ids": findings, "signal_refs": list(refs.values()),
            "observation": observation,
            "interpretation": interpretation,
            "observations": [responses.observation(v) for v in values],
            "artifacts": artifacts, "limitations": ["Overlapping windows are correlated and do not count as independent confirmations.",
                "A high window response does not identify a specific word, speaker identity or an exact edit boundary."],
            "anchor": {"original_sha256": analysis.sha256, "pcm_sha256": binding["pcm_sha256"],
                       "coordinate_space": "analysed-mono-pcm", "sample_rate": SAMPLE_RATE,
                       "source_timestamp_ranges": source_ranges(mapping, a, b)},
            "derivation": {"version": VERSION, "clip_encoding": "PCM signed 16-bit WAV, 16000 Hz mono, no further normalisation",
                           "selection": "calibrated reference window in either direction or without a direction; otherwise raw response >= 0.5; descending AI response order", "context_padding_samples": 2 * SAMPLE_RATE}})
    if not candidates:
        document["limitations"].append("No recorded window reached the inspection threshold. This does not establish authenticity."
                                       if windows else "No voice windows were scored; no local synthesis claim can be made.")
    from app.evidence.audio_background import build_content as build_acoustic_context
    acoustic_source = {key: document["source"][key] for key in
                       ("sha256", "pcm_sha256", "samples", "sample_rate", "coordinate_space")}
    document["acoustic_context"] = build_acoustic_context(pcm, mapping, acoustic_source, document["packets"],
        lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata), _chart)
    from app.evidence.sound_events import build as build_sound_events
    sound_context = build_sound_events(store, record, binding, document["source"], pcm, mapping,
        lambda data, media_type, **metadata: _asset(db, store, analysis, set_id, data, media_type, **metadata), _chart)
    if sound_context is not None:
        document["sound_events"] = sound_context
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
    store.verify(document["source"]["mapping_sha256"])
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
