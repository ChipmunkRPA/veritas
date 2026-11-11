"""Immutable, source-checked evidence of actual video frame observations.

An uncalibrated frame/face score supports inspecting that sampled input. It is
never converted into a pixel edit mask or continuous temporal coverage. Native
frame commitments written during inference bind every saved crop to the pixels
actually decoded, even with VFR, nonzero PTS origins and evicted frame caches.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from copy import deepcopy
from typing import Any

from PIL import Image, ImageDraw
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.public_projection import alias_signal
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.video.decode import FrameTable, decode_frames
from app.evidence import (
    face_alignment_context,
    face_boundary_binding,
    face_interval_binding,
    face_intervals,
    face_lighting_context,
    picture_hold_binding,
    whole_face_binding,
)
from app.evidence.image_packets import PacketIntegrityError, _bounds, _box, _png, _verified
from app.evidence.mp4 import validate_clip_binding
from app.evidence.video_clips import ClipUnavailable, build_clip, select_range
from app.evidence.video_packet_versions import (
    ACOUSTIC_VERSIONS,
    ALIGNMENT_CONTEXT_VERSIONS,
    BOUNDARY_VERSION,
    CONTINUITY_VERSIONS,
    LIGHTING_CONTEXT_VERSIONS,
    NESTED_VERSIONS,
    PICTURE_HOLD_VERSIONS,
    VERSIONS,
)
from app.storage.artefacts import get_artefact, register_artefact
from app.storage.db import Analysis, AnalysisStage, Artefact
from app.storage.objects import ObjectStore

SCHEMA = "chipmunk.video-evidence/1"
VERSION = BOUNDARY_VERSION
CREATOR = "video-evidence-packets/11"
CREATORS = ("video-evidence-packets/1", "video-evidence-packets/2", "video-evidence-packets/3", "video-evidence-packets/4",
            "video-evidence-packets/5", "video-evidence-packets/6", "video-evidence-packets/7", "video-evidence-packets/8",
            "video-evidence-packets/9", "video-evidence-packets/10", CREATOR)
ANCHOR_VERSION = "stream-pts-rgb24/1"
MAX_PACKETS = 6
MAX_FRAMES_PER_PACKET = 3
PREVIEW_MAX = 960
_NS = uuid.UUID("ab90189d-875b-4d57-a2cb-f56d6cad158f")
_SHA = re.compile(r"[a-f0-9]{64}")


def packet_hash(document: dict[str, Any]) -> str:
    return canonical_hash({k: v for k, v in document.items() if k != "sha256"})


def current_set_id(analysis: Analysis) -> str:
    return str(uuid.uuid5(_NS, f"{analysis.id}:{analysis.result_hash}:{VERSION}"))


def _stage(db: Session, store: ObjectStore, analysis_id: str, name: str) -> tuple[dict, str] | None:
    row = db.get(AnalysisStage, (analysis_id, name))
    if row is None or row.status != "done" or row.output_artefact_id is None:
        return None
    art = get_artefact(db, row.output_artefact_id, analysis_id=analysis_id)
    if art.role != "stage_output" or art.media_type != "application/json":
        raise PacketIntegrityError("Video source record is not a stage output")
    return json.loads(_verified(store, art.sha256)), art.sha256


def _records(db: Session, store: ObjectStore, analysis: Analysis, result: dict) -> tuple[FrameTable, dict, dict, list, dict | None]:
    probe, render = _stage(db, store, analysis.id, "probe"), _stage(db, store, analysis.id, "render")
    sampling = result.get("manifest", {}).get("sampling", {})
    if (probe is None or render is None or sampling.get("frame_anchor_version") != ANCHOR_VERSION
            or render[0].get("frame_anchor_version") != ANCHOR_VERSION):
        raise PacketIntegrityError("This video has no source-frame commitments; reanalyse it to create frame evidence")
    table_doc, anchors = probe[0].get("frame_table"), render[0].get("frame_anchors")
    if not isinstance(table_doc, dict) or not isinstance(anchors, list):
        raise PacketIntegrityError("Video frame records are missing")
    if (canonical_hash(table_doc) != sampling.get("frame_table_sha256")
            or canonical_hash(anchors) != sampling.get("frame_anchor_sha256")):
        raise PacketIntegrityError("Video source records differ from the published commitments")
    table = FrameTable.from_json(table_doc)
    if not len(table) or min(table.width, table.height) <= 0:
        raise PacketIntegrityError("Video frame table has no valid frames")
    by_index = {}
    for anchor in anchors:
        i = anchor["frame_index"]
        if (not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(table) or i in by_index
                or anchor.get("pts") != table.pts[i] or anchor.get("time_base") != table.time_base
                or (anchor.get("width"), anchor.get("height")) != (table.width, table.height)
                or not _SHA.fullmatch(str(anchor.get("decoded_rgb_sha256", "")))):
            raise PacketIntegrityError("Video frame anchor is inconsistent with its frame table")
        by_index[i] = anchor
    # Source start time is frozen with the result. For older files without it,
    # leave browser navigation unknown rather than silently assume zero.
    origin = sampling.get("timeline_origin_s")
    if origin is not None and (not isinstance(origin, int | float) or not math.isfinite(origin)):
        raise PacketIntegrityError("Video timeline origin is invalid")
    source = {"sha256": analysis.sha256, "stream_selector": "v:0", "time_base": table.time_base,
              "width": table.width, "height": table.height, "frame_count": len(table),
              "timeline_origin_s": origin, "coordinate_space": "display-oriented-native-frame",
              "frame_table_sha256": sampling["frame_table_sha256"], "frame_anchor_sha256": sampling["frame_anchor_sha256"]}
    hold = None
    if VERSION in PICTURE_HOLD_VERSIONS:
        hold = render[0].get("picture_hold")
        source["picture_hold_sha256"] = sampling.get("picture_hold_sha256")
        if (canonical_hash(hold) if hold is not None else None) != source["picture_hold_sha256"]:
            raise PacketIntegrityError("Video picture hold differs from its published commitment")
    return table, source, by_index, [{"stage": "probe", "sha256": probe[1]}, {"stage": "render", "sha256": render[1]}], hold


def _signals(result: dict, exhibit: dict) -> list[dict]:
    refs = exhibit.get("source_signal_ids", [])
    return [s for s in result.get("signals", []) if s.get("status", "ok") == "ok"
            and (s["id"] in refs or exhibit["id"] in s.get("exhibit_ids", []))]


def _series_points(db: Session, store: ObjectStore, aid: str, series: dict) -> list:
    if series.get("points") is not None:
        return series["points"]
    if not series.get("payload_artefact_id"):
        return []
    row = get_artefact(db, series["payload_artefact_id"], analysis_id=aid)
    if row.media_type != "application/json":
        return []
    data = json.loads(_verified(store, row.sha256))
    return data if isinstance(data, list) else []


def _observations(db: Session, store: ObjectStore, analysis: Analysis, result: dict,
                  table: FrameTable, anchors: dict) -> tuple[list[dict], set[int], int]:
    # A rounded second is merely a lookup aid. Every match must be unique and
    # agree with exact PTS time within the six-decimal serialization precision.
    by_time: dict[float, list[int]] = {}
    for i in anchors:
        if i not in table.interpolated:
            by_time.setdefault(round(table.seconds(i), 6), []).append(i)
    candidates, scored_frames, omitted = [], set(), 0
    for exhibit in sorted(result.get("exhibits", []), key=lambda e: e["id"]):
        if exhibit.get("kind") != "series" or exhibit.get("x_unit") != "s" or exhibit.get("media_ref") != "frame":
            continue
        signals = _signals(result, exhibit)
        if not signals:
            continue
        for series in exhibit.get("series", []):
            name = series.get("name")
            # Smoothed/fused curves are not raw frame measurements. Interval
            # window scores also need an explicit input-window binding first.
            if name not in {"p", "logit"}:
                continue
            rows = []
            for point in _series_points(db, store, analysis.id, series):
                if not isinstance(point, list | tuple) or len(point) != 2:
                    continue
                time_s, score = point
                if any(not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v) for v in point):
                    continue
                possible = by_time.get(round(time_s, 6), [])
                if len(possible) != 1 or abs(table.seconds(possible[0]) - time_s) > 0.0000011:
                    omitted += 1
                    continue
                i = possible[0]
                rows.append({"frame_index": i, "score": score})
                scored_frames.add(i)
            if not rows:
                continue
            rows.sort(key=lambda r: r["frame_index"])
            threshold = 0.0 if name == "logit" else 0.5
            peak = max(range(len(rows)), key=lambda j: rows[j]["score"])
            if rows[peak]["score"] < threshold:
                continue
            selected = sorted({max(0, peak - 1), peak, min(len(rows) - 1, peak + 1)})
            candidate = {"exhibit": exhibit, "signals": signals, "series_name": name,
                         "track_id": series.get("track_id"), "peak": rows[peak], "threshold": threshold,
                         "rows": [rows[j] for j in selected], "scored_frame_count": len(rows)}
            candidates.append(candidate)
            if name == "p" and exhibit.get("created_by") == face_intervals.FRAME_CHECK:
                tracks = face_intervals.bind_tracks(rows, result["exhibits"], table)
                for interval in face_intervals.localize(tracks):
                    peak_sample = interval.peak
                    candidates.append({**candidate, "track_id": interval.track_id,
                                       "face_interval": face_intervals.record(interval, table),
                                       "peak": {"frame_index": peak_sample.frame_index, "score": peak_sample.score},
                                       "rows": [{"frame_index": sample.frame_index, "score": sample.score, "interval_role": role}
                                                for role, sample in (("before", interval.before[-1]), ("during", peak_sample),
                                                                     ("after", interval.after[0]))]})
    return candidates, scored_frames, omitted


def _face_context(result: dict, track_id: str | None, index: int, table: FrameTable, *, face_check: bool) -> dict | None:
    width, height = table.width, table.height
    if track_id is None:
        # The served per-frame face path selects the largest detected face. This
        # box is context only: aligned crops or whole-portrait inputs may differ.
        sampling = result.get("manifest", {}).get("sampling", {})
        ms = round(table.seconds(index) * 1000)
        if (not face_check or sampling.get("exhibit_frame_pts_unit") != "milliseconds"
                or sum(round(table.seconds(i)*1000) == ms for i in range(len(table))) != 1):
            return None
        boxes = [r["region"]["bbox"] for e in result.get("exhibits", []) if e.get("kind") == "regions"
                 and e.get("created_by") == "vid.faces.yunet" for r in e.get("regions", [])
                 if r.get("region", {}).get("kind") == "bbox" and r["region"].get("frame_pts") == ms]
        return _box(_bounds(max(boxes, key=lambda b: b["w"]*b["h"]), width, height)) if boxes else None
    matches = [frame for e in result.get("exhibits", []) if e.get("kind") == "face_track"
               and e.get("track", {}).get("track_id") == track_id for frame in e.get("frames", [])
               if frame.get("frame_index") == index and frame.get("pts") == table.pts[index]]
    if len(matches) != 1:
        return None
    return _box(_bounds(matches[0].get("bbox"), width, height))


def _whole_face_candidate(result: dict, table: FrameTable, source: dict, anchors: dict) -> list[dict]:
    matches = [(signal, (signal.get("details") or {}).get("whole_face_review")) for signal in result.get("signals", [])
               if signal.get("status", "ok") == "ok" and (signal.get("details") or {}).get("whole_face_review") is not None]
    if not matches:
        return []
    if len(matches) != 1:
        raise PacketIntegrityError("Expected one paired face review")
    signal, record = matches[0]
    indices = whole_face_binding.validate(record, source, pts=table.pts)
    if (signal.get("effective_weight") != 0.0 or set(indices) & set(table.interpolated)
            or any(index not in anchors for index in indices)):
        raise PacketIntegrityError("Paired face review lacks the committed native inputs or inspection-only scope")
    samples = whole_face_binding.representatives(record)
    return [{"signals": [signal], "exhibit": {"id": None}, "series_name": "paired_face_review", "track_id": None,
             "whole_face_review": record, "window": {"frame_indices": list(indices)},
             "rows": [{"frame_index": row["frame_index"], "score": None, "box": row["box"]} for row in samples],
             "peak": {"frame_index": samples[1]["frame_index"], "score": float.fromhex(record["score_hex"])},
             "scored_frame_count": len(indices), "threshold": whole_face_binding.REVIEW_POINT}]


def _window_observations(result: dict, table: FrameTable, anchors: dict) -> tuple[list[dict], set[int]]:
    """Temporal feature statistics must cite every frame used by that window.

    Select the lowest observed variation for inspection, without borrowing the
    whole-video median threshold as a threshold on individual windows.
    """
    candidates, scored = [], set()
    for signal in result.get("signals", []):
        details = signal.get("details", {})
        if signal.get("status", "ok") != "ok" or details.get("video_window_binding_version") != "source-frame-window/1":
            continue
        windows = []
        for window in details.get("evidence_windows", []):
            indices = window.get("frame_indices", [])
            value = window.get("value")
            if (window.get("measure") != "temporal_feature_variation" or not isinstance(value, int | float)
                    or isinstance(value, bool) or not math.isfinite(value) or value < 0):
                continue
            if (not indices or indices != sorted(set(indices)) or any(i not in anchors or i in table.interpolated for i in indices)
                    or window.get("time_base") != table.time_base or window.get("start_pts") != table.pts[indices[0]]
                    or window.get("end_pts") != table.pts[indices[-1]]):
                raise PacketIntegrityError("A video scoring window does not match its committed input frames")
            for point in window.get("curve", []):
                if (point.get("frame_index") not in indices or not set(point.get("support_frame_indices", [])).issubset(indices)
                        or len(point.get("support_frame_indices", [])) != 3 or not isinstance(point.get("value"), int | float)
                        or not math.isfinite(point["value"])):
                    raise PacketIntegrityError("A temporal measurement is not bound to its three source frames")
            scored.update(indices)
            windows.append(window)
        if windows:
            selected = min(windows, key=lambda w: (w["value"], w["frame_indices"][0]))
            indices = selected["frame_indices"]
            shown = sorted({indices[0], indices[len(indices)//2], indices[-1]})
            candidates.append({"window": selected, "signals": [signal], "track_id": None, "series_name": "temporal_variation",
                               "exhibit": {"id": next(iter(signal.get("exhibit_ids", [])), "")},
                               "rows": [{"frame_index": i, "score": None} for i in shown],
                               "peak": {"frame_index": shown[len(shown)//2], "score": selected["value"]},
                               "threshold": None, "scored_frame_count": len(indices)})
    return candidates, scored


def _context(source: Image.Image, region: dict | None, label: str) -> Image.Image:
    image = source.copy()
    image.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    if region:
        sx, sy = image.width / source.width, image.height / source.height
        x, y, w, h = (region[k] for k in ("x", "y", "w", "h"))
        draw.rectangle((round(x*sx), round(y*sy), round((x+w-1)*sx), round((y+h-1)*sy)), outline=(250, 170, 40), width=3)
    draw.rectangle((0, 0, min(image.width - 1, 230), 20), fill=(30, 35, 30))
    draw.text((5, 4), label, fill=(255, 230, 165))
    return image


def _sequence_observations(result: dict, table: FrameTable, source: dict, anchors: dict) -> tuple[list[dict], set[int]]:
    candidates, scored = [], set()
    for signal in result.get("signals", []):
        details = signal.get("details") or {}
        record = details.get("face_sequence")
        if signal.get("status", "ok") != "ok" or details.get("face_sequence_binding_version") not in face_interval_binding.SEQUENCE_METHODS:
            continue
        if record is not None and record.get('method') != details['face_sequence_binding_version']:
            raise PacketIntegrityError('Face review measurement method differs from its binding version')
        for window in details.get("face_sequence_windows", []):
            if window.get('method') != details['face_sequence_binding_version']:
                raise PacketIntegrityError('Face review coverage method differs from its binding version')
            face_interval_binding.validate_sequence_record(window, source, table_pts=table.pts,
                                                          interpolated=table.interpolated, require_flag=False)
            for item in window["frame_inputs"]:
                index = item["frame_index"]
                if index not in anchors or any(item[key] != anchors[index][key] for key in ("pts", "decoded_rgb_sha256")):
                    raise PacketIntegrityError("Face sequence coverage differs from committed decoded source frames")
                scored.add(index)
        if record is None:
            continue
        face_interval_binding.validate_sequence_record(record, source, table_pts=table.pts, interpolated=table.interpolated)
        inputs = {row["frame_index"]: row for row in record["frame_inputs"]}
        for index, row in inputs.items():
            if index not in anchors or any(row[key] != anchors[index][key] for key in ("pts", "decoded_rgb_sha256")):
                raise PacketIntegrityError("Face sequence inputs differ from the committed decoded source frames")
        indices = sorted(inputs)
        groups = record["centers"]
        shown = [("before", groups["before"][-1]), ("during", groups["during"][len(groups["during"])//2]),
                 ("after", groups["after"][0])]
        candidates.append({"face_sequence": record, "window": {"frame_indices": indices, "value": record["response"],
                           "start_pts": table.pts[indices[0]], "end_pts": table.pts[indices[-1]]},
                           "signals": [signal], "track_id": record["track_id"], "series_name": "face_sequence_change",
                           "exhibit": {"id": next(iter(signal.get("exhibit_ids", [])), "")},
                           "rows": [{"frame_index": i, "score": None, "interval_role": role} for role, i in shown],
                           "peak": {"frame_index": shown[1][1], "score": record["response"]},
                           "threshold": None, "scored_frame_count": len(indices)})
        scored.update(indices)
        if boundary := details.get("face_boundary"):
            face_boundary_binding.validate_record(boundary, record, source, table_pts=table.pts, interpolated=table.interpolated)
            if signal.get("effective_weight") != 0.0:
                raise PacketIntegrityError("Refined face timing is limited to a confidence-neutral review check")
            for item in boundary["frame_inputs"]:
                index = item["frame_index"]
                if index not in anchors or any(item[key] != anchors[index][key] for key in ("pts", "decoded_rgb_sha256")):
                    raise PacketIntegrityError("Refined face timing differs from committed decoded source frames")
                scored.add(index)
            candidates[-1]["face_boundary"] = boundary
    return candidates, scored


def _contact(images: list[Image.Image], labels: list[str]) -> Image.Image:
    panel_w, panel_h = 320, 220
    sheet = Image.new("RGB", (panel_w * len(images), panel_h), (247, 247, 242))
    draw = ImageDraw.Draw(sheet)
    for n, (image, label) in enumerate(zip(images, labels, strict=True)):
        thumb = image.copy()
        thumb.thumbnail((panel_w - 8, panel_h - 28), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (n * panel_w + (panel_w - thumb.width) // 2, 0))
        draw.text((n * panel_w + 6, panel_h - 20), label, fill=(25, 30, 25))
    return sheet


def _packet_image(db: Session, store: ObjectStore, analysis: Analysis, set_id: str, image: Image.Image) -> dict:
    asset = register_artefact(db, store, _png(image), media_type="image/png", role="preview", created_by=CREATOR,
                              analysis_id=analysis.id)
    return {"sha256": asset.sha256, "media_type": "image/png", "width": image.width, "height": image.height,
            "url": f"/api/v2/analyses/{analysis.id}/evidence-packets/{set_id}/assets/{asset.sha256}.png"}


def _picture_hold_packet(db: Session, store: ObjectStore, analysis: Analysis, set_id: str,
                         source: dict, match: dict, decoded: dict) -> dict:
    packet_id = picture_hold_binding.PACKET_ID
    indices = (match["start_frame_index"], match["end_frame_index"])
    frames = []
    for position, (role, index) in enumerate(zip(("start", "end"), indices, strict=True), 1):
        if index not in decoded or hashlib.sha256(decoded[index]).hexdigest() != match["decoded_rgb_sha256"]:
            raise PacketIntegrityError("Picture hold endpoint differs from the native inspection")
        frames.append({"id": f"{packet_id}.{position}", "anchor": {
            "frame_index": index, "pts": match[f"{role}_pts"], "time_base": source["time_base"],
            "decoded_rgb_sha256": match["decoded_rgb_sha256"], "original_sha256": source["sha256"],
            "width": source["width"], "height": source["height"], "stream_selector": source["stream_selector"],
            "coordinate_space": source["coordinate_space"], "stream_time_s": match[f"{role}_stream_s"],
            "playback_time_s": match[f"{role}_playback_s"],
        }, "score": None, "is_peak": False, "context_region": None, "region_meaning": "whole_frame_context",
            "artifact_refs": {"original": "held_picture", "context": "held_picture"}})
    image = Image.fromarray(decoded[indices[0]])
    sheet = _contact([image, image], [f"First | frame {indices[0]}", f"Last | frame {indices[1]}"])
    return {"id": packet_id, "title": "Repeated picture across consecutive frames", "evidence_type": "anomaly",
            "claim_scope": "consecutive_frames", "semantics": picture_hold_binding.METHOD, "picture_hold": match,
            "finding_ids": [], "signal_refs": [], "exhibit_refs": [], "frames": frames,
            "observation": picture_hold_binding.observation(match),
            "interpretation": picture_hold_binding.CONTEXT,
            "limitations": ["The check covers a bounded opening sequence. Compression can change pixels even when a picture looks still."],
            "measurements": picture_hold_binding.measurements(match),
            "artifacts": {"held_picture": _packet_image(db, store, analysis, set_id, image),
                          "contact_sheet": _packet_image(db, store, analysis, set_id, sheet)},
            "clip": {"status": "unavailable", "reason": "Use the original video to inspect either measured endpoint"},
            "derivation": {"version": VERSION, "selection": "longest_exact_hold_in_bounded_opening_prefix",
                           "decoder": "ffmpeg-rgb24-display-v1", "frame_pts_unit": "stream_ticks", "source_pixels": "exact_native_rgb"}}


def _channel_coverage(db: Session, store: ObjectStore, analysis: Analysis, result: dict) -> dict:
    probe, render = _stage(db, store, analysis.id, "probe"), _stage(db, store, analysis.id, "render")
    has_audio = probe[0].get("info", {}).get("has_audio") if probe else None
    mapping = render[0].get("audio_mapping") if render else None
    sampling = result.get("manifest", {}).get("sampling", {})
    mapping_verified = bool(mapping and mapping.get("source_sha256") == analysis.sha256
                            and canonical_hash(mapping) == sampling.get("video_audio_mapping_sha256"))
    decode_status = render[0].get("audio_status") if render else None
    channels = {}
    for key, family in (("audio", "voice"), ("av_sync", "av-sync")):
        checks = [s for s in result.get("signals", []) if s.get("family") == family]
        successful = [s for s in checks if s.get("status", "ok") == "ok"]
        if has_audio is False:
            status, reason = "absent", "The source has no audio stream; voice and audio/video synchronization cannot be checked."
        elif successful:
            status, reason = "evidence_unavailable", "This channel produced responses, but source-bound playable audio/window evidence has not been frozen for this video."
        else:
            status, reason = "not_assessed", "No successful check with source-bound evidence is available for this channel."
        channels[key] = {"status": status, "reason": reason, "source_audio_present": has_audio,
                         "source_mapping_verified": mapping_verified, "decode_status": decode_status,
                         "successful_check_count": len(successful), "check_count": len(checks),
                         "check_refs": [{"id": alias_signal(s["id"]), "display_id": s.get("display_id"), "status": s.get("status", "ok")} for s in checks]}
    return channels


def build_packet_set(db: Session, analysis: Analysis, store: ObjectStore, result: dict[str, Any]) -> dict[str, Any]:
    if analysis.kind != "video" or not analysis.result_hash:
        raise ValueError("Video evidence needs a published video result")
    if canonical_hash(result) != analysis.result_hash or result.get("input", {}).get("sha256") != analysis.sha256:
        raise PacketIntegrityError("Video result or original identity does not match")
    store.verify(analysis.sha256)
    table, source, anchors, records, hold = _records(db, store, analysis, result)
    candidates, scored_frames, omitted = _observations(db, store, analysis, result, table, anchors)
    window_candidates, window_frames = _window_observations(result, table, anchors)
    sequence_candidates, sequence_frames = _sequence_observations(result, table, source, anchors)
    whole_candidates = _whole_face_candidate(result, table, source, anchors)
    window_frames.update(row["frame_index"] for candidate in whole_candidates for row in candidate["whole_face_review"]["samples"]
                         if any(value is not None for value in row["responses_hex"]))
    window_frames |= sequence_frames
    sequence_checks = sum(1 for signal in result.get("signals", []) if signal.get("status", "ok") == "ok"
                          and (signal.get("details") or {}).get("face_sequence_binding_version") in face_interval_binding.SEQUENCE_METHODS
                          and ((signal.get("details") or {}).get("face_sequence_windows") or (signal.get("details") or {}).get("face_sequence")))
    face_candidates = [item for item in candidates if "face_interval" in item]
    existing = window_candidates + [item for item in candidates if "face_interval" not in item]
    chosen = whole_candidates + sequence_candidates[:1] + face_candidates[:2-len(sequence_candidates[:1])] + existing[:MAX_PACKETS]
    scored_frames |= window_frames
    set_id = current_set_id(analysis)
    document = {"schema": SCHEMA, "version": VERSION, "id": set_id, "status": "ready", "analysis_id": analysis.id,
                "revision": analysis.revision, "original_result_hash": analysis.result_hash, "source": source, "packets": [],
                "coverage": {"kind": "point_and_window_samples" if window_frames else "point_samples", "frame_count": len(table), "decoded_frame_indices": sorted(anchors),
                             "scored_frame_indices": sorted(scored_frames), "scored_frame_count": len(scored_frames),
                             "window_frame_indices": sorted(window_frames), "window_check_count": len(window_candidates) + sequence_checks + len(whole_candidates),
                             "table_truncated": table.truncated, "interpolated_frame_indices": list(table.interpolated),
                             "unmapped_observations": omitted, "channels": _channel_coverage(db, store, analysis, result)},
                "source_records": records, "omitted_candidates": len(existing) + len(face_candidates) + len(sequence_candidates) + len(whole_candidates) - len(chosen),
                "limitations": ["These are sampled frame observations. Unscored frames and gaps are not negative findings.",
                                "Frame and face classification scores do not locate edited pixels or prove AI editing.",
                                "The contact sheet shows neighbouring scored samples, not necessarily consecutive video frames."]}
    if VERSION in PICTURE_HOLD_VERSIONS:
        document["picture_hold"] = (hold["coverage"] if hold is not None else picture_hold_binding.coverage(
            "unavailable", picture_hold_binding.HISTORICAL, source["time_base"]))
    match = hold["match"] if hold is not None else None
    wanted = {r["frame_index"] for c in chosen for r in c["rows"]}
    wanted.update(i for c in chosen for i in c.get("face_boundary", {}).get("anchors", {}).values())
    endpoints = {match["start_frame_index"], match["end_frame_index"]} if match is not None else set()
    selected = sorted(wanted | endpoints)
    decoded = {f.index: f.rgb for f in decode_frames(store.path_for(analysis.sha256), table, selected)} if selected else {}
    for i in wanted:
        if i not in decoded or hashlib.sha256(decoded[i].tobytes()).hexdigest() != anchors[i]["decoded_rgb_sha256"]:
            raise PacketIntegrityError("Video evidence pixels differ from the frame decoded during analysis")
    clip_cache = {}
    for number, candidate in enumerate(chosen, 1):
        packet_id = f"V{number:02d}"
        artifacts, frames, context_images, labels = {}, [], [], []

        def add(role: str, image: Image.Image, target: dict = artifacts) -> None:
            target[role] = _packet_image(db, store, analysis, set_id, image)

        def write_frame(i: int, region: dict | None, frame_id: str, key: str, *,
                        score: float | None, is_peak: bool, role: str | None) -> tuple[dict, Image.Image]:
            image = Image.fromarray(decoded[i])
            stream_s = table.seconds(i)
            player_s = None if source["timeline_origin_s"] is None else stream_s - source["timeline_origin_s"]
            context = _context(image, region, f"{frame_id} - frame {i}, {stream_s:.3f} s")
            original_key, context_key = f"{key}_original", f"{key}_context"
            add(original_key, image)
            add(context_key, context)
            refs = {"original": original_key, "context": context_key}
            if region:
                crop_key = f"{key}_crop"
                add(crop_key, image.crop(_bounds(region, table.width, table.height)))
                refs["crop"] = crop_key
            frame = {"id": frame_id, "anchor": {**anchors[i], "original_sha256": analysis.sha256,
                     "stream_selector": "v:0", "stream_time_s": stream_s, "playback_time_s": player_s,
                     "coordinate_space": "display-oriented-native-frame"}, "score": score,
                     "is_peak": is_peak, "context_region": region,
                     "region_meaning": "detected_face_context" if region else "whole_frame_context", "artifact_refs": refs}
            if role:
                frame["interval_role"] = role
            return frame, context

        for pos, row in enumerate(candidate["rows"], 1):
            i = row["frame_index"]
            region = _face_context(result, candidate["track_id"], i, table,
                                   face_check=any(s.get("family") == "learned-face" for s in candidate["signals"]))
            if sequence := candidate.get("face_sequence"):
                measured = next(row for row in sequence["frame_inputs"] if row["frame_index"] == i)
                region = face_interval_binding.face_context_box(measured["bbox"], source)
            if candidate.get("whole_face_review"):
                region = row["box"]
            role = row.get("interval_role")
            frame, context = write_frame(i, region, f"{packet_id}.{pos}", f"frame_{pos}", score=row["score"],
                                         is_peak="window" not in candidate and i == candidate["peak"]["frame_index"], role=role)
            frames.append(frame)
            context_images.append(context)
            stream_s = frame["anchor"]["stream_time_s"]
            labels.append(f"{role.title()} | {stream_s:.3f} s | frame {i}" if role else
                          f"Frame {i}; score {row['score']:.3f}" if row["score"] is not None else f"Frame {i}; window context")
        add("contact_sheet", _contact(context_images, labels))
        signal_ids = {s["id"] for s in candidate["signals"]}
        exhibit_id = candidate["exhibit"]["id"]
        findings = [f["id"] for f in result.get("findings", []) if (signal_ids | {exhibit_id}).intersection(f.get("evidence_ids", []))]
        face = candidate["track_id"] is not None or any(s.get("family") == "learned-face" for s in candidate["signals"])
        measure = "Frame logit" if candidate["series_name"] == "logit" else "Frame detection score"
        packet = {"id": packet_id, "title": "Face-forgery inspection clue" if face else "Frame synthesis inspection clue",
                                   "evidence_type": "statistical", "claim_scope": "sampled_frames", "semantics": "classifier_response",
                                   "finding_ids": findings, "exhibit_refs": [exhibit_id] if exhibit_id else [],
                                   "signal_refs": [{"id": alias_signal(s["id"]), "display_id": s.get("display_id")} for s in candidate["signals"]],
                                   "observation": f"{measure} reached {candidate['peak']['score']:.3f} at frame {candidate['peak']['frame_index']}; "
                                                  f"the recorded decision boundary is {candidate['threshold']}.",
                                   "interpretation": "This recorded response supports examining the sampled input and its neighbours. "
                                                     "Compression, motion blur and unfamiliar content can also affect the response.",
                                   "measurements": [{"label": measure, "value": candidate["peak"]["score"], "unit": "uncalibrated score"},
                                                    {"label": "Recorded decision boundary", "value": candidate["threshold"], "unit": "score"},
                                                    {"label": "Frames scored by this series", "value": candidate["scored_frame_count"], "unit": "frames"}],
                                   "limitations": ["No calibrated probability or pixel-level edit boundary is established by this score.",
                                                    "A face outline is detected-face context, not a manipulation mask or exact aligned classifier input.",
                                                    "Only the displayed sampled frames carry these observations; playback between them is context."],
                                   "frames": frames, "artifacts": artifacts,
                                   "derivation": {"version": VERSION, "selection": "highest_recorded_point_and_adjacent_samples",
                                                  "decoder": "ffmpeg-rgb24-display-v1", "frame_pts_unit": "stream_ticks",
                                                  "source_pixels": "exact_native_rgb", "contact_sheet": "sampled_sequence"}}
        if "window" in candidate and "whole_face_review" not in candidate:
            window = candidate["window"]
            indices = window["frame_indices"]
            origin = source["timeline_origin_s"]
            packet.update({"title": "Temporal variation inspection clue", "evidence_type": "anomaly",
                           "claim_scope": "sampled_window", "semantics": "temporal_feature_variation",
                           "observation": f"Temporal feature variation was {window['value']:.5f} across {len(indices)} sampled frames. "
                                          "This is the lowest recorded clip variation for this check.",
                           "interpretation": "A relatively flat sequence of visual features is a synthesis clue measured by this check. "
                                             "Static scenes and ordinary camera footage can also have low variation.",
                           "measurements": [{"label": "Temporal feature variation", "value": window["value"], "unit": "feature-distance standard deviation"},
                                            {"label": "Input frames in this window", "value": len(indices), "unit": "frames"}],
                           "window": {"frame_indices": indices, "start_pts": window["start_pts"], "end_pts": window["end_pts"],
                                      "time_base": table.time_base, "start_stream_s": table.seconds(indices[0]),
                                      "end_stream_s": table.seconds(indices[-1]),
                                      "start_playback_s": None if origin is None else table.seconds(indices[0]) - origin,
                                      "end_playback_s": None if origin is None else table.seconds(indices[-1]) - origin},
                           "observed_curve": window.get("curve", []),
                           "limitations": ["This is an uncalibrated window statistic, not a probability of AI generation.",
                                            "No individual-window decision threshold is established here; the lowest observed variation is selected for inspection.",
                                            "The contact sheet contains representative input frames. The listed input frames establish actual sample coverage."],
                           "derivation": {**packet["derivation"], "selection": "lowest_observed_temporal_variation"}})
        if whole := candidate.get("whole_face_review"):
            indices = candidate["window"]["frame_indices"]
            origin = source["timeline_origin_s"]
            start, end = table.seconds(indices[0]), table.seconds(indices[-1])
            packet.update(title="Sampled faces to review" if whole["flagged"] else "Sampled face review",
                          evidence_type="anomaly", claim_scope="sampled_window", semantics=whole_face_binding.METHOD,
                          whole_face_review=whole, affects_overall_confidence=False,
                          observation=candidate["signals"][0]["observed"],
                          interpretation="The source frames show the faces at sampled positions. The check compares responses across the clip and carries no weight in overall AI confidence.",
                          measurements=[{"label": "Planned face-check positions", "value": len(indices), "unit": "frames"},
                                        *[{"label": f"Usable positions for check {arm + 1}",
                                           "value": sum(row["responses_hex"][arm] is not None for row in whole["samples"]),
                                           "unit": "frames"} for arm in range(2)]],
                          window={"frame_indices": indices, "time_base": table.time_base,
                                  "start_pts": table.pts[indices[0]], "end_pts": table.pts[indices[-1]],
                                  "start_stream_s": start, "end_stream_s": end,
                                  "start_playback_s": None if origin is None else start - origin,
                                  "end_playback_s": None if origin is None else end - origin},
                          limitations=["The check assesses the largest detected face at each sampled position. These samples provide neither a continuous edit range nor an identity comparison."],
                          derivation={**packet["derivation"], "selection": "representative_paired_face_samples"},
                          clip={"status": "unavailable", "reason": "Use the original video for playback at the sampled positions"})
            document["packets"].append(packet)
            continue
        if sequence := candidate.get("face_sequence"):
            packet.update(title="Face-change range to review", semantics=sequence['method'],
                          face_sequence=sequence,
                          observation=face_interval_binding.sequence_observation(sequence),
                          interpretation="This pattern supports reviewing a possible face edit. Head motion, expression, lighting and compression can also cause it. The displayed frames show before, during and after context; none has an individual AI confidence score.",
                          measurements=[{"label": "Face-sequence change", "value": sequence["response"], "unit": "uncalibrated feature distance"},
                                        {"label": "Input frames in this window", "value": len(sequence["frame_inputs"]), "unit": "frames"}],
                          limitations=["This is a window-level review cue, not proof of a face swap or an authenticity probability.",
                                       "Three consecutive native face crops support each listed center. Gaps between centers are not continuously assessed.",
                                       "The face crops shown here are source context, not pixel-level manipulation masks.",
                                       "This cue does not change overall confidence."],
                          derivation={**packet["derivation"], "selection": "highest_source_bound_face_sequence_change"})
            if VERSION in ALIGNMENT_CONTEXT_VERSIONS and sequence["method"] == face_interval_binding.APPEARANCE_METHOD:
                packet["measurements"].extend(face_alignment_context.measure(sequence))
                packet["interpretation"] += (
                    " Compare the eye spacing and the adjacent frames for changes in head angle or face visibility."
                    " Eye spacing describes how the face was tracked in the sampled frames."
                )
            if VERSION in LIGHTING_CONTEXT_VERSIONS and sequence["method"] == face_interval_binding.APPEARANCE_METHOD:
                lighting = face_lighting_context.record(packet, source, decoded)
                packet["face_lighting"] = lighting
                packet["measurements"].extend(face_lighting_context.measurements(lighting))
                packet["interpretation"] += face_lighting_context.explanation(lighting)
                if lighting["status"] == "measured":
                    packet["limitations"].append(
                        "Brightness uses three displayed frames. Camera motion can change the scene pixels being compared."
                    )
        if boundary := candidate.get("face_boundary"):
            boundary_frames = []
            inputs = {row["frame_index"]: row for row in boundary["frame_inputs"]}
            for position, role in enumerate(face_boundary_binding.ROLES, 1):
                i = boundary["anchors"][role]
                region = face_interval_binding.face_context_box(inputs[i]["bbox"], source)
                frame, _ = write_frame(i, region, f"{packet_id}.B{position}", f"boundary_frame_{position}",
                                       score=None, is_peak=False, role=role)
                boundary_frames.append(frame)
            packet.update(face_boundary=boundary, face_boundary_frames=boundary_frames, affects_overall_confidence=False)
        if interval := candidate.get("face_interval"):
            start, end = interval["start_stream_s"], interval["end_stream_s"]
            packet.update(title="Face-swap interval to review", evidence_type="anomaly", claim_scope="sampled_window",
                          semantics=face_intervals.METHOD, face_interval=interval,
                          observation=f"The face check stayed elevated across {len(interval['samples']['during'])} sampled frames from {start:.3f} to {end:.3f} s.",
                          interpretation="The responses rose and fell on the same tracked face. Review the surrounding frames for changes in the face, lighting, blur, or compression.",
                          measurements=[{"label": "Rise above both neighboring groups", "value": interval["lift"], "unit": "uncalibrated score"}],
                          limitations=["The range runs from the first to the last flagged sample. Exact edit boundaries and pixels remain undetermined.",
                                       "A face track joins nearby detections. It does not establish anyone's identity."],
                          derivation={**packet["derivation"], "selection": "sustained_face_response_with_flanking_context"})
        try:
            bounds = select_range(table, candidate["peak"]["frame_index"])
            if bounds not in clip_cache:
                clip_cache[bounds] = build_clip(store.path_for(analysis.sha256), table, source_sha256=analysis.sha256,
                                               focus_index=candidate["peak"]["frame_index"], sampled_frame_indices=[],
                                               timeline_origin_s=source["timeline_origin_s"])
            data, metadata, description = clip_cache[bounds]
            art = register_artefact(db, store, data, media_type="video/mp4", role="preview", created_by=CREATOR, analysis_id=analysis.id)
            artifacts["context_clip"] = {**metadata, "sha256": art.sha256, "media_type": "video/mp4",
                                         "url": f"/api/v2/analyses/{analysis.id}/evidence-packets/{set_id}/assets/{art.sha256}.mp4"}
            packet["clip"] = deepcopy(description)
            measured = packet.get("window", {}).get("frame_indices") or [f["anchor"]["frame_index"] for f in frames]
            packet["clip"]["sampled_frame_indices"] = [i for i in measured if bounds[0] <= i < bounds[1]]
        except ClipUnavailable as error:
            packet["clip"] = {"status": "unavailable", "reason": str(error)}
        document["packets"].append(packet)
    if match is not None:
        document["packets"].append(_picture_hold_packet(db, store, analysis, set_id, source, match, decoded))
    if VERSION in NESTED_VERSIONS:
        from app.evidence import video_audio, video_av

        document["audio"] = video_audio.build_channel(db, analysis, store, result, set_id=set_id)
        document["av"] = video_av.build_channel(db, analysis, store, result, set_id=set_id,
            video_source=source, table=table, anchors=anchors, audio_channel=document["audio"])
        document["coverage"]["channels"] = {
            key: {"status": document[key]["status"], "reason": document[key].get("reason") or
                  ("Actual voice windows and source samples are retained." if key == "audio" else
                   "Measured mouth/audio correlation windows and source inputs are retained; this is not AI proof.")}
            for key in ("audio", "av")}
    document = json.loads(canonical_json(document))
    document["sha256"] = packet_hash(document)
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    try:
        picture_hold_binding.validate_record(document, hold, table_pts=table.pts, interpolated=table.interpolated,
                                             canonical_hash=canonical_hash)
    except (KeyError, TypeError, ValueError) as error:
        raise PacketIntegrityError(str(error)) from error
    register_artefact(db, store, canonical_json(document), media_type="application/json", role="exhibit_payload",
                     created_by=CREATOR, analysis_id=analysis.id, artefact_id=set_id)
    return document


def validate_packet_set(document: dict[str, Any], *, analysis_id: str, revision: int,
                        original_result_hash: str, source_sha256: str) -> None:
    if (document.get("schema") != SCHEMA or document.get("version") not in VERSIONS
            or document.get("status") != "ready" or document.get("analysis_id") != analysis_id
            or document.get("revision") != revision or document.get("original_result_hash") != original_result_hash
            or document.get("source", {}).get("sha256") != source_sha256 or document.get("sha256") != packet_hash(document)):
        raise PacketIntegrityError("Video evidence differs from its source, result or recorded hash")
    source = document["source"]
    try:
        picture_hold_binding.validate(document)
    except (KeyError, TypeError, ValueError) as error:
        raise PacketIntegrityError(str(error)) from error
    table_base = FrameTable(time_base=source["time_base"], pts=(), key_frames=(), pict_types="", width=source["width"],
                            height=source["height"], coded_width=source["width"], coded_height=source["height"]).tb
    ids = set()
    for packet in document["packets"]:
        if (packet["id"] in ids or (packet.get("evidence_type"), packet.get("claim_scope"))
                not in {("statistical", "sampled_frames"), ("anomaly", "sampled_window"), ("anomaly", "consecutive_frames")}):
            raise PacketIntegrityError("Video evidence has duplicate or misrepresented observations")
        ids.add(packet["id"])
        artifacts = packet["artifacts"]
        for asset in artifacts.values():
            if (not _SHA.fullmatch(str(asset.get("sha256", ""))) or asset.get("media_type") not in {"image/png", "video/mp4"}
                    or min(asset.get("width", 0), asset.get("height", 0)) <= 0):
                raise PacketIntegrityError("Video evidence has invalid image references")
        try:
            validate_clip_binding(packet, source)
            face_interval_binding.validate_packet(packet, source)
            face_boundary_binding.validate_packet(packet, source, document["version"])
            whole_face_binding.validate_packet(packet, source)
            if document["version"] in ALIGNMENT_CONTEXT_VERSIONS:
                face_alignment_context.validate(packet)
            if document["version"] in LIGHTING_CONTEXT_VERSIONS:
                face_lighting_context.validate(packet, source)
        except (KeyError, TypeError, ValueError) as error:
            raise PacketIntegrityError(str(error)) from error
        if document["version"] in {"video-evidence-2", *NESTED_VERSIONS} and "clip" not in packet:
            raise PacketIntegrityError("Video evidence omits its context-clip availability")
        for frame in (*packet["frames"], *packet.get("face_boundary_frames", ())):
            anchor = frame["anchor"]
            if (anchor.get("original_sha256") != source_sha256 or anchor.get("time_base") != source["time_base"]
                    or not 0 <= anchor.get("frame_index", -1) < source["frame_count"]
                    or not _SHA.fullmatch(str(anchor.get("decoded_rgb_sha256", "")))
                    or abs(float(anchor["pts"] * table_base) - anchor["stream_time_s"]) > 0.0000011):
                raise PacketIntegrityError("Video evidence frame has an invalid source anchor")
            origin, playback = source.get("timeline_origin_s"), anchor.get("playback_time_s")
            if (origin is None) != (playback is None) or (origin is not None and abs(anchor["stream_time_s"] - origin - playback) > 0.000002):
                raise PacketIntegrityError("Video evidence mixes stream timestamps and playback time")
            if not set(frame["artifact_refs"].values()).issubset(artifacts):
                raise PacketIntegrityError("Video frame refers to an image outside its packet")
            if frame.get("context_region"):
                _bounds(frame["context_region"], source["width"], source["height"])
        if packet.get("window"):
            window = packet["window"]
            indices = window["frame_indices"]
            if (not indices or indices != sorted(set(indices)) or any(not 0 <= i < source["frame_count"] for i in indices)
                    or window.get("time_base") != source["time_base"]
                    or not {f["anchor"]["frame_index"] for f in packet["frames"]}.issubset(indices)):
                raise PacketIntegrityError("Video window evidence has invalid frame support")
    if document["version"] in NESTED_VERSIONS:
        from app.evidence.av_binding import validate_audio_clock_coverage, validate_av_channel
        from app.evidence.video_audio import validate_channel

        try:
            validate_channel(document["audio"], source_sha256=source_sha256)
            validate_av_channel(document["av"], source)
            validate_audio_clock_coverage(document)
            if (document["version"] in ACOUSTIC_VERSIONS and document["audio"]["coverage"] is not None
                    and not document["audio"].get("acoustic_context")):
                raise ValueError("Current video evidence omits its acoustic context measurements")
            if (document["version"] in CONTINUITY_VERSIONS and document["audio"]["coverage"] is not None
                    and document["audio"]["acoustic_context"]["schema"] != "chipmunk.acoustic-context/2"):
                raise ValueError("Current video evidence omits continuity measurements")
        except (KeyError, TypeError, ValueError) as error:
            raise PacketIntegrityError("Video audio/AV evidence has invalid source bindings") from error
        all_ids = [p["id"] for part in (document, document["audio"], document["av"]) for p in part["packets"]]
        all_ids.extend(p["id"] for p in (document["audio"].get("acoustic_context") or {}).get("events", []))
        if len(all_ids) != len(set(all_ids)):
            raise PacketIntegrityError("Video channels contain ambiguous packet identifiers")
    elif any(key in document for key in ("audio", "av")):
        raise PacketIntegrityError("Historical video evidence cannot introduce undeclared nested channels")


def load_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, set_id: str | None = None) -> dict[str, Any] | None:
    query = select(Artefact).where(Artefact.analysis_id == analysis.id, Artefact.created_by.in_(CREATORS),
                                   Artefact.role == "exhibit_payload", Artefact.media_type == "application/json")
    if set_id is not None:
        query = query.where(Artefact.id == set_id)
    row = db.scalars(query.order_by(Artefact.created_at.desc(), Artefact.id)).first()
    if row is None:
        return None
    document = json.loads(_verified(store, row.sha256))
    if document.get("id") != row.id:
        raise PacketIntegrityError("Video evidence identity differs from its stored object")
    validate_packet_set(document, analysis_id=analysis.id, revision=analysis.revision,
                        original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    store.verify(analysis.sha256)
    if document["version"] in NESTED_VERSIONS:
        for key in ("audio", "av"):
            if document[key]["coverage"] is not None:
                for name in ("wav_sha256", "mapping_sha256"):
                    store.verify(document[key]["source"][name])
        if stereo_sha := document["audio"].get("stereo_context_sha256"):
            store.verify(stereo_sha)
    return document


def persist_packet_set(db: Session, analysis: Analysis, store: ObjectStore, *, result: dict | None = None) -> dict | None:
    if analysis.kind != "video" or not analysis.result_hash:
        return None
    existing = load_packet_set(db, analysis, store, set_id=current_set_id(analysis))
    if existing is not None:
        return existing
    return build_packet_set(db, analysis, store, result or json.loads(_verified(store, analysis.result_hash)))


def unavailable_packet_set(analysis: Analysis) -> dict:
    return {"schema": SCHEMA, "status": "not_generated" if analysis.kind == "video" else "not_applicable", "id": None,
            "version": VERSION, "sha256": None, "analysis_id": analysis.id, "revision": analysis.revision,
            "original_result_hash": analysis.result_hash, "source": {"sha256": analysis.sha256}, "packets": [],
            "coverage": None, "omitted_candidates": 0,
            "limitations": ["Source-bound video evidence is not available for this revision. Reanalyse the video to record exact frame commitments."]}
