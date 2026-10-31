"""Specimen report snapshots (BUILD_SPEC R-2, R-3): one per kind (image, video, audio, document) for the summary and forensic
profiles, the newsroom variant and the standard profile of the image, and a stress specimen of more than 30 forensic pages.

Fixed ids, times and values, so the golden text tests, the generation timing record and page reviews render the same pages every
time. They are test specimens, not measurements: detector ids start with `specimen.`, the error-rate rows exercise the icon-array
rule (≥ 100 test items) and the "k of n" rule, and the PDF of a specimen never leaves the tests and QA captures. Every result
document is validated against `AnalysisResultV2` first, so a specimen can only carry what a published result can.
"""
from __future__ import annotations

import copy
import hashlib
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from app.audit import chain as audit
from app.reports import engine, view
from app.reports.snapshot import (
    FULL_PROFILES,
    ReportSnapshot,
    SnapshotAnalysis,
    SnapshotAnnotation,
    SnapshotCase,
    SnapshotConclusion,
    SnapshotCoverage,
    SnapshotMedia,
    SnapshotReport,
    SnapshotReview,
    SnapshotSignoff,
    display_id,
)
from app.schemas.evidence import AnalysisResultV2
from app.version import VERSION

KINDS = ("image", "video", "audio", "document")
CASES = (("image", "summary"), ("video", "summary"), ("audio", "summary"), ("document", "summary"), ("image", "newsroom"))
FULL_CASES = (("image", "forensic"), ("video", "forensic"), ("audio", "forensic"), ("document", "forensic"), ("image", "standard"))
RECEIVED_AT = datetime(2026, 9, 11, 9, 0, tzinfo=UTC)
ACQUIRED_AT = datetime(2026, 9, 10, 16, 45, tzinfo=UTC)  # as the custodian stated it (N-19)
CREATED_AT = datetime(2026, 9, 11, 9, 10, tzinfo=UTC)
FINISHED_AT = datetime(2026, 9, 11, 9, 12, 4, tzinfo=UTC)
REQUESTED_AT = datetime(2026, 9, 11, 14, 2, tzinfo=UTC)
FUSER_SHA256 = hashlib.sha256(b"specimen fuser").hexdigest()
EXAMINER = "specimen-examiner"
NAMES = {"bootstrap-admin": "Bootstrap administrator", EXAMINER: "R. Okafor", audit.SYSTEM_ACTOR: "Veritas"}


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _finding(fid: str, rank: int, family: str, strength: str, direction: str, title: str, sentence: str, more: str = "", *,
             hypothesis: str | None = None, dissent: str | None = None, evidence: tuple[str, ...] = ("decision",)) -> dict[str, Any]:
    return {"id": fid, "rank": rank, "family": family, "hypothesis": hypothesis, "strength": strength, "direction": direction,
            "review_priority": {"strong": "high", "moderate": "medium"}.get(strength, "low"), "title": title,
            "statement": f"{sentence} {more}".strip(), "sentences": [{"text": sentence, "evidence_ids": list(evidence)}],
            "evidence_ids": list(evidence), "dissent_note": dissent}


def _decision(code: str, *, p: float | None, mode: str, confidence: str, reason: str, ci: tuple[float, float] | None = None) -> dict[str, Any]:
    band = view.CODE_BAND.get(code, "uncertain")
    out: dict[str, Any] = {"code": code, "verbal": view.HEADLINE[band] if code != "not_assessable" else view.HEADLINE["insufficient"],
                           "band": band, "confidence": confidence, "reason": reason, "fusion_mode": mode, "ci90": ci}
    if mode == "artefact":
        out.update(probability_ai_calibrated=p, fuser_sha256=FUSER_SHA256)
    else:
        out.update(probability_ai_uncalibrated=p)
    return out


def _rates(modality: str, context: str, *, tpr: float, fpr: float, n_pos: int, n_neg: int) -> list[dict[str, Any]]:
    return [{"subject": f"fuser-{modality}-1.0.0", "method": "platt", "artefact_sha256": FUSER_SHA256, "bench_version": "vb-1.0",
             "error_rates": [{"context": context, "threshold_kind": "conformal_0.05", "threshold": 0.71, "tpr": tpr, "fpr": fpr,
                              "tpr_ci95": (round(tpr - 0.04, 2), round(tpr + 0.04, 2)), "fpr_ci95": (max(0.0, round(fpr - 0.02, 2)), round(fpr + 0.02, 2)),
                              "n_pos": n_pos, "n_neg": n_neg, "bench_version": "vb-1.0"}]}]


COVERAGE_CHECKS = "Checks that ran"
SPECIMENS: dict[str, dict[str, Any]] = {
    "image": {
        "analysis_id": "7f3a2c1e-4b1d-4e0a-9c55-2f1d3b8a6e10", "filename": "Q3_board_deck_photo.jpg", "content_type": "image/jpeg",
        "size_bytes": 2481223, "coverage": {"ok": 11}, "supplied_by": "J. Rivera, Finance", "source": "USB drive from the custodian",
        "decision": _decision("ai_supported_strong", p=0.91, mode="artefact", confidence="high", ci=(0.85, 0.95),
                              reason="Localisation, two classifiers and the metadata history point toward manipulation."),
        "condition_profile": {"width": 4032, "height": 3024, "jpeg_quality_est": 88, "min_face_px": 212, "median_face_px": 212,
                              "slice_id": "image/jpeg/q>=70/face", "within_validated_envelope": True, "limiting_conditions": []},
        "calibration": _rates("image", "portrait face", tpr=0.61, fpr=0.02, n_pos=480, n_neg=520),
        "findings": [
            _finding("F1", 1, "F4_localised_manipulation", "moderate", "for", "Moderate support for manipulation",
                     "A region around the left hand has pixel statistics that differ from the rest of the image.",
                     "Marked regions are candidates for editing, not confirmed edits.", hypothesis="H2_localised_manipulation"),
            _finding("F2", 2, "F8_metadata_consistency", "weak", "for", "Weak support for manipulation",
                     "The metadata names Adobe Photoshop 25.1 as the last software to save the file.", hypothesis="H2_localised_manipulation"),
            _finding("F3", 3, "F3_whole_image_generation", "weak", "for", "Weak support for AI generation",
                     "Two classifiers trained on different generators scored the whole image high.", hypothesis="H1_whole_content_synthesis"),
            _finding("F4", 4, "F10_face", "weak", "against", "Weak support against manipulation", "No signs of a swapped face were found.",
                     hypothesis="H2_localised_manipulation", dissent="That concerns a different hypothesis: a swap changes only the face."),
            _finding("F5", 5, "coverage", "none", "neutral", COVERAGE_CHECKS, "11 of 11 checks produced a result."),
        ],
    },
    "video": {
        "analysis_id": "3c9e41d0-8a52-4f6b-b1e7-5d20c4a9f3b8", "filename": "press_briefing_clip.mp4", "content_type": "video/mp4",
        "size_bytes": 18734112, "coverage": {"not_applicable": 1, "ok": 6}, "supplied_by": "Newsdesk", "source": "Social media repost",
        "decision": _decision("ai_supported", p=0.72, mode="fallback", confidence="medium",
                              reason="Lip-sync and face-track checks point toward manipulation between 00:12 and 00:19."),
        "condition_profile": {"width": 1280, "height": 720, "duration_s": 12.4, "n_frames": 372, "min_face_px": 150, "median_face_px": 180,
                              "speech_share": 0.8, "slice_id": "video/h264/720p/face", "within_validated_envelope": False,
                              "limiting_conditions": ["platform re-encode"]},
        "calibration": [],
        "findings": [
            _finding("F1", 1, "V2_lip_sync", "moderate", "for", "Moderate support for manipulation (lip-sync)",
                     "Lip movement and speech drift apart between 00:12 and 00:19.", hypothesis="H2_localised_manipulation"),
            _finding("F2", 2, "V1_face_track", "weak", "for", "Weak support for manipulation (face swap)",
                     "Face A scores slightly high in the same interval.", hypothesis="H2_localised_manipulation"),
            _finding("F3", 3, "V3_synthetic_video", "moderate", "against", "Moderate support against AI generation",
                     "Whole-scene checks found no signs of a fully generated video.", hypothesis="H1_whole_content_synthesis",
                     dissent="That does not contradict the lip-sync finding: a lip-sync edit changes only the mouth region."),
            _finding("F4", 4, "coverage", "none", "neutral", COVERAGE_CHECKS, "6 of 7 checks produced a result."),
        ],
    },
    "audio": {
        "analysis_id": "b41f06a2-2d7c-4c19-8e3a-96f0d5e2a741", "filename": "voicemail_0914.mp3", "content_type": "audio/mpeg",
        "size_bytes": 1504220, "coverage": {"failed": 1, "ok": 3}, "supplied_by": "Fraud desk", "source": "Customer voicemail export",
        "decision": _decision("no_ai_indicators", p=0.22, mode="artefact", confidence="high", ci=(0.14, 0.31),
                              reason="Voice windows scored low across the recording."),
        "condition_profile": {"duration_s": 94.0, "speech_share": 0.91, "slice_id": "audio/mp3/speech", "within_validated_envelope": True,
                              "limiting_conditions": ["lossy compression (mp3)"]},
        "calibration": _rates("audio", "audio lossy", tpr=0.83, fpr=0.09, n_pos=64, n_neg=71),
        "findings": [
            _finding("F1", 1, "A1_voice", "weak", "against", "Weak support against a cloned voice", "Voice windows scored low across the recording.",
                     hypothesis="H2_localised_manipulation"),
            _finding("F2", 2, "A1_voice", "weak", "for", "Weak support for a cloned voice",
                     "One window between 01:02 and 01:06 scored higher than the rest.", hypothesis="H2_localised_manipulation"),
            _finding("F3", 3, "coverage", "none", "neutral", COVERAGE_CHECKS, "3 of 4 checks produced a result."),
        ],
    },
    "document": {
        "analysis_id": "e8d2c7b4-61a9-4f30-a2d5-0b7c9e14f6d3", "filename": "statement_draft.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size_bytes": 18220,
        "coverage": {"not_applicable": 2}, "supplied_by": "Legal", "source": "Email attachment",
        "decision": _decision("not_assessable", p=None, mode="fallback", confidence="low",
                              reason="The document has 31 words; at least 40 are needed."),
        "condition_profile": {"slice_id": "document/short", "within_validated_envelope": False, "limiting_conditions": ["fewer than 40 words"]},
        "calibration": [],
        "findings": [_finding("F1", 1, "coverage", "none", "neutral", COVERAGE_CHECKS, "0 of 2 checks produced a result.")],
    },
}


# --- forensic extras (R-3) -------------------------------------------------------------------------------------------------------

def _signal(sid: str, display: str, name: str, family: str, category: str, modality: str, hypothesis: str | None, *, lane: str | None = None,
            contribution: float | None = None, calibrated: float | None = None, raw: float | None = None, log_lr: float | None = None,
            observed: str = "", reference: str = "", exhibits: tuple[str, ...] = (), fp_modes: tuple[str, ...] = (), status: str = "ok",
            reason: str | None = None, segments: tuple[dict[str, Any], ...] = ()) -> dict[str, Any]:
    detector = sid.split(":")[0]
    return {"id": sid, "display_id": display, "detector_id": detector, "detector_version": "1.0.0", "name": name, "family": family,
            "modality": modality, "category": category, "status": status, "status_reason": reason, "hypothesis": hypothesis, "lane": lane,
            "segments": list(segments), "fp_modes": list(fp_modes), "contribution_logit": contribution, "raw_score": raw,
            "raw_kind": "probability", "calibrated_probability": calibrated, "log_lr": log_lr, "observed": observed, "reference": reference,
            "exhibit_ids": list(exhibits)}


def _segment(display: str, start_s: float, end_s: float, channel: str, lane: str, peak: float, sources: tuple[str, ...],
             track: str | None = None) -> dict[str, Any]:
    return {"display_id": display, "start_pts": round(start_s * 30000), "end_pts": round(end_s * 30000), "time_base": "1/30000",
            "start_s": start_s, "end_s": end_s, "channel": channel, "track_id": track, "peak": peak, "lane": lane,
            "hypothesis": "H2_localised_manipulation", "sources": list(sources)}


def _contributions(p: float, rows: list[tuple[str | None, str, str, float]]) -> list[dict[str, Any]]:
    """The starting row (D-63: every check at its neutral point; balancing, 6 dp) then the rows, with percentage-point effects
    along the running total."""
    start = round(math.log(p / (1 - p)) - math.fsum(r[3] for r in rows), 6)
    out, running = [], start
    out.append({"signal_id": None, "label": "every check at its neutral point", "kind": "neutral_start", "logit": start, "pp": 0.0})
    for signal_id, label, kind, logit in rows:
        before = 1 / (1 + math.exp(-running))
        running += logit
        out.append({"signal_id": signal_id, "label": label, "kind": kind, "logit": logit, "pp": round((1 / (1 + math.exp(-running)) - before) * 100, 2),
                    **({"baseline": "neutral_point"} if kind == "detector" else {})})
    return out


def _detector(ident: str, *, licence: str = "Apache-2.0", weights: bool = True) -> dict[str, Any]:
    return {"id": ident, "version": "1.0.0", "admission": {"decision": "pending", "date": None},
            "weights_sha256": _h(f"weights {ident}") if weights else None, "licence": licence,
            "coverage_date": "2025-06", "device": "mps" if weights else "cpu", "precision": "fp16" if weights else "fp32"}


def _card(ident: str, name: str, fp_modes: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"id": ident, "name": name, "version": "1.0.0", "citation": "Specimen publication (fixture)", "source_url": None,
            "coverage_date": "2025-06", "admission": "pending", "admission_decision": "pending", "admission_date": None,
            "eval": [], "fp_modes": list(fp_modes), "known_limitations": [],
            "licences": [{"model": f"{ident}-weights", "spdx": "Apache-2.0", "class": "commercial"}]}


IMG = "image"
LOC, CLS_A, CLS_B, FACE, META = "specimen.localiser:global", "specimen.classifier_a:global", "specimen.classifier_b:global", "specimen.face:face:0", "specimen.metadata:global"
ELA, NOISE, COPY, FREQ, C2PA, MARK = ("specimen.ela:global", "specimen.noise:global", "specimen.copymove:global", "specimen.spectrum:global",
                                      "specimen.c2pa:global", "specimen.watermark:global")
SWAP, SYNC, SYNTH, CODEC, VOICE = "specimen.swap:track:0", "specimen.lipsync:track:0", "specimen.synthesis:global", "specimen.codec:global", "specimen.voice:global"
SEG_A = _segment("Seg. a", 12.0, 19.3, "visual", "face_swap", 0.66, (SWAP,), track="track:0")
SEG_B = _segment("Seg. b", 12.4, 19.0, "av", "lip_sync", 0.7, (SYNC,), track="track:0")
SEG_V = _segment("Seg. a", 62.0, 66.0, "audio", "voice", 0.61, ("specimen.voice_windows:global",))
H1, H2, H3 = "H1_whole_content_synthesis", "H2_localised_manipulation", "H3_provenance_integrity"

EXTRAS: dict[str, dict[str, Any]] = {
    "image": {
        "coverage": {"not_applicable": 1, "ok": 10},
        "signals": [
            _signal(LOC, "S01", "Localisation (specimen)", "localisation", "forensic", IMG, H2, lane="partial_edit", contribution=2.4, calibrated=0.87,
                    log_lr=2.1, observed="integrity score 0.91", reference="unaltered test images mostly below 0.30", exhibits=("ex-heat", "ex-regions"),
                    fp_modes=("heavy HDR tone-mapping", "text overlays and screenshots")),
            _signal(CLS_A, "S02", "Whole-image classifier A (specimen)", "learned-whole-image", "model", IMG, H1, contribution=0.9, calibrated=0.83,
                    log_lr=1.6, observed="0.88", reference="unaltered test images mostly below 0.40"),
            _signal(CLS_B, "S03", "Whole-image classifier B (specimen)", "learned-whole-image", "model", IMG, H1, contribution=0.5, calibrated=0.76,
                    log_lr=1.2, observed="0.79", reference="unaltered test images mostly below 0.40"),
            _signal(FACE, "S04", "Face check (specimen)", "face-swap", "face", IMG, H2, lane="face_swap", contribution=-0.75, calibrated=0.21,
                    log_lr=-0.9, observed="0.18 on face 1", reference="swapped test faces mostly above 0.60"),
            _signal(META, "S05", "Metadata history (specimen)", "metadata", "metadata", IMG, H2, contribution=0.7, log_lr=0.8,
                    observed="Software=Adobe Photoshop 25.1", reference="camera originals name the device firmware", exhibits=("ex-meta",)),
            _signal(ELA, "S06", "Error level (specimen)", "filter", "forensic", IMG, H2, contribution=0.12, raw=0.64, observed="mean difference 0.64",
                    reference="0.20 to 0.70 on re-saved photographs"),
            _signal(NOISE, "S07", "Noise residual (specimen)", "filter", "forensic", IMG, H2, contribution=0.08, raw=0.58, observed="0.58",
                    reference="0.30 to 0.60 on camera photographs"),
            _signal(COPY, "S08", "Copy-move (specimen)", "filter", "forensic", IMG, H2, contribution=-0.03, raw=0.1, observed="no matched blocks",
                    reference="matched blocks on duplicated regions"),
            _signal(FREQ, "S09", "Frequency spectrum (specimen)", "filter", "forensic", IMG, H1, contribution=0.04, raw=0.55, observed="no periodic peaks",
                    reference="periodic peaks on some generators"),
            _signal(C2PA, "S10", "Content Credentials (specimen)", "provenance", "provenance", IMG, H3, observed="absent", reference="most files carry none"),
            _signal(MARK, "S11", "Watermark decoder (specimen)", "provenance", "provenance", IMG, H3, status="not_applicable",
                    reason="no decoder for this format"),
        ],
        "contributions": [(LOC, "Localisation (specimen)", "detector", 2.4), (CLS_A, "Whole-image classifier A (specimen)", "detector", 0.9),
                          (CLS_B, "Whole-image classifier B (specimen)", "detector", 0.5), (FACE, "Face check (specimen)", "detector", -0.75),
                          (META, "Metadata history (specimen)", "detector", 0.7), (ELA, "Error level (specimen)", "detector", 0.12),
                          (NOISE, "Noise residual (specimen)", "detector", 0.08), (COPY, "Copy-move (specimen)", "detector", -0.03),
                          (FREQ, "Frequency spectrum (specimen)", "detector", 0.04), (None, "missing: attribution (specimen)", "missing", 0.1)],
        "exhibits": [
            {"id": "ex-heat", "display_id": "Fig. 1", "kind": "heatmap", "title": "Anomaly map (specimen localiser)", "source_signal_ids": [LOC],
             "created_by": "specimen.localiser", "media_ref": "original", "coordinate_space": {"width": 64, "height": 64, "scale_to_original": 63.0},
             "sha256": _h("specimen heatmap"), "semantics": "p_manipulated", "calibrated": False, "grid": (64, 64), "value_range": (0.0, 1.0),
             "payload_artefact_id": "specimen-payload", "reliability_mask_artefact_id": "specimen-mask", "preview_artefact_id": "specimen-preview"},
            {"id": "ex-regions", "display_id": "Fig. 2", "kind": "regions", "title": "Regions segmented from the anomaly map", "source_signal_ids": [LOC],
             "created_by": "specimen.localiser", "media_ref": "original", "sha256": _h("specimen regions"),
             "regions": [{"display_id": "R1", "region": {"kind": "bbox", "bbox": {"x": 1210, "y": 1830, "w": 520, "h": 610}}, "probability": 0.87, "label": "left hand"},
                         {"display_id": "R2", "region": {"kind": "bbox", "bbox": {"x": 2900, "y": 400, "w": 180, "h": 150}}, "score": 0.58, "label": "mostly unreliable"}]},
            {"id": "ex-meta", "display_id": "Fig. 3", "kind": "table", "title": "Metadata fields (specimen)", "source_signal_ids": [META],
             "created_by": "specimen.metadata", "media_ref": "original", "sha256": _h("specimen metadata"), "columns": ["Field", "Value"],
             "rows": [["Make", "Apple"], ["Model", "iPhone 15 Pro"], ["Software", "Adobe Photoshop 25.1"], ["DateTimeOriginal", "2026:08:21 13:58:02"]]},
        ],
        "evidence": {"F1": (LOC, "ex-heat"), "F2": (META, "ex-meta"), "F3": (CLS_A, CLS_B), "F4": (FACE,)},
        "assessments": [{"hypothesis": H1, "applicable": True, "probability": 0.62, "ci90": (0.5, 0.72)},
                        {"hypothesis": H2, "applicable": True, "probability": 0.91},
                        {"hypothesis": H3, "applicable": True, "c2pa_state": "absent"}],
        "detectors": ["specimen.localiser", "specimen.classifier_a", "specimen.classifier_b", "specimen.face", "specimen.metadata", "specimen.ela",
                      "specimen.noise", "specimen.copymove", "specimen.spectrum", "specimen.c2pa", "specimen.watermark"],
        "caveats": ["The image was saved at least twice, so pixel-level checks are weaker than on a camera original."],
        "review": "reviewed",
    },
    "video": {
        "coverage": {"failed": 1, "ok": 4},
        "condition_profile": {"width": 1280, "height": 720, "duration_s": 24.0, "n_frames": 720, "min_face_px": 150, "median_face_px": 180,
                              "speech_share": 0.8, "slice_id": "video/h264/720p/face", "within_validated_envelope": False,
                              "limiting_conditions": ["platform re-encode"]},
        "signals": [
            _signal(SWAP, "S01", "Face-swap track scorer (specimen)", "face-swap", "face", "video", H2, lane="face_swap", raw=0.64,
                    observed="top-30 % window mean 0.64", reference="unaltered test tracks mostly below 0.45", exhibits=("ex-track", "ex-swap-series"),
                    segments=(SEG_A,)),
            _signal(SYNC, "S02", "Lip-sync (specimen)", "lip-sync", "temporal", "video", H2, lane="lip_sync", raw=0.7,
                    observed="offset step of +180 ms at 00:12.4", reference="offset within ±80 ms on unaltered speech", exhibits=("ex-sync",), segments=(SEG_B,),
                    fp_modes=("voice-over and dubbing", "occluded mouth")),
            _signal(SYNTH, "S03", "Whole-frame synthesis (specimen)", "learned-whole-image", "model", "video", H1, raw=0.21, observed="median 0.21",
                    reference="generated test clips mostly above 0.60"),
            _signal(CODEC, "S04", "Codec history (specimen)", "codec", "forensic", "video", None, observed="2 encodes: camera H.264, platform re-encode",
                    reference="a camera original has 1 encode"),
            _signal(VOICE, "S05", "Voice windows (specimen)", "voice", "model", "video", H2, lane="voice", status="failed",
                    reason="the audio decoder raised an error"),
        ],
        "exhibits": [
            {"id": "ex-track", "display_id": "Fig. 1", "kind": "face_track", "title": "Face A", "source_signal_ids": [SWAP], "created_by": "specimen.tracker",
             "media_ref": "frame", "sha256": _h("specimen track"), "track": {"track_id": "track:0", "first_pts": 0, "last_pts": 719000, "n_frames": 720, "identity": "A"},
             "frames": [{"pts": 0, "frame_index": 0, "bbox": {"x": 540, "y": 120, "w": 180, "h": 180}},
                        {"pts": 360000, "frame_index": 360, "bbox": {"x": 548, "y": 126, "w": 176, "h": 176}},
                        {"pts": 719000, "frame_index": 719, "bbox": {"x": 552, "y": 130, "w": 172, "h": 172}}]},
            {"id": "ex-swap-series", "display_id": "Fig. 2", "kind": "series", "title": "Face-swap score by window (Face A)", "source_signal_ids": [SWAP],
             "created_by": "specimen.swap", "media_ref": "frame", "sha256": _h("specimen series"), "x_unit": "s", "thresholds": [0.5],
             "series": [{"name": "face swap, Face A", "track_id": "track:0",
                         "points": [(0, 0.2), (4, 0.24), (8, 0.3), (12, 0.6), (14, 0.66), (16, 0.64), (18, 0.62), (20, 0.34), (24, 0.25)]}]},
            {"id": "ex-sync", "display_id": "Fig. 3", "kind": "segments", "title": "Lip-sync segments", "source_signal_ids": [SYNC], "created_by": "specimen.lipsync",
             "media_ref": "frame", "sha256": _h("specimen segments"), "lane": "lip_sync", "segments": [SEG_B]},
        ],
        "evidence": {"F1": (SYNC, "ex-sync"), "F2": (SWAP, "ex-track"), "F3": (SYNTH,)},
        "assessments": [{"hypothesis": H1, "applicable": True, "probability": 0.21},
                        {"hypothesis": H2, "applicable": True, "probability": 0.72,
                         "lanes": [{"lane": "face_swap", "probability": 0.58, "band": "uncertain", "track_id": "track:0", "segments": [SEG_A]},
                                   {"lane": "lip_sync", "probability": 0.71, "band": "likely_ai", "track_id": "track:0", "segments": [SEG_B]}]},
                        {"hypothesis": H3, "applicable": True, "c2pa_state": "absent"}],
        "detectors": ["specimen.swap", "specimen.lipsync", "specimen.synthesis", "specimen.codec", "specimen.voice", "specimen.tracker"],
        "caveats": [],
        "review": "machine",
    },
    "audio": {
        "signals": [
            _signal("specimen.voice_windows:global", "S01", "Voice windows (specimen)", "voice", "model", "audio", H2, lane="voice", contribution=-0.9,
                    calibrated=0.24, log_lr=-1.1, observed="median window score 0.19", reference="cloned test voices mostly above 0.55", exhibits=("ex-voice",),
                    segments=(SEG_V,)),
            _signal("specimen.gate:global", "S02", "Speech gate (specimen)", "content-gate", "statistical", "audio", None, observed="speech 91 % of 94.0 s",
                    reference="at least 1 s of speech"),
            _signal("specimen.codec_audio:global", "S03", "Codec (specimen)", "codec", "forensic", "audio", H2, contribution=-0.12, raw=0.3,
                    observed="1 encode, MP3 128 kbit/s", reference="re-encoded exports show 2 or more"),
            _signal("specimen.spectral:global", "S04", "Spectral check (specimen)", "filter", "forensic", "audio", H2, status="failed",
                    reason="the check timed out after 60 s"),
        ],
        "contributions": [("specimen.voice_windows:global", "Voice windows (specimen)", "detector", -0.9),
                          ("specimen.codec_audio:global", "Codec (specimen)", "detector", -0.12)],
        "exhibits": [{"id": "ex-voice", "display_id": "Fig. 1", "kind": "series", "title": "Voice window scores", "source_signal_ids": ["specimen.voice_windows:global"],
                      "created_by": "specimen.voice_windows", "media_ref": "audio", "sha256": _h("specimen voice"), "x_unit": "s", "thresholds": [0.5],
                      "series": [{"name": "voice", "points": [(0, 0.2), (10, 0.18), (20, 0.22), (40, 0.19), (62, 0.61), (66, 0.3), (80, 0.2), (94, 0.21)]}]}],
        "evidence": {"F1": ("specimen.voice_windows:global",), "F2": ("specimen.spectral:global", "ex-voice")},
        "assessments": [{"hypothesis": H2, "applicable": True, "probability": 0.22,
                         "lanes": [{"lane": "voice", "probability": 0.24, "band": "likely_authentic", "segments": [SEG_V]}]}],
        "detectors": ["specimen.voice_windows", "specimen.gate", "specimen.codec_audio", "specimen.spectral"],
        "caveats": [],
        "review": "machine",
    },
    "document": {
        "signals": [
            _signal("specimen.text_classifier:global", "S01", "AI-text classifier (specimen)", "learned-text", "model", "image", H1, status="not_applicable",
                    reason="fewer than 40 words"),
            _signal("specimen.zero_shot:global", "S02", "Zero-shot text check (specimen)", "zero-shot-text", "statistical", "image", H1,
                    status="not_applicable", reason="fewer than 40 words"),
        ],
        "exhibits": [],
        "evidence": {},
        "assessments": [{"hypothesis": H1, "applicable": False, "note": "Not addressed: the document is too short for the text checks."}],
        "detectors": ["specimen.text_classifier", "specimen.zero_shot"],
        "caveats": [],
        "review": "machine",
    },
}


def result(kind: str, *, full: bool = False) -> dict[str, Any]:
    spec = SPECIMENS[kind]
    doc = {"analysis_id": spec["analysis_id"], "kind": kind, "manifest": {"veritas_version": VERSION, "pipeline_version": f"{kind}-specimen", "profile": "full"},
           "input": {"sha256": sha256(kind), "size_bytes": spec["size_bytes"], "mime_sniffed": spec["content_type"], "filename": spec["filename"]},
           "condition_profile": {**spec["condition_profile"],
                                 "validation_support": "direct" if spec["condition_profile"]["within_validated_envelope"] else "unsupported"},
           "decision": spec["decision"], "findings": spec["findings"],
           "calibration": spec["calibration"]}
    if full:
        extra = EXTRAS[kind]
        doc = copy.deepcopy(doc)
        if kind == "document":
            doc["signals"] = [{**s, "modality": "image"} for s in extra["signals"]]  # the evidence schema has no document modality
        else:
            doc["signals"] = extra["signals"]
        doc.update(exhibits=extra["exhibits"], assessments=extra["assessments"], caveats=extra["caveats"],
                   condition_profile=extra.get("condition_profile", doc["condition_profile"]))
        if extra.get("contributions"):
            doc["decision"] = {**doc["decision"], "contributions": _contributions(float(doc["decision"]["probability_ai_calibrated"]), extra["contributions"])}
        doc["manifest"].update(detectors=[_detector(i, weights=i not in ("specimen.metadata", "specimen.c2pa", "specimen.codec", "specimen.gate"))
                                          for i in extra["detectors"]], git_commit=_h("specimen commit")[:40], config_sha256=_h("specimen config"),
                               device="mps", precision="fp16", template_set=f"{kind}-specimen-templates", calibration_id=FUSER_SHA256,
                               licence_posture="commercial", library_versions={"python": "3.12", "specimen": "1.0.0"}, sampling={"seed": 7})
        for f in doc["findings"]:
            if f["id"] in extra["evidence"]:
                f["evidence_ids"] = list(extra["evidence"][f["id"]])
                f["sentences"] = [{**s, "evidence_ids": list(extra["evidence"][f["id"]])} for s in f["sentences"]]
        # L-3: a finding carries the contribution of the signals it cites, as the synthesis stage records it
        by_signal: dict[str, float] = {}
        for row in doc["decision"].get("contributions") or []:
            if row.get("signal_id"):
                by_signal[row["signal_id"]] = by_signal.get(row["signal_id"], 0.0) + float(row["logit"])
        for f in doc["findings"]:
            cited = [by_signal[e] for e in f["evidence_ids"] if e in by_signal]
            if cited:
                f["contribution_logit"] = round(math.fsum(cited), 6)
    return AnalysisResultV2.model_validate(doc).model_dump(mode="json")


def sha256(kind: str) -> str:
    return hashlib.sha256(f"veritas specimen {kind}".encode()).hexdigest()


def _custody(kind: str, count: int = 6) -> list[dict[str, Any]]:
    """A custody extract whose events carry recomputable hashes (seq from 41; the first `prev_hash` names an earlier event)."""
    aid = SPECIMENS[kind]["analysis_id"]
    actions = ("intake", "hash_verified", "analysis_started", "analysis_finished", "conclusion_recorded", "signoff")
    events, prev = [], _h("specimen earlier event")
    for i in range(count):
        signoff = actions[i % len(actions)] == "signoff"  # names the specimen sign-off and how its signer authenticated (D-61)
        body = {"seq": 41 + i, "ts": (RECEIVED_AT + timedelta(minutes=3 * i)).strftime(audit.TS_FORMAT),
                "actor_id": audit.SYSTEM_ACTOR if actions[i % len(actions)] in ("hash_verified", "analysis_started", "analysis_finished") else EXAMINER,
                "action": actions[i % len(actions)], "case_id": "case-specimen", "media_id": f"media-{kind}", "analysis_id": aid, "report_id": None,
                "object_ids": ["signoff-specimen", "conclusion-specimen"] if signoff else [], "before_hash": None,
                "after_hash": sha256(kind) if i < 2 else None,
                "details": {"kind": "examiner", "decision": "concur", "auth": "session"} if signoff else {}}
        event = {**body, "prev_hash": prev, "event_hash": audit.event_hash(prev, body)}
        events.append(event)
        prev = event["event_hash"]
    return events


def specimen(kind: str, profile: str = "summary") -> ReportSnapshot:
    spec = SPECIMENS[kind]
    aid = spec["analysis_id"]
    full = profile in FULL_PROFILES
    number = (KINDS.index(kind) + 1) * {"newsroom": 10, "forensic": 100, "standard": 1000}.get(profile, 1)
    extra = EXTRAS[kind] if full else {}
    by_status = dict(sorted(extra.get("coverage", spec["coverage"]).items()))
    state = extra.get("review", "machine")
    revision = 2 if state == "reviewed" else 1
    conclusion, signoffs, annotations = None, [], []
    if state == "reviewed":
        conclusion = SnapshotConclusion(id="conclusion-specimen", author_id=EXAMINER, author_name=NAMES[EXAMINER], scale_position="ai_supported_strong",
                                        machine_code="ai_supported_strong", deviation_steps=0,
                                        rationale="Localisation, two classifiers and the metadata history agree; the face result concerns a different "
                                                  "hypothesis. The original file was requested from the custodian.",
                                        limitations_acknowledged=["L-01"], next_steps="Compare with the original file once the custodian supplies it.",
                                        declaration_text_id=None, deviation_justification=None, created_at=datetime(2026, 9, 11, 13, 50, tzinfo=UTC),
                                        signature_valid=True)
        signoffs = [SnapshotSignoff(id="signoff-specimen", kind="examiner", decision="concur", principal_id=EXAMINER, principal_name=NAMES[EXAMINER],
                                    comment=None, signed_at=datetime(2026, 9, 11, 13, 58, tzinfo=UTC), signature_valid=True)]
    if full and kind == "image":
        annotations = [
            SnapshotAnnotation(id="ann-1", author_id=EXAMINER, target_type="finding", target_id="F1", shape="none", geometry=None, examiner_status="accepted",
                               body="Consistent with the custodian's account of an edited slide photo.", visibility="in_report",
                               created_at=datetime(2026, 9, 11, 13, 20, tzinfo=UTC), edited_at=None, revision=1),
            SnapshotAnnotation(id="ann-2", author_id=EXAMINER, target_type="exhibit", target_id="ex-heat", shape="box",
                               geometry={"x": 1210.0, "y": 1830.0, "w": 520.0, "h": 610.0}, examiner_status=None,
                               body="Hand region; compare with the source slide.", visibility="in_report",
                               created_at=datetime(2026, 9, 11, 13, 25, tzinfo=UTC), edited_at=None, revision=1),
            SnapshotAnnotation(id="ann-3", author_id=EXAMINER, target_type="finding", target_id="F4", shape="none", geometry=None, examiner_status="needs_check",
                               body="Check the face against the second photograph.", visibility="internal",
                               created_at=datetime(2026, 9, 11, 13, 30, tzinfo=UTC), edited_at=None, revision=1),
        ]
    if full and kind == "video":
        annotations = [SnapshotAnnotation(id="ann-v1", author_id=EXAMINER, target_type="interval", target_id=None, shape="interval",
                                          geometry={"start_pts": 360000, "end_pts": 579000, "time_base": "1/30000"}, examiner_status=None,
                                          body="Speaker turns away at 00:15; compare with the second camera.", visibility="in_report",
                                          created_at=datetime(2026, 9, 11, 13, 40, tzinfo=UTC), edited_at=None, revision=1)]
    revisions = ([{"id": f"VR-2026-{number - 1:06d}", "display_id": display_id(aid, 1, False), "revision": 1, "requested_at": "2026-09-11T10:30:00Z",
                   "review_state": "machine", "revision_reason": None, "status": "done", "sha256": _h(f"specimen report {kind} r1")}]
                 if full and revision == 2 else [])
    return ReportSnapshot(
        report=SnapshotReport(id=f"VR-2026-{number:06d}", display_id=display_id(aid, revision, state == "reviewed"), revision=revision, profile=profile,  # type: ignore[arg-type]
                              redacted=profile == "newsroom", supersedes_report_id=revisions[0]["id"] if revisions else None,
                              revision_reason="Signed off by the examiner." if revisions else None, requested_at=REQUESTED_AT,
                              requested_by="bootstrap-admin", requested_by_name="Bootstrap administrator", template_set=engine.TEMPLATE_SET,
                              veritas_version=VERSION),
        analysis=SnapshotAnalysis(id=aid, kind=kind, revision=1, profile="full", supersedes_id=None, created_at=CREATED_AT, finished_at=FINISHED_AT,
                                  coverage=SnapshotCoverage(detectors_total=sum(by_status.values()), by_status=by_status)),
        media=SnapshotMedia(filename=spec["filename"], content_type=spec["content_type"], size_bytes=spec["size_bytes"], sha256=sha256(kind),
                            md5=hashlib.md5(f"veritas specimen {kind}".encode(), usedforsecurity=False).hexdigest(), received_at=RECEIVED_AT,
                            acquisition_method="upload", supplied_by=spec["supplied_by"], source=spec["source"],
                            date_acquired=ACQUIRED_AT, intake_notes="Handed over at reception in a sealed bag; the seal was intact."),
        case=SnapshotCase(id="case-specimen", reference="VX-2026-014", title="Vendor payment fraud", classification="restricted", exhibit_label="B"),
        result=result(kind, full=full),
        review=SnapshotReview(state=state, issuable=state == "reviewed", requires_second_review=False, annotations_seq=len(annotations),
                              conclusion=conclusion, signoffs=signoffs),
        annotations=annotations,
        custody=_custody(kind) if full else [],
        # N-19k: a figure names the check that drew it; the tracker of the video specimen has no signal to take a name from
        detector_names={"specimen.tracker": "Face tracker (specimen)"} if full else {},
        model_cards=[_card(i, f"{i.removeprefix('specimen.').replace('_', ' ').capitalize()} (specimen)") for i in EXTRAS[kind]["detectors"]] if full else [],
        revisions=revisions,
        actor_names=dict(NAMES) if full else {},
        custody_verification={"valid": True, "checked": 46, "first_broken_seq": None, "reason": None} if full else None,
    )


def stress(n_signals: int = 90, n_events: int = 160, n_notes: int = 70) -> ReportSnapshot:
    """The image forensic specimen with many signals, custody events and notes: more than 30 pages (RP-03 timing, R-3 ≤ 10 s)."""
    base = specimen("image", "forensic")
    doc = copy.deepcopy(base.result)
    for i in range(n_signals):
        doc["signals"].append(_signal(f"specimen.extra_{i:03d}:global", f"S{12 + i:02d}", f"Extra check {i + 1} (specimen)", "filter", "forensic", IMG, H2,
                                      raw=0.4 + (i % 20) / 100, observed=f"score {0.4 + (i % 20) / 100:.2f}", reference="0.30 to 0.60 on camera photographs",
                                      fp_modes=("flat backgrounds",)))
        doc["manifest"]["detectors"].append(_detector(f"specimen.extra_{i:03d}"))
    validated = AnalysisResultV2.model_validate(doc).model_dump(mode="json")
    events = _custody("image", n_events)
    notes = [SnapshotAnnotation(id=f"ann-x{i}", author_id=EXAMINER, target_type="exhibit", target_id="ex-heat", shape="box",
                                geometry={"x": 100.0 + i, "y": 200.0, "w": 40.0, "h": 40.0}, examiner_status=None,
                                body=f"Region {i + 1} checked against the source slide.", visibility="in_report",
                                created_at=datetime(2026, 9, 11, 13, 0, tzinfo=UTC) + timedelta(seconds=i), edited_at=None, revision=1)
             for i in range(n_notes)]
    return base.model_copy(update={"result": validated, "custody": events, "annotations": [*base.annotations, *notes],
                                   "model_cards": [*base.model_cards, *(_card(f"specimen.extra_{i:03d}", f"Extra check {i + 1} (specimen)") for i in range(n_signals))]})
