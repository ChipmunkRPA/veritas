"""Stage helpers shared by the image, document and video pipelines (ARCHITECTURE §3.1–3.2).

- **Detector plumbing.** `detector_entry` turns a `DetectorResult` into the resume-safe JSON entry a stage output
  carries. `DbRunStore` persists `detector_runs` rows. `run_detectors` fans out the enabled detectors of one or more
  modalities.
- **Media.** `media(ctx)` is the store-backed handle for the analysis kind. `media_path(ctx)` is a local file for
  tools that need a path (ffprobe, OpenCV, PyMuPDF): the upload when it survives, else a copy of the stored original.
- **Legacy view.** `legacy_signal_v2` maps a v0.3 aggregate signal (documents and video, where v0.3 signals are
  derived from several detectors) onto `SignalV2`, keeping the v0.3 id in `legacy.v1_id` so the v1 view is unchanged.
  `fallback_decision` builds the uncalibrated fallback `Decision` (§4.4) from a v0.3 verdict.
- **Publish.** `build_result` and `publish` store the canonical v2 result, project the evidence tables and write the v1
  view the Folio UI and the v1 PDF read.
"""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import effect
from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.core.logging import get_logger
from app.engine import build_info, card_copy, profiles, rerun, synthesis
from app.engine.boundary import NOT_STORED, Cancelled, DetectorRunRecord, run_detector, run_device, store_run
from app.engine.detector import (
    ON_THIS_SERVER,
    SERVER_FEWER,
    UNNAMED_CHECK,
    DetectorResult,
    DetectorSpec,
    Modality,
    Profile,
    RunContext,
    StageOutputs,
    UnitScore,
    of_the_server,
)
from app.engine.events import emit
from app.engine.fusion import abstain, decisive
from app.engine.fusion import runtime as fusion_runtime
from app.engine.fusion.features import face_gated
from app.engine.fusion.registry import Selection, candidates, record_fuser, release_bench, select_for_registry
from app.engine.media import StoredMedia
from app.engine.runner import StageContext
from app.schemas.evidence import AnalysisResultV2, Decision, SignalV2
from app.schemas.legacy import (
    BAND_CODE,
    BAND_HEADLINE,
    FALLBACK_CAVEAT,
    H3_VALUE,
    V1_NOTE,
    fallback_cap,
    fallback_reason,
    uncalibrated_score,
    v1_label,
)
from app.storage.artefacts import (
    ensure_original_in_store,
    index_object,
    persist_result,
    register_artefact,
    verify_artefact,
)
from app.storage.db import Analysis, AnalysisStage, DetectorRun, MediaObject
from app.storage.files import safe_filename, untrusted_strings, untrusted_text
from app.storage.objects import ObjectStore

log = get_logger(__name__)
MAX_UNIT_SCORES = 500  # per detector entry; documents raise it to cover every sentence
FACE_KINDS = ("image", "video")  # the kinds whose pipeline has a `faces` stage
LEGACY_ASSESSMENTS = [{"hypothesis": "H1_whole_content_synthesis", "applicable": False, "note": V1_NOTE}]


# --- detector plumbing -------------------------------------------------------------------------------------

def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def unit_score_json(score: UnitScore | None) -> dict[str, Any] | None:
    if score is None or _finite(score.raw) is None:
        return None
    return {"unit_ref": score.unit_ref, "raw": float(score.raw), "raw_kind": score.raw_kind,
            "quality": {k: float(v) for k, v in score.quality.items() if _finite(v) is not None}}


def detector_entry(spec: DetectorSpec, result: DetectorResult, run_id: str | None, *,
                   max_units: int = MAX_UNIT_SCORES) -> dict[str, Any]:
    aggregate = unit_score_json(result.aggregate)
    status, reason = result.status, result.reason
    if status == "ok" and result.aggregate is not None and aggregate is None:
        status, reason = "failed", "the detector returned a non-finite score"
    elif status == "ok" and aggregate is not None and aggregate["raw_kind"] == "probability" and not 0.0 <= aggregate["raw"] <= 1.0:
        status, reason, aggregate = "failed", "the detector returned an unusable score", None  # D-49
    return {
        "detector_id": spec.id, "version": spec.version, "name": spec.name, "family": spec.family,
        "modality": spec.modality.value, "hypotheses": [h.value for h in spec.hypotheses], "run_id": run_id,
        "status": status, "reason": reason, "aggregate": aggregate,
        "unit_scores": [s for s in (unit_score_json(u) for u in result.unit_scores[:max_units]) if s is not None],
        "observed": result.observed, "reference": result.reference, "reasoning": result.reasoning,
        "metrics": {k: float(v) for k, v in result.metrics.items() if _finite(v) is not None},
        "details": json_safe(result.details),  # N-19n: what the check found and where, for the finding's sentences
        "exhibit_ids": list(result.exhibit_ids), "model_revisions": dict(result.model_revisions),
        "device": result.device or None, "dtype": result.dtype or None, "provenance_override": result.provenance_override,
    }


def signal_id(detector_id: str, aggregate: dict[str, Any] | None) -> str:
    return f"{detector_id}:{aggregate['unit_ref'] if aggregate else 'global'}"


class DbRunStore:
    """Persists `detector_runs` rows with a raw-output artefact, and emits `detector_finished` (no scores)."""

    def __init__(self, sessions: Any, store: ObjectStore):
        self.sessions = sessions
        self.store = store
        self._ids: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def persist(self, record: DetectorRunRecord) -> None:
        result = record.result
        entry = detector_entry(_SpecView(record), result, None)
        run_id = str(uuid.uuid4())
        with self.sessions() as db:
            raw = register_artefact(db, self.store, canonical_json(entry), media_type="application/json", role="raw_output",
                                    created_by=record.detector_id, analysis_id=record.analysis_id)
            db.add(DetectorRun(id=run_id, analysis_id=record.analysis_id, detector_id=record.detector_id,
                               detector_version=record.detector_version, model_revisions=dict(result.model_revisions),
                               device=record.device or None, dtype=record.dtype or None, status=entry["status"],
                               reason=entry["reason"], started_at=datetime.fromisoformat(record.started_at),
                               duration_ms=round(record.duration_ms), peak_memory_mb=result.peak_memory_mb,
                               raw_output_artefact_id=raw.id))
            db.commit()
        with self._lock:
            self._ids[(record.analysis_id, record.detector_id)] = run_id
        payload = {"detector_id": record.detector_id, "status": entry["status"], "reason": entry["reason"],
                   "duration_ms": record.duration_ms}
        if entry["status"] == "ok":
            payload["signal_id"] = signal_id(record.detector_id, entry["aggregate"])
        try:
            emit(self.sessions, record.analysis_id, "detector_finished", payload)
        except Exception as e:
            # the run is stored; its progress event is not what the record rests on, and an error here would have the
            # boundary store the run a second time (N-10a)
            log.warning("detector.finished_event_failed", analysis_id=record.analysis_id, detector_id=record.detector_id,
                        error=f"{type(e).__name__}: {e}"[:300])

    def run_id(self, analysis_id: str, detector_id: str) -> str | None:
        with self._lock:
            return self._ids.get((analysis_id, detector_id))


@dataclasses.dataclass(frozen=True)
class _SpecView:
    """The subset of a spec the raw-output record needs (the record carries id and version only)."""

    record: DetectorRunRecord

    @property
    def id(self) -> str:
        return self.record.detector_id

    @property
    def version(self) -> str:
        return self.record.detector_version

    name = family = ""
    hypotheses: tuple = ()

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE


def media(ctx: StageContext) -> StoredMedia:
    lock = ctx.state.setdefault("media_lock", threading.Lock())
    with lock:
        handle = ctx.state.get("media")
        if handle is None:
            ingest = ctx.outputs["ingest"]
            upload = settings.uploads_dir / ctx.analysis_id / safe_filename(ingest["filename"])
            handle = StoredMedia(ingest["sha256"], ctx.state.get("kind", "image"), ctx.store,
                                 path=upload if upload.is_file() else None)
            ctx.state["media"] = handle
        return handle


def media_path(ctx: StageContext) -> Path:
    """A local file with the original bytes and the upload's suffix. Temporary copies are removed by `cleanup`."""
    handle = media(ctx)
    if handle.path is not None and handle.path.is_file():
        return handle.path
    lock = ctx.state.setdefault("media_lock", threading.Lock())
    with lock:
        copy = ctx.state.get("media_copy")
        if copy is None:
            filename = safe_filename(ctx.outputs["ingest"]["filename"])
            tmp = settings.data_dir / "tmp" / ctx.analysis_id
            tmp.mkdir(parents=True, exist_ok=True)
            copy = tmp / filename
            copy.write_bytes(bytes(handle.original_bytes()))
            ctx.state["media_copy"] = copy
        return copy


def cleanup(state: dict[str, Any], *, keep_copy: bool = False) -> None:
    """Close the media handle and remove the run's copy of the original. `keep_copy`: the job lease was lost, and the new
    owner's run reads the same `data/tmp/<analysis id>` (N-8); the sweep of that owner's job removes it."""
    handle = state.get("media")
    if handle is not None:
        handle.close()
    copy = state.get("media_copy")
    if copy is not None and not keep_copy:
        shutil.rmtree(Path(copy).parent, ignore_errors=True)


def sweep_tmp(*, only: str | None = None, min_age_s: float | None = None) -> list[str]:
    """Remove `data/tmp/<analysis id>` copies of originals (D-48; code review P2-15); returns the removed ids.

    A run removes its own copy in `cleanup`; this catches what a crashed or killed process left. A directory goes when it is named
    by an analysis id whose analysis is not queued or running, and is older than `min_age_s` (default
    `settings.tmp_sweep_min_age_s`, so the live copy of another process on the same data directory is left alone). With `only`,
    that analysis's directory is judged by its status alone (its job has just ended). Other entries (package work files) stay."""
    import time
    import uuid

    from sqlalchemy import select

    from app.storage.db import Analysis, get_session

    root = settings.data_dir / "tmp"
    if not root.is_dir():
        return []
    candidates = []
    for path in [root / only] if only is not None else list(root.iterdir()):
        try:
            uuid.UUID(path.name)
        except ValueError:
            continue
        if path.is_dir():
            candidates.append(path)
    if only is None:
        cutoff = time.time() - (settings.tmp_sweep_min_age_s if min_age_s is None else min_age_s)
        candidates = [p for p in candidates if p.stat().st_mtime < cutoff]
    if not candidates:
        return []
    with get_session() as db:
        active = set(db.scalars(select(Analysis.id).where(Analysis.id.in_([p.name for p in candidates]),
                                                          Analysis.status.in_(("queued", "running")))))
    removed = []
    for path in candidates:
        if path.name not in active:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    return removed


def face_stage_failure(ctx: StageContext) -> str | None:
    """Why the `faces` stage gave no usable face list (the stage failed, or a face detector did not report ok or not
    applicable), else None. Face-dependent detectors are then `failed`, not "no face" (D-46)."""
    faces = ctx.outputs.get("faces")
    if faces is None:
        return "the face-detection stage failed"
    broken = [e for e in faces.get("detectors") or [] if e["status"] not in ("ok", "not_applicable")]
    return "; ".join(f"{e.get('name') or e['detector_id']} {e['status']}" + (f": {e['reason']}" if e.get("reason") else "")
                     for e in broken) or None


def _face_failure(ctx: StageContext) -> str | None:
    """`face_stage_failure` for the kinds whose pipeline has a `faces` stage (image, video); None for the others."""
    return face_stage_failure(ctx) if ctx.state.get("kind") in FACE_KINDS else None


def face_gate_failure(ctx: StageContext) -> str | None:
    """Why fusion cannot evaluate a face gate, else None: the face stage gave no usable face list, or (K-3) the `condition` stage
    failed, so `face_px` is unknown and would read as no face."""
    failure = _face_failure(ctx)
    if failure is None and ctx.state.get("kind") in FACE_KINDS and ctx.outputs.get("condition") is None:
        return "the condition stage failed"
    return failure


def _needs_faces(registry: Any, detector_id: str) -> bool:
    spec = registry.adapter_spec(detector_id) if registry is not None else None
    return spec is not None and "faces" in spec.requirements.needs


def fusion_entries(ctx: StageContext, stages: tuple[str, ...]) -> list[dict[str, Any]]:
    """The detector entries fusion, findings and the result read (D-46, D-49; J-3e). While the face stage did not complete, a
    stored `not_applicable` entry of a detector that needs faces is `failed` ("face detection did not complete (…)"), so a revision
    re-fused from a run stored before J-1 caps as a fresh run does. `needs` comes from the adapter class without the enablement
    check (`DetectorRegistry.adapter_spec`): a detector disabled since the source ran is still rewritten; one the manifest no longer
    lists, or whose adapter does not import, keeps its stored entry."""
    found = entries(ctx, stages)
    failure = _face_failure(ctx)
    if failure is None:
        return found
    registry, reason = ctx.state.get("registry"), f"face detection did not complete ({failure})"[:300]
    return [{**e, "status": "failed", "reason": reason} if e["status"] == "not_applicable" and _needs_faces(registry, e["detector_id"])
            else e for e in found]


def run_detectors(ctx: StageContext, modalities: Iterable[Modality] = (Modality.IMAGE,), *, include: set[str] | None = None,
                  exclude: set[str] = frozenset(), outputs: StageOutputs | None = None,
                  max_units: int = MAX_UNIT_SCORES, face_failure: str | None = None,
                  audio_failure: str | None = None) -> dict[str, Any]:
    """`face_failure` (`face_stage_failure`): detectors that need faces are recorded as failed instead of running.
    `audio_failure` (D-49; the video `detect` stage): detectors that need audio are recorded as failed with this reason, without
    loading a model, when the probe reports a sound track that could not be decoded."""
    if ctx.state.get("refusion"):
        raise RuntimeError(f"Re-fusion does not run detectors, but the stored output of stage {ctx.stage_id} is not usable")
    registry, run_store = ctx.state["registry"], ctx.state["run_store"]
    handle = media(ctx)
    if outputs is None:
        outputs = StageOutputs(probe=ctx.outputs.get("probe"), faces=(ctx.outputs.get("faces") or {}).get("faces"))
    entries: list[dict[str, Any]] = []
    ran: set[str] = set()
    for modality in modalities:
        for detector_id in registry.enabled(modality, ctx.profile):
            detector = registry.instance(detector_id)
            family = detector.spec.family
            if (include is not None and family not in include) or family in exclude:
                continue
            ctx.check(f"detector {detector_id}")
            device = run_device(detector)
            run_ctx = RunContext(analysis_id=ctx.analysis_id, media=handle, profile=ctx.profile, device=device, dtype=None,
                                 lease=None, outputs=outputs, deadline=ctx.deadline, exhibits=ctx.exhibits.bind(detector_id),
                                 log=ctx.log.bind(detector_id=detector_id))
            needs = detector.spec.requirements.needs
            failure = (f"face detection did not complete ({face_failure})" if face_failure is not None and "faces" in needs
                       else audio_failure if audio_failure is not None and "audio" in needs else None)
            if failure is not None:
                result = DetectorResult.failed(failure[:300])
                result = store_run(run_store, DetectorRunRecord(ctx.analysis_id, detector_id, detector.spec.version, result.status,
                                                                result.reason, "", "", datetime.now(UTC).isoformat(), 0.0, result),
                                   run_ctx.log, ctx.deadline).result
            else:
                result = run_detector(detector, run_ctx, store=run_store, models=ctx.state.get("models"))
            ran.add(detector_id)
            entry = detector_entry(detector.spec, result, run_store.run_id(ctx.analysis_id, detector_id), max_units=max_units)
            entries.append(entry)
            if entry["status"] == "ok":
                ctx.emit("partial_signal", {"detector_id": detector_id, "signal_id": signal_id(detector_id, entry["aggregate"]),
                                            "name": detector.spec.name, "status": "ok"})
    return {"detectors": entries, **stage_exhibits(ctx, ran - not_stored(entries))}


def not_stored(entries: Iterable[dict[str, Any]]) -> set[str]:
    """The detectors whose run could not be stored (N-10a; N-12b, code re-check R3-03). Such a run is failed "so its result
    is not used", and its result is more than its score: the exhibits and artefacts it wrote before the write failed (face
    regions, a response map) are left out of the stage's output too. Whatever later stages derive from a stage's exhibits
    (faces and the condition profile, regions cut from a map, findings, the report's plates) then has nothing to derive from."""
    return {e["detector_id"] for e in entries if e.get("status") == "failed" and e.get("reason") == NOT_STORED}


def stage_exhibits(ctx: StageContext, creators: set[str]) -> dict[str, Any]:
    writer = ctx.exhibits
    return {"exhibits": [e.model_dump(mode="json") for e in writer.exhibits if e.created_by in creators],
            "artefacts": [dataclasses.asdict(a) for a in writer.artefacts if a.created_by in creators]}


def entries(ctx: StageContext, stages: tuple[str, ...]) -> list[dict[str, Any]]:
    return [e for stage in stages for e in (ctx.outputs.get(stage) or {}).get("detectors", [])]


def stage_exhibit_list(ctx: StageContext, stages: tuple[str, ...]) -> list[dict[str, Any]]:
    return [x for stage in stages for x in (ctx.outputs.get(stage) or {}).get("exhibits", [])]


def json_safe(value: Any) -> Any:
    """numpy scalars → Python numbers, NaN/Inf → None, tuples → lists: strictly valid JSON for the stored views."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --- stages ------------------------------------------------------------------------------------------------

def ingest(ctx: StageContext) -> dict[str, Any]:
    """Media object row, write-once original in the store, re-hash before acting (CC-01). The re-hash is recorded as
    `hash_verified` or `hash_mismatch` (C-3), and a mismatch fails the analysis."""
    from app.audit import chain as audit
    from app.storage.objects import ObjectIntegrityError, ObjectNotFound

    failure: Exception | None = None
    with ctx.sessions() as db:
        analysis = db.get(Analysis, ctx.analysis_id)
        if analysis is None:
            raise LookupError("The analysis record no longer exists")
        actor, expected = analysis.submitted_by or audit.SYSTEM_ACTOR, analysis.sha256
        media_row = db.scalars(select(MediaObject).where(MediaObject.sha256 == analysis.sha256)).first()
        if media_row is None:
            media_row = MediaObject(id=str(uuid.uuid4()), sha256=analysis.sha256, size_bytes=analysis.size_bytes,
                                    mime_sniffed=analysis.content_type, original_filename=analysis.filename)
            db.add(media_row)
        analysis.media_id = media_row.id
        db.flush()
        media_id = media_row.id
        try:
            original = ensure_original_in_store(db, ctx.store, media_row)
            verify_artefact(db, ctx.store, original.id)
        except (ObjectIntegrityError, ObjectNotFound) as e:
            failure = e
            db.rollback()
        else:
            db.commit()
            out = {"media_id": media_row.id, "sha256": media_row.sha256, "md5": media_row.md5,
                   "size_bytes": int(media_row.size_bytes), "mime": media_row.mime_sniffed, "filename": analysis.filename,
                   "original_artefact_id": original.id}
    # N-9: `hash_mismatch` and `hash_verified` record a check that ran, appended once the ingest transaction is closed. A
    # failed append fails the ingest stage, which is critical: the analysis does not go on without the record
    if failure is not None:
        audit.record("hash_mismatch", actor_id=actor, analysis_id=ctx.analysis_id, before_hash=expected,
                     details={"purpose": "analysis", "error": str(failure)[:300]}, sessions=ctx.sessions)
        raise failure
    audit.record("hash_verified", actor_id=actor, analysis_id=ctx.analysis_id, media_id=media_id, before_hash=expected,
                 after_hash=expected, details={"purpose": "analysis"}, sessions=ctx.sessions)
    return out


def index_artefacts(ctx: StageContext, stages: tuple[str, ...], extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Index the artefacts written by earlier stages, plus `extra` records (idempotent on their deterministic ids)."""
    records = [a for stage in stages for a in (ctx.outputs.get(stage) or {}).get("artefacts", [])] + list(extra or [])
    with ctx.sessions() as db:
        for a in records:
            index_object(db, a["sha256"], a["size_bytes"], media_type=a["media_type"], role=a["role"],
                         created_by=a["created_by"], analysis_id=ctx.analysis_id, artefact_id=a["id"])
        db.commit()
    return {"artefacts_indexed": len(records), "bytes": sum(a["size_bytes"] for a in records)}


# --- legacy (v0.3) views -----------------------------------------------------------------------------------

def fallback_decision(verdict: dict[str, Any], *, informative: bool, subject: str) -> dict[str, Any]:
    band, confidence = fallback_cap(verdict["band"], verdict["confidence"])  # L-3: never "very likely" or high
    if informative:  # N-7: the sentence is built from the capped band and confidence, never copied from the legacy fuser
        code, verbal, reason = BAND_CODE[band], BAND_HEADLINE[band], fallback_reason(subject, band, confidence, verdict)
    else:
        code, verbal = "not_assessable", "Not assessable: no detector produced a usable result"
        reason = f"No enabled detector returned a score that can be used for this {subject}."  # N-17 (D-68): not "a probability"
    return Decision(code=code, verbal=verbal, probability_ai_uncalibrated=round(verdict["probability_ai"], 4) if informative else None,
                    band=band, confidence=confidence if informative else "low", reason=reason,
                    fusion_mode="fallback").model_dump(mode="json")


def legacy_signal_v2(s: dict[str, Any], n: int, *, entry: dict[str, Any] | None, fallback_detector: str, modality: str,
                     version: str, hypothesis: str | None = "H1_whole_content_synthesis", family: str = "legacy",
                     exhibit_ids: list[str] | None = None) -> dict[str, Any]:
    """A v0.3 aggregate signal as `SignalV2`. With a source detector entry the id, version, revisions and raw score
    come from that detector; otherwise the signal is derived by the legacy fuser (`<fallback_detector>:<v1 id>`)."""
    prob = s.get("ai_probability")
    metrics: dict[str, float] = {}
    if prob is not None:
        metrics["legacy_probability"] = float(prob)
    if entry is not None:
        aggregate = entry["aggregate"]
        detector_id, sid = entry["detector_id"], signal_id(entry["detector_id"], aggregate)
        detector_version, revisions = entry["version"], entry["model_revisions"]
        raw, raw_kind = (aggregate["raw"], aggregate["raw_kind"]) if aggregate is not None else (prob, "probability")
        family, exhibits = entry["family"], [*entry["exhibit_ids"], *(exhibit_ids or [])]
        metrics.update(entry["metrics"])
    else:
        detector_id, sid = fallback_detector, f"{fallback_detector}:{s['id']}"
        detector_version, revisions, raw, raw_kind, exhibits = version, {}, prob, "probability", list(exhibit_ids or [])
    return SignalV2.model_validate({
        "id": sid, "display_id": f"S{n:02d}", "detector_id": detector_id, "detector_version": detector_version,
        "model_revisions": revisions, "name": s["name"], "family": family, "modality": modality, "category": s["category"],
        "status": "ok", "scope": "global", "scope_ref": "global", "hypothesis": hypothesis if prob is not None else None,
        # N-19p (SEC6-01): a v0.3 signal can carry text from the file (a video's container tags), and reaches no detector run's
        # filter: `observed`, `reasoning` and the strings of `details` are filtered here, at write time
        "raw_score": raw, "raw_kind": raw_kind, "effective_weight": s["weight"], "observed": untrusted_text(s["observed"]),
        "reference": s["reference"], "reasoning": untrusted_text(s["reasoning"]), "metrics": metrics,
        "details": untrusted_strings(json_safe(s.get("details") or {})),
        "exhibit_ids": exhibits,
        "legacy": {"v1_id": s["id"], "value": s.get("value"), "direction": s["direction"], "strength": s["strength"],
                   "decisive": bool(s.get("decisive"))},
    }).model_dump(mode="json")


def current_synthesis_output(output: Any) -> bool:
    """Only reuse findings produced by this template set; old objects remain immutable."""
    return isinstance(output, dict) and output.get("template_set") == synthesis.TEMPLATE_SET


def synthesise_findings(ctx: StageContext, *, kind: str, entry_stages: tuple[str, ...], exhibit_stages: tuple[str, ...],
                        family_by_v1_id: dict[str, str] | None = None) -> dict[str, Any]:
    """Findings synthesis v2 (§4.5–4.6, M-4) over the fuse output, the detector entries, exhibits and model-card fields."""
    fused = ctx.outputs.get("fuse") or {}
    detector_entries = fusion_entries(ctx, entry_stages)
    manifest = getattr(ctx.state.get("registry"), "manifest", None)
    cards: dict[str, dict[str, Any]] = {}
    for e in detector_entries:
        card = manifest.detector(e["detector_id"]) if manifest is not None else None
        if card is not None:
            # L-7: a result publishes the card's false-positive modes in client words; the card keeps its own record verbatim
            cards[e["detector_id"]] = {"fp_modes": card_copy.client_modes(card.fp_modes), "eval": [r.model_dump(mode="json") for r in card.eval]}
    findings = synthesis.synthesise(synthesis.SynthesisInput(
        kind=kind, decision=fused.get("decision") or {}, signals=fused.get("signals") or [], entries=detector_entries,
        exhibits=stage_exhibit_list(ctx, exhibit_stages), condition_profile=fused.get("condition_profile"),
        calibration=fused.get("calibration") or [], cards=cards, family_by_v1_id=family_by_v1_id or {},
        not_run=(fused.get("fusion") or {}).get("not_run") or [], failed_steps=failed_steps(ctx)))
    # A calibrated speech decision includes abstention guards. Generic LR wording
    # must not turn an uncertain decision back into an AI/original conclusion.
    for finding in findings:
        if finding.get("family") != "A1_voice" or finding.get("scope"):
            continue
        linked = [s for s in fused.get("signals", []) if s["id"] in finding.get("evidence_ids", [])]
        screen = next((s.get("details", {}).get("audio_screening") for s in linked
                       if s.get("details", {}).get("audio_screening")), None)
        if screen:
            direction = screen.get("decision", "inconclusive")
            statement = screen["reason"] + " " + screen["limitation"]
            finding.update(title={"ai": "Leaning towards AI-generated speech", "original": "Leaning towards original speech"}.get(
                               direction, "Inconclusive speech assessment"),
                           statement=statement, sentences=[], probability=screen.get("probability_ai"),
                           direction={"ai": "for", "original": "against"}.get(direction, "neutral"),
                           strength="weak" if direction != "inconclusive" else "none",
                           standalone_strength=None, standalone_direction=None)
    return {"findings": findings, "template_set": synthesis.TEMPLATE_SET}


V1_OUTSIDE = "Inconclusive: this input is outside what was validated ({reason})."


def v1_summary(decision: Mapping[str, Any], profile: Mapping[str, Any] | None = None) -> str:
    """The summary of the v1 view of a calibrated result: the decision's own headline, or that the input is outside what
    was validated, with why. N-19k said so for a capped video, by the reason of the cap; N-19n (the rest of Q2-14): for an
    image too (its v1 view is `schemas.legacy.downgrade_v2`'s, which printed "the checks don't point clearly either
    way"), and with `profile` also where the cap changed no code (`abstain.outside_why`)."""
    # only of a result that is inconclusive: under a record that decides outside what was validated the headline is the record's
    outside = (abstain.outside_why(decision, profile) or abstain.cap_reason(decision)) if decision.get("code") == "inconclusive" else None
    return V1_OUTSIDE.format(reason=outside) if outside else str(decision["verbal"])


def legacy_verdict(verdict: dict[str, Any], decision: dict[str, Any], profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The v0.3 verdict block with the published decision's numbers: the fallback caveat first on the fallback path; the
    calibrated probability, band label and headline with a fuser artefact. `probability_ai` is None whenever the decision
    publishes no estimate (D-49: capped because a check did not run, or not assessable); the legacy number is not kept.
    `profile`: the result's condition profile, by which a calibrated result outside what was validated says so (N-19n)."""
    out = dict(verdict)
    probability = decision.get("probability_ai_uncalibrated")
    if probability is None:
        probability = decision.get("probability_ai_calibrated")
    out["probability_ai"] = probability
    out["band"], out["confidence"] = decision["band"], decision["confidence"]
    # L-3, D-66: the label follows the published decision (its capped band; "Not assessable") on both paths
    out["label"] = v1_label(decision.get("code"), decision["band"])
    caveats = [c for c in verdict.get("caveats", []) if c != FALLBACK_CAVEAT]
    if decision.get("fusion_mode") == "artefact":
        # N-19k (browser QA, part 2): a result outside what was validated says so, with the reason of its cap; its own
        # sentence ("the checks don't point clearly either way") is not why it is inconclusive
        out.update(summary=v1_summary(decision, profile), caveats=caveats)
    else:  # N-7: the published reason (capped band and confidence, a documentary override, a cap), not the fuser's sentence
        out.update(summary=decision["reason"], caveats=[FALLBACK_CAVEAT, *caveats])
    return out


# --- fusion (M-2) ------------------------------------------------------------------------------------------

def select_fusion(ctx: StageContext, kind: str) -> Selection:
    """The newest compatible fuser artefact (or the pinned one, for refusion and re-runs), else the fallback. A re-run of a
    fallback analysis stays on the fallback (CC-03). Records the artefact."""
    if ctx.state.get("force_fallback"):
        ctx.log.info("fusion.fallback", kind=kind, reason="re-run of a fallback analysis")
        return Selection(kind=kind, mode="fallback", reason="re-run of an analysis that used the uncalibrated fallback (CC-03)",
                         installed=bool(candidates(kind)))
    pin = ctx.state.get("fuser_pin")
    if ctx.state.get("refusion"):  # D-47: the stored runs (copied from the source) decide compatibility, not what is enabled now
        from app.engine import refusion

        with ctx.sessions() as db:
            selection = refusion.select_for_source(db, ctx.state["registry"], kind, ctx.analysis_id, pin)
    else:
        selection = select_for_registry(ctx.state["registry"], kind, ctx.profile, pin=pin)
    if selection.mode == "artefact":
        with ctx.sessions() as db:
            added = record_fuser(db, selection)
            db.commit()
        if added:  # the first use of this artefact (C-3)
            from app.audit import chain as audit

            # N-9: the first use of an artefact, appended after its row. A failed append fails the fuse stage (critical),
            # and the re-run finds the row and records nothing: a lost `fuser_loaded` is acceptable, because every result
            # names its artefact in `manifest_recorded` and in its own decision
            audit.record("fuser_loaded", actor_id=audit.SYSTEM_ACTOR, analysis_id=ctx.analysis_id, object_ids=[selection.sha256 or ""],
                         details={"kind": kind, "version": selection.version}, sessions=ctx.sessions)
        from app.engine.fusion import review_log

        # N-9 (D-57): a result names the reviewer of its artefact, so the review it is fused under is in the custody log
        # first. A sidecar that came with a release, or was copied, replaced or removed by hand, is recorded here once
        # (the row and its event in one transaction); a failed append fails the fuse stage and the retry records it
        subject = review_log.of_selection(selection, kind)
        if subject is not None:
            review_log.reconcile(subject, selection.review, source=review_log.FOUND, sessions=ctx.sessions)
    elif pin is not None:
        raise RuntimeError(f"The pinned fuser artefact cannot be used: {selection.reason}")
    else:
        ctx.log.info("fusion.fallback", kind=kind, reason=selection.reason)
    return selection


def artefact_fusion(ctx: StageContext, kind: str, selection: Selection, legacy: dict[str, Any],
                    detector_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace the fallback decision of a legacy fuse output with the artefact's calibrated decision (§4.4).

    - **Face gates** (D-49): while the face stage did not complete, or the condition stage failed (K-3), an artefact detector
      gated on a face term is fused as `failed`, because a missing face size reads as no face and would open a `!has_face` gate.
    - **Condition stage** (N-8): a failed condition stage of an image or video is itself a check that did not run, whatever the
      gates of the artefact (the served video fuser has none), so no estimate is published and the reason names it.
    - **Re-fusion** (J-3e): the revision names a pipeline or build change since its source ran (`refusion.revision_caveat`)."""
    assert selection.fuser is not None
    failure = face_gate_failure(ctx)
    if failure is not None:
        gated = sorted(d for d, literals in (selection.fuser.doc.get("gates") or {}).items() if face_gated(literals))
        by_id, registry = {e["detector_id"]: e for e in detector_entries}, ctx.state.get("registry")
        cause = failure if _face_failure(ctx) is None else f"face detection did not complete ({failure})"
        reason = f"its face gate cannot be evaluated: {cause}"[:300]
        blank = {"version": "", "hypotheses": [], "unit_scores": [], "metrics": {}, "provenance_override": None}
        detector_entries = [e for e in detector_entries if e["detector_id"] not in gated] + [
            {**(by_id.get(d) or {**blank, "detector_id": d, "name": (registry.detector_name(d) if registry is not None else None) or d}),
             "status": "failed", "reason": reason, "aggregate": None} for d in gated]
    fused = fusion_runtime.fuse_with_artefact(selection, detector_entries, ctx.outputs.get("condition"), kind=kind,
                                              prior_ai=settings.prior_ai, h3_bound=settings.h3_influence_bound,
                                              policy=ctx.state.get("policy"),
                                              condition_failed=kind in FACE_KINDS and ctx.outputs.get("condition") is None,
                                              check_names=_check_names(ctx, selection.fuser.detectors))
    provenance: list[str] = []
    if ctx.state.get("refusion"):
        from app.engine import refusion
        from app.engine.pipelines import PIPELINE_VERSIONS

        with ctx.sessions() as db:
            caveat = refusion.revision_caveat(db, ctx.store, ctx.analysis_id, PIPELINE_VERSIONS[kind], build_info.git_commit())
        provenance = [caveat] if caveat else []
    fused_caveats = [*fused["caveats"], *provenance]
    signals = fusion_runtime.annotate(legacy["signals"], fused["signal_fields"], artefact_detectors=selection.fuser.detectors,
                                      outside=fused.get("signal_outside"))
    kept = [c for c in legacy.get("caveats", []) if c != FALLBACK_CAVEAT]
    out = {**legacy, "decision": fused["decision"], "assessments": fused["assessments"], "signals": signals,
           "caveats": [*fused_caveats, *kept], "calibration": fused["calibration"],
           "condition_profile": fused["condition_profile"], "fusion": fused["fusion"]}
    if isinstance(legacy.get("v1"), dict) and "verdict" in legacy["v1"]:
        verdict = legacy["v1"]["verdict"]
        # D-49: the fused caveats (D-46 did-not-run, envelope) reach the v1 verdict, as `downgrade_v2` gives images
        carried = [*fused_caveats, *(c for c in verdict.get("caveats", []) if c not in fused_caveats)]
        out["v1"] = {**legacy["v1"], "verdict": legacy_verdict({**verdict, "caveats": carried}, fused["decision"], fused["condition_profile"])}
    return out


def fallback_checks(signals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """D-64 (N-18): the signals of a fallback result as `abstain.content_check_scored` reads them. A signal scored when
    the legacy fuser read a probability from it and gave it a weight; a signal of the provenance hypothesis (H3) is a
    record about the file whatever its family (the v0.3 video tags have the family of the legacy fuser)."""
    out = []
    for s in signals:
        probability = (s.get("metrics") or {}).get("legacy_probability") is not None or (
            s.get("raw_kind") == "probability" and s.get("raw_score") is not None)
        out.append({"family": "provenance" if s.get("hypothesis") == H3_VALUE else s.get("family"),
                    "scored": bool(s.get("status") == "ok" and probability and (s.get("effective_weight") or 0) > 0)})
    return out


def _outside(ctx: StageContext, kind: str, profile: dict[str, Any]) -> str | None:
    """Why the input of a fallback result is outside what the release bench validated: below its size floor, content it
    does not hold, or, without a condition output for an image or a video, that nothing was measured. None inside."""
    if kind in FACE_KINDS and ctx.outputs.get("condition") is None:  # the calibrated path: `condition_failed`
        return abstain.CONDITIONS_NOT_MEASURED
    return abstain.outside_reason(kind, profile, (ctx.outputs.get("condition") or {}).get("context"), release_bench(kind))


def fallback_fusion(ctx: StageContext, kind: str, selection: Selection, legacy: dict[str, Any]) -> dict[str, Any]:
    """The fallback result of a kind. Documentary evidence (H3) is not a statistical result: its code and its number stay.

    - **A check of the content (N-18; D-64):** a statistical fallback result in which no check of the pixels, frames, sound
      or text scored (metadata and provenance records only) is `not_assessable`: metadata is never evidence of authenticity.
    - **Size floor and content (N-7, N-18; D-65):** below the size floor of the release bench, or for content that bench
      does not hold, a statistical fallback is capped at inconclusive, as the calibrated result is (code review 13): the
      fallback has no validation of its own, and the only validation the product claims does not reach such inputs.
    - **Without a condition output (N-19f; code review R4-03):** the condition stage is not critical for an image or a
      video, so it can fail and the run goes on. Nothing then says whether the input is below the floor or of such a
      content (`outside_reason` reads an output that is not there and finds no reason), so the result is outside what was
      validated and capped in the same way, with the score before the cap in its reason. N-18b's caveat says that the
      step failed. `not_assessable` is left as it is.
    - **A result that a record decides (N-19h; E-20, 8):** the same look is made for it. It keeps its code and its
      number; it says that the input is outside what was validated, and its statistical score is not printed."""
    profile = fusion_runtime.fallback_profile(ctx.outputs.get("condition"), kind, installed=selection.installed)
    out = {**legacy, "condition_profile": profile, "fusion": selection.as_json()}
    decision = legacy["decision"]
    if decision.get("driving_hypothesis") == H3_VALUE:
        # N-19h (E-20, 8): the look that caps a statistical result is made for a result that a record decides too. Its
        # code, its number and its headline are the record's and stay; what changes is that the result says that the
        # input is outside what was validated (in its reason, which is where a fallback result says it:
        # `abstain.outside_validated`), and that the statistical score is printed beside no record: the caveat says
        # that it is not shown, and why; the decision's field keeps the score
        below = _outside(ctx, kind, profile)
        if below is None or decision["code"] == "not_assessable":
            return out
        named = Decision.model_validate({
            **decision, "reason": f"{decision['reason']} {abstain.OUTSIDE_RECORD_SENTENCE.format(reason=below)}"}).model_dump(mode="json")
        profile["limiting_conditions"] = [*profile.get("limiting_conditions", []), below]
        out.update(decision=named, caveats=[*decisive.withhold_statistical(legacy.get("caveats", []), fallback=True),
                                            f"Not validated for inputs like this: {below}."])
        return out
    if decision["code"] != "not_assessable" and not abstain.content_check_scored(fallback_checks(legacy.get("signals") or [])):
        reason = abstain.no_content_check_reason(kind)
        withdrawn = Decision.model_validate({
            **decision, "code": "not_assessable", "verbal": abstain.NOT_ASSESSABLE_VERBAL, "probability_ai_uncalibrated": None,
            "band": "uncertain", "confidence": "low", "reason": reason}).model_dump(mode="json")
        out.update(decision=withdrawn)
        if isinstance(legacy.get("v1"), dict) and "verdict" in legacy["v1"]:
            out["v1"] = {**legacy["v1"], "verdict": legacy_verdict(legacy["v1"]["verdict"], withdrawn)}
        return out
    below = _outside(ctx, kind, profile)
    if below is None or abstain.cap(decision["code"], False) == decision["code"]:
        return out
    score = decision.get("probability_ai_uncalibrated")
    before = f" Before the cap the result was {uncalibrated_score(score)} (\"{decision['verbal']}\")." if score is not None else ""
    capped = Decision.model_validate({
        **decision, "code": "inconclusive", "band": "uncertain", "confidence": "low", "verbal": BAND_HEADLINE["uncertain"],
        "reason": f"Capped at inconclusive: {below}.{before}"}).model_dump(mode="json")
    profile["limiting_conditions"] = [*profile.get("limiting_conditions", []), below]
    out.update(decision=capped, caveats=[*legacy.get("caveats", []), f"Not validated for inputs like this: {below}."])
    if isinstance(legacy.get("v1"), dict) and "verdict" in legacy["v1"]:
        out["v1"] = {**legacy["v1"], "verdict": legacy_verdict(legacy["v1"]["verdict"], capped)}
    return out


# --- result and publish ------------------------------------------------------------------------------------

def _profile_omissions(ctx: StageContext, kind: str, detector_entries: list[dict[str, Any]]) -> list[str]:
    """Primary route completeness, including policy omissions (D-08/PL-03), without loading adapters.

    Face/track dependencies count even without a score. Optional second-pass absence does not count, but a recorded
    failed refinement does. System stages and legacy fused signals are not detector executions.
    """
    registry = ctx.state["registry"]
    modalities = {"image": (Modality.IMAGE,), "document": (Modality.TEXT,), "audio": (Modality.AUDIO,),
                  "video": (Modality.VIDEO, Modality.AUDIO)}[kind]

    def routed(modality: Modality, family: str | None) -> bool:
        # Video provenance schedules VIDEO only; its mixed detect stage excludes these dependency families.
        return not (kind == "video" and modality == Modality.AUDIO
                    and family in {"provenance", "face-detection", "face-tracking"})

    expected = {d for modality in modalities for d in registry.enabled(modality, ctx.profile)
                if routed(modality, registry.detector_family(d))}
    policy_omitted: set[str] = set()
    # Registry overrides replace the entire catalogue (model-free fixtures), including its policy exclusions.
    if getattr(registry, "_override", None) is None:
        for card in registry.manifest.detectors:
            if (not card.default_enabled or not card.adapter or card.modality not in modalities
                    or ctx.profile.value not in card.profiles or not routed(card.modality, card.family)):
                continue
            decision = registry.decision(card.id)
            if any(reason.startswith("superseded by ") for reason in decision.reasons):
                continue  # an installed selected replacement fulfils this alternative
            if any(reason == "listed in deny_detectors" or " has licence class " in reason for reason in decision.reasons):
                expected.add(card.id)
                policy_omitted.add(card.id)

    selected = set(expected)
    detected = ctx.outputs.get("detect") or {}
    content = (ctx.outputs.get("condition") or {}).get("content") or {}
    if kind == "audio" and content.get("assessable") is False and detected.get("not_run"):
        # audio.detect deliberately emits no rows after its persisted content-applicability gate.
        exempt = {d for d in expected if registry.detector_family(d) != "provenance"}
        expected -= exempt
        policy_omitted -= exempt
    present = {e["detector_id"] for e in detector_entries}
    recorded = [*detector_entries, *(ctx.outputs.get("second_pass") or {}).get("detectors", [])]
    broken = {e["detector_id"] for e in recorded if e["detector_id"] in selected
              and e["status"] not in {"ok", "not_applicable"}}
    return sorted(policy_omitted | (expected - present) | broken)


def _check_names(ctx: StageContext, detectors: Iterable[str]) -> dict[str, str]:
    """The registry's name of each of these checks (N-19k; D2-06), for a caveat about one that has no run entry."""
    registry = ctx.state.get("registry")
    if registry is None:
        return {}
    return {d: name for d in detectors if (name := registry.detector_name(d))}


# N-19k (D2-06): a step of the run as a caveat names it; never "stage:second_pass"
STEP_NAMES = {"second_pass": "the second pass over the flagged segments"}
OTHER_STEP = "a later step of the analysis"


def check_name(item: str, entries: Mapping[str, Mapping[str, Any]], registry: Any) -> str:
    """N-19k (D2-06; browser QA lead 4): a check as the page names it (its run entry's name, else the registry's), a
    step of the run in words; never a detector id or a stage name."""
    if item.startswith("stage:"):
        return STEP_NAMES.get(item.removeprefix("stage:"), OTHER_STEP)
    return ((entries.get(item) or {}).get("name") or (registry.detector_name(item) if registry is not None else None)
            or UNNAMED_CHECK)


def reduced_caveats(caveats: list[str], omitted: Sequence[str], detector_entries: Sequence[Mapping[str, Any]], registry: Any,
                    profile: str) -> list[str]:
    """The caveats of a result whose profile ran fewer checks than it holds (N-19k; browser QA leads 3 and 4; D2-06).

    - **One fact once:** the checks that could not run on this server (its memory budget, model files that are not
      installed) are one caveat with their names, "2 checks could not run on this server: A (memory budget); B (memory
      budget). This result rests on fewer checks than the profile holds." It takes the place of one caveat for each such
      check ("A check that this analysis depends on did not run: A (could not run on this server: memory budget).") and
      of their ids in the caveat of the reduced profile, which named them a second time. The reports then add no
      sentence of their own (`forensic.applied_limitations`). This is how a result is composed from now on: a stored
      result keeps its caveats and their numbers, under which conclusions hold acknowledgements.
    - **Names:** every other omitted or incomplete check stands in "Requested <profile> profile reduced: …" by the
      name the page gives it, and a step of the run in words."""
    by_id = {str(e.get("detector_id")): e for e in detector_entries}
    server = [e for e in detector_entries if of_the_server(e.get("status"), e.get("reason"))]
    out = list(caveats)
    if server:
        single = {fusion_runtime.NOT_RUN_CAVEAT.format(name=e.get("name") or "", reason=f"{ON_THIS_SERVER}: {e.get('reason')}") for e in server}
        out = [c for c in out if c not in single]
        named = "; ".join(f"{check_name(str(e['detector_id']), by_id, registry)} ({e.get('reason')})" for e in server)
        out.append(f"{len(server)} {'check' if len(server) == 1 else 'checks'} {ON_THIS_SERVER}: {named}. {SERVER_FEWER}")
    rest = [item for item in omitted if item not in {str(e["detector_id"]) for e in server}]
    if rest:
        out.append(f"Requested {profile} profile reduced: omitted or incomplete checks: "
                   f"{', '.join(check_name(item, by_id, registry) for item in rest)}.")
    return out


def build_result(ctx: StageContext, *, kind: str, pipeline_version: str, entry_stages: tuple[str, ...],
                 exhibit_stages: tuple[str, ...], exhibits: list[dict[str, Any]] | None = None) -> AnalysisResultV2:
    ingested, fused = ctx.outputs["ingest"], ctx.outputs.get("fuse")
    if fused is None:
        raise RuntimeError("fusion did not produce a decision")
    detector_entries = fusion_entries(ctx, entry_stages)
    fusion = fused.get("fusion") or {}
    registry = ctx.state["registry"]
    with ctx.sessions() as db:
        analysis = db.get(Analysis, ctx.analysis_id)
        revision, case_id = (analysis.revision or 1, analysis.case_id) if analysis is not None else (1, None)
        intake_event = build_info.intake_event_id(db, analysis) if analysis is not None else None  # CC-01
        source = (db.get(Analysis, analysis.supersedes_id)
                  if ctx.state.get("refusion") and analysis is not None and analysis.supersedes_id else None)
        source_effective = source.effective_profile if source is not None else None
        second_pass_incomplete = False
        if kind == "video" and ctx.profile == Profile.DEEP and not ctx.state.get("refusion"):
            # AnalysisStage stores the latest attempt in one row. Do not reinterpret historical refusion stages.
            second = db.get(AnalysisStage, (ctx.analysis_id, "second_pass"))
            second_pass_incomplete = bool(
                (ctx.outputs.get("second_pass") or {}).get("budget_exhausted")
                or (second is not None and (second.status == "failed"
                    or (second.status == "skipped" and second.error != "not in this analysis profile"))))
    if ctx.state.get("refusion"):
        # Stored execution belongs to its source selection, not today's registry. Unknown historical completeness stays
        # unknown; a newly rewritten D-49 dependency failure can still reduce this revision.
        omitted = sorted({after["detector_id"] for before, after in zip(entries(ctx, entry_stages), detector_entries, strict=True)
                          if before["status"] != after["status"] and after["status"] not in {"ok", "not_applicable"}})
        effective = profiles.effective_profile(ctx.profile, omitted) if omitted else source_effective
    else:
        omitted = _profile_omissions(ctx, kind, detector_entries)
        if second_pass_incomplete:
            omitted = sorted({*omitted, "stage:second_pass"})
        effective = profiles.effective_profile(ctx.profile, omitted)
    caveats = list(fused["caveats"])
    if (fused.get("decision") or {}).get("code") == "not_assessable":
        # N-17 (D-68; design review DS-10): a result that is not assessable speaks of no estimate and no probability, so the
        # caveats about one ("treat the probability as indicative only", the fallback's "probability is not a frequency
        # estimate") are left out, on the page and in the reports alike
        caveats = [c for c in caveats if not effect.speaks_of_estimate(c)]
    caveats = reduced_caveats(caveats, omitted, detector_entries, registry, ctx.profile.value)
    # N-18b (operations re-check O4-01): a result says when a step of its run failed, and what it lacks for it
    caveats += [c for c in failed_steps(ctx) if c not in caveats]
    device, precision = build_info.compute(detector_entries)
    return AnalysisResultV2.model_validate({
        "analysis_id": ctx.analysis_id, "kind": kind, "revision": revision, "case_id": case_id,
        "input": {"sha256": ingested["sha256"], "md5": ingested["md5"], "size_bytes": ingested["size_bytes"],
                  "mime_sniffed": ingested["mime"], "filename": ingested["filename"], "intake_event_id": intake_event},
        "manifest": {"veritas_version": settings.version, "pipeline_version": pipeline_version, "profile": ctx.profile.value,
                     "effective_profile": effective, "licence_posture": registry.policy.label,
                     # §6.6 build, configuration, compute, libraries and sampling (Q-4d)
                     "git_commit": build_info.git_commit(), "config_sha256": build_info.config_sha256(),
                     "models_manifest_sha256": build_info.models_manifest_sha256(), "device": device,
                     "precision": precision, "library_versions": build_info.library_versions(),
                     "sampling": build_info.sampling(kind, ctx.outputs),
                     "disclosures": registry.disclosures(sorted({e["detector_id"] for e in detector_entries})),  # D-34
                     "detectors": rerun.detector_refs(registry, detector_entries),  # pinned weights (CC-03)
                     "calibration_id": f"fuser-{kind}-{fusion['version']}" if fusion.get("mode") == "artefact" else None,
                     "policy_thresholds_id": (ctx.state.get("policy") or {}).get("id"),  # API-03
                     "template_set": (ctx.outputs.get("synthesise") or {}).get("template_set")},
        "condition_profile": fused.get("condition_profile"),
        "decision": fused["decision"], "assessments": fused["assessments"], "signals": fused["signals"],
        "exhibits": stage_exhibit_list(ctx, exhibit_stages) if exhibits is None else exhibits,
        "findings": (ctx.outputs.get("synthesise") or {}).get("findings", []),
        "calibration": fused.get("calibration") or [{"subject": "legacy-fuser", "method": "legacy-heuristic"}],
        "caveats": caveats,
    })


def failed_steps(ctx: StageContext) -> list[str]:
    """N-18b: the caveats of the stages of this run that are not critical and failed (`Runner.run` records them in the run's
    shared state), in the order of their ids: "The step that marks regions failed: this result shows no regions from it." """
    from app.engine.runner import STAGE_FAILURES

    return [effect.step_failed(stage) for stage in sorted(ctx.state.get(STAGE_FAILURES) or {})]


class EvidenceMissing(ValueError):
    """An object that a stage of this analysis wrote is not in the store (N-18a). A `ValueError`: its sentence is the
    analysis's reason (`core.errors.friendly_error`)."""


def missing_objects(ctx: StageContext) -> dict[str, list[dict[str, Any]]]:
    """By stage, the objects its output names that are not in the store now."""
    from app.engine.runner import listed_objects

    found = {stage: [a for a in listed_objects(output) if not ctx.store.exists(a["sha256"])] for stage, output in ctx.outputs.items()}
    return {stage: gone for stage, gone in found.items() if gone}


def publish(ctx: StageContext, result: AnalysisResultV2, v1: dict[str, Any], *, pipeline_version: str,
            run_stages: tuple[str, ...]) -> dict[str, Any]:
    """Project the evidence tables, store the canonical result and the v1 view, then record `manifest_recorded` (C-3).
    Reports are separate jobs (B11). On a re-fusion, the revision's copied run row of each entry `fusion_entries` rewrote is set
    to `failed` with the same reason, so coverage and the report limitations agree with the cap; the raw-output artefact stays
    the source's record.

    N-18a (operations re-check O3-01): nothing is published while an object that a stage of this run wrote is missing from the
    store. The runner reuses a stage only with its objects, so this is the object that went after that check (a release
    that had set it aside at that moment, a removal by hand). The analysis fails with the stage's name; its next run finds the
    stage without its objects and runs it again."""
    from app.audit import chain as audit

    if (missing := missing_objects(ctx)):
        from app.engine.runner import evidence_missing

        raise EvidenceMissing(evidence_missing({stage: len(gone) for stage, gone in missing.items()}))
    result_hash = canonical_hash(result)
    manifest = getattr(result, "manifest", None)

    def unit() -> str:
        # N-9: the published result and its `manifest_recorded` event are one transaction: a result is never in force
        # without the custody record of what produced it. The chain's append lock comes before the write lock (`audit.writing`).
        with audit.writing(ctx.sessions) as db:
            # Publication and expired-lease finalization serialize on the same analysis row. If terminal cancellation
            # committed first, the late worker cannot publish a contradictory result; if this commit wins, recovery is done.
            conn = db.connection()
            if conn.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            analysis = db.scalar(select(Analysis).where(Analysis.id == ctx.analysis_id).with_for_update())
            if analysis is None:
                raise LookupError("The analysis record no longer exists")
            if analysis.status in ("cancelled", "failed") and not analysis.result_hash:
                raise Cancelled("The analysis was finalized before this worker could publish")
            actor = analysis.submitted_by or audit.SYSTEM_ACTOR
            if ctx.state.get("refusion"):
                stored = entries(ctx, run_stages)
                rewritten = {f["run_id"]: f["reason"] for e, f in zip(stored, fusion_entries(ctx, run_stages), strict=True)
                             if f["status"] != e["status"] and f["run_id"]}
                for run in db.scalars(select(DetectorRun).where(DetectorRun.analysis_id == ctx.analysis_id,
                                                                DetectorRun.id.in_(list(rewritten)))):
                    run.status, run.reason = "failed", rewritten[run.id]
            if analysis.result_hash != result_hash:
                runs = {e["detector_id"]: e["run_id"] for e in entries(ctx, run_stages) if e["run_id"]}
                persist_result(db, result, store=ctx.store, detector_run_ids=existing_runs(db, runs))
            artefact = register_artefact(db, ctx.store, canonical_json(result), media_type="application/json",
                                         role="stage_output", created_by="publish", analysis_id=ctx.analysis_id)
            analysis.result_json = json.dumps(json_safe(v1), allow_nan=False)
            analysis.probability_ai = result.decision.probability_ai
            analysis.band = result.decision.band
            analysis.pipeline_version = pipeline_version
            evidence_events = []
            if analysis.kind in {"image", "video", "audio", "document"}:
                from app.evidence.service import current_set_id, load_packet_set, persist_packet_set

                previous = load_packet_set(db, analysis, ctx.store, set_id=current_set_id(analysis))
                packet_set = persist_packet_set(db, analysis, ctx.store, result=result.model_dump(mode="json"))
                if previous is None and packet_set:
                    evidence_events.append({"action": f"{analysis.kind}_evidence_generated", "actor_id": actor,
                                            "analysis_id": ctx.analysis_id, "object_ids": [packet_set["id"]],
                                            "after_hash": packet_set["sha256"], "details": {"version": packet_set["version"]}})
            artefact_id = artefact.id
            audit.commit_with(db, [{
                "action": "manifest_recorded", "actor_id": actor, "analysis_id": ctx.analysis_id, "object_ids": [artefact_id],
                "after_hash": result_hash,
                "details": {"pipeline_version": pipeline_version,
                            "manifest_hash": canonical_hash(manifest) if manifest is not None else None}}, *evidence_events])
            return artefact_id

    artefact_id = audit.repeated(unit)
    ctx.state["done_payload"] = {"decision_code": result.decision.code}
    return {"result_hash": result_hash, "result_artefact_id": artefact_id, "decision_code": result.decision.code}


def existing_runs(db: Session, runs: dict[str, str]) -> dict[str, str]:
    known = set(db.scalars(select(DetectorRun.id).where(DetectorRun.id.in_(list(runs.values())))))
    return {k: v for k, v in runs.items() if v in known}


def legacy_artifact_dir(ctx: StageContext) -> Path:
    out_dir = settings.artifacts_dir / ctx.analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
