"""Artefact index and evidence-table projection (ARCHITECTURE §4.7, §4.9).

- `register_artefact` puts bytes into the object store and indexes them in `artefacts`.
- `ensure_original_in_store` performs the lazy copy of a v0.3 upload into the store (§4.9 step 2).
- `persist_result` projects an `AnalysisResultV2` into the query tables (signals, assessments, findings, exhibits).
- `delete_evidence` removes an analysis's evidence rows. Objects stay in the store: they are shared by sha256.

Nothing here commits. The caller owns the transaction.
"""
from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.evidence import AnalysisResultV2
from app.storage.db import (
    Analysis,
    AnalysisEvent,
    AnalysisStage,
    AnnotationRevisionRow,
    AnnotationRow,
    Artefact,
    AssessmentRow,
    ConclusionRow,
    DetectorRun,
    ExhibitRow,
    FindingRow,
    MediaObject,
    PackageRow,
    ReportRow,
    SignalRow,
    SignoffRow,
)
from app.storage.files import safe_filename
from app.storage.objects import CHUNK, ObjectIntegrityError, ObjectNotFound, ObjectStore, check_sha256

log = get_logger(__name__)

ARTEFACT_ROLES = frozenset({
    "original", "working_copy", "exhibit_payload", "mask", "preview", "tile", "still", "stage_output", "raw_output",
    "report", "package",
})
EXHIBIT_INLINE_LIMIT = 64 * 1024  # canonical exhibit JSON above this goes to an artefact (§4.7)


class ArtefactNotFound(LookupError):
    """No artefact with this id (or it belongs to another analysis)."""


class EvidenceConflict(RuntimeError):
    """The machine layer of this analysis is already published and immutable (RW-01)."""


def register_artefact(session: Session, store: ObjectStore, data: bytes, *, media_type: str, role: str, created_by: str,
                      analysis_id: str | None = None, artefact_id: str | None = None) -> Artefact:
    """Store bytes and index them. Idempotent on (sha256, role, analysis_id) and on an explicit `artefact_id`."""
    if role not in ARTEFACT_ROLES:
        raise ValueError(f"unknown artefact role {role!r}")
    if not media_type or not created_by:
        raise ValueError("media_type and created_by are required")
    obj = store.put_bytes(data)
    return index_object(session, obj.sha256, obj.size_bytes, media_type=media_type, role=role, created_by=created_by,
                        analysis_id=analysis_id, artefact_id=artefact_id)


def index_object(session: Session, sha256: str, size_bytes: int, *, media_type: str, role: str, created_by: str,
                 analysis_id: str | None = None, artefact_id: str | None = None) -> Artefact:
    """Index an object that is already in the store. The row is flushed, so the caller may reference it from a row it adds next
    (a media object, a stage, a detector run)."""
    if role not in ARTEFACT_ROLES:
        raise ValueError(f"unknown artefact role {role!r}")
    check_sha256(sha256)
    if artefact_id is not None:
        existing = session.get(Artefact, artefact_id)
        if existing is not None:
            if existing.sha256 != sha256:
                raise EvidenceConflict(f"artefact {artefact_id} already indexes different bytes")
            return existing
    else:
        existing = session.scalars(select(Artefact).where(
            Artefact.sha256 == sha256, Artefact.role == role, Artefact.analysis_id == analysis_id)).first()
        if existing is not None:
            return existing
    row = Artefact(id=artefact_id or str(uuid.uuid4()), sha256=sha256, bytes=size_bytes, media_type=media_type, role=role,
                   analysis_id=analysis_id, created_by=created_by)
    session.add(row)
    session.flush()
    return row


def get_artefact(session: Session, artefact_id: str, *, analysis_id: str | None = None) -> Artefact:
    """Look up an artefact. With `analysis_id`, an artefact of another analysis is reported as not found."""
    row = session.get(Artefact, artefact_id)
    if row is None or (analysis_id is not None and row.analysis_id != analysis_id):
        raise ArtefactNotFound(f"artefact {artefact_id} not found")
    return row


def verify_artefact(session: Session, store: ObjectStore, artefact_id: str, *, analysis_id: str | None = None) -> Artefact:
    """Re-hash the artefact's object. Raises ArtefactNotFound, ObjectNotFound or ObjectIntegrityError."""
    row = get_artefact(session, artefact_id, analysis_id=analysis_id)
    size = store.verify(row.sha256)
    if size != row.bytes:
        raise ObjectIntegrityError(f"artefact {artefact_id} has {size} bytes, the index records {row.bytes}")
    return row


def _hash_file(path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def ensure_original_in_store(session: Session, store: ObjectStore, media: MediaObject) -> Artefact:
    """Copy a legacy upload of this media object into the store, once. Idempotent.

    The upload file stays where v1 code reads it until the ingest stage (P-4) writes originals to the store directly.
    Raises ObjectNotFound when no upload file survives and ObjectIntegrityError when the surviving bytes do not match.
    """
    if media.storage_artefact_id is not None:
        return get_artefact(session, media.storage_artefact_id)
    analyses = session.scalars(select(Analysis).where(Analysis.media_id == media.id).order_by(Analysis.created_at)).all()
    candidates = [settings.uploads_dir / a.id / safe_filename(a.filename) for a in analyses]
    present = [p for p in candidates if p.is_file()]
    if not present:
        raise ObjectNotFound(f"no upload of media {media.id} survives on disk")
    sha, md5 = _hash_file(present[0])
    if sha != media.sha256:
        log.error("artefacts.original_hash_mismatch", media_id=media.id, expected=media.sha256, found=sha)
        raise ObjectIntegrityError(f"the upload of media {media.id} no longer matches its recorded sha256")
    obj = store.put_file(present[0])
    if obj.sha256 != sha:  # the file changed between hashing and copying
        raise ObjectIntegrityError(f"the upload of media {media.id} changed while it was copied")
    row = index_object(session, obj.sha256, obj.size_bytes, media_type=media.mime_sniffed or "application/octet-stream",
                       role="original", created_by="ingest")
    media.storage_artefact_id = row.id
    media.md5 = media.md5 or md5
    session.flush()
    return row


def persist_result(session: Session, result: AnalysisResultV2, *, store: ObjectStore,
                   detector_run_ids: Mapping[str, str] | None = None) -> None:
    """Project a v2 result into the evidence tables and the `analyses` v2 columns."""
    analysis = session.get(Analysis, result.analysis_id)
    if analysis is None:
        raise LookupError(f"analysis {result.analysis_id} not found")
    if analysis.locked_at is not None or session.scalars(
            select(SignalRow.signal_id).where(SignalRow.analysis_id == result.analysis_id).limit(1)).first() is not None:
        raise EvidenceConflict(f"analysis {result.analysis_id} already has published evidence; re-runs create a new revision")
    aid = result.analysis_id
    runs = detector_run_ids or {}

    for s in result.signals:
        session.add(SignalRow(
            analysis_id=aid, signal_id=s.id, detector_run_id=runs.get(s.detector_id), hypothesis=s.hypothesis, lane=s.lane,
            track_id=s.track_id, status=s.status, raw_score=s.raw_score, calibrated_probability=s.calibrated_probability,
            contribution_logit=s.contribution_logit, payload=s.model_dump(mode="json")))

    for a in result.assessments:
        ci = a.ci90 or (None, None)
        session.add(AssessmentRow(analysis_id=aid, hypothesis=a.hypothesis, lane=None, track_id=None, probability=a.probability,
                                  ci_low=ci[0], ci_high=ci[1], band=None, payload=a.model_dump(mode="json")))
        for lane in a.lanes:
            session.add(AssessmentRow(analysis_id=aid, hypothesis=a.hypothesis, lane=lane.lane, track_id=lane.track_id,
                                      probability=lane.probability, band=lane.band, payload=lane.model_dump(mode="json")))

    for f in result.findings:
        session.add(FindingRow(analysis_id=aid, finding_id=f.id, rank=f.rank, family=f.family, hypothesis=f.hypothesis,
                               lane=f.lane, strength=f.strength, review_priority=f.review_priority,
                               payload=f.model_dump(mode="json")))

    for e in result.exhibits:
        body = canonical_json(e)
        payload, payload_artefact_id = e.model_dump(mode="json"), None
        if len(body) > EXHIBIT_INLINE_LIMIT:
            art = register_artefact(session, store, body, media_type="application/json", role="exhibit_payload",
                                    created_by=e.created_by, analysis_id=aid)
            payload, payload_artefact_id = None, art.id
        session.add(ExhibitRow(id=e.id, analysis_id=aid, kind=e.kind, title=e.title[:255], payload=payload,
                               payload_artefact_id=payload_artefact_id, sha256=e.sha256))

    d, m = result.decision, result.manifest
    analysis.result_schema_version = 2
    analysis.result_hash = canonical_hash(result)
    analysis.revision = result.revision
    analysis.case_id = result.case_id
    analysis.decision_code = d.code
    analysis.driving_hypothesis = d.driving_hypothesis
    analysis.fusion_mode = d.fusion_mode
    analysis.fuser_sha256 = d.fuser_sha256
    analysis.slice_id = result.condition_profile.slice_id if result.condition_profile else None
    analysis.profile = m.profile
    analysis.effective_profile = m.effective_profile
    analysis.licence_posture = m.licence_posture
    analysis.pipeline_version = m.pipeline_version
    analysis.engine_version = m.veritas_version
    analysis.build_commit = m.git_commit
    analysis.config_hash = m.config_sha256
    analysis.template_set = m.template_set
    analysis.policy_thresholds_id = m.policy_thresholds_id
    session.flush()


def delete_evidence(session: Session, analysis_id: str) -> None:
    """Delete an analysis's evidence rows (children first; SQLite does not enforce ON DELETE CASCADE by default), including the
    examiner layer: annotation revisions, annotations, sign-offs and conclusions (D-47)."""
    annotation_ids = select(AnnotationRow.id).where(AnnotationRow.analysis_id == analysis_id)
    session.execute(delete(AnnotationRevisionRow).where(AnnotationRevisionRow.annotation_id.in_(annotation_ids)))
    for model in (SignalRow, AssessmentRow, FindingRow, ExhibitRow, AnalysisEvent, DetectorRun, AnalysisStage, ReportRow,
                  PackageRow, AnnotationRow, SignoffRow, ConclusionRow):
        session.execute(delete(model).where(model.analysis_id == analysis_id))
    session.execute(delete(Artefact).where(Artefact.analysis_id == analysis_id))
