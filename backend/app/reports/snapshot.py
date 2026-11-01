"""The frozen input of a report (ARCHITECTURE §6.2, §6.3 rule 1; D-25; BUILD_SPEC R-1, R-2).

- **What it holds.** The report's identity (id, D-25 display id, revision, profile, `requested_at`), the analysis with its
  detector coverage, the media identity with the intake record's received date and source fields, the case and exhibit label,
  the published v2 result document, the review state with the current conclusion and sign-offs (signer names resolved), the
  live annotations with the annotation count at request time, the custody extract of the analysis up to the request, and the
  store objects of key-finding thumbnails (finding id → SHA-256 of a heatmap preview).
- **Frozen at request time.** `POST /analyses/{id}/reports` builds it and stores its canonical JSON as a `stage_output` artefact;
  the report job renders only that stored form, so a later review change never alters a report already requested.
- **Hash.** `snapshot_hash = sha256(canonical_json(snapshot))`, unchanged by a dump and load round trip.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import BOOTSTRAP_ID, LOCAL_ID, bootstrap_principal, local_principal
from app.audit import chain as audit
from app.cases import intake as case_intake
from app.cases import review
from app.core import times
from app.core.canonical import canonical_hash, canonical_json
from app.engine import runs
from app.reports.citations import finding_heatmaps
from app.storage.db import (
    Analysis,
    AnnotationRow,
    Artefact,
    AuditEvent,
    CaseMediaRow,
    CaseRow,
    DetectorRun,
    IntakeRecordRow,
    MediaObject,
    PrincipalRow,
    ReportRow,
)
from app.storage.objects import ObjectStore, default_store

SNAPSHOT_SCHEMA = "veritas.report-snapshot/1"
Profile = Literal["summary", "standard", "forensic", "newsroom"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _without_unset(out: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Leave out keys added after a release when they are unset, so a snapshot frozen without them keeps its bytes and hash."""
    for key in keys:
        if out.get(key) is None:
            out.pop(key, None)
    return out


class SnapshotReport(_Frozen):
    id: str
    display_id: str
    revision: int
    profile: Profile
    redacted: bool
    supersedes_report_id: str | None
    revision_reason: str | None
    requested_at: datetime
    requested_by: str
    requested_by_name: str
    template_set: str
    veritas_version: str


class SnapshotCoverage(_Frozen):
    detectors_total: int
    by_status: dict[str, int]
    # N-14a: checks skipped for a reason of the server (`detector.SERVER_REASONS`), by reason; None when there is none, and
    # in snapshots frozen before N-14a
    not_run_on_server: dict[str, int] | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_server(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_unset(handler(self), "not_run_on_server")


class SnapshotAnalysis(_Frozen):
    id: str
    kind: str
    revision: int
    profile: str | None
    supersedes_id: str | None
    created_at: datetime | None
    finished_at: datetime | None
    coverage: SnapshotCoverage | None = None
    mode: str | None = None  # how this revision was made (`rerun.mode_of`): rerun, reanalyse or refusion; None for revision 1
    # K-2: the revision number of `supersedes_id`; after a cancelled or failed revision it is not `revision - 1`. None in older
    # snapshots, which render `revision - 1`
    supersedes_revision: int | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_supersedes_revision(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave the K-2 key out when unset, so a snapshot without it keeps its bytes and hash."""
        out = handler(self)
        if out.get("supersedes_revision") is None:
            out.pop("supersedes_revision", None)
        return out


class SnapshotMedia(_Frozen):
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    md5: str | None
    received_at: datetime | None = None
    acquisition_method: str | None = None
    supplied_by: str | None = None
    source: str | None = None
    # N-3: the custodian-stated hash check of `intake.custodian_check`; None in snapshots frozen before it was recorded
    custodian_hash_match: str | None = None
    custodian_stated_hash: str | None = None
    # N-19 (browser QA P1-8; design review DS-01): the rest of the intake record, as the custodian stated it; None in
    # snapshots frozen before, and when nothing was stated
    date_acquired: datetime | None = None
    intake_notes: str | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_custodian(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_unset(handler(self), "custodian_hash_match", "custodian_stated_hash", "date_acquired", "intake_notes")


class SnapshotCase(_Frozen):
    id: str
    reference: str
    title: str
    classification: str
    exhibit_label: str | None
    legal_hold: bool | None = None  # N-19: True while the case is under legal hold when the snapshot is frozen; else unset

    @model_serializer(mode="wrap")
    def _omit_unset_legal_hold(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_unset(handler(self), "legal_hold")


class SnapshotConclusion(_Frozen):
    id: str
    author_id: str
    author_name: str
    scale_position: str
    machine_code: str | None
    deviation_steps: int | None
    steps_from: str | None = None  # N-19n (D4-01): the record whose step the steps are counted from; unset before, and without one
    rationale: str
    limitations_acknowledged: list[str]
    next_steps: str | None
    declaration_text_id: str | None
    deviation_justification: str | None
    created_at: datetime
    signature_valid: bool | None = None  # N-3: the record's HMAC check when the snapshot was frozen; None: frozen before N-3

    @model_serializer(mode="wrap")
    def _omit_unset_signature_valid(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_unset(handler(self), "signature_valid", "steps_from")


class SnapshotSignoff(_Frozen):
    id: str
    kind: str
    decision: str
    principal_id: str
    principal_name: str
    comment: str | None
    signed_at: datetime
    signature_valid: bool | None = None  # N-3, as on the conclusion

    @model_serializer(mode="wrap")
    def _omit_unset_signature_valid(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_unset(handler(self), "signature_valid")


class SnapshotReview(_Frozen):
    state: str
    issuable: bool
    requires_second_review: bool | None = None  # N-3: the case policy when frozen; None: frozen before N-3
    annotations_seq: int
    conclusion: SnapshotConclusion | None
    signoffs: list[SnapshotSignoff]

    @model_serializer(mode="wrap")
    def _omit_unset_policy(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_unset(handler(self), "requires_second_review")


class SnapshotAnnotation(_Frozen):
    id: str
    author_id: str
    target_type: str
    target_id: str | None
    shape: str
    geometry: dict[str, Any] | None
    examiner_status: str | None
    body: str | None
    visibility: str
    created_at: datetime
    edited_at: datetime | None
    revision: int


class ReportSnapshot(_Frozen):
    snapshot_schema: Literal["veritas.report-snapshot/1"] = SNAPSHOT_SCHEMA
    report: SnapshotReport
    analysis: SnapshotAnalysis
    media: SnapshotMedia
    case: SnapshotCase | None
    result: dict[str, Any]
    review: SnapshotReview
    annotations: list[SnapshotAnnotation]
    custody: list[dict[str, Any]]
    thumbnails: dict[str, str] = Field(default_factory=dict)
    # Scalar/mask references for selected finding thumbnails in every profile; full profiles also freeze exhibit plates.
    # Keep the existing field/default so loading a historical snapshot does not add new canonical keys.
    exhibit_images: dict[str, str] = Field(default_factory=dict)
    # standard and forensic profiles (R-3): model cards, earlier revisions, actor names and whole-chain verification
    model_cards: list[dict[str, Any]] = Field(default_factory=list)
    revisions: list[dict[str, Any]] = Field(default_factory=list)
    actor_names: dict[str, str] = Field(default_factory=dict)
    # valid, checked, first_broken_seq, reason; since N-10a also seq and event_hash, the last event that verified (a
    # snapshot frozen before it has the four keys and keeps its bytes)
    custody_verification: dict[str, Any] | None = None
    detector_names: dict[str, str] = Field(default_factory=dict)  # names the findings quote (coverage "Not run: <name>")
    # Public, immutable packet set frozen at request time. Historical snapshots omit this added field entirely.
    image_evidence: dict[str, Any] | None = None
    # Optional non-image packet set; absent historical snapshots must retain their canonical bytes.
    evidence: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_image_evidence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _without_unset(handler(self), "image_evidence", "evidence")


FULL_PROFILES = frozenset({"standard", "forensic"})
STILLS_PER_TRACK = 6
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def snapshot_hash(snapshot: ReportSnapshot) -> str:
    return canonical_hash(snapshot)


def dump(snapshot: ReportSnapshot) -> bytes:
    return canonical_json(snapshot)


def load(data: bytes) -> ReportSnapshot:
    return ReportSnapshot.model_validate_json(data)


def _supersedes_revision(db: Session, analysis: Analysis) -> int | None:
    """Set only when it differs from `revision - 1` (after a cancelled or failed revision), so other snapshots are unchanged."""
    source = db.get(Analysis, analysis.supersedes_id) if analysis.supersedes_id else None
    number = None if source is None else source.revision or 1
    return number if number is not None and number != (analysis.revision or 1) - 1 else None


def display_id(analysis_id: str, revision: int, issuable: bool) -> str:
    """D-25, D-56: `R-<analysis id8>-<revision, 2 digits>` for a report whose snapshot is issuable, else
    `R-<analysis id8>-DRAFT-<revision>`: before the examiner's sign-off, while a required second review is outstanding, and when
    the peer review does not concur."""
    return f"R-{analysis_id[:8]}-{revision:02d}" if issuable else f"R-{analysis_id[:8]}-DRAFT-{revision}"


_utc = times.to_utc  # one form on every engine: a time is printed in UTC (D-72)


def principal_names(db: Session, ids: set[str]) -> dict[str, str]:
    """Display names of principals; the built-in principals by their fixed names, `system` as Chipmunk Verify, unknown ids as themselves."""
    names = {BOOTSTRAP_ID: bootstrap_principal().display_name, LOCAL_ID: local_principal().display_name, audit.SYSTEM_ACTOR: "Chipmunk Verify"}
    wanted = ids - set(names)
    if wanted:
        names.update({row.id: row.display_name for row in db.scalars(select(PrincipalRow).where(PrincipalRow.id.in_(wanted)))})
    return {i: names.get(i, i) for i in ids}


def thumbnail_artefacts(db: Session, analysis_id: str, result: dict[str, Any]) -> dict[str, str]:
    """Finding id → SHA-256 of the preview of the first heatmap exhibit the finding cites, directly or through a cited signal."""
    exhibits = {e["id"]: e for e in result.get("exhibits") or []}
    signals = {s["id"]: s for s in result.get("signals") or []}
    out: dict[str, str] = {}
    for finding in result.get("findings") or []:
        cited = dict.fromkeys([*finding.get("evidence_ids", []), *(i for s in finding.get("sentences") or [] for i in s.get("evidence_ids", []))])
        candidates = [i for c in cited for i in ([c] if c in exhibits else signals.get(c, {}).get("exhibit_ids", []))]
        preview = next((exhibits[i]["preview_artefact_id"] for i in candidates
                        if exhibits.get(i, {}).get("kind") == "heatmap" and exhibits[i].get("preview_artefact_id")), None)
        row = db.get(Artefact, preview) if preview is not None else None
        if row is not None and row.analysis_id == analysis_id:
            out[str(finding["id"])] = row.sha256
    return out


def exhibit_images(db: Session, analysis: Analysis, result: dict[str, Any], store: ObjectStore | None = None, *,
                   finding_thumbnails_only: bool = False) -> dict[str, str]:
    """Plate key → SHA-256 of a stored image of this analysis: `<exhibit id>` a heatmap preview or an image exhibit's payload,
    `<exhibit id>#scalar` a heatmap's declared scalar payload,
    `<exhibit id>#reliability` its reliability mask, `<exhibit id>#still-<n>` up to 6 face-track stills, and `source`
    the original of an image analysis when the object store holds it. With ``finding_thumbnails_only``, freeze only the
    selected cited maps' scalar/mask references: no preview, source photograph, still or unrelated exhibit."""
    wanted: dict[str, str] = {}
    scalar_exhibits: dict[str, dict[str, Any]] = {}
    mask_exhibits: dict[str, dict[str, Any]] = {}
    records = finding_heatmaps(result).values() if finding_thumbnails_only else result.get("exhibits") or []
    for exhibit in records:
        kind, eid = exhibit.get("kind"), str(exhibit.get("id"))
        if kind == "heatmap":
            if not finding_thumbnails_only and exhibit.get("preview_artefact_id"):
                wanted[eid] = exhibit["preview_artefact_id"]
            if exhibit.get("payload_artefact_id"):
                wanted[f"{eid}#scalar"] = exhibit["payload_artefact_id"]
                scalar_exhibits[f"{eid}#scalar"] = exhibit
            if exhibit.get("reliability_mask_artefact_id"):
                wanted[f"{eid}#reliability"] = exhibit["reliability_mask_artefact_id"]
                mask_exhibits[f"{eid}#reliability"] = exhibit
        elif kind == "image":
            wanted[eid] = exhibit["payload_artefact_id"]
        elif kind == "face_track":
            stills = [f["still_artefact_id"] for f in exhibit.get("frames") or [] if f.get("still_artefact_id")]
            wanted.update({f"{eid}#still-{n}": artefact for n, artefact in enumerate(stills[:STILLS_PER_TRACK], 1)})
    out: dict[str, str] = {}
    for key, artefact_id in sorted(wanted.items()):
        row = db.get(Artefact, artefact_id)
        if row is None or row.analysis_id != analysis.id:
            continue
        if key in scalar_exhibits:
            exhibit = scalar_exhibits[key]
            if (row.sha256 != exhibit.get("sha256") or row.role != "exhibit_payload" or row.media_type != "image/png"
                    or row.created_by != exhibit.get("created_by")):
                continue
        if key in mask_exhibits and (row.role != "mask" or row.media_type != "image/png"
                                    or row.created_by != mask_exhibits[key].get("created_by")):
            continue
        out[key] = row.sha256
    if not finding_thumbnails_only:
        store = store or default_store()
        if analysis.kind == "image" and store.exists(analysis.sha256):
            out["source"] = analysis.sha256
    return out


def admission_words(admission: Any) -> str | None:
    """The card decision; a rejected / retired detector that the served fusion model still holds (D-51 `served_until`) says so."""
    if admission is None:
        return None
    if admission.served_until and admission.decision in ("rejected", "retired"):
        return f"{admission.decision}; still part of the served fusion model {admission.served_until} until its refit"
    return admission.decision


def model_cards(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Card extracts of the detectors in the result's manifest (VG-01): name, version, citation, source, coverage date, admission,
    measured rows, known false alarms and limitations, and licences of their models and transitive dependencies.
    Unknown detector ids are left out; stored snapshots are never refreshed by this function."""
    from app.engine.registry import get_registry

    registry = get_registry()
    manifest = registry.manifest
    out = []
    for ref in sorted({str(d["id"]) for d in (result.get("manifest") or {}).get("detectors") or []}):
        det = manifest.detector(ref)
        if det is None:
            continue
        models = [m for m in (manifest.model(i) for i in registry.required_models(det)) if m is not None]
        out.append({"id": det.id, "name": det.name, "version": det.version, "citation": det.citation, "source_url": det.source_url,
                    "coverage_date": det.coverage_date, "admission": admission_words(det.admission),
                    "admission_decision": det.admission.decision if det.admission else None,
                    "admission_date": det.admission.date if det.admission else None,
                    "eval": [row.model_dump(mode="json") for row in det.eval], "fp_modes": list(det.fp_modes),
                    "known_limitations": list(det.known_limitations),
                    "licences": [{"model": m.id, "spdx": m.licence.spdx, "class": m.licence.class_} for m in models]})
    return out


def revisions(db: Session, analysis_id: str, profile: str, revision: int) -> list[dict[str, Any]]:
    """The earlier report revisions of this analysis and profile, oldest first (RP-06)."""
    rows = db.scalars(select(ReportRow).where(ReportRow.analysis_id == analysis_id, ReportRow.profile == profile,
                                              ReportRow.revision < revision).order_by(ReportRow.revision))
    return [{"id": r.id, "display_id": r.display_id, "revision": r.revision, "requested_at": _utc(r.requested_at).strftime(TS_FORMAT),  # type: ignore[union-attr]
             "review_state": r.review_state, "issuable": bool(r.issuable), "revision_reason": r.revision_reason, "status": r.status,
             "sha256": r.sha256} for r in rows]


def custody_verification(db: Session, verified: audit.Verification | None = None) -> dict[str, Any]:
    """The whole audit chain verified when the report is requested (CC-02), read through the request's own session: one
    connection per request (N-9a; Postgres QA X-2: a second session per request used up the pool under 15 requests).

    `verified` is a walk the request made before it took the revision's lock and the append lock (N-10a; code
    re-verification R2-02: the whole walk under those locks stopped every other writer for its length). Only the events
    appended since are hashed here, from the seq and hash that walk reached, so what is recorded is still the whole chain
    as it stands while the snapshot is frozen. `seq` and `event_hash` name the last event that verified."""
    result = audit.verify_chain(db=db, after=verified)
    return {"valid": result.valid, "checked": result.checked, "first_broken_seq": result.first_broken_seq, "reason": result.reason,
            "seq": result.seq, "event_hash": result.head}


def coverage(db: Session, analysis: Analysis, result: dict[str, Any]) -> SnapshotCoverage:
    """Detector run statuses (the analysis detail's `coverage`); a v1 result counts its signals."""
    # one run per detector (N-8); a detector the result names whose run could not be stored counts as failed (N-10a)
    statuses = runs.statuses(db, analysis.id, result if analysis.result_schema_version == 2 else None)
    if not statuses and analysis.result_schema_version == 1:
        statuses = [str(s.get("status", "ok")) for s in result.get("signals") or []]
    on_server = runs.not_run_on_server(db, analysis.id) if analysis.result_schema_version == 2 else {}
    return SnapshotCoverage(detectors_total=len(statuses), by_status=dict(sorted(Counter(statuses).items())),
                            not_run_on_server=on_server or None)


def _mode(db: Session, analysis: Analysis) -> str | None:
    from app.engine import rerun

    return rerun.mode_of(db, analysis.id) if analysis.supersedes_id else None


def review_of(db: Session, analysis: Analysis) -> SnapshotReview:
    """The review of a revision as a report freezes it: state, the issuable flag, the current conclusion and the sign-offs, each
    with the result of its signature check (N-3). Also read by the v1 report for its status line."""
    brief = review.brief(db, analysis)
    signoffs = review.signoffs(db, analysis.id)
    conclusion = review.current_conclusion(db, analysis.id, signoffs)
    names = principal_names(db, {*(s.principal_id for s in signoffs), *([conclusion.author_id] if conclusion else [])})
    return SnapshotReview(
        state=brief["state"], issuable=brief["issuable"], requires_second_review=brief["requires_second_review"],
        annotations_seq=review.annotations_seq(db, analysis.id),
        conclusion=SnapshotConclusion(
            id=conclusion.id, author_id=conclusion.author_id, author_name=names[conclusion.author_id],
            scale_position=conclusion.scale_position, machine_code=conclusion.machine_code, deviation_steps=conclusion.deviation_steps,
            steps_from=conclusion.steps_from,
            rationale=conclusion.rationale, limitations_acknowledged=list(conclusion.limitations_acknowledged or []),
            next_steps=conclusion.next_steps, declaration_text_id=conclusion.declaration_text_id,
            deviation_justification=conclusion.deviation_justification, created_at=_utc(conclusion.created_at),  # type: ignore[arg-type]
            signature_valid=review.conclusion_valid(conclusion))
        if conclusion is not None else None,
        signoffs=[SnapshotSignoff(id=s.id, kind=s.kind, decision=s.decision, principal_id=s.principal_id, principal_name=names[s.principal_id],
                                  comment=s.comment, signed_at=_utc(s.signed_at), signature_valid=review.signoff_valid(s))  # type: ignore[arg-type]
                  for s in signoffs])


def build(db: Session, analysis: Analysis, result: dict[str, Any], *, report_id: str, revision: int, profile: str, redacted: bool,
          supersedes_report_id: str | None, revision_reason: str | None, requested_at: datetime, requested_by: str,
          template_set: str, veritas_version: str, verified: audit.Verification | None = None,
          store: ObjectStore | None = None) -> ReportSnapshot:
    from app.reports.image_evidence import capture
    from app.reports.multimedia_evidence import capture as capture_multimedia

    frozen_review = review_of(db, analysis)
    names = principal_names(db, {requested_by})
    media = db.get(MediaObject, analysis.media_id) if analysis.media_id else None
    case = db.get(CaseRow, analysis.case_id) if analysis.case_id else None
    label = (db.scalars(select(CaseMediaRow.label).where(CaseMediaRow.case_id == case.id, CaseMediaRow.media_id == analysis.media_id)).first()
             if case is not None and analysis.media_id else None)
    intake = (db.scalars(select(IntakeRecordRow).where(IntakeRecordRow.case_id == case.id, IntakeRecordRow.media_id == analysis.media_id)
                         .order_by(IntakeRecordRow.intake_at, IntakeRecordRow.id)).first() if case is not None and analysis.media_id else None)
    annotations = db.scalars(select(AnnotationRow).where(AnnotationRow.analysis_id == analysis.id, AnnotationRow.deleted_at.is_(None))
                             .order_by(AnnotationRow.created_at, AnnotationRow.id))
    custody = [audit.exported(row) for row in db.scalars(select(AuditEvent).where(AuditEvent.analysis_id == analysis.id).order_by(AuditEvent.seq))]
    annotations = list(annotations)
    custodian = case_intake.custodian_check(db, analysis.case_id, analysis.media_id)
    full = profile in FULL_PROFILES
    return ReportSnapshot(
        report=SnapshotReport(id=report_id, display_id=display_id(analysis.id, revision, frozen_review.issuable), revision=revision,
                              profile=profile, redacted=redacted, supersedes_report_id=supersedes_report_id,  # type: ignore[arg-type]
                              revision_reason=revision_reason, requested_at=requested_at, requested_by=requested_by,
                              requested_by_name=names[requested_by], template_set=template_set, veritas_version=veritas_version),
        analysis=SnapshotAnalysis(id=analysis.id, kind=analysis.kind, revision=analysis.revision or 1, profile=analysis.profile,
                                  supersedes_id=analysis.supersedes_id, created_at=_utc(analysis.created_at), finished_at=_utc(analysis.finished_at),
                                  coverage=coverage(db, analysis, result), mode=_mode(db, analysis),
                                  supersedes_revision=_supersedes_revision(db, analysis)),
        media=SnapshotMedia(filename=analysis.filename, content_type=analysis.content_type, size_bytes=analysis.size_bytes,
                            sha256=analysis.sha256, md5=media.md5 if media is not None else None,
                            received_at=_utc(intake.intake_at) if intake is not None else None,
                            acquisition_method=intake.acquisition_method if intake is not None else None,
                            supplied_by=intake.supplied_by if intake is not None else None, source=intake.source if intake is not None else None,
                            custodian_hash_match=custodian.custodian_hash_match if custodian is not None else None,
                            custodian_stated_hash=custodian.custodian_stated_hash if custodian is not None else None,
                            date_acquired=_utc(intake.date_acquired) if intake is not None else None,
                            intake_notes=(intake.notes or None) if intake is not None else None),
        case=SnapshotCase(id=case.id, reference=case.reference, title=case.title, classification=case.classification,
                          exhibit_label=label, legal_hold=True if case.legal_hold else None) if case is not None else None,
        result=result,
        review=frozen_review,
        annotations=[SnapshotAnnotation(id=a.id, author_id=a.author_id, target_type=a.target_type, target_id=a.target_id, shape=a.shape,
                                        geometry=a.geometry, examiner_status=a.examiner_status, body=a.body, visibility=a.visibility,
                                        created_at=_utc(a.created_at), edited_at=_utc(a.edited_at), revision=a.revision)  # type: ignore[arg-type]
                     for a in annotations],
        custody=custody,
        thumbnails=thumbnail_artefacts(db, analysis.id, result),
        exhibit_images=exhibit_images(db, analysis, result, store=store, finding_thumbnails_only=not full),
        model_cards=model_cards(result) if full else [],
        revisions=revisions(db, analysis.id, profile, revision) if full else [],
        actor_names=principal_names(db, {str(e["actor_id"]) for e in custody} | {a.author_id for a in annotations}) if full else {},
        custody_verification=custody_verification(db, verified) if full else None,
        detector_names=detector_names(db, analysis.id, result) if full else {},
        image_evidence=capture(db, analysis, store=store),
        evidence=capture_multimedia(db, analysis, result=result, store=store),
    )


def detector_names(db: Session, analysis_id: str, result: dict[str, Any]) -> dict[str, str]:
    """Display names of the detectors in the manifest and the detector runs of this analysis."""
    from app.engine.registry import get_registry

    registry = get_registry()
    ids = {str(d["id"]) for d in (result.get("manifest") or {}).get("detectors") or []}
    ids |= set(db.scalars(select(DetectorRun.detector_id).where(DetectorRun.analysis_id == analysis_id)))
    return {i: name for i in sorted(ids) if (name := registry.detector_name(i))}
