"""Persistence for analyses, the v2 evidence tables and the job queue. SQLite by default; Postgres via VERITAS_DATABASE_URL.

The schema is managed by Alembic (`alembic/versions/`). These models must match the migrations exactly:
`tests/test_migrations.py` compares them.

Insert order (D-72). The models declare no `relationship()`, so within one flush the unit of work inserts the new rows of
different tables in the order of their class names, not of their foreign keys: `AnnotationRevisionRow` before `AnnotationRow`,
`BatchItemRow` before `BatchRow`, `CaseMemberRow` and `CaseMediaRow` before `CaseRow`, `IntakeRecordRow` before `MediaObject`,
`AnalysisStage` before `Artefact`, `WebhookDeliveryRow` before `WebhookRow`. Both engines refuse a child whose parent row is not
there yet, so a site that creates a parent and its child in one transaction calls `flush()` after adding the parent (or gets
the parent from a helper that flushes: `artefacts.index_object`, `intake.new_case`) before it adds the child.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    event,
    false,
    make_url,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core import times
from app.core.config import settings


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC)


# K-2: statuses of a revision that never published and never will; such a successor does not supersede its source.
UNPUBLISHED_TERMINAL = ("cancelled", "failed")
UNPUBLISHED_SUCCESSOR_SQL = "status NOT IN ('cancelled', 'failed')"


def _analysis_fk() -> ForeignKey:
    return ForeignKey("analyses.id", ondelete="CASCADE")


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_case_id_created_at", "case_id", "created_at"),
        Index("ix_analyses_media_id", "media_id"),
        # the join of the recursive supersession query (`review.state_expression`; migration 0021): the partial index below cannot
        # serve it
        Index("ix_analyses_supersedes_id", "supersedes_id"),
        # one live newer revision per revision (D-47, migration 0014); a cancelled or failed one never published and leaves the
        # slot free (K-2, migration 0015)
        Index("ux_analyses_supersedes_id", "supersedes_id", unique=True, sqlite_where=text(UNPUBLISHED_SUCCESSOR_SQL),
              postgresql_where=text(UNPUBLISHED_SUCCESSOR_SQL)),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # image | video | document
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)  # migration 0018: a file of 2 GiB or more
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default="queued")  # queued|running|done|failed|cancelled
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    probability_ai: Mapped[float | None] = mapped_column(Float, nullable=True)
    band: Mapped[str | None] = mapped_column(String(32), nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # --- v2 (migration 0002, ARCHITECTURE §4.7) ---
    case_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    media_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    supersedes_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    profile: Mapped[str | None] = mapped_column(String(16), nullable=True)
    effective_profile: Mapped[str | None] = mapped_column(String(16), nullable=True)
    licence_posture: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    engine_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    build_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manifest_lock_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fuser_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fusion_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    template_set: Mapped[str | None] = mapped_column(String(32), nullable=True)
    policy_thresholds_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    result_schema_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    driving_hypothesis: Mapped[str | None] = mapped_column(String(40), nullable=True)
    slice_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    progress: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    submitted_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # --- review (migration 0009, D-27 `withdrawn`) ---
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    withdrawn_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    withdrawn_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # --- re-run pins (migration 0013, CC-03): fusion_mode, fuser_sha256, policy_thresholds_id, profile, devices ---
    rerun_pins: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    @property
    def result(self) -> dict | None:
        return json.loads(self.result_json) if self.result_json else None

    def to_summary(self) -> dict:
        iso = times.iso_z

        return {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "status": self.status,
            "created_at": iso(self.created_at),
            "started_at": iso(self.started_at),
            "finished_at": iso(self.finished_at),
            "error": self.error,
            "probability_ai": self.probability_ai,
            "band": self.band,
            "has_report": bool(self.report_path),
        }


class BillingOrderRow(Base):
    """One fixed-price purchase; no analysis is queued until verified payment fulfillment."""
    __tablename__ = "billing_orders"
    __table_args__ = (Index("ux_billing_owner_request", "owner_id", "request_key", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(36), index=True)
    request_key: Mapped[str] = mapped_column(String(64))
    case_id: Mapped[str] = mapped_column(String(36))
    media_id: Mapped[str] = mapped_column(String(36))
    kind: Mapped[str] = mapped_column(String(16))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    profile: Mapped[str] = mapped_column(String(16))
    amount_minor: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3))
    mode: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(24), index=True)
    stripe_session_id: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    checkout_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    payment_intent_id: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    analysis_id: Mapped[str | None] = mapped_column(String(36), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BillingEventRow(Base):
    """Processed Stripe event IDs, committed in the same transaction as fulfillment."""
    __tablename__ = "billing_events"
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(36), index=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Artefact(Base):
    """Index row for one object in the content-addressed store (several rows may share a sha256)."""
    __tablename__ = "artefacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    bytes: Mapped[int] = mapped_column(BigInteger)
    media_type: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(16))
    analysis_id: Mapped[str | None] = mapped_column(String(36), _analysis_fk(), index=True, nullable=True)
    created_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MediaObject(Base):
    __tablename__ = "media_objects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    md5: Mapped[str | None] = mapped_column(String(32), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    mime_sniffed: Mapped[str | None] = mapped_column(String(100), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    storage_artefact_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("artefacts.id"), nullable=True, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisStage(Base):
    __tablename__ = "analysis_stages"

    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk(), primary_key=True)
    stage: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(16))
    attempt: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    output_artefact_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("artefacts.id"), nullable=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DetectorRun(Base):
    __tablename__ = "detector_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk(), index=True)
    detector_id: Mapped[str] = mapped_column(String(100))
    detector_version: Mapped[str] = mapped_column(String(32))
    model_revisions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    device: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dtype: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peak_memory_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_output_artefact_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("artefacts.id"), nullable=True, index=True)


class SignalRow(Base):
    """Machine layer; immutable after publish (RW-01). `payload` is the full SignalV2 JSON."""
    __tablename__ = "signals"
    __table_args__ = (Index("ix_signals_analysis_id_hypothesis_lane", "analysis_id", "hypothesis", "lane"),)

    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk(), primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    detector_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("detector_runs.id", ondelete="CASCADE"), nullable=True, index=True)
    hypothesis: Mapped[str | None] = mapped_column(String(40), nullable=True)
    lane: Mapped[str | None] = mapped_column(String(16), nullable=True)
    track_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    raw_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    contribution_logit: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class AssessmentRow(Base):
    """One row per hypothesis assessment (lane NULL) and one per H2 lane assessment."""
    __tablename__ = "assessments"
    __table_args__ = (Index("ix_assessments_analysis_id_hypothesis", "analysis_id", "hypothesis"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk())
    hypothesis: Mapped[str] = mapped_column(String(40))
    lane: Mapped[str | None] = mapped_column(String(16), nullable=True)
    track_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    band: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class FindingRow(Base):
    __tablename__ = "findings"

    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk(), primary_key=True)
    finding_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    family: Mapped[str] = mapped_column(String(48))
    hypothesis: Mapped[str | None] = mapped_column(String(40), nullable=True)
    lane: Mapped[str | None] = mapped_column(String(16), nullable=True)
    strength: Mapped[str] = mapped_column(String(16))
    review_priority: Mapped[str] = mapped_column(String(8))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ExhibitRow(Base):
    """Exhibit metadata; a canonical payload larger than 64 KB lives in an artefact instead of `payload`."""
    __tablename__ = "exhibits"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk(), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(255))
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    payload_artefact_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("artefacts.id"), nullable=True, index=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AnalysisEvent(Base):
    """Progress / SSE events (§3.3); not the audit log."""
    __tablename__ = "analysis_events"
    __table_args__ = (Index("ix_analysis_events_analysis_id_seq", "analysis_id", "seq"),)

    seq: Mapped[int] = mapped_column(BigInteger().with_variant(Integer(), "sqlite"), primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk())
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class FuserRow(Base):
    """Fuser artefacts in use (ARCHITECTURE §4.7 `fusers`): one row per artefact sha256, written when it is first used."""

    __tablename__ = "fusers"
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    modality: Mapped[str] = mapped_column(String(16))
    version: Mapped[str] = mapped_column(String(64))
    bench_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FuserReviewRow(Base):
    """The review of a fuser artefact as the custody log names it last (migration 0019; N-9, D-57): one row per artefact
    sha256, written with its `fuser_review_recorded` or `fuser_review_withdrawn` event in one transaction
    (`app.engine.fusion.review_log`). `review_sha256` is None after a withdrawal."""

    __tablename__ = "fuser_reviews"
    artefact_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    review_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    statement_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    audit_event_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PrincipalRow(Base):
    """Actors (§4.7 `principals`; CM-02, PL-04; migration 0005): one role each, an Argon2id password hash for people and an
    HMAC API-key hash (`<key id>:<hex>`) when a key was issued. Written through `app.api.auth` and `app.api.v2_auth`."""

    __tablename__ = "principals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(16))  # administrator | examiner | reviewer | viewer | api_client
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    api_key_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SessionRow(Base):
    """Browser sessions (§4.7 `sessions`; migration 0005): the id is the SHA-256 of the cookie value, never the value."""

    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    principal_id: Mapped[str] = mapped_column(String(36), ForeignKey("principals.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    csrf_secret: Mapped[str] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(256), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CaseRow(Base):
    """Cases (§4.7 `cases`; CM-01; migration 0006). `implicit` marks the one-item case a quick analysis creates."""

    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    reference: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    matter_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    classification: Mapped[str] = mapped_column(String(16))  # public | internal | confidential | restricted
    case_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    requires_second_review: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    status: Mapped[str] = mapped_column(String(16))  # open | in_review | signed_off | closed
    legal_hold: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    implicit: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_by: Mapped[str] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CaseMemberRow(Base):
    """Per-case access (§4.7 `case_members`; CM-02; migration 0006): one case role per principal (owner | examiner |
    reviewer | viewer)."""

    __tablename__ = "case_members"
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True)
    principal_id: Mapped[str] = mapped_column(String(36), ForeignKey("principals.id", ondelete="CASCADE"), primary_key=True,
                                              index=True)
    role: Mapped[str] = mapped_column(String(16))
    added_by: Mapped[str] = mapped_column(String(36))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CaseMediaRow(Base):
    """The items of a case (BUILD_SPEC C-2 `case_media`; migration 0006): a media object once per case, with its exhibit label."""

    __tablename__ = "case_media"
    __table_args__ = (Index("ix_case_media_case_id_label", "case_id", "label", unique=True),)
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True)
    media_id: Mapped[str] = mapped_column(String(36), ForeignKey("media_objects.id"), primary_key=True, index=True)
    label: Mapped[str] = mapped_column(String(8))
    added_by: Mapped[str] = mapped_column(String(36))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IntakeRecordRow(Base):
    """One row per arrival of media into a case (§4.7 `intake_records`; CC-01; migration 0006). `client_sha256` is the
    browser pre-hash (UX §5.1) when the client sent one; `audit_event_seq` is filled from C-3."""

    __tablename__ = "intake_records"
    __table_args__ = (Index("ix_intake_records_case_id_intake_at", "case_id", "intake_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    media_id: Mapped[str] = mapped_column(String(36), ForeignKey("media_objects.id"), index=True)
    case_id: Mapped[str] = mapped_column(String(36), ForeignKey("cases.id", ondelete="CASCADE"))
    supplied_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    acquisition_method: Mapped[str] = mapped_column(String(32))
    source: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    date_acquired: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    custodian_stated_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    custodian_hash_match: Mapped[str] = mapped_column(String(16))  # match | mismatch | not_provided
    client_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    intake_principal_id: Mapped[str] = mapped_column(String(36))
    intake_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    audit_event_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class AuditEvent(Base):
    """Custody and administrative audit log (§4.7 `audit_events`, §4.8; CC-02; migration 0007). Append-only: written by
    `app.audit.chain.record` only, and database triggers abort UPDATE and DELETE."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_case_id_seq", "case_id", "seq"),)
    seq: Mapped[int] = mapped_column(BigInteger().with_variant(Integer(), "sqlite"), primary_key=True, autoincrement=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    actor_id: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(32))
    case_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    media_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    report_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    object_ids: Mapped[list[str]] = mapped_column(JSON)
    before_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    after_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON)
    prev_hash: Mapped[str] = mapped_column(String(64))
    event_hash: Mapped[str] = mapped_column(String(64))


ANNOTATION_NOTE_RULE = ("examiner_status IS NULL OR examiner_status NOT IN ('disputed', 'not_relevant') "
                        "OR length(trim(coalesce(body, ''))) > 0")


class AnnotationRow(Base):
    """Examiner layer (§4.7 `annotations`; RW-01, RW-02; migration 0008): statuses, notes and geometry in source-media
    coordinates. Soft-deleted. Every state, the first included, is kept in `annotation_revisions`. Written through
    `app.api.v2_annotations` only; the machine layer is never touched."""

    __tablename__ = "annotations"
    __table_args__ = (CheckConstraint(ANNOTATION_NOTE_RULE, name="ck_annotations_note_required"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk(), index=True)
    case_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    author_id: Mapped[str] = mapped_column(String(64))
    target_type: Mapped[str] = mapped_column(String(16))  # finding | signal | exhibit | region | interval | sentence | item | case
    target_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    shape: Mapped[str] = mapped_column(String(16))  # box | polygon | arrow | text | interval | sentence | none
    geometry: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    examiner_status: Mapped[str | None] = mapped_column(String(16), nullable=True)  # accepted | disputed | needs_check | not_relevant
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    visibility: Mapped[str] = mapped_column(String(16), default="internal", server_default="internal")  # internal | in_report
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, default=1, server_default="1")


class AnnotationRevisionRow(Base):
    """One state of an annotation (§4.7 `annotation_revisions`; migration 0008): revision 1 is the created state."""

    __tablename__ = "annotation_revisions"
    annotation_id: Mapped[str] = mapped_column(String(36), ForeignKey("annotations.id", ondelete="CASCADE"), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    examiner_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    geometry: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    visibility: Mapped[str] = mapped_column(String(16))
    edited_by: Mapped[str] = mapped_column(String(64))
    edited_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConclusionRow(Base):
    """An examiner conclusion (§4.7 `conclusions`; RW-03; migration 0009). Each PUT adds a row; the newest is current.
    `version` numbers an analysis's conclusions 1, 2, … (L-5; migration 0016). `signature` is an HMAC-SHA256 over the signed
    fields (`app.cases.review.conclusion_body`)."""

    __tablename__ = "conclusions"
    __table_args__ = (Index("ux_conclusions_analysis_id_version", "analysis_id", "version", unique=True),)  # L-5; migration 0016
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk(), index=True)
    analysis_revision: Mapped[int] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer)
    author_id: Mapped[str] = mapped_column(String(64))
    scale_position: Mapped[str] = mapped_column(String(32))  # a decision code on the five-step scale, or no_opinion
    machine_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rationale: Mapped[str] = mapped_column(Text)
    limitations_acknowledged: Mapped[list[str]] = mapped_column(JSON)
    next_steps: Mapped[str | None] = mapped_column(Text, nullable=True)
    declaration_text_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    deviation_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # N-19n (D4-01; migration 0023): the record whose step on the scale the steps are counted from (`review.STEPS_FROM`);
    # NULL: from the decision code, as every conclusion recorded before, and as one beside a result that no record decides
    steps_from: Mapped[str | None] = mapped_column(String(32), nullable=True)
    deviation_justification: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    signature: Mapped[str] = mapped_column(String(64))


class SignoffRow(Base):
    """A sign-off (§4.7 `signoffs`; RW-04; migration 0009): the examiner's locks the revision; a peer review follows it."""

    __tablename__ = "signoffs"
    __table_args__ = (Index("ux_signoffs_analysis_id_kind", "analysis_id", "kind", unique=True),)  # D-47; migration 0014
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk(), index=True)
    analysis_revision: Mapped[int] = mapped_column(Integer)
    conclusion_id: Mapped[str] = mapped_column(String(36), ForeignKey("conclusions.id"), index=True)
    principal_id: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))  # examiner | peer_review
    decision: Mapped[str] = mapped_column(String(32))  # concur | concur_with_changes | non_concur
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    annotations_seq: Mapped[int] = mapped_column(Integer)
    signed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    audit_event_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    signature: Mapped[str] = mapped_column(String(64))


class ReportRow(Base):
    """A report revision (§4.7 `reports`, §6; RP-03, RP-06; D-10, D-25; migration 0010). The snapshot is frozen when the report
    is requested; the `report` job renders it. Immutable once `done`. Written through `app.reports.service` only. `issuable` is
    the snapshot's frozen flag (D-56; migration 0017): it, not the state, decides the id, the word "issued" and the issue gate."""

    __tablename__ = "reports"
    __table_args__ = (
        Index("ix_reports_analysis_id_revision", "analysis_id", "revision"),
        Index("ix_reports_analysis_id_profile_revision", "analysis_id", "profile", "revision", unique=True),
    )
    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # VR-<year>-<6 digits>
    display_id: Mapped[str] = mapped_column(String(48))  # D-25, D-56: R-<analysis id8>-DRAFT-<rev> until issuable, then R-<id8>-<rev:02>
    revision: Mapped[int] = mapped_column(Integer)  # per (analysis, profile)
    supersedes_report_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    revision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk())
    analysis_revision: Mapped[int] = mapped_column(Integer)
    case_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    profile: Mapped[str] = mapped_column(String(16))  # summary | standard | forensic | newsroom
    redacted: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    review_state: Mapped[str] = mapped_column(String(16))  # D-27 displayed state at request time
    issuable: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())  # D-56, frozen at request time
    template_set: Mapped[str] = mapped_column(String(32))
    engine: Mapped[str] = mapped_column(String(16))  # weasyprint
    engine_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16))  # queued | running | done | failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot_hash: Mapped[str] = mapped_column(String(64))
    snapshot_artefact_id: Mapped[str] = mapped_column(String(36))
    artefact_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    signature: Mapped[str | None] = mapped_column(Text, nullable=True)  # Ed25519, base64 (R-4)
    signing_key_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    requested_by: Mapped[str] = mapped_column(String(64))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    audit_event_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class SigningKeyRow(Base):
    """The public half of an instance Ed25519 key (§4.7 `signing_keys`; RP-06, D-11; migration 0011). The private key stays in
    `data/keys/instance_ed25519.pem`. Written through `app.reports.signing` only."""

    __tablename__ = "signing_keys"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # ed25519:<first 16 hex of sha256(raw public key)>
    public_key: Mapped[bytes] = mapped_column(LargeBinary(32))  # raw 32 bytes
    key_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ReportCounterRow(Base):
    """The last report number given in a year (migration 0020; N-9a): `VR-<year>-<last>`. A delete does not lower it, so an id
    is given once (`app.reports.service.next_report_id`)."""

    __tablename__ = "report_counters"
    year: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    last: Mapped[int] = mapped_column(Integer)


class PackageRow(Base):
    """An evidence-package export (§4.7 `packages`, §6.5; RP-05, D-27, D-30; migration 0011). Immutable once `done`. Written
    through `app.reports.package` only."""

    __tablename__ = "packages"
    __table_args__ = (Index("ix_packages_analysis_id", "analysis_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(36), _analysis_fk())
    report_id: Mapped[str] = mapped_column(String(32))
    case_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    kind: Mapped[str] = mapped_column(String(8))  # draft | issue (D-27)
    schema_id: Mapped[str] = mapped_column(String(64), default="veritas.evidence/2.0",
                                         server_default="veritas.evidence/2.0")
    include_original: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    status: Mapped[str] = mapped_column(String(16))  # queued | running | done | failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    artefact_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)  # of the ZIP
    manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signature: Mapped[str | None] = mapped_column(Text, nullable=True)  # Ed25519 over manifest.sha256, base64
    signing_key_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    audit_event_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class BatchRow(Base):
    """A batch submission (§4.7 `batches`, §5.2; API-01; migration 0012). Written through `app.batches.service` only."""

    __tablename__ = "batches"
    __table_args__ = (Index("ix_batches_created_at", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(36))
    created_by: Mapped[str] = mapped_column(String(64))
    profile: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(16))  # upload | archive | local_path
    source_ref: Mapped[str | None] = mapped_column(Text, nullable=True)  # the archive's filename or the resolved local path
    status: Mapped[str] = mapped_column(String(16))  # expanding | done | failed (reads add running)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expanded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BatchItemRow(Base):
    """One file of a batch and its intake outcome (§4.7 `batch_items`; API-01; migration 0012)."""

    __tablename__ = "batch_items"
    __table_args__ = (
        Index("ix_batch_items_batch_id_seq", "batch_id", "seq", unique=True),
        Index("ix_batch_items_analysis_id", "analysis_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(36), ForeignKey("batches.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer)  # 0-based position in the request, archive or directory listing
    source_name: Mapped[str] = mapped_column(Text)
    kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(16))  # staged | rejected | submitted
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WebhookRow(Base):
    """A user-configured webhook endpoint (§4.7 `webhooks`; API-02; migration 0012). The signing secret is derived from the instance
    HMAC key and `secret_nonce`; only its hash is stored. Written through `app.webhooks.service` only."""

    __tablename__ = "webhooks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    url: Mapped[str] = mapped_column(Text)
    events: Mapped[list[str]] = mapped_column(JSON)
    secret_nonce: Mapped[str] = mapped_column(String(64))
    secret_hash: Mapped[str] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class WebhookDeliveryRow(Base):
    """One delivery attempt of one event to one webhook (§4.7 `webhook_deliveries`; API-02; migration 0012)."""

    __tablename__ = "webhook_deliveries"
    __table_args__ = (
        Index("ix_webhook_deliveries_hook_id_created_at", "hook_id", "created_at"),
        Index("ix_webhook_deliveries_event_id", "event_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    hook_id: Mapped[str] = mapped_column(String(36), ForeignKey("webhooks.id", ondelete="CASCADE"))
    event: Mapped[str] = mapped_column(String(32))
    event_id: Mapped[str] = mapped_column(String(36))
    payload: Mapped[str] = mapped_column(Text)
    payload_sha256: Mapped[str] = mapped_column(String(64))
    attempt: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))  # pending | delivered | failed (retry scheduled) | abandoned
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class PolicyThresholdsRow(Base):
    """One version of the workspace policy thresholds (§4.4 "Abstention"; API-03; migration 0013). Never updated: a change adds the
    next version and records `policy_changed` (`app.policy.thresholds`)."""

    __tablename__ = "policy_thresholds"
    __table_args__ = (Index("ix_policy_thresholds_version", "version", unique=True),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    version: Mapped[int] = mapped_column(Integer)
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSON)  # alpha_strong, alpha_moderate, min_slice_auc
    sha256: Mapped[str] = mapped_column(String(64))  # canonical JSON of `thresholds`
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Job(Base):
    """DB job queue with leases (§3.4; migration 0003). Written through `app.engine.queue` only."""
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status_priority_created_at", "status", "priority", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    type: Mapped[str] = mapped_column(String(32))  # analysis | report | package | refusion | webhook_delivery | batch_expand
    subject_id: Mapped[str] = mapped_column(String(64), index=True)
    priority: Mapped[int] = mapped_column(Integer, default=100, server_default="100")  # interactive 100, batch 10
    status: Mapped[str] = mapped_column(String(16))  # queued | leased | done | failed | cancelled
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    lease_owner: Mapped[str | None] = mapped_column(String(100), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    not_before: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WorkerHeartbeatRow(Base):
    """One row per running worker (migration 0022; N-13), written through `app.engine.heartbeat` only: the worker says that
    it lives and what it runs on, where the API reads it. `ready`: every thread reached the database lately. `state`: the
    worker's fusion state and what else the probes print of it."""

    __tablename__ = "worker_heartbeats"
    worker_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(255))
    pid: Mapped[int] = mapped_column(Integer)
    in_process: Mapped[bool] = mapped_column(Boolean)  # the worker threads of an API process
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    beat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    device: Mapped[str] = mapped_column(String(32))
    profile: Mapped[str] = mapped_column(String(16))
    threads: Mapped[int] = mapped_column(Integer)
    ready: Mapped[bool] = mapped_column(Boolean)
    state: Mapped[dict[str, Any]] = mapped_column(JSON)


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


@event.listens_for(Engine, "connect")
def _enforce_sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    """Every SQLite connection of the process enforces its foreign keys, as Postgres always does (D-72): an insert of a child
    before its parent, or a delete that leaves a reference behind, fails in the tests on SQLite instead of in service on Postgres.
    The listener is on the engine class, so an engine a test or a command makes itself is covered. Migrations switch the
    enforcement off for their own connection (`migrations.migration`)."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")


SQLITE_BUSY_TIMEOUT_MS = 30_000
# What the write-ahead log keeps on disk once a checkpoint has reset it (N-12a; code re-check R3-01). A read that stays open
# holds the log back from being reset, so a busy writer grows it without bound meanwhile (427 MB in 20 s was measured); with
# the limit the file is cut back to this size at the first reset after the read ends, instead of staying at its peak.
SQLITE_JOURNAL_SIZE_LIMIT = 64 * 1024 * 1024
SQLITE_STOP_CHECKPOINT_WAIT_S = 5.0  # how long a stopping process waits for another one's read before it leaves the log


@event.listens_for(Engine, "connect")
def _sqlite_busy_timeout(dbapi_connection: Any, _record: Any) -> None:
    """Every SQLite connection of the process waits up to 30 seconds for the write lock (D-73; the driver's own limit is 5),
    and bounds the write-ahead log it leaves after a checkpoint (`SQLITE_JOURNAL_SIZE_LIMIT`)."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        dbapi_connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        dbapi_connection.execute(f"PRAGMA journal_size_limit={int(SQLITE_JOURNAL_SIZE_LIMIT)}")


def journal_mode(engine: Engine) -> str | None:
    """The journal mode of an SQLite database, after switching a database on disk to WAL (D-73): a read then no longer blocks a
    write. The mode is a property of the file and stays; the statement runs outside a transaction and its answer is read back.
    None on another engine."""
    if engine.dialect.name != "sqlite":
        return None
    # N-14 (code re-check R3-02): the switch needs the database to itself for a moment and does not wait as a write does:
    # beside another process's write transaction it failed at once with "database is locked", and the start with it. It is
    # tried again for as long as a write would wait.
    deadline = time.monotonic() + SQLITE_BUSY_TIMEOUT_MS / 1000
    wait = 0.02
    while True:
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                mode = str(conn.exec_driver_sql("PRAGMA journal_mode").scalar()).lower()
                if mode not in ("wal", "memory"):
                    mode = str(conn.exec_driver_sql("PRAGMA journal_mode=WAL").scalar()).lower()
            return mode
        except OperationalError as e:
            if "locked" not in str(e).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(wait)
            wait = min(0.5, wait * 2)


POSTGRES_URL_FORM = "postgresql+psycopg://user:password@host/database"


class DatabaseUrlError(ValueError):
    """A database URL whose driver this installation cannot serve."""


def engine_url(url: str) -> str:
    """The URL the engine is made from. A bare `postgresql://` (or `postgres://`) URL is served by psycopg 3, the driver of the
    `postgres` extra; a Postgres driver that is not installed is refused with the accepted form named."""
    parsed = make_url(url)
    if parsed.get_backend_name() not in ("postgresql", "postgres"):
        return url
    bare = parsed.drivername in ("postgresql", "postgres")
    if bare:
        parsed = parsed.set(drivername="postgresql+psycopg")
    driver = parsed.get_driver_name()
    if importlib.util.find_spec(driver) is None:
        raise DatabaseUrlError(
            f"the database URL needs the Postgres driver '{driver}', which is not installed; use the form {POSTGRES_URL_FORM} "
            "and install the 'postgres' extra (psycopg 3)")
    return parsed.render_as_string(hide_password=False) if bare else url


def _utc_session(dbapi_connection: Any, _record: Any) -> None:
    """Pin a new Postgres session to UTC (D-72 "Time"; PG-03, PG-04, PG-06).

    Postgres reads a time without an offset, and prints every time, in the zone of the session, which is the zone of the
    server's host unless something sets it. A statement on the new connection, not a start-up option: a pooler may drop the
    options of the start-up packet. It is committed, so the pool's rollback on return does not undo it."""
    autocommit = dbapi_connection.autocommit
    dbapi_connection.autocommit = True
    try:
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET TIME ZONE 'UTC'")
    finally:
        dbapi_connection.autocommit = autocommit


def make_engine(url: str) -> Engine:
    """The engine of the service and of its command line, with the settings each database needs (D-72).

    Postgres: a connection is tested before use (`pool_pre_ping`), so the first request after a database restart reconnects
    instead of failing; the sessions carry an `application_name`; the pool size is a setting; every session is in UTC,
    whatever the zone of the server or of the database (`_utc_session`)."""
    url = engine_url(url)
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite":
        return create_engine(url, future=True, connect_args={"check_same_thread": False})
    if parsed.get_backend_name() == "postgresql":
        named = {} if "application_name" in parsed.query else {"application_name": settings.db_application_name}
        engine = create_engine(url, future=True, pool_pre_ping=True, pool_size=settings.db_pool_size,
                               max_overflow=settings.db_max_overflow, connect_args=named)
        event.listen(engine, "connect", _utc_session)
        return engine
    return create_engine(url, future=True)


_init_lock = threading.RLock()


def init_db() -> None:
    """Connect and migrate to the Alembic head (stamping unversioned v0.3 databases at the baseline first)."""
    global _engine, _SessionLocal
    from app.storage.migrations import upgrade_to_head

    # one at a time: two threads that find no engine (a worker beside the first request; in the tests, a job the test before left
    # running, beside the next fixture) would both create the tables of a new database. The second finds it at the head
    with _init_lock:
        settings.ensure_dirs()
        engine = make_engine(settings.sqlalchemy_url)
        journal_mode(engine)
        upgrade_to_head(engine)
        _engine = engine
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def shutdown() -> None:
    """When the API or the worker stops (N-12a; D-73; code re-check R3-01): fold the write-ahead log into the database file and
    close the connections, so that a stopped SQLite instance is `veritas.db` alone.

    SQLite removes the `-wal` and `-shm` files when the last connection of the last process closes; a process that exits with
    its connections open leaves them, and `veritas.db` on its own is then stale or empty (4,096 bytes beside a log of 2 MB
    after one analysis). A stop that is killed skips this: the three files are then the database, and the next start reads
    the log. The engine stays usable: a later use opens new connections. Nothing to do on Postgres but close the pool."""
    engine = _engine
    if engine is None:
        return
    engine.dispose()  # the pool's connections first: the checkpoint below then meets no reader of this process
    path = engine.url.database if engine.dialect.name == "sqlite" else None
    if not path or path == ":memory:" or not Path(path).is_file():
        return
    from app.core.logging import get_logger

    try:
        # a connection of its own with a short wait: a stop does not hang for the 30 seconds of the pool's connections on
        # another process that is reading. Its close is the last close of this process, which removes an emptied log
        last = sqlite3.connect(path, timeout=SQLITE_STOP_CHECKPOINT_WAIT_S)
        try:
            busy = last.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        finally:
            last.close()
        if busy is not None and busy[0]:  # another process still reads or writes: it folds the log in when it stops
            get_logger(__name__).info("db.checkpoint_left_to_another_process")
    except sqlite3.Error as e:  # a stop is never failed by its tidying
        get_logger(__name__).warning("db.checkpoint_failed", error=f"{type(e).__name__}: {e}"[:200])


def sessions_as_found() -> sessionmaker[Session]:
    """Sessions on the configured database as it is, for a command that only reads (N-14; `audit verify`, `custody export`):
    nothing is migrated and the journal mode is not switched, so the command neither changes a database it was asked to
    check nor takes part in a start. In a process that has already started on the database, its own sessions.

    Raises FileNotFoundError when the SQLite file is not there: opening it would create an empty database."""
    if _SessionLocal is not None:
        return _SessionLocal
    engine = make_engine(settings.sqlalchemy_url)
    path = engine.url.database if engine.dialect.name == "sqlite" else None
    if path and path != ":memory:" and not Path(path).is_file():
        raise FileNotFoundError(f"no database at {path}: the instance has not been started on this data directory")
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_engine() -> Engine:
    if _engine is None:
        init_db()
    assert _engine is not None
    return _engine


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    return _SessionLocal()
