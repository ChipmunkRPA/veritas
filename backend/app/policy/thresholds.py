"""Workspace policy thresholds (ARCHITECTURE §4.4 "Abstention"; PRD API-03; BUILD_SPEC X-2).

- **What the policy sets.** The decision codes of a calibrated (artefact) decision:
  - `alpha_strong`: the conformal α whose singleton prediction set issues `ai_supported_strong` / `no_ai_indicators_strong`
    (default 0.01);
  - `alpha_moderate`: the α whose singleton set issues `ai_supported` / `no_ai_indicators` (default 0.05);
  - `min_slice_auc`: the measured slice AUC below which a decision is capped at `inconclusive` (EA-02; default and floor 0.70).
  The α levels are the ones `make calibrate` fits (0.05, 0.01), with α_strong ≤ α_moderate. The uncalibrated fallback keeps its
  own caps (D-31), and Stage 0 documentary overrides are not capped.
- **Versions.** Rows are never updated. Version 1 holds the defaults and is written the first time the policy is read. A change
  adds version n+1 and records `policy_changed` with the before and after hashes, the changed fields and the reason. A policy
  identical to the version in force is refused.
- **Reference.** Each analysis records the version in force in `manifest.policy_thresholds_id` (§6.6); a re-run reuses the
  version its analysis recorded (CC-03).
"""
from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.canonical import canonical_json
from app.storage.db import PolicyThresholdsRow, get_session, utcnow

Alpha = Literal[0.01, 0.05]
SYSTEM_ACTOR = "system"
INITIAL_REASON = "Initial defaults (ML §4.5 conformal levels, EA-02 envelope)"


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha_strong: Alpha = Field(default=0.01, description="Conformal α of the *_strong codes (a singleton set at this level)")
    alpha_moderate: Alpha = Field(default=0.05, description="Conformal α of ai_supported / no_ai_indicators")
    min_slice_auc: float = Field(default=0.70, ge=0.70, le=1.0,
                                 description="Minimum measured slice AUC; below it a decision is capped at inconclusive (EA-02)")

    @model_validator(mode="after")
    def _order(self) -> Thresholds:
        if self.alpha_strong > self.alpha_moderate:
            raise ValueError("alpha_strong must not exceed alpha_moderate: a strong code needs the stricter level")
        return self


@dataclass(frozen=True)
class Policy:
    id: str
    version: int
    thresholds: Thresholds
    sha256: str

    def as_json(self) -> dict[str, Any]:
        """The form the fusion runtime reads (and the analysis's `fusion.policy` block records)."""
        return {"id": self.id, "version": self.version, **self.thresholds.model_dump()}


class PolicyUnchanged(ValueError):
    """The requested thresholds equal the version in force."""


class PolicyConflict(RuntimeError):
    """Another change saved the next version first."""


def digest(thresholds: Thresholds) -> str:
    return hashlib.sha256(canonical_json(thresholds.model_dump())).hexdigest()


def _policy(row: PolicyThresholdsRow) -> Policy:
    return Policy(row.id, row.version, Thresholds.model_validate(row.thresholds), row.sha256)


def _latest(db: Session) -> PolicyThresholdsRow | None:
    return db.scalars(select(PolicyThresholdsRow).order_by(PolicyThresholdsRow.version.desc()).limit(1)).first()


def current_row(db: Session) -> PolicyThresholdsRow:
    """The version in force; writes version 1 (the defaults) when none exists. Commits, so call it with no pending changes."""
    row = _latest(db)
    if row is not None:
        return row
    defaults = Thresholds()
    row = PolicyThresholdsRow(id=str(uuid.uuid4()), version=1, thresholds=defaults.model_dump(), sha256=digest(defaults),
                              reason=INITIAL_REASON, created_by=SYSTEM_ACTOR, created_at=utcnow())
    db.add(row)
    try:
        db.commit()
    except IntegrityError:  # another process wrote version 1 first
        db.rollback()
        row = _latest(db)
        assert row is not None
    return row


def resolve(policy_id: str | None = None, sessions: Callable[[], Session] | None = None) -> Policy:
    """The pinned version (a re-run) or the version in force. LookupError for an unknown id."""
    with (sessions or get_session)() as db:
        if policy_id is None:
            return _policy(current_row(db))
        row = db.get(PolicyThresholdsRow, policy_id)
        if row is None:
            raise LookupError(f"policy thresholds {policy_id} not found")
        return _policy(row)


def versions(db: Session) -> list[PolicyThresholdsRow]:
    current_row(db)
    return list(db.scalars(select(PolicyThresholdsRow).order_by(PolicyThresholdsRow.version.desc())))


def change(thresholds: Thresholds, *, actor_id: str, reason: str) -> tuple[Policy, Policy]:
    """Save the next version and record `policy_changed`; returns (previous, new)."""
    from app.audit import chain as audit

    after = digest(thresholds)

    def unit() -> tuple[Policy, Policy]:
        # N-9: the thresholds in force decide every later verdict, so a version exists with its `policy_changed` event or
        # not at all
        with audit.writing() as db:
            previous = _policy(current_row(db))
            if after == previous.sha256:
                raise PolicyUnchanged(f"The thresholds equal version {previous.version}, which is in force; nothing was changed.")
            row = PolicyThresholdsRow(id=str(uuid.uuid4()), version=previous.version + 1, thresholds=thresholds.model_dump(),
                                      sha256=after, reason=reason, created_by=actor_id, created_at=utcnow())
            db.add(row)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                raise PolicyConflict("Another change saved the next version first; read the policy again and retry.") from None
            policy = _policy(row)
            before_fields, after_fields = previous.thresholds.model_dump(), thresholds.model_dump()
            changes = {k: {"from": before_fields[k], "to": v} for k, v in after_fields.items() if before_fields[k] != v}
            audit.commit_with(db, [{
                "action": "policy_changed", "actor_id": actor_id, "object_ids": [policy.id], "before_hash": previous.sha256,
                "after_hash": policy.sha256,
                "details": {"policy_thresholds_id": policy.id, "version": policy.version, "from_id": previous.id,
                            "from_version": previous.version, "changes": changes, "reason": reason}}])
            return previous, policy

    return audit.repeated(unit)
