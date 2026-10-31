"""The review of a fuser artefact in the custody log (BUILD_SPEC N-9; D-57).

A review is a sidecar file (`registry.read_review`): the review command writes it, a release can bring it, and it can be
replaced or removed. The custody log names each of those once:

- `fuser_review_recorded`: the kind, version, file and sha256 of the artefact, the reviewer, the review's date, the SHA-256 of
  the statement and of the review record, and `replaces` when the log named another review of this artefact before;
- `fuser_review_withdrawn`: the same for the review that no longer counts.

`fuser_reviews` holds what the log names last, per artefact. `reconcile` compares it with the review that counts now and, when
they differ, writes the row and its event in one transaction (`audit.commit_with`): a failed append leaves the row as it was
and the next caller records the event. The row is changed only from the state the caller read (an insert on the primary key,
or an update that names the digest it replaces), so two processes that find the same new review record it once.

Callers: the review commands (`python -m app.cli fusers review`, `fusers withdraw-review`; source "command line") and the fuse
stage (`stages.common.select_fusion`; source "found on use"), which is how a review that came with a release, or a sidecar
copied or deleted by hand, reaches the log: the first time an analysis is fused under it. A review that no analysis has used
yet and that is withdrawn by the command is recorded first (source "found on withdrawal"), so the log holds both ends.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from app.audit import chain as audit
from app.core.canonical import canonical_json
from app.engine.fusion.registry import Review, Selection
from app.storage.db import FuserReviewRow, get_session, utcnow

RECORDED, WITHDRAWN = "fuser_review_recorded", "fuser_review_withdrawn"
COMMAND, FOUND, FOUND_ON_WITHDRAWAL = "command line", "found on use", "found on withdrawal"
Source = Literal["command line", "found on use", "found on withdrawal"]
REASON_MAX = 500


@dataclass(frozen=True)
class Subject:
    """The artefact a review is about."""
    kind: str
    version: str | None
    file: str | None
    sha256: str


class _Raced(RuntimeError):
    """Another process changed the recorded review of this artefact first; the caller reads it again."""


def statement_sha256(statement: str) -> str:
    """The SHA-256 of a review statement: its text without surrounding white space, as UTF-8."""
    return hashlib.sha256(statement.strip().encode("utf-8")).hexdigest()


def review_sha256(artefact_sha256: str, review: Review | None) -> str | None:
    """The digest of a review record (the artefact it names, the reviewer, the date and the statement); None for no review."""
    if review is None:
        return None
    return hashlib.sha256(canonical_json({"artefact_sha256": artefact_sha256, "reviewer": review.reviewer, "date": review.date,
                                          "statement": review.statement})).hexdigest()


def of_selection(selection: Selection, kind: str) -> Subject | None:
    if selection.mode != "artefact" or selection.sha256 is None:
        return None
    return Subject(kind=kind, version=selection.version, file=selection.path.name if selection.path else None, sha256=selection.sha256)


def reconcile(subject: Subject, review: Review | None, *, source: Source, reason: str | None = None,
              sessions: audit.Sessions | None = None) -> dict[str, Any] | None:
    """Bring the custody log in line with the review that counts for this artefact now. Returns the event recorded (its
    `action` and `seq`), or None when the log already names this state (or the artefact was never reviewed)."""
    sessions = sessions or get_session
    current = review_sha256(subject.sha256, review)
    for _ in range(3):
        with sessions() as db:
            row = db.get(FuserReviewRow, subject.sha256)
            seen = None if row is None else (row.review_sha256, row.reviewer, row.statement_sha256)
        if (seen[0] if seen is not None else None) == current:
            return None
        try:
            return audit.repeated(lambda seen=seen: _record(subject, review, current, seen, source, reason, sessions))
        except _Raced:
            continue
    raise RuntimeError(f"the recorded review of fuser artefact {subject.sha256[:12]} kept changing")


def _record(subject: Subject, review: Review | None, current: str | None, seen: tuple[str | None, str | None, str | None] | None,
            source: Source, reason: str | None, sessions: audit.Sessions) -> dict[str, Any]:
    key = FuserReviewRow.artefact_sha256 == subject.sha256
    values = {"review_sha256": current, "reviewer": review.reviewer if review is not None else None,
              "statement_sha256": statement_sha256(review.statement) if review is not None else None}
    with audit.writing(sessions) as db:
        # the transaction's first statement is the write, conditional on what the caller read
        if seen is None:
            db.add(FuserReviewRow(artefact_sha256=subject.sha256, recorded_at=utcnow(), **values))
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                raise _Raced from None
        else:
            unchanged = FuserReviewRow.review_sha256.is_(None) if seen[0] is None else FuserReviewRow.review_sha256 == seen[0]
            changed = db.execute(update(FuserReviewRow).where(key, unchanged).values(recorded_at=utcnow(), **values)
                                 .execution_options(synchronize_session=False))
            if changed.rowcount != 1:
                db.rollback()
                raise _Raced
        previous = None if seen is None or seen[0] is None else {"reviewer": seen[1], "statement_sha256": seen[2], "review_sha256": seen[0]}
        details: dict[str, Any] = {"kind": subject.kind, "version": subject.version, "artefact": subject.file, "source": source}
        if review is not None:
            action = RECORDED
            details.update(reviewer=review.reviewer, review_date=review.date, statement_sha256=values["statement_sha256"],
                           review_sha256=current)
            if previous is not None:
                details["replaces"] = previous
        else:
            action = WITHDRAWN
            details.update(previous or {})
        if reason:
            details["reason"] = reason
        [seq] = audit.commit_with(
            db, [{"action": action, "actor_id": audit.SYSTEM_ACTOR, "object_ids": [subject.sha256], "details": details}],
            before_commit=lambda seqs: db.execute(update(FuserReviewRow).where(key).values(audit_event_seq=seqs[0])
                                                  .execution_options(synchronize_session=False)))
        return {"action": action, "seq": seq}
