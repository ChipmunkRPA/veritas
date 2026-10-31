"""Review and sign-off rules (ARCHITECTURE §4.7 `conclusions`, `signoffs`, §5.4; PRD RW-03, RW-04; D-27; UX §5.14; BUILD_SPEC C-5).

- **Verbal scale.** Five positions: ai_supported_strong (+2), ai_supported (+1), inconclusive (0), no_ai_indicators (−1),
  no_ai_indicators_strong (−2). `not_assessable` and the examiner's `no_opinion` have no position, so no deviation is counted.
- **Displayed state** (D-27, D-56), first match wins: withdrawn, superseded (a newer revision exists that is not cancelled or
  failed, directly or through revisions that were cancelled or failed before they published, N-8; or this revision was
  cancelled or failed and the one it would have replaced has been replaced, K-2), approved (the peer
  review concurs, with or without changes), disputed (the peer review does not concur), reviewed (examiner signed), in_review (a
  conclusion or examiner statuses exist), machine.
- **Issuable** (D-56): approved, or reviewed on a case that does not require a second review. Disputed is never issuable: the
  way forward is a new revision.
  An invalid signature on the current conclusion or on a sign-off also stops the issue (N-3).
- **Signatures.** HMAC-SHA256 with the instance signing key (ARCHITECTURE §5.3) over the canonical JSON of a record's signed
  fields. The key derives from `VERITAS_SIGNING_KEY`, else `VERITAS_API_KEY`: changing it makes older records read as invalid.
"""
from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from sqlalchemy import and_, case, func, inspect, or_, select, update
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.elements import ColumnElement

from app.api import tokens
from app.core import times
from app.core.canonical import canonical_hash, canonical_json
from app.storage.db import (
    UNPUBLISHED_TERMINAL,
    Analysis,
    AnnotationRevisionRow,
    AnnotationRow,
    CaseRow,
    ConclusionRow,
    SignoffRow,
)

SCALE: dict[str, int] = {"ai_supported_strong": 2, "ai_supported": 1, "inconclusive": 0, "no_ai_indicators": -1,
                         "no_ai_indicators_strong": -2}
POSITIONS = (*SCALE, "no_opinion")
STATES = ("machine", "in_review", "reviewed", "approved", "disputed", "superseded", "withdrawn")
PEER_APPROVES = ("concur", "concur_with_changes")  # D-56: any other peer decision disputes the conclusion
CONCURRING = ("concur", "concur_with_changes")  # N-19 (browser QA P1-7): an examiner whose conclusion departs signs neither
JUSTIFY_FROM_STEPS = 2  # RW-03


_utc = times.to_utc  # one form on every engine: a time is printed in UTC (D-72)


# N-19n (design review D4-01, P1: the rest of D3-01): under a trusted capture record and under a trusted declaration the
# examiner's position is compared with the step of that record on the scale, the step of the result's basis: the first
# under a capture, the fifth under a declaration. With a fusion model that is the code's own step. On the fallback the code
# stands one step inside (the fallback issues no strongest code) while the scale fills the record's segment, so that an
# examiner on the record's own step was "1 step from the machine decision" and could not sign as concurring. What the steps
# of a conclusion are counted from (`ConclusionRow.steps_from`), by the kind of the record that decides, and that step
STEPS_FROM_CAPTURE, STEPS_FROM_DECLARATION = "capture_record", "declaration"
STEPS_FROM = {"trusted_capture": STEPS_FROM_CAPTURE, "decisive_ai": STEPS_FROM_DECLARATION}
RECORD_STEP = {STEPS_FROM_CAPTURE: "no_ai_indicators_strong", STEPS_FROM_DECLARATION: "ai_supported_strong"}
# the step in words, as a refusal and the lines of the page and the reports name it; the code's own without a record
STEP_NAMES = {STEPS_FROM_CAPTURE: "the step of the capture record", STEPS_FROM_DECLARATION: "the step of the declaration"}


def steps_from(record_kind: str | None) -> str | None:
    """What the steps of a new conclusion are counted from: the record that decides the result, for a trusted capture and
    a trusted declaration; None for a marker and without a record (the decision code, as before)."""
    return STEPS_FROM.get(str(record_kind))


def deviation(machine_code: str | None, position: str, counted_from: str | None = None) -> int | None:
    """Steps to the examiner's position (positive: towards AI), or None without two positions: from the step of the record
    `counted_from` names (`steps_from`), else from the machine decision."""
    reference = RECORD_STEP.get(str(counted_from), machine_code)
    if machine_code not in SCALE or reference not in SCALE or position not in SCALE:
        return None
    return SCALE[position] - SCALE[reference]


def departs(machine_code: str | None, position: str, steps: int | None = None) -> bool | None:
    """Whether the examiner's position differs from the machine's (L-5): another scale position, or no opinion where the
    machine gave one. None where there is nothing to compare with: no machine decision (a v1 result), or a result that
    is not assessable (N-19j, D2-12: an opinion beside such a result was a departure on the page, "differs from the machine
    assessment", while the report said that no comparison is recorded). A sign-off then has no object.

    N-19n (D4-01): agreement is read from the stored `steps` (0: on the step), no longer from the code against the
    position: on the fallback a conclusion on the step of a capture record holds the second step as its `machine_code`,
    with 0 steps. A conclusion recorded before holds the position less the code, so it reads as it did."""
    if machine_code is None or machine_code == "not_assessable":
        return None
    if steps is None or position not in SCALE:  # no opinion on the scale: nothing is counted, and it is no agreement
        return position != machine_code
    return steps != 0


def departs_of(conclusion: Any) -> bool | None:
    """`departs` of a conclusion as it is stored, or as a snapshot holds it (one frozen without its steps is read from its
    code, as before)."""
    return departs(conclusion.machine_code, conclusion.scale_position, getattr(conclusion, "deviation_steps", None))


def counts_from_capture(conclusion: Any) -> bool:
    """Whether the steps of a conclusion under a trusted capture are those from the step of the capture record: it says so
    (`steps_from`), or the code's step is the record's own (a fusion model; also a conclusion recorded before N-19n). One
    recorded before on the fallback, whose steps were counted from the code's step one further in, is not."""
    capture = STEPS_FROM_CAPTURE
    return getattr(conclusion, "steps_from", None) == capture or conclusion.machine_code == RECORD_STEP[capture]


def counts_from_declaration(conclusion: Any, machine_code: str | None) -> bool:
    """N-19o (design review D5-01): whether the examiner's layer names the step of the declaration beside this conclusion:
    its steps are counted from the declaration (`steps_from`), and the code is not the declaration's own step (the
    fallback, whose code stands one step inside). With a fusion model the code is that step, and the words of before stand;
    a conclusion recorded before N-19n, whose steps were counted from the code, keeps them too."""
    return getattr(conclusion, "steps_from", None) == STEPS_FROM_DECLARATION and machine_code != RECORD_STEP[STEPS_FROM_DECLARATION]


_H3 = "H3_provenance_integrity"
_CLEAN_CODES = ("no_ai_indicators", "no_ai_indicators_strong")


def reference_of(analysis: Analysis, conclusion: Any) -> str | None:
    """`capture_record` where the page and the lists name the step of the capture record beside this conclusion (N-19n,
    D4-01; `reports/view.review_reference` for the reports, which hold the result): a trusted capture decides the
    analysis, and the conclusion's steps are the record's. From the columns of the analysis, as a list reads a row: a
    record that decides with a clean code is a capture's (a declaration and a marker give an AI code).

    N-19o (D5-01): `declaration` where a trusted declaration decides on the fallback and the conclusion's steps are the
    declaration's (`counts_from_declaration`; only a trusted declaration gives a conclusion that key)."""
    if conclusion is None or analysis.driving_hypothesis != _H3:
        return None
    if analysis.decision_code not in _CLEAN_CODES:
        return STEPS_FROM_DECLARATION if counts_from_declaration(conclusion, analysis.decision_code) else None
    return STEPS_FROM_CAPTURE if counts_from_capture(conclusion) else None


# N-19j (browser QA Q2-03): a sign-off decision is said with its object. The examiner concurs or does not concur with the
# machine assessment; the reviewer concurs or does not concur with the examiner's opinion. Where the machine's result is not
# assessable the examiner's sign-off has no object: it records the opinion without a comparison
SIGNOFF_OBJECT = {"examiner": "machine_assessment", "peer_review": "examiner_opinion"}
NEEDS_COMMENT = ("concur_with_changes", "non_concur")  # N-19j (Q2-02, Q2-08): the reason, or the changes asked for, is on record


def signoff_object(kind: str, conclusion: Any) -> str | None:
    """What the decision of a sign-off is about, or None for an examiner's sign-off without a comparison."""
    if kind == "examiner" and (conclusion is None or departs_of(conclusion) is None):
        return None
    return SIGNOFF_OBJECT[kind]


def conclusion_body(row: ConclusionRow) -> dict[str, Any]:
    """The signed fields of a conclusion. N-19n (D4-01): a conclusion whose steps are counted from the step of a record
    says so in a key that is present only there, so that every signature made before verifies as it did."""
    body = {"record": "conclusion", "id": row.id, "analysis_id": row.analysis_id, "analysis_revision": row.analysis_revision,
            "author_id": row.author_id, "scale_position": row.scale_position, "machine_code": row.machine_code,
            "deviation_steps": row.deviation_steps, "rationale": row.rationale,
            "limitations_acknowledged": list(row.limitations_acknowledged or []), "next_steps": row.next_steps,
            "declaration_text_id": row.declaration_text_id, "deviation_justification": row.deviation_justification,
            "created_at": _utc(row.created_at)}
    return body if row.steps_from is None else {**body, "steps_from": row.steps_from}


def signoff_body(row: SignoffRow) -> dict[str, Any]:
    return {"record": "signoff", "id": row.id, "analysis_id": row.analysis_id, "analysis_revision": row.analysis_revision,
            "conclusion_id": row.conclusion_id, "principal_id": row.principal_id, "kind": row.kind, "decision": row.decision,
            "comment": row.comment, "result_hash": row.result_hash, "annotations_seq": row.annotations_seq,
            "signed_at": _utc(row.signed_at)}


def sign(body: dict[str, Any], key: bytes | None = None) -> str:
    return hmac.new(key or tokens.signing_key(), canonical_json(body), hashlib.sha256).hexdigest()


def signature_valid(body: dict[str, Any], signature: str | None, key: bytes | None = None) -> bool:
    """Against the signing key of the instance; `key` names another key to try (`keyguard.signed_under`)."""
    return bool(signature) and hmac.compare_digest(sign(body, key), str(signature))


def stored_conclusions(db: Session) -> list[Any]:
    """Every conclusion with the fields that its signature covers, read from a database at any revision since 0016: a
    column that the database does not hold yet (`steps_from`, migration 0023) reads as None, which is what it holds for
    every conclusion recorded before. `backup verify` reads a copy as it was backed up, and the key guard reads the
    database before the start migrates it: read through the model, a copy of an earlier build "cannot be read by this
    build", and its upgrade lost the check of its signatures."""
    columns = list(ConclusionRow.__table__.columns)
    present = {c["name"] for c in inspect(db.connection()).get_columns(ConclusionRow.__tablename__)}
    rows = db.execute(select(*[c for c in columns if c.name in present])).mappings()
    return [SimpleNamespace(**{**{c.name: None for c in columns}, **row}) for row in rows]


def conclusion_valid(row: ConclusionRow) -> bool:
    return signature_valid(conclusion_body(row), row.signature)


def signoff_valid(row: SignoffRow) -> bool:
    return signature_valid(signoff_body(row), row.signature)


def latest_conclusion(db: Session, analysis_id: str) -> ConclusionRow | None:
    return db.scalars(select(ConclusionRow).where(ConclusionRow.analysis_id == analysis_id)
                      .order_by(ConclusionRow.version.desc()).limit(1)).first()


def conclusion_version(db: Session, analysis_id: str) -> int:
    """The version of the newest conclusion of an analysis, 0 before the first (L-5)."""
    return int(db.scalar(select(func.max(ConclusionRow.version)).where(ConclusionRow.analysis_id == analysis_id)) or 0)


def current_conclusion(db: Session, analysis_id: str, records: list[SignoffRow] | None = None) -> ConclusionRow | None:
    """The conclusion the examiner signed once signed off, else the newest (D-47): a conclusion written after the lock never reads
    as current."""
    records = records if records is not None else signoffs(db, analysis_id)
    signed = next((s for s in records if s.kind == "examiner"), None)
    if signed is not None and (row := db.get(ConclusionRow, signed.conclusion_id)) is not None:
        return row
    return latest_conclusion(db, analysis_id)


def hold_unlocked(db: Session, analysis_id: str) -> bool:
    """Take the analysis row's write lock while the revision is unlocked (a conditional no-op update), so a lock check and the
    write that follows commit atomically (D-47). False when the revision is locked or gone; the caller refuses and rolls back."""
    result = db.execute(update(Analysis).where(Analysis.id == analysis_id, Analysis.locked_at.is_(None)).values(locked_at=None)
                        .execution_options(synchronize_session=False))
    return result.rowcount == 1


def hold_revision(db: Session, analysis_id: str) -> bool:
    """Take the analysis row's write lock, whatever its review state (a conditional no-op update, as `hold_unlocked`): a
    purge or a delete that holds the row finishes first, and the caller then sees whether the revision is still there
    (N-9; Postgres QA X-7, PG-16). False when the revision is gone; the caller answers 404 and rolls back."""
    result = db.execute(update(Analysis).where(Analysis.id == analysis_id).values(kind=Analysis.kind)
                        .execution_options(synchronize_session=False))
    return result.rowcount == 1


def gone(db: Session, analysis_id: str) -> bool:
    """The revision no longer exists, read from the database and not from the session (which may hold the row it read before
    it waited for a purge): a request that lost its revision answers 404, not "locked" (N-9)."""
    return db.scalar(select(func.count()).select_from(Analysis).where(Analysis.id == analysis_id)) == 0


def lock_revision(db: Session, analysis_id: str, at: datetime) -> bool:
    """Set `locked_at` only if it is unset (the examiner sign-off; D-47). False when another sign-off locked it first."""
    result = db.execute(update(Analysis).where(Analysis.id == analysis_id, Analysis.locked_at.is_(None)).values(locked_at=at)
                        .execution_options(synchronize_session=False))
    return result.rowcount == 1


def signoffs(db: Session, analysis_id: str) -> list[SignoffRow]:
    return list(db.scalars(select(SignoffRow).where(SignoffRow.analysis_id == analysis_id).order_by(SignoffRow.signed_at, SignoffRow.id)))


def annotations_seq(db: Session, analysis_id: str) -> int:
    """Annotation states plus deletions recorded for the analysis: any later change to the examiner layer raises it."""
    states = db.scalar(select(func.count()).select_from(AnnotationRevisionRow)
                       .join(AnnotationRow, AnnotationRow.id == AnnotationRevisionRow.annotation_id).where(AnnotationRow.analysis_id == analysis_id))
    deleted = db.scalar(select(func.count()).select_from(AnnotationRow).where(AnnotationRow.analysis_id == analysis_id,
                                                                              AnnotationRow.deleted_at.is_not(None)))
    return int(states or 0) + int(deleted or 0)


def later_revisions(db: Session, analysis_id: str) -> list[tuple[str, str, int | None]]:
    """(id, status, revision) of every revision that follows this one directly, or through revisions that were cancelled or
    failed before they published (N-8): nearest first. A chain is not followed past a live revision."""
    out: list[tuple[str, str, int | None]] = []
    seen, frontier = {analysis_id}, [analysis_id]
    while frontier:
        rows = db.execute(select(Analysis.id, Analysis.status, Analysis.revision).where(Analysis.supersedes_id.in_(frontier))
                          .order_by(Analysis.revision, Analysis.id)).tuples().all()
        fresh = [r for r in rows if r[0] not in seen]
        seen.update(r[0] for r in fresh)
        out.extend(fresh)
        frontier = [r[0] for r in fresh if r[1] in UNPUBLISHED_TERMINAL]
    return out


def successor(db: Session, analysis_id: str) -> str | None:
    """The newer revision that supersedes this one: queued, running or published. A cancelled or failed successor never
    published, so the revision it would have replaced stays current and actionable (K-2), unless a live revision follows that
    dead one: a chain "done ← failed ← done" has one current revision, the last (N-8)."""
    return next((rid for rid, status, _ in later_revisions(db, analysis_id) if status not in UNPUBLISHED_TERMINAL), None)


def superseded(db: Session, analysis_id: str) -> bool:
    """A newer live revision replaces this one; or this one never published and the revision it would have replaced has since
    been replaced (K-2). Both through any number of revisions that never published (N-8)."""
    if successor(db, analysis_id) is not None:
        return True
    row = db.get(Analysis, analysis_id)
    if row is None or row.status not in UNPUBLISHED_TERMINAL:
        return False
    base = revision_base(db, row)
    return base.id != row.id and successor(db, base.id) is not None


def revision_base(db: Session, analysis: Analysis) -> Analysis:
    """The revision a new re-run, reanalyse or re-fusion of `analysis` supersedes: `analysis` itself, or, for a revision that
    was cancelled or failed before it published, the revision that one would have replaced (K-2)."""
    seen = {analysis.id}
    while analysis.status in UNPUBLISHED_TERMINAL and analysis.supersedes_id:
        source = db.get(Analysis, analysis.supersedes_id)
        if source is None or source.id in seen:
            break
        seen.add(source.id)
        analysis = source
    return analysis


def next_revision(db: Session, base: Analysis) -> int:
    """The revision number of a new successor of `base`: one above `base` and above every unpublished revision that follows it."""
    later = [revision or 0 for _, _, revision in later_revisions(db, base.id)]
    return max([base.revision or 1, *later]) + 1


def peer_state(decision: str) -> str:
    """D-56: a peer review that concurs (with or without changes) approves; any other decision disputes."""
    return "approved" if decision in PEER_APPROVES else "disputed"


def is_issuable(state: str, requires_second_review: bool) -> bool:
    """D-56: approved, or reviewed on a case that does not require a second review."""
    return state == "approved" or (state == "reviewed" and not requires_second_review)


def state(db: Session, analysis: Analysis, records: Sequence[SignoffRow] | None = None) -> str:
    if analysis.withdrawn_at is not None:
        return "withdrawn"
    if superseded(db, analysis.id):
        return "superseded"
    records = records if records is not None else signoffs(db, analysis.id)
    peer = next((s for s in records if s.kind == "peer_review"), None)
    if peer is not None:
        return peer_state(peer.decision)
    if any(s.kind == "examiner" for s in records):
        return "reviewed"
    if latest_conclusion(db, analysis.id) is not None or db.scalars(select(AnnotationRow.id).where(
            AnnotationRow.analysis_id == analysis.id, AnnotationRow.examiner_status.is_not(None), AnnotationRow.deleted_at.is_(None)).limit(1)).first():
        return "in_review"
    return "machine"



def state_expression() -> ColumnElement[str]:
    """D-27 and D-56 precedence for collection queries; correlated indexed EXISTS avoids per-row API/SQL calls."""
    # N-8: (root, node) for every revision `node` that follows `root` directly or through revisions that never published,
    # as `later_revisions` walks it. UNION (not UNION ALL) ends a cycle in damaged data.
    first, later = aliased(Analysis), aliased(Analysis)
    reach = (select(first.supersedes_id.label("root"), first.id.label("node"), first.status.label("status"))
             .where(first.supersedes_id.is_not(None)).cte("revision_reach", recursive=True))
    reach = reach.union(select(reach.c.root, later.id, later.status)
                        .where(later.supersedes_id == reach.c.node, reach.c.status.in_(UNPUBLISHED_TERMINAL)))
    replaced = select(reach.c.node).where(reach.c.root == Analysis.id,
                                          reach.c.status.not_in(UNPUBLISHED_TERMINAL)).exists().correlate(Analysis)
    up, down = reach.alias("reach_up"), reach.alias("reach_down")
    obsolete = and_(Analysis.status.in_(UNPUBLISHED_TERMINAL), select(down.c.node).where(
        up.c.node == Analysis.id, down.c.root == up.c.root, down.c.status.not_in(UNPUBLISHED_TERMINAL)).exists().correlate(Analysis))
    def signed(kind: str, decisions: tuple[str, ...] = ()):
        found = select(SignoffRow.id).where(SignoffRow.analysis_id == Analysis.id, SignoffRow.kind == kind)
        return (found.where(SignoffRow.decision.in_(decisions)) if decisions else found).exists().correlate(Analysis)
    concluded = select(ConclusionRow.id).where(ConclusionRow.analysis_id == Analysis.id).exists().correlate(Analysis)
    annotated = select(AnnotationRow.id).where(AnnotationRow.analysis_id == Analysis.id,
        AnnotationRow.examiner_status.is_not(None), AnnotationRow.deleted_at.is_(None)).exists().correlate(Analysis)
    return case((Analysis.withdrawn_at.is_not(None), "withdrawn"), (or_(replaced, obsolete), "superseded"),
                (signed("peer_review", PEER_APPROVES), "approved"), (signed("peer_review"), "disputed"),
                (signed("examiner"), "reviewed"),
                (or_(concluded, annotated), "in_review"), else_="machine")


def states(db: Session, analyses: Sequence[Analysis]) -> dict[str, str]:
    """Read states for an already access-scoped collection in one query."""
    if not analyses:
        return {}
    return {aid: value for aid, value in db.execute(
        select(Analysis.id, state_expression()).where(Analysis.id.in_([a.id for a in analyses]))).tuples()}

def second_review_cases() -> Any:
    """The cases whose sign-off needs a second reviewer, as a subquery of their ids."""
    return select(CaseRow.id).where(CaseRow.requires_second_review.is_(True))


def awaits_second_review() -> ColumnElement[bool]:
    """N-19a (browser QA P1-12; design review DS-09): signed by the examiner on a case that requires a second review, and
    no peer review recorded: the state `reviewed` that is not issuable (D-56). Such a revision reads "awaiting second
    review", never "reviewed"."""
    return and_(state_expression() == "reviewed", Analysis.case_id.in_(second_review_cases()))


def awaiting_second(db: Session, analyses: Sequence[Analysis], review_states: dict[str, str]) -> set[str]:
    """Of an already access-scoped collection with its states, the ids that await their second review."""
    signed = [a for a in analyses if review_states.get(a.id) == "reviewed" and a.case_id]
    if not signed:
        return set()
    needing = set(db.scalars(second_review_cases().where(CaseRow.id.in_({a.case_id for a in signed}))))
    return {a.id for a in signed if a.case_id in needing}


def repair_signoff_events() -> tuple[int, int]:
    """Give every sign-off the custody event it lacks (N-9; code review 12). Before N-9 the event was appended after the
    sign-off had committed, so a failed append left a locked, "reviewed" revision with `audit_event_seq` NULL. For each such
    row: when the chain holds its `signoff` event, the row takes that seq; else the event is appended now, marked
    `recorded_late`, with the row's own time of signing in its details. Returns (linked, appended). Run at start."""
    from app.audit import chain as audit
    from app.storage import locks
    from app.storage.db import AuditEvent, PrincipalRow, get_session

    with get_session() as db:
        pending = list(db.scalars(select(SignoffRow.id).where(SignoffRow.audit_event_seq.is_(None)).order_by(SignoffRow.signed_at)))
    linked = appended = 0
    for signoff_id in pending:
        def unit(signoff_id: str = signoff_id) -> str | None:
            with audit.writing() as db:
                # N-10a (code re-verification, the reviewer's notes): the row is read under a lock that another process
                # waits for. Two processes that start together both found the row pending and each appended a `signoff`
                # event; the second now reads the row after the first has committed and finds its event recorded
                locks.serialise(db, "signoff-repair", signoff_id)
                row = db.get(SignoffRow, signoff_id, populate_existing=True)
                if row is None or row.audit_event_seq is not None:
                    db.rollback()
                    return None
                recorded = next((e.seq for e in db.scalars(select(AuditEvent).where(
                    AuditEvent.action == "signoff", AuditEvent.analysis_id == row.analysis_id).order_by(AuditEvent.seq))
                    if (e.object_ids or [None])[0] == row.id), None)
                if recorded is not None:
                    row.audit_event_seq = recorded
                    db.commit()
                    return "linked"
                signer = db.get(PrincipalRow, row.principal_id)
                audit.commit_with(db, [{
                    "action": "signoff", "actor_id": row.principal_id, "analysis_id": row.analysis_id,
                    "object_ids": [row.id, row.conclusion_id], "after_hash": canonical_hash(signoff_body(row)),
                    "details": {"kind": row.kind, "decision": row.decision,
                                "signer": signer.display_name if signer is not None else row.principal_id,
                                "signer_role": signer.role if signer is not None else None,
                                "analysis_revision": row.analysis_revision, "locks_revision": row.kind == "examiner",
                                "recorded_late": True, "signed_at": times.iso_z(row.signed_at)}}],
                    before_commit=lambda seqs: setattr(row, "audit_event_seq", seqs[0]))
                return "appended"

        outcome = audit.repeated(unit)
        linked, appended = linked + (outcome == "linked"), appended + (outcome == "appended")
    return linked, appended


def positions(db: Session, analyses: Sequence[Analysis]) -> dict[str, tuple[str, bool | None, str | None]]:
    """The current conclusion's scale position of each analysis, with whether it departs from the machine decision (L-5): the
    signed conclusion once signed off, else the newest. Analyses without a conclusion are absent. N-19n (D4-01): and what
    the position is compared with where that is the step of a capture record (`reference_of`)."""
    ids = [a.id for a in analyses]
    by_id = {a.id: a for a in analyses}
    if not ids:
        return {}
    signed = {aid: cid for aid, cid in db.execute(select(SignoffRow.analysis_id, SignoffRow.conclusion_id)
                                                  .where(SignoffRow.analysis_id.in_(ids), SignoffRow.kind == "examiner")).tuples()}
    out: dict[str, tuple[str, bool | None, str | None]] = {}
    for row in db.scalars(select(ConclusionRow).where(ConclusionRow.analysis_id.in_(ids)).order_by(ConclusionRow.version)):
        if row.analysis_id not in signed or signed[row.analysis_id] == row.id:
            out[row.analysis_id] = (row.scale_position, departs_of(row), reference_of(by_id[row.analysis_id], row))
    return out


def deciding_records(analyses: Sequence[Analysis]) -> dict[str, str]:
    """N-19p (design review D6-01, P2): what a list row says decides its result, where a trusted record does (`steps_from`:
    `capture_record`, `declaration`), read from the stored result. A row holds the decision code and the driving hypothesis,
    and on the fallback a trusted declaration and a marker give the same two (`ai_supported`, H3): the Analyses list, the
    batch page and the case row printed the code's word ("Indications") under a result that the page heads by the
    declaration. Only rows that a provenance record drives are read; analyses without one are absent."""
    return {a.id: kind for a in analyses
            if a.status == "done" and a.result_schema_version == 2 and a.result_hash and a.driving_hypothesis == _H3
            and (kind := steps_from(_record_kind(a.result_hash)))}


_RECORD_KINDS: dict[str, str | None] = {}  # by result hash: a published result is immutable and stored by its hash


def _record_kind(result_hash: str) -> str | None:
    """The kind of the provenance record of the published result with this hash, or None: none in it, or the object cannot
    be read or does not match its hash (the row then reads by its code, and the analysis's own page reports the object; a
    failed read is not remembered)."""
    if result_hash in _RECORD_KINDS:
        return _RECORD_KINDS[result_hash]
    import json

    from app.storage.objects import ObjectStoreError, default_store

    try:
        data = default_store().read_bytes(result_hash)
    except ObjectStoreError:
        return None
    if hashlib.sha256(data).hexdigest() != result_hash:
        return None
    record = (json.loads(data).get("decision") or {}).get("provenance_record")
    kind = record.get("kind") if isinstance(record, dict) else None
    if len(_RECORD_KINDS) >= 4096:
        _RECORD_KINDS.clear()
    _RECORD_KINDS[result_hash] = kind
    return kind


def brief(db: Session, analysis: Analysis) -> dict[str, Any]:
    """The review block of an analysis detail: displayed state, issuable, lock, withdrawal, current conclusion, sign-offs."""
    records = signoffs(db, analysis.id)
    current = state(db, analysis, records)
    case = db.get(CaseRow, analysis.case_id) if analysis.case_id else None
    requires_second = bool(case.requires_second_review) if case is not None else False
    conclusion = current_conclusion(db, analysis.id, records)
    # N-3: a review record whose HMAC no longer matches its content (a row edited outside the API, a changed signing key)
    valid = (conclusion is None or conclusion_valid(conclusion)) and all(signoff_valid(s) for s in records)
    return {"state": current, "issuable": is_issuable(current, requires_second) and valid, "signatures_valid": valid,
            "requires_second_review": requires_second, "locked_at": _utc(analysis.locked_at), "withdrawn_reason": analysis.withdrawn_reason,
            "conclusion_id": conclusion.id if conclusion is not None else None,
            "conclusion_version": conclusion.version if conclusion is not None else None,
            "examiner_position": conclusion.scale_position if conclusion is not None else None,
            "examiner_departs": departs_of(conclusion) if conclusion is not None else None,
            "examiner_reference": reference_of(analysis, conclusion),
            "signoffs": [s.kind for s in records]}
