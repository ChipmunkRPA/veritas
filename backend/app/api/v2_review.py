"""API v2 review and sign-off (ARCHITECTURE §4.7 `conclusions`, `signoffs`, §5.2, §5.4; PRD RW-03, RW-04; D-27; UX §5.14;
BUILD_SPEC C-5).

- **Conclusion** (`GET/PUT /analyses/{aid}/conclusion`; PUT needs the examiner role on the case).
  - Fields: the verbal-scale position (`ai_supported_strong` … `no_ai_indicators_strong`, or `no_opinion`), rationale,
    limitations acknowledged, next steps, declaration text id.
  - Each PUT records a new conclusion, and the newest is current. Conclusions are versioned 1, 2, …; a PUT names the version it
    edits from (`base_version`, 0 for the first) and is refused with 409 when a newer one exists (L-5).
  - A position 2 or more steps from the machine decision needs `deviation_justification` (422, RW-03). Under a trusted
    capture record and under a trusted declaration the steps are counted from the step of that record on the scale (the
    first, the fifth), on both fusion paths, and the conclusion says so (`steps_from`; N-19n, D4-01).
- **Sign-off** (`GET/POST /analyses/{aid}/signoffs`).
  - Both name the conclusion id and version the signer saw; another current conclusion is refused with 409 (L-5).
  - `examiner`: needs a conclusion, is made by its author, and locks the revision (`locked_at`). A conclusion that departs from
    the machine decision cannot be signed `concur` (422; L-5) or `concur_with_changes` (422; N-19, browser QA P1-7): a
    record never reads "concurs" beside a departure.
  - `peer_review` (RW-04): after the examiner sign-off, by an examiner, reviewer or administrator with the reviewer or owner role
    on the case (L-5: no instance reviewer role is needed), who is neither the signing examiner nor the conclusion's author.
  - Each record stores the result hash and the annotation count (`annotations_seq`) and is HMAC-signed; reads report
    `signature_valid`.
- **Withdraw** (`POST /analyses/{aid}/withdraw` {reason}): D-27 `withdrawn`, the revision should not be relied on.
- **Reanalyse** (`POST /analyses/{aid}/reanalyse` {profile, reason}): re-hashes the original, then queues revision n+1
  superseding this one. Locked revisions may be reanalysed; that is how later changes are made. Superseding a signed revision
  takes the rule of a withdrawal (D-60): a named examiner who signed it or wrote its conclusion, a case owner or an
  administrator, and a reason that the `rerun` custody event records. `rerun` applies the same rule.
- **Refusals.**
  - 409: the analysis is not finished, withdrawn or superseded; the revision is locked (conclusion); already signed; no
    conclusion yet; a peer review before the examiner sign-off; a newer revision exists (reanalyse); the conclusion changed since
    the request's version (L-5).
  - 403: another examiner's conclusion; a peer review of one's own analysis; a role or case role that cannot sign this kind; an
    API client, the bootstrap administrator key or a single-user principal (conclusion, sign-off, withdraw; D-47).
- **Atomicity** (D-47): the lock check and the write of a conclusion commit together; the examiner sign-off sets `locked_at`
  only if unset; unique `signoffs(analysis_id, kind)` and `analyses.supersedes_id` turn a concurrent duplicate into 409. Once
  signed, the current conclusion is the signed one.
  - 500: the original fails its re-hash (reanalyse).
- **Custody** (C-3): `conclusion_recorded`, `signoff` with the signer's identity, `status_changed` (withdraw), `rerun`.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api import jobs
from app.api.auth import BOOTSTRAP_ID, LOCAL_ID, Principal
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.api.v2_annotations import clean_body
from app.api.v2_evidence import AnalysisSummaryV2, ReviewBrief, _result
from app.api.v2_evidence import _summary as analysis_summary
from app.audit import chain as audit
from app.cases import access, review
from app.core import times
from app.core.canonical import canonical_hash
from app.core.logging import get_logger
from app.engine import rerun
from app.reports import view
from app.reports.snapshot import principal_names
from app.storage.db import Analysis, ConclusionRow, SignoffRow, get_session, utcnow

log = get_logger(__name__)
router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
ScalePosition = Literal["ai_supported_strong", "ai_supported", "inconclusive", "no_ai_indicators", "no_ai_indicators_strong", "no_opinion"]
SignoffKind = Literal["examiner", "peer_review"]
SignoffDecision = Literal["concur", "concur_with_changes", "non_concur"]
Profile = Literal["quick", "full", "deep"]
LimitationId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]{1,64}$")]
TERMINAL = frozenset({"done", "failed", "cancelled"})


class ConclusionPut(RequestModel):
    model_config = ConfigDict(extra="forbid")
    scale_position: ScalePosition
    rationale: str = Field(min_length=1, max_length=10000)
    limitations_acknowledged: list[LimitationId] = Field(default_factory=list, max_length=50)
    next_steps: str | None = Field(default=None, max_length=4000)
    declaration_text_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,64}$")
    deviation_justification: str | None = Field(default=None, max_length=10000)
    base_version: int = Field(ge=0, description="The conclusion version this edit starts from, 0 for the first (L-5); a newer version "
                                                "is refused with 409, so a stale tab cannot overwrite it")


class ConclusionOut(BaseModel):
    id: str
    analysis_id: str
    analysis_revision: int
    version: int = Field(description="1, 2, … per analysis revision (L-5)")
    author_id: str
    scale_position: ScalePosition
    machine_code: str | None
    deviation_steps: int | None = Field(description="Steps from the machine decision (positive: towards AI); null without two "
                                                    "positions. Where `steps_from` is present they are counted from the step "
                                                    "of that record on the scale")
    steps_from: Literal["capture_record", "declaration"] | None = Field(
        default=None, description="Present only where the steps are counted from the step of a provenance record that decides the "
                                  "result: a trusted capture record (the first step) or a trusted declaration (the fifth). With a "
                                  "fusion model that is the step of `machine_code`; on the uncalibrated fallback the code stands one "
                                  "step inside. Absent for a conclusion recorded before N-19n and beside any other result (N-19n, D4-01)")
    departs_from_machine: bool | None = Field(description="The position is not on the step the steps are counted from (0 steps); the "
                                                          "examiner cannot then sign 'concur' or 'concur_with_changes' (L-5, N-19), "
                                                          "as a position on it cannot be signed 'non_concur' (N-19j). Null where "
                                                          "there is nothing to compare with: no machine decision, or a result that "
                                                          "is not assessable")
    rationale: str
    limitations_acknowledged: list[str]
    next_steps: str | None
    declaration_text_id: str | None
    deviation_justification: str | None
    created_at: datetime
    signature: str
    signature_valid: bool


class SignoffPost(RequestModel):
    model_config = ConfigDict(extra="forbid")
    kind: SignoffKind
    decision: SignoffDecision
    comment: str | None = Field(default=None, max_length=4000, description="Printed in the reports. Required with 'non_concur' (the "
                                "reason) and with 'concur_with_changes' (the changes asked for): 422 without one (N-19j)")
    conclusion_id: str = Field(max_length=36, description="The conclusion the signer saw (L-5); another current conclusion is refused with 409")
    conclusion_version: int = Field(ge=1, description="Its version (L-5)")


class SignoffOut(BaseModel):
    id: str
    analysis_id: str
    analysis_revision: int
    conclusion_id: str
    principal_id: str
    principal_name: str = Field(description="The signer's display name as it is stored now; the id when the account is gone")
    kind: SignoffKind
    decision: SignoffDecision
    comment: str | None
    result_hash: str | None
    annotations_seq: int
    signed_at: datetime
    audit_event_seq: int | None
    signature: str
    signature_valid: bool


class SignoffList(BaseModel):
    items: list[SignoffOut]


class WithdrawPost(RequestModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=4000)


class ReanalyseRequest(RequestModel):
    model_config = ConfigDict(extra="forbid")
    profile: Profile | None = Field(default=None, description="Default: the profile of this revision")
    reason: str | None = Field(default=None, max_length=4000, description="Why the analysis is reanalysed (recorded in the custody "
                                                                          "log). Required when the revision is signed off.")


def _json(model: BaseModel, status_code: int = 200) -> JSONResponse:
    return JSONResponse(jsonable_encoder(model), status_code=status_code, headers=NO_STORE)


def _conclusion_json(out: ConclusionOut) -> JSONResponse:
    """N-19n (D4-01): `steps_from` is a key that is present only where the steps are counted from a record's step."""
    body = jsonable_encoder(out)
    if body.get("steps_from") is None:
        body.pop("steps_from", None)
    return JSONResponse(body, headers=NO_STORE)


_utc = times.to_utc  # one form on every engine: a time is printed in UTC (D-72)


def _analysis(db: Session, aid: str) -> Analysis:
    analysis = db.get(Analysis, aid)
    if analysis is None:
        raise ProblemError(404, *access.NOT_FOUND["analysis"])
    return analysis


def _gone(db: Session, aid: str) -> None:
    """404 when the revision no longer exists (it was purged or deleted while this request waited for its row)."""
    if review.gone(db, aid):
        db.rollback()
        raise ProblemError(404, *access.NOT_FOUND["analysis"])


def _reviewable(db: Session, analysis: Analysis) -> None:
    if analysis.status != "done":
        raise ProblemError(409, "Analysis not finished", f"The analysis is {analysis.status}; only a finished analysis is reviewed.")
    if analysis.withdrawn_at is not None:
        raise ProblemError(409, "Revision withdrawn", "This revision was withdrawn and should not be relied on.")
    if review.superseded(db, analysis.id):
        raise ProblemError(409, "Revision superseded", "A newer revision of this analysis exists; review the latest revision.")


def _case_roles(db: Session, principal: Principal, analysis: Analysis) -> set[str] | None:
    """The principal's roles on the analysis's case; None for an administrator (every role)."""
    if principal.satisfies("administrator"):
        return None
    return set(access.case_roles(db, principal.id, {analysis.case_id}).values()) if analysis.case_id else set()


def _conclusion_out(row: ConclusionRow) -> ConclusionOut:
    return ConclusionOut(id=row.id, analysis_id=row.analysis_id, analysis_revision=row.analysis_revision, version=row.version,
                         author_id=row.author_id, scale_position=row.scale_position, machine_code=row.machine_code,  # type: ignore[arg-type]
                         deviation_steps=row.deviation_steps, steps_from=row.steps_from,  # type: ignore[arg-type]
                         departs_from_machine=review.departs_of(row),
                         rationale=row.rationale, limitations_acknowledged=list(row.limitations_acknowledged or []),
                         next_steps=row.next_steps, declaration_text_id=row.declaration_text_id,
                         deviation_justification=row.deviation_justification, created_at=_utc(row.created_at),  # type: ignore[arg-type]
                         signature=row.signature, signature_valid=review.signature_valid(review.conclusion_body(row), row.signature))


def _signoff_out(row: SignoffRow, names: dict[str, str]) -> SignoffOut:
    """`names`: the signers' display names (`snapshot.principal_names`). A signer need not be a member of the case (an
    administrator signs on a single-user instance), so the sign-off says who signed and the page asks nobody else (N-10b;
    code re-verification R2-08: "Reviewed by principal ca320a67-…")."""
    return SignoffOut(id=row.id, analysis_id=row.analysis_id, analysis_revision=row.analysis_revision, conclusion_id=row.conclusion_id,
                      principal_id=row.principal_id, principal_name=names.get(row.principal_id, row.principal_id),
                      kind=row.kind, decision=row.decision, comment=row.comment,  # type: ignore[arg-type]
                      result_hash=row.result_hash, annotations_seq=row.annotations_seq, signed_at=_utc(row.signed_at),  # type: ignore[arg-type]
                      audit_event_seq=row.audit_event_seq, signature=row.signature,
                      signature_valid=review.signature_valid(review.signoff_body(row), row.signature))


# --- conclusion --------------------------------------------------------------------------------------------------------------

@router.get("/analyses/{aid}/conclusion", response_model=ConclusionOut, responses=problem_responses(401, 404))
def get_conclusion(aid: str) -> JSONResponse:
    with get_session() as db:
        _analysis(db, aid)
        row = review.current_conclusion(db, aid)
        if row is None:
            raise ProblemError(404, "No conclusion", "No conclusion has been recorded for this analysis.")
        return _conclusion_json(_conclusion_out(row))


@router.put("/analyses/{aid}/conclusion", response_model=ConclusionOut, responses=problem_responses(401, 403, 404, 409, 422))
def put_conclusion(aid: str, req: ConclusionPut, request: Request) -> JSONResponse:
    principal: Principal = request.state.principal
    named_examiner(principal, "Recording a conclusion")
    rationale, justification = clean_body(req.rationale), clean_body(req.deviation_justification)
    if rationale is None:
        raise ProblemError(422, "Invalid request", "rationale is required (RW-03).")
    def unit() -> ConclusionOut:
        # N-9: the conclusion and its custody event are one transaction; neither exists without the other
        with audit.writing() as db:
            analysis = _analysis(db, aid)
            _reviewable(db, analysis)
            if analysis.locked_at is not None:
                raise ProblemError(409, "Revision locked", "This revision is signed off and read-only; reanalyse to change the conclusion.")
            result = _result(db, analysis) or {}
            machine = result.get("decision", {}).get("code")
            # N-19n (D4-01): under a trusted capture and a trusted declaration the steps are counted from the record's step
            # on the scale (the step of the basis), and the refusal names the step it counts from
            record = view.record_of(view.decision_of(result)) if result.get("decision") else None
            counted_from = review.steps_from((record or {}).get("kind")) if machine in review.SCALE else None
            steps = review.deviation(machine, req.scale_position, counted_from)
            if steps is not None and abs(steps) >= review.JUSTIFY_FROM_STEPS and justification is None:
                place = review.STEP_NAMES[counted_from] if counted_from else f"the machine decision ({machine})"
                raise ProblemError(422, "Invalid request", f"The position is {abs(steps)} steps from {place}; "
                                                           "deviation_justification is required (RW-03).")
            if not review.hold_unlocked(db, aid):  # D-47: a sign-off since the check above wins
                _gone(db, aid)  # N-9: or a purge that this request waited for
                raise ProblemError(409, "Revision locked", "This revision was signed off while the conclusion was being recorded; "
                                                           "reanalyse to change the conclusion.")
            current = review.conclusion_version(db, aid)
            if current != req.base_version:  # L-5: an edit from a stale copy never replaces a conclusion its author did not see
                raise ProblemError(409, "Conclusion changed", _changed(current, req.base_version))
            row = ConclusionRow(id=str(uuid.uuid4()), analysis_id=aid, analysis_revision=analysis.revision, version=current + 1,
                                author_id=principal.id, scale_position=req.scale_position, machine_code=machine, rationale=rationale,
                                limitations_acknowledged=list(req.limitations_acknowledged), next_steps=clean_body(req.next_steps),
                                declaration_text_id=req.declaration_text_id, deviation_steps=steps, steps_from=counted_from,
                                deviation_justification=justification, created_at=utcnow())
            row.signature = review.sign(review.conclusion_body(row))
            db.add(row)
            try:
                db.flush()
            except IntegrityError:  # unique conclusions(analysis_id, version): a concurrent edit from the same base committed first
                db.rollback()
                raise ProblemError(409, "Conclusion changed", "Another conclusion was recorded for this revision at the same time; "
                                                              "reload it and edit again.") from None
            out = _conclusion_out(row)
            audit.commit_with(db, [{
                "action": "conclusion_recorded", "actor_id": principal.id, "analysis_id": aid, "object_ids": [out.id],
                "after_hash": canonical_hash(review.conclusion_body(row)),
                "details": {"scale_position": out.scale_position, "machine_code": out.machine_code,
                            "deviation_steps": out.deviation_steps, "auth": principal.auth,  # D-61: session or api_key
                            # N-19n (D4-01): what the steps are counted from; an event recorded before has no such field
                            **({"steps_from": counted_from} if counted_from else {})}}])
            return out

    return _conclusion_json(audit.repeated(unit))


def _changed(current: int, seen: int) -> str:
    return (f"The conclusion is now version {current}, and this request was made from version {seen}. Reload the analysis to see "
            "the current conclusion, then try again." if current else "The conclusion this request was made from no longer exists; "
            "reload the analysis.")


def _departure(conclusion: ConclusionRow) -> str:
    steps = conclusion.deviation_steps
    return f"by {abs(steps)} step{'s' if abs(steps) != 1 else ''}" if steps else "(no opinion)"


def _review_reference(db: Session, analysis: Analysis, conclusion: ConclusionRow) -> str | None:
    """N-19n (supervisor check 40): "the step of the capture record" where the reports' line names it for this conclusion
    (`view.review_reference`), else None (the words of before)."""
    result = _result(db, analysis) or {}
    return view.review_reference(view.decision_of(result), conclusion) if result.get("decision") else None


def _concur_refused(conclusion: ConclusionRow, reference: str | None) -> str:
    """The refusal of a concurring sign-off beside a conclusion that departs. Under a capture (`reference`), the dialog's
    sentences (`SignoffBlock.tsx`): "Your conclusion is 1 step from the step of the capture record, so it cannot be signed
    as concurring with the machine assessment."; it said "Your conclusion departs from the machine assessment by 1 step;
    …" of a place that no surface states there."""
    if reference is None:
        return (f"Your conclusion departs from the machine assessment {_departure(conclusion)}; sign it as not concurring with the "
                "machine assessment.")
    steps = conclusion.deviation_steps
    if not steps:  # no opinion on the scale: no position to be away from
        return "Your conclusion records no opinion on the scale, so it cannot be signed as concurring with the machine assessment."
    return f"Your conclusion is {abs(steps)} step{'s' if abs(steps) != 1 else ''} from {reference}, so it cannot be signed as concurring with the machine assessment."


COMMENT_REQUIRED = {
    "non_concur": "A sign-off that does not concur needs a comment that gives the reason. It is printed in the report.",
    "concur_with_changes": "A sign-off that concurs with changes needs a comment that says which changes. It is printed in the report."}


# --- sign-off ----------------------------------------------------------------------------------------------------------------

@router.get("/analyses/{aid}/signoffs", response_model=SignoffList, responses=problem_responses(401, 404))
def list_signoffs(aid: str) -> JSONResponse:
    with get_session() as db:
        _analysis(db, aid)
        rows = review.signoffs(db, aid)
        names = principal_names(db, {r.principal_id for r in rows})
        return _json(SignoffList(items=[_signoff_out(r, names) for r in rows]))


@router.post("/analyses/{aid}/signoffs", status_code=201, response_model=SignoffOut, responses=problem_responses(401, 403, 404, 409, 422))
def post_signoff(aid: str, req: SignoffPost, request: Request) -> JSONResponse:
    principal: Principal = request.state.principal
    named_examiner(principal, "Signing off a revision" if req.kind == "examiner" else "A peer review")
    # N-19j (browser QA Q2-02, Q2-08): "Do not concur" was accepted with an empty comment, so no reason was on record, and
    # the condition of "Concur with changes" was optional. Text in which nothing can be seen is no comment (`clean_body`)
    if req.decision in review.NEEDS_COMMENT and clean_body(req.comment) is None:
        raise ProblemError(422, "Invalid request", COMMENT_REQUIRED[req.decision])

    def unit() -> tuple[SignoffOut, int, str | None]:
        # N-9: the sign-off, the revision's lock and the `signoff` custody event are one transaction, and the row holds the
        # seq of its event: a failed append leaves no locked revision without its event (code review 12)
        with audit.writing() as db:
            analysis = _analysis(db, aid)
            _reviewable(db, analysis)
            roles = _case_roles(db, principal, analysis)
            records = review.signoffs(db, aid)
            examiner_signoff = next((s for s in records if s.kind == "examiner"), None)
            now = utcnow()
            if req.kind == "examiner":
                if not principal.satisfies("examiner") or (roles is not None and not roles & {"owner", "examiner"}):
                    raise ProblemError(403, "Forbidden", "Signing off as examiner needs the examiner role on this case.")
                if examiner_signoff is not None:
                    raise ProblemError(409, "Already signed off", "The examiner has already signed off this revision.")
                if not review.lock_revision(db, aid, now):  # D-47: set only if unset; the conclusion is read after taking the lock
                    _gone(db, aid)  # N-9: a sign-off that waited for a purge finds no revision, not a lock
                    raise ProblemError(409, "Already signed off", "Another sign-off locked this revision first.")
                conclusion = review.latest_conclusion(db, aid)
                if conclusion is None:
                    raise ProblemError(409, "Conclusion required", "Record a conclusion before signing off (RW-03).")
                if conclusion.author_id != principal.id:
                    raise ProblemError(403, "Forbidden", "Only the examiner who recorded the conclusion signs it off.")
                if (conclusion.id, conclusion.version) != (req.conclusion_id, req.conclusion_version):  # L-5: lock only what the signer saw
                    raise ProblemError(409, "Conclusion changed", _changed(conclusion.version, req.conclusion_version))
                departs = review.departs_of(conclusion)  # N-19n: from the stored steps (0: on the step)
                refused = (req.decision in review.CONCURRING and departs) or (req.decision == "non_concur" and departs is False)
                # N-19n (check 40): under a trusted capture the refusals name the step of the capture record, as the dialog does
                reference = _review_reference(db, analysis, conclusion) if refused else None
                if req.decision in review.CONCURRING and departs:
                    raise ProblemError(422, "Invalid request", _concur_refused(conclusion, reference))
                # N-19j (Q2-03, the mirror of P1-7): a conclusion equal to the machine's was signed "Different opinion" and
                # recorded as "Do not concur" under "agrees with the machine assessment"
                if req.decision == "non_concur" and departs is False:
                    raise ProblemError(422, "Invalid request", f"Your conclusion is on {reference}, so it cannot be signed as not concurring with "
                                       "the machine assessment." if reference else
                                       "Your conclusion is the machine assessment's own; sign it as concurring with the machine assessment, "
                                       "or revise the conclusion first.")
            else:
                # PRD RW-04 (L-5): a second examiner peer-reviews; the case reviewer (or owner) role is the grant, so an examiner
                # needs no instance reviewer role. A viewer never signs (CM-02).
                if not (principal.satisfies("examiner") or principal.satisfies("reviewer")) or (
                        roles is not None and not roles & {"owner", "reviewer"}):
                    raise ProblemError(403, "Forbidden", "A peer review needs the reviewer role on this case.")
                if examiner_signoff is None:
                    raise ProblemError(409, "Examiner sign-off required", "The examiner signs off before the peer review.")
                if not review.hold_revision(db, aid):  # N-9: a peer review takes the row's lock too, and 404 when it is gone
                    _gone(db, aid)
                if any(s.kind == "peer_review" for s in records):
                    raise ProblemError(409, "Already reviewed", "This revision already has a peer review.")
                conclusion = db.get(ConclusionRow, examiner_signoff.conclusion_id)
                if conclusion is None or principal.id in (examiner_signoff.principal_id, conclusion.author_id):
                    raise ProblemError(403, "Forbidden", "An examiner cannot peer-review their own analysis.")
                if (conclusion.id, conclusion.version) != (req.conclusion_id, req.conclusion_version):
                    raise ProblemError(409, "Conclusion changed", "The examiner signed a different conclusion from the one this peer review "
                                                                  "was made from. Reload the analysis.")
            row = SignoffRow(id=str(uuid.uuid4()), analysis_id=aid, analysis_revision=analysis.revision, conclusion_id=conclusion.id,
                             principal_id=principal.id, kind=req.kind, decision=req.decision, comment=clean_body(req.comment),
                             result_hash=analysis.result_hash, annotations_seq=review.annotations_seq(db, aid), signed_at=now)
            row.signature = review.sign(review.signoff_body(row))
            db.add(row)
            try:
                db.flush()
            except IntegrityError:  # unique signoffs(analysis_id, kind): a concurrent sign-off of this kind committed first
                db.rollback()
                raise ProblemError(409, "Already signed off" if req.kind == "examiner" else "Already reviewed",
                                   "A sign-off of this kind was recorded for this revision at the same time.") from None
            revision, case_id = analysis.revision, analysis.case_id
            audit.commit_with(db, [{
                "action": "signoff", "actor_id": principal.id, "analysis_id": aid, "object_ids": [row.id, conclusion.id],
                "after_hash": canonical_hash(review.signoff_body(row)),
                # N-19j: what the decision is about, so that the custody log prints it with its object (None: no comparison)
                "details": {"kind": req.kind, "decision": req.decision, "object": review.signoff_object(req.kind, conclusion),
                            "signer": principal.display_name,
                            "signer_role": principal.role, "analysis_revision": revision,
                            "locks_revision": req.kind == "examiner", "auth": principal.auth}}],
                before_commit=lambda seqs: setattr(row, "audit_event_seq", seqs[0]))
            return _signoff_out(row, principal_names(db, {row.principal_id})), revision, case_id

    out, revision, case_id = audit.repeated(unit)
    # a webhook is a notification of what is stored: one that is lost here leaves the sign-off and its custody event whole
    jobs.emit("analysis.signed_off", {"analysis_id": aid, "case_id": case_id, "analysis_revision": revision, "kind": req.kind,
                                      "decision": req.decision})
    log.info("review.signoff", analysis_id=aid, kind=req.kind, by=principal.id)
    return _json(out, 201)


# --- withdraw and reanalyse --------------------------------------------------------------------------------------------------

def named_examiner(principal: Principal, action: str) -> None:
    """Conclusions, sign-offs and withdrawals carry a named human actor (D11; RW-03, RW-04): an api_client is refused (Q-2 F6), and
    so are the bootstrap administrator key and the single-user principal, which anyone holding the key or the loopback shares, so
    one person could otherwise satisfy two-person review (D-47)."""
    if "api_client" in principal.roles:
        raise ProblemError(403, "Forbidden", f"{action} needs a named examiner; an API client cannot do it.")
    if principal.id in (BOOTSTRAP_ID, LOCAL_ID):
        raise ProblemError(403, "Forbidden", f"{action} needs a named examiner account; the bootstrap administrator key and a "
                                             "single-user instance cannot do it.")


def revision_custodian(db: Session, principal: Principal, analysis: Analysis, action: str, verb: str) -> None:
    """Who may take a revision out of use (Q-2 F5; D-60): a named examiner who signed it or wrote its conclusion, an owner of its
    case, or an administrator. Withdrawing a revision, and re-running or reanalysing a signed one, all take this rule."""
    named_examiner(principal, action)
    roles = _case_roles(db, principal, analysis)
    if roles is not None and "owner" not in roles:  # not any examiner of the case
        conclusion = review.current_conclusion(db, analysis.id)
        signers = {s.principal_id for s in review.signoffs(db, analysis.id) if s.kind == "examiner"}
        if principal.id not in signers and (conclusion is None or conclusion.author_id != principal.id):
            raise ProblemError(403, "Forbidden", "Only the examiner who recorded or signed this revision's conclusion, a case owner "
                                                 f"or an administrator can {verb}.")


def signed_rerun_reason(db: Session, principal: Principal, base: Analysis, reason: str | None, action: str, verb: str) -> str | None:
    """D-60: a new revision supersedes `base`. When `base` is signed off (`locked_at`), that needs the people of a withdrawal and
    a reason, which the `rerun` custody event records. Returns the reason to record, None for an unsigned revision."""
    if base.locked_at is None:
        return None
    revision_custodian(db, principal, base, action, verb)
    if reason is None:
        raise ProblemError(422, "Invalid request", f"{action} needs a reason: the revision is signed off, and the new revision "
                                                   "will supersede it.")
    return reason


@router.post("/analyses/{aid}/withdraw", response_model=ReviewBrief, responses=problem_responses(401, 403, 404, 409, 422))
def withdraw(aid: str, req: WithdrawPost, request: Request) -> JSONResponse:
    principal: Principal = request.state.principal
    named_examiner(principal, "Withdrawing a revision")
    reason = clean_body(req.reason)
    if reason is None:
        raise ProblemError(422, "Invalid request", "A withdrawal needs a reason.")
    def unit() -> None:
        with audit.writing() as db:  # N-9: the withdrawal and its `status_changed` event are one transaction
            analysis = _analysis(db, aid)
            revision_custodian(db, principal, analysis, "Withdrawing a revision", "withdraw it")
            if analysis.withdrawn_at is not None:
                raise ProblemError(409, "Already withdrawn", "This revision is already withdrawn.")
            previous = review.state(db, analysis)
            analysis.withdrawn_at, analysis.withdrawn_reason, analysis.withdrawn_by = utcnow(), reason, principal.id
            audit.commit_with(db, [{"action": "status_changed", "actor_id": principal.id, "analysis_id": aid,
                                    "details": {"from": previous, "to": "withdrawn", "reason": reason}}])

    audit.repeated(unit)
    with get_session() as db:
        return _json(ReviewBrief(**review.brief(db, _analysis(db, aid))))


@router.post("/analyses/{aid}/reanalyse", status_code=202, response_model=AnalysisSummaryV2, responses=problem_responses(401, 403, 404, 409, 422, 500))
def reanalyse(aid: str, request: Request, body: ReanalyseRequest | None = None) -> JSONResponse:
    """Revision n+1 of the same media, superseding this revision; the original is re-hashed first (CC-01). A signed-off
    revision is reanalysed only by its signer, its conclusion's author, a case owner or an administrator, with a reason."""
    principal: Principal = request.state.principal
    wanted = body or ReanalyseRequest()
    with get_session() as db:
        analysis = _analysis(db, aid)
        if analysis.status not in TERMINAL:
            raise ProblemError(409, "Analysis still running", f"The analysis is {analysis.status}; reanalyse it once it has finished.")
        base = review.revision_base(db, analysis)
        if review.superseded(db, base.id):
            raise ProblemError(409, "Revision superseded", "A newer revision exists; reanalyse the latest revision.")
        reason = signed_rerun_reason(db, principal, base, clean_body(wanted.reason), "Reanalysing a signed revision", "reanalyse it")
        base_id = base.id
    try:
        sha256 = audit.verify_original("rerun", actor_id=principal.id, analysis_id=aid)
    except audit.OriginalIntegrityError as e:
        raise ProblemError(500, "Evidence integrity failure", f"The original {e.problem}; the analysis was not re-run.") from None
    def unit() -> dict:
        # N-9 (code review 12): the revision, its job and its `rerun` event are one transaction, as in `v2_rerun.request_rerun`
        revision_id = rerun.prepare_revision(base_id, sha256)  # N-10a: an upload that is a copy, before the locks
        with audit.writing() as db:
            try:
                source = _analysis(db, base_id)  # K-2: a dead revision's reanalyse supersedes the revision it would have replaced
                revision = rerun.create_revision(db, source, rerun.original_path(db, source, sha256), principal_id=principal.id,
                                                 profile=wanted.profile or source.profile, revision_id=revision_id)
                jobs.submit_in(db, revision.id)
                summary = analysis_summary(revision)
                audit.commit_with(db, [{
                    "action": "rerun", "actor_id": principal.id, "analysis_id": revision.id, "object_ids": [base_id],
                    "details": {"mode": "reanalyse", "supersedes_id": base_id, "revision": summary["revision"],
                                "profile": summary["profile"], **({"retry_of": aid} if base_id != aid else {}),
                                **({"reason": reason, "supersedes_signed": True} if reason is not None else {})}}])
                return summary
            except rerun.SourceGone:  # N-9: purged or deleted while this request waited for the base row
                db.rollback()
                rerun.discard_upload(revision_id)
                raise ProblemError(404, *access.NOT_FOUND["analysis"]) from None
            except IntegrityError:  # unique analyses.supersedes_id: a concurrent reanalyse or re-fusion committed first (D-47)
                db.rollback()
                rerun.discard_upload(revision_id)
                raise ProblemError(409, "Revision superseded", "A newer revision of this analysis was created at the same time; "
                                                               "reanalyse the latest revision.") from None
            except BaseException:  # nothing was committed: the linked original goes too
                rerun.discard_upload(revision_id)
                raise

    summary = audit.repeated(unit)
    jobs.wake()
    log.info("review.reanalyse", analysis_id=aid, revision_id=summary["id"], by=principal.id)
    return JSONResponse(jsonable_encoder(AnalysisSummaryV2(**summary)), status_code=202, headers=NO_STORE)
