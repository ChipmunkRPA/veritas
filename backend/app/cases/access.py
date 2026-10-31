"""Per-case access (ARCHITECTURE §5.3 RBAC; PRD CM-02, CM-04; BUILD_SPEC C-2).

- **Case roles.** A member holds one role on a case: owner, examiner, reviewer or viewer. The principal's own role still
  applies first (`security.authorize`); on a case resource the case role must also satisfy the route's minimum role:
  - `viewer` routes: any membership;
  - `examiner` routes: owner or examiner;
  - `reviewer` routes: owner or reviewer.
  A peer review (RW-04) is granted by the case: the reviewer or owner case role, held by an examiner, reviewer or administrator
  principal; no instance reviewer role is needed (L-5, ADR 0011).
  Changing members needs the owner role (`app.api.v2_cases`).
- **Administrators** see and act on every case without membership.
- **Resources.**
  - An analysis belongs to its `case_id`; an exhibit and a report to its analysis's case; an annotation to its `case_id` (C-4).
  - A media object may be an item of several cases (`case_media`); one membership is enough.
  - An analysis or media object in no case (v0.3 rows, benchmark rows) is visible to administrators only.
- **No leakage** (CM-04): to a principal that is not a member, a case resource is not found (404), never 403. This holds
  before the principal's own role is weighed (L-7): a role refusal (403) is given only to a member, or on a route that names
  no case resource.
  An id that names nothing gets the same answer at the same point, before the request is read (N-14b): what a request
  holds (a body or a query that does not validate, a role the handler refuses) cannot tell a real id from an invented one.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import ColumnElement, Select, select
from sqlalchemy.orm import Session

from app.storage.db import (
    Analysis,
    AnnotationRow,
    BatchRow,
    CaseMediaRow,
    CaseMemberRow,
    CaseRow,
    ExhibitRow,
    MediaObject,
    PackageRow,
    ReportRow,
)

CASE_ROLES = ("owner", "examiner", "reviewer", "viewer")
# D-67 (N-19a, N-19j; browser QA Q2-10): the case roles a principal of an instance role can use, which are the roles the
# member form offers (roles.ts `CASE_ROLES_FOR`). An examiner owns, examines and peer-reviews (ADR 0011: the case grants
# the peer review); a reviewer reviews and reads; a viewer reads. A case role above them gains no right, and the record
# of it is wrong: the owner gave a viewer the case role examiner
CASE_ROLES_FOR: dict[str, tuple[str, ...]] = {
    "administrator": CASE_ROLES, "examiner": CASE_ROLES, "api_client": CASE_ROLES, "reviewer": ("reviewer", "viewer"), "viewer": ("viewer",)}


def case_roles_for(instance_role: str | None) -> tuple[str, ...]:
    return CASE_ROLES_FOR.get(instance_role or "", ("viewer",))
CASE_SATISFIES: dict[str, frozenset[str]] = {
    "any": frozenset(CASE_ROLES),
    "viewer": frozenset(CASE_ROLES),
    "examiner": frozenset({"owner", "examiner"}),
    "reviewer": frozenset({"owner", "reviewer"}),
    "administrator": frozenset(),
}
NOT_FOUND: dict[str, tuple[str, str]] = {
    "case": ("Case not found", "No case exists with this id."),
    "analysis": ("Analysis not found", "No analysis exists with this id."),
    "exhibit": ("Exhibit not found", "No exhibit exists with this id."),
    "media": ("Media not found", "No media object exists with this id."),
    "annotation": ("Annotation not found", "No annotation exists with this id."),
    "report": ("Report not found", "No report exists with this id."),
    "package": ("Package not found", "No evidence package exists with this id."),
    "batch": ("Batch not found", "No batch exists with this id."),
}


class Actor(Protocol):
    @property
    def id(self) -> str: ...

    def satisfies(self, minimum: str) -> bool: ...


@dataclass(frozen=True)
class Denial:
    status: int  # 404: not a member (reads as not found); 403: a member whose case role is too low
    title: str
    detail: str


@dataclass(frozen=True)
class Scope:
    kind: str  # case | analysis | exhibit | media | annotation
    exists: bool
    case_ids: frozenset[str]


def resource_scope(db: Session, params: Mapping[str, Any]) -> Scope | None:
    """The cases the resource named by a route's path parameters belongs to, or None when it names no case resource."""
    if case_id := params.get("case_id"):
        return Scope("case", db.get(CaseRow, case_id) is not None, frozenset({case_id}))
    if aid := params.get("aid"):
        analysis = db.get(Analysis, aid)
        return Scope("analysis", analysis is not None, _cases_of(analysis))
    if exhibit_id := params.get("exhibit_id"):
        row = db.get(ExhibitRow, exhibit_id)
        return Scope("exhibit", row is not None, _cases_of(db.get(Analysis, row.analysis_id) if row is not None else None))
    if media_id := params.get("media_id"):
        cases = db.scalars(select(CaseMediaRow.case_id).where(CaseMediaRow.media_id == media_id))
        return Scope("media", db.get(MediaObject, media_id) is not None, frozenset(cases))
    if annotation_id := params.get("annotation_id"):
        annotation = db.get(AnnotationRow, annotation_id)
        cases = frozenset({annotation.case_id}) if annotation is not None and annotation.case_id else frozenset()
        return Scope("annotation", annotation is not None, cases)
    if report_id := params.get("report_id"):
        report = db.get(ReportRow, report_id)
        return Scope("report", report is not None, _cases_of(db.get(Analysis, report.analysis_id) if report is not None else None))
    if package_id := params.get("package_id"):
        package = db.get(PackageRow, package_id)
        return Scope("package", package is not None, _cases_of(db.get(Analysis, package.analysis_id) if package is not None else None))
    if batch_id := params.get("batch_id"):
        batch = db.get(BatchRow, batch_id)
        return Scope("batch", batch is not None, frozenset({batch.case_id}) if batch is not None else frozenset())
    return None


def _cases_of(analysis: Analysis | None) -> frozenset[str]:
    return frozenset({analysis.case_id}) if analysis is not None and analysis.case_id else frozenset()


def case_roles(db: Session, principal_id: str, case_ids: frozenset[str] | set[str]) -> dict[str, str]:
    """The principal's role on each of these cases it is a member of."""
    if not case_ids:
        return {}
    rows = db.scalars(select(CaseMemberRow).where(CaseMemberRow.principal_id == principal_id, CaseMemberRow.case_id.in_(case_ids)))
    return {row.case_id: row.role for row in rows}


def outsider(db: Session, principal: Actor, params: Mapping[str, Any]) -> Denial | None:
    """The not-found denial of a case resource this principal has no part in: the resource named by `params` does not exist, or
    the principal is a member of none of its cases. Checked before a role refusal, so a 403 never tells a non-member that the
    resource exists (CM-04; L-7: a viewer outside the case got 403 on cancel where an examiner got 404). None for an
    administrator, for a member, and when `params` name no case resource."""
    if principal.satisfies("administrator"):
        return None
    scope = resource_scope(db, params)
    if scope is None:
        return None
    if not scope.exists or not case_roles(db, principal.id, scope.case_ids):
        return Denial(404, *NOT_FOUND[scope.kind])
    return None


def denial(db: Session, principal: Actor, params: Mapping[str, Any], minimum: str) -> Denial | None:
    """Why the principal may not act on the case resource named by `params` with a route of this minimum role, or None.

    A resource that does not exist is not found here, as one of a case the principal has no part in is (N-14b; S2-02):
    `security.authorize` asks before the body and the query are read, so the two answers are the same whatever the
    request holds. The handler's own 404 answers an administrator, who is never refused here."""
    if principal.satisfies("administrator"):
        return None
    scope = resource_scope(db, params)
    if scope is None:
        return None
    roles = case_roles(db, principal.id, scope.case_ids) if scope.exists else {}
    if not roles:
        return Denial(404, *NOT_FOUND[scope.kind])
    allowed = CASE_SATISFIES.get(minimum, frozenset(CASE_ROLES))
    if not set(roles.values()) & allowed:
        needed = " or ".join(r for r in CASE_ROLES if r in allowed) or "administrator"
        return Denial(403, "Forbidden", f"Your role on this case does not allow this action (it needs {needed}).")
    return None


def member_case_ids(principal_id: str) -> Select[tuple[str]]:
    return select(CaseMemberRow.case_id).where(CaseMemberRow.principal_id == principal_id)


def visible_cases(principal: Actor) -> ColumnElement[bool] | None:
    """A WHERE clause over `cases` limiting rows to the principal's cases, or None for an administrator."""
    return None if principal.satisfies("administrator") else CaseRow.id.in_(member_case_ids(principal.id))


def visible_analyses(principal: Actor) -> ColumnElement[bool] | None:
    """A WHERE clause over `analyses` limiting rows to the principal's cases, or None for an administrator."""
    return None if principal.satisfies("administrator") else Analysis.case_id.in_(member_case_ids(principal.id))
