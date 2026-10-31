"""API v2 sign-in, `/me` and principal administration (ARCHITECTURE §5.2–5.3; PRD CM-02, PL-04; D-09; BUILD_SPEC C-1).

- **Sign-in and sign-out.**
  - `POST /auth/login` (public, rate limited per client) checks an Argon2id password and sets the session and CSRF cookies.
    Failures are logged (`auth.login_failed`); the audit hash chain arrives with C-3.
  - `POST /auth/logout` (public, N-9b) revokes the session its cookie names and deletes both cookies; with no session, or
    a cookie that names none any more, it deletes the cookies all the same. An open session needs its CSRF token.
  - `GET /me` names the calling principal, how it authenticated and whether the instance is unauthenticated.
  - `POST /me/password` (N-6a): a principal with a password replaces it, giving the current one. The other sessions end. A
    wrong current password is a refused credential: it counts against the client's refused-credential limit.
- **Principals** (administrator): `GET/POST /principals`, `PATCH /principals/{id}`, `POST /principals/{id}/api-keys`.
  - People have a password of at least 12 characters; api_client principals have none.
  - Changing a password or a role, or disabling a principal, revokes its sessions. A disabled principal's key and tokens
    stop working. The operator's commands (`python -m app.cli principals …`) go through the same functions of `app.api.auth`.
  - Issuing a key replaces the previous one (rotation); the key is shown once.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from app.api import auth
from app.api.security import (
    CheckedRoute,
    authorize,
    client_id,
    credential_limiter,
    login_limiter,
    rate_limit,
    session,
)
from app.api.text import RequestModel, holds_nul
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.audit import chain as audit
from app.core import times
from app.core.logging import get_logger
from app.storage.db import PrincipalRow, get_session, utcnow

log = get_logger(__name__)
public_router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit)])
router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
Role = Literal["administrator", "examiner", "reviewer", "viewer", "api_client"]
AuthMethod = Literal["api_key", "bootstrap", "session", "token", "single_user"]


class LoginRequest(BaseModel):  # not a RequestModel (N-10): a name that holds a NUL is answered 401 like any name that is no account
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


class PrincipalOut(BaseModel):
    id: str
    username: str
    display_name: str
    email: str | None
    role: Role
    created_at: datetime
    disabled_at: datetime | None
    last_login_at: datetime | None
    has_password: bool
    has_api_key: bool


class PrincipalList(BaseModel):
    items: list[PrincipalOut]


class LoginOut(BaseModel):
    principal: PrincipalOut
    csrf_token: str = Field(description="Send as X-CSRF-Token on every non-GET request of this session")
    expires_at: datetime


class MeOut(BaseModel):
    id: str
    username: str
    display_name: str
    role: Role
    roles: list[Role]
    auth: AuthMethod
    unauthenticated_instance: bool = Field(description="Single-user mode: no credentials were checked (D-09)")


class PrincipalCreate(RequestModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    email: str | None = Field(default=None, max_length=255)
    role: Role
    password: str | None = Field(default=None, min_length=auth.MIN_PASSWORD_CHARS, max_length=1024)


class PrincipalPatch(RequestModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    email: str | None = Field(default=None, max_length=255)
    role: Role | None = None
    password: str | None = Field(default=None, min_length=auth.MIN_PASSWORD_CHARS, max_length=1024)
    disabled: bool | None = None


class PasswordChange(RequestModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=auth.MIN_PASSWORD_CHARS, max_length=1024)


class ApiKeyOut(BaseModel):
    principal_id: str
    key_id: str
    api_key: str = Field(description="vk_<id8>_<secret32>; shown once, only its HMAC is stored")
    issued_at: datetime


def _json(model: BaseModel, status_code: int = 200) -> JSONResponse:
    return JSONResponse(jsonable_encoder(model), status_code=status_code, headers=NO_STORE)


def principal_out(row: PrincipalRow) -> PrincipalOut:
    return PrincipalOut(id=row.id, username=row.username, display_name=row.display_name, email=row.email, role=row.role,  # type: ignore[arg-type]
                        created_at=times.to_utc(row.created_at), disabled_at=times.to_utc(row.disabled_at),  # type: ignore[arg-type]
                        last_login_at=times.to_utc(row.last_login_at),
                        has_password=row.password_hash is not None, has_api_key=row.api_key_hash is not None)


def _password_rule(role: str, has_password: bool) -> None:
    refused = auth.password_rule(role, has_password)
    if refused is not None:
        raise ProblemError(422, "Invalid request", refused)


def _row(db, principal_id: str) -> PrincipalRow:
    row = db.get(PrincipalRow, principal_id)
    if row is None:
        raise ProblemError(404, "Principal not found", "No principal exists with this id.")
    return row


# --- sign-in -------------------------------------------------------------------------------------------------------------

@public_router.post("/auth/login", response_model=LoginOut, responses=problem_responses(401, 422, 429))
def login(req: LoginRequest, request: Request) -> JSONResponse:
    """A browser session: the `veritas_session` cookie (HttpOnly) and the CSRF token (`veritas_csrf` cookie and body)."""
    client = client_id(request)
    if not login_limiter.check(f"login:{client}"):
        raise ProblemError(429, "Too many sign-in attempts", "Wait a minute before signing in again.")
    with get_session() as db:
        # N-10 (PG-05): a name that holds a NUL is no account, and PostgreSQL refuses a statement that carries one
        row = None if holds_nul(req.username) else db.scalars(select(PrincipalRow).where(PrincipalRow.username == req.username)).first()
        refused = not auth.verify_password(row, req.password) or row is None or row.disabled_at is not None
        known = row is not None
        if not refused:
            value, session = auth.create_session(db, row, request.headers.get("user-agent"))
            db.commit()
            out = LoginOut(principal=principal_out(row), csrf_token=session.csrf_secret, expires_at=times.to_utc(session.expires_at))  # type: ignore[arg-type]
    if refused and not known:
        # N-10 (SEC-13): a name that is no account is not written anywhere. Anyone can send one, as often as the limiter allows,
        # and what people type there by mistake is their password; the custody log does not grow for it
        log.warning("auth.login_failed", known=False, client=client)
        raise ProblemError(401, "Sign-in failed", "The username or password is not correct, or the account is disabled.")
    if refused:  # recorded once the read transaction is closed (SQLite)
        log.warning("auth.login_failed", known=True, username=req.username, client=client)
        # N-9: appended after the fact on purpose. Nothing was changed, and a failed append fails the request as a refusal does
        audit.record("login_failed", actor_id=audit.ANONYMOUS_ACTOR, details={"username": req.username, "client": client})
        raise ProblemError(401, "Sign-in failed", "The username or password is not correct, or the account is disabled.")
    log.info("auth.login", principal_id=out.principal.id, client=client)
    # N-9: appended after the session row. A lost `login` is acceptable: when the append fails the request fails before the
    # cookie is set, so the stored session is never used (it expires), and the client signs in again
    audit.record("login", actor_id=out.principal.id, details={"client": client, "auth": "session"})
    response = _json(out)
    auth.set_session_cookies(response, value, out.csrf_token)
    return response


@public_router.post("/auth/logout", status_code=204, responses=problem_responses(403, 429))
async def logout(request: Request) -> Response:
    """Ends the session the request's cookie names and deletes both cookies. A request with no session, or with a cookie
    that names none any more, is answered the same way, so a browser can always drop its cookies. A session that is open
    needs its CSRF token, as on every other write."""
    principal = await session(request)
    if principal is not None and principal.session_id is not None:
        await run_in_threadpool(auth.revoke_session, principal.session_id)
        log.info("auth.logout", principal_id=principal.id)
    response = Response(status_code=204, headers=NO_STORE)
    auth.clear_session_cookies(response)
    return response


@router.get("/me", response_model=MeOut, responses=problem_responses(401))
def me(request: Request) -> JSONResponse:
    p: auth.Principal = request.state.principal
    return _json(MeOut(id=p.id, username=p.username, display_name=p.display_name, role=p.role,  # type: ignore[arg-type]
                       roles=[r for r in auth.ROLES if r in p.roles], auth=p.auth,  # type: ignore[arg-type]
                       unauthenticated_instance=p.unauthenticated_instance))


@router.post("/me/password", status_code=204, responses=problem_responses(401, 403, 409, 422, 429))
def change_own_password(req: PasswordChange, request: Request) -> Response:
    """The calling principal replaces its own password. The current password is checked again, whatever the request is
    authenticated by. The principal's other sessions end; the calling session stays."""
    p: auth.Principal = request.state.principal
    client = client_id(request)

    # N-10a (code re-verification, the reviewer's notes): the two Argon2 computations, the check of the current password and
    # the hash of the new one, come before the unit; under the append lock they stopped every custody append of the process
    with get_session() as db:
        row = db.get(PrincipalRow, p.id) if p.auth in ("session", "api_key") else None
        if row is None or row.password_hash is None:
            raise ProblemError(409, "No password to change", "This account does not sign in with a password.")
        checked, correct = row.password_hash, auth.verify_password(row, req.current_password)
    if correct and req.new_password == req.current_password:
        raise ProblemError(422, "Invalid request", "The new password is the current password.")
    new_hash = auth.hash_password(req.new_password) if correct else None

    def unit() -> tuple[bool, int]:
        with audit.writing() as db:  # N-9: the new password, the ended sessions and `permission_changed` are one transaction
            row = db.get(PrincipalRow, p.id)
            if row is None or row.password_hash is None:
                raise ProblemError(409, "No password to change", "This account does not sign in with a password.")
            if row.password_hash != checked:
                # the stored password changed since the check above (another request of this principal, or an
                # administrator): the answer above is about a password that is gone, so it is asked again, here
                if not auth.verify_password(row, req.current_password):
                    return True, 0
            elif not correct:
                return True, 0
            ended = auth.set_password_hash(db, row, new_hash or auth.hash_password(req.new_password), keep=p.session_id)
            audit.commit_with(db, [{"action": "permission_changed", "actor_id": p.id, "object_ids": [p.id],
                                    "details": {"change": "password_changed", "principal_id": p.id, "by": "the principal",
                                                "auth": p.auth, "sessions_ended": ended}}])
            return False, ended

    refused, ended = audit.repeated(unit)
    if refused:  # recorded once the transaction is closed; nothing was changed (as `login_failed` of a sign-in)
        credential_limiter.check(client)
        log.warning("auth.password_change_refused", principal_id=p.id, client=client)
        audit.record("login_failed", actor_id=p.id, object_ids=[p.id],
                     details={"username": p.username, "client": client, "at": "password change", "auth": p.auth})
        raise ProblemError(403, "Current password not correct", "The current password is not correct.")
    log.info("auth.password_changed", principal_id=p.id, sessions_ended=ended)
    return Response(status_code=204, headers=NO_STORE)


# --- principals ----------------------------------------------------------------------------------------------------------

@router.get("/principals", response_model=PrincipalList, responses=problem_responses(401, 403))
def list_principals() -> JSONResponse:
    with get_session() as db:
        rows = db.scalars(select(PrincipalRow).order_by(PrincipalRow.created_at, PrincipalRow.username)).all()
        return _json(PrincipalList(items=[principal_out(r) for r in rows]))


@router.post("/principals", status_code=201, response_model=PrincipalOut, responses=problem_responses(401, 403, 409, 422))
def create_principal(req: PrincipalCreate, request: Request) -> JSONResponse:
    _password_rule(req.role, req.password is not None)
    if req.username in auth.RESERVED_USERNAMES:
        raise ProblemError(409, "Username reserved", f"{req.username} names a built-in principal.")
    actor = request.state.principal.id

    def unit() -> PrincipalOut:
        with audit.writing() as db:  # N-9: an account exists with its `permission_changed` event or not at all
            if db.scalars(select(PrincipalRow.id).where(PrincipalRow.username == req.username)).first() is not None:
                raise ProblemError(409, "Username taken", f"A principal named {req.username} already exists.")
            row = auth.new_principal(req.username, req.role, display_name=req.display_name, email=req.email,
                                     password=req.password)
            db.add(row)
            try:
                db.flush()
            except IntegrityError as e:
                db.rollback()
                raise ProblemError(409, "Username taken", f"A principal named {req.username} already exists.") from e
            out = principal_out(row)
            audit.commit_with(db, [{"action": "permission_changed", "actor_id": actor, "object_ids": [out.id],
                                    "details": {"change": "principal_created", "principal_id": out.id,
                                                "username": out.username, "role": out.role}}])
            return out

    out = audit.repeated(unit)
    log.info("principal.created", principal_id=out.id, role=out.role, by=actor)
    return _json(out, 201)


@router.patch("/principals/{principal_id}", response_model=PrincipalOut, responses=problem_responses(401, 403, 404, 422))
def update_principal(principal_id: str, req: PrincipalPatch, request: Request) -> JSONResponse:
    actor, fields = request.state.principal.id, sorted(req.model_fields_set)

    def unit() -> PrincipalOut:
        with audit.writing() as db:  # N-9: a role, a password or a disabled account changes with its event or not at all
            row = _row(db, principal_id)
            role = req.role or row.role
            _password_rule(role, req.password is not None or row.password_hash is not None)
            if req.display_name is not None:
                row.display_name = req.display_name
            if "email" in req.model_fields_set:
                row.email = req.email
            auth.set_role(db, row, role)
            if req.password is not None:
                auth.set_password(db, row, req.password)
            if req.disabled is not None:
                auth.set_disabled(db, row, req.disabled)
            db.flush()
            out = principal_out(row)
            audit.commit_with(db, [{"action": "permission_changed", "actor_id": actor, "object_ids": [principal_id],
                                    "details": {"change": "principal_updated", "principal_id": principal_id, "fields": fields,
                                                "role": out.role, "disabled": out.disabled_at is not None}}])
            return out

    out = audit.repeated(unit)
    log.info("principal.updated", principal_id=principal_id, fields=fields, by=actor)
    return _json(out)


@router.post("/principals/{principal_id}/api-keys", status_code=201, response_model=ApiKeyOut,
             responses=problem_responses(401, 403, 404, 409))
def issue_api_key(principal_id: str, request: Request) -> JSONResponse:
    actor = request.state.principal.id

    def unit() -> tuple[str, str]:
        with audit.writing() as db:  # N-9: a key that replaces the old one is issued with its event or not at all
            row = _row(db, principal_id)
            if row.disabled_at is not None:
                raise ProblemError(409, "Principal disabled", "Enable the principal before issuing it an API key.")
            key, key_id = auth.issue_api_key(db, row)
            audit.commit_with(db, [{"action": "permission_changed", "actor_id": actor, "object_ids": [principal_id],
                                    "details": {"change": "api_key_issued", "principal_id": principal_id, "key_id": key_id}}])
            return key, key_id

    key, key_id = audit.repeated(unit)
    log.info("principal.api_key_issued", principal_id=principal_id, key_id=key_id, by=actor)
    return _json(ApiKeyOut(principal_id=principal_id, key_id=key_id, api_key=key, issued_at=utcnow()), 201)
