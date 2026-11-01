"""Principals, API keys, browser sessions and request authentication (ARCHITECTURE §5.3; PRD CM-02, PL-04; D-09; BUILD_SPEC C-1).

- **Principals** are `principals` rows with one role: administrator, examiner, reviewer, viewer or api_client. Two synthetic
  principals are not rows:
  - `bootstrap-admin`: the `VERITAS_API_KEY` (administrator), so existing deployments and clients keep working;
  - `local-examiner`: single-user mode, an unauthenticated loopback request (administrator and examiner) carrying the
    "unauthenticated instance" marker. Single-user mode is `VERITAS_SINGLE_USER`, by default on when no API key is set.
- **API keys** look like `vk_<id8>_<secret32>`. `api_key_hash` stores `<id8>:HMAC-SHA256(pepper, secret)`. The pepper is
  `VERITAS_API_KEY_PEPPER`, or a random 32-byte key at `data/keys/api_key_pepper.key` (0600). Issuing a key replaces the
  previous one (rotation). Keys travel in `X-API-Key` only, never in a query string.
- **Sessions.** Passwords are Argon2id hashes (argon2-cffi defaults). A sign-in stores a `sessions` row whose id is the
  SHA-256 of the random cookie value (`veritas_session`: HttpOnly, SameSite=Strict, Secure outside dev). Every non-GET
  request on a session must send `X-CSRF-Token` equal to the session's `csrf_secret` (double submit: the value also travels
  in the readable `veritas_csrf` cookie and in the sign-in response).
- **Resource tokens** (`?token=`, GET and HEAD only) act as their subject principal on routes whose resource matches.
- **Order:** `X-API-Key`, then the session cookie, then a resource token, then single-user mode; otherwise 401. A cookie that
  names no open session is passed over (N-9b): the request is resolved as if it carried none, and its response, the 401
  included, deletes both cookies (`session_refused`; the middleware in `app.main`), so a browser is never held by a cookie
  it cannot remove.
- **Case scoping** reads `case_members` in `app.cases.access` (C-2), applied by `security.authorize` and `security.require`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import ipaddress
import re
import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import HTTPException, Request, Response, status
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.api import rbac, tokens
from app.core.config import settings
from app.storage.db import PrincipalRow, SessionRow, get_session, utcnow

ROLES = rbac.ROLES
BOOTSTRAP_ID, LOCAL_ID = "bootstrap-admin", "local-examiner"
RESERVED_USERNAMES = frozenset({BOOTSTRAP_ID, LOCAL_ID})
SESSION_COOKIE, CSRF_COOKIE, CSRF_HEADER = "veritas_session", "veritas_csrf", "x-csrf-token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
TOKEN_METHODS = frozenset({"GET", "HEAD"})
MIN_PASSWORD_CHARS = 12
UNAUTHORIZED = "Invalid or missing API key"
_KEY = re.compile(r"vk_([0-9a-f]{8})_([0-9a-f]{32})")
_hasher = PasswordHasher()  # Argon2id with the argon2-cffi defaults (RFC 9106 low-memory profile)
_DUMMY_HASH = _hasher.hash(secrets.token_hex(16))  # checked for unknown usernames, so timing does not reveal them


@dataclass(frozen=True)
class Principal:
    id: str
    username: str
    display_name: str
    roles: frozenset[str]
    auth: str  # api_key | bootstrap | session | token | single_user
    unauthenticated_instance: bool = False
    session_id: str | None = None  # the stored (hashed) session id when authenticated by a session
    csrf_secret: str | None = None

    @property
    def role(self) -> str:
        return next(r for r in ROLES if r in self.roles)

    def satisfies(self, minimum: str) -> bool:
        return rbac.satisfies(self.roles, minimum)


def bootstrap_principal(auth: str = "bootstrap") -> Principal:
    return Principal(BOOTSTRAP_ID, BOOTSTRAP_ID, "Bootstrap administrator", frozenset({"administrator"}), auth)


def local_principal(auth: str = "single_user") -> Principal:
    return Principal(LOCAL_ID, LOCAL_ID, "Local examiner (unauthenticated instance)", frozenset({"administrator", "examiner"}), auth,
                     unauthenticated_instance=True)


def row_principal(row: PrincipalRow, auth: str, **kw: str) -> Principal:
    return Principal(row.id, row.username, row.display_name, frozenset({row.role}), auth, **kw)


def single_user_enabled() -> bool:
    """Loopback requests act as local-examiner (D-09), by default when no API key is set. Never in prod, and never behind a proxy
    the API is told to trust, where the socket address says nothing about the client (Q-2 F7)."""
    if settings.environment == "prod" or settings.trust_proxy or settings.billing_mode != "disabled":
        return False
    return settings.single_user if settings.single_user is not None else not settings.api_key


def _unauthorized(detail: str = UNAUTHORIZED) -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


# --- API keys ------------------------------------------------------------------------------------------------------

PEPPER_FILE = "api_key_pepper.key"


def _pepper() -> bytes:
    if settings.api_key_pepper:
        return settings.api_key_pepper.encode()
    path = settings.data_dir / "keys" / PEPPER_FILE
    if not path.is_file():  # N-12: in prod no new pepper over principal keys hashed with the earlier one
        from app.storage import keyguard

        keyguard.refuse_generation(keyguard.PEPPER)
    return tokens.instance_key(path)


def pepper_source() -> tuple[str, bytes | None]:
    """Where `_pepper()` takes the API-key pepper from, and the pepper, without creating one (N-12): `environment`
    (`VERITAS_API_KEY_PEPPER`), `file` (`data/keys/api_key_pepper.key`) or `absent`: the next use would generate the file."""
    if settings.api_key_pepper:
        return "environment", settings.api_key_pepper.encode()
    try:
        return "file", tokens._read_key(settings.data_dir / "keys" / PEPPER_FILE)
    except FileNotFoundError:
        return "absent", None


def _key_mac(secret: str) -> str:
    return hmac.new(_pepper(), secret.encode("ascii"), hashlib.sha256).hexdigest()


def issue_api_key(db: Session, row: PrincipalRow) -> tuple[str, str]:
    """A new `vk_<id8>_<secret32>` key for the principal, replacing any previous key. Returns (key, key id)."""
    while True:
        key_id = secrets.token_hex(4)
        if db.scalars(select(PrincipalRow.id).where(PrincipalRow.api_key_hash.like(f"{key_id}:%"))).first() is None:
            break
    secret = secrets.token_hex(16)
    row.api_key_hash = f"{key_id}:{_key_mac(secret)}"
    return f"vk_{key_id}_{secret}", key_id


def principal_for_key(db: Session, key: str) -> PrincipalRow | None:
    match = _KEY.fullmatch(key)
    if match is None:
        return None
    key_id, secret = match.groups()
    row = db.scalars(select(PrincipalRow).where(PrincipalRow.api_key_hash.like(f"{key_id}:%"))).first()
    if row is None or row.api_key_hash is None:
        return None
    return row if hmac.compare_digest(row.api_key_hash.split(":", 1)[1], _key_mac(secret)) else None


# --- passwords and sessions ----------------------------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(row: PrincipalRow | None, password: str) -> bool:
    stored = row.password_hash if row is not None else None
    try:
        return _hasher.verify(stored or _DUMMY_HASH, password) and stored is not None
    except (VerificationError, InvalidHashError):
        return False


def _session_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogateescape")).hexdigest()


def create_session(db: Session, row: PrincipalRow, user_agent: str | None) -> tuple[str, SessionRow]:
    """(cookie value, session row); the row stores only the SHA-256 of the value."""
    value, now = secrets.token_urlsafe(32), utcnow()
    session = SessionRow(id=_session_hash(value), principal_id=row.id, created_at=now,
                         expires_at=now + timedelta(seconds=settings.session_ttl_s), csrf_secret=secrets.token_urlsafe(32),
                         user_agent=(user_agent or "")[:256] or None)
    db.add(session)
    row.last_login_at = now
    return value, session


def revoke_sessions(db: Session, principal_id: str, *, keep: str | None = None) -> int:
    """Ends the principal's open sessions, all but `keep` (the stored id of the session that asks). Returns how many ended."""
    ended = update(SessionRow).where(SessionRow.principal_id == principal_id, SessionRow.revoked_at.is_(None))
    if keep is not None:
        ended = ended.where(SessionRow.id != keep)
    return db.execute(ended.values(revoked_at=utcnow())).rowcount  # type: ignore[attr-defined]


def revoke_session(session_id: str) -> None:
    with get_session() as db:
        db.execute(update(SessionRow).where(SessionRow.id == session_id, SessionRow.revoked_at.is_(None)).values(revoked_at=utcnow()))
        db.commit()


def set_session_cookies(response: Response, value: str, csrf_secret: str) -> None:
    secure = settings.environment != "dev"
    response.set_cookie(SESSION_COOKIE, value, max_age=settings.session_ttl_s, httponly=True, samesite="strict", secure=secure, path="/")
    response.set_cookie(CSRF_COOKIE, csrf_secret, max_age=settings.session_ttl_s, httponly=False, samesite="strict", secure=secure, path="/")


def clear_session_cookies(response: Response) -> None:
    """Deletes both cookies, with the attributes they were set with."""
    secure = settings.environment != "dev"
    for name in (SESSION_COOKIE, CSRF_COOKIE):
        response.delete_cookie(name, path="/", samesite="strict", secure=secure, httponly=name == SESSION_COOKIE)


def _session_principal(value: str) -> Principal:
    with get_session() as db:
        session = db.get(SessionRow, _session_hash(value))
        row = db.get(PrincipalRow, session.principal_id) if session is not None else None
    if session is None or row is None or session.revoked_at is not None or _aware(session.expires_at) <= utcnow():
        raise _unauthorized("The session is not valid; sign in again")
    if row.disabled_at is not None:
        raise _unauthorized("The principal is disabled")
    return row_principal(row, "session", session_id=session.id, csrf_secret=session.csrf_secret)


def session_refused(request: Request) -> str | None:
    """Why the request's session cookie was refused, when it was looked at and names no open session."""
    return getattr(request.state, "session_refused", None)


def session_principal(request: Request) -> Principal | None:
    """The principal of the session the request's cookie names (403 for an unsafe method without the session's CSRF token).
    None without a cookie, and for a cookie that names no open session, which is noted on the request (`session_refused`)."""
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return None
    try:
        principal = _session_principal(cookie)
    except HTTPException as refused:
        request.state.session_refused = str(refused.detail)
        return None
    given = request.headers.get(CSRF_HEADER) or ""
    if request.method not in SAFE_METHODS and not hmac.compare_digest(given.encode(), (principal.csrf_secret or "").encode()):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token missing or invalid")
    return principal


# --- request authentication ------------------------------------------------------------------------------------------

def principal_for_subject(sub: str) -> Principal | None:
    """The principal a resource token was issued to, when it can still act."""
    if sub == BOOTSTRAP_ID:
        return bootstrap_principal("token") if settings.api_key else None
    if sub == LOCAL_ID and settings.billing_mode != "disabled":
        return None  # earlier anonymous download tokens must not reopen paid customers' cases
    if sub == LOCAL_ID:
        return local_principal("token") if single_user_enabled() else None
    with get_session() as db:
        row = db.get(PrincipalRow, sub)
    return row_principal(row, "token") if row is not None and row.disabled_at is None else None


def _loopback_name(host: str) -> bool:
    host = host.strip().strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() in ("localhost", "localhost.")


def _authority_host(authority: str) -> str:
    """The host of a `Host` / `Origin` authority: `[::1]:8160` → `::1`, `localhost:5230` → `localhost`."""
    authority = authority.split("://", 1)[-1].split("/", 1)[0]
    if authority.startswith("["):
        return authority[1:].split("]", 1)[0]
    return authority.rsplit(":", 1)[0] if authority.count(":") == 1 else authority


def _loopback(request: Request) -> bool:
    """A request single-user mode may act for: a loopback socket, a loopback `Host` (DNS rebinding sends the attacker's name),
    no forwarding header (a same-host proxy would make every remote client look local), and not cross-site (a page on another
    origin posting to the local API). Q-2 F7."""
    if not _loopback_name(request.client.host if request.client is not None else ""):
        return False
    headers = request.headers
    if not _loopback_name(_authority_host(headers.get("host", ""))):
        return False
    if headers.get("x-forwarded-for") or headers.get("forwarded"):
        return False
    origin = headers.get("origin")
    if origin is not None and (origin == "null" or not _loopback_name(_authority_host(origin))):
        return False
    return headers.get("sec-fetch-site", "") != "cross-site"


def presents_key_or_session(request: Request) -> bool:
    """Whether the request carries a credential that `resolve` compares with a stored secret: an API key or a session."""
    return bool(request.headers.get("x-api-key") or request.cookies.get(SESSION_COOKIE))


def resolve(request: Request, token_resources: Callable[[Request], set[str]]) -> Principal:
    """The principal behind a request, or 401 (403 for a session request without a valid CSRF token). May read the DB."""
    key = request.headers.get("x-api-key")
    if key:
        if settings.api_key and hmac.compare_digest(key.encode(), settings.api_key.encode()):
            return bootstrap_principal()
        with get_session() as db:
            row = principal_for_key(db, key)
        if row is None:
            raise _unauthorized()
        if row.disabled_at is not None:
            raise _unauthorized("The principal is disabled")
        return row_principal(row, "api_key")
    by_session = session_principal(request)
    if by_session is not None:
        return by_session
    refused = session_refused(request) or UNAUTHORIZED  # a cookie that was passed over names the 401, when nothing else answers
    raw = request.query_params.get("token")
    if request.method in TOKEN_METHODS and raw:
        claims = tokens.verify_token(raw)
        if claims is not None and claims.res in token_resources(request):
            principal = principal_for_subject(claims.sub)
            if principal is not None and (claims.sub != LOCAL_ID or _loopback(request)):  # a local-examiner token acts from loopback only (Q-2 F8)
                return principal
        if not single_user_enabled():
            raise _unauthorized(refused)
    if single_user_enabled() and _loopback(request):
        return local_principal()
    raise _unauthorized(refused)


# --- principal rows ------------------------------------------------------------------------------------------------------

def new_principal(username: str, role: str, *, display_name: str | None = None, email: str | None = None,
                  password: str | None = None) -> PrincipalRow:
    return PrincipalRow(id=str(uuid.uuid4()), username=username, display_name=display_name or username, email=email, role=role,
                        password_hash=hash_password(password) if password else None, created_at=utcnow())


def password_rule(role: str, has_password: bool) -> str | None:
    """Why a principal of this role cannot be with, or without, a password; None when it can."""
    if role == "api_client" and has_password:
        return "An api_client principal authenticates with an API key and has no password."
    if role != "api_client" and not has_password:
        return f"A {role} principal needs a password of at least {MIN_PASSWORD_CHARS} characters."
    return None


def set_password(db: Session, row: PrincipalRow, password: str, *, keep: str | None = None) -> int:
    """Stores the hash of a new password and ends the principal's sessions: all of them when an administrator or the operator
    sets it, all but `keep` when the principal changes it from that session. Returns how many sessions ended."""
    return set_password_hash(db, row, hash_password(password), keep=keep)


def set_password_hash(db: Session, row: PrincipalRow, password_hash: str, *, keep: str | None = None) -> int:
    """`set_password` with a hash computed already: Argon2 takes a tenth of a second and more, and a caller that holds a lock
    computes it first (N-10a)."""
    row.password_hash = password_hash
    return revoke_sessions(db, row.id, keep=keep)


def set_role(db: Session, row: PrincipalRow, role: str) -> int:
    """A changed role ends the principal's sessions (the next sign-in starts with the new role); the API key and resource
    tokens act with the stored role at once. Returns how many sessions ended."""
    if role == row.role:
        return 0
    row.role = role
    return revoke_sessions(db, row.id)


def set_disabled(db: Session, row: PrincipalRow, disabled: bool) -> int:
    """A disabled principal's sessions end, and its API key and resource tokens are refused while it is disabled; enabling
    restores the key, not the sessions. Returns how many sessions ended."""
    if disabled and row.disabled_at is None:
        row.disabled_at = utcnow()
        return revoke_sessions(db, row.id)
    if not disabled:
        row.disabled_at = None
    return 0


def replace_principal(principal: Principal, **changes: object) -> Principal:
    return dataclasses.replace(principal, **changes)  # type: ignore[arg-type]
