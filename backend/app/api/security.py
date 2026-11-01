"""Request authentication and route authorisation (ARCHITECTURE §5.3; PRD CM-02, CM-04, PL-04; BUILD_SPEC C-1, C-2), the
resources a resource token may open, and per-client token-bucket rate limiters.

- **`authorize`** is the dependency of every authenticated router:
  - it resolves the principal (`app.api.auth.resolve`) and keeps it on `request.state.principal` for the handler;
  - it looks the matched route up in `app.api.rbac.ROUTES` and fails closed with 403 when the route has no rule;
  - it checks the principal's role against the route's minimum role;
  - on a case resource (a path naming a case, an analysis, an exhibit or a media object), a non-administrator must be a
    member of the case whose case role satisfies the minimum role (`app.cases.access`). A non-member gets the resource's
    not-found answer, so other cases do not leak.
- **`require(role, case_from=…)`** is a per-route dependency for stricter checks. With `case_from`, the same case check
  applies to a case the request names outside its path.
- **Before the body (SEC-03).** Every authenticated router has the route class `CheckedRoute`, which asks `rate_limit` and
  `authorize` before the framework reads the request body: an upload that will be refused is never spooled.
- **NUL (PG-05).** `refuse_nul` answers 404 for a path or query value that holds a NUL character, before anything is looked up.
- **Limiters (SEC-07).** `rate_limit` is declared before `authorize` on every router, so a request without valid credentials
  is counted like any other. A client whose API keys or sessions were refused `credential_failures_per_minute` times is
  answered 429 before its next key or session is looked at, right or wrong: a guess is then not answered at all. A
  session cookie that names no open session counts as refused even when the request is then answered by a resource token
  or by single-user mode (N-9b).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import HTTPException, Request, Response, status
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool

from app.api import auth, rbac, text, tokens
from app.cases import access
from app.core.config import settings
from app.storage.db import get_session

BOOTSTRAP_PRINCIPAL = auth.BOOTSTRAP_ID  # VERITAS_API_KEY authenticates as this principal (§5.3)
LOCAL_PRINCIPAL = auth.LOCAL_ID  # single-user mode (D-09 "unauthenticated instance")


def make_download_token(analysis_id: str, principal_id: str) -> str:
    """v1 `download-token`: an `analysis:<id>` resource token acting as the calling principal, so browsers never put a key
    in a URL."""
    token, _exp = tokens.make_token(f"analysis:{analysis_id}", principal_id, min(settings.download_token_ttl_s, tokens.MAX_TTL_S))
    return token


def _exhibit_owner(exhibit_id: str) -> str | None:
    from app.storage.db import ExhibitRow

    with get_session() as db:
        row = db.get(ExhibitRow, exhibit_id)
        return row.analysis_id if row is not None else None


def token_resources(request: Request) -> set[str]:
    """The token resources a GET route accepts, from its path parameters (§5.3). May read the DB (exhibit owner)."""
    params, path = request.path_params, request.url.path
    accepted: set[str] = set()
    if aid := params.get("aid"):
        accepted.add(f"analysis:{aid}")
        if path.endswith(("/events", "/progress")):
            accepted.add(f"events:{aid}")
    if exhibit_id := params.get("exhibit_id"):
        accepted.add(f"exhibit:{exhibit_id}")
        if owner := _exhibit_owner(exhibit_id):
            accepted.add(f"analysis:{owner}")
    for key, kind in (("media_id", "media"), ("report_id", "report")):
        if value := params.get(key):
            accepted.add(f"{kind}:{value}")
    return accepted


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def _case_denial(principal: auth.Principal, params: Mapping[str, Any], minimum: str) -> access.Denial | None:
    with get_session() as db:
        return access.denial(db, principal, params, minimum)


def _outsider(principal: auth.Principal, params: Mapping[str, Any]) -> access.Denial | None:
    with get_session() as db:
        return access.outsider(db, principal, params)


def _denied(request: Request, denial: access.Denial) -> Exception:
    """problem+json under /api/v2, with the title and detail of the handler's own not found; v1 bodies elsewhere."""
    if request.url.path.startswith("/api/v2/"):
        from app.api.v2 import ProblemError  # app.api.v2 imports this module

        return ProblemError(denial.status, denial.title, denial.detail)
    return HTTPException(status_code=denial.status, detail=denial.title if denial.status == 404 else denial.detail)


async def authenticate(request: Request) -> auth.Principal:
    """The request's principal, resolved once per request (401 without valid credentials; 429 for a key or a session from a
    client whose keys and sessions were refused too often in the last minute)."""
    principal = getattr(request.state, "principal", None)
    if principal is None:
        guessable = auth.presents_key_or_session(request)  # a resource token is a MAC of the signing key, not a secret to guess
        client = client_id(request)
        if guessable and not credential_limiter.available(client):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Too many requests with credentials that were refused; wait a minute")
        try:
            principal = await run_in_threadpool(auth.resolve, request, token_resources)
        except HTTPException as refused:
            if guessable and refused.status_code == status.HTTP_401_UNAUTHORIZED:
                credential_limiter.check(client)
            raise
        if auth.session_refused(request) is not None:  # N-9b: a cookie that was passed over is a refused credential,
            credential_limiter.check(client)  # whatever the request then acted by (a resource token, single-user mode)
        request.state.principal = principal
    return principal


async def session(request: Request) -> auth.Principal | None:
    """The principal of the request's session cookie alone, for the sign-out (N-9b): None without a cookie and for one that
    names no open session, which counts as a refused credential here as in `authenticate`; 403 for a session without its
    CSRF token; 429 from a client whose keys and sessions were refused too often."""
    if not request.cookies.get(auth.SESSION_COOKIE):
        return None
    client = client_id(request)
    if not credential_limiter.available(client):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Too many requests with credentials that were refused; wait a minute")
    principal = await run_in_threadpool(auth.session_principal, request)
    if principal is None:
        credential_limiter.check(client)
    return principal


async def authorize(request: Request) -> None:
    """Authentication, the declared rule of the matched route (fail closed), then case membership on case resources. Decided
    once per request: `CheckedRoute` asks before the body is read, the router dependency after."""
    if getattr(request.state, "authorized", False):
        return
    principal = await authenticate(request)
    minimum = rbac.minimum_role(request.method, getattr(request.scope.get("route"), "path", ""))
    if minimum is None:
        raise _forbidden("No access rule is declared for this route")
    if not principal.satisfies(minimum):
        # L-7: to a principal with no part in the case the resource is not found, whatever role the route asks for
        if request.path_params:
            hidden = await run_in_threadpool(_outsider, principal, dict(request.path_params))
            if hidden is not None:
                raise _denied(request, hidden)
        raise _forbidden(f"This action needs the {minimum} role")
    if request.path_params and not principal.satisfies("administrator"):
        denial = await run_in_threadpool(_case_denial, principal, dict(request.path_params), minimum)
        if denial is not None:
            raise _denied(request, denial)
    from app.billing.service import guard_submission

    guard_submission(request.method, getattr(request.scope.get("route"), "path", ""))
    request.state.authorized = True


def require(minimum: str, *, case_from: Callable[[Request], str | None] | None = None) -> Callable[[Request], Awaitable[auth.Principal]]:
    """A per-route dependency: the principal must satisfy `minimum`. With `case_from`, a non-administrator must also be a member
    of the case the request names, with a case role satisfying `minimum` (a non-member reads the case as not found)."""
    if minimum not in rbac.SATISFIES:
        raise ValueError(f"unknown role {minimum!r}")

    async def dependency(request: Request) -> auth.Principal:
        principal = await authenticate(request)
        case_id = case_from(request) if case_from is not None else None
        if not principal.satisfies(minimum):
            if case_id is not None:
                hidden = await run_in_threadpool(_outsider, principal, {"case_id": case_id})
                if hidden is not None:
                    raise _denied(request, hidden)
            raise _forbidden(f"This action needs the {minimum} role")
        if case_id is not None and not principal.satisfies("administrator"):
            denial = await run_in_threadpool(_case_denial, principal, {"case_id": case_id}, minimum)
            if denial is not None:
                raise _denied(request, denial)
        return principal

    return dependency


class RateLimiter:
    def __init__(self, per_minute: int, max_clients: int = 10000):
        self.rate = per_minute / 60.0
        self.cap = float(per_minute)
        self.max_clients = max_clients
        self.buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self.lock = threading.Lock()

    def check(self, client: str) -> bool:
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.pop(client, (self.cap, now))
            tokens = min(self.cap, tokens + (now - last) * self.rate)
            allowed = tokens >= 1
            self.buckets[client] = (tokens - 1 if allowed else tokens, now)  # re-inserted => most recently seen
            while len(self.buckets) > self.max_clients:
                self.buckets.popitem(last=False)  # evict least-recently-seen, never the active ones
            return allowed

    def available(self, client: str) -> bool:
        """Whether `check` would pass now, without taking anything."""
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(client, (self.cap, now))
            return min(self.cap, tokens + (now - last) * self.rate) >= 1


limiter = RateLimiter(settings.rate_limit_per_minute)
login_limiter = RateLimiter(settings.login_rate_per_minute)  # sign-in attempts per client (§5.3)
credential_limiter = RateLimiter(settings.credential_failures_per_minute)  # refused API keys and sessions per client (SEC-07)


def client_id(request: Request) -> str:
    """The address the limiters and the sign-in events know a client by (SEC-11).

    Without `trust_proxy` it is the socket peer, and no header changes it. With it, each of the operator's
    `trusted_proxy_hops` proxies appended the address it saw to `X-Forwarded-For`, so the client is that many entries from
    the right. A header with fewer entries did not pass every proxy: its leftmost entry is the furthest one known."""
    if settings.trust_proxy:
        hops = [hop.strip() for hop in (request.headers.get("x-forwarded-for") or "").split(",") if hop.strip()]
        if hops:
            return hops[-min(settings.trusted_proxy_hops, len(hops))]
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    """One request, one count, whoever asks first (`CheckedRoute` before the body is read, the router dependency after)."""
    if getattr(request.state, "rate_counted", False):
        return
    request.state.rate_counted = True
    if not limiter.check(client_id(request)):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")


async def refuse_nul(request: Request) -> None:
    """A dependency of every route, and asked by `CheckedRoute` before the credentials (N-10; PG-05): a decoded path value or a
    query name or value that holds a NUL character names nothing. It is not found, for every caller, before any lookup:
    PostgreSQL refuses a statement that carries one."""
    values = (*request.path_params.values(), *(part for pair in request.query_params.multi_items() for part in pair))
    if any(text.holds_nul(value) for value in values):
        if request.url.path.startswith("/api/v2/"):
            from app.api.v2 import ProblemError  # app.api.v2 imports this module

            raise ProblemError(404, "Not found", "No resource exists at this address.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


class CheckedRoute(APIRoute):
    """The route class of every authenticated router (N-10; SEC-03). FastAPI reads a request's body, and spools an upload's
    file parts to disk, before it runs the route's dependencies. So the limiter, the credentials and the route's rule are
    asked here, before the handler that reads: a request that will be refused (429, 401, 403, or the not-found of a case the
    caller has no part in) is refused with none of its body consumed. The router dependencies stay declared, in the same
    order, and find the request already counted and decided."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def checked(request: Request) -> Response:
            await rate_limit(request)
            await refuse_nul(request)
            await authorize(request)
            return await handler(request)

        return checked
