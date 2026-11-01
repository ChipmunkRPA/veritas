"""API v2 progress routes (ARCHITECTURE §3.3, §5.1–5.2): polling and Server-Sent Events over `analysis_events`.

- Errors are `application/problem+json`, and JSON responses are `no-store`.
- The SSE stream polls the table, which works across processes and SQLite. It honours `Last-Event-ID` (or `?after=`),
  sends a heartbeat comment, closes after a terminal event or `sse_max_s`, and sets `X-Accel-Buffering: no` for nginx.
- Browsers cannot send headers with EventSource, so GET routes also accept a resource token (`?token=`, §5.3):
  `events:<id>` or `analysis:<id>`.

Shared v2 conventions live here too: `ProblemError` and the problem+json handlers for every `/api/v2` error, and the
cursor helpers behind `{items, next_cursor}` collections (§5.1).
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from http import HTTPStatus
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.exc import DataError, OperationalError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import holds_nul
from app.core.config import settings
from app.core.lifecycle import STOPPING
from app.core.logging import get_logger
from app.engine.events import TERMINAL_EVENTS, event_json, list_events
from app.storage import space
from app.storage.db import Analysis, get_session

log = get_logger(__name__)
router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
PREFIX = "/api/v2"
NO_STORE = {"Cache-Control": "no-store"}
TERMINAL_STATUSES = ("done", "failed", "cancelled")
MAX_LIMIT, DEFAULT_LIMIT = 200, 50
Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT, description="Page size (≤ 200)")]
Cursor = Annotated[str | None, Query(max_length=512, description="Opaque cursor from `next_cursor`")]
T = TypeVar("T")


class Problem(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str


class ProblemError(Exception):
    """Raised by v2 routes; rendered as application/problem+json (RFC 9457)."""

    def __init__(self, status: int, title: str, detail: str):
        super().__init__(detail)
        self.status, self.title, self.detail = status, title, detail


def problem_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {code: {"model": Problem, "content": {"application/problem+json": {}}} for code in codes}


def install_problem_handlers(app: FastAPI) -> None:
    """problem+json for every error under /api/v2; other paths keep FastAPI's default bodies (v1 clients). A value the database
    refuses is a 422 on both."""

    async def on_problem(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, ProblemError)
        return problem(exc.status, exc.title, exc.detail, request.url.path)

    async def on_http(request: Request, exc: Exception) -> Any:
        assert isinstance(exc, StarletteHTTPException)
        if not request.url.path.startswith(PREFIX):
            return await http_exception_handler(request, exc)
        detail = exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
        response = problem(exc.status_code, HTTPStatus(exc.status_code).phrase, detail, request.url.path)
        for name, value in (exc.headers or {}).items():
            response.headers[name] = value
        return response

    async def on_validation(request: Request, exc: Exception) -> Any:
        assert isinstance(exc, RequestValidationError)
        if not request.url.path.startswith(PREFIX):
            return await request_validation_exception_handler(request, exc)
        parts = [f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', 'invalid')}" for e in exc.errors()[:5]]
        return problem(422, "Invalid request", "; ".join(parts) or "The request is invalid.", request.url.path)

    async def on_data_error(request: Request, exc: Exception) -> Any:
        """N-10 (PG-05): the database refused a value (on PostgreSQL a NUL character in text, a number out of range), which is
        the request's fault and not a failure of the server. The checks in front (`app.api.text`) name what is known; this is
        for what they do not name. Neither the statement nor the value is repeated, in the answer or in the log."""
        assert isinstance(exc, DataError)
        log.warning("api.data_error", method=request.method, route=getattr(request.scope.get("route"), "path", None),
                    error=type(exc.orig).__name__, sqlstate=getattr(exc.orig, "sqlstate", None))
        detail = "A value in the request cannot be stored or compared."
        if request.url.path.startswith(PREFIX):
            return problem(422, "Invalid request", detail, request.url.path)
        return JSONResponse(status_code=422, content={"detail": detail}, headers=NO_STORE)

    async def on_storage_error(request: Request, exc: Exception) -> Any:
        """N-13 (OPS-19): a write that met a full volume, or a database with no room left, is answered 507 in words instead
        of a bare 500 (the floor of `storage.space` refuses most uploads before this). Any other operating system or
        database error is raised again and stays the 500 it was."""
        if not space.is_full(exc):
            raise exc
        log.error("api.storage_full", method=request.method, route=getattr(request.scope.get("route"), "path", None),
                  error=type(exc).__name__)
        if request.url.path.startswith(PREFIX):
            return problem(507, "Insufficient storage", space.FULL, request.url.path)
        return JSONResponse(status_code=507, content={"detail": space.FULL}, headers=NO_STORE)

    app.add_exception_handler(ProblemError, on_problem)
    app.add_exception_handler(DataError, on_data_error)
    app.add_exception_handler(OSError, on_storage_error)
    app.add_exception_handler(OperationalError, on_storage_error)
    app.add_exception_handler(StarletteHTTPException, on_http)
    app.add_exception_handler(RequestValidationError, on_validation)


# --- cursor pagination (§5.1) ------------------------------------------------------------------------------

class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None


def encode_cursor(key: Sequence[Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(list(key), separators=(",", ":")).encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None, arity: int | None, *, candidates: Sequence[str] = ()) -> list[Any] | None:
    """The key after which the next page starts, or None for the first page. Raises ProblemError(400) when malformed."""
    if cursor is None:
        return None
    try:
        key = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        key = None
    if (not isinstance(key, list) or not key or (arity is not None and len(key) != arity)
            or not all(isinstance(k, str | int | float) and not isinstance(k, bool) for k in key)
            or any(holds_nul(k) for k in key)):  # N-10: a key is compared in a statement
        raise ProblemError(400, "Invalid cursor", "The cursor is malformed; use the next_cursor value of a previous page.")
    from app.api.public_projection import original_id

    try:
        return [original_id(item, candidates=candidates) if isinstance(item, str) else item for item in key]
    except ValueError:
        raise ProblemError(400, "Invalid cursor", "The cursor does not uniquely identify a record in this collection.") from None


def paginate[I](items: Sequence[I], key: Callable[[I], tuple[Any, ...]], cursor: str | None,
                limit: int) -> tuple[list[I], str | None]:
    """Keyset pagination over a small, fully loaded collection already sorted by `key`."""
    candidates = tuple(part for item in items for part in key(item) if isinstance(part, str))
    after = decode_cursor(cursor, len(key(items[0])) if items else None, candidates=candidates)
    if after is not None and any(isinstance(part, str) and part.startswith(('check.', 'component.'))
                                 and part not in candidates for part in after):
        raise ProblemError(400, "Invalid cursor", "The cursor does not belong to this collection.")
    try:
        remaining = [item for item in items if after is None or list(key(item)) > after]
    except TypeError:  # a cursor from another collection (mismatched key types)
        raise ProblemError(400, "Invalid cursor", "The cursor does not belong to this collection.") from None
    page = remaining[:limit]
    return page, encode_cursor(key(page[-1])) if len(remaining) > limit else None


class ProgressEvent(BaseModel):
    seq: int
    ts: str
    type: str
    payload: dict[str, Any]


class Progress(BaseModel):
    analysis_id: str
    status: str
    progress: float | None
    current_stage: str | None
    events: list[ProgressEvent]
    next_after: int


def problem(status: int, title: str, detail: str, instance: str) -> JSONResponse:
    return JSONResponse(status_code=status, media_type="application/problem+json", headers=NO_STORE,
                        content=Problem(title=title, status=status, detail=detail, instance=instance).model_dump())


def _not_found(request: Request) -> JSONResponse:
    return problem(404, "Analysis not found", "No analysis exists with this id.", request.url.path)


MAX_SEQ = 2**63 - 1  # event sequence numbers are BIGINT; a larger cursor is a 422, never a 500 (Q-2 S-3)
_EVENT_ID = re.compile(r"[0-9]{1,18}")


@router.get("/analyses/{aid}/progress", response_model=Progress,
            responses={404: {"model": Problem, "description": "Unknown analysis"}})
def analysis_progress(aid: str, request: Request, after: int = Query(0, ge=0, le=MAX_SEQ), limit: int = Query(200, ge=1, le=500)):
    """Progress events after `after` (exclusive), oldest first. No provisional probability appears before `fuse`."""
    with get_session() as db:
        analysis = db.get(Analysis, aid)
        if analysis is None:
            return _not_found(request)
        batch = list_events(db, aid, after=after, limit=limit)
        body = Progress(analysis_id=aid, status=analysis.status, progress=analysis.progress,
                        current_stage=analysis.current_stage, events=[ProgressEvent(**event_json(e)) for e in batch],
                        next_after=batch[-1].seq if batch else after)
    from app.api.public_projection import project

    return JSONResponse(content=project(body.model_dump(mode="json")), headers={**NO_STORE, 'X-Chipmunk-View': 'customer-projection'})


@router.get("/analyses/{aid}/events", response_class=StreamingResponse,
            responses={200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events"},
                       400: {"model": Problem, "description": "Invalid Last-Event-ID"},
                       404: {"model": Problem, "description": "Unknown analysis"},
                       429: {"model": Problem, "description": "This principal has the maximum number of open streams"},
                       503: {"model": Problem, "description": "The server has the maximum number of open streams"}})
def analysis_events(aid: str, request: Request, after: int | None = Query(None, ge=0, le=MAX_SEQ),
                    last_event_id: str | None = Header(None, alias="Last-Event-ID")):
    """Server-Sent Events: `stage_started`, `stage_finished`, `detector_finished`, `progress`, then `done|failed|cancelled`.
    Open streams are capped per principal (`sse_max_per_principal`, 429) and per process (`sse_max_connections`, 503)."""
    cursor = after or 0
    if last_event_id is not None:
        if not _EVENT_ID.fullmatch(last_event_id.strip()):
            return problem(400, "Invalid Last-Event-ID", "Last-Event-ID must be a non-negative event sequence number.",
                           request.url.path)
        cursor = int(last_event_id.strip())
    with get_session() as db:
        if db.get(Analysis, aid) is None:
            return _not_found(request)
    lease = STREAMS.acquire(request.state.principal.id)
    if isinstance(lease, int):
        detail = ("This principal already has the maximum number of open event streams; close one, or poll /progress." if lease == 429
                  else "The server has the maximum number of open event streams; retry shortly, or poll /progress.")
        refused = problem(lease, "Too many open streams", detail, request.url.path)
        refused.headers["Retry-After"] = "5"
        return refused
    return StreamingResponse(_stream(aid, cursor, request, lease), media_type="text/event-stream",
                             headers={**NO_STORE, "X-Accel-Buffering": "no", 'X-Chipmunk-View': 'customer-projection'},
                             background=BackgroundTask(lease.release))


class StreamLease:
    """One open event stream, released once by whichever of the stream's `finally` and the response's background task runs first."""

    def __init__(self, registry: StreamRegistry, principal_id: str):
        self.registry, self.principal_id, self.released = registry, principal_id, False

    def release(self) -> None:
        self.registry.release(self)


class StreamRegistry:
    """Open event streams per principal and in this process (Q-2 S-1)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: dict[str, int] = {}

    def open(self, principal_id: str | None = None) -> int:
        with self._lock:
            return sum(self._open.values()) if principal_id is None else self._open.get(principal_id, 0)

    def acquire(self, principal_id: str) -> StreamLease | int:
        """A lease, or the refusal status: 503 at `sse_max_connections` in the process, 429 at `sse_max_per_principal`."""
        with self._lock:
            if sum(self._open.values()) >= settings.sse_max_connections:
                return 503
            if self._open.get(principal_id, 0) >= settings.sse_max_per_principal:
                return 429
            self._open[principal_id] = self._open.get(principal_id, 0) + 1
        return StreamLease(self, principal_id)

    def release(self, lease: StreamLease) -> None:
        with self._lock:
            if lease.released:
                return
            lease.released = True
            remaining = self._open.get(lease.principal_id, 0) - 1
            if remaining > 0:
                self._open[lease.principal_id] = remaining
            else:
                self._open.pop(lease.principal_id, None)


STREAMS = StreamRegistry()


def _frame(event: str, data: dict[str, Any], seq: int | None = None) -> str:
    from app.api.public_projection import project

    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(project(data), separators=(',', ':'))}\n\n"


def _poll(aid: str, cursor: int) -> tuple[list[tuple[int, str, dict[str, Any]]], str | None]:
    with get_session() as db:
        batch = [(e.seq, e.type, event_json(e)) for e in list_events(db, aid, after=cursor, limit=200)]
        analysis = db.get(Analysis, aid)
        return batch, (analysis.status if analysis is not None else None)


def still_allowed(request: Request, aid: str) -> bool:
    """Re-checked at each heartbeat: the request's credentials still resolve (session not revoked, principal enabled, token not
    expired) and the principal can still read this analysis (Q-2 S-2)."""
    from app.api import auth, security

    try:
        principal = auth.resolve(request, security.token_resources)
    except StarletteHTTPException:
        return False
    return principal.satisfies("administrator") or security._case_denial(principal, {"aid": aid}, "viewer") is None


async def _stream(aid: str, cursor: int, request: Request, lease: StreamLease) -> AsyncIterator[str]:
    """An async generator: an idle stream waits on the event loop, not on a worker thread, and each database poll borrows the
    thread pool only while it runs (Q-2 S-1)."""
    try:
        started = last_beat = time.monotonic()
        yield "retry: 2000\n\n"
        while True:
            batch, status = await run_in_threadpool(_poll, aid, cursor)
            for seq, type_, data in batch:
                cursor = seq
                yield _frame(type_, data, seq)
                if type_ in TERMINAL_EVENTS:
                    return
            if status is None:
                yield _frame("failed", {"type": "failed", "payload": {"reason": "The analysis was deleted"}})
                return
            if not batch and status in TERMINAL_STATUSES:  # e.g. a v1-pipeline analysis that recorded no events
                yield _frame(status, {"type": status, "payload": {"reason": "No progress events were recorded"}})
                return
            now = time.monotonic()
            if now - started >= settings.sse_max_s:
                return
            if STOPPING.is_set():  # N-14 (OPS-15): the process is stopping; the client reconnects with Last-Event-ID
                return
            if now - last_beat >= settings.sse_heartbeat_s:
                if not await run_in_threadpool(still_allowed, request, aid):
                    return
                yield ": heartbeat\n\n"
                last_beat = now
            await asyncio.sleep(settings.sse_poll_s)
    finally:
        lease.release()
