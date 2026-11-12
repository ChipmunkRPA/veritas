"""Veritas API entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import auth, jobs, state, v2_image_reference
from app.api.demo import router as demo_router
from app.api.routes import KIND_LIMITS, public, router
from app.api.security import refuse_nul
from app.api.v2 import install_problem_handlers, problem
from app.api.v2 import router as v2_router
from app.api.v2_annotations import router as v2_annotations_router
from app.api.v2_audit import router as v2_audit_router
from app.api.v2_auth import public_router as v2_auth_public_router
from app.api.v2_auth import router as v2_auth_router
from app.api.v2_batches import router as v2_batches_router
from app.api.v2_billing import public_router as v2_billing_public_router
from app.api.v2_billing import router as v2_billing_router
from app.api.v2_cancel import router as v2_cancel_router
from app.api.v2_cases import router as v2_cases_router
from app.api.v2_evidence import router as v2_evidence_router
from app.api.v2_evidence_packets import router as v2_evidence_packets_router
from app.api.v2_models import router as v2_models_router
from app.api.v2_policy import router as v2_policy_router
from app.api.v2_refusion import router as v2_refusion_router
from app.api.v2_reports import router as v2_reports_router
from app.api.v2_rerun import router as v2_rerun_router
from app.api.v2_review import router as v2_review_router
from app.api.v2_webhooks import router as v2_webhooks_router
from app.core import lifecycle
from app.core.config import settings
from app.core.frontend import mount_frontend
from app.core.logging import RequestLogMiddleware, configure_logging, get_logger
from app.engine.registry import get_registry
from app.storage import db, keyguard, refusals, running
from app.storage.db import init_db

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """The process is marked as running on its data directory from before the database is opened until after it is closed
    (N-12a): `backup restore` refuses such a directory, and a start is refused while a restore writes into it."""
    settings.ensure_dirs()
    with refusals.logged(log):  # N-19e (E-17): a refused start is one `startup.refused` with the sentence
        held = running.hold()
    try:
        async with _serving(app):
            yield
    finally:
        running.release(held)


@asynccontextmanager
async def _serving(app: FastAPI):
    with refusals.logged(log):
        init_db()
        keyguard.check_start()  # N-12: a prod start generates no key over records made with the earlier one
    registry = get_registry()  # startup check: manifest lint, licence policy, card completeness (VG-01, D-04)
    log.info("registry.loaded", licence_policy=registry.policy.label, models=len(registry.manifest.models),
             detectors=len(registry.manifest.detectors), enabled_v2=sum(e.enabled for e in registry.decisions()))
    jobs.start()
    log.info("startup", version=settings.version, env=settings.environment, data_dir=str(settings.data_dir))
    from app.engine.fusion.registry import warn_fallbacks, warn_shadowed

    warn_shadowed()  # D-57: a data-directory artefact that the packaged one of its version is served in place of is named
    if settings.environment == "prod":  # OPS-03: an uncalibrated start is said once per kind, with the reason
        from app.engine.pipelines import default_profile

        warn_fallbacks(registry, default_profile())
    # The start is over. There is no warm-up (N-14; OPS-17): the models of an analysis load when it runs and are evicted
    # under the memory budget (D-21); the setting and the route that loaded the v0.3 model set are gone.
    state.ready.set()
    restore_signals = lifecycle.watch_signals()  # N-14 (OPS-15): open event streams end when the process is told to stop
    try:
        yield
    finally:
        restore_signals()
    jobs.stop()
    db.shutdown()  # N-12a: a stopped SQLite instance is one file


_docs = settings.environment != "prod"
# N-10 (PG-05): `refuse_nul` on every route; the authenticated routers ask it again before the credentials (`CheckedRoute`)
app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan, docs_url="/api/docs" if _docs else None, openapi_url="/api/openapi.json" if _docs else None, redoc_url=None,
              dependencies=[Depends(refuse_nul)])
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


def body_limit(path: str) -> int:
    """The request body cap of a path: upload routes get their kind's limit plus multipart overhead, everything else
    `max_json_bytes` (Q-2 L-1, L-2)."""
    if path.startswith("/api/analyze/"):
        return KIND_LIMITS.get(path.rsplit("/", 1)[-1], settings.max_video_bytes) + 1024 * 1024
    if (path.startswith("/api/v2/cases/") and path.endswith("/items")) or path == "/api/v2/billing/checkout":
        return max(KIND_LIMITS.values()) + 1024 * 1024  # the declared kind is a form field; intake enforces its own limit
    if path == "/api/v2/batches":
        return settings.batch_max_bytes + 16 * 1024 * 1024
    if v2_image_reference.is_comparison_path(path) and v2_image_reference.enabled():
        return v2_image_reference.upload_limit() + 1024 * 1024
    return settings.max_json_bytes


def upload_limiter(scope):
    """The per-kind caps of an upload's file parts, counted while they stream (D-48; `app.api.upload_limits`); None off the
    upload routes."""
    from app.api import upload_limits
    from app.batches.service import detect_kind

    path, declared, field_caps = scope["path"], None, {}
    if path.startswith("/api/analyze/"):
        declared = path.rsplit("/", 1)[-1]
    elif path == "/api/v2/batches":
        field_caps = {"archive": settings.batch_max_bytes}
    elif v2_image_reference.is_comparison_path(path) and v2_image_reference.enabled():
        declared, field_caps = "image", {"reference": v2_image_reference.upload_limit()}
    elif not ((path.startswith("/api/v2/cases/") and path.endswith("/items")) or path == "/api/v2/billing/checkout"):
        return None
    content_type = next((v for k, v in scope["headers"] if k == b"content-type"), b"")
    return upload_limits.for_body(content_type, KIND_LIMITS, declared_kind=declared, field_caps=field_caps, detect=detect_kind)


def _too_large(limit: int) -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": f"Request body exceeds the {max(limit // 1048576, 1)} MB limit"})


class BodyLimitMiddleware:
    """Every request body is bounded while it arrives, for every method and with or without Content-Length: a declared length
    above the path's cap is refused before any read, and a chunked body is refused once the received bytes pass it (raised as
    HTTP 413 from `receive`, so form parsing stops before the spooled file grows further). On the upload routes each file part is
    also bounded by its kind's cap while it streams (D-48): the chunk that passes it never reaches the form parser."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = body_limit(scope["path"])
        declared = next((v for k, v in scope["headers"] if k == b"content-length"), None)
        if declared is not None and declared.isdigit() and int(declared) > limit:
            if v2_image_reference.is_comparison_path(scope["path"]):
                response = problem(413, "Request body exceeds the limit", "The image upload is larger than allowed.", scope["path"])
                return await response(scope, receive, send)
            return await _too_large(limit)(scope, receive, send)
        limiter = upload_limiter(scope)
        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                received += len(chunk)
                if limiter is not None:  # first: when one message passes both caps, the part's own cap names the kind
                    from app.api.upload_limits import PartTooLarge

                    try:
                        limiter.write(chunk)
                    except PartTooLarge as e:
                        raise StarletteHTTPException(413, str(e)) from None
                if received > limit:
                    raise StarletteHTTPException(413, f"Request body exceeds the {max(limit // 1048576, 1)} MB limit")
            return message

        return await self.app(scope, counting_receive, send)


app.add_middleware(BodyLimitMiddleware)


V1_EXEMPT = ("/api/v2/", "/api/health", "/api/ready", "/api/docs", "/api/openapi.json")


@app.middleware("http")
async def v1_deprecation_headers(request, call_next):
    """v1 routes (`routes.py`, `demo.py`) stay through v2.1 and say so on every response, refusals included (ARCHITECTURE §5.1;
    PRD API-03): `Deprecation: true`, `Sunset` (RFC 8594) and the successor version."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/") and not path.startswith(V1_EXEMPT):
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = settings.v1_sunset
        response.headers["Link"] = '</api/v2>; rel="successor-version"'
    return response


SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
DOCS_PATHS = ("/api/docs", "/api/openapi.json")  # Swagger UI loads its assets from a CDN (dev only; off in prod)


@app.middleware("http")
async def security_headers(request, call_next):
    """PL-04 secure headers on every response, refusals and the served SPA included (ARCHITECTURE §5.1). A route that sets its own
    policy keeps it (the report preview's stricter CSP)."""
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        if name == "Content-Security-Policy" and request.url.path.startswith(DOCS_PATHS):
            continue
        response.headers.setdefault(name, value)
    return response


@app.middleware("http")
async def stale_session_cookies(request, call_next):
    """A request whose session cookie names no open session has both cookies deleted by its response, whatever the status
    (the 401, or the answer a resource token or single-user mode gave): the session cookie is HttpOnly, so only the server
    can remove it (N-9b; code review 15)."""
    response = await call_next(request)
    if auth.session_refused(request) is not None:
        auth.clear_session_cookies(response)
    return response


# N-13 (OPS-11): added last, so outermost: every response carries X-Request-ID, the refusals of the layers above included,
# and each request leaves one log line with its route template, status and duration (never its query string)
app.add_middleware(RequestLogMiddleware)


app.include_router(public)
app.include_router(router)
app.include_router(demo_router)
app.include_router(v2_router)
app.include_router(v2_auth_public_router)
app.include_router(v2_auth_router)
app.include_router(v2_audit_router)
app.include_router(v2_cases_router)
app.include_router(v2_cancel_router)
app.include_router(v2_annotations_router)
app.include_router(v2_review_router)
app.include_router(v2_rerun_router)
app.include_router(v2_evidence_router)
app.include_router(v2_evidence_packets_router)
app.include_router(v2_image_reference.router)
app.include_router(v2_models_router)
app.include_router(v2_refusion_router)
app.include_router(v2_reports_router)
app.include_router(v2_batches_router)
app.include_router(v2_billing_public_router)
app.include_router(v2_billing_router)
app.include_router(v2_webhooks_router)
app.include_router(v2_policy_router)
install_problem_handlers(app)

# Serve the built frontend if present (single-container deployment)
_dist = settings.frontend_dir or Path(__file__).resolve().parents[2] / "frontend" / "dist"
if settings.frontend_dir is not None or _dist.is_dir():
    mount_frontend(app, _dist)
