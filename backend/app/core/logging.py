"""Structured logging (structlog). Every analysis logs analysis_id, stage, duration.

`configure_logging()` is called by each process that serves or works (BUILD_SPEC N-13; ops review OPS-11): the API when
`app.main` is imported, the worker (`python -m app.engine.worker`) and the command line (`python -m app.cli`).

- **Prod:** every line is one JSON object with `event`, `level` and a UTC `timestamp`, at `VERITAS_LOG_LEVEL`. That holds
  for the service's own events and for the lines other libraries write through the standard `logging` module (uvicorn's
  start and stop lines, the migrations, warnings), which are rendered by the same processors; an exception that ends a
  process is logged as one such line as well.
- **Dev:** the console renderer, as before.
- **Where:** services log to standard output. The command line logs to standard error, so that what a command prints
  (a table, a `--json` document) is its answer and nothing else.
- **Requests** (`RequestLogMiddleware`): each gets an `X-Request-ID` (a proxy's plain token is kept, anything else is
  replaced), bound to every log line of the request, and one line `request` with method, route template, status and
  duration. The query string is never logged: resource tokens travel there (`?token=`). The line is `error` for a
  5xx, except a 503 of a probe path, which is a `warning`: a probe that says "not ready" while a process starts is the
  probe working, and `ready.not_ready` beside it says what is not ready.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
import uuid
from typing import Any, TextIO

import structlog

from .config import settings

PROBE_PATHS = frozenset({"/api/health", "/api/ready", "/api/metrics"})  # polled every few seconds: logged at debug
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
# Libraries whose lines below a warning say nothing to an operator. `weasyprint.progress` writes a line per layout step
# and page of every report, and `alembic.runtime.plugins` seven "setup plugin" lines at every process start (BUILD_SPEC
# N-14c, E-12); the renderer's warnings (`weasyprint`) and the migrations' revision lines (`alembic.runtime.migration`)
# are other loggers and stay.
_NOISY = ("urllib3", "httpx", "PIL", "transformers", "huggingface_hub", "weasyprint.progress", "alembic.runtime.plugins")
_UVICORN = ("uvicorn", "uvicorn.error")


def _processors() -> list[Any]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def configure_logging(stream: TextIO | None = None) -> None:
    """Set up structlog and, in prod, the standard `logging` module on one stream (default: standard output)."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    prod = settings.environment == "prod"
    renderer = structlog.processors.JSONRenderer() if prod else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[*_processors(), renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=True,
    )
    if prod:
        _standard_logging_as_json(stream or sys.stdout, level)
    else:
        logging.basicConfig(format="%(message)s", stream=stream or sys.stdout, level=level)
    quieten()


def quieten() -> None:
    """The noisy libraries log from a warning up."""
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def _standard_logging_as_json(stream: TextIO, level: int) -> None:
    """Lines written through `logging` (uvicorn, alembic, warnings) leave as the same JSON objects as the service's own."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[structlog.contextvars.merge_contextvars, structlog.stdlib.add_log_level, structlog.stdlib.add_logger_name,
                           structlog.processors.TimeStamper(fmt="iso", utc=True), structlog.processors.StackInfoRenderer(),
                           structlog.processors.format_exc_info],
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, structlog.processors.JSONRenderer()],
    ))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in _UVICORN:  # uvicorn sets its own handlers before it imports the application: its lines go the same way
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    # uvicorn's access log prints the query string, where resource tokens travel: it stays off whatever the command line
    # says (uvicorn asks `hasHandlers()`), and the `request` line of the middleware stands in its place
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    logging.captureWarnings(True)
    log = structlog.get_logger("app.process")

    def uncaught(kind: type[BaseException], value: BaseException, tb: Any) -> None:
        if issubclass(kind, KeyboardInterrupt):
            return
        log.error("process.uncaught_exception", exc_info=(kind, value, tb))

    def uncaught_in_thread(args: Any) -> None:
        if args.exc_type is not SystemExit:
            log.error("process.uncaught_exception", thread=getattr(args.thread, "name", None),
                      exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = uncaught
    threading.excepthook = uncaught_in_thread


def get_logger(name: str):
    return structlog.get_logger(name)


def request_id(given: str | None) -> str:
    """The id of a request: the one a proxy sent when it is a plain token (letters, digits, `.`, `_`, `-`; 64 characters at
    most), else a new one. Anything else could carry line breaks or markup into the logs and the response."""
    return given if given is not None and _REQUEST_ID.fullmatch(given) else uuid.uuid4().hex


class RequestLogMiddleware:
    """`X-Request-ID` on every response, bound to the request's log lines, and one `request` line per request (module
    docstring). Plain ASGI, outermost: it sees the refusals of every inner layer, and a stream is not buffered."""

    def __init__(self, app: Any):
        self.app = app
        self.log = structlog.get_logger("app.request")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        rid = request_id(headers.get("x-request-id"))
        started = time.perf_counter()
        status: list[int] = []

        async def send_with_id(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status.append(int(message["status"]))
                message = {**message, "headers": [(k, v) for k, v in message.get("headers") or [] if k.lower() != b"x-request-id"]
                           + [(b"x-request-id", rid.encode("ascii"))]}
            await send(message)

        structlog.contextvars.bind_contextvars(request_id=rid)
        try:
            await self.app(scope, receive, send_with_id)
        except BaseException:
            status[:] = status or [500]  # nothing was sent: the server answers 500
            raise
        finally:
            route = getattr(scope.get("route"), "path", None)  # the template ("/api/v2/cases/{case_id}"), never the query string
            code = status[0] if status else 500
            fields = {"request_id": rid, "method": scope.get("method"), "route": route or "unmatched", "status": code,
                      "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
            if code == 503 and route in PROBE_PATHS:
                self.log.warning("request", **fields)
            elif code >= 500:
                self.log.error("request", **fields)
            elif route in PROBE_PATHS:
                self.log.debug("request", **fields)
            else:
                self.log.info("request", **fields)
            structlog.contextvars.unbind_contextvars("request_id")
