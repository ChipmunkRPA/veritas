"""The two probes (BUILD_SPEC N-13; ops review OPS-07, OPS-18; security review SEC-15).

- **`/api/health` is liveness:** it answers `{"status": "ok"}` from the process alone, on the event loop. It asks neither
  the database nor a thread of the pool, so a locked database or a full volume does not get a living API restarted.
- **`/api/ready` is readiness:** 200 only when the instance can take work and do it.
  - `database`: on SQLite the write lock is taken and given back on a connection of its own within `DATABASE_WAIT_S`
    (a read does not see another connection's write lock in WAL, and an instance that cannot write can store nothing);
    on another engine `SELECT 1`.
  - `storage`: one small file is written in the data directory, and the volume has at least `VERITAS_MIN_FREE_BYTES` free.
  - `worker`: when this process runs no worker (`VERITAS_RUN_WORKER=false`), a worker's heartbeat row is at most
    `VERITAS_WORKER_STALE_S` old and says that every thread of it reached the database lately (`app.engine.heartbeat`).
  Each check runs on a thread of the probe's own and is given `CHECK_TIMEOUT_S`; one that does not answer has failed.
  One round of checks runs at a time, and callers that arrive meanwhile share its answer.
- **What a caller is told.** Without credentials: the `status` field and nothing else. With credentials: the checks, and
  version, build commit, device, profile, fusion state, models in memory and queue depth, and `capacity`: the detectors
  of the worker's default profile that cannot run under its model memory budget (N-14a). With a separate worker these are
  the worker's, read from its row, not those of the API process, which may sit on another device.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Request
from sqlalchemy import text

from app.api import jobs, state
from app.core.config import settings
from app.core.logging import get_logger
from app.engine import heartbeat
from app.storage import db, space

log = get_logger(__name__)

DATABASE_WAIT_S = 2.0  # how long the SQLite check waits for the write lock
CHECK_TIMEOUT_S = 4.0  # a check that has not answered by then has failed
LOG_EVERY_S = 60.0  # a state that lasts is logged again this often
IDENTIFY_TIMEOUT_S = 3.0  # the caller's credentials are read within this, or the caller is told the status only
_checks = ThreadPoolExecutor(max_workers=6, thread_name_prefix="veritas-ready")


def check_database() -> None:
    """Raise when the database cannot be written (SQLite) or reached (another engine)."""
    engine = db.get_engine()
    path = engine.url.database if engine.dialect.name == "sqlite" else None
    if path and path != ":memory:":
        if not Path(path).is_file():
            raise RuntimeError("the database file is missing")
        conn = sqlite3.connect(path, timeout=DATABASE_WAIT_S, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
        finally:
            conn.close()
        return
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def check_storage() -> dict[str, Any]:
    """Raise when the data directory cannot be written or its volume is below the floor; else the free bytes and the floor."""
    space.writable()
    space.require()
    return {"free_bytes": space.free_bytes(), "min_free_bytes": settings.min_free_bytes}


def check_worker() -> dict[str, Any]:
    """The worker the instance is served by. In a process that runs its worker threads: this process (`in_process`). Else
    the live worker that wrote its heartbeat last; raises when there is none."""
    if settings.run_worker:
        return {"in_process": True}
    views = heartbeat.workers(db.get_engine())
    described = heartbeat.newest(views)
    if described is None:
        raise RuntimeError("no worker has written a heartbeat")
    if not described.live():
        age = described.age_s()
        if age > settings.worker_stale_s:
            raise RuntimeError(f"the last worker heartbeat is {age:.0f} s old")
        raise RuntimeError("the worker's threads do not reach the database")
    return {"in_process": False, "worker_id": described.worker_id, "heartbeat_age_s": round(described.age_s(), 1),
            "threads": described.threads, "workers": sum(v.live() for v in views), "on_hold": described.state.get("on_hold"),
            "_view": described}


async def _run(check: Any) -> dict[str, Any]:
    """One check as `{"ok": …}`, with what it returned or why it failed. The reason is for an authenticated caller and the log."""
    try:
        found = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(_checks, check), CHECK_TIMEOUT_S)
        return {"ok": True, **(found or {})}
    except TimeoutError:
        return {"ok": False, "detail": f"no answer within {CHECK_TIMEOUT_S:g} s"}
    except space.StorageLow as low:
        return {"ok": False, "detail": "the data volume is below its free-space floor", "free_bytes": low.free, "min_free_bytes": low.floor}
    except Exception as e:  # noqa: BLE001  a probe answers
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"[:200]}


async def _round() -> dict[str, dict[str, Any]]:
    database, storage, worker = await asyncio.gather(_run(check_database), _run(check_storage), _run(check_worker))
    return {"database": database, "storage": storage, "worker": worker}


_running: asyncio.Task[dict[str, dict[str, Any]]] | None = None


async def checks() -> dict[str, dict[str, Any]]:
    """The three checks. One round runs at a time: a request that arrives while a round runs is given that round's answer,
    so the probe, which needs no credentials, takes the write lock and writes its file once however many callers ask."""
    global _running
    loop = asyncio.get_running_loop()
    if _running is None or _running.done() or _running.get_loop() is not loop:
        _running = loop.create_task(_round())
    return await asyncio.shield(_running)


def status_of(found: dict[str, dict[str, Any]]) -> str:
    # `state.ready`: the API's start is over. There is no warm-up, so no third answer (N-14; OPS-17)
    return "ready" if state.ready.is_set() and all(check["ok"] for check in found.values()) else "not_ready"


async def identified(request: Request) -> bool:
    """The request carries credentials this instance accepts. Asked as every route asks (`security.authenticate`), so a
    refused key counts against its client as anywhere else and the probe is no place to guess keys; any refusal, error or
    delay means the caller is told the status only."""
    from app.api import security

    try:
        await asyncio.wait_for(security.authenticate(request), IDENTIFY_TIMEOUT_S)
        return True
    except Exception:  # noqa: BLE001
        return False


def _engine_detail(worker: dict[str, Any], profile: str | None) -> dict[str, Any]:
    """Device, profile, fusion state and models of the worker that serves: this process, or the row of a separate worker.
    With `profile`, what serves under that profile. Empty when no worker is known."""
    view = worker.get("_view")
    if settings.run_worker:
        own = heartbeat.engine_state()
        if profile is not None:
            own = {**own, **heartbeat.serving(profile)}
        engine = own
    elif view is not None:
        engine = {"device": view.device, "profile": view.profile, **view.state}
        if profile is not None:
            engine = {**engine, **(engine.get("profiles") or {}).get(profile, {"fusion": None, "fusers": None, "text_classifier": None})}
    else:
        return {}
    keep = ("version", "build_commit", "device", "profile", "fusion", "fusers", "text_classifier", "loaded", "capacity")
    return {key: engine.get(key) for key in keep}


def _queue_depth() -> int | None:
    try:
        return jobs.queue_depth()
    except Exception:  # noqa: BLE001  the database check says why
        return None


_logged: tuple[tuple[str, ...], float] = ((), 0.0)


def _log(status: str, found: dict[str, dict[str, Any]]) -> None:
    """One warning when the failing checks change and once a minute while they stay, one line when the instance is ready
    again: a probe every few seconds does not fill the log."""
    global _logged
    failing = tuple(sorted(name for name, check in found.items() if not check["ok"]))
    before, at = _logged
    now = time.monotonic()
    if failing and (failing != before or now - at >= LOG_EVERY_S):
        log.warning("ready.not_ready", **{name: found[name].get("detail") for name in failing})
        _logged = (failing, now)
    elif not failing and before:
        log.info("ready.ready_again", status=status)
        _logged = ((), now)


async def ready_body(request: Request, profile: str | None) -> tuple[int, dict[str, Any]]:
    """The status code and the body of `/api/ready` for this caller."""
    found = await checks()
    status = status_of(found)
    code = 200 if status == "ready" else 503
    _log(status, found)
    if not await identified(request):
        return code, {"status": status}
    loop = asyncio.get_running_loop()
    try:
        engine = await asyncio.wait_for(loop.run_in_executor(_checks, _engine_detail, found["worker"], profile), CHECK_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        log.warning("ready.engine_detail_unavailable", error=f"{type(e).__name__}: {e}"[:200])
        engine = {}
    try:
        depth = await asyncio.wait_for(loop.run_in_executor(_checks, _queue_depth), CHECK_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        depth = None
    shown = {name: {k: v for k, v in check.items() if not k.startswith("_")} for name, check in found.items()}
    return code, {"status": status, "checks": shown, "version": settings.version, **engine, "queue_depth": depth}
