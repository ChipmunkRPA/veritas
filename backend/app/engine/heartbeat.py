"""The worker's heartbeat (BUILD_SPEC N-13; ops review OPS-06, OPS-18; migration 0022).

A worker writes one row of `worker_heartbeats` at its start and then every `worker_heartbeat_s`: that it lives, whether
every one of its threads reached the database lately (`Worker.health_snapshot`), and what it runs on (`engine_state`:
device, default profile, fusion state per kind, models in memory). The API reads the rows, so its metrics and its readiness
describe the worker that serves the queue and not the API process, which in a split deployment may sit on another device.

- **Who writes.** `python -m app.engine.worker` and the worker threads of an API process (`VERITAS_RUN_WORKER=true`).
  `worker --once` writes none: it serves no queue.
- **A stop** removes the row. A worker that was killed leaves its row; it grows old (`worker_stale_s`) and a later start
  removes it (`PRUNE_AFTER_S`).
- **A write that fails** (a locked or unreachable database, a Postgres role without the grants of migration 0022) is logged
  and tried again at the next beat. The heartbeat never stops a worker: the row growing old is the signal.
- **Clocks.** `beat_at` is the worker's clock in UTC, as the leases of the queue are; its age is taken on the reader's.
"""
from __future__ import annotations

import os
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Connection, Engine, delete, insert, select, update

from app.core import times
from app.core.config import settings
from app.core.logging import get_logger
from app.storage.db import WorkerHeartbeatRow, utcnow

log = get_logger(__name__)

PRUNE_AFTER_S = 3600.0  # the row of a worker that was killed is removed by a start this long after its last beat
_ROWS = WorkerHeartbeatRow.__table__


def loaded_models() -> list[str]:
    """The models this process holds in memory. Reads the model manager when it exists and never builds it."""
    from app.engine import models

    manager = models._manager
    if manager is None:
        return []
    return sorted({info.id for info in manager.snapshot()} | set(manager.v1_loaded()))


def serving(profile: Any = None) -> dict[str, Any]:
    """What serves under a profile (the default one when none is named), per kind: `fusion` (`artefact` or `fallback`),
    `fusers` (the artefact's version, the start of its sha256 and its review state) and the primary `text_classifier`."""
    from app.engine.fusion.registry import fuser_status
    from app.engine.pipelines import default_profile
    from app.engine.registry import get_registry, text_classifier_status

    detectors, profile = get_registry(), profile or default_profile()
    status = fuser_status(detectors, profile)
    return {"fusion": {kind: row["mode"] for kind, row in status.items()},
            "fusers": {kind: {"version": row["version"], "sha256": row["sha256"][:12] if row["sha256"] else None, "review": row["review"]}
                       for kind, row in status.items()},
            "text_classifier": text_classifier_status(detectors, profile)}


def engine_state() -> dict[str, Any]:
    """What this process runs analyses on: device, default profile, what serves under it (`serving`) and under each other
    profile (`profiles`, for `/api/ready?profile=`), the models in memory, version and build commit, and `capacity`: which
    detectors of the default profile cannot run under the model memory budget (N-14a; `capacity.summary`, None until the
    plan of the start is made)."""
    from app.core.device import DEVICE
    from app.engine import capacity
    from app.engine.build_info import git_commit
    from app.engine.detector import Profile
    from app.engine.pipelines import default_profile

    profile = default_profile()
    return {"device": str(DEVICE), "profile": str(profile.value), **serving(profile),
            "profiles": {str(p.value): serving(p) for p in Profile},
            "loaded": loaded_models(), "version": settings.version, "build_commit": git_commit(),
            "capacity": capacity.process_summary()}


@dataclass(frozen=True)
class WorkerView:
    worker_id: str
    hostname: str
    pid: int
    in_process: bool
    started_at: datetime
    beat_at: datetime
    device: str
    profile: str
    threads: int
    ready: bool
    state: dict[str, Any]

    def age_s(self, now: datetime | None = None) -> float:
        return max(0.0, ((now or utcnow()) - self.beat_at).total_seconds())

    def live(self, now: datetime | None = None, stale_s: float | None = None) -> bool:
        """It wrote lately and every one of its threads reached the database lately."""
        return self.ready and self.age_s(now) <= (settings.worker_stale_s if stale_s is None else stale_s)


def _view(row: Any) -> WorkerView:
    m = row._mapping
    return WorkerView(worker_id=m["worker_id"], hostname=m["hostname"], pid=m["pid"], in_process=bool(m["in_process"]),
                      started_at=times.to_utc(m["started_at"]), beat_at=times.to_utc(m["beat_at"]), device=m["device"],
                      profile=m["profile"], threads=m["threads"], ready=bool(m["ready"]), state=dict(m["state"] or {}))


def workers_in(conn: Connection) -> list[WorkerView]:
    """Every heartbeat row, the newest beat first."""
    return [_view(row) for row in conn.execute(select(_ROWS).order_by(_ROWS.c.beat_at.desc(), _ROWS.c.worker_id))]


def workers(engine: Engine) -> list[WorkerView]:
    with engine.connect() as conn:
        return workers_in(conn)


def newest(views: list[WorkerView], now: datetime | None = None) -> WorkerView | None:
    """The worker the probes describe: the live one that wrote last, else the one that wrote last."""
    return next((v for v in views if v.live(now)), views[0] if views else None)


class Publisher:
    """Writes the row of one worker. `start` writes the first beat in the calling thread, so the row exists when the worker
    takes its first job; a daemon thread writes the later ones."""

    def __init__(self, worker: Any, engine: Engine, *, in_process: bool, interval_s: float | None = None,
                 state: Any = engine_state, clock: Any = utcnow):
        self.worker = worker
        self.worker_id = str(worker.worker_id)[:100]
        self.engine = engine  # of the start: a beat after a stop never reaches another instance's database
        self.in_process = in_process
        self.interval_s = settings.worker_heartbeat_s if interval_s is None else interval_s
        self.state = state
        self.clock = clock
        self.started_at: datetime | None = None
        self._stop = threading.Event()
        self._writing = threading.Lock()  # a beat and the removal of the row never cross: no row is written after a stop
        self._thread: threading.Thread | None = None
        self._failing = False

    def _values(self) -> dict[str, Any]:
        state = dict(self.state())
        snapshot = self.worker.health_snapshot(max_age_s=settings.worker_stale_s)
        # a worker that has not asked the queue yet (the first beat) is not ready; its next beat says so
        return {"beat_at": self.clock(), "device": str(state.pop("device", "unknown"))[:32],
                "profile": str(state.pop("profile", "unknown"))[:16], "threads": int(snapshot["configured_threads"]),
                "ready": bool(snapshot["ready"]),
                "state": {**state, "alive_threads": int(snapshot["alive_threads"]), "stopping": bool(snapshot["stopping"]),
                          "on_hold": snapshot.get("on_hold")}}

    def beat(self) -> bool:
        """Write the row now. False when the write failed: logged once per run of failures, and once when it works again."""
        try:
            values = self._values()
            with self._writing:
                if self._stop.is_set():
                    return False
                with self.engine.begin() as conn:
                    key = _ROWS.c.worker_id == self.worker_id
                    if conn.execute(update(_ROWS).where(key).values(**values)).rowcount == 0:
                        conn.execute(insert(_ROWS).values(worker_id=self.worker_id, hostname=socket.gethostname()[:255],
                                                          pid=os.getpid(), in_process=self.in_process,
                                                          started_at=self.started_at or values["beat_at"], **values))
        except Exception as e:  # noqa: BLE001  the heartbeat never stops a worker
            if not self._failing:
                log.error("worker.heartbeat_not_written", worker_id=self.worker_id, error=f"{type(e).__name__}: {e}"[:300])
            self._failing = True
            return False
        if self._failing:
            log.info("worker.heartbeat_written_again", worker_id=self.worker_id)
        self._failing = False
        return True

    def _prune(self) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(delete(_ROWS).where(_ROWS.c.beat_at < self.clock() - timedelta(seconds=PRUNE_AFTER_S)))
        except Exception as e:  # noqa: BLE001
            log.warning("worker.heartbeat_prune_failed", error=f"{type(e).__name__}: {e}"[:300])

    def _run(self) -> None:
        # the second beat comes soon: by then the worker threads have asked the queue once and the row says `ready`
        wait = min(self.interval_s, 1.0)
        while not self._stop.wait(wait):
            self.beat()
            wait = self.interval_s

    def start(self) -> None:
        self._stop.clear()
        self.started_at = self.clock()
        self._prune()
        self.beat()
        self._thread = threading.Thread(target=self._run, name="veritas-worker-heartbeat", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """End the beats and remove the row: a worker that stopped is no worker."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if not self._writing.acquire(timeout=timeout):  # a beat still waits for the database: its row grows old instead
            log.warning("worker.heartbeat_not_removed", worker_id=self.worker_id, error="a beat is still being written")
            return
        try:
            with self.engine.begin() as conn:
                conn.execute(delete(_ROWS).where(_ROWS.c.worker_id == self.worker_id))
        except Exception as e:  # noqa: BLE001  the row grows old instead
            log.warning("worker.heartbeat_not_removed", worker_id=self.worker_id, error=f"{type(e).__name__}: {e}"[:300])
        finally:
            self._writing.release()
