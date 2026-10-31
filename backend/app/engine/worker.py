"""Queue worker (ARCHITECTURE §3.4).

- **How it runs.** Worker threads claim jobs from the DB queue and run the handler for the job type in that thread.
  A heartbeat thread keeps the lease alive while the handler runs.
- **Cancellation.** When a heartbeat reports `cancel_requested`, the worker cancels the job cooperatively: it sets
  `ctx.cancel_event` and cancels `ctx.deadline`. A handler observes this with `ctx.check()`, or with
  `app.core.deadline.check()` after binding the event.
- **Lease loss** (K-3). When a heartbeat reports the lease is no longer held, the job is stopped the same way, but with
  `ctx.lease_lost` (and `Deadline.lease_lost`) set: this is not a cancel request. The runner then writes no terminal state, and
  the worker neither completes the job nor cleans its temporary files; the new lease owner decides the outcome.
- **Stop** (N-14; OPS-15, OPS-16). `stop(timeout)` stops claiming and waits that long, in all, for the jobs that run. A
  job that is still running then is released: its context is marked as for a lost lease (the runner writes nothing more
  and its outcome is not recorded), and `queue.release` puts it back as `queued` without counting this attempt, with one
  `worker.job_released` line. The next worker claims it at once, where it used to wait for the lease to expire and to
  count the stop as an interruption. `on_released` is then told (N-17a; O2-05): the record the job was working on goes
  back to waiting with it (`app.api.jobs`: an analysis reads `queued` again, not `running`, until a worker claims it).
- **Outcomes.** A handler returns the final status (`done`, `failed` or `cancelled`; `None` means done). An exception
  fails the job, and the stored error is sanitised by `friendly_error`.

In-process workers start from the API lifespan (`app.api.jobs.start`). Scale-out containers run
`python -m app.engine.worker`.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from app.core.deadline import JobCancelled
from app.core.errors import friendly_error
from app.core.logging import get_logger
from app.engine.boundary import Cancelled, Deadline
from app.engine.queue import JobQueue, JobView

log = get_logger(__name__)

LOOP_BACKOFF_MAX_S = 30.0  # the longest wait between two tries of a worker thread whose loop keeps failing
HOLD_POLL_S = 2.0  # how often a worker on hold asks again whether the reason still holds
STOP_TIMEOUT_S = 30.0  # a stopped standalone worker waits this long for its running job, then releases it (Compose: 40 s grace)


@dataclass
class JobContext:
    job: JobView
    worker_id: str
    deadline: Deadline
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lease_event: threading.Event = field(default_factory=threading.Event)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def lease_lost(self) -> bool:
        return self.lease_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.deadline.cancel()

    def lose_lease(self) -> None:
        self.lease_event.set()
        self.cancel_event.set()
        self.deadline.lose_lease()

    def check(self, label: str = "") -> None:
        self.deadline.check(label)


Handler = Callable[[JobContext], "str | None"]


class Worker:
    def __init__(self, queue: JobQueue, handlers: Mapping[str, Handler], *, threads: int = 1, poll_s: float = 0.5,
                 heartbeat_s: float = 10.0, timeout_s: float | None = None, sweep_s: float = 5.0,
                 worker_id: str | None = None, on_terminal: Callable[[JobView], None] | None = None,
                 hold: Callable[[], str | None] | None = None, on_released: Callable[[JobView], None] | None = None):
        if not handlers:
            raise ValueError("a worker needs at least one handler")
        self.queue = queue
        self.handlers = dict(handlers)
        self.threads = max(1, threads)
        self.poll_s = poll_s
        self.heartbeat_s = heartbeat_s
        self.timeout_s = timeout_s
        self.sweep_s = sweep_s
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.on_terminal = on_terminal
        self.on_released = on_released  # N-17a: a job given back at a stop, queued again
        self.hold = hold  # N-13: a reason to claim nothing for now (the data volume below its floor), or None
        self._held: str | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_sweep = 0.0
        self._sweep_lock = threading.Lock()
        self._health_lock = threading.Lock()
        self._health_generation = 0
        self._last_db_success: list[float | None] = [None] * self.threads
        self._slot = threading.local()
        self._running: dict[str, JobContext] = {}  # job id -> context, while its handler runs in this process
        self._running_lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        with self._health_lock:
            self._health_generation += 1
            generation = self._health_generation
            self._last_db_success = [None] * self.threads
        for i in range(self.threads):
            t = threading.Thread(target=self._loop, args=(i, generation), name=f"veritas-worker-{i}", daemon=True)
            t.start()
            with self._health_lock:
                self._threads.append(t)
        log.info("worker.started", worker_id=self.worker_id, threads=self.threads, types=sorted(self.handlers))

    def stop(self, timeout: float = 2.0) -> list[JobView]:
        """Stop claiming and wait up to `timeout` seconds, in all, for the jobs that run. One that is still running then
        is released (module docstring): queued again, this attempt not counted. Returns the released jobs."""
        self._stop.set()
        self._wake.set()
        with self._health_lock:
            threads = list(self._threads)
        until = time.monotonic() + timeout
        for t in threads:
            t.join(max(0.0, until - time.monotonic()))
        with self._health_lock:
            self._threads = []
        return self.release_running()

    def release_running(self) -> list[JobView]:
        """Give back every job whose handler still runs in this process. The lock is held throughout: a handler that
        returns meanwhile waits, then finds its lease gone and records nothing."""
        released: list[JobView] = []
        with self._running_lock:
            for ctx in list(self._running.values()):
                ctx.lose_lease()  # first: from here on the runner writes no result and no terminal state
                try:
                    view = self.queue.release(ctx.job.id, self.worker_id)
                except Exception as e:  # noqa: BLE001  the lease then expires as before; the stop goes on
                    log.error("worker.release_failed", job_id=ctx.job.id, error=f"{type(e).__name__}: {e}"[:300])
                    continue
                if view is None:  # ended or taken over since
                    continue
                log.warning("worker.job_released", job_id=view.id, type=view.type, subject_id=view.subject_id,
                            outcome=view.status, attempts=view.attempts, reason="the worker was stopped before the job ended")
                released.append(view)
        for view in released:
            if view.status == "cancelled" and self.on_terminal is not None:
                try:
                    self.on_terminal(view)
                except Exception as e:  # noqa: BLE001
                    log.error("worker.on_terminal_failed", job_id=view.id, error=str(e))
            elif view.status == "queued" and self.on_released is not None:
                try:
                    self.on_released(view)
                except Exception as e:  # noqa: BLE001  the job is queued either way; the stop goes on
                    log.error("worker.on_released_failed", job_id=view.id, error=f"{type(e).__name__}: {e}"[:300])
        return released

    def wake(self) -> None:
        self._wake.set()

    def backoff_s(self, failures: int) -> float:
        """How long a thread waits after its n-th failure in a row: the poll interval doubled each time, up to
        `LOOP_BACKOFF_MAX_S` (N-13; OPS-19: a full volume gave 97 `worker.loop_error` lines in 30 seconds)."""
        return min(LOOP_BACKOFF_MAX_S, self.poll_s * 2 ** min(failures, 16))

    def _on_hold(self) -> str | None:
        """The reason this worker claims nothing for now, said once when it begins and once when it ends. The queue is
        still read, so the worker is seen to live and to reach the database while it waits."""
        reason = None
        if self.hold is not None:
            try:
                reason = self.hold()
            except Exception as e:  # noqa: BLE001  a check that fails holds nothing
                log.error("worker.hold_check_failed", error=f"{type(e).__name__}: {e}"[:300])
        with self._health_lock:
            before, self._held = self._held, reason  # the probes read the reason of the moment, with its numbers
        # N-17a (O2-05): the reason holds the free byte count, and a line for each change of it was a line every few seconds
        if reason is not None and before is None:
            log.warning("worker.on_hold", worker_id=self.worker_id, reason=reason)
        elif reason is None and before is not None:
            log.info("worker.hold_ended", worker_id=self.worker_id, reason=before)
        return reason

    def _loop(self, slot: int, generation: int) -> None:
        self._slot.value = (generation, slot)
        failures = 0
        while not self._stop.is_set():
            ran = False
            try:
                if self._on_hold() is not None:
                    self.queue.depth()
                    self._record_db_success(self._slot.value)
                    self._stop.wait(HOLD_POLL_S)
                    continue
                self.sweep()
                ran = self.run_once()
                if failures:
                    log.info("worker.loop_recovered", worker_id=self.worker_id, after_failures=failures)
                failures = 0
            except Exception as e:  # a DB hiccup must not kill the worker thread
                failures += 1
                log.error("worker.loop_error", error=str(e), tb=traceback.format_exc()[-800:], consecutive=failures,
                          retry_in_s=round(self.backoff_s(failures), 2))
                self._stop.wait(self.backoff_s(failures))  # not `_wake`: a submit does not cut the wait short
                continue
            if not ran:
                self._wake.wait(self.poll_s)
                self._wake.clear()

    def _record_db_success(self, slot: tuple[int, int] | None) -> None:
        if slot is None:
            return
        generation, index = slot
        with self._health_lock:
            if generation == self._health_generation and not self._stop.is_set():
                self._last_db_success[index] = time.monotonic()

    def health_snapshot(self, max_age_s: float = 45.0) -> dict:
        """All configured slots must be alive and have recently claimed or renewed a held lease."""
        with self._health_lock:
            now = time.monotonic()
            stopping = self._stop.is_set()
            slots = []
            for index, last in enumerate(self._last_db_success):
                alive = index < len(self._threads) and self._threads[index].is_alive()
                slots.append({"slot": index, "alive": alive,
                              "last_db_success_age_s": None if last is None else now - last})
            alive_threads = sum(slot["alive"] for slot in slots)
            ready = (not stopping and len(self._threads) == self.threads and alive_threads == self.threads
                     and all(slot["last_db_success_age_s"] is not None
                             and 0 <= slot["last_db_success_age_s"] <= max_age_s for slot in slots))
            return {"ready": ready, "stopping": stopping, "configured_threads": self.threads,
                    "alive_threads": alive_threads, "slots": slots, "on_hold": self._held}

    # --- one step --------------------------------------------------------------------------------------------

    def sweep(self, force: bool = False) -> list[JobView]:
        """Re-queue expired leases (at most every `sweep_s` unless forced) and report terminal outcomes."""
        with self._sweep_lock:
            if not force and time.monotonic() - self._last_sweep < self.sweep_s:
                return []
            self._last_sweep = time.monotonic()
        changed = self.queue.requeue_expired()
        for job in changed:
            if job.status in ("failed", "cancelled") and self.on_terminal is not None:
                try:
                    self.on_terminal(job)
                except Exception as e:
                    log.error("worker.on_terminal_failed", job_id=job.id, error=str(e))
        return changed

    def run_once(self) -> bool:
        """Claim and run at most one job in the calling thread. Returns False when the queue had nothing to claim."""
        job = self.queue.claim(self.worker_id, types=sorted(self.handlers))
        self._record_db_success(getattr(self._slot, "value", None))
        if job is None:
            return False
        self._execute(job)
        return True

    def _execute(self, job: JobView) -> None:
        ctx = JobContext(job=job, worker_id=self.worker_id, deadline=Deadline(self.timeout_s))
        done = threading.Event()
        slot = getattr(self._slot, "value", None)

        def beat() -> None:
            while not done.wait(self.heartbeat_s):
                try:
                    hb = self.queue.heartbeat(job.id, self.worker_id)
                except Exception as e:
                    log.error("worker.heartbeat_failed", job_id=job.id, error=str(e))
                    continue
                if not hb.held:
                    log.warning("worker.lease_lost", job_id=job.id)
                    ctx.lose_lease()
                    return
                self._record_db_success(slot)
                if hb.cancel_requested and not ctx.cancelled:
                    log.info("worker.cancel_requested", job_id=job.id)
                    ctx.cancel()

        heart = threading.Thread(target=beat, name=f"veritas-heartbeat-{job.id[:8]}", daemon=True)
        heart.start()
        error = None
        with self._running_lock:
            self._running[job.id] = ctx
        try:
            status = self.handlers[job.type](ctx) or "done"
        except (Cancelled, JobCancelled):
            status, error = "cancelled", "Cancelled on request"
        except Exception as e:
            log.error("worker.job_failed", job_id=job.id, error=str(e), tb=traceback.format_exc()[-1200:])
            status, error = "failed", friendly_error(e)
        finally:
            with self._running_lock:  # the handler has ended: a stop from here on leaves the job to this thread
                self._running.pop(job.id, None)
            done.set()
            heart.join(self.heartbeat_s + 1.0)
        if ctx.lease_lost:  # another worker holds the job now; its outcome is not this worker's to record
            log.warning("worker.job_abandoned", job_id=job.id, type=job.type, reason="lease lost")
            return
        if status == "cancelled" and error is None:
            error = "Cancelled on request"
        self.queue.complete(job.id, self.worker_id, status, error)
        log.info("worker.job_finished", job_id=job.id, type=job.type, status=status)


def _check_configured_device() -> str:
    """Exercise the selected device before standalone readiness; no model weights or warmup."""
    import torch

    from app.core.device import pick_device

    device = pick_device()
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    value = (torch.ones(1, device=device) + 1).sum()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    if value.item() != 2.0:
        raise RuntimeError("configured device startup operation failed")
    return str(device)


def _warn_uncalibrated() -> None:
    """A worker names, at its start, an artefact file that a packaged one is served in place of (D-57); a prod worker also says
    once per kind when an artefact exists and the uncalibrated fallback serves, with the reason (OPS-03)."""
    from app.core.config import settings
    from app.engine.fusion.registry import warn_fallbacks, warn_shadowed

    warn_shadowed()
    if settings.environment != "prod":
        return
    from app.engine.pipelines import default_profile
    from app.engine.registry import get_registry

    warn_fallbacks(get_registry(), default_profile())


def main(argv: list[str] | None = None) -> int:
    """Standalone worker for scale-out deployments (`VERITAS_RUN_WORKER=false` on the API containers)."""
    parser = argparse.ArgumentParser(prog="python -m app.engine.worker", description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="sweep expired leases, run at most one job, then exit")
    parser.add_argument("--health-file", help="atomically publish standalone worker readiness at this path")
    parser.add_argument("--stop-timeout", type=float, default=STOP_TIMEOUT_S, metavar="SECONDS",
                        help="on SIGTERM, how long to wait for the running job before it is released to the queue "
                             f"(default {STOP_TIMEOUT_S:g}; keep the container's stop grace period above it)")
    args = parser.parse_args(argv)
    if args.once and args.health_file:
        parser.error("--health-file requires the continuously running worker, not --once")

    health = None
    if args.health_file:
        from app.engine.worker_health import HealthFile
        health = HealthFile(args.health_file)
        health.prepare()
    worker = None
    started = False
    held = None
    beats = None
    try:
        from app.api import jobs
        from app.core.config import settings
        from app.storage import keyguard, refusals, running
        from app.storage.db import init_db

        settings.ensure_dirs()
        with refusals.logged(log):  # N-19e (E-17): a refused start is one `startup.refused` with the sentence
            held = running.hold()  # N-12a: as the API; `backup restore` refuses a data directory a worker runs on
            init_db()
            keyguard.check_start()  # N-12: as the API's start
        worker = jobs.build_worker()
        _warn_uncalibrated()
        device = _check_configured_device() if health is not None else None
        if args.once:
            worker.sweep(force=True)
            worker.run_once()
            return 0
        stopping = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stopping.set())
        started = True
        worker.start()
        beats = jobs.publish_heartbeat(worker, in_process=False)  # N-13: the API reads that this worker lives, and on what
        if health is None:
            stopping.wait()
        else:
            while not stopping.is_set():
                health.write({**worker.health_snapshot(), "device": device, "device_startup_checked": True})
                stopping.wait(2.0)
        return 0
    finally:
        try:
            if health is not None:
                health.clear()
        finally:
            try:
                if beats is not None:
                    beats.stop()
                if started and worker is not None:
                    worker.stop(timeout=max(0.0, args.stop_timeout))  # a job that outlasts it is released, not abandoned
            finally:
                from app.storage import db, running

                db.shutdown()  # N-12a: the log is folded in and the connections are closed, as when the API stops
                running.release(held)


if __name__ == "__main__":
    from app.core.logging import configure_logging

    configure_logging()  # N-13: the worker's lines are the API's (JSON in prod, UTC, at VERITAS_LOG_LEVEL)
    sys.exit(main())
