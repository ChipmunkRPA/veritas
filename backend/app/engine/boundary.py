"""Runner boundary (ARCHITECTURE §2.6): applicable → lease → nested deadline → load → run, under a watchdog.

- Every run ends as a persisted `DetectorResult` with status ok | not_applicable | skipped | failed | timeout and a
  sanitised reason. No detector error ever propagates to the analysis.
- A result whose run is not stored is not used (`store_run`; N-10a, code re-verification R2-03). The write is tried again
  after a pause; a run that still cannot be stored is failed, and the stage gets that failed result, never the output of
  a run of which no row and no raw output exist.
- Cooperative checks (`ctx.deadline.check`) raise `DetectorTimeout` between batches.
- A torch op cannot be pre-empted, so a watchdog marks the run `timeout` once the budget (plus a short grace for
  cooperative checks) is spent. It then cancels the detector's deadline and discards the late result when the op
  returns. The model lease is held until then, so memory still in use is never evicted.
- `isolation: process` detectors are refused (`skipped: process isolation unavailable`) until the subprocess worker
  lands with the codec and decode adapters.
"""
from __future__ import annotations

import contextvars
import dataclasses
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral
from typing import Any, Protocol

from app.core.errors import friendly_error, sanitise_message
from app.core.logging import get_logger
from app.engine.detector import ActivationUse, Detector, DetectorResult, RunContext
from app.engine.models import MemoryBudgetExceeded, ModelLease, ModelManager, ModelUnavailable
from app.engine.profiles import detector_timeout

log = get_logger(__name__)

WATCHDOG_GRACE_S = 0.25  # lets a cooperative DetectorTimeout end the run (and release its lease) before the watchdog
PERSIST_PAUSES_S: tuple[float, ...] = (0.5, 2.0)  # between the attempts to store a run: three attempts in all
NOT_STORED = ("The record of this check's run could not be stored, so its result is not used; details are in the server log "
              "under this analysis id.")


class DetectorTimeout(Exception):
    """Raised by `Deadline.check` once the budget is spent."""


class Cancelled(Exception):
    """Raised by `Deadline.check` after cancellation (analysis cancelled or watchdog fired)."""


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds >= 10 else f"{seconds:.1f}s"


class Deadline:
    """Cooperative deadline. `nested(seconds)` narrows to min(parent remaining, seconds). Cancelling a parent cancels
    its children, never the reverse."""

    def __init__(self, seconds: float | None = None, *, parent: Deadline | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.budget_s = seconds
        own_end = None if seconds is None else clock() + max(0.0, seconds)
        parent_end = parent._end if parent is not None else None
        if parent is not None and parent_end is not None and (own_end is None or parent_end < own_end):
            self._end, self.limit_label = parent_end, parent.limit_label
        else:
            self._end = own_end
            self.limit_label = "no" if seconds is None else f"{_fmt_s(seconds)} budget"
        self._event = threading.Event()
        self._events: tuple[threading.Event, ...] = (parent._events if parent is not None else ()) + (self._event,)
        self._lost = threading.Event()
        self._losts: tuple[threading.Event, ...] = (parent._losts if parent is not None else ()) + (self._lost,)

    def remaining(self) -> float | None:
        return None if self._end is None else max(0.0, self._end - self._clock())

    def check(self, label: str = "") -> None:
        during = f" during {label}" if label else ""
        if any(event.is_set() for event in self._events):
            raise Cancelled(f"cancelled{during}")
        if self._end is not None and self._clock() > self._end:
            raise DetectorTimeout(f"exceeded {self.limit_label}{during}")

    def nested(self, seconds: float | None) -> Deadline:
        return Deadline(seconds, parent=self, clock=self._clock)

    def cancel(self) -> None:
        self._event.set()

    def lose_lease(self) -> None:
        """Cancel because the job lease was lost (K-3): work stops as on a cancel, but the new lease owner, not this run, decides
        the outcome, so a runner that sees `lease_lost` writes no terminal state."""
        self._lost.set()
        self._event.set()

    @property
    def lease_lost(self) -> bool:
        return any(event.is_set() for event in self._losts)


@dataclass(frozen=True)
class DetectorRunRecord:
    analysis_id: str
    detector_id: str
    detector_version: str
    status: str
    reason: str | None
    device: str
    dtype: str
    started_at: str
    duration_ms: float
    result: DetectorResult


class RunStore(Protocol):
    def persist(self, record: DetectorRunRecord) -> None: ...


class InMemoryRunStore:
    """Detector-run persistence until the evidence tables land (P-2)."""

    def __init__(self) -> None:
        self._records: list[DetectorRunRecord] = []
        self._lock = threading.Lock()

    def persist(self, record: DetectorRunRecord) -> None:
        with self._lock:
            self._records.append(record)

    def for_analysis(self, analysis_id: str) -> list[DetectorRunRecord]:
        with self._lock:
            return [r for r in self._records if r.analysis_id == analysis_id]


def _persisted(store: RunStore, record: DetectorRunRecord, bound: Any, deadline: Deadline | None,
               pauses: tuple[float, ...]) -> str | None:
    """Store a run, again after each pause while the write fails. None once it is stored, else the last traceback."""
    for attempt, pause in enumerate((*pauses, None), 1):
        try:
            store.persist(record)
            return None
        except Exception:
            tb = traceback.format_exc()
        if deadline is not None and deadline.lease_lost:  # N-8: the job has another owner; this run stores nothing more
            raise Cancelled("The job lease was lost while the detector's run was stored")
        if pause is None:
            return tb
        bound.warning("detector.persist_retry", attempt=attempt, status=record.status, tb=tb[-1200:])
        time.sleep(pause)
    return None


def store_run(store: RunStore, record: DetectorRunRecord, bound: Any, deadline: Deadline | None = None) -> DetectorRunRecord:
    """Store a detector run and return the record whose result the stage uses (N-10a; code re-verification R2-03).

    A write that fails ("database is locked" behind a long write of another connection) is tried again after a pause, three
    times in all. A run that still cannot be stored is failed: the record returned is a failed run that holds nothing of
    the detector's output, with a reason that says so, and that record is stored if the store takes it now. The caller
    never gets the result of a run of which no row and no raw output exist. Raises `Cancelled` once the job lease is lost."""
    tb = _persisted(store, record, bound, deadline, PERSIST_PAUSES_S)
    if tb is None:
        return record
    bound.error("detector.persist_failed", status=record.status, attempts=len(PERSIST_PAUSES_S) + 1, tb=tb)
    result = DetectorResult.failed(NOT_STORED)
    failed = dataclasses.replace(record, status=result.status, reason=result.reason, device="", dtype="", result=result)
    if _persisted(store, failed, bound, deadline, ()) is not None:
        bound.error("detector.failed_run_not_stored")  # coverage counts the detector as failed without a row (`runs.statuses`)
    return failed


def run_detector(detector: Detector, ctx: RunContext, *, store: RunStore, models: ModelManager | None = None,
                 timeout_s: float | None = None, lease_wait_s: float = 0.0) -> DetectorResult:
    """Run one detector behind the boundary. Persists the result and does not raise, except after the job lease was lost:
    then nothing is persisted and `Cancelled` ends the stage (N-8). The result of a run that cannot be stored is replaced
    by a failed one (`store_run`)."""
    spec = detector.spec
    bound = log.bind(analysis_id=ctx.analysis_id, detector_id=spec.id)
    parent = ctx.deadline if isinstance(ctx.deadline, Deadline) else Deadline(None)
    prefer = str(ctx.device) if ctx.device else (models.default_device if models is not None else "cpu")
    budget = timeout_s if timeout_s is not None else detector_timeout(
        spec.requirements.timeout_s, ctx.profile, prefer, remaining_s=parent.remaining())
    nested = parent.nested(budget)
    started_at = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()
    result = _guarded(detector, ctx, nested, bound, models=models, prefer=prefer, budget=budget, lease_wait_s=lease_wait_s)
    if parent.lease_lost:
        # N-8 (K-3): the job has another owner. This run stores nothing more: no run row, no raw output, and no
        # "skipped: analysis cancelled" for an analysis nobody cancelled. The stage ends as the lease loss, not as a result.
        bound.warning("detector.discarded_after_lease_loss", status=result.status)
        raise Cancelled("The job lease was lost while the detector ran")
    elapsed = time.perf_counter() - t0
    result.timings.setdefault("total_s", round(elapsed, 4))
    record = DetectorRunRecord(ctx.analysis_id, spec.id, spec.version, result.status, result.reason, result.device,
                               result.dtype, started_at, round(elapsed * 1000, 2), result)
    record = store_run(store, record, bound, parent)
    bound.info("detector.finished", status=record.status, reason=record.reason, duration_ms=record.duration_ms)
    return record.result


def _guarded(detector: Detector, ctx: RunContext, nested: Deadline, bound: Any, *, models: ModelManager | None,
             prefer: str, budget: float, lease_wait_s: float) -> DetectorResult:
    spec = detector.spec
    if spec.requirements.isolation == "process":
        return DetectorResult.skipped("process isolation unavailable")
    try:
        applicability = detector.applicable(ctx)
    except Exception as e:
        bound.error("detector.applicable_failed", error=f"{type(e).__name__}: {e}"[:300], tb=traceback.format_exc())
        return DetectorResult.failed(friendly_error(e))
    if not applicability.ok:
        return DetectorResult.not_applicable(sanitise_message(applicability.reason or "not applicable"))

    holder: dict[str, Any] = {}
    done, timed_out = threading.Event(), threading.Event()

    def body() -> None:
        try:
            holder["result"] = _execute(detector, ctx, nested, models, prefer, lease_wait_s)
        except BaseException as e:  # mapped to a status below
            holder["error"], holder["tb"] = e, traceback.format_exc()
        finally:
            done.set()
            if timed_out.is_set():
                late = holder.get("result")
                bound.warning("detector.late_result_discarded", late_status=getattr(late, "status", None),
                              late_error=type(holder["error"]).__name__ if "error" in holder else None)

    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(body,), name=f"detector:{spec.id}", daemon=True).start()
    if not done.wait(timeout=max(0.0, budget) + WATCHDOG_GRACE_S):
        timed_out.set()
        nested.cancel()
        bound.warning("detector.watchdog_timeout", budget_s=budget)
        return DetectorResult.timeout(f"exceeded {_fmt_s(budget)} budget")

    error = holder.get("error")
    if error is None:
        return holder["result"]
    if isinstance(error, DetectorTimeout):
        return DetectorResult.timeout(str(error))
    if isinstance(error, Cancelled):
        return DetectorResult.skipped("analysis cancelled")
    if isinstance(error, MemoryBudgetExceeded | ModelUnavailable):
        bound.warning("detector.models_unavailable", error=str(error))
        return error.result()
    bound.error("detector.failed", error=f"{type(error).__name__}: {error}"[:300], tb=holder.get("tb"))
    return DetectorResult.failed(friendly_error(error))


def run_device(detector: Detector) -> str | None:
    """The device a stage asks for when it runs this detector: CPU for an adapter that declares CPU alone, else None (the
    model manager's default). The capacity plan asks the same rule (N-14a)."""
    return "cpu" if tuple(detector.spec.requirements.devices) == ("cpu",) else None


def planned_allowance(detector: Detector, ctx: RunContext, models: ModelManager) -> tuple[int, tuple[ActivationUse, ...]]:
    """The activation allowance of the adapter's plan in this manager's estimates, with the validated plan. Nothing is
    acquired and nothing is logged: a run asks it through `_activation_allowance`, the capacity plan directly (N-14a)."""
    plan = detector.activation_plan(ctx)
    required = set(detector.spec.requirements.model_ids)
    counts: dict[str, int] = {}
    if not isinstance(plan, tuple):
        raise ModelUnavailable("model unavailable: activation plan must be a tuple")
    for use in plan:
        if (not isinstance(use, ActivationUse) or not isinstance(use.model_id, str)
                or use.model_id not in required or use.model_id in counts
                or isinstance(use.max_items, bool) or not isinstance(use.max_items, Integral) or use.max_items < 1
                or not isinstance(use.unit, str) or not use.unit.strip()
                or not isinstance(use.input_bound, str) or not use.input_bound.strip()
                or not isinstance(use.limitations, tuple)
                or any(not isinstance(x, str) or not x.strip() for x in use.limitations)):
            raise ModelUnavailable("model unavailable: invalid activation plan")
        counts[use.model_id] = int(use.max_items)
    if set(counts) != required:
        raise ModelUnavailable("model unavailable: activation plan does not cover all leased models")
    return models.activation_bytes_for_items(counts), plan


def _activation_allowance(detector: Detector, ctx: RunContext, models: ModelManager) -> int:
    """Validate the adapter envelope before acquiring anything; use this manager's estimates."""
    allowance, plan = planned_allowance(detector, ctx, models)
    log.info("detector.activation_allowance", analysis_id=ctx.analysis_id, detector_id=detector.spec.id,
             estimated=True, activation_bytes=allowance,
             plan=[dataclasses.asdict(use) for use in plan])
    return allowance


def _execute(detector: Detector, ctx: RunContext, nested: Deadline, models: ModelManager | None, prefer: str,
             lease_wait_s: float) -> DetectorResult:
    model_ids = detector.spec.requirements.model_ids
    if model_ids and models is None:
        raise ModelUnavailable("model unavailable: no ModelManager is bound")
    allowance = _activation_allowance(detector, ctx, models) if model_ids and models is not None else 0
    remaining = nested.remaining()
    wait_s = lease_wait_s if remaining is None else min(lease_wait_s, remaining)
    lease_cm = models.lease(model_ids, prefer=prefer, activation_bytes=allowance, wait_s=wait_s) \
        if model_ids and models is not None else nullcontext(ModelLease(prefer, "float32", {}))
    with lease_cm as lease:
        nested.check("model lease")
        run_ctx = dataclasses.replace(ctx, lease=lease, device=lease.device, dtype=lease.dtype, deadline=nested)
        detector.load(lease)
        nested.check("model load")
        result = detector.run(run_ctx)
        if not isinstance(result, DetectorResult):
            raise TypeError(f"{detector.spec.id}.run() returned {type(result).__name__}, not a DetectorResult")
        result.device = result.device or lease.device
        result.dtype = result.dtype or lease.dtype
        return result
