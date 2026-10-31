"""Cooperative per-job deadline and cancellation. Pipelines call check() between expensive stages."""
from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Callable

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("veritas_deadline", default=None)
_cancel: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("veritas_cancel", default=None)


class JobTimeout(Exception):
    pass


class StepTimeout(JobTimeout):
    """One step with a hard limit of its own (the reading and segmenting of a document) reached it. Its sentence names
    that limit and is the reason the person reads; the job's own time budget is named by one fixed sentence."""


class JobCancelled(Exception):
    """A cancellation was requested for the running job (the worker sets the event on heartbeat)."""


def set_deadline(seconds: float) -> None:
    _deadline.set(time.monotonic() + seconds)


def set_cancel_event(event: threading.Event | None) -> None:
    _cancel.set(event)


def clear() -> None:
    _deadline.set(None)
    _cancel.set(None)


def check(stage: str = "") -> None:
    during = f" during {stage}" if stage else ""
    event = _cancel.get()
    if event is not None and event.is_set():
        raise JobCancelled(f"Analysis was cancelled{during}")
    d = _deadline.get()
    if d is not None and time.monotonic() > d:
        raise JobTimeout(f"Analysis exceeded the time budget{during}")


def remaining() -> float | None:
    d = _deadline.get()
    return None if d is None else max(0.0, d - time.monotonic())


def run_within[T](seconds: float | None, call: Callable[[], T], *, what: str = "The job",
                  late: Callable[[T], None] | None = None) -> T:
    """Run `call` and wait at most `seconds` for it (None: no limit, run in the calling thread). A call that cannot be
    interrupted (a PDF render, a ZIP build) runs in a helper thread: at the limit the caller gets `JobTimeout` and goes on,
    so a hung call does not hold the worker thread. The helper thread is left to end by itself; what it returns after the
    limit is dropped, after `late` has seen it (to remove a work file). An exception of the call is raised in the caller."""
    if seconds is None:
        return call()
    box: dict[str, object] = {}
    finished, abandoned, guard = threading.Event(), threading.Event(), threading.Lock()

    def run() -> None:
        try:
            box["value"] = call()
        except BaseException as e:  # noqa: BLE001 - handed to the caller
            box["error"] = e
        with guard:
            finished.set()
            dropped = abandoned.is_set()
        if dropped and late is not None and "value" in box:
            with contextlib.suppress(Exception):  # nobody is left to tell
                late(box["value"])  # type: ignore[arg-type]

    context = contextvars.copy_context()
    threading.Thread(target=lambda: context.run(run), name="veritas-deadline", daemon=True).start()
    finished.wait(seconds)
    with guard:
        if not finished.is_set():
            abandoned.set()
            raise JobTimeout(f"{what} did not finish within the time limit of {seconds:g} s")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]
