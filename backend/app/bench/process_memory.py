"""Observed RSS maxima for explicitly selected local process trees.

This measures sampled resident pages, not allocator/GPU memory or a true peak.
Any failed observation makes the entire recording incomplete. Valid earlier
samples remain available as lower bounds, never as a successful measurement.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

import psutil


class ProcessMemorySampler:
    """Sample unique processes under explicit roots; bind every observed PID to its creation time.

    The context must finish before ``report()['complete']`` can be true. Root
    processes must stay alive through the final sample. Descendants may normally
    exit between samples; a descendant lost during an observation invalidates it.
    No process is started, signalled or otherwise modified by this sampler.
    """

    def __init__(self, pids: Sequence[int], interval_s: float = 0.25, *,
                 process_factory: Callable[[int], Any] = psutil.Process,
                 clock: Callable[[], float] = time.monotonic) -> None:
        roots = list(pids)
        if any(isinstance(pid, bool) or not isinstance(pid, Integral) or pid <= 0 for pid in roots):
            raise ValueError("process roots must be positive integer PIDs")
        if isinstance(interval_s, bool) or not isinstance(interval_s, Real) or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("sampling interval must be positive and finite")
        self._pids = tuple(dict.fromkeys(int(pid) for pid in roots))
        self.interval_s = float(interval_s)
        self._process = process_factory
        self._clock = clock
        self._identities: dict[int, float] = {}
        self._errors: list[dict[str, Any]] = []
        self._sample_count = 0
        self._attempts = 0
        self._maximum: int | None = None
        self._start: float | None = None
        self._end: float | None = None
        self._finished = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _error(self, operation: str, pid: int | None, error: str) -> None:
        self._errors.append({"sample": self._attempts, "operation": operation, "pid": pid, "error": error})

    def _bind(self, pid: int, created: float) -> None:
        if isinstance(created, bool) or not isinstance(created, Real) or not math.isfinite(created) or created <= 0:
            raise ValueError("invalid process creation time")
        created = float(created)
        if pid in self._identities and self._identities[pid] != created:
            raise ProcessLookupError("PID was reused")
        self._identities[pid] = created

    def _identity(self, pid: int):
        # Use a fresh handle: psutil caches create_time within each Process object.
        process = self._process(pid)
        self._bind(pid, process.create_time())
        if process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            raise ProcessLookupError("process has exited")
        return process

    def _sample(self) -> None:
        with self._lock:
            self._attempts += 1
            if not self._pids:
                self._error("roots", None, "no process roots supplied")
                return
            errors_before = len(self._errors)
            pending = list(self._pids)
            seen: set[int] = set()
            total = 0
            while pending:
                pid = pending.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    process = self._identity(pid)
                except Exception as exc:
                    self._error("identity", pid, type(exc).__name__)
                    continue
                try:
                    children = list(process.children(recursive=False))
                except Exception as exc:
                    self._error("children", pid, type(exc).__name__)
                    children = []
                for child in children:
                    child_pid = None
                    try:
                        child_pid = child.pid
                        if isinstance(child_pid, bool) or not isinstance(child_pid, Integral) or child_pid <= 0:
                            raise ValueError("invalid child PID")
                        # Bind the handle returned by enumeration before acquiring a
                        # fresh one, otherwise a recycled child PID could be accepted.
                        self._bind(int(child_pid), child.create_time())
                        pending.append(int(child_pid))
                    except Exception as exc:
                        safe_pid = int(child_pid) if isinstance(child_pid, Integral) and not isinstance(child_pid, bool) else None
                        self._error("child_identity", safe_pid, type(exc).__name__)
                try:
                    rss = process.memory_info().rss
                    if isinstance(rss, bool) or not isinstance(rss, Integral) or rss < 0:
                        raise ValueError("invalid RSS")
                    self._identity(pid)  # Refuse disappearance or PID reuse during the read.
                    total += int(rss)
                except Exception as exc:
                    self._error("rss", pid, type(exc).__name__)
            # A root can disappear while another tree is being sampled.
            for pid in self._pids:
                try:
                    self._identity(pid)
                except Exception as exc:
                    self._error("root_after_sample", pid, type(exc).__name__)
            if len(self._errors) == errors_before:
                self._sample_count += 1
                self._maximum = total if self._maximum is None else max(self._maximum, total)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> ProcessMemorySampler:
        with self._lock:
            if self._start is not None:
                raise RuntimeError("a process memory recording cannot be reused")
            self._start = self._clock()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="process-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        with self._lock:
            self._end = self._clock()
            self._finished = True

    def report(self) -> dict[str, Any]:
        with self._lock:
            end = self._end if self._finished else self._clock()
            return {
                "complete": self._finished and self._sample_count > 0 and not self._errors,
                "metric": "sampled_process_tree_rss",
                "sampled_max_rss_bytes": self._maximum,
                "interval_s": self.interval_s,
                "sample_count": self._sample_count,
                "sample_attempts": self._attempts,
                "duration_s": None if self._start is None else max(0.0, end - self._start),
                "finished": self._finished,
                "roots": [{"pid": pid, "create_time": self._identities.get(pid)} for pid in self._pids],
                "errors": [dict(error) for error in self._errors],
                "limitations": [
                    "Maximum of successful RSS samples; short-lived processes and between-sample peaks may be missed.",
                    "Process RSS totals can double-count shared pages; not GPU memory or a true process/system peak.",
                    "Only the supplied process roots and observed descendants are in scope; other workers are not inferred.",
                ],
            }
