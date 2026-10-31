"""`/api/metrics`: the Prometheus text exposition, built from the database when it is asked for (BUILD_SPEC N-13; ops review
OPS-06).

Before, the counters lived in the memory of the process that ran a v1 analysis: under the v2 pipeline nothing counted, and
an API beside a separate worker had nothing to count. Now every figure but the last two is read from the database or the
data volume at scrape time, so every process of an instance gives the same answer:

- `veritas_jobs{type,status}`: rows of the queue; every type and status is printed, a zero included;
- `veritas_jobs_oldest_queued_age_seconds`: 0 when nothing is queued;
- `veritas_jobs_attempts_exhausted`: jobs failed because their run was interrupted `max_requeue_attempts` times;
- `veritas_analyses{kind,status,fusion}`: revisions; `fusion` is `artefact`, `fallback` or `none` (no result yet, or v1);
- `veritas_analysis_duration_seconds{kind,quantile}` and `veritas_analysis_duration_samples{kind}`: seconds from
  `started_at` to `finished_at` of the last `DURATION_WINDOW` finished analyses of a kind (nearest rank);
- `veritas_workers`, `veritas_worker_heartbeat_age_seconds`, `veritas_worker_info{…}`, `veritas_worker_fusion{kind,mode}`:
  from `worker_heartbeats` (`app.engine.heartbeat`). The age is `+Inf` when no worker has a row;
- `veritas_data_free_bytes`, `veritas_data_total_bytes`: the volume of the data directory;
- `veritas_build_info{version,commit}`;
- `veritas_database_up`: 0 when the database could not be read; the figures above that come from it are then left out;
- `veritas_uptime_seconds` and `veritas_ready`: of the answering process.

Rows are counted as they are now: a deleted or purged revision leaves the counts, so these are gauges, not counters.
"""
from __future__ import annotations

import math
import shutil
import time
from datetime import datetime

from sqlalchemy import Connection, Engine, func, select

from app.api import state
from app.core import times
from app.core.config import settings
from app.core.logging import get_logger
from app.engine import heartbeat
from app.engine.queue import JOB_TYPES
from app.storage.db import Analysis, Job, utcnow

log = get_logger(__name__)

JOB_STATUSES = ("queued", "leased", "done", "failed", "cancelled")
DURATION_WINDOW = 500
QUANTILES = (0.5, 0.95)
EXHAUSTED_ERROR = "Processing was interrupted %"  # `JobQueue.requeue_expired`, the poison guard


def _label(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _sample(name: str, value: float | int, **labels: object) -> str:
    text = "+Inf" if value == math.inf else f"{value:.3f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    if not labels:
        return f"{name} {text}"
    return f"{name}{{{','.join(f'{k}="{_label(v)}"' for k, v in labels.items())}}} {text}"


def _family(lines: list[str], name: str, help_: str, samples: list[str]) -> None:
    lines += [f"# HELP {name} {help_}", f"# TYPE {name} gauge", *samples]


def quantile(ordered: list[float], q: float) -> float:
    """Nearest rank: the smallest value with at least `q` of the values at or below it."""
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _queue(conn: Connection, now: datetime, lines: list[str]) -> None:
    counts = {(t, s): 0 for t in sorted(JOB_TYPES) for s in JOB_STATUSES}
    for type_, status, n in conn.execute(select(Job.type, Job.status, func.count()).group_by(Job.type, Job.status)):
        counts[(type_, status)] = n
    _family(lines, "veritas_jobs", "Jobs of the queue by type and status.",
            [_sample("veritas_jobs", n, type=t, status=s) for (t, s), n in sorted(counts.items())])
    oldest = times.to_utc(conn.execute(select(func.min(Job.created_at)).where(Job.status == "queued")).scalar())
    _family(lines, "veritas_jobs_oldest_queued_age_seconds", "Age of the oldest queued job; 0 when nothing is queued.",
            [_sample("veritas_jobs_oldest_queued_age_seconds", 0.0 if oldest is None else max(0.0, (now - oldest).total_seconds()))])
    exhausted = conn.execute(select(func.count()).select_from(Job.__table__)
                             .where(Job.status == "failed", Job.error.like(EXHAUSTED_ERROR))).scalar_one()
    _family(lines, "veritas_jobs_attempts_exhausted", "Jobs failed because their run was interrupted too often.",
            [_sample("veritas_jobs_attempts_exhausted", exhausted)])


def _analyses(conn: Connection, lines: list[str]) -> None:
    grouped = conn.execute(select(Analysis.kind, Analysis.status, Analysis.fusion_mode, func.count())
                           .group_by(Analysis.kind, Analysis.status, Analysis.fusion_mode)).all()
    _family(lines, "veritas_analyses", "Analyses (revisions) by kind, status and fusion mode.",
            [_sample("veritas_analyses", n, kind=kind, status=status, fusion=fusion or "none")
             for kind, status, fusion, n in sorted(grouped, key=lambda r: (r[0], r[1], r[2] or ""))])
    durations, samples = [], []
    for kind in sorted({row[0] for row in grouped}):
        rows = conn.execute(select(Analysis.started_at, Analysis.finished_at)
                            .where(Analysis.kind == kind, Analysis.status == "done", Analysis.started_at.is_not(None),
                                   Analysis.finished_at.is_not(None))
                            .order_by(Analysis.finished_at.desc()).limit(DURATION_WINDOW)).all()
        seconds = sorted(max(0.0, (times.to_utc(end) - times.to_utc(start)).total_seconds()) for start, end in rows)
        if seconds:
            durations += [_sample("veritas_analysis_duration_seconds", quantile(seconds, q), kind=kind, quantile=q) for q in QUANTILES]
            samples.append(_sample("veritas_analysis_duration_samples", len(seconds), kind=kind))
    _family(lines, "veritas_analysis_duration_seconds",
            f"Seconds from start to finish of the last {DURATION_WINDOW} finished analyses of a kind.", durations)
    _family(lines, "veritas_analysis_duration_samples", "How many analyses the duration quantiles of a kind are over.", samples)


def _workers(conn: Connection, now: datetime, lines: list[str]) -> None:
    views = heartbeat.workers_in(conn)
    live = [v for v in views if v.live(now)]
    _family(lines, "veritas_workers", f"Workers whose heartbeat is at most {settings.worker_stale_s:g} s old and whose threads all reach the database.",
            [_sample("veritas_workers", len(live))])
    _family(lines, "veritas_worker_heartbeat_age_seconds", "Age of the newest worker heartbeat; +Inf when no worker has one.",
            [_sample("veritas_worker_heartbeat_age_seconds", views[0].age_s(now) if views else math.inf)])
    _family(lines, "veritas_worker_info", "One sample per live worker: what it runs on.",
            [_sample("veritas_worker_info", 1, worker_id=v.worker_id, hostname=v.hostname, device=v.device, profile=v.profile,
                     in_process=str(v.in_process).lower(), version=v.state.get("version") or "unknown") for v in live])
    described = heartbeat.newest(views, now)
    fusion = dict((described.state.get("fusion") or {}) if described is not None and described.live(now) else {})
    _family(lines, "veritas_worker_fusion", "Per kind, whether the live worker that wrote last serves a fuser artefact or the fallback.",
            [_sample("veritas_worker_fusion", 1, kind=kind, mode=mode) for kind, mode in sorted(fusion.items())])


def render(engine: Engine, now: datetime | None = None) -> str:
    from app.engine.build_info import git_commit

    now = now or utcnow()
    lines: list[str] = []
    _family(lines, "veritas_build_info", "Version and build commit of the answering process.",
            [_sample("veritas_build_info", 1, version=settings.version, commit=git_commit() or "unknown")])
    read: list[str] = []
    try:
        with engine.connect() as conn:  # one read transaction: the figures are of one moment
            _queue(conn, now, read)
            _analyses(conn, read)
            _workers(conn, now, read)
        up = 1
    except Exception as e:  # noqa: BLE001  a scrape answers while the database does not
        log.warning("metrics.database_unreadable", error=f"{type(e).__name__}: {e}"[:300])
        read, up = [], 0
    _family(lines, "veritas_database_up", "1 when the figures of the queue, the analyses and the workers were read from the database.",
            [_sample("veritas_database_up", up)])
    lines += read
    try:
        usage = shutil.disk_usage(settings.data_dir)
        _family(lines, "veritas_data_free_bytes", "Free bytes of the volume of the data directory.", [_sample("veritas_data_free_bytes", usage.free)])
        _family(lines, "veritas_data_total_bytes", "Size of the volume of the data directory.", [_sample("veritas_data_total_bytes", usage.total)])
    except OSError as e:
        log.warning("metrics.data_volume_unreadable", error=f"{type(e).__name__}: {e}"[:300])
    _family(lines, "veritas_uptime_seconds", "Seconds since the answering process started.",
            [_sample("veritas_uptime_seconds", int(time.time() - state.started_at))])
    _family(lines, "veritas_ready", "1 once the answering process finished its start.",
            [_sample("veritas_ready", 1 if state.ready.is_set() else 0)])
    return "\n".join(lines) + "\n"
