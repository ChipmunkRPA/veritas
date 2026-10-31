"""DAG runner (ARCHITECTURE §3.2–3.3).

- **Scheduling.** Stages are sorted topologically (stable in definition order). A stage starts once every dependency
  has finished. Accelerator and mixed stages share one serial lane; CPU stages share a small thread pool. Stages whose
  dependencies are done therefore run at the same time when they sit in different lanes, or in the CPU pool.
- **Persistence and resume.**
  - Every attempt writes an `analysis_stages` row. A `persist=True` stage stores its output as a canonical-JSON
    `stage_output` artefact.
  - On a later run, a stage is reused when its row is `done`, its artefact re-hashes correctly, its optional
    reuse predicate accepts the output, and no dependency recomputed a persisted output, failed or was cancelled.
  - `persist=False` stages (decoded pixels, media handles) always recompute. They are deterministic, so they do not
    invalidate their dependents.
  - A stored output that fails re-hashing is quarantined and recomputed.
  - **The objects a stage names (N-18a; operations re-check O3-01).** A stage's output lists the store objects written for
    it under `artefacts` (`ExhibitWriter`: maps, masks, previews, thumbnails, stills). They are indexed in the transaction
    that marks the stage `done`, so a finished stage's objects are named by rows while its run waits (after a stop or a
    kill) and `store gc` or a purge leaves them. And a stage is reused only when every object it lists is in the store
    and re-hashes: otherwise the stage runs again, since it is the stage that wrote them (a stage finished by an earlier
    build has no such rows; an object can be removed by hand).
- **Failures.** A failing critical stage stops scheduling, so stages not yet started are marked `skipped`. A failing
  non-critical stage lets its dependents run without that output.
- **Cancellation and budget.** Cancelling the deadline, or exceeding it, stops scheduling. Running stages observe it
  cooperatively. Completed stage outputs are kept.
- **Events.** `stage_started`, `stage_finished`, `progress` and one terminal `done` / `failed` / `cancelled`. The
  terminal event is written before the analysis status changes.
"""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.canonical import canonical_json
from app.core.deadline import JobCancelled, JobTimeout, StepTimeout
from app.core.errors import friendly_error
from app.core.logging import get_logger
from app.engine.boundary import Cancelled, Deadline, DetectorTimeout
from app.engine.detector import Profile
from app.engine.events import emit, finish_analysis
from app.engine.exhibits import ExhibitWriter
from app.storage.artefacts import EvidenceConflict, index_object, register_artefact
from app.storage.db import Analysis, AnalysisStage, Artefact, Job, utcnow
from app.storage.objects import ObjectIntegrityError, ObjectNotFound, ObjectStore

log = get_logger(__name__)

TIMEOUT_MESSAGE = "The analysis exceeded its time budget"
_MISS = object()


class PipelineError(ValueError):
    """The stage graph is invalid (duplicate ids, unknown dependencies or a cycle)."""


@dataclass(frozen=True)
class StageDef:
    id: str
    fn: Callable[[StageContext], Any]
    after: tuple[str, ...] = ()
    lane: Literal["cpu", "accelerator", "mixed"] = "cpu"
    persist: bool = True  # False ⇒ recomputed on resume (deterministic from persisted inputs)
    critical: bool = False  # failure fails the analysis
    profiles: frozenset[Profile] = frozenset(Profile)
    weight: float = 1.0  # share of the progress fraction
    reuse_if: Callable[[Any], bool] | None = None  # semantic compatibility, after stored-byte verification


@dataclass
class StageContext:
    analysis_id: str
    stage_id: str
    profile: Profile
    deadline: Deadline
    store: ObjectStore
    sessions: Callable[[], Session]
    outputs: Mapping[str, Any]  # JSON outputs of finished stages (missing when a non-critical stage failed)
    state: dict[str, Any]  # shared in-memory objects for this run (media handle, decoded pixels); never persisted
    exhibits: ExhibitWriter
    log: Any

    def check(self, label: str = "") -> None:
        self.deadline.check(label or self.stage_id)

    def emit(self, type: str, payload: Mapping[str, Any] | None = None) -> int:
        return emit(self.sessions, self.analysis_id, type, payload)


@dataclass
class RunOutcome:
    status: Literal["done", "failed", "cancelled", "lease_lost"]  # lease_lost: stopped, nothing terminal written (K-3)
    stages: dict[str, str]
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def toposort(stages: Sequence[StageDef]) -> list[StageDef]:
    by_id: dict[str, StageDef] = {}
    for stage in stages:
        if stage.id in by_id:
            raise PipelineError(f"duplicate stage id {stage.id!r}")
        by_id[stage.id] = stage
    for stage in stages:
        unknown = [d for d in stage.after if d not in by_id]
        if unknown:
            raise PipelineError(f"stage {stage.id!r} depends on unknown stages {unknown}")
    ordered: list[StageDef] = []
    placed: set[str] = set()
    remaining = list(stages)
    while remaining:
        ready = [s for s in remaining if all(d in placed for d in s.after)]
        if not ready:
            raise PipelineError(f"the stage graph has a cycle among {sorted(s.id for s in remaining)}")
        for stage in ready:
            ordered.append(stage)
            placed.add(stage.id)
        remaining = [s for s in remaining if s.id not in placed]
    return ordered


STAGE_FAILURES = "stage_failures"  # in a run's shared state: stage id -> reason, for the stages that are not critical and failed


def listed_objects(output: Any) -> list[dict[str, Any]]:
    """The store objects a stage's output names: the records of what `ExhibitWriter` wrote for it (`artefacts`). They are
    what `render` indexes (`stages.common.index_artefacts`) and what later stages read."""
    items = output.get("artefacts") if isinstance(output, dict) else None
    return [a for a in items or [] if isinstance(a, dict) and a.get("sha256")]


def evidence_missing(by_stage: Mapping[str, int], *, reader: str | None = None) -> str:
    """The reason of an analysis that is not published because an object of the evidence store is missing (N-18a), by the
    stage that wrote it; `reader` is the stage that could not read an object no stage of this run lists (N-18b)."""
    total = sum(by_stage.values()) or 1
    where = (", ".join(f"`{stage}` ({n})" for stage, n in sorted(by_stage.items())) if by_stage else None)
    return (f"The result was not published: {total} evidence object{'s' if total != 1 else ''} that this analysis wrote "
            f"{'are' if total != 1 else 'is'} no longer in the evidence store "
            f"({'written by stage ' + where if where else 'read by stage `' + str(reader) + '`'}). Run the analysis "
            "again: a stage whose objects are missing runs again and writes them.")


def missing_object(error: BaseException) -> ObjectNotFound | None:
    """The `ObjectNotFound` a stage failed on, itself or as the cause of what it raised."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, ObjectNotFound):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


# N-19f (code review R4-04): the analysis did not write its original, and no run writes it again
ORIGINAL_MISSING = ("The original file of this analysis is no longer in the evidence store, and the analysis cannot run without "
                    "it. Restore it from a backup of the store, or upload the same file again, which writes the object again; "
                    "the analysis can then be analysed again.")


def _named_prefix(missing: ObjectNotFound) -> str:
    """The start of the hash by which the store names a missing object ("object 3fa9c1…"); empty when the error names none."""
    words = str(missing).split()
    return words[1].rstrip("…") if len(words) > 1 and words[0] == "object" else ""


def is_original(missing: ObjectNotFound, sha256: str | None) -> bool:
    """Whether the missing object is the original of the analysis (N-19f, R4-04): it has the analysis's own hash, whichever
    stage read it, or `ingest` could neither find nor restore it (`ensure_original_in_store`, the error that names no
    hash)."""
    if str(missing).startswith("no upload of media "):
        return True
    prefix = _named_prefix(missing)
    return len(prefix) >= 8 and bool(sha256) and str(sha256).startswith(prefix)


def writer_of(missing: ObjectNotFound, outputs: Mapping[str, Any]) -> str | None:
    """The stage of this run whose output lists the object (the store names it by the start of its hash)."""
    prefix = _named_prefix(missing)
    if len(prefix) < 8:
        return None
    return next((stage for stage, output in outputs.items() if any(str(a["sha256"]).startswith(prefix) for a in listed_objects(output))),
                None)


class Runner:
    def __init__(self, stages: Sequence[StageDef], *, sessions: Callable[[], Session], store: ObjectStore,
                 cpu_workers: int = 2, exhibit_budget_bytes: int | None = None, update_analysis: bool = True):
        self.stages = toposort(stages)
        self.sessions = sessions
        self.store = store
        self.cpu_workers = max(1, cpu_workers)
        self.exhibit_budget_bytes = exhibit_budget_bytes
        self.update_analysis = update_analysis

    # --- bookkeeping -----------------------------------------------------------------------------------------

    def _row(self, analysis_id: str, stage: str, **values: Any) -> None:
        with self.sessions() as db:
            self._set_row(db, analysis_id, stage, **values)
            db.commit()

    def _original_sha256(self, analysis_id: str) -> str | None:
        """The hash of the analysed file, by which the store holds its original."""
        with self.sessions() as db:
            return db.scalar(select(Analysis.sha256).where(Analysis.id == analysis_id))

    @staticmethod
    def _set_row(db: Session, analysis_id: str, stage: str, **values: Any) -> None:
        row = db.get(AnalysisStage, (analysis_id, stage))
        if row is None:
            row = AnalysisStage(analysis_id=analysis_id, stage=stage, status=values.get("status", "pending"), attempt=0)
            db.add(row)
        if values.pop("new_attempt", False):
            row.attempt = (row.attempt or 0) + 1
            row.error, row.finished_at, row.output_artefact_id = None, None, None
        for key, value in values.items():
            setattr(row, key, value)

    @staticmethod
    def _index_listed(db: Session, analysis_id: str, output: Any) -> None:
        """N-18a: name the objects a stage wrote by rows, in the transaction that marks it done (the ids are deterministic,
        so `render` finds them indexed). A stage that ran again may have written other bytes under an id (a model is not
        bit-exact between runs): the row then follows the stage's output while the analysis is not published; a published
        result's evidence is immutable."""
        for a in listed_objects(output):
            existing = db.get(Artefact, a["id"])
            if existing is not None and existing.sha256 != a["sha256"]:
                analysis = db.get(Analysis, analysis_id)
                if existing.analysis_id != analysis_id or (analysis is not None and analysis.result_hash):
                    raise EvidenceConflict(f"artefact {a['id']} already indexes different bytes")
                existing.sha256, existing.bytes, existing.media_type = a["sha256"], a["size_bytes"], a["media_type"]
                continue
            index_object(db, a["sha256"], a["size_bytes"], media_type=a["media_type"], role=a["role"],
                         created_by=a["created_by"], analysis_id=analysis_id, artefact_id=a["id"])

    def _analysis(self, analysis_id: str, **values: Any) -> None:
        """Progress bookkeeping. A terminal projection committed by recovery or cancellation is never rewritten."""
        if not self.update_analysis:
            return
        with self.sessions() as db:
            db.execute(update(Analysis).where(Analysis.id == analysis_id, Analysis.status.in_(("queued", "running")))
                       .values(**values).execution_options(synchronize_session=False))
            db.commit()

    def _start(self, analysis_id: str) -> str | None:
        """Mark the analysis running unless a cancellation committed first; returns what blocked the start.

        Serializes with a cancellation request on the analysis row: the request locks it while flagging the job, so a
        start that wins the lock is seen by the worker's heartbeat and one that loses sees the flag here.
        """
        if not self.update_analysis:
            return None
        with self.sessions() as db:
            conn = db.connection()
            if conn.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            analysis = db.scalar(select(Analysis).where(Analysis.id == analysis_id).with_for_update())
            if analysis is None:
                return None
            if analysis.status == "cancelled":
                return "cancelled"
            if db.scalar(select(Job.id).where(Job.subject_id == analysis_id, Job.type.in_(("analysis", "refusion")),
                                              Job.status.in_(("queued", "leased")), Job.cancel_requested.is_(True)).limit(1)):
                return "cancel_requested"
            analysis.status, analysis.started_at, analysis.progress, analysis.error = "running", utcnow(), 0.0, None
            db.commit()
        return None

    def _reuse(self, analysis_id: str, stage: StageDef) -> Any:
        if not stage.persist:
            return _MISS
        with self.sessions() as db:
            row = db.get(AnalysisStage, (analysis_id, stage.id))
            if row is None or row.status != "done" or row.output_artefact_id is None:
                return _MISS
            artefact = db.get(Artefact, row.output_artefact_id)
        if artefact is None:
            return _MISS
        try:
            if self.store.verify(artefact.sha256) != artefact.bytes:
                raise ValueError("the stored size does not match the artefact index")
            data = json.loads(self.store.read_bytes(artefact.sha256))
            if stage.reuse_if is not None and not stage.reuse_if(data):
                log.info("runner.stage_output_incompatible", analysis_id=analysis_id, stage=stage.id)
                return _MISS
        except ObjectIntegrityError as e:
            log.error("runner.stage_output_corrupt", analysis_id=analysis_id, stage=stage.id, error=str(e))
            self.store.quarantine(artefact.sha256)
            return _MISS
        except (ObjectNotFound, ValueError) as e:
            log.warning("runner.stage_output_unusable", analysis_id=analysis_id, stage=stage.id, error=str(e))
            return _MISS
        # N-18a: the stage is reused only with every object it names. One that is gone (`store gc` or a purge while the run
        # waited, before a build that indexed them with the stage; a removal by hand) or damaged is written again by the stage
        for a in listed_objects(data):
            try:
                if self.store.verify(a["sha256"]) != a["size_bytes"]:
                    raise ValueError("the stored size does not match the stage's record")
            except ObjectIntegrityError as e:
                log.error("runner.stage_object_corrupt", analysis_id=analysis_id, stage=stage.id, sha256=a["sha256"], error=str(e))
                self.store.quarantine(a["sha256"])
                return _MISS
            except (ObjectNotFound, ValueError) as e:
                log.warning("runner.stage_object_missing", analysis_id=analysis_id, stage=stage.id, sha256=a["sha256"],
                            artefact_id=a.get("id"), error=str(e)[:200])
                return _MISS
        return data

    # --- run -------------------------------------------------------------------------------------------------

    def run(self, analysis_id: str, *, profile: Profile | str = Profile.FULL, deadline: Deadline | None = None,
            state: dict[str, Any] | None = None) -> RunOutcome:
        profile = Profile(profile)
        deadline = deadline or Deadline(None)
        state = state if state is not None else {}
        writer = ExhibitWriter(analysis_id, self.store, budget_bytes=self.exhibit_budget_bytes)
        state.setdefault("exhibits", writer)
        bound = log.bind(analysis_id=analysis_id)
        outputs: dict[str, Any] = {}
        status: dict[str, str] = {s.id: "pending" for s in self.stages}
        invalidated: set[str] = set()  # stages whose output in this run differs from, or replaces, a stored one
        total = sum(s.weight for s in self.stages) or 1.0
        finished_weight = 0.0
        abort: str | None = None
        cancelled = False
        running: dict[Future, tuple[StageDef, float]] = {}
        if (blocked := self._start(analysis_id)) is not None:
            # a committed cancellation keeps its finish time and event; a pending one is acknowledged before any stage runs
            if blocked == "cancel_requested":
                finish_analysis(self.sessions, analysis_id, "cancelled", {"reason": "Cancelled on request"})
            bound.info("runner.not_started", reason=blocked)
            return RunOutcome("cancelled", dict(status), outputs, "Cancelled on request")
        pools = {"accelerator": ThreadPoolExecutor(1, thread_name_prefix="veritas-lane-acc"),
                 "cpu": ThreadPoolExecutor(self.cpu_workers, thread_name_prefix="veritas-lane-cpu")}

        def progress(current: str | None) -> None:
            fraction = round(min(1.0, finished_weight / total), 4)
            self._analysis(analysis_id, progress=fraction, current_stage=current)
            emit(self.sessions, analysis_id, "progress", {"fraction": fraction, "current_stage": current, "eta_s": None})

        def finish(stage: StageDef, final: str, *, duration_ms: float = 0.0, reason: str | None = None) -> None:
            nonlocal finished_weight
            status[stage.id] = final
            finished_weight += stage.weight
            if final in ("failed", "cancelled") or (final == "done" and stage.persist):
                invalidated.add(stage.id)
            payload: dict[str, Any] = {"stage": stage.id, "status": final, "duration_ms": round(duration_ms, 1)}
            if reason:
                payload["reason"] = reason
            emit(self.sessions, analysis_id, "stage_finished", payload)
            progress(stage.id)

        try:
            while True:
                scheduled = False
                for stage in self.stages:
                    if status[stage.id] != "pending" or abort or cancelled:
                        continue
                    if any(status[d] in ("pending", "running") for d in stage.after):
                        continue
                    if deadline.lease_lost:  # N-8: nothing more is scheduled or written, not even a skipped row
                        cancelled = True
                        break
                    if profile not in stage.profiles:
                        self._row(analysis_id, stage.id, status="skipped", error="not in this analysis profile")
                        finish(stage, "skipped", reason="not in this analysis profile")
                        scheduled = True
                        continue
                    try:
                        deadline.check(f"before {stage.id}")
                    except Cancelled:
                        cancelled = True
                        break
                    except DetectorTimeout:
                        abort = TIMEOUT_MESSAGE
                        break
                    stale = any(d in invalidated for d in stage.after)
                    reused = _MISS if stale else self._reuse(analysis_id, stage)
                    if reused is not _MISS:
                        outputs[stage.id] = reused
                        finish(stage, "reused")
                        scheduled = True
                        continue
                    status[stage.id] = "running"
                    self._row(analysis_id, stage.id, status="running", started_at=utcnow(), new_attempt=True)
                    emit(self.sessions, analysis_id, "stage_started", {"stage": stage.id, "lane": stage.lane})
                    ctx = StageContext(analysis_id=analysis_id, stage_id=stage.id, profile=profile, deadline=deadline,
                                       store=self.store, sessions=self.sessions, outputs=dict(outputs), state=state,
                                       exhibits=state["exhibits"].bind(stage.id), log=bound.bind(stage=stage.id))
                    pool = pools["cpu"] if stage.lane == "cpu" else pools["accelerator"]
                    running[pool.submit(stage.fn, ctx)] = (stage, time.perf_counter())
                    scheduled = True
                if not running:
                    if scheduled and not (abort or cancelled):
                        continue  # a skip or reuse may have unblocked further stages
                    break
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for future in done:
                    stage, t0 = running.pop(future)
                    duration_ms = (time.perf_counter() - t0) * 1000
                    try:
                        data = future.result()
                        if deadline.lease_lost:
                            # N-8: a stage that returned after the lease was lost is not stored. Its detectors were stopped
                            # as on a cancel, so its output is not what an undisturbed run gives, and the new owner would reuse it
                            raise Cancelled("The job lease was lost while the stage ran")
                        if stage.persist:
                            body = canonical_json(data)
                            data = json.loads(body)
                            # N-18a: the stage's output, the rows that name the objects it wrote and the stage's `done` are
                            # one transaction, so a finished stage never has an object that no row names
                            with self.sessions() as db:
                                artefact_id = register_artefact(db, self.store, body, media_type="application/json",
                                                                role="stage_output", created_by=stage.id,
                                                                analysis_id=analysis_id).id
                                self._index_listed(db, analysis_id, data)
                                self._set_row(db, analysis_id, stage.id, status="done", finished_at=utcnow(),
                                              output_artefact_id=artefact_id)
                                db.commit()
                        else:
                            self._row(analysis_id, stage.id, status="done", finished_at=utcnow(), output_artefact_id=None)
                        outputs[stage.id] = data
                        finish(stage, "done", duration_ms=duration_ms)
                    except (Cancelled, JobCancelled):
                        cancelled = True
                        if deadline.lease_lost:  # K-3: the stage row stays for the new lease owner
                            status[stage.id] = "lease_lost"
                        else:
                            self._row(analysis_id, stage.id, status="cancelled", finished_at=utcnow(), error="cancelled")
                            finish(stage, "cancelled", duration_ms=duration_ms)
                    except (DetectorTimeout, JobTimeout) as e:
                        if deadline.lease_lost:  # N-8: whatever ended the stage, this run records nothing about it
                            cancelled, status[stage.id] = True, "lease_lost"
                            continue
                        # N-14b: a step that reached a limit of its own (a document's reading) says which
                        abort = friendly_error(e) if isinstance(e, StepTimeout) else TIMEOUT_MESSAGE
                        self._row(analysis_id, stage.id, status="failed", finished_at=utcnow(), error=abort)
                        finish(stage, "failed", duration_ms=duration_ms, reason=abort)
                    except Exception as e:
                        if deadline.lease_lost:
                            bound.warning("runner.stage_error_after_lease_loss", stage=stage.id, error=f"{type(e).__name__}: {e}"[:300])
                            cancelled, status[stage.id] = True, "lease_lost"
                            continue
                        reason = friendly_error(e)
                        bound.error("runner.stage_failed", stage=stage.id, error=f"{type(e).__name__}: {e}"[:300],
                                    tb=traceback.format_exc()[-1200:])
                        # N-18b (operations re-check O4-01): a stage that could not read an object of the evidence store
                        # fails the analysis, critical or not, with N-18a's sentence. A release may put the object back
                        # before `publish` asks for it (`common.missing_objects`), and the result was then published
                        # `done` without what this stage gives. The run ends here: `publish` is never reached
                        gone = missing_object(e)
                        if gone is not None:
                            writer = writer_of(gone, outputs)
                            # N-19f (R4-04): the original has a sentence of its own; every other object keeps N-18a's
                            original = writer is None and is_original(gone, self._original_sha256(analysis_id))
                            reason = ORIGINAL_MISSING if original else evidence_missing({writer: 1} if writer else {}, reader=stage.id)
                            bound.error("runner.stage_object_missing_in_run", stage=stage.id, written_by=writer, original=original,
                                        error=str(gone)[:200])
                        self._row(analysis_id, stage.id, status="failed", finished_at=utcnow(), error=reason)
                        finish(stage, "failed", duration_ms=duration_ms, reason=reason)
                        if stage.critical or gone is not None:
                            abort = reason
                        else:
                            # N-18b: a stage that is not critical failed for another reason and the run goes on. The stages
                            # that build the result read this (`common.failed_steps`) and say what the result lacks
                            state.setdefault(STAGE_FAILURES, {})[stage.id] = reason
        finally:
            for pool in pools.values():
                pool.shutdown(wait=True)

        if deadline.lease_lost:
            # K-3: the lease was lost (not a cancel request). Another worker holds the job; recovery or that owner decides the
            # outcome, so this run writes no skipped stages and no terminal projection, whether it stopped as cancelled, with
            # a failed stage or with every stage finished (N-8)
            bound.warning("runner.lease_lost", stages=dict(status))
            return RunOutcome("lease_lost", dict(status), outputs, "The job lease was lost; its new owner decides the outcome")
        why = "cancelled" if cancelled else ("upstream failure" if abort else "dependency did not finish")
        for stage in self.stages:
            if status[stage.id] == "pending":
                self._row(analysis_id, stage.id, status="skipped", error=f"not run: {why}")
                finish(stage, "skipped", reason=f"not run: {why}")

        # Recovery after a lease expires and a late runner share one terminal projection. A committed result wins.
        final = "cancelled" if cancelled else "failed" if abort else "done"
        error = "Cancelled on request" if cancelled else abort
        payload = {"reason": error} if error else state.get("done_payload", {})
        if self.update_analysis:
            final = finish_analysis(self.sessions, analysis_id, final, payload)
        else:
            emit(self.sessions, analysis_id, final, payload)
        outcome = RunOutcome(final, dict(status), outputs, None if final == "done" else error)
        bound.info("runner.finished", status=outcome.status, stages=outcome.stages)
        return outcome
