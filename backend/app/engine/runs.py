"""The detector runs of an analysis that count (BUILD_SPEC N-8; code review 6).

`detector_runs` holds one row per run, and nothing retires the rows of an attempt that was interrupted (a worker restart while
`detect` ran): the requeued job runs the stage again and adds a second row per detector. Coverage ("k of n checks"), the run
list, the report snapshot and a re-fusion's copies therefore read `current`, which gives one run per detector:

- the run a stored stage output references (`detectors[].run_id`), which is what the published result was built from;
- else the latest run of the detector by start time (an analysis that is still running, or a stage whose output is not stored).

The rows of an interrupted attempt stay in the table: they are a record of what ran, and are not counted.

A detector whose run could not be stored has no row (N-10a; code re-verification R2-03): the stage records it as failed and
the published result names it in its manifest and its coverage finding. `statuses` counts it as failed, so the coverage of
the API, of the page and of a report snapshot is that of the result.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import times
from app.engine.detector import SERVER_REASONS, of_the_server
from app.evidence.audio_long_review_binding import DETECTOR_ID as LONG_VOICE_REVIEW_ID
from app.storage.db import AnalysisStage, Artefact, DetectorRun
from app.storage.objects import ObjectStore, default_store


def _started(run: DetectorRun) -> tuple[int, str]:
    """Order of the times, not of their text or of the engine's NULL placement (no time sorts first); the id breaks a tie."""
    return times.micros(run.started_at), run.id


def referenced(db: Session, analysis_id: str, store: ObjectStore | None = None) -> set[str]:
    """Run ids that the stored stage outputs of an analysis reference. A stage output that cannot be read references none."""
    store = store or default_store()
    out: set[str] = set()
    rows = db.execute(select(AnalysisStage.stage, Artefact.sha256)
                      .join(AnalysisStage, AnalysisStage.output_artefact_id == Artefact.id)
                      .where(AnalysisStage.analysis_id == analysis_id, AnalysisStage.status == "done"))
    for stage, sha256 in rows:
        try:
            output = json.loads(store.read_bytes(sha256))
        except (OSError, LookupError, ValueError):
            continue
        if isinstance(output, dict):
            out.update(str(e["run_id"]) for e in output.get("detectors") or [] if isinstance(e, dict) and e.get("run_id"))
            if stage == "detect" and isinstance(review := output.get("long_voice_review"), dict) and review.get("run_id"):
                out.add(str(review["run_id"]))
    return out


def select_current(rows: Sequence[DetectorRun], referenced_ids: set[str]) -> list[DetectorRun]:
    """One run per detector, in start order: a referenced run before an unreferenced one, the later of two equals."""
    ordered = sorted(rows, key=_started)
    chosen: dict[str, DetectorRun] = {}
    for run in ordered:
        held = chosen.get(run.detector_id)
        if held is None or (run.id in referenced_ids) >= (held.id in referenced_ids):
            chosen[run.detector_id] = run
    return [run for run in ordered if chosen[run.detector_id] is run]


def current(db: Session, analysis_id: str, store: ObjectStore | None = None, *, include_auxiliary: bool = False) -> list[DetectorRun]:
    """Return public scoring runs in start order, optionally including a pinned supplemental receipt for re-fusion."""
    query = select(DetectorRun).where(DetectorRun.analysis_id == analysis_id)
    if not include_auxiliary:
        query = query.where(DetectorRun.detector_id != LONG_VOICE_REVIEW_ID)
    rows = list(db.scalars(query))
    has_auxiliary = any(run.detector_id == LONG_VOICE_REVIEW_ID for run in rows)
    if not has_auxiliary and len({r.detector_id for r in rows}) == len(rows):
        return sorted(rows, key=_started)
    pins = referenced(db, analysis_id, store)
    return select_current([run for run in rows if run.detector_id != LONG_VOICE_REVIEW_ID or run.id in pins], pins)


def statuses(db: Session, analysis_id: str, result: Mapping[str, Any] | None = None, store: ObjectStore | None = None) -> list[str]:
    """One status per detector, for coverage: that of its current run, and `failed` for a detector that the published v2
    result names in its manifest and that has no run. Only a run that could not be stored leaves such a detector
    (`boundary.store_run`), and the result holds it as failed."""
    rows = current(db, analysis_id, store)
    have = {run.detector_id for run in rows}
    named = [str(d["id"]) for d in ((result or {}).get("manifest") or {}).get("detectors") or [] if isinstance(d, dict) and d.get("id")]
    return [run.status for run in rows] + ["failed" for detector_id in dict.fromkeys(named) if detector_id not in have]


def not_run_on_server(db: Session, analysis_id: str, store: ObjectStore | None = None) -> dict[str, int]:
    """How many detectors' runs that count were skipped for a reason of the server, by reason (`detector.SERVER_REASONS`;
    N-14a). Empty when none was: a part of the coverage of an analysis, beside `statuses`."""
    counts = Counter(run.reason for run in current(db, analysis_id, store) if of_the_server(run.status, run.reason))
    return {reason: counts[reason] for reason in SERVER_REASONS if counts[reason]}
