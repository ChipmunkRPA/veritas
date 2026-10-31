"""Analysis progress events (ARCHITECTURE §3.3): an append-only `analysis_events` table read by polling and SSE.

Partial-results policy: no provisional probability before `fuse`. `emit()` refuses probability-like keys in every
non-terminal event type. Readers can therefore show coverage and stage progress, and cannot anchor on incomplete
evidence.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import times
from app.storage.db import Analysis, AnalysisEvent, utcnow

EVENT_TYPES = frozenset({"stage_started", "stage_finished", "detector_finished", "progress", "warning", "partial_signal",
                         "done", "failed", "cancelled"})
TERMINAL_EVENTS = frozenset({"done", "failed", "cancelled"})
PROBABILITY_KEYS = frozenset({"probability", "probability_ai", "probability_ai_calibrated", "probability_ai_uncalibrated",
                              "calibrated_probability", "raw_score", "raw", "p_ai", "ai_probability", "ci90", "band"})


def probability_keys(payload: Any) -> set[str]:
    """Probability-like keys anywhere in a payload (recursive)."""
    found: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in PROBABILITY_KEYS:
                found.add(key)
            found |= probability_keys(value)
    elif isinstance(payload, list | tuple):
        for value in payload:
            found |= probability_keys(value)
    return found


def emit(sessions: Callable[[], Session], analysis_id: str, type: str, payload: Mapping[str, Any] | None = None) -> int:
    """Append one event and return its sequence number."""
    if type not in EVENT_TYPES:
        raise ValueError(f"unknown event type {type!r}")
    body = dict(payload or {})
    if type not in TERMINAL_EVENTS and (leaked := probability_keys(body)):
        raise ValueError(f"{type} events must not carry provisional probabilities ({', '.join(sorted(leaked))})")
    with sessions() as db:
        event = AnalysisEvent(analysis_id=analysis_id, ts=utcnow(), type=type, payload=body)
        db.add(event)
        db.commit()
        return event.seq


def list_events(db: Session, analysis_id: str, *, after: int = 0, limit: int = 200) -> list[AnalysisEvent]:
    return list(db.scalars(select(AnalysisEvent).where(AnalysisEvent.analysis_id == analysis_id, AnalysisEvent.seq > after)
                           .order_by(AnalysisEvent.seq).limit(limit)))


def event_json(event: AnalysisEvent) -> dict[str, Any]:
    return {"seq": event.seq, "ts": times.iso_z(event.ts), "type": event.type, "payload": event.payload}


def finish_analysis(sessions: Callable[[], Session], analysis_id: str,
                    status: Literal["done", "failed", "cancelled"], payload: Mapping[str, Any] | None = None) -> str:
    """Commit terminal status and its event together, preserving an already published result.

    A lease may expire after publication commits but before the runner finalizes. Recovery and a late runner
    serialize on the analysis row, so published evidence ends done and only one matching terminal event is emitted.
    """
    with sessions() as db:
        conn = db.connection()
        if conn.dialect.name == "sqlite":
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        analysis = db.scalar(select(Analysis).where(Analysis.id == analysis_id).with_for_update())
        if analysis is None:
            return status
        effective = ("done" if analysis.result_hash else
                     analysis.status if analysis.status in TERMINAL_EVENTS else status)
        body = dict(payload or {}) if effective == status else {}
        if effective == "done":
            body.setdefault("decision_code", analysis.decision_code)
        elif not body:
            body = {"reason": analysis.error or ("Cancelled on request" if effective == "cancelled" else None)}
        existing = (db.scalar(select(AnalysisEvent.seq).where(AnalysisEvent.analysis_id == analysis_id,
                                                             AnalysisEvent.type == effective).limit(1))
                    if analysis.status == effective else None)
        if existing is None:
            db.add(AnalysisEvent(analysis_id=analysis_id, ts=utcnow(), type=effective, payload=body))
        finished_at = analysis.finished_at if analysis.status in TERMINAL_EVENTS else None
        analysis.status = effective
        analysis.finished_at = finished_at or utcnow()
        analysis.current_stage = None
        if effective == "done":
            analysis.progress, analysis.error = 1.0, None
        else:
            analysis.error = analysis.error or body.get("reason")
        db.commit()
        return effective
