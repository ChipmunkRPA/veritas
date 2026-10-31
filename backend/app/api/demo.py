"""Guided demo: analyse a bundled set of realistic samples and return them in walkthrough order.

POST /api/demo/start reuses an existing completed analysis for each sample (matched by SHA-256 and kind) that the caller can
see and that an earlier demo created (N-19d: its intake record says so), so the demo is instant after the first run and never
walks through an examiner's own analysis of the same bytes; otherwise it takes the sample in as a quick analysis (implicit
case, C-2).
Only analyses with the running pipeline's result schema are reused: the tour anchors on the v2 workspace (E-8), and a v1 row
would render through the legacy views, which are kept for v1 results only.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import and_, exists, or_

from app.api import jobs
from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2_evidence import _stage_output
from app.cases import access, intake
from app.core.config import settings
from app.storage import files
from app.storage.db import Analysis, IntakeRecordRow, get_session

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
router = APIRouter(prefix="/api/demo", deprecated=True, dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)  # v1: through v2.1 (API-03)


def _schema_version() -> int:
    """The result schema the running pipeline writes (`VERITAS_PIPELINE=v1` keeps v0.3 results)."""
    return 2 if settings.pipeline == "v2" else 1


def _opens_in_workspace(db, a: Analysis) -> bool:
    """A finished v2 video row from a build before the probe wrote its frame table (V-1) cannot open the video workspace: the
    tour's lanes steps would find nothing, so the sample is taken in again. Other rows open as they are."""
    if a.status != "done" or a.kind != "video" or a.result_schema_version != 2:
        return True
    return bool((_stage_output(db, a.id, "probe") or {}).get("frame_table"))


def made_by_demo():
    """The analyses whose media arrived in their case through the demo (`intake.DEMO_SUPPLIED_BY`, `DEMO_SOURCE`)."""
    return exists().where(IntakeRecordRow.media_id == Analysis.media_id, IntakeRecordRow.case_id == Analysis.case_id,
                          IntakeRecordRow.supplied_by == intake.DEMO_SUPPLIED_BY, IntakeRecordRow.source == intake.DEMO_SOURCE)


def manifest() -> list[dict]:
    p = DEMO_DIR / "manifest.json"
    if not p.exists():
        raise HTTPException(503, "Demo samples are not installed on this server")
    return json.loads(p.read_text())


@router.get("")
def describe():
    return {"items": [{k: v for k, v in it.items() if k != "file"} for it in manifest()]}


@router.post("/start", status_code=202)
def start(request: Request):
    principal = request.state.principal
    out = []
    for it in manifest():
        path = DEMO_DIR / it["file"]
        if not path.is_file():
            continue
        sha = files.sha256_bytes(path.read_bytes())
        with get_session() as db:
            # A queued or running row runs on the current pipeline; a finished one is reused only with the current result schema.
            current = or_(Analysis.status.in_(["queued", "running"]),
                          and_(Analysis.status == "done", Analysis.result_schema_version == _schema_version()))
            query = db.query(Analysis).filter(Analysis.sha256 == sha, Analysis.kind == it["kind"], current, made_by_demo())
            if (visible := access.visible_analyses(principal)) is not None:
                query = query.filter(visible)
            existing = next((a for a in query.order_by(Analysis.created_at.desc()) if _opens_in_workspace(db, a)), None)
            if existing:
                out.append({**{k: v for k, v in it.items() if k != "file"}, "id": existing.id, "status": existing.status,
                            "decision_code": existing.decision_code})  # L-2: the tour follows the decision
                continue
        try:
            with path.open("rb") as src:
                outcome = intake.take_in(src, it["file"], it["kind"], principal_id=principal.id, analyze=True,
                                         acquisition=intake.Acquisition(method="quick_upload", supplied_by=intake.DEMO_SUPPLIED_BY,
                                                                        source=intake.DEMO_SOURCE),
                                         job_priority=jobs.PRIORITY_INTERACTIVE)  # N-9: queued with the row
        except intake.IntakeError as e:
            raise HTTPException(e.status, e.detail) from e
        assert outcome.analysis is not None
        jobs.submit(outcome.analysis.id)
        out.append({**{k: v for k, v in it.items() if k != "file"}, "id": outcome.analysis.id, "status": "queued"})
    return {"items": out}
