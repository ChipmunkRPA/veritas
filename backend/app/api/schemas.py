from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.api.text import RequestModel


class AnalysisSummary(BaseModel):
    id: str
    kind: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    error: str | None
    probability_ai: float | None
    band: str | None
    has_report: bool


class AnalysisDetail(AnalysisSummary):
    result: dict | None = None


class TextRequest(RequestModel):
    text: str = Field(min_length=1, max_length=200_000)
    title: str = "pasted-text.txt"
    profile: Literal["quick", "full", "deep"] | None = Field(None, description="Analysis profile; the instance default when absent")


class Health(BaseModel):
    """Liveness says that the process answers and nothing else (N-13; SEC-15): the details are `/api/ready`'s, with credentials."""

    status: str
