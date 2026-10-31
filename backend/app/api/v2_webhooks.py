"""API v2 webhooks (ARCHITECTURE §5.2; PRD API-02; BUILD_SPEC X-1). Administrator only.

- `GET /webhooks`; `POST /webhooks` {url, events, enabled?} → 201 with the secret, shown once; `PATCH /webhooks/{webhook_id}` {url?,
  events?, enabled?, rotate_secret?} (a rotation returns the new secret once); `DELETE /webhooks/{webhook_id}` → 204 with its delivery
  log; `GET /webhooks/{webhook_id}/deliveries`: every attempt, newest first, with its status code or error.
- Each change records `webhook_configured` (action, host, events, enabled; never the secret).
- **Refusals.** 401 / 403; 404 unknown webhook; 409 webhooks are off on this instance (`VERITAS_WEBHOOKS_ENABLED`); 422 invalid body
  or URL.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.text import RequestModel
from app.api.v2 import NO_STORE, ProblemError, problem_responses
from app.audit import chain as audit
from app.core import times
from app.core.config import settings
from app.storage.db import WebhookDeliveryRow, WebhookRow, get_session, utcnow
from app.webhooks import service as webhooks

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)
EventName = Literal["analysis.completed", "analysis.failed", "analysis.signed_off", "batch.completed"]


class WebhookCreate(RequestModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1, max_length=2048)
    events: list[EventName] = Field(min_length=1)
    enabled: bool = True


class WebhookPatch(RequestModel):
    model_config = ConfigDict(extra="forbid")
    url: str | None = Field(default=None, min_length=1, max_length=2048)
    events: list[EventName] | None = Field(default=None, min_length=1)
    enabled: bool | None = None
    rotate_secret: bool = False


class WebhookOut(BaseModel):
    id: str
    url: str
    events: list[EventName]
    enabled: bool
    secret_sha256_prefix: str = Field(description="First 12 hex of SHA-256 over the secret, to tell secrets apart")
    secret: str | None = Field(default=None, description="The signing secret: returned once, at creation and at rotation")
    created_by: str
    created_at: datetime
    updated_at: datetime


class WebhookList(BaseModel):
    deliveries_enabled: bool = Field(description="VERITAS_WEBHOOKS_ENABLED: false means no event is delivered")
    items: list[WebhookOut]


class DeliveryOut(BaseModel):
    id: str
    hook_id: str
    event: EventName
    event_id: str
    attempt: int
    status: Literal["pending", "delivered", "failed", "abandoned"]
    status_code: int | None
    error: str | None
    payload_sha256: str
    created_at: datetime
    attempted_at: datetime | None
    next_attempt_at: datetime | None
    delivered_at: datetime | None
    duration_ms: int | None


class DeliveryList(BaseModel):
    items: list[DeliveryOut]


_utc = times.to_utc  # one form on every engine: a time is printed in UTC (D-72)


def _out(row: WebhookRow, secret: str | None = None) -> WebhookOut:
    return WebhookOut(id=row.id, url=row.url, events=list(row.events), enabled=row.enabled, secret_sha256_prefix=row.secret_hash[:12],  # type: ignore[arg-type]
                      secret=secret, created_by=row.created_by, created_at=_utc(row.created_at), updated_at=_utc(row.updated_at))  # type: ignore[arg-type]


def _json(model: BaseModel, status_code: int = 200) -> JSONResponse:
    return JSONResponse(jsonable_encoder(model), status_code=status_code, headers=NO_STORE)


def _hook(db: Session, webhook_id: str) -> WebhookRow:
    row = db.get(WebhookRow, webhook_id)
    if row is None:
        raise ProblemError(404, "Webhook not found", "No webhook exists with this id.")
    return row


def _url(url: str) -> str:
    if (problem := webhooks.url_problem(url)) is not None:
        raise ProblemError(422, "Invalid request", problem)
    return url


def _configured(request: Request, action: str, row: WebhookRow, **details: object) -> dict:
    """The `webhook_configured` event of a change, for `audit.commit_with` (N-9): where results are sent to changes with
    its event or not at all."""
    return {"action": "webhook_configured", "actor_id": request.state.principal.id, "object_ids": [row.id],
            "details": {"action": action, "webhook_id": row.id, "host": urlsplit(row.url).hostname, "events": list(row.events),
                        "enabled": row.enabled, **details}}


@router.get("/webhooks", response_model=WebhookList, responses=problem_responses(401, 403))
def list_webhooks() -> JSONResponse:
    with get_session() as db:
        rows = db.scalars(select(WebhookRow).order_by(WebhookRow.created_at))
        return _json(WebhookList(deliveries_enabled=settings.webhooks_enabled, items=[_out(r) for r in rows]))


@router.post("/webhooks", status_code=201, response_model=WebhookOut, responses=problem_responses(401, 403, 409, 422))
def create_webhook(req: WebhookCreate, request: Request) -> JSONResponse:
    if not settings.webhooks_enabled:
        raise ProblemError(409, "Webhooks are off", "Webhooks are off on this instance: set VERITAS_WEBHOOKS_ENABLED=true to configure them.")
    now, hook_id, nonce = utcnow(), str(uuid.uuid4()), webhooks.new_nonce()
    secret = webhooks.secret_for(hook_id, nonce)
    url = _url(req.url)

    def unit() -> WebhookOut:
        with audit.writing() as db:
            row = WebhookRow(id=hook_id, url=url, events=sorted(set(req.events)), secret_nonce=nonce,
                             secret_hash=webhooks.secret_hash(secret), enabled=req.enabled,
                             created_by=request.state.principal.id, created_at=now, updated_at=now)
            db.add(row)
            out = _out(row, secret)
            audit.commit_with(db, [_configured(request, "created", row)])
            return out

    return _json(audit.repeated(unit), 201)


@router.patch("/webhooks/{webhook_id}", response_model=WebhookOut, responses=problem_responses(401, 403, 404, 422))
def update_webhook(webhook_id: str, req: WebhookPatch, request: Request) -> JSONResponse:
    def unit() -> WebhookOut:
        with audit.writing() as db:
            row = _hook(db, webhook_id)
            changed = sorted(k for k in ("url", "events", "enabled") if getattr(req, k) is not None)
            if req.url is not None:
                row.url = _url(req.url)
            if req.events is not None:
                row.events = sorted(set(req.events))
            if req.enabled is not None:
                row.enabled = req.enabled
            secret = None
            if req.rotate_secret:
                row.secret_nonce = webhooks.new_nonce()
                secret = webhooks.secret_for(row.id, row.secret_nonce)
                row.secret_hash = webhooks.secret_hash(secret)
            row.updated_at = utcnow()
            out = _out(row, secret)
            audit.commit_with(db, [_configured(request, "updated", row, changed=changed, rotated=req.rotate_secret)])
            return out

    return _json(audit.repeated(unit))


@router.delete("/webhooks/{webhook_id}", status_code=204, response_class=Response, responses=problem_responses(401, 403, 404))
def delete_webhook(webhook_id: str, request: Request) -> Response:
    def unit() -> None:
        with audit.writing() as db:
            row = _hook(db, webhook_id)
            event = _configured(request, "deleted", row)
            db.execute(delete(WebhookDeliveryRow).where(WebhookDeliveryRow.hook_id == webhook_id))
            db.delete(row)
            audit.commit_with(db, [event])

    audit.repeated(unit)
    return Response(status_code=204)


@router.get("/webhooks/{webhook_id}/deliveries", response_model=DeliveryList, responses=problem_responses(401, 403, 404))
def list_deliveries(webhook_id: str) -> JSONResponse:
    with get_session() as db:
        _hook(db, webhook_id)
        rows = db.scalars(select(WebhookDeliveryRow).where(WebhookDeliveryRow.hook_id == webhook_id)
                          .order_by(WebhookDeliveryRow.created_at.desc(), WebhookDeliveryRow.attempt.desc()).limit(500))
        return _json(DeliveryList(items=[DeliveryOut(id=r.id, hook_id=r.hook_id, event=r.event, event_id=r.event_id, attempt=r.attempt,  # type: ignore[arg-type]
                                                     status=r.status, status_code=r.status_code, error=r.error,  # type: ignore[arg-type]
                                                     payload_sha256=r.payload_sha256, created_at=_utc(r.created_at),  # type: ignore[arg-type]
                                                     attempted_at=_utc(r.attempted_at), next_attempt_at=_utc(r.next_attempt_at),
                                                     delivered_at=_utc(r.delivered_at), duration_ms=r.duration_ms) for r in rows]))
