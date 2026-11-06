"""Hosted checkout endpoints. Only a signed Stripe webhook may release an analysis."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Body, Depends, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool
from starlette.concurrency import run_in_threadpool

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2 import NO_STORE, ProblemError
from app.billing import service
from app.cases import intake

public_router = APIRouter(prefix="/api/v2/billing", tags=["billing"])
router = APIRouter(prefix="/api/v2/billing", tags=["billing"], dependencies=[Depends(rate_limit), Depends(authorize)],
                   route_class=CheckedRoute)


def _error(error: service.BillingError) -> JSONResponse:
    content = {"type": "about:blank", "title": "Checkout unavailable" if error.status == 503 else "Checkout request refused",
               "status": error.status, "detail": error.detail}
    if error.order_id:
        content["order_id"] = error.order_id
    return JSONResponse(content, status_code=error.status, media_type="application/problem+json", headers=NO_STORE)


@public_router.get("/config")
def configuration():
    return JSONResponse(service.configuration(), headers=NO_STORE)


@router.post("/checkout", status_code=201)
def create_checkout(request: Request, file: UploadFile = File(...),
                    kind: Literal["image", "video", "audio", "document"] = Form(...),
                    profile: Literal["quick", "full", "deep"] = Form("full"),
                    client_sha256: str | None = Form(None, pattern=r"^[0-9A-Fa-f]{64}$"),
                    risk_accepted: bool = Form(False), risk_notice_version: str | None = Form(None, max_length=80),
                    risk_notice_sha256: str | None = Form(None, max_length=64),
                    idempotency_key: str = Header(..., min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")):
    try:
        service.require_owner_principal(request.state.principal)
        order = service.create_order(file.file, file.filename or f"upload.{kind}", kind, owner_id=request.state.principal.id,
                                     request_key=idempotency_key, profile=profile, client_sha256=client_sha256,
                                     risk_accepted=risk_accepted, risk_notice_version=risk_notice_version,
                                     risk_notice_sha256=risk_notice_sha256)
        return JSONResponse(service.checkout(order.id, request.state.principal.id), status_code=201, headers=NO_STORE)
    except service.BillingError as error:
        return _error(error)
    except intake.IntakeError as error:
        raise ProblemError(error.status, error.title, error.detail) from error


class PurchaseAcceptance(BaseModel):
    risk_accepted: StrictBool = False
    risk_notice_version: str | None = Field(None, max_length=80)
    risk_notice_sha256: str | None = Field(None, max_length=64)


@router.post("/orders/{order_id}/checkout")
def retry_checkout(order_id: str, request: Request, acceptance: PurchaseAcceptance | None = Body(None)):
    try:
        service.require_owner_principal(request.state.principal)
        service.acknowledge_order(order_id, request.state.principal.id,
                                  accepted=acceptance.risk_accepted if acceptance else False,
                                  version=acceptance.risk_notice_version if acceptance else None,
                                  sha256=acceptance.risk_notice_sha256 if acceptance else None)
        return JSONResponse(service.checkout(order_id, request.state.principal.id), headers=NO_STORE)
    except service.BillingError as error:
        return _error(error)


@router.get("/orders/{order_id}")
def read_order(order_id: str, request: Request):
    try:
        service.require_owner_principal(request.state.principal)
        return JSONResponse(service.order_status(order_id, request.state.principal.id), headers=NO_STORE)
    except service.BillingError as error:
        return _error(error)


@public_router.post("/webhook")
async def webhook(request: Request):
    # Raw bytes are required: decoding and re-serialising JSON would invalidate Stripe's signature.
    try:
        event = service.verify_event(await request.body(), request.headers.get("stripe-signature", ""))
        result = await run_in_threadpool(service.process_event, event)
        return JSONResponse(result, headers=NO_STORE)
    except service.BillingError as error:
        return _error(error)
