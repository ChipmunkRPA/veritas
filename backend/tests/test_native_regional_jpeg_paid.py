"""Opt-in real local JPEG detector, authenticated queue, signed PDF, and export."""
from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import subprocess
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import sessionmaker
from test_native_regional_paid_flow_v49 import FORBIDDEN, ROOT, wait_for
from test_stripe_billing import billing as billing
from test_stripe_billing import counts, event_for, signed

from app.api import jobs
from app.billing import purchase_notice
from app.core.config import settings
from app.detectors.image import regional, regional_support
from app.engine import models, pipelines
from app.engine.registry import reset_registry
from app.evidence import image_packets_v4
from app.reports import signing, snapshot
from app.storage import db as storage_db
from app.storage.db import Analysis, BillingOrderRow, DetectorRun, ReportRow


@pytest.mark.slow
@pytest.mark.parametrize(("input_var", "proof_var", "expected_source", "expected_mask"), [
    ("CHIPMUNK_REGIONAL_JPEG_FLOW_INPUT", "CHIPMUNK_REGIONAL_JPEG_FLOW_PROOF",
     "35caa98a2cb4a3cba85c5910b3d11b5cfe6fa211728ecd1a160478f9bafb8df3",
     "a0394a1a157a0427db88c388aa9aeab9c4f8534dd139f97fefefef171f9b874c"),
    ("CHIPMUNK_REGIONAL_PROGRESSIVE_FLOW_INPUT", "CHIPMUNK_REGIONAL_PROGRESSIVE_FLOW_PROOF",
     "8bd8de273e668a77aa63e4caacd7a87374a6801930a2cc38124722fad247ad6b",
     "a872832fbaf89e4631b0b10bacb3850eb4cafc7f10b55472be574b901a7a9233"),
], ids=["baseline", "progressive"])
def test_paid_jpeg_uses_same_source_regions_in_browser_pdf_and_json(
    billing, monkeypatch, tmp_path, input_var, proof_var, expected_source, expected_mask,
):
    import fitz
    import torch
    import uvicorn

    path = os.getenv(input_var)
    if not path:
        pytest.skip("Supply the digest-pinned held JPEG from the retained transfer panel")
    source = Path(path).read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    assert source_hash == expected_source
    f = billing
    for key, value in {"environment": "dev", "device": "cpu", "image_region_mode": "legacy",
                       "image_region_jpeg_review": True, "image_region_flip_review": True,
                       "model_memory_budget_mb": "4096", "worker_threads": 1, "job_timeout_s": 600,
                       "run_worker": False, "offline": True, "webhooks_enabled": False}.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(storage_db, "_engine", f.engine)
    monkeypatch.setattr(storage_db, "_SessionLocal", sessionmaker(f.engine, expire_on_commit=False))
    monkeypatch.setattr(jobs, "get_session", f.sessions)
    monkeypatch.setattr(pipelines, "get_session", f.sessions)
    monkeypatch.setattr(pipelines, "default_store", lambda: f.store)
    reset_registry()
    models.reset_model_manager()
    torch.set_num_threads(2)
    with f.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        connection.commit()

    state = {"analysis_id": None, "source_mask": None, "forwards": 0}

    def require_paid():
        assert state["analysis_id"] is not None
        with f.sessions() as db:
            order = db.query(BillingOrderRow).filter_by(analysis_id=state["analysis_id"]).one()
            assert order.status == "fulfilled" and order.paid_at is not None
            assert (order.amount_minor, order.currency, order.mode) == (999, "usd", "test")

    original_import = models._import_callable

    def import_loader(path):
        loader = original_import(path)

        @functools.wraps(loader)
        def checked(*args, **kwargs):
            require_paid()
            return loader(*args, **kwargs)

        return checked

    native_predict = regional.RegionalModel.predict

    @functools.wraps(native_predict)
    def predict(self, image):
        require_paid()
        state["forwards"] += 1
        result = native_predict(self, image)
        selected = regional_support.readout(result.raw_mask, result.candidate_eligibility["eligible"],
                                            regional.PIXEL_THRESHOLD, source, np.asarray(image))
        state["source_mask"] = selected.source_mask(*image.size)
        return result

    monkeypatch.setattr(models, "_import_callable", import_loader)
    monkeypatch.setattr(regional.RegionalModel, "predict", predict)
    from app.main import app

    f.client.close()
    f.client = TestClient(app)
    data = {"kind": "image", "profile": "full", "risk_accepted": "true",
            "risk_notice_version": purchase_notice.VERSION, "risk_notice_sha256": purchase_notice.SHA256}
    response = f.client.post("/api/v2/billing/checkout",
                             headers={**f.headers, "Idempotency-Key": str(uuid.uuid4())}, data=data,
                             files={"file": ("regional-feathered.jpg", source, "image/jpeg")})
    assert response.status_code == 201, response.text
    order_id = response.json()["order_id"]
    assert counts(f) == (0, 0, 0) and not state["forwards"]
    assert signed(f, event_for(f, order_id, paid=False)).status_code == 200
    assert counts(f) == (0, 0, 0) and not state["forwards"]
    assert signed(f, event_for(f, order_id)).status_code == 200
    with f.sessions() as db:
        aid = state["analysis_id"] = db.get(BillingOrderRow, order_id).analysis_id
    worker = jobs.build_worker()
    worker.start()
    try:
        def terminal_analysis():
            with f.sessions() as db:
                analysis = db.get(Analysis, aid)
                return analysis.status, analysis.error, analysis.result_hash

        status, error, result_hash = wait_for(terminal_analysis, lambda row: row[0] in {"done", "failed"}, "JPEG analysis")
        assert status == "done", error
        assert state["forwards"] == 1
        assert hashlib.sha256(state["source_mask"].tobytes()).hexdigest() == expected_mask
        result = json.loads(f.store.read_bytes(result_hash))
        with f.sessions() as db:
            run = db.query(DetectorRun).filter_by(analysis_id=aid, detector_id="loc.native_regional").one()
            assert run.status == "ok", run.reason
        assert f.client.get(f"/api/v2/analyses/{aid}/evidence-packets", headers=f.other_headers).status_code == 404
        api = f.client.get(f"/api/v2/analyses/{aid}", headers=f.headers)
        export = f.client.get(f"/api/analyses/{aid}/export.json", headers=f.headers)
        evidence = f.client.get(f"/api/v2/analyses/{aid}/evidence-packets", headers=f.headers)
        for response in (api, export, evidence):
            assert response.status_code == 200, response.text
            assert FORBIDDEN.search(response.text) is None
        document = evidence.json()
        assert document == export.json()["image_evidence"]
        image_packets_v4.validate_packet_source_mapping(document, result)
        packet = next(p for p in document["packets"]
                      if p.get("localisation", {}).get("resolution_policy", {}).get("source_format") == "JPEG")
        assert packet["anchor"]["original_sha256"] == source_hash
        assert "JPEG compression can obscure" in " ".join(packet["limitations"])
        assert set(packet["artifacts"]) == {"context", "crop", "overlay", "mask"}
        for asset in packet["artifacts"].values():
            response = f.client.get(asset["url"], headers=f.headers)
            assert response.status_code == 200
            assert hashlib.sha256(response.content).hexdigest() == asset["sha256"]
        asset = packet["artifacts"]["mask"]
        response = f.client.get(asset["url"], headers=f.headers)
        with Image.open(io.BytesIO(response.content)) as measured, Image.open(io.BytesIO(source)) as original:
            restored = np.asarray(measured.convert("L").resize(original.size, Image.Resampling.NEAREST)) > 0
        assert np.array_equal(restored, state["source_mask"])

        requested = f.client.post(f"/api/v2/analyses/{aid}/reports", headers=f.headers, json={"profile": "standard"})
        assert requested.status_code == 202, requested.text
        report_id = requested.json()["id"]

        def report_state():
            response = f.client.get(f"/api/v2/reports/{report_id}", headers=f.headers)
            assert response.status_code == 200, response.text
            return response.json()

        report = wait_for(report_state, lambda row: row["status"] in {"done", "failed"}, "signed JPEG PDF")
        assert report["status"] == "done", report
        pdf = f.client.get(f"/api/v2/reports/{report_id}/file", headers=f.headers)
        assert pdf.status_code == 200
        key = f.client.get("/api/v2/instance/public-key", headers=f.headers)
        assert key.status_code == 200
        assert signing.verify(signing.load_public_key(key.json()["public_key_pem"].encode()),
                              pdf.content, report["signature"])
        with fitz.open(stream=pdf.content, filetype="pdf") as rendered:
            pdf_text = " ".join(" ".join(page.get_text().split()) for page in rendered)
        assert "JPEG compression can obscure small or softly blended edits" in pdf_text
        assert FORBIDDEN.search(pdf_text) is None
        with f.sessions() as db:
            frozen = snapshot.load(f.store.read_bytes(db.get(ReportRow, report_id).snapshot_hash))
        assert frozen.image_evidence == document

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8164, lifespan="off",
                                               log_config=None, access_log=False))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            wait_for(lambda: server.started, bool, "isolated browser server", 20)
            credentials = tmp_path / "browser-input.json"
            credentials.write_text(json.dumps({"analysis_id": aid, "headers": f.headers, "packet_id": packet["id"],
                                               **{f"{role}_sha256": packet["artifacts"][role]["sha256"]
                                                  for role in ("context", "crop", "mask")}}))
            credentials.chmod(0o600)
            result = subprocess.run(["node", str(ROOT / "frontend/e2e/native-regional-jpeg-paid.mjs"), str(tmp_path)],
                                    check=False, timeout=90, capture_output=True, text=True)
            (tmp_path / "browser.log").write_text(result.stdout + result.stderr)
            assert result.returncode == 0, "Private browser verification failed"
            assert json.loads((tmp_path / "browser-result.json").read_text())["modelNamesHidden"]
        finally:
            server.should_exit = True
            thread.join(timeout=15)
        proof = os.getenv(proof_var)
        if proof:
            with Path(proof).open("x") as output:
                json.dump({"source_sha256": source_hash, "simulated_stripe_only": True, "price_usd_minor": 999,
                           "payment_before_weight_load_and_inference": True, "regional_forwards": state["forwards"],
                           "original_crop_overlay_mask_hashes_verified": True, "exact_source_mask_replayed": True,
                           "pdf_signature_verified": True, "browser_crop_and_mask_verified": True,
                           "json_and_pdf_snapshot_match": True, "model_names_hidden": True}, output, indent=2)
                output.write("\n")
    finally:
        worker.stop(timeout=30)
        models.reset_model_manager()
        reset_registry()
