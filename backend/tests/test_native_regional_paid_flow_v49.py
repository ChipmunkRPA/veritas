"""Real local detectors/queue/report pipeline; isolated DB and simulated Stripe only."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from test_stripe_billing import billing as billing  # shared infrastructure fixture, no detector doubles
from test_stripe_billing import counts, event_for, signed, upload

from app.api import jobs
from app.core.config import settings
from app.storage.db import Analysis, BillingOrderRow, DetectorRun

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "backend/data/independent-evaluation/native-regional-paid-flow-v49"
CASES = ("candidate-small", "legacy-ai", "legacy-original", "candidate-ai", "candidate-original")
FORBIDDEN = re.compile(r"RelayFormer|TIDAL|local\.native_regional_v1|loc\.native_regional|tidal\.splice_v30_1", re.I)


def wait_for(fn, predicate, label, seconds=600):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        value = fn()
        if predicate(value):
            return value
        time.sleep(.1)
    raise AssertionError(f"Timed out waiting for {label}")


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES)
def test_real_prepaid_image_flow(billing, monkeypatch, case):
    import fitz
    import torch

    from app.core import device
    from app.detectors.image import regional, splice
    from app.engine import models, pipelines
    from app.engine.customer_assessment import customer_assessment
    from app.engine.registry import get_registry, reset_registry
    from app.reports import image_evidence, snapshot
    from app.storage import db as storage_db
    from scripts.native_regional_paid_flow_v49 import admission, sha

    frozen = admission()
    assert os.environ.get("CHIPMUNK_REGIONAL_FLOW_CASE") == case
    assert str(device.DEVICE) == "cpu"
    f = billing
    number = CASES.index(case)
    folder = TRIAL / f"phase-{number:03d}"
    mode, kind = case.split("-", 1)
    item = json.loads((TRIAL / "inputs.json").read_text())[kind]
    source = TRIAL / item["file"]
    assert sha(source) == item["sha256"]
    for key, value in {"device": "cpu", "image_region_mode": mode, "model_memory_budget_mb": "4096",
                       "worker_threads": 1, "job_timeout_s": 600., "run_worker": False,
                       "offline": True, "webhooks_enabled": False}.items():
        monkeypatch.setattr(settings, key, value)
    # All components, including functions imported before this fixture, resolve
    # the isolated DB. Never initialize or modify the user's running database.
    monkeypatch.setattr(storage_db, "_engine", f.engine)
    monkeypatch.setattr(storage_db, "_SessionLocal", sessionmaker(f.engine, expire_on_commit=False))
    monkeypatch.setattr(jobs, "get_session", f.sessions)
    monkeypatch.setattr(pipelines, "get_session", f.sessions)
    monkeypatch.setattr(pipelines, "default_store", lambda: f.store)
    reset_registry()
    models.reset_model_manager()
    torch.set_num_threads(2)  # owned offline test-worker startup, same in both arms
    registry = get_registry()
    expected = set(frozen["enabled_detectors"][mode])
    assert set(registry.enabled("image", "full")) == expected
    assert registry._override is None
    with f.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        connection.commit()

    observed = {"regional": 0, "splice": 0, "weight_loads": 0}
    active = {"analysis_id": None}

    def require_paid():
        assert active["analysis_id"] is not None
        with f.sessions() as db:
            order = db.query(BillingOrderRow).filter_by(analysis_id=active["analysis_id"]).one()
            assert order.status == "fulfilled" and order.paid_at is not None
            assert order.amount_minor == 999 and order.currency == "usd" and order.mode == "test"

    original_native, original_splice = regional.RegionalModel.predict, splice.predict

    @functools.wraps(original_native)
    def native_call(self, image):
        require_paid()
        observed["regional"] += 1
        return original_native(self, image)

    @functools.wraps(original_splice)
    def splice_call(*args, **kwargs):
        require_paid()
        observed["splice"] += 1
        return original_splice(*args, **kwargs)

    original_import = models._import_callable

    def import_loader(path):
        original = original_import(path)

        @functools.wraps(original)
        def loaded(*args, **kwargs):
            require_paid()
            observed["weight_loads"] += 1
            return original(*args, **kwargs)
        return loaded

    monkeypatch.setattr(regional.RegionalModel, "predict", native_call)
    monkeypatch.setattr(splice, "predict", splice_call)
    monkeypatch.setattr(models, "_import_callable", import_loader)

    # Use every production HTTP router and the built application UI. The DB,
    # storage and Stripe transport are isolated; detector/pipeline math is real.
    from app.main import app

    f.client.close()
    f.client = TestClient(app)
    worker = jobs.build_worker()
    worker.start()
    try:
        response = upload(f, content=source.read_bytes(), filename=f"regional-{kind}.png")
        assert response.status_code == 201, response.text
        oid = response.json()["order_id"]
        assert counts(f) == (0, 0, 0) and not any(observed.values())
        assert signed(f, event_for(f, oid, paid=False)).status_code == 200
        assert counts(f) == (0, 0, 0) and not any(observed.values())
        event = event_for(f, oid)
        # The paid webhook queues work. Set the expected entitlement identity
        # before the worker wakes; the provider session already binds this order.
        worker.stop(timeout=30)
        assert signed(f, event).status_code == 200
        with f.sessions() as db:
            active["analysis_id"] = db.get(BillingOrderRow, oid).analysis_id
        aid = active["analysis_id"]
        worker = jobs.build_worker()
        worker.start()

        def analysis_state():
            with f.sessions() as db:
                a = db.get(Analysis, aid)
                return {"status": a.status, "error": a.error, "hash": a.result_hash}

        state = wait_for(analysis_state, lambda a: a["status"] in {"done", "failed", "cancelled"}, "analysis")
        assert state["status"] == "done", state
        detail = f.client.get(f"/api/v2/analyses/{aid}", headers=f.headers)
        assert detail.status_code == 200, detail.text
        exported = f.client.get(f"/api/analyses/{aid}/export.json", headers=f.headers)
        assert exported.status_code == 200, exported.text
        evidence = f.client.get(f"/api/v2/analyses/{aid}/evidence-packets", headers=f.headers)
        assert evidence.status_code == 200, evidence.text
        assert exported.json()["image_evidence"] == evidence.json()
        for response in (detail, exported, evidence):
            assert FORBIDDEN.search(response.text) is None
        document = evidence.json()
        assert document["status"] == "ready"
        with f.sessions() as db:
            analysis = db.get(Analysis, aid)
            result = json.loads(f.store.read_bytes(analysis.result_hash))
            checks = {r.detector_id: {"status": r.status, "reason": r.reason} for r in
                      db.query(DetectorRun).filter_by(analysis_id=aid)}
        assert set(checks) == expected
        if mode == "candidate":
            assert checks["loc.native_regional"]["status"] == ("not_applicable" if kind == "small" else "ok")
        assert observed["regional"] == int(mode == "candidate" and kind != "small")
        assert observed["splice"] == int(kind != "small")
        assessment = customer_assessment({**result, "kind": "image", "status": "done"})
        if kind == "small":
            assert assessment["direction"] not in {"ai", "original"}
        if mode == "candidate" and kind == "ai":
            assert any(p.get("evidence_type") == "localisation" for p in document["packets"])
        # Every declared crop, overlay and component mask is served from the same
        # immutable evidence set, not an unrelated raw-object URL.
        assets = []
        for packet in document["packets"]:
            for role, asset in packet["artifacts"].items():
                response = f.client.get(asset["url"], headers=f.headers)
                assert response.status_code == 200
                assert hashlib.sha256(response.content).hexdigest() == asset["sha256"]
                assets.append({"packet": packet["id"], "role": role, "sha256": asset["sha256"]})

        request = f.client.post(f"/api/v2/analyses/{aid}/reports", headers=f.headers, json={"profile": "standard"})
        assert request.status_code == 202, request.text
        report_id = request.json()["id"]

        def report_state():
            response = f.client.get(f"/api/v2/reports/{report_id}", headers=f.headers)
            assert response.status_code == 200
            return response.json()

        report = wait_for(report_state, lambda r: r["status"] in {"done", "failed"}, "PDF report")
        assert report["status"] == "done", report
        response = f.client.get(f"/api/v2/reports/{report_id}/file", headers=f.headers)
        assert response.status_code == 200
        pdf = response.content
        folder.joinpath("report.pdf").write_bytes(pdf)
        with fitz.open(stream=pdf, filetype="pdf") as parsed:
            text = "\n".join(p.get_text() for p in parsed)
            assert FORBIDDEN.search(text) is None
            assert re.search(r"examiner.s conclusion|not signed off|unauthenticated instance|local examiner", text, re.I) is None
            pages = parsed.page_count
        with f.sessions() as db:
            from app.storage.db import ReportRow

            report_row = db.get(ReportRow, report_id)
            frozen_report = snapshot.load(f.store.read_bytes(report_row.snapshot_hash))
        assert frozen_report.image_evidence == document
        manifest = image_evidence.manifest(frozen_report)
        assert all(a in manifest["artifacts"] for a in
                   [{"packet_id": a["packet"], "role": a["role"], "sha256": a["sha256"]} for a in assets])
        folder.joinpath("detail.json").write_bytes(detail.content)
        folder.joinpath("export.json").write_bytes(exported.content)
        folder.joinpath("evidence.json").write_bytes(evidence.content)
        folder.joinpath("report.txt").write_text(text)

        if case == "candidate-ai":
            import uvicorn

            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8164, lifespan="off", log_config=None,
                                                 access_log=False))
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                wait_for(lambda: server.started, bool, "isolated browser server", 20)
                credentials = folder / "browser-input.json"
                credentials.write_text(json.dumps({"analysis_id": aid, "headers": f.headers}))
                os.chmod(credentials, 0o600)
                process = subprocess.run(["node", str(ROOT / "frontend/e2e/native-regional-paid-flow-v49.mjs"),
                                          str(folder)], check=False, timeout=90, capture_output=True, text=True)
                (folder / "browser.log").write_text(process.stdout + process.stderr)
                assert process.returncode == 0, "Browser verification failed; private log retained"
            finally:
                server.should_exit = True
                thread.join(timeout=15)
        result_summary = {"case": case, "status": "complete", "mode": mode, "input_sha256": item["sha256"],
                          "all_enabled_checks_recorded": True, "checks": checks, "native_and_splice_calls": observed,
                          "payment_before_weight_load_and_inference": True, "simulated_stripe_only": True,
                          "assessment": {k: assessment[k] for k in ("direction", "headline", "confidence_pct", "probability_ai_pct", "interval_pct")},
                          "decision": {k: result["decision"].get(k) for k in ("code", "probability_ai_calibrated", "probability_ai_uncalibrated", "ci90")},
                          "packet_count": len(document["packets"]), "assets_verified": len(assets), "pdf_pages": pages,
                          "source_crop_mask_report_export_equality": True, "model_names_hidden": True,
                          "browser_checked": case == "candidate-ai", "new_accuracy_claim": False}
        (folder / "completed.json").write_text(json.dumps(result_summary, indent=2) + "\n")
    finally:
        worker.stop(timeout=30)
        models.reset_model_manager()
        reset_registry()
