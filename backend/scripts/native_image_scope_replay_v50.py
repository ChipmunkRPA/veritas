"""Replay saved full-flow results through corrected live API, report and browser presentation; zero inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "backend/data/independent-evaluation/native-regional-paid-flow-v49"
TRIAL = ROOT / "backend/data/independent-evaluation/native-image-scope-replay-v50"
PUBLIC = ROOT / "docs/evaluation-panels/native-image-scope-replay-v50-freeze.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def need(value, message):
    if not value:
        raise ValueError(message)


def prepare():
    from scripts.prepare_regional_calibration_v42 import runtime_versions

    need(read(PARENT / "summary.json")["status"] == "automatic_checks_complete", "Completed full flows required")
    parent = read(PARENT / "freeze.json")
    files = {ROOT / name for name in parent["code_sha256"] if not name.startswith("frontend/dist/")}
    files.update(p for p in (ROOT / "frontend/dist").rglob("*") if p.is_file())
    files.update(ROOT / p for p in (
        "backend/app/engine/image_assessment_scope.py", "backend/scripts/native_image_scope_replay_v50.py",
        "backend/scripts/native_image_scope_replay_runtime_v50.py", "backend/tests/test_image_assessment_scope_v50.py",
        "backend/tests/fixtures/image_scope_cases_v50.json", "frontend/src/features/analysis/imageAssessmentScope.ts",
        "frontend/src/features/analysis/customerAssessment.ts", "frontend/src/features/analysis/AnswerModule.tsx",
        "frontend/e2e/native-image-scope-replay-v50.mjs", "docs/native-image-scope-replay-v50-protocol.md"))
    paths = []
    for n in range(5):
        p = PARENT / f"phase-{n:03d}"
        database = next((p / "work").glob("*/billing.sqlite"))
        paths.append({"phase": n, "database": str(database), "database_sha256": sha(database),
                      "completed_sha256": sha(p / "completed.json"), "evidence_sha256": sha(p / "evidence.json"),
                      "old_pdf_sha256": sha(p / "report.pdf")})
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "private-paths.json", paths)
    frozen = {"schema": "chipmunk.image-scope-presentation-replay/50",
              "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
              "runtime_versions": runtime_versions(), "private_paths_sha256": sha(TRIAL / "private-paths.json"),
              "cases": 5, "new_model_forwards_permitted": 0, "model_or_threshold_changes": False,
              "expected_candidate_edit": {"direction": "inconclusive", "overall_confidence": None, "component_confidence": 96.6},
              "expected_original": {"direction": "original", "scope": "image_components", "confidence": 97.7},
              "numeric_decisions_masks_and_original_reports_unchanged": True, "default_changed": False,
              "report_template": "veritas-report-2.0.r79", "assessment_rule": "evidence-led-assessment/v6"}
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"prepared_cases": 5, "new_model_forwards": 0}), flush=True)


def admission():
    from scripts.prepare_regional_calibration_v42 import runtime_versions

    f, start = read(TRIAL / "freeze.json"), read(TRIAL / "execution-start.json")
    need(sha(PUBLIC) == sha(TRIAL / "freeze.json") == start["freeze_sha256"]
         and start["remote_verified_before_first_replay"] is True, "Published exact replay freeze required")
    need(all(sha(ROOT / k) == v for k, v in f["code_sha256"].items()), "Frozen presentation code changed")
    need(runtime_versions() == f["runtime_versions"] and sha(TRIAL / "private-paths.json") == f["private_paths_sha256"],
         "Runtime or private paths changed")
    p = TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json")
    need(sha(p) == start["preparation_profile_sha256"], "Preparation changed")
    r = read(p)
    need(r["child_exit_code"] == 0 and r["guard"] is None and r["sample_errors"] == 0
         and not r["resource_limit_observations"], "Successful bounded preparation required")
    return f


def phase(number):
    import fitz
    import httpx
    import uvicorn
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import sessionmaker

    admission()
    need(0 <= number < 5, "One explicit case required")
    if number:
        p = read(TRIAL / f"phase-{number - 1:03d}-profile/profile.json")
        need(read(TRIAL / f"phase-{number - 1:03d}/completed.json")["status"] == "complete"
             and p["child_exit_code"] == 0 and p["guard"] is None, "Prior successful replay required")
    entry = read(TRIAL / "private-paths.json")[number]
    source_db = Path(entry["database"])
    old = PARENT / f"phase-{number:03d}"
    need(sha(source_db) == entry["database_sha256"] and sha(old / "completed.json") == entry["completed_sha256"],
         "Original flow evidence changed")
    folder = TRIAL / f"phase-{number:03d}"
    folder.mkdir(exist_ok=False)
    work = folder / "work"
    work.mkdir()
    shutil.copytree(source_db.parent / "objects", work / "objects")
    shutil.copytree(source_db.parent / "keys", work / "keys")
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as source, sqlite3.connect(work / "replay.sqlite") as target:
        source.backup(target)
    os.environ.update(VERITAS_DATA_DIR=str(work), VERITAS_DATABASE_URL=f"sqlite:///{work / 'replay.sqlite'}",
                      VERITAS_DEVICE="cpu", VERITAS_RUN_WORKER="false", VERITAS_BILLING_MODE="test",
                      VERITAS_API_KEY_PEPPER="isolated-replay-pepper", VERITAS_SIGNING_KEY="isolated-token-key")
    from app.api import auth, jobs
    from app.core.config import settings
    from app.detectors.image import regional, splice
    from app.engine import models
    from app.storage import db
    from app.storage.db import Analysis, PrincipalRow

    # The admission helper may already have imported Settings. Override the
    # object as well as the environment, before opening any application DB.
    settings.data_dir = work
    settings.database_url = f"sqlite:///{work / 'replay.sqlite'}"
    settings.device = "cpu"
    settings.run_worker = False
    settings.billing_mode = "test"
    settings.api_key_pepper = "isolated-replay-pepper"
    settings.signing_key = "isolated-token-key"
    settings.webhooks_enabled = False
    need(settings.sqlalchemy_url == f"sqlite:///{work / 'replay.sqlite'}", "Isolated replay database required")
    engine = db.make_engine(settings.sqlalchemy_url)
    db._engine, db._SessionLocal = engine, sessionmaker(engine, expire_on_commit=False)
    detail_old = read(old / "detail.json")
    aid = detail_old["id"]
    with db.get_session() as session:
        a = session.get(Analysis, aid)
        original_result_hash = a.result_hash
        principal = session.get(PrincipalRow, a.submitted_by)
        key, _ = auth.issue_api_key(session, principal)
        session.commit()
    attempted = []

    def no_inference(*_args, **_kwargs):
        attempted.append(True)
        raise AssertionError("Replay may not load models or perform new inference")

    models._import_callable = no_inference
    regional.RegionalModel.predict = no_inference
    splice.predict = no_inference
    from app.main import app

    client, headers = TestClient(app), {"X-API-Key": key}
    worker = jobs.build_worker()
    worker.start()
    try:
        detail = client.get(f"/api/v2/analyses/{aid}", headers=headers)
        exported = client.get(f"/api/analyses/{aid}/export.json", headers=headers)
        evidence = client.get(f"/api/v2/analyses/{aid}/evidence-packets", headers=headers)
        need(all(r.status_code == 200 for r in (detail, exported, evidence)), "Replay API read failed")
        got = detail.json()["customer_assessment"]
        need(got == exported.json()["customer_assessment"], "API and JSON presentation differ")
        need(evidence.json() == read(old / "evidence.json") == exported.json()["image_evidence"], "Evidence or masks changed")
        if number == 3:
            need(got["direction"] == "inconclusive" and got["confidence_pct"] is None
                 and got["component_estimate"]["confidence_pct"] == 96.6, "Editing conflict remains a whole-file original verdict")
        elif number in (2, 4):
            need(got["direction"] == "original" and got["scope"] == "image_components"
                 and got["confidence_pct"] == 97.7, "Original image estimate is not scoped")
        elif number == 1:
            need(got["scope"] == "image_components" and got["confidence_pct"] == 96.6, "Legacy estimate was numerically changed")
        else:
            need(got["direction"] not in {"ai", "original"}, "Undersized image was cleared")
        response = client.post(f"/api/v2/analyses/{aid}/reports", headers=headers, json={"profile": "standard"})
        need(response.status_code == 202, "New versioned report was not queued")
        rid = response.json()["id"]
        until = time.monotonic() + 300
        while True:
            report = client.get(f"/api/v2/reports/{rid}", headers=headers).json()
            if report["status"] in {"done", "failed"}:
                break
            need(time.monotonic() < until, "Report timed out")
            time.sleep(.1)
        need(report["status"] == "done", "Revised report failed")
        response = client.get(f"/api/v2/reports/{rid}/file", headers=headers)
        need(response.status_code == 200, "Revised report is unavailable")
        (folder / "report.pdf").write_bytes(response.content)
        with fitz.open(stream=response.content, filetype="pdf") as pdf:
            text = " ".join("\n".join(p.get_text() for p in pdf).split())
            pages = pdf.page_count
        if number == 3:
            need("Local editing cues found; AI origin is inconclusive" in text
                 and "No whole-file confidence percentage is assigned" in text, "PDF did not preserve corrected scope")
        (folder / "report.txt").write_text(text)
        (folder / "detail.json").write_bytes(detail.content)
        (folder / "export.json").write_bytes(exported.content)
        if number == 3:
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8164, lifespan="off", log_config=None, access_log=False))
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                until = time.monotonic() + 20
                while not server.started:
                    need(time.monotonic() < until, "Browser server did not start")
                    time.sleep(.1)
                # Assert loopback before handing the private test key to the browser.
                need(httpx.get("http://127.0.0.1:8164/").status_code == 200, "Local frontend unavailable")
                (folder / "browser-input.json").write_text(json.dumps({"analysis_id": aid, "headers": headers}))
                os.chmod(folder / "browser-input.json", 0o600)
                run = subprocess.run(["node", str(ROOT / "frontend/e2e/native-image-scope-replay-v50.mjs"), str(folder)],
                                     timeout=90, capture_output=True, text=True)
                (folder / "browser.log").write_text(run.stdout + run.stderr)
                need(run.returncode == 0, "Corrected ordinary browser presentation failed")
            finally:
                server.should_exit = True
                thread.join(timeout=15)
        with db.get_session() as session:
            need(session.get(Analysis, aid).result_hash == original_result_hash, "Recorded numerical decision was rewritten")
        need(not attempted, "Unexpected model/inference attempt")
        need(sha(source_db) == entry["database_sha256"] and sha(old / "report.pdf") == entry["old_pdf_sha256"],
             "Original DB or PDF changed")
        write(folder / "completed.json", {"status": "complete", "phase": number, "case": read(old / "completed.json")["case"],
              "assessment": got, "raw_result_hash_unchanged": True, "evidence_and_masks_unchanged": True,
              "old_database_and_pdf_unchanged": True, "new_model_calls": 0, "new_weight_load_attempts": 0,
              "pdf_pages": pages, "pdf_sha256": sha(folder / "report.pdf"), "browser_verified": number == 3,
              "pdf_visual_review_pending": True})
    finally:
        worker.stop(timeout=30)
        client.close()
        engine.dispose()


def seal():
    admission()
    rows = []
    for n in range(5):
        r = read(TRIAL / f"phase-{n:03d}/completed.json")
        p = read(TRIAL / f"phase-{n:03d}-profile/profile.json")
        need(r["status"] == "complete" and p["child_exit_code"] == 0 and p["guard"] is None
             and p["sample_errors"] == 0 and not p["resource_limit_observations"], "Every bounded replay must pass")
        rows.append(r)
    write(TRIAL / "summary.json", {"schema": "chipmunk.image-scope-presentation-replay-result/50",
          "status": "automatic_checks_complete", "cases": rows, "new_model_calls": 0,
          "raw_decisions_and_evidence_unchanged": True, "pdf_visual_review_pending": True,
          "default_changed": False, "github_workflows_started": 0, "deployment_done": False, "outreach_sent": False})
    print(json.dumps({"replayed_cases": 5, "new_model_calls": 0, "pdf_visual_review_pending": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "phase", "seal"))
    parser.add_argument("number", nargs="?", type=int)
    args = parser.parse_args()
    {"prepare": prepare, "phase": lambda: phase(args.number), "seal": seal}[args.command]()
