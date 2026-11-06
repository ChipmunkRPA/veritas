"""Controlled background edit -> real inference -> evidence/export/PDF integration, isolated database.

Uses consumed research fixtures, not a fresh accuracy evaluation. Only the Stripe
HTTP transport is doubled; the payment gate, signed webhook, queue, models,
calibration, evidence and report pipeline are production code. No real charge.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(Path(__file__).parent)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["candidate"], default="candidate")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--with-video", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    data = folder / "data"
    assert not (data / "veritas.db").exists(), "Use a fresh output directory; completed runs are immutable."
    local_weights = folder / "model-links" / "weights"
    local_weights.mkdir(parents=True, exist_ok=True)
    for original in (ROOT / "models" / "weights").iterdir():
        if not original.name.startswith(".veritas-"):
            (local_weights / original.name).symlink_to(original)
    os.environ.update(
        VERITAS_DATA_DIR=str(data),
        VERITAS_DATABASE_URL=f"sqlite:///{data}/veritas.db",
        VERITAS_MODELS_DIR=str(local_weights.parent),
        VERITAS_AUDIO_DETECTOR_MODE=args.mode,
        VERITAS_BILLING_MODE="test",
        VERITAS_STRIPE_SECRET_KEY="sk_test_isolated_validation",
        VERITAS_STRIPE_WEBHOOK_SECRET="whsec_isolated_validation",
        VERITAS_API_KEY=secrets.token_hex(32),
        VERITAS_API_KEY_PEPPER=secrets.token_hex(32),
        VERITAS_SIGNING_KEY=secrets.token_hex(32),
        VERITAS_SINGLE_USER="false",
        VERITAS_DEVICE="cpu",
        VERITAS_OFFLINE="true",
        VERITAS_JOB_TIMEOUT_S="900",
        VERITAS_MIN_FREE_BYTES="0",
        VERITAS_RUN_WORKER="true",
        MPLCONFIGDIR=str(folder / "matplotlib-cache"),
    )
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    from fastapi.testclient import TestClient
    from verify_multimedia_exports import run as exports
    from verify_multimedia_exports import scan

    from app.api import auth
    from app.billing import service
    from app.main import app
    from app.storage.db import Analysis, DetectorRun, Job, get_session
    from app.storage.objects import default_store

    remote = {}

    def provider(method, path, *, data=None, params=None, idempotency_key=None):
        if method == "POST":
            sid = "cs_test_" + data["client_reference_id"].replace("-", "")
            remote.setdefault(
                sid,
                {
                    "id": sid,
                    "object": "checkout.session",
                    "mode": "payment",
                    "livemode": False,
                    "client_reference_id": data["client_reference_id"],
                    "amount_total": 999,
                    "amount_subtotal": 999,
                    "currency": "usd",
                    "metadata": {k[9:-1]: v for k, v in data.items() if k.startswith("metadata[")},
                    "payment_status": "unpaid",
                    "status": "open",
                    "payment_intent": "pi_test_" + sid,
                    "url": "https://checkout.stripe.com/c/pay/" + sid,
                    "line_items": {
                        "has_more": False,
                        "data": [
                            {
                                "quantity": 1,
                                "amount_total": 999,
                                "amount_subtotal": 999,
                                "currency": "usd",
                                "price": {"unit_amount": 999, "currency": "usd", "type": "one_time"},
                            }
                        ],
                    },
                },
            )
            return copy.deepcopy(remote[sid])
        return copy.deepcopy(remote[path.rsplit("/", 1)[-1]])

    service._stripe = provider
    original = args.input.resolve()
    assert original.is_file()
    cases = [{"name": "background-edit", "kind": "audio", "path": original}]
    if args.with_video:
        video = folder / "background-edit-video.mov"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x254d42:s=320x180:r=25:d=4",
                "-i",
                str(original),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "pcm_s16le",
                "-shortest",
                str(video),
            ],
            check=True,
        )
        cases.append({"name": "background-edit-video", "kind": "video", "path": video})
    results = []
    with TestClient(app) as client:
        with get_session() as db:
            owner = auth.new_principal("rollout-validation", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        public_key = client.get("/api/v2/instance/public-key").json()["public_key_pem"].encode()
        for case in cases:
            started = time.monotonic()
            with get_session() as db:
                before = (db.query(Analysis).count(), db.query(Job).count())
            response = client.post(
                "/api/v2/billing/checkout",
                headers={"Idempotency-Key": str(uuid.uuid4())},
                data={"kind": case["kind"], "profile": "full"},
                files={"file": (case["path"].name, case["path"].read_bytes(), "application/octet-stream")},
            )
            assert response.status_code == 201, response.text
            oid = response.json()["order_id"]
            with get_session() as db:
                assert before == (db.query(Analysis).count(), db.query(Job).count()), (
                    "Analysis ran before payment"
                )
            sid = next(s for s, r in remote.items() if r["client_reference_id"] == oid)
            remote[sid].update(payment_status="paid", status="complete")
            event = {
                "id": "evt_test_" + oid.replace("-", ""),
                "type": "checkout.session.completed",
                "livemode": False,
                "data": {"object": {"id": sid}},
            }
            payload = json.dumps(event, separators=(",", ":")).encode()
            stamp = str(int(time.time()))
            sig = hmac.new(
                b"whsec_isolated_validation", stamp.encode() + b"." + payload, hashlib.sha256
            ).hexdigest()
            paid = client.post(
                "/api/v2/billing/webhook",
                content=payload,
                headers={"Stripe-Signature": f"t={stamp},v1={sig}"},
            )
            assert paid.status_code == 200, paid.text
            order = client.get(f"/api/v2/billing/orders/{oid}").json()
            aid = order["analysis_id"]
            print(
                json.dumps({"case": case["name"], "analysis_id": aid, "stage": "paid_and_queued"}), flush=True
            )
            until = time.monotonic() + 950
            while True:
                with get_session() as db:
                    row = db.get(Analysis, aid)
                    status, error, result_hash = row.status, row.error, row.result_hash
                if status in ("done", "failed", "cancelled"):
                    break
                assert time.monotonic() < until, "Analysis timeout"
                time.sleep(1)
            assert status == "done", (case["name"], status, error)
            result = json.loads(default_store().read_bytes(result_hash))
            with get_session() as db:
                assert db.query(DetectorRun).filter_by(analysis_id=aid, status="ok").count(), "No real checks ran"
            evidence = client.get(f"/api/v2/analyses/{aid}/evidence-packets").json()
            acoustic = (
                evidence["acoustic_context"]
                if case["kind"] == "audio"
                else evidence["audio"]["acoustic_context"]
            )
            assert acoustic["schema"] == "chipmunk.acoustic-context/2"
            event = next(e for e in acoustic["events"] if e["category"] == "spectral_transition")
            assert event["change_sample"] == 32000
            assert event["interval"] == {"start_sample": 24000, "end_sample": 40000}
            assert event["evidence_type"] == "descriptive"
            detail = client.get(f"/api/v2/analyses/{aid}")
            detail.raise_for_status()
            scan(detail.text)
            case_folder = folder / case["name"]
            case_folder.mkdir()
            (case_folder / "api.json").write_bytes(detail.content)
            # Complete export, evidence asset/range, PDF, signature and package correspondence.
            check = exports(client, aid, case_folder, default_store(), public_key)
            check.update(
                case=case["name"],
                seconds=round(time.monotonic() - started, 2),
                decision=result["decision"]["code"],
                probability_ai=result["decision"]["probability_ai_calibrated"],
                payment_before_analysis=True,
                controlled_edit=True,
                expected_edit_sample=32000,
                continuity_event=event,
                evidence_version=evidence["version"],
            )
            results.append(check)
            (folder / "results.json").write_text(
                json.dumps({"mode": args.mode, "purpose": __doc__, "results": results}, indent=2)
            )
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "status": "verified",
                        "seconds": check["seconds"],
                        "decision": check["decision"],
                        "p": check["probability_ai"],
                    }
                ),
                flush=True,
            )
    print("LOCAL BACKGROUND ROLLOUT VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    main()
