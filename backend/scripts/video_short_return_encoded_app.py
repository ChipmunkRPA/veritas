# SPDX-License-Identifier: Apache-2.0
"""Run the real local video worker on a frozen 360p panel with simulated payment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import io
import json
import os
import secrets
import time
import uuid
from fractions import Fraction
from pathlib import Path

ALLOW = {"vid.faces.yunet", "vid.tracks.yunet_dinov2", "vid.review.face_appearance"}
WEBHOOK_SECRET = "whsec_isolated_short_return_transfer"


class PaymentStub:
    """Local test transport only. The app validates the simulated responses."""
    def __init__(self):
        self.sessions = {}

    def __call__(self, method, path, *, data=None, params=None, idempotency_key=None):
        if method == "POST":
            sid = "cs_test_" + data["client_reference_id"].replace("-", "")
            self.sessions[sid] = {"id": sid, "object": "checkout.session", "mode": "payment", "livemode": False,
                "client_reference_id": data["client_reference_id"], "amount_total": 999, "amount_subtotal": 999,
                "currency": "usd", "metadata": {key[9:-1]: value for key, value in data.items() if key.startswith("metadata[")},
                "payment_status": "unpaid", "status": "open", "payment_intent": "pi_test_" + sid,
                "url": "https://checkout.stripe.com/c/pay/" + sid, "line_items": {"has_more": False, "data": [
                    {"quantity": 1, "amount_total": 999, "amount_subtotal": 999, "currency": "usd",
                     "price": {"unit_amount": 999, "currency": "usd", "type": "one_time"}}]}}
        else:
            sid = path.rsplit("/", 1)[-1]
        return copy.deepcopy(self.sessions[sid])


def body(response):
    response.raise_for_status()
    return response.json()


def fetch(client, url):
    response = client.get(url)
    response.raise_for_status()
    return response.content


def event(client, sid, nonce):
    event_id = "evt_test_" + nonce
    content = json.dumps({"id": event_id, "type": "checkout.session.completed", "livemode": False,
                          "data": {"object": {"id": sid}}}, separators=(",", ":")).encode()
    timestamp = str(int(time.time()))
    signature = hmac.new(WEBHOOK_SECRET.encode(), timestamp.encode() + b"." + content, hashlib.sha256).hexdigest()
    return body(client.post("/api/v2/billing/webhook", content=content, headers={"Stripe-Signature": f"t={timestamp},v1={signature}"}))


def wait(client, url, timeout=180):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = body(client.get(url))
        if result["status"] in ("done", "failed", "cancelled"):
            if result["status"] != "done":
                raise RuntimeError(f"Local worker ended with {result['status']}")
            return result
        time.sleep(.6)
    raise TimeoutError("Local worker did not finish")


def counts():
    from app.storage.db import Analysis, Job, get_session
    with get_session() as db:
        return db.query(Analysis).count(), db.query(Job).count()


def checkout(client, stub, path):
    from app.billing import purchase_notice
    before = counts()
    order = body(client.post("/api/v2/billing/checkout", headers={"Idempotency-Key": str(uuid.uuid4())},
        data={"kind": "video", "profile": "full", "risk_accepted": "true", "risk_notice_version": purchase_notice.VERSION,
              "risk_notice_sha256": purchase_notice.SHA256},
        files={"file": ("exact-360p-video.mp4", path.read_bytes(), "video/mp4")}))
    assert counts() == before
    sid = next(reversed(stub.sessions))
    nonce = uuid.uuid4().hex
    assert event(client, sid, nonce + "unpaid")["received"] and counts() == before
    stub.sessions[sid].update(payment_status="paid", status="complete")
    assert event(client, sid, nonce + "paid")["received"] and event(client, sid, nonce + "paid")["received"]
    order = body(client.get(f"/api/v2/billing/orders/{order['order_id']}"))
    assert order["status"] == "fulfilled" and counts() == tuple(value + 1 for value in before)
    aid = order["analysis_id"]
    wait(client, f"/api/analyses/{aid}")
    return aid


def read_result(aid, expected):
    from app.storage.db import Analysis, AnalysisStage, Artefact, get_session
    from app.storage.objects import default_store
    from sqlalchemy import select
    with get_session() as db:
        analysis = db.get(Analysis, aid)
        assert analysis.sha256 == expected and analysis.profile == "full"
        store = default_store()
        result = json.loads(store.read_bytes(analysis.result_hash))
        stage = db.scalar(select(AnalysisStage).where(AnalysisStage.analysis_id == aid, AnalysisStage.stage == "detect"))
        assert stage.status == "done"
        output = json.loads(store.read_bytes(db.get(Artefact, stage.output_artefact_id).sha256))
        [entry] = [row for row in output["detectors"] if row["detector_id"] == "vid.review.face_appearance"]
        [signal] = [row for row in result["signals"] if row["detector_id"] == "vid.review.face_appearance"]
        assert signal["effective_weight"] == 0 and signal["calibrated_probability"] is None
        return result["decision"], entry


def proof_files(client, packet, folder):
    folder.mkdir(mode=0o700)
    images = []
    for frame in packet["frames"]:
        for name in ("original", "crop"):
            entry = packet["artifacts"][frame["artifact_refs"][name]]
            data = fetch(client, entry["url"])
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            output = folder / f"{frame['interval_role']}-{name}.png"
            output.write_bytes(data)
            images.append({"role": frame["interval_role"], "asset": name, "file": output.name, "sha256": entry["sha256"]})
    assert len(images) == 6
    return images


def encode_report(client, aid, source, evidence, images, folder):
    from zipfile import ZipFile

    import pymupdf
    from app.reports.verify import verify_package
    from PIL import Image
    from scripts.verify_multimedia_exports import scan
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    def signature(image):
        rgb = image.convert("RGB")
        return rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()
    targets = {signature(Image.open(folder / entry["file"])) for entry in images}
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        words = " ".join(" ".join(page.get_text().split()) for page in doc)
        scan(words)
        assert "The brief face-change range spans" in words and "with an exclusive end" in words
        extracted = {signature(Image.open(io.BytesIO(doc.extract_image(xref)["image"])))
                     for xref in {img[0] for page in doc for img in page.get_images(full=True)}}
        assert targets <= extracted
    (folder / "report.pdf").write_bytes(pdf)
    pack = body(client.post(f"/api/v2/analyses/{aid}/packages",
                            json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    pack = wait(client, f"/api/v2/packages/{pack['id']}", timeout=120)
    payload = fetch(client, f"/api/v2/packages/{pack['id']}/file")
    destination = folder / "signed-package.zip"
    destination.write_bytes(payload)
    key = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    verified = verify_package(destination, public_key_pem=key, source_video=source)
    assert verified.outcome == "verified", verified.as_dict()
    assert any(row.ok and row.name == "original video frames" for row in verified.checks)
    with ZipFile(destination) as package:
        assert json.loads(package.read("evidence.json")) == evidence and package.read("report.pdf") == pdf
    return {"source_frame_crop_images_verified": len(images), "signed_original_replay_verified": True}


def configure(output, root):
    output.mkdir(mode=0o700, exist_ok=False)
    data = output / "app"
    os.environ.update(VERITAS_DATA_DIR=str(data), VERITAS_DATABASE_URL=f"sqlite:///{data}/veritas.db",
        VERITAS_MODELS_DIR=str(root / "models"), VERITAS_HF_HOME=str(root / "models/hf"), VERITAS_DEVICE="cpu",
        VERITAS_OFFLINE="true", VERITAS_RUN_WORKER="true", VERITAS_WORKER_THREADS="1",
        VERITAS_VIDEO_FACE_REVIEW_MODE="appearance", VERITAS_VIDEO_FACE_BOUNDARY_REFINEMENT="true",
        VERITAS_VIDEO_WHOLE_FACE_REVIEW="false", VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="true", VERITAS_BILLING_MODE="test",
        VERITAS_STRIPE_SECRET_KEY="sk_test_isolated_short_return_transfer", VERITAS_STRIPE_WEBHOOK_SECRET=WEBHOOK_SECRET,
        VERITAS_SINGLE_USER="false", VERITAS_API_KEY=secrets.token_hex(32), VERITAS_API_KEY_PEPPER=secrets.token_hex(32),
        VERITAS_SIGNING_KEY=secrets.token_hex(32), VERITAS_MIN_FREE_BYTES=str(1024 ** 3), VERITAS_JOB_TIMEOUT_S="180",
        MPLCONFIGDIR=str(output / "matplotlib"))


def run(folder, replay_names, codec_transfer=False):
    root = Path(__file__).resolve().parents[2]
    output = folder / ("serving" if replay_names else "scoring")
    configure(output, root)

    import cv2
    import torch
    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.detectors.video import face_return
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session
    from fastapi.testclient import TestClient
    from scripts.verify_multimedia_exports import scan

    if codec_transfer:
        from scripts import video_short_return_codec_transfer as study
    else:
        from scripts import video_short_return_encoded_transfer as study

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    lock = study.common.read(folder / "freeze.json")
    manifest = study.common.read(folder / "media.json")
    assert manifest["freeze_sha256"] == study.common.digest(folder / "freeze.json")
    assert (lock["delivery_size"], lock["delivery_rate_hz"]) == (list(study.DIMENSIONS), study.RATE)
    step = Fraction(1, study.RATE)
    if replay_names:
        assert step in face_return.PROFILE_STEPS.get(study.DIMENSIONS, ())
        roster = [row for row in manifest["media"] if row["file"] in replay_names]
        assert {row["file"] for row in roster} == set(replay_names)
        assert sum(row["truth_frames"] is not None for row in roster) == (2 if codec_transfer else 1)
    else:
        study.frozen(folder)
        steps = face_return.PROFILE_STEPS.get(study.DIMENSIONS, ())
        if step not in steps:
            face_return.PROFILE_STEPS[study.DIMENSIONS] = (*steps, step)
        roster = manifest["media"]
    registry = get_registry()
    enabled = registry.enabled
    assert ALLOW.issubset(enabled("video", "full"))
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in ALLOW]
    stub = PaymentStub()
    service._stripe = stub
    baseline = {}
    with (output / "journal.jsonl").open("x") as journal, TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("short-return-360p-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for active in ((False, True) if replay_names else (True,)):
            settings.video_face_short_return_review = active
            for row in roster:
                filename = row["file"]
                path = folder / "media" / filename
                assert study.common.digest(path) == row["sha256"]
                aid = checkout(client, stub, path)
                decision, entry = read_result(aid, row["sha256"])
                details = entry["details"]
                check = details.get("face_return_check")
                incumbent = details.get("face_sequence")
                if not active:
                    assert not check and not details.get("face_return")
                    baseline[filename] = (decision, incumbent)
                    continue
                if replay_names:
                    assert baseline[filename] == (decision, incumbent)
                evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
                exported = fetch(client, f"/api/analyses/{aid}/export.json")
                scan(exported.decode())
                assert json.loads(exported)["evidence"] == evidence
                packets = [p for p in evidence["packets"] if p.get("face_return")]
                assert len(packets) == int(details.get("face_return") is not None)
                images = []
                report = None
                packet = packets[0] if packets else None
                if packet is not None:
                    assert packet["face_return"] == details["face_return"] and packet["affects_overall_confidence"] is False
                    target = output / path.stem
                    images = proof_files(client, packet, target)
                    if replay_names:
                        assert row["truth_frames"] is not None
                        report = encode_report(client, aid, path, evidence, images, target)
                if replay_names:
                    assert bool(packets) == bool(row["truth_frames"])
                journal.write(json.dumps({"file": filename, "input_sha256": row["sha256"], "incumbent": incumbent,
                    "return_check": check, "packet": packet, "images": images, "report_replay": report,
                    "paid_before_analysis_and_idempotency": True, "decision_parity": bool(replay_names)},
                    allow_nan=False, sort_keys=True) + "\n")
                journal.flush()
                print(filename, (check or {}).get("status"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--replay", type=str, action="append", default=[])
    parser.add_argument("--codec-transfer", action="store_true")
    args = parser.parse_args()
    run(args.folder.resolve(), args.replay, args.codec_transfer)


if __name__ == "__main__":
    main()
