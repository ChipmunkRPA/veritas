"""Run the held face proposals through the local prepaid video worker."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts.audio_stereo_timing_change_app import local_model_view
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_common_sources_v89 import read, sha, write
from scripts.video_face_local_fit import overlap
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def run(receipt, inputs, output, *, all_checks=False):
    source = read(receipt)
    candidates = [row for row in source["rows"] if row["proposal"] is not None]
    paid.configure(output, ROOT)
    if all_checks:
        os.environ.pop("VERITAS_VIDEO_WHOLE_FACE_REVIEW")
    models = local_model_view(output)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_FACE_LOCAL_RETURN_REVIEW="true",
                      VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"), VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_MODELS_DIR=str(models), VERITAS_HF_HOME=str(models / "hf"))

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence.face_local_return_binding import METHOD
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    assert paid.ALLOW.issubset(enabled("video", "full"))
    if not all_checks:
        registry.enabled = lambda modality, profile: [item for item in enabled(modality, profile) if item in paid.ALLOW]
    stub = paid.PaymentStub()
    service._stripe = stub
    comparison_verified = False
    cases, browser, demonstrated = [], [], set()
    with TestClient(app, client=("127.0.0.1", 50000)) as client, (output / "worker.jsonl").open("x") as journal:
        with get_session() as db:
            principal = auth.new_principal("local-video-face-shot-guard", "examiner")
            db.add(principal)
            db.flush()
            key, _ = auth.issue_api_key(db, principal)
            db.commit()
        client.headers["X-API-Key"] = key
        for row in candidates:
            media = inputs / row["file"]
            if sha(media) != row["input_sha256"]:
                raise ValueError("The prepaid test received different input bytes")
            aid = paid.checkout(client, stub, media, filename=media.name)
            _, entry = paid.read_result(aid, row["input_sha256"], require_signal=False)
            details = entry["details"]
            coverage = details.get("face_return_check", {}).get("local_duration_fallback")
            selected = details.get("face_return") or {}
            marked = [selected["start_frame_index"], selected["end_frame_index"]] if selected.get("method") == METHOD else None
            prior = [[round(record[f"{edge}_stream_s"] * 25) for edge in ("start", "end")]
                     for name in ("face_sequence", "face_boundary", "face_return")
                     if (record := details.get(name)) and record["method"] != METHOD]
            if marked and (row["cuts"] or marked != row["proposal"]):
                raise ValueError("Worker accepted an abrupt shot change or moved a measured range")
            added = overlap(marked, row["truth"]) >= .5 and not any(overlap(bounds, row["truth"]) >= .5 for bounds in prior)
            case = {field: row[field] for field in ("file", "group", "family", "delivery", "input_sha256", "truth")}
            case.update(prior_ranges=prior, new_range=marked, extra_useful=added, coverage=coverage)
            if added and row["delivery"] not in demonstrated:
                route = f"/api/v2/analyses/{aid}"
                evidence = paid.body(client.get(f"{route}/evidence-packets"))
                [packet] = [item for item in evidence["packets"] if item.get("semantics") == METHOD]
                export = paid.fetch(client, f"/api/analyses/{aid}/export.json").decode()
                assert json.loads(export)["evidence"] == evidence and packet["face_return"] == selected
                assert "no abrupt shot change" in packet["interpretation"]
                for payload in (export, paid.fetch(client, route).decode(), paid.fetch(client, f"{route}/detector-runs").decode()):
                    scan(payload)
                target = output / media.stem
                artifacts = paid.proof_files(client, packet, target)
                case.update(paid.encode_report(client, aid, media, evidence, artifacts, target,
                                               expected_text="no abrupt shot change"))
                case["playback_s"] = [packet["window"][f"{edge}_playback_s"] for edge in ("start", "end")]
                case["anchors"] = [frame["anchor"]["frame_index"] for frame in packet["frames"]]
                browser.append({"id": aid, "name": media.stem, "packet": packet["id"], "title": packet["title"],
                                "target": packet["frames"][1]["anchor"], "frames": packet["window"]["frame_indices"]})
                demonstrated.add(row["delivery"])
                if not comparison_verified:
                    settings.video_face_local_return_review = False
                    comparison = paid.checkout(client, stub, media, filename=media.name)
                    assert tokens_assessed(comparison) == tokens_assessed(aid)
                    old = paid.body(client.get(f"/api/v2/analyses/{comparison}/evidence-packets"))
                    assert all(item.get("semantics") != METHOD for item in old["packets"])
                    settings.video_face_local_return_review = True
                    comparison_verified = True
            cases.append(case)
            journal.write(json.dumps(case, sort_keys=True) + "\n")
            journal.flush()
            print(media.name, "previous", prior, "new", marked, "added", added, flush=True)
        shown = browser_check(output, app, key, browser, scenario="video-face-shot-guard.mjs") if browser else None
    totals = Counter()
    for case in cases:
        cut_reason = (case["coverage"] or {}).get("reason") == "local_shot_change"
        totals.update({"extra_useful": int(case["extra_useful"]), "ordinary_new_flags": int(case["truth"] is None and bool(case["new_range"])),
                       "misplaced_edit_ranges": int(case["truth"] is not None and bool(case["new_range"])
                                                     and overlap(case["truth"], case["new_range"]) < .5),
                       "shot_vetoes": int(cut_reason)})
    result = {"comparisons": len(cases), "cases": cases, "counts": dict(totals), "browser": shown,
              "comparison_sha256": sha(receipt), "simulated_paid_jobs": len(stub.sessions), "unpaid_jobs": 0,
              "confidence_toggle_verified": comparison_verified, "all_checks": all_checks}
    write(output / "receipt.json", result)
    print(dict(totals), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("receipt", "inputs", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--all-checks", action="store_true")
    args = parser.parse_args()
    run(args.receipt, args.inputs, args.output, all_checks=args.all_checks)
