# SPDX-License-Identifier: Apache-2.0
"""Verify both distinct AVC face ranges through simulated prepaid analysis and signed exports."""

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.video_brief_face_avc_app import ROOT
from scripts.video_face_distinct_avc import BASE_FLAGS
from scripts.video_short_return_encoded_app import (
    ALLOW,
    PaymentStub,
    body,
    checkout,
    configure,
    fetch,
    read_result,
    wait,
)


def run(bank, output):
    configure(output, ROOT)
    os.environ.update(
        BASE_FLAGS, VERITAS_VIDEO_FACE_SECOND_AVC_REVIEW="false", VERITAS_VIDEO_WHOLE_FACE_REVIEW="false"
    )

    import cv2
    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence import face_short_proposal_binding as proof
    from app.main import app
    from app.reports.verify import verify_package
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan
    from scripts.video_face_paired_edge_hevc_app import image_signature
    from scripts.video_face_unrouted_short_app import native_frames

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in ALLOW]
    assert set(registry.enabled("video", "full")) == ALLOW
    stub = PaymentStub()
    service._stripe = stub
    roster = {row["file"]: row for row in json.loads((bank / "media.json").read_text())["rows"]}
    cases = (
        "g1-two_swaps.mp4",
        "g2-tandem_two_swaps.mp4",
        "g2-swap_then_jump.mp4",
        "g1-cross_recording_cuts.mp4",
    )
    baseline, results = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-avc-two-range-trial", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for active in (False, True):
            settings.video_face_second_avc_review = active
            for case in cases:
                row = roster[case]
                source = bank / "media" / case
                aid = checkout(client, stub, source, filename="local-avc-trial.mp4")
                decision, _ = read_result(aid, row["sha256"])
                route = f"/api/v2/analyses/{aid}"
                evidence = body(client.get(f"{route}/evidence-packets"))
                ranges = [packet for packet in evidence["packets"] if "face_return" in packet]
                incumbent = [
                    packet["face_return"]
                    for packet in ranges
                    if packet["semantics"] != proof.SECOND_AVC_METHOD
                ]
                if not active:
                    assert len(incumbent) == len(ranges)
                    baseline[case] = decision, incumbent
                    continue
                assert (decision, incumbent) == baseline[case]
                additional = [packet for packet in ranges if packet["semantics"] == proof.SECOND_AVC_METHOD]
                dual = len(row["truth"]) == 2
                assert len(additional) == int(dual)
                one_change = len(row["truth"]) == 1
                second_scan = evidence["coverage"]["brief_face_review"]["passes"].get("second_avc")
                if dual or one_change:
                    assert second_scan["status"] == ("localized" if dual else "no_distinct_range")
                else:
                    assert second_scan is None
                exported = fetch(client, f"/api/analyses/{aid}/export.json")
                assert json.loads(exported)["evidence"] == evidence
                scan(exported.decode())
                if not dual:
                    results[case] = {
                        "additional_ranges": 0,
                        "assessment_retained": True,
                        "second_scan": second_scan["status"] if second_scan else "not_run",
                    }
                    continue
                assert evidence["version"] == "video-evidence-27"
                second = additional[0]["face_return"]
                assert [second["start_frame_index"], second["end_frame_index"]] == row["truth"][1]
                case_output = output / source.stem
                case_output.mkdir()
                images, frames = set(), []
                for packet in ranges:
                    witness = case_output / packet["id"]
                    witness.mkdir()
                    anchors = packet["face_return"]["anchors"]
                    representatives = {role: anchors[role] for role in ("before", "during", "after")}
                    found, _ = native_frames(
                        client, packet, source, row, witness, representatives=representatives
                    )
                    images.update(found)
                    frames.append(list(representatives.values()))
                assert len(images) == 12
                report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
                report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
                pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
                with pymupdf.open(stream=pdf, filetype="pdf") as document:
                    words = " ".join(" ".join(page.get_text().split()) for page in document)
                    scan(words)
                    assert "Second brief face-change scan. A separate range was recorded." in words
                    for packet in ranges:
                        change = packet["face_return"]
                        assert (
                            f"brief face-change range spans {change['start_stream_s']:.3f}-{change['end_stream_s']:.3f} s"
                            in words
                        )
                    assert images <= {
                        image_signature(document.extract_image(xref)["image"])
                        for xref in {image[0] for page in document for image in page.get_images(full=True)}
                    }
                package = body(
                    client.post(
                        f"{route}/packages",
                        json={"report_id": report["id"], "kind": "draft", "include_original": False},
                    )
                )
                package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
                signed = case_output / "signed-package.zip"
                signed.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
                signer = body(client.get("/api/v2/instance/public-key"))
                replay = verify_package(
                    signed,
                    public_key_pem=signer["public_key_pem"].encode(),
                    expect_key_id=signer["key_id"],
                    source_video=source,
                )
                assert replay.outcome == "verified" and replay.signer_authenticated, replay.as_dict()
                assert any(
                    item.ok
                    and item.name == "original video frames"
                    and "each reported proposal" in item.detail
                    for item in replay.checks
                ), replay.as_dict()
                with ZipFile(signed) as archive:
                    assert json.loads(archive.read("evidence.json")) == evidence
                    bundle = json.loads(archive.read("bundle.json"))
                    assert archive.read(bundle["reports"][0]["file"]) == pdf
                (case_output / "report.pdf").write_bytes(pdf)
                (case_output / "export.json").write_bytes(exported)
                results[case] = {
                    "additional_ranges": 1,
                    "assessment_retained": True,
                    "second_scan": second_scan["status"],
                    "source_frames": frames,
                    "exact_source_images": len(images),
                    "pdf_json_and_source_verified": True,
                }
                print(case, results[case], flush=True)
    receipt = {"cases": results, "simulated_paid_runs": 2 * len(cases), "unpaid_jobs": 0, "default_off": True}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.bank.resolve(), args.output.resolve())
