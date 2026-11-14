# SPDX-License-Identifier: Apache-2.0
"""Verify moving-face recovery through prepaid local analysis, exports and signed replay."""

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts.video_face_distinct_avc import BASE_FLAGS
from scripts.video_face_moving_shortlist import ROOT
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


def run(bank, output, *, primary_track=False, position_track=False, pane_track=False):
    configure(output, ROOT)
    os.environ.update(
        BASE_FLAGS, VERITAS_VIDEO_FACE_SECOND_AVC_REVIEW="true", VERITAS_VIDEO_FACE_MOVING_AVC_REVIEW="false",
        VERITAS_VIDEO_FACE_PRIMARY_TRACK_AVC_REVIEW="false",
        VERITAS_VIDEO_FACE_POSITION_TRACK_AVC_REVIEW=str(pane_track).lower(),
        VERITAS_VIDEO_FACE_PANE_TRACK_AVC_REVIEW="false",
    )

    import cv2
    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence.face_short_proposal_binding import (
        MOVING_AVC_METHOD,
        PANE_TRACK_AVC_METHOD,
        POSITION_TRACK_AVC_METHOD,
        PRIMARY_TRACK_AVC_METHOD,
    )
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
    if pane_track:
        cases = ("g1-swaps_left.mp4", "g1-swaps_right.mp4", "g1-ordinary_speaker_cut.mp4")
        serving_mode = "video_face_pane_track_avc_review"
        review_method, packet_version = PANE_TRACK_AVC_METHOD, "video-evidence-31"
    elif position_track:
        cases = ("g2-tandem_two_swaps.mp4", "g2-guest_cut.mp4", "g6-original.mp4")
        serving_mode = "video_face_position_track_avc_review"
        review_method, packet_version = POSITION_TRACK_AVC_METHOD, "video-evidence-30"
    elif primary_track:
        cases = ("g4-tandem_two_swaps.mp4", "g4-cross_recording_cuts.mp4", "g1-tandem_two_swaps.mp4")
        serving_mode = "video_face_primary_track_avc_review"
        review_method, packet_version = PRIMARY_TRACK_AVC_METHOD, "video-evidence-29"
    else:
        cases = ("g5-tandem_two_swaps.mp4", "g5-cross_recording_cuts.mp4", "g8-tandem_two_swaps.mp4")
        serving_mode = "video_face_moving_avc_review"
        review_method, packet_version = MOVING_AVC_METHOD, "video-evidence-28"
    before, results = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-moving-face-trial", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for active in (False, True):
            setattr(settings, serving_mode, active)
            for case in cases:
                row = roster[case]
                source = bank / "media" / case
                aid = checkout(client, stub, source, filename="local-moving-face.mp4")
                decision, _ = read_result(aid, row["sha256"])
                route = f"/api/v2/analyses/{aid}"
                evidence = body(client.get(f"{route}/evidence-packets"))
                recovered = [r for r in evidence["packets"] if r.get("semantics") == review_method]
                existing = [
                    r["face_return"] for r in evidence["packets"] if "face_return" in r and r not in recovered
                ]
                if not active:
                    assert not recovered
                    before[case] = decision, existing
                    continue
                assert before[case] == (decision, existing)
                assert len(recovered) == int(case in cases[:2 if pane_track else 1])
                exported = fetch(client, f"/api/analyses/{aid}/export.json")
                assert json.loads(exported)["evidence"] == evidence
                scan(exported.decode())
                if not recovered:
                    results[case] = {"ranges_kept": len(existing), "additional_ranges": 0}
                    if position_track:
                        coverage = evidence["coverage"]["brief_face_review"]["passes"]["bounded_proposal"]
                        results[case]["bounded_scan"] = coverage["status"]
                        if coverage["reason"]:
                            results[case]["coverage_reason"] = coverage["reason"]
                    continue
                [packet] = recovered
                assert evidence["version"] == packet_version
                change = packet["face_return"]
                interval = [change["start_frame_index"], change["end_frame_index"]]
                if pane_track:
                    assert interval in row["truth"] and change["pane"] == row["expected_pane"]
                else:
                    assert interval == row["truth"][1]
                anchors = {role: change["anchors"][role] for role in ("before", "during", "after")}
                images, _ = native_frames(client, packet, source, row, output, representatives=anchors)
                report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
                report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
                pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
                with pymupdf.open(stream=pdf, filetype="pdf") as document:
                    words = " ".join(" ".join(page.get_text().split()) for page in document)
                    scan(words)
                    assert (
                        f"brief face-change range spans {change['start_stream_s']:.3f}-{change['end_stream_s']:.3f} s"
                        in words
                    )
                    assert "Source frames compared at each boundary" in words
                    if primary_track:
                        assert "followed the larger face by its position and size" in words
                        assert any(len(row["boxes"]) >= 2 for row in change["face_detection_boxes"])
                    if position_track:
                        assert "followed the same face position when another face changed size" in words
                        assert any(len(row["boxes"]) >= 2 for row in change["face_detection_boxes"])
                        [references] = [m["value"] for m in packet["measurements"]
                                        if m["label"] == "Face position reference frames"]
                        assert f"Face position reference frames {references}" in words
                    if pane_track:
                        assert f"followed the face on the {change['pane']} side" in words
                        for label, count in zip(("Start", "End"), change["off_face_motion_tiles"], strict=True):
                            assert f"{label} abrupt changes outside the face on the {change['pane']} {count}" in words
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
                case_output = output / Path(case).stem if pane_track else output
                case_output.mkdir(exist_ok=True)
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
                    and "bounded face shortlist" in item.detail
                    for item in replay.checks
                ), replay.as_dict()
                with ZipFile(signed) as archive:
                    bundle = json.loads(archive.read("bundle.json"))
                    assert json.loads(archive.read("evidence.json")) == evidence
                    assert archive.read(bundle["reports"][0]["file"]) == pdf
                (case_output / "report.pdf").write_bytes(pdf)
                (case_output / "export.json").write_bytes(exported)
                results[case] = {
                    "additional_ranges": 1,
                    "frames": interval,
                    "exhibit_frames": list(anchors.values()),
                    "exact_source_images": len(images),
                    "signed_source_verified": True,
                }
                if position_track:
                    results[case]["face_position_reference_frames"] = [
                        int(index) for index in references.split(" and ")
                    ]
                if pane_track:
                    results[case]["side"] = change["pane"]
                    results[case]["off_face_motion_tiles"] = change["off_face_motion_tiles"]
    receipt = {
        "cases": results,
        "simulated_paid_runs": 2 * len(cases),
        "unpaid_jobs": 0,
        "overall_assessment_retained": True,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--primary-track", action="store_true")
    mode.add_argument("--position-track", action="store_true")
    mode.add_argument("--pane-track", action="store_true")
    args = parser.parse_args()
    run(args.bank.resolve(), args.output.resolve(), primary_track=args.primary_track,
        position_track=args.position_track, pane_track=args.pane_track)
