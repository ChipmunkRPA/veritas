# SPDX-License-Identifier: Apache-2.0
"""Verify whole-picture cut context in an isolated prepaid local analysis."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from scripts.video_face_paired_edge_hevc_app import image_signature, sha, write_json
from scripts.video_face_unrouted_short_app import METHOD as SHORT_METHOD
from scripts.video_face_unrouted_short_app import VERSION as SHORT_VERSION
from scripts.video_face_unrouted_short_app import fixture_plan
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

FILES = {"ordinary_cut": "speaker_cuts-o1-person_switch.mp4", "short_swap": "source_edits-g1-two_swaps.mp4",
         "ordinary_original": "source_edits-g1-original.mp4"}
METHOD = "face-picture-cut-context/1"
VERSION = "video-evidence-22"
EDGES = (63, 74)
REPRESENTATIVES = dict(zip(("before", "during", "after"), (62, 68, 74), strict=True))


def verify_surfaces(client, aid, source, commitment, evidence, packet, folder):
    import pymupdf
    from PIL import Image

    from app.detectors.video.decode import decode_frames, probe_frames
    from app.evidence.video_packets import _context
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    folder.mkdir(mode=0o700)
    table = probe_frames(source)
    assert list(table.pts) == commitment["clock"]["pts"] and table.rotation == 0 and not table.interpolated and not table.truncated
    frames = {row.index: row for row in decode_frames(source, table, sorted(REPRESENTATIVES.values()))}
    signatures, assets = set(), []
    for row, (role, index) in zip(packet["face_spatial_cut_frames"], REPRESENTATIVES.items(), strict=True):
        assert row["interval_role"] == role and row["score"] is None and row["context_region"] is None
        assert row["region_meaning"] == "whole_frame_context" and set(row["artifact_refs"]) == {"original", "context"}
        anchor = row["anchor"]
        frame = frames[index]
        assert frame.pts_verified and anchor["frame_index"] == index and frame.pts == table.pts[index] == anchor["pts"]
        assert anchor["stream_time_s"] == round(float(table.pts[index] * table.tb), 6)
        rgb_hash = sha(frame.rgb.tobytes())
        assert rgb_hash == commitment["rgb_sha256"][index] == anchor["decoded_rgb_sha256"]
        context = _context(Image.fromarray(frame.rgb), None, f"{row['id']} - frame {index}, {anchor['stream_time_s']:.3f} s")
        expected = {"original": ((table.width, table.height), rgb_hash), "context": (context.size, sha(context.tobytes()))}
        for kind, signature in expected.items():
            asset = packet["artifacts"][row["artifact_refs"][kind]]
            data = fetch(client, asset["url"])
            assert sha(data) == asset["sha256"] and image_signature(data) == signature
            signatures.add(signature)
            assets.append({"role": role, "kind": kind, "frame": index, "pts": table.pts[index], "sha256": asset["sha256"]})
    assert len(signatures) == 6 and any("crop" in row["artifact_refs"] for row in packet["frames"])
    exported = fetch(client, f"/api/analyses/{aid}/export.json")
    assert json.loads(exported)["evidence"] == evidence
    scan(exported.decode())
    (folder / "export.json").write_bytes(exported)
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        words = " ".join(" ".join(page.get_text().split()) for page in document)
        scan(words)
        assert "Picture-cut context for a face change" in words and "2.100" in words and "2.467" in words
        assert "plausible explanation" in words and "exclusive end" in words
        assert all(f"source frame {index}" in words for index in REPRESENTATIVES.values())
        xrefs = {item[0] for page in document for item in page.get_images(full=True)}
        assert signatures <= {image_signature(document.extract_image(xref)["image"]) for xref in xrefs}
    (folder / "report.pdf").write_bytes(pdf)
    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                              json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
    signed = folder / "signed-package.zip"
    signed.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    signer = body(client.get("/api/v2/instance/public-key"))
    key = signer["public_key_pem"].encode()
    replay = verify_package(signed, public_key_pem=key, expect_key_id=signer["key_id"], source_video=source)
    assert replay.ok and replay.signer_authenticated, replay.as_dict()
    assert any(item.ok and item.name == "original video frames"
               and "Recomputed the surrounding-picture cut context and both native boundaries" in item.detail
               for item in replay.checks), replay.as_dict()
    with ZipFile(signed) as archive:
        bundle = json.loads(archive.read("bundle.json"))
        assert json.loads(archive.read("evidence.json")) == evidence
        assert archive.read(bundle["reports"][0]["file"]) == pdf
    changed = folder / "altered-original.mp4"
    changed.write_bytes(source.read_bytes() + b"\n")
    assert not verify_package(signed, public_key_pem=key, source_video=changed).ok
    return {"exact_source_images": assets, "pdf_api_json_agree": True, "signed_original_replay": True,
            "different_original_rejected": True}


def run(study, output):
    roster = fixture_plan(study)
    root = Path(__file__).resolve().parents[2]
    configure(output, root)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW="true",
                      VERITAS_VIDEO_FACE_SPATIAL_CUT_CONTEXT="true", VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT="true",
                      VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="true", VERITAS_VIDEO_FACE_BOUNDARY_REFINEMENT="true",
                      VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW="true", VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW="true",
                      VERITAS_MIN_FREE_BYTES=str(256 * 1024**2), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    # A later isolated API process can read the same fixture-only pepper from disk.
    os.environ["VERITAS_API_KEY_PEPPER"] = ""
    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session

    assert settings.data_dir == (output / "app").resolve() and settings.video_face_spatial_cut_context
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    assert set(enabled("video", "full")) >= ALLOW
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in ALLOW]
    stub = PaymentStub()
    service._stripe = stub
    results, decisions = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("spatial-cut-test-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for case in (*FILES, "feature_disabled"):
            selected = case if case in FILES else "ordinary_cut"
            source = study / FILES[selected]
            commitment = roster[selected]
            settings.video_face_spatial_cut_context = case != "feature_disabled"
            aid = checkout(client, stub, source)
            decision, detector = read_result(aid, commitment["sha256"])
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            assert evidence["source"]["sha256"] == commitment["sha256"]
            coarse = [packet for packet in evidence["packets"] if packet.get("face_sequence")]
            picture_cut = [packet for packet in evidence["packets"] if packet.get("face_spatial_cut")]
            brief = [packet for packet in evidence["packets"] if packet.get("face_return", {}).get("method") == SHORT_METHOD]
            row = {"file_sha256": commitment["sha256"], "version": evidence["version"], "coarse_packets": len(coarse),
                   "picture_cut_packets": len(picture_cut), "brief_face_change_packets": len(brief)}
            print(case, row, flush=True)
            assert bool(coarse) == (selected == "ordinary_cut")
            if case == "ordinary_cut":
                assert evidence["version"] == VERSION and not brief
                [packet] = picture_cut
                assert packet is coarse[0] and packet["affects_overall_confidence"] is False
                assert packet["title"] == "Picture-cut context for a face change"
                measured = packet["face_spatial_cut"]
                assert measured == detector["details"]["face_spatial_cut"] and measured["method"] == METHOD
                assert measured["anchors"] == REPRESENTATIVES
                tb = Fraction(evidence["source"]["time_base"])
                assert [measured[k] * tb for k in ("start_pts", "end_pts")] == [Fraction(index, 30) for index in EDGES]
                assert [measured[k] for k in ("start_frame_index", "end_frame_index")] == list(EDGES)
                assert [round(measured[k], 3) for k in ("start_stream_s", "end_stream_s")] == [2.100, 2.467]
                row.update(verify_surfaces(client, aid, source, commitment, evidence, packet, output / "ordinary_cut"))
                row.update(measured_frames=list(EDGES), source_time_s=[2.100, 2.467], representative_frames=REPRESENTATIVES)
                write_json(output / "browser-input.json", {"analysis_id": aid, "packet_id": packet["id"]})
                (output / "browser-key.txt").write_text(key)
                (output / "browser-key.txt").chmod(0o600)
            else:
                assert not picture_cut and not detector["details"].get("face_spatial_cut")
                assert bool(brief) == (case == "short_swap")
                if brief:
                    assert evidence["version"] == SHORT_VERSION
            results[case], decisions[case] = row, decision
    assert decisions["ordinary_cut"] == decisions["feature_disabled"]
    write_json(output / "receipt.json", {"cases": results, "unpaid_jobs": 0, "simulated_paid_jobs": len(results),
                                         "overall_decision_identical_flag_on_off": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Frozen 720p HEVC fixture directory")
    parser.add_argument("output", type=Path, help="New isolated output directory")
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
