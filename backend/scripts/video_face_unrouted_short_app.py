# SPDX-License-Identifier: Apache-2.0
"""Verify the six-picture face-change fallback in a local prepaid analysis."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from scripts.video_face_paired_edge_hevc_app import image_signature, sha, sha_path, write_json
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

FILES = {"short_swap": "source_edits-g1-two_swaps.mp4", "ordinary_cut": "speaker_cuts-o1-person_switch.mp4",
         "ordinary_original": "source_edits-g1-original.mp4"}
METHOD = "face-short-return/9"
VERSION = "video-evidence-21"
EDGES = (62, 68)
REPRESENTATIVES = dict(zip(("before", "during", "after"), (61, 65, 68), strict=True))


def fixture_plan(study: Path) -> dict:
    media_file = study / "media.json"
    media = json.loads(media_file.read_bytes())
    measurements = json.loads((study / "held.json").read_bytes())
    baseline = json.loads((study / "incumbent.json").read_bytes())
    assert media["freeze_sha256"] == sha_path(study / "freeze.json")
    assert measurements["media_sha256"] == baseline["media_sha256"] == sha_path(media_file)
    measured = {row["sha256"]: row for row in measurements["rows"]}
    roster = {row["file"]: row for row in media["rows"]}
    result = {}
    for case, name in FILES.items():
        row = roster[name]
        assert row["sha256"] == sha_path(study / name)
        assert measured[row["sha256"]]["truth"] == row["truth"]
        assert (row["clock"]["width"], row["clock"]["height"]) == (1280, 720)
        assert row["clock"]["time_base"] == "1/30000" and len(row["clock"]["pts"]) == 120
        assert bool(baseline["rows"][row["sha256"]]) == (case == "ordinary_cut")
        result[case] = row
    positive = result["short_swap"]
    assert list(EDGES) in positive["truth"]
    assert any(item["frames"] == list(EDGES) and item["response"] > .10
               for item in measured[positive["sha256"]]["candidates"]["shortlist"])
    assert result["ordinary_cut"]["truth"] == result["ordinary_original"]["truth"] == []
    return result


def native_frames(client, packet, source, commitment, folder, *, representatives=REPRESENTATIVES):
    from app.detectors.video.decode import decode_frames, probe_frames

    table = probe_frames(source)
    assert list(table.pts) == commitment["clock"]["pts"]
    assert table.rotation == 0 and not table.truncated and not table.interpolated
    originals = {item.index: item for item in decode_frames(source, table, sorted(representatives.values()))}
    signatures, assets = set(), []
    for item, (role, index) in zip(packet["frames"], representatives.items(), strict=True):
        anchor = item["anchor"]
        assert item["interval_role"] == role and item["score"] is None
        assert anchor["frame_index"] == index and anchor["pts"] == table.pts[index]
        assert anchor["stream_time_s"] == round(float(table.pts[index] * table.tb), 6)
        decoded = originals[index]
        assert decoded.pts_verified
        rgb = decoded.rgb
        rgb_sha = sha(rgb.tobytes())
        assert rgb_sha == anchor["decoded_rgb_sha256"] == commitment["rgb_sha256"][index]
        box = item["context_region"]
        x, y, width, height = (box[key] for key in "xywh")
        expected = {"original": ((table.width, table.height), rgb_sha),
                    "crop": ((width, height), sha(rgb[y:y + height, x:x + width].tobytes()))}
        for kind, signature in expected.items():
            asset = packet["artifacts"][item["artifact_refs"][kind]]
            data = fetch(client, asset["url"])
            assert sha(data) == asset["sha256"] and image_signature(data) == signature
            signatures.add(signature)
            (folder / f"{role}-{kind}.png").write_bytes(data)
            assets.append({"role": role, "kind": kind, "frame": index, "pts": table.pts[index], "sha256": asset["sha256"]})
    assert len(signatures) == 6
    return signatures, assets


def verified_surfaces(client, aid, source, commitment, evidence, packet, folder, *, representatives=REPRESENTATIVES,
                      initial_coverage="Initial scan. This clip is outside the supported frame or face-size profile."):
    import pymupdf

    from app.reports import engine
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    folder.mkdir(mode=0o700)
    images, assets = native_frames(client, packet, source, commitment, folder, representatives=representatives)
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
        assert "brief face-change range spans 2.067-2.267 s" in words and "with an exclusive end" in words
        if initial_coverage is not None:
            assert initial_coverage in words
        assert "Bounded native scan. A brief face-change range was recorded." in words
        assert all(f"source frame {index}" in words for index in representatives.values())
        xrefs = {item[0] for page in document for item in page.get_images(full=True)}
        embedded = {image_signature(document.extract_image(xref)["image"]) for xref in xrefs}
        assert images <= embedded
    (folder / "report.pdf").write_bytes(pdf)
    package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                              json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
    signed = folder / "signed-package.zip"
    signed.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    signer = body(client.get("/api/v2/instance/public-key"))
    key = signer["public_key_pem"].encode()
    replay = verify_package(signed, public_key_pem=key, expect_key_id=signer["key_id"], source_video=source)
    assert replay.outcome == "verified" and replay.signer_authenticated, replay.as_dict()
    assert any(item.ok and item.name == "original video frames"
               and "Recomputed the bounded face shortlist and both near-head motion boundaries" in item.detail
               for item in replay.checks), replay.as_dict()
    with ZipFile(signed) as archive:
        bundle = json.loads(archive.read("bundle.json"))
        assert json.loads(archive.read("evidence.json")) == evidence
        assert archive.read(bundle["reports"][0]["file"]) == pdf
    changed = folder / "altered-original.mp4"
    changed.write_bytes(source.read_bytes() + b"\n")
    assert not verify_package(signed, public_key_pem=key, source_video=changed).ok
    return {"exact_source_images": assets, "pdf_api_json_agree": True, "signed_original_replay": True,
            "different_original_rejected": True, "template_set": engine.TEMPLATE_SET}


def run(study, output):
    roster = fixture_plan(study)
    root = Path(__file__).resolve().parents[2]
    configure(output, root)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW="true",
                      VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT="true", VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="true",
                      VERITAS_VIDEO_FACE_BOUNDARY_REFINEMENT="true", VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW="true",
                      VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW="true", VERITAS_MIN_FREE_BYTES=str(256 * 1024**2),
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session

    assert settings.data_dir == (output / "app").resolve() and settings.video_face_unrouted_short_review
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    assert set(enabled("video", "full")) >= ALLOW
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile) if key in ALLOW]
    stub = PaymentStub()
    service._stripe = stub
    decisions, cases = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("unrouted-short-test-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for case in (*FILES, "feature_disabled"):
            selected = case if case in FILES else "short_swap"
            source = study / FILES[selected]
            settings.video_face_unrouted_short_review = case != "feature_disabled"
            aid = checkout(client, stub, source)
            decision, detector = read_result(aid, roster[selected]["sha256"])
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            assert evidence["source"]["sha256"] == roster[selected]["sha256"]
            coarse = [packet for packet in evidence["packets"] if packet.get("face_sequence")]
            fallback = [packet for packet in evidence["packets"] if packet.get("face_return", {}).get("method") == METHOD]
            coverage = evidence["coverage"]["brief_face_review"]
            assert coverage["passes"]["initial"]["status"] == "not_applicable"
            assert coverage["passes"]["initial"]["reason"] == "native_scope"
            bounded = coverage["passes"].get("bounded_proposal")
            row = {"file_sha256": roster[selected]["sha256"], "coarse_packets": len(coarse),
                   "short_fallback_packets": len(fallback), "version": evidence["version"],
                   "brief_face_coverage": coverage["passes"]}
            print(case, row, flush=True)
            decisions[case] = decision
            if case == "short_swap":
                assert not coarse and evidence["version"] == VERSION
                [packet] = fallback
                measured = packet["face_return"]
                assert bounded == {"status": "localized", "reason": None, "pixel_frames": 120,
                                   "scheduled_frames": len(measured["scheduled_frame_indices"]),
                                   "measured_face_frames": len(measured["frame_inputs"])}
                assert packet["affects_overall_confidence"] is False and detector["details"]["face_return"] == measured
                assert measured["anchors"] == REPRESENTATIVES
                tb = Fraction(evidence["source"]["time_base"])
                assert [measured[k] * tb for k in ("start_pts", "end_pts")] == [Fraction(index, 30) for index in EDGES]
                assert [round(measured[k], 3) for k in ("start_stream_s", "end_stream_s")] == [2.067, 2.267]
                row.update(verified_surfaces(client, aid, source, roster[selected], evidence, packet, output / "short_swap"))
                row.update(known_edit_frames=list(EDGES), measured_frames=list(EDGES), source_time_s=[2.067, 2.267],
                           representative_frames=REPRESENTATIVES)
                write_json(output / "browser-input.json", {"analysis_id": aid, "packet_id": packet["id"]})
            else:
                assert not fallback and detector["details"].get("face_return", {}).get("method") != METHOD
                assert bool(coarse) == (case == "ordinary_cut")
                if case == "feature_disabled":
                    assert bounded is None
                else:
                    assert bounded["status"] == "no_distinct_range" and bounded["pixel_frames"] == 120
            cases[case] = row
    assert decisions["short_swap"] == decisions["feature_disabled"]
    write_json(output / "receipt.json", {"cases": cases, "unpaid_jobs": 0, "simulated_paid_jobs": len(cases),
                                         "overall_decision_identical_flag_on_off": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Frozen 720p HEVC study fixtures")
    parser.add_argument("output", type=Path, help="New isolated output directory")
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
