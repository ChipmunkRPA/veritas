# SPDX-License-Identifier: Apache-2.0
"""Replay a bounded HEVC face-boundary holdout through the local paid worker."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import uuid
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

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

FILES = {
    "held_swap": "hevc-g1-swap_b.mp4",
    "ordinary_cut": "hevc-c1-cutaway.mp4",
    "short_swap": "hevc-g1-swap_a.mp4",
}
METHOD = "face-appearance-paired-edge/1"
VERSION = "video-evidence-20"
ROLES = ("before", "during", "after")
EDGES = (87, 97)
REPRESENTATIVES = dict(zip(ROLES, (86, 92, 97), strict=True))


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_path(path: Path) -> str:
    return sha(path.read_bytes())


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def fixture_plan(study: Path) -> dict:
    manifest = json.loads((study / "media.json").read_bytes())
    reference = json.loads((study / "receipt.json").read_bytes())
    assert manifest["freeze_sha256"] == sha_path(study / "freeze.json")
    assert reference["media_sha256"] == sha_path(study / "media.json")
    assert reference["freeze"] == json.loads((study / "freeze.json").read_bytes())
    roster = {row["file"]: row for row in manifest["rows"]}
    measured = {row["sha256"]: row for row in reference["rows"]}
    result = {}
    for case, name in FILES.items():
        commitment = roster[name]
        assert commitment["sha256"] == sha_path(study / name)
        assert measured[commitment["sha256"]]["truth"] == commitment["truth"]
        assert (commitment["clock"]["width"], commitment["clock"]["height"]) == (1280, 720)
        assert commitment["clock"]["time_base"] == "1/30000"
        result[case] = commitment
    assert result["held_swap"]["truth"] == [list(EDGES)]
    assert result["ordinary_cut"]["truth"] == []
    assert result["short_swap"]["truth"] == [[72, 78]]
    return result


def image_signature(data: bytes) -> tuple:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        return rgb.size, sha(rgb.tobytes())


def verify_images(client, packet: dict, source: Path, commitment: dict, folder: Path) -> tuple:
    from app.detectors.video.decode import decode_frames, probe_frames

    table = probe_frames(source)
    assert list(table.pts) == commitment["clock"]["pts"]
    assert table.rotation == 0 and not table.truncated and not table.interpolated
    originals = {item.index: item for item in decode_frames(source, table, sorted(REPRESENTATIVES.values()))}
    targets, assets = set(), []
    for item, (role, index) in zip(packet["face_boundary_frames"], REPRESENTATIVES.items(), strict=True):
        assert item["interval_role"] == role
        assert item["anchor"]["frame_index"] == index and item["anchor"]["pts"] == table.pts[index]
        assert item["anchor"]["stream_time_s"] == round(float(table.pts[index] * table.tb), 6)
        rgb = originals[index].rgb
        rgb_sha = sha(rgb.tobytes())
        assert originals[index].pts_verified
        assert rgb_sha == item["anchor"]["decoded_rgb_sha256"] == commitment["rgb_sha256"][index]
        box = item["context_region"]
        x, y, width, height = (box[name] for name in ("x", "y", "w", "h"))
        expected = {"original": ((table.width, table.height), rgb_sha),
                    "crop": ((width, height), sha(rgb[y:y+height, x:x+width].tobytes()))}
        for kind, signature in expected.items():
            asset = packet["artifacts"][item["artifact_refs"][kind]]
            data = fetch(client, asset["url"])
            assert sha(data) == asset["sha256"]
            assert image_signature(data) == signature
            targets.add(signature)
            (folder / f"{role}-{kind}.png").write_bytes(data)
            assets.append({"role": role, "kind": kind, "frame": index, "pts": table.pts[index], "sha256": asset["sha256"]})
    assert len(targets) == 6
    return targets, assets


def verified_surfaces(client, aid, source: Path, commitment: dict, evidence: dict, packet: dict, folder: Path) -> dict:
    import pymupdf

    from app.reports import engine
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    folder.mkdir(mode=0o700)
    images, assets = verify_images(client, packet, source, commitment, folder)
    exported = fetch(client, f"/api/analyses/{aid}/export.json")
    assert json.loads(exported)["evidence"] == evidence
    scan(exported.decode())
    (folder / "export.json").write_bytes(exported)
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    report = wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        text = " ".join(" ".join(page.get_text().split()) for page in doc)
        scan(text)
        assert "Refined boundary estimate" in text and "2.900" in text and "3.233" in text
        assert all(f"source frame {index}" in text for index in REPRESENTATIVES.values())
        xrefs = {entry[0] for page in doc for entry in page.get_images(full=True)}
        embedded = {image_signature(doc.extract_image(xref)["image"]) for xref in xrefs}
        assert images <= embedded
    (folder / "report.pdf").write_bytes(pdf)
    pack = body(client.post(f"/api/v2/analyses/{aid}/packages",
                            json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    pack = wait(client, f"/api/v2/packages/{pack['id']}", timeout=120)
    target = folder / "signed-package.zip"
    target.write_bytes(fetch(client, f"/api/v2/packages/{pack['id']}/file"))
    signer = body(client.get("/api/v2/instance/public-key"))
    key = signer["public_key_pem"].encode()
    replay = verify_package(target, public_key_pem=key, expect_key_id=signer["key_id"], source_video=source)
    assert replay.outcome == "verified" and replay.signer_authenticated, replay.as_dict()
    assert any(item.ok and item.name == "original video frames"
               and "Recomputed both paired face boundary motion measurements and verified the source carrier" in item.detail
               for item in replay.checks)
    with ZipFile(target) as package:
        bundle = json.loads(package.read("bundle.json"))
        assert json.loads(package.read("evidence.json")) == evidence
        assert package.read(bundle["reports"][0]["file"]) == pdf
    altered = folder / "changed-original.mp4"
    altered.write_bytes(source.read_bytes() + b"\n")
    assert not verify_package(target, public_key_pem=key, source_video=altered).ok
    return {"exact_source_images": assets, "pdf_api_json_agree": True, "signed_original_replay": True,
            "different_original_rejected": True, "template_set": engine.TEMPLATE_SET}


def run(study: Path, output: Path) -> None:
    roster = fixture_plan(study)
    root = Path(__file__).resolve().parents[2]
    configure(output, root)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT="true",
                      VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="false", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session

    assert settings.data_dir == (output / "app").resolve() and settings.video_face_paired_edge_refinement
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
            owner = auth.new_principal("paired-edge-test-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        for case in (*FILES, "feature_disabled"):
            selected = case if case in FILES else "held_swap"
            source = study / FILES[selected]
            settings.video_face_paired_edge_refinement = case != "feature_disabled"
            aid = checkout(client, stub, source)
            decision, detector = read_result(aid, roster[selected]["sha256"])
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            assert evidence["source"]["sha256"] == roster[selected]["sha256"]
            coarse = [packet for packet in evidence["packets"] if packet.get("face_sequence")]
            boundaries = [packet for packet in coarse if packet.get("face_boundary", {}).get("method") == METHOD]
            row = {"file_sha256": roster[selected]["sha256"], "coarse_packets": len(coarse),
                   "paired_edge_packets": len(boundaries), "version": evidence["version"]}
            print(case, row, flush=True)
            decisions[case] = decision
            if case == "held_swap":
                [packet] = boundaries
                boundary = packet["face_boundary"]
                assert evidence["version"] == VERSION and packet["affects_overall_confidence"] is False
                assert detector["details"]["face_boundary"] == boundary
                assert boundary["anchors"] == REPRESENTATIVES
                time_base = Fraction(evidence["source"]["time_base"])
                assert [boundary[k] * time_base for k in ("start_pts", "end_pts")] == [Fraction(i, 30) for i in EDGES]
                assert [round(boundary[k], 3) for k in ("start_stream_s", "end_stream_s")] == [2.9, 3.233]
                row.update(verified_surfaces(client, aid, source, roster[selected], evidence, packet, output / "held_swap"))
                row.update(known_edit_frames=list(EDGES), measured_frames=list(EDGES), source_time_s=[2.9, 3.233],
                           representative_frames=REPRESENTATIVES)
                write_json(output / "browser-input.json", {"analysis_id": aid, "packet_id": packet["id"]})
            else:
                assert not boundaries and detector["details"].get("face_boundary", {}).get("method") != METHOD
                assert bool(coarse) == (case != "short_swap")
            cases[case] = row
    assert decisions["held_swap"] == decisions["feature_disabled"]
    write_json(output / "receipt.json", {"cases": cases, "unpaid_jobs": 0, "simulated_paid_jobs": len(cases),
                                         "overall_decision_identical_flag_on_off": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Retained HEVC transfer fixtures and commitments")
    parser.add_argument("output", type=Path, help="New isolated output directory")
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
