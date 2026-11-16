"""Replay mirrored footage through the prepaid app, the browser and signed exports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from app.evidence import picture_sequence as binding
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_face_paired_edge_hevc_app import image_signature, sha_path, write_json
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]
KEY = binding.MOTION_REFLECTED_KEY
PACKET = binding.MOTION_REFLECTED_PACKET_ID
MIME = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
HD_REPLAY_PROFILES = ("avc-1080-2997", "webm-720-2997", "hevc-portrait-30", "avc-portrait-23976")


def examples(rows, groups):
    for label in groups:
        group = [row for row in rows if row.get("delivery_profile", row["codec"]) == label]
        positive = next(row for row in group if row["role"] == "later_mirrored" and row["exact"]
                        and all(other["reflected"] is None for other in group
                                if other["source"] == row["source"] and other["role"] in {"original", "in_place_mirror"}))
        yield positive, False
        yield positive, True
        for role in ("original", "in_place_mirror"):
            yield next(row for row in group if (row["source"], row["role"]) == (positive["source"], role)), True


def exports(client, aid, evidence, original, packet, output):
    import pymupdf

    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    route = f"/api/v2/analyses/{aid}"
    report = body(client.post(f"{route}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    if packet:
        pair = packet["picture_sequence"]
        images = set()
        for frame in packet["frames"]:
            asset = packet["artifacts"][frame["artifact_refs"]["original"]]
            signature = image_signature(fetch(client, asset["url"]))
            assert signature == ((evidence["source"]["width"], evidence["source"]["height"]), frame["anchor"]["decoded_rgb_sha256"])
            images.add(signature)
        with pymupdf.open(stream=pdf, filetype="pdf") as document:
            words = " ".join(" ".join(page.get_text().split()) for page in document)
            scan(words)
            ranges = " and ".join(f"{pair[name]['start_stream_s']:.3f}-{pair[name]['end_stream_s']:.3f} s"
                                  for name in ("first", "second"))
            assert f"Mirrored ranges {ranges}" in words
            assert binding.observation(pair) in words
            xrefs = {image[0] for page in document for image in page.get_images(full=True)}
            assert images <= {image_signature(document.extract_image(xref)["image"]) for xref in xrefs}
    package = body(client.post(f"{route}/packages",
                               json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{package['id']}", timeout=120)
    output.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    key = body(client.get("/api/v2/instance/public-key"))
    result = verify_package(output, public_key_pem=key["public_key_pem"].encode(), expect_key_id=key["key_id"], source_video=original)
    assert result.outcome == "verified" and result.signer_authenticated, result.as_dict()
    assert any(check.ok and check.name == "original video frames" for check in result.checks)
    with ZipFile(output) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == pdf
    return len(packet["frames"]) if packet else 0


def run(clips, held, output, browser, *, groups=("h264", "vp9")):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"),
                      VERITAS_VIDEO_RETIMED_SEQUENCE_REVIEW="true", VERITAS_LOG_LEVEL="WARNING")

    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    # Keep the regular worker and export pipeline. Learned detectors are outside this smoke test.
    registry = get_registry()
    registry.enabled = lambda modality, profile: []
    stub = PaymentStub()
    service._stripe = stub
    receipts, browser_cases, assessments = [], [], {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            examiner = auth.new_principal("mirrored-footage-local-test", "examiner")
            db.add(examiner)
            db.flush()
            key, _ = auth.issue_api_key(db, examiner)
            db.commit()
        client.headers["X-API-Key"] = key
        for row, enabled in examples(held, groups):
            settings.video_reflected_sequence_review = enabled
            profile = row.get("delivery_profile")
            source = clips / (profile or "") / row["delivery"]
            assert sha_path(source) == row["sha256"]
            aid = checkout(client, stub, source, media_type=MIME[source.suffix], filename=source.name)
            route = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{route}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            for public in (exported, fetch(client, route)):
                scan(public.decode())
            assert json.loads(exported)["evidence"] == evidence
            name = f"{source.stem}-{profile or row['codec']}-{'on' if enabled else 'off'}"
            check = evidence["picture_hold"].get(KEY)
            assert (check is not None) == enabled
            packets = [packet for packet in evidence["packets"] if packet["id"] == PACKET]
            positive = enabled and row["role"] == "later_mirrored"
            assert len(packets) == int(positive)
            packet = packets[0] if packets else None
            if check:
                assert check["status"] == ("matched" if positive else "no_match")
                assert check["schema"] == (binding.MOTION_REFLECTED_HD_SCHEMA if profile else binding.MOTION_REFLECTED_SCHEMA)
            if packet:
                assert packet["affects_overall_confidence"] is False
                assert list(binding.endpoints(packet["picture_sequence"])) == [tuple(anchor) for anchor in row["anchors"]]
            decision = tokens_assessed(aid)
            if not enabled:
                assessments[source] = decision
            elif positive:
                assert assessments[source] == decision
            count = exports(client, aid, evidence, source, packet, output / f"{name}.zip") if enabled else 0
            clock = [[packet["picture_sequence"][side][f"{bound}_stream_s"] for bound in ("start", "end")]
                     for side in ("first", "second")] if packet else []
            receipts.append({"fixture": name, "source_sha256": row["sha256"], "enabled": enabled, "ranges_s": clock,
                             "source_frames_in_pdf": count, "signed_original_verified": enabled})
            if enabled:
                browser_cases.append({"analysis_id": aid, "name": name, "matched": positive, "ranges_s": clock,
                                      "checked_frames": check["frames_scanned"], "seek_verified_in_browser": row["codec"] != "hevc",
                                      "dimensions": [evidence["source"][key] for key in ("width", "height")],
                                      "frames": [entry["anchor"] for entry in packet["frames"]] if packet else []})
            print(name, clock, flush=True)
        ui = browser_check(output, app, key, browser_cases, scenario="mirrored-footage-replay.mjs") if browser else None
    write_json(output / "receipt.json", {"cases": receipts, "browser": ui, "paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                                         "unchanged_assessment_on_matched_pairs": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--hd", action="store_true", help="Use the combined numeric panel for the HD replay")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if args.hd:
        rows = [row | {"delivery_profile": panel["delivery_profile"], "codec": panel["method"]["codec"]}
                for panel in receipt["profiles"] for row in panel["cases"]]
        run(args.clips, rows, args.output, args.browser, groups=HD_REPLAY_PROFILES)
    else:
        run(args.clips, receipt["cases"], args.output, args.browser)
