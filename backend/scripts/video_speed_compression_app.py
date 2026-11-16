"""Verify the recompressed retiming check through the local prepaid workflow."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from app.evidence import picture_sequence as binding
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_face_paired_edge_hevc_app import sha_path, write_json
from scripts.video_face_temporal_fit import read
from scripts.video_reflected_sequence_app import exports
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch
from scripts.video_speed_compression_panel import useful, verify
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def examples(rows):
    controls = {(row["source"], row["role"]): row for row in rows if not row["truth"] and not row["other_repeat"]
                and not (row["current"] or row["added"])}
    extras = [row for row in rows if row["role"].endswith("_tandem") and useful(row["added"], row["truth"])
              and all((row["source"], role) in controls for role in ("original_tandem", "in_place_speed"))]
    first = extras[0]
    second = next(row for row in extras if row["source"] != first["source"])
    previous = next(row for row in rows if useful(row["current"], row["truth"]))
    return ((first, False), (first, True), (second, True), (previous, True),
            (controls[first["source"], "original_tandem"], True), (controls[first["source"], "in_place_speed"], True))


def run(media, receipt_path, output, *, browser):
    receipt = read(receipt_path)
    verify(receipt)
    output = output.resolve()
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"),
                      VERITAS_VIDEO_RETIMED_SEQUENCE_REVIEW="true", VERITAS_LOG_LEVEL="WARNING")

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session
    from fastapi.testclient import TestClient
    from scripts.verify_multimedia_exports import scan

    get_registry().enabled = lambda modality, profile: []
    stub = PaymentStub()
    service._stripe = stub
    results, browser_cases = [], []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as database:
            principal = auth.new_principal("retimed-compression-local-test", "examiner")
            database.add(principal)
            database.flush()
            key, _ = auth.issue_api_key(database, principal)
            database.commit()
        client.headers["X-API-Key"] = key
        baseline = None
        for row, enabled in examples(receipt["cases"]):
            settings.video_retimed_sequence_review = enabled
            source = media / row["delivery"]
            assert sha_path(source) == row["sha256"]
            aid = checkout(client, stub, source, filename=source.name)
            api = f"/api/v2/analyses/{aid}"
            evidence = body(client.get(f"{api}/evidence-packets"))
            exported = fetch(client, f"/api/analyses/{aid}/export.json")
            for public in (exported, fetch(client, api)):
                scan(public.decode())
            assert evidence["source"]["sha256"] == row["sha256"] and json.loads(exported)["evidence"] == evidence
            review = evidence["picture_hold"].get(binding.MOTION_RETIMED_KEY)
            assert enabled == (review is not None)
            packets = [packet for packet in evidence["packets"] if packet["id"] == binding.MOTION_RETIMED_PACKET_ID]
            expected = enabled and (row["added"] or row["current"])
            assert len(packets) == bool(expected)
            packet = packets[0] if packets else None
            ranges, frames = [], []
            if review:
                assert review["status"] == ("matched" if expected else "no_match")
                assert review["schema"] == (binding.MOTION_RETIMED_SCHEMA if row["current"]
                                            else binding.MOTION_RETIMED_COMPRESSED_SCHEMA)
            if packet:
                pair = packet["picture_sequence"]
                assert packet["affects_overall_confidence"] is False
                frames = [exhibit["anchor"] for exhibit in packet["frames"]]
                for i, side in enumerate(("first", "second")):
                    span = pair[side]
                    assert [span["start_frame_index"], span["end_frame_index"], pair["frame_steps"][side]] == expected[i]
                    ranges.append([span["start_stream_s"], span["end_stream_s"]])
                    assert [[item["pts"], item["decoded_rgb_sha256"]] for item in frames[2*i:2*i+2]] == row["anchors"][i]
                altered = deepcopy(evidence)
                altered["picture_hold"][binding.MOTION_RETIMED_KEY]["parameters"]["luma_grid"] = [4, 4]
                try:
                    binding.validate(altered["picture_hold"], altered)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Changed retiming parameters passed evidence validation")
            assessment = tokens_assessed(aid)
            if not enabled:
                baseline = (row["sha256"], assessment)
            elif baseline is not None and row["sha256"] == baseline[0]:
                assert assessment == baseline[1]
            name = f"{source.stem}-{'on' if enabled else 'off'}"
            originals = exports(client, aid, evidence, source, packet, output / f"{name}.zip",
                                locator="Ranges at two playback speeds") if enabled else 0
            results.append({"fixture": name, "schema": review["schema"] if review else None, "source_sha256": row["sha256"],
                            "source_ranges_s": ranges, "original_frames_verified_in_pdf": originals,
                            "signed_original_verified": enabled, "api_and_json_agree": True})
            if enabled:
                browser_cases.append({"mode": "retimed", "analysis_id": aid, "name": name, "matched": bool(expected),
                                      "ranges_s": ranges, "checked_frames": review["frames_scanned"],
                                      "seek_verified_in_browser": True, "dimensions": [640, 360], "frames": frames})
            print(name, ranges, flush=True)
        ui = browser_check(output, app, key, browser_cases, scenario="mirrored-footage-replay.mjs") if browser else None
    write_json(output / "receipt.json", {"cases": results, "browser": ui, "paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                                         "overall_assessment_unchanged_with_additional_check": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("media", "receipt", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    run(args.media, args.receipt, args.output, browser=args.browser)
