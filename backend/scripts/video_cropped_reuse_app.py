"""Verify cropped footage evidence through prepaid analysis and its exports."""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from app.evidence import picture_sequence as binding
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_face_paired_edge_hevc_app import sha_path, write_json
from scripts.video_reflected_sequence_app import exports
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def examples(report):
    if "plan" not in report:
        profile = report["profile"]
        role = "offset_reuse_tandem" if profile == "avc-720" else "offset_reuse"
        positive = next(row for row in report["cases"] if row["role"] == role and row["candidate"]
                        and row["candidate"]["ranges"] == row["truth"]["ranges"])
        control = next(row for row in report["cases"] if row["source"] == positive["source"] and row["role"] == "in_place_crop")
        assert not control["candidate"]
        return report, positive, control
    plan = report["plan"]
    wanted = "centered" if plan["profile"] == "avc-360" else "landmarks"
    roles = {"center_reuse"} if wanted == "centered" else {"center_fractional_reuse_tandem", "between_grid_reuse_tandem"}
    positive = next(row for row in plan["cases"] if row["role"] in roles and row["selected"] == wanted
                    and row["results"][wanted]["ranges"] == row["truth"]["ranges"])
    control = next(row for row in plan["cases"] if row["source"] == positive["source"] and row["role"] == "in_place_crop")
    assert not control["selected"]
    candidate = positive["results"][wanted] | {"alignment": wanted}
    return plan, positive | {"candidate": candidate, "candidate_anchors": candidate["pts"]}, control


def run(clips, reports, output, browser):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"), VERITAS_LOG_LEVEL="WARNING")

    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    get_registry().enabled = lambda modality, profile: []
    stub = PaymentStub()
    service._stripe = stub
    receipts, browser_cases = [], []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as database:
            principal = auth.new_principal("cropped-footage-local-test", "examiner")
            database.add(principal)
            database.flush()
            key, _ = auth.issue_api_key(database, principal)
            database.commit()
        client.headers["X-API-Key"] = key
        for path in reports:
            report, matched, control = examples(json.loads(path.read_text()))
            baseline = None
            for row, enabled in ((matched, False), (matched, True), (control, True)):
                settings.video_cropped_sequence_review = enabled
                source = clips / path.stem.removesuffix("-result") / row["delivery"]
                assert sha_path(source) == row["sha256"]
                aid = checkout(client, stub, source, filename=source.name)
                route = f"/api/v2/analyses/{aid}"
                evidence = body(client.get(f"{route}/evidence-packets"))
                result = fetch(client, route)
                exported = fetch(client, f"/api/analyses/{aid}/export.json")
                for public in (result, exported):
                    scan(public.decode())
                assert json.loads(exported)["evidence"] == evidence
                check = evidence["picture_hold"].get(binding.MOTION_CROP_KEY)
                positive = enabled and row is matched
                assert (check is not None) == enabled
                assert not check or check["status"] == ("matched" if positive else "no_match")
                packets = [packet for packet in evidence["packets"] if packet["id"] == binding.MOTION_CROP_PACKET_ID]
                assert len(packets) == int(positive)
                decision = tokens_assessed(aid)
                if not enabled:
                    baseline = decision
                elif positive:
                    assert decision == baseline
                packet = packets[0] if packets else None
                name = f"{report['profile']}-{source.stem}-{'on' if enabled else 'off'}"
                frames, crop_sizes, ranges = [], [], []
                if packet:
                    pair = packet["picture_sequence"]
                    assert packet["affects_overall_confidence"] is False
                    candidate = row["candidate"]
                    assert pair["framing"] == {field: candidate[field] for field in ("wider_range", "crop_box", "alignment")
                                                if field in candidate}
                    assert check["schema"] == (binding.MOTION_CROP_ALIGNED_SCHEMA if "alignment" in candidate else binding.MOTION_CROP_SCHEMA)
                    assert [[pair[side]["pts"][0], pair[side]["pts"][-1]] for side in ("first", "second")] == row["candidate_anchors"]
                    frames = [frame["anchor"] for frame in packet["frames"]]
                    crop_sizes = [[region["w"], region["h"]] for frame in packet["frames"] if (region := frame.get("context_region"))]
                    assert len(frames) == 4 and len(crop_sizes) == 2
                    ranges = [[pair[side]["start_stream_s"], pair[side]["end_stream_s"]] for side in ("first", "second")]
                    altered = deepcopy(evidence)
                    changed = next(item for item in altered["packets"] if item["id"] == packet["id"])
                    next(frame["context_region"] for frame in changed["frames"] if frame["context_region"])["x"] += 1
                    try:
                        binding.validate(altered["picture_hold"], altered)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Changed crop coordinates passed evidence validation")
                verified = exports(client, aid, evidence, source, packet, output / f"{name}.zip",
                                   locator="Ranges with matching footage after cropping") if enabled else 0
                receipts.append({"fixture": name, "enabled": enabled, "matched": positive, "source_sha256": row["sha256"],
                                 "ranges_s": ranges, "original_frames_in_pdf": verified, "crops_in_pdf": len(crop_sizes),
                                 "signed_original_verified": enabled})
                if enabled:
                    browser_cases.append({"mode": "crop", "analysis_id": aid, "name": name, "matched": positive,
                                          "ranges_s": ranges, "checked_frames": check["frames_scanned"], "seek_verified_in_browser": True,
                                          "dimensions": report["dims"], "frames": frames, "crop_dimensions": crop_sizes})
                print(name, ranges, flush=True)
        ui = browser_check(output, app, key, browser_cases, scenario="mirrored-footage-replay.mjs") if browser else None
    write_json(output / "receipt.json", {"cases": receipts, "browser": ui, "paid_analyses": len(stub.sessions), "unpaid_jobs": 0,
                                         "unchanged_assessment_on_matched_pairs": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    run(args.clips, args.reports, args.output, args.browser)
