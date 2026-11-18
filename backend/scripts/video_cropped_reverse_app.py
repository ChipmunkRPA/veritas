"""Verify centered, reversed footage through local checkout and signed exports."""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from scripts import video_cropped_reverse_panel as panel
from scripts.image_jpeg_turned_copy_app import browser_check
from scripts.video_reflected_sequence_app import exports
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch
from scripts.writing_character_view_app import tokens_assessed

ROOT = Path(__file__).resolve().parents[2]


def examples(value):
    by_id = {row["id"]: row for row in value["source_roster"]}
    cases = value["rows"]
    for order in ("first", "second"):
        hit = next(row for row in cases if row["truth"] and row["truth"]["wider_range"] == order
                   and row["role"] == "center_reverse_85" and row["candidate"]
                   and panel._useful(row["candidate"], row["truth"]))
        control = next(row for row in cases if row["source"] == hit["source"] and row["role"] == "in_place_reverse_crop")
        assert control["candidate"] is None and not control["baseline"]
        yield by_id[hit["source"]], hit, control


def run(receipt, source_folder, output):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_FRONTEND_DIR=str(ROOT / "frontend/dist"), VERITAS_LOG_LEVEL="WARNING")

    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence import picture_sequence as binding
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    get_registry().enabled = lambda modality, profile: []
    stub = PaymentStub()
    service._stripe = stub
    value = panel.load(receipt)
    receipts, browser_cases = [], []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as database:
            principal = auth.new_principal("centered-cropped-reverse-local-test", "examiner")
            database.add(principal)
            database.flush()
            key, _ = auth.issue_api_key(database, principal)
            database.commit()
        client.headers["X-API-Key"] = key
        for source, hit, control in examples(value):
            path = source_folder / f"{source['id']}.webm"
            assert panel.sha(path) == source["sha256"]
            prepared = panel._prepare(source, path)
            clips = {}
            for row in (hit, control):
                clip = output / row["file"]
                panel.render(prepared, source["sha256"], row["role"], clip)
                assert panel.sha(clip) == row["file_sha256"]
                clips[row["role"]] = clip
            del prepared
            baseline = None
            for row, enabled in ((hit, False), (hit, True), (control, True)):
                settings.video_cropped_sequence_review = enabled
                clip = clips[row["role"]]
                case = f"{clip.stem}-{'on' if enabled else 'off'}"
                aid = checkout(client, stub, clip, filename=clip.name)
                route = f"/api/v2/analyses/{aid}"
                evidence = body(client.get(f"{route}/evidence-packets"))
                result = fetch(client, route)
                exported = fetch(client, f"/api/analyses/{aid}/export.json")
                assert json.loads(exported)["evidence"] == evidence
                for document in (result, exported):
                    scan(document.decode())
                record = evidence["picture_hold"].get(binding.MOTION_CROP_KEY)
                passed = enabled and row is hit
                assert (record is not None) == enabled
                assert not record or record["status"] == ("matched" if passed else "no_match")
                packets = [packet for packet in evidence["packets"] if packet["id"] == binding.MOTION_CROP_PACKET_ID]
                assert len(packets) == int(passed)
                assessment = tokens_assessed(aid)
                if not enabled:
                    baseline = assessment
                elif passed:
                    assert assessment == baseline
                packet = packets[0] if packets else None
                frames, crop_dims, ranges = [], [], []
                if packet:
                    pair = packet["picture_sequence"]
                    expected = row["candidate"]
                    assert record["schema"] == binding.MOTION_CROP_REVERSE_SCHEMA
                    assert pair["order"] == "reverse" and packet["affects_overall_confidence"] is False
                    assert pair["framing"] == {field: expected[field] for field in ("wider_range", "crop_box", "alignment")}
                    assert pair["supporting_edge_offsets"] == expected["support"]
                    frames = [item["anchor"] for item in packet["frames"]]
                    for item, label in zip(frames, ("first_start", "first_end", "second_start", "second_end"), strict=True):
                        assert [item["frame_index"], item["pts"], item["decoded_rgb_sha256"]] == expected["anchors"][label]
                    ranges = [[pair[side]["start_stream_s"], pair[side]["end_stream_s"]] for side in ("first", "second")]
                    crop_dims = [[c["w"], c["h"]] for item in packet["frames"] if (c := item["context_region"])]
                    assert len(crop_dims) == 2
                    for change in ("order", "crop"):
                        altered = deepcopy(evidence)
                        wrong = next(item for item in altered["packets"] if item["id"] == packet["id"])
                        if change == "order":
                            saved_pair = deepcopy(wrong["picture_sequence"])
                            saved_pair.pop("order")
                            wrong["picture_sequence"] = saved_pair
                            altered["picture_hold"][binding.MOTION_CROP_KEY]["pair"] = saved_pair
                        else:
                            next(item["context_region"] for item in wrong["frames"] if item["context_region"])["x"] += 1
                        try:
                            binding.validate(altered["picture_hold"], altered)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError(f"Changed {change} passed source-evidence validation")
                verified = exports(client, aid, evidence, clip, packet, output / f"{case}.zip",
                                   locator="Ranges with matching footage after cropping") if enabled else 0
                receipts.append({"fixture": case, "enabled": enabled, "matched": passed, "source_sha256": row["file_sha256"],
                                 "ranges_s": ranges, "wider_occurrence": packet["picture_sequence"]["framing"]["wider_range"] if packet else None,
                                 "pdf_frames": verified, "pdf_crops": len(crop_dims), "signed_original_verified": enabled})
                if enabled:
                    browser_cases.append({"mode": "crop_reverse", "analysis_id": aid, "name": case, "matched": passed,
                                          "ranges_s": ranges, "checked_frames": record["frames_scanned"], "seek_verified_in_browser": True,
                                          "dimensions": list(panel.DIMENSIONS), "frames": frames, "crop_dimensions": crop_dims})
                print(case, ranges, flush=True)
        browser = browser_check(output, app, key, browser_cases, scenario="mirrored-footage-replay.mjs")
    panel.save(output / "receipt.json", {"cases": receipts, "browser": browser, "paid_analyses": len(stub.sessions),
                                         "unpaid_jobs": 0, "unchanged_ai_estimate": True, "altered_crop_and_order_rejected": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("source_folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.receipt, args.source_folder, args.output)
