# SPDX-License-Identifier: Apache-2.0
"""Verify the near-end face check after simulated checkout and in signed reports."""

import argparse
import os
from fractions import Fraction
from pathlib import Path

from scripts.video_common_sources_v89 import read, write
from scripts.video_face_clip_boundary_transfer import review
from scripts.video_face_distinct_avc import BASE_FLAGS
from scripts.video_short_return_encoded_app import ALLOW, PaymentStub, body, checkout, configure, read_result


def run(study, output, split, *, rate=30):
    panel = read(study / f"{split}.json")
    review(panel, rate=rate)
    media = {(row["group"], row["condition"]): row for row in read(study / split / "media.json")}

    def example(condition):
        return next(row for row in panel["rows"] if row["condition"] == condition
                    and row["readings"]["current"]["selected"] is None
                    and ([row["readings"]["boundary"]["selected"]] == row["truth"] if row["truth"]
                         else row["readings"]["boundary"]["selected"] is None))

    cases = [example(condition) for condition in ("early_swap", "late_swap", "late_cut")]
    configure(output, Path(__file__).resolve().parents[2])
    os.environ.update(BASE_FLAGS, VERITAS_VIDEO_WHOLE_FACE_REVIEW="false",
                      VERITAS_VIDEO_FACE_MOVING_AVC_REVIEW="true", VERITAS_VIDEO_FACE_BOUNDARY_AVC_REVIEW="false")

    import cv2
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence.face_short_proposal_binding import BOUNDARY_AVC_METHOD
    from app.evidence.video_packet_versions import BOUNDARY_AVC_PROPOSAL_VERSION
    from app.main import app
    from app.storage.db import get_session
    from scripts.video_face_unrouted_short_app import verified_surfaces

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    registry.enabled = lambda modality, profile: [item for item in enabled(modality, profile) if item in ALLOW]
    assert set(registry.enabled("video", "full")) == ALLOW
    stub = PaymentStub()
    service._stripe = stub
    baseline, receipts = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-clip-boundary-review", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        for active in (False, True):
            settings.video_face_boundary_avc_review = active
            if rate != 30:
                settings.video_face_short_avc_review = active
            for row in cases:
                name = row["condition"]
                source_row = media[row["group"], name]
                source = study / split / source_row["file"]
                aid = checkout(client, stub, source, filename="local-clip-boundary.mp4")
                decision, _ = read_result(aid, row["sha256"])
                evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
                new = [packet for packet in evidence["packets"] if packet.get("semantics") == BOUNDARY_AVC_METHOD]
                existing = [packet["face_return"] for packet in evidence["packets"]
                            if "face_return" in packet and packet not in new]
                if not active:
                    assert not new
                    baseline[name] = decision, existing
                    continue
                assert (decision, existing) == baseline[name]
                assert len(new) == len(row["truth"]), (name, evidence.get("coverage"), existing)
                if new:
                    assert evidence["version"] == BOUNDARY_AVC_PROPOSAL_VERSION
                    packet = new[0]
                    change = packet["face_return"]
                    interval = [change["start_frame_index"], change["end_frame_index"]]
                    assert [interval] == row["truth"] and packet["affects_overall_confidence"] is False
                    ticks = [change["start_pts"], change["end_pts"]]
                    assert ticks == [source_row["clock"]["pts"][i] for i in interval]
                    anchors = {role: change["anchors"][role] for role in ("before", "during", "after")}
                    timing = f"{change['start_stream_s']:.3f}-{change['end_stream_s']:.3f}"
                    surfaces = verified_surfaces(client, aid, source, source_row, evidence, packet, output / name,
                                                 representatives=anchors, initial_coverage=None, expected_range=timing)
                    receipts[name] = {"range": interval, "source_seconds": timing, **surfaces}
                    if rate != 30:
                        receipts[name].update(time_base=change["time_base"], source_ticks=ticks)
                else:
                    receipts[name] = {"range": None, "added_flags": 0}
                print(name, receipts[name], flush=True)
    result = {"cases": receipts, "split": split, "simulated_paid_runs": len(stub.sessions),
              "unpaid_jobs": 0, "overall_assessments_retained": True}
    if rate != 30:
        result["frame_rate"] = str(rate)
    write(output / "receipt.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("development", "held"), default="held")
    parser.add_argument("--rate", type=Fraction, default=Fraction(30))
    args = parser.parse_args()
    run(args.study, args.output, args.split, rate=args.rate)
