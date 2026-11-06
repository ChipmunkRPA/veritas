"""Compare the production adapter with frozen sequence receipts on consumed inputs.

This checks implementation parity, source bindings and activation boundaries. It
is not a new accuracy study and never changes an operating point or calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    import cv2
    import structlog
    import torch

    from app.core.canonical import canonical_json
    from app.detectors.fakes import InMemoryExhibitWriter, InMemoryMedia
    from app.detectors.video.adapters.face_sequence import FaceSequenceReviewDetector, SequenceInputs
    from app.detectors.video.decode import FrameTable
    from app.detectors.video.face_sequence import REVIEW_POINT
    from app.detectors.video.frame_store import FrameStore
    from app.engine.boundary import Deadline, InMemoryRunStore, run_detector
    from app.engine.detector import Profile, RunContext, StageOutputs
    from app.engine.fetchers import ensure_verified
    from app.engine.models import get_model_manager
    from app.engine.registry import get_registry
    from app.evidence.face_interval_binding import validate_sequence_record

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with args.baseline.open() as stream:
        baseline_header, *cases = [json.loads(line) for line in stream]
    with args.reference.open() as stream:
        reference_header, *references = [json.loads(line) for line in stream]
    assert reference_header["baseline_sha256"] == sha(args.baseline)
    assert reference_header["manifest_sha256"] == baseline_header["manifest_sha256"]
    assert len(cases) == len(references) and [r["id"] for r in cases] == [r["id"] for r in references]
    registry = get_registry()
    detector = FaceSequenceReviewDetector()
    for model_id in detector.spec.requirements.model_ids:
        model = registry.manifest.model(model_id)
        assert model is not None and model.licence.class_ == "commercial"
        assert reference_header["weights"][model_id] == [{"path": f.path, "sha256": f.sha256} for f in model.source.files]
        ensure_verified(model)
    manager = get_model_manager()
    runs, counts = InMemoryRunStore(), {"cases": 0, "selected": 0, "unassessed": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps({"scope": __doc__, "baseline_sha256": sha(args.baseline),
            "reference_sha256": sha(args.reference), "device": args.device,
            "adapter_sha256": sha(Path(__file__).parents[1]/"app/detectors/video/adapters/face_sequence.py"),
            "math_sha256": sha(Path(__file__).parents[1]/"app/detectors/video/face_sequence.py"),
            "planned_cases": min(args.limit or len(cases), len(cases))})+"\n")
        for case, reference in list(zip(cases, references, strict=True))[:args.limit]:
            path = args.media_root/case["path"]
            assert sha(path) == case["sha256"] == reference["source_sha256"]
            table = FrameTable.from_json(case["table"])
            store = FrameStore(path, table, max_bytes=64*1024*1024)
            inputs = SequenceInputs(table, store, tuple(p["frame_index"] for p in case["points"]), tuple(case["exhibits"]))
            context = RunContext(analysis_id=case["id"], media=InMemoryMedia(case["sha256"], "video", path=path),
                profile=Profile.FULL, device=args.device, dtype=None, lease=None, outputs=StageOutputs(face_review=inputs),
                deadline=Deadline(120), exhibits=InMemoryExhibitWriter(), log=structlog.get_logger())
            actual = run_detector(detector, context, store=runs, models=manager)
            expected = reference["raw_windows"]
            selected = None
            if expected:
                assert actual.status == "ok", (case["id"], actual.status, actual.reason)
                assert abs(actual.aggregate.raw-expected[0]["response"]) <= 2e-6, (case["id"], actual.aggregate.raw, expected[0]["response"])
                selected = actual.details["face_sequence"]
                assert (selected is not None) == (expected[0]["response"] >= REVIEW_POINT)
                if selected is not None:
                    assert selected["start_stream_s"] == expected[0]["start_s"] and selected["end_stream_s"] == expected[0]["end_s"]
                    assert sum(selected["centers"].values(), []) == expected[0]["centers"]
                    source = {"time_base": table.time_base, "frame_count": len(table), "width": table.width, "height": table.height}
                    # Exercise the stored six-decimal canonical form as well as the live floats.
                    validate_sequence_record(json.loads(canonical_json(selected)), source, table_pts=table.pts, interpolated=table.interpolated)
                    anchors = {a["frame_index"]: a for a in store.native_anchors()}
                    assert all(all(item[key] == anchors[item["frame_index"]][key] for key in ("pts", "decoded_rgb_sha256"))
                               for item in selected["frame_inputs"])
                    counts["selected"] += 1
            else:
                assert actual.status == "not_applicable", (case["id"], actual.status, actual.reason)
                counts["unassessed"] += 1
            counts["cases"] += 1
            output.write(json.dumps({"id": case["id"], "status": actual.status, "source_sha256": case["sha256"],
                "raw": actual.aggregate.raw if actual.aggregate is not None else None,
                "selected": selected, "numerical_parity": True, "source_binding": True})+"\n")
            output.flush()
            print(json.dumps({"id": case["id"], "status": actual.status, **counts}), flush=True)
        output.write(json.dumps({"complete": counts["cases"] == len(cases), **counts})+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("media_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
