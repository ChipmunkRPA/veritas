"""Independently decode every compared video input and replay native measurement geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.audit_mouth_probe_v6 import check_chip, detector
from scripts.liveportrait_mouth_benchmark import digest
from scripts.run_mouth_learning_v8 import check_lock, write


def audit(lock_path, receipt, output):
    lock = check_lock(lock_path)
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    expected = []
    for panel in lock["development"]:
        _, *references = [json.loads(line) for line in Path(panel["receipt"]).read_text().splitlines()]
        for case, reference in zip(
            json.loads(Path(panel["manifest"]).read_text())["items"], references, strict=True
        ):
            expected.append((case, reference, Path(panel["manifest"]).parent))
    if header["lock_sha256"] != digest(lock_path) or [r["id"] for r in rows] != [
        r[0]["id"] for r in expected
    ]:
        raise ValueError("Every frozen source outcome is required")
    face_detector = detector()
    records = []
    for row, (case, reference, root) in zip(rows, expected, strict=True):
        record = {"id": row["id"], "status": "failed", "verified_inputs": 0}
        try:
            path = root / case["path"]
            if (
                row["status"] != "ok"
                or not row.get("baseline_replay_parity")
                or row["base_windows"] != reference["base_windows"]
                or row["sha256"] != case["sha256"]
                or digest(path) != case["sha256"]
            ):
                raise ValueError("Changed source/baseline or incomplete comparison")
            table = probe_frames(path)
            bindings = row["model_inputs"]
            indices = [r["frame_index"] for r in bindings]
            selected = {
                i + k for i in reference["sampling_centers"] if 0 < i < len(table) - 1 for k in (-1, 0, 1)
            }
            if indices != sorted(selected):
                raise ValueError("Input schedule/order differs from the baseline native triplets")
            for binding, frame in zip(bindings, decode_frames(path, table, indices), strict=True):
                if (
                    binding["frame_index"] != frame.index
                    or binding["pts"] != frame.pts
                    or binding["time_base"] != table.time_base
                    or binding["time_s"] != table.seconds(frame.index)
                    or binding["decoded_rgb_sha256"] != hashlib.sha256(frame.rgb.tobytes()).hexdigest()
                ):
                    raise ValueError("Learned response is bound to different native pixels or source time")
                check_chip(frame.rgb, binding, face_detector)
                record["verified_inputs"] += 1
            record["status"] = "ok"
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)
    body = {
        "schema": "chipmunk.mouth-learning-video-source-audit/8",
        "lock_sha256": digest(lock_path),
        "score_receipt_sha256": digest(receipt),
        "implementation_sha256": digest(Path(__file__)),
        "shared_auditor_sha256": digest(Path(__file__).with_name("audit_mouth_probe_v6.py")),
        "planned": 88,
        "passed": sum(r["status"] == "ok" for r in records),
        "verified_inputs": sum(r["verified_inputs"] for r in records),
        "cases": records,
        "scope": "Native RGB/PTS/alignment/mouth-chip/model-input correspondence; not outside review or causal attribution",
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k != "cases"}))
    if body["passed"] != 88:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    audit(args.lock, args.receipt, args.output)
