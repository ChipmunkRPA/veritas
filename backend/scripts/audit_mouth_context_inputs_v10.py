"""Independently replay both heads' source bindings across calibration and development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.audit_mouth_probe_v6 import check_chip, detector
from scripts.liveportrait_mouth_benchmark import digest
from scripts.run_mouth_context_v10 import check_lock
from scripts.run_mouth_learning_v8 import write
from scripts.run_mouth_paired_v9 import native_parity


def audit(lock_path, calibration, development, output):
    lock = check_lock(lock_path)
    face_detector = detector()
    records = []
    for stage, receipt in (("calibration", calibration), ("development", development)):
        header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
        if header["lock_sha256"] != digest(lock_path) or header["stage"] != stage:
            raise ValueError("Wrong frozen source/context receipt")
        expected = []
        if stage == "calibration":
            manifest = Path(lock["calibration_manifest"])
            expected = [(r, manifest.parent) for r in json.loads(manifest.read_text())["items"]]
        else:
            for panel in lock["development"]:
                manifest = Path(panel["manifest"])
                expected.extend((r, manifest.parent) for r in json.loads(manifest.read_text())["items"])
        if [r["id"] for r in rows] != [r[0]["id"] for r in expected]:
            raise ValueError("Every planned case is required for source replay")
        for row, (case, root) in zip(rows, expected, strict=True):
            record = {
                "id": row["id"],
                "stage": stage,
                "status": "failed",
                "native_inputs": 0,
                "head_input_bindings": 0,
            }
            try:
                path = root / case["path"]
                if (
                    row.get("failure_reason")
                    or digest(path) != case["sha256"]
                    or row["sha256"] != case["sha256"]
                ):
                    raise ValueError("Source changed or scoring failed")
                a, b = row["heads"]["v8"], row["heads"]["v9"]
                if any(
                    head["status"] != "ok" or not head.get("baseline_replay_parity") for head in (a, b)
                ) or not native_parity(a["model_inputs"], b["model_inputs"]):
                    raise ValueError("Heads have different native source inputs or incomplete replays")
                table = probe_frames(path)
                indices = [r["frame_index"] for r in a["model_inputs"]]
                expected_indices = {
                    i + k
                    for i in row["current_reference"]["sampling_centers"]
                    if 0 < i < len(table) - 1
                    for k in (-1, 0, 1)
                }
                if indices != sorted(expected_indices):
                    raise ValueError("Input schedule differs from the production native triplets")
                for binding, frame in zip(
                    a["model_inputs"], decode_frames(path, table, indices), strict=True
                ):
                    if (
                        binding["frame_index"] != frame.index
                        or binding["pts"] != frame.pts
                        or binding["time_base"] != table.time_base
                        or binding["time_s"] != table.seconds(frame.index)
                        or binding["decoded_rgb_sha256"] != hashlib.sha256(frame.rgb.tobytes()).hexdigest()
                    ):
                        raise ValueError("Raw response is attached to different source pixels/time")
                    check_chip(frame.rgb, binding, face_detector)
                    record["native_inputs"] += 1
                    record["head_input_bindings"] += 2
                record["status"] = "ok"
            except Exception as error:
                record["reason"] = f"{type(error).__name__}: {error}"
            records.append(record)
    body = {
        "schema": "chipmunk.mouth-context-native-input-audit/10",
        "lock_sha256": digest(lock_path),
        "calibration_scores_sha256": digest(calibration),
        "development_scores_sha256": digest(development),
        "implementation_sha256": digest(Path(__file__)),
        "planned": 112,
        "passed": sum(r["status"] == "ok" for r in records),
        "native_inputs": sum(r["native_inputs"] for r in records),
        "head_input_bindings": sum(r["head_input_bindings"] for r in records),
        "cases": records,
        "scope": "Every retained RGB/PTS/crop/normalized-input binding across two heads,24 calibration/88 development clips; not neural output truth, causal localization or outside review.",
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k != "cases"}))
    if body["passed"] != 112:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("calibration", type=Path)
    p.add_argument("development", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.lock, a.calibration, a.development, a.output)
