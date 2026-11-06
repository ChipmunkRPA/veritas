"""Publish V9 source/learning/response receipts without media or private filesystem paths."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts.liveportrait_mouth_benchmark import digest
from scripts.run_mouth_learning_v8 import write
from scripts.run_mouth_paired_v9 import check_lock


def encoded(body):
    data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
    if b"/Users/" in data or b"/private/" in data:
        raise ValueError("Private path in allowlisted public research artifact")
    return data


def export(root, destination):
    lock_path = root / "lock.json"
    lock = check_lock(lock_path)
    result = json.loads((root / "summary.json").read_text())
    head = root / "trained/head.json"
    receipt = root / "development-scores.jsonl"
    if result["head_sha256"] != digest(head) or result["score_receipt_sha256"] != digest(receipt):
        raise ValueError("Summary refers to a different model or score receipt")
    fit = json.loads((root / "fit-replay.json").read_text())
    forward = json.loads((root / "full-forward-replay.json").read_text())
    video = json.loads((root / "video-source-audit.json").read_text())
    if not fit["passed"] or not forward["passed"] or video["passed"] != 88:
        raise ValueError("Complete fitting/full-forward/native-video source replays required")
    for name, source in (
        ("mouth-v9-head.json", head),
        ("mouth-v9-full-forward-replay.json", root / "full-forward-replay.json"),
        ("mouth-v9-video-source-audit.json", root / "video-source-audit.json"),
    ):
        text = source.read_text()
        encoded(json.loads(text))
        (destination / name).write_text(text)
    public_lock = {
        k: v
        for k, v in lock.items()
        if k.endswith("_sha256")
        or k
        in (
            "schema",
            "method",
            "implementation_sha256",
            "pairs",
            "fitting_teacher_offsets",
            "deployment_eligible",
        )
    }
    public_lock["original_lock_sha256"] = digest(lock_path)
    public_lock["development"] = [
        {k: p[k] for k in ("manifest_sha256", "receipt_sha256", "family", "planned")}
        for p in lock["development"]
    ]
    encoded(public_lock)
    write(destination / "mouth-v9-lock.json", public_lock)
    (destination / "mouth-v9-fit-replay.json.gz").write_bytes(gzip.compress(encoded(fit), mtime=0))
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    allowed = (
        "id",
        "group",
        "family",
        "condition",
        "label",
        "status",
        "reason",
        "sha256",
        "source_sha256",
        "source_url",
        "source_license",
        "frames",
        "time_base",
        "pts",
        "pcm_sha256",
        "audio_samples",
        "edit_frame_range",
        "edit_time_range_s",
        "source_ranges",
        "original_rgb_sha256",
        "base_windows",
        "probe_windows",
        "model_inputs",
        "baseline_replay_parity",
        "prior_native_input_parity",
        "roi_bindings",
    )
    public_responses = {
        "schema": "chipmunk.mouth-source-paired-public-responses/9",
        "original_receipt_sha256": digest(receipt),
        "header": header,
        "current_review_point": 0.007326911389827726,
        "cases": [{k: row[k] for k in allowed if k in row} for row in rows],
    }
    (destination / "mouth-v9-review-responses.json.gz").write_bytes(
        gzip.compress(encoded(public_responses), mtime=0)
    )
    encoded(result)
    (destination.parent / "mouth-paired-v9-evaluation.json").write_text((root / "summary.json").read_text())
    print(
        json.dumps(
            {
                "published_cases": len(rows),
                "fit_pairs": len(fit["pairs"]),
                "teacher_fit_targets": len(fit["teacher_fit_targets"]),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    export(a.root, a.destination)
