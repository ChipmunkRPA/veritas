"""Publish allowlisted source/evaluation metadata without media or private paths."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts.liveportrait_mouth_benchmark import digest
from scripts.run_mouth_learning_v8 import check_lock, write


def packed(path, value):
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if b"/Users/" in encoded or b"/private/" in encoded:
        raise ValueError("Private filesystem path in allowlisted research metadata")
    with path.open("xb") as stream:
        stream.write(gzip.compress(encoded, mtime=0))


def export(root, destination):
    experiment = root / "experiment"
    lock_path = experiment / "lock.json"
    lock = check_lock(lock_path)
    summary_path = experiment / "summary.json"
    result = json.loads(summary_path.read_text())
    head_path = experiment / "trained/head.json"
    receipt = experiment / "development-scores.jsonl"
    if result["score_receipt_sha256"] != digest(receipt) or result["head_sha256"] != digest(head_path):
        raise ValueError("Evaluation result does not bind the retained responses/head")
    source_audit = json.loads((root / "training-source-audit.json").read_text())
    video_audit = json.loads((experiment / "video-source-audit.json").read_text())
    replay = json.loads((experiment / "full-forward-replay.json").read_text())
    if (
        not source_audit["passed"]
        or source_audit["planned"] != 1600
        or video_audit["passed"] != 88
        or not replay["passed"]
    ):
        raise ValueError("Complete source and full-model replay audits are required")
    for name, source in (
        ("mouth-v8-training-source-audit.json", root / "training-source-audit.json"),
        ("mouth-v8-video-source-audit.json", experiment / "video-source-audit.json"),
        ("mouth-v8-full-forward-replay.json", experiment / "full-forward-replay.json"),
        ("mouth-v8-head.json", head_path),
    ):
        text = source.read_text()
        if "/Users/" in text or "/private/" in text:
            raise ValueError("Private path in audit/head copy")
        (destination / name).write_text(text)
    public_lock = {
        k: lock[k]
        for k in (
            "schema",
            "method",
            "training_manifest_sha256",
            "implementation_sha256",
            "deployment_eligible",
        )
    }
    public_lock.update(
        original_lock_sha256=digest(lock_path),
        development=[
            {k: p[k] for k in ("manifest_sha256", "receipt_sha256", "family", "planned")}
            for p in lock["development"]
        ],
    )
    write(destination / "mouth-v8-lock.json", public_lock)
    plan = json.loads((root / "training-plan.json").read_text())
    write(
        destination / "mouth-v8-training-plan.json",
        {k: v for k, v in plan.items() if k not in ("parent_plan", "parent_manifest", "source_root")},
    )
    training = json.loads(Path(lock["training_manifest"]).read_text())
    packed(
        destination / "mouth-v8-training-bindings.json.gz",
        {
            "schema": "chipmunk.mouth-learning-public-training/8",
            "original_manifest_sha256": lock["training_manifest_sha256"],
            "parent_manifest_sha256": training["parent_sha256"],
            "extension_manifest_sha256": training["extension_sha256"],
            "items": [
                {k: v for k, v in r.items() if k not in ("path", "mask_path")} for r in training["items"]
            ],
            "terms": "CREMA-D ODbL/DbCL source/derived data; checkpoint/source declarations separately recorded. No media or pretrained/learned weights included.",
        },
    )
    cache_path = experiment / "cache/receipt.json"
    packed(
        destination / "mouth-v8-prefix-bindings.json.gz",
        {"original_receipt_sha256": digest(cache_path), **json.loads(cache_path.read_text())},
    )
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
        "roi_bindings",
    )
    packed(
        destination / "mouth-v8-review-responses.json.gz",
        {
            "schema": "chipmunk.mouth-learning-public-responses/8",
            "original_receipt_sha256": digest(receipt),
            "header": header,
            "current_review_point": 0.007326911389827726,
            "cases": [{k: r[k] for k in allowed if k in r} for r in rows],
        },
    )
    result_target = destination.parent / "mouth-learning-v8-evaluation.json"
    result_target.write_text(summary_path.read_text())
    print(
        json.dumps(
            {
                "summary": str(result_target),
                "published_cases": len(rows),
                "training_stills": len(training["items"]),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    export(a.root, a.destination)
