"""Reuse the unchanged V8 source/full-forward audit formats with explicit V9 bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from scripts import audit_mouth_video_v8 as video_audit
from scripts import replay_mouth_learning_v8 as forward_audit
from scripts.liveportrait_mouth_benchmark import digest
from scripts.run_mouth_paired_v9 import check_lock, source_cache


def audit(stage, lock_path, input_path, output):
    lock = check_lock(lock_path)
    if stage == "video":
        with patch.object(video_audit, "check_lock", check_lock):
            video_audit.audit(lock_path, input_path, output)
    else:

        def cache(_lock, _cache):
            return source_cache(lock_path)

        with patch.object(forward_audit, "load_cache", side_effect=cache):
            forward_audit.replay(lock_path, Path(lock["cache"]), input_path, output)
    result = json.loads(output.read_text())
    if result["lock_sha256"] != digest(lock_path):
        raise ValueError("Audit output uses a different V9 experiment")
    print(
        json.dumps(
            {"experiment": 9, "shared_format": result["schema"], "mode": stage, "passed": result["passed"]}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["video", "forward"])
    p.add_argument("lock", type=Path)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.stage, a.lock, a.input, a.output)
