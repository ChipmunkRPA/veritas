"""Replay every fitting-only teacher target and bind each matched pair to immutable sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import mouth_paired_v9 as method
from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_learner_v8 import sha_tensor
from scripts.run_mouth_learning_v8 import setup, write
from scripts.run_mouth_paired_v9 import initialized, source_cache


def audit(lock_path, head_path, output):
    lock, rows, receipt, tokens = source_cache(lock_path)
    head = json.loads(head_path.read_text())
    if head["lock_sha256"] != digest(lock_path):
        raise ValueError("Different V9 head/source lock")
    setup()
    teacher = initialized(lock).eval().requires_grad_(False)
    targets, eligible = method.teacher_targets(teacher, tokens, rows, receipt["bindings"])
    actual_identity = [{"name": k, "sha256": sha_tensor(p)} for k, p in teacher.named_parameters()]
    if (
        actual_identity != head["learning"]["teacher_parameter_identity"]
        or sha_tensor(targets) != head["learning"]["teacher_target_f32_sha256"]
        or sha_tensor(eligible) != head["learning"]["teacher_eligible_mask_sha256"]
    ):
        raise ValueError("Replayed frozen teacher, targets or eligible inputs differ")
    pair_rows = []
    for p, n in method.make_pairs(rows):
        positive, negative = rows[p], rows[n]
        if (
            positive["role"] != "fit"
            or negative["role"] != "fit"
            or any(
                positive[k] != negative[k]
                for k in ("group", "frame_index", "source_sha256", "original_rgb_sha256")
            )
        ):
            raise ValueError("Pair differs in source identity/time or includes reserved actors")
        pair_rows.append({"positive_id": positive["id"], "negative_id": negative["id"]})
    if pair_rows != lock["pairs"] or pair_rows != head["learning"]["pairs"]:
        raise ValueError("Teacher fitting used different matched source pairs")
    public_targets = []
    for row, binding in zip(rows, receipt["bindings"], strict=True):
        for view in binding["views"]:
            k = view["offset"]
            if bool(eligible[k]) != method.teacher_eligible(row):
                raise ValueError("A reserved/first-family AI input entered teacher retention")
            if eligible[k]:
                public_targets.append(
                    {"id": row["id"], "view": view["view"], "offset": k, "target": float(targets[k])}
                )
    body = {
        "schema": "chipmunk.mouth-paired-fit-source-replay/9",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "implementation_sha256": digest(Path(__file__)),
        "cache_receipt_sha256": lock["cache_receipt_sha256"],
        "passed": len(pair_rows) == 768 and len(public_targets) == 4096,
        "pairs": pair_rows,
        "teacher_fit_targets": public_targets,
        "all_target_f32_sha256": sha_tensor(targets),
        "eligible_bool_sha256": sha_tensor(eligible),
        "point_teacher_offsets": 0,
        "target_payload_bytes": len(targets) * 4,
        "target_count": len(targets),
        "scope": "All 768 source-matched fitting pairs and 4096 frozen teacher fitting targets; held-out offsets excluded; not outside review or detection accuracy.",
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k not in ("pairs", "teacher_fit_targets")}))
    if not body["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.lock, a.head, a.output)
