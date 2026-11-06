"""Compare reloaded full-model inference to retained-prefix inference on every native still."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from app.detectors.image import gend
from scripts import mouth_learner_v8 as method
from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_probe_v6 import margins, mouth_chip
from scripts.run_mouth_learning_v8 import load_cache, loaded_candidate, setup, write


def replay(lock_path, cache_path, head_path, output):
    _, rows, receipt, tokens = load_cache(lock_path, cache_path)
    head = json.loads(head_path.read_text())
    if head["lock_sha256"] != digest(lock_path):
        raise ValueError("Head uses a different source lock")
    setup()
    dino = loaded_candidate(head_path).obj
    learner = method.MouthLearner(dino.model).to("mps")
    # The temporary wrapper's head initialization is separate. The published
    # linear parameters are used below; its backbone is the reloaded learned one.
    point = {r["id"]: r["margin"] for r in head["point_responses"]}
    records = []
    maximum_feature_error = maximum_margin_error = maximum_point_error = 0.0
    for start in range(0, len(rows), 12):
        batch, bindings = rows[start : start + 12], receipt["bindings"][start : start + 12]
        chips = []
        for row, binding in zip(batch, bindings, strict=True):
            path = Path(row["path"])
            if digest(path) != row["sha256"]:
                raise ValueError("Native fitting PNG changed")
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            matrix = np.array([[float.fromhex(v) for v in r] for r in binding["alignment_matrix_hex"]])
            aligned = gend.warp(rgb, matrix, binding["alignment_side"])
            chips.append(mouth_chip(aligned)[0])
        values, _ = method.features(chips, dino)
        offsets = [b["views"][0]["offset"] for b in bindings]
        with torch.no_grad():
            retained = learner.representation(tokens[offsets].to("mps")).cpu().numpy()
        full_margins, retained_margins = margins(values, head), margins(retained, head)
        for j, row in enumerate(batch):
            feature_error = float(np.max(np.abs(values[j] - retained[j])))
            margin_error = abs(float(full_margins[j] - retained_margins[j]))
            point_error = abs(float(full_margins[j] - point[row["id"]])) if row["role"] == "point" else None
            maximum_feature_error = max(maximum_feature_error, feature_error)
            maximum_margin_error = max(maximum_margin_error, margin_error)
            maximum_point_error = max(maximum_point_error, point_error or 0.0)
            records.append(
                {
                    "id": row["id"],
                    "role": row["role"],
                    "label": row["label"],
                    "full_margin": float(full_margins[j]),
                    "cached_margin": float(retained_margins[j]),
                    "max_feature_difference": feature_error,
                    "margin_difference": margin_error,
                    "retained_point_margin_difference": point_error,
                }
            )
    passed = (
        len(records) == 1600
        and maximum_feature_error <= 1e-5
        and maximum_margin_error <= 1e-5
        and maximum_point_error <= 1e-5
    )
    body = {
        "schema": "chipmunk.mouth-learning-full-forward-replay/8",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "implementation_sha256": digest(Path(__file__)),
        "planned": 1600,
        "passed": passed,
        "maximum_feature_difference": maximum_feature_error,
        "maximum_margin_difference": maximum_margin_error,
        "maximum_point_margin_difference": maximum_point_error,
        "non_ai_point_stills_over_fixed_point_full_forward": sum(
            r["role"] == "point" and not r["label"] and r["full_margin"] > head["review_point"]
            for r in records
        ),
        "cases": records,
        "scope": "Reloaded final state; all native fitting/point stills; full forward versus cached frozen prefix. Numerical parity, not accuracy or outside validation.",
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k != "cases"}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    replay(a.lock, a.cache, a.head, a.output)
