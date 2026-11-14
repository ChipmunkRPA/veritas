# SPDX-License-Identifier: Apache-2.0
"""Replay speech activity on the exact mouth-hold noise panel."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.review_video_mouth_hold_noise_transfer import SCENES

SCHEMA = "chipmunk.video-mouth-hold-activity-transfer/1"
MIN_SAMPLES = 3200
MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"


def summarize(plan, rows):
    keys = set()
    counts = {split: {scene: Counter() for scene in ("clean", *SCENES)}
              for split in ("development", "held")}
    for row in rows:
        group, side, scene = row["group"], row["side"], row["scene"]
        key = group, side, scene
        if key in keys or len(row["activity_samples"]) != 4 or any(
                type(n) is not int or not 0 <= n <= 6400 for n in row["activity_samples"]):
            raise ValueError("Invalid or repeated speech-activity measurement")
        keys.add(key)
        split = plan["groups"][group]["split"]
        background = scene.startswith("background_")
        label = scene.removeprefix("background_")
        active = sum(value >= MIN_SAMPLES for value in row["activity_samples"])
        tag = "background" if background else "delivery"
        counts[split][label].update({tag + "_positive": active, tag + "_windows": 4})
    if keys != {(group["group"], side, scene) for group in plan["groups"] for side in ("original", "variant")
                for scene in ("clean", *SCENES, *(f"background_{name}" for name in SCENES))}:
        raise ValueError("Incomplete paired speech-activity results")
    return {split: {scene: dict(stats) for scene, stats in scenes.items()} for split, scenes in counts.items()}


def measure(pcm, activity, windows):
    import numpy as np

    spans = [(round(a * 16000), round(b * 16000))
             for a, b in activity.spans(pcm.astype(np.float32) / 32768)]
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [sum(max(0, min(end, right) - max(start, left)) for start, end in merged)
            for left, right in windows]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("baseline", type=Path)
    cli.add_argument("receipt", type=Path)
    cli.add_argument("--source", type=Path)
    cli.add_argument("--sounds", type=Path)
    cli.add_argument("--checkpoint", type=Path)
    args = cli.parse_args()
    baseline = json.loads(args.baseline.read_bytes())
    plan = baseline["plan"]
    refs = [{k: row[k] for k in ("group", "side", "scene", "pcm_sha256")} for row in baseline["rows"]]
    if args.source:
        import torch

        from app.detectors.audio.upgraded import SpeechActivity
        from scripts.video_mouth_hold_noise_transfer import _inputs, _source_plan, sha

        if sha(args.checkpoint.read_bytes()) != MODEL_SHA256 or _source_plan(args.source, args.sounds) != plan:
            raise ValueError("The local model or source plan differs from the freeze")
        torch.set_num_threads(1)
        activity = SpeechActivity(args.checkpoint)
        rows = []
        for expected, (group, side, scene, _, pcm) in zip(refs, _inputs(args.source, plan, args.sounds), strict=True):
            observed = {"group": group, "side": side, "scene": scene, "pcm_sha256": sha(pcm.tobytes())}
            if observed != expected:
                raise ValueError("Speech activity measurement does not share the paired acoustic input")
            rows.append(observed | {"activity_samples": measure(pcm, activity, plan["windows"])})
        record = {"schema": SCHEMA, "model_sha256": MODEL_SHA256, "min_samples": MIN_SAMPLES,
                  "rows": rows, "summary": summarize(plan, rows)}
        with args.receipt.open("x") as stream:
            json.dump(record, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
    else:
        record = json.loads(args.receipt.read_bytes())
    if (record["schema"] != SCHEMA or record["model_sha256"] != MODEL_SHA256 or record["min_samples"] != MIN_SAMPLES
            or [{key: row[key] for key in refs[0]} for row in record["rows"]] != refs
            or record["summary"] != summarize(plan, record["rows"])):
        raise ValueError("The speech activity receipt differs from its paired replay")
    print(json.dumps(record["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
