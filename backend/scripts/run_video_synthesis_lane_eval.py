"""Score both deployed synthesis checks on the same inputs as the temporal candidate."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from prepare_genbuster_video_eval import POLICY as GENBUSTER_POLICY
from prepare_video_upgrade_eval import POLICY, sha

CHECKS = ("vid.synth.frames.commfor384", "vid.synth.frames.dda")


def run(manifest: Path, output: Path, variant: str) -> None:
    import torch

    from app.bench.score import ScoreTask
    from app.bench.score_v2 import V2VideoScorer
    from app.engine.registry import build_registry, get_registry

    torch.set_num_threads(2)
    body = json.loads(manifest.read_text())
    if body["policy"] not in {POLICY, GENBUSTER_POLICY}:
        raise ValueError("unexpected video manifest policy")
    spec = get_registry().manifest
    registry = build_registry(
        policy={"preset": "commercial", "deny_detectors": [d.id for d in spec.detectors if d.id not in CHECKS]}
    )
    if any(not registry.decision(check).enabled for check in CHECKS):
        raise ValueError("a current synthesis check is not enabled")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file, tempfile.TemporaryDirectory(prefix="video-synthesis-") as temp:
        scorer = V2VideoScorer(Path(temp) / "database", registry=registry, profile="full")
        try:
            keys = {check: scorer.keys[check] for check in CHECKS}
            if any(key.endswith("@not-installed") for key in keys.values()):
                raise ValueError("current synthesis weights are not installed")
            file.write(
                json.dumps({"run": {
                    "system": "current-synthesis-lane",
                    "manifest_sha256": sha(manifest),
                    "scorer_sha256": sha(Path(__file__)),
                    "checks": keys,
                    "versions": {check: scorer.versions[check] for check in CHECKS},
                    "variants": [variant],
                }}) + "\n"
            )
            file.flush()
            for item in body["items"]:
                row = {key: item[key] for key in ("id", "group", "label", "split", "family")}
                meta = item.get(variant, {})
                row.update(status=item["status"], sha256=meta.get("sha256"), variant=variant)
                if item["status"] == "ready":
                    path = manifest.parent / meta["path"]
                    if sha(path) != meta["sha256"]:
                        raise ValueError(f"source bytes changed: {item['id']}")
                    try:
                        scores = scorer.score(ScoreTask(item["id"], variant, "none", path))
                        row.update(status="ok", checks={check: scores[key] for check, key in keys.items()})
                        values = [row["checks"][check]["prob"] for check in CHECKS]
                        row["score"] = sum(values) / len(values) if all(p is not None for p in values) else None
                    except Exception as exc:
                        row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                else:
                    row.update(reason=item.get("reason"))
                file.write(json.dumps(row, allow_nan=False) + "\n")
                file.flush()
                print(f"{item['id']} {variant}: {row['status']}", flush=True)
        finally:
            scorer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=["original", "matched"], default="matched")
    args = parser.parse_args()
    run(args.manifest, args.output, args.variant)
