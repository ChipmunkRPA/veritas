"""Score face-swap candidate and current face check in an isolated video run."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from prepare_face_swap_upgrade_eval import POLICY
from prepare_video_upgrade_eval import sha

CHECKS = ("vid.frames.dfdc_b7", "vid.swap.gend_clip_l14")
INPUT_CHECKS = ("vid.faces.yunet", "vid.tracks.yunet_dinov2")


def run(manifest: Path, output: Path, *, retain_frames: bool = False) -> None:
    import torch

    from app.bench.score import ScoreTask
    from app.bench.score_v2 import V2VideoScorer
    from app.engine.registry import build_registry, get_registry

    torch.set_num_threads(2)
    body = json.loads(manifest.read_text())
    if body["policy"] != POLICY:
        raise ValueError("unexpected face-swap manifest policy")
    spec = get_registry().manifest
    registry = build_registry(
        policy={
            "preset": "commercial",
            "allow_detectors": [CHECKS[1]],
            "deny_detectors": [d.id for d in spec.detectors if d.id not in CHECKS + INPUT_CHECKS],
        },
    )
    for check in CHECKS + INPUT_CHECKS:
        if not registry.decision(check).enabled:
            raise ValueError(f"required check unavailable: {check}")
    class FaceScorer(V2VideoScorer):
        def rows_from_outputs(self, outputs, runtime_ms, durations=None):
            rows = super().rows_from_outputs(outputs, runtime_ms, durations)
            if retain_frames:
                for entry in outputs["detect"]["detectors"]:
                    key = self.keys.get(entry["detector_id"])
                    if key and entry["detector_id"] in CHECKS:
                        rows[key]["scored_units"] = entry["unit_scores"]
            return rows

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file, tempfile.TemporaryDirectory(prefix="face-swap-upgrade-") as temp:
        scorer = FaceScorer(Path(temp) / "database", registry=registry, profile="full")
        try:
            keys = {check: scorer.keys[check] for check in CHECKS + INPUT_CHECKS}
            if any(value.endswith("@not-installed") for value in keys.values()):
                raise ValueError("face-swap benchmark weights are not installed")
            lock = {
                "manifest_sha256": sha(manifest),
                "scorer_sha256": sha(Path(__file__)),
                "checks": keys,
                "versions": {key: scorer.versions[key] for key in keys},
                "variant": "matched",
            }
            file.write(json.dumps({"run": lock}) + "\n")
            file.flush()
            for item in body["items"]:
                row = {key: item[key] for key in ("id", "group", "label", "split", "family")}
                row.update(status=item["status"])
                if item["status"] != "ready":
                    row.update(reason=item.get("reason"))
                else:
                    path = manifest.parent / item["matched"]["path"]
                    row["sha256"] = sha(path)
                    if row["sha256"] != item["matched"]["sha256"]:
                        raise ValueError(f"source bytes changed: {item['id']}")
                    try:
                        scores = scorer.score(ScoreTask(item["id"], "matched", "none", path))
                        row.update(status="ok", scores={key: scores[value] for key, value in keys.items()})
                    except Exception as exc:
                        row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                file.write(json.dumps(row, allow_nan=False) + "\n")
                file.flush()
                print(f"{item['id']}: {row['status']}", flush=True)
        finally:
            scorer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retain-frames", action="store_true")
    args = parser.parse_args()
    run(args.manifest, args.output, retain_frames=args.retain_frames)
