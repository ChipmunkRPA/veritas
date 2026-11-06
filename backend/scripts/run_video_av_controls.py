"""Record existing synchronization and voice responses to fixed audiovisual controls."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from prepare_video_av_controls import POLICY
from prepare_video_upgrade_eval import sha

SYNC = "vid.lipsync.landmark_xcorr"
VOICE = "aud.pella_v2"
INPUTS = ("vid.tracks.yunet_dinov2", "vid.faces.yunet")


def run(manifest: Path, output: Path, *, timing_only: bool = False) -> None:
    import torch

    from app.bench.score import ScoreTask
    from app.bench.score_v2 import V2VideoScorer
    from app.core.config import settings
    from app.engine.registry import build_registry, get_registry

    torch.set_num_threads(2)
    if not timing_only and settings.audio_detector_mode != "candidate":
        raise ValueError("this comparison requires the local candidate audio mode")
    body = json.loads(manifest.read_text())
    if timing_only:
        from prepare_video_timing_upgrade import POLICY as TIMING_POLICY

    if body["policy"] != (TIMING_POLICY if timing_only else POLICY):
        raise ValueError("unexpected audiovisual controls policy")
    checks = (SYNC,) if timing_only else (SYNC, VOICE)
    spec = get_registry().manifest
    registry = build_registry(policy={
        "preset": "commercial",
        "deny_detectors": [d.id for d in spec.detectors if d.id not in (*INPUTS, *checks)],
    })

    class AVScorer(V2VideoScorer):
        def rows_from_outputs(self, outputs, runtime_ms, durations=None):
            entries = {e["detector_id"]: e for e in outputs["detect"]["detectors"]}
            readings = {}
            for detector in checks:
                entry = entries.get(detector)
                if entry is None:
                    readings[detector] = {"status": "not_run"}
                    continue
                readings[detector] = {key: entry[key] for key in ("status", "reason", "aggregate", "metrics", "unit_scores")}
                if detector == VOICE:
                    readings[detector]["screening"] = (entry.get("details") or {}).get("audio_screening")
            return readings

    with output.open("x") as file, tempfile.TemporaryDirectory(prefix="video-av-") as temp:
        scorer = AVScorer(Path(temp) / "database", registry=registry, profile="full")
        try:
            file.write(json.dumps({"manifest_sha256": sha(manifest), "scorer_sha256": sha(Path(__file__))}) + "\n")
            for row in body["items"]:
                path = manifest.parent / row["path"]
                if sha(path) != row["sha256"]:
                    raise ValueError(f"source bytes changed: {row['id']}")
                try:
                    values = scorer.score(ScoreTask(row["id"], row["category"], "none", path))
                    row = {**row, "status": "ok", "checks": values}
                except Exception as exc:
                    row = {**row, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                file.write(json.dumps(row, allow_nan=False) + "\n")
                file.flush()
                print(f"{row['id']}: {row['status']}", flush=True)
        finally:
            scorer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    run(args.manifest, args.output, timing_only=args.timing_only)
