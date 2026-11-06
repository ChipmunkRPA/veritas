"""Record the current face scores and track boxes from the production video stages."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from prepare_face_interval_eval import POLICY
from prepare_video_upgrade_eval import sha

CHECK = "vid.frames.dfdc_b7"
INPUT_CHECKS = ("vid.faces.yunet", "vid.tracks.yunet_dinov2")


def main() -> None:
    import torch

    from app.bench.score import ScoreTask
    from app.bench.score_v2 import V2VideoScorer
    from app.engine.registry import build_registry, get_registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    if manifest["policy"] != POLICY:
        raise ValueError("unexpected source manifest")
    registry = build_registry(policy={"preset": "commercial", "deny_detectors": [
        d.id for d in get_registry().manifest.detectors if d.id not in (CHECK, *INPUT_CHECKS)]})

    class FaceScorer(V2VideoScorer):
        def rows_from_outputs(self, outputs, runtime_ms, durations=None):
            entry = next(e for e in outputs["detect"]["detectors"] if e["detector_id"] == CHECK)
            decode = {f["index"]: f for f in outputs["decode"]["frames"]}
            points = [{"frame_index": decode[int(u["unit_ref"].split(":")[1])]["frame_index"], "score": u["raw"]}
                      for u in entry["unit_scores"]]
            return {"check_status": entry["status"], "points": points, "table": outputs["probe"]["frame_table"],
                    "exhibits": [*outputs["tracks"]["exhibits"], *outputs["faces"]["exhibits"]]}

    with args.output.open("x") as out, tempfile.TemporaryDirectory(prefix="face-interval-") as temp:
        scorer = FaceScorer(Path(temp) / "trial", registry=registry, profile="full")
        try:
            if any(scorer.keys.get(check, "@not-installed").endswith("@not-installed") for check in (CHECK, *INPUT_CHECKS)):
                raise ValueError("required detector weights are absent")
            out.write(json.dumps({"manifest_sha256": sha(args.manifest), "keys": scorer.keys,
                                  "versions": scorer.versions, "split": args.split}) + "\n")
            for row in manifest["items"]:
                if row["split"] != args.split:
                    continue
                if row["status"] == "ready":
                    path = args.manifest.parent / row["path"]
                    if sha(path) != row["sha256"]:
                        raise ValueError(f"input changed: {row['id']}")
                    try:
                        row = {**row, **scorer.score(ScoreTask(row["id"], row["family"], "none", path))}
                    except Exception as error:
                        row = {**row, "check_status": "failed", "error": str(error)}
                out.write(json.dumps(row, allow_nan=False) + "\n")
                out.flush()
                print(row["id"], row.get("check_status", row["status"]), flush=True)
        finally:
            scorer.close()


if __name__ == "__main__":
    main()
