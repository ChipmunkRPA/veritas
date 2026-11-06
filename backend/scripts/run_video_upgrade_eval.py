"""Compare the local production video pipeline with a pinned temporal candidate."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

from prepare_genbuster_video_eval import POLICY as GENBUSTER_POLICY
from prepare_video_upgrade_eval import POLICY, sha

CHECKPOINT = "7df233979f9d3ef340e101d0d635a4d074577d43e6d1591d677cde31f80e44ba"
SOURCE_FILES = {
    "consistency_branch.py": "bc0687fdc835037e18bafe28a80511e23fdc4f8c0603171a3c4897057b5c434b",
    "detector_model.py": "bdfcfddba6dfe305f462d0a094b28314ea208b2f4b7fa119978fd226d88f05a6",
    "motion_branch.py": "5d809994c1597bcfc2a7c7cdf59faa1da15ba87b0701df1ab580ce138a3d7c34",
    "pixel_branch.py": "e8d6d188729662187fcd04653d54337101bd8287820d3a87f2a4cf09e8bca26a",
    "video_io.py": "bab6ba4b8b1c07ff9bc392cf40487be8e8d31fdd248a4494563057e812a043dc",
}


class AegisScorer:
    def __init__(self, source: Path, checkpoint: Path, device: str):
        import timm
        import torch

        if sha(checkpoint) != CHECKPOINT:
            raise ValueError("unexpected candidate checkpoint")
        for filename, digest in SOURCE_FILES.items():
            if sha(source / filename) != digest:
                raise ValueError(f"unexpected upstream source: {filename}")
        sys.path.insert(0, str(source.resolve()))
        try:
            pixel = importlib.import_module("pixel_branch")
            model = importlib.import_module("detector_model")
            self.video_io = importlib.import_module("video_io")
        finally:
            sys.path.pop(0)
        # The checkpoint already contains the resized encoder weights. Construct
        # its architecture locally in place of the upstream hard-coded cache path.
        def encoder():
            return timm.create_model(
                "vit_base_patch14_dinov2", pretrained=False, num_classes=0,
                global_pool="avg", img_size=224,
            )

        pixel._load_dinov2_224 = encoder
        self.torch = torch
        self.device = device
        self.model = model.VideoForensicsDetector(freeze_dino=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(state["model_state"], strict=True)
        self.model.to(device).eval()
        self.video_io.DECORD_AVAILABLE = False

    def score(self, path: Path) -> dict:
        started = time.perf_counter()
        bundle = self.video_io.load_video(str(path), random_start=False)
        with self.torch.inference_mode():
            outputs = self.model(bundle.frames_all.unsqueeze(0).to(self.device))
        logit = float(outputs["ai_logit"].item())
        if not math.isfinite(logit):
            raise ValueError("non-finite candidate score")
        return {
            "status": "ok",
            "logit": logit,
            "score": float(outputs["ai_probability"].item()),
            "pixel_score": float(outputs["pixel_prob"].item()),
            "motion_score": float(outputs["motion_prob"].item()),
            "consistency_score": float(outputs["consistency_prob"].item()),
            "runtime_s": round(time.perf_counter() - started, 4),
        }


def production_scorer(work_dir: Path, profile: str):
    from app.bench.score import ScoreTask
    from app.bench.score_v2 import V2VideoScorer
    from app.engine.customer_assessment import customer_assessment

    class ProductionScorer(V2VideoScorer):
        def rows_from_outputs(self, outputs, runtime_ms, durations=None):
            rows = super().rows_from_outputs(outputs, runtime_ms, durations)
            fused = outputs["fuse"]
            decision = fused["decision"]
            answer = customer_assessment(
                {
                    "kind": "video",
                    "decision": decision,
                    "caveats": fused.get("caveats"),
                    "condition_profile": (outputs.get("condition") or {}).get("profile"),
                }
            )
            return {
                "status": "ok",
                "runtime_s": round(runtime_ms / 1000, 4),
                "score": decision.get("probability_ai_calibrated"),
                "customer_direction": answer["direction"],
                "confidence_pct": answer["confidence_pct"],
                "fusion_mode": decision.get("fusion_mode"),
                "fusion_hash": decision.get("fuser_sha256"),
                "detectors": rows,
            }

        def score_file(self, path: Path):
            return self.score(ScoreTask(path.stem, "original", "none", path))

    scorer = ProductionScorer(work_dir, profile=profile)
    # Restore the installed fusion selection. V2VideoScorer normally excludes
    # fusion artefacts because its raw detector rows are used to train them.
    scorer._settings.fuser_dir = scorer._saved["fuser_dir"]
    return scorer


def run(args: argparse.Namespace) -> None:
    import torch

    torch.set_num_threads(args.threads)
    body = json.loads(args.manifest.read_text())
    if body["policy"] not in {POLICY, GENBUSTER_POLICY}:
        raise ValueError("unexpected source manifest policy")
    path_root = args.manifest.parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.system == "candidate" and args.candidate == "cakelens":
        from video_cakelens_candidate import CHECKPOINT as cake_checkpoint
        from video_cakelens_candidate import SOURCE_FILES as cake_sources
        from video_cakelens_candidate import CakeLensScorer

        candidate_type, checkpoint, source_files = CakeLensScorer, cake_checkpoint, cake_sources
    else:
        candidate_type, checkpoint, source_files = AegisScorer, CHECKPOINT, SOURCE_FILES
    lock = {
        "system": args.system,
        "manifest_sha256": sha(args.manifest),
        "scorer_sha256": sha(Path(__file__)),
        "variants": args.variant,
        "profile": args.profile if args.system == "current" else None,
        "candidate": args.candidate if args.system == "candidate" else None,
        "candidate_checkpoint": checkpoint if args.system == "candidate" else None,
        "candidate_source": source_files if args.system == "candidate" else None,
        "torch": torch.__version__,
    }
    with args.output.open("x") as file, tempfile.TemporaryDirectory(prefix="video-upgrade-") as temp:
        file.write(json.dumps({"run": lock}) + "\n")
        file.flush()
        if args.system == "candidate":
            if args.upstream is None or args.checkpoint is None:
                raise ValueError("candidate requires --upstream and --checkpoint")
            scorer = candidate_type(args.upstream, args.checkpoint, args.device)
            score = scorer.score
        else:
            scorer = production_scorer(Path(temp) / "database", args.profile)
            score = scorer.score_file
        try:
            for item in body["items"]:
                for variant in args.variant:
                    meta = item.get(variant)
                    row = {key: item[key] for key in ("id", "group", "label", "split", "family")}
                    row.update(variant=variant, sha256=(meta or {}).get("sha256"))
                    if item["status"] != "ready":
                        row.update(status="unavailable", reason=item.get("reason"))
                    else:
                        path = path_root / meta["path"]
                        if sha(path) != meta["sha256"]:
                            raise ValueError(f"source bytes changed: {item['id']} ({variant})")
                        try:
                            row.update(score(path))
                        except Exception as exc:
                            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                    file.write(json.dumps(row, allow_nan=False) + "\n")
                    file.flush()
                    print(f"{item['id']} {variant}: {row['status']}", flush=True)
        finally:
            if args.system == "current":
                scorer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=["current", "candidate"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=["original", "matched"], nargs="+", default=["original", "matched"])
    parser.add_argument("--profile", choices=["full", "deep"], default="full")
    parser.add_argument("--candidate", choices=["aegis", "cakelens"], default="aegis")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())
