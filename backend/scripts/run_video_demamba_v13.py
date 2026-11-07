"""Complete consumed-source temporal-video comparison; no serving admission.

Freeze all inputs, implementations, assets and statistical rules before any
recording forward. Preserve both formats, unavailable sources and unassessed
checks. A passing development result only permits separately frozen confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import video_demamba_clock_v13 as clock
from scripts import video_demamba_v13 as candidate

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.video-demamba-development-lock/13"
DATASETS = {
    "deepaction": ("deepaction/manifest.json", "3b5688815c31dab4111e00c33cc3de7f6928ca0576bbf321a52db8bba8a08030"),
    "genbuster": ("genbuster/locked/manifest.json", "2780f76362c3a081e30cf6fc0eb63fb26a0168748d350b03d3d611f12f803cd6"),
}
CHECKS = ("vid.synth.frames.commfor384", "vid.synth.frames.dda")
BASE_ASSETS = {
    "hf/hub/models--OwensLab--commfor-model-384/blobs/b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387": "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387",
    "hf/hub/models--Junwei-Xi--Dual-Data-Alignment/blobs/b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e": "b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e",
    "weights/dinov2.vitl14/hub/facebookresearch_dinov2_b8931f7bf91576930313be2c6d6af376033b35f0.zip": "218d1e159ebd7d928aa76e5412ac04455c92aad9ecb3c48ab8542a98ccc1b357",
}
METHOD = {
    "candidate": "Strict original complete checkpoint; CPU float32, eager attention,2threads; one8-picture center window from first32 sampled pictures; original preprocessing; no padding/guessed clocks",
    "baseline": "Existing two production synthesis adapters; commercial policy/full profile, MPS float32,2CPUthreads; mean of available finite raw probabilities. Individual failed statuses remain failures; all-inapplicable remains unassessed. Candidate CPU versus baseline MPS latency is not a like-hardware speed comparison.",
    "scope": "Component general-picture synthesis evidence on consumed development; not whole-file authenticity, face-swap/lip-sync validation or calibrated source-time localization",
    "calibration": "Each method separately: same common applicable original calibration rows per dataset, both formats together; equal truth classes; clipped logits[-20,20], nonnegative Platt slope, ridge .01*(slope-1)^2, L-BFGS-B start[1,0] unchanged published numerical fitter",
    "research_reference": .8,
    "missing": "Every288 planned rows retained; no score means inconclusive, never original; expected unavailable4 remain in denominator; no failure-driven case replacement",
    "statistics": "Shared prior video summary implementation; paired Brier/logloss uncertainty resamples source groups2000times seed901; variants do not constitute independent originals",
    "native_audit": "First source by SHA256(id) in each dataset/truth/format/role cell:16 cases, regardless of response. Separate FFmpeg CLI capture versus OpenCV exact RGB/tensor, native decoded PTS within1.1e-6s, candidate/baseline responses absolute1e-5. Paths share FFmpeg decoder lineage; not independently authored codecs. Missing support explicit.",
    "gates": {
        "complete_matrix": "All288 planned rows and exact source correspondence for both methods",
        "no_failures": "Zero unexpected decode/inference failures; unsupported/known unavailable distinct",
        "calibration_support": "Both truth classes and at least8 common independent calibration groups in each dataset",
        "brier_gain": "Paired mean candidate-minus-baseline Brier <=-.005 in each dataset/format",
        "brier_interval": "Upper paired Brier interval below0 in each dataset/format",
        "wrong_directions": "At most1 additional false AI and1 false original in each dataset/format and overall",
        "ai_usefulness": "At least4 additional distinct AI sources with an AI conclusion in some format, and lose at most2 correct AI conclusions in any dataset/format",
        "original_retention": "Lose at most2 correct original conclusions in any dataset/format and overall",
        "total_correct": "More correct directional conclusions overall",
        "scope_retention": "Every baseline-assessed input still assessed by candidate",
        "native_correspondence": "All16 fixed native/source replays pass or retain genuinely unassessed support; no failed audit",
    },
    "fresh_confirmation": "Not permitted before every frozen development gate passes; new source/data freeze required",
    "serving_admitted": False,
}
FILES = {
    "backend/scripts/run_video_demamba_v13.py", "backend/scripts/video_demamba_v13.py",
    "backend/scripts/video_demamba_clock_v13.py", "backend/scripts/review_av_candidate_v12.py",
    "backend/scripts/summarize_video_upgrade_eval.py", "backend/scripts/prepare_video_upgrade_eval.py",
    "backend/tests/test_video_demamba_v13.py", "backend/tests/test_video_demamba_clock_v13.py",
    "backend/tests/test_video_demamba_execution_v13.py", "docs/video-demamba-v13-protocol.md",
}


def digest(path):
    return candidate.digest(path)


def write_new(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def implementation():
    files = FILES | {str(p.relative_to(PROJECT)) for p in (PROJECT / "backend/app").rglob("*") if p.suffix in {".py", ".yaml"}}
    return {name: digest(PROJECT / name) for name in sorted(files)}


def runtime():
    import cv2
    names = ("numpy", "scipy", "scikit-learn", "torch", "transformers", "timm")
    return {"python": platform.python_version(), "platform": platform.system(), "architecture": platform.machine(),
            "opencv": cv2.__version__, "opencv_distributions": {name: importlib.metadata.version(name) for name in importlib.metadata.packages_distributions().get("cv2", [])},
            **{name: importlib.metadata.version(name) for name in names}}


def installed():
    modules = ("transformers.models.x_clip.modeling_x_clip", "transformers.models.x_clip.configuration_x_clip",
               "scipy.optimize._lbfgsb_py", "scipy.optimize._lbfgsb", "torch.nn.functional",
               "timm.models.vision_transformer", "timm.layers.patch_embed", "timm.layers.mlp")
    result = {name: digest(importlib.import_module(name).__file__) for name in modules}
    import cv2
    result["cv2_binary"] = digest(next(Path(cv2.__file__).parent.glob("cv2*.so")))
    for name in ("ffprobe", "ffmpeg"):
        result[name + "_binary"] = digest(shutil.which(name))
    return result


def source_rows(base):
    rows, manifests = [], {}
    for dataset, (relative, expected) in DATASETS.items():
        path = base / relative
        if digest(path) != expected:
            raise ValueError("Original consumed manifest changed")
        manifests[dataset] = json.loads(path.read_text())
        if len(manifests[dataset]["items"]) != 72:
            raise ValueError("Every original source required")
        roles = {}
        for item in manifests[dataset]["items"]:
            if item["group"] in roles and roles[item["group"]] != item["split"]:
                raise ValueError("Original source groups cross calibration/development")
            roles[item["group"]] = item["split"]
            for variant in ("original", "matched"):
                meta = item.get(variant) or {}
                row = {k: item[k] for k in ("id", "group", "label", "split", "family", "status")}
                row.update(dataset=dataset, variant=variant, sha256=meta.get("sha256"),
                           relative_path=str(Path(relative).parent / meta["path"]) if meta.get("path") else None)
                rows.append(row)
    if len(rows) != 288 or Counter(r["status"] for r in rows) != {"ready": 284, "unavailable": 4}:
        raise ValueError("Complete original288-input matrix required")
    return manifests, rows


def asset_hashes(model_root, baseline_root):
    values = {"candidate/" + name: digest(model_root / name) for name in candidate.ASSETS}
    if any(values["candidate/" + name] != expected for name, expected in candidate.ASSETS.items()):
        raise ValueError("Candidate source/notice changed")
    values["candidate/best_acc.pth"] = digest(model_root / "best_acc.pth")
    if values["candidate/best_acc.pth"] != candidate.WEIGHT_SHA:
        raise ValueError("Original candidate weights required")
    for name, expected in BASE_ASSETS.items():
        values["baseline/" + name] = digest(baseline_root / name)
        if values["baseline/" + name] != expected:
            raise ValueError("Original baseline weights required")
    return values


def native_plan(rows):
    native = []
    for dataset in DATASETS:
        for label in (0, 1):
            for variant in ("original", "matched"):
                for split in ("calibration", "test"):
                    pool = [r for r in rows if r["dataset"] == dataset and r["label"] == label and r["variant"] == variant and r["split"] == split and r["status"] == "ready"]
                    selected = min(pool, key=lambda r: hashlib.sha256(r["id"].encode()).digest())
                    native.append([selected["dataset"], selected["id"], selected["variant"]])
    return native


def preparation():
    path = PROJECT / "docs/evaluation-panels/video-demamba-v13-synthetic-review.json"
    body = json.loads(path.read_text())
    if (body["passed"] is not True or body["source_recordings_scored"] != 0
            or body["weight_sha256"] != candidate.WEIGHT_SHA
            or body["reviewer_sha256"] != digest(PROJECT / "backend/scripts/video_demamba_v13.py")
            or len(body["synthetic_cases"]) != 3 or not all(r["repeat_exact"] for r in body["synthetic_cases"])):
        raise ValueError("Verified original synthetic preparation required")
    return digest(path)


def freeze(base, model_root, baseline_root, output):
    if output.exists() or any((output.parent / n).exists() for n in ("candidate.jsonl", "baseline.jsonl", "summary.json")):
        raise ValueError("Unscored output directory required")
    _, rows = source_rows(base)
    for row in rows:
        if row["status"] == "ready" and digest(base / row["relative_path"]) != row["sha256"]:
            raise ValueError("Source bytes changed before freeze")
    write_new(output, {"schema": SCHEMA, "base": str(base.resolve()), "model_root": str(model_root.resolve()),
        "baseline_root": str(baseline_root.resolve()), "method": METHOD, "rows": rows, "native_cases": native_plan(rows),
        "implementation_sha256": implementation(), "runtime": runtime(), "installed_sha256": installed(),
        "assets_sha256": asset_hashes(model_root, baseline_root), "preparation_sha256": preparation(),
        "ffprobe_version": subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0],
        "recordings_scored": 0, "calibration_fitted": False, "deployment_eligible": False})
    print(json.dumps({"lock_sha256": digest(output), "planned_inputs": len(rows), "recordings_scored": 0}), flush=True)


def check(lock_path, assets=True):
    body = json.loads(lock_path.read_text())
    if (body["schema"] != SCHEMA or body["method"] != METHOD or body["implementation_sha256"] != implementation()
            or body["runtime"] != runtime() or body["installed_sha256"] != installed()
            or body["rows"] != source_rows(Path(body["base"]))[1]
            or body["native_cases"] != native_plan(body["rows"]) or body["preparation_sha256"] != preparation()
            or body["recordings_scored"] != 0 or body["calibration_fitted"] is not False or body["deployment_eligible"] is not False):
        raise ValueError("Complete original prospective freeze required")
    if assets and body["assets_sha256"] != asset_hashes(Path(body["model_root"]), Path(body["baseline_root"])):
        raise ValueError("Frozen original model assets changed")
    return body


def public(lock_path, output):
    body = check(lock_path)
    clean = {k: v for k, v in body.items() if k not in {"base", "model_root", "baseline_root"}}
    clean["implementation_sha256"] = [{"path": name, "sha256": value} for name, value in clean["implementation_sha256"].items()]
    value = {"schema": "chipmunk.video-demamba-public-freeze/13", "private_lock_sha256": digest(lock_path),
             "lock": clean, "scope": "Consumed development and predeclared comparison; no raw media, weights, customer data or predictive results"}
    if any(x in json.dumps(value) for x in ("/Users/", "/private/", "/home/")):
        raise ValueError("Private paths in public freeze")
    write_new(output, value)


def open_journal(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x")
    stream.write(json.dumps(header, allow_nan=False) + "\n")
    stream.flush()
    return stream


def score_candidate(lock_path, output):
    body = check(lock_path)
    import torch
    torch.set_num_threads(2)
    model, _, loading = candidate.load(Path(body["model_root"]))
    with open_journal(output, {"lock_sha256": digest(lock_path), "system": "candidate", "loading": loading}) as stream:
        for row in body["rows"]:
            result = dict(row)
            started = time.monotonic()
            try:
                if row["status"] != "ready":
                    result.update(status="unavailable", score=None, logit=None, support=None)
                else:
                    path = Path(body["base"]) / row["relative_path"]
                    if digest(path) != row["sha256"]:
                        raise ValueError("Frozen video changed")
                    data, support = clock.read_native(path)
                    with torch.inference_mode():
                        logit = float(model(torch.from_numpy(data))[0, 0])
                    if not math.isfinite(logit):
                        raise ValueError("Nonfinite candidate response")
                    result.update(status="ok", score=float(torch.sigmoid(torch.tensor(logit))), logit=logit, support=support)
            except clock.UnsupportedSource as error:
                result.update(status="not_applicable", score=None, logit=None, support=None, reason=str(error))
            except Exception as error:
                result.update(status="failed", score=None, logit=None, support=None, error_type=type(error).__name__)
            result["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({"dataset": row["dataset"], "id": row["id"], "variant": row["variant"], "status": result["status"]}), flush=True)
            if result["status"] == "failed":
                raise ValueError("Candidate failed; retain journal before proceeding")


def baseline_rows(outputs, keys, durations=None):
    from app.bench.score_v2 import DETECTOR_STAGES_BY_KIND, detector_row

    entries = {entry["detector_id"]: entry for stage in DETECTOR_STAGES_BY_KIND["video"]
               for entry in (outputs.get(stage) or {}).get("detectors") or []}
    return {keys[identifier]: detector_row(entries.get(identifier, {"status": "skipped", "reason": "No synthesis result"}), (durations or {}).get(identifier))
            for identifier in CHECKS}


def baseline_scorer(body, scratch):
    from app.bench.score_v2 import V2VideoScorer
    from app.core.config import settings
    from app.engine.registry import build_registry, get_registry

    class SynthesisScorer(V2VideoScorer):
        def rows_from_outputs(self, outputs, runtime_ms, durations=None):
            return baseline_rows(outputs, self.keys, durations)

    if settings.models_dir is None or settings.models_dir.resolve() != Path(body["baseline_root"]) or settings.device != "mps":
        raise ValueError("Use the frozen local model root and MPS baseline device")
    spec = get_registry().manifest
    registry = build_registry(policy={"preset": "commercial", "deny_detectors": [d.id for d in spec.detectors if d.id not in CHECKS]})
    if any(not registry.decision(name).enabled for name in CHECKS):
        raise ValueError("Every original synthesis adapter must be enabled")
    scorer = SynthesisScorer(Path(scratch) / "database", registry=registry, profile="full")
    if any(scorer.keys[name].endswith("@not-installed") for name in CHECKS):
        scorer.close()
        raise ValueError("Original baseline assets missing")
    return scorer


def score_baseline(lock_path, output):
    body = check(lock_path)
    import torch

    from app.bench.score import ScoreTask
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="demamba-v13-baseline-") as scratch:
        scorer = baseline_scorer(body, scratch)
        try:
            with open_journal(output, {"lock_sha256": digest(lock_path), "system": "baseline", "versions": {k: scorer.versions[k] for k in CHECKS}}) as stream:
                for row in body["rows"]:
                    result, started = dict(row), time.monotonic()
                    try:
                        if row["status"] != "ready":
                            result.update(status="unavailable", score=None, checks={})
                        else:
                            path = Path(body["base"]) / row["relative_path"]
                            if digest(path) != row["sha256"]:
                                raise ValueError("Frozen baseline video changed")
                            scored = scorer.score(ScoreTask(row["id"], row["variant"], "none", path))
                            values = {k: scored[scorer.keys[k]] for k in CHECKS}
                            if any(v["status"] == "failed" for v in values.values()):
                                raise ValueError("Baseline adapter failed")
                            usable = [v["prob"] for v in values.values() if v["status"] == "ok" and v["prob"] is not None]
                            result.update(status="ok" if usable else "not_applicable", score=float(np.mean(usable)) if usable else None, checks=values)
                    except Exception as error:
                        result.update(status="failed", score=None, checks={}, error_type=type(error).__name__)
                    result["runtime_s"] = time.monotonic() - started
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps({"dataset": row["dataset"], "id": row["id"], "variant": row["variant"], "status": result["status"]}), flush=True)
                    if result["status"] == "failed":
                        raise ValueError("Baseline failed; retain journal before proceeding")
        finally:
            scorer.close()


def key(row):
    return (row["dataset"], row["id"], row["variant"])


def valid_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_support(support):
    if (support is None or support["schema"] != "chipmunk.video-demamba-native-input/13"
            or support["tensor_shape"] != [1, 8, 3, 224, 224] or support["synthetic_padding_frames"] != 0
            or support["source_frame_count"] != support["decoded_clock_entries"]
            or support["plan"] != clock.sample_plan(support["plan"]["fps"], support["source_frame_count"])):
        raise ValueError("Complete original eight-picture clock/input support required")
    frames = support["frames"]
    if len(frames) != 8 or [r["native_index"] for r in frames] != support["plan"]["selected_indices"]:
        raise ValueError("Missing, duplicate or reordered native frames")
    for i, row in enumerate(frames):
        if (not math.isfinite(row["pts_s"]) or (i and row["pts_s"] <= frames[i - 1]["pts_s"])
                or not math.isclose(row["relative_to_first_decoded_pts_s"], row["pts_s"] - support["first_decoded_pts_s"], abs_tol=1e-12)):
            raise ValueError("Changed or guessed source-frame clock")
        width, height = row["native_size"]
        if row["crop_box_xyxy"] != list(clock.crop_box(height, width)):
            raise ValueError("Native crop geometry changed")
        for name in ("native_rgb_sha256", "resized_rgb_sha256", "normalized_tensor_sha256"):
            value = row[name]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Original pixel/tensor commitments required")
    value = support["tensor_sha256"]
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Complete input commitment required")


def read_scores(path, lock_path, body, system):
    with path.open() as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    if header["lock_sha256"] != digest(lock_path) or header["system"] != system or len(rows) != 288:
        raise ValueError("Complete original source-bound288-input journal required")
    for source, row in zip(body["rows"], rows, strict=True):
        if type(row["label"]) is not int or any(source[k] != row[k] for k in source if k != "status"):
            raise ValueError("Journal source, truth, role or format changed")
        if row["status"] not in {"ok", "not_applicable", "unavailable", "failed"}:
            raise ValueError("Unknown assessment status")
        if (source["status"] == "unavailable") != (row["status"] == "unavailable"):
            raise ValueError("Originally available sources cannot disappear")
        if row["status"] == "ok":
            if not valid_probability(row["score"]):
                raise ValueError("Finite assessed probability required")
            if system == "candidate":
                if type(row["logit"]) not in (int, float) or not math.isfinite(row["logit"]):
                    raise ValueError("Candidate raw logit required")
                logit = row["logit"]
                expected = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
                if abs(row["score"] - expected) > 1e-7:
                    raise ValueError("Original float32 sigmoid/raw correspondence changed")
                validate_support(row["support"])
        elif row["score"] is not None:
            raise ValueError("Unassessed/failed sources cannot acquire votes")
        if system == "baseline" and source["status"] == "ready":
            checks = row["checks"]
            if set(checks) != set(CHECKS):
                raise ValueError("Both original baseline checks/statuses required")
            if any(v["status"] not in {"ok", "not_applicable", "skipped"} for v in checks.values()):
                raise ValueError("A baseline failure is not missing evidence")
            if any(v["status"] != "ok" and v["prob"] is not None for v in checks.values()):
                raise ValueError("Unassessed baseline check acquired a vote")
            values = [v["prob"] for v in checks.values() if v["status"] == "ok"]
            if any(not valid_probability(value) for value in values):
                raise ValueError("Invalid original baseline response")
            if values and (row["status"] != "ok" or not math.isclose(float(np.mean(values)), row["score"], abs_tol=1e-12)):
                raise ValueError("Original mean-of-available baseline response changed")
            if not values and row["status"] == "ok":
                raise ValueError("Missing baseline checks cannot acquire a score")
    return rows


def numerical():
    # Retain the original reviewed calibration/statistical implementation.
    sys.path.insert(0, str(PROJECT / "backend/scripts"))
    return importlib.import_module("summarize_video_upgrade_eval")


def direction(p):
    return "inconclusive" if not valid_probability(p) else "ai" if p >= .8 else "original" if p <= .2 + 1e-12 else "inconclusive"


def panel_counts(rows):
    counts = Counter()
    for row in rows:
        decision, label = row["decision"], row["label"]
        counts["planned"] += 1
        counts["assessed"] += row["status"] == "ok"
        counts["failed"] += row["status"] == "failed"
        counts["inconclusive"] += decision == "inconclusive"
        counts["correct_ai"] += label == 1 and decision == "ai"
        counts["correct_original"] += label == 0 and decision == "original"
        counts["false_ai"] += label == 0 and decision == "ai"
        counts["false_original"] += label == 1 and decision == "original"
        counts["ai_without_ai_conclusion"] += label == 1 and decision != "ai"
    return dict(counts)


def compare(lock_path, baseline_path, candidate_path, output):
    body = check(lock_path)
    old = read_scores(baseline_path, lock_path, body, "baseline")
    new = read_scores(candidate_path, lock_path, body, "candidate")
    stats = numerical()
    calibrated, comparisons, fitted = {"baseline": [], "candidate": []}, {}, {}
    calibration_support = True
    for dataset in DATASETS:
        cal = [i for i, row in enumerate(body["rows"]) if row["dataset"] == dataset and row["split"] == "calibration" and old[i]["status"] == new[i]["status"] == "ok"]
        labels = [old[i]["label"] for i in cal]
        groups = {old[i]["group"] for i in cal}
        if set(labels) != {0, 1} or len(groups) < 8:
            raise ValueError("Frozen common calibration support required before comparison")
        fitted[dataset] = {"common_rows": len(cal), "groups": len(groups),
            "baseline": stats.fit([float(stats.logit(np.clip(old[i]["score"], 1e-7, 1 - 1e-7))) for i in cal], labels),
            "candidate": stats.fit([new[i]["logit"] for i in cal], labels)}
        for system, rows in (("baseline", old), ("candidate", new)):
            for i, row in enumerate(rows):
                if row["dataset"] != dataset or row["split"] != "test":
                    continue
                raw = (float(stats.logit(np.clip(row["score"], 1e-7, 1 - 1e-7))) if system == "baseline" else row["logit"]) if row["status"] == "ok" else None
                p = float(stats.calibrated([raw], fitted[dataset][system])[0]) if raw is not None else None
                calibrated[system].append({**row, "p": p, "decision": direction(p)})
        by_system = {name: {key(row): row for row in rows} for name, rows in calibrated.items()}
        for variant in ("original", "matched"):
            keys = [key(r) for r in body["rows"] if r["dataset"] == dataset and r["split"] == "test" and r["variant"] == variant]
            paired = [k for k in keys if all(by_system[s][k]["status"] == "ok" for s in by_system)]
            y = np.asarray([by_system["baseline"][k]["label"] for k in paired])
            if set(y) != {0, 1}:
                raise ValueError("Paired development must retain both truth classes")
            groups = np.asarray([by_system["baseline"][k]["group"] for k in paired])
            a, b = [np.asarray([by_system[s][k]["p"] for k in paired]) for s in ("baseline", "candidate")]
            comparisons[dataset + ":" + variant] = {
                "paired": len(paired), "source_groups": len(set(groups)),
                "baseline_metrics": stats.metrics(y, a), "candidate_metrics": stats.metrics(y, b),
                "paired_brier_difference": stats.paired_delta(groups, (y - b)**2 - (y - a)**2),
                "paired_logloss_difference": stats.paired_delta(groups, stats.logloss(y, b) - stats.logloss(y, a)),
                **{s: panel_counts([by_system[s][k] for k in keys]) for s in by_system}}
    totals = {s: panel_counts(rows) for s, rows in calibrated.items()}
    distinct = {s: len({(r["dataset"], r["id"]) for r in rows if r["label"] == 1 and r["decision"] == "ai"}) for s, rows in calibrated.items()}
    wrong = all(new <= old + 1 for name in ("false_ai", "false_original")
                for old, new in [(totals["baseline"][name], totals["candidate"][name])]
                + [(p["baseline"][name], p["candidate"][name]) for p in comparisons.values()])
    gates = {"complete_matrix": len(old) == len(new) == 288,
        "no_failures": not any(r["status"] == "failed" for r in old + new),
        "calibration_support": calibration_support,
        "brier_gain": all(p["paired_brier_difference"]["mean"] <= -.005 for p in comparisons.values()),
        "brier_interval": all(p["paired_brier_difference"]["group_bootstrap_95"][1] < 0 for p in comparisons.values()),
        "wrong_directions": wrong,
        "ai_usefulness": distinct["candidate"] >= distinct["baseline"] + 4 and all(p["candidate"]["correct_ai"] >= p["baseline"]["correct_ai"] - 2 for p in comparisons.values()),
        "original_retention": totals["candidate"]["correct_original"] >= totals["baseline"]["correct_original"] - 2 and all(p["candidate"]["correct_original"] >= p["baseline"]["correct_original"] - 2 for p in comparisons.values()),
        "total_correct": totals["candidate"]["correct_ai"] + totals["candidate"]["correct_original"] > totals["baseline"]["correct_ai"] + totals["baseline"]["correct_original"],
        "scope_retention": all(a["status"] != "ok" or b["status"] == "ok" for a, b in zip(old, new, strict=True))}
    write_new(output, {"schema": "chipmunk.video-demamba-development-result/13", "lock_sha256": digest(lock_path),
        "baseline_sha256": digest(baseline_path), "candidate_sha256": digest(candidate_path), "fitted_calibration": fitted,
        "comparisons": comparisons, "totals": totals, "distinct_correct_ai_sources": distinct,
        "gates_before_native_audit": gates, "native_audit_pending": True, "advance_to_fresh_confirmation": False,
        "deployment_eligible": False, "calibrated_development": calibrated,
        "scope": "Consumed component comparison, own calibration per method on common calibration sources, all192 development variants retained; not fresh or whole-file service confidence"})


def replay_pixels(path, support):
    """Separate CLI RGB/PTS capture; codec lineage remains shared with OpenCV."""
    validate_support(support)
    frames = support["frames"]
    sizes = {tuple(r["native_size"]) for r in frames}
    if len(sizes) != 1:
        raise ValueError("Constant native geometry required for raw-video replay")
    width, height = next(iter(sizes))
    expression = "+".join(f"eq(n\\,{r['native_index']})" for r in frames)
    result = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", "-copyts", "-i", str(path),
                             "-map", "0:v:0", "-vf", "select=" + expression + ",showinfo", "-frames:v", "8",
                             "-fps_mode", "passthrough", "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                            capture_output=True, timeout=120, check=True)
    times = [float(v) for v in re.findall(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([^\s]+)", result.stderr.decode(errors="replace"))]
    if len(times) != 8 or len(result.stdout) != 8 * width * height * 3:
        raise ValueError("Complete independent RGB/clock capture required")
    pixels = np.frombuffer(result.stdout, np.uint8).reshape(8, height, width, 3)
    tensors, seen = [], []
    for rgb, timestamp, trace in zip(pixels, times, frames, strict=True):
        if abs(timestamp - trace["pts_s"]) > 1.1e-6:
            raise ValueError("Separate native clock correspondence failed")
        if hashlib.sha256(rgb.tobytes()).hexdigest() != trace["native_rgb_sha256"]:
            raise ValueError("Separate native RGB correspondence failed")
        tensor, again = clock.preprocess(np.ascontiguousarray(rgb[:, :, ::-1]))
        if any(again[k] != trace[k] for k in again):
            raise ValueError("Separate geometry/tensor correspondence failed")
        tensors.append(tensor)
        seen.append(trace["native_index"])
    data = np.ascontiguousarray(np.stack(tensors)[None], dtype=np.float32)
    if seen != support["plan"]["selected_indices"] or hashlib.sha256(data.astype("<f4").tobytes()).hexdigest() != support["tensor_sha256"]:
        raise ValueError("Complete independent source input required")
    return data


def audit_native(lock_path, baseline_path, candidate_path, summary_path, output):
    body = check(lock_path)
    old = {key(r): r for r in read_scores(baseline_path, lock_path, body, "baseline")}
    new = {key(r): r for r in read_scores(candidate_path, lock_path, body, "candidate")}
    summary = json.loads(summary_path.read_text())
    if summary["lock_sha256"] != digest(lock_path) or summary["baseline_sha256"] != digest(baseline_path) or summary["candidate_sha256"] != digest(candidate_path):
        raise ValueError("Original complete comparison required")
    with tempfile.TemporaryDirectory(prefix="demamba-v13-numerical-") as scratch:
        reconstructed = Path(scratch) / "summary.json"
        compare(lock_path, baseline_path, candidate_path, reconstructed)
        if json.loads(reconstructed.read_text()) != summary:
            raise ValueError("Original complete calibration/count/gate reconstruction differs")
    import torch

    from app.bench.score import ScoreTask
    torch.set_num_threads(2)
    model, _, _ = candidate.load(Path(body["model_root"]))
    cases = []
    with tempfile.TemporaryDirectory(prefix="demamba-v13-native-") as scratch:
        scorer = baseline_scorer(body, scratch)
        try:
            for value in body["native_cases"]:
                identity = tuple(value)
                a, b = old[identity], new[identity]
                record = {"identity": value, "passed": False}
                try:
                    path = Path(body["base"]) / b["relative_path"]
                    if digest(path) != b["sha256"]:
                        raise ValueError("Audited source bytes changed")
                    try:
                        data, support = clock.read_native(path)
                    except clock.UnsupportedSource:
                        if b["status"] != "not_applicable":
                            raise ValueError("Candidate applicability changed") from None
                        record.update(candidate_unassessed_reproduced=True, native_frames=0)
                    else:
                        if b["status"] != "ok" or support != b["support"]:
                            raise ValueError("Candidate native input correspondence changed")
                        independent = replay_pixels(path, support)
                        with torch.inference_mode():
                            repeated = float(model(torch.from_numpy(independent))[0, 0])
                        if abs(repeated - b["logit"]) > 1e-5:
                            raise ValueError("Candidate native response differs")
                        record.update(candidate_max_logit_difference=abs(repeated - b["logit"]), native_frames=8)
                    baseline = scorer.score(ScoreTask(a["id"], a["variant"], "none", path))
                    differences = []
                    for check_id in CHECKS:
                        actual, expected = baseline[scorer.keys[check_id]], a["checks"][check_id]
                        if actual["status"] != expected["status"] or (actual["prob"] is None) != (expected["prob"] is None):
                            raise ValueError("Baseline applicability differs")
                        if actual["prob"] is not None:
                            differences.append(abs(actual["prob"] - expected["prob"]))
                    if max(differences, default=0.) > 1e-5:
                        raise ValueError("Baseline native response differs")
                    record.update(passed=True, baseline_max_probability_difference=max(differences, default=0.))
                except Exception as error:
                    record.update(error_type=type(error).__name__, reason=str(error))
                cases.append(record)
                print(json.dumps(record), flush=True)
        finally:
            scorer.close()
    gates = {**summary["gates_before_native_audit"], "native_correspondence": len(cases) == 16 and all(r["passed"] for r in cases)}
    write_new(output, {"schema": "chipmunk.video-demamba-native-audit/13", "lock_sha256": digest(lock_path),
        "summary_sha256": digest(summary_path), "cases": cases, "gates": gates,
        "advance_to_fresh_confirmation": all(gates.values()), "deployment_eligible": False,
        "outside_reviewers": 0, "source_clips_independent": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "public", "candidate", "baseline", "compare", "native"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.model_root is None or args.baseline_root is None:
            parser.error("Freeze requires both local model roots")
        freeze(args.input, args.model_root, args.baseline_root, args.output)
    elif args.mode == "public":
        public(args.input, args.output)
    elif args.mode == "candidate":
        score_candidate(args.input, args.output)
    elif args.mode == "baseline":
        score_baseline(args.input, args.output)
    else:
        if args.baseline is None or args.candidate is None or (args.mode == "native" and args.summary is None):
            parser.error("Complete original score paths required")
        if args.mode == "compare":
            compare(args.input, args.baseline, args.candidate, args.output)
        else:
            audit_native(args.input, args.baseline, args.candidate, args.summary, args.output)
