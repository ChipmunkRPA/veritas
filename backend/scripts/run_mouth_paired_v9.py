"""Frozen source-matched mouth experiment over immutable V8 inputs and baseline."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.video.adapters.loaders import load_dinov2_small
from app.detectors.video.face_sequence import REVIEW_POINT
from scripts import compare_mouth_probe_v6 as comparison
from scripts import compare_mouth_sequence_v4 as baseline
from scripts import mouth_learner_v8 as previous_method
from scripts import mouth_paired_v9 as method
from scripts import run_mouth_learning_v8 as previous
from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_probe_v6 import margins


def implementation():
    root = Path(__file__).resolve().parents[2]
    names = [
        "backend/scripts/run_mouth_paired_v9.py",
        "backend/scripts/mouth_paired_v9.py",
        "backend/scripts/audit_mouth_paired_v9.py",
        "backend/scripts/audit_mouth_video_v8.py",
        "backend/scripts/replay_mouth_learning_v8.py",
        "backend/scripts/audit_mouth_probe_v6.py",
        "docs/mouth-paired-v9-development.md",
    ]
    return {**previous.pins(), **{p: digest(root / p) for p in names}}


def freeze(parent, output):
    old_lock = previous.check_lock(parent / "lock.json")
    head = json.loads((parent / "trained/head.json").read_text())
    result = json.loads((parent / "summary.json").read_text())
    if (
        head["lock_sha256"] != digest(parent / "lock.json")
        or head["learned_sha256"] != digest(parent / "trained/learned.safetensors")
        or result["head_sha256"] != digest(parent / "trained/head.json")
        or result["score_receipt_sha256"] != digest(parent / "development-scores.jsonl")
    ):
        raise ValueError("Immutable V8 fit/evaluation identities differ")
    _, rows, receipt, _ = previous.load_cache(parent / "lock.json", parent / "cache")
    pairs = method.make_pairs(rows)
    if len(pairs) != 768 or sum(method.teacher_eligible(r) for r in rows) != 1024:
        raise ValueError("The complete fixed fitting-pair/teacher panel is required")
    sources = {
        "parent_lock": parent / "lock.json",
        "parent_head": parent / "trained/head.json",
        "parent_weights": parent / "trained/learned.safetensors",
        "parent_scores": parent / "development-scores.jsonl",
        "parent_summary": parent / "summary.json",
        "training_audit": parent.parent / "training-source-audit.json",
        "parent_video_audit": parent / "video-source-audit.json",
        "parent_forward_replay": parent / "full-forward-replay.json",
    }
    if (
        not json.loads(sources["training_audit"].read_text())["passed"]
        or json.loads(sources["parent_video_audit"].read_text())["passed"] != 88
        or not json.loads(sources["parent_forward_replay"].read_text())["passed"]
    ):
        raise ValueError("All V8 source/numerical audits must pass")
    output.mkdir(parents=True, exist_ok=False)
    lock = {
        "schema": "chipmunk.mouth-source-paired-lock/9",
        "method": method.METHOD,
        **{k: str(v.resolve()) for k, v in sources.items()},
        **{k + "_sha256": digest(v) for k, v in sources.items()},
        "cache": str((parent / "cache").resolve()),
        "cache_receipt_sha256": digest(parent / "cache/receipt.json"),
        "training_manifest_sha256": old_lock["training_manifest_sha256"],
        "development": old_lock["development"],
        "implementation_sha256": implementation(),
        "pairs": [{"positive_id": rows[p]["id"], "negative_id": rows[n]["id"]} for p, n in pairs],
        "fitting_teacher_offsets": sum(
            len(b["views"])
            for r, b in zip(rows, receipt["bindings"], strict=True)
            if method.teacher_eligible(r)
        ),
        "deployment_eligible": False,
    }
    previous.write(output / "lock.json", lock)
    print(
        json.dumps(
            {
                "lock_sha256": digest(output / "lock.json"),
                "unique_fit_pairs": len(pairs),
                "teacher_fit_offsets": lock["fitting_teacher_offsets"],
            }
        )
    )


def check_lock(path):
    lock = json.loads(path.read_text())
    if lock["implementation_sha256"] != implementation():
        raise ValueError("Frozen V9 learning/scoring/audit implementation changed")
    for key in (
        "parent_lock",
        "parent_head",
        "parent_weights",
        "parent_scores",
        "parent_summary",
        "training_audit",
        "parent_video_audit",
        "parent_forward_replay",
    ):
        if digest(Path(lock[key])) != lock[key + "_sha256"]:
            raise ValueError("Immutable parent input changed: " + key)
    previous.check_lock(Path(lock["parent_lock"]))
    return lock


def source_cache(lock_path):
    lock = check_lock(lock_path)
    _, rows, receipt, tokens = previous.load_cache(Path(lock["parent_lock"]), Path(lock["cache"]))
    if (
        digest(Path(lock["cache"]) / "receipt.json") != lock["cache_receipt_sha256"]
        or receipt["manifest_sha256"] != lock["training_manifest_sha256"]
    ):
        raise ValueError("Reused source/cache bindings changed")
    pairs = method.make_pairs(rows)
    if [{"positive_id": rows[p]["id"], "negative_id": rows[n]["id"]} for p, n in pairs] != lock["pairs"]:
        raise ValueError("Frozen fitting pairs changed")
    return lock, rows, receipt, tokens


def initialized(lock):
    head = json.loads(Path(lock["parent_head"]).read_text())
    backbone = previous.loaded_candidate(Path(lock["parent_head"])).obj.model
    learner = previous_method.MouthLearner(backbone).to("mps")
    state = load_file(Path(lock["parent_weights"]))
    learner.load_state_dict(state, strict=False)
    expected = {k for k, p in learner.named_parameters() if p.requires_grad}
    if (
        set(state) != expected
        or not all(torch.equal(learner.state_dict()[k].detach().cpu(), v) for k, v in state.items())
        or previous_method.linear_head(learner)
        != {k: head[k] for k in ("coef", "intercept", "mean", "scale")}
    ):
        raise ValueError("Initialization differs from the exact V8 learned state/head")
    return learner


def train(lock_path, output):
    lock, rows, receipt, tokens = source_cache(lock_path)
    output.mkdir(parents=True, exist_ok=False)
    weights = previous.setup()
    torch.manual_seed(method.SEED)
    learner = initialized(lock)
    teacher = copy.deepcopy(learner).eval().requires_grad_(False)
    result = method.optimize(learner, teacher, tokens, rows, receipt["bindings"])
    head = previous_method.linear_head(learner)
    point_rows = [r for r in rows if r["role"] == "point"]
    offsets = [
        b["views"][0]["offset"]
        for r, b in zip(rows, receipt["bindings"], strict=True)
        if r["role"] == "point"
    ]
    values = []
    with torch.no_grad():
        for start in range(0, len(offsets), method.BATCH):
            values.extend(
                learner.representation(tokens[offsets[start : start + method.BATCH]].to("mps")).cpu().numpy()
            )
    responses = margins(values, head)
    point = previous_method.choose_point(point_rows, responses)
    target = output / "learned.safetensors"
    save_file(
        {k: p.detach().cpu().contiguous() for k, p in learner.named_parameters() if p.requires_grad}, target
    )
    body = {
        "schema": "chipmunk.mouth-source-paired-head/9",
        "method": method.METHOD,
        "lock_sha256": digest(lock_path),
        "cache_receipt_sha256": lock["cache_receipt_sha256"],
        "parent_head_sha256": lock["parent_head_sha256"],
        "learned_sha256": digest(target),
        "weights": weights,
        **head,
        "review_point": point,
        "learning": result,
        "fit_groups": sorted({r["group"] for r in rows if r["role"] == "fit"}),
        "point_groups": sorted({r["group"] for r in point_rows}),
        "fit_stills": 1280,
        "point_stills": 320,
        "point_negative_stills": sum(not r["label"] for r in point_rows),
        "point_responses": [
            {"id": r["id"], "label": r["label"], "margin": float(v)}
            for r, v in zip(point_rows, responses, strict=True)
        ],
        "response_kind": "raw signed margin; not calibrated AI probability",
        "torch_version": torch.__version__,
        "device": "mps",
        "deployment_eligible": False,
    }
    previous.write(output / "head.json", body)
    print(
        json.dumps(
            {
                "head_sha256": digest(output / "head.json"),
                "learned_sha256": digest(target),
                "review_point": point,
                "trainable_parameters": sum(result["trainable_parameters"].values()),
            }
        ),
        flush=True,
    )


def native_parity(new, old):
    omitted = {"features_float32_sha256", "margin"}
    return [{k: v for k, v in r.items() if k not in omitted} for r in new] == [
        {k: v for k, v in r.items() if k not in omitted} for r in old
    ]


def score(lock_path, head_path, output):
    lock = check_lock(lock_path)
    head = json.loads(head_path.read_text())
    if head["lock_sha256"] != digest(lock_path):
        raise ValueError("V9 head uses a different experiment")
    weights = previous.setup()
    if weights != head["weights"]:
        raise ValueError("Verified pretrained weights changed")
    models = {
        "yunet.2023mar": load_yunet("yunet.2023mar", "cpu", "float32"),
        "dinov2.small": load_dinov2_small("dinov2.small", "cpu", "float32"),
        "probe.dino": previous.loaded_candidate(head_path),
    }
    _, *prior = [json.loads(line) for line in Path(lock["parent_scores"]).read_text().splitlines()]
    expected = []
    for panel in lock["development"]:
        manifest = Path(panel["manifest"])
        _, *refs = [json.loads(line) for line in Path(panel["receipt"]).read_text().splitlines()]
        expected.extend(
            (case, reference, manifest.parent, panel["family"])
            for case, reference in zip(json.loads(manifest.read_text())["items"], refs, strict=True)
        )
    if [r["id"] for r in prior] != [r[0]["id"] for r in expected]:
        raise ValueError("V8 outcomes no longer correspond to the complete planned panel")
    header = {
        "schema": "chipmunk.mouth-source-paired-scores/9",
        "method": method.METHOD,
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "review_point": head["review_point"],
        "parent_scores_sha256": lock["parent_scores_sha256"],
        "planned": 88,
        "weights": weights,
        "candidate_device": "mps",
        "baseline_device": "cpu",
        "torch_version": torch.__version__,
    }
    complete = []
    if output.exists():
        old_header, *complete = [json.loads(line) for line in output.read_text().splitlines()]
        if old_header != header or [r["id"] for r in complete] != [
            r[0]["id"] for r in expected[: len(complete)]
        ]:
            raise ValueError("Cannot resume changed V9 experiment")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    with output.open("a") as stream:
        for (case, reference, root, family), old in zip(
            expected[len(complete) :], prior[len(complete) :], strict=True
        ):
            started = time.monotonic()
            row = {
                **case,
                "family": family,
                "mask_root": str(root),
                "status": "failed",
                "base_windows": reference["base_windows"],
                "probe_windows": [],
            }
            try:
                if case["group"] in set(head["fit_groups"] + head["point_groups"]):
                    raise ValueError("Video actor leaked into fitting/point roles")
                path = root / case["path"]
                if case["status"] != "ready" or digest(path) != case["sha256"]:
                    raise ValueError("Prepared development source changed")
                with patch.object(comparison, "features", previous_method.features):
                    row.update(comparison.score_case(path, case, reference, models, head))
                if not native_parity(row["model_inputs"], old["model_inputs"]):
                    raise ValueError("V9 and V8 use different native picture/time/crop/tensor inputs")
                row["prior_native_input_parity"] = True
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            row["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "reason": row.get("reason"),
                        "seconds": row["runtime_s"],
                    }
                ),
                flush=True,
            )


def gates(a, b, families, rows):
    current, candidate = baseline.summary(a), baseline.summary(b)
    losses = sum(x["localized"] and not y["localized"] for x, y in zip(a, b, strict=True))
    lp = [r for r, family in zip(b, families, strict=True) if family == "liveportrait"]
    muse = [r for r, family in zip(b, families, strict=True) if family == "musetalk"]
    ordinary = {r["condition"] for r in a if not r["label"]}
    return {
        "six_net_additional_locations": candidate["localized_ai"] >= current["localized_ai"] + 6,
        "first_family_six_locations": sum(r["localized"] for r in lp) >= 6,
        "both_first_family_conditions": all(
            sum(r["localized"] for r in lp if r["condition"] == c) >= 2 for c in ("ai-redraw", "ai-retime")
        ),
        "retained_second_family": sum(r["localized"] for r in muse) >= 13,
        "at_most_one_lost_location": losses <= 1,
        "total_ordinary_flags": candidate["non_ai_flags"] <= current["non_ai_flags"] + 1,
        "condition_ordinary_flags": all(
            sum(r["flagged"] for r in b if r["condition"] == c)
            <= sum(r["flagged"] for r in a if r["condition"] == c) + 1
            for c in ordinary
        ),
        "spatial_iou_gain": candidate["mean_ai_spatial_iou"] >= current["mean_ai_spatial_iou"] + 0.02,
        "retained_temporal_iou": candidate["mean_ai_temporal_iou"] >= current["mean_ai_temporal_iou"],
        "retained_coverage": all(not x["assessed"] or y["assessed"] for x, y in zip(a, b, strict=True)),
        "complete_replays": len(rows) == 88
        and all(
            r["status"] == "ok" and r.get("baseline_replay_parity") and r.get("prior_native_input_parity")
            for r in rows
        ),
    }


def review(lock_path, receipt, output):
    lock = check_lock(lock_path)
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    parent = json.loads(Path(lock["parent_summary"]).read_text())
    expected = [r["id"] for r in parent["cases"]["candidate"]]
    if header["lock_sha256"] != digest(lock_path) or [r["id"] for r in rows] != expected:
        raise ValueError("Every frozen development outcome is required")
    current, candidate = [], []
    for row in rows:
        current += baseline.evaluate([row], "base", REVIEW_POINT, Path(row["mask_root"]))
        candidate += baseline.evaluate([row], "probe", header["review_point"], Path(row["mask_root"]))
    if current != parent["cases"]["baseline"]:
        raise ValueError("Current source/mask evaluation changed from V8")
    prior = parent["cases"]["candidate"]
    families = [r["family"] for r in rows]
    advancement = gates(prior, candidate, families, rows)
    body = {
        "schema": "chipmunk.mouth-source-paired-summary/9",
        "lock_sha256": digest(lock_path),
        "head_sha256": header["head_sha256"],
        "score_receipt_sha256": digest(receipt),
        "parent_scores_sha256": lock["parent_scores_sha256"],
        "review_point": header["review_point"],
        "current": baseline.summary(current),
        "baseline": baseline.summary(prior),
        "candidate": baseline.summary(candidate),
        "gates": advancement,
        "advance_to_confirmation_design": all(advancement.values()),
        "deployment_eligible": False,
        "terminal_statuses": dict(Counter(r["status"] for r in rows)),
        "families": {
            f: {
                "baseline": baseline.summary(
                    [r for r, name in zip(prior, families, strict=True) if name == f]
                ),
                "candidate": baseline.summary(
                    [r for r, name in zip(candidate, families, strict=True) if name == f]
                ),
            }
            for f in sorted(set(families))
        },
        "conditions": {
            c: {
                "baseline": baseline.summary([r for r in prior if r["condition"] == c]),
                "candidate": baseline.summary([r for r in candidate if r["condition"] == c]),
            }
            for c in sorted({r["condition"] for r in rows})
        },
        "cases": {"current": current, "baseline": prior, "candidate": candidate},
        "limitations": [
            "All source actors, sentences and point data were consumed development; no fresh or outside confirmation.",
            "Teacher/paired labels occur only during fitting; single model at inference, no original reference or generator label.",
            "Raw margins and measurement regions are not calibrated authenticity confidence or causal pixel proof.",
        ],
    }
    previous.write(output, body)
    print(
        json.dumps(
            {k: v for k, v in body.items() if k not in ("cases", "families", "conditions", "limitations")}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "train", "score", "review"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--receipt", type=Path)
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.input, a.output)
    elif a.stage == "train":
        train(a.input, a.output)
    elif a.stage == "score" and a.head:
        score(a.input, a.head, a.output)
    elif a.stage == "review" and a.receipt:
        review(a.input, a.receipt, a.output)
    else:
        p.error("score requires --head; review requires --receipt")
