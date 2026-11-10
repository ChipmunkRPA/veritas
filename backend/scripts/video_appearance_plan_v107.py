"""Prospective source-clock support for a fixed appearance-score transfer test.

No decoder or model import. Constructed-byte binding remains a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
SOURCE = BASE / "video-native-inserts-v95"
CALIBRATION = BASE / "video-joint-scalars-v106/evaluation/fitted-models.json"
TRIAL = BASE / "video-appearance-plan-v107"
PUBLIC = ROOT / "docs/evaluation-panels/video-appearance-plan-freeze-v107.json"
PANEL = ROOT / "docs/evaluation-panels"
CONDITIONS = ("original", "swap_a", "swap_b", "ordinary_light", "ordinary_blur", "ordinary_time_step")
CODE = (
    "backend/scripts/video_appearance_plan_v107.py",
    "backend/tests/test_video_appearance_plan_v107.py",
    "docs/video-appearance-plan-v107-method.md",
    "backend/scripts/review_video_native_inserts_v96.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def windows(frames, step, ai_frames):
    require(type(frames) is int and frames >= 3, "Observed positive frame extent required")
    step = Fraction(step)
    require(step > 0, "Positive rational source-frame step required")
    require(
        all(type(i) is int and 0 <= i < frames for i in ai_frames),
        "AI source frames exceed constructed extent",
    )
    times = [i * step for i in range(frames)]
    duration = frames * step
    centers = sorted(
        {
            min(range(frames), key=lambda i: (abs(times[i] - Fraction(k, 2)), i))
            for k in range(math.ceil(duration * 2))
            if Fraction(k, 2) < duration
        }
    )
    triplets = [i for i in centers if 0 < i < frames - 1 and 2 * step <= Fraction(13, 100)]
    result = []
    for start in range(2, len(triplets) - 4):
        before, middle, after = (
            triplets[start - 2 : start],
            triplets[start : start + 3],
            triplets[start + 3 : start + 5],
        )
        support = sorted({i + k for i in middle for k in (-1, 0, 1)})
        context = sorted({i + k for i in before + after for k in (-1, 0, 1)})
        result.append(
            {
                "id": f"face-appearance.{middle[0]}.{middle[-1]}",
                "before_centers": before,
                "middle_centers": middle,
                "after_centers": after,
                "location": {
                    "kind": "time",
                    "unit": "nanoseconds",
                    "start": round(support[0] * step * 10**9),
                    "end": round((support[-1] + 1) * step * 10**9),
                    "extent": round(duration * 10**9),
                    "frames": {"start": support[0], "end": support[-1] + 1, "extent": frames},
                    "sampled_frames": support,
                    "context_frames": context,
                },
                "sampled_ai_frame_presence": int(bool(set(support) & set(ai_frames))),
                "context_contains_ai_frame": bool(set(context) & set(ai_frames)),
            }
        )
    return {
        "centers": centers,
        "units": result,
        "model_frame_indices": sorted(
            {i for u in result for key in ("sampled_frames", "context_frames") for i in u["location"][key]}
        ),
    }


def case_plan(index, group, condition):
    require(condition in CONDITIONS, "One fixed source condition required")
    construction = group["construction"]
    n = construction["frames"]
    step = Fraction(construction["frame_step_seconds"])
    a, b = construction["edit_frames"]
    require(0 < a < b < n, "Known bounded source edit required")
    original = group["sources"][0]["rgb_sha256"]
    slot = {"swap_a": 1, "swap_b": 2}.get(condition)
    changed = (
        []
        if slot is None
        else [i for i in range(a, b) if group["sources"][slot]["rgb_sha256"][i] != original[i]]
    )
    support = windows(n, step, changed)
    observed = {i for u in support["units"] for i in u["location"]["sampled_frames"]}
    return {
        "group_index": index,
        "source_group": group["group"],
        "source_role": group["role"],
        "condition": condition,
        "publisher_manipulation_label": int(slot is not None),
        "expected_clock": {
            "status": "construction_recipe_expected_not_newly_observed",
            "frame_count": n,
            "time_base": str(Fraction(1, step.denominator)),
            "pts": [i * step.numerator for i in range(n)],
            "duration_seconds": str(n * step),
            "width": group["sources"][0]["frame_table"]["width"],
            "height": group["sources"][0]["frame_table"]["height"],
        },
        "known_insert_frames": None if condition == "original" else [a, b],
        "known_insert_seconds": None if condition == "original" else [str(a * step), str(b * step)],
        "intended_changed_ai_frames": changed,
        "planned_scope_label": int(any(u["sampled_ai_frame_presence"] for u in support["units"])),
        "support_status": "planned" if support["units"] else "unsupported",
        "support_unavailable_reason": None
        if support["units"]
        else "Insufficient observed cadence/context for fixed appearance windows; no originality verdict.",
        "unobserved_changed_ai_frames": len(set(changed) - observed),
        **support,
        "scope": "Fixed single-visible-face source-frame presence. RGB change is not AI-pixel or identity truth; no whole-video originality claim.",
    }


def identity():
    old = read(PANEL / "video-face-specialists-v20-freeze.json")
    models = {}
    for name in ("yunet.2023mar", "opencv.sface.2021dec"):
        model = old["models"][name]
        require(model["licence"]["class_"] == "commercial", "Commercial weight record required")
        require(all(sha(ROOT / p) == h for p, h in model["files"].items()), "Pinned local model file changed")
        models[name] = model
    return {
        "schema": "chipmunk.video-appearance-plan-freeze/107",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "constructor_freeze_sha256": sha(SOURCE / "freeze.json"),
        "constructor_plan_sha256": sha(SOURCE / "plan.json"),
        "candidate_fitted_models_sha256": sha(CALIBRATION),
        "candidate_result_sha256": sha(PANEL / "video-joint-scalar-result-v106.json"),
        "fixed_local_heads_sha256": sha(PANEL / "video-face-specialists-v20-all-artifact.json"),
        "model_records": models,
        "evaluated_role": "selection",
        "planned_groups": 23,
        "planned_cases": 138,
        "new_neural_calls": 0,
        "scope": "Source-only support preparation for fixed v106 appearance-map transfer. Existing source roles and 24 confirmation groups stay untouched; not native scoring or a full production pipeline.",
    }


def freeze():
    TRIAL.mkdir(exist_ok=False)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def build(lock, output, observe):
    source = read(SOURCE / "plan.json")
    frozen = read(SOURCE / "freeze.json")
    require(
        sha(SOURCE / "plan.json") == frozen["private_plan_sha256"]
        and source["identity"] == frozen["identity"],
        "Unchanged native construction plan required",
    )
    candidate = read(PANEL / "video-joint-scalar-result-v106.json")
    require(candidate["fitted_model_sha256"] == sha(CALIBRATION), "Sealed candidate models differ")
    groups = [(i, g) for i, g in enumerate(source["groups"]) if g["role"] == "selection"]
    require(len(groups) == 23, "All selection groups required")
    cases = []
    for i, g in groups:
        cases.extend(case_plan(i, g, c) for c in CONDITIONS)
        observe()
    plan = {
        "schema": "chipmunk.video-appearance-transfer-plan/107",
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "cases": cases,
        "batch_group_indices": [[i for i, g in groups][k : k + 3] for k in range(0, len(groups), 3)],
        "candidate_point": candidate["new_points"]["appearance_only"],
        "raw_point": 0.11243900750879454,
        "native_scoring_ready": False,
        "new_neural_calls": 0,
    }
    write(output / "plan.json", plan)
    roles = Counter(g["role"] for g in source["groups"])
    result = {
        "schema": "chipmunk.video-appearance-plan-result/107",
        "status": "complete",
        "selection_groups": 23,
        "cases": len(cases),
        "manipulated_cases": sum(c["publisher_manipulation_label"] for c in cases),
        "genuine_cases": sum(not c["publisher_manipulation_label"] for c in cases),
        "planned_windows": sum(len(c["units"]) for c in cases),
        "planned_native_model_frames": sum(len(c["model_frame_indices"]) for c in cases),
        "manipulated_cases_without_positive_planned_unit": sum(
            c["publisher_manipulation_label"] and not c["planned_scope_label"] for c in cases
        ),
        "unchanged_confirmation_groups": roles["confirmation"],
        "unchanged_fitting_groups": roles["fusion_fit"],
        "unchanged_calibration_groups": roles["confidence_cal"],
        "planned_neural_batches": len(plan["batch_group_indices"]),
        "private_plan_sha256": sha(output / "plan.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": lock["scope"]
        + " Constructed-byte/actual-clock binding, guarded native extraction and localization/false-alarm comparison remain pending.",
    }
    write(output / "result.json", result)
    write(PANEL / "video-appearance-plan-result-v107.json", result)
    print(json.dumps(result), flush=True)


def prepare():
    import time

    from scripts.review_video_native_inserts_v96 import Budget

    lock = read(TRIAL / "freeze.json")
    require(lock == identity() == read(PUBLIC), "Published unchanged preparation freeze required")
    output = TRIAL / "preparation"
    output.mkdir(exist_ok=False)
    budget = Budget(output)
    status, error = "failed", None
    try:
        budget.observe(initial=True)
        build(lock, output, budget.observe)
        status = "complete"
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        budget.stream.close()
        write(
            output / "profile.json",
            {
                "status": status,
                "error_type": error,
                "duration_s": time.monotonic() - budget.started,
                "maximum_observed_self_rss_bytes": max(
                    (r["combined_rss_bytes"] for r in budget.rows), default=0
                ),
                "minimum_available_ram_bytes": min(
                    (r["available_ram_bytes"] for r in budget.rows), default=None
                ),
                "resource_journal_sha256": sha(output / "resources.jsonl"),
                "new_neural_calls": 0,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare"))
    freeze() if parser.parse_args().command == "freeze" else prepare()
