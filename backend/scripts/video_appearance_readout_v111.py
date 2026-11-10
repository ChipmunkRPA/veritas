"""Fixed-point readout of native selection responses; no model or fit.

This elaborates v108's predeclared errors/coverage/time-support reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
NATIVE = BASE / "video-appearance-native-v108"
PREP = BASE / "video-appearance-plan-v107/preparation"
TRIAL = BASE / "video-appearance-readout-v111"
PANEL = ROOT / "docs/evaluation-panels"
PUBLIC = PANEL / "video-appearance-readout-freeze-v111.json"
FITTED = BASE / "video-joint-scalars-v106/evaluation/fitted-models.json"
HEADS = PANEL / "video-face-specialists-v20-all-artifact.json"
CODE = (
    "backend/scripts/video_appearance_readout_v111.py",
    "backend/tests/test_video_appearance_readout_v111.py",
    "docs/video-appearance-readout-v111-method.md",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def logit(model, values):
    require(
        model["status"] == "fitted"
        and len(values) == len(model["center"]) == len(model["scale"]) == len(model["coefficients"]),
        "Complete fitted scalar map required",
    )
    value = model["intercept"]
    for raw, center, scale, coefficient in zip(
        values, model["center"], model["scale"], model["coefficients"], strict=True
    ):
        require(
            type(raw) in (float, int)
            and math.isfinite(raw)
            and all(math.isfinite(v) for v in (center, scale, coefficient))
            and scale > 0,
            "Finite measured scalar required",
        )
        value += coefficient * max(-8.0, min(8.0, (raw - center) / scale))
    require(math.isfinite(value), "Finite mapped response required")
    return value


def reference_probability(raws, local, model):
    if not raws or any(raw is None for raw in raws):
        return None
    feature = max(0.0, max(logit(local, [raw]) for raw in raws))
    eta = logit(model["calibration"], [logit(model["response"], [feature])])
    return 1 / (1 + math.exp(-max(-700.0, min(700.0, eta))))


def merge(intervals):
    output = []
    for a, b in sorted(intervals):
        require(a < b, "Positive bounding interval required")
        if output and a <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], b))
        else:
            output.append((a, b))
    return output


def support(units, flags, ai_frames, interval):
    chosen = [u for u, flag in zip(units, flags, strict=True) if flag]
    sparse = {i for u in chosen for i in u["location"]["sampled_frames"]}
    scheduled = {i for u in units for i in u["location"]["sampled_frames"]}
    spans = merge(
        [(Fraction(u["location"]["start"], 10**9), Fraction(u["location"]["end"], 10**9)) for u in chosen]
    )
    seconds = sum((b - a for a, b in spans), Fraction(0))
    overlap = (
        Fraction(0)
        if interval is None
        else sum((max(Fraction(0), min(b, interval[1]) - max(a, interval[0])) for a, b in spans), Fraction(0))
    )
    union = seconds if interval is None else seconds + interval[1] - interval[0] - overlap
    return {
        "flagged_windows": len(chosen),
        "flagged_sample_positions": len(sparse),
        "flagged_ai_sample_positions": len(sparse & ai_frames),
        "false_flagged_sample_positions": len(sparse - ai_frames),
        "scheduled_ai_sample_positions": len(scheduled & ai_frames),
        "ai_source_frames_not_scheduled": len(ai_frames - scheduled),
        "ai_sample_support_found": bool(sparse & ai_frames),
        "bounding_seconds": float(seconds),
        "bounding_false_seconds": float(seconds - overlap),
        "bounding_interval_union_iou": None if interval is None else float(overlap / union) if union else 0.0,
    }


def overlap(first, second):
    x, y, w, h = first
    a, b, c, d = second
    intersection = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
    union = w * h + c * d - intersection
    return intersection / union if union > 0 else 0.0


def review_case(row, bound, local, model, raw_point, candidate_point):
    planned = bound["planned"]
    for key in (
        "group_index",
        "condition",
        "source_group",
        "source_role",
        "publisher_manipulation_label",
        "planned_scope_label",
        "known_insert_frames",
        "known_insert_seconds",
        "intended_changed_ai_frames",
    ):
        require(row[key] == planned[key], "Native response identity or truth changed")
    require(
        row["input_sha256"] == bound["input_sha256"] and len(row["units"]) == len(planned["units"]),
        "Exact complete native response required",
    )
    inputs = row["frame_inputs"]
    require(
        [r["index"] for r in inputs] == planned["model_frame_indices"],
        "All exact scheduled model inputs required",
    )
    frames = {r["index"]: r for r in inputs}
    for r in inputs:
        i = r["index"]
        require(
            r["rgb_sha256"] == bound["expected_rgb_sha256"][str(i)]
            and r["pts"] * Fraction(r["time_base"])
            == bound["table"]["pts"][i] * Fraction(bound["table"]["time_base"]),
            "Native frame pixels/time binding differs",
        )
        require(r["status"] in {"ok", "unassessed"}, "Explicit frame availability required")
        if r["status"] == "ok":
            require(
                r["faces"] == 1
                and len(r["bbox"]) == 4
                and len(r["landmarks"]) == 5
                and len(r["feature_sha256"]) == 64
                and len(r["aligned_crop_sha256"]) == 64,
                "Complete measured face input proof required",
            )
    raws = []
    for unit, expected in zip(row["units"], planned["units"], strict=True):
        require(all(unit[k] == v for k, v in expected.items()), "Planned unit/time/support changed")
        indices = sorted(set(unit["location"]["sampled_frames"] + unit["location"]["context_frames"]))
        eligible = all(frames[i]["status"] == "ok" for i in indices)
        if eligible:
            eligible = all(
                overlap(frames[a]["bbox"], frames[b]["bbox"]) >= 0.1
                for a, b in zip(indices, indices[1:], strict=False)
            )
        require((unit["status"] == "ok") == eligible, "Unexpected assessed/missing fixed geometric support")
        raw = unit["raw"] if unit["status"] == "ok" else None
        require(
            unit["raw"] is None if not eligible else type(raw) in (float, int) and math.isfinite(raw),
            "Unavailable response cannot become zero",
        )
        raws.append(raw)
    raw_flag = None if any(r is None for r in raws) else any(r > raw_point for r in raws)
    p = reference_probability(raws, local, model)
    require(
        row["raw_file_flag"] == raw_flag
        and (
            p is None
            and row["candidate_file_reference_probability"] is None
            or p is not None
            and abs(p - row["candidate_file_reference_probability"]) <= 1e-12
        ),
        "Independent fixed scalar readout differs",
    )
    candidate_flag = None if p is None else p > candidate_point
    require(row["candidate_file_flag"] == candidate_flag, "Strict frozen file point changed")
    interval = (
        None
        if not planned["publisher_manipulation_label"]
        else tuple(Fraction(s) for s in planned["known_insert_seconds"])
    )
    ai = set(planned["intended_changed_ai_frames"])
    raw_local = [r is not None and r > raw_point for r in raws]
    candidate_local = [
        r is not None and reference_probability([r], local, model) > candidate_point for r in raws
    ]
    return {
        "group": planned["source_group"],
        "condition": planned["condition"],
        "label": planned["planned_scope_label"],
        "publisher_label": planned["publisher_manipulation_label"],
        "raw_flag": raw_flag,
        "candidate_flag": candidate_flag,
        "candidate_reference_probability": p,
        "planned_units": len(raws),
        "assessed_units": sum(r is not None for r in raws),
        "face_detection_inputs": len(inputs),
        "appearance_inputs": sum(r["status"] == "ok" for r in inputs),
        "appearance_repeat_inputs": int(any(r["status"] == "ok" for r in inputs)),
        "raw_support": support(row["units"], raw_local, ai, interval),
        "candidate_support": support(row["units"], candidate_local, ai, interval),
    }


def summarize(rows, arm):
    counts = Counter(
        {
            k: 0
            for k in (
                "ai",
                "controls",
                "assessed",
                "inconclusive",
                "ai_found",
                "missed_ai_including_inconclusive",
                "false_ai",
                "inconclusive_ai",
                "inconclusive_controls",
            )
        }
    )
    loss, false_groups = {0: defaultdict(list), 1: defaultdict(list)}, set()
    local = Counter()
    ious = []
    for r in rows:
        label, flag, s = r["label"], r[arm + "_flag"], r[arm + "_support"]
        counts["ai" if label else "controls"] += 1
        counts["inconclusive" if flag is None else "assessed"] += 1
        if flag is None:
            counts["inconclusive_ai" if label else "inconclusive_controls"] += 1
        if label:
            counts["ai_found" if flag is True else "missed_ai_including_inconclusive"] += 1
            ious.append(s["bounding_interval_union_iou"])
            local["ai_clips_with_flagged_ai_sample"] += s["ai_sample_support_found"]
        elif flag is True:
            counts["false_ai"] += 1
            false_groups.add(r["group"])
        if not label:
            local["control_clips_with_local_support"] += bool(s["flagged_windows"])
        for key in (
            "flagged_windows",
            "flagged_sample_positions",
            "flagged_ai_sample_positions",
            "false_flagged_sample_positions",
            "scheduled_ai_sample_positions",
            "ai_source_frames_not_scheduled",
            "bounding_false_seconds",
        ):
            local[key] += s[key]
        if arm == "candidate" and r["candidate_reference_probability"] is not None:
            loss[label][r["group"]].append((r["candidate_reference_probability"] - label) ** 2)
    means = [
        sum(sum(v) / len(v) for v in cells.values()) / len(cells) if cells else None
        for cells in loss.values()
    ]
    return {
        "planned": len(rows),
        **counts,
        "false_ai_groups": len(false_groups),
        "local_support": dict(local),
        "mean_ai_bounding_interval_union_iou": sum(ious) / len(ious) if ious else None,
        "source_class_balanced_reference_brier": sum(means) / 2
        if arm == "candidate" and all(v is not None for v in means)
        else None,
    }


def identity():
    lock = read(NATIVE / "freeze.json")
    require(
        all(sha(ROOT / p) == h for p, h in lock["code_sha256"].items()),
        "Frozen native implementation changed",
    )
    require(
        sha(FITTED) == lock["fitted_models_sha256"] and sha(HEADS) == lock["local_heads_sha256"],
        "Unchanged sealed scalar maps required",
    )
    return {
        "schema": "chipmunk.video-appearance-readout-freeze/111",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "native_freeze_sha256": sha(NATIVE / "freeze.json"),
        "prepared_plan_sha256": sha(PREP / "plan.json"),
        "fitted_models_sha256": sha(FITTED),
        "local_heads_sha256": sha(HEADS),
        "raw_point": lock["raw_point"],
        "candidate_point": lock["candidate_point"],
        "planned_cases": 138,
        "scope": "Implementation of predeclared v108 fixed-point counts and continuous sparse/bounding-support readout; no fitting, point selection or new local probability.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "planned_cases": 138, "new_neural_calls": 0}), flush=True)


def evaluate():
    from scripts.review_video_native_inserts_v96 import Budget

    require(read(TRIAL / "freeze.json") == identity() == read(PUBLIC), "Unchanged published readout required")
    for i in range(8):
        require(
            (NATIVE / f"extract-{i:02d}/output/result.json").is_file()
            and (NATIVE / f"extract-{i:02d}/profile.json").is_file(),
            "All eight distinct native batches must complete",
        )
    output = TRIAL / "evaluation"
    output.mkdir(exist_ok=False)
    budget = Budget(output)
    budget.observe(initial=True)
    lock = read(PUBLIC)
    local = read(HEADS)["specialists"]["face-appearance|single-face"]
    model = read(FITTED)["models"]["appearance_only"]
    reviewed, batches = [], []
    with (output / "reviewed-cases.jsonl").open("x") as journal:
        for i in range(8):
            folder = NATIVE / f"extract-{i:02d}"
            result, profile = read(folder / "output/result.json"), read(folder / "profile.json")
            require(
                result["status"] == "complete"
                and result["batch"] == i
                and profile["child_exit_code"] == 0
                and profile["guard"] is None
                and profile["sample_errors"] == 0,
                "Successful exact native stage required",
            )
            require(
                result["responses_sha256"] == sha(folder / "output/responses.jsonl")
                and profile["worker_receipt_sha256"] == sha(folder / "output/result.json")
                and result["freeze_sha256"] == sha(NATIVE / "freeze.json"),
                "Exact native response commitment required",
            )
            binding = NATIVE / f"bind-{i:02d}/output"
            require(
                result["binding_receipt_sha256"] == sha(binding / "result.json")
                and read(binding / "result.json")["bound_cases_sha256"] == sha(binding / "bound-cases.json"),
                "Source binding commitment changed",
            )
            rows = [json.loads(line) for line in (folder / "output/responses.jsonl").read_text().splitlines()]
            bound = read(binding / "bound-cases.json")
            require(len(rows) == len(bound) == result["cases"], "No missing/substituted native cases")
            values = [
                review_case(r, b, local, model, lock["raw_point"], lock["candidate_point"])
                for r, b in zip(rows, bound, strict=True)
            ]
            for key in ("face_detection_inputs", "appearance_inputs", "appearance_repeat_inputs"):
                require(
                    sum(r[key] for r in values) == result["native_input_counts"][key],
                    "Native call counters differ",
                )
            for r in values:
                journal.write(json.dumps(r, allow_nan=False) + "\n")
            reviewed.extend(values)
            batches.append(
                {
                    "batch": i,
                    "cases": len(rows),
                    "result_sha256": sha(folder / "output/result.json"),
                    "responses_sha256": sha(folder / "output/responses.jsonl"),
                    "profile_sha256": sha(folder / "profile.json"),
                    "neural_calls": result["new_neural_calls"],
                    "duration_s": profile["duration_s"],
                    "sampled_tree_rss_lower_bound_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
                }
            )
            budget.observe()
    require(
        len(reviewed) == 138 and len({r["group"] for r in reviewed}) == 23,
        "All planned selection groups/cases required",
    )
    paired = {}
    for label, name in ((1, "ai"), (0, "controls")):
        rows = [r for r in reviewed if r["label"] == label]
        paired[name] = {
            "added_flags": sum(r["candidate_flag"] is True and r["raw_flag"] is not True for r in rows),
            "removed_flags": sum(r["raw_flag"] is True and r["candidate_flag"] is not True for r in rows),
            "changed_availability": sum(
                (r["raw_flag"] is None) != (r["candidate_flag"] is None) for r in rows
            ),
        }
    result = {
        "schema": "chipmunk.video-appearance-selection-result/111",
        "status": "complete",
        "cases": len(reviewed),
        "source_groups": 23,
        "raw": summarize(reviewed, "raw"),
        "candidate": summarize(reviewed, "candidate"),
        "by_condition": {
            c: {a: summarize([r for r in reviewed if r["condition"] == c], a) for a in ("raw", "candidate")}
            for c in sorted({r["condition"] for r in reviewed})
        },
        "paired": paired,
        "assessed_units": sum(r["assessed_units"] for r in reviewed),
        "planned_units": sum(r["planned_units"] for r in reviewed),
        "raw_point": lock["raw_point"],
        "candidate_point": lock["candidate_point"],
        "private_reviewed_cases_sha256": sha(output / "reviewed-cases.jsonl"),
        "batches": batches,
        "new_neural_calls_in_readout": 0,
        "new_native_neural_calls": sum(r["neural_calls"] for r in batches),
        "defaults_changed": False,
        "scope": "Fixed selection-source development comparison, not reserved confirmation, full production tracking/fallback, lip/audio accuracy or customer probability. Local candidate support is a single-window replay into the file head, not a calibrated local probability. Bounding intervals include unsampled gaps; sparse support and source labels are not AI-pixel or identity proof.",
    }
    budget.stream.close()
    write(output / "result.json", result)
    write(PANEL / "video-appearance-selection-result-v111.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"batches", "by_condition"}}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate"))
    freeze() if parser.parse_args().command == "freeze" else evaluate()
