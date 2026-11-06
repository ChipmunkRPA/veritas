"""Frozen prospective fresh-audio metrics and gates; never fitting or serving."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from scripts.review_audio_noise_v4 import metrics, valid, weights

CONDITIONS = (
    "clean",
    "noise_0",
    "noise_5",
    "noise_15",
    "music_5",
    "music_15",
    "speech_0",
    "speech_5",
    "speech_15",
)
HEAD_SHA256 = "fa1a0faec12fd6926cefb8f7ffe9c9736f517ea89b44a2ff13ab17f4d028c408"
PREPARED_SHA256 = "2027c36079c29cf0f5fb3c365b2d3effa50fac4fd0010712b94caefcad563e8a"
SEED = 461109
DRAWS = 2000


def paired(rows, draws=DRAWS):
    selected = [r for r in rows if valid(r.get("original_p")) and valid(r.get("candidate_p"))]
    w = weights(selected) if selected else None
    if w is None:
        return {
            "common_inputs": len(selected),
            "brier_difference": None,
            "log_loss_difference": None,
            "brier_difference_ci95": None,
            "log_loss_difference_ci95": None,
            "bootstrap_defined": 0,
            "bootstrap_requested": draws,
        }
    y = np.asarray([r["label"] for r in selected])
    a = np.asarray([r["original_p"] for r in selected])
    b = np.asarray([r["candidate_p"] for r in selected])
    brier = (b - y) ** 2 - (a - y) ** 2

    def loss(p):
        return -y * np.log(np.clip(p, 1e-12, 1)) - (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))

    logs = loss(b) - loss(a)
    kinds = defaultdict(set)
    for row in selected:
        kinds[row["group"]].add(f"{row['label']}:{row['generator']}")
    strata = defaultdict(list)
    for group, values in sorted(kinds.items()):
        strata["|".join(sorted(values))].append(group)
    rng, resamples = np.random.default_rng(SEED), []
    for _ in range(draws):
        mult = dict.fromkeys(kinds, 0)
        for kind in sorted(strata):
            groups = strata[kind]
            for group in rng.choice(groups, len(groups), replace=True):
                mult[group] += 1
        current = weights(selected, mult)
        if current is not None:
            resamples.append((float(current @ brier), float(current @ logs)))
    values = np.asarray(resamples)
    return {
        "common_inputs": len(selected),
        "brier_difference": float(w @ brier),
        "log_loss_difference": float(w @ logs),
        "brier_difference_ci95": np.quantile(values[:, 0], [0.025, 0.975]).tolist() if len(values) else None,
        "log_loss_difference_ci95": np.quantile(values[:, 1], [0.025, 0.975]).tolist()
        if len(values)
        else None,
        "bootstrap_defined": len(values),
        "bootstrap_requested": draws,
        "strata": {k: len(v) for k, v in sorted(strata.items())},
        "interpretation": "Fresh source comparison conditional on fixed fitting/calibration and selected backgrounds; excludes fitting/calibration and background-population uncertainty. Not outside validation or per-upload accuracy.",
    }


def arm_metrics(rows, key, decision):
    result = metrics(rows, key, decision)
    real, ai, total = result["real_inputs"], result["ai_inputs"], len(rows)
    result.update(
        ai_without_ai_conclusion=ai - result["correct_ai"],
        genuine_without_original_conclusion=real - result["correct_original"],
        false_ai_rate=result["false_ai"] / real if real else None,
        false_original_rate=result["false_original"] / ai if ai else None,
        ai_without_ai_conclusion_rate=(ai - result["correct_ai"]) / ai if ai else None,
        inconclusive_rate=result["decisions"].get("inconclusive", 0) / total if total else None,
    )
    return result


def summarize(rows):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate outcome identity")
    panels = {c: [r for r in rows if c == "all" or r["condition"] == c] for c in ("all", *CONDITIONS)}
    panels["background_speech"] = [r for r in rows if r["condition"].startswith("speech_")]
    for generator in ("ElevenLabs", "Chatterbox"):
        panels["generator:" + generator] = [r for r in rows if r["generator"] == generator]
    groups = {
        name: {
            "service": arm_metrics(panel, "original_p", "original_decision"),
            "candidate": arm_metrics(panel, "candidate_p", "candidate_decision"),
        }
        for name, panel in panels.items()
    }
    differences = {name: paired(panel) for name, panel in panels.items() if not name.startswith("generator:")}

    def delta(name, field):
        return groups[name]["candidate"][field] - groups[name]["service"][field]

    def gain(name):
        return delta(name, "correct_ai") + delta(name, "correct_original")

    overall = differences["all"]
    interval = overall["brier_difference_ci95"]
    gates = {
        "brier_gain": overall["brier_difference"] is not None and overall["brier_difference"] <= -0.005,
        "log_loss_gain": overall["log_loss_difference"] is not None
        and overall["log_loss_difference"] <= -0.01,
        "brier_interval_below_zero": interval is not None and interval[1] < 0,
        "strong_condition_brier": all(
            differences[c]["brier_difference"] is not None and differences[c]["brier_difference"] < 0
            for c in ("noise_0", "music_5", "speech_0", "speech_5", "speech_15")
        ),
        "total_false_calls": all(delta("all", k) <= 1 for k in ("false_ai", "false_original")),
        "condition_false_calls": all(
            delta(c, k) <= 1 for c in CONDITIONS for k in ("false_ai", "false_original")
        ),
        "generator_false_original_calls": all(
            delta("generator:" + g, "false_original") <= 1 for g in ("ElevenLabs", "Chatterbox")
        ),
        "clean_retention": all(delta("clean", k) >= -2 for k in ("correct_ai", "correct_original")),
        "generator_ai_retention": all(
            delta("generator:" + g, "correct_ai") >= -2 for g in ("ElevenLabs", "Chatterbox")
        ),
        "strong_condition_usefulness": all(gain(c) >= 5 for c in ("noise_0", "music_5")),
        "background_speech_usefulness": gain("background_speech") >= 5,
        "background_speech_class_retention": all(
            delta(c, k) >= -2
            for c in ("speech_0", "speech_5", "speech_15")
            for k in ("correct_ai", "correct_original")
        ),
        "same_assessed": {r["id"] for r in rows if r["original_p"] is not None}
        == {r["id"] for r in rows if r["candidate_p"] is not None},
        "complete_cases": len(rows) == 1080
        and len({r["base_id"] for r in rows}) == 120
        and all(
            Counter(r["condition"] for r in rows if r["base_id"] == base) == Counter(CONDITIONS)
            for base in {r["base_id"] for r in rows}
        ),
        "no_extraction_failures": all(r["status"] != "failed" for r in rows),
        "no_probe_failures": all(r["probe"]["status"] != "failed" for r in rows),
    }
    strata = {
        field: {
            str(value): {
                "service": arm_metrics(
                    [r for r in rows if r.get(field) == value], "original_p", "original_decision"
                ),
                "candidate": arm_metrics(
                    [r for r in rows if r.get(field) == value], "candidate_p", "candidate_decision"
                ),
            }
            for value in sorted({r.get(field) for r in rows}, key=str)
        }
        for field in ("generator", "dataset", "background_id", "label")
    }
    return {
        "groups": groups,
        "differences": differences,
        "gates": gates,
        "passes": all(gates.values()),
        "strata": strata,
        "execution_status": dict(Counter(r["status"] for r in rows)),
        "probe_status": dict(Counter(r["probe"]["status"] for r in rows)),
        "deployment_eligible": False,
    }
