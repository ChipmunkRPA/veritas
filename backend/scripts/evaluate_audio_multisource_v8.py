"""Prospective multi-generator gates; failures and abstentions stay in totals."""

from __future__ import annotations

from collections import Counter

from scripts.audio_confirmation_v7 import CONDITIONS, arm_metrics, paired


def summarize(rows, expected_sources, draws=2000):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Unique complete outcome identities required")
    generators = sorted({r["generator"] for r in rows if r["label"] == 1})
    panels = {c: [r for r in rows if c == "all" or r["condition"] == c] for c in ("all", *CONDITIONS)}
    panels["background_speech"] = [r for r in rows if r["condition"].startswith("speech_")]
    panels.update({"generator:" + g: [r for r in rows if r["generator"] == g] for g in generators})
    groups = {
        name: {
            "service": arm_metrics(panel, "original_p", "original_decision"),
            "candidate": arm_metrics(panel, "candidate_p", "candidate_decision"),
        }
        for name, panel in panels.items()
    }
    differences = {
        name: paired(panel, draws=draws)
        for name, panel in panels.items()
        if not name.startswith("generator:")
    }

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
            delta("generator:" + g, "false_original") <= 1 for g in generators
        ),
        "clean_retention": all(delta("clean", k) >= -2 for k in ("correct_ai", "correct_original")),
        "generator_ai_retention": all(delta("generator:" + g, "correct_ai") >= -2 for g in generators),
        "strong_condition_usefulness": all(gain(c) >= 5 for c in ("noise_0", "music_5")),
        "background_speech_usefulness": gain("background_speech") >= 5,
        "background_speech_class_retention": all(
            delta(c, k) >= -2
            for c in ("speech_0", "speech_5", "speech_15")
            for k in ("correct_ai", "correct_original")
        ),
        "same_assessed": {r["id"] for r in rows if r["original_p"] is not None}
        == {r["id"] for r in rows if r["candidate_p"] is not None},
        "complete_cases": len(rows) == expected_sources * len(CONDITIONS)
        and len({r["base_id"] for r in rows}) == expected_sources
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
        "schema": "chipmunk.audio-multisource-comparison/8",
        "groups": groups,
        "paired_differences": differences,
        "strata": strata,
        "gates": gates,
        "advance_to_fresh_confirmation": all(gates.values()),
        "complete_inputs": len(rows),
        "source_files": len({r["base_id"] for r in rows}),
        "connected_foreground_groups": len({r["group"] for r in rows}),
        "deployment_eligible": False,
        "scope": "Development results may permit a separately frozen fresh comparison only. Neither development nor a numerical pass alone enables serving or overall confidence.",
    }
