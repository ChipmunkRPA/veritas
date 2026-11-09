"""Supplemental joint strata and nuisance-exposure review; no scoring or selection."""
from __future__ import annotations

from collections import Counter

CONDITIONS = ("clean", "noise_0", "noise_5", "noise_15", "music_5", "music_15", "speech_0", "speech_5", "speech_15")
KINDS = ("genuine", "clone", "expressive", "stock")
DECISIONS = {"ai", "original", "inconclusive", "unassessed"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def count_outcomes(rows, choices, current):
    require(len(rows) == len(choices) == len(current), "Complete aligned case decisions required")
    require(all(type(row["label"]) is int and row["label"] in {0, 1} for row in rows)
            and all(value in DECISIONS for value in [*choices, *current]), "Known truths and explicit outcomes required")
    result = {"cases": len(rows), "source_recordings": len({row["base_id"] for row in rows}),
              "connected_foreground_groups": len({row["group"] for row in rows}),
              "genuine_cases": sum(row["label"] == 0 for row in rows),
              "ai_cases": sum(row["label"] == 1 for row in rows),
              "ai_found": 0, "genuine_correct": 0, "false_ai": 0, "false_original": 0,
              "inconclusive": choices.count("inconclusive"), "unassessed": choices.count("unassessed"),
              "new_correct_vs_current": 0, "lost_correct_vs_current": 0}
    for row, decision, baseline in zip(rows, choices, current, strict=True):
        truth = "ai" if row["label"] else "original"
        result["new_correct_vs_current"] += baseline != truth and decision == truth
        result["lost_correct_vs_current"] += baseline == truth and decision != truth
        if decision == "ai":
            result["ai_found" if row["label"] else "false_ai"] += 1
        elif decision == "original":
            result["false_original" if row["label"] else "genuine_correct"] += 1
    require(sum(result[key] for key in ("ai_found", "genuine_correct", "false_ai", "false_original", "inconclusive", "unassessed"))
            == result["cases"], "Every case must retain exactly one outcome")
    result["ai_without_ai_tendency"] = result["ai_cases"] - result["ai_found"]
    result["false_ai_rate"] = result["false_ai"] / result["genuine_cases"] if result["genuine_cases"] else None
    result["missed_ai_rate_including_unassessed"] = result["ai_without_ai_tendency"] / result["ai_cases"] if result["ai_cases"] else None
    return result


def joint_table(rows, predictions):
    require(bool(rows) and "current" in predictions
            and all(len(values) == len(rows) for values in predictions.values()), "All methods on the same complete inputs required")
    require(all(row["condition"] in CONDITIONS and row["kind"] in KINDS
                and (row["kind"] == "genuine") == (row["label"] == 0) for row in rows), "Known source family and condition required")
    result = {}
    for condition in CONDITIONS:
        result[condition] = {}
        for kind in KINDS:
            indices = [index for index, row in enumerate(rows) if row["condition"] == condition and row["kind"] == kind]
            subset = [rows[index] for index in indices]
            baseline = [predictions["current"][index] for index in indices]
            result[condition][kind] = {name: count_outcomes(subset, [values[index] for index in indices], baseline)
                                       for name, values in predictions.items()}
    return {"conditions": result, "scope": "All 36 predetermined condition/family cells, including empty cells. Correlated variants/groups and shared backgrounds; no confidence interval or primary-selection change. Class-only cells do not measure probability calibration."}


def background_exposure(manifest, plan):
    foregrounds = {row["raw_sha256"]: row for row in manifest["foregrounds"]}
    backgrounds = {row["raw_sha256"]: row for row in manifest["backgrounds"]}
    result = {}
    for role in ("fusion_fit", "confidence_cal", "selection", "confirmation"):
        jobs = [row for row in plan if row["role"] == role]
        require(len(jobs) == 116, "Every planned foreground in each source role required")
        result[role] = {}
        for family in (*KINDS, "all_ai"):
            selected = [job for job in jobs if (foregrounds[job["source_sha256"]]["label"] == 1 if family == "all_ai"
                                               else foregrounds[job["source_sha256"]]["kind"] == family)]
            table = {}
            for kind in ("noise", "music", "speech"):
                pool = {key for key, row in backgrounds.items() if row["role"] == role and row["kind"] == kind}
                require(len(pool) == 4, "Four original background sources of every kind per role required")
                counts = Counter(dict.fromkeys(pool, 0))
                for job in selected:
                    require(job["backgrounds"][kind]["raw_sha256"] in pool
                            and foregrounds[job["source_sha256"]]["role"] == role,
                            "Background or foreground crosses assigned role")
                    counts[job["backgrounds"][kind]["raw_sha256"]] += 1
                table[kind] = sorted(counts.values())
            result[role][family] = table
    return {"sorted_foreground_use_counts_per_four_background_sources": result,
            "scope": "Anonymous sorted count histograms only, no source IDs or per-record pairing. Truth-class totals are balanced across backgrounds; individual AI-family exposure may be uneven. Histograms are not independent sample counts or weighted calibration probabilities."}
