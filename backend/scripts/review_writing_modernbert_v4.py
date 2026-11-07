"""Separate arithmetic reconstruction of all four writing development arms.

Reuses frozen source/journal validation and NumPy's declared bootstrap RNG;
rebuilds ranks, character counts, rates, shifts, paired means and decisions.
This local numerical check is not an outside reviewer or new accuracy test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts import run_writing_modernbert_v4 as run
from scripts.writing_modernbert_v4 import digest


def ranks(values):
    result = [0.] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def rank_auc(truth, scores):
    positives = sum(truth)
    negatives = len(truth) - positives
    if not positives or not negatives:
        raise ValueError("Both retained-origin classes required for rank AUC")
    mass = math.fsum(rank for rank, label in zip(ranks(scores), truth, strict=True) if label)
    return (mass - positives * (positives + 1) / 2) / (positives * negatives)


def rank_correlation(first, second):
    a, b = ranks(first), ranks(second)
    avg_a, avg_b = math.fsum(a) / len(a), math.fsum(b) / len(b)
    x, y = [v - avg_a for v in a], [v - avg_b for v in b]
    return math.fsum(u * v for u, v in zip(x, y, strict=True)) / math.sqrt(
        math.fsum(v * v for v in x) * math.fsum(v * v for v in y))


def one_document(source, values, point):
    foci = source["foci"]
    flagged = [f for f, value in zip(foci, values, strict=True) if value >= point]
    return {
        "bot_characters": source["total_ai_characters"], "human_characters": source["total_human_characters"],
        "located_bot_characters": sum(f["ai_characters"] for f in flagged),
        "highlighted_human_characters": sum(f["human_characters"] for f in flagged),
        "pure_bot_foci": sum(f["human_characters"] == 0 for f in foci),
        "recovered_pure_bot_foci": sum(f["human_characters"] == 0 for f in flagged),
        "pure_human_foci": sum(f["ai_characters"] == 0 for f in foci),
        "highlighted_pure_human_foci": sum(f["ai_characters"] == 0 for f in flagged),
        "highlighted_mixed_foci": sum(f["human_characters"] > 0 and f["ai_characters"] > 0 for f in flagged),
    }


def fractions(c):
    def divide(a, b):
        return a / b if b else None

    tp, fp = c["located_bot_characters"], c["highlighted_human_characters"]
    return {
        "bot_character_recall": divide(tp, c["bot_characters"]),
        "human_origin_character_highlight_rate": divide(fp, c["human_characters"]),
        "bot_character_precision": divide(tp, tp + fp),
        "character_iou": divide(tp, c["bot_characters"] + fp),
        "pure_bot_recovery_rate": divide(c["recovered_pure_bot_foci"], c["pure_bot_foci"]),
        "pure_human_origin_highlight_rate": divide(c["highlighted_pure_human_foci"], c["pure_human_foci"]),
    }


def same(actual, expected, path=""):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Arithmetic fields differ: " + path)
        for key in expected:
            same(actual[key], expected[key], path + "/" + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("Arithmetic list differs: " + path)
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            same(a, e, path + "/" + str(index))
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, abs_tol=5e-12, rel_tol=0):
            raise ValueError("Arithmetic response differs: " + path)
    elif actual != expected:
        raise ValueError("Arithmetic count/decision differs: " + path)


def review(lock_path, baseline, candidate, baseline_native, candidate_native, summary, output):
    import numpy as np

    lock, sources = run.check(lock_path)
    reported = json.loads(summary.read_text())
    journals = {name: run.read_responses(lock_path, name, path, sources)
                for name, path in (("baseline", baseline), ("candidate", candidate))}
    if (reported["lock_sha256"] != digest(lock_path) or reported["baseline_response_sha256"] != digest(baseline)
            or reported["candidate_response_sha256"] != digest(candidate)
            or reported["serving_adoption_permitted"] is not False or reported["fresh_confirmation"] is not False
            or reported["fitting_performed"] is not False or reported["outside_reviewers"] != 0):
        raise ValueError("Summary must retain source/execution/interpretation boundaries")
    foci = [f for s in sources for f in s["foci"]]
    pure = [i for i, f in enumerate(foci) if f["human_characters"] == 0 or f["ai_characters"] == 0]
    truth = [foci[i]["human_characters"] == 0 for i in pure]
    origin_fraction = [f["ai_characters"] / (f["ai_characters"] + f["human_characters"]) for f in foci]
    arms, author_iou, shifts = {}, {}, {}
    for name, rows in journals.items():
        for view in ("context", "focus"):
            scores = [f[view]["score"] for row in rows for f in row["foci"]]
            arm = arms[name + "/" + view] = {
                "pure_origin_rank_auc": rank_auc(truth, [scores[i] for i in pure]),
                "focus_provenance_spearman": rank_correlation(origin_fraction, scores),
                "coverage": {"inputs": 674, **{key: sum(f[view]["coverage"][key] for row in rows for f in row["foci"])
                                                for key in ("tokens", "token_windows")}},
            }
            for point in (0.5, 0.8):
                counts = [one_document(s, [f[view]["score"] for f in r["foci"]], point)
                          for s, r in zip(sources, rows, strict=True)]
                total = {key: sum(c[key] for c in counts) for key in counts[0]}
                ious = [fractions(c)["character_iou"] for c in counts]
                author_iou[name, view, str(point)] = ious
                arm[str(point)] = {"counts": total, "rates": fractions(total),
                                   "author_mean_iou": math.fsum(ious) / 40}
        pairs = [(f["context"]["score"], f["focus"]["score"]) for row in rows for f in row["foci"]]
        shifts[name] = {"mean_focus_minus_context": math.fsum(f - c for c, f in pairs) / 674,
                        "mean_absolute_shift": math.fsum(abs(f - c) for c, f in pairs) / 674}
        for point in (0.5, 0.8):
            shifts[name][str(point)] = {"context_only_highlights": sum(c >= point and f < point for c, f in pairs),
                                        "focus_only_highlights": sum(f >= point and c < point for c, f in pairs)}
    draws = np.random.default_rng(20261106).integers(0, 40, size=(2000, 40))
    paired, gates = {}, {"complete_no_failures": True}
    for point in ("0.5", "0.8"):
        differences = [c - b for c, b in zip(author_iou["candidate", "focus", point],
                                            author_iou["baseline", "context", point], strict=True)]
        means = [math.fsum(differences[i] for i in row) / 40 for row in draws]
        paired[point] = {"author_mean_iou_gain": math.fsum(differences) / 40,
                         "percentile95": np.quantile(means, [0.025, 0.975]).tolist()}
        b, c = arms["baseline/context"][point], arms["candidate/focus"][point]
        gates[point + "_more_pure_bot_recoveries"] = c["counts"]["recovered_pure_bot_foci"] > b["counts"]["recovered_pure_bot_foci"]
        gates[point + "_human_origin_highlights"] = c["rates"]["pure_human_origin_highlight_rate"] - b["rates"]["pure_human_origin_highlight_rate"] <= 0.02 + 1e-15
        gates[point + "_bot_character_retention"] = c["rates"]["bot_character_recall"] >= b["rates"]["bot_character_recall"]
        gates[point + "_author_iou_gain"] = paired[point]["percentile95"][0] > 0
    for key, rebuilt in (("arms", arms), ("focus_context_shifts", shifts), ("primary_author_paired_comparison", paired),
                         ("development_gates", gates), ("qualifies_for_further_development_validation", all(gates.values()))):
        same(reported[key], rebuilt, key)
    native_counts = {}
    for name, path in (("baseline", baseline_native), ("candidate", candidate_native)):
        audit = json.loads(path.read_text())
        expected = {(s["id"], f["sentence"], view) for s, row in zip(sources, journals[name], strict=True)
                    for f, r in zip(s["foci"], row["foci"], strict=True) for view in ("context", "focus")
                    if f["sentence"] == 0 or r[view]["coverage"]["token_windows"] > 1}
        observed = {(r["id"], r["sentence"], r["view"]) for r in audit["audits"]}
        journal_path = baseline if name == "baseline" else candidate
        if (audit["lock_sha256"] != digest(lock_path) or audit["response_sha256"] != digest(journal_path)
                or audit["model"] != name or observed != expected or len(audit["audits"]) != len(expected)
                or audit["passed"] is not True or any(r["passed"] is not True or r["coverage_exact"] is not True
                    or not 0 <= r["score_difference"] <= 1e-5 for r in audit["audits"])):
            raise ValueError("Complete predeclared native correspondence audit required")
        native_counts[name] = len(expected)
    historical = {(r["id"], r["sentence"]): r for r in json.loads(Path(lock["historical"]).read_text())["records"]}
    errors = [abs(f["context"]["score"] - historical[r["id"], f["sentence"]]["score"])
              for r in journals["baseline"] for f in r["foci"]]
    same(reported["historical_baseline_context_replay"]["max_absolute_response_difference"], max(errors))
    same(reported["historical_baseline_context_replay"]["differences_above1e_minus5"], sum(e > 1e-5 for e in errors))
    result = {"schema": "chipmunk.writing-modernbert-numerical-review/4", "passed": True,
              "lock_sha256": digest(lock_path), "summary_sha256": digest(summary),
              "review_code_sha256": digest(Path(__file__)), "documents": 40, "foci": 674, "arms": 4,
              "all_counts_ranks_rates_shifts_intervals_and_decisions_reproduced": True,
              "native_probes": native_counts, "native_sha256": {"baseline": digest(baseline_native), "candidate": digest(candidate_native)},
              "outside_reviewers": 0, "serving_changed": False}
    run.write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "baseline", "candidate", "baseline_native", "candidate_native", "summary", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    review(a.lock, a.baseline, a.candidate, a.baseline_native, a.candidate_native, a.summary, a.output)
