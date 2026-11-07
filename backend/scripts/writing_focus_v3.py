"""Frozen, consumed-development comparison of writing focus and scored context.

Raw classifier responses are not calibrated authorship probabilities. Research
inputs and per-passage labels stay private; public summaries contain aggregates.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

INPUT_SHA = "fb7dbfa725728926c86c6627e7871ba8fcc9e0faa50bba384b13992c624a9752"
BASELINE_REVISION = "f1795c86806e6838d4afa33d0b1427f8430c9615"
CANDIDATE_REVISION = "0ceddea903015ef99cbaa040a4d8a216aed9c683"
BASELINE_ASSETS = {
    "NOTICE": "4fe6835a2928bcfd7467888a9422d805dfa2fe6364437c1af01cf1945c3edd57",
    "config.json": "deac399da91c0cc96f5570fa5eae7cb716e32f6feb00860ce3e42c84f86c01a3",
    "model.safetensors": "4a1561fadf44ec72934edd6158ff8c76e9388ade1384dbeeea6eb15f93251087",
    "serving_head.json": "098ce573e07d8d99221d9eac41f7acae7b324071490a06023f837efe43df4cab",
    "tokenizer.json": "2c070ff353c36f85e20919da787a30e1e8e52b626d0e026e044f373378f6d0bf",
    "tokenizer_config.json": "d8089020d294f20cb21eca751b5f7f68a0b62ea2716a78576cbd587605382432",
}
CANDIDATE_ASSETS = {
    "README.md": "9a2e93fb2848c5707322588e7685d5abd4f38e08df9b346065d7f4ba390cd687",
    "config.json": "96530fd73c11eae27beb5e13a86297b8c720fc7bc9ea30b31ba51978070dc44d",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": "a74ea30f5db6a1094510be903ce374426191addaf7d860b766077de246d32358",
    "special_tokens_map.json": "f23c8e6099631c233c16d9bf8dab198f610826cdd1b358f270f6d55c1863e857",
    "tokenizer.json": "1f33749d010b4d63908e5c174c341622cb45039dd73a139dcd95bd74cc7e304b",
    "tokenizer_config.json": "288b4077af1ffb3beead6d96fccfc93beb2df9b689cbb038c4eb329165efc43a",
    "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
}
METHOD = {
    "schema": "chipmunk.writing-focus-method/3",
    "design": "2x2: unchanged baseline and candidate, each on three-sentence context and exact focus",
    "documents": 40, "author_groups": 40, "foci": 674,
    "models": ["baseline", "candidate"], "views": ["context", "focus"],
    "device": "cpu", "dtype": "float32", "threads": 4, "batch": 4, "max_len": 512,
    "baseline_response": "1 minus human softmax; includes ai, ai_edited and humanized classes",
    "candidate_response": "softmax class1 AI; no temperature or fitted calibration",
    "coverage": "all native content tokens, stride0 overflow; content-token-weighted mean",
    "native_audit": "first focus of every author, both views and models; plus every input with tokenizer overflow; direct single-window forwards, absolute tolerance1e-5",
    "reference_points": [0.5, 0.8],
    "comparison": "candidate/focus versus baseline/context; all four arms retained",
    "author_bootstrap": {"seed": 20261106, "replicates": 2000, "interval": [0.025, 0.975]},
    "development_continuation": {
        "complete_no_failures": True, "more_pure_bot_recoveries_at_both_points": True,
        "human_origin_highlight_rate_increase_at_most": 0.02,
        "bot_character_recall_not_lower_at_both_points": True,
        "author_mean_iou_gain_lower_interval_above": 0.0,
    },
    "fitting": False, "confirmation": False, "serving_adoption_permitted": False,
    "scope": "Consumed mixed-author development and retained-character provenance, not unassisted-human false alarms, word-origin proof, calibrated confidence or fresh validation",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def finite_score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Finite bounded raw response required")
    return float(value)


def validate_inputs(header, rows):
    if (header["schema"] != "chipmunk.writing-focus-inputs/3" or header["window_sentences"] != 3
            or header["documents"] != 40 or header["foci"] != 674 or header["classifier_forwards"] != 0
            or len(rows) != 40 or len({r["id"] for r in rows}) != 40
            or len({r["author_group"] for r in rows}) != 40):
        raise ValueError("Complete original consumed document/author roles required")
    total = 0
    for row in rows:
        previous = -1
        human = bot = 0
        for index, focus in enumerate(row["foci"]):
            if (focus["sentence"] != index or type(focus["start"]) is not int
                    or type(focus["end"]) is not int or focus["start"] < previous
                    or focus["end"] <= focus["start"]
                    or focus["end"] - focus["start"] != len(focus["focus"])
                    or focus["offset_unit"] != "Unicode code points, end exclusive"
                    or any(text_sha(focus[k]) != focus[k + "_sha256"] for k in ("focus", "context"))):
                raise ValueError("Exact bound and non-overlapping focus/context identities required")
            h, b = focus["human_characters"], focus["ai_characters"]
            if (type(h) is not int or type(b) is not int or min(h, b) < 0 or h + b == 0
                    or h + b != sum(not c.isspace() for c in focus["focus"])
                    or focus["focus_ai_fraction"] != b / (h + b)):
                raise ValueError("Exact non-whitespace retained-character counts required")
            human += h
            bot += b
            previous = focus["end"]
            total += 1
        if (bot + row["uncovered_ai_characters"] != row["total_ai_characters"]
                or human + row["uncovered_human_characters"] != row["total_human_characters"]
                or row["uncovered_ai_characters"] != 0):
            raise ValueError("Complete source character denominators required")
    if total != 674:
        raise ValueError("Every original focus retained")


def counts(row, scores, point):
    if len(scores) != len(row["foci"]):
        raise ValueError("Score positions cannot be omitted")
    result = {key: 0 for key in ("bot_characters", "human_characters", "located_bot_characters",
                                "highlighted_human_characters", "pure_bot_foci", "recovered_pure_bot_foci",
                                "pure_human_foci", "highlighted_pure_human_foci", "highlighted_mixed_foci")}
    result["bot_characters"] = row["total_ai_characters"]
    result["human_characters"] = row["total_human_characters"]
    for focus, value in zip(row["foci"], scores, strict=True):
        flagged = finite_score(value) >= point
        h, b = focus["human_characters"], focus["ai_characters"]
        result["pure_bot_foci"] += h == 0
        result["pure_human_foci"] += b == 0
        if flagged:
            result["located_bot_characters"] += b
            result["highlighted_human_characters"] += h
            result["recovered_pure_bot_foci"] += h == 0
            result["highlighted_pure_human_foci"] += b == 0
            result["highlighted_mixed_foci"] += h > 0 and b > 0
    return result


def rates(c):
    def ratio(numerator, denominator):
        return c[numerator] / c[denominator] if c[denominator] else None

    intersection = c["located_bot_characters"]
    union = c["bot_characters"] + c["highlighted_human_characters"]
    predicted = intersection + c["highlighted_human_characters"]
    return {
        "bot_character_recall": ratio("located_bot_characters", "bot_characters"),
        "human_origin_character_highlight_rate": ratio("highlighted_human_characters", "human_characters"),
        "bot_character_precision": intersection / predicted if predicted else None,
        "character_iou": intersection / union if union else None,
        "pure_bot_recovery_rate": ratio("recovered_pure_bot_foci", "pure_bot_foci"),
        "pure_human_origin_highlight_rate": ratio("highlighted_pure_human_foci", "pure_human_foci"),
    }


def development_gates(baseline, candidate, interval_low, complete):
    gates = {"complete_no_failures": bool(complete)}
    for point in METHOD["reference_points"]:
        key = str(point)
        b, c = baseline[key], candidate[key]
        gates[key + "_more_pure_bot_recoveries"] = c["counts"]["recovered_pure_bot_foci"] > b["counts"]["recovered_pure_bot_foci"]
        gates[key + "_human_origin_highlights"] = c["rates"]["pure_human_origin_highlight_rate"] <= b["rates"]["pure_human_origin_highlight_rate"] + 0.02
        gates[key + "_bot_character_retention"] = c["rates"]["bot_character_recall"] >= b["rates"]["bot_character_recall"]
        gates[key + "_author_iou_gain"] = interval_low[key] > 0
    return gates
