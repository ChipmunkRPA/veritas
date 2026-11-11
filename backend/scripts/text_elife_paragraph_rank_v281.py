"""Rank the frozen target paragraph on archived current passage responses."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import mean

from scripts import text_elife_passage_score_v275 as previous

RESULT = previous.PANEL / "text-elife-passage-score-complete-v275.json"
RESULT_SHA = "4058cca94716819cca2302b80cc29f36d37e28c9b2f259b24414f4f91c32efea"
FREEZE_SHA = "6c11cb1bf13ceccf6ad7c6851eacee58d4992c8475701a25bf855145745eb141"
RULE = (
    "At each non-whitespace source position take the maximum raw current response covering it. "
    "Rank the frozen target's exact mean against all paragraph means only with full source coverage. "
    "Equal means tie, so the target wins only when its mean is strictly highest."
)
require, read, sha = previous.require, previous.read, previous.sha


def protocol():
    require(sha(RESULT) == RESULT_SHA and sha(previous.PUBLIC) == FREEZE_SHA, "Exact sealed v275 required")
    return {
        "schema": "chipmunk.text-elife-paragraph-rank-rule/281",
        "v275_result_sha256": RESULT_SHA,
        "v275_freeze_sha256": FREEZE_SHA,
        "code_sha256": sha(Path(__file__)),
        "rule": RULE,
        "exploratory_saved_cohort": True,
    }


def paragraph_edges(original, case):
    target = original["edit_target_codepoints"]
    paragraphs = original["paragraph_codepoints"]
    index = paragraphs.index(target)
    if case["ai_spans"]:
        left, right = case["ai_spans"][0]
        require(left == target[0], "The replacement must begin at the original target start")
        shift = right - target[1]
        paragraphs = [
            *paragraphs[:index],
            [left, right],
            *([a + shift, b + shift] for a, b in paragraphs[index + 1 :]),
        ]
    text = case["text"]
    require(
        paragraphs[0][0] == 0
        and paragraphs[-1][1] == len(text)
        and all(0 <= a < b <= len(text) for a, b in paragraphs)
        and all(text[b:c] == "\n\n" for (_, b), (c, _) in zip(paragraphs, paragraphs[1:]))
        and [text[a:b] for a, b in paragraphs] == text.split("\n\n"),
        "Every mapped paragraph must reconstruct the exact article",
    )
    return paragraphs, index


def current_positions(case, response):
    if response["status"] == "unassessed":
        require(response["reason"] == "extraction_correspondence", "Unknown v275 unassessed reason")
        return {}
    require(response["status"] == "measured", "Expected the saved native current response")
    proposals, proof = response["primary_current"], response["native_proofs"][0]
    if proposals is None:
        require(proof is None, "Missing proposals must have no proof")
        return {}
    view, _ = previous.prior.views(case["text"])
    require(
        proof is not None
        and proof["foci"] == len(proposals)
        and all(
            type(row["start"]) is type(row["end"]) is int
            and 0 <= row["start"] < row["end"] <= len(view["text"])
            for row in proposals
        ),
        "Exact bounded current proposals and proof required",
    )
    return previous.position_scores(view, proposals)


def measure(text, paragraphs, index, scores):
    positions = [[i for i in range(a, b) if not text[i].isspace()] for a, b in paragraphs]
    known = {i for span in positions for i in span}
    require(all(positions) and scores.keys() <= known, "Mapped current support must be inside paragraph text")
    coverage = {
        "known_positions": len(known),
        "covered_positions": len(scores),
        "target_known_positions": len(positions[index]),
        "target_covered_positions": sum(i in scores for i in positions[index]),
    }
    if len(scores) != len(known):
        return {"status": "partial" if scores else "unassessed", **coverage}
    exact = {value: Fraction(value) for value in set(scores.values())}
    means = [mean(exact[scores[i]] for i in span) for span in positions]
    target = means[index]
    rank = 1 + sum(value > target for value in means)
    tied_peers = sum(value == target for value in means) - 1
    return {
        "status": "full",
        **coverage,
        "paragraphs": len(paragraphs),
        "target_paragraph": index + 1,
        "target_rank": rank,
        "tied_peers": tied_peers,
        "unique_highest": rank == 1 and tied_peers == 0,
        "target_raw_mean": float(target),
        "target_minus_best_other": float(target - max(means[:index] + means[index + 1 :])),
    }


def compare(original, mixed):
    if original["status"] != "full" or mixed["status"] != "full":
        return None
    return {
        "rank_gain": original["target_rank"] - mixed["target_rank"],
        "target_raw_mean_gain": mixed["target_raw_mean"] - original["target_raw_mean"],
        "target_vs_best_other_gain": mixed["target_minus_best_other"] - original["target_minus_best_other"],
        "new_unique_highest": mixed["unique_highest"] and not original["unique_highest"],
    }


def assess(rows, responses, originals):
    by_group, measured = {row["group"]: row for row in originals}, {}
    coverage = Counter()
    controls, pairs = [], []
    for case, response in zip(rows, responses, strict=True):
        require(case["case"] == response["case"], "Exact v275 response order required")
        group, family = case["article_group"], case["generator"]
        if case["status"] == "unavailable":
            require(
                response == {"case": case["case"], "status": "preparation_unavailable"},
                "Unavailable rewrite cannot have predictions",
            )
            item = {"status": "preparation_unavailable"}
        else:
            paragraphs, index = paragraph_edges(by_group[group], case)
            item = measure(case["text"], paragraphs, index, current_positions(case, response))
        coverage[item["status"]] += 1
        if family is None:
            measured[group] = item
            controls.append({"source": group, **item})
        else:
            original = measured[group]
            pairs.append(
                {
                    "source": group,
                    "family": family,
                    "original": original,
                    "mixed": item,
                    "paired_change": compare(original, item),
                }
            )
    return {
        "coverage": dict(coverage),
        "controls": controls,
        "pairs": pairs,
        "by_family": {
            family: summarize([row for row in pairs if row["family"] == family])
            for family in ("qwen", "smol2")
        },
    }


def summarize(rows):
    paired = [row for row in rows if row["paired_change"] is not None]
    return {
        "planned_rewrites": len(rows),
        "paired_ranks": len(paired),
        "preparation_unavailable": sum(row["mixed"]["status"] == "preparation_unavailable" for row in rows),
        "mixed_target_unique_highest": sum(row["mixed"]["unique_highest"] for row in paired),
        "same_original_target_unique_highest": sum(row["original"]["unique_highest"] for row in paired),
        "mixed_target_top_ties": sum(
            row["mixed"]["target_rank"] == 1 and row["mixed"]["tied_peers"] > 0 for row in paired
        ),
        "same_original_target_top_ties": sum(
            row["original"]["target_rank"] == 1 and row["original"]["tied_peers"] > 0 for row in paired
        ),
        "rank_gains": [row["paired_change"]["rank_gain"] for row in paired],
        "new_unique_highest": sum(row["paired_change"]["new_unique_highest"] for row in paired),
    }


def readout(frozen_rule):
    require(frozen_rule == protocol(), "The exploratory rule must match the prior preflight")
    rows = previous.verify()
    responses, profiles, calls = previous.scoped(
        previous.opai.observations,
        {
            "TRIAL": previous.TRIAL,
            "PUBLIC": previous.PUBLIC,
            "WIDTH": previous.WIDTH,
            "__file__": previous.__file__,
        },
    )(rows)
    prior = read(RESULT)
    require(
        profiles == prior["resource_profile_sha256"] and calls == prior["actual_forwards"],
        "Exact v275 resource and model receipts required",
    )
    return {
        "schema": "chipmunk.text-elife-paragraph-rank-result/281",
        "protocol": frozen_rule,
        **assess(rows, responses, previous.generation.inputs()),
        "new_neural_calls": 0,
        "serving_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    modes.add_parser("preflight")
    modes.add_parser("readout").add_argument("prior_rule", type=Path)
    args = parser.parse_args()
    value = protocol() if args.command == "preflight" else readout(read(args.prior_rule))
    print(json.dumps(value, indent=2, allow_nan=False))
