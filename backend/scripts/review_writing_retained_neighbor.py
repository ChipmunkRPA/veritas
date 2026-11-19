# SPDX-License-Identifier: Apache-2.0
"""Recount the held writing trial using only anonymous geometry and readings."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.review_writing_neighbor_factorial import INTERIOR, best, interval_match, normalized, require

SCHEMA = "chipmunk.writing-retained-neighbor/1"
CANDIDATES = ("isolated", "retained_neighbors")
METHODS = ("current", *CANDIDATES)
ORDINARY = {"original", "archive_revision"}
LABELS = {"preserved", "changed", "unclear"}


def rank(row, references):
    words, input_scores = row["words"], row["readings_ppm"]
    isolated = [normalized(input_scores["isolated"][i], words[i], references["isolated"]) for i in INTERIOR]
    left = [normalized(input_scores["left"][i - 2], words[i - 1] + words[i], references["context"]) for i in INTERIOR]
    right = [normalized(input_scores["right"][i - 2], words[i] + words[i + 1], references["context"]) for i in INTERIOR]
    return {"isolated": best(isolated), "retained_neighbors": best([min(values) for values in zip(isolated, left, right, strict=True)])}


def generation(packet, held):
    attempts, eligible = set(), set()
    reasons, factual = Counter(), Counter()
    for work, prompt, outcome, label in packet["attempts"]:
        require(work in held and type(prompt) is int and prompt in {0, 1} and (work, prompt) not in attempts,
                "Invalid generator attempt identity")
        attempts.add((work, prompt))
        if outcome == "eligible":
            require(label in LABELS, "The attempt lacks its pre-score factual review")
            factual[label] += 1
            if label == "preserved":
                eligible.add(work)
        else:
            require(label is None, "An excluded attempt cannot have a factual label")
            reasons[outcome] += 1
    require(attempts == {(work, prompt) for work in held for prompt in (0, 1)}, "Incomplete generator attempts")
    return {"attempts": len(attempts), "supported_outputs": factual.total(), "factual": dict(factual),
            "exclusions": dict(reasons), "admitted_work_groups": len(eligible)}, eligible


def read_works(packet, eligible):
    works, seen = {}, set()
    for row in packet["rows"]:
        work, role = row["work"], row["role"]
        require(work not in {None, 0} and (work, role) not in seen, "Repeated or invalid work identity")
        require(all(len(row[key]) == 12 and all(type(n) is int and n > 0 for n in row[key])
                    for key in ("sentence_ids", "words", "nonspace_chars")), "Invalid sentence geometry")
        ranges = row["current_ranges"]
        require(len(ranges) <= 6 and all(type(a) is int and type(b) is int and 0 <= a < b <= 12 for a, b in ranges),
                "Invalid saved passage range")
        readings = row["readings_ppm"]
        require(set(readings) == {"isolated", "left", "right"} and all(
                len(values) == (12 if key == "isolated" else 8)
                and all(type(x) is int and 0 <= x <= 1_000_000 for x in values) for key, values in readings.items()),
                "Invalid model reading")
        works.setdefault(work, {})[role] = row
        seen.add((work, role))
    for group in packet["groups"]:
        work, split, focal = group["id"], group["split"], group["focal"]
        require(split in {"development", "held"} and type(focal) is int and focal in INTERIOR, "Invalid source split or focus")
        files = works[work]
        expected = ORDINARY | ({"generated"} if work in eligible else set())
        require(set(files) == expected, "Incomplete matched ordinary or generated controls")
        original = files["original"]["sentence_ids"]
        for role, row in files.items():
            difference = {i for i, (a, b) in enumerate(zip(original, row["sentence_ids"], strict=True)) if a != b}
            require(difference == (set() if role == "original" else {focal}), "Wrong changed sentence identity")
    return works


def point(ranks, works, development, method, allowed):
    scores = [max((ranks[work, role][method] or (-1e9, None))[0] for role in works[work]) for work in development]
    maxima = sorted(scores, reverse=True)
    return maxima[allowed] if allowed < len(maxima) else -1e9


def recount(packet):
    require(packet["schema"] == SCHEMA, "Unsupported writing trial receipt")
    groups = {group["id"]: group for group in packet["groups"]}
    require(len(groups) == len(packet["groups"]), "Repeated source group")
    held = {work for work, row in groups.items() if row["split"] == "held"}
    development = set(groups) - held
    intake, eligible = generation(packet, held)
    works = read_works(packet, eligible)
    require(set(groups) == set(works) and bool(development), "Missing source group or development controls")
    ranks = {(work, role): rank(row, packet["normalization"]) for work, item in works.items() for role, row in item.items()}
    allowed = min(len(development), 1 + sum(any(row["current_ranges"] for row in works[work].values()) for work in development))
    points = {method: point(ranks, works, development, method, allowed) for method in CANDIDATES}
    totals = {method: Counter() for method in METHODS}
    ordinary = {method: set() for method in METHODS}
    ranked = {method: Counter() for method in CANDIDATES}
    same_correct = same_rank = 0
    for work in sorted(held):
        focal = groups[work]["focal"]
        for role, row in works[work].items():
            picks = ranks[work, role]
            spans = {method: [[pick[1], pick[1] + 1]] if (pick := picks[method]) and pick[0] > points[method] else []
                     for method in CANDIDATES}
            spans["current"] = row["current_ranges"]
            if role in ORDINARY:
                for method, selection in spans.items():
                    if selection:
                        ordinary[method].add(work)
                        totals[method][role + "_alert"] += 1
                continue
            guesses = tuple(picks[method][1] if picks[method] else None for method in CANDIDATES)
            same_rank += guesses[0] is not None and guesses[0] == guesses[1]
            same_correct += all(guess == focal for guess in guesses)
            for method, guess in zip(CANDIDATES, guesses, strict=True):
                ranked[method].update(available=guess is not None, true_target=guess == focal,
                                      wrong_remote=guess is not None and abs(guess - focal) > 1)
            weights = row["nonspace_chars"]
            for method, selection in spans.items():
                exact = any(interval_match(weights, span, focal) for span in selection)
                flags = sum(any(interval_match(weights, span, i) for i in (focal - 1, focal + 1))
                            and not interval_match(weights, span, focal) for span in selection)
                totals[method].update(generated_alert=bool(selection), distinct_sites=exact,
                                      wrong_highlights=len(selection) - int(exact), neighbor_displacements=flags)
    for method, selected in ordinary.items():
        totals[method]["ordinary_groups"] = len(selected)
    return {"development_groups": len(development), "held_groups": len(held), "ordinary_files": 2 * len(held),
            "exact_generated_rewrites": len(eligible), "generation": intake, "points": points,
            "comparison": {method: dict(c) for method, c in totals.items()},
            "ranks_before_point": {method: dict(c) for method, c in ranked.items()},
            "same_top_index": same_rank, "same_correct_site": same_correct}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    packet = json.loads(args.receipt.read_text())
    summary = recount(packet)
    require(summary == packet["summary"], "Summary differs from the anonymous measurements")
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
