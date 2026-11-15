# SPDX-License-Identifier: MIT
"""Test source-position-preserving sentence ablation on matched essay rewrites."""

from __future__ import annotations

import argparse
import json
import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_independent_range_transfer as independent
from scripts import writing_width_calibration_transfer as width

pool, words, intake = width.pool, width.words, width.intake
SCHEMA = "chipmunk.writing-attention-attribution/1"
PARENT = width.PANELS / "writing-width-calibration-2026-11-15.json.gz"
FAMILIES = dict(zip(intake.FAMILIES, intake.PINS["essays"], strict=True)) | dict(reversed(list(width.ESSAY_PINS.items())))
CENTERS = independent.CENTERS
LEVELS = independent.previous.LEVELS
PER_POSITION = 6
ARMS = ("isolated", "ablation", "both")
METHOD = {
    "scope": "Three exact interior sentences in a five-sentence essay excerpt. Compare the existing isolated "
             "logit with full-excerpt minus hidden-center logit. Retain the original token IDs and positions and "
             "change only the attention mask on the completely covered center. No prior draft enters inference.",
    "combination": "Rank each measurement against ordinary essays at the same position. The joint arm uses the "
                   "smaller of both empirical ranks. Select at most one sentence, withhold ties, and calibrate "
                   "each arm using leave-one-work-out document maxima at the existing two ordinary budgets.",
    "holdout": "Reuse a frozen panel for comparison, then test separate works chosen "
               "only by excerpt geometry. No source work or normalized 12-word sequence may recur from the "
               "prior essay plans, relevant exposure receipts, or other admitted works. Scan the four pinned families "
               "in fixed order, with a cap of six works per family and position. Record shortfalls without relaxing the "
               "matching criteria. Use the same frozen calibration points for both groups.",
    "word_evidence": "On the retained panel use the prior width-adjusted word proposal only inside the chosen "
                     "sentence. On the separate holdout evaluate exact sentence overlap and changed-word "
                     "coverage of the full selected sentence, separately from any refined word focus.",
    "measurement": "Count correct and wrong sentence picks, matched ordinary collisions at the same sentence, "
                   "paired gains and losses, unassessed inputs, and useful word overlap. No authorship probability.",
}
require, read, save = width.require, width.read, width.save


def prior_exposure(parent):
    known = {row["group"] for row in parent["essays"]["rows"]}
    for name in width.EXPOSURE:
        path = width.PANELS / name
        raw = path.read_text()
        known.update(re.findall(r'"([0-9a-f]{64})"', raw))
    return known


def fresh_cases(paths, parent):
    tables = {family: intake.shard(paths[family], "essays", family, sha) for family, sha in FAMILIES.items()}
    known = prior_exposure(parent)
    used = set()
    for table in tables.values():
        for key, versions in table.items():
            if pool.sha(key) in known and "v0" in versions:
                used.update(intake.fingerprints(versions["v0"]["text"]))
    groups, output = Counter(), []
    for family, table in tables.items():
        for key in sorted(table, key=lambda name: pool.sha(f"{SCHEMA}/{family}/{name}")):
            group = pool.sha(key)
            if group in known or "v0" not in table[key]:
                continue
            phrases = intake.fingerprints(table[key]["v0"]["text"])
            if phrases & used:
                continue
            try:
                options = intake.excerpt_options(key, table[key])
            except (KeyError, ValueError):
                continue
            positions = [index for index in options if groups[family, index] < PER_POSITION]
            if not positions:
                continue
            focus = min(positions, key=lambda index: (groups[family, index], index))
            pair = [{"role": "held", "group": group, "family": label, "position": focus, "version": version,
                     "sha256": pool.sha(text), "spans": spans, "text": text}
                    for label, (text, spans, version) in zip(("original", family), options[focus], strict=True)]
            for row in pair:
                row["words"] = [list(item[:3]) for item in words.items(row)]
                row["changed"] = sorted(words.edit_words(pair[0], row)) if row["version"] else []
            if not pair[1]["changed"]:
                continue
            output.extend(pair)
            used.update(phrases)
            known.add(group)
            groups[family, focus] += 1
            if all(groups[family, index] == PER_POSITION for index in CENTERS):
                break
        print(f"Source-only {family} admissions by position: {[groups[family, i] for i in CENTERS]}", flush=True)
    check_admissions(groups)
    return output


def check_admissions(groups):
    require(all(groups["gemini-2.5-flash", i] == PER_POSITION for i in CENTERS)
            and sum(groups.values()) >= 3 * PER_POSITION + 2
            and all(family in FAMILIES and index in CENTERS and 0 < total <= PER_POSITION
                    for (family, index), total in groups.items() if total),
            "The source admissions differ from the declared strata or coverage")


def prepare(args):
    parent = read(PARENT)
    prior = width.assay_text(args.flash, args.nano)["rows"]
    require([public(row) for row in prior] == parent["essays"]["rows"], "Prior local essay inputs do not replay")
    fresh = fresh_cases(dict(zip(FAMILIES, (args.qwen, args.gpt, args.nano, args.flash), strict=True)), parent)
    return {"schema": SCHEMA, "parent": pool.file_hash(PARENT), "method": METHOD,
            "sources": FAMILIES, "retained": prior, "held": fresh}


def public(row):
    return {key: value for key, value in row.items() if key != "text"}


def token_selection(text, offsets, special, span):
    start, stop = span
    selected = []
    coverage = set()
    for index, ((a, b), marker) in enumerate(zip(offsets, special, strict=True)):
        if marker or a == b or a >= stop or b <= start:
            continue
        if any(not text[pos].isspace() for pos in range(a, min(start, b))) or any(
            not text[pos].isspace() for pos in range(max(a, stop), b)
        ):
            return []
        coverage.update(range(max(a, start), min(b, stop)))
        selected.append(index)
    return selected if all(i in coverage or text[i].isspace() for i in range(start, stop)) else []


def capture(plan, assets):
    import numpy as np

    model, tokenizer, runtime = independent.load_classifier(assets)
    require(runtime == "1.24.3", "The classifier runtime differs")
    require(plan["schema"] == SCHEMA and plan["method"] == METHOD and plan["parent"] == pool.file_hash(PARENT)
            and plan["sources"] == FAMILIES, "The frozen source plan differs")

    def score(ids, masks):
        inputs = np.repeat(np.array([ids], dtype=np.int64), len(masks), axis=0)
        [logits] = model.run(["logits"], {"input_ids": inputs, "attention_mask": np.array(masks, dtype=np.int64)})
        return [round(float(result[1]) - float(result[0]), 7) for result in logits]

    def measure(row):
        text, spans = row["text"], row["spans"]
        require(pool.sha(text) == row["sha256"], "The native excerpt changed")
        encoded = tokenizer.encode(text)
        record = {"tokens": len(encoded.ids), "removed": [0, 0, 0], "isolated": [], "ablation": []}
        for index in CENTERS:
            a, b = spans[index]
            sentence = tokenizer.encode(text[a:b]).ids
            require(2 < len(sentence) <= 512, "The isolated sentence is outside the previous input extent")
            record["isolated"].extend(score(sentence, [[1] * len(sentence)]))
        if not 2 < len(encoded.ids) <= 512:
            return record
        selected = [token_selection(text, encoded.offsets, encoded.special_tokens_mask, spans[i]) for i in CENTERS]
        if not all(selected):
            return record
        full = [1] * len(encoded.ids)
        masks = [full.copy() for _ in selected]
        for mask, indices in zip(masks, selected, strict=True):
            for index in indices:
                mask[index] = 0
        original, *hidden = score(encoded.ids, [full, *masks])
        return record | {"removed": list(map(len, selected)), "ablation": [round(original - x, 7) for x in hidden]}

    controls = [row for row in plan["retained"] if row["role"] == "calibration"]
    held = [row for row in plan["retained"] if row["role"] == "held"] + plan["held"]
    readings = {row["sha256"]: measure(row) for row in controls}
    _, points = calibrate(controls, readings)
    print(f"Froze two operating points per comparison on {len(controls)} originals before held inference.", flush=True)
    for index, row in enumerate(held, 1):
        readings[row["sha256"]] = measure(row)
        if index % 36 == 0:
            print(f"Measured {index}/{len(held)} held excerpts.", flush=True)
    result = {"schema": SCHEMA, "parent": plan["parent"], "method": METHOD, "sources": plan["sources"],
              "model": {"revision": independent.RELEASE, "assets": independent.ASSETS, "runtime": runtime},
              "rows": list(map(public, plan["held"])), "readings": readings, "points": points}
    return result | {"results": evaluate(result)}


def winner(row, readings, refs, *, exclude=None):
    sample = readings[row["sha256"]]
    ranks = {}
    for signal in ("isolated", "ablation"):
        choices = {}
        for index, value in zip(CENTERS, sample[signal]):
            values = sorted(score for group, score in refs[signal, index].items() if group != exclude)
            choices[index] = 10_000 * bisect_left(values, value) // (len(values) + 1)
        ranks[signal] = choices
    ranks["both"] = {i: min(ranks["isolated"][i], score) for i, score in ranks["ablation"].items()}
    output = {}
    for name, choices in ranks.items():
        top = sorted(((value, index) for index, value in choices.items()), reverse=True)
        output[name] = top[0] if len(top) > 1 and top[0][0] > top[1][0] else (0, None)
    return output


def calibrate(rows, readings):
    refs = defaultdict(dict)
    for row in rows:
        require(row["role"] == "calibration" and row["family"] == "original", "Calibration requires untouched source works")
        for signal in ("isolated", "ablation"):
            for index, value in zip(CENTERS, readings[row["sha256"]][signal]):
                refs[signal, index][row["group"]] = value
    require(len(refs) == 6 and all(len(table) >= 24 for table in refs.values()), "Insufficient ordinary calibration evidence")
    picks = [winner(row, readings, refs, exclude=row["group"]) for row in rows]
    points = {}
    for arm in ARMS:
        scores = sorted((entry[arm][0] for entry in picks), reverse=True)
        points[arm] = {name: scores[len(scores) // divisor] for name, divisor in LEVELS.items()}
    return refs, points


def evaluate_panel(rows, readings, refs, points, proposals=None):
    picked = {row["sha256"]: winner(row, readings, refs) for row in rows}
    originals = {row["group"]: picked[row["sha256"]] for row in rows if row["family"] == "original"}
    original_strata = {row["group"]: row["family"] for row in rows if row["family"] != "original"}
    output = defaultdict(Counter)
    for row in rows:
        group, family, sha = row["group"], row["family"], row["sha256"]
        ordinary = family == "original"
        stratum = f"original/{original_strata[group]}" if ordinary else family
        options = picked[sha]
        changed = set(row["changed"])
        for arm, (rank, index) in options.items():
            for level, cutoff in points[arm].items():
                flagged = index is not None and rank > cutoff
                hit = flagged and not ordinary and index == row["position"]
                original_rank, original_index = originals[group][arm]
                base_rank, base_index = options["isolated"]
                base_flag = base_index is not None and base_rank > points["isolated"][level]
                base_hit = base_flag and not ordinary and base_index == row["position"]
                span = ({i for i, item in enumerate(row["words"]) if item[2] == index}
                        if proposals is None else proposals[sha].get(index, set()))
                output[f"{arm}/{level}/{stratum}"].update(
                    files=1, assessed=bool(readings[sha]["isolated" if arm == "isolated" else "ablation"]),
                    flagged=flagged, correct_sentence=hit,
                    wrong_sentence=flagged and not ordinary and not hit,
                    correct_without_matched_collision=hit and not (original_rank > cutoff and original_index == index),
                    useful_word_overlap=flagged and not ordinary and words.useful_words(span, changed),
                    gained=hit and not base_hit, lost=base_hit and not hit,
                    ordinary_added=ordinary and flagged and not base_flag, ordinary_cleared=ordinary and base_flag and not flagged,
                )
    return output


def matched_movement(rows, readings):
    """Diagnose score movement with the known pairing, outside the selection rule."""
    originals = {row["group"]: readings[row["sha256"]] for row in rows if row["family"] == "original"}
    result = defaultdict(Counter)
    for row in rows:
        if row["family"] == "original":
            continue
        old, new = originals[row["group"]], readings[row["sha256"]]
        position = CENTERS.index(row["position"])
        for arm in ("isolated", "ablation"):
            if not old[arm] or not new[arm]:
                continue
            changes = [round(b - a, 7) for a, b in zip(old[arm], new[arm], strict=True)]
            elsewhere = max(abs(change) for i, change in enumerate(changes) if i != position)
            result[f"{arm}/{row['family']}"].update(
                complete_pairs=1,
                target_increased=changes[position] > 0,
                unchanged_sentence_moved=elsewhere > 0,
                target_increase_stronger_than_neighbor_movement=changes[position] > elsewhere,
            )
    return result


def evaluate(data):
    parent = read(PARENT)
    require(data["schema"] == SCHEMA and data["parent"] == pool.file_hash(PARENT) and data["method"] == METHOD
            and data["sources"] == FAMILIES
            and data["model"] == {"revision": independent.RELEASE, "assets": independent.ASSETS, "runtime": "1.24.3"},
            "The model, source stencils, or analysis rule differ")
    controls, retained = width.essay_rows(parent["essays"])
    fresh, readings = data["rows"], data["readings"]
    require(len({row["sha256"] for row in controls + retained + fresh}) == len(controls + retained + fresh)
            and set(readings) == {row["sha256"] for row in controls + retained + fresh}, "Unexpected source hashes")
    by_group = defaultdict(list)
    known = prior_exposure(parent)
    for row in fresh:
        spans = row["spans"]
        require("text" not in row and row["role"] == "held" and row["group"] not in known
                and row["position"] in CENTERS and len(spans) == 5
                and all(0 <= a < b for a, b in spans) and all(x[1] <= y[0] for x, y in zip(spans, spans[1:]))
                and all(0 <= i < len(spans) and spans[i][0] <= a < b <= spans[i][1] for a, b, i in row["words"])
                and all(x[1] <= y[0] for x, y in zip(row["words"], row["words"][1:]))
                and row["changed"] == sorted(set(row["changed"]))
                and all(0 <= i < len(row["words"]) and row["words"][i][2] == row["position"] for i in row["changed"]),
                "Invalid held source coordinates or exposure")
        by_group[row["group"]].append(row)
    for rows in by_group.values():
        require(len(rows) == 2 and {row["position"] for row in rows} == {rows[0]["position"]}
                and {row["family"] for row in rows} & {"original"} and len({row["family"] for row in rows} & FAMILIES.keys()) == 1
                and all(bool(row["changed"]) == (row["family"] != "original") for row in rows), "Invalid matched pair geometry")
    check_admissions(Counter((row["family"], row["position"]) for row in fresh if row["family"] != "original"))
    for item in readings.values():
        require(len(item["isolated"]) == len(item["removed"]) == 3 and len(item["ablation"]) in {0, 3}
                and all(math.isfinite(v) for v in (*item["isolated"], *item["ablation"]))
                and all(type(n) is int and n >= 0 for n in item["removed"])
                and sum(item["removed"]) <= item["tokens"] and bool(item["ablation"]) == all(item["removed"])
                and (not item["ablation"] or 2 < item["tokens"] <= 512), "Invalid model reading coverage")
    for sha, original in parent["essay_sentence_readings"]["readings"].items():
        require(readings[sha]["isolated"] == original, "The published isolated scores did not reproduce")
    refs, points = calibrate(controls, readings)
    require(points == data["points"] and points["isolated"] == parent["report"]["sentence_then_words"]["points"]["essays"],
            "The original or comparison points changed")
    scored = {row["sha256"]: width.range_buckets(row, parent["essays"]["readings"][row["sha256"]]) for row in controls + retained}
    profile = width.fit(controls, scored)
    proposals = {}
    for row in retained:
        sha = row["sha256"]
        proposals[sha] = {index: set(range(candidate[2], candidate[3]))
                          for index in CENTERS
                          if (candidate := width.candidates(scored[sha], profile, focus=index).get("width_absolute"))}
    return {"retained": evaluate_panel(retained, readings, refs, points, proposals),
            "separate": evaluate_panel(fresh, readings, refs, points)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("output", type=Path)
    for name in ("flash", "nano", "qwen", "gpt"):
        freeze.add_argument(name, type=Path)
    run = commands.add_parser("capture")
    for name in ("output", "plan", "assets"):
        run.add_argument(name, type=Path)
    review = commands.add_parser("review")
    review.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        save(args.output, prepare(args))
    elif args.mode == "capture":
        save(args.output, capture(read(args.plan), args.assets))
    else:
        data = read(args.receipt)
        results = evaluate(data)
        require(data["results"] == results, "The published match counts did not reproduce")
        _, retained = width.essay_rows(read(PARENT)["essays"])
        movement = {"retained": matched_movement(retained, data["readings"]),
                    "separate": matched_movement(data["rows"], data["readings"])}
        print(json.dumps({"points": data["points"], "results": results, "matched_score_movement": movement},
                         sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
