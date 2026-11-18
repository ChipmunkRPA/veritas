# SPDX-License-Identifier: Apache-2.0
"""Compare unigram, sequence, and clipped sequence scores for short rewrites."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from app.bench.writing_entropy_prominence import compact_range, matched
from scripts.verify_writing_entropy_prominence import matching
from scripts.writing_matched_lexical import GENERATORS, fit_pairs, terms, usable
from scripts.writing_transition_deviation import COMPARATORS, PINS, file_hash, read, require, save, text

SCHEMA = "chipmunk.writing-clipped-markov/1"
ARMS = ("unigram", "markov", "clipped_markov")
SETTINGS = {"backoff_mass": 8.0, "unigram_mass": 0.5, "clip": 0.5, "min_frequency": 8,
            "quantile": 0.95, "max_ranges": 6, "units": 100_000}


class LanguageModel:
    def __init__(self, tokens, vocabulary):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.histories = Counter()
        self.history_pairs = Counter()
        self.vocabulary = vocabulary
        for sequence in tokens:
            context = ["^", "^", *self.encode(sequence)]
            for first, previous, token in zip(context, context[1:], context[2:]):
                self.unigrams[token] += 1
                self.bigrams[previous, token] += 1
                self.trigrams[first, previous, token] += 1
                self.histories[previous] += 1
                self.history_pairs[first, previous] += 1
        self.unigram_denominator = self.unigrams.total() + SETTINGS["unigram_mass"] * len(vocabulary)

    def encode(self, sequence):
        return [token if token in self.vocabulary else "<unknown>" for token in sequence]

    def probabilities(self, first, previous, token):
        mass = SETTINGS["backoff_mass"]
        unigram = (self.unigrams[token] + SETTINGS["unigram_mass"]) / self.unigram_denominator
        bigram = (self.bigrams[previous, token] + mass * unigram) / (self.histories[previous] + mass)
        trigram = (self.trigrams[first, previous, token] + mass * bigram) / (self.history_pairs[first, previous] + mass)
        return unigram, trigram


def train(path):
    require(path, PINS["fit"])
    plan = read(path)
    names, original, revised = zip(*fit_pairs(plan), strict=True)
    counts = Counter(names)
    if set(counts) != set(GENERATORS) or min(counts.values()) < 800:
        raise ValueError("Too few mapped training revisions per generator")
    sources, rewrites = ([terms(sentence)[1:] for sentence in passages] for passages in (original, revised))
    tokens = Counter(token for sequence in sources + rewrites for token in sequence)
    vocab = {token for token, count in tokens.items() if count >= SETTINGS["min_frequency"]} | {"<unknown>"}
    return (LanguageModel(sources, vocab), LanguageModel(rewrites, vocab)), sorted(
        row["group"] for row in plan["works"]), dict(counts)


def scores(sentence, models):
    ordinary, edited = models
    context = ["^", "^", *ordinary.encode(terms(sentence)[1:])]
    terms_by_arm = [[] for _ in ARMS]
    for first, previous, token in zip(context, context[1:], context[2:]):
        human_unigram, human_markov = ordinary.probabilities(first, previous, token)
        edit_unigram, edit_markov = edited.probabilities(first, previous, token)
        ratio = math.log(edit_markov / human_markov)
        clip = SETTINGS["clip"]
        terms_by_arm[0].append(math.log(edit_unigram / human_unigram))
        terms_by_arm[1].append(ratio)
        terms_by_arm[2].append(min(clip, max(-clip, ratio)))
    return [round(SETTINGS["units"] * math.fsum(items) / len(items)) for items in terms_by_arm]


def measure(works, models, prior=()):
    reference = {(row["group"], row["native_sha256"]): row for row in prior}
    for work in works:
        for case in work["cases"]:
            if not prior and case["kind"] != "ordinary":
                continue
            row = {"group": work["group"], "native_sha256": case["native_text_sha256"], "kind": case["kind"],
                   "genre": work["genre"], "generator": work["generator"], "sentences": case["sentences"],
                   "extent": case["nonspace_characters"], "truth": [item["compact"] for item in case["truth"]],
                   "scores": [[i, *scores(sentence, models)] for i, sentence in enumerate(text(case)) if usable(sentence)]}
            if prior:
                incumbent = reference.pop((row["group"], row["native_sha256"]))
                if any(row[key] != incumbent[key] for key in ("kind", "sentences", "extent", "truth")):
                    raise ValueError("Incumbent scored a different source position")
                row.update(incumbent=incumbent["incumbent"], incumbent_assessed=incumbent["incumbent_assessed"])
            yield row
    if reference:
        raise ValueError("The comparison omitted retained incumbent readings")


def thresholds(maxima):
    if len(maxima) < 64 or any(len(row) != len(ARMS) or any(type(value) is not int for value in row) for row in maxima):
        raise ValueError("At least 64 complete ordinary-file readings are required for calibration")
    rank = math.ceil(len(maxima) * SETTINGS["quantile"]) - 1
    return dict(zip(ARMS, (sorted(values)[rank] for values in zip(*maxima, strict=True)), strict=True))


def develop(fit, calibration, output):
    models, fit_groups, pairs = train(fit)
    require(calibration, PINS["calibration"])
    works = read(calibration)["works"]
    calibration_groups = sorted(row["group"] for row in works)
    if set(fit_groups) & set(calibration_groups):
        raise ValueError("Fit and calibration source groups overlap")
    maxima = [list(map(max, zip(*(scores[1:] for scores in row["scores"]), strict=True)))
              for row in measure(works, models) if row["scores"]]
    save(output, {"schema": SCHEMA, "code_sha256": file_hash(Path(__file__)), "settings": SETTINGS,
                  "inputs": PINS, "comparators": COMPARATORS, "fit_groups": fit_groups,
                  "calibration_groups": calibration_groups, "pairs": pairs,
                  "vocabulary": len(models[0].vocabulary), "maxima": maxima, "thresholds": thresholds(maxima)})


def tally(rows, operating, match):
    totals = {arm: Counter() for arm in ("incumbent", *ARMS, "incumbent_plus_clipped")}
    for row in rows:
        baseline = {tuple(value) for value in row["incumbent"]}
        proposals = {"incumbent": baseline}
        for offset, arm in enumerate(ARMS, 1):
            ranked = sorted(row["scores"], key=lambda values: (-values[offset], values[0]))[:SETTINGS["max_ranges"]]
            proposals[arm] = {tuple(compact_range(row["sentences"], (values[0], values[0] + 1)))
                              for values in ranked if values[offset] > operating[arm]}
        proposals["incumbent_plus_clipped"] = baseline | proposals["clipped_markov"]
        for arm, ranges in proposals.items():
            total = totals[arm]
            if row["kind"] == "ordinary":
                total.update(ordinary=1, ordinary_alerts=bool(ranges))
            else:
                hits = match(sorted(ranges), row["truth"])
                total.update(rewrites=1, known=len(row["truth"]), located=hits, missed=len(row["truth"]) - hits,
                             misplaced=len(ranges) - hits)
    return {arm: dict(counts) for arm, counts in totals.items()}


def evaluate(fit, freeze, inputs, output):
    frozen = read(freeze)
    if (frozen["schema"] != SCHEMA or frozen["settings"] != SETTINGS or frozen["inputs"] != PINS
            or frozen["comparators"] != COMPARATORS or frozen["thresholds"] != thresholds(frozen["maxima"])
            or frozen["code_sha256"] != file_hash(Path(__file__))):
        raise ValueError("Calibration differs from the frozen settings")
    models, groups, pairs = train(fit)
    if groups != frozen["fit_groups"] or pairs != frozen["pairs"]:
        raise ValueError("Fitted groups differ from the frozen fit")
    used = set(groups + frozen["calibration_groups"])
    panels = {}
    for name, (plan, comparator) in zip(COMPARATORS, zip(inputs[::2], inputs[1::2], strict=True), strict=True):
        require(plan, PINS[name])
        require(comparator, COMPARATORS[name])
        works, saved = read(plan)["works"], read(comparator)
        if used & {row["group"] for row in works}:
            raise ValueError("Evaluation includes a calibration or fit source group")
        rows = list(measure(works, models, saved.get("cases", saved.get("readings"))))
        totals = tally(rows, frozen["thresholds"], lambda proposals, truth: len(matched(proposals, truth)))
        panels[name] = {"cases": rows, "plan_sha256": PINS[name], "comparator_sha256": COMPARATORS[name],
                        "results": totals}
    save(output, {"frozen": frozen, "panels": panels})


def review(receipt):
    record = read(receipt)
    frozen = record["frozen"]
    operating = thresholds(frozen["maxima"])
    expected_metadata = (SCHEMA, SETTINGS, PINS, COMPARATORS, operating)
    if tuple(frozen[key] for key in ("schema", "settings", "inputs", "comparators", "thresholds")) != expected_metadata:
        raise ValueError("Recorded thresholds or fixture pins differ")
    if set(record["panels"]) != COMPARATORS.keys():
        raise ValueError("Evaluation panels differ from the frozen selection")
    used = set(frozen["fit_groups"] + frozen["calibration_groups"])
    for name, panel in record["panels"].items():
        rows = panel["cases"]
        if (panel["plan_sha256"] != PINS[name] or panel["comparator_sha256"] != COMPARATORS[name]
                or {row["group"] for row in rows} & used
                or len({(row["group"], row["native_sha256"]) for row in rows}) != len(rows)):
            raise ValueError("Evaluation contains an unpinned input or a reused source")
        for row in rows:
            if row["kind"] not in ("ordinary", "mixed") or bool(row["truth"]) != (row["kind"] == "mixed"):
                raise ValueError("A document has an unexpected label")
            native_end = compact_end = 0
            for start, stop, first, last, *_ in row["sentences"]:
                if not (native_end <= start < stop and compact_end <= first < last <= row["extent"]
                        and last - first <= stop - start):
                    raise ValueError("A sentence falls outside the source order")
                native_end, compact_end = stop, last
            if any(not 0 <= first < last <= row["extent"] for first, last in row["truth"] + row["incumbent"]):
                raise ValueError("A region falls outside the source document")
            previous = -1
            for index, *values in row["scores"]:
                if (len(values) != len(ARMS) or any(type(value) is not int for value in values)
                        or type(index) is not int or not previous < index < len(row["sentences"])):
                    raise ValueError("A reading has invalid sentence coordinates or scores")
                previous = index
        if tally(rows, operating, matching) != panel["results"]:
            raise ValueError("Independent location counting differs")
        print(name, panel["results"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration_cmd = commands.add_parser("develop")
    for name in ("fit", "calibration", "output"):
        calibration_cmd.add_argument(name, type=Path)
    evaluation_cmd = commands.add_parser("evaluate")
    for name in ("fit", "freeze", "output"):
        evaluation_cmd.add_argument(name, type=Path)
    evaluation_cmd.add_argument("inputs", nargs=6, type=Path,
                                metavar="PLAN_OR_COMPARATOR", help="Three plan and incumbent-receipt pairs in held, GAO, economic order")
    commands.add_parser("review").add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "develop":
        develop(args.fit, args.calibration, args.output)
    elif args.command == "evaluate":
        evaluate(args.fit, args.freeze, args.inputs, args.output)
    else:
        review(args.receipt)
