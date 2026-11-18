# SPDX-License-Identifier: Apache-2.0
"""Learn lexical changes from matched edits and score single submitted excerpts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

from app.bench.writing_entropy_prominence import compact_range, matched, overlap
from scripts import writing_multi_passage_panel as native
from scripts.verify_writing_entropy_prominence import matching
from scripts.writing_entropy_prominence import existing
from scripts.writing_token_supervision_eval import bind

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
SCHEMA = "chipmunk.writing-matched-lexical/1"
TOKEN = re.compile(r"[^\W\d_]+(?:['’\u2011-][^\W\d_]+)*|\d+(?:[.,]\d+)*|[.,;:!?()–—]", re.UNICODE)
WORD = re.compile(r"[^\W\d_]", re.UNICODE)
ARMS = ("absolute_words", "paired_words", "paired_characters", "within_document", "generator_agreement")
POINTS = {"tighter": .95, "wider": .90}
GENERATORS = ("gpt-5.4", "gpt-5.4-nano")
UNITS = 100_000
MAX_RANGES = 6


def terms(text):
    return ["^", *(("<number>" if token[0].isdigit() else token)
                    for token in TOKEN.findall(text.casefold().replace("’", "'"))), "$"]


def usable(text):
    return 8 <= sum(bool(WORD.search(token)) for token in TOKEN.findall(text)) <= 100


def sentences(case):
    return [case["text"][start:end] for start, end, *_ in case["sentences"]]


def fit_pairs(plan):
    for work in plan["works"]:
        original, edited = work["cases"]
        if (original["kind"], edited["kind"]) != ("ordinary", "mixed"):
            raise ValueError("Unexpected fit case order")
        before, after = sentences(original), sentences(edited)
        alignment = SequenceMatcher(None, list(map(native.compact, before)), list(map(native.compact, after)), autojunk=False)
        for tag, a, b, c, d in alignment.get_opcodes():
            if tag != "replace" or b - a != d - c:
                continue
            for index, former, revised in zip(range(c, d), before[a:b], after[c:d], strict=True):
                source = edited["sentences"][index]
                if usable(former) and usable(revised) and matched([source[2:4]], [label["compact"] for label in edited["truth"]]):
                    yield work["generator"], former, revised


def fit_models(plan):
    import numpy as np
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    groups, before, after = zip(*fit_pairs(plan), strict=True)
    sizes = Counter(groups)
    if set(sizes) != set(GENERATORS) or min(sizes.values()) < 800:
        raise ValueError("Too few matched sentence changes from each generator")
    word_vectorizer = TfidfVectorizer(tokenizer=terms, token_pattern=None, lowercase=False, ngram_range=(1, 3),
                                     min_df=4, max_features=24_000, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=8,
                                     max_features=20_000, sublinear_tf=True)
    x_words = word_vectorizer.fit_transform(before + after)
    x_chars = char_vectorizer.fit_transform(before + after)
    count = len(before)
    target = np.r_[np.zeros(count), np.ones(count)]

    def train(matrix, labels):
        model = LogisticRegression(C=.5, solver="liblinear", fit_intercept=False, random_state=0, max_iter=300)
        model.fit(matrix, labels)
        if model.n_iter_[0] >= 300:
            raise ValueError("Lexical fit did not converge")
        return model

    def pairwise(matrix):
        difference = matrix[count:] - matrix[:count]
        return vstack((-difference, difference), format="csr")

    paired = pairwise(x_words)
    model = {"word_vectorizer": word_vectorizer, "char_vectorizer": char_vectorizer,
             "absolute_words": train(x_words, target),
             "paired_words": train(paired, target), "paired_characters": train(pairwise(x_chars), target)}
    for generator in GENERATORS:
        rows = np.asarray([name == generator for name in groups] * 2)
        model[generator] = train(paired[rows], target[rows])
    return model, dict(sizes)


def readings(case, model):
    focus = [(i, text) for i, text in enumerate(sentences(case)) if usable(text)]
    if not focus:
        return {}
    positions, text = zip(*focus, strict=True)
    word_input = model["word_vectorizer"].transform(text)
    char_input = model["char_vectorizer"].transform(text)
    raw = {}
    for key in (*ARMS[:3], *GENERATORS):
        features = char_input if key == "paired_characters" else word_input
        raw[key] = [float(score) for score in model[key].decision_function(features)]
    center = median(raw["paired_words"])
    raw["within_document"] = [score - center for score in raw["paired_words"]]
    if "agreement_scales" in model:
        left, right = ([((score - midpoint) / spread) for score in raw[name]]
                       for name, (midpoint, spread) in model["agreement_scales"].items())
        raw["generator_agreement"] = [min(a, b) for a, b in zip(left, right, strict=True)]
    return {key: [[i, round(score * UNITS)] for i, score in zip(positions, values, strict=True)]
            for key, values in raw.items()}


def measure(plan, model, comparator=None):
    records = {(work["group"], row["version"]): row for work in comparator["works"]
               for row in work["variants"]} if comparator else {}
    result = []
    for work in plan["works"]:
        for case in work["cases"]:
            if native.digest(case["text"].encode()) != case["native_text_sha256"]:
                raise ValueError("The measured source differs from its recorded text")
            values = readings({"text": case["text"], "sentences": case["sentences"]}, model)
            row = {"kind": case["kind"], "group": work["group"], "genre": work["genre"], "generator": work["generator"],
                   "operation": work["operation"], "native_sha256": case["native_text_sha256"],
                   "extent": case["nonspace_characters"], "sentences": case["sentences"],
                   "truth": [label["compact"] for label in case["truth"]],
                   "readings": {name: values.get(name, []) for name in ARMS}}
            if comparator:
                prior = records.pop((work["group"], case["kind"]))
                ranges, assessed = existing(case, prior)
                row.update(incumbent=[compact_range(case["sentences"], interval) for interval in ranges],
                           incumbent_assessed=assessed)
            result.append(row)
    if records:
        raise ValueError("The current-check readings contain unmatched files")
    return result


def _locate(case, arm, cutoff, *, limit=MAX_RANGES):
    ranked = sorted(case["readings"][arm], key=lambda item: (-item[1], item[0]))
    return [compact_range(case["sentences"], (i, i + 1)) for i, score in ranked[:limit] if score > cutoff]


def _cohorts(row):
    return ("all", row["genre"], row["generator"], f"edit/{row['operation']}")


def _match_size(proposals, labels):
    return len(matched(proposals, labels))


def tally(cases, arm=None, cutoff=None, *, combine=False, match=_match_size):
    totals = defaultdict(Counter)
    for case in cases:
        prior = case.get("incumbent", [])
        proposed = _locate(case, arm, cutoff) if arm else prior
        assessed = bool(case["readings"][arm]) if arm else case.get("incumbent_assessed", False)
        if combine:
            proposed = list({tuple(x) for x in prior + proposed})
            assessed |= case.get("incumbent_assessed", False)
        truth = case["truth"]
        correct = match(proposed, truth)
        for key in _cohorts(case):
            totals[key].update(assessed=assessed, files=1, ranges=len(proposed))
            if case["kind"] == "ordinary":
                totals[key].update(ordinary=1, ordinary_flags=bool(proposed))
            else:
                totals[key].update(edited=1, edits=len(truth), correct=correct, missed=len(truth) - correct,
                                   misplaced=len(proposed) - correct, files_located=bool(correct))
    return dict(totals)


def points(rows):
    operating = {}
    training = {}
    for arm in ARMS:
        peaks = sorted(max(value for _, value in row["readings"][arm]) for row in rows
                       if row["kind"] == "ordinary" and row["readings"][arm])
        if len(peaks) < 80:
            raise ValueError("The ordinary calibration is too sparse")
        for name, fraction in POINTS.items():
            key = f"{arm}/{name}"
            operating[key] = peaks[math.ceil(fraction * len(peaks)) - 1]
            row = tally(rows, arm, operating[key], match=_match_size)["all"]
            training[key] = {"counts": row, "utility": 2 * row["correct"] - 4 * row["ordinary_flags"] - row["misplaced"]}
    preferred = max(training, key=lambda name: (training[name]["utility"], -training[name]["counts"]["ordinary_flags"], name))
    return operating, training, preferred


def _check_partition(works, others=()):
    groups = {work["group"] for work in works}
    digests = {case["native_text_sha256"] for work in works for case in work["cases"]}
    if (len(groups) != len(works) or len(digests) != sum(len(work["cases"]) for work in works)
            or groups.intersection(others)):
        raise ValueError("Matched sources or documents overlap across partitions")
    return sorted(groups)


def fit(fit_plan, calibration, output):
    import joblib
    import sklearn

    pins = native.read(PANELS / "writing-token-supervision-independent-2026-11-17.json")["plans"]
    roles = {"fit": fit_plan, "calibration": calibration}
    if any(bind(path) != pins[role] for role, path in roles.items()):
        raise ValueError("The training and calibration must use the published, separate source rosters")
    training, ordinary = (native.read(path) for path in roles.values())
    fit_groups = _check_partition(training["works"])
    calibration_groups = _check_partition(ordinary["works"], fit_groups)
    model, counts = fit_models(training)
    reference = [readings(work["cases"][0], model) for work in ordinary["works"]]
    scales = {}
    for name in GENERATORS:
        values = [value / UNITS for item in reference for _, value in item.get(name, [])]
        midpoint = median(values)
        scales[name] = (midpoint, max(.0001, 1.4826 * median(abs(value - midpoint) for value in values)))
    model["agreement_scales"] = scales
    rows = measure(ordinary, model)
    point, development, preferred = points(rows)
    output.mkdir(mode=0o700)
    joblib.dump(model, output / "coefficients.joblib", compress=3)
    native.save(output / "freeze.json.gz", {"schema": SCHEMA, "implementation": bind(Path(__file__)),
                "sklearn": sklearn.__version__, "plans": pins, "fit_groups": fit_groups, "calibration_groups": calibration_groups,
                "fit_pairs": counts, "model_sha256": bind(output / "coefficients.joblib"), "agreement_scales": scales,
                "points": point, "development": development, "preferred": preferred, "calibration": rows})
    print(preferred, json.dumps(development[preferred]), flush=True)


def _outcomes(cases, seal, match):
    result = {"incumbent": tally(cases, match=match)}
    for name, cutoff in seal["points"].items():
        arm, _ = name.split("/")
        result[name] = tally(cases, arm, cutoff, match=match)
        result[f"{name}/combined"] = tally(cases, arm, cutoff, combine=True, match=match)
    return result


def _rank_diagnostic(cases):
    ranks = defaultdict(Counter)
    for case in cases:
        if case["kind"] != "mixed":
            continue
        for arm in ARMS:
            regions = _locate(case, arm, -10**9, limit=1)
            total = len(case["readings"][arm])
            if not total:
                continue
            for cohort in _cohorts(case):
                ranks[cohort][arm] += bool(matched(regions, case["truth"]))
                ranks[cohort]["assessed_by_" + arm] += 1
        eligible = [compact_range(case["sentences"], (index, index + 1)) for index, _ in case["readings"]["absolute_words"]]
        ordinary_chance = sum(any(overlap(region, truth) for truth in case["truth"]) for region in eligible) / max(1, len(eligible))
        for cohort in _cohorts(case):
            ranks[cohort]["expected_random_top_one_units"] += round(ordinary_chance * UNITS)
    return dict(ranks)


def held(plan, comparator, frozen, output):
    import joblib

    seal = native.read(frozen / "freeze.json.gz")
    reading = native.read(plan)
    current = native.read(comparator)
    model_file = frozen / "coefficients.joblib"
    if (seal["implementation"] != bind(Path(__file__)) or seal["model_sha256"] != bind(model_file)
            or current["plan_sha256"] != bind(plan)):
        raise ValueError("The fit or current-check comparison differs from the frozen inputs")
    groups = _check_partition(reading["works"], seal["calibration_groups"] + seal["fit_groups"])
    model = joblib.load(model_file)
    cases = measure(reading, model, current)
    results = _outcomes(cases, seal, _match_size)
    native.save(output, {"schema": SCHEMA, "freeze_sha256": bind(frozen / "freeze.json.gz"), "plan_sha256": bind(plan),
                         "comparator_sha256": bind(comparator), "group_ids": groups, "results": results,
                         "top_one": _rank_diagnostic(cases), "cases": cases})
    print(json.dumps({"preferred": seal["preferred"], "current": results["incumbent"]["all"],
                      "combined": results[f"{seal['preferred']}/combined"]["all"]}), flush=True)


def _verify_positions(rows):
    seen = set()
    for row in rows:
        identity = (row["group"], row["kind"])
        if (identity in seen or re.fullmatch(r"[a-f0-9]{64}", row["native_sha256"]) is None
                or row["kind"] not in ("ordinary", "mixed") or bool(row["truth"]) != (row["kind"] == "mixed")):
            raise ValueError("A delivery has an invalid source identifier or edit label")
        seen.add(identity)
        native_end = compact_end = 0
        for start, stop, a, b, _, _ in row["sentences"]:
            if not (native_end <= start < stop and compact_end <= a < b <= row["extent"] and b - a <= stop - start):
                raise ValueError("A sentence falls outside the source order")
            native_end, compact_end = stop, b
        if any(not 0 <= a < b <= row["extent"] for a, b in row["truth"] + row.get("incumbent", [])):
            raise ValueError("An edit label or current-check range falls outside the document")
        if set(row["readings"]) != set(ARMS):
            raise ValueError("A frozen measurement arm is missing")
        for values in row["readings"].values():
            previous = -1
            for index, score in values:
                if type(index) is not int or type(score) is not int or not previous < index < len(row["sentences"]):
                    raise ValueError("A reading falls outside its sentence positions")
                previous = index


def review(freeze, result):
    seal, outcome = native.read(freeze), native.read(result)
    if outcome["freeze_sha256"] != bind(freeze) or {seal["schema"], outcome["schema"]} != {SCHEMA}:
        raise ValueError("The result does not match the published freeze")
    _verify_positions(seal["calibration"])
    _verify_positions(outcome["cases"])
    points_, counts, choice = points(seal["calibration"])
    if (points_, counts, choice) != (seal["points"], seal["development"], seal["preferred"]):
        raise ValueError("The selected settings do not follow the calibration readings")
    cases = outcome["cases"]
    if (sorted({case["group"] for case in cases}) != outcome["group_ids"]
            or set(outcome["group_ids"]).intersection(seal["calibration_groups"] + seal["fit_groups"])):
        raise ValueError("A confirmation source overlaps development")
    if _outcomes(cases, seal, matching) != outcome["results"] or _rank_diagnostic(cases) != outcome["top_one"]:
        raise ValueError("Published locations, ordinary alerts or rank counts could not be reproduced")
    print(json.dumps({"preferred": choice, "results": outcome["results"], "top_one": outcome["top_one"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, fields in (("fit", ("fit_plan", "calibration", "output")), ("held", ("plan", "comparator", "frozen", "output")),
                         ("review", ("freeze", "result"))):
        action = commands.add_parser(name)
        for field in fields:
            action.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    globals()[args.pop("action")](**args)
