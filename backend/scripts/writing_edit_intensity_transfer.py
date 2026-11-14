"""Evaluate sentence edit-intensity estimates using retained native source measurements."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from math import fsum, isfinite, log1p, sqrt
from pathlib import Path
from statistics import median

from scripts import writing_revision_trajectory_transfer as prior
from scripts.review_writing_long_position import select as select_long

SCHEMA = "chipmunk.writing-edit-intensity-transfer/1"
PANELS = prior.ROOT / "docs/evaluation-panels"
PUBLIC = {
    "long_development": ("revision-trajectory", "2f905e55222da0b0e024cbaa955feb4fc053d0124f2c3ce4f0c86c4d09c36657"),
    "short_development": ("temperature-transfer", "d95deabfbe775cf79bfaf7235dc1f901153b8c5f8762b2dfb6b9ca6d8f77ddf5"),
    "short_a": ("word-edit-transfer", "86837d924e3c5256f1bd731ffb56fcf0363090df48f3883bce3ac8a9da4367ec"),
    "short_b": ("surprisal-dynamics", "157eea47ab989440608a0d0f91f19a38dab2a1dd795a6024032b4adc33032900"),
    "long": ("interval-transfer", "92fa746a81c2e60a2241080de36eb5fd80d2799bf14882b675d910995257b71e"),
    "f1000": ("f1000-position-transfer", "2de36e115c061e9cf4e353a567a6987ccf116adb00d49a18478c2cf12e5a6b3d"),
    "elife": ("elife-long-position-transfer", "227d5287d17e3f8cf92cbccfedc30c6e134b4e04767be56805d024538bd15e99"),
}
PRIVATE = {
    "long_development": "456ea8b149fee03d13448aa20fa47693ad825f32bcd13c5c48516661f307c174",
    "short_a": "217eb06cc145d060cfdb1242d28219d1a0a9817a3278744bb57bdcc45a739ee9",
    "short_b": "dfb6a52072e32af9acf75b42f84b671850d402a0ba534f414e4d8fa31aa6ad58",
    "long": "417edacc01afbc6248f1065f7803f900f9e01d8fcebbb1cc19eeff5883347429",
    "f1000": "9b07d5ee0afa19051f11063e5e1615a72222eeae4042661df3b653d42145e44c",
    "elife": "597573e182a68e0edf5800a5c470fee23aa5bc79592b514f6a8c87811258a61c",
}
MODEL_FEATURES = ("primary context log odds", "context odds minus file median", "context odds minus median of up to two surrounding scores per side")
STYLE_FEATURES = ("log word count", "log mean word length", "distinct-word share", "connector share", "comma and semicolon share",
                  "other nonterminal punctuation share", "digit share", "content overlap with both neighboring sentences",
                  "word count contrast", "word diversity contrast", "connector contrast", "comma and semicolon contrast")
ARMS = ("binary", "intensity", "intensity_with_style")
METHOD = {
    "fit": "20 previously used GAO report works. Train work-balanced regularized linear heads on unchanged and publisher-edited exact native sentence targets. Byte-identical edited versions contribute no extra observations.",
    "supervision": "Binary and intensity heads use the same three uncalibrated context features. The binary target is any annotated edit in the sentence. The intensity target is the fraction of sentence word tokens both inside an annotation and outside a monotonic exact word alignment with the original. A third arm adds the fixed text-only features. The source original is used for labels only.",
    "proposal": "Select one native sentence of at least 40 nonspace characters or one run of up to six such sentences in the same paragraph. Short files (up to eight sentences) use only single sentences. Longer files use a length-normalized sum with a .015 cost per added sentence. Abstain on a tied top range.",
    "point": "Strictly above the second-highest original-work maximum in each of two length bands. Use 24 consumed short-report originals and 16 consumed long-report originals, disjoint from fit and assessment works.",
    "metric": "Strict localization is at least half of all edited nonspace characters and at least .30 intersection over union in one reported range. Comparators are existing first six native primary contexts at .50, plus the reported long position check for F1000Research and eLife.",
    "advance": "At least six novel report localizations across both generators and both report lengths, plus at least two novel localizations across scholarly works, at most two additional genuine groups in any cohort, and at least 90 percent assessed coverage. All evaluation cases already have project exposure. A passing method needs a fresh panel before serving.",
}
TOKEN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
CONNECTORS = frozenset({"a", "an", "the", "in", "on", "at", "for", "of", "to", "as", "and", "or", "but", "so", "by", "from", "with", "into",
                        "between", "while", "when", "whether", "over", "under", "that", "which", "who", "because", "if", "though", "although",
                        "however", "moreover", "therefore", "meanwhile", "additionally", "consequently", "then"})
COUNT = {"long_development": 156, "short_development": 72, "short_a": 124, "short_b": 60, "long": 140, "f1000": 36, "elife": 45}


def require(test, message):
    prior.require(test, message)


def open_public(name):
    suffix, digest = PUBLIC[name]
    path = PANELS / f"writing-{suffix}-2026-11-13.json"
    require(prior.digest(path) == digest, f"Changed published source ledger: {name}")
    return prior.read(path)


def open_private(path, name):
    require(prior.digest(path) == PRIVATE[name], f"Changed retained native cases: {name}")
    return prior.read(path)


def sentence_features(text, ranges):
    excerpts = [text[a:b] for a, b, _, _ in ranges]
    words = [[m.group().casefold() for m in TOKEN.finditer(t)] for t in excerpts]
    contents = [set(w) - CONNECTORS for w in words]
    styles = []
    for i, (sentence, items) in enumerate(zip(excerpts, words, strict=True)):
        neighbors = set().union(*contents[max(0, i - 1):i], *contents[i + 1:i + 2])
        denominator = max(1, len(items))
        styles.append([log1p(len(items)), log1p(sum(map(len, items)) / denominator), len(set(items)) / denominator,
                       sum(w in CONNECTORS for w in items) / denominator,
                       (sentence.count(",") + sentence.count(";")) / denominator,
                       sum(sentence.count(v) for v in (":", "(", ")", "—", "–", "\"", "“", "”")) / denominator,
                       sum(c.isdigit() for c in sentence) / max(1, len(sentence)),
                       len(contents[i] & neighbors) / max(1, len(contents[i] | neighbors))])
    centers = {k: median(v[k] for v in styles) for k in (0, 2, 3, 4)}
    return [v + [v[k] - center for k, center in centers.items()] for v in styles]


def intensities(original, text, ranges, gold):
    old = [word.group().casefold() for word in TOKEN.finditer(original)]
    new = list(TOKEN.finditer(text))
    values = [word.group().casefold() for word in new]
    matched = {i for block in SequenceMatcher(None, old, values, autojunk=False).get_matching_blocks()
               for i in range(block.b, block.b + block.size)}
    frequencies = []
    for start, stop, *_ in ranges:
        sentence = [(index, word) for index, word in enumerate(new) if start <= word.start() < word.end() <= stop]
        changed = sum(index not in matched and any(word.start() < end and lo < word.end() for lo, end in gold)
                      for index, word in sentence)
        frequencies.append(changed / max(1, len(sentence)))
    return frequencies


def standard(row, text, scores, panel, raw_ranges, gold, baseline=None, intensity=None):
    raw_positions = [(a, b, paragraph) for a, b, paragraph in raw_ranges]
    nonspace = [sum(not c.isspace() for c in text[a:b]) for a, b, _ in raw_positions]
    flagged = [sum(sum(not c.isspace() for c in text[max(a, lo):min(b, hi)]) for lo, hi in gold if a < hi and lo < b)
               for a, b, _ in raw_positions]
    chars = sum(sum(not c.isspace() for c in text[a:b]) for a, b in gold)
    ranges = [[a, b, paragraph, n, overlap] for (a, b, paragraph), n, overlap in zip(raw_positions, nonspace, flagged, strict=True)]
    require(prior.sha(text) == row["text_sha256"] and len(scores) == len(ranges) and chars == sum(flagged)
            and all(isfinite(p) and 0 <= p <= 1 for p in scores), "Source text or original scores do not match")
    odds = [prior.odds(p) for p in scores]
    center = median(odds)
    model = [[value, value - center, value - median(odds[max(0, i - 2):i] + odds[i + 1:i + 3])]
             if len(odds) > 1 else [value, 0., 0.] for i, value in enumerate(odds)]
    style = sentence_features(text, [r[:4] for r in ranges])
    if baseline is None:
        baseline = [[max(0, i - 1), min(len(ranges), i + 2)] for i, p in enumerate(scores) if p >= .50][:6]
    data = {"cohort": panel, "group": row["group"], "family": row.get("family", row.get("variant")),
            "id": row.get("record_id", row.get("id", row.get("variant"))), "text_sha256": row["text_sha256"],
            "characters": len(text), "truth_characters": chars, "ranges": ranges, "baseline": baseline,
            "features": [[round(x, 9) for x in a + b] for a, b in zip(model, style, strict=True)]}
    if intensity is not None:
        data["intensities"] = intensities(intensity, text, ranges, gold)
    return data


def long_rows(data, local, role):
    rows = {case["record_id"]: case for case in local}
    originals = {case["group"]: case["text"] for case in local if case["family"] == "original"}
    for raw in data["rows"]:
        row = rows[raw["record_id"]]
        require(all(raw[key] == row[key] for key in ("group", "family", "text_sha256", "ranges", "gold", "truth_characters")),
                "Different publisher metadata or geometry")
        fit = role == "long_development" and raw["role"] == "held"
        original = originals[raw["group"]]
        if fit and row["family"] != "original" and row["text"] == original:
            continue
        positions = [(v["start"], v["end"], v["paragraph"]) for v in raw["ranges"]]
        yield standard(raw, row["text"], [p for _, p in raw["primary"]], "fit" if fit else "point_long" if role == "long_development" else role,
                       positions, raw["gold"], intensity=original if fit else None)


def paragraphs_from_gaps(text, spans):
    paragraph = previous = 0
    for a, b in spans:
        paragraph += "\n\n" in text[previous:a]
        yield a, b, paragraph
        previous = b


def short_rows(data, local, panel):
    rows = {case["id"]: case for case in local}
    if panel == "point_short":
        by_hash = {case["text_sha256"]: case for case in local if case["cohort"] == "fit" and case["family"] == "original"}
        raw_rows = [r for r in data["freeze"]["roster"] if r["cohort"] == "report" and r["family"] == "original"]
    else:
        raw_rows = [r for r in data["rows"] if panel == "short_a" or r["cohort"] == "report"]
    for raw in raw_rows:
        if panel == "point_short":
            row = by_hash[raw["text_sha256"]]
            scores = [p for _, p in data["measurements"]["primary"]["rows"][raw["id"]]]
        elif panel == "short_a":
            row = rows[raw["id"]]
            scores = raw["primary"]
        else:
            row = rows[raw["id"]]
            scores = data["primary"][raw["id"]]
        require(all(raw[key] == row[key] for key in ("group", "family", "text_sha256", "spans", "truth")), "Changed short source geometry")
        cohort = raw["cohort"] if panel == "short_a" and raw["cohort"] != "report" else panel
        yield standard(raw, row["text"], scores, cohort, list(paragraphs_from_gaps(row["text"], raw["spans"])),
                       [raw["truth"]] if raw["truth"] else [])


def article_rows(data, local, panel):
    rows = {(str(case["group"]), case["variant"]): case for case in local}
    for raw in data["cases"]:
        row = rows[str(raw["group"]), raw["variant"]]
        require(row["truth"] == raw["truth"], "Changed journal edit extent")
        scores = [r["score"] for r in raw["native_current"]]
        baseline = [[max(0, i - 1), min(len(scores), i + 2)] for i, p in enumerate(scores) if p >= .5][:6]
        if (position := select_long(raw, 3.6821095168341915)) is not None:
            baseline.append([position, position + 1])
        yield standard(raw, row["text"], scores, panel, [r[:3] for r in raw["positions"]],
                       [raw["truth"]] if raw["truth"] else [], baseline)


def build_rows(paths, held=False):
    if not held:
        long = open_private(paths["long_development"], "long_development")
        short = open_private(paths["short_a"], "short_a")
        return [*long_rows(open_public("long_development"), long, "long_development"),
                *short_rows(open_public("short_development"), short, "point_short")]
    rows = []
    choices = {"short_a": short_rows, "short_b": short_rows, "long": long_rows, "f1000": article_rows, "elife": article_rows}
    for cohort, build in choices.items():
        cases = list(build(open_public(cohort), open_private(paths[cohort], cohort), cohort))
        require(len(cases) == COUNT[cohort], f"Wrong assessed file count: {cohort}")
        rows.extend(cases)
    return rows


def band(row):
    return "short" if len(row["ranges"]) <= 8 else "long"


def fit(rows, arm):
    grouped = defaultdict(lambda: defaultdict(list))
    size = len(MODEL_FEATURES) if arm != "intensity_with_style" else len(MODEL_FEATURES + STYLE_FEATURES)
    for row in rows:
        for i, (a, b, paragraph, characters, changed) in enumerate(row["ranges"]):
            if characters >= 40:
                target = float(changed > 0) if arm == "binary" else row["intensities"][i]
                grouped[row["group"]][bool(changed)].append((row["features"][i][:size], target))
    require(len(grouped) == 20 and all(set(v) == {False, True} for v in grouped.values()), "Incomplete supervised works")
    examples = [(x, target, 1 / (2 * len(grouped) * len(items))) for pairs in grouped.values()
                for items in pairs.values() for x, target in items]
    centers = [fsum(w * x[i] for x, _, w in examples) for i in range(size)]
    scales = [max(.01, sqrt(fsum(w * (x[i] - u) ** 2 for x, _, w in examples))) for i, u in enumerate(centers)]
    matrix = [([1., *((a - b) / c for a, b, c in zip(x, centers, scales, strict=True))], y, w) for x, y, w in examples]
    coefs = [0.] * (size + 1)
    for _ in range(100):
        res = [(x, w * (fsum(a * b for a, b in zip(x, coefs, strict=True)) - y)) for x, y, w in matrix]
        for i in range(len(coefs)):
            coefs[i] -= .045 * (fsum(x[i] * val for x, val in res) + (.10 * coefs[i] if i else 0.))
    return {"centers": centers, "scales": scales, "coefficients": coefs, "fit_sentences": len(examples)}


def propose(row, head):
    centers, scales, coefficients = (head[key] for key in ("centers", "scales", "coefficients"))
    scores = [coefficients[0] + fsum((a - b) / s * c for a, b, s, c in
              zip(v[:len(centers)], centers, scales, coefficients[1:], strict=True)) for v in row["features"]]
    n = len(scores)
    ranges = row["ranges"]
    background = median(scores)
    candidates = []
    for start, (_, _, paragraph, _, _) in enumerate(ranges):
        for end in range(start + 1, min(n, start + (1 if n <= 8 else 6)) + 1):
            if ranges[end - 1][2] != paragraph or ranges[end - 1][3] < 40:
                break
            length = end - start
            contrast = fsum(v - background for v in scores[start:end]) / sqrt(length)
            response = background + contrast - .015 * (length - 1)
            candidates.append((round(response, 9), start, end))
    candidates.sort(key=lambda v: (-v[0], v[1], v[2]))
    if not candidates or len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0]


def freeze(rows):
    development = [row for row in rows if row["cohort"] == "fit"]
    arms = {}
    require(Counter(r["cohort"] for r in rows) == {"fit": 125, "point_short": 24, "point_long": 16}, "Wrong selection roster")
    for arm in ARMS:
        head = fit(development, arm)
        maxima = {"short": {}, "long": {}}
        for row in rows:
            cohort = row["cohort"]
            if cohort != "fit":
                length = band(row)
                top = propose(row, head)
                require(cohort == f"point_{length}" and row["truth_characters"] == 0 and top is not None, "Unassessed fitting original")
                maxima[length][row["group"]] = top[0]
        arms[arm] = {**head, "maxima": maxima, "points": {k: sorted(v.values())[-2] for k, v in maxima.items()}}
    return {"schema": SCHEMA, "method": METHOD, "features": list(MODEL_FEATURES + STYLE_FEATURES),
            "runner_sha256": prior.digest(Path(__file__)), "public": {k: list(v) for k, v in PUBLIC.items()}, "native": PRIVATE, "arms": arms}


def decision(row, spans):
    choices = []
    useful = False
    seen = set()
    for a, b in spans:
        selected = row["ranges"][a:b]
        common = sum(v[4] for v in selected)
        union = row["truth_characters"] + sum(v[3] for v in selected) - common
        useful |= common > 0 and 2 * common >= row["truth_characters"] and 10 * common >= 3 * union
        choices.append([selected[0][0], selected[-1][1]])
        seen.update(range(a, b))
    ordinary = sum(row["ranges"][i][3] - row["ranges"][i][4] for i in seen)
    return {"useful": useful, "ranges": choices, "ordinary_characters": ordinary}


def evaluate(lock, development, held):
    require(not {r["group"] for r in development} & {r["group"] for r in held}, "Work group intersects development")
    counts = defaultdict(lambda: defaultdict(Counter))
    gains = defaultdict(lambda: defaultdict(set))
    alarms = defaultdict(lambda: defaultdict(set))
    selected = []
    for row in held:
        cohort = row["cohort"]
        truth = row["truth_characters"]
        base = decision(row, row["baseline"])
        result = {"existing": base}
        assessed = {"existing": True}
        best = {"existing": False}
        for arm, head in lock["arms"].items():
            proposed = propose(row, head)
            assessed[arm] = proposed is not None
            highest = [proposed[1:]] if proposed else []
            best[arm] = decision(row, highest)["useful"]
            above = highest if proposed and proposed[0] > head["points"][band(row)] else []
            result[arm] = decision(row, above)
            if result[arm]["useful"] and not base["useful"]:
                gains[arm][cohort].add((row["group"], row["family"], row["id"]))
            if not truth and above and not base["ranges"]:
                alarms[arm][cohort].add(row["group"])
            result[f"plus_{arm}"] = decision(row, row["baseline"] + above)
            assessed[f"plus_{arm}"] = True
            best[f"plus_{arm}"] = False
        for arm, outcome in result.items():
            counts[f"{cohort}/{row['family']}" if truth else f"{cohort}/ordinary"][arm].update(
                files=1, useful=outcome["useful"], selected=bool(outcome["ranges"]), assessed=assessed[arm],
                wrong=bool(truth and outcome["ranges"] and not outcome["useful"]), useful_before_point=best[arm],
                ordinary_characters=outcome["ordinary_characters"])
        selected.append({"cohort": cohort, "group": row["group"], "id": row["id"], "selections": {k: v["ranges"] for k, v in result.items()}})
    summary = {"counts": counts, "gains": {k: {p: [list(x) for x in sorted(g)] for p, g in v.items()} for k, v in gains.items()},
               "new_ordinary_groups": {k: {p: sorted(g) for p, g in v.items()} for k, v in alarms.items()}}
    for arm in ARMS:
        reports = set().union(*(gains[arm][cohort] for cohort in ("short_a", "short_b", "long")))
        journal = set().union(*(gains[arm][cohort] for cohort in ("f1000", "elife")))
        assessments = [cell[arm] for k, cell in counts.items() if not k.endswith("/ordinary")]
        summary.setdefault("advance", {})[arm] = (len(reports) >= 6 and len({family for _, family, _ in reports}) == 2
            and bool(gains[arm]["long"]) and bool(gains[arm]["short_a"] or gains[arm]["short_b"]) and len(journal) >= 2
            and max(map(len, alarms[arm].values()), default=0) <= 2
            and sum(c["assessed"] for c in assessments) / sum(c["files"] for c in assessments) >= .9)
    summary["selections"] = selected
    return summary


def replay(receipt):
    lock = freeze(receipt["development"])
    require(prior.serialize(lock) == prior.serialize(receipt["freeze"]), "Frozen point or model differs")
    outcome = evaluate(lock, receipt["development"], receipt["held"])
    require(prior.serialize(outcome) == prior.serialize(receipt["outcome"]), "Frozen decisions differ")
    return outcome


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "measure", "review", "native"))
    p.add_argument("file", type=Path)
    p.add_argument("--output", type=Path)
    for cohort in PRIVATE:
        p.add_argument(f"--{cohort.replace('_', '-')}", type=Path)
    args = p.parse_args()
    paths = {key: getattr(args, key) for key in PRIVATE}
    if args.phase == "freeze":
        development = build_rows(paths)
        prior.seal(args.file, {"development": development, "freeze": freeze(development)})
    elif args.phase == "measure":
        data = prior.read(args.file)
        require(data["freeze"] == freeze(data["development"]), "Different frozen method")
        data["held"] = build_rows(paths, held=True)
        data["outcome"] = evaluate(data["freeze"], data["development"], data["held"])
        prior.seal(args.output, data)
    else:
        receipt = prior.read(args.file)
        if args.phase == "native":
            require(receipt["development"] == build_rows(paths) and receipt["held"] == build_rows(paths, held=True),
                    "Native source measurements differ")
        print(prior.serialize(replay(receipt)["counts"]), end="")


if __name__ == "__main__":
    main()
