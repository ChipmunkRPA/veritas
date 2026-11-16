# SPDX-License-Identifier: MIT
"""Compare masked grammar patterns on source-aligned local rewrites."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_editorial_direction_transfer as editorial
from scripts import writing_editorial_negative_transfer as parent

require = editorial.require

SCHEMA = "chipmunk.writing-function-sequence-transfer/1"
PANEL = editorial.PANEL
BASE = PANEL / "writing-editorial-direction-2026-11-15.json.gz"
ARMS = ("surface", "sequence", "ordinary_revisions")
LEVELS = {"tight": .10, "wide": .25}
QUOTA = 24
GENRES = ("news", "reports")
FAMILIES = ("gpt-5.4", "gemini-2.5-flash")
SINGLES = {(i,) for i in range(1, 5)}
DOUBLES = {(i, i+1) for i in range(1, 4)}
METHOD = {
    "fit": "Use paired original and rewritten report sentences. Fit a surface baseline, a sequence model using "
           "function words and masked content classes, and the same sequence model with penalties for score "
           "changes across historical ordinary revisions. No encoder or source document is used during scoring.",
    "location": "Use the strongest interior sentence or consecutive pair of sentences in one supplied excerpt. "
                "Set operating points from separate ordinary news or report excerpts and report the exact sentence "
                "selection and coverage of source-aligned changed words separately.",
    "scope": "The probe keeps grammatical words, punctuation and coarse word forms. No content word identities, "
             "verbatim keyword phrases, topics, author identifiers or generator names enter the individual score.",
    "fresh": "Choose unused six-sentence news and report work pairs from pinned source shards. Preserve their "
             "exact source text, publisher labels and earlier and later boundaries. Exclude work hashes and "
             "normalized twelve-word overlaps before reading any scores.",
}

FUNCTION = frozenset((
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "i", "me", "my", "mine", "we", "us",
    "our", "ours", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "they", "them", "their",
    "theirs", "who", "whom", "whose", "which", "what", "someone", "anyone", "everyone", "something", "anything",
    "everything", "is", "are", "was", "were", "be", "being", "been", "am", "do", "does", "did", "doing", "done",
    "have", "has", "had", "having", "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "and", "or", "but", "nor", "so", "yet", "if", "then", "than", "because", "although", "though", "while",
    "when", "where", "as", "for", "of", "by", "from", "in", "into", "out", "onto", "on", "upon", "at", "with",
    "within", "without", "under", "over", "before", "after", "between", "through", "to", "until", "during",
    "since", "about", "against", "along", "across", "despite", "per", "via", "among", "up", "down", "off",
    "near", "beyond", "around", "further", "not", "no", "there", "here", "also", "only", "just", "more", "most",
    "less", "least", "very", "too", "such", "each", "every", "all", "any", "some", "many", "much", "both", "either",
    "neither", "few", "several", "once", "twice", "again", "already", "still", "even", "rather", "instead",
    "whether", "how", "why", "whereby", "whereupon", "otherwise", "whereas", "thereby", "together", "thus",
    "therefore", "hence", "however", "moreover", "furthermore",
))
TOKEN = re.compile(r"[^\W_]+(?:[’'][^\W_]+)*|[^\s\w]", re.UNICODE)


def term(raw):
    word = raw.casefold().replace("’", "'")
    if word in FUNCTION:
        return word
    if not any(char.isalnum() for char in word):
        return "-" if word in "–—−" else "'" if word in "’‘" else '"' if word in "“”" else word
    if any(char.isdigit() for char in word):
        return "<number>"
    if word.endswith(("n't", "'d", "'ll", "'re", "'s", "'ve", "'m")):
        return "<contraction>"
    if len(word) > 5:
        for suffix in ("ation", "ment", "ness", "ity", "ing", "ed", "ly", "al", "ive", "ous", "es"):
            if word.endswith(suffix):
                return f"<{suffix}>"
    return "<word>"


def features(sentence, *, sequence):
    raw = [match.group() for match in TOKEN.finditer(sentence)]
    words = [word for word in raw if any(c.isalnum() for c in word)]
    if not 6 <= len(words) <= 110:
        return {}
    size = len(words)
    mapped = [term(token) for token in raw]
    structural = Counter(mapped)
    counts = {
        "length": math.log(size) - 3,
        "average_word_length": (sum(map(len, words)) / size - 5.) / 3.,
        "function_share": sum(structural[key] for key in structural.keys() & FUNCTION) / size,
        "punctuation_share": (len(raw) - size) / size,
        "number_share": structural["<number>"] / size,
        "capital_share": sum(word[0].isupper() for word in words[1:]) / size,
        "repeated_share": (size - len({word.casefold() for word in words})) / size,
    }
    for symbol in (",", ";", ":", "-", "(", "/", '"'):
        counts[f"punct/{symbol}"] = structural[symbol] / size
    if sequence:
        marked = ["^", *mapped, "$"]
        for n in (1, 2, 3, 4):
            normalizer = math.sqrt(len(marked) - n + 1)
            grams = Counter(" ".join(marked[i:i+n]) for i in range(len(marked) - n + 1))
            counts.update({f"gram/{n}/{gram}": value / normalizer for gram, value in grams.items()})
    return counts


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def save(path, body):
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("xb") as store:
        store.write(gzip.compress(payload, mtime=0))


def prior_plan(path):
    pinned = read(BASE)
    require(parent.sha(path.read_bytes()) == pinned["source_plan_sha256"], "The retained editorial and rewrite plan differs")
    return read(path)


def _difference(earlier, later):
    return {key: later.get(key, 0.) - earlier.get(key, 0.) for key in earlier.keys() | later.keys()}


def _pairs(rows, versions):
    a, b = versions
    groups = defaultdict(dict)
    for row in rows:
        if len(row["sentences"]) == 1:
            groups[row["group"]][row["delivery"]] = row["sentences"][0]["text"]
    return [(pair[a], pair[b]) for pair in groups.values() if set(pair) == {a, b}]


def train(plan, arm):
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    sequence = arm != "surface"
    measured = []
    scores, weights = [], []
    reports = _pairs(plan["fit"], ("original", "edited"))
    revision = _pairs(plan["wiki_fit"], ("before", "after")) if arm == "ordinary_revisions" else []
    for ordinary, pairs in ((False, reports), (True, revision)):
        for earlier, later in pairs:
            a, b = (features(text, sequence=sequence) for text in (earlier, later))
            if not a or not b:
                continue
            delta = _difference(a, b)
            measured.extend((delta, {key: -value for key, value in delta.items()}))
            scores.extend((1, 0))
            weights.extend((1., 1.) if not ordinary else (.8, .8))
            if ordinary:
                measured.extend((delta, {key: -value for key, value in delta.items()}))
                scores.extend((0, 1))
                weights.extend((.8, .8))
    counts = Counter(key for row in measured for key, value in row.items() if value)
    features_kept = {key for key, count in counts.items() if count >= 10 or not key.startswith("gram/")}
    matrix_rows = [{key: value for key, value in row.items() if key in features_kept} for row in measured]
    vocabulary = DictVectorizer(sparse=False)
    matrix = vocabulary.fit_transform(matrix_rows)
    learner = LogisticRegression(C=4, solver="liblinear", fit_intercept=False, max_iter=600, tol=1e-8, random_state=1)
    learner.fit(matrix, np.asarray(scores), sample_weight=np.asarray(weights))
    terms = dict(zip(vocabulary.get_feature_names_out().tolist(), learner.coef_[0].round(9).tolist(), strict=True))
    return {"counts": {"reports": len(reports), "historical": len(revision), "feature_terms": len(terms)}, "weights": terms}


def measure(row, fits):
    result = {arm: [] for arm in ARMS}
    for sentence in row["sentences"]:
        raw = features(sentence["text"], sequence=True)
        for arm, fit in fits.items():
            weights = fit["weights"]
            value = round(math.fsum(weights.get(key, 0.) * number for key, number in raw.items()), 7) if raw else None
            result[arm].append(value)
    return {key: row[key] for key in ("group", "genre", "family", "delivery", "sha256", "changed", "words")} | {
        "positions": [[sentence["index"], sentence["offset"], sentence["offset"] + len(sentence["text"])] for sentence in row["sentences"]],
        "readings": result,
    }


def ranges(row, arm):
    present = {index: score for (index, _, _), score in zip(row["positions"], row["readings"][arm], strict=True) if score is not None}
    options = {(i,): value for i, value in present.items()}
    options.update({(i, i+1): (value + present[i+1]) * .5 for i, value in present.items() if i+1 in present})
    return options


def baseline_centers(ordinary, arm, width):
    by_work = []
    for row in ordinary:
        candidates = [weight for slot, weight in ranges(row, arm).items() if len(slot) == width]
        if candidates:
            by_work.append(max(candidates))
    return by_work


def calibration(ordinary):
    thresholds = {}
    for genre in GENRES:
        controls = [row for row in ordinary if row["genre"] == genre]
        require(len(controls) == 36, "The ordinary calibration roster differs")
        thresholds[genre] = {}
        for arm in ARMS:
            points = {}
            for width in (1, 2):
                choices = sorted(baseline_centers(controls, arm, width), reverse=True)
                require(len(choices) >= 24, "Too few independent ordinary observations")
                points[str(width)] = {key: round(choices[math.floor(len(choices)*budget/2)], 7) for key, budget in LEVELS.items()}
            thresholds[genre][arm] = points
    return thresholds


def _select(row, arm, points, level):
    distances = [(score - points[str(len(slot))][level], slot) for slot, score in ranges(row, arm).items()]
    if not distances:
        return None
    excess, slot = max(distances, key=lambda value: (value[0], -len(value[1]), tuple(-i for i in value[1])))
    if excess <= 0:
        return ()
    return slot


def result(rows, points):
    by_group = defaultdict(dict)
    for row in rows:
        by_group[row["group"]][row["delivery"]] = row
    ordinary = len(by_group)
    results = {}
    for arm in ARMS:
        for level in LEVELS:
            counts = defaultdict(Counter)
            for pair in by_group.values():
                controls = set(pair) == {"before", "after"}
                original, edited = (pair["before"], pair["after"]) if controls else (pair["original"], pair["edited"])
                genre = edited["genre"] if edited["genre"] in points else "reports"
                point = points[genre][arm]
                reference, selected = (_select(row, arm, point, level) for row in (original, edited))
                truth = set(edited["changed"])
                exact = {edited["words"][index][2] for index in truth}
                kept = {i for i, word in enumerate(edited["words"]) if selected and word[2] in selected}
                hits = len(kept & truth)
                useful = bool(truth and 2 * hits >= len(truth) and 10 * hits >= 3 * len(kept | truth))
                selected_correct = bool(selected and set(selected) == exact)
                flags = bool(selected)
                counts["all"].update(works=1, original_flags=bool(reference), revised_flags=flags,
                    ordinary_after=controls and flags, correct_sentence=selected_correct and not controls,
                    useful=useful and not controls, original_assessed=reference is not None, revised_assessed=selected is not None,
                    same_location=bool(useful and selected == reference and not controls),
                    misplaced=bool(flags and not selected_correct and not controls))
                bucket = f"{edited['genre']}/{edited['family']}"
                counts[bucket].update(works=1, original_flags=bool(reference), revised_flags=flags,
                    correct_sentence=selected_correct and not controls, useful=useful and not controls,
                    original_assessed=reference is not None, revised_assessed=selected is not None,
                    same_location=bool(useful and selected == reference and not controls))
            require(counts["all"]["works"] == ordinary, "A matched group is missing")
            results[f"{arm}/{level}"] = {key: dict(value) for key, value in counts.items()}
    return results


def fit(path, output):
    plan = prior_plan(path)
    fits = {arm: train(plan, arm) for arm in ARMS}
    ordinary = [measure(row, fits) for row in plan["calibration"] if row["genre"] in GENRES]
    points = calibration(ordinary)
    phases = {name: [measure(row, fits) for row in rows] for name, rows in plan["held"].items()}
    summaries = {name: result(rows, points) for name, rows in phases.items()}
    save(output, {"schema": SCHEMA, "method": METHOD, "parent_sha256": parent.sha(path.read_bytes()), "models": fits,
                  "calibration": ordinary, "points": points, "prior": phases, "summaries": summaries})
    print(json.dumps({name: {arm: stats["all"] for arm, stats in values.items()} for name, values in summaries.items()}, indent=2))


def sources():
    from scripts import writing_layer_transfer as earlier
    from scripts import writing_structured_range_transfer as current

    return [(genre, family, pin) for genre in GENRES for family in FAMILIES
            for pin in (("test", *current.SHARDS[genre][family]),)
            + ((earlier.SHARDS[genre, family],) if family == "gpt-5.4" else ())]


def inventory(source_dir, shards, known, fingerprints):
    for genre, family, (split, size, sha) in shards:
        path = source_dir / split / f"{genre}_{family}.csv"
        require(path.stat().st_size == size and parent.sha(path.read_bytes()) == sha, "The pinned publisher shard differs")
        csv.field_size_limit(1024 * 1024)
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, strict=True):
                if row["version"] == "v0" and parent.sha(row["id"]) in known:
                    yield from fingerprints(row["text"])


def holdout(folder, earlier, output, reference):
    from scripts import writing_ratio_fusion_transfer as native

    old = prior_plan(earlier)
    if reference:
        paths = editorial.exposure_paths(read(reference)["fresh_intake"]["exposures"])
    else:
        paths = editorial.exposure_files(None)
    paths = [path for path in paths if not path.name.startswith("writing-function-sequence-")]
    ledger, seen = editorial.exposure_snapshot(paths)
    if reference:
        require(ledger == read(reference)["fresh_intake"]["exposures"], "The replay exposure snapshot differs")
    exposed = old["fit"] + old["calibration"] + old["wiki_fit"] + [row for rows in old["held"].values() for row in rows]
    seen.update(row["group"] for row in exposed)
    fingerprints = native.adjacent.intake.fingerprints
    terms = {term for row in exposed for sentence in row["sentences"] for term in fingerprints(sentence["text"])}
    shards = sources()
    terms.update(inventory(folder, shards, seen, fingerprints))
    counts = defaultdict(Counter)
    rows = []
    for genre, family, pin in shards:
        slots = counts[genre, family]
        if slots.total() >= QUOTA:
            continue
        table = native.load_shard(folder, genre, family, pin)
        for key in sorted(table, key=lambda value: parent.sha(f"{SCHEMA}/{genre}/{family}/{value}")):
            if slots.total() == QUOTA:
                break
            group, versions = parent.sha(key), table[key]
            if group in seen or "v0" not in versions:
                continue
            marks = set(fingerprints(versions["v0"]["text"]))
            if not marks or marks & terms:
                continue
            eligible = {slot for slot in SINGLES | DOUBLES if slots[slot] < (5 if len(slot) == 1 else 2)}
            try:
                chosen = next(native.local_examples(versions, eligible), None)
            except (KeyError, ValueError):
                continue
            if chosen is None:
                continue
            slot, stage, original, revision = chosen
            common = {"group": group, "genre": genre, "family": family, "target": list(slot)}
            pair = (original | {"delivery": "original", "changed": []}, revision | {"delivery": "edited"})
            rows.extend(parent.source_record(item | common) | {"variant": stage, "partition": pin[0]} for item in pair)
            seen.add(group)
            terms.update(marks | set(fingerprints(revision["text"])))
            slots[slot] += 1
        print(f"{genre}, {family}, {pin[0]}: {slots.total()} matched works in all inspected partitions", flush=True)
        del table
    total = sum(arm.total() for arm in counts.values())
    require(total >= 40 and {row["family"] for row in rows} == set(FAMILIES), "The held study needs both generators and at least 40 works")
    save(output, {"schema": SCHEMA, "method": METHOD, "exposures": ledger, "previous_plan_sha256": parent.sha(earlier.read_bytes()),
                  "sources": [[genre, family, list(pin)] for genre, family, pin in shards],
                  "quota": {f"{genre}/{family}": {",".join(map(str, slot)): count for slot, count in values.items()}
                            for (genre, family), values in counts.items()}, "rows": rows})


def evaluate(frozen, fit_path, output):
    held, measurements = read(frozen), read(fit_path)
    require(held["schema"] == measurements["schema"] == SCHEMA and held["method"] == measurements["method"] == METHOD
            and held["previous_plan_sha256"] == measurements["parent_sha256"]
            and calibration(measurements["calibration"]) == measurements["points"], "The retained points or fresh plan differs")
    measured = [measure(row, measurements["models"]) for row in held["rows"]]
    got = result(measured, measurements["points"])
    intake = {key: held[key] for key in ("exposures", "sources", "quota")}
    packet = measurements | {"fresh_intake": intake | {"plan_sha256": parent.sha(frozen.read_bytes())},
                             "held": measured, "held_results": got}
    replay(packet)
    save(output, packet)
    print(json.dumps(got, indent=2))


def replay(packet):
    require(packet["schema"] == SCHEMA and packet["method"] == METHOD and calibration(packet["calibration"]) == packet["points"],
            "The published fit metadata or ordinary points do not match")
    require(all(result(rows, packet["points"]) == packet["summaries"][name] for name, rows in packet["prior"].items()),
            "The earlier source diagnostics differ")
    known = {row["group"] for row in packet["calibration"]} | {row["group"] for phase in packet["prior"].values() for row in phase}
    seen = set()
    quota = defaultdict(Counter)
    for original, edited in zip(packet["held"][::2], packet["held"][1::2], strict=True):
        group = original["group"]
        require(group not in known and group not in seen and group == edited["group"]
                and [row["delivery"] for row in (original, edited)] == ["original", "edited"], "The held groups overlap")
        seen.add(group)
        for row in (original, edited):
            require(set(row["readings"]) == set(ARMS) and all(len(value) == len(row["positions"]) for value in row["readings"].values())
                    and all(0 <= a < b <= len(row["words"]) for a, b in ((i, i+1) for i in row["changed"])), "Broken source positions")
        slot = tuple(sorted({edited["words"][i][2] for i in edited["changed"]}))
        require(slot in SINGLES | DOUBLES and not original["changed"], "The exact edit label differs")
        quota[f"{edited['genre']}/{edited['family']}"][",".join(map(str, slot))] += 1
    require(all(dict(quota[key]) == count for key, count in packet["fresh_intake"]["quota"].items()),
            "The frozen source counts differ")
    current = result(packet["held"], packet["points"])
    require(current == packet["held_results"], "The fresh holdout results differ on replay")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("fit")
    train_parser.add_argument("parent", type=Path)
    train_parser.add_argument("output", type=Path)
    freeze = sub.add_parser("freeze")
    for name in ("sources", "parent", "output"):
        freeze.add_argument(name, type=Path)
    freeze.add_argument("--reference", type=Path)
    test = sub.add_parser("test")
    for name in ("frozen", "fit", "output"):
        test.add_argument(name, type=Path)
    verify = sub.add_parser("review")
    verify.add_argument("receipt", type=Path)
    opt = parser.parse_args()
    if opt.command == "fit":
        fit(opt.parent, opt.output)
    elif opt.command == "freeze":
        holdout(opt.sources, opt.parent, opt.output, opt.reference)
    elif opt.command == "test":
        evaluate(opt.frozen, opt.fit, opt.output)
    else:
        print(json.dumps(replay(read(opt.receipt)), indent=2))


if __name__ == "__main__":
    main()
