"""Compare learned POS and dependency readings for isolated writing revisions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
from math import fsum, log1p, sqrt
from pathlib import Path

from scripts import writing_bidirectional_transfer as prior
from scripts import writing_interval_transfer as interval
from scripts import writing_revision_trajectory_transfer as source
from scripts import writing_temperature_transfer as short
from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION

SCHEMA = "chipmunk.writing-dependency-transfer/1"
PRIOR_ROSTER = "e1db77bebd9e6b0c99a1353b661bda9b7ac3ede45e8832070943d819752c2457"
PRIOR_PRIMARY = "6dc7cd5341c9ac42d83eed8a619239d58de301547806ef455f504a2bec9eb06c"
PARSER_VERSION = "3.8.0"
PARSER_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
LOCAL_VERSION = "3.8.7"
ARMS = ("pos", "dependency")
DEVELOPMENT = {"fit", "calibration"}
POS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB")
RELATIONS = {
    "subject": {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"},
    "argument": {"obj", "dobj", "iobj", "dative", "attr", "oprd"},
    "clause": {"advcl", "acl", "relcl", "ccomp", "xcomp"},
    "adverb": {"advmod", "npadvmod", "neg"},
    "nominal": {"amod", "compound", "nummod", "poss", "appos", "quantmod"},
    "prep": {"prep", "agent", "pobj", "pcomp"},
    "coordinate": {"cc", "conj", "preconj"},
    "aux": {"aux", "auxpass", "cop", "mark"},
}
METHOD = {
    "selection": "Twenty previously unused GAO works with matched original and one isolated publisher-labeled polish "
                 "from each of two generators; require identical flanks in five native sentences. Balance eligible "
                 "target positions 1, 2 and 3 by fill count with digest tie breaks. Exclude prior or Qwen works and "
                 "normalized 12-word overlap. Select before extracting parser features.",
    "head": "Fit a regularized, work-balanced direction to differences between paired edited and original target "
            "sentences, after subtracting the average feature vector of the two exact adjacent sentences. "
            "Compare punctuation, length, and part-of-speech features with a second arm adding labeled attachment "
            "shape. At inference, use only the submitted text and pick a unique highest interior sentence.",
    "point": "For each arm require strictly exceeding the second-highest per-work maximum of 35 dated genuine "
             "journal revision pairs. Fit only on 24 retained GAO source works before measuring held features.",
    "comparison": "Use the incumbent point .50 for up to six context windows. Locations require half edited "
                  "nonspace characters covered and .30 intersection over union. Do not drop unassessed files.",
    "advance": "At least four additional fresh edits, including both generators and at least three source works; "
               "at least two more additions than POS alone; at most two additional fresh original works flagged; "
               "at least 36 of 40 edits assessed. Every response contributes zero to authorship confidence.",
}


def intake(cache, history):
    source.require(all(source.digest(cache / name) == checksum for name, checksum in source.METADATA.items()),
                   "Publisher metadata differs")
    source.require("released under Apache-2.0" in (cache / "README.md").read_text(), "Publisher license differs")
    tables = {family: interval.read_shard(cache, family, checksum)
              for family, checksum in (interval.FAMILIES | interval.EXCLUDED_SHARD).items()}
    gem, nano, qwen = tables.values()
    seen = {row["group"] for row in history}
    for ledger in short.known_sources():
        seen.update(row["group"] for row in source.read(ledger)["rows"])
    works = set(qwen) | {work for work in gem.keys() | nano.keys() if source.sha(work) in seen}
    phrases = set()
    for work in works:
        for table in tables.values():
            if work in table:
                phrases.update(interval.fingerprints(table[work]["v0"]["text"]))
    order = sorted((gem.keys() & nano.keys()) - works, key=lambda work: source.sha(f"{SCHEMA}/{work}"))
    rows, stats = [], Counter(eligible=len(order))
    counts = Counter({1: 0, 2: 0, 3: 0})
    for work in order:
        if stats["admitted"] == 20:
            break
        stats["examined"] += 1
        original = gem[work]["v0"]
        if original["text"] != nano[work]["v0"]["text"]:
            stats["different_original"] += 1
            continue
        fingerprint = interval.fingerprints(original["text"])
        if phrases & fingerprint:
            stats["overlap"] += 1
            continue
        group = source.sha(work)
        try:
            base, gold, units = short.annotated(original)
            source.require(not gold, "Original has edit labels")
            published = {f: tables[f][work]["v1"] for f in interval.FAMILIES}
            source.require(all(row["edit_operation"] == "polish" for row in published.values()), "Unexpected edit label")
            variants = {f: short.annotated(row) for f, row in published.items()}
        except ValueError:
            stats["invalid_source"] += 1
            continue
        for left in sorted(counts, key=lambda i: (counts[i], source.sha(f"{group}/{i}"))):
            right = 4 - left
            matches = {f: short.align(edited, base, units, (left, right)) for f, edited in variants.items()}
            shared = set.intersection(*(set(v) for v in matches.values()))
            if not shared:
                continue
            target = min(shared, key=lambda i: source.sha(f"{group}/{left}/{i}"))
            start, end = units[target-left].start, units[target+right].end
            try:
                cases = [short.record(base[start:end], "fresh", group, "original", target_index=left,
                                      source_range=[start, end])]
                for f, matched in matches.items():
                    text, truth, bounds = matched[target]
                    cases.append(short.record(text, "fresh", group, f, truth, target_index=left, source_range=bounds))
                source.require(all(len(r["spans"]) == 5 and r["truth"] in (None, r["spans"][left]) for r in cases),
                               "Native excerpt geometry differs")
            except ValueError:
                continue
            rows.extend(cases)
            phrases.update(fingerprint)
            counts[left] += 1
            stats["admitted"] += 1
            break
        else:
            stats["no_isolated_match"] += 1
    source.require(stats["admitted"] == 20, f"Too few fresh works {stats}")
    return rows, {"counts": dict(stats), "target_positions": {str(k): v for k, v in counts.items()}}


def freeze(run, cache, previous, wheel):
    source.require(source.digest(previous / "roster.json") == PRIOR_ROSTER, "Retained source roster differs")
    source.require(source.digest(wheel) == PARSER_SHA256, "Parser release file differs")
    history = source.read(previous / "roster.json")
    fresh, admission = intake(cache, history)
    held = [r | {"cohort": "exposed_gao"} if r["cohort"] == "fresh" else r for r in history]
    source.seal(run / "roster.json", held + fresh)
    source.seal(run / "freeze.json", {
        "schema": SCHEMA, "method": METHOD, "roster_sha256": source.digest(run / "roster.json"),
        "runner_sha256": source.digest(Path(__file__)), "admission": admission,
        "versions": {name: version(name) for name in ("transformers", "torch", "tokenizers")},
        "publisher_revision": source.REVISION, "prior_roster_sha256": PRIOR_ROSTER,
        "parser": {"spacy": LOCAL_VERSION, "en-core-web-sm": PARSER_VERSION, "sha256": PARSER_SHA256, "terms": "MIT"},
        "incumbent_revision": BASELINE_REVISION, "incumbent_assets": BASELINE_ASSETS,
    })
    print(admission, flush=True)


def locked(run):
    lock = source.read(run / "freeze.json")
    source.require(lock["runner_sha256"] == source.digest(Path(__file__)) and
                   lock["roster_sha256"] == source.digest(run / "roster.json"), "Frozen plan or source differs")
    return lock, source.read(run / "roster.json")


def features(doc):
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    if not 6 <= len(tokens) <= 100 or not doc.has_annotation("DEP", require_complete=True):
        return None
    n = len(tokens)
    parts = Counter(t.pos_ for t in tokens)
    counts = Counter(t.text for t in doc if t.is_punct)
    vector = [log1p(n), sum(counts.values()) / n, counts[","] / n, (counts[":"] + counts[";"]) / n]
    vector.extend(parts[pos] / n for pos in POS)
    depth = [sum(1 for _ in token.ancestors) for token in tokens]
    distances = [abs(t.i - t.head.i) for t in tokens]
    subtrees = [t.n_lefts + t.n_rights for t in tokens]
    vector.extend((fsum(depth) / n, max(depth), fsum(log1p(d) for d in distances) / n,
                   sum(d >= 4 for d in distances) / n, sum(s == 0 for s in subtrees) / n,
                   sum(s >= 3 for s in subtrees) / n, sum(t.dep_ in {"nsubjpass", "auxpass"} for t in tokens) / n))
    collations = Counter((t.dep_, t.head.i > t.i) for t in tokens)
    vector.extend(sum(collations[label, direction] for label in labels) / n
                  for labels in RELATIONS.values() for direction in (False, True))
    return [round(x, 9) for x in vector]


def read_features(run, stage):
    import en_core_web_sm

    _, cases = locked(run)
    if stage == "held":
        points = source.read(run / "points.json")
        source.require(points["freeze"] == source.digest(run / "freeze.json"), "No matching frozen operating points")
    source.require(version("spacy") == LOCAL_VERSION and version("en-core-web-sm") == PARSER_VERSION,
                   "Incorrect parser runtime")
    nlp = en_core_web_sm.load(exclude=["ner", "lemmatizer"])
    cases = [r for r in cases if (r["cohort"] in DEVELOPMENT) == (stage == "development")]
    relevant = {r["text"][start:end] for r in cases for start, end in r["spans"]}
    texts = sorted(relevant, key=source.sha)
    lookup = dict(zip(texts, (features(doc) for doc in nlp.pipe(texts, batch_size=32)), strict=True))
    rows = {r["id"]: [lookup[r["text"][s:e]] for s, e in r["spans"]] for r in cases}
    source.seal(run / f"{stage}.json", {"freeze": source.digest(run / "freeze.json"), "rows": rows})
    print(stage, len(cases), len(texts), flush=True)


def positions(rows, arm):
    size = 4 + len(POS) if arm == "pos" else 4 + len(POS) + 7 + 2 * len(RELATIONS)
    return {i: [b - (a + c) / 2 for a, b, c in zip(left[:size], focus[:size], right[:size], strict=True)]
            for i, (left, focus, right) in enumerate(zip(rows[:-2], rows[1:-1], rows[2:], strict=True), 1)
            if left is not None and focus is not None and right is not None}


def fitted(cases, observations, arm):
    groups = defaultdict(list)
    for row in cases:
        if row["cohort"] == "fit":
            groups[row["group"]].append(row)
    work_differences = []
    for group in groups.values():
        original = next(r for r in group if r["truth"] is None)
        clean = positions(observations[original["id"]], arm)
        differences = []
        for r in group:
            if r["truth"] is not None:
                index = r["spans"].index(r["truth"])
                edited = positions(observations[r["id"]], arm)
                if index in clean and index in edited:
                    differences.append([b-a for a, b in zip(clean[index], edited[index], strict=True)])
        if differences:
            work_differences.append(differences)
    source.require(len(work_differences) >= 18, "Too few matched fit works")
    pairs = [(d, 1/(len(work_differences)*len(group))) for group in work_differences for d in group]
    size = len(pairs[0][0])
    scales = [max(.05, sqrt(fsum(w * d[i]**2 for d, w in pairs))) for i in range(size)]
    observations = [([v/s for v, s in zip(d, scales, strict=True)], w) for d, w in pairs]
    weights = [0.] * size
    for _ in range(180):
        errors = [(d, w * (interval.sigmoid(fsum(x*y for x, y in zip(d, weights, strict=True))) - 1))
                  for d, w in observations]
        weights = [value - .20 * (fsum(d[i] * e for d, e in errors) + .25 * value)
                   for i, value in enumerate(weights)]
    return {"scales": scales, "weights": weights, "works": len(work_differences), "pairs": len(pairs)}


def top(rows, arm, model):
    candidates = sorted((round(fsum(x * w / s for x, w, s in zip(values, model["weights"], model["scales"],
                             strict=True)), 9), i) for i, values in positions(rows, arm).items())
    if len(candidates) >= 2 and candidates[-1][0] > candidates[-2][0]:
        return candidates[-1]
    return None


def calibrate(cases, observations):
    models = {}
    for arm in ARMS:
        model = fitted(cases, observations, arm)
        maxima = defaultdict(list)
        for row in cases:
            if row["cohort"] == "calibration" and (peak := top(observations[row["id"]], arm, model)) is not None:
                maxima[row["group"]].append(peak[0])
        source.require(len(maxima) >= 32, "Insufficient independent calibration works")
        maxima = {group: max(values) for group, values in maxima.items()}
        models[arm] = model | {"maxima": maxima, "point": sorted(maxima.values())[-2]}
    return models


def point(run):
    _, cases = locked(run)
    dev = source.read(run / "development.json")
    source.require(dev["freeze"] == source.digest(run / "freeze.json"), "Development sources differ")
    models = calibrate(cases, dev["rows"])
    source.seal(run / "points.json", {"freeze": dev["freeze"], "models": models})
    print({arm: (len(m["maxima"]), m["point"]) for arm, m in models.items()}, flush=True)


def primary(run, previous, snapshot):
    from app.detectors.text.classifier import FourClassTextClassifier

    lock, cases = locked(run)
    prior.torch_runtime(lock, snapshot, BASELINE_ASSETS)
    source.require(source.digest(previous / "primary.json") == PRIOR_PRIMARY, "Retained primary rows differ")
    old = source.read(previous / "primary.json")["rows"]
    fresh = [r for r in cases if r["cohort"] == "fresh"]
    texts = sorted({text for row in fresh for text in row["context"]}, key=source.sha)
    classifier = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    values = {}
    for start in range(0, len(texts), 48):
        batch = texts[start:start+48]
        values.update(zip(batch, classifier.predict(batch, batch=6, check=lambda _: None), strict=True))
        print("incumbent", len(values), len(texts), flush=True)
    rows = {r["id"]: [values[t] for t in r["context"]] if r["cohort"] == "fresh" else old[r["id"]]
            for r in cases if r["cohort"] not in DEVELOPMENT}
    source.seal(run / "primary.json", {"freeze": source.digest(run / "freeze.json"), "rows": rows})


def evaluate(cases, readings, old, models):
    metrics = defaultdict(lambda: defaultdict(Counter))
    extra, alarms = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    decisions = {}
    for row in cases:
        cohort = row["cohort"]
        if cohort in DEVELOPMENT:
            continue
        spans = row["spans"]
        predictions = old[row["id"]]
        source.require(len(spans) == len(predictions) and all(0 <= p <= 1 for p in predictions), "Invalid context scores")
        ranges = {"incumbent": [[spans[max(0, i-1)][0], spans[min(len(spans)-1, i+1)][1]]
                                for i, p in enumerate(predictions) if p >= .50][:6]}
        assessed = {"incumbent": True}
        spaces = set(row["spaces"])

        def useful(locations, truth=row["truth"], excluded=spaces):
            return any(short.useful(span, truth, excluded) for span in locations)

        for arm in ARMS:
            peak = top(readings[row["id"]], arm, models[arm])
            assessed[arm] = peak is not None
            ranges[arm] = [spans[peak[1]]] if peak and peak[0] > models[arm]["point"] else []
            ranges[f"plus_{arm}"] = ranges["incumbent"] + ranges[arm]
            assessed[f"plus_{arm}"] = True
            if useful(ranges[arm]) and not useful(ranges["incumbent"]):
                extra[arm][cohort].append([row["group"], row["family"]])
            if row["truth"] is None and ranges[arm] and not ranges["incumbent"]:
                alarms[arm][cohort].append(row["group"])
        for arm, locations in ranges.items():
            hit = useful(locations)
            counts = metrics[f"{cohort}/{row['family']}"][arm]
            counts.update(files=1, assessed=assessed[arm], flagged=bool(locations), useful=hit,
                          wrong=bool(row["truth"] and locations and not hit))
        decisions[row["id"]] = ranges
    for by_cohort in extra.values():
        for items in by_cohort.values():
            items.sort()
    for by_cohort in alarms.values():
        for cohort, items in by_cohort.items():
            by_cohort[cohort] = sorted(set(items))
    recovered = extra["dependency"]["fresh"]
    new_pos = extra["pos"]["fresh"]
    passed = (len(recovered) >= max(4, len(new_pos)+2) and {family for _, family in recovered} == set(interval.FAMILIES)
              and len({group for group, _ in recovered}) >= 3 and len(alarms["dependency"]["fresh"]) <= 2
              and sum(metrics[f"fresh/{f}"]["dependency"]["assessed"] for f in interval.FAMILIES) >= 36)
    return {"metrics": metrics, "extra": extra, "alarms": alarms, "passes": passed, "decisions": decisions}


def replay(packet):
    cases = packet["geometry"]
    readings = packet["readings"]
    source.require(packet["freeze"]["schema"] == SCHEMA and packet["freeze"]["method"] == METHOD, "Wrong plan")
    source.require(set(readings) == {r["id"] for r in cases} and len(readings) == len(cases), "Incomplete position rows")
    role_groups = {role: {r["group"] for r in cases if r["cohort"] == role}
                   for role in ("fit", "calibration", "fresh")}
    source.require(all(not a & b for i, a in enumerate(role_groups.values()) for b in list(role_groups.values())[i+1:]),
                   "Source-work leakage")
    for r in cases:
        source.require(len(readings[r["id"]]) == len(r["spans"]) and r["truth"] in [None, *r["spans"]],
                       "Invalid position geometry")
    models = calibrate(cases, readings)
    return {"models": models, **evaluate(cases, readings, packet["primary"], models)}


def finish(run, output):
    lock, cases = locked(run)
    dev, held, old, points = (source.read(run / f"{part}.json") for part in ("development", "held", "primary", "points"))
    source.require(all(x["freeze"] == source.digest(run / "freeze.json") for x in (dev, held, old, points)),
                   "Measurements do not share one freeze")
    packet = {
        "freeze": lock,
        "geometry": [{key: row[key] for key in ("id", "cohort", "group", "family", "spans", "truth", "spaces", "text_sha256")}
                     for row in cases],
        "readings": dev["rows"] | held["rows"], "primary": old["rows"],
    }
    result = replay(packet)
    source.require(result["models"] == points["models"], "Frozen operating points differ from replay")
    source.seal(output, packet | {"result": result})
    print(source.serialize({key: result[key] for key in ("metrics", "extra", "alarms", "passes")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "point", "held", "primary", "finish", "review"))
    parser.add_argument("path", type=Path)
    for key in ("cache", "previous", "wheel", "snapshot", "output"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.path, args.cache, args.previous, args.wheel)
    elif args.stage in ("development", "held"):
        read_features(args.path, args.stage)
    elif args.stage == "point":
        point(args.path)
    elif args.stage == "primary":
        primary(args.path, args.previous, args.snapshot)
    elif args.stage == "finish":
        finish(args.path, args.output)
    else:
        packet = source.read(args.path)
        result = replay(packet)
        source.require(source.serialize(result) == source.serialize(packet["result"]), "Receipt differs")
        print("Independent arithmetic replay matched.")


if __name__ == "__main__":
    main()
