"""Compare typed decisions about individual sentences on matched English polishes."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from scripts import writing_bidirectional_transfer as prior
from scripts import writing_dependency_transfer as intake
from scripts import writing_interval_transfer as interval
from scripts import writing_revision_trajectory_transfer as source
from scripts import writing_temperature_transfer as short
from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION

SCHEMA = "chipmunk.writing-typed-decision-transfer/1"
CALIBRATION_LEDGER = "writing-dependency-transfer-2026-11-13.json"
CALIBRATION_LEDGER_SHA = "8a6b5e0bb2fdaee0a57383031102cd854768f467802d601c599dee3817d85d86"
PUBLISHER_LEDGER = "writing-long-position-transfer-2026-11-12.json"
PUBLISHER_LEDGER_SHA = "1b4713fb1fc38053a1f28d2c4e564fa587cb16950d261070e8dd7ab6d6871801"
JOURNAL_PAIRS_SHA = "e8633d587e1943f1f9dd7c7dc1d93f7809590fcb82a7956ddaa6621e2499ea69"
MODEL_REVISION = "f2b4faf51023039425946074e2cf1361d2db11d5"
MODEL_MANIFEST = "deb2de8ec6b1520b74866249bda632f40501440cdac29fdd90af80a3c422c778"
GEOMETRY = ("id", "cohort", "group", "family", "spans", "truth", "spaces", "text_sha256")
RUNTIME = {"laya-mlx": "0.2.0", "mlx": "0.32.2", "tokenizers": "0.23.2", "numpy": "2.4.6"}
OPTIONS = (
    ("human", "Written by a person without a language model generating or rewriting this sentence."),
    ("ai", "Generated or rewritten by an AI language model."),
)
QUESTION = "Who produced the target sentence? If surrounding sentences are present, assess only the target."
ARMS = ("isolated", "context")
METHOD = {
    "input": "Read the target sentence by itself and with its exact two native neighbors. Use the same fixed typed question "
             "with both orders of the two labeled choices and average the AI responses. Do not score truncated input.",
    "decision": "Require complete measurement of all interior sentences. The highest untied interior sentence above "
                "that arm's locked ordinary-work maximum point gives one exact location.",
    "point": "Freeze a separate point per input using the second-highest maximum across 35 earlier dated authentic "
             "journal revision works, after requiring at least 30 measurable calibration works. Reconstruct the exact "
             "70 source excerpts using prior published hashes and measure the candidate from scratch.",
    "holdout": "Use twenty GAO reports with matching original excerpts and one exact labeled edit from each of two "
               "generators. Exclude works appearing in earlier published writing ledgers and normalized 12-word "
               "overlap with earlier eligible source works or retained comparison shards.",
    "validation": "Measure both arms against the incumbent native three-sentence context response at its unchanged "
                  ".50 operating point. Keep the first six incumbent ranges. A successful span intersects at least "
                  "half of the edited nonwhitespace characters and achieves .30 nonwhitespace intersection over union.",
    "advancement": "Require at least four additional fresh edit locations across three works and both generators, "
                   "no more than two additional fresh original works, and at least 36 of 40 edits assessed. "
                   "Adding this check cannot affect authorship confidence.",
}


def checked_ledger(name, checksum):
    path = short.PANELS / name
    source.require(source.digest(path) == checksum, f"Retained ledger differs: {name}")
    return source.read(path)


def calibration(journal_pairs, publisher_xml):
    from scripts import verify_writing_long_position_sources as publisher

    source.require(source.digest(journal_pairs) == JOURNAL_PAIRS_SHA, "Retained journal pairs differ")
    history = checked_ledger(CALIBRATION_LEDGER, CALIBRATION_LEDGER_SHA)
    originals = checked_ledger(PUBLISHER_LEDGER, PUBLISHER_LEDGER_SHA)
    expected = {(r["group"], r["family"]): r for r in history["geometry"] if r["cohort"] == "calibration"}
    cases = {(r["group"], r["variant"]): r for r in originals["cases"]}
    texts = {}
    for row in source.read(journal_pairs):
        texts[row["group"], "before"] = row["before"]["text"]
        texts[row["group"], "edited"] = row["after"]["text"]
    for work in originals["sources"]:
        if work["publisher"] != "f1000":
            continue
        for side, (_, filename) in publisher.source_paths(work).items():
            key = f"f1000-{work['group']}", side
            case = cases[work["group"], side]
            digest, _, _, article = publisher.verify_version(work, side, case, publisher_xml / filename)
            source.require(digest == work["source_sha256"][side], "Retained publisher XML differs")
            units = short.native(article)
            windows = [article[units[i - 2].start:units[i + 2].end] for i in range(2, len(units) - 2)]
            matches = [text for text in windows if source.sha(text) == expected[key]["text_sha256"]]
            source.require(len(matches) == 1, "Exact historical calibration excerpt differs")
            texts[key] = matches[0]
    source.require(texts.keys() == expected.keys() and len(texts) == 70, "Incomplete historical calibration")
    rows = [short.record(texts[group, side], "calibration", group, side) | {"id": known["id"]}
            for (group, side), known in expected.items()]
    source.require(all({key: r[key] for key in GEOMETRY} == expected[r["group"], r["family"]] for r in rows),
                   "Reconstructed calibration geometry differs")
    return rows


def freeze(run, cache, journal_pairs, publisher_xml):
    calibrators = calibration(journal_pairs, publisher_xml)
    own_ledger = "writing-typed-decision-transfer-2026-11-13.json"
    ledgers = sorted(path for path in short.PANELS.glob("writing-*.json") if path.name != own_ledger)
    ids = set()
    for path in ledgers:
        ids.update(re.findall(r'(?<=")[0-9a-f]{64}(?=")', path.read_text()))
    fresh, stats = intake.intake(cache, [{"group": group} for group in ids])
    source.seal(run / "roster.json", calibrators + fresh)
    source.seal(run / "freeze.json", {
        "schema": SCHEMA, "method": METHOD, "model_revision": MODEL_REVISION, "model_manifest": MODEL_MANIFEST,
        "runtime": RUNTIME, "question": QUESTION, "options": OPTIONS, "admission": stats,
        "earlier_ledgers": {path.name: source.digest(path) for path in ledgers},
        "journal_pairs_sha256": JOURNAL_PAIRS_SHA,
        "source_files": source.METADATA | {f"reports_{f}.csv": checksum
                                           for f, checksum in (interval.FAMILIES | interval.EXCLUDED_SHARD).items()},
        "source_revision": source.REVISION, "roster_sha256": source.digest(run / "roster.json"),
        "runner_sha256": source.digest(Path(__file__)), "incumbent_revision": BASELINE_REVISION,
        "versions": {n: version(n) for n in ("torch", "transformers", "tokenizers")},
    })
    print(stats, flush=True)


def locked(run):
    lock = source.read(run / "freeze.json")
    source.require(lock["schema"] == SCHEMA and lock["roster_sha256"] == source.digest(run / "roster.json")
                   and lock["runner_sha256"] == source.digest(Path(__file__)), "Source or method changed")
    return lock, source.read(run / "roster.json")


def state(row, index, arm):
    def sentence(i):
        start, end = row["spans"][i]
        return row["text"][start:end]

    target = "Target sentence: " + sentence(index)
    return target if arm == "isolated" else "\n".join((
        "Previous sentence: " + sentence(index - 1), target, "Next sentence: " + sentence(index + 1),
    ))


def measures(run, path, role):
    import laya_mlx
    from laya_mlx.common import build_prefix, serialize_state

    _, rows = locked(run)
    source.require(os.getenv("HF_HUB_OFFLINE") == "1" and all(version(name) == v for name, v in RUNTIME.items()),
                   "Use the pinned offline runtime")
    if role == "held":
        source.require(source.read(run / "points.json")["freeze"] == source.digest(run / "freeze.json"),
                       "Record genuine points before any held measurements")
    manifest = path / "manifest.json"
    source.require(source.digest(manifest) == MODEL_MANIFEST, "Conversion manifest differs")
    for name, record in source.read(manifest)["files"].items():
        source.require(source.digest(path / name) == record["sha256"], f"Converted {name} differs")
    agent = laya_mlx.load(path, dtype="float16")
    questions = {k: {"type": "choice", "instructions": QUESTION, "criteria": dict(v)}
                 for k, v in (("forward", OPTIONS), ("reverse", OPTIONS[::-1]))}
    prefix = max(len(build_prefix(agent.tok, agent._to_internal(q), agent.cfg["head_max_len"])[0])
                 for q in questions.values())
    rows = [row for row in rows if (row["cohort"] == "calibration") == (role == "calibration")]
    inputs = {state(r, i, arm) for r in rows for i in range(1, len(r["spans"]) - 1) for arm in ARMS}
    observed = {}
    for index, text in enumerate(sorted(inputs, key=source.sha), 1):
        ids = agent.tok(serialize_state(text).replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        if len(ids) > agent.cfg["max_len"] - prefix - 1:
            observed[text] = None
            continue
        output = agent.predict(text, questions)["answers"]
        observed[text] = [output[q]["probabilities"]["ai"] for q in questions]
        if index % 100 == 0:
            print(role, index, len(inputs), flush=True)
    scores = {r["id"]: {arm: [observed[state(r, i, arm)] for i in range(1, len(r["spans"]) - 1)]
                        for arm in ARMS} for r in rows}
    source.seal(run / f"{role}.json", {"freeze": source.digest(run / "freeze.json"), "rows": scores,
                                     "unique_inputs": len(inputs)})
    print(role, "measured", len(inputs), flush=True)


def peak(scores):
    if any(votes is None for votes in scores):
        return None
    source.require(all(len(votes) == 2 and all(0 <= v <= 1 for v in votes) for votes in scores),
                   "Invalid paired readout")
    values = [(round(sum(votes) / 2, 5), index) for index, votes in enumerate(scores, 1)]
    values.sort(reverse=True)
    if not values or len(values) > 1 and values[0][0] == values[1][0]:
        return None
    return values[0]


def calibrate(rows, scores):
    points = {}
    for arm in ARMS:
        maxima = defaultdict(list)
        for r in rows:
            if r["cohort"] == "calibration" and (top := peak(scores[r["id"]][arm])) is not None:
                maxima[r["group"]].append(top[0])
        source.require(len(maxima) >= 30, "Too few assessable genuine works for calibration")
        frozen = {group: max(values) for group, values in maxima.items()}
        points[arm] = {"cutoff": sorted(frozen.values())[-2], "maxima": frozen}
    return points


def point(run):
    _, rows = locked(run)
    data = source.read(run / "calibration.json")
    source.require(data["freeze"] == source.digest(run / "freeze.json"), "Calibration does not match the freeze")
    cutoffs = calibrate(rows, data["rows"])
    source.seal(run / "points.json", {"freeze": data["freeze"], "arms": cutoffs})
    print("frozen", {a: p["cutoff"] for a, p in cutoffs.items()}, flush=True)


def primary(run, snapshot):
    from app.detectors.text.classifier import FourClassTextClassifier

    lock, rows = locked(run)
    prior.torch_runtime(lock, snapshot, BASELINE_ASSETS)
    fresh = [r for r in rows if r["cohort"] == "fresh"]
    new_inputs = sorted({text for r in fresh for text in r["context"]}, key=source.sha)
    clf = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    readings = {}
    for index in range(0, len(new_inputs), 48):
        texts = new_inputs[index:index + 48]
        readings.update(zip(texts, clf.predict(texts, batch=6, check=lambda _: None), strict=True))
        print("incumbent", len(readings), len(new_inputs), flush=True)
    result = {r["id"]: [readings[t] for t in r["context"]] for r in fresh}
    source.seal(run / "primary.json", {"freeze": source.digest(run / "freeze.json"), "rows": result})


def evaluate(rows, readings, incumbent, points):
    mapping = {}
    counters = defaultdict(lambda: defaultdict(Counter))
    extra = defaultdict(list)
    alarms = defaultdict(set)
    for row in rows:
        spans, label, scope, group = (row[k] for k in ("spans", "truth", "cohort", "group"))
        if scope == "calibration":
            continue
        old_scores = incumbent[row["id"]]
        source.require(len(old_scores) == len(spans) and all(0 <= p <= 1 for p in old_scores), "Bad comparator")
        old = [[spans[max(0, i-1)][0], spans[min(len(spans)-1, i+1)][1]]
               for i, p in enumerate(old_scores) if p >= .5][:6]
        spaces = set(row["spaces"])

        def hit(ranges, truth=label, ignore=spaces):
            return any(short.useful(span, truth, ignore) for span in ranges)

        checked = {"incumbent": (True, old)}
        for arm in ARMS:
            scores = readings[row["id"]][arm]
            source.require(len(scores) == len(spans) - 2, "Mismatched sentence readings")
            top = peak(scores)
            chosen = [spans[top[1]]] if top and top[0] > points[arm]["cutoff"] else []
            checked[arm] = (top is not None, chosen)
            checked[f"plus_{arm}"] = (True, old + chosen)
            if hit(chosen) and not hit(old):
                extra[arm].append((group, row["family"]))
            if label is None and chosen and not old:
                alarms[arm].add(group)
        for arm, (available, ranges) in checked.items():
            counters[f"{scope}/{row['family']}"][arm].update(total=1, assessed=available, flagged=bool(ranges),
                                                              useful=hit(ranges), misplaced=bool(ranges and label and not hit(ranges)))
        mapping[row["id"]] = {arm: ranges for arm, (_, ranges) in checked.items()}
    passes = {arm: len(extra[arm]) >= 4 and len({group for group, _ in extra[arm]}) >= 3 and
              {f for _, f in extra[arm]} == set(interval.FAMILIES) and len(alarms[arm]) <= 2 and
              sum(counters[f"fresh/{f}"][arm]["assessed"] for f in interval.FAMILIES) >= 36 for arm in ARMS}
    return {"metrics": counters, "locations": mapping, "new_hits": {a: sorted(extra[a]) for a in ARMS},
            "new_alarms": {a: sorted(alarms[a]) for a in ARMS}, "passes": passes}


def review(data):
    lock, rows, scores = (data[key] for key in ("freeze", "geometry", "rows"))
    source.require(lock["schema"] == SCHEMA and lock["method"] == METHOD
                   and lock["runner_sha256"] == source.digest(Path(__file__)), "Wrong replay procedure")
    source.require(len({r["id"] for r in rows}) == len(rows) == len(scores), "Missing or duplicate rows")
    points = calibrate(rows, scores)
    return {"points": points, **evaluate(rows, scores, data["primary"], points)}


def publish(run, output):
    lock, rows = locked(run)
    parts = {k: source.read(run / f"{k}.json") for k in ("calibration", "held", "primary", "points")}
    source.require(all(v["freeze"] == source.digest(run / "freeze.json") for v in parts.values()), "Run parts differ")
    data = {"freeze": lock, "geometry": [{k: r[k] for k in GEOMETRY} for r in rows],
            "rows": parts["calibration"]["rows"] | parts["held"]["rows"],
            "primary": parts["primary"]["rows"]}
    result = review(data)
    source.require(result["points"] == parts["points"]["arms"], "Operating points changed after held readings")
    source.seal(output, data | {"result": result})
    print(source.serialize({k: result[k] for k in ("metrics", "passes")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "calibration", "point", "held", "primary", "publish", "review"))
    parser.add_argument("run", type=Path)
    for key in ("cache", "journal-pairs", "publisher-xml", "snapshot", "output"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.run, args.cache, args.journal_pairs, args.publisher_xml)
    elif args.stage in ("calibration", "held"):
        measures(args.run, args.snapshot, args.stage)
    elif args.stage == "point":
        point(args.run)
    elif args.stage == "primary":
        primary(args.run, args.snapshot)
    elif args.stage == "publish":
        publish(args.run, args.output)
    else:
        packet = source.read(args.run)
        source.require(source.serialize(review(packet)) == source.serialize(packet["result"]), "Packet differs")
        print("Independent arithmetic replay matched.")


if __name__ == "__main__":
    main()
