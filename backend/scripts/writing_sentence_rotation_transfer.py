"""Compare the same three sentences with each sentence in the center."""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_context_margin_transfer as prior
from scripts import writing_typed_decision_transfer as typed

source = typed.source
SCHEMA = "chipmunk.writing-sentence-rotation/1"
TRAIN = prior.PRIOR
ARCHIVED_HELD = typed.short.PANELS / "writing-context-margin-transfer-2026-11-13.json"
DEV_BANK = source.ROOT / "backend/data/independent-evaluation/writing-typed-decision-transfer-2026-11-13"
HELD_BANK = source.ROOT / "backend/data/independent-evaluation/writing-context-margin-transfer-2026-11-13"
ARMS = ("native", "rotated")
METHOD = {
    "native": "Read exact left-middle-right text. Each score concerns all three sentences.",
    "rotated": "Read all three cyclic rotations of the same sentences. Keep original middle score minus the mean of the two rotations where each neighbor occupies the middle. These alternate inputs are diagnostic and their responses do not measure independent authorship.",
    "localization": "Select a unique highest interior value strictly above its frozen point. Keep the incumbent three-sentence ranges at its original 0.50 point as the baseline. Report outright wins, wrong locations, and paired gains versus a highest-native-center selector.",
    "calibration": "Use the second-highest genuine work maximum from 20 earlier GAO reports. Do not reuse edits for threshold fitting. Reuse 20 later previously measured report works as transfer and reserve an available unused-work follow-up.",
    "criterion": "At least four new useful locations on three works and both generators with at most two new untouched flags. Requires a separate fresh assessment before serving.",
}
GEOMETRY = ("id", "group", "family", "spans", "target_index", "truth", "spaces", "text_sha256")


def inputs(row, index):
    text = row["text"]
    pieces = [text[a:b] for a, b in row["spans"][index-1:index+2]]
    return [" ".join(pieces[shift:] + pieces[:shift]) for shift in range(3)]


def original_ledger(path):
    recorded = source.read(path)
    return recorded["freeze"]["roster_sha256"], {
        key: {row["id"]: row for row in recorded["geometry"] if row.get("cohort") == key}
        for key in ("fresh", "calibration")
    }


def retained(path, ledger):
    expected, rows = original_ledger(ledger)
    file = path / "roster.json"
    source.require(source.digest(file) == expected, "The native writing roster differs from its published binding")
    returned = [row for row in source.read(file) if row["cohort"] == "fresh"]
    expected = rows["fresh"]
    source.require(len(returned) == len(expected) == 60 and all(
        all(row[key] == expected[row["id"]][key] for key in typed.GEOMETRY) for row in returned
    ), "Historical source geometry differs")
    return returned


def followup(cache, excluded):
    """Select before scoring from the remaining same-position report polishes."""
    return typed.intake.intake(cache, [{"group": group} for group in excluded], target_works=12)


def freeze(run, cache):
    train, held = retained(DEV_BANK, TRAIN), retained(HELD_BANK, ARCHIVED_HELD)
    earlier = sorted(p for p in typed.short.PANELS.glob("writing-*.json")
                     if p.name != "writing-sentence-rotation-transfer-2026-11-13.json")
    excluded = {match for path in earlier for match in re.findall(r'(?<=")[0-9a-f]{64}(?=")', path.read_text())}
    fresh, admission = followup(cache, excluded)
    roles = {"point": train, "transfer": held, "fresh": fresh}
    source.require(all(not ({row["group"] for row in rows} & {row["group"] for row in later})
                       for i, rows in enumerate(roles.values()) for later in list(roles.values())[i+1:]),
                   "A writing work crosses selection roles")
    for rows in roles.values():
        source.require(all(source.sha(row["text"]) == row["text_sha256"] and all(
            inputs(row, i)[0] == row["context"][i] for i in (1, 2, 3)
        ) for row in rows), "The native reading changed before rotation")
    run.mkdir(mode=0o700)
    source.seal(run / "roster.json", roles)
    source.seal(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "runner_sha256": source.digest(Path(__file__)),
        "roster_sha256": source.digest(run / "roster.json"), "train_sha256": source.digest(TRAIN), "transfer_sha256": source.digest(ARCHIVED_HELD),
        "earlier_ledgers": {p.name: source.digest(p) for p in earlier}, "fresh_admission": admission,
        "checkpoint": typed.BASELINE_ASSETS, "revision": typed.BASELINE_REVISION,
        "runtime": source.read(ARCHIVED_HELD)["freeze"]["versions"]})
    print(admission, flush=True)


def locked(run):
    lock = source.read(run / "freeze.json")
    source.require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["runner_sha256"] == source.digest(Path(__file__))
                   and lock["roster_sha256"] == source.digest(run / "roster.json"), "Writing experiment freeze differs")
    return lock, source.read(run / "roster.json")


def measure(run, snapshot, stage):
    import os
    from importlib.metadata import version

    from app.detectors.text.classifier import FourClassTextClassifier

    lock, roles = locked(run)
    if stage != "point":
        fit = source.read(run / "point.json")
        source.require(fit["freeze"] == source.digest(run / "freeze.json")
                       and fit["points"] == points(roles["point"], fit["readings"]), "The selected points changed")
    source.require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
                   and all(version(name) == value for name, value in lock["runtime"].items())
                   and all(source.digest(snapshot / name) == checksum for name, checksum in lock["checkpoint"].items()),
                   "The pinned offline CPU or checkpoint differs")
    classifier = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    rows = roles[stage]
    if stage == "point":
        rows = [row for row in rows if row["family"] == "original"]
    contexts = {value for row in rows for index in (1, 2, 3) for value in inputs(row, index)}
    contexts.update(item for row in rows for item in row["context"])
    ordered = sorted(contexts, key=source.sha)
    observed, single = {}, {}
    for start in range(0, len(ordered), 36):
        batch = ordered[start:start + 36]
        values = classifier.predict(batch, batch=6, check=lambda _: None)
        for text, value, coverage in zip(batch, values, classifier.last_coverage, strict=True):
            tokens = coverage["tokens"]
            source.require(tokens > 0 and tokens == coverage.get("covered_tokens", tokens) and not coverage.get("omitted_tokens", 0),
                           "Incomplete native writing input")
            observed[text] = float(value)
            single[text] = observed[text] if coverage["token_windows"] == 1 else None
        print(stage, len(observed), len(ordered), flush=True)
    observed_rows = {row["id"]: {"incumbent": [observed[text] for text in row["context"]],
                     "native_and_rotated": [[single[text] for text in inputs(row, i)] for i in (1, 2, 3)],
                     "input_sha256": [[source.sha(text) for text in inputs(row, i)] for i in (1, 2, 3)]}
                     for row in rows}
    record = {"freeze": source.digest(run / "freeze.json"), "readings": observed_rows}
    if stage == "point":
        record["points"] = points(roles["point"], observed_rows)
    source.seal(run / f"{stage}.json", record)


def responses(reading, method):
    triplets = reading["native_and_rotated"]
    source.require(len(triplets) == 3 and all(len(t) == 3 and all(v is None or math.isfinite(v) and 0 <= v <= 1 for v in t)
                   for t in triplets), "Invalid native and rotated responses")
    if any(t[0] is None or method == "rotated" and any(v is None for v in t) for t in triplets):
        return None
    if method == "native":
        return [t[0] for t in triplets]
    return [t[0] - (t[1] + t[2]) * .5 for t in triplets]


def points(rows, readings):
    originals = [row for row in rows if row["family"] == "original"]
    source.require(len(originals) == len(readings) == 20, "Exactly twenty untouched calibration works required")
    cutoffs = {}
    for arm in ARMS:
        values = [responses(readings[row["id"]], arm) for row in originals]
        peaks = sorted(max(value) for value in values if value is not None)
        source.require(len(peaks) >= 16, "Insufficient measured untouched works")
        cutoffs[arm] = {"point": peaks[-2], "works": len(peaks)}
    return cutoffs


def evaluate(rows, scores, cutoffs):
    summary, groups = defaultdict(lambda: defaultdict(Counter)), defaultdict(set)
    recovered, extra = {}, defaultdict(set)
    predicted, paired = {}, defaultdict(Counter)
    for row in rows:
        name, family = row["id"], row["family"]
        groups[row["group"]].add(family)
        spans, known, spaces = row["spans"], row["truth"], set(row["spaces"])
        reading = scores[name]
        current = reading["incumbent"]
        source.require(len(current) == len(spans) == 5
                       and all(v is None or math.isfinite(v) and 0 <= v <= 1 for v in current), "Invalid original responses")
        baseline = [[spans[max(0, i-1)][0], spans[min(4, i+1)][1]]
                     for i, value in enumerate(current) if value is not None and value >= .5]
        base_found = any(typed.short.useful(span, known, spaces) for span in baseline)
        summary[family]["incumbent"].update(files=1, assessed=all(v is not None for v in current), flagged=bool(baseline),
                                               useful=base_found, wrong=known is not None and bool(baseline) and not base_found)
        predicted[name] = {}
        for arm in ARMS:
            values = responses(reading, arm)
            winner = None
            if values and values.count(max(values)) == 1 and max(values) > cutoffs[arm]["point"]:
                winner = values.index(max(values))+1
            found = winner is not None and typed.short.useful(spans[winner], known, spaces)
            summary[family][arm].update(files=1, assessed=values is not None, flagged=winner is not None,
                                          useful=found, wrong=known is not None and winner is not None and not found,
                                          additional=found and not base_found, union=found or base_found)
            if found and not base_found:
                recovered.setdefault(arm, set()).add((row["group"], family))
            if family == "original" and winner is not None and not baseline:
                extra[arm].add(row["group"])
            predicted[name][arm] = winner
        native, rotated = (predicted[name][arm] for arm in ARMS)
        if family == "original":
            paired[family].update(rotated_only_flag=native is None and rotated is not None,
                                  native_only_flag=rotated is None and native is not None)
        else:
            native_hit = native is not None and typed.short.useful(spans[native], known, spaces)
            rotated_hit = rotated is not None and typed.short.useful(spans[rotated], known, spaces)
            paired[family].update(rotated_only_useful=rotated_hit and not native_hit,
                                  native_only_useful=native_hit and not rotated_hit)
    source.require(len(rows) == len(scores) and all(roles == {"original", *typed.interval.FAMILIES} for roles in groups.values()),
                   "The complete two-generator matched panel is required")
    result = {}
    for arm in ARMS:
        gains = recovered.get(arm, set())
        passing = (sum(summary[family][arm]["additional"] for family in typed.interval.FAMILIES) >= 4
                   and len({key for key, _ in gains}) >= 3 and {family for _, family in gains} == set(typed.interval.FAMILIES)
                   and len(extra[arm]) <= 2)
        result[arm] = {"passing": passing, "additional_genuine_groups": len(extra[arm]), "gained_works": len({key for key, _ in gains})}
    return {"summary": summary, "decisions": result, "paired": paired, "selected_indices": predicted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "point", "transfer", "fresh", "publish", "review"))
    parser.add_argument("path", type=Path)
    for name in ("cache", "snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.path, args.cache)
    elif args.phase in {"point", "transfer", "fresh"}:
        measure(args.path, args.snapshot, args.phase)
    elif args.phase == "publish":
        lock, roles = locked(args.path)
        scores = {name: source.read(args.path / f"{name}.json") for name in roles}
        source.require(all(row["freeze"] == source.digest(args.path / "freeze.json") for row in scores.values()),
                       "Scores belong to another freeze")
        geometry = {name: [{key: row[key] for key in GEOMETRY} for row in rows] for name, rows in roles.items()}
        packet = {"freeze": lock, "geometry": geometry, "readings": {key: row["readings"] for key, row in scores.items()},
                  "points": scores["point"]["points"]}
        source.seal(args.output, packet | {"result": verify(packet)})
    else:
        packet = source.read(args.path)
        result = verify(packet)
        source.require(result == packet["result"], "Reported results differ")
        print(source.serialize({key: {name: row[name] for name in ("summary", "decisions", "paired")} for key, row in result.items()}), end="")


def verify(packet):
    lock, rows, readings = (packet[k] for k in ("freeze", "geometry", "readings"))
    source.require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["runner_sha256"] == source.digest(Path(__file__))
                   and lock["train_sha256"] == source.digest(TRAIN) and lock["transfer_sha256"] == source.digest(ARCHIVED_HELD),
                   "The frozen comparison differs")
    cutoff = points(rows["point"], readings["point"])
    source.require(packet["points"] == cutoff, "Operating points do not reproduce")
    return {name: evaluate(rows[name], readings[name], cutoff) for name in ("transfer", "fresh")}


if __name__ == "__main__":
    main()
