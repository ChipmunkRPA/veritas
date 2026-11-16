# SPDX-License-Identifier: MIT
"""Test comparable false-alarm points for combining writing-range proposals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import writing_missed_passage_transfer as ratio
from scripts import writing_structured_range_transfer as ranges

anchor, adjacent, pool = ranges.anchor, ranges.adjacent, ranges.pool
read, save, require = ranges.read, ranges.save, ranges.require
PUBLISHED_TEST = ranges.REFERENCE.with_name("writing-structured-range-transfer-2026-11-15.json.gz")
SCHEMA = "chipmunk.writing-ratio-fusion-transfer/1"
SHARDS = {
    "news": {
        "gpt-5.4-nano": ("train", 210015836, "1476d26ac5891e6b0797ef5cec1a51a6ab6c70931b6edb238eb53ca8f8805d11"),
        "gemini-2.5-flash": ("test", *ranges.SHARDS["news"]["gemini-2.5-flash"]),
        "qwen3-8b": ("test", 6035138, "e9b830b2e376ac017c944b67bf5620deebf7d9aeec5edecc3734517de2f16a21"),
    },
    "reports": {
        "gpt-5.4-nano": ("train", 254277559, "c0b0189df99b6be49f4917dd3db32e268deddd366a641c337f3a0e07789f2d9e"),
        "gemini-2.5-flash": ("test", *ranges.SHARDS["reports"]["gemini-2.5-flash"]),
        "qwen3-8b": ("test", 5919856, "23e2b29270910b2a54d1ae2859e9e2b1e89cee1dd179f16d2ea2b39c7e60af25"),
    },
}
PER_SLOT = 4
STAGES = tuple(f"v{i}" for i in range(1, 9))
ARMS = ("broad", "two_runs", "ratio", "relative_ratio", "either_word", "either_all", "word_with_ratio")
COMPONENTS = ARMS[:4]
METHOD = {
    "inputs": "One native six-sentence paragraph per file. Read GPT-5.4 Nano from the publisher's train partition "
              "and Gemini and Qwen from its test partition. Exclude the frozen encoder's known fit groups, "
              "work hashes from prior receipts and word-sequence overlap with earlier reconstructed plans, "
              "identifiable exposed works or other selected originals. Use unique unchanged first and last "
              "sentences to link the six-sentence excerpts, even if the rest of the document changed. Both native "
              "paragraphs must preserve every untagged sentence and have only one or two adjacent labeled "
              "interior sentences changed. Allow up to four work pairs per interior edit slot and family.",
    "ratio": "Use the two pinned local causal checkpoints at the established Apple GPU precision. "
             "A sentence must have at least 12 tokens and complete non-space character coverage. "
             "Use the negative observed-to-cross loss ratio and its margin over the other interior sentences. "
             "Each ratio pick displays the top 60% of words from the existing word head on the picked sentence.",
    "word": "Use the unchanged original frozen word head, the broad highlight and the one-or-two-run selector. "
            "Use the published word profiles for the existing two-run decoder.",
    "combined": "Take the strongest standardized candidate among the two word checks, or among all four "
                "components. The anchored blend instead adjusts each broad sentence score by 0.25 times "
                "the relative ratio on the same sentence, limited to plus or minus two standardized units.",
    "calibration": "Use 36 prior ordinary works per genre. Normalize each component by the median and "
                   "positive 90th-percentile gap of ordinary file maxima with a .05 word or .02 ratio floor. "
                   "Set points on leave-one-work-out ordinary maxima at the previous tenth and quarter "
                   "budgets. Require strict score excess and keep the raw readings separate.",
    "evaluation": "Assess useful changed-word overlap, touched sentence positions, missing coverage, flags on genuine "
                  "inputs and matched originals, and paired gains and losses. No signals reach the app.",
}


def contract():
    return {"schema": SCHEMA, "method": METHOD, "code_sha256": pool.file_hash(Path(__file__)),
            "word_parent": pool.file_hash(PUBLISHED_TEST), "ratio_code": pool.file_hash(Path(ratio.__file__)),
            "ratio_models": [list(item) for item in ratio.MODELS], "ratio_tokenizer": ratio.TOKENIZER,
            "publisher_revision": adjacent.SOURCE_REVISION, "quota": PER_SLOT,
            "shards": {genre: {family: list(pin) for family, pin in families.items()} for genre, families in SHARDS.items()}}


def load_shard(folder, genre, family, pin):
    split, size, digest = pin
    path = folder / split / f"{genre}_{family}.csv"
    require(path.stat().st_size == size and pool.file_hash(path) == digest, "Publisher file bytes differ")
    csv.field_size_limit(1024 * 1024)
    fields = ("text", "text_tagged", "sentences", "sentence_labels", "ai_spans_char")
    table = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, strict=True):
            require((row["domain"], row["generator"], row["split"], row["subset"]) == (genre, family, split, "main"),
                    "Publisher partition metadata differs")
            version, key = row["version"], row["id"]
            require(version in ("v0", *STAGES) and version not in table[key], "Repeated or unexpected publisher version")
            table[key][version] = {field: row[field] for field in fields}
    return table


def local_examples(versions, allowed):
    original, before, before_spans, original_labels = adjacent.positions(versions["v0"])
    require(not any(original_labels), "An original contains a publisher edit flag")
    unique_before = Counter(before)
    for stage in STAGES:
        if stage not in versions:
            continue
        try:
            revised, after, after_spans, labels = adjacent.positions(versions[stage])
        except ValueError:
            continue
        unique_after = Counter(after)
        at = {text: i for i, text in enumerate(after) if unique_after[text] == 1}
        for start in range(len(before)-5):
            first, last = before[start], before[start+5]
            if unique_before[first] != 1 or unique_before[last] != 1 or first not in at or last not in at:
                continue
            current = at[first]
            if at[last] != current+5:
                continue
            flags = labels[current:current+6]
            goals = tuple(i for i, flag in enumerate(flags) if flag)
            if goals not in allowed or any((old == new) == bool(flag) for old, new, flag in zip(
                    before[start:start+6], after[current:current+6], flags, strict=True)):
                continue
            left = adjacent.native_slice(original, before_spans[start:start+6])
            right = adjacent.native_slice(revised, after_spans[current:current+6])
            if left and right:
                changed = set().union(*(adjacent.words.edit_words(left, right | {"position": i}) for i in goals))
                tokens = adjacent.words.items(right)
                word_slots = {tokens[i][2] for i in changed}
                if word_slots == set(goals):
                    yield goals, stage, left, right | {"changed": sorted(changed)}


def qualified_sources(folder, dev_path, prior_path, exposures=None):
    development, prior = read(dev_path), read(prior_path)
    tested = read(PUBLISHED_TEST)
    ref = ranges.reference()
    require(pool.file_hash(prior_path) == tested["plan_sha256"] and list(map(adjacent.public, prior["rows"])) == tested["rows"],
            "Earlier official test inputs differ")
    require(list(map(adjacent.public, development["rows"])) == [r for r in ref["rows"] if r["family"] != "editorial"],
            "Earlier development inputs differ")
    known, inventory = adjacent.context.abstract_exclusions(exposures)
    known.update(row["group"] for plan in (development, prior) for row in plan["rows"])
    known.update(ref["fit"]["groups"])
    fingerprints = adjacent.intake.fingerprints
    used = {mark for plan in (development, prior) for row in plan["rows"] for mark in fingerprints(row["text"])}
    tables = {(genre, family): load_shard(folder, genre, family, pin)
              for genre, families in SHARDS.items() for family, pin in families.items()}
    used.update(mark for table in tables.values() for key, versions in table.items()
                if pool.sha(key) in known and "v0" in versions for mark in fingerprints(versions["v0"]["text"]))
    admitted, cases, quota = set(), [], {}
    for (genre, family), table in tables.items():
        counts = Counter()
        for key in sorted(table, key=lambda value: pool.sha(f"{SCHEMA}/{genre}/{family}/{value}")):
            group, versions = pool.sha(key), table[key]
            if group in known or "v0" not in versions:
                continue
            marks = set(fingerprints(versions["v0"]["text"]))
            if not marks or marks & used or marks & admitted:
                continue
            allowed = {slot for slot in ranges.SLOTS if counts[slot] < PER_SLOT}
            try:
                candidate = next(local_examples(versions, allowed), None)
            except ValueError:
                continue
            if candidate is None:
                continue
            slot, stage, original, edited = candidate
            labels = {"group": group, "genre": genre, "family": family, "stage": stage, "target": list(slot)}
            cases.extend((labels | original | {"delivery": "original", "changed": []},
                          labels | edited | {"delivery": "edited"}))
            known.add(group)
            admitted.update(marks)
            counts[slot] += 1
            if all(counts[slot] == PER_SLOT for slot in ranges.SLOTS):
                break
        quota[f"{genre}/{family}"] = {"/".join(map(str, slot)): counts[slot] for slot in sorted(ranges.SLOTS)}
    calibration = [row for row in development["rows"] if row["family"] == "calibration"]
    return {"freeze": {"contract": contract(), "exposures": inventory, "quota": quota,
                       "development_sha256": pool.file_hash(dev_path), "earlier_test_sha256": pool.file_hash(prior_path)},
            "calibration": calibration, "held": cases}


def ordinary_words():
    ref = ranges.reference()
    rows = [row for row in ref["rows"] if row["family"] == "calibration"]
    return rows, {row["sha256"]: ref["readings"][row["sha256"]]["original"] for row in rows}


def sentence_proposals(row, word, readings, profile):
    words = row["words"]
    interior = range(1, len(row["spans"])-1)
    broad = {}
    for sentence in interior:
        ranked = sorted((i for i, (_, _, s) in enumerate(words) if s == sentence),
                        key=lambda i: (-word[i], i)) if word else []
        if len(ranked) >= 6:
            broad[sentence] = (round(sum(word[i] for i in ranked[:6]) / 6, 7),
                               tuple(sorted(ranked[:math.ceil(.60*len(ranked))])))
    split = ranges.proposals(row, word, profile)["two_ranges"]
    signed = {i: -value for i, value in enumerate(ratio.ratio_values({"ratio": readings}))
              if i in interior and value is not None}
    relative = {}
    for sentence, value in signed.items():
        others = [other for i, other in signed.items() if i != sentence]
        if len(others) >= 2:
            relative[sentence] = value - median(others)
    return {"broad": broad, "two_runs": {None: (split[0], tuple(split[1]))} if split else {},
            "ratio": {i: (value, broad[i][1]) for i, value in signed.items() if i in broad},
            "relative_ratio": {i: (value, broad[i][1]) for i, value in relative.items() if i in broad}}


def profile(rows, all_proposals):
    profile = {}
    for arm in COMPONENTS:
        values = sorted(max(item[0] for item in all_proposals[key][arm].values()) for key in rows
                        if all_proposals[key][arm])
        require(len(values) >= 24, f"Ordinary scores lack complete {arm} coverage")
        middle, high = median(values), values[math.ceil(.9*len(values))-1]
        profile[arm] = [middle, max(.02 if "ratio" in arm else .05, high-middle)]
    return profile


def pick(options):
    if not options:
        return None
    ordered = sorted(set(options), reverse=True)
    return ordered[0] if len(ordered) == 1 or ordered[0][0] != ordered[1][0] else None


def arms(proposals, scales):
    normalized = {arm: {i: (round((score-scales[arm][0])/scales[arm][1], 7), words)
                        for i, (score, words) in proposals[arm].items()}
                  for arm in COMPONENTS}
    result = {arm: pick(list(values.values())) for arm, values in normalized.items()}
    result["either_word"] = pick([entry for arm in COMPONENTS[:2] if (entry := result[arm])])
    result["either_all"] = pick([entry for arm in COMPONENTS if (entry := result[arm])])
    relative = normalized["relative_ratio"]
    result["word_with_ratio"] = pick([
        (round(value + .25*max(-2., min(2., relative[i][0])), 7), indices)
        for i, (value, indices) in normalized["broad"].items() if i in relative
    ])
    return result


def proposals(rows, word, traces):
    locked = read(PUBLISHED_TEST)["freeze"]["calibration"]
    return {row["sha256"]: sentence_proposals(row, word[row["sha256"]], traces[row["sha256"]],
                                             locked[f'{row["genre"]}/original']["profile"]) for row in rows}


def calibrate(rows, word, traces):
    readings = proposals(rows, word, traces)
    result = {}
    for genre in SHARDS:
        sources = [row["sha256"] for row in rows if row["genre"] == genre]
        require(len(sources) == 36, "Unexpected calibration source count")
        values = defaultdict(list)
        for source in sources:
            scales = profile([other for other in sources if other != source], readings)
            for name, option in arms(readings[source], scales).items():
                if option:
                    values[name].append(option[0])
        points = {}
        for name in ARMS:
            ranked = sorted(values[name], reverse=True)
            require(len(ranked) >= 24, "Insufficient leave-one-out reference scores")
            points[name] = {level: ranked[len(ranked)//divisor] for level, divisor in ranges.POINTS.items()}
        result[genre] = {"profiles": profile(sources, readings), "points": points,
                         "work_counts": {name: len(values[name]) for name in ARMS}}
    return result


def evaluate(rows, word, traces, calibration):
    prep = proposals(rows, word, traces)
    choices = {row["sha256"]: arms(prep[row["sha256"]], calibration[row["genre"]]["profiles"]) for row in rows}
    pairs = {row["group"]: row for row in rows if row["delivery"] == "original"}
    output = {}
    decisions = {}
    for arm in ARMS:
        for level in ranges.POINTS:
            counts = defaultdict(Counter)
            for row in rows:
                stats = counts[f'{row["genre"]}/{row["family"]}']
                key, genre = row["sha256"], row["genre"]
                point = calibration[genre]["points"][arm][level]
                baseline_point = calibration[genre]["points"]["broad"][level]
                chosen, earlier = choices[key][arm], choices[key]["broad"]
                value, indices = chosen or (None, ())
                alert = chosen is not None and value > point
                old_alert = earlier is not None and earlier[0] > baseline_point
                target = set(row["changed"])
                sentences = {row["words"][i][2] for i in indices}
                decisions[f"{key}/{arm}/{level}"] = [value, list(indices) if alert else [], alert]
                if row["delivery"] == "original":
                    stats.update(ordinary=1, false_alarm=alert, ordinary_unassessed=chosen is None,
                                 added_ordinary=alert and not old_alert, cleared_ordinary=old_alert and not alert)
                    continue
                useful = alert and anchor.old.useful_words(set(indices), target)
                old_useful = old_alert and anchor.old.useful_words(set(earlier[1]), target)
                stats.update(edited=1, edited_unassessed=chosen is None, edited_alert=alert, useful=useful,
                             exact_sentences=alert and sentences == set(row["target"]), added=useful and not old_useful,
                             lost=old_useful and not useful, wrong=alert and sentences.isdisjoint(row["target"]))
                if useful:
                    match = pairs[row["group"]]
                    ordinary = choices[match["sha256"]][arm]
                    stats["matched_collision"] += bool(ordinary and ordinary[0] > point and
                                                       sentences & {match["words"][i][2] for i in ordinary[1]})
            output[f"{arm}/{level}"] = {name: dict(value) for name, value in counts.items()}
    return output, decisions


def verify_readings(rows, words, traces):
    require(len(rows) == len(words) and {row["sha256"] for row in rows} == set(traces) == set(words),
            "Writing readouts lack source coverage")
    for row in rows:
        key = row["sha256"]
        require(len(row["spans"]) == len(traces[key]) == 6 and
                (not words[key] or len(words[key]) == len(row["words"]) and all(map(math.isfinite, words[key]))),
                "The complete token and word geometry differs")
        words_in_order = row["words"]
        require(all(b <= c for (_, b, _), (c, _, _) in zip(words_in_order, words_in_order[1:])),
                "Word ranges overlap or change order")
        for a, b, sentence in words_in_order:
            require(0 <= sentence < 6 and row["spans"][sentence][0] <= a < b <= row["spans"][sentence][1],
                    "A word is outside its source sentence")
        require(all(0 <= i < len(row["words"]) and row["words"][i][2] in row["target"] for i in row["changed"]),
                "The edited-word truth falls outside its sentence")
        for response in traces[key]:
            if response is not None:
                values = float.fromhex(response["obs_hex"]), float.fromhex(response["cross_hex"])
                require(all(math.isfinite(value) for value in values) and values[0] >= 0 and values[1] > 0
                        and response["tokens"] >= 12 and len(response["offset_sha256"]) == 64,
                        "Invalid loss or exact coverage")


def review(packet):
    freeze = packet["freeze"]
    require(freeze["contract"] == contract(), "Frozen study differs")
    clean, clean_word = ordinary_words()
    verify_readings(clean, clean_word, packet["calibration_ratio"])
    lock = calibrate(clean, clean_word, packet["calibration_ratio"])
    require(lock == packet["calibration"], "Ordinary points differ from their frozen readings")
    rows, word, traces = (packet[key] for key in ("rows", "word", "ratio"))
    verify_readings(rows, word, traces)
    known, _ = adjacent.context.abstract_exclusions(freeze["exposures"])
    pairs = defaultdict(list)
    for row in rows:
        require(row["group"] not in known and tuple(row["target"]) in ranges.SLOTS and
                row["family"] in SHARDS.get(row["genre"], {}), "A known or unsupported work entered the holdout")
        pairs[row["group"]].append(row)
    counts = Counter()
    for before, after in pairs.values():
        require((before["delivery"], after["delivery"]) == ("original", "edited") and not before["changed"] and after["changed"]
                and all(before[field] == after[field] for field in ("family", "genre", "stage", "target")),
                "A retained source pair is incomplete")
        counts[f'{after["genre"]}/{after["family"]}', "/".join(map(str, after["target"]))] += 1
    quotas = freeze["quota"]
    require(set(quotas) == {f"{genre}/{family}" for genre, families in SHARDS.items() for family in families}
            and all(set(by_slot) == {"/".join(map(str, slot)) for slot in ranges.SLOTS} for by_slot in quotas.values())
            and all(counts[group, slot] == n <= PER_SLOT for group, by_slot in quotas.items()
                    for slot, n in by_slot.items()), "The source intake quota changed")
    results, decisions = evaluate(rows, word, traces, lock)
    require(packet.get("results", results) == results and packet.get("decisions", decisions) == decisions,
            "The held outcomes differ")
    return results, decisions


def native_word(plan, encoder, checkpoint):
    tokenizer, runtime, head = anchor.native_runtime(encoder, checkpoint)
    results = anchor.native_rows(plan["held"], tokenizer, runtime, {"original": head})
    return {key: value["original"] for key, value in results.items()}


def native_ratio(plan_path, word_path, output):
    plan, prior = read(plan_path), read(word_path)
    require(plan["freeze"]["contract"] == contract() and prior["plan_sha256"] == pool.file_hash(plan_path),
            "Native readouts use different frozen sources")
    runtime, other = ratio.load_pair()

    def measure(rows):
        values = {}
        for i, row in enumerate(rows, 1):
            require(pool.sha(row["text"]) == row["sha256"], "Native excerpt bytes differ")
            values[row["sha256"]] = ratio.reduce_tokens(runtime, other, row["text"], row["spans"])
            if i % 16 == 0 or i == len(rows):
                print(f"Measured {i}/{len(rows)} exact-ratio excerpts", flush=True)
        return values

    clean, clean_word = ordinary_words()
    require(list(map(adjacent.public, plan["calibration"])) == clean, "Ordinary source coordinates differ")
    ordinary = measure(plan["calibration"])
    points = calibrate(clean, clean_word, ordinary)
    output.mkdir(mode=0o700)
    save(output / "calibration.json", {"ratio": ordinary, "points": points})
    print("Sealed ordinary points before the held ratio inference", flush=True)
    trace = measure(plan["held"])
    packet = {"freeze": plan["freeze"], "plan_sha256": pool.file_hash(plan_path),
              "calibration_ratio": ordinary, "calibration": points,
              "rows": list(map(adjacent.public, plan["held"])), "word": prior["readings"], "ratio": trace}
    packet["results"], packet["decisions"] = review(packet)
    save(output / "receipt.json.gz", packet)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    phases = cli.add_subparsers(dest="phase", required=True)
    for name, fields in (("freeze", ("shards", "development", "earlier_test", "output")),
                         ("verify-source", ("shards", "development", "earlier_test", "plan")),
                         ("word", ("plan", "encoder", "checkpoint", "output")),
                         ("ratio", ("plan", "word", "output")), ("review", ("receipt",))):
        phase = phases.add_parser(name)
        for field in fields:
            phase.add_argument(field, type=Path)
        if name == "freeze":
            phase.add_argument("--reference", type=Path)
    args = cli.parse_args()
    if args.phase in ("freeze", "verify-source"):
        existing = read(args.plan) if args.phase == "verify-source" else None
        input_receipt = existing or (read(args.reference) if args.reference else None)
        plan = qualified_sources(args.shards, args.development, args.earlier_test,
                                 input_receipt["freeze"]["exposures"] if input_receipt else None)
        if existing is None:
            require(not args.output.exists(), "Source plan already exists")
            save(args.output, plan)
        else:
            require(plan == existing, "The native held source plan does not replay")
        print(f'{len(plan["held"])//2} held work pairs. {plan["freeze"]["quota"]}', flush=True)
    elif args.phase == "word":
        plan = read(args.plan)
        require(plan["freeze"]["contract"] == contract() and not args.output.exists(), "Word stage source or output differs")
        scores = native_word(plan, args.encoder, args.checkpoint)
        save(args.output, {"plan_sha256": pool.file_hash(args.plan), "readings": scores})
        print(f"Measured {len(scores)} word-score excerpts", flush=True)
    elif args.phase == "ratio":
        native_ratio(args.plan, args.word, args.output)
    else:
        print(json.dumps(review(read(args.receipt))[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
