# SPDX-License-Identifier: MIT
"""Compare transition-penalized word ranges on held mixed-authorship excerpts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import writing_word_anchor_transfer as anchor

adjacent = anchor.adjacent
pool = adjacent.pool
read, save, require = adjacent.read, adjacent.save, adjacent.require
SCHEMA = "chipmunk.writing-structured-range-transfer/1"
REFERENCE = anchor.PANEL.with_name("writing-word-anchor-transfer-2026-11-15.json.gz")
SHARDS = {
    "news": {
        "gemini-2.5-flash": (93083015, "44161adb78c0dfa4cc06d61d1af6a3a2a63ecc1b4f0c41ba0525d1cd0fab69ab"),
        "gpt-5.4": (35629525, "a5cdc0a802abf7ac317c91d26159a2491395260fb112f57e40f5d37539944d90"),
    },
    "reports": {
        "gemini-2.5-flash": (122947488, "e4a52b8cd3a56ac9d748d101f9d1ea75e6ae13b6eb6c38dc95e41f64488c45b5"),
        "gpt-5.4": (52442189, "9e28d157d336a343106487c8acab242ee41829f49ceb748dcb5f9fd6a9f9fb33"),
    },
}
PER_SLOT = 5
ARMS = ("earlier_broad", "one_range", "two_ranges", "two_ranges_with_context")
HEADS = ("original", "anchors_and_local")
POINTS = {"10pct": 10, "25pct": 4}
MIN_WORDS, MAX_WORDS, MAX_GAP = 2, 18, 6
WORD_COST, RUN_COST, CLIP = .8, 1.5, 3.
SLOTS = {tuple(range(start, start + width)) for start, width in adjacent.SLOTS}
METHOD = {
    "source": "Use the pinned official test split rather than the prior development split. Select at most five "
              "matched work pairs per genre, generator and one- or two-sentence interior edit slot. Exclude "
              "known work hashes, exact shared 12-word fingerprints among identifiable test and prior works, "
              "and overlap with the previous fully reconstructed development excerpt roster.",
    "selection": "Keep both existing word heads fixed. Compare the earlier broad proposal, one consecutive "
                 "range, up to two consecutive ranges, and the two-range rule after removing positive "
                 "whole-excerpt median elevation. A run stays within one sentence; a second run may occupy "
                 "the same or the next interior sentence. Each selected word and each run has a fixed cost. "
                 "Use only the submitted excerpt to compute its proposals.",
    "calibration": "Use the previous 36 original news and 36 original report excerpts. Center every word "
                   "score by its ordinary genre median and scale by the positive 90th-percentile gap "
                   "with a .05 floor. Clip standardized word values at three. Freeze the tenth and quarter "
                   "points of leave-one-work-out ordinary maxima. Use strict greater-than decisions.",
    "outcome": "Record useful changed-word overlap (half the change and .30 intersection over union), "
               "exact touched sentences, paired same-sentence ordinary collisions, and missing coverage. "
               "Also replay the earlier journal revisions at the news/report points used for the retained bank.",
}


def reference():
    ref = read(REFERENCE)
    anchor.check_sources(ref)
    return ref


def contract():
    return {"schema": SCHEMA, "method": METHOD, "code_sha256": pool.file_hash(Path(__file__)),
            "reference_sha256": pool.file_hash(REFERENCE), "source_revision": adjacent.SOURCE_REVISION,
            "split": "test", "shards": {genre: {family: list(pin) for family, pin in families.items()}
                                       for genre, families in SHARDS.items()},
            "quota_per_slot": PER_SLOT, "heads": list(HEADS), "arms": list(ARMS),
            "range_geometry": [MIN_WORDS, MAX_WORDS, MAX_GAP], "costs": [WORD_COST, RUN_COST, CLIP]}


def load_shard(path, genre, family, pin):
    size, digest = pin
    require(path.stat().st_size == size and pool.file_hash(path) == digest, "Publisher test bytes differ")
    csv.field_size_limit(1024 * 1024)
    works = defaultdict(dict)
    fields = ("text", "text_tagged", "sentences", "sentence_labels", "ai_spans_char")
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, strict=True):
            require((row["domain"], row["generator"], row["split"], row["subset"]) == (genre, family, "test", "main"),
                    "Publisher test partition metadata differs")
            version, work = row["version"], row["id"]
            if version == "v0" or version in adjacent.STAGES:
                require(version not in works[work], "Repeated publisher work version")
                works[work][version] = {field: row[field] for field in fields}
    return works


def source_rows(folder, retained, exposures):
    known, bindings = adjacent.context.abstract_exclusions(exposures)
    rep = reference()
    require(list(map(adjacent.public, retained["rows"])) == [r for r in rep["rows"] if r["family"] != "editorial"],
            "Reconstructed development excerpts differ from their published word scores")
    known.update(row["group"] for row in rep["rows"])
    known.update(rep["fit"]["groups"])
    fingerprints = adjacent.intake.fingerprints
    used = {mark for row in retained["rows"] for mark in fingerprints(row["text"])}
    admitted, rows, admission = set(), [], {}
    tested = {genre: {family: load_shard(folder / f"{genre}_{family}.csv", genre, family, pin)
                      for family, pin in families.items()} for genre, families in SHARDS.items()}
    used.update(mark for tables in tested.values() for table in tables.values() for key, versions in table.items()
                if pool.sha(key) in known and "v0" in versions for mark in fingerprints(versions["v0"]["text"]))
    for genre, tables in tested.items():
        for family, table in tables.items():
            counts = Counter()
            for key in sorted(table, key=lambda value: pool.sha(f"{SCHEMA}/{genre}/{family}/{value}")):
                group, versions = pool.sha(key), table[key]
                if group in known or "v0" not in versions:
                    continue
                marks = set(fingerprints(versions["v0"]["text"]))
                if not marks or marks & used or marks & admitted:
                    continue
                permitted = {slot for slot in SLOTS if counts[slot] < PER_SLOT}
                try:
                    pair = next(adjacent.examples(versions, permitted, calibrating=False), None)
                except ValueError:
                    continue
                if pair is None:
                    continue
                targets, stage, original, revised = pair
                common = {"group": group, "genre": genre, "family": family, "stage": stage, "target": list(targets)}
                rows.extend((common | original | {"delivery": "original", "changed": []},
                             common | revised | {"delivery": "edited"}))
                counts[targets] += 1
                admitted.update(marks)
                known.add(group)
                if all(counts[slot] == PER_SLOT for slot in SLOTS):
                    break
            admission[f"{genre}/{family}"] = {"/".join(map(str, slot)): counts[slot] for slot in sorted(SLOTS)}
    return rows, bindings, admission


def ordinary_profile(rows, readings, head):
    values = sorted(score for row in rows for index, score in enumerate(readings[row["sha256"]][head])
                    if 0 < row["words"][index][2] < len(row["spans"]) - 1)
    require(values, "No assessed calibration words")
    middle = median(values)
    return middle, max(.05, values[math.ceil(len(values)*.9)-1] - middle)


def rank(item):
    score, bounds = item
    return -score, sum(b-a for a, b in bounds), bounds


def ranges(sentence, terms):
    """Get the exact best one- and up-to-two-run path under the frozen costs."""
    starts, stops = {}, {}
    for first, current in enumerate(sentence):
        if current is None:
            continue
        score = -RUN_COST
        for last in range(first, min(len(terms), first+MAX_WORDS)):
            if sentence[last] != current:
                break
            score += terms[last]
            if last-first+1 < MIN_WORDS:
                continue
            item = (round(score, 7), ((first, last+1),))
            starts[first] = min(starts.get(first, item), item, key=rank)
            stops[last+1] = min(stops.get(last+1, item), item, key=rank)
    single = min(starts.values(), key=rank, default=None)
    together = single
    for end, left in stops.items():
        for start in range(end+1, min(end+MAX_GAP+1, len(terms))):
            if start not in starts or sentence[start] - sentence[end-1] not in (0, 1):
                continue
            right = starts[start]
            joined = (round(left[0]+right[0], 7), left[1]+right[1])
            together = min(together, joined, key=rank)
    return single, together


def proposals(row, scores, profile):
    if not scores:
        return dict.fromkeys(ARMS)
    words, span_count = row["words"], len(row["spans"])
    sentence = [s if 0 < s < span_count-1 else None for _, _, s in words]
    center, scale = profile
    signal = [min(CLIP, max(-CLIP, (value-center)/scale)) for value in scores]
    context = max(0., median(signal[i] for i, s in enumerate(sentence) if s is not None))
    one, two = ranges(sentence, [value-WORD_COST for value in signal])
    _, local = ranges(sentence, [value-WORD_COST-context for value in signal])
    broad = anchor.broad(words, scores, span_count)
    result = {"earlier_broad": None if broad is None else (broad["score"], broad["indices"])}
    for arm, selection in zip(ARMS[1:], (one, two, local), strict=True):
        result[arm] = (selection[0], [i for a, b in selection[1] for i in range(a, b)]) if selection else None
    return result


def calibrate(ref):
    readings = ref["readings"]
    calibration = {}
    for genre in SHARDS:
        rows = [row for row in ref["rows"] if row["genre"] == genre and row["family"] == "calibration"]
        require(len(rows) == 36 and all(row["delivery"] == "original" for row in rows), "Ordinary calibration changed")
        for head in HEADS:
            values = defaultdict(list)
            for row in rows:
                profile = ordinary_profile([case for case in rows if case != row], readings, head)
                for arm, item in proposals(row, readings[row["sha256"]][head], profile).items():
                    if item is not None:
                        values[arm].append(item[0])
            points = {}
            for arm in ARMS:
                ranked = sorted(values[arm], reverse=True)
                require(len(ranked) >= 24, "Insufficient scored ordinary work groups")
                points[arm] = {key: ranked[len(ranked)//divisor] for key, divisor in POINTS.items()}
            calibration[f"{genre}/{head}"] = {
                "profile": list(ordinary_profile(rows, readings, head)), "points": points,
                "assessed": {arm: len(values[arm]) for arm in ARMS},
            }
    return calibration


def report(rows, readings, calibration):
    output = {}
    candidates = {
        head: {row["sha256"]: proposals(row, readings[row["sha256"]][head],
                                       calibration[f'{row["genre"]}/{head}']["profile"]) for row in rows}
        for head in HEADS
    }
    counterparts = {row["group"]: row for row in rows if row["delivery"] == "original"}
    for head in HEADS:
        for arm in ARMS:
            for level in POINTS:
                counts = defaultdict(Counter)
                for row in rows:
                    genre, family = row["genre"], row["family"]
                    stats = counts[genre, family]
                    key = row["sha256"]
                    point = calibration[f"{genre}/{head}"]["points"][arm][level]
                    baseline_point = calibration[f"{genre}/original"]["points"]["earlier_broad"][level]
                    selected, earlier = candidates[head][key][arm], candidates["original"][key]["earlier_broad"]
                    flag = bool(selected and selected[0] > point)
                    baseline_flag = bool(earlier and earlier[0] > baseline_point)
                    role = "editorial" if family == "editorial" else "rewrite" if row["delivery"] == "edited" else "genuine"
                    stats[f"{role}_files"] += 1
                    stats[f"{role}_unassessed"] += selected is None
                    stats[f"{role}_flags"] += flag
                    if role != "rewrite":
                        stats[f"{role}_flags_added"] += flag and not baseline_flag
                        stats[f"{role}_flags_cleared"] += baseline_flag and not flag
                        continue
                    changed = set(row["changed"])
                    overlap = flag and anchor.old.useful_words(set(selected[1]), changed)
                    baseline_overlap = baseline_flag and anchor.old.useful_words(set(earlier[1]), changed)
                    stats["useful"] += overlap
                    stats["useful_added"] += overlap and not baseline_overlap
                    stats["useful_lost"] += baseline_overlap and not overlap
                    if not flag:
                        continue
                    locations = {row["words"][i][2] for i in selected[1]}
                    stats["exact_sentences"] += locations == set(row["target"])
                    stats["wrong_sentences"] += locations.isdisjoint(row["target"])
                    if overlap:
                        other = counterparts[row["group"]]
                        match = candidates[head][other["sha256"]][arm]
                        collision = bool(match and match[0] > point and
                                         locations & {other["words"][i][2] for i in match[1]})
                        stats["useful_with_same_sentence_ordinary_flag"] += collision
                output[f"{head}/{arm}/{level}"] = {"/".join(group): dict(values) for group, values in counts.items()}
    return output


def verify_holdout(rows, readings, ref, freeze):
    groups = defaultdict(list)
    seen = set()
    known, _ = adjacent.context.abstract_exclusions(freeze["exposures"])
    known.update(row["group"] for row in ref["rows"])
    known.update(ref["fit"]["groups"])
    for row in rows:
        group, key = row["group"], row["sha256"]
        require(group not in known and key not in seen and row["genre"] in SHARDS
                and row["family"] in SHARDS[row["genre"]], "Known or repeated held source")
        seen.add(key)
        groups[group].append(row)
        words, spans, targets = row["words"], row["spans"], row["target"]
        require(len(spans) == 6 and tuple(targets) in SLOTS, "Source sentence ranges differ")
        require(all(0 <= s < len(spans) and spans[s][0] <= a < b <= spans[s][1] for a, b, s in words)
                and all(b <= c for (_, b, _), (c, _, _) in zip(words, words[1:]))
                and all(0 <= i < len(words) and words[i][2] in targets for i in row["changed"]),
                "Invalid word or change coordinates")
        values = readings[key]
        require(set(values) == set(HEADS) and len({len(a) for a in values.values()}) == 1 and
                all(not a or len(a) == len(words) and all(map(math.isfinite, a)) for a in values.values()),
                "Invalid held word-score geometry")
    require(seen == set(readings), "Unexpected held data")
    counts = Counter()
    for original, edited in groups.values():
        keys = ("genre", "family", "stage", "target")
        require((original["delivery"], edited["delivery"]) == ("original", "edited")
                and all(original[k] == edited[k] for k in keys) and not original["changed"] and edited["changed"],
                "A held work pair is incomplete")
        slot = "/".join(map(str, edited["target"]))
        counts["/".join((edited["genre"], edited["family"], slot))] += 1
    require(set(freeze["admission"]) == {f"{genre}/{family}" for genre, families in SHARDS.items() for family in families}
            and all(set(slots) == {"/".join(map(str, slot)) for slot in SLOTS} for slots in freeze["admission"].values())
            and all(counts[f"{key}/{slot}"] == count <= PER_SLOT for key, slots in freeze["admission"].items()
                for slot, count in slots.items()) and sum(counts.values()) == len(rows)//2, "Source quota differs")


def review(packet):
    ref = reference()
    freeze = packet["freeze"]
    require(freeze["contract"] == contract() and freeze["calibration"] == calibrate(ref), "Frozen procedure differs")
    rows, readings = packet["rows"], packet["readings"]
    verify_holdout(rows, readings, ref, freeze)
    retained = [row for row in ref["rows"] if row["family"] != "calibration"]
    results = {"official_test": report(rows, readings, freeze["calibration"]),
               "retained_development_and_journals": report(retained, ref["readings"], freeze["calibration"])}
    require(packet.get("results", results) == results, "Stored outcome differs")
    return results


def capture(plan, encoder, checkpoint, fit_dir):
    from safetensors.torch import load_file
    from torch.nn import Linear

    ref = reference()
    data = read(plan)
    freeze = data["freeze"]
    require(freeze["contract"] == contract() and freeze["calibration"] == calibrate(ref), "Frozen procedure differs")
    rows = data["rows"]
    require(all(pool.sha(row["text"]) == row["sha256"] for row in rows), "Frozen excerpt bytes changed")
    tokenizer, runtime, baseline = anchor.native_runtime(encoder, checkpoint)
    path = fit_dir / "anchors_and_local.safetensors"
    require(pool.file_hash(path) == ref["fit"]["heads"]["anchors_and_local"]["weights_sha256"], "Second head differs")
    second = Linear(baseline.in_features, 1)
    second.load_state_dict(load_file(path), strict=True)
    readings = anchor.native_rows(rows, tokenizer, runtime, {"original": baseline, "anchors_and_local": second.eval()})
    packet = {"freeze": freeze, "plan_sha256": pool.file_hash(plan), "rows": list(map(adjacent.public, rows)),
              "readings": readings}
    packet["results"] = review(packet)
    return packet


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    stages = cli.add_subparsers(dest="phase", required=True)
    for phase, fields in (("freeze", ("shards", "retained", "output")),
                          ("verify-source", ("shards", "retained", "plan")),
                          ("capture", ("plan", "encoder", "checkpoint", "fit_dir", "output")),
                          ("review", ("receipt",))):
        sub = stages.add_parser(phase)
        for name in fields:
            sub.add_argument(name, type=Path)
        if phase == "freeze":
            sub.add_argument("--reference", type=Path, help="Use the exact earlier receipt inventory.")
    args = cli.parse_args()
    if args.phase in {"freeze", "verify-source"}:
        before = read(args.plan) if args.phase == "verify-source" else None
        reference_packet = before or (read(args.reference) if args.reference else None)
        inventory = reference_packet["freeze"]["exposures"] if reference_packet else None
        rows, exposures, admission = source_rows(args.shards, read(args.retained), inventory)
        freeze = {"contract": contract(), "calibration": calibrate(reference()), "exposures": exposures,
                  "retained_plan_sha256": pool.file_hash(args.retained), "admission": admission}
        plan = {"freeze": freeze, "rows": rows}
        if before is None:
            require(not args.output.exists(), "Source plan already exists")
            save(args.output, plan)
        else:
            require(plan == before, "Sealed held source replay differs")
        print(f"{len(rows)//2} held work pairs. {admission}", flush=True)
    elif args.phase == "capture":
        require(not args.output.exists(), "Held receipt already exists")
        packet = capture(args.plan, args.encoder, args.checkpoint, args.fit_dir)
        save(args.output, packet)
        print(f"{len(packet['rows'])} native readings captured", flush=True)
    else:
        print(json.dumps(review(read(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
