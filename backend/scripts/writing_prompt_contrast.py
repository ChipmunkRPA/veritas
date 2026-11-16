# SPDX-License-Identifier: MIT
"""Compare two learned likelihood prompts on native sentence ranges."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

SCHEMA = "chipmunk.writing-prompt-contrast/1"
GENRES = ("news", "abstracts", "essays")
FAMILIES = ("gemini-2.5-flash", "gpt-5.4-nano")
ADDITIONAL = {
    "news": ("train", "11d6c9b371c3f43aa25a108c2d557a92adaa2db75b9ae68cc813b2692bd570be"),
    "abstracts": ("dev", "ea88c02630e2bc02ccdf239be81172aec0c1d91956b9088073141c63a83f8191"),
    "essays": ("test", "1e87bd7063aec665bd7f6c5bfc3fcb1ac86507946b2b76dcbbb395a68a7992b8"),
}
REPORT_SHARDS = {
    "gemini-2.5-flash": ("test", "e4a52b8cd3a56ac9d748d101f9d1ea75e6ae13b6eb6c38dc95e41f64488c45b5"),
    "gpt-5.4": ("train", "a205bd7102225997b9e2245589cdf416e42499677b31f76382e28bd1b0b9fa68"),
}
SOURCE_REVISION = "072c03c7051409346aaba05168d183cf6d863ee9"
SHARDS = dict(zip(GENRES, (
    ("44161adb78c0dfa4cc06d61d1af6a3a2a63ecc1b4f0c41ba0525d1cd0fab69ab",
     "8e7c2b3dc843c0665be33e45f9708cdf724c59aff8953d58dc108a4cf1c43a6b"),
    ("2c626b0597f6082908a0183b31c48884a45d310fea8bb1a84043ed2432568b30",
     "9b8873c000ee712f0d1a7bf325cba9ea8245bab5b34b206fbad79968b4af4ea9"),
    ("d4805b719907ea7e3e7215c0cbfd5d47e95f7bad394a51fcdc851d4bc029222e",
     "b1177c4cbda63ccd623dc903037aa8e3ac20b31fcb5e4fe8bcf93c32980f3045"),
), strict=True))
CENTERS = (1, 2, 3, 4)
POSITIONS = tuple((i,) for i in CENTERS)
QUOTAS = {"fit": 24, "calibration": 24, "held": 24}
ARM_NAMES = ("current", "unconditioned", "fixed_prompt", "learned_prompt", "combined")
LEVELS = {"tighter": 10, "wider": 4}
PUBLIC_FIELDS = ("sha256", "group", "genre", "generator", "role", "stage", "target", "label", "spans")
PREFIX_TOKEN_COUNT = 12
BATCH = 3
EPOCHS = 2
SEED = 446713
LEARNING_RATE = .006
HINGE = .25
MAX_INPUT_TOKENS, MAX_FOCUS_TOKENS = 180, 80
FIXED = (
    "An extract from an article written and revised by a human journalist follows.\n",
    "An extract from an article polished and rewritten by an AI assistant follows.\n",
)
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def save(path, payload):
    body = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.write_bytes(gzip.compress(body, mtime=0) if path.suffix == ".gz" else body)


def load_shard(path, genre, family, pin, split="test"):
    require(file_hash(path) == pin, "The publisher shard differs")
    csv.field_size_limit(1024 * 1024)
    table = {}
    columns = ("text", "text_tagged", "sentences", "sentence_labels", "ai_spans_char")
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, strict=True):
            require((row["split"], row["domain"], row["generator"], row["subset"]) == (split, genre, family, "main"),
                    "Unexpected source metadata")
            version = row["version"]
            if version not in ("v0", "v1", "v2", "v3"):
                continue
            variants = table.setdefault(row["id"], {})
            require(version not in variants, "Duplicate source version")
            variants[version] = {column: row[column] for column in columns}
    return table


def valid_slice(raw, selected):
    from app.detectors.text.extract import _normalise, from_string
    from app.detectors.text.segment import segment

    document = from_string(raw[selected[0][0]:selected[-1][1]])
    sentences = segment(document)
    expected = [_normalise(raw[a:b])[0] for a, b in selected]
    if (len(sentences) != 6 or [unit.text for unit in sentences] != expected
            or any(unit.n_words < 8 for unit in sentences)):
        return None
    return {"text": document.text, "sha256": sha(document.text), "spans": [[s.start, s.end] for s in sentences]}


def examples(variants, allowed, ordinary):
    from scripts import writing_adjacent_range_transfer as adjacent

    original, before, spans, initial = adjacent.positions(variants["v0"])
    require(not any(initial), "An ordinary original contains an edit label")
    if ordinary:
        for start in range(len(spans) - 5):
            entry = valid_slice(original, spans[start:start+6])
            if entry:
                yield (), "v0", entry, None
                return
        return
    counts = Counter(before)
    for stage in ("v1", "v2", "v3"):
        if stage not in variants:
            continue
        try:
            edited, after, updated, marks = adjacent.positions(variants[stage])
        except ValueError:
            continue
        counts_after = Counter(after)
        offsets = {value: position for position, value in enumerate(after) if counts_after[value] == 1}
        for start in range(len(spans) - 5):
            lft, rgt = before[start], before[start+5]
            if counts[lft] != 1 or counts[rgt] != 1 or lft not in offsets or offsets.get(rgt) != offsets[lft] + 5:
                continue
            current = offsets[lft]
            changed = tuple(index for index, value in enumerate(marks[current:current+6]) if value)
            if changed not in allowed or any(
                (a == b) != (i not in changed)
                for i, (a, b) in enumerate(zip(before[start:start+6], after[current:current+6], strict=True))
            ):
                continue
            first = valid_slice(original, spans[start:start+6])
            second = valid_slice(edited, updated[current:current+6])
            if first and second and first["sha256"] != second["sha256"]:
                yield changed, stage, first, second


def prepare(root, output, prior_exposures=None, *, fit_plan=None, extra=None, reports_only=False):
    from scripts import writing_adjacent_range_transfer as adjacent
    from scripts.writing_interval_transfer import fingerprints
    from scripts.writing_range_context_transfer import abstract_exclusions

    known, exposures = abstract_exclusions(prior_exposures)
    require(not reports_only or (fit_plan is not None and extra is not None), "Report transfer needs the original fit and shards")
    prior = read(fit_plan) if fit_plan else None
    selected = [row for row in prior["rows"] if row["role"] == "fit"] if prior else []
    if reports_only:
        known.update(row["group"] for row in prior["rows"])
    chosen = {row["group"] for row in selected}
    accepted = {phrase for row in (prior["rows"] if reports_only else selected) for phrase in fingerprints(row["text"])}
    intake = {}
    if prior:
        intake.update({key: value for key, value in prior["intake"].items() if "/fit/" in key})
    for genre in ("reports",) if reports_only else GENRES:
        if genre == "reports":
            tables = {family: load_shard(extra / f"{genre}_{family}.csv", genre, family, pin, split=split)
                      for family, (split, pin) in REPORT_SHARDS.items()}
        else:
            tables = {family: load_shard(root / genre / f"{genre}_{family}.csv", genre, family, pin)
                      for family, pin in zip(FAMILIES, SHARDS[genre], strict=True)}
            if extra is not None:
                split, pin = ADDITIONAL[genre]
                tables["gpt-5.4"] = load_shard(extra / f"{genre}_gpt-5.4.csv", genre, "gpt-5.4", pin, split=split)
        documented = {phrase for table in tables.values() for work, options in table.items()
                      if sha(work) in known | chosen and "v0" in options for phrase in fingerprints(options["v0"]["text"])}
        routes = [("calibration", FAMILIES[0]), *(("held", family) for family in tables)]
        if not fit_plan:
            routes.insert(0, ("fit", FAMILIES[0]))
        for role, family in routes:
            counts, source_counts = Counter(), Counter()
            table = tables[family]
            for work in sorted(table, key=lambda value: sha(f"{SCHEMA}/{genre}/{role}/{family}/{value}")):
                digest, variants = sha(work), table[work]
                if digest in chosen or (role != "calibration" and digest in known) or "v0" not in variants:
                    counts["excluded_work"] += 1
                    continue
                marks = set(fingerprints(variants["v0"]["text"]))
                if not marks or marks & accepted or (role != "calibration" and marks & documented):
                    counts["excluded_phrase_or_empty"] += 1
                    continue
                allowed = {p for p in POSITIONS if source_counts[p] < QUOTAS[role] // len(POSITIONS)}
                try:
                    selections = (adjacent.examples(variants, allowed, calibrating=False) if role == "fit"
                                  else examples(variants, allowed, ordinary=role == "calibration"))
                    example = next(selections, None)
                except ValueError:
                    example = None
                if example is None:
                    counts["ineligible"] += 1
                    continue
                slots, stage, original, rewrite = example
                if role != "calibration" and any(item and item["sha256"] in known for item in (original, rewrite)):
                    counts["excluded_payload"] += 1
                    continue
                record = {"group": digest, "genre": genre, "generator": family, "role": role, "stage": stage,
                          "target": slots[0] if slots else None}
                selected.append(record | original | {"label": "ordinary"})
                if rewrite:
                    selected.append(record | rewrite | {"label": "edited"})
                chosen.add(digest)
                accepted.update(marks)
                source_counts[slots] += 1
                counts["selected"] += 1
                if counts["selected"] == QUOTAS[role]:
                    break
            intake[f"{genre}/{role}/{family}"] = dict(counts)
            print(f"{genre} {role} {family} {counts['selected']} works", flush=True)
    save(output, {"schema": SCHEMA, "publisher_revision": SOURCE_REVISION, "inputs": SHARDS,
                  "additional_inputs": ({"reports": REPORT_SHARDS} if reports_only
                                        else ADDITIONAL if extra is not None else {}),
                  "fit_plan_sha256": file_hash(fit_plan) if fit_plan else None,
                  "exposures": exposures, "intake": intake, "rows": selected})


def native_spans(row):
    require(sha(row["text"]) == row["sha256"] and len(row["spans"]) == 6
            and all(0 <= a < b <= len(row["text"]) for a, b in row["spans"]), "Unbound writing input")
    return row["text"], row["spans"]


def rebuild(receipt, root, extra, output):
    from scripts import writing_adjacent_range_transfer as adjacent

    published = receipt["fit_roster"] + receipt["roster"]
    requests = {}
    for row in published:
        requests.setdefault((row["genre"], row["generator"]), {}).setdefault(row["group"], []).append(row)
    bound = {}
    for (genre, family), groups in requests.items():
        if genre == "reports":
            split, pin = REPORT_SHARDS[family]
            table = load_shard(extra / f"{genre}_{family}.csv", genre, family, pin, split=split)
        elif family == "gpt-5.4":
            split, pin = ADDITIONAL[genre]
            table = load_shard(extra / f"{genre}_{family}.csv", genre, family, pin, split=split)
        else:
            pin = SHARDS[genre][FAMILIES.index(family)]
            table = load_shard(root / genre / f"{genre}_{family}.csv", genre, family, pin)
        for work, variants in table.items():
            digest = sha(work)
            if digest not in groups or "v0" not in variants:
                continue
            expected = {row["label"]: row for row in groups[digest]}
            reference = expected["ordinary"]
            target = reference["target"]
            allowed = {(target,)} if target is not None else set()
            if reference["role"] == "fit":
                candidates = adjacent.examples(variants, allowed, calibrating=False)
            else:
                candidates = examples(variants, allowed, ordinary=target is None)
            for _, stage, first, second in candidates:
                if stage != reference["stage"]:
                    continue
                items = {"ordinary": first, "edited": second} if second is not None else {"ordinary": first}
                if set(expected) == set(items) and all(
                    expected[side]["sha256"] == item["sha256"] and expected[side]["spans"] == item["spans"]
                    for side, item in items.items()
                ):
                    for side, item in items.items():
                        bound[item["sha256"]] = expected[side] | {"text": item["text"]}
                    break
    require(set(bound) == {row["sha256"] for row in published}, "The publisher did not reproduce every source excerpt")
    values = {key: receipt[key] for key in ("schema", "publisher_revision", "inputs", "additional_inputs",
                                           "fit_plan_sha256", "exposures", "intake")}
    save(output, values | {"rows": [bound[row["sha256"]] for row in published]})


def incumbent(plan, snapshot, output):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier
    from scripts.writing_word_edit_transfer import model_assets

    model_assets(snapshot)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    classifier = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    inputs, groups = [], []
    for row in plan["rows"]:
        if row["role"] == "fit":
            continue
        text, bounds = native_spans(row)
        groups.append(row["sha256"])
        inputs.extend(text[bounds[i-1][0]:bounds[i+1][1]] for i in CENTERS)
    unique = list(dict.fromkeys(inputs))
    predictions = classifier.predict(unique, batch=8, check=lambda _: None)
    responses = {text: round(float(number), 8) for text, number in zip(unique, predictions, strict=True)}
    rows = {key: [responses[v] for v in inputs[index:index+len(CENTERS)]]
            for index, key in zip(range(0, len(inputs), len(CENTERS)), groups, strict=True)}
    save(output, {"schema": SCHEMA, "rows": rows, "unique_inputs": len(unique)})


def position_scales(readings):
    sorted_values = [sorted(row[i] for row in readings if row[i] is not None) for i in range(len(CENTERS))]
    return [(median(v), max(1e-5, v[(9 * len(v)) // 10] - median(v))) if v else None for v in sorted_values]


def scale_reading(scores, scales):
    return [round((value - rule[0]) / rule[1], 8) if value is not None and rule else None
            for value, rule in zip(scores, scales, strict=True)]


def maxima(values):
    found = sorted(((v, center) for v, center in zip(values, CENTERS, strict=True) if v is not None), reverse=True)
    return found[0] if found and (len(found) == 1 or found[0][0] != found[1][0]) else None


def rows_with_points(rows, panels):
    base_arms = ARM_NAMES[:-1]

    def apply(genre, exclude=None):
        calibration = [row for row in rows if row["genre"] == genre and row["role"] == "calibration"
                       and row["group"] != exclude]
        return {arm: position_scales([panels[row["sha256"]][arm] for row in calibration]) for arm in base_arms}

    def normalized(key, rules):
        x = {arm: scale_reading(panels[key][arm], rules[arm]) for arm in base_arms}
        x["combined"] = [max(a, b) if a is not None and b is not None else a if b is None else b
                         for a, b in zip(x["current"], x["learned_prompt"], strict=True)]
        return x

    points, selected = {}, {}
    for genre in dict.fromkeys(row["genre"] for row in rows if row["role"] == "held"):
        ord_ = [row for row in rows if row["genre"] == genre and row["role"] == "calibration"]
        require(len(ord_) >= 12, "Too few genuine works to measure localization cutoffs")
        views = [normalized(row["sha256"], apply(genre, row["group"])) for row in ord_]
        rules = apply(genre)
        points[genre] = {}
        for arm in ARM_NAMES:
            peaks = sorted((result[0] for entry in views if (result := maxima(entry[arm])) is not None), reverse=True)
            require(len(peaks) >= 12, "Too few assessed calibration works")
            points[genre][arm] = {level: peaks[len(peaks) // value] for level, value in LEVELS.items()}
        for row in rows:
            if row["role"] == "held" and row["genre"] == genre:
                selected[row["sha256"]] = {arm: maxima(value) for arm, value in normalized(row["sha256"], rules).items()}
    return points, selected


def audit(record):
    rows, panels = record["roster"], record["scores"]
    require(record["schema"] == SCHEMA and {r["sha256"] for r in rows} == set(panels), "Study rows differ")
    require(len(rows) == len(panels) and all(row["role"] in {"calibration", "held"} for row in rows),
            "Repeated input or unexpected source role")
    roles = {role: {r["group"] for r in rows if r["role"] == role} for role in ("calibration", "held")}
    require(roles["calibration"].isdisjoint(roles["held"]), "Calibration and held source works overlap")
    require(set(record["fit_groups"]).isdisjoint(set.union(*roles.values())), "Fit works overlap evaluation")
    require({r["group"] for r in record["fit_roster"]} == set(record["fit_groups"]), "The fit roster differs")
    require(set(record["training"]["optimized_source_groups"]) <= set(record["fit_groups"]), "Training used a held work")
    require(all(set(p) == set(ARM_NAMES[:-1]) and all(
        len(v) == len(CENTERS) and all(n is None or math.isfinite(n) for n in v) for v in p.values()
    ) for p in panels.values()), "Invalid sentence measurements")
    points, peaks = rows_with_points(rows, panels)
    bucket, paired_rank = {}, {}
    by_pair = {(r["group"], r["label"]): r for r in rows if r["role"] == "held"}
    require(len(by_pair) == 2 * len(roles["held"]) and all(
        (group, "ordinary") in by_pair and (group, "edited") in by_pair
        and by_pair[group, "ordinary"]["target"] == by_pair[group, "edited"]["target"] in CENTERS
        for group in roles["held"]
    ), "Held edits must have one original and one known edited sentence")
    words = record["edit_word_counts"]
    require(set(words) == {row["sha256"] for row in rows if row["label"] == "edited"}
            and all(type(total) is int and type(changed) is int and 0 <= changed <= total
                    for changed, total in words.values()), "Edit word totals differ from the held set")
    for row in rows:
        if row["role"] != "held":
            continue
        genre, target = row["genre"], row["target"]
        is_edit = row["label"] == "edited"
        found = peaks[row["sha256"]]
        if is_edit:
            original = by_pair[row["group"], "ordinary"]
            index = CENTERS.index(target)
            for arm in ARM_NAMES[:-1]:
                a, b = (panels[item["sha256"]][arm][index] for item in (original, row))
                key = f"{genre}/{row['generator']}/{arm}"
                ranked = paired_rank.setdefault(key, Counter())
                result = "unassessed" if a is None or b is None else "higher" if b > a else "tie" if b == a else "lower"
                ranked[result] += 1
        for arm, hit in found.items():
            for level, point in points[genre][arm].items():
                key = f"{genre}/{row['generator']}/{arm}/{level}"
                outcomes = bucket.setdefault(key, Counter())
                marked = hit is not None and hit[0] > point
                useful = marked and is_edit and hit[1] == target
                name = "edited" if is_edit else "ordinary"
                outcomes[name] += 1
                outcomes["unassessed_" + name] += all(v is None for v in panels[row["sha256"]].get(arm, ())) if arm != "combined" else (
                    all(v is None for v in panels[row["sha256"]]["current"]) and
                    all(v is None for v in panels[row["sha256"]]["learned_prompt"])
                )
                if is_edit:
                    changed, total = words[row["sha256"]]
                    outcomes["correct"] += useful
                    outcomes["correct_before_point"] += hit is not None and hit[1] == target
                    outcomes["useful_full_sentence_mask"] += useful and changed > 0 and 10 * changed >= 3 * total
                    outcomes["wrong"] += marked and not useful
                    counterpart = peaks[original["sha256"]][arm]
                    outcomes["correct_also_ordinary"] += (useful and counterpart is not None and
                                                         counterpart[0] > point and counterpart[1] == target)
                else:
                    outcomes["false_flags"] += marked
    return {"points": points, "outcomes": {k: dict(v) for k, v in sorted(bucket.items())},
            "paired_rank": {k: dict(v) for k, v in sorted(paired_rank.items())}}


def count_changed_words(rows):
    originals = {row["group"]: row for row in rows if row["role"] == "held" and row["label"] == "ordinary"}
    counts = {}

    def focus(row, target):
        lower, upper = row["spans"][target]
        return WORD.findall(row["text"][lower:upper].casefold())

    for edited in rows:
        if edited["role"] != "held" or edited["label"] != "edited":
            continue
        target = edited["target"]
        original = originals[edited["group"]]

        prior, current = focus(original, target), focus(edited, target)
        changes = SequenceMatcher(None, prior, current, autojunk=False)
        counts[edited["sha256"]] = [sum(b-a for tag, _, _, a, b in changes.get_opcodes() if tag != "equal"),
                                    len(current)]
    return counts


def merge(plan, causal, current, output):
    require(set(causal["rows"]) == set(current["rows"]), "Current and new score rosters differ")
    record = {"schema": SCHEMA, "intake": plan["intake"], "exposures": plan["exposures"],
              "publisher_revision": plan["publisher_revision"], "inputs": plan["inputs"],
              "additional_inputs": plan["additional_inputs"], "fit_plan_sha256": plan["fit_plan_sha256"],
              "roster": [{key: r[key] for key in PUBLIC_FIELDS} for r in plan["rows"] if r["role"] != "fit"],
              "fit_roster": [{key: r[key] for key in PUBLIC_FIELDS} for r in plan["rows"] if r["role"] == "fit"],
              "fit_groups": sorted({r["group"] for r in plan["rows"] if r["role"] == "fit"}),
              "training": causal["training"], "edit_word_counts": count_changed_words(plan["rows"]),
              "scores": {key: row | {"current": current["rows"][key]}
                                                          for key, row in causal["rows"].items()}}
    record["summary"] = audit(record)
    save(output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fields in (("prepare", ("sources", "output")), ("incumbent", ("plan", "snapshot", "output")),
                         ("train", ("plan", "snapshot", "weights", "output")),
                         ("score", ("plan", "snapshot", "weights", "output", "training")),
                         ("merge", ("plan", "causal", "current", "output")), ("review", ("receipt",)),
                         ("rebuild", ("receipt", "sources", "additional", "output"))):
        command = sub.add_parser(name)
        for field in fields:
            command.add_argument(field, type=Path)
        if name == "prepare":
            command.add_argument("--prior-receipt", type=Path)
            command.add_argument("--fit-from", type=Path)
            command.add_argument("--additional-shards", type=Path)
            command.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.sources, args.output, read(args.prior_receipt)["exposures"] if args.prior_receipt else None,
                fit_plan=args.fit_from, extra=args.additional_shards, reports_only=args.reports_only)
    elif args.command == "incumbent":
        incumbent(read(args.plan), args.snapshot, args.output)
    elif args.command in {"train", "score"}:
        from scripts import writing_prompt_runtime

        inputs = (read(args.plan), args.snapshot, args.weights, args.output)
        if args.command == "train":
            writing_prompt_runtime.train(*inputs)
        else:
            writing_prompt_runtime.score(*inputs, args.training)
    elif args.command == "merge":
        merge(read(args.plan), read(args.causal), read(args.current), args.output)
    elif args.command == "rebuild":
        rebuild(read(args.receipt), args.sources, args.additional, args.output)
    else:
        receipt = read(args.receipt)
        require(audit(receipt) == receipt["summary"], "Independent outcome reconstruction differs")
        print(json.dumps(receipt["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
