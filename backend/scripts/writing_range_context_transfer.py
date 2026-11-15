# SPDX-License-Identifier: MIT
"""Compare exact writing ranges against the other sentences in one upload."""

import argparse
import gzip
import json
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_range_report_replication as replication
from scripts import writing_supervised_ranges as ranges

PANELS = replication.pool.PANEL_DIR
ELIFE = PANELS / "writing-elife-exact-sentence-confirmation-2026-11-12.json"
FRESH = PANELS / "writing-paired-ranges-fresh-2026-11-15.json.gz"
EXTRA = PANELS / "writing-range-context-elife-readings-2026-11-15.json.gz"
ABSTRACTS = PANELS / "writing-range-context-abstract-readings-2026-11-15.json.gz"
INPUTS = {
    "report": (ranges.RANGE_REPORT, PANELS / "writing-paired-ranges-report-2026-11-15.json.gz"),
    "transfer": (ranges.RANGE_TRANSFER, PANELS / "writing-paired-ranges-transfer-2026-11-15.json.gz"),
}
SCHEMA = "chipmunk.writing-range-context/1"
CAPTURE = "chipmunk.writing-range-context-elife/1"
ABSTRACT_SCHEMA = "chipmunk.writing-range-context-abstracts/1"
ABSTRACT_PINS = {
    "gemini-2.5-flash": "2c626b0597f6082908a0183b31c48884a45d310fea8bb1a84043ed2432568b30",
    "gpt-5.4": "e657011ad11434eb9f5860dff3dbe72790843c0dd8626ad7755abcae4c5562a7",
}
ABSTRACT_LIMIT = 8
METHOD = {
    "scope": "Both previous heads read only one submission at a time, using the same original range geometry. "
             "The native test centers a five-sentence excerpt on the known change, with two unchanged sentences either side.",
    "contrast": "For each head, subtract the strongest other interior sentence response from each position. "
                "Compare that contrast alone and the smaller of its rank and the absolute response rank at that position.",
    "agreement": "Take the smaller rank across both heads' absolute-and-contrast scores at the same sentence. "
                 "One arm shows the paired head's range. The other requires three shared words covering half of each "
                 "range and shows the shared words only.",
    "calibration": "Rank against original calibration works at the same sentence position. Leave each calibration "
                   "work out of its own ranking. For each arm, set separate points from the strongest response per work "
                   "at the existing 10 and 25 percent budgets. Require a unique highest result strictly above the point.",
    "held": "Use the report points for held reports, genre-specific points for news and abstracts, and compare "
            "report and abstract points separately for the dated journals and eLife excerpts. The additional abstract "
            "holdout uses the same abstract points and fixed fitted heads.",
    "outcomes": "Count useful word overlap, false ordinary flags, off-target alerts, and genuine matched "
                "uploads that also flagged the edit's sentence. Count ordinary editorial and same-work replacements separately.",
}
KEYS = ranges.KEYS
SIGNED_ARMS = ("direct", "paired")
NEW_ARMS = ("contrast", "absolute_and_contrast", "same_sentence", "shared_words")
LEVELS = {"10pct": 10, "25pct": 4}
ORDINARY = {"original", "before", "edited", "ordinary"}
read, require = ranges.spans.read, replication.require
pool, words = replication.pool, replication.words


def digest(path):
    return pool.file_hash(path)


def signed_inputs():
    report, transfer = (tuple(map(read, INPUTS[name])) for name in ("report", "transfer"))
    for split in (report, transfer):
        require(split[0]["encoder_sha256"] == split[1]["encoder_sha256"], "The two heads used different encoders")
    for initial, final, path in zip(report, transfer, INPUTS["report"], strict=True):
        ranges.review_record(initial)
        ranges.review_record(final, path)
    return report, transfer


def align(direct, paired):
    for a, b in zip(direct, paired, strict=True):
        require(all(a[key] == b[key] for key in KEYS) and a.get("domain") == b.get("domain"),
                "The two measurements use different excerpts or change labels")
        yield b, {"direct": a, "paired": b}


def elife_cases(path, excluded):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    previous = read(ELIFE)
    require(previous["private_sha256"]["plan.json"] == digest(path), "Dated eLife source plan differs")
    cases = read(path)["cases"]
    receipts = {(item["sha256"], item["role"]): item["group"] for item in previous["cases"]}
    sources = defaultdict(dict)
    for row in cases:
        require(row["sha256"] == pool.sha(row["text"]) and row["role"] not in sources[row["group"]], "Historical text changed")
        sources[row["group"]][row["role"]] = row

    rows, truth = [], {}
    disposition = Counter()
    for group, variants in sources.items():
        public_groups = {receipts[row["sha256"], family] for family, row in variants.items()}
        require(len(public_groups) == 1, "The public groups disagree with the native archive")
        public, = public_groups
        if {group, public, pool.sha(group)} & excluded:
            disposition["previous_group"] += 1
            continue
        target = variants["ordinary"]["truth"][0]
        pending = []
        for family, row in variants.items():
            document = from_string(row["text"])
            spans = [(sentence.start, sentence.end) for sentence in segment(document)]
            focus = next((i for i, bound in enumerate(spans) if bound[0] == target), -1)
            if document.text != row["text"] or focus < 2 or focus + 2 >= len(spans):
                break
            if row["truth"] is not None and list(spans[focus]) != row["truth"]:
                break
            local = spans[focus - 2:focus + 3]
            start, end = local[0][0], local[-1][1]
            text = document.text[start:end]
            pending.append({"role": "elife_transfer", "domain": "journals", "family": family, "group": public,
                            "text": text, "sha256": pool.sha(text), "spans": [[a - start, b - start] for a, b in local],
                            "position": 2, "original_excerpt_sha256": row["sha256"], "offset_in_archive": start})
        if len(pending) != len(variants):
            disposition["insufficient_neighbors"] += 1
            continue
        if any(row["sha256"] in excluded for row in pending):
            disposition["previous_excerpt"] += 1
            continue
        original, = [row for row in pending if row["family"] == "original"]
        neighbors = [original["text"][a:b] for i, (a, b) in enumerate(original["spans"]) if i != 2]
        if any([row["text"][a:b] for i, (a, b) in enumerate(row["spans"]) if i != 2] != neighbors for row in pending):
            disposition["different_neighbors"] += 1
            continue
        rows.extend(pending)
        truth.update({row["sha256"]: words.edit_words(original, row) for row in pending if row["family"] != "original"})
        disposition["admitted"] += 1
    require(rows, "No separate eLife groups have the target geometry")
    return rows, truth, dict(disposition)


def capture(args):
    [report, _] = signed_inputs()
    exclusions = {value for view in report for item in view["fit"] for value in item[1::2]}
    exclusions.update(value for view in report for row in view["calibration"] + view["held"] for value in (row["group"], row["sha256"]))
    native, changes, disposition = elife_cases(args.elife, exclusions)
    document = {"schema": CAPTURE, "source": digest(args.elife), "original_receipt": digest(ELIFE), "disposition": disposition}
    document.update(native_measurement(args, report, native, changes))
    save(args.output, document)
    print(json.dumps({"source_groups": disposition, "files": len(native)}, sort_keys=True))


def native_measurement(args, report, native, changes):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(3)
    require(digest(args.encoder_weights) == report[0]["encoder_sha256"], "The exact contextual encoder differs")
    tokenizer, encoder, _, device = words.parent.init_encoder(args.encoder)
    raw = load_file(args.encoder_weights)
    encoder.load_state_dict({key.removeprefix("encoder."): tensor for key, tensor in raw.items() if key.startswith("encoder.")}, strict=True)
    del raw
    encoder.eval()
    result = {}
    for arm, path, parent in zip(SIGNED_ARMS, (args.direct, args.paired), report, strict=True):
        require(digest(path) == parent["head_sha256"], "A previously fitted range head differs")
        state = load_file(path)
        model = ranges.head()
        model.load_state_dict({key.removeprefix("head."): tensor for key, tensor in state.items() if key.startswith("head.")}, strict=True)
        result[arm] = list(ranges.measure(native, changes, tokenizer, encoder, device, (model.eval(), state["mean"], state["scale"])))
    inputs = [{key: value for key, value in row.items() if key != "text"} for row in native]
    return {"heads": {arm: receipt["head_sha256"] for arm, receipt in zip(SIGNED_ARMS, report, strict=True)},
            "inputs": inputs, "readings": result}


def abstract_exclusions(commitments=None):
    if commitments is None:
        paths = sorted((*PANELS.glob("writing-*.json"), *PANELS.glob("writing-*.json.gz"), *PANELS.glob("text-opai-*.json")))
        commitments = {path.name: digest(path) for path in paths if not path.name.startswith("writing-range-context-")}
    known = set()
    for name, pin in commitments.items():
        path = PANELS / name
        require(path.name == name and digest(path) == pin, "The abstract exposure inventory differs")
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        known.update(item.decode() for item in re.findall(rb'"([0-9a-f]{64})"', raw))
    return known, commitments


def capture_abstracts(args):
    from scripts import writing_focus_inside_context as intake
    from scripts.writing_interval_transfer import fingerprints

    exclusion_receipt = read(args.reference)["exposure_receipts"] if args.reference else None
    known, exposure = abstract_exclusions(exclusion_receipt)
    tables = {family: intake.shard(path, "abstracts", family, pin)
              for (family, pin), path in zip(ABSTRACT_PINS.items(), (args.gemini, args.gpt), strict=True)}
    known_phrases = {phrase for table in tables.values() for work, variants in table.items()
                     if pool.sha(work) in known and "v0" in variants for phrase in fingerprints(variants["v0"]["text"])}
    native, changes, disposition = [], {}, {}
    for family, table in tables.items():
        counts, positions = Counter(), Counter()
        for work in sorted(table, key=lambda item: pool.sha(f"{ABSTRACT_SCHEMA}/{family}/{item}")):
            group, variants = pool.sha(work), table[work]
            if group in known or "v0" not in variants:
                counts["work_excluded"] += 1
                continue
            phrases = set(fingerprints(variants["v0"]["text"]))
            if phrases & known_phrases:
                counts["phrase_excluded"] += 1
                continue
            try:
                options = intake.excerpt_options(work, variants)
            except (ValueError, KeyError):
                counts["ineligible"] += 1
                continue
            eligible = [position for position in options if positions[position] < ABSTRACT_LIMIT]
            if not eligible:
                counts["unavailable_position"] += 1
                continue
            position = min(eligible, key=lambda item: (positions[item], item))
            pair = [{"role": "abstract_holdout", "domain": "abstracts", "family": role, "group": group,
                     "text": text, "spans": spans, "sha256": pool.sha(text), "position": position, "variant": variant}
                    for (text, spans, variant), role in zip(options[position], ("original", family), strict=True)]
            fragments = {fragment for row in pair for fragment in fingerprints(row["text"])}
            if fragments & known_phrases:
                counts["excerpt_overlap"] += 1
                continue
            native.extend(pair)
            changes[pair[1]["sha256"]] = words.edit_words(*pair)
            known.add(group)
            known_phrases.update(phrases | fragments)
            positions[position] += 1
        disposition[family] = dict(counts) | {"positions": {str(key): count for key, count in positions.items()}}
    require(native, "No additional matched abstract work is eligible")
    lock = {"schema": ABSTRACT_SCHEMA, "shards": ABSTRACT_PINS, "exposure_receipts": exposure,
            "source_groups": disposition, "max_per_position": ABSTRACT_LIMIT}
    public_plan = [{key: value for key, value in row.items() if key != "text"} for row in native]
    save(args.output.with_suffix(".plan.json.gz"), lock | {"inputs": public_plan})
    report, _ = signed_inputs()
    save(args.output, lock | native_measurement(args, report, native, changes))
    print(json.dumps(disposition, sort_keys=True))


def features(views):
    proposals, responses = {}, {}
    for arm, row in views.items():
        locations = {item["sentence"]: item for item in row["by_sentence"]}
        if set(locations) != {1, 2, 3}:
            return None, None
        proposals[arm] = locations
        for section, item in locations.items():
            value = item["score"]
            other = max(other["score"] for j, other in locations.items() if j != section)
            responses[arm, section, "absolute"] = value
            responses[arm, section, "contrast"] = round(value - other, 7)
    return proposals, responses


def calibration(rows):
    views = {}
    for row, pair in rows:
        require(row["family"] == "original" and row["group"] not in views and not row["changed"],
                "Calibration requires one ordinary original per source work")
        views[row["group"]] = pair
    require(len(views) >= 24, "Insufficient ordinary calibration works")
    by_response = defaultdict(dict)
    for group, pair in views.items():
        _, values = features(pair)
        for key, value in (values or {}).items():
            by_response[key][group] = value
    require(all(len(values) >= 22 for values in by_response.values()) and len(by_response) == 12,
            "Too few complete ordinary response vectors")
    return views, dict(by_response)


def derived(views, reference, *, exclude=None):
    proposals, responses = features(views)
    if not responses:
        return dict.fromkeys(NEW_ARMS)
    ranked = {}
    for key, value in responses.items():
        population = sorted(reading for group, reading in reference[key].items() if group != exclude)
        ranked[key] = (10000 * bisect_left(population, value)) // (len(population) + 1)
    choices = defaultdict(list)
    for section in (1, 2, 3):
        direct, paired = (proposals[arm][section] for arm in SIGNED_ARMS)
        contrast = ranked["paired", section, "contrast"]
        paired_strength = min(contrast, ranked["paired", section, "absolute"])
        strength = min(paired_strength, *(ranked["direct", section, feature] for feature in ("absolute", "contrast")))
        choices["contrast"].append(paired | {"score": contrast})
        choices["absolute_and_contrast"].append(paired | {"score": paired_strength})
        choices["same_sentence"].append(paired | {"score": strength})
        shared = set(direct["words"]) & set(paired["words"])
        if len(shared) >= 3 and 2 * len(shared) >= max(len(direct["words"]), len(paired["words"])):
            choices["shared_words"].append({"score": strength, "sentence": section, "words": sorted(shared)})
    output = {}
    for arm in NEW_ARMS:
        ordered = sorted(choices[arm], key=lambda item: -item["score"])
        output[arm] = ordered[0] if ordered and (len(ordered) == 1 or ordered[0]["score"] > ordered[1]["score"]) else None
    return output


def operating(rows, reference):
    decisions = [derived(views, reference, exclude=group) for group, views in rows.items()]
    points = {}
    for arm in NEW_ARMS:
        scores = sorted((choice[arm]["score"] if choice[arm] else 0 for choice in decisions), reverse=True)
        points[arm] = {level: scores[len(scores) // divisor] for level, divisor in LEVELS.items()}
    return points


def project(row, views, reference):
    choices = {name: item["best"]["range"] for name, item in views.items()}
    candidates = derived(views, reference)
    return {key: row[key] for key in ("group", "family", "sha256", "changed", "position", "assessed")} | {
        "reports": choices | candidates,
    }


def count(rows, points):
    measures = defaultdict(Counter)
    ordinary_flags = defaultdict(list)
    for row in rows:
        if row["family"] in ORDINARY:
            for arm, limits in points.items():
                candidate = row["reports"][arm]
                for level, cutoff in limits.items():
                    if candidate and candidate["score"] > cutoff:
                        ordinary_flags[arm, level, row["group"]].append(candidate["sentence"])
    for row in rows:
        for arm, limits in points.items():
            reading = row["reports"][arm]
            before = row["reports"]["paired"]
            ordinary = row["family"] in ORDINARY
            for level, cutoff in limits.items():
                flagged = reading is not None and reading["score"] > cutoff
                old = before is not None and before["score"] > points["paired"][level]
                useful = flagged and (ordinary or words.useful_words(set(reading["words"]), set(row["changed"])))
                prior = old and (ordinary or words.useful_words(set(before["words"]), set(row["changed"])))
                counter = measures[arm, level, row["family"]]
                counter.update(files=1, unassessed=not row["assessed"], alerts=flagged, useful=useful,
                               gained=useful and not prior, lost=prior and not useful)
                if ordinary:
                    counter.update(flagged_changed_sentence=flagged and reading["sentence"] == row["position"])
                else:
                    collision = row["position"] in ordinary_flags[arm, level, row["group"]]
                    counter.update(off_target=flagged and not useful,
                                   useful_with_matched_alert=useful and collision,
                                   useful_without_matched_alert=useful and not collision)
    return {"/".join(key): dict(value) for key, value in sorted(measures.items())}


def captures(path, report):
    data = read(path)
    original = read(ELIFE)
    require(data["schema"] == CAPTURE and data["original_receipt"] == digest(ELIFE)
            and data["source"] == original["private_sha256"]["plan.json"], "The new readings refer to different source files")
    require(data["heads"] == {arm: record["head_sha256"] for arm, record in zip(SIGNED_ARMS, report, strict=True)},
            "The new readings refer to different fitted heads")
    # Recheck source digests and group roles against the earlier public measurement.
    known = {(row["group"], row["role"]): row for row in original["cases"]}
    rows = list(align(*(data["readings"][arm] for arm in SIGNED_ARMS)))
    for commitment, (row, views) in zip(data["inputs"], rows, strict=True):
        require(all(row[key] == commitment[key] for key in ("role", "domain", "family", "group", "sha256", "spans", "position"))
                and commitment["original_excerpt_sha256"] == known[row["group"], row["family"]]["sha256"],
                "Native excerpt or role differs")
        for reading in views.values():
            ranges.validate_range_row(reading)
    return rows


def abstract_capture(path, report):
    data = read(path)
    require(data["schema"] == ABSTRACT_SCHEMA and data["shards"] == ABSTRACT_PINS
            and data["max_per_position"] == ABSTRACT_LIMIT, "The frozen abstract cohort differs")
    require(data["heads"] == {arm: record["head_sha256"] for arm, record in zip(SIGNED_ARMS, report, strict=True)},
            "The abstract readings refer to different fitted heads")
    excluded, _ = abstract_exclusions(data["exposure_receipts"])
    result = list(align(*(data["readings"][arm] for arm in SIGNED_ARMS)))
    for witness, (row, views) in zip(data["inputs"], result, strict=True):
        require(all(row[key] == witness[key] for key in ("role", "domain", "family", "group", "sha256", "spans", "position"))
                and row["group"] not in excluded and row["domain"] == "abstracts" and row["role"] == "abstract_holdout",
                "Abstract corpus roles or source positions differ")
        for reading in views.values():
            ranges.validate_range_row(reading)
    groups, strata = set(), defaultdict(Counter)
    for (before, _), (after, _) in zip(result[::2], result[1::2], strict=True):
        require(before["family"] == "original" and not before["changed"] and after["family"] in ABSTRACT_PINS
                and before["group"] == after["group"] and before["position"] == after["position"] and after["group"] not in groups,
                "An abstract pair or source group is invalid")
        groups.add(after["group"])
        strata[after["family"]][str(after["position"])] += 1
    require(set(data["source_groups"]) == set(ABSTRACT_PINS)
            and all(dict(strata[family]) == entry["positions"] for family, entry in data["source_groups"].items()),
            "Abstract language-model or position counts differ")
    return result


def evaluate(extra, fresh_abstracts):
    report, transfer = signed_inputs()
    datasets = {name: list(align(*(record["held"] for record in split)))
                for name, split in (("report", report), ("transfer", transfer))}
    fresh = read(FRESH)
    replication.review(fresh, INPUTS["report"][1])
    datasets["fresh"] = list(align(fresh["unpaired"], fresh["held"]))
    datasets["elife"] = captures(extra, report)
    calibration_sets = {"report": list(align(*(record["calibration"] for record in report)))}
    by_genre = list(align(*(record["calibration"] for record in transfer)))
    for name in ("news", "abstracts"):
        calibration_sets[name] = [item for item in by_genre if item[0]["domain"] == name]
    points, references = {}, {}
    for name, rows in calibration_sets.items():
        eligible, references[name] = calibration(rows)
        points[name] = operating(eligible, references[name])
        for arm, before in zip(SIGNED_ARMS, report if name == "report" else transfer, strict=True):
            points[name][arm] = (before["points"] if name == "report" else before["points"][name])["range"]
    panels = {"report": ("report", datasets["report"]), "fresh_reports": ("report", datasets["fresh"])}
    for genre in ("news", "abstracts"):
        panels[genre] = genre, [item for item in datasets["transfer"] if item[0]["domain"] == genre]
    journals = [item for item in datasets["transfer"] if item[0]["domain"] == "journals"]
    for genre in ("report", "abstracts"):
        panels[f"dated_journals/{genre}"] = genre, journals
        panels[f"elife/{genre}"] = genre, datasets["elife"]
    panels["fresh_abstracts"] = "abstracts", abstract_capture(fresh_abstracts, report)
    outcomes, readings = {}, {}
    for name, (genre, sources) in panels.items():
        readings[name] = [project(row, views, references[genre]) for row, views in sources]
        outcomes[name] = count(readings[name], points[genre])
    files = [path for paths in INPUTS.values() for path in paths] + [ELIFE, FRESH]
    return {"schema": SCHEMA, "method": METHOD, "points": points, "native_source_commitments": {
              file.name: digest(file) for file in files} | {EXTRA.name: digest(extra), ABSTRACTS.name: digest(fresh_abstracts)},
            "readings": readings, "outcomes": outcomes}


def save(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    phases = cli.add_subparsers(dest="phase", required=True)
    weights = ("encoder", "encoder_weights", "direct", "paired", "output")
    for phase, fields in (("capture", ("elife", *weights)), ("capture-abstracts", ("gemini", "gpt", *weights)),
                          ("measure", ("receipt",)), ("review", ("receipt",))):
        command = phases.add_parser(phase)
        for field in fields:
            command.add_argument(field, type=Path)
        if phase == "capture-abstracts":
            command.add_argument("--reference", type=Path)
        if phase in {"measure", "review"}:
            command.add_argument("--elife", type=Path, default=EXTRA)
            command.add_argument("--abstracts", type=Path, default=ABSTRACTS)
    args = cli.parse_args()
    if args.phase == "capture":
        capture(args)
    elif args.phase == "capture-abstracts":
        capture_abstracts(args)
    else:
        measured = evaluate(args.elife, args.abstracts)
        if args.phase == "measure":
            save(args.receipt, measured)
        else:
            require(read(args.receipt) == measured, "The exact writing comparison did not reproduce")
        print(json.dumps(measured["outcomes"], indent=2, sort_keys=True))
