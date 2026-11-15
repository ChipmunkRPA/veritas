# SPDX-License-Identifier: MIT
"""Test training on GPT-5.4 rewrite examples against disjoint report works."""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from scripts import writing_edit_occurrence_transfer as occurrence
from scripts import writing_focus_inside_context as intake
from scripts import writing_range_report_replication as replication
from scripts import writing_supervised_ranges as ranges
from scripts import writing_word_localization as words
from scripts.writing_interval_transfer import fingerprints

SCHEMA = "chipmunk.writing-range-generator-adaptation/1"
PANEL = replication.pool.PANEL_DIR
BASE = PANEL / "writing-paired-ranges-report-2026-11-15.json.gz"
OLDER, HELD = replication.PREVIOUS_REPLICATION, PANEL / "writing-paired-ranges-fresh-2026-11-15.json.gz"
FAMILY = replication.FAMILY
SPLITS = ("calibration", "held", "transfer")
TRANSFER = {genre: intake.PINS[genre][1] for genre in ("news", "abstracts")}
PER_POSITION = 8
EXCLUSIONS = (*replication.EXPOSURE, *replication.pool.EXPOSURE, intake.PARENT.name, "writing-focus-inside-context-2026-11-14.json")
EXPOSURE_AUDIT = ("writing-joint-tail-matched-2026-11-14.json", "writing-joint-tail-transfer-2026-11-14.json",
                  "writing-word-localization-transfer-2026-11-14.json", "writing-local-word-spans-2026-11-15.json.gz",
                  "writing-context-response-transfer-2026-11-15.json.gz", "writing-supervised-ranges-transfer-2026-11-15.json.gz",
                  "writing-paired-ranges-transfer-2026-11-15.json.gz")
KEYS = ("role", "group", "family", "sha256", "spans", "position", "variant")
pool, require = replication.pool, replication.require
read, write = ranges.spans.read, words.parent.previous.write


def contract():
    return {"schema": SCHEMA, "source_sha256": replication.SOURCE_SHA,
            "references": {p.name: pool.file_hash(p) for p in (BASE, OLDER, HELD)},
            "selection_exclusions": {name: pool.file_hash(PANEL / name) for name in EXCLUSIONS},
            "transfer_exposure_audit": {name: pool.file_hash(PANEL / name) for name in EXPOSURE_AUDIT},
            "source_roster_sha256": read(words.RECEIPT)["roster_sha256"],
            "transfer_shards": TRANSFER, "transfer_works_per_position": PER_POSITION}


def pair(group, position, versions, role):
    return [{"role": role, "group": group, "position": position, "family": family,
             "text": text, "sha256": pool.sha(text), "spans": spans, "variant": variant}
            for (text, spans, variant), family in zip(versions, ("original", FAMILY), strict=True)]


def earlier_report_groups():
    rows = read(BASE)["held"]
    return {r["group"] for r in rows if r["family"] == FAMILY}


def work_identifiers(references):
    for name in references:
        path = PANEL / name
        contents = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
        yield from re.findall(r'"([a-f0-9]{64})"', contents)


def freeze(source, roster, transfer_paths):
    base, _, _ = words.input_plan(roster)
    exposed = {row["group"] for row in base}
    exposed.update(work_identifiers(EXCLUSIONS))
    table = words.parent.load_shard(source, "test", FAMILY, replication.SOURCE_SHA)
    by_group = {pool.sha(key): key for key in table}
    adapt_groups = earlier_report_groups()
    rows = {"fit_addition": [r | {"role": "fit_addition"} for r in base if r["role"] == "held" and r["group"] in adapt_groups],
            "calibration": [r for r in base if r["role"] == "calibration"], "held": [], "transfer": []}
    for role, ref in (("fit_addition", OLDER), ("held", HELD)):
        prior = read(ref)["held"]
        for before, after in zip(prior[::2], prior[1::2], strict=True):
            group, position = before["group"], after["position"]
            require(group == after["group"], "A historical comparison is no longer paired")
            key = by_group[group]
            admission = words.parent.intake.excerpt_options(key, table[key])[position]
            rebuilt = pair(group, position, admission, role)
            require([item["sha256"] for item in rebuilt] == [before["sha256"], after["sha256"]], "A historical pair changed")
            rows[role].extend(rebuilt)
            exposed.add(group)

    seen = {piece for row in base + rows["fit_addition"] + rows["held"] for piece in fingerprints(row["text"])}
    for genre, sha in TRANSFER.items():
        table = intake.shard(transfer_paths[genre], genre, FAMILY, sha)
        for key, versions in table.items():
            if pool.sha(key) in exposed and "v0" in versions:
                seen.update(fingerprints(versions["v0"]["text"]))
        counts = Counter()
        for key in sorted(table, key=lambda work: pool.sha(f"{SCHEMA}/{genre}/{work}")):
            if sum(counts.values()) == 3 * PER_POSITION:
                break
            group, versions = pool.sha(key), table[key]
            if group in exposed or "v0" not in versions:
                continue
            pieces = set(fingerprints(versions["v0"]["text"]))
            if seen & pieces:
                continue
            try:
                options = words.parent.intake.excerpt_options(key, versions)
            except (KeyError, ValueError):
                continue
            positions = [i for i in options if counts[i] < PER_POSITION]
            if not positions:
                continue
            position = min(positions, key=lambda i: (counts[i], i))
            excerpt = pair(group, position, options[position], "transfer")
            excerpt_pieces = {piece for row in excerpt for piece in fingerprints(row["text"])}
            if seen & excerpt_pieces:
                continue
            rows["transfer"].extend(r | {"domain": genre} for r in excerpt)
            counts[position] += 1
            exposed.add(group)
            seen.update(pieces | excerpt_pieces)
        require(counts == dict.fromkeys((1, 2, 3), PER_POSITION), f"Insufficient separate works for {genre}: {dict(counts)}")
    return contract() | {"rows": rows}


def roster_commitments(plan):
    return {role: [{key: row[key] for key in KEYS} | ({"domain": row["domain"]} if role == "transfer" else {})
                   for row in rows] for role, rows in plan["rows"].items()}


def review(record):
    require(record["contract"] == contract(), "A source or comparison binding changed")
    roster = record["roster"]
    require(set(roster) == {"fit_addition", *SPLITS}, "Missing source role")
    def identities(rows):
        return {(row["group"], row["family"], row["sha256"], row["position"]) for row in rows}

    reference = read(BASE)
    require(record["baseline_sha256"] == reference["head_sha256"] and record["encoder_sha256"] == reference["encoder_sha256"],
            "The baseline weights differ")
    additions = identities(read(OLDER)["held"]) | identities(r for r in reference["held"] if r["group"] in earlier_report_groups())
    require(identities(roster["held"]) == identities(read(HELD)["held"]) and len(roster["held"]) == 96,
            "The report holdout changed")
    require(identities(roster["calibration"]) == identities(reference["calibration"]) and len(roster["calibration"]) == 72,
            "The ordinary calibration works changed")
    for role, rows in roster.items():
        require(all(row["role"] == role for row in rows), "An excerpt changed role")
        if role != "calibration":
            pairs = list(zip(rows[::2], rows[1::2], strict=True))
            require(len({base["group"] for base, _ in pairs}) == len(pairs), "A source pair repeats")
            require(all(before["group"] == after["group"] and before["position"] == after["position"]
                        and (before["family"], after["family"]) == ("original", FAMILY)
                        for before, after in pairs), "A paired excerpt differs")
            anchors = rows[::2]
        else:
            anchors = rows
            require(all(r["family"] == "original" for r in rows), "Calibration contains a rewrite")
        if role == "transfer":
            require(Counter((r["domain"], r["position"]) for r in anchors) == {
                (genre, i): PER_POSITION for genre in TRANSFER for i in (1, 2, 3)}, "A source quota differs")
    require(len(roster["fit_addition"]) == len(additions) and additions == identities(roster["fit_addition"]), "Adaptation examples changed")
    sources = {role: {r["group"] for r in rows} for role, rows in roster.items()}
    require(len(sources["calibration"]) == len(roster["calibration"])
            and len(set.union(*sources.values())) == sum(map(len, sources.values())),
            "Source work leaked between development and evaluation")
    require(sources["transfer"].isdisjoint(work_identifiers(EXCLUSIONS)), "A selection exclusion was reused")
    previously_recorded = sources["transfer"].intersection(work_identifiers(EXPOSURE_AUDIT))
    before, after = record["baseline"], record["adapted"]
    require(before["points"] == reference["points"], "The earlier baseline points did not reproduce")
    other_groups = {r["group"] for r in reference["held"] if r["family"] == words.parent.LOCAL_FAMILY}
    other_identities = identities(r for r in reference["held"] if r["group"] in other_groups)
    for arm in (before, after):
        require(len(arm["other_generator"]) == len(other_identities) and identities(arm["other_generator"]) == other_identities,
                "The other-generator controls changed")
        for role in SPLITS:
            for reading, frozen in zip(arm[role], roster[role], strict=True):
                require(all(reading[key] == value for key, value in frozen.items() if key != "variant"), "A reading used a different source excerpt")
        for row in arm["calibration"] + arm["held"] + arm["transfer"] + arm["other_generator"]:
            ranges.validate_range_row(row)
        require(arm["points"] == ranges.operating(arm["calibration"]), "Ordinary-only points changed")
    held = ranges.workflow_summary(after["held"], after["points"], before["held"], before["points"]["range"])
    other = ranges.workflow_summary(after["other_generator"], after["points"], before["other_generator"], before["points"]["range"])
    transfer = {genre: ranges.workflow_summary([r for r in after["transfer"] if r["domain"] == genre], after["points"],
        [r for r in before["transfer"] if r["domain"] == genre], before["points"]["range"]) for genre in TRANSFER}
    exposed = {name: ranges.workflow_summary([r for r in after["transfer"] if (r["group"] in previously_recorded) == seen],
        after["points"], [r for r in before["transfer"] if (r["group"] in previously_recorded) == seen],
        before["points"]["range"]) for name, seen in (("previously_recorded", True), ("unlisted", False))}
    joint = ranges.joint_comparison(after, before, "range", before["points"]["range"])
    summary = {"reports": held, "other_generator": other, "joint_reports": joint, "transfer_by_genre": transfer,
               "transfer_by_exposure": exposed, "previously_recorded_transfer_groups": sorted(previously_recorded)}
    require("summary" not in record or record["summary"] == summary, "Reported outcomes do not reproduce")
    return summary


def run(args):
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(4)
    plan = read(args.plan)
    require(plan == freeze(args.source, args.roster, {genre: getattr(args, genre) for genre in TRANSFER}), "The frozen source selection changed")
    base, truth, _ = words.input_plan(args.roster)
    prior = read(BASE)
    require(pool.file_hash(args.encoder_weights) == prior["encoder_sha256"]
            and pool.file_hash(args.baseline) == prior["head_sha256"], "Pinned weights differ")
    tokenizer, encoder, _, device = words.parent.init_encoder(args.encoder)
    parameters = load_file(args.encoder_weights)
    encoder.load_state_dict({name.removeprefix("encoder."): value for name, value in parameters.items()
                             if name.startswith("encoder.")}, strict=True)
    encoder.eval()
    del parameters
    pairs = plan["rows"]["fit_addition"]
    truth.update({b["sha256"]: words.edit_words(a, b) for a, b in zip(pairs[::2], pairs[1::2], strict=True)})
    others = {r["group"] for r in base if r["role"] == "held" and r["family"] == words.parent.LOCAL_FAMILY}
    groups = plan["rows"] | {"other_generator": [r for r in base if r["role"] == "held" and r["group"] in others]}
    old = load_file(args.baseline)
    baseline = ranges.head()
    baseline.load_state_dict({name.removeprefix("head."): value for name, value in old.items() if name.startswith("head.")}, strict=True)
    adapted, mean, scale, coverage = ranges.fit(occurrence.fit_roster(base) + groups["fit_addition"], truth,
                                               tokenizer, encoder, device, paired_margin=True)
    head = args.output / "adapted.safetensors"
    save_file({"mean": mean, "scale": scale, **{f"head.{key}": value for key, value in adapted.state_dict().items()}}, head)
    result = {"contract": contract(), "roster": roster_commitments(plan), "coverage": coverage,
              "encoder_sha256": prior["encoder_sha256"], "baseline_sha256": prior["head_sha256"],
              "adapted_sha256": pool.file_hash(head), "device": device}
    for role in ("held", "transfer"):
        pairs = groups[role]
        truth.update({b["sha256"]: words.edit_words(a, b) for a, b in zip(pairs[::2], pairs[1::2], strict=True)})
    for name, fitted in (("baseline", (baseline.eval(), old["mean"], old["scale"])), ("adapted", (adapted, mean, scale))):
        readings = {"calibration": list(ranges.measure(groups["calibration"], truth, tokenizer, encoder, device, fitted))}
        readings["points"] = ranges.operating(readings["calibration"])
        write(args.output / f"{name}-points.json", readings["points"])
        for role in ("held", "transfer", "other_generator"):
            readings[role] = list(ranges.measure(groups[role], truth, tokenizer, encoder, device, fitted))
        result[name] = readings
    result["summary"] = review(result)
    write(args.output / "receipt.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action, fields in (("freeze", ("source", "roster", "output")), ("review", ("receipt",)),
                           ("run", ("source", "roster", "output", "plan", "encoder", "encoder_weights", "baseline"))):
        command = sub.add_parser(action)
        for field in fields:
            command.add_argument(field, type=Path)
        if action != "review":
            for genre in TRANSFER:
                command.add_argument(f"--{genre}", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        write(args.output, freeze(args.source, args.roster, {genre: getattr(args, genre) for genre in TRANSFER}))
    elif args.action == "run":
        args.output.mkdir(mode=0o700)
        run(args)
    else:
        print(json.dumps(review(read(args.receipt)), sort_keys=True, indent=2))
