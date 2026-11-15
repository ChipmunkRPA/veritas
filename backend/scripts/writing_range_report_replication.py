# SPDX-License-Identifier: MIT
"""Replicate the frozen writing range checks on new publisher report works."""

import argparse
import re
from collections import Counter
from pathlib import Path

from scripts import writing_edit_occurrence_transfer as occurrence
from scripts import writing_supervised_ranges as ranges
from scripts import writing_word_localization as words

SCHEMA = "chipmunk.writing-range-report-replication/1"
FAMILY = "gpt-5.4"
PER_POSITION = 16
SOURCE_SHA = words.parent.previous.REPORT_PINS[1]
EXPOSURE = (
    "text-opai-bench-intake-complete-v257.json",
    "text-opai-bench-score-complete-v260.json",
    "writing-sentence-probe-2026-11-14.json",
    "writing-topic-response-transfer-2026-11-14.json",
    "writing-matched-encoder-2026-11-14.json",
)
RULE = {"selection": "Select 16 new source works per interior edit position by source hash, before scoring. "
                     "Exclude recorded work hashes and overlapping normalized 12-word phrases.",
        "measurement": "Use both frozen weights and the earlier report operating points, including the combined ordinary-work ranks.",
        "input": "Select a single isolated publisher rewrite and its exact earlier version, with four unchanged adjacent sentences."}
pool = words.parent.pool


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def make_plan(source, native):
    from scripts.writing_interval_transfer import fingerprints

    exposed_native, _, _ = words.input_plan(native)
    fingerprints_seen = {piece for row in exposed_native for piece in fingerprints(row["text"])}
    groups = {row["group"] for row in exposed_native}
    exposure = {}
    for name in EXPOSURE:
        path = pool.PANEL_DIR / name
        exposure[name] = pool.file_hash(path)
        groups.update(re.findall(r'"([a-f0-9]{64})"', path.read_text()))
    table = words.parent.load_shard(source, "test", FAMILY, SOURCE_SHA)
    for key, versions in table.items():
        if pool.sha(key) in groups and "v0" in versions:
            fingerprints_seen.update(fingerprints(versions["v0"]["text"]))
    order = sorted(table, key=lambda key: pool.sha(f"{SCHEMA}/{key}"))
    rows, count = [], Counter()
    for key in order:
        if len(rows) == PER_POSITION * 3 * 2:
            break
        group = pool.sha(key)
        variants = table[key]
        if group in groups or "v0" not in variants:
            continue
        content = set(fingerprints(variants["v0"]["text"]))
        if content & fingerprints_seen:
            continue
        try:
            options = words.parent.intake.excerpt_options(key, variants)
        except (KeyError, ValueError):
            continue
        available = [pos for pos in options if count[pos] < PER_POSITION]
        if not available:
            continue
        position = min(available, key=lambda pos: (count[pos], pos))
        choices = options[position]
        excerpt_pieces = {piece for text, _, _ in choices for piece in fingerprints(text)}
        if excerpt_pieces & fingerprints_seen:
            continue
        for (text, spans, variant), family in zip(choices, ("original", FAMILY), strict=True):
            rows.append({"role": "replication", "group": group, "text": text,
                         "family": family, "sha256": pool.sha(text), "spans": spans, "position": position, "variant": variant})
        fingerprints_seen.update(content | excerpt_pieces)
        count[position] += 1
    require(count == {i: PER_POSITION for i in (1, 2, 3)}, "There were too few unused source works at a required position")
    return {"schema": SCHEMA, "rule": RULE, "source_sha256": SOURCE_SHA,
            "native_roster_sha256": pool.file_hash(native), "prior_exposure": exposure, "rows": rows}


def review(record, report):
    primary = ranges.spans.read(report)
    reference = ranges.spans.read(ranges.REFERENCE)
    rows, old = record["held"], record["incumbent"]
    require(record["schema"] == SCHEMA and record["rule"] == RULE and record["source_sha256"] == SOURCE_SHA
            and record["native_roster_sha256"] == ranges.spans.read(words.RECEIPT)["roster_sha256"]
            and record["candidate_sha256"] == words.receipt_digest(primary)
            and record["baseline_sha256"] == words.receipt_digest(reference), "The frozen models or source pin differs")
    require(record["prior_exposure"] == {name: pool.file_hash(pool.PANEL_DIR / name) for name in EXPOSURE},
            "The prior work exclusion receipts differ")
    require(len(rows) == len(old) == 6 * PER_POSITION, "A held pair is missing")
    require(Counter(row["position"] for row in rows if row["family"] == FAMILY) == dict.fromkeys((1, 2, 3), PER_POSITION),
            "A held position count differs")
    groups = set()
    for row in rows:
        ranges.validate_range_row(row)
    for earlier, later in zip(rows[::2], rows[1::2], strict=True):
        require((earlier["family"], later["family"]) == ("original", FAMILY)
                and earlier["group"] == later["group"] and earlier["group"] not in groups
                and earlier["position"] == later["position"] and not earlier["changed"], "Invalid matching group")
        groups.add(earlier["group"])
    counts = ranges.workflow_summary(rows, primary["points"], old, reference["points"]["edit_and_origin"])
    joint = ranges.joint_comparison({"calibration": primary["calibration"], "held": rows},
                                    {"calibration": reference["calibration"], "held": old},
                                    "edit_and_origin", reference["points"]["edit_and_origin"])
    result = {"ranges": counts, "joint": joint}
    require("results" not in record or record["results"] == result, "Replication counts do not reproduce")
    return result


def run(args):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    primary_file = args.fitted / "receipt.json"
    ranges.review(primary_file)
    primary = ranges.spans.read(primary_file)
    incumbent = ranges.spans.read(ranges.REFERENCE)
    require(pool.file_hash(args.encoder_weights) == primary["encoder_sha256"]
            and pool.file_hash(args.context_head) == incumbent["head_sha256"], "A pinned weight file differs")
    weights = args.fitted / "range.safetensors"
    require(pool.file_hash(weights) == primary["head_sha256"], "The range weights differ")
    fresh = make_plan(args.source, args.roster)
    words.parent.previous.write(args.output / "frozen.json", fresh)
    native = fresh["rows"]
    truth = {after["sha256"]: words.edit_words(before, after) for before, after in zip(native[::2], native[1::2], strict=True)}
    tokenizer, encoder, _, device = words.parent.init_encoder(args.encoder)
    contents = load_file(args.encoder_weights)
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in contents.items() if k.startswith("encoder.")}, strict=True)
    encoder.eval()
    del contents
    values = load_file(weights)
    model = ranges.head()
    model.load_state_dict({k.removeprefix("head."): v for k, v in values.items() if k.startswith("head.")}, strict=True)
    actual = list(ranges.measure(native, truth, tokenizer, encoder, device, (model.eval(), values["mean"], values["scale"])))
    old = list(occurrence.measure(native, truth, tokenizer, encoder, device,
                                  occurrence.load_fitted_heads(args.context_head), context_response=True))
    record = {key: value for key, value in fresh.items() if key != "rows"}
    record.update(candidate_sha256=words.receipt_digest(primary), baseline_sha256=words.receipt_digest(incumbent),
                  held=actual, incumbent=old,
                  variants=dict(Counter(row["variant"] for row in native if row["variant"] is not None)))
    record["results"] = review(record, primary_file)
    words.parent.previous.write(args.output / "receipt.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("output", type=Path)
    for option in ("report", "source", "roster", "encoder", "encoder-weights", "context-head", "fitted"):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        import json

        if args.report is None:
            parser.error("Review requires --report")
        print(json.dumps(review(ranges.spans.read(args.output), args.report), sort_keys=True, indent=2))
    else:
        missing = [name for name in ("source", "roster", "encoder", "encoder-weights", "context-head", "fitted")
                   if getattr(args, name.replace("-", "_")) is None]
        if missing:
            parser.error("Missing options: " + ", ".join(f"--{name}" for name in missing))
        args.output.mkdir(mode=0o700)
        run(args)
