# SPDX-License-Identifier: MIT
"""Compare draft figure matching on independent rewrites and staged factual edits."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

from app.evidence.document_revision_values import compare, readings

SCHEMA = "chipmunk.draft-values-evaluation/1"
INPUTS = {
    "test.jsonl": (622837, "e8142517c08744d123aa11ad585d0d7b44fb1c384e599b72100d0f656c717f98"),
    "samples.json": (73331, "9ef4749a8d368f92de228bbb513a55c1d9766a7b9df4f7730a91811da6245a56"),
    "results.jsonl": (1716714, "f5b69b75c63e4ac535873b15670824d03ef690da1303cef23328444d8fcc0069"),
    "fact_units.jsonl": (344521, "747477ea96d2f2c5d59d2433cd5e2ddff5a5c417fe6f35f49b462762ce838c6d"),
}
ARMS = {"three_words": 3, "two_words": 2}
METHOD = (
    "Compare only figures with unique exact anchors in both texts. Use three adjacent words on both sides, "
    "or a verified sentence/paragraph end and four words on the other side. Separately try two words on "
    "each side. Count exact source ranges on staged value changes, equivalent formatting, identity and "
    "different articles. Check original figure surfaces against the published altered-unit labels. "
    "An altered-unit label does not specify the rewrite's new position or whether it is a substitution."
)
STAGED_NUMBER = re.compile(r"(?<![\w.,/:+−–-])[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?(?![\w/+:−–-]|[.,][0-9])")
BLANK = re.compile(r"\r?\n[ \t]*\r?\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(folder, name):
    raw = (folder / name).read_bytes()
    require((len(raw), hashlib.sha256(raw).hexdigest()) == INPUTS[name], f"Changed source file {name}")
    return [json.loads(line) for line in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)


def blocks(text):
    boundaries = [0, *(match.end() for match in BLANK.finditer(text)), len(text)]
    return [(start, end) for start, end in zip(boundaries, boundaries[1:]) if start < end]


def label_spans(text, units):
    labels = {}
    for unit in units:
        if unit["group"] != "num":
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(unit['surface'])}(?!\w)", re.I)
        matches = list(pattern.finditer(text))
        if len(matches) == 1:
            labels[matches[0].span()] = "retained" if unit["kept"] else "altered"
    return labels


def case(group, category, before, after, *, truth=(), units=()):
    counts = Counter(unit["kept"] for unit in units if unit["group"] == "num")
    labels = label_spans(before, units)
    spans = {}
    for name, width in ARMS.items():
        pairs = compare(readings(before, blocks(before), words_per_side=width),
                        readings(after, blocks(after), words_per_side=width))
        hits = []
        for old, new in pairs:
            if old.value != new.value:
                # Labels cover just the earlier value. Keep uncertain coverage
                # separate from a label matched at the exact same source range.
                mark = labels.get(old.span, "unlabelled")
                hits.append([list(old.span), list(new.span), mark])
        spans[name] = {"compared": len(pairs), "hits": hits}
    return {"group": group, "category": category, "files": [digest(before), digest(after)],
            "sizes": [len(before), len(after)], "truth": list(truth), "readings": spans,
            "audit_altered": counts[False], "audit_retained": counts[True]}


def staged(original):
    # Ground truth is authored from an independent plain-decimal locator.
    for match in STAGED_NUMBER.finditer(original):
        visible = match.group()
        if len(visible) > 12 or original[match.end():].startswith(("-", "/", ":")):
            continue
        normal = visible.replace(",", "")
        decimals = len(normal.partition(".")[2])
        next_value = f"{Decimal(normal) + Decimal(1).scaleb(-decimals):.{decimals}f}"
        same = visible + ("0" if decimals else ".0")
        a, b = match.span()
        yield "stage_changed", original[:a] + next_value + original[b:], ([a, b], [a, a + len(next_value)])
        yield "stage_format", original[:a] + same + original[b:], ()
        return


def run(folder):
    go = load(folder, "test.jsonl")
    samples = {item["id"]: item for item in load(folder, "samples.json")}
    benchmark = load(folder, "results.jsonl")
    audit = {(item["tool"], item["sample_id"]): item for item in load(folder, "fact_units.jsonl")
             if item["set"] == "commercial" and item["aihb_cycle"] == "2026-10"}
    require(len(go) == 254 and len(samples) == 33 and len(audit) == len(benchmark) == 363, "Source counts differ")
    rows = []
    modern = [item for item in go if item["source"] == "us-federal"]
    require(len(modern) == 54 and len({item["url"] for item in modern}) == 30, "Federal sources differ")
    for item in go:
        federal = item["source"] == "us-federal"
        source = item["url"] if federal else str(item["gutenberg_id"])
        category = ("federal_" if federal else "historic_") + item["generator"]
        rows.append(case(digest(source), category, item["output"], item["input"]))
    for item in modern:
        group = digest(item["url"])
        original = item["output"]
        rows.append(case(group, "same_text", original, original))
        for category, rewritten, truth in staged(original):
            rows.append(case(group, category, original, rewritten, truth=truth))
        other = min((row for row in modern if row["url"] != item["url"]), key=lambda row: digest(item["id"] + row["id"]))
        rows.append(case(group, "different_official_articles", original, other["output"]))
    for item in benchmark:
        tool, sample = item["humanizer"], item["test_id"].rsplit("-", 1)[1]
        unit_row = audit.pop((tool, sample))
        require(unit_row["category"] == item["category"] and samples[sample]["text"] == item["input"]
                and item["cycle_id"] == "2026-10" and item["status"] == "ok", "Mismatched benchmark case")
        rows.append(case(digest(sample), "published_rewrite", item["input"], item["output"], units=unit_row["units"]))
    require(not audit, "Incomplete published judgment coverage")
    packet = {"schema": SCHEMA, "method": METHOD, "inputs": {key: list(value) for key, value in INPUTS.items()}, "rows": rows}
    return packet | {"summary": summarize(packet)}


def summarize(packet):
    require(packet["schema"] == SCHEMA and packet["method"] == METHOD
            and packet["inputs"] == {key: list(value) for key, value in INPUTS.items()},
            "Frozen inputs or method differ")
    total = {}
    groups = {}
    for row in packet["rows"]:
        category = row["category"]
        groups.setdefault(category, set()).add(row["group"])
        require(set(row["readings"]) == set(ARMS) and len(row["sizes"]) == 2, "Case measurements differ")
        targets = row["truth"]
        if targets:
            require(category == "stage_changed" and len(targets) == 2, "Invalid staged comparison")
        for name, reading in row["readings"].items():
            hits = reading["hits"]
            require(len(hits) <= reading["compared"], "Too many paired ranges")
            report = total.setdefault(category, {}).setdefault(name, Counter())
            report.update(files=1, comparable=bool(reading["compared"]), flagged=bool(hits), pairs=len(hits),
                          audit_altered=row["audit_altered"], audit_retained=row["audit_retained"])
            for a, b, label in hits:
                require(all(0 <= start < end <= size for (start, end), size in zip((a, b), row["sizes"], strict=True))
                        and label in {"altered", "retained", "unlabelled"}, "Invalid paired range or judgment")
                report[label] += 1
                if targets:
                    # Full printed spans may also contain a symbol or unit.
                    useful = all(lo <= first < last <= hi for (lo, hi), (first, last) in zip((a, b), targets, strict=True))
                    report.update(staged_exact=useful, staged_mispaired=not useful)
    return {category: {"works": len(groups[category]), **{key: dict(value) for key, value in arms.items()}}
            for category, arms in total.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("run")
    score.add_argument("sources", type=Path)
    score.add_argument("receipt", type=Path)
    review = commands.add_parser("review")
    review.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        packet = run(args.sources)
        args.receipt.write_bytes(gzip.compress(json.dumps(packet, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    else:
        packet = json.loads(gzip.decompress(args.receipt.read_bytes()))
    summary = summarize(packet)
    require(summary == packet["summary"], "The published recount differs")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
