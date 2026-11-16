"""Test whether already flagged writing context supports a narrower reading location."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_context_pool_transfer as pool
from scripts.writing_interval_transfer import fingerprints

SCHEMA = "chipmunk.writing-focus-inside-context/1"
PARENT = pool.PANEL_DIR / "writing-context-pool-transfer-2026-11-14.json"
FAMILIES = ("qwen3-8b", "gpt-5.4")
PINS = {
    "news": (
        "e9b830b2e376ac017c944b67bf5620deebf7d9aeec5edecc3734517de2f16a21",
        "a5cdc0a802abf7ac317c91d26159a2491395260fb112f57e40f5d37539944d90",
    ),
    "abstracts": (
        "5c8bf202a9e8173d3156120665d548ce1b32eab628b8be711fc2be7f3bb595d0",
        "e657011ad11434eb9f5860dff3dbe72790843c0dd8626ad7755abcae4c5562a7",
    ),
    "essays": (
        "fc6aeea7cc97634d2d5c303ef0c743878dabfdad3c448253bab460c078f9543a",
        "1e87bd7063aec665bd7f6c5bfc3fcb1ac86507946b2b76dcbbb395a68a7992b8",
    ),
}
METHOD = {
    "sources": "Use Qwen and full GPT-5.4 releases absent from the parent fit, with pinned publisher labels. "
    "Within each domain admit a matched Qwen and GPT-5.4 work at each position in a round, up to three pairs "
    "per position. Retain one unchanged control per work. Carry forward every position with a matched pair. "
    "Exclude earlier recorded work hashes and 12-word phrase overlap against prior and admitted works.",
    "comparison": "On files with AI-like native context, compare the largest eligible context response with "
    "three policies using the parent fitted heads and fixed points. Consider sentences wholly within at least "
    "one AI-like native context. Keep the original findings available for every policy.",
    "pooled": "The combined head must choose a unique highest sentence, exceed the parent's calibration-matched "
    "point and select an eligible sentence.",
    "agreement": "Apply the pooled rule and require the short-context head to choose the same sentence.",
    "separated": "Apply the agreement rule and require the combined winning score to exceed the runner-up by .10, "
    "chosen from the previously exposed position-balanced panel and fixed before this test.",
    "advance": "A policy can be considered for a local trial if it gets five precise locations never chosen by the "
    "native policy on at least four different works with gains on both families, and produces no more new wrong "
    "locations than corrections. It must also have at most two precise focuses on genuine works and leave "
    "every prior finding and confidence value intact. No verdicts or new findings can be emitted.",
}


def read(path):
    return json.loads(path.read_text())


def shard(path, domain, family, expected, *, split="test"):
    if pool.file_hash(path) != expected:
        raise ValueError("Publisher shard differs")
    rows = defaultdict(dict)
    csv.field_size_limit(1024 * 1024)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, strict=True):
            if (row["domain"], row["generator"], row["split"], row["subset"]) != (domain, family, split, "main"):
                raise ValueError("Unexpected publisher source")
            version = row["version"]
            work = rows[row["id"]]
            if version in work:
                raise ValueError("Repeated source work/version")
            work[version] = row
    return rows


def excerpt_options(work_id, versions):
    raw, labels = pool.truth_text(versions["v0"])
    if labels:
        raise ValueError("Genuine source has labels")
    original = list(pool.source_windows(raw))
    result = {}
    for position in (1, 2, 3):
        keys = [pool.signature(raw, spans, position) for spans in original]
        counts = Counter(keys)
        unique = {key: spans for key, spans in zip(keys, original, strict=True) if counts[key] == 1}
        options = pool.candidates(versions, unique, raw, position)
        for key in sorted(options, key=lambda k: pool.sha(work_id + "/" + "/".join(k))):
            edited, spans, version = options[key]
            try:
                seed, a = pool.normalize(raw, unique[key])
                changed, b = pool.normalize(edited, spans)
            except ValueError:
                continue
            if (seed != changed and len({seed[x:y] for x, y in a}) == len({changed[x:y] for x, y in b}) == 5
                    and [seed[x:y] for i, (x, y) in enumerate(a) if i != position]
                    == [changed[x:y] for i, (x, y) in enumerate(b) if i != position]):
                result[position] = ((seed, a, None), (changed, b, version))
                break
    return result


def fresh_cases(paths):
    exposure = {name: pool.PANEL_DIR / name for name in (*pool.EXPOSURE, PARENT.name)}
    excluded = {value for path in exposure.values() for value in re.findall(r'"([a-f0-9]{64})"', path.read_text())}
    sources = {
        domain: [shard(paths[domain, family], domain, family, pin)
                 for family, pin in zip(FAMILIES, PINS[domain], strict=True)]
        for domain in pool.DOMAINS
    }
    prior_phrases = set()
    for tables in sources.values():
        for table in tables:
            for key, rows in table.items():
                if pool.sha(key) in excluded:
                    prior_phrases.update(fingerprints(rows["v0"]["text"]))
    cases, admissions = [], {}
    for domain, tables in sources.items():
        # Find eligible options without using the classifiers, then fill each position by source hash.
        options = [defaultdict(list) for _ in FAMILIES]
        for collection, table in zip(options, tables, strict=True):
            for name in sorted(table, key=lambda key: pool.sha(f"{SCHEMA}/{domain}/{key}")):
                if pool.sha(name) in excluded:
                    continue
                try:
                    positions = excerpt_options(name, table[name])
                except ValueError:
                    continue
                if positions:
                    phrases = fingerprints(table[name]["v0"]["text"])
                    if not phrases.intersection(prior_phrases):
                        for position, versions in positions.items():
                            collection[position].append((name, phrases, versions))
        pairs = Counter()
        order = sorted((1, 2, 3), key=lambda pos: (len(options[0][pos]), pos))
        for _ in range(3):
            for position in order:
                pair = matched_pair(options, position, excluded, prior_phrases)
                if pair is None:
                    continue
                pairs[str(position)] += 1
                for (name, phrases, versions), family in zip(pair, FAMILIES, strict=True):
                    group = pool.sha(name)
                    excluded.add(group)
                    prior_phrases.update(phrases)
                    for (text, ranges, variant), source in zip(versions, ("original", family), strict=True):
                        cases.append({"group": group, "domain": domain, "position": position, "family": source,
                                      "sha256": pool.sha(text), "text": text, "spans": ranges, "variant": variant,
                                      "truth": ranges[position] if variant else None,
                                      "spaces": [i for i, c in enumerate(text) if c.isspace()]})
        if not pairs:
            raise ValueError(f"No matched position pairs in {domain}")
        admissions[domain] = {"pairs_by_position": dict(pairs), "works": sum(pairs.values()) * 2}
        print(domain, admissions[domain], flush=True)
    return cases, admissions, {name: pool.file_hash(path) for name, path in exposure.items()}


def matched_pair(options, position, excluded, prior):
    for first in options[0][position]:
        first_id, phrases, _ = first
        if pool.sha(first_id) in excluded or phrases.intersection(prior):
            continue
        second = next((case for case in options[1][position] if case[0] != first_id
                       and pool.sha(case[0]) not in excluded and not case[1].intersection(phrases, prior)), None)
        if second is not None:
            return first, second
    return None


def read_scores(row, head, mode):
    features = pool.features(row, mode)
    if features is None:
        return []
    b = head["coefficients"]
    return [b[0] + pool.math.fsum(c * (v - m) / s for c, v, m, s in zip(
        b[1:], values, head["mean"], head["scale"], strict=True,
    )) for values in features]


def best(scores):
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return order[0] if len(order) >= 2 and scores[order[0]] > scores[order[1]] else None


def choices(row, heads):
    contexts = pool.baseline(row)
    eligible = [i for i, (start, end) in enumerate(row["spans"][1:4])
                if any(a <= start < end <= b for a, b in contexts)]
    if not eligible:
        return {}
    native = max(eligible, key=lambda i: row["baseline"][i + 1])
    result = {"native": native}
    combined = read_scores(row, heads["combined"], "combined")
    winner = best(combined)
    if winner is None or winner not in eligible or combined[winner] <= heads["combined"]["matched"]:
        return result
    result["pooled"] = winner
    if best(read_scores(row, heads["short"], "short")) != winner:
        return result
    result["agreement"] = winner
    if combined[winner] - max(score for i, score in enumerate(combined) if i != winner) > .10:
        result["separated"] = winner
    return result


def summary(rows, heads):
    totals = {}
    for domain in (*pool.DOMAINS, "total"):
        counts = Counter()
        corrected_works = defaultdict(set)
        selected = (r for r in rows if domain in (r["domain"], "total"))
        for row in selected:
            truth = row["truth"]
            picked = choices(row, heads)
            base = pool.baseline(row)
            counts["edited" if truth else "genuine"] += 1
            counts["existing_edit_flags" if truth else "existing_genuine_flags"] += bool(base)
            counts["existing_useful_contexts"] += any(pool.useful(row, span) for span in base)
            for rule, index in picked.items():
                span = row["spans"][index + 1]
                counts[f"{rule}/used"] += 1
                if truth is None:
                    counts[f"{rule}/genuine_focus"] += 1
                elif span == truth:
                    counts[f"{rule}/exact"] += 1
                    if index != picked["native"]:
                        counts[f"{rule}/corrected"] += 1
                        counts[f"{rule}/gain/{row['family']}"] += 1
                        corrected_works[rule].add(row["group"])
                else:
                    counts[f"{rule}/wrong"] += 1
                    counts[f"{rule}/changed_to_wrong"] += index != picked["native"]
        for rule, works in corrected_works.items():
            counts[f"{rule}/corrected_works"] = len(works)
        totals[domain] = dict(counts)
    counts = totals["total"]
    totals["advance"] = {rule: (
        counts.get(f"{rule}/corrected", 0) >= 5
        and counts.get(f"{rule}/corrected_works", 0) >= 4
        and all(counts.get(f"{rule}/gain/{family}", 0) for family in FAMILIES)
        and counts.get(f"{rule}/changed_to_wrong", 0) <= counts.get(f"{rule}/corrected", 0)
        and counts.get(f"{rule}/genuine_focus", 0) <= 2
    ) for rule in ("pooled", "agreement", "separated")}
    return totals


def validate(rows, freeze):
    matches = Counter()
    excluded = {digest for name in freeze["exposure"] for digest in re.findall(
        r'"([a-f0-9]{64})"', (pool.PANEL_DIR / name).read_text(),
    )}
    for group, pair in pool.group_rows(rows).items():
        if len(pair) != 2 or pair[0]["family"] != "original" or group in excluded:
            raise ValueError("Unexpected or previously measured held work")
        original, edit = pair
        domain, family, position = (edit[key] for key in ("domain", "family", "position"))
        if (domain not in pool.DOMAINS or family not in FAMILIES or position not in (1, 2, 3)
                or (original["domain"], original["position"], original["truth"]) != (domain, position, None)
                or edit["truth"] != edit["spans"][position] or original["sha256"] == edit["sha256"]
                or any("text" in row or len(row["spans"]) != 5 or row["input_sha256"][-1] != row["sha256"]
                       for row in pair)):
            raise ValueError("The source geometry or held labels differ")
        matches[domain, family, position] += 1
    expected = {(domain, family, int(position)): count for domain, info in freeze["admissions"].items()
                for family in FAMILIES for position, count in info["pairs_by_position"].items()}
    if matches != expected:
        raise ValueError("Incomplete held balance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parent = read(PARENT)
    head = parent["parameters"]
    marker = {"schema": SCHEMA, "method": METHOD, "parent_sha256": pool.file_hash(PARENT),
              "runner_sha256": pool.file_hash(Path(__file__)), "native_sha256": pool.file_hash(Path(pool.__file__)),
              "publisher_sha256": PINS}
    if args.mode == "freeze":
        rows, admissions, exposure = fresh_cases({
            (domain, family): args.cache / f"{domain}_{family}.csv" for domain in pool.DOMAINS for family in FAMILIES
        })
        freeze = marker | {"admissions": admissions, "exposure": exposure}
        pool.write(args.path, {"freeze": freeze, "rows": rows})
        return
    packet = read(args.path)
    freeze = packet["freeze"]
    if any(marker[key] != freeze[key] for key in marker if key != "publisher_sha256") or any(
        list(pins) != freeze["publisher_sha256"][domain] for domain, pins in PINS.items()
    ) or any(pool.file_hash(pool.PANEL_DIR / name) != sha for name, sha in freeze["exposure"].items()):
        raise ValueError("Study commitments differ")
    if args.mode == "review":
        rows = packet["rows"]
        validate(rows, freeze)
        if summary(rows, head) != packet["summary"]:
            raise ValueError("Saved localization results differ")
        print(json.dumps(packet["summary"], indent=2))
        return
    if args.output is None or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("--output and offline inference are required")
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    if any(pool.file_hash(pool.SNAPSHOT / file) != sha for file, sha in pool.BASELINE_ASSETS.items()):
        raise ValueError("Pinned model files differ")
    torch.set_num_threads(3)
    clf = FourClassTextClassifier.from_snapshot(pool.SNAPSHOT, "cpu")
    rows = []
    for row in pool.measurements(packet["rows"], clf):
        rows.append(row)
        if len(rows) % 18 == 0:
            print(f"{len(rows)}/{len(packet['rows'])} files scored", flush=True)
    validate(rows, freeze)
    counts = summary(rows, head)
    pool.write(args.output, {"freeze": freeze, "roster_sha256": pool.file_hash(args.path), "rows": rows, "summary": counts})
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
