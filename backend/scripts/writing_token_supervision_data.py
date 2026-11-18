# SPDX-License-Identifier: Apache-2.0
"""Freeze verified writer, source and edit positions for a word-supervised comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from scripts import writing_entropy_sources as exposure
from scripts import writing_multi_passage_panel as native
from scripts import writing_report_sources as reports
from scripts import writing_revision_trajectory_transfer as publisher

SCHEMA = "chipmunk.writing-token-supervision-inputs/1"
STAGES = {"v1": "polish", "v2": "paraphrase", "v3": "style", "v4": "compress"}
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*", re.UNICODE)
SHARDS = (
    ("held", "test", "qwen3-8b", 35, publisher.SHARDS["qwen3-8b"]),
    ("held", "test", "gemini-2.5-flash", 96, reports.OPAI_SHARDS["gemini-2.5-flash"]),
    ("calibration", "dev", "gpt-5.4-nano", 96, "d62fb56d81ee6e4d0ce34a116770b86af910308fd7102bef5e79036dd87e9767"),
    ("fit", "train", "gpt-5.4-nano", 400, reports.OPAI_EXTRA),
    ("fit", "train", "gpt-5.4", 400, "a205bd7102225997b9e2245589cdf416e42499677b31f76382e28bd1b0b9fa68"),
)


def group_key(identifier):
    return native.digest(identifier.encode())


def order(label):
    return group_key(f"{SCHEMA}/{label}")


def raw_sentence_bounds(text, sentences):
    cursor = 0
    for sentence in sentences:
        start = text.index(sentence, cursor)
        cursor = start + len(sentence)
        yield start, cursor


def changed_words(before, after, start, prefix):
    original = [word.group().casefold() for word in WORD.finditer(before)]
    revised = list(WORD.finditer(after))
    delta = SequenceMatcher(None, original, [word.group().casefold() for word in revised], autojunk=False)
    return [[prefix[start + word.start()], prefix[start + word.end()]]
            for operation, _, _, lo, hi in delta.get_opcodes() if operation in {"replace", "insert"}
            for word in revised[lo:hi]]


def pair(original, edited):
    before, old_labels = publisher.publisher_case(original)
    after, _ = publisher.publisher_case(edited)
    if old_labels:
        raise ValueError("The unedited publisher record has edited positions")
    sources, replacements = (json.loads(row["sentences"]) for row in (original, edited))
    changed = json.loads(edited["sentence_labels"])
    if len(sources) != len(replacements):
        return None
    prefix = native._prefix(after)
    truth, words = [], []
    for earlier, latest, label, (a, b) in zip(sources, replacements, changed, raw_sentence_bounds(after, replacements), strict=True):
        if native.compact(earlier) == native.compact(latest):
            continue
        if not label:
            return None
        truth.append([a, b])
        words.extend(changed_words(earlier, latest, a, prefix))
    if not 1 <= len(truth) < len(sources) or not words:
        return None
    clean, mixed = native.native(before, ()), native.native(after, truth)
    if clean is None or mixed is None:
        return None
    return [clean | {"kind": "ordinary", "changed_words": []}, mixed | {"kind": "mixed", "changed_words": words}]


def shard(path, split, generator, expected):
    if publisher.digest(path) != expected:
        raise ValueError("The report shard differs from the pinned publisher material")
    records = defaultdict(dict)
    csv.field_size_limit(1024 * 1024)
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, strict=True):
            if row["version"] not in {"v0", *STAGES}:
                continue
            identity, version = row["id"], row["version"]
            if ((row["split"], row["domain"], row["generator"], row["subset"]) != (split, "reports", generator, "main")
                    or re.fullmatch(r"GAO_[A-Z]+-[0-9A-Z-]+", identity) is None or version in records[identity]
                    or row["edit_operation"] != STAGES.get(version, "none")):
                raise ValueError("Unexpected source grouping, operation or duplicate version")
            records[identity][version] = {k: row[k] for k in (
                "text", "text_tagged", "ai_spans_char", "sentences", "sentence_labels", "record_id",
            )}
    return records


def qualification(archives, document, versions):
    authority = archives.get(document.removeprefix("GAO_"))
    if (authority is None or "v0" not in versions or authority["published_date"] > "2020-12-31"
            or native.compact(authority["text"]) != native.compact(versions["v0"]["text"])):
        return None
    return authority


def overlap_keys(text):
    words = re.findall(r"\w+", text.casefold())
    return {group_key(" ".join(words[i:i + 24])) for i in range(len(words) - 23)}


def prepare(shards, archive, earlier, output, reference=None):
    if publisher.digest(archive) != reports.GAO_SHA256:
        raise ValueError("The verified archive differs from the government text source")
    archives = native.read(archive)["reports"]
    frozen = native.read(reference) if reference else None
    history, old_digests, commitments = exposure.exposures(frozen["prior_exposures"] if frozen else None)
    transfer = native.read(earlier)
    retained = [work for work in transfer["works"] if work["split"] == "test" and work["genre"] != "gao_report"]
    external = {work["group_sha256"]: work for work in sorted(retained, key=lambda row: order(row["document"]))}
    if not external:
        raise ValueError("The earlier cross-genre transfer panel is missing")
    groups, texts, phrases = set(), set(), set()
    plans = defaultdict(list)
    for role, split, generator, maximum, digest in SHARDS:
        name = f"{split}-reports_{generator}.csv"
        rows = shard(shards / name, split, generator, digest)
        observed = Counter()
        selected = 0
        for identity in sorted(rows, key=lambda key: order(f"{role}/{generator}/{key}")):
            if selected == maximum:
                break
            source = rows[identity]
            authority = qualification(archives, identity, source)
            group = group_key(identity)
            if not authority or group in groups:
                continue
            original = source["v0"]
            exposed = (identity.removeprefix("GAO_") in history or group in old_digests
                       or group_key(original["text"]) in old_digests)
            if role == "held" and generator == "gemini-2.5-flash" and exposed:
                continue
            phrases_in_source = overlap_keys(original["text"])
            if phrases_in_source & phrases:
                observed["shared_passage"] += 1
                continue
            options = sorted(STAGES, key=lambda version: (observed[version], order(f"{identity}/{version}")))
            for stage in options:
                if stage not in source:
                    continue
                try:
                    cases = pair(original, source[stage])
                except ValueError:
                    observed["inconsistent_publisher_labels"] += 1
                    continue
                if not cases:
                    continue
                checksums = {case["native_text_sha256"] for case in cases}
                if len(checksums) != 2 or checksums & texts:
                    continue
                if role == "held" and generator == "gemini-2.5-flash" and checksums & old_digests:
                    continue
                plans[role].append({"group": group, "document": identity, "generator": generator,
                                    "genre": "gao_report", "stage": stage, "operation": STAGES[stage],
                                    "original_source_sha256": authority["source_record_sha256"], "publisher_shard": name,
                                    "publisher_records": [original["record_id"], source[stage]["record_id"]],
                                    "prior_project_exposure": exposed or bool(checksums & old_digests), "cases": cases})
                groups.add(group)
                texts.update(checksums)
                phrases.update(phrases_in_source)
                observed[stage] += 1
                selected += 1
                break
        print(role, generator, selected, dict(observed), flush=True)

    for work in external.values():
        cases = [version | {"kind": version["version"], "changed_words": []}
                 for version in work["variants"] if version["layout"] == "published"]
        inputs = {case["native_text_sha256"] for case in cases}
        if not cases or work["group_sha256"] in groups or inputs & texts:
            raise ValueError("A cross-genre file comes from a source reserved in another role")
        plans["held"].append({"group": work["group_sha256"], "document": work["document"],
                              "generator": work.get("generator_backend"), "genre": work["genre"],
                              "operation": "retained_rewrite", "prior_project_exposure": True, "cases": cases})
        groups.add(work["group_sha256"])
        texts.update(inputs)
    if len(plans["fit"]) < 600 or len(plans["calibration"]) < 64:
        raise ValueError("Too few separated fitting and genuine calibration works")
    if sum(work["generator"] == "gemini-2.5-flash" for work in plans["held"]) < 48:
        raise ValueError("Too few fresh valid reports from the held generator")
    output.mkdir(mode=0o700)
    specs = {f"{split}-reports_{gen}.csv": pin for _, split, gen, _, pin in SHARDS}
    for role, works in plans.items():
        path = output / f"{role}.json.gz"
        native.save(path, {"schema": SCHEMA, "role": role, "works": works, "shards": specs,
                   "archive_sha256": reports.GAO_SHA256, "earlier_transfer_sha256": publisher.digest(earlier),
                   "prior_exposures": commitments if role == "held" else {}})
        if frozen and publisher.digest(path) != frozen["plans"][role]:
            raise ValueError("A rebuilt roster differs from the frozen source plan")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("shards", "archive", "earlier", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reference", type=Path)
    prepare(**vars(cli.parse_args()))
