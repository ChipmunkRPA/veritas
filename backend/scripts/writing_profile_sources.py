# SPDX-License-Identifier: MIT
"""Prepare a fresh matched writing panel and genuine out-of-profile controls."""

from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_local_baseline_transfer as paired
from scripts import writing_multi_passage_panel as source

SCHEMA = "chipmunk.writing-profile-sources/1"
PUBLISHER_FILE = "mixed-train.parquet"
PUBLISHER_SHA = "0c6d7436a25b099b807462de394f4ec38595af6cd33e1984e652acd0774d4307"
FRUS_SHA = "29c3e843f74bcaa244ee15af6f5cebd572091051614d554823bebab46b7ca24e"
BOARD_URL = "https://www.federalreserve.gov/monetarypolicy/"
REPORTS = [f"{year}{month}" for year, months in ((2017, "01 03 04 05 07 09 10 11"),
                                               (2018, "01 03 04 05 07 09 10 12")) for month in months.split()]


def order(document):
    return source.digest(f"{SCHEMA}/{document}".encode())


def mixed_panel(path, prior):
    import pyarrow.parquet as pq

    assert source.digest(path.read_bytes()) == PUBLISHER_SHA
    cases = pq.read_table(path, filters=[("source_dataset", "in", sorted(source.GENRES))]).to_pylist()
    groups, references, documents, bodies = prior
    eligible, exclusions = defaultdict(list), Counter()
    for row in sorted(cases, key=lambda row: order(row["id"])):
        try:
            group = source.digest(source._source_id(row).encode())
        except AssertionError:
            exclusions["source_terms_or_identity_unverified"] += 1
            continue
        if group in groups or row["source_reference"] in references or row["id"] in documents:
            exclusions["earlier_source"] += 1
            continue
        if not 1 <= sum(item["label"] == "ai" for item in row["spans"]) <= 2:
            exclusions["passage_count"] += 1
            continue
        assert row["split"] == "train"
        source._validate(row)
        variants = paired.variants(row)
        if variants is None:
            exclusions["native_geometry"] += 1
            continue
        if any(variant["native_text_sha256"] in bodies for variant in variants):
            exclusions["earlier_body"] += 1
            continue
        collection = row["source_dataset"]
        metadata = {key: row[key] for key in ("source_reference", "source_author", "source_title", "source_license",
                                             "generator_backend", "requested_model", "model_identity_status")}
        eligible[group].append(metadata | {"document": row["id"], "split": "test", "publisher_split": "train",
                                           "genre": source.GENRES[collection], "source_collection": collection,
                                           "group_sha256": group, "variants": variants})
    chosen, by_generator = [], Counter()
    # Select one eligible pair per group. Break ties with a fixed hash and use all available generators.
    for group in sorted(eligible, key=order):
        options = eligible[group]
        work = min(options, key=lambda item: (by_generator[item["genre"], item["generator_backend"]], order(item["document"])))
        chosen.append(work)
        by_generator[work["genre"], work["generator_backend"]] += 1
    return chosen, dict(exclusions)


def ordinary(text, document, group, genre, **metadata):
    variants = []
    for layout, raw in (("published", text), ("neutral", paired._neutral(text))):
        variant = source.native(raw, ())
        if variant is None:
            return None
        variants.append(variant | {"layout": layout, "version": "ordinary"})
    return metadata | {"document": document, "split": "test", "group_sha256": source.digest(group.encode()),
                       "genre": genre, "generator_backend": None, "variants": variants}


def archive_panel(path):
    assert source.digest(path.read_bytes()) == FRUS_SHA
    counts, works = Counter(), []
    for key, entry in sorted(source.read(path).items(), key=lambda item: order(item[0])):
        volume, doc = entry["volume"], entry["doc_id"]
        if counts[volume] == 10:
            continue
        body = "\n\n".join(paragraph["text"] for paragraph in entry["paragraphs"])
        work = ordinary(body, key, volume, "diplomatic_correspondence", source_collection="frus",
                        source_reference=f"https://history.state.gov/historicaldocuments/{volume}/{doc}",
                        source_license="U.S. Department of State publications, public domain and CC0")
        if work is not None:
            works.append(work)
            counts[volume] += 1
    assert len(counts) == 4 and sum(counts.values()) == 40
    return works


def report_panel(folder):
    from lxml import html

    works, hashes = [], {}
    for month in REPORTS:
        filename = f"beigebook{month}.htm"
        raw = (folder / filename).read_bytes()
        hashes[filename] = source.digest(raw)
        article, = html.fromstring(raw).xpath('//*[@id="article"]')
        heading, = article.xpath('./h4[normalize-space() = "Federal Reserve Bank of Boston"]')
        paragraphs = []
        for item in heading.itersiblings():
            if item.tag in ("h2", "h3", "h4"):
                break
            if item.tag == "p" and not item.xpath('.//a'):
                # Keep body paragraphs and line breaks. Drop the navigation paragraph after each district.
                paragraphs.append("\n".join(" ".join(line.split()) for line in
                                           item.xpath('string()').splitlines() if line.strip()))
        work = ordinary("\n\n".join(paragraphs), filename, f"beigebook-boston:{month}", "economic_report",
                        source_collection="board_reports", source_reference=BOARD_URL + filename,
                        source_author="Board of Governors of the Federal Reserve System",
                        source_license="Board website public domain, unless otherwise indicated")
        assert work is not None, filename
        works.append(work)
    return works, hashes


def prepare(publisher, archive, previous_calibration, previous_held, output):
    earlier = source.read(previous_calibration)["works"] + source.read(previous_held)["held"]
    identities = ({work[field] for work in earlier} for field in ("group_sha256", "source_reference", "document"))
    prior = (*identities, {variant["native_text_sha256"] for work in earlier for variant in work["variants"]})
    mixed, excluded = mixed_panel(publisher / PUBLISHER_FILE, prior)
    archive_works = archive_panel(archive)
    board_works, hashes = report_panel(publisher)
    works = mixed + archive_works + board_works
    assert not {work["source_reference"] for work in works} & prior[1]
    assert not {work["group_sha256"] for work in works} & prior[0]
    assert not {variant["native_text_sha256"] for work in works for variant in work["variants"]} & prior[3]
    assert len({work["document"] for work in works}) == len(works)
    source.save(output, {"schema": SCHEMA, "dataset": source.DATASET, "revision": paired.PUBLISHER_REVISION,
                         "publisher_files_sha256": {PUBLISHER_FILE: PUBLISHER_SHA, archive.name: FRUS_SHA} | hashes,
                         "excluded_publisher_rows": excluded, "works": works})
    print(json.dumps({"pairs": dict(Counter(work["genre"] for work in mixed)),
                      "publisher_generators": dict(Counter(work["generator_backend"] for work in mixed)),
                      "controls": dict(Counter(work["source_collection"] for work in archive_works + board_works))},
                     sort_keys=True), flush=True)


def verify_counts(receipt):
    """Recount edits with the shared independent matcher's bipartite assignment."""
    from app.bench.independent_locations import _matches

    counts = defaultdict(Counter)
    for work in receipt["checks"]:
        for row in work["versions"]:
            for method, options in row["choices"].items():
                key = f"{work['genre']}/{row['layout']}/{method}"
                ranges = [selection["compact"] for selection in options["ranges"]]
                if row["version"] == "mixed":
                    found = _matches(row["truth"], ranges, .30, .50)
                    counts[key].update(mixed=1, targets=len(row["truth"]), located=found, unmatched=len(ranges) - found)
                else:
                    counts[key].update(ordinary=1, ordinary_flagged=bool(ranges))
                counts[key].update(unassessed=not row["assessed"])
    assert counts.keys() == receipt["summary"].keys()
    for key, counted in counts.items():
        assert all(value == receipt["summary"][key][field] for field, value in counted.items()), key
    return {"verified_summary_strata": len(counts)}


def verify_plan(receipt, plan):
    """Verify the recorded source text and locations without evaluating a model."""
    measured = receipt["source"]
    assert measured["input_plan_sha256"] == source.digest(json.dumps(plan, sort_keys=True).encode())
    counts, model_inputs = Counter(), set()
    checkpoints = {work["document"]: work["versions"] for work in receipt["checks"]}
    for work, reading in zip(plan["works"], measured["works"], strict=True):
        assert work["document"] == reading["document"]
        layouts = defaultdict(set)
        replay = {(item["layout"], item["version"]): item for item in checkpoints[work["document"]]}
        for variant, observed in zip(work["variants"], reading["variants"], strict=True):
            assert all(observed[key] == value for key, value in variant.items() if key != "text")
            body = variant["text"]
            assert variant["native_text_sha256"] == source.digest(body.encode())
            layouts[variant["version"]].add(source.digest(re.sub(r"\s+", "", body).encode()))
            character_positions = [item.start() for item in re.finditer(r"\S", body)]
            assert len(character_positions) == variant["nonspace_characters"]
            sentences = variant["sentences"]
            excerpts = [body[start:end] for start, end, *_ in sentences]
            windows = [" ".join(excerpts[max(i - 1, 0):i + 2]) for i in range(len(sentences))]
            assert observed["inputs_sha256"] == [source.digest(window.encode()) for window in windows]
            model_inputs.update(excerpts)
            model_inputs.update(windows)
            row = replay[(variant["layout"], variant["version"])]
            for options in row["choices"].values():
                for selected in options["ranges"]:
                    first, stop = selected["native"]
                    actual = [bisect_left(character_positions, point) for point in (first, stop)]
                    assert actual == selected["compact"] and 0 <= first < stop <= len(body)
                    counts["source_ranges"] += 1
            counts.update(deliveries=1, context_windows=len(windows), known_passages=len(variant["truth"]))
        assert all(len(digests) == 1 for digests in layouts.values())
    assert len(model_inputs) == measured["input_count"]
    return dict(counts) | {"unique_model_inputs": len(model_inputs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="step", required=True)
    build = subparsers.add_parser("prepare")
    verify = subparsers.add_parser("verify")
    for name in ("publisher", "archive", "previous_calibration", "previous_held", "output"):
        build.add_argument(name, type=Path)
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.publisher, args.archive, args.previous_calibration, args.previous_held, args.output)
    else:
        packet = source.read(args.receipt)
        counts = verify_counts(packet)
        if args.plan is not None:
            counts |= verify_plan(packet, source.read(args.plan))
        print(json.dumps(counts, sort_keys=True))
