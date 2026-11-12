"""Freeze publisher sentence locations and controls for the writing option-order experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lxml import html

from app.detectors.text.segment import spans
from scripts.review_writing_option_order_transfer import publisher_body
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.writing_blind_rewrite_probe import digest, load, require, save, text_hash, words

PUBLISHERS = {
    "faa": ("FAA", "https://www.faa.gov/newsroom/"),
    "usda": ("USDA FSA", "https://www.fsa.usda.gov/news-events/news/"),
}
DATES = {
    "FAA": '//h1/following-sibling::div[contains(@class,"mb-4")][1]',
    "USDA FSA": '//div[contains(@class,"field--name-field-release-date")]/div[contains(@class,"field__item")]',
}
GEOMETRY = {
    "window": "Four consecutive native sentences. Interior sentences have 12 to 40 lexical words.",
    "donor": "First same-release sentence at least two sentence indices beyond the window, with 12 to 40 lexical words and a target word-count ratio of .75 to 1.33.",
    "selection": "First eligible excerpt and donor in publisher order; target alternates between interior sentences for each publisher.",
    "disjointness": "Exclude shared normalized 12-word fragments with the declared retained history and other admitted releases.",
}


def sentences(value):
    return [(start, end, value[start:end]) for start, end, _ in spans(value, [(0, len(value))])]


def compatible(sentences_expected):
    return [value for _, _, value in sentences(" ".join(sentences_expected))] == sentences_expected


def possibilities(body, index):
    records = sentences(body)
    sizes = [len(words(value)) for _, _, value in records]
    for start in range(len(records) - 3):
        if not all(12 <= size <= 40 for size in sizes[start + 1:start + 3]):
            continue
        window = records[start:start + 4]
        excerpt = [value for _, _, value in window]
        if not compatible(excerpt):
            continue
        target_size = sizes[start + index]
        for donor, size in enumerate(sizes):
            if start - 1 <= donor <= start + 4 or not 12 <= size <= 40 or not .75 <= size / target_size <= 1.33:
                continue
            replacement = records[donor][2]
            revised = [*excerpt[:index], replacement, *excerpt[index + 1:]]
            if replacement not in excerpt and compatible(revised):
                yield start, window, donor, records[donor]


def benchmark_texts(path):
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("lang", "eng") == "eng" and isinstance(row.get("text"), str) and row["text"].strip():
                require(row.get("text_sha256", text_hash(row["text"])) == text_hash(row["text"]),
                        "A historical text row changed")
                yield row["text"]


def prepare(run, history):
    urls = load(run / "urls.json")
    inventories, candidates, rejected = {}, {}, {}
    for group, url in urls.items():
        prefix = group.split("-", 1)[0]
        if prefix == "gao":
            rejected[group] = "Publisher returned HTTP 403 during initial intake"
            continue
        publisher, base = PUBLISHERS[prefix]
        require(url.startswith(base) and group.isascii() and group.replace("-", "").isalnum(), "Unexpected publisher source")
        path = run / f"{group}.html"
        tree = html.fromstring(path.read_text())
        (title,) = tree.xpath('//h1[contains(@class,"node__title") or @class="margin-0"]')
        (date,) = tree.xpath(DATES[publisher])
        published = " ".join(date.text_content().split())
        require(published.endswith(("2019", "2020")), "Unexpected release date")
        body = publisher_body(path, publisher)
        inventories[group] = {
            "publisher": publisher, "url": url, "title": " ".join(title.text_content().split()),
            "date": published, "html_sha256": digest(path), "normalized_body_sha256": text_hash(body),
        }
        for index in (1, 2):
            candidates[group, index] = list(possibilities(body, index))

    signatures = {
        (group, index, n): fingerprints(" ".join(value for _, _, value in window)) | fingerprints(donor[2])
        for (group, index), choices in candidates.items()
        for n, (_, window, _, donor) in enumerate(choices)
    }
    history_packets = [(name, digest(path), None, benchmark_texts(path)) for name, path in history]
    compared, external, _ = overlap(history_packets, signatures)
    admitted, originals, used = {}, [], set()
    for group, publisher_source in inventories.items():
        publisher = publisher_source["publisher"]
        index = 1 + sum(source["publisher"] == publisher for source in admitted.values()) % 2
        for n, (start, window, donor_index, donor) in enumerate(candidates[group, index]):
            key = group, index, n
            if key in external or signatures[key] & used:
                continue
            used.update(signatures[key])
            anchors = [
                {"publisher_codepoints": [a, b], "sentence_index": number, "sha256": text_hash(value)}
                for number, (a, b, value) in zip([*range(start, start + 4), donor_index], [*window, donor], strict=True)
            ]
            admitted[group] = {**publisher_source, "anchors": anchors, "target_index": index}
            originals.append({"group": group, "sentences": [value for _, _, value in window],
                              "donor": donor[2], "target_index": index})
            break
        else:
            rejected[group] = "No non-overlapping four-sentence excerpt and compatible same-release donor"
    counts = {publisher: sum(source["publisher"] == publisher for source in admitted.values())
              for publisher, _ in PUBLISHERS.values()}
    require(all(count >= 4 for count in counts.values()), "Fewer than four source groups for a publisher")
    save(run / "sources.json", {
        "sources": admitted, "rejected": rejected, "geometry": GEOMETRY,
        "preparation_sha256": digest(Path(__file__)), "history": compared,
        "prior_source_packets": len(compared), "prior_text_entries": sum(row["text_entries"] for row in compared),
        "shared_admitted_12_word_fragments": 0,
        "rights": "Federal government-duty release prose, excluding quotes, captions, and contacts.",
        "rights_urls": ["https://www.usa.gov/government-copyright", "https://www.nal.usda.gov/programs/speccoll"],
    })
    with (run / "originals.jsonl").open("x") as stream:
        for row in originals:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(counts, len(compared), sum(row["text_entries"] for row in compared), rejected, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--history", action="append", required=True, metavar="NAME=JSONL")
    args = parser.parse_args()
    history = [item.partition("=") for item in args.history]
    require(all(name and equals and path for name, equals, path in history), "Use NAME=JSONL for each benchmark packet")
    require(len({name for name, _, _ in history}) == len(history), "History packet names must be unique")
    os.umask(0o077)
    prepare(args.directory, [(name, Path(path)) for name, _, path in history])


if __name__ == "__main__":
    main()
