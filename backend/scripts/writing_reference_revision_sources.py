"""Freeze F1000Research version pairs before running any writing model."""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import sleep
from urllib.parse import urlencode

from lxml import etree

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from scripts.prepare_writing_option_order_transfer import benchmark_texts
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.writing_agreement_revision_sources import read, require, retrieve, save
from scripts.writing_blind_rewrite_probe import distance, words
from scripts.writing_focus_v3 import digest

SCHEMA = "chipmunk.writing-reference-revision-sources/1"
PLAN = {
    "schema": SCHEMA,
    "publisher": "F1000Research",
    "catalog": "https://api.crossref.org/journals/2046-1402/works",
    "query": {"filter": "from-pub-date:2017-01-01,until-pub-date:2017-06-30", "rows": 500,
              "sort": "published", "order": "asc", "select": "DOI,link,license,published,container-title"},
    "pool": 28,
    "admit_at_most": 12,
    "license": "https://creativecommons.org/licenses/by/4.0/",
    "selection": "Complete Crossref result. Keep version-2 DOI records with the published CC BY 4.0 term. "
                 "Sort by SHA-256 of the version-2 DOI and cache the first 28 version pairs. Use both publisher "
                 "XML versions with the same author list and pre-July-2017 online dates. Use original body "
                 "prose only. Exclude quoted, imported or separately marked third-party content.",
    "geometry": "First source-ordered seven-sentence excerpt with three exact preceding and following sentences, "
                "one uniquely anchored interior lexical edit and a 12–40-word target in both versions. "
                "No cross-section, figure, table, list, block quote, or third-party inline range. "
                "Check 12-token overlap with the frozen prior panels and the other chosen article groups.",
}
EXPANDED = {
    **PLAN,
    "pool": 70,
    "skip_first": 28,
    "selection": "Freeze the remaining 70 CC BY version pairs after excluding the first 28 source-only scout groups. "
                 "Keep the strict author, date and publisher metadata commitments.",
    "geometry": "First source-ordered five-sentence excerpt with two exact preceding and two exact following "
                "sentences and one interior lexical revision. Omit only JATS bibliographic cross-references and "
                "record exact normalized body blocks after this transformation. Preserve the other rights and "
                "overlap checks from the strict source scout.",
}
URL = re.compile(r"https://f1000research\.com/articles/[0-9]+-[0-9]+/v2/xml")
DOI = re.compile(r"10\.12688/f1000research\.[0-9]+\.2")
INLINE = {"bold", "italic", "monospace", "sup", "sub", "underline"}
LINK = "{http://www.w3.org/1999/xlink}href"
AUTHOR_YEAR = re.compile(r"\([^()]{0,160}\b(?:18|19|20)\d{2}[a-z]?[^()]{0,40}\)")


def license_match(value):
    return isinstance(value, str) and value.replace("http://", "https://").rstrip("/") == PLAN["license"].rstrip("/")


def chosen(catalog, plan):
    record = read(catalog)["message"]
    rows = record["items"]
    require(len(rows) == record["total-results"] and len(rows) <= PLAN["query"]["rows"], "Catalog is incomplete")
    pool = {}
    for row in rows:
        doi = row["DOI"]
        if not DOI.fullmatch(doi) or not any(license_match(lic["URL"]) for lic in row.get("license", [])):
            continue
        xml = [link["URL"] for link in row.get("link", [])
               if link["content-type"] == "application/xml" and URL.fullmatch(link["URL"])]
        require(len(xml) == 1 and row["container-title"] == [PLAN["publisher"]], "Catalog did not bind a version")
        require(doi not in pool and len(row["published"]["date-parts"][0]) == 3, "Duplicate or undated catalog item")
        pool[doi] = {"after": xml[0], "before": xml[0].replace("/v2/", "/v1/"), "date": row["published"]["date-parts"][0]}
    start = plan.get("skip_first", 0)
    stop = start + plan["pool"]
    require(len(pool) >= stop, "Insufficient catalog version pairs")
    return {doi: pool[doi] for doi in sorted(pool, key=text_hash)[start:stop]}


def fetch(run, source_plan):
    publisher = run / "publisher"
    publisher.mkdir(exist_ok=True)
    plan = run / "source-plan.json"
    if not plan.exists():
        save(plan, source_plan)
    require(read(plan) == source_plan, "Source plan changed")
    url = PLAN["catalog"] + "?" + urlencode(PLAN["query"])
    catalog = retrieve(publisher, "catalog.json", url)
    pool = chosen(catalog, source_plan)
    lock = {"source_plan_sha256": digest(plan), "catalog_sha256": digest(catalog), "pool": pool}
    path = run / "catalog-freeze.json"
    if not path.exists():
        save(path, lock)
    require(read(path) == lock, "Catalog selection changed")
    for doi in sorted(pool, key=text_hash):
        key = doi.split(".")[-2]
        for version in ("before", "after"):
            name = f"{key}-{version}.xml"
            if not (publisher / name).exists():
                retrieve(publisher, name, pool[doi][version])
                sleep(.2)
        print(f"Cached pair {key}", flush=True)


def article(path, doi):
    root = etree.fromstring(path.read_bytes(), parser=etree.XMLParser(resolve_entities=False, no_network=True))
    meta = root.find("front/article-meta")
    require(root.tag == "article" and root.get("{http://www.w3.org/XML/1998/namespace}lang") == "en"
            and meta is not None and root.findtext("front/journal-meta/journal-title-group/journal-title") == PLAN["publisher"]
            and meta.findtext("article-id[@pub-id-type='doi']") == doi, "Publisher version or language differs")
    terms = meta.findall("permissions/license")
    require(len(terms) == 1 and license_match(terms[0].get(LINK)), "Publisher XML lacks the exact commercial-use license")
    date = meta.find("pub-date[@pub-type='epub']")
    if date is None:
        date = meta.find("pub-date[@date-type='pub']")
    require(date is not None, "Publisher XML has no online publication date")
    published = [int(date.findtext(key)) for key in ("year", "month", "day")]
    contributors = [" ".join(" ".join(author.itertext()).split())
                    for author in meta.findall("contrib-group/contrib[@contrib-type='author']/name")]
    require(contributors and all(contributors), "Publisher XML does not identify contributors")
    return root.find("body"), contributors, published


def blocks(body, expanded):
    if body is None:
        return []
    if expanded:
        for reference in list(body.iter("xref")):
            if reference.get("ref-type") != "bibr":
                continue
            parent, previous, tail = reference.getparent(), reference.getprevious(), reference.tail or ""
            if previous is None:
                parent.text = (parent.text or "") + tail
            else:
                previous.tail = (previous.tail or "") + tail
            parent.remove(reference)
    collected = []
    sections = [section for section in body.iter("sec")
                if all(node.tag in {"article", "body", "sec"} for node in section.iterancestors())]
    for section in (body, *sections):
        paragraphs = []
        for item in section:
            if item.tag == "p" and all(node.tag in INLINE for node in item.iterdescendants()):
                value = " ".join("".join(item.itertext()).split())
                if value and not any(char in value for char in ('"', '“', '”', '«', '»')):
                    paragraphs.append(value)
                    continue
            if paragraphs:
                collected.append("\n\n".join(paragraphs))
                paragraphs = []
        if paragraphs:
            collected.append("\n\n".join(paragraphs))
    return collected


def windows(body, expanded):
    sides = 2 if expanded else 3
    for block_index, prose in enumerate(blocks(body, expanded)):
        doc = from_string(prose)
        units = segment(doc)
        for index in range(sides, len(units) - sides):
            sample = units[index - sides:index + sides + 1]
            focal = sample[sides]
            word_limits = (55, 200) if expanded else (100, 245)
            if not 12 <= focal.n_words <= 40 or not word_limits[0] <= sum(unit.n_words for unit in sample) <= word_limits[1]:
                continue
            if not all(len(unit.text) > 6 and unit.text[-1] in ".?!" for unit in sample):
                continue
            start = sample[0].start
            text = doc.text[start:sample[-1].end]
            flanks = tuple(unit.text for unit in (*sample[:sides], *sample[sides + 1:]))
            yield flanks, {"text": text, "focus": [focal.start - start, focal.end - start],
                           "focus_text": focal.text, "block": block_index,
                           "block_span": [start, sample[-1].end], "block_sha256": text_hash(doc.text)}


def excerpt(left, right, expanded):
    before, after = defaultdict(list), defaultdict(list)
    for body, index in ((left, before), (right, after)):
        for key, value in windows(body, expanded):
            index[key].append(value)
    for key, rows in after.items():
        if len(rows) != 1 or len(before[key]) != 1:
            continue
        old, new = before[key][0], rows[0]
        a, b = (AUTHOR_YEAR.sub("", item["focus_text"]) for item in (old, new))
        word_lengths = len(words(a)), len(words(b))
        if min(word_lengths) < 1 or not .60 <= word_lengths[1] / word_lengths[0] <= 1.5:
            continue
        if distance(a, b) * max(word_lengths) < 2 - 1e-12:
            continue
        yield old, new


def prepare(run, histories, source_plan):
    from datetime import date

    plan, catalog = run / "source-plan.json", run / "publisher/catalog.json"
    frozen = read(run / "catalog-freeze.json")
    require(read(plan) == source_plan and frozen == {
        "source_plan_sha256": digest(plan), "catalog_sha256": digest(catalog), "pool": chosen(catalog, source_plan),
    }, "Catalog or plan changed")
    require(histories and len(histories) == len({name for name, _ in histories}), "Supply distinct frozen overlap inputs")
    eligible, signatures = {}, {}
    ledger, selected, dispositions = {}, [], Counter()
    for doi, entry in sorted(frozen["pool"].items(), key=lambda item: text_hash(item[0])):
        key = doi.split(".")[-2]
        paths = [run / "publisher" / f"{key}-{variant}.xml" for variant in ("before", "after")]
        ledger[key] = {side: digest(path) for side, path in zip(("before", "after"), paths, strict=True)}
        try:
            old, new = (article(path, doi[:-1] + version) for path, version in zip(paths, ("1", "2"), strict=True))
        except (ValueError, etree.XMLSyntaxError):
            dispositions["unsupported_publisher_metadata"] += 1
            continue
        if old[1] != new[1] or not (date(*old[2]) < date(*new[2]) < date(2017, 7, 1)) or new[2] != entry["date"]:
            dispositions["author_or_date_mismatch"] += 1
            continue
        candidates = list(excerpt(old[0], new[0], "skip_first" in source_plan))
        if not candidates:
            dispositions["no_qualified_exact_edit"] += 1
            continue
        eligible[key] = [{"group": key, "before": before, "after": after, "after_doi": doi, "urls": entry,
                          "before_date": old[2], "after_date": new[2], "attribution": old[1], "license": PLAN["license"]}
                         for before, after in candidates]
        for index, (before, after) in enumerate(candidates):
            signatures[key, index] = fingerprints(before["text"]) | fingerprints(after["text"])
    packets = [(name, digest(path), None, benchmark_texts(path)) for name, path in histories]
    history, external, _ = overlap(packets, signatures)
    retained, authors = set(), set()
    for key, candidates in eligible.items():
        if len(selected) == source_plan["admit_at_most"]:
            dispositions["eligible_after_quota"] += 1
            continue
        for index, item in enumerate(candidates):
            identity = key, index
            contributors = {name.casefold() for name in item["attribution"]}
            if identity not in external and not retained & signatures[identity] and not authors & contributors:
                selected.append(item)
                retained.update(signatures[identity])
                authors.update(contributors)
                dispositions["selected"] += 1
                break
        else:
            dispositions["author_or_text_overlap"] += 1
    require(len(ledger) == source_plan["pool"], "Missing publisher records")
    return selected, {"schema": SCHEMA, "catalog_freeze_sha256": digest(run / "catalog-freeze.json"), "history": history,
                      "source_code_sha256": digest(Path(__file__)),
                      "publisher_xml": ledger, "dispositions": dict(dispositions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fetch", "prepare", "verify"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--history", action="append", default=[])
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    source_plan = EXPANDED if args.expanded else PLAN
    if args.phase == "fetch":
        fetch(args.run, source_plan)
        return
    histories = [(key, Path(path)) for key, path in (entry.split("=", 1) for entry in args.history)]
    selected, record = prepare(args.run, histories, source_plan)
    cases = args.run / "originals.json"
    receipt = args.run / "source-receipt.json"
    if args.phase == "prepare":
        save(cases, selected)
        save(receipt, {**record, "originals_sha256": digest(cases)})
    else:
        require(read(cases) == selected and read(receipt) == {**record, "originals_sha256": digest(cases)}, "Source replay differs")
    print(record["dispositions"], flush=True)


if __name__ == "__main__":
    main()
