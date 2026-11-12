"""Qualify licensed PeerJ and Copernicus revision pairs from publisher-deposited abstracts."""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path
from time import sleep
from urllib.parse import urlencode

from lxml import etree

from scripts.prepare_writing_option_order_transfer import benchmark_texts
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.writing_agreement_revision_sources import read, require, retrieve, save
from scripts.writing_focus_v3 import digest
from scripts.writing_reference_revision_sources import excerpt, text_hash

SCHEMA = "chipmunk.writing-publisher-revision-holdout/1"
PUBLISHERS = {
    "peerj": {"member": "4443", "prefix": "10.7287/peerj.preprints.", "relationship": "replaces",
              "catalog_side": "after", "dates": ["2016-01-01", "2019-12-31"]},
    "copernicus": {"member": "3145", "prefix": "10.5194/", "relationship": "is-preprint-of",
                   "catalog_side": "before", "dates": ["2018-01-01", "2018-06-30"]},
}
API = "https://api.crossref.org"
CATALOG_LIMIT = 250
POOL_PER_PUBLISHER = 48
QUOTA_PER_PUBLISHER = 8
FIELDS = "DOI,abstract,relation,published,license,author,member,type,publisher"
LICENSE = re.compile(r"https?://creativecommons\.org/licenses/by/(3\.0|4\.0)/?")
PLAN = {
    "schema": SCHEMA, "publishers": PUBLISHERS, "catalog_limit": CATALOG_LIMIT,
    "pool_per_publisher": POOL_PER_PUBLISHER, "quota_per_publisher": QUOTA_PER_PUBLISHER,
    "catalog": "Use first date-ascending Crossref member page with the specified relationship, posted content and abstracts. "
               "Freeze the first 48 eligible linked works in SHA-256 order per publisher before reading the counterpart.",
    "licenses": "Use publisher-deposited article abstracts only. Require a CC BY 3.0 or 4.0 term on both exact records.",
    "geometry": "Require five exact adjoining abstract sentences with one native interior lexical edit of 12–40 words, "
                "unchanged two-sentence flanks on each side, and no qualifying quoted or marked third-party text.",
    "selection": "Accept up to eight distinct author groups per publisher after excluding known project overlaps.",
}


def licensed(record):
    return sorted({item["URL"].rstrip("/") for item in record.get("license", []) if LICENSE.fullmatch(item["URL"])})


def endpoint(config):
    start, stop = config["dates"]
    relationship = config["relationship"]
    filters = (f"from-pub-date:{start},until-pub-date:{stop},type:posted-content,"
               f"relation.type:{relationship},has-abstract:true")
    return f"{API}/members/{config['member']}/works?" + urlencode({
        "filter": filters, "rows": CATALOG_LIMIT, "sort": "published", "order": "asc", "select": FIELDS,
    })


def entries(record, config):
    result = {}
    for row in record["items"]:
        doi = row["DOI"].lower()
        if not doi.startswith(config["prefix"]) or not licensed(row) or not row.get("abstract"):
            continue
        linked = {link["id"].lower() for link in row.get("relation", {}).get(config["relationship"], [])
                  if link["id-type"] == "doi" and link["id"].lower().startswith(config["prefix"])}
        if len(linked) != 1:
            continue
        (other,) = linked
        if config["catalog_side"] == "after":
            if not doi.endswith("v2") or other != doi[:-1] + "1":
                continue
            key = doi.removesuffix("v2")
        else:
            if row["type"] != "posted-content" or other == doi:
                continue
            key = doi
        result.setdefault(key, (row, other))
    ordered = sorted(result, key=text_hash)
    require(len(ordered) >= POOL_PER_PUBLISHER, "Publisher catalog contains too few eligible works")
    return {key: result[key] for key in ordered[:POOL_PER_PUBLISHER]}


def frozen(run, publisher):
    config = PUBLISHERS[publisher]
    cache = run / "publisher" / publisher
    catalog = cache / "catalog.json"
    record = read(catalog)["message"]
    require(len(record["items"]) == CATALOG_LIMIT, "Fixed catalog page is incomplete")
    pool = entries(record, config)
    freeze = {"source_plan_sha256": digest(run / "source-plan.json"), "catalog_sha256": digest(catalog),
              "catalog_total": record["total-results"],
              "pool": {key: {"catalog": row["DOI"], "counterpart": other} for key, (row, other) in pool.items()}}
    return pool, freeze


def fetch(run, publisher):
    config = PUBLISHERS[publisher]
    require(read(run / "source-plan.json") == PLAN, "Publisher intake was not frozen")
    cache = run / "publisher" / publisher
    cache.mkdir(parents=True, exist_ok=True)
    retrieve(cache, "catalog.json", endpoint(config))
    pool, freeze = frozen(run, publisher)
    binding = cache / "catalog-freeze.json"
    if not binding.exists():
        save(binding, freeze)
    require(read(binding) == freeze, "Publisher candidate roster changed")
    for i, (_, other) in enumerate(pool.values()):
        path = cache / f"{i:03}.json"
        if not path.exists():
            retrieve(cache, path.name, f"{API}/works/{other}")
            sleep(.15)
        if (i + 1) % 8 == 0:
            print(publisher, i + 1, flush=True)


def metadata(row, config):
    published, = row["published"]["date-parts"]
    require(len(published) == 3 and row["member"] == config["member"] and licensed(row),
            "A publisher version lacks its dates or use terms")
    authors = [(author.get("given", "").strip().casefold(), author["family"].strip().casefold())
               for author in row["author"] if author.get("family")]
    require(authors, "A publisher version lacks attributable authors")
    return date(*published), authors


def prose(record):
    abstract = record.get("abstract")
    if not abstract:
        return None
    wrapper = '<body xmlns:jats="http://www.ncbi.nlm.nih.gov/JATS1">' + abstract + "</body>"
    container = etree.fromstring(wrapper.encode(), parser=etree.XMLParser(resolve_entities=False, no_network=True))
    for node in container.iter():
        if isinstance(node.tag, str):
            node.tag = etree.QName(node).localname
    root = container.find("abstract")
    if root is None:
        root = container
    root.tag = "body"
    if not root.findall(".//p") and root.text:
        paragraph = etree.Element("p")
        paragraph.text = root.text
        root.text = None
        root.insert(0, paragraph)
    return root


def inspect(run, histories, exposed):
    require(read(run / "source-plan.json") == PLAN and histories, "Unfrozen source or missing overlap records")
    publisher_audit, candidates, signatures = {}, {}, {}
    for publisher, config in PUBLISHERS.items():
        pool, freeze = frozen(run, publisher)
        cache = run / "publisher" / publisher
        binding = cache / "catalog-freeze.json"
        require(read(binding) == freeze, "Publisher source roster differs")
        receipts, dispositions = {}, Counter()
        for i, (key, (entry, other)) in enumerate(pool.items()):
            file = cache / f"{i:03}.json"
            second = read(file)["message"]
            require(second["DOI"].lower() == other, "DOI record binding differs")
            receipts[key] = {"counterpart_sha256": digest(file), "ordinal": i}
            old, new = (entry, second) if config["catalog_side"] == "before" else (second, entry)
            try:
                before_date, before_authors = metadata(old, config)
                after_date, after_authors = metadata(new, config)
                require(before_date < after_date < date(2020, 1, 1) and before_authors == after_authors,
                        "Publisher dates or authors differ")
                edits = list(excerpt(prose(old), prose(new), expanded=True))
            except (KeyError, TypeError, ValueError, etree.XMLSyntaxError):
                dispositions["metadata_or_parse_rejected"] += 1
                continue
            if not edits:
                dispositions["no_exact_edit"] += 1
                continue
            group = text_hash(key)[:16]
            candidates[group] = [
                {"group": group, "publisher": publisher, "before": earlier, "after": later,
                 "before_doi": old["DOI"], "after_doi": new["DOI"], "before_date": before_date.isoformat(),
                 "after_date": after_date.isoformat(), "authors": before_authors,
                 "before_license": licensed(old), "after_license": licensed(new)}
                for earlier, later in edits
            ]
            for index, (earlier, later) in enumerate(edits):
                signatures[group, index] = fingerprints(earlier["text"]) | fingerprints(later["text"])
        publisher_audit[publisher] = {"catalog_freeze_sha256": digest(binding), "records": receipts,
                                      "dispositions": dispositions}
    packets = [(name, digest(path), None, benchmark_texts(path)) for name, path in histories]
    prior = read(exposed)
    packets.append(("f1000-previous-source-panel", digest(exposed), len(prior) * 2,
                    (row[side]["text"] for row in prior for side in ("before", "after"))))
    history, external, _ = overlap(packets, signatures)
    authors = {name.casefold() for row in prior for name in row["attribution"]}
    chosen, used, counts = [], set(), Counter()
    for group, edits in candidates.items():
        publisher = edits[0]["publisher"]
        disposition = publisher_audit[publisher]["dispositions"]
        if counts[publisher] >= QUOTA_PER_PUBLISHER:
            disposition["eligible_after_quota"] += 1
            continue
        for index, row in enumerate(edits):
            row_authors = {" ".join(author).strip() for author in row["authors"]}
            identity = group, index
            if identity in external or authors & row_authors or used & signatures[identity]:
                continue
            chosen.append({**row, "authors": [list(author) for author in row["authors"]]})
            authors.update(row_authors)
            used.update(signatures[identity])
            counts[publisher] += 1
            disposition["selected"] += 1
            break
        else:
            disposition["author_or_text_overlap"] += 1
    receipt = {"schema": SCHEMA, "source_plan_sha256": digest(run / "source-plan.json"),
               "source_code_sha256": digest(Path(__file__)), "publishers": publisher_audit, "history": history}
    return chosen, receipt


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "fetch", "prepare", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("--publisher", choices=PUBLISHERS)
    cli.add_argument("--history", action="append", default=[])
    cli.add_argument("--exposed", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    if args.phase == "freeze":
        save(args.run / "source-plan.json", PLAN)
    elif args.phase == "fetch":
        require(args.publisher, "Select the frozen publisher lane")
        fetch(args.run, args.publisher)
    else:
        require(args.exposed, "Supply the prior F1000 source manifest")
        history = [(key, Path(path)) for key, path in (entry.split("=", 1) for entry in args.history)]
        rows, receipt = inspect(args.run, history, args.exposed)
        originals = args.run / "originals.json"
        committed = args.run / "source-receipt.json"
        if args.phase == "prepare":
            save(originals, rows)
            save(committed, {**receipt, "originals_sha256": digest(originals)})
        else:
            require(read(originals) == rows and read(committed) == {**receipt, "originals_sha256": digest(originals)},
                    "Publisher holdout source replay differs")
        print({key: dict(value["dispositions"]) for key, value in receipt["publishers"].items()}, flush=True)


if __name__ == "__main__":
    main()
