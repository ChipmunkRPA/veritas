"""Freeze exact dated eLife body versions from an immutable pre-2020 publisher archive."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen

from lxml import etree

from scripts.prepare_writing_option_order_transfer import benchmark_texts
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.writing_agreement_revision_sources import read, require, retrieve, save
from scripts.writing_copernicus_body_holdout import material_change
from scripts.writing_focus_v3 import digest
from scripts.writing_publisher_revision_holdout import SCHEMA as PUBLISHER_SCHEMA
from scripts.writing_reference_revision_sources import excerpt, text_hash

SCHEMA = "chipmunk.writing-elife-version-body-holdout/2"
COMMIT = "58e688ec3d4523eb1f3b03bed6bb440bf91e4786"
TREE = "634a0477c305095ae24842488ad2a2e97325d32f"
API = "https://api.github.com/repos/elifesciences/elife-article-xml"
RAW = f"https://raw.githubusercontent.com/elifesciences/elife-article-xml/{COMMIT}/articles"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
FILENAME = re.compile(r"elife-(\d{5})-v([23])\.xml")
EXCLUDED = {"34311"}
POOL = 30
MIN_XML_BYTES = 18_000
MAX_TREE_BYTES = 8_000_000
PLAN = {
    "schema": SCHEMA, "publisher": "eLife", "archive_commit": COMMIT, "archive_articles_tree": TREE,
    "pool_size": POOL, "excluded_scout_ids": sorted(EXCLUDED),
    "selection": "Use the pinned publisher repository. Freeze 30 works in fixed SHA-256 order with both v2 and v3 "
                 "XML files of at least 18,000 bytes. Require both versions of record before 2020. Exclude previous "
                 "project article IDs and the known source-only example before selecting the pool.",
    "geometry": "Require a dated ordinary body revision with two exact preceding and following sentences, "
                "one changed 12–40-word center and at least two material lexical changes. Exclude citation-only and "
                "typographic changes, and compare exact source author prose under both version-specific CC BY licenses.",
    "exposure": "eLife previously appeared in development, including source-exposed ordinary edits. Retain a separate "
                "new-work holdout with exact article bodies and exclude known project source overlaps.",
}
PANELS = Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
PRIOR_INTAKE = (PANELS / "text-elife-transfer-intake-complete-v286.json",
                PANELS / "text-elife-source-intake-complete-v272.json")


def roster(run):
    archive = run / "publisher"
    root = read(archive / "root-tree.json")
    article_tree = read(archive / "articles-tree.json")
    require(not root["truncated"] and not article_tree["truncated"] and article_tree["sha"] == TREE
            and any(row["path"] == "articles" and row["sha"] == TREE for row in root["tree"]),
            "Publisher archive tree is incomplete")
    versions = defaultdict(dict)
    excluded = EXCLUDED | {work["source_id"] for path in PRIOR_INTAKE for work in read(path)["works"]}
    for row in article_tree["tree"]:
        if match := FILENAME.fullmatch(row["path"]):
            article, revision = match.groups()
            if article not in excluded and int(article) < 55_000 and row["type"] == "blob" and MIN_XML_BYTES <= row["size"] <= 450_000:
                versions[article][revision] = {key: row[key] for key in ("sha", "path", "size")}
    ranked = sorted((key for key, items in versions.items() if set(items) == {"2", "3"}), key=text_hash)
    require(len(ranked) >= POOL, "Publisher archive has too few bounded version pairs")
    return {"source_plan_sha256": digest(run / "source-plan.json"), "root_sha256": digest(archive / "root-tree.json"),
            "tree_sha256": digest(archive / "articles-tree.json"), "prior_intake": {path.name: digest(path) for path in PRIOR_INTAKE},
            "pool": {key: versions[key] for key in ranked[:POOL]}}


def fetch(run):
    require(read(run / "source-plan.json") == PLAN, "Source plan differs")
    archive = run / "publisher"
    archive.mkdir(exist_ok=True)
    retrieve(archive, "root-tree.json", f"{API}/git/trees/{COMMIT}")
    tree_path = archive / "articles-tree.json"
    if not tree_path.exists():
        request = Request(f"{API}/git/trees/{TREE}",
                          headers={"User-Agent": "Chipmunk-Verify-public-revision-research/1.0", "Accept-Encoding": "gzip"})
        with urlopen(request, timeout=30) as response:
            if response.headers.get("Content-Encoding") == "gzip":
                with gzip.GzipFile(fileobj=response) as compressed:
                    tree = compressed.read(MAX_TREE_BYTES + 1)
            else:
                tree = response.read(MAX_TREE_BYTES + 1)
        require(0 < len(tree) <= MAX_TREE_BYTES, "Publisher archive index exceeded its local cap")
        with tree_path.open("xb") as stream:
            stream.write(tree)
    frozen = run / "catalog-freeze.json"
    lock = roster(run)
    if not frozen.exists():
        save(frozen, lock)
    require(read(frozen) == lock, "Publisher version roster differs")
    for i, (key, versions) in enumerate(lock["pool"].items()):
        retrieve(archive, f"{key}-versions.json", f"https://api.elifesciences.org/articles/{key}/versions")
        for item in versions.values():
            retrieve(archive, item["path"], f"{RAW}/{item['path']}")
            sleep(.1)
        if (i + 1) % 5 == 0:
            print(f"Cached {i + 1}/{POOL} eLife version pairs.", flush=True)


def dated_api(path, article):
    rows = {row["version"]: row for row in read(path)["versions"] if row["version"] in (2, 3)}
    require(rows.keys() == {2, 3}, "Public version history lacks both releases")
    dates = [datetime.fromisoformat(rows[i]["versionDate"]).astimezone(UTC) for i in (2, 3)]
    require(dates[0] < dates[1] < datetime(2020, 1, 1, tzinfo=UTC)
            and all(row["id"] == article and row["doi"].lower() == f"10.7554/elife.{article}"
                    and row["copyright"]["license"] == "CC-BY-4.0" and row["status"] == "vor"
                    for row in rows.values()), "Version dates, terms or article identity differ")
    return dates, [rows[i]["status"] for i in (2, 3)]


def body(path, commitment, article):
    data = path.read_bytes()
    oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    require(oid == commitment["sha"] and len(data) == commitment["size"], "Archive XML differs from its Git blob")
    root = etree.fromstring(data, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    meta, value = root.find("front/article-meta"), root.find("body")
    require(meta is not None and value is not None
            and meta.findtext("article-id[@pub-id-type='doi']", "").lower() == f"10.7554/elife.{article}",
            "Archive XML lacks the exact article body")
    terms = {item.get("{http://www.w3.org/1999/xlink}href", "").replace("http://", "https://").rstrip("/")
             for item in meta.findall("permissions/license")}
    require(LICENSE.rstrip("/") in terms, "Archive XML license differs")
    authors = [" ".join(" ".join(node.itertext()).split()).casefold()
               for node in meta.findall("contrib-group/contrib[@contrib-type='author']/name")]
    require(authors and all(authors), "Archive XML lacks author attribution")
    return value, authors


def collect(run, histories, prior_sources):
    lock = run / "catalog-freeze.json"
    plan = roster(run)
    require(read(lock) == plan and read(run / "source-plan.json") == PLAN, "Publisher source roster differs")
    require(histories or prior_sources, "Missing previous source exposure commitments")
    archive = run / "publisher"
    sources, candidates, signatures, counts = {}, {}, {}, Counter()
    for article, version in plan["pool"].items():
        api = archive / f"{article}-versions.json"
        paths = [archive / version[key]["path"] for key in ("2", "3")]
        sources[article] = {"history_sha256": digest(api), "xml_sha256": [digest(path) for path in paths]}
        try:
            dates, status = dated_api(api, article)
            (old, authors), (new, later_authors) = (
                body(path, version[key], article) for path, key in zip(paths, ("2", "3"), strict=True)
            )
            require(authors == later_authors, "Source version authors differ")
        except (ValueError, etree.XMLSyntaxError, KeyError, TypeError) as error:
            if str(error) == "Archive XML differs from its Git blob":
                raise
            counts["version_metadata_rejected"] += 1
            continue
        matches = [(a, b) for a, b in excerpt(old, new, expanded=True) if material_change(a, b)]
        if not matches:
            counts["no_material_matched_body_edit"] += 1
            continue
        counts["eligible_before_overlap"] += 1
        group = text_hash("elife:" + article)[:16]
        candidates[group] = []
        for i, (before, after) in enumerate(matches):
            candidates[group].append({
                "group": group, "publisher": "elife", "doi": f"10.7554/eLife.{article}", "article": article,
                "version_dates": [item.isoformat() for item in dates], "version_statuses": status, "authors": authors,
                "license": LICENSE, "before": before, "after": after, **sources[article],
            })
            signatures[group, i] = fingerprints(before["text"]) | fingerprints(after["text"])
    packets = [(name, digest(path), None, benchmark_texts(path)) for name, path in histories]
    names = set()
    for name, path in prior_sources:
        rows = read(path)
        packets.append((name, digest(path), len(rows) * 2, (row[side]["text"] for row in rows for side in ("before", "after"))))
        for row in rows:
            names.update(str(author).casefold() for author in row.get("attribution", []))
            names.update(" ".join(author) if isinstance(author, list) else author for author in row.get("authors", []))
    history, excluded, _ = overlap(packets, signatures)
    used, selected = set(), []
    for group, rows in candidates.items():
        for i, row in enumerate(rows):
            key = group, i
            authors = set(row["authors"])
            if key in excluded or used & signatures[key] or names & authors:
                continue
            selected.append(row)
            names.update(authors)
            used.update(signatures[key])
            counts["selected"] += 1
            break
        else:
            counts["author_or_text_overlap"] += 1
    return selected, {"schema": PUBLISHER_SCHEMA, "body_method_schema": SCHEMA, "archive_commit": COMMIT,
                      "source_plan_sha256": digest(run / "source-plan.json"), "catalog_freeze_sha256": digest(lock),
                      "source_code_sha256": digest(Path(__file__)), "source_versions": sources, "history": history,
                      "dispositions": dict(counts)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "fetch", "prepare", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("--history", action="append", default=[])
    cli.add_argument("--prior", action="append", default=[])
    args = cli.parse_args()
    os.umask(0o077)
    if args.phase == "freeze":
        save(args.run / "source-plan.json", PLAN)
    elif args.phase == "fetch":
        fetch(args.run)
    else:
        histories = [(key, Path(path)) for key, path in (entry.split("=", 1) for entry in args.history)]
        priors = [(key, Path(path)) for key, path in (entry.split("=", 1) for entry in args.prior)]
        rows, receipt = collect(args.run, histories, priors)
        originals, record = args.run / "originals.json", args.run / "source-receipt.json"
        if args.phase == "prepare":
            save(originals, rows)
            save(record, {**receipt, "originals_sha256": digest(originals)})
        else:
            require(read(originals) == rows and read(record) == {**receipt, "originals_sha256": digest(originals)},
                    "Source and rights replay differs")
        print(receipt["dispositions"], flush=True)


if __name__ == "__main__":
    main()
