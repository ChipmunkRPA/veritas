# SPDX-License-Identifier: Apache-2.0
"""Measure separately located sentences using verified earlier report wording."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts import writing_entropy_sources as intake
from scripts import writing_multi_passage_panel as data
from scripts.verify_writing_entropy_prominence import matching, public_roster
from scripts.writing_report_sources import GAO_SHA256

SCHEMA = "chipmunk.writing-archive-sentences/1"


def reserve(previous, shards, pack, output):
    selected = data.read(previous)
    identities, digests, _ = intake.exposures(selected["exposed"])
    for work in selected["works"]:
        identities.add(work["document"].removeprefix("GAO_").upper())
        digests.add(work["group_sha256"])
        digests.update(row["native_text_sha256"] for row in work["variants"])
    works = intake.report_sources(shards, pack, identities, digests)
    if not works or len({work["generator_backend"] for work in works}) < 2:
        raise ValueError("The confirmation needs fresh sources with diverse recorded generators")
    data.save(output, selected | {"cohort": "fresh-archive-confirmation", "parent_sha256": data.digest(previous.read_bytes()),
                                 "works": works})


def summarize(rows):
    totals = Counter()
    for row in rows:
        truth = row["truth"]
        totals.update(deliveries=1, ordinary=row["role"] == "ordinary", edits=len(truth),
                      qualified=row["reference"] is not None, grouped=matching(row["grouped"], truth),
                      separate=matching(row["separate"], truth),
                      false_highlights=row["role"] == "ordinary" and bool(row["separate"]))
    return dict(totals)


def compare(pack, earlier, reserved, confirmation, output):
    from app.core.canonical import canonical_hash
    from app.detectors.text.extract import from_string
    from app.evidence import document_gao_archive
    from app.evidence import document_published_wording as wording

    reports, index = document_gao_archive.library(pack, GAO_SHA256)
    panels, parents = {}, {}
    for cohort, path in (("earlier", earlier), ("reserved", reserved), ("confirmation", confirmation)):
        rows = []
        parents[cohort] = data.digest(path.read_bytes())
        for work in data.read(path)["works"]:
            for version in work["variants"]:
                if version["layout"] != "published":
                    continue
                doc = from_string(version["text"])
                source_hash = data.digest(doc.text.encode())
                if source_hash != version["native_text_sha256"]:
                    raise ValueError("The native source differs from the frozen report delivery")
                metadata = doc.meta | {"method": "plain", "original_sha256": source_hash}
                record = wording.capture(doc.text, doc.paragraphs, doc.source_map, metadata, None, GAO_SHA256, reports, index)
                grouped, separate, source = [], [], None
                if record:
                    commitment = canonical_hash(record)
                    evidence = {"coverage": {wording.KEY: record},
                                "source": {"method": "plain", "chars": len(doc.text), wording.COMMITMENT: commitment}}
                    wording.verify_source(evidence, doc.text, doc.source_map, metadata, None, paragraphs=doc.paragraphs)
                    source = record["reference"]["id"]
                    if source != work["document"].removeprefix("GAO_").upper():
                        raise ValueError("The exact source differs from the publisher's known report")
                    spans = [(item["start"], item["end"]) for item in record["changes"]]
                    grouped = document_gao_archive._context_ranges(doc, spans)
                    separate = [(row["start"], row["end"]) for row in record["changed_sentences"]]
                prefix = data._prefix(doc.text)
                rows.append({"document": work["document"], "source_sha256": source_hash, "role": version["version"],
                             "reference": source, "truth": [mark["compact"] for mark in version["truth"]],
                             "grouped": [[prefix[a], prefix[b]] for a, b in grouped],
                             "separate": [[prefix[a], prefix[b]] for a, b in separate]})
        panels[cohort] = {"rows": rows, "counts": summarize(rows)}
        print(cohort, panels[cohort]["counts"], flush=True)
    data.save(output, {"schema": SCHEMA, "archive_sha256": GAO_SHA256, "parents": parents, "panels": panels,
                       "confirmation_sources": public_roster(confirmation)})


def verify(receipt):
    value = data.read(receipt)
    if value["schema"] != SCHEMA:
        raise ValueError("Unknown archive sentence receipt")
    for cohort, panel in value["panels"].items():
        totals = summarize(panel["rows"])
        if totals != panel["counts"]:
            raise ValueError("The reported sentence counts differ from their coordinates")
        print(cohort, totals, flush=True)
    sources = value["confirmation_sources"]
    expected = {(work["document"], row["version"]): row["native_text_sha256"]
                for work in sources["works"] for row in work["variants"]}
    actual = {(row["document"], row["role"]): row["source_sha256"] for row in value["panels"]["confirmation"]["rows"]}
    if actual != expected or sources["native_plan_sha256"] != value["parents"]["confirmation"]:
        raise ValueError("The confirmation results differ from the original report list")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="action", required=True)
    for name, method, args in (("reserve", reserve, ("previous", "shards", "pack", "output")),
                               ("compare", compare, ("pack", "earlier", "reserved", "confirmation", "output")),
                               ("verify", verify, ("receipt",))):
        stage = modes.add_parser(name)
        stage.set_defaults(method=method)
        for argument in args:
            stage.add_argument(argument, type=Path)
    options = vars(cli.parse_args())
    options.pop("action")
    options.pop("method")(**options)
