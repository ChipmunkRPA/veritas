# SPDX-License-Identifier: Apache-2.0
"""Verify local report attribution and the scoped wording comparison on held files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from app.bench.writing_entropy_prominence import matched
from app.core.canonical import canonical_hash
from app.detectors.text.extract import extract
from app.evidence import document_economic_archive as release
from app.evidence import document_published_wording as wording
from scripts.writing_transition_deviation import PINS, file_hash, read, require, save

SCHEMA = "chipmunk.economic-release-wording/1"


def _review(plan, archive, output, scratch):
    require(plan, PINS["economic_rewrites"])
    pack = file_hash(archive)
    releases = release.library(archive, pack)
    works = read(plan)["works"]
    receipts = []
    totals = Counter()
    for index, work in enumerate(works):
        source = releases[work["document"]]
        other = releases[works[(index + 1) % len(works)]["document"]]
        related = releases[next(row["document"] for row in works if row["genre"] == work["genre"] and row is not work)]
        if source["source_record_sha256"] != work["original_source_sha256"]:
            raise ValueError("The archive lacks the exact held publisher version")
        header = f"{source['title']}\n\nSource: {source['source_url']}\n\n"
        for offset, case in enumerate(work["cases"]):
            text = case["text"]
            negative = {
                "anonymous": "",
                "missing_title": f"Source: {source['source_url']}\n\n",
                "missing_url": f"{source['title']}\n\n",
                "wrong_url": f"{source['title']}\n\nSource: {other['source_url']}\n\n",
                "wrong_release": f"{related['title']}\n\nSource: {related['source_url']}\n\n",
            }
            for label, attribution in negative.items():
                scratch.write_text(attribution + text)
                control = extract(scratch, ".txt")
                if wording.capture_release(control.text, control.paragraphs, control.source_map, control.meta, pack, releases):
                    raise ValueError(f"Unexpected binding for {label}")
                totals.update({f"{label}_abstained": 1})
            scratch.write_text(header + text)
            doc = extract(scratch, ".txt")
            if doc.text != header + text:
                raise ValueError("The attribution changed body positions")
            record = wording.capture_release(doc.text, doc.paragraphs, doc.source_map, doc.meta, pack, releases)
            if record is None or record["reference"] != source:
                raise ValueError("A verified publisher release was not bound")
            evidence = {"coverage": {wording.KEY: record}, "source": {
                "method": "plain", "chars": len(doc.text), wording.COMMITMENT: canonical_hash(record),
            }}
            wording.verify_source(evidence, doc.text, doc.source_map, doc.meta, doc.word_review, paragraphs=doc.paragraphs)
            outcome = "rewritten" if case["kind"] == "mixed" else "unchanged" if offset == 0 else "ordinary_edit"
            truth = [row["compact"] for row in case["truth"]]
            ranges = [[row["start"] - len(header), row["end"] - len(header)] for row in record["changed_sentences"]]
            prefix = [0]
            for char in text:
                prefix.append(prefix[-1] + (not char.isspace()))
            compact = [[prefix[a], prefix[b]] for a, b in ranges]
            if outcome == "unchanged" and record["changes"] or outcome != "unchanged" and not record["changes"]:
                raise ValueError("The comparison omitted an edit or marked unchanged text")
            found = len(matched(compact, truth))
            totals.update({outcome: 1, f"{outcome}_marked_sentences": len(compact),
                           f"{outcome}_known_sentences": len(truth), f"{outcome}_located_sentences": found})
            receipts.append({"group": work["group"], "original_sha256": source["source_record_sha256"],
                             "case_sha256": case["native_text_sha256"], "outcome": outcome,
                             "submitted_ranges": ranges, "compact_ranges": compact, "truth": truth,
                             "located": found, "abstained_on": sorted(negative)})
    if totals["rewritten_known_sentences"] != totals["rewritten_located_sentences"]:
        raise ValueError("A known rewritten sentence is missing from the scoped comparison")
    data = {"schema": SCHEMA, "implementation_sha256": file_hash(Path(release.__file__)),
            "wording_sha256": file_hash(Path(wording.__file__)), "pack_sha256": pack,
            "plan_sha256": PINS["economic_rewrites"], "results": dict(totals), "cases": receipts}
    save(output, data)
    print(dict(totals))


def review(plan, archive, output):
    with TemporaryDirectory(prefix="chipmunk-economic-review-") as folder:
        _review(plan, archive, output, Path(folder) / "body.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plan", "archive", "output"):
        parser.add_argument(field, type=Path)
    review(**vars(parser.parse_args()))
