"""Measure earlier wording evidence and exact suppression of writing-style highlights."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from app.bench.independent_locations import _matches
from app.core.canonical import canonical_hash
from scripts import writing_multi_passage_panel as panel
from scripts import writing_report_sources as sources
from scripts.writing_archive_competition import mixed_sources

SCHEMA = "chipmunk.writing-published-wording/1"
PUBLIC = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
SHARDS = (
    ("test", "gpt-5.4", "9e28d157d336a343106487c8acab242ee41829f49ceb748dcb5f9fd6a9f9fb33"),
    ("train", "gemini-2.5-flash", "c4af2313124b86f5003f6635bf451960f6cbd015a849fb33cd35935c89cf778e"),
    ("train", "gpt-5.4", "a205bd7102225997b9e2245589cdf416e42499677b31f76382e28bd1b0b9fa68"),
)


def intake(cache, archive, validation, development, previous, output, receipt=None):
    raw = archive.read_bytes()
    if panel.digest(raw) != sources.GAO_SHA256:
        raise ValueError("The GovReport highlights pack differs from the expected edition")
    library = json.loads(raw)["reports"]
    commitments = panel.read(receipt)["fresh_sources"]["earlier_receipts"] if receipt is not None else None
    seen, digests, receipts = sources.exposed_reports(PUBLIC, commitments=commitments)
    for path in (validation, development, previous):
        for work in panel.read(path)["works"]:
            seen.add(work["document"].removeprefix("GAO_").upper())
            digests.update(v["native_text_sha256"] for v in work["variants"])
    works, counts = [], Counter()
    for split, generator, digest in SHARDS:
        key = f"{split}/{generator}"
        entries = sources.test_pairs(cache / f"default/{split}/reports_{generator}.csv", generator, split, digest)
        for identifier in sorted(entries, key=lambda item: panel.digest(f"{SCHEMA}/{key}/{item}".encode())):
            if counts[key] == 16:
                break
            report_id = identifier.removeprefix("GAO_").upper()
            if report_id in seen or set(entries[identifier]) != {"v0", "v1"}:
                continue
            work, _ = sources.verified_gao_work(identifier, entries[identifier], library.get(report_id), generator, split,
                                                max_edits=4, changed_only=True)
            if work is None or any(v["native_text_sha256"] in digests for v in work["variants"]):
                continue
            # Test only the publisher layout, to keep related whitespace variants out of the denominators.
            work["variants"] = [v for v in work["variants"] if v["layout"] == "published"]
            work["generator"] = generator
            seen.add(report_id)
            digests.update(v["native_text_sha256"] for v in work["variants"])
            works.append(work)
            counts[key] += 1
        print(key, counts[key], flush=True)
    panel.save(output, {"schema": SCHEMA, "works": works, "sources": {"opai_revision": sources.OPAI_REVISION,
        "shards_sha256": {f"{split}/{generator}": digest for split, generator, digest in SHARDS},
        "excluded_plans_sha256": [panel.digest(p.read_bytes()) for p in (validation, development, previous)],
        "earlier_receipts": receipts, "admitted": counts}})


def _summary(rows):
    totals = defaultdict(Counter)
    for row in rows:
        truth, style, retained, changes = (row[name] for name in ("truth", "style", "retained", "changes"))
        totals[f"{row['panel']}/{row['case']}"].update(files=1, qualified_sources=bool(row["reference"]),
            wrong_sources=bool(row["reference"]) and row["reference"] != row["source"],
            known_changed_ranges=len(truth), style_located=_matches(truth, style, .30, .50),
            archive_located=_matches(truth, changes, .30, .50),
            combined_located=_matches(truth, list(dict.fromkeys(map(tuple, retained + changes))), .30, .50),
            style_flagged=bool(style), style_remaining=bool(retained), with_archive_differences=bool(changes),
            archived_style_ranges=len(style) - len(retained))
    return {key: dict(counts) for key, counts in sorted(totals.items())}


def compare(archive, development, development_readings, fresh, fresh_readings, output):
    from app.detectors.text.extract import from_string
    from app.evidence import document_gao_archive, document_passage_intervals, document_published_wording

    started = time.monotonic()
    library, index = document_gao_archive.library(archive, sources.GAO_SHA256)
    index_seconds = round(time.monotonic() - started, 3)
    rows = []

    def inspect(group, case, source, text, truth, style=None):
        doc = from_string(text)
        metadata = doc.meta | {"method": "plain", "original_sha256": panel.digest(text.encode())}
        record = document_published_wording.capture(doc.text, doc.paragraphs, doc.source_map, metadata, style,
                                                     sources.GAO_SHA256, library, index)
        comparison, reference = [], None
        if record:
            document = {"coverage": {document_published_wording.KEY: record, document_passage_intervals.KEY: style},
                        "source": {"method": "plain", "chars": len(doc.text),
                                   document_published_wording.COMMITMENT: canonical_hash(record)}}
            document_published_wording.verify_source(document, doc.text, doc.source_map, metadata, None)
            reference = record["reference"]["id"]
            comparison = document_gao_archive._context_ranges(doc, [(v["start"], v["end"]) for v in record["changes"]])
        prefix = panel._prefix(doc.text)
        def ranges(values):
            return [[prefix[a], prefix[b]] for a, b in values]

        displayed = document_published_wording.display_intervals(style, record) if style else []
        rows.append({"panel": group, "case": case, "source": source, "reference": reference, "sha256": panel.digest(doc.text.encode()),
                     "truth": truth, "style": ranges((v["start"], v["end"]) for v in style["ranges"]) if style else [],
                     "retained": ranges((v["start"], v["end"]) for v in displayed), "changes": ranges(comparison)})

    for group, plan_path, readings_path in (("previous", development, development_readings), ("fresh", fresh, fresh_readings)):
        plan = panel.read(plan_path)
        measurements = {(work["document"], v["version"]): v for work in panel.read(readings_path)["works"]
                        for v in work["variants"] if v["layout"] == "published"}
        for work in plan["works"]:
            for variant in work["variants"]:
                if variant["layout"] != "published":
                    continue
                text = variant["text"]
                version = variant["version"]
                reading = measurements[work["document"], version]
                if (reading["native_text_sha256"] != variant["native_text_sha256"]
                        or variant["native_text_sha256"] != panel.digest(text.encode())):
                    raise ValueError("Writing readings differ from the exact extracted text")
                _, contexts = panel._inputs(variant)
                if reading["inputs_sha256"] != [panel.digest(value.encode()) for value in contexts]:
                    raise ValueError("Writing readings differ from their context inputs")
                doc = from_string(text)
                geometry = [[s[0], s[1], s[5]] for s in variant["sentences"]]
                style = document_passage_intervals.capture(doc.text, doc.source_map, geometry,
                    {name: reading[name] for name in ("isolated", "context")}, document_baseline=True, report_review=True)
                inspect(group, version, work["document"].removeprefix("GAO_").upper(), text,
                        [v["compact"] for v in variant["truth"]], style)
        for case, _, text in mixed_sources(plan["works"], library, index):
            inspect(f"{group}_mixed_sources", case, None, text, [])
        if group == "fresh":
            reserve = sorted(plan["works"], key=lambda item: item["group_sha256"])[24:48]
            for case, _, text in mixed_sources(reserve, library, index):
                inspect("fresh_reserved_mixed_sources", case, None, text, [])
    inputs = [development, development_readings, fresh, fresh_readings]
    receipt = {"schema": SCHEMA, "archive_sha256": sources.GAO_SHA256, "fresh_sources": panel.read(fresh)["sources"],
               "inputs_sha256": {p.name: panel.digest(p.read_bytes()) for p in inputs}, "cold_index_seconds": index_seconds,
               "rows": rows, "summary": _summary(rows)}
    panel.save(output, receipt)
    print(json.dumps(verify(receipt), sort_keys=True), flush=True)


def verify(receipt):
    rows = receipt["rows"]
    summary = _summary(rows)
    if receipt["schema"] != SCHEMA or receipt["summary"] != summary:
        raise ValueError("Published wording counts differ from their recorded locations")
    if any(counts["wrong_sources"] or (key.endswith("/ordinary") and counts["with_archive_differences"])
           for key, counts in summary.items()):
        raise ValueError("A study result attributed an incorrect source or reported a change in an unchanged file")
    if any(span not in row["style"] for row in rows for span in row["retained"]):
        raise ValueError("An existing style highlight moved during the archive comparison")
    return summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    stages = cli.add_subparsers(dest="step", required=True)
    pick, run, recount = (stages.add_parser(step) for step in ("intake", "compare", "verify"))
    for name in ("cache", "archive", "validation", "development", "previous", "output"):
        pick.add_argument(name, type=Path)
    pick.add_argument("--receipt", type=Path)
    for name in ("archive", "development", "development_readings", "fresh", "fresh_readings", "output"):
        run.add_argument(name, type=Path)
    recount.add_argument("receipt", type=Path)
    args = vars(cli.parse_args())
    step = args.pop("step")
    if step == "verify":
        print(json.dumps(verify(panel.read(args["receipt"])), sort_keys=True))
    else:
        {"intake": intake, "compare": compare}[step](**args)


if __name__ == "__main__":
    main()
