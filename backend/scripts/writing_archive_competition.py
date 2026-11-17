"""Recount short wording differences and competing archive references."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.bench.independent_locations import _matches
from scripts import writing_multi_passage_panel as panel

SCHEMA = "chipmunk.writing-archive-competition/1"
METHODS = ("previous", "competing_phrases")


def evaluated(document, digest, sha, catalog, index):
    from app.evidence import document_gao_archive as archive

    prefix = panel._prefix(document.text)
    rows = {}
    for method in METHODS:
        result = archive.compare(document, digest, sha, catalog, index, resolve_competition=method != METHODS[0])
        rows[method] = {
            "status": result["status"], "reference": (result["reference"] or {}).get("id"),
            "ranges": [[prefix[a], prefix[b]] for a, b in result["submitted_sentence_ranges"]],
        }
    return rows


def mixed_sources(works, catalog, index):
    from app.evidence import document_gao_archive as archive

    def body(work):
        text = next(v["text"] for v in work["variants"] if v["layout"] == "published" and v["version"] == "ordinary")
        return text, archive.token_stream(text)

    ordered = sorted(works, key=lambda item: item["group_sha256"])
    for left, right in zip(ordered[:24:2], ordered[1:24:2], strict=True):
        a, a_terms = body(left)
        b, b_terms = body(right)
        b_id = right["document"][4:].upper()
        votes, _ = archive._votes(archive.token_values(b_terms), index, catalog, complete=True)
        starts = [start for start, _ in votes.get(b_id, ()) if start + 64 < len(b_terms)]
        if not starts:
            continue
        start = starts[len(starts) // 2]
        middle = a_terms[len(a_terms) // 2][1]
        short = b[b_terms[start][1]:b_terms[start + 15][2]]
        long = b[b_terms[start][1]:b_terms[start + 63][2]]
        sources = [left["document"], right["document"]]
        yield "balanced", sources, a[:middle].rstrip() + "\n\n" + b[b_terms[start][1]:]
        yield "large_insertion", sources, a[:middle] + long + " " + a[middle:]
        yield "short_append", sources, a + "\n\n" + short


def summarize(rows):
    totals = defaultdict(Counter)
    for row in rows:
        for name, result in row["checks"].items():
            correct = result["reference"] == row["reference"] and row["reference"] is not None
            key = f"{row['role']}/{row['case']}/{name}"
            totals[key].update(files=1, correct_sources=correct, none=result["reference"] is None,
                               wrong_sources=result["reference"] is not None and not correct,
                               flags=bool(result["ranges"]), truth=len(row["truth"]),
                               matches=_matches(row["truth"], result["ranges"], .30, .50) if correct else 0)
            totals[key][result["status"]] += 1
    return {key: dict(value) for key, value in sorted(totals.items())}


def compare(archive_path, validation, test, output):
    from app.detectors.text.extract import from_string
    from app.evidence.document_gao_archive import library
    from scripts.writing_report_sources import GAO_SHA256

    catalog, index = library(archive_path, GAO_SHA256)
    rows = []
    for source, role in ((validation, "earlier"), (test, "held")):
        works = panel.read(source)["works"]
        for work in works:
            for variant in work["variants"]:
                if variant["layout"] != "published":
                    continue
                document = from_string(variant["text"])
                digest = panel.digest(document.text.encode())
                if digest != variant["native_text_sha256"]:
                    raise ValueError("The extracted document differs from the exact source")
                rows.append({"role": role, "case": variant["version"], "source": work["document"],
                             "reference": work["document"][4:].upper(), "sha256": digest,
                             "truth": [span["compact"] for span in variant["truth"]],
                             "checks": evaluated(document, digest, GAO_SHA256, catalog, index)})
    for case, sources, text in mixed_sources(panel.read(test)["works"], catalog, index):
        document = from_string(text)
        digest = panel.digest(document.text.encode())
        rows.append({"role": "control", "case": case, "source": sources, "reference": None, "sha256": digest,
                     "truth": [], "checks": evaluated(document, digest, GAO_SHA256, catalog, index)})
    panel.save(output, {"schema": SCHEMA, "archive_sha256": GAO_SHA256,
                        "source_plans": [panel.digest(item.read_bytes()) for item in (validation, test)],
                        "rows": rows, "summary": summarize(rows)})
    print(json.dumps(verify(panel.read(output)), sort_keys=True), flush=True)


def verify(receipt):
    if receipt["schema"] != SCHEMA or summarize(receipt["rows"]) != receipt["summary"]:
        raise ValueError("The exact comparison ranges differ from the reported counts")
    return {"files": len(receipt["rows"]), "outcomes": len(receipt["summary"])}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    stages = cli.add_subparsers(dest="step", required=True)
    score, check = stages.add_parser("compare"), stages.add_parser("verify")
    for name in ("pack", "earlier", "held", "output"):
        score.add_argument(name, type=Path)
    check.add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.step == "compare":
        compare(args.pack, args.earlier, args.held, args.output)
    else:
        print(json.dumps(verify(panel.read(args.receipt)), sort_keys=True))
