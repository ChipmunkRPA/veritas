"""Replay the GAO exact-excerpt comparison against fresh held report excerpts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_revision_trajectory_transfer as source
from scripts.writing_temperature_transfer import useful

SCHEMA = "chipmunk.writing-gao-exact-excerpt/1"
PACK_SHA256 = "671287085feaa971c501be2e2db7d50a36ca2a9adba0642c9540f4add35b341e"
CONTROL_STATUS = {"quoted": "out_of_scope", "extraneous_edge": "no_qualified_reference",
                  "split_sources": "ambiguous_reference", "reversed": "no_qualified_reference"}


def native(roster_path, writing_path, pack, output):
    from app.detectors.text.extract import from_string
    from app.evidence import document_gao_archive as archive

    writing = source.read(writing_path)
    if source.digest(roster_path) != writing["freeze"]["roster_sha256"]:
        raise ValueError("The private roster differs from the frozen writing inputs")
    rows = [r for r in source.read(roster_path) if r["cohort"] == "report"]
    reports, index = archive.library(pack, PACK_SHA256)
    results = []
    for row in rows:
        doc = from_string(row["text"])
        earlier, current = [archive.compare(doc, row["text_sha256"], PACK_SHA256, reports, index, accept_exact_excerpt=accept)
                            for accept in (False, True)]
        expected_source = row["source"].rsplit("/", 1)[-1].upper()
        reference = current["reference"]
        if reference:
            source.require(reference["id"] == expected_source, "Unexpected source attribution")
        if earlier != current:
            source.require(row["family"] == "original" and earlier["status"] == "no_qualified_reference"
                           and not current["changes"] and reference, "Unexpected changed result")
            a, b = reference["submitted_range"]
            x, y = reference["transcript_range"]
            submitted = archive.token_values(archive.token_stream(doc.text[a:b]))
            transcript = archive.token_values(archive.token_stream(reports[expected_source]["text"][x:y]))
            source.require(a == 0 and b == len(doc.text) and submitted == transcript, "Exact normalized tokens differ")
        ranges = current["submitted_sentence_ranges"]
        changes = [[[r["submitted"]["start"], r["submitted"]["end"]], [r["archive"]["start"], r["archive"]["end"]]]
                   for r in current["changes"]]
        results.append({"id": row["id"], "prior_status": earlier["status"], "status": current["status"], "reference": reference,
                        "changes": changes, "sentence_ranges": ranges,
                        "strict": any(useful(span, row["truth"], set(row["spaces"])) for span in ranges)})
    originals = [r for r in rows if r["family"] == "original"]
    controls = []
    for i, row in enumerate(originals):
        first, second = row["text"], originals[(i + 1) % len(originals)]
        starts = row["spans"]
        specimens = {
            "quoted": "> " + first,
            "extraneous_edge": "Separately, the orchard reopened after the band played. " + first,
            "split_sources": first[starts[0][0]:starts[2][1]] + " " + second["text"][:second["spans"][2][1]],
            "reversed": " ".join(first[a:b] for a, b in reversed(starts)),
        }
        for role, text in specimens.items():
            evidence = archive.compare(from_string(text), source.sha(text), PACK_SHA256, reports, index)
            controls.append({"id": row["id"], "role": role, "text_sha256": source.sha(text), "status": evidence["status"]})
    receipt = {"schema": SCHEMA, "pack_sha256": PACK_SHA256, "writing_sha256": source.digest(writing_path),
               "catalog_reports": len(reports), "implementation_sha256": source.digest(Path(archive.__file__)),
               "rows": results, "controls": controls}
    receipt["summary"] = summarize(receipt, writing)
    source.seal(output, receipt)


def summarize(receipt, writing):
    roster = {r["id"]: r for r in writing["freeze"]["roster"] if r["cohort"] == "report"}
    source.require(len(roster) == 72 and {r["id"] for r in receipt["rows"]} == roster.keys()
                   and len(receipt["rows"]) == len(roster), "Unexpected held partitions")
    counts = defaultdict(Counter)
    originals = set()
    for row in receipt["rows"]:
        fixture = roster[row["id"]]
        original = fixture["family"] == "original"
        if original:
            originals.add(row["id"])
        matches = row["status"] == "reference_bound"
        reference = row["reference"]
        source.require(bool(reference) == matches, "Unexpected report reference")
        if matches:
            source.require(reference["id"] == fixture["source"].rsplit("/", 1)[-1].upper(), "Report identity differs")
            start, stop = reference["submitted_range"]
            source.require(0 <= start < stop <= fixture["length"], "Submitted range exceeds the file")
            if original:
                source.require([start, stop] == [0, fixture["length"]] and not row["changes"], "Exact excerpt differs")
        source.require(all(0 <= start <= stop <= fixture["length"] and 0 <= a <= b for (start, stop), (a, b) in row["changes"])
                       and all(0 <= a < b <= fixture["length"] for a, b in row["sentence_ranges"]),
                       "Invalid exact wording or sentence coordinates")
        strict = any(useful(span, fixture["truth"], set(fixture["spaces"])) for span in row["sentence_ranges"])
        source.require(strict == row["strict"], "Held localization differs")
        counts[fixture["family"]].update(files=1, prior_bound=row["prior_status"] == "reference_bound",
                                        bound=matches, changes=bool(row["changes"]), strict=strict)
    controls = receipt["controls"]
    source.require(len(controls) == 96
                   and {(r["id"], r["role"]) for r in controls} == {(i, role) for i in originals for role in CONTROL_STATUS},
                   "Incomplete controls")
    for row in controls:
        source.require(row["status"] == CONTROL_STATUS[row["role"]], "Unexpected control disposition")
        counts[f"control/{row['role']}"].update(files=1, **{row["status"]: 1})
    return {k: dict(v) for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("writing", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--roster", type=Path)
    args = parser.parse_args()
    if args.pack and args.roster:
        native(args.roster, args.writing, args.pack, args.receipt)
    receipt, writing = source.read(args.receipt), source.read(args.writing)
    source.require(receipt["schema"] == SCHEMA and receipt["catalog_reports"] == 12228 and receipt["pack_sha256"] == PACK_SHA256
                   and receipt["writing_sha256"] == source.digest(args.writing), "Exact-excerpt study inputs differ")
    result = summarize(receipt, writing)
    source.require(result == receipt["summary"], "Exact-excerpt study totals differ")
    print(source.serialize(result), end="")


if __name__ == "__main__":
    main()
