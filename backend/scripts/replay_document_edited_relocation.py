# SPDX-License-Identifier: MIT
"""Compare edited paragraph locations from published rewrites in two draft formats."""

import argparse
import gzip
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

from scripts import writing_range_report_replication as prior
from scripts.writing_interval_transfer import fingerprints

SCHEMA = "chipmunk.document-edited-relocation-review/1"
REFERENCE = prior.pool.PANEL_DIR / "writing-paired-ranges-fresh-2026-11-15.json.gz"
FAMILIES = {prior.FAMILY: prior.SOURCE_SHA, "qwen3-8b": prior.words.parent.previous.REPORT_PINS[0]}
SIZES = {prior.FAMILY: 48, "qwen3-8b": 24}
FORMATS = (".txt", ".docx")
ROLES = ("moved_edit", "same_position_edit", "adjacent_swap_edit", "unchanged_move", "unrelated_replacement", "repeated_anchor")
EXTRA = "The custodian recorded the release number in the archive before sending the notification."
UNRELATED = "A courier collected the sealed box from a spare instrument cabinet after the visitors had gone."
require, digest = prior.require, prior.pool.sha


def plan(paths):
    table = {family: prior.words.parent.load_shard(path, "test", family, FAMILIES[family])
             for family, path in zip(FAMILIES, paths, strict=True)}
    records = prior.ranges.spans.read(REFERENCE)["held"]
    ids = {digest(key): key for key in table[prior.FAMILY]}
    reserved, seen, cases = set(), set(), []
    for original, revision in zip(records[::2], records[1::2], strict=True):
        group, position = revision["group"], revision["position"]
        key = ids[group]
        versions = table[prior.FAMILY][key]
        pair = prior.words.parent.intake.excerpt_options(key, versions)[position]
        require([digest(item[0]) for item in pair] == [original["sha256"], revision["sha256"]], "Report pair differs")
        cases.append((prior.FAMILY, group, position, pair))
        reserved.add(group)
        seen.update(fingerprints(versions["v0"]["text"]))
    second = table["qwen3-8b"]
    for key in sorted(second, key=lambda value: digest(f"{SCHEMA}/{value}")):
        if len(cases) == sum(SIZES.values()):
            break
        group = digest(key)
        if group in reserved or "v0" not in second[key]:
            continue
        phrases = set(fingerprints(second[key]["v0"]["text"]))
        if seen & phrases:
            continue
        options = prior.words.parent.intake.excerpt_options(key, second[key])
        if options:
            position = min(options, key=lambda value: digest(f"{group}/{value}"))
            cases.append(("qwen3-8b", group, position, options[position]))
            seen.update(phrases)
            reserved.add(group)
    require(Counter(row[0] for row in cases) == SIZES, "The two report rosters are incomplete")
    return cases


def variants(pair, position):
    (text, spans, _), (revised, revised_spans, _) = pair
    old, new = [body[rows[position - 1][0]:rows[position + 1][1]]
                for body, rows in ((text, spans), (revised, revised_spans))]
    # Keep both immediate neighbors inside the passage and the other two publisher sentences outside it.
    a, b = (text[start:end] for index, (start, end) in enumerate(spans) if abs(position - index) > 1)
    base = [a, old, b, EXTRA]
    return {
        "moved_edit": (base, [a, b, EXTRA, new]),
        "same_position_edit": (base, [a, new, b, EXTRA]),
        "adjacent_swap_edit": (base, [a, b, new, EXTRA]),
        "unchanged_move": (base, [a, b, EXTRA, old]),
        "unrelated_replacement": (base, [a, b, EXTRA, UNRELATED]),
        "repeated_anchor": ([a, old, b, old + " " + EXTRA], [a, b, old + " " + EXTRA, new]),
    }


def encode(paragraphs, extension):
    if extension == ".txt":
        return "\n\n".join(paragraphs).encode()
    from docx import Document

    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    data = BytesIO()
    document.save(data)
    return data.getvalue()


def observe(result, bodies):
    excerpts = {side: [row[side] for row in (*result["changes"], *result.get("relocated_passages", ())) if row[side]]
                for side in ("earlier", "later")}
    spans = []
    for side, body in zip(excerpts, bodies, strict=True):
        rows = [span for item in excerpts[side] for span in item["changed_tokens"]]
        require(all(0 <= start < end <= len(body) for start, end in rows), "A reported token is outside its source")
        spans.append(rows)
    passages = result.get("relocated_passages", ())
    bounds = [[row[side]["start"], row[side]["end"]] for row in passages for side in ("earlier", "later")]
    return {"schema": result["schema"], "passage_ranges": bounds, "tokens": spans}


def measure(paths):
    from app.evidence import document_revision as drafts

    records = []
    for family, group, position, pair in plan(paths):
        cases = variants(pair, position)
        mapped = []
        for role, paragraphs in cases.items():
            for extension in FORMATS:
                earlier, later = (encode(body, extension) for body in paragraphs)
                before, old_text, new_text, _ = drafts.compare_with_sources(earlier, later, extension, include_edited_relocations=False)
                after = drafts.compare(earlier, later, extension)
                bodies = old_text, new_text
                truth, bounds = [], []
                if role == "moved_edit":
                    for body, index, version in zip(bodies, (1, 3), pair, strict=True):
                        start, end = version[1][position]
                        decoded = body.split("\n\n")
                        passage_start = sum(len(paragraph) + 2 for paragraph in decoded[:index])
                        passage_end = passage_start + len(decoded[index])
                        left = passage_start + start - version[1][position - 1][0]
                        truth.append([left, left + end - start])
                        bounds.append([passage_start, passage_end])
                mapped.append({"role": role, "format": extension, "truth": truth, "passage": bounds, "identical_comparison": before == after,
                               "decoded_sha256": [digest(text) for text in bodies],
                               "decoded_characters": [len(text) for text in bodies],
                               "previous": observe(before, bodies),
                               "updated": observe(after, bodies)})
        records.append({"family": family, "group": group, "rows": mapped})
    return {"schema": SCHEMA, "source_shards": FAMILIES, "reference_sha256": prior.pool.file_hash(REFERENCE), "works": records}


def outside(reading, truth):
    return any(not low <= start < end <= high for (low, high), spans in zip(truth, reading["tokens"], strict=True)
               for start, end in spans)


def review(record):
    require(record["schema"] == SCHEMA and record["source_shards"] == FAMILIES
            and record["reference_sha256"] == prior.pool.file_hash(REFERENCE), "The source plan differs")
    groups = record["works"]
    require(Counter(row["family"] for row in groups) == SIZES and len({row["group"] for row in groups}) == len(groups), "A report is absent or repeated")
    totals = {family: {extension: Counter() for extension in FORMATS} for family in FAMILIES}
    for work in groups:
        require([(row["role"], row["format"]) for row in work["rows"]] == [(role, ext) for role in ROLES for ext in FORMATS],
                "A format or control is missing")
        for row in work["rows"]:
            old, new = row["previous"], row["updated"]
            truth, lengths = row["truth"], row["decoded_characters"]
            if row["role"] == "moved_edit":
                require(all(0 <= begin <= low < high <= end <= size
                            for (begin, end), (low, high), size in zip(row["passage"], truth, lengths, strict=True)),
                        "A target sentence is outside the moved paragraph")
            else:
                require(not truth and not row["passage"], "A control has an edited movement label")
            for reading in (old, new):
                require(all(0 <= start < end <= size for size, spans in zip(lengths, reading["tokens"], strict=True)
                            for start, end in spans), "A reported token is outside its source")
            old_outside, new_outside = (bool(truth) and outside(reading, truth) for reading in (old, new))
            totals[work["family"]][row["format"]].update({
                row["role"]: 1,
                "old_unrelated_highlights": int(old_outside),
                "new_unrelated_highlights": int(new_outside),
                "exact_edited_passage": int(bool(truth) and row["passage"] == new["passage_ranges"]
                                            and not new_outside and any(new["tokens"])),
                "control_changes": int(not row["truth"] and not row["identical_comparison"]),
            })
    require("summary" not in record or totals == record["summary"], "The measured counts differ")
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-shards", type=Path, nargs=2, metavar=("GPT_REPORTS", "QWEN_REPORTS"))
    args = parser.parse_args()
    if args.source_shards:
        record = measure(args.source_shards)
        record["summary"] = review(record)
        args.receipt.write_bytes(gzip.compress((json.dumps(record, sort_keys=True, indent=2) + "\n").encode(), mtime=0))
    else:
        record = json.loads(gzip.decompress(args.receipt.read_bytes()))
    print(json.dumps(review(record), sort_keys=True, indent=2))
