"""Verify Word paragraph coordinates and score parity on previously held draft pairs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from scripts import audio_vocal_event_mixtures as io
from scripts.writing_paired_scale_transfer import cases, increases, sha

SCHEMA = "chipmunk.document-word-anchor-transfer/1"
DOMAINS = ("news", "abstracts")
TITLE = "This additional review note records the changes requested for the next meeting."
PREFIX = TITLE + "\n\n"
DELIVERIES = ("same_body", "extra_paragraph")
SHA = re.compile("[0-9a-f]{64}")


def parent_paths(domain):
    return tuple(io.PUBLIC / f"writing-{name}-{domain}-2026-11-14.json" for name in ("paired-scale", "independent-anchor"))


def frozen_readings(row):
    if row["scope"] == "current":
        return {(tuple(extent["earlier"]), tuple(extent["later"])): readings["context"]
                for extent, readings in zip(row["ranges"], row["readings"], strict=True)}
    return {(tuple(record["earlier"]["context_spans"][1]), tuple(record["later"]["context_spans"][1])):
            record["reading"]["context"] for record in row["independent"]}


def native_pairs(root, directory, domain):
    base, independent = (io.read(path) for path in parent_paths(domain))
    *inputs, _ = cases(root, directory, domain)
    for item, row, saved in zip(inputs, base["rows"], independent["rows"], strict=True):
        bank, earlier, later, labels, variant = item
        hashes = [sha(earlier), sha(later)]
        if (bank, hashes, labels, variant) != (row["bank"], row["source_sha256"], row["truth"], row["variant"]):
            raise ValueError("Native texts differ from their prior committed selection")
        if hashes != saved["source_sha256"] or labels != saved["truth"]:
            raise ValueError("The earlier model readings apply to different texts")
        yield earlier, later, saved


def transfer(earlier, later, saved, delivery):
    from app.evidence.document_revision import compare_with_sources
    from app.evidence.document_revision_writing import (
        WORD_INDEPENDENT_SCHEMA,
        independently_anchored_revisions,
        multiple_result,
        paired_input,
        separate_revisions,
        verify,
    )
    from scripts.rebuild_writing_word_boundary_sources import docx

    formatted = PREFIX + later if delivery == "extra_paragraph" else later
    older, newer = docx(earlier), docx(formatted)
    comparison, old, new, layouts = compare_with_sources(older, newer, ".docx")
    if (old, new) != (earlier, formatted):
        raise ValueError("Word extraction altered the native sentence inputs")
    existing = paired_input(comparison, old, new, allow_boundary=True, word_layouts=layouts)
    options = [] if existing else independently_anchored_revisions(comparison, old, new, word_layouts=layouts)
    prior = frozen_readings(saved)
    prefix = len(PREFIX) if delivery == "extra_paragraph" else 0
    reviews = []
    if options:
        plain, before, after = compare_with_sources(earlier.encode(), later.encode(), ".txt")[:3]
        text_candidates = (separate_revisions(plain, before, after) if saved["scope"] == "current"
                           else independently_anchored_revisions(plain, before, after))
        reference = {tuple(tuple(item.geometry[side]["context_spans"][1]) for side in ("earlier", "later")):
                     [item.geometry[side]["input_sha256"] for side in ("earlier", "later")] for item in text_candidates}
        expected = []
        for candidate in options:
            keys = tuple(tuple(candidate.geometry[side]["context_spans"][1]) for side in ("earlier", "later"))
            key = keys[0], tuple(n - prefix for n in keys[1])
            if key not in prior or [candidate.geometry[side]["input_sha256"] for side in ("earlier", "later")] != reference.get(key):
                raise ValueError("The Word comparison inferred a different passage from the frozen source")
            expected.append(prior[key])
        exported = multiple_result(options, expected, independent=True, draft_format="docx")
        if exported["schema"] != WORD_INDEPENDENT_SCHEMA:
            raise ValueError("The Word neighborhoods have the wrong comparison schema")
        verify(comparison, old, new, exported, word_layouts=layouts)
        for revision in exported["revisions"]:
            spans = [revision[side]["context_spans"][1] for side in ("earlier", "later")]
            contexts = [revision[side]["context_paragraphs"] for side in ("earlier", "later")]
            reviews.append({"spans": spans, "paragraphs": contexts, "status": revision["status"]})
    return {"word_sha256": [io.digest(raw) for raw in (older, newer)], "existing_assessed": bool(existing), "reviews": reviews}


def check(doc):
    if (doc["schema"] != SCHEMA
            or doc["sources"] != {domain: [io.sha(path) for path in parent_paths(domain)] for domain in DOMAINS}):
        raise ValueError("Source receipts changed")
    summary = {}
    for domain in DOMAINS:
        parent_hashes = io.read(parent_paths(domain)[1])["rows"]
        rows = doc["domains"][domain]
        if len(rows) != len(parent_hashes):
            raise ValueError("Missing frozen Word inputs")
        for index, (row, saved) in enumerate(zip(rows, parent_hashes, strict=True)):
            if row["text_sha256"] != saved["source_sha256"] or set(row["deliveries"]) != set(DELIVERIES):
                raise ValueError("Wrong transport source")
            prior = frozen_readings(saved)
            labels = {tuple(item["later"]): item for item in saved["truth"]}
            for delivery, outcome in row["deliveries"].items():
                counts = summary.setdefault(f"{domain}/{delivery}", Counter())
                counts.update(pairs=1, existing_assessed=outcome["existing_assessed"],
                              eligible=len(outcome["reviews"]), prior_measured=len(prior))
                if (len(outcome["reviews"]) > len(prior) or outcome["existing_assessed"] and outcome["reviews"]
                        or len(outcome["word_sha256"]) != 2 or not all(SHA.fullmatch(pin) for pin in outcome["word_sha256"])):
                    raise ValueError(f"Unexpected or repeated assessment on source {index}")
                offset = len(PREFIX) if delivery == "extra_paragraph" else 0
                seen = set()
                for review in outcome["reviews"]:
                    (old, new), (old_paras, new_paras) = review["spans"], review["paragraphs"]
                    key = tuple(old), tuple(n - offset for n in new)
                    if (key in seen or key not in prior or any(len(group) != 3 or len({p["paragraph"] for p in group}) != 1
                                                for group in (old_paras, new_paras))
                            or any(b["end"] - b["start"] != span[1] - span[0]
                                   for b, span in zip((old_paras[1], new_paras[1]), (old, new), strict=True))):
                        raise ValueError("The measured context differs from the original sentence")
                    seen.add(key)
                    measured = prior[key]
                    status = ("joint_increase" if min(increases(measured)) >= math.log(2)
                              else "no_joint_increase") if measured else "unavailable"
                    if review["status"] != status:
                        raise ValueError("Word and text score dispositions differ")
                    label = labels.get(key[1])
                    role = "unassigned" if label is None else "ordinary" if label["family"] == "ordinary" else "generated"
                    counts.update({f"{role}_mapped": 1, f"{role}_flagged": status == "joint_increase"})
                    counts["word_positions_changed"] += new_paras[1]["paragraph"] != old_paras[1]["paragraph"]
    return summary


def run(root, directories):
    result = {}
    for domain, directory in zip(DOMAINS, directories, strict=True):
        result[domain] = [
            {"text_sha256": saved["source_sha256"], "deliveries": {
                case: transfer(earlier, later, saved, case) for case in DELIVERIES}}
            for earlier, later, saved in native_pairs(root, directory, domain)
        ]
        print("Completed", domain, len(result[domain]), flush=True)
    return {"schema": SCHEMA, "sources": {domain: [io.sha(path) for path in parent_paths(domain)] for domain in DOMAINS},
            "domains": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--abstracts", type=Path)
    args = parser.parse_args()
    if args.root:
        if not (args.news and args.abstracts) or args.receipt.exists():
            parser.error("Native replay requires all source banks and a new receipt path")
        doc = run(args.root, (args.news, args.abstracts))
        doc["results"] = check(doc)
        io.save(args.receipt, doc)
    else:
        doc = io.read(args.receipt)
    observed = check(doc)
    if observed != doc["results"]:
        raise ValueError("The receipt counts differ")
    print(json.dumps(observed, indent=2))


if __name__ == "__main__":
    main()
