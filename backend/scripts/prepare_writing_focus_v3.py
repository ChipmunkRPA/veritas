"""Prepare consumed sentence/context positions, without authorship inference.

Output contains research source text and provenance labels; keep it private.
No new training or redistribution grant is inferred from conflicting data terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import from_string
from app.detectors.text.pipeline import windows
from app.detectors.text.segment import segment
from scripts.audit_writing_sources_v2 import PRIOR_SHA, SPAN_SHA, check_row

HISTORICAL_PASSAGES_SHA = "33fd38ce148fbbaf507d4df8a5ddfac4df215aa3cd597573bf23d535657b53b3"


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_labels(doc, labels):
    if any(v not in ("User", "Bot") for v in labels):
        raise ValueError("Explicit source character provenance required")
    result = {}
    for paragraph in doc.source_map:
        for start, end, first, last in paragraph["source_ranges"]:
            if not 0 <= start <= end <= len(doc.text) or not 0 <= first <= last <= len(labels):
                raise ValueError("Source position outside immutable text")
            if end - start != last - first:
                if any(not c.isspace() for c in doc.text[start:end]):
                    raise ValueError("Normalization cannot preserve this non-whitespace provenance")
                continue
            for position, label in zip(range(start, end), labels[first:last], strict=True):
                if position in result and result[position] != label:
                    raise ValueError("Conflicting source attribution")
                result[position] = label
    if any(i not in result for i, c in enumerate(doc.text) if not c.isspace()):
        raise ValueError("Extracted text is missing character provenance")
    return result


def focus_record(doc, labels, start, end, context):
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(doc.text):
        raise ValueError("Exact nonempty extracted focus coordinates required")
    counts = Counter(labels[i] for i in range(start, end) if not doc.text[i].isspace())
    if not sum(counts.values()):
        raise ValueError("Whitespace alone cannot supply writing evidence")
    focus = doc.text[start:end]
    return {"start": start, "end": end, "offset_unit": "Unicode code points, end exclusive",
            "focus": focus, "focus_sha256": sha(focus), "context": context,
            "context_sha256": sha(context), "human_characters": counts["User"],
            "ai_characters": counts["Bot"], "focus_ai_fraction": counts["Bot"] / sum(counts.values())}


def prepare(manifest_path, spans_path, previous_path, output, window_sentences):
    import pyarrow.parquet as pq

    if output.exists():
        raise ValueError("Refuse to overwrite an existing research input packet")
    if type(window_sentences) is not int or window_sentences < 1 or window_sentences % 2 != 1:
        raise ValueError("Positive odd sentence context size required")
    if (file_sha(manifest_path) != PRIOR_SHA or file_sha(spans_path) != SPAN_SHA
            or file_sha(previous_path) != HISTORICAL_PASSAGES_SHA):
        raise ValueError("Exact consumed source and original passage references required")
    manifest = json.loads(manifest_path.read_text())
    previous = json.loads(previous_path.read_text())
    historical = {(r["id"], r["sentence"]): r for r in previous["records"]}
    annotations = pq.read_table(spans_path).to_pylist()
    records, matched = [], set()
    for case in manifest["cases"]:
        if case["label"] != "mixed":
            continue
        row = annotations[int(case["id"].split(":")[1])]
        original, _ = check_row(row)
        if original != case["text"] or sha(original) != case["text_sha256"]:
            raise ValueError("Historical case no longer binds to its source")
        labels = [part["author"] for part in row["data"] for _ in part["text"]]
        doc = from_string(original)
        projected = project_labels(doc, labels)
        sentences = segment(doc)
        contexts = windows(sentences, window_sentences)
        foci, covered = [], set()
        for index, (sentence, context) in enumerate(zip(sentences, contexts, strict=True)):
            focus = focus_record(doc, projected, sentence.start, sentence.end, context)
            positions = {i for i in range(sentence.start, sentence.end) if not doc.text[i].isspace()}
            if covered & positions:
                raise ValueError("Focus evidence units overlap and would double count characters")
            covered.update(positions)
            key = (case["id"], index)
            prior = historical[key]
            if (any(focus[k] != prior[k] for k in ("start", "end", "context_sha256", "focus_ai_fraction"))
                    or prior["extracted_text_sha256"] != sha(doc.text)):
                raise ValueError("Current extraction/context differs from consumed passage inputs")
            matched.add(key)
            foci.append({"sentence": index, **focus})
        uncovered = Counter(projected[i] for i, c in enumerate(doc.text) if not c.isspace() and i not in covered)
        total = Counter(projected[i] for i, c in enumerate(doc.text) if not c.isspace())
        records.append({"id": case["id"], "author_group": sha("writing-focus-author:" + row["metadata"]["author"]),
                        "source_text_sha256": sha(original), "extracted_text_sha256": sha(doc.text),
                        "total_human_characters": total["User"], "total_ai_characters": total["Bot"],
                        "uncovered_human_characters": uncovered["User"], "uncovered_ai_characters": uncovered["Bot"],
                        "foci": foci})
    if matched != set(historical) or len(records) != 40 or sum(len(r["foci"]) for r in records) != 674:
        raise ValueError("Every consumed40-document674-focus input must remain")
    header = {"schema": "chipmunk.writing-focus-inputs/3", "manifest_sha256": PRIOR_SHA,
              "spans_sha256": SPAN_SHA, "historical_passages_sha256": HISTORICAL_PASSAGES_SHA,
              "preparation_code_sha256": file_sha(Path(__file__)), "window_sentences": window_sentences,
              "documents": len(records), "foci": 674, "classifier_forwards": 0,
              "scope": "Consumed development, publisher character-provenance labels; private research input text. No training, calibrated confidence or fresh validation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for value in [header, *records]:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
    print(json.dumps({"documents": len(records), "author_groups": len({r["author_group"] for r in records}),
                      "foci": 674, "historical_inputs_matched": len(matched),
                      "uncovered_ai_characters": sum(r["uncovered_ai_characters"] for r in records),
                      "classifier_forwards": 0, "input_packet_sha256": file_sha(output)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("spans", type=Path)
    p.add_argument("previous", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--window-sentences", type=int, default=3)
    a = p.parse_args()
    prepare(a.manifest, a.spans, a.previous, a.output, a.window_sentences)
