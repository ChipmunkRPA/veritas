# SPDX-License-Identifier: Apache-2.0
"""Verify saved-version source prefixes and check selected word boxes with Poppler."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from defusedxml import ElementTree

from scripts.audio_native_alignment_panel import digest, read, save
from scripts.pdf_saved_wording_panel import SCHEMA, totals


def _poppler(data, page):
    xml = subprocess.run(  # noqa: S603 - fixed local reader, stdin only
        ["pdftotext", "-f", str(page), "-l", str(page), "-bbox", "-", "-"],
        input=data, capture_output=True, check=True, timeout=20).stdout
    return [(item.text or "", [float(item.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax")])
            for item in ElementTree.fromstring(xml).iter() if item.tag.rsplit("}", 1)[-1] == "word"]


def _rotated(box, side):
    a, b, c, d = box
    width, height, rotation = (side[key] for key in ("width", "height", "rotation"))
    return {0: box, 90: [height - d, a, height - b, c], 180: [width - c, height - d, width - a, height - b],
            270: [b, width - c, d, width - a]}[rotation]


def _overlap(a, b):
    shared_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    shared_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = min((box[2] - box[0]) * (box[3] - box[1]) for box in (a, b))
    return shared_width * shared_height / area if area > 0 else 0


def _check_words(data, location, phrase):
    words = _poppler(data, location["page"])
    expected = phrase.split()
    if len(expected) != len(location["boxes"]):
        raise ValueError("Independent excerpt differs from the number of committed native boxes")
    for literal, bbox in zip(expected, location["boxes"], strict=True):
        measured = _rotated(bbox, location)
        if not any(word == literal and _overlap(box, measured) >= .6 for word, box in words):
            raise ValueError(f"The selected word has no corresponding independent bounds on page {location['page']}")
    return len(expected)


def verify_row(folder, row):
    from app.evidence.document_pdf_history import inspect_path

    delivered = folder / row["file"]
    raw = delivered.read_bytes()
    parent = delivered if row["role"] == "publisher" else folder / f"{row['id']}-base.pdf"
    if digest(delivered) != row["input_sha256"] or digest(parent) != row["parent_sha256"]:
        raise ValueError(f"Byte commitment differs for {delivered.name}")
    record = inspect_path(delivered)
    if record != row["report"]:
        raise ValueError(f"Released inspection differs from the measured reading for {delivered.name}")
    result = Counter(files=1, prefix_bindings=0, independent_word_boxes=0)
    revisions = {}
    for revision in record["saves"]:
        prefix = raw[:revision["prefix_bytes"]]
        if hashlib.sha256(prefix).hexdigest() != revision["sha256"]:
            raise ValueError("PDF revision prefix hash differs")
        revisions[revision["version"]] = prefix
        result["prefix_bindings"] += 1
    if row["role"] not in {"publisher", "full_rewrite"} and not raw.startswith(parent.read_bytes()):
        raise ValueError("Incremental PDF output does not start with the exact working copy")
    if record["status"] == "changed":
        report = record["comparisons"][-1]
        older = revisions[report["earlier_version"]]
        for change in report["changes"]:
            if change["current"] and row["expected_words"]:
                result["independent_word_boxes"] += _check_words(raw, change["current"], row["expected_words"])
            if change["earlier"] and row["older_word"]:
                result["independent_word_boxes"] += _check_words(older, change["earlier"], row["older_word"])
    return result


def audit(receipt, folder, output):
    from app.evidence import document_pdf_history, document_pdf_history_binding

    record = read(receipt)
    if record["schema"] != SCHEMA or record["totals"] != totals(record["rows"]):
        raise ValueError("Published PDF wording counts did not reproduce")
    counts = Counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda row: verify_row(folder, row), record["rows"]):
            counts.update(result)
    result = {"schema": "chipmunk.pdf-saved-wording-audit/1", "result_sha256": digest(receipt),
              "released_reader_sha256": digest(Path(document_pdf_history.__file__)),
              "released_binding_sha256": digest(Path(document_pdf_history_binding.__file__)), **counts}
    save(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "folder", "output"):
        cli.add_argument(name, type=Path)
    audit(**vars(cli.parse_args()))
