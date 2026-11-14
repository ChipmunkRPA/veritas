# SPDX-License-Identifier: Apache-2.0
"""Measure how much changed wording the editorial worker's evidence actually covers."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from scripts import writing_editorial_worker_audit as audit
from scripts.audio_vocal_event_mixtures import read, save, sha

SCHEMA = "chipmunk.writing-editorial-word-overlap/1"
TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def changed_tokens(earlier, later):
    old = [span.group().casefold() for span in TOKEN.finditer(earlier)]
    new = list(TOKEN.finditer(later))
    edits = SequenceMatcher(None, old, [span.group().casefold() for span in new], autojunk=False)
    changed = []
    for action, _, _, start, end in edits.get_opcodes():
        if action in {"insert", "replace"}:
            changed.extend([span.start(), span.end()] for span in new[start:end])
    return changed


def compare(worker, rows):
    counts = defaultdict(Counter)
    if len(rows) != len(worker["rows"]):
        raise ValueError("Missing planned cases")
    for measured, row in zip(worker["rows"], rows, strict=True):
        if ((measured["group"], measured["role"], measured["text_sha256"])
                != (row["group"], row["role"], row["text_sha256"])):
            raise ValueError("Word evidence differs from the worker input")
        focus = measured["focus"]
        changes = row["changed_words"]
        if (not all(focus[0] <= a < b <= focus[1] for a, b in changes)
                or any(x[1] > y[0] for x, y in zip(changes, changes[1:]))
                or (row["role"] == "before" and changes)):
            raise ValueError("Changed word intervals leave the known center")
        marks = {i for card in measured["cards"] for a, b in card["spans"] for i in range(a, b)} - set(measured["spaces"])
        hit = sum(all(i in marks for i in range(a, b)) for a, b in changes)
        touched = {i for a, b in changes for i in range(a, b)}
        cell = counts[measured["role"]]
        cell.update(files=1, files_with_changed_words=bool(changes), changed_words=len(changes), covered_changed_words=hit,
                    files_with_any_changed_word_covered=hit > 0, files_with_half_words_covered=bool(changes) and 2 * hit >= len(changes),
                    marked_characters=len(marks), marked_changed_word_characters=len(marks.intersection(touched)))
    return {key: dict(value) for key, value in counts.items()}


def run(source, worker, output):
    inputs = audit.plan(source)
    receipt = read(worker)
    originals = {row["group"]: row["text"][slice(*row["focus"])] for row in inputs if row["role"] == "before"}
    records = []
    for native in inputs:
        start, stop = native["focus"]
        words = changed_tokens(originals[native["group"]], native["text"][start:stop])
        records.append({key: native[key] for key in ("group", "role", "text_sha256")} | {
            "changed_words": [[start + a, start + b] for a, b in words],
        })
    result = {"schema": SCHEMA, "worker_sha256": sha(worker), "rows": records, "results": compare(receipt, records)}
    save(output, result)
    return result["results"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.phase == "run":
        if args.source is None or args.path.exists():
            parser.error("Use a new output path and the exact retained text roster")
        result = run(args.source, args.worker, args.path)
    else:
        receipt = read(args.path)
        if receipt["schema"] != SCHEMA or receipt["worker_sha256"] != sha(args.worker):
            raise ValueError("Changed text and evidence ledger bindings differ")
        result = compare(read(args.worker), receipt["rows"])
        if result != receipt["results"]:
            raise ValueError("Saved word overlap results do not recount")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
