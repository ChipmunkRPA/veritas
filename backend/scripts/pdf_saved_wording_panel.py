# SPDX-License-Identifier: Apache-2.0
"""Rebuild retained PDF wording and ordinary save controls from pinned government publications."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audio_native_alignment_panel import digest, read, save

SCHEMA = "chipmunk.pdf-saved-wording-panel/1"
MARKERS = {"replace": "updated", "insert": "additional wording", "append": "revised"}


def _canonical_save(pdf, path, *, incremental=False, streams=0):
    import pymupdf

    if incremental:
        pdf.save(path, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP, no_new_id=True)
    else:
        pdf.save(path, no_new_id=True, use_objstms=streams)


def _working_copy(source, path, slot):
    import pymupdf

    with pymupdf.open(source) as publisher, pymupdf.open() as pdf:
        page = publisher[0]
        text = " ".join(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.,;()/-]*", page.get_text()))
        lines = textwrap.wrap(" ".join(text.split()[20:200]), 66)
        if len(lines) < 6:
            raise ValueError(f"Publisher page has too little native text: {source.name}")
        choices = [(index, word) for index, line in enumerate(lines[2:-1], 2) for word in line.split()
                   if re.fullmatch(r"[A-Za-z]{4,13}", word) and line.split().count(word) == 1 and word not in MARKERS.values()]
        if not choices:
            raise ValueError(f"Publisher page has no unambiguous test word: {source.name}")
        target_line, target_word = choices[slot % len(choices)]
        pdf.insert_pdf(publisher, from_page=0, to_page=0, links=False, annots=False, widgets=False)
        work = pdf.new_page(width=612, height=792)
        work.insert_text((56, 66), "Working copy for local evaluation", fontsize=12)
        line_stream = None
        for index, line in enumerate(lines):
            work.insert_text((56, 94 + index * 17), line, fontsize=10)
            if target_line == index:
                line_stream = work.get_contents()[-1]
        widget = pymupdf.Widget()
        widget.field_name = "review_status"
        widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
        widget.field_value = "Pending review"
        widget.rect = pymupdf.Rect(56, 595, 280, 618)
        work.add_widget(widget)
        if slot % 3 == 1:
            work.set_rotation(90)
        pdf.set_metadata({"title": "Local revision evaluation working copy"})
        _canonical_save(pdf, path, streams=slot % 2)
        return line_stream, target_word


def _edit_stream(pdf, stream, word, replacement):
    data = pdf.xref_stream(stream)
    [text] = re.finditer(rb"<([0-9a-f]+)>", data)
    words = bytes.fromhex(text[1].decode()).decode("ascii").split(" ")
    if words.count(word) != 1:
        raise ValueError("The test word does not identify one native line position")
    words[words.index(word)] = replacement
    pdf.update_stream(stream, data[:text.start(1)] + " ".join(words).encode().hex().encode() + data[text.end(1):])


def _alter(path, role, stream, word):
    import pymupdf

    with pymupdf.open(path) as pdf:
        page = pdf[1]
        if role in {"replace", "append", "full_rewrite"}:
            _edit_stream(pdf, stream, word, MARKERS.get(role, MARKERS["replace"]))
        elif role == "insert":
            _edit_stream(pdf, stream, word, f"{word} {MARKERS['insert']}")
        elif role == "delete":
            _edit_stream(pdf, stream, word, "")
        elif role == "metadata":
            pdf.set_metadata(pdf.metadata | {"subject": "A reviewer changed a file property"})
        elif role == "comment":
            page.add_text_annot((470, 90), "The reviewer left an ordinary comment.")
        elif role == "form":
            field = next(page.widgets())
            field.field_value = "Review completed"
            field.update()
        elif role == "reorder":
            streams = page.get_contents()
            pdf.xref_set_key(page.xref, "Contents", "[" + " ".join(f"{n} 0 R" for n in reversed(streams)) + "]")
        elif role == "added_page":
            pdf.new_page()
        if role == "full_rewrite":
            replacement = pdf.tobytes(no_new_id=True)
        else:
            _canonical_save(pdf, path, incremental=True)
    if role == "full_rewrite":
        path.write_bytes(replacement)
    elif role == "append":
        _alter(path, "comment", stream, word)


def prepare(sources, roster, split, output):
    import pymupdf

    from app.evidence import document_pdf_history

    output.mkdir(mode=0o700)
    rows = []
    records = read(roster)
    prepared = [record for record in records if record["split"] == split and "sha256" in record]
    for slot, record in enumerate(prepared):
        name = record["name"]
        source = sources / f"{name}.pdf"
        if digest(source) != record["sha256"]:
            raise ValueError(f"Publisher file differs: {source.name}")
        path = output / f"{name}-source.pdf"
        path.write_bytes(source.read_bytes())
        rows.append({"id": name, "role": "publisher", "file": path.name, "input_sha256": digest(path),
                     "parent_sha256": record["sha256"], "expected_words": None, "older_word": None,
                     "report": document_pdf_history.inspect_path(path)})
        base = output / f"{name}-base.pdf"
        stream, word = _working_copy(source, base, slot)
        content = base.read_bytes()
        for role in ("replace", "append", "insert", "delete", "metadata", "comment", "form", "reorder", "added_page", "full_rewrite"):
            path = output / f"{name}-{role}.pdf"
            path.write_bytes(content)
            _alter(path, role, stream, word)
            reading = document_pdf_history.inspect_path(path)
            rows.append({"id": name, "role": role, "file": path.name, "input_sha256": digest(path),
                         "parent_sha256": digest(base), "expected_words": MARKERS.get(role),
                         "older_word": word if role in {"replace", "append", "delete"} else None,
                         "report": reading})
    save(output / "result.json.gz", {"schema": SCHEMA, "split": split, "rows": rows, "totals": totals(rows),
         "roster_sha256": digest(roster), "implementation_sha256": digest(Path(document_pdf_history.__file__)),
         "recipe_sha256": digest(Path(__file__)), "pymupdf": pymupdf.VersionBind})


def totals(rows):
    table = defaultdict(Counter)
    for row in rows:
        role, record = row["role"], row["report"]
        steps = record["comparisons"]
        changes = steps[-1]["changes"] if steps else []
        wanted = row["expected_words"]
        localized = bool(wanted and any(change["current"] and change["current"]["page"] == 2
                                       and wanted == change["current"]["excerpt"] for change in changes))
        if role == "delete":
            localized = any(change["kind"] == "delete" and change["earlier"]["page"] == 2 for change in changes)
        if role == "added_page":
            localized = bool(steps and steps[-1]["current_only_pages"] == [3])
        table[role].update(n=1, changed=record["status"] == "changed", located=localized,
                           no_change=record["status"] == "no_change", no_history=record["status"] == "no_history",
                           unavailable=record["status"] == "unavailable")
    return table


def review(receipt):
    result = read(receipt)
    counts = totals(result["rows"])
    if result["schema"] != SCHEMA or result["totals"] != counts:
        raise ValueError("Published PDF revision results do not recount")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="command", required=True)
    build = actions.add_parser("prepare")
    build.add_argument("sources", type=Path)
    build.add_argument("roster", type=Path)
    build.add_argument("split", choices=("development", "confirmation"))
    build.add_argument("output", type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    arguments = vars(cli.parse_args())
    globals()[arguments.pop("command")](**arguments)
