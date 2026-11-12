"""Replay dated Python PEP revisions with the publisher's legacy text layout."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import text_hash
from scripts import writing_pep_revision_intake as prior
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import distance, valid_sentence
from scripts.writing_focus_v3 import digest

LEGACY_COPYRIGHT = re.compile(r"(?m)^Copyright(?: and Licen[cs]e)?[ \t]*\n(?=\n| {4})")
PLAIN_CONTENT = re.compile(r"(?m)^Content-Type:[ \t]*text/plain[ \t]*$")
INLINE_LITERAL = re.compile(r"(?<!`)``([\w.:-]+)``(?!`)")
REFERENCE = re.compile(r"\s*\[\d{1,3}\]_")
LAYOUT = {
    "geometry": "Keep exact five-sentence center edits, length, word distance, and overlap conditions. In legacy TXT "
                "normalize the four-space body indent and match the plain Copyright heading. In either format render "
                "only simple inline literal names and numeric reference labels. Leave other markup unassessed.",
    "binding": "Keep original Git blob hashes and normalized-document and paragraph offsets. Replay the renderer from the Git blob.",
}


def source(repo, commit, path):
    raw = prior.git(repo, "show", f"{commit}:{path}", limit=prior.MAX_FILE_BYTES)
    old_text = path.endswith(".txt") and not re.search(r"(?m)^Content-Type:\s*(?!text/plain\s*$)\S+", raw)
    old_text = old_text or bool(PLAIN_CONTENT.search(raw))
    mark = (LEGACY_COPYRIGHT.search(raw) if old_text else None) or prior.HEADING.search(raw)
    people = prior.authors(raw)
    if not mark or not people:
        return None
    rights = next((s.strip() for s in raw[mark.end():].split("\n\n", maxsplit=2) if s.strip()), "")
    if not prior.LICENSE.search(rights) or re.search(r"all rights reserved|non.commercial|except ", rights, re.I):
        return None
    prose = raw[:mark.start()]
    if old_text:
        prose = re.sub(r"(?m)^ {4}(?!\s)", "", prose)
    prose = INLINE_LITERAL.sub(r"\1", prose)
    prose = REFERENCE.sub("", prose)
    return raw, prose, people


def pair(repo, row):
    try:
        old = source(repo, row["before"], row["path"])
        new = source(repo, row["after"], row["path"])
    except (subprocess.CalledProcessError, UnicodeError, ValueError):
        return "unavailable_blob", None
    if not old or not new:
        return "missing_explicit_terms_or_authors", None
    if old[2] != new[2]:
        return "author_header_changed", None
    a, b = (prior.windows(record[1], len(record[1])) for record in (old, new))
    options = []
    for key in a.keys() & b.keys():
        if len(a[key]) != 1 or len(b[key]) != 1:
            continue
        first, second = a[key][0], b[key][0]
        old_focus, new_focus = first["focus_text"], second["focus_text"]
        if distance(old_focus, new_focus) >= .15 and valid_sentence(
            {"text": new_focus, "eos": True}, old_focus, generated=False
        ):
            options.append((first, second))
    if not options:
        return "no_material_center", None
    first, second = min(options, key=lambda values: text_hash(values[0]["text"] + "\0" + values[1]["text"]))
    parent = int(prior.git(repo, "show", "-s", "--format=%ct", row["before"]).strip())
    if parent > row["after_committed"]:
        return "nonchronological_parent", None
    return "qualified", {
        "publisher": "python-peps", "license": "Public-domain or CC0 statement in both exact Git blobs",
        "attribution": old[2], "before": first, "after": second,
        "source": row | {"before_committed": parent, "before_blob_sha256": text_hash(old[0]),
                         "after_blob_sha256": text_hash(new[0]), "before_prose_sha256": text_hash(old[1]),
                         "after_prose_sha256": text_hash(new[1])},
    }


def qualify(repo, roster):
    found, dispositions = {}, Counter()
    for number, candidates in sorted(roster.items(), key=lambda item: text_hash(item[0])):
        for row in candidates:
            status, record = pair(repo, row)
            dispositions[status] += 1
            if record is not None:
                found[number] = record | {"group": f"pep-{number}"}
                break
    snippets = {key: fingerprints(row["before"]["text"]) | fingerprints(row["after"]["text"])
                for key, row in found.items()}
    history, external, internal = overlap(
        [(key, digest(path), None, prior.historical_texts(path)) for key, path in prior.EXPOSURE.items()], snippets
    )
    selected, names, editors = [], set(), set()
    for key, row in found.items():
        editor = row["source"]["editor"]
        if key in external or key in internal:
            dispositions["text_overlap_rejected"] += 1
        elif set(row["attribution"]) & names or editor in editors:
            dispositions["identity_overlap_rejected"] += 1
        elif len(selected) == prior.MAX_GROUPS:
            dispositions["qualified_over_cap"] += 1
        else:
            names.update(row["attribution"])
            editors.add(editor)
            selected.append(row)
    return selected, {"schema": prior.SCHEMA, "layout": LAYOUT, "work_pool": len(roster), "counts": dict(dispositions),
                      "mechanical_groups": len(found), "qualified_groups": len(selected), "history": history}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "qualify", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("previous", type=Path)
    cli.add_argument("repo", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    lock, previous = args.run / "source-freeze.json", args.previous / "source-freeze.json"
    prior_plan = prior.freeze(args.repo)
    require(read(previous) == prior_plan, "Prior fixed metadata roster or history differs")
    frozen = {"layout": LAYOUT, "source_code_sha256": digest(Path(__file__)), "prior_freeze_sha256": digest(previous),
              "prior_result_sha256": digest(args.previous / "source-receipt.json")}
    if args.phase == "freeze":
        args.run.mkdir()
        write(lock, frozen)
        return
    require(read(lock) == frozen, "Frozen method differs")
    selected, summary = qualify(args.repo, prior_plan["work_roster"])
    originals, receipt = args.run / "originals.json", args.run / "source-receipt.json"
    if args.phase == "qualify":
        write(originals, selected)
        write(receipt, summary | {"source_freeze_sha256": digest(lock), "originals_sha256": digest(originals)})
    else:
        require(read(originals) == selected and read(receipt) == summary | {
            "source_freeze_sha256": digest(lock), "originals_sha256": digest(originals)
        }, "Independent source replay differs")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
