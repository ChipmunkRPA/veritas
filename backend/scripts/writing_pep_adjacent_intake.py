"""Replay five-sentence PEP excerpts with independently witnessed adjacent edits."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from scripts import writing_pep_revision_intake as prior
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import distance, valid_sentence, words
from scripts.writing_focus_v3 import digest
from scripts.writing_pep_legacy_intake import source as legacy_source

PLAN = {
    "comparison": "Source qualification for a single-file test. Each published version supplies five consecutive body "
                  "sentences. Require one material 12–40-word center edit between exactly matched immediate neighbors. "
                  "The two outermost context sentences may independently differ across historical versions. "
                  "No editing or substitution in either authentic control.",
    "scope": "Require at least two eligible internal positions in each five-sentence passage. "
             "Use the exact legacy renderer, dated Git roster and prior attribution and exposure checks.",
    "difference": "Require normalized lexical edit distance at least 0.15 in the anchored natural center. "
                  "The human variants are independent chronological controls, and may contain other revisions.",
}


def windows(text):
    result = defaultdict(list)
    for run in prior.sentence_runs(text, len(text)):
        for index in range(2, len(run) - 2):
            items = run[index - 2:index + 3]
            bodies = [item["text"] for item in items]
            counts = [len(words(value)) for value in bodies]
            if (not 12 <= counts[2] <= 40 or sum(12 <= n <= 40 for n in counts[1:4]) < 2
                    or not all(6 <= n <= 65 for n in counts) or not 60 <= sum(counts) <= 225):
                continue
            value = " ".join(bodies)
            units = segment(from_string(value))
            if [unit.text for unit in units] != bodies:
                continue
            focus = units[2]
            result[bodies[1], bodies[3]].append({"text": value, "focus": [focus.start, focus.end],
                                                 "focus_text": focus.text, "witness": items})
    return result


def pair(repo, row):
    try:
        old = legacy_source(repo, row["before"], row["path"])
        new = legacy_source(repo, row["after"], row["path"])
    except (subprocess.CalledProcessError, UnicodeError, ValueError):
        return "unavailable_blob", None
    if old is None or new is None:
        return "missing_explicit_terms_or_authors", None
    if old[2] != new[2]:
        return "author_header_changed", None
    a, b = (windows(record[1]) for record in (old, new))
    options = []
    for key in a.keys() & b.keys():
        if len(a[key]) != 1 or len(b[key]) != 1:
            continue
        first, second = a[key][0], b[key][0]
        if distance(first["focus_text"], second["focus_text"]) >= .15 and valid_sentence(
            {"text": second["focus_text"], "eos": True}, first["focus_text"], generated=False
        ):
            options.append((first, second))
    if not options:
        return "no_material_center", None
    first, second = min(options, key=lambda values: text_hash(values[0]["text"] + "\0" + values[1]["text"]))
    parent = int(prior.git(repo, "show", "-s", "--format=%ct", row["before"]).strip())
    if parent > row["after_committed"]:
        return "nonchronological_parent", None
    outer_match = all(first["witness"][i]["text"] == second["witness"][i]["text"] for i in (0, 4))
    return "qualified", {
        "publisher": "python-peps", "license": "Public-domain or CC0 statement in both exact Git blobs",
        "attribution": old[2], "before": first, "after": second, "outer_sentences_match": outer_match,
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
    return selected, {"schema": prior.SCHEMA, "plan": PLAN, "work_pool": len(roster), "counts": dict(dispositions),
                      "mechanical_groups": len(found), "qualified_groups": len(selected),
                      "unchanged_outer_context_groups": sum(row["outer_sentences_match"] for row in selected),
                      "history": history}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "qualify", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("prior", type=Path)
    cli.add_argument("legacy", type=Path)
    cli.add_argument("repo", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    lock = args.run / "source-freeze.json"
    initial = prior.freeze(args.repo)
    require(read(args.prior / "source-freeze.json") == initial, "Initial Git roster or history differs")
    frozen = {"method": PLAN, "source_code_sha256": digest(Path(__file__)),
              "earlier_source_sha256": digest(args.prior / "source-freeze.json"),
              "legacy_intake_sha256": digest(args.legacy / "source-receipt.json"),
              "legacy_layout_code_sha256": digest(Path(__file__).with_name("writing_pep_legacy_intake.py"))}
    if args.phase == "freeze":
        args.run.mkdir()
        write(lock, frozen)
        return
    require(read(lock) == frozen, "Frozen source plan differs")
    selected, summary = qualify(args.repo, initial["work_roster"])
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
