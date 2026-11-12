"""Qualify archived human PEP revisions by a bounded prose difference."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections import Counter
from difflib import SequenceMatcher
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
    "scope": "Single-file prospective admission. Use the same 200 PEP histories, exact publisher licenses, source "
             "renderer, recorded identity separation and nine retained history packets.",
    "human_pair": "Within a bounded Git parent-to-child prose diff, locate an interior one-to-one sentence replacement "
                  "bracketed by exact matched prose in the full source. Five consecutive sentences in each authentic "
                  "file may differ outside the changed center. Preserve both texts and source anchors as published.",
    "length": "Require 12–40 words per revised center, 6–65 words per neighboring sentence, 60–225 words per "
              "native five-sentence excerpt, and at least two eligible interior candidate positions.",
    "difference": "Select only an equal-length diff replacement opcode with 1–3 sentences and exact equal opcodes "
                  "on either side. Lexical edit distance between 0.10 and 0.70 at the scored center.",
}


def views(text):
    sequence, selected = [], {}
    for run in prior.sentence_runs(text, len(text)):
        start = len(sequence)
        sequence.extend(item["text"] for item in run)
        for index in range(2, len(run) - 2):
            items = run[index - 2:index + 3]
            sentences = [item["text"] for item in items]
            sizes = [len(words(value)) for value in sentences]
            if (not 12 <= sizes[2] <= 40 or sum(12 <= n <= 40 for n in sizes[1:4]) < 2
                    or not all(6 <= n <= 65 for n in sizes) or not 60 <= sum(sizes) <= 225):
                continue
            text = " ".join(sentences)
            units = segment(from_string(text))
            if [unit.text for unit in units] != sentences:
                continue
            center = units[2]
            selected[start + index] = {"text": text, "focus": [center.start, center.end],
                                       "focus_text": center.text, "witness": items}
    return sequence, selected


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
    (a, before), (b, after) = (views(record[1]) for record in (old, new))
    if not before or not after:
        return "no_eligible_excerpt", None
    opcodes = SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    options = []
    for previous, item, following in zip(opcodes[:-2], opcodes[1:-1], opcodes[2:], strict=True):
        tag, lo, hi, first, last = item
        if tag != "replace" or previous[0] != "equal" or following[0] != "equal" or not 1 <= hi - lo == last - first <= 3:
            continue
        for i, j in zip(range(lo, hi), range(first, last), strict=True):
            if i not in before or j not in after:
                continue
            if not .10 <= distance(a[i], b[j]) <= .70:
                continue
            if valid_sentence({"text": b[j], "eos": True}, a[i], generated=False):
                options.append((before[i], after[j], {"earlier_indexes": [lo, hi], "later_indexes": [first, last],
                                                     "previous_exact_sha256": text_hash(a[lo - 1]),
                                                     "next_exact_sha256": text_hash(a[hi])}))
    if not options:
        return "no_anchored_center_replacement", None
    earlier, later, hunk = min(options, key=lambda values: text_hash(values[0]["text"] + "\0" + values[1]["text"]))
    parent = int(prior.git(repo, "show", "-s", "--format=%ct", row["before"]).strip())
    if parent > row["after_committed"]:
        return "nonchronological_parent", None
    return "qualified", {
        "publisher": "python-peps", "license": "Public-domain or CC0 statement in both exact Git blobs",
        "attribution": old[2], "before": earlier, "after": later, "diff": hunk,
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
                      "mechanical_groups": len(found), "qualified_groups": len(selected), "history": history}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "qualify", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("prior", type=Path)
    cli.add_argument("legacy", type=Path)
    cli.add_argument("adjacent", type=Path)
    cli.add_argument("repo", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    lock = args.run / "source-freeze.json"
    initial = prior.freeze(args.repo)
    require(read(args.prior / "source-freeze.json") == initial, "Initial metadata or prior text files changed")
    frozen = {"method": PLAN, "source_code_sha256": digest(Path(__file__)),
              "initial_freeze_sha256": digest(args.prior / "source-freeze.json"),
              "legacy_source_sha256": digest(args.legacy / "source-receipt.json"),
              "adjacent_source_sha256": digest(args.adjacent / "source-receipt.json"),
              "legacy_renderer_sha256": digest(Path(__file__).with_name("writing_pep_legacy_intake.py"))}
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
