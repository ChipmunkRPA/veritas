"""Extend the frozen PEP prose-diff intake to every remaining dated work."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.extract import text_hash
from scripts import writing_pep_prose_diff_intake as comparison
from scripts import writing_pep_revision_intake as prior
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_focus_v3 import digest

METHOD = {
    "scope": "All remaining work IDs after the first 200 SHA-256-ordered dated PEP histories. "
             "Preserve the same prose-diff and source-rights conditions.",
    "combination": "Keep earlier admitted works first. Omit repeated documented authors, recorded editors, "
                   "and normalized 12-token overlap across the first and extension blocks.",
}


def full_catalog(repo):
    log = prior.git(repo, "log", prior.ANCHOR, "--since=2015-01-01T00:00:00Z", "--no-merges",
                    "--format=%x1e%H%x1f%P%x1f%at%x1f%ct%x1f%an%x1f%s", "--name-only", "--",
                    "pep-*.txt", "pep-*.rst")
    rows = defaultdict(list)
    for record in log.split("\x1e")[1:]:
        header, *paths = record.strip().splitlines()
        commit, parent, author_time, committer_time, editor, subject = header.split("\x1f", 5)
        if (not re.fullmatch(r"[0-9a-f]{40}", parent)
                or not all(prior.FROM <= int(x) < prior.UNTIL for x in (author_time, committer_time))
                or re.match(r"(?i)^(?:revert|merge|auto)", subject)
                or re.search(r"(?i)\[bot\]|^(?:github|pep bot)$", editor)):
            continue
        for path in paths:
            found = prior.PATH.fullmatch(path)
            if found and int(found[1]) >= 100:
                rows[found[1]].append({"path": path, "after": commit, "before": parent,
                                       "after_authored": int(author_time), "after_committed": int(committer_time),
                                       "editor": " ".join(editor.casefold().split())})
    return {number: sorted(rows[number], key=lambda row: text_hash(row["after"]))[:prior.COMMITS_PER_WORK]
            for number in sorted(rows, key=text_hash)}


def combine(previous, extra):
    roster = list(previous)
    names = {name for row in previous for name in row["attribution"]}
    editors = {row["source"]["editor"] for row in previous}
    fragments = set.union(*(fingerprints(row["before"]["text"]) | fingerprints(row["after"]["text"]) for row in previous))
    counts = Counter()
    for row in extra:
        incoming = fingerprints(row["before"]["text"]) | fingerprints(row["after"]["text"])
        editor = row["source"]["editor"]
        if fragments & incoming:
            counts["cross_block_text_overlap"] += 1
        elif set(row["attribution"]) & names or editor in editors:
            counts["cross_block_identity_overlap"] += 1
        else:
            names.update(row["attribution"])
            editors.add(editor)
            fragments.update(incoming)
            roster.append(row)
            counts["extension_admitted_to_combined"] += 1
    return roster, dict(counts)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "qualify", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("initial", type=Path)
    cli.add_argument("previous", type=Path)
    cli.add_argument("repo", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    work = full_catalog(args.repo)
    first = dict(list(work.items())[:prior.POOL])
    extra = dict(list(work.items())[prior.POOL:])
    require(first == read(args.initial / "source-freeze.json")["work_roster"] and not first.keys() & extra.keys(),
            "First or extension metadata roster differs")
    locked = args.run / "source-freeze.json"
    freeze = {"method": METHOD, "code_sha256": digest(Path(__file__)),
              "prior_commitments": {name: digest(args.previous / name)
                                    for name in ("source-freeze.json", "source-receipt.json", "originals.json")},
              "comparison_sha256": digest(Path(comparison.__file__)), "first_works": len(first), "extension": extra}
    if args.phase == "freeze":
        args.run.mkdir()
        write(locked, freeze)
        return
    require(read(locked) == freeze, "Frozen disjoint metadata changed")
    supplemental, summary = comparison.qualify(args.repo, extra)
    first_groups = read(args.previous / "originals.json")
    roster, combined = combine(first_groups, supplemental)
    originals, receipt, extra_file = (args.run / f"{name}.json" for name in ("originals", "source-receipt", "extension-only"))
    summary.update({"method": METHOD, "extension_qualification": summary["qualified_groups"],
                    "extension_work_pool": summary["work_pool"], "qualified_groups": len(roster),
                    "first_qualified_groups": len(first_groups), "combined": combined})
    if args.phase == "qualify":
        write(extra_file, supplemental)
        write(originals, roster)
        write(receipt, summary | {"source_freeze_sha256": digest(locked), "originals_sha256": digest(originals)})
    else:
        require(read(extra_file) == supplemental and read(originals) == roster and read(receipt) == summary | {
            "source_freeze_sha256": digest(locked), "originals_sha256": digest(originals)
        }, "Independent extension source replay differs")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
